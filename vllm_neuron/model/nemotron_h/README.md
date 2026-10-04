# NemotronH (text backbone)

Serving implementation of the `NemotronHForCausalLM` text backbone of
**Nemotron-3-Nano-Omni-30B-A3B** (the Omni vision/audio encoders are out of scope; this is the
language model only). NemotronH is a hybrid decoder that interleaves **Mamba2 (SSM)**, **MoE**, and
**Attention** layers, selected per layer by `hybrid_override_pattern`.

## Architecture

| Parameter | Value |
|---|---|
| hidden_size | 2688 |
| num_hidden_layers | 52 (`hybrid_override_pattern`: 23 Mamba2 `M` / 23 MoE `E` / 6 Attention `*`) |
| vocab_size | 131072 |
| tie_word_embeddings | false |
| **Attention** | GQA, 32 query heads / 2 KV heads, head_dim 128, **NoPE** (no rotary; position information is carried by the Mamba2 layers) |
| **MoE** | 128 routed experts, top-6, + 1 shared expert; DeepSeek-style router (sigmoid + `e_score_correction_bias`, `routed_scaling_factor` 2.5, `norm_topk_prob`); `n_group=1`; relu² expert activation; `moe_intermediate_size` 1856, shared 3712 |
| **Mamba2** | `mamba_num_heads` 64, `mamba_head_dim` 64, `ssm_state_size` 128, `n_groups` 8, `conv_kernel` 4; grouped gated RMSNorm (`group_size` = intermediate/`n_groups` = 512); `time_step_limit` (0.0, inf) |
| residual stream | fp32 (`residual_in_fp32`) |
| dtype | bfloat16 |

## Key Differences from Reference (GPT-OSS BF16)

- **Hybrid backbone.** Unlike the attention-only reference models, layers dispatch by
  `hybrid_override_pattern` to one of three mixers (Mamba2 / MoE / Attention). MoE and Attention
  follow the existing plugin patterns; Mamba2 is the new piece.
- **Mamba2 SSM, native to the plugin compile path (no `torch_neuronx.trace` delegation).** Prefill
  uses a **vectorized SSD (quadratic / attention-form) selective scan** — matmuls + cumsum + a causal
  mask + a bounded (exponent ≤ 0) decay, the same op shapes as attention — so it compiles on the
  neuronx-cc path. The causal mask is applied to the decay exponent **before** `exp` (the upper
  triangle would otherwise overflow to +inf on real `dt` and produce `inf*0 = NaN`).
- **Recurrent state in a model-side pool.** Each Mamba2 layer keeps its ssm/conv state in a pool of
  `2 * max_num_seqs + 1` rows held in module buffers and updated in place, so the state persists
  across the runner's graph calls. The runner hands the model no per-request index and a request's
  batch row changes from step to step, so the row is found from the request's first KV block id,
  which stays fixed for the request's lifetime (`state_slots.py`, below).
- **DGE-free MoE router.** The top-k gate is built with reductions + elementwise comparisons only
  (no data-dependent `scatter`/`gather`), which avoids a neuronx-cc miscompilation that surfaced as a
  `scatter/gather (vector DGE) out-of-bound` once several MoE layers were stacked. Math is identical
  to the argmax+scatter router. Decode then reads the selected experts' weights by index (an indirect
  DMA in the NKI kernel; an `index_select` on the `NEMOTRONH_MOE_DECODE=torch` path).
- **NoPE attention.** No rotary embedding — matches the HF `NemotronHAttention`, which carries
  position information through the Mamba2 layers.
- **Config unwrapping.** `config.py` unwraps the Omni wrapper (`llm_config`/`language_model`/
  `text_config`) and recovers HF `attribute_map`-aliased fields; the weight loader auto-detects the
  checkpoint prefix (`backbone.*` for the text checkpoint, `language_model.backbone.*` for Omni).

## Feature Status

| Feature | Status | Notes |
|---|---|---|
| TP (tensor parallel) | ✅ | Attention Q-head + Mamba head + MoE expert-intermediate sharding; 30B-A3B needs TP=4 on one trn2 chip |
| SP (sequence parallel) | ✅ | Prefill SP-scatter at the embedding; Mamba all-gathers to full at the SSM boundary |
| DP (data parallel) | N/A | Not wired for this backbone yet |
| EP (expert parallel) | N/A | Experts are sharded on the intermediate axis (TP). Prefill runs every expert over every token as batched GEMMs over groups of experts; decode reads only the selected experts' weights |
| Cross-DP EP | N/A | See EP |
| Eagle3 spec decode | N/A | Not applicable |
| FP8 KV cache | N/A | bf16 only today (FP8/NVFP4 is future work; `factory.py` rejects other quantizations) |
| On-device sampling | ✅ | via `Sampler` when `on_device_sampling_config` is set |

## Known limitations

- **Prefill length.** The DEFAULT prefill scan is the chunked SSD (`ssd.py`, `chunked_ssd_scan`):
  O(l·C + T²) in sequence length (T = l/C chunks), so for a realistic `max_model_len` the linear
  `l·C` term dominates and long prefills fit where the full-sequence O(l²) form does not. It splits
  the sequence into chunks of `NEMOTRONH_CHUNK` (default 128) and combines an intra-chunk diagonal
  pass with an inter-chunk state pass solved in closed form on the chunk axis — no O(l) Python loop
  and no strided chunk split, so it compiles on neuronx-cc (verified on trn2). The full-sequence
  O(l²) vectorized form is opt-in via `NEMOTRONH_SCAN=quadratic` (short-seq / debugging). CPU
  equivalence to the sequential recurrence is pinned by `test_chunked_ssd_matches_sequential`
  (incl. an fp32 long-sequence stress test). The practical `max_model_len` ceiling is the per-bucket
  NEFF compile time (which grows with the number of sequence-length buckets), not the scan.
- **Segmented / continuation prefill (supported).** When `max_num_batched_tokens < max_model_len`
  (a supported segment size, e.g. `--max-num-batched-tokens 512`), the plugin auto-enables segmented
  prefill: the prompt is processed in `kv_segment_size` segments and the model carries state across
  the boundary — the Mamba2 layers carry BOTH the SSM state and the causal-conv1d state (in fp32,
  via the same in-place buffers + `AliasingOutputRewritePass` used for decode), and the Attention
  layers use `NF.segmented_attention` to read the prior segments' KV. `forward_prefill` takes a
  runtime `cached_seq_len`; the first-segment case (`cached_seq_len == 0`) is handled graph-statically
  by a mask that zeroes the carried state (no Python branch on a runtime value). This is what lets a
  `max_model_len` that a single-shot prefill cannot fit in host RAM run on one trn2 chip: e.g.
  `max_model_len 1024` with `--max-num-batched-tokens 512` serves 1024-token contexts on
  trn2.3xlarge (a single-shot 1024 prefill OOMs the 30B model's compile there), verified on-device by
  recalling a fact placed in the first 512-token segment when asked in the second. Split-vs-single-shot
  numerical equivalence (including the conv-history carry) is pinned by
  `test_segmented_prefill_matches_single_shot`. The opt-in quadratic form does not support this.
- **Single-shot prefill length is host-RAM-bound on one chip.** Without segmentation, the peak
  compile + NEFF/weight-load memory grows with `max_model_len`; on trn2.3xlarge (≈125 GiB) the 30B
  model fits single-shot up to a few hundred tokens (256 verified) but a 1024 single-shot prefill
  OOMs — use segmented prefill (above) or a larger instance for longer single-shot contexts.
- **Concurrent requests.** Decode batches up to `max_num_seqs` requests. The Mamba2 state pool row
  of each request is resolved inside the graph from its first KV block id (`state_slots.py`): a
  prefill's first segment takes the lowest free row, later segments and decode steps look the row up
  by owner, and a row is released when the runner reports its request finished
  (`release_request_state`, called between steps by `NeuronModelRunner._update_states`) or when its
  request is absent from a decode batch. This relies on the
  Neuron scheduler running either one prefill request (all its segments in a row) or a decode batch
  of every running request, never a mix; the pool holds `2 * max_num_seqs + 1` rows because requests
  that finished in the last decode step still look live to a prefill admitted before the next one.
  Prefill takes one request per step and raises otherwise. The KV cache must hold `max_num_seqs`
  full-length requests for the batch to fill (`--num-gpu-blocks-override`, or the scheduler holds
  requests back).
- **Automatic Prefix Caching (APC) is not supported.** Do not set `--enable-prefix-caching`. The
  attention KV is addressable by block hash and can be reused across requests, but the Mamba2
  recurrent state is per request with no block-hash addressing, so a reused prefix would silently
  continue from the wrong SSM/conv state. Shared first blocks would also break the state-row lookup
  above, which needs every running request to own its first KV block. (The stock runner only checks that APC implies
  segmented prefill; it does not know this model is recurrent, so the guard is the operator's.)
- **Speculative decoding is not supported.** The SSM `forward_decode` advances the state by exactly
  one token; a multi-token verify step would silently process only the first. `forward_decode` raises
  on `T != 1` rather than mis-generating.
- **Bucket padding is masked out of the SSM/conv state.** Neuron pads a prefill up to a fixed bucket
  width and points the padding tokens' KV writes at a slot no request owns (`PAD_SLOT_ID = -1`, or the
  reserved null block 0). The Mamba path treats a token as real exactly when its slot is in an
  allocated block (`slot_mapping >= block_size`), zeroes `dt` on the padding (identity recurrence
  steps) and gathers the conv state from the real tail, so the state handed to decode is independent
  of the padding (pinned by `test_prefill_pad_invariance` and, for both padding conventions, by
  `test_decode_after_prefill_matches_single_shot`).
- **Precision.** bf16 only (FP8/NVFP4 is future work).
- **Layer types.** Only the `M` (Mamba2), `E` (MoE), and `*` (Attention) `hybrid_override_pattern`
  entries are implemented. Plain-MLP layers (`-`) are not supported and raise at construction; the
  30B-A3B checkpoint does not use them.

## NKI kernels

On a Neuron device the hot paths run as NKI kernels; on CPU (tests) and for shapes a kernel does not
support, the PyTorch path computes the same thing. Each kernel splits its work across the two
physical cores of a logical core (LNC=2).

| Kernel | Used in | What it does |
|---|---|---|
| `ssd_prefill_kernel.py` | Mamba2 prefill | Chunked SSD scan (128-token chunks), one SSM group per core. Exponents are formed as clamped differences `cs_i - cs_j` (never `exp(cumsum)`), so a large `|A| * sum(dt)` within a chunk cannot overflow. |
| `mamba_decode_kernel.py` | Mamba2 decode | Conv step + SSM step + gated RMSNorm for a batch of requests, one SSM group per core. Per-channel and per-head work is done for all requests at once (requests on the free axis); per-head values reach the (head, p) partitions through a matmul against a head selector. |
| `moe_decode_kernel.py` | MoE decode | Reads only each token's selected experts (indirect DMA on the expert index), relu² expert MLP, cores split the intermediate axis. |
| `matvec_kernel.py` | decode projections | `x @ W` for a few rows: Mamba in/out, attention qkv/o, the vocabulary projection. The shared expert keeps the compiler matmul (faster on trn2). |

## Environment variables

`run.py` sets `NEURON_SKIP_EFA_AFFINITY=1` because the TP=4 target (trn2.3xlarge) has a single EFA
card and the Neuron EFA-affinity probe expects a co-located EFA under each NeuronCore's PCI path
(true only on multi-card instances like trn2.48xlarge); the affinity is a CPU-locality optimization,
not a correctness requirement. On a multi-card instance you can leave it unset.

Prefill-scan selection:

| Variable | Default | Effect |
|---|---|---|
| `NEMOTRONH_SCAN` | chunked | Prefill scan. `chunked` (default): O(l·C + T²), long sequences. `quadratic`: O(l²) vectorized form (short-seq / debugging). `sequential`: the 1-step-recurrence oracle (numerically equivalent, but does not compile on the neuronx-cc path). |
| `NEMOTRONH_CHUNK` | 128 | Chunk size C for the chunked SSD (only used when `NEMOTRONH_SCAN=chunked`). Must be `<= max_model_len` so a prefill spans at least one full chunk without heavy right-padding — see the compile caveat below. |

**Compile caveat — keep `NEMOTRONH_CHUNK <= max_model_len`.** When a compiled prefill bucket is
shorter than the chunk size, the sequence is right-padded up to a single chunk (T = 1, large pad),
and neuronx-cc fails that graph with `NCC_IMPR902` (MaskPropagation, `isl_set_union … spaces don't
match`). A bucket that spans two or more chunks with no heavy padding (T >= 2) compiles cleanly. The
shipped preset (`max_model_len` 512, chunk 128 → T = 4) is safe; only a `max_model_len` below the
default chunk size needs a smaller `NEMOTRONH_CHUNK` (e.g. 16 for `max_model_len` 32). This is a
compile-time limit on the degenerate short-bucket case, not a numerical one — the scan math is
identical for any chunk size (pinned by `test_chunked_ssd_matches_sequential`, which covers small
chunk sizes down to T = 2 at l = 32).

Numerics vs the opt-in quadratic form: the two scans are mathematically identical (fp64 agreement is
machine-epsilon) but reduce in a different order, so in bf16 they can pick a different greedy token on
a near-tie — exactly as the quadratic form itself differs from the 1-step recurrence in bf16 (the
chunked scan adds no error beyond that reformulation-rounding envelope).

Kernel selection (each defaults to the NKI kernel on a Neuron device; `torch` selects the PyTorch path):

| Variable | Default | Effect |
|---|---|---|
| `NEMOTRONH_SSD_PREFILL` | nki | Mamba2 prefill scan kernel. |
| `NEMOTRONH_MAMBA_DECODE` | nki | Mamba2 decode-step kernel. |
| `NEMOTRONH_MOE_DECODE` | nki | MoE decode kernel. |
| `NEMOTRONH_MATVEC` | nki | Decode projection kernel. |
| `NEMOTRONH_MATVEC_OFF` | shared | Comma-separated call sites (`mamba`, `attn`, `shared`, `lm_head`) that keep the plain matmul. |
| `NEMOTRONH_MOE_GROUP` | 16 | Experts per batched GEMM in the MoE prefill (keeps the intermediate in SBUF). |

The following are **diagnostic only — do not set them in production**; they change or disable numerics:

| Variable | Default | Effect |
|---|---|---|
| `NEMOTRONH_MAMBA_STUB=1` | off | Skip the SSM recurrence entirely (returns non-sense output). For isolating the Mamba path only. |
| `NEMOTRONH_DEBUG_LOAD=1` | off | Log weight-loading coverage (unmapped params, still-on-meta tensors) at DEBUG level. |

## Checkpoint license

This integration code is Apache-2.0 (see the SPDX headers). The model **weights** carry their own
license — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` is published under the NVIDIA Open Model
License; review and comply with it before redistributing weights or a derived checkpoint. (The
`transformers` `modeling_nemotron_h.py` referenced in comments for math semantics is Apache-2.0.)

## Verification

On trn2.3xlarge at TP=4 (real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` weights, `max_model_len
8192`, `--max-num-batched-tokens 512`), compared with the same checkpoint on vLLM GPU (bf16, TP=2):

| Check | GPU | Trainium |
|---|---|---|
| Greedy probes (facts, two-step arithmetic, ordering, chat, and multi-fact questions over 1k / 4k / 7k-token contexts) | 12/12 | 12/12, same text |
| gsm8k 5-shot, first 250 test questions, strict-match | 217/250 (86.8%) | 215/250 (86.0%) |

On the 250 gsm8k questions the two backends disagree on 16 (9 solved only on GPU, 7 only on Trainium; exact McNemar p = 0.80).

Throughput on the same setup (`max_model_len 8192`, `--max-num-batched-tokens 512`):

| Measure | Value |
|---|---|
| Decode, one request | 9.2 ms/token |
| Decode, 8 concurrent requests (`--max-num-seqs 8`, 128 tokens each, including their prefills) | 215 tokens/s aggregate |
| Prefill | about 2600 tokens/s (487 to 7165-token prompts) |
| Greedy probes, all 12 submitted at once (8 decoded concurrently) | 12/12 |

A request's output in a batch does not depend on which other requests share the batch (bit-identical
over different partners). It can differ from the same request run alone: a batch of `b` runs a
different compiled graph than a batch of 1, and bf16 rounding differs between the two.

## Module Structure

```text
vllm_neuron/model/nemotron_h/
├── __init__.py        # env-workaround note + package re-exports
├── README.md          # This file
├── config.py          # NemotronHConfig: Omni-wrapper unwrap + attribute_map alias recovery
├── factory.py         # NemotronHForCausalLM factory (bf16 today; FP8/NVFP4 future)
├── ssd.py             # chunked_ssd_scan + segmented_causal_conv1d: Mamba2 SSD prefill + conv carry
├── ops.py             # gated_rmsnorm + dense_moe_gate (single source; test compares vs HF reference)
├── state_slots.py     # which Mamba2 state-pool row each request of a step uses
├── ssd_prefill_kernel.py, mamba_decode_kernel.py, moe_decode_kernel.py, matvec_kernel.py
│                      # NKI kernels (see above)
└── model_bf16.py      # RMSNorm / Attention / MoE / Mamba2 mixers + model + HF weight loader
```

## Testing

```text
test/vllm_neuron/model/nemotron_h/bf16/
├── test_nemotron_h_kernels.py   # CPU equivalence tests (no Neuron device / no checkpoint):
│                                #  - chunked SSD == sequential recurrence (chunk boundaries, long
│                                #    sequences, small chunks, prefix state) + fp32 stress
│                                #  - segmented/continuation prefill == single-shot (SSM + conv1d
│                                #    carry, first-segment mask); bucket-padding does not change the
│                                #    state (pad-invariance) + negative tests (missing mask diverges)
│                                #  - mask-before-exp is required to avoid inf*0 = NaN
│                                #  - gated RMSNorm and DGE-free MoE gate match the ACTUAL HF
│                                #    reference (MambaRMSNormGated / NemotronHTopkRouter)
├── test_nemotron_h_tp.py        # the real model on CPU ranks over gloo, from a tiny random
│                                # checkpoint loaded through load_weights:
│                                #  - prefill at TP=2 and TP=4 == TP=1 (sharding + SP collectives)
│                                #  - prefill (unpadded, or padded with PAD_SLOT_ID or the null block)
│                                #    + one decode step == single-shot prefill, at TP=1 and TP=4
│                                #  - two requests decoded in one reordered, padded batch == each
│                                #    request on its own, at TP=1 and TP=4
│                                #  - the grouped-GEMM MoE == the per-expert sum over the checkpoint
├── test_nemotron_h_state_slots.py  # state-pool rows under a simulated scheduler (admissions,
│                                #    multi-segment prefills, shuffled padded decode batches,
│                                #    finished requests, reused first blocks)
├── test_nemotron_h_*_kernel.py  # each NKI kernel on the NKI CPU simulator, against the shipped
│                                #    PyTorch scan (SSD) or a NumPy transcription of the decode math
└── test_nemotron_h_kernel_trace.py # each NKI kernel through the device compiler frontend on meta
                                 # tensors (catches constructs the simulator accepts but the device
                                 # frontend rejects)
```

Run with `pytest test/vllm_neuron/model/nemotron_h/bf16/`. `test_nemotron_h_tp.py` needs the
plugin's Python dependencies (it skips without them) but no Neuron device; the kernel tests need the
NKI package, and the trace test the Neuron compiler stack (`NEURON_PLATFORM_TARGET_OVERRIDE=trn2` is
set for it, so no device is needed). The on-device numerics
(NKI kernels, compiled graphs) are covered by the verification above.
