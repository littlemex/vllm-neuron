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
- **Recurrent state without a runner state pool.** The Mamba2 ssm/conv state is carried across decode
  steps via in-place module buffers; the plugin's `AliasingOutputRewritePass` turns the `copy_` into
  an HLO `input_output_alias`, so state persists across decode-step graphs (batch=1 / `max_num_seqs=1`).
- **DGE-free MoE router.** The top-k gate is built with reductions + elementwise comparisons only
  (no data-dependent `scatter`/`gather`), which avoids a neuronx-cc miscompilation that surfaced as a
  `scatter/gather (vector DGE) out-of-bound` once several MoE layers were stacked. Math is identical
  to the argmax+scatter router.
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
| EP (expert parallel) | N/A | MoE runs a dense per-expert loop on the local TP shard (fast NF.moe path is future work) |
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
- **Batch size.** batch=1 (`max_num_seqs=1`): the Mamba2 recurrent state is a single per-layer
  buffer (no per-slot pool), so concurrent sequences would corrupt each other. Both the Mamba prefill
  (batch != 1) and attention decode raise rather than producing wrong output.
- **Automatic Prefix Caching (APC) is not supported.** Do not set `--enable-prefix-caching`. The
  attention KV is addressable by block hash and can be reused across requests, but the Mamba2
  recurrent state is a single mutable buffer with no block-hash addressing, so a reused prefix would
  silently continue from the wrong SSM/conv state. (The stock runner only checks that APC implies
  segmented prefill; it does not know this model is recurrent, so the guard is the operator's.)
- **Speculative decoding is not supported.** The SSM `forward_decode` advances the state by exactly
  one token; a multi-token verify step would silently process only the first. `forward_decode` raises
  on `T != 1` rather than mis-generating.
- **Bucket padding is masked out of the SSM/conv state.** Neuron pads a prefill up to a fixed bucket
  width; attention ignores the pad via `slot_mapping = -1`, and the Mamba path derives the same
  real/pad mask from `slot_mapping` to zero `dt` on pad positions (identity recurrence steps) and to
  gather the conv state from the real tail — so the state handed to decode is independent of the
  padding (pinned by `test_prefill_pad_invariance`).
- **Precision.** bf16 only (FP8/NVFP4 is future work).
- **Layer types.** Only the `M` (Mamba2), `E` (MoE), and `*` (Attention) `hybrid_override_pattern`
  entries are implemented. Plain-MLP layers (`-`) are not supported and raise at construction; the
  30B-A3B checkpoint does not use them.

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

On trn2.3xlarge + EFA at TP=4 (real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` weights), greedy
generation matches the HF reference: "The capital of France is" → " Paris" (first token matches the
HF CPU reference exactly); "2 + 2 =" → " 2 + 2 = 4"; "日本の首都は" → "東京…".

## Module Structure

```text
vllm_neuron/model/nemotron_h/
├── __init__.py        # env-workaround note + package re-exports
├── README.md          # This file
├── config.py          # NemotronHConfig: Omni-wrapper unwrap + attribute_map alias recovery
├── factory.py         # NemotronHForCausalLM factory (bf16 today; FP8/NVFP4 future)
├── ssd.py             # chunked_ssd_scan + segmented_causal_conv1d: Mamba2 SSD prefill + conv carry
├── ops.py             # gated_rmsnorm + dense_moe_gate (single source; test compares vs HF reference)
└── model_bf16.py      # RMSNorm / Attention / MoE / Mamba2 mixers + model + HF weight loader
```

## Testing

```text
test/vllm_neuron/model/nemotron_h/bf16/
└── test_nemotron_h_kernels.py   # CPU equivalence tests (no Neuron device / no checkpoint):
                                  #  - chunked SSD == sequential recurrence (chunk boundaries, long
                                  #    sequences, small chunks, prefix state) + fp32 stress
                                  #  - segmented/continuation prefill == single-shot (SSM + conv1d
                                  #    carry, first-segment mask); bucket-padding does not change the
                                  #    state (pad-invariance) + negative tests (missing mask diverges)
                                  #  - mask-before-exp is required to avoid inf*0 = NaN
                                  #  - gated RMSNorm and DGE-free MoE gate match the ACTUAL HF
                                  #    reference (MambaRMSNormGated / NemotronHTopkRouter), not a
                                  #    self-authored oracle
```

Run with `pytest test/vllm_neuron/model/nemotron_h/bf16/`. Full-model / HF-parity correctness is
covered by the on-device verification above.
