# Qwen3.5-MoE

Serving implementation of **`Qwen/Qwen3.6-35B-A3B`** — whose HF `architectures` entry is
`Qwen3_5MoeForConditionalGeneration` and whose `model_type` is `qwen3_5_moe`.

Two architectures are registered, and the name selects which one runs:

| Architecture | What it serves | State |
|---|---|---|
| `Qwen3_5MoeForCausalLM` | The text backbone | Verified on device |
| `Qwen3_5MoeForConditionalGeneration` | Text backbone + vision tower | Wired; **not yet run on a device** |

The multi-token-prediction head is implemented (`mtp.py`: the four modules, the weight map, a KV spec and
a cache binding) and **unreachable**: the runner accepts `eagle3` as the only speculative method, so
reaching a draft head needs an `mtp` branch and a proposer in a shared file. What is verified about it is
on CPU and is listed under "Multi-token verification" below.

Qwen3.5-MoE is a hybrid decoder that interleaves **Gated DeltaNet** linear-attention layers with
**gated GQA** full-attention layers, and puts a **256-expert MoE** block on every layer.

## Architecture

| Parameter | Value |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 40 (`layer_types`: three `linear_attention` then one `full_attention`, repeating — `full_attention_interval` 4, so 30 GDN / 10 GQA) |
| vocab_size | 248320 |
| tie_word_embeddings | false |
| **Gated DeltaNet** | 16 key heads x 128, 32 value heads x 128 (query/key are repeat-interleaved to the value head count), depthwise causal conv kernel 4 (no bias), SiLU conv activation; recurrent state `[32, 128, 128]` per layer in fp32 |
| **Full attention** | GQA, 16 query heads / 2 KV heads, head_dim 256; `q_proj` is double width and emits an **output gate** alongside the query; per-head `q_norm`/`k_norm`; **partial rotary** (`partial_rotary_factor` 0.25 → the leading 64 of 256 channels rotate), `rope_theta` 1e7 |
| **MoE** | 256 routed experts, top-8, `moe_intermediate_size` 512, plus one **sigmoid-gated** shared expert of the same width; router = softmax over all experts (fp32) → top-8 → L1 renormalise |
| RMSNorm | `x_norm * (1 + weight)`, weights stored as offsets from unity |
| residual stream | fp32 (a deliberate deviation — see below) |
| dtype | bfloat16 |

## Key differences from the reference models in this plugin

- **Hybrid backbone.** Layers dispatch by `layer_types` to a Gated DeltaNet or a gated GQA mixer.
  The MoE half is identical for both.
- **Gated DeltaNet, native to the plugin compile path.** Prefill uses the chunked gated delta rule:
  a fully vectorised intra-chunk pass plus a sequential pass over the (few) chunks. Unlike a Mamba2
  SSM, the update subtracts the state's own prediction `S^T k`, so the operator that carries state
  across a chunk boundary is **not diagonal** and the inter-chunk pass has no closed form on the
  chunk axis — hence the loop over chunks (`T = ceil(l / 64)`, e.g. 8 for a 512-token prefill),
  which is bounded and unrolls cleanly, unlike a loop over the sequence.
- **The chunk-internal triangular system is inverted by recursive block inversion.**
  `torch.linalg.solve_triangular` is not available on the compile path, and the HF export fallback's
  `chunk_size`-step forward substitution unrolls 64 shrinking slices. Instead the unit lower
  triangular system is inverted exactly with a constant mask and two matmuls per level:
  `Inv_2s = Inv_s - Inv_s @ Lsub @ Inv_s`, `ceil(log2(chunk_size))` levels. All static shapes, no
  solver, no data-dependent control flow. **The algebraically equivalent Neumann / repeated-squaring
  form was measured and rejected**: it materialises powers up to `N^(C-1)` that then cancel, and once
  `||N||_2 > 1` it loses fp32 entirely (max error 4.4 on a system whose true inverse has condition
  number 17, where this form and forward substitution both stay near 1e-7). `||N||_2 > 1` is the
  ordinary regime here — it happens whenever the log decay is weak and the keys inside a chunk are
  correlated, which is normal for adjacent tokens.
- **Recurrent state, two ways.** By default the conv and recurrent state are carried across decode steps
  by in-place module buffers; the plugin's `AliasingOutputRewritePass` turns the `copy_` into an HLO
  `input_output_alias` so the state persists across the runner's per-step graphs, and the runner needs no
  change. That path is single-sequence. With `QWEN3_5_MOE_STATE_POOL=1` the state instead lives in a
  runner-allocated pool with a slot axis, which is what more than one sequence requires; see the
  concurrency entry under "Limits".
- **Two norm conventions.** The residual-stream norm scales by `1 + weight` (weights near zero), while
  the Gated DeltaNet's gated norm scales by `weight` (weights near one) and applies the gate after
  normalising. Using the plugin's usual `weight * x` for the former would collapse the residual stream.
- **Gated attention with partial rotary.** `q_proj` packs `[query | gate]` per head; the attention
  output is multiplied by `sigmoid(gate)` before `o_proj`. Only the leading 64 of 256 head channels
  rotate. `NF.qkv_proj`'s fused rotary is delimited by `num_kv_heads` and knows nothing about either,
  so the projection is a plain matmul.
- **MRoPE is collapsed, not approximated.** The reference builds three positional axes and interleaves
  their frequencies. For a text-only prompt all three carry the same position, so the interleave
  selects from three identical tensors and the result is exactly the single-axis partial rotary table
  used here (pinned by `test_rotary_tables_match_hf_for_text_only_positions`). Image and video inputs
  break that equality, which is why the vision path is excluded rather than approximated.

## Deliberate deviations from the checkpoint

- **fp32 residual stream.** The checkpoint accumulates the residual in bf16. Over a 40-layer stack
  with a recurrent mixer that was empirically unstable (the same finding as the NemotronH port, where
  it produced all-zero tokens on real weights), so the residual is kept in fp32 and each block
  normalises down to bf16 for the weights. This is a real deviation and not a free improvement: HF
  rounds each residual add to bf16 before the next norm reads it, so a component near a rounding
  boundary can give a different normalised activation here and change MoE routing or a near-tie greedy
  token. Higher precision is not the same model.
- **No sequence parallelism.** Hidden states stay full on every rank and each mixer all-reduces its
  output, rather than scatter/gather around every mixer boundary. Correctness-first; it also removes
  the "prefill token count must be a multiple of TP" constraint. SP is a later optimisation.

## Feature status

| Feature | Status | Notes |
|---|---|---|
| TP (tensor parallel) | Yes | Attention Q heads, GDN key/value heads, MoE expert intermediate; 35B-A3B needs TP=4 on one trn2 chip |
| SP (sequence parallel) | No | See the deviation above |
| DP (data parallel) | No | Not wired for this backbone |
| EP (expert parallel) | No | Every rank holds all 256 experts and shards the intermediate dimension; rejected at construction |
| Eagle3 / speculative decode | No | Rejected at construction and in `forward` — see below |
| FP8 / NVFP4 weights | No | bf16 only; the factory rejects other quantizations and the config rejects a non-bf16 checkpoint dtype |
| `--dtype` other than bfloat16 | No | Rejected at construction. Every parameter, the Gated DeltaNet kernels and the declared KV dtype are bf16, so another dtype disagrees with the graph rather than trading precision for it |
| Quantized KV cache (`--kv-cache-dtype`) | No | Rejected at construction. The runner allocates the cache from `cache_config`, not from this model's KV spec, so an fp8 cache would be read as bf16 with nothing raising. `bind_caches` also checks the dtype it is handed against the dtype the attention reads with |
| On-device sampling | Yes | Via `Sampler` when `on_device_sampling_config` is set |
| Image input | Wired, unverified | `Qwen3_5MoeForConditionalGeneration` builds the vision tower, encodes into the runner's encoder cache and merges at prefill. **No device run yet** — see below for what that leaves open |
| Video input | No | Frames pack per frame rather than per item; `embed_multimodal` refuses video rather than packing it as one item |
| Prompt embeddings | No | The model embeds `input_ids` only; the factory rejects `enable_prompt_embeds` and `forward` raises on `inputs_embeds` |
| Decode-context parallel (DCP) | No | Refused for both `apply_prefill_dcp` and `decode_context_parallel_size > 1`: the decode attention reads the local KV cache with no cross-rank gather |

## Known limitations

- **Long contexts are bounded by two things before they are bounded by the scan.** The decode path
  gathers the sequence's whole KV context densely and in fp32 (`[heads, context, head_dim]` per
  attention layer), so its cost grows with the bucket's `max_blocks_per_seq` rather than with the real
  context; and the chunked prefill scan unrolls `ceil(bucket / chunk)` chunk bodies per Gated DeltaNet
  layer, so a very wide prefill bucket produces a very large graph and a long compile. Raising
  `QWEN3_5_MOE_GDN_CHUNK` trades chunk-internal work for fewer unrolled bodies. Neither is a numerical
  limit; both are why the shipped example starts at a modest `max_model_len`.

- **Concurrency needs the state pool, and is verified on CPU only.** `max_num_seqs > 1` is refused unless
  `QWEN3_5_MOE_STATE_POOL=1` asks for the per-slot state; with the pool on it proceeds and logs a warning
  that the multi-request path has not run on a device. What is verified, at rung 2 with the whole runner:
  four requests in one scheduler step produce token-for-token what the same four produce at
  `max_num_seqs=1`. The pieces behind that are a slot axis on the conv and recurrent state, a packed
  chunk-aligned prefill whose scan carries per-request masks and whose convolution gathers each request's
  own history, and a decode attention that is batch-general (`ops.paged_decode_attention`). Nothing here
  has been measured on a device, so the seat count that pays for itself is not yet known.

- **Automatic prefix caching is not supported.** Do not set `--enable-prefix-caching`. The attention
  KV is addressable by block hash and would be reused across requests, but the recurrent state has no
  block-hash addressing, so a reused prefix would silently continue from the wrong state. The factory
  rejects it.

  Note that enabling prefix caching also selects a **recurrent-state reuse mode** on this model's
  behalf: `MambaModelConfig.verify_and_update_config` forces `cache_config.mamba_cache_mode` to `none`
  whenever prefix caching is off, and derives `align` or `all` whenever it is on. Nothing on the command
  line says so, which is why the two are refused together rather than separately — a standalone
  `mamba_cache_mode` guard cannot be reached while prefix caching is refused, and reports a setting the
  operator never chose.
- **Speculative decoding is not supported.** The decode path advances the recurrent state by exactly
  one token; a multi-token verify step would leave the state inconsistent with the accepted tokens.
  The factory refuses a speculative config, and `gated_delta_net_decode` raises rather than
  mis-generating if one gets through.
- **Decode-context-parallel prefill (`apply_prefill_dcp`) is not supported.** The recurrent layers need
  the whole prefix in one pass, and derive their real/pad mask from an unsliced `slot_mapping`. The
  factory refuses it.
- **A one-token request is indistinguishable from a decode step by token count**, so the model does not
  rely on the distinction: the carried conv and recurrent state are zeroed whenever a step's first
  token sits at absolute position 0. That is also what makes a scheduler preemption which resumes by
  recomputing from position 0 safe (pinned by
  `test_decode_first_token_mask_starts_from_zero_state`).
- **A mostly-padded prefill bucket costs as much as a full one.** The MoE dispatch schedule is built
  from the unmasked affinity pattern and sized by the bucket width, so the padding mask handed to
  `build_blockwise_mapping` only zeroes the pad tokens' contribution to the result — it does not make a
  short prompt in a wide bucket cheaper. Choosing bucket widths is how that cost is controlled.
- **Bucket padding is masked out of the recurrent state.** Neuron right-pads a prefill to a fixed
  bucket width. Attention ignores the pad via `slot_mapping = -1`; the Gated DeltaNet derives the same
  real/pad mask from `slot_mapping` and forces the pad steps to the identity (zero log decay, zero
  beta) and gathers the conv history from the real tail. Without this the state handed to decode is
  the state after convolving the padding, which is silently wrong from the second decode token on.
  The mask's contract is narrower than its name: the real tokens must form a prefix and the padding a
  contiguous suffix, which is what Neuron's bucket padding of a single sequence produces.
- **Padding K/V is neutralised, not redirected to a reserved slot.** The sentinel
  `slot_mapping = -1` must not be scattered arithmetically: `-1 // block_size` is `-1`, which resolves
  to the last block of the cache. A fixed "reserved" sink is no better, because nothing reserves a
  physical block — the allocator may have given the active sequence exactly that one. The pad rows are
  therefore pointed at a slot that is guaranteed to be real and given that row's value, so every
  duplicate writes the same thing and the result does not depend on which write wins
  (`ops.redirect_padded_slots`, pinned by the padded-cache-write tests including the case where the
  sequence ends in the final slot). This assumes duplicate scatter destinations are merely unordered,
  not faulting — which is why the duplicated values are made identical rather than relying on an
  ordering.
- **Precision.** bf16 only. The config rejects a checkpoint whose declared dtype is not bfloat16 —
  the quantization field does not reveal it, and an fp32 checkpoint would roughly double the footprint.
- **An untied checkpoint with no `lm_head.weight` is refused** rather than served with the embedding
  matrix substituted as a tied head.
- **Layer types.** Only `linear_attention` and `full_attention` are implemented; any other entry in
  `layer_types` raises at construction. A stack with no `full_attention` layer is also rejected,
  because the prefill/decode phase and the real/pad mask are both derived from attention metadata.

## Serving knobs: honoured, refused, or warned about

Every setting an operator can turn is a promise. There are three honest states, and the fourth one --
read and silently discarded -- is what this table exists to prevent. The line between refusing and
warning is **whether the output would be wrong or merely incomplete**, not whether the feature is
implemented.

| Knob | State | Effect here |
|---|---|---|
| `--tensor-parallel-size` | honoured | Every layer and the vocabulary are sharded; 35B-A3B needs 4 on one trn2 chip |
| `--max-num-batched-tokens` | honoured | The runner derives the prefill buckets from it. Below `max_model_len` it enables segmented prefill, which reproduces single-shot output token for token |
| `--block-size` | honoured | Raised by the platform's hybrid alignment because this model declares recurrent state; the runner reads the aligned value rather than choosing one |
| `max_logprobs` | honoured | At 0 the vocabulary all-gather is skipped entirely, so a deployment that never asks for logprobs does not pay for it |
| `on_device_sampling_config` | honoured | Keeps logits sharded and samples on device |
| `--async-scheduling` | **warned** | Generation is correct, but the runner's async path returns the sampler output without logprobs, so that field comes back empty on a successful request. Warned rather than refused: refusing it would refuse the default configuration |
| `--kv-cache-dtype` | refused | See the feature table |
| `--dtype` other than bfloat16 | refused | See the feature table |
| `--enable-prefix-caching` | refused | See above, including the state-reuse mode it selects implicitly |
| `--max-num-seqs > 1` | refused | See the limitations |
| speculative config | refused | See the limitations |
| `--enable-expert-parallel` / `ep_degree` | refused | Every rank holds all experts |
| `--enable-prompt-embeds` | refused | The model embeds `input_ids` only |
| `decode_context_parallel_size` / `apply_prefill_dcp` | refused | No cross-rank gather in decode attention; recurrent layers need the whole prefix |

Every refusal happens at construction, before compilation or device capacity is spent, and every
message names this model so that a log search can attribute it.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `QWEN3_5_MOE_STATE_POOL` | unset (off) | Give the conv and recurrent state a slot axis so several requests can hold state at once. Off by default because `max_num_seqs > 1` is still refused: the pool is verified on CPU (identical tokens with the pool on, at several pool sizes, compiled and eager) and not yet on the device |
| `QWEN3_5_MOE_SLOT_INDEX` | 1 | With the pool on, read a slot with `index_select` (1) or with a one-hot product (0). The one-hot form materialises the whole pool and exists as a correctness reference, not as a serving path |
| `QWEN3_5_MOE_GDN_CHUNK` | 64 | Chunk width for the Gated DeltaNet prefill scan. A tiling choice only — the result is independent of it (pinned by `test_chunk_size_does_not_change_the_result`). **Measured: there is nothing to gain here.** At a 2048 prefill bucket, 32 gives TTFT 0.394 s against 0.376 s at the default 64, and 128 and 256 do not compile at all (`neuronx-cc compilation failed with 70`). The default sits just under the compiler's ceiling. |

On a `trn2.3xlarge` (a single EFA card) set `NEURON_SKIP_EFA_AFFINITY=1`: the Neuron EFA-affinity
probe expects a co-located EFA under each NeuronCore's PCI path, which only holds on multi-card
instances. The affinity is a CPU-locality optimisation, not a correctness requirement.

## Checkpoint license

This integration code is Apache-2.0 (see the SPDX headers). The model **weights** carry their own
license: review `Qwen/Qwen3.6-35B-A3B`'s license before redistributing weights or a derived
checkpoint. The `transformers` `modeling_qwen3_5_moe.py` referenced in comments for math semantics is
Apache-2.0.

## Module structure

```text
vllm_neuron/model/qwen3_5_moe/
├── __init__.py     # package re-exports
├── README.md       # this file
├── config.py       # Qwen3_5MoeConfig: unwraps text_config, lifts rope_parameters, validates topology;
│                   #   Qwen3_5MoeVisionConfig and Qwen3_5MoeMultimodalConfig (the wrapper's own fields)
├── factory.py      # implementation selection + serving-option validation, per architecture
├── gdn.py          # the Gated DeltaNet: chunked and one-step scans, the conv carry, and the whole
│                   #   mixer (single source of truth shared with the CPU tests)
├── layout.py       # weight layout transforms and the checkpoint key mapping (both CPU-tested)
├── mtp.py          # the multi-token-prediction head; unreachable from the runner, see above
├── multimodal.py   # Qwen3_5MoeForConditionalGeneration: the text model plus the vision tower
├── ops.py          # both RMSNorm conventions, the partial rotary, the one- and three-axis rotary
├── vision_inputs.py # the encoder's host-side tensors, over this repository's own preprocessing helpers
└── model_bf16.py   # attention / MoE / mixer modules, the model, and the HF weight loader
```

## Testing

```text
test/vllm_neuron/model/qwen3_5_moe/bf16/
└── test_qwen3_5_moe_kernels.py   # CPU equivalence (no Neuron device, no checkpoint):
                                  #  - the scans vs the ACTUAL HF reference kernels and vs the
                                  #    token-by-token recurrence; chunk-size independence; prefix
                                  #    state; extreme decay; mask-before-exp
                                  #  - the triangular inverse vs solve_triangular, including the
                                  #    large-operator-norm regime that rules out a Neumann form
                                  #  - the WHOLE mixer vs HF Qwen3_5MoeGatedDeltaNet: prefill,
                                  #    prefill-then-decode, an 8-step decode carry, bucket-padding
                                  #    invariance, segmented prefill
                                  #  - both norm conventions, the partial rotary and its text-only
                                  #    MRoPE collapse, the decay/beta projections, the router order
                                  #  - every weight-layout shard, reassembled and compared against
                                  #    the HF experts / attention / MLP modules at TP 1, 2 and 4
                                  #  - the padded paged-cache scatter, including the case where the
                                  #    sequence ends in the final slot of the cache
                                  #  - parsing the published config, and the shapes it must refuse
                                  #  - the checkpoint key mapping: complete, decoder-only, no
                                  #    duplicates. Point QWEN3_5_MOE_CHECKPOINT_INDEX at a real
                                  #    model.safetensors.index.json to diff it against a checkpoint
                                  #    (verified against Qwen/Qwen3.6-35B-A3B: nothing declared is
                                  #    absent, nothing in the decoder goes unconsumed)
                                  #  - negative tests for each load-bearing choice (offset norm,
                                  #    gate-after-norm, partial-vs-full rotary, the gate/up split,
                                  #    the per-head query/gate regroup, unmasked padding, the
                                  #    sentinel slot arithmetic)
```

Run with `pytest test/vllm_neuron/model/qwen3_5_moe/bf16/`. The tests import the shipped modules by
path, so they need neither a Neuron install nor model weights; they do need a `transformers` that
provides `qwen3_5_moe` (or `qwen3_next`, whose Gated DeltaNet kernels are the same) and skip the
reference comparisons otherwise.

The MoE and paged-attention kernels (`NF.router`, `NF.build_blockwise_mapping`, `NF.moe_cte`,
`NF.moe_block_tkg`, `NF.flash_attention`) cannot run on CPU. What the tests can and do pin is the
math those kernels are *configured* to compute — for example that the router's
softmax-then-top-k-then-renormalise order is the reference's, and that every weight shard handed to
them reassembles into the reference computation. Whether the kernels then compute it on the device is
what on-device verification covers.

## Verification status

CPU equivalence is complete (see Testing above), and `ruff` with the repository's configuration passes
clean on these files.

**On device: verified on a `trn2.3xlarge` at TP=4** (1 Trainium2 device, 4 NeuronCores, 96 GB, LNC=2),
plugin 0.21.0.1.0.0 / vLLM 0.21.0 / Neuron SDK 2.31, `max_model_len` 128, `max_num_seqs` 1.

*Compilation and state persistence*, on a reduced checkpoint with the same architectural shape (8
layers of the same `layer_types` period, the same head counts and dims, 32 experts, random weights):
compiles, generates without exception, and the recurrent state does not leak between requests — the
same prompt run three times with a different prompt interleaved gives byte-identical output. A
one-token prompt also completes, which is the case that would otherwise continue from the previous
request's state.

*Greedy agreement with the HuggingFace reference* on the real `Qwen/Qwen3.6-35B-A3B` weights (72 GB
loaded at TP=4), against `Qwen3_5MoeForCausalLM` run on CPU in the same container, comparing token IDs
for 8 greedy steps:

| Prompt | First token | Greedy prefix |
|---|---|---|
| `The capital of France is` | match | 8/8 |
| `2 + 2 =` | match | 8/8 |
| `日本の首都は` | match | 8/8 |
| `The largest planet in the solar system is` | match | 8/8 |
| `水の沸点は` | match | 6/8 (diverges at step 6) |

First token matches on 5/5 prompts; 38/40 tokens agree. The single divergence is at step 6 of one
prompt, which is the expected bf16 near-tie behaviour amplified over a 40-layer stack — and this port
deliberately keeps the residual in fp32, so exact end-to-end token equality is not the criterion (see
Deliberate deviations).

*Bucket independence.* A prefill graph is compiled per bucket, so a verification taken at one bucket
is a statement about that graph. The same five prompts were rerun with a single 2048-token prefill
bucket: byte-identical output. With a single 1024-token bucket, one of the five differs at step 7. See
Single-stream latency below for why that is the expected consequence of padding width changing the
reduction order, and Margins for the measurement that bounds it.

### Margins: what a token disagreement is worth

Free-running generations stop being comparable once they diverge, since every later step is
conditioned on a different prefix. Feeding the reference's own sequence back in makes each step an
independent comparison at the same input. Doing that for the five prompts (8 steps each, top-20
logprobs) gives 38/40 steps where the device and the reference pick the same token, and for the two
that differ:

| Step | Device top-2 margin | Reference top-2 margin |
|---|---|---|
| `The capital of France is` step 7 | 0.1250 | **0.0000** |
| `水の沸点は` step 6 | **0.0000** | 0.0625 |

Against a median margin of 1.3750 over the agreeing steps. Every observed margin is an integer
multiple of 0.0625, which is one bf16 ULP at logit magnitudes of 8–16 — so a margin of 0.0000 means
the reference's own bf16 logits do not separate the two candidates at all. **The disagreements are
confined to steps where the reference cannot distinguish the tokens either.** That is the bound worth
having, because exact token equality is not achievable here by construction (see Deliberate
deviations).

As a numerical floor for comparison, one Gated DeltaNet layer with the real weights, fp32 against
fp64, gives max_abs 1.8e-08 and rel_p99 2.6e-05 (cosine 1 - 9e-14). The attention layer's floor was
not measured — its `forward` needs precomputed position embeddings and a mask, which this harness does
not build.

## Single-stream latency

`max_num_seqs` is 1 by construction, so there is no batching dimension and these are not serving
throughput numbers. Measured on the same `trn2.3xlarge` at TP=4, bf16, best of three, with nothing
else running on the node.

| Prefill bucket | Prompt lengths landing in it | TTFT | per token |
|---|---|---|---|
| 128 | 100, 128 | 0.159 s | 1.24 ms |
| 512 | 130, 300, 512 | 0.509 s | 0.99 ms |
| 1024 | 520, 700, 1024 | 1.119 s | 1.09 ms |
| 2048 | 1030, 1500, 1900 | 0.377 s | 0.18 ms |

Two things follow. **TTFT tracks the bucket, not the prompt**: the steps fall exactly on the bucket
boundaries, and actual length costs nothing within a bucket. And **cost is not monotonic in bucket
width**: the 2048 bucket is 3x faster than the 1024 bucket and 5.5x more efficient per token. That is
not a "last bucket is special" effect — recompiling with `max_model_len` 1024, so that 1024 is the
last bucket, still gives 1.119 s. Only the 2048 shape lands on the efficient kernel.

Read that as advice and it says something specific: **the 512 and 1024 buckets are worse than the
2048 bucket they exist to avoid.** A request landing in them pays more than if it had been padded all
the way. So the bucket set to use is the ladder with that middle removed.

| Prompt | `[128, 512, 1024, 2048]` | `[128, 2048]` | |
|---|---|---|---|
| 100 | 0.157 s | 0.157 s | unchanged |
| 512 | 0.507 s | 0.373 s | **1.36x** |
| 1024 | 1.115 s | 0.374 s | **2.98x** |

TPOT is 9.2 ms in every cell, so this is TTFT only and costs nothing elsewhere. Back to back at a
512-token prompt and 32 output tokens, that carries through to throughput: **1.512 requests/s against
1.258, and 48.4 output tok/s against 40.2 — 20% either way.** The serving path is not what limits
this: measured request throughput is within 0.6-1.1 ms per request of the ideal implied by TTFT and
TPOT, so scheduling, detokenisation and the Python round trip are together under 0.2% of a request.

Two cautions on the prefill tok/s figure. It differs by 4x depending on what is counted: at a
512-token prompt in a 2048 bucket it is 5461 tok/s counting the bucket and 1365 tok/s counting the
prompt. And there is no batched throughput here at all, since `max_num_seqs` is 1 by construction —
these are single-stream numbers and multiplying them by a concurrency is wrong.

Decode is 9.2 ms per token (about 109 tok/s). **Narrowing the decode context buckets is not worth
anything measurable**: with the prefill bucket, prompt and generation length all held fixed, a
`[128, 512, 1024]` context ladder gives 9.1 ms against 9.2 ms for the single `max_model_len`-wide
bucket. An earlier reading of "about 8%" here was an artefact of comparing runs with different
generation lengths, which biases TPOT because it is derived as `(total - TTFT) / (n - 1)`. Weight
reading dominates decode, and the KV gather is not where the time goes.

Adding the logprobs gather (see below) changed neither number: 0.375 s / 9.2 ms against 0.376 s /
9.2 ms without it.

**The per-bucket cost is a property of the width, not of `max_model_len`.** Recompiled at
`max_model_len` 4096 with buckets `[128, 512, 1024, 2048, 4096]`, the same widths cost the same:

| Bucket | at `max_model_len` 2048 | at 4096 | per token |
|---|---|---|---|
| 128 | 0.157 s | 0.157 s | 1.23 ms |
| 512 | 0.507 s | 0.510 s | 1.00 ms |
| 1024 | 1.115 s | 1.119 s | 1.09 ms |
| 2048 | 0.373 s | 0.375 s | **0.183 ms** |
| 4096 | — | 0.667 s | **0.163 ms** |

So the efficient widths are 128 and everything from 2048 up, and the example keeps exactly those:
`[128, 2048]` at `max_model_len` 2048, `[128, 2048, 4096]` at 4096. Dropping 2048 from the 4096 case
would cost 1.78x on a 1500-token prompt (0.667 s in the 4096 bucket against 0.375 s in the 2048 one),
which is what the first version of this rule did before it was measured at a second `max_model_len`.

Compilation and loading, for planning device time: loading the 72 GB checkpoint from the shared
filesystem cold takes about 16 minutes; compiling one 2048 prefill bucket about 12 minutes; four
prefill buckets plus three decode context buckets about 20 minutes. With a warm compile cache a
rebuild is 2-5 minutes. The cache lives under `~/.cache/vllm/neuron`, not at
`NEURON_COMPILE_CACHE_URL`, so it does not survive a container that keeps only that path.

**A failed compile poisons that cache.** After `QWEN3_5_MOE_GDN_CHUNK=128` failed with
`[NCC_ITRF901] TritiumFusion assertion error: Unexpected remat axes`, every subsequent configuration
failed the same way — including ones that had compiled and run minutes earlier, and including a
`max_model_len` of 512 whose graph is unrelated in size. Removing `~/.cache/vllm/neuron` restored it,
and the reconstructed measurement matched the pre-poisoning one (0.510 s against 0.509 s at a 512
bucket). So: if a configuration that worked stops compiling, suspect the cache before the model, and
run limit-finding experiments after the measurements you actually want.

## Serving through the OpenAI-compatible server

Verified on the same node, not just the offline `LLM` API:

```
vllm serve <checkpoint> --served-model-name qwen36 \
  --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 2048 \
  --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --hf-overrides '{"architectures": ["Qwen3_5MoeForCausalLM"]}' \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [128, 2048]},
                        "vision_neuron_config": {"num_vision_tokens_buckets": [2048],
                                                 "vision_attention_block_size": 2048}}'
```

`/v1/models`, `/v1/completions`, `/v1/chat/completions` (the checkpoint's chat template is picked up)
and streaming all work. `--max-num-seqs 1` is not a tuning choice here — the factory refuses anything
else, so a server started without it fails at construction rather than at the first concurrent request.

## Multi-token verification (the groundwork for MTP)

`gated_delta_net_verify` advances the recurrent and convolution state by a number of tokens that is only
known at runtime. Speculative decoding needs this: the target is handed k proposed tokens, produces an
output for each, and learns the accepted count only after those outputs have been compared — by which
point a one-token-at-a-time state has already passed the rejected ones.

The states it needs are already being computed. Stepping k tokens yields k successive states; with the
incoming state as the zero-accepted case that is k + 1 candidates, and the accepted count selects one as
a sum of `{0, 1}` masks. `k` comes from the tensor's static shape so the loop unrolls at trace time, and
the count stays a tensor, so one traced graph serves every outcome.

Tested for each accepted count from 0 to 3 against the state reached by stepping only that prefix, byte
for byte, with 0 leaving both states untouched; for every proposed token's output against the reference;
and for graph-staticness under `fullgraph=True`.

**Speculative decoding is still refused.** This removes the reason it had to be, not the refusal. The
failure it guards against is silent — a rejected suffix leaves the state ahead of the sequence and every
later token is conditioned on tokens the model never emitted, with fluent output throughout.

What exists on the model side is in `mtp.py`: the head's four modules, its weight map (`layout.py`,
`mtp_checkpoint_mappings`), a KV spec and a cache binding for the one extra full-attention layer, and a
constructor that refuses a layer index inside the main model's range — a draft layer that reused a
decoder layer's index would share that layer's cache with every shape matching. What is missing is not
in this directory: the runner accepts `eagle3` as the only speculative method, so reaching a draft head
at all needs an `mtp` branch and a proposer in a shared file. Refusing it here is therefore redundant
with the runner's own refusal, and stays only so the reason is stated where the head is.

The head also has **no differential reference on CPU**: `transformers` drops `mtp.*` on load and vLLM's
own module cannot be built without a platform attention backend. What is pinned instead is every
component in isolation, the weight map against the real checkpoint index in both directions, and the one
composition step that cannot be inferred — the concatenation feeding `fc` puts the embedding first, which
`ops.concat_draft_inputs` names so a test can hold it. The swapped order loads every weight without
complaint and reads the wrong learned columns.

## Interleaved MRoPE

The rotary path now has both forms. `rotary_tables` builds the single-axis table, exact for text because
a text prompt makes the three MRoPE axes carry the same position. `mrope_tables` builds the real
three-axis form for when they do not, with slot i taking height at `i % 3 == 1` and width at `i % 3 == 2`
inside each axis's section (`[11, 11, 10]`, the default the reference applies when the config omits it,
as this checkpoint does) and time everywhere else.

The test that matters is that the two agree **bit for bit** where the axes agree. Without it, adding
vision would put a second possible cause behind every text disagreement.

## Vision: what is wired, and what a device run still has to show

The checkpoint's vision keys are identical to this repository's Qwen3-VL reference, so the encoder is
reused unchanged and `vision_checkpoint_mappings` is generated by the encoder itself rather than written
out. The map covers all 333 `model.visual.*` keys, diffed against the real checkpoint index in both
directions; the only unmapped source prefix left is `mtp` — 19 keys — which a test asserts.

What `Qwen3_5MoeForConditionalGeneration` adds:

| Piece | Where |
|---|---|
| Vision shape and the four special token ids, which live beside `text_config` rather than in it | `Qwen3_5MoeMultimodalConfig` |
| The six host-side tensors the encoder's `forward` takes, keyed by its parameter names | `vision_inputs.build_vision_inputs` |
| Encoding into the runner's encoder cache, and the encoder-block to cache-block mapping | `embed_multimodal`, `vision_inputs.write_block_ids` |
| Shape-only inputs for warming the encoder graph, from the same builder as the real path | `build_vision_synthetic_inputs` |
| Three-axis MRoPE, replacing the text-only collapse | `get_mrope_input_positions`, `ops.mrope_tables` |
| Merging the encoder cache into the token embeddings at prefill | `Qwen3_5MoeModel.forward` |
| One weight map covering both halves, so the completeness and still-on-meta checks see the encoder | `checkpoint_mappings` |

The host-side pieces are covered by CPU tests that read the contract out of the encoder's own source —
its `forward` parameter names and the dtypes its warmup declares — so a change upstream reaches them.
The three-axis rotary is pinned bit-identical to the single-axis table when the axes agree, which is what
makes a later text-output change attributable to the vision path rather than to the rotary.

**Three things only a device can show, and none of them has run:**

1. that the encoder cache's `input_output_alias` holds across the encoder's NEFF
2. that the encoder graph compiles and one image generates
3. that text output is unchanged with the vision path present

Until those pass, treat image input as unimplemented. `docs/DESIGN-vision-mtp.md` in the project
directory has the sequence.

## Logprobs

Under on-device sampling the LM head returns this rank's vocabulary shard, so the full distribution
only exists across ranks. The model gathers it and returns `(sampled_tokens, gathered_logits)` when
`neuron_config.max_logprobs` is non-zero or logits are being dumped, matching the other models in this
repository. Returning `None` there does not refuse `logprobs=N` — it makes the field come back
**empty**, which reads as "this model has no logprobs" while the request reports success.

One caveat is outside this model: the runner's async-scheduling path returns the sampler output
without logprobs regardless. `async_scheduling=False` is required to actually receive them.

The example was run at `max_model_len` 4096 with the bucket set this rule produces,
`[128, 2048, 4096]`, and generates the same output as at 2048 — so the shipped configuration is the one
that was measured, not an extrapolation from it.

**Not verified:** contexts beyond 4096 (6144 compiles but fails to load with an allocation failure, and
8192 fails inside the compiler), segmented prefill, any concurrency (there is none to have), and every
device-side claim about the vision path (see above). `mypy` is clean on this model's files; it is not
clean elsewhere in this repository.

## Serving a multimodal checkpoint as the text backbone

Both architectures are registered, so serving the text backbone of this checkpoint is a choice rather
than a workaround — and it is the choice that keeps image input out, since the runner decides a model is
multimodal from its architecture name. Two things have to be arranged, and the example `run.py` does
both:

- **Override `architectures`** to `Qwen3_5MoeForCausalLM`.
- **Neutralise the vision buckets** rather than removing the vision config. The runner decides a model
  is multimodal from `hasattr(hf_config, "vision_config")`, which is true for this config class even
  when the checkpoint's JSON omits it; deleting the attribute breaks vLLM's own config for this
  `model_type`, and changing `model_type` makes vLLM reject the config class. So
  `vision_neuron_config` is given `num_vision_tokens_buckets` and `vision_attention_block_size` that
  pass validation against the prefill buckets. Nothing runs through them: no image or video input is
  accepted, and `get_mrope_input_positions` refuses a request carrying multimodal features.
