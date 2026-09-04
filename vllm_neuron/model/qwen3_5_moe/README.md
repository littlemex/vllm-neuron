# Qwen3.5-MoE (text backbone)

Serving implementation of the text backbone of **`Qwen/Qwen3.6-35B-A3B`** — whose HF
`architectures` entry is `Qwen3_5MoeForConditionalGeneration` and whose `model_type` is
`qwen3_5_moe`. The vision tower and the multi-token-prediction head are out of scope; this is the
language model only.

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
- **Recurrent state without a runner state pool.** The conv and recurrent state are carried across
  decode steps by in-place module buffers; the plugin's `AliasingOutputRewritePass` turns the `copy_`
  into an HLO `input_output_alias` so the state persists across the runner's per-step graphs. The
  runner is unmodified, at the cost of batch=1 (`max_num_seqs=1`).
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
| FP8 / NVFP4 | No | bf16 only; the factory rejects other quantizations and the config rejects a non-bf16 checkpoint dtype |
| On-device sampling | Yes | Via `Sampler` when `on_device_sampling_config` is set |
| Vision / video input | No | Text backbone only; a `vision_neuron_config` is rejected rather than silently ignored |
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

- **Batch size 1 only** (`max_num_seqs=1`). The conv and recurrent state are single per-layer buffers
  with no per-slot pool, so concurrent sequences would read each other's state. Both the attention
  decode and the factory raise rather than producing wrong output.
- **Automatic prefix caching is not supported.** Do not set `--enable-prefix-caching`. The attention
  KV is addressable by block hash and would be reused across requests, but the recurrent state has no
  block-hash addressing, so a reused prefix would silently continue from the wrong state. The factory
  rejects it.
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

## Environment variables

| Variable | Default | Effect |
|---|---|---|
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
├── config.py       # Qwen3_5MoeConfig: unwraps text_config, lifts rope_parameters, validates topology
├── factory.py      # implementation selection + serving-option validation (bf16, batch=1, no APC/EP)
├── gdn.py          # the Gated DeltaNet: chunked and one-step scans, the conv carry, and the whole
│                   #   mixer (single source of truth shared with the CPU tests)
├── layout.py       # weight layout transforms and the checkpoint key mapping (both CPU-tested)
├── ops.py          # both RMSNorm conventions, the partial rotary, the decay/beta projections
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

The example's default bucket set follows from this: `[128, 2048]`, not a power-of-two ladder.

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

## Logprobs

Under on-device sampling the LM head returns this rank's vocabulary shard, so the full distribution
only exists across ranks. The model gathers it and returns `(sampled_tokens, gathered_logits)` when
`neuron_config.max_logprobs` is non-zero or logits are being dumped, matching the other models in this
repository. Returning `None` there does not refuse `logprobs=N` — it makes the field come back
**empty**, which reads as "this model has no logprobs" while the request reports success.

One caveat is outside this model: the runner's async-scheduling path returns the sampler output
without logprobs regardless. `async_scheduling=False` is required to actually receive them.

**Not verified:** contexts beyond 2048 (nothing wider was compiled), segmented prefill, any
concurrency (there is none to have), and the vision and MTP paths (out of scope). `mypy` is not clean
on these files, nor anywhere in this repository.

## Serving a multimodal checkpoint as the text backbone

The published checkpoint declares `Qwen3_5MoeForConditionalGeneration`; this model is registered as
`Qwen3_5MoeForCausalLM`, the text architecture. Two things therefore have to be arranged, and the
example `run.py` does both:

- **Override `architectures`** to the text name, since that is what is registered.
- **Neutralise the vision buckets** rather than removing the vision config. The runner decides a model
  is multimodal from `hasattr(hf_config, "vision_config")`, which is true for this config class even
  when the checkpoint's JSON omits it; deleting the attribute breaks vLLM's own config for this
  `model_type`, and changing `model_type` makes vLLM reject the config class. So
  `vision_neuron_config` is given `num_vision_tokens_buckets` and `vision_attention_block_size` that
  pass validation against the prefill buckets. Nothing runs through them: no image or video input is
  accepted, and `get_mrope_input_positions` refuses a request carrying multimodal features.
