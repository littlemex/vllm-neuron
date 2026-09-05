# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-MoE (bf16) -- vLLM-Neuron serving implementation of the text backbone.
=============================================================================
Language backbone of ``Qwen/Qwen3.6-35B-A3B`` (HF ``Qwen3_5MoeForConditionalGeneration``,
``model_type`` ``qwen3_5_moe``): 40 layers alternating three Gated DeltaNet (``linear_attention``)
layers and one gated GQA (``full_attention``) layer, with a 256-expert MoE block on every layer.
~72 GB in bf16, so TP=4 on one trn2 chip; the attention Q heads, the Gated DeltaNet key/value heads
and the MoE expert intermediate are all sharded. The Gated DeltaNet arithmetic is in ``gdn.py``.

The architecture, the feature matrix and the reasoning behind each deviation from the checkpoint are
in README.md next to this file. Two of those deviations are easy to undo here by accident and
expensive to notice: the residual stream is fp32 while the checkpoint accumulates in bf16, and there
is no sequence parallelism, so every mixer must reduce its own partials.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, cast

import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoECTEImplementation,
)
from nkilib.core.utils.common_types import RouterActFnType
from torch import nn
from torch.distributed._functional_collectives import all_gather_tensor
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

import vllm_neuron.functional as NF
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.qwen3_vl.utils.merge_vision_embeds import merge_vision_embeddings
from vllm_neuron.nn import ColumnParallelLinear
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5MoeConfig
from .gdn import (
    DEFAULT_CHUNK_SIZE,
    gated_delta_net_decode,
    gated_delta_net_prefill,
)
from .layout import (
    checkpoint_mappings,
    fuse_attention_qkvg,
    gdn_qkv_rows,
    shard_columns_transposed,
    shard_expert_down,
    shard_expert_gate_up,
    shard_heads,
    shard_rows_transposed,
)
from .ops import (
    apply_partial_rotary,
    chunk_aligned_layout,
    mrope_tables,
    read_state_slot,
    read_state_slots,
    redirect_padded_slots,
    rmsnorm,
    rotary_tables,
    temporal_axis,
    write_state_slot,
    write_state_slots,
)

# Chunk width for the gated-delta-rule prefill scan. 64 is the reference kernel's default; it is a
# tiling choice only (pinned by test_chunk_size_does_not_change_the_result), so it can be retuned for
# compile time without changing numerics.
_DEFAULT_GDN_CHUNK = DEFAULT_CHUNK_SIZE
# MoE blockwise dispatch width, and the point past which loading every expert beats loading the
# selected ones.
_MOE_BLOCK_SIZE = 256
_SELECTIVE_LOADING_THRESHOLD = 1.0

logger = init_logger(__name__)


def attention_metadata_key(layer_idx: int) -> str:
    """The name a full-attention layer is known by in ``attn_metadata`` and in the KV spec.

    One definition: ``get_kv_spec`` declares these names, ``bind_kv_cache`` looks them up, and both
    attention phases and the recurrent layers' phase/pad-mask derivation read them. A second spelling
    anywhere silently produces a KeyError at best and an unbound cache at worst.
    """
    return f"layers.{layer_idx}.self_attn"


def state_metadata_key(layer_idx: int) -> str:
    """The name a Gated DeltaNet layer is known by when its state is pooled by the runner.

    Distinct from the attention key for the same reason that one is defined once: the cache is keyed by
    name, so a collision would let one layer's entry shadow another's.
    """
    return f"layers.{layer_idx}.linear_attn"


def state_pool_requested(vllm_config) -> bool:
    """Whether to ask the runner for a pooled state instead of carrying it in module buffers.

    Opt-in, and deliberately not derived from ``max_num_seqs``. The runner's spec conversion and cache
    allocation understand a state layer, but the metadata builders do not yet hand out a per-request
    slot (see the project's docs/DESIGN-concurrency.md), so a deployment that asked for concurrency and
    got a pool would run with every sequence writing slot 0. Until that is finished the pool has to be
    something a caller asks for on purpose, so the verified single-sequence path stays the default.
    """
    return bool(os.environ.get("QWEN3_5_MOE_STATE_POOL") == "1" and vllm_config is not None)


def _gdn_chunk_size():
    """Prefill chunk width, overridable for experiments via ``QWEN3_5_MOE_GDN_CHUNK``.

    Validated here rather than trusted: 0 divides by zero inside the scan and a negative or absurd
    value fails much later, during tracing, after compile time has already been spent.
    """
    raw = os.environ.get("QWEN3_5_MOE_GDN_CHUNK")
    if raw is None:
        return _DEFAULT_GDN_CHUNK
    try:
        chunk = int(raw)
    except ValueError as error:
        raise ValueError(f"QWEN3_5_MOE_GDN_CHUNK must be an integer; got {raw!r}") from error
    if chunk < 2:
        raise ValueError(f"QWEN3_5_MOE_GDN_CHUNK must be at least 2; got {chunk}")
    return chunk


def resolve_text_neuron_config(neuron_config, text_neuron_config):
    """The one text NeuronConfig, from the two spellings the runner may use.

    Both names reach the model depending on how the architecture was entered, and the factory normally
    resolves which applies before getting here. This exists so the choice has ONE implementation: two
    places picking between the same two arguments is how they end up picking differently, and the symptom
    would be a model built with the wrong bucket lists rather than an error.

    Both supplied and disagreeing is refused rather than ranked. A precedence is a guess about which
    caller was right, and the wrong guess produces a served model with someone else's configuration.
    """
    if (neuron_config is not None and text_neuron_config is not None
            and neuron_config is not text_neuron_config):
        raise ValueError(
            "neuron_config and text_neuron_config were both supplied and are different objects; "
            "which one applies is the caller's to decide, and ranking them here would serve a model "
            "configured by the loser."
        )
    return neuron_config if neuron_config is not None else text_neuron_config


class Qwen3_5MoeRMSNorm(nn.Module):
    """HF ``Qwen3_5MoeRMSNorm``: scales by ``1 + weight``, with weight stored as an offset from unity.

    <-- MODEL-SPECIFIC: the ``1 +`` is not the plugin's usual convention. The checkpoint's norm
    weights are near zero, so a ``weight * x`` norm would collapse the residual stream.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `rmsnorm` comes from an untyped helper module, so its return is Any; naming it here keeps the
        # ambiguity from spreading to every caller of this layer.
        return cast(torch.Tensor, rmsnorm(x, self.weight, self.eps))


# ============================================================
# Full attention: GQA + output gate + partial rotary
# ============================================================
class Qwen3_5MoeAttention(nn.Module):
    """GQA (16 Q / 2 KV, head_dim 256) with three model-specific twists.

    >>> PARALLELISM: TP (Q-head sharding, KV replication when TP > num_kv_heads) <<<

    <-- MODEL-SPECIFIC 1: ``q_proj`` is TWICE as wide as the query. Per head it emits
    ``[query(head_dim) | gate(head_dim)]``, and the attention output is multiplied by
    ``sigmoid(gate)`` before ``o_proj``. Because the packing is per head, sharding by head keeps
    query and gate together and no separate gate projection is needed.

    <-- MODEL-SPECIFIC 2: ``q_norm``/``k_norm`` are RMSNorms over head_dim, applied per head before
    the rotary embedding (Qwen3 family), and they use the ``1 + weight`` convention.

    <-- MODEL-SPECIFIC 3: the rotary embedding is PARTIAL -- ``partial_rotary_factor`` 0.25 rotates
    the leading 64 of 256 channels and passes the remaining 192 through untouched.
    """

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.window_size = None      # <-- MODEL-SPECIFIC: full attention only, no sliding window
        self.eps = config.rms_norm_eps

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads = config.num_attention_heads       # 16
        self.num_key_value_heads = config.num_key_value_heads       # 2
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by "
                f"tensor_parallel_size={self.world_size}."
            )
        # >>> PARALLELISM: GQA head sharding. KV heads either divide the TP size (each rank owns whole
        # KV heads) or are divided by it (a rank group shares one KV head, replicated). Reject other
        # sizes up front rather than reading an out-of-range KV-head slice in the weight loader. <<<
        if not (self.world_size % self.num_key_value_heads == 0
                or self.num_key_value_heads % self.world_size == 0):
            raise ValueError(
                f"tensor_parallel_size={self.world_size} is incompatible with "
                f"num_key_value_heads={self.num_key_value_heads}: one must divide the other."
            )
        # NOTE for anyone switching the decode path to NF.attention_decode: that megakernel also
        # requires (num_attention_heads / tp_size) to be EVEN, and its bias tensors to be 2-D
        # [1, size]. Neither constraint applies to the pure-PyTorch decode used here, so neither is
        # checked — a TP size that passes construction today may not pass with the megakernel.
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        if self.num_attention_heads_per_rank % self.num_key_value_heads_per_rank != 0:
            # Divisibility of the GLOBAL head counts by the TP size is not enough: the per-rank counts
            # must divide too, or `repeat_interleave` cannot expand this rank's KV heads to cover its
            # query heads (e.g. 12 Q / 8 KV at TP=4 gives 3 local Q against 2 local KV).
            raise ValueError(
                f"per-rank query heads ({self.num_attention_heads_per_rank}) must be divisible by "
                f"per-rank KV heads ({self.num_key_value_heads_per_rank}); "
                f"num_attention_heads={self.num_attention_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}, "
                f"tensor_parallel_size={self.world_size} cannot be sharded."
            )
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = self.num_key_value_heads_per_rank * self.head_dim
        # Per-rank fused projection: [query | gate | key | value]. Query and gate are separated here
        # (they are interleaved per head in the checkpoint) so the forward needs no per-head regroup.
        fused_size = 2 * self.q_size + 2 * self.kv_size
        self.qkvg_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, fused_size, dtype=self.dtype))
        self.qkvg_split_indices = [self.q_size, 2 * self.q_size, 2 * self.q_size + self.kv_size]
        self.o_proj_weight = nn.Parameter(torch.empty(self.q_size, self.hidden_size, dtype=self.dtype))
        self.q_norm_weight = nn.Parameter(torch.zeros(self.head_dim, dtype=self.dtype))
        self.k_norm_weight = nn.Parameter(torch.zeros(self.head_dim, dtype=self.dtype))

        # Optional only until the runner calls bind_kv_cache. Reads go through the properties below,
        # which fail with the reason rather than with an AttributeError on None.
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        self._setup_weight_loaders()

    @property
    def k_cache(self) -> torch.Tensor:
        """The key cache the runner bound to this layer.

        A property rather than a plain attribute so that using the layer before ``bind_kv_cache`` says
        so. Left as ``Tensor | None``, every read site would have to narrow it, and the narrowing would
        be noise at forty call sites for a condition that is really a lifecycle fact.
        """
        if self._k_cache is None:
            raise RuntimeError(
                f"layer {self.layer_idx} has no key cache; the runner must call bind_kv_cache "
                "before the first forward"
            )
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        """The value cache the runner bound to this layer. See ``k_cache``."""
        if self._v_cache is None:
            raise RuntimeError(
                f"layer {self.layer_idx} has no value cache; the runner must call bind_kv_cache "
                "before the first forward"
            )
        return self._v_cache

    def bind_caches(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        """Give this layer the caches the runner allocated for it.

        The dtype is checked because the runner does NOT allocate the dtype this model declared: it
        overrides it with ``cache_config.cache_dtype``. The factory refuses a cache dtype this attention
        cannot read, and this is the destination-side half of that -- the check that fires if the refusal
        is ever bypassed or the runner's override changes.
        """
        for label, cache in (("K", k_cache), ("V", v_cache)):
            if cache.dtype != self.dtype:
                raise RuntimeError(
                    f"Qwen3.5-MoE on Neuron, layer {self.layer_idx}: the {label} cache was allocated "
                    f"as {cache.dtype} but this attention reads it as {self.dtype}. The runner takes "
                    "the cache dtype from cache_config, not from the model's KV spec."
                )
        self._k_cache = k_cache
        self._v_cache = v_cache

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: Q sharded by head, KV sliced per rank group (replicated when TP > KV) <<<

        The layout arithmetic lives in ``layout.py`` so it can be pinned on CPU against the HF
        reference; see ``fuse_attention_qkvg`` for the head-major query/gate packing.
        """
        num_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads
        head_dim = self.head_dim
        world_size = self.world_size

        set_weight_loader(self.qkvg_proj_weight, SafetensorsWeightLoader(
            transform=lambda slices, rank: fuse_attention_qkvg(
                slices[0][:], slices[1][:], slices[2][:], rank, world_size,
                num_heads, num_kv_heads, head_dim)))
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.q_size,
                num_shards=world_size, is_storage_transposed=True,
            ),
        )
        # Head-dim norms are replicated: every rank normalises its own heads with the same weight.
        for param in (self.q_norm_weight, self.k_norm_weight):
            set_weight_loader(param, SafetensorsWeightLoader(transform=lambda s, r: s[0][:]))

    def _project(self, hidden_states, cos, sin):
        """Project, normalise per head, rotate partially. Returns q/k/v as ``[heads, T, head_dim]``
        and the gate as ``[T, q_size]``.

        <-- MODEL-SPECIFIC: a plain matmul instead of ``NF.qkv_proj`` -- that kernel's fused rotary is
        delimited by ``num_kv_heads`` and would rotate the wrong channels of a q_proj that also
        carries the output gate, and it applies full-width rotary where this model needs partial.
        """
        tokens = hidden_states.shape[0]
        fused = hidden_states @ self.qkvg_proj_weight
        query, gate, key, value = torch.tensor_split(fused, self.qkvg_split_indices, dim=-1)

        query = query.view(tokens, self.num_attention_heads_per_rank, self.head_dim)
        key = key.view(tokens, self.num_key_value_heads_per_rank, self.head_dim)
        value = value.view(tokens, self.num_key_value_heads_per_rank, self.head_dim)
        # Per-head RMSNorm before the rotary embedding (Qwen3 family), then partial rotary. cos/sin
        # are [T, rotary_dim]; unsqueeze the head axis so they broadcast over heads.
        query = rmsnorm(query, self.q_norm_weight, self.eps)
        key = rmsnorm(key, self.k_norm_weight, self.eps)
        query = apply_partial_rotary(query, cos.unsqueeze(1), sin.unsqueeze(1))
        key = apply_partial_rotary(key, cos.unsqueeze(1), sin.unsqueeze(1))
        return (query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
                gate.reshape(tokens, self.q_size))

    def forward(self, hidden_states, positions, cos, sin, attn_metadata, is_prefill):
        if is_prefill:
            return self._prefill(hidden_states, cos, sin, attn_metadata)
        return self._decode(hidden_states, positions, cos, sin, attn_metadata)

    def _gate_and_project_out(self, attn_output, gate):
        """Apply the output gate and ``o_proj``. ``attn_output`` is ``[T, q_size]``.

        <-- MODEL-SPECIFIC: the gate. It is applied in the token-major layout, which is also why
        ``o_proj`` is a plain matmul here rather than ``NF.o_proj`` (that helper consumes the
        head-major ``[1, heads, head_dim, T]`` layout the attention kernels emit).
        """
        gated = attn_output.to(self.dtype) * torch.sigmoid(gate.to(self.dtype))
        output = gated @ self.o_proj_weight
        if self.world_size > 1:
            # >>> PARALLELISM: each rank holds a slice of o_proj's input, so the partial products sum
            # across ranks. No SP here, so it is an all-reduce in both phases. <<<
            output = self.tp_group.all_reduce(output)
        return output.contiguous()

    def _prefill(self, hidden_states, cos, sin, attn_metadata):
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        metadata = attn_metadata[attention_metadata_key(self.layer_idx)]
        query, key, value, gate = self._project(hidden_states, cos, sin)

        block_size = metadata["block_size"]
        self._write_kv_cache(key, value, metadata["slot_mapping"], block_size)

        kv_segment_size = metadata.get("kv_segment_size")
        if kv_segment_size:
            # Segmented / continuation prefill: this segment's queries attend to every prior
            # segment's KV in the paged cache plus this segment's own. GQA is handled inside the
            # kernel, so key/value keep their KV-head count.
            attn_output = NF.segmented_attention(
                query, k_cache=self.k_cache, v_cache=self.v_cache,
                block_tables=metadata["block_table_tensor"],
                prior_tokens=metadata.get("cached_seq_len"),
                block_size=block_size, kv_segment_size=kv_segment_size,
                scale=self.scaling, tp_q=True, tp_out=True,
                sliding_window=None, sink=None, fp8_packed=False,
            )                                                       # [heads, head_dim, T]
        else:
            key = key.repeat_interleave(self.num_key_value_groups, dim=0)
            value = value.repeat_interleave(self.num_key_value_groups, dim=0)
            # The kernel's annotation covers several paths, one of which returns a tuple. This call
            # is the plain path, so the type is named rather than narrowed at every later use.
            attn_output = cast(torch.Tensor, NF.flash_attention(
                query.transpose(1, 2), key.transpose(1, 2), value,
                scale=self.scaling, causal_mask=True, tp_q=False, tp_out=True,
            ))                                                      # [heads, head_dim, T]
        # [heads, head_dim, T] -> [T, heads * head_dim] so the gate lines up with q_proj's packing.
        attn_output = attn_output.permute(2, 0, 1).reshape(-1, self.q_size)
        return self._gate_and_project_out(attn_output, gate)

    def _decode(self, hidden_states, positions, cos, sin, attn_metadata):
        """Pure-PyTorch fp32 decode.

        ``NF.flash_attention`` does not handle the asymmetric ``q=1`` / ``k=S_ctx`` decode shape, and
        fp32 also removes bf16 accumulation drift over a long context. With batch=1 and 10 attention
        layers the cost is negligible.
        """
        metadata = attn_metadata[attention_metadata_key(self.layer_idx)]
        block_size = metadata["block_size"]
        max_blocks_per_seq = metadata["max_blocks_per_seq"]
        block_table = metadata["block_table_tensor"]

        batch = block_table.shape[0]
        tokens = hidden_states.shape[0]
        if batch != 1:
            raise NotImplementedError(
                "Qwen3.5-MoE serving supports max_num_seqs=1 only: the Gated DeltaNet conv and "
                "recurrent state are single per-layer buffers (no per-slot pool), so concurrent "
                f"sequences would corrupt each other. Got batch size {batch}."
            )
        decode_tokens = tokens // batch
        hidden_states = hidden_states.to(self.dtype)
        context_len = max_blocks_per_seq * block_size
        kv_heads = self.num_key_value_heads_per_rank

        query, key, value, gate = self._project(hidden_states, cos, sin)
        self._write_kv_cache(key, value, metadata["slot_mapping"], block_size)

        # Block tables are padded to the bucket width with a sentinel; clamp before gathering. A -1
        # would wrap to the last block and a value past the end is an out-of-bounds device gather. The
        # causal mask discards those positions anyway, so the clamped read is harmless.
        flat_blocks = block_table.reshape(-1).clamp(0, self.k_cache.shape[0] - 1)
        key_dense = (self.k_cache[flat_blocks]
                     .view(batch, max_blocks_per_seq, kv_heads, block_size, self.head_dim)
                     .permute(0, 2, 1, 3, 4).reshape(batch * kv_heads, context_len, self.head_dim))
        value_dense = (self.v_cache[flat_blocks]
                       .view(batch, max_blocks_per_seq, kv_heads, block_size, self.head_dim)
                       .permute(0, 2, 1, 3, 4).reshape(batch * kv_heads, context_len, self.head_dim))
        key_full = key_dense.repeat_interleave(self.num_key_value_groups, dim=0).to(torch.float32)
        value_full = value_dense.repeat_interleave(self.num_key_value_groups, dim=0).to(torch.float32)

        scores = torch.matmul(query.to(torch.float32),
                              key_full.transpose(-2, -1)) * self.scaling
        # Per-query causal mask over the padded context. Each decode row attends only up to its own
        # absolute position; masking with the last position alone would let earlier rows of a decode
        # bucket read future keys.
        context_index = torch.arange(context_len, device=scores.device)
        valid = context_index.view(1, -1) <= positions.view(-1, 1)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(weights, value_full)             # [heads, decode_tokens, head_dim]

        attn_output = attn_output.transpose(0, 1).reshape(-1, self.q_size)
        return self._gate_and_project_out(attn_output, gate)

    def _write_kv_cache(self, key, value, slot_mapping, block_size):
        """Scatter K/V into the paged cache at ``slot_mapping``, neutralising the padded rows.

        The padding sentinel handling lives in ``ops.redirect_padded_slots`` (which documents why a
        reserved sink slot is not a valid approach) and is CPU-tested there.
        """
        head_dim = self.head_dim
        kv_heads = self.num_key_value_heads_per_rank
        num_slots = slot_mapping.shape[0]
        # Token-major rows so a pad row can take a whole real token's K/V in one `where`.
        key_rows = key.transpose(0, 1).reshape(num_slots, kv_heads * head_dim)
        value_rows = value.transpose(0, 1).reshape(num_slots, kv_heads * head_dim)
        safe_slot, key_rows = redirect_padded_slots(slot_mapping, key_rows)
        _, value_rows = redirect_padded_slots(slot_mapping, value_rows)

        block_index = (safe_slot // block_size).repeat(kv_heads)
        position_index = (safe_slot % block_size).repeat(kv_heads)
        head_index = torch.arange(kv_heads, dtype=torch.long,
                                  device=key.device).repeat_interleave(num_slots)
        index = (block_index, head_index, position_index)
        # Back to head-major [heads * tokens, head_dim] to match the index triple's ordering.
        key_flat = key_rows.view(num_slots, kv_heads, head_dim).transpose(0, 1).reshape(-1, head_dim)
        value_flat = value_rows.view(num_slots, kv_heads, head_dim).transpose(0, 1).reshape(-1, head_dim)
        self.k_cache.index_put_(index, key_flat.to(self.k_cache.dtype))
        self.v_cache.index_put_(index, value_flat.to(self.v_cache.dtype))


# ============================================================
# MoE: 256 routed experts (top-8) + one sigmoid-gated shared expert
# ============================================================
class Qwen3_5MoeMoE(nn.Module):
    """Routed experts on the plugin's NKI MoE path, plus a dense shared expert.

    >>> PARALLELISM: TP (expert intermediate sharded; every rank holds all 256 experts, and the
    factory refuses a deployment that asks for expert parallelism) <<<

    The kernel sequence follows gpt_oss, the canonical shape for this path; diverging needs a reason.
    Routing is ``softmax`` over ALL experts in fp32, then top-k, then an L1 renormalisation, which the
    kernels express as ``PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER`` in prefill and
    ``router_pre_norm=True`` with ``norm_topk_prob=True`` in decode.

    <-- MODEL-SPECIFIC: ``moe_block_tkg`` can host a shared expert but has no hook for this model's
    ``sigmoid(shared_expert_gate(x))`` scaling, so the shared expert is a dense SwiGLU MLP here.
    """

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.eps = config.rms_norm_eps
        self.block_size = _MOE_BLOCK_SIZE

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        intermediate = config.moe_intermediate_size
        shared_intermediate = config.shared_expert_intermediate_size
        if intermediate % self.world_size != 0 or shared_intermediate % self.world_size != 0:
            raise ValueError(
                f"moe_intermediate_size={intermediate} and shared_expert_intermediate_size="
                f"{shared_intermediate} must both be divisible by tensor_parallel_size="
                f"{self.world_size}."
            )
        self.intermediate_per_rank = intermediate // self.world_size
        self.shared_intermediate_per_rank = shared_intermediate // self.world_size

        # The MoE kernels shard the hidden dim 128-ways per LNC core and the intermediate dim 128-ways;
        # check here so a bad TP size fails at construction instead of inside a kernel.
        if self.hidden_size % 256 != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by 256 for the MoE kernels "
                "(H_par * n_prgs = 128 * 2 under LNC-2)."
            )
        if self.intermediate_per_rank % 128 != 0:
            raise ValueError(
                f"moe_intermediate_size / tensor_parallel_size = {self.intermediate_per_rank} must be "
                "a multiple of 128 for the MoE kernels."
            )

        # Router: replicated (small). Kept in the model dtype rather than promoted to fp32.
        # DECISION, not an oversight: the general advice for MoE ports is to hold the router weight in
        # fp32, because a bf16 rounding difference in a softmax over many experts can select a
        # different expert. The reference computes the logits from a bf16 weight, and this port's
        # acceptance test is agreement with the reference, so matching it wins over being more stable
        # than it. The prefill router still accumulates in fp32 (``computation_dtype``). Promoting the
        # parameter would be a deliberate divergence from the reference, not a free fix.
        self.router_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size,
                                                      dtype=self.dtype))
        # The kernels want gate/up fused as [E, hidden, 2, intermediate] and down as
        # [E, intermediate, hidden].
        self.gate_up_proj_weight = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_size, 2, self.intermediate_per_rank, dtype=self.dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(
            self.num_experts, self.intermediate_per_rank, self.hidden_size, dtype=self.dtype))
        # Shared expert: separate gate/up in the checkpoint, kept separate here.
        self.shared_gate_proj_weight = nn.Parameter(torch.empty(
            self.hidden_size, self.shared_intermediate_per_rank, dtype=self.dtype))
        self.shared_up_proj_weight = nn.Parameter(torch.empty(
            self.hidden_size, self.shared_intermediate_per_rank, dtype=self.dtype))
        self.shared_down_proj_weight = nn.Parameter(torch.empty(
            self.shared_intermediate_per_rank, self.hidden_size, dtype=self.dtype))
        self.shared_expert_gate_weight = nn.Parameter(torch.empty(
            self.hidden_size, 1, dtype=self.dtype))
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: shard the expert and shared-expert intermediate dimension <<<

        Layout arithmetic lives in ``layout.py`` (pinned on CPU against the HF reference experts):
        the fused ``gate_up_proj`` stores gate rows first and must be split, not reshaped, and
        ``down_proj`` is stored ``[out, in]`` per expert where the kernels want ``[in, out]``.
        """
        world_size = self.world_size

        set_weight_loader(self.router_weight,
                          SafetensorsWeightLoader(transform=lambda s, r: s[0][:].contiguous()))
        set_weight_loader(self.gate_up_proj_weight, SafetensorsWeightLoader(
            transform=lambda s, r: shard_expert_gate_up(s[0][:], r, world_size)))
        set_weight_loader(self.down_proj_weight, SafetensorsWeightLoader(
            transform=lambda s, r: shard_expert_down(s[0][:], r, world_size)))
        for param in (self.shared_gate_proj_weight, self.shared_up_proj_weight):
            set_weight_loader(param, SafetensorsWeightLoader(
                transform=lambda s, r: shard_rows_transposed(s[0][:], r, world_size)))
        set_weight_loader(self.shared_down_proj_weight, SafetensorsWeightLoader(
            transform=lambda s, r: shard_columns_transposed(s[0][:], r, world_size)))
        # Replicated: [1, hidden] -> [hidden, 1]. Its output is a per-token scalar, so sharding it
        # would need an extra reduction for no benefit.
        set_weight_loader(self.shared_expert_gate_weight,
                          SafetensorsWeightLoader(transform=lambda s, r: s[0][:].T.contiguous()))

    def _shared_expert(self, normed_hidden):
        """SwiGLU shared expert, scaled by ``sigmoid(shared_expert_gate(x))``.

        <-- MODEL-SPECIFIC: the sigmoid gate. Its input is the same normalised hidden states the
        routed experts see, so the two halves of the block cannot drift apart.
        """
        gate = normed_hidden @ self.shared_gate_proj_weight
        up = normed_hidden @ self.shared_up_proj_weight
        out = (F.silu(gate) * up) @ self.shared_down_proj_weight
        # The gate weight is replicated, so every rank computes the same scalar; scaling here rather
        # than after the cross-rank reduction keeps the routed and shared partials additive.
        return out * torch.sigmoid(normed_hidden @ self.shared_expert_gate_weight)

    def forward(self, hidden_states, norm_weight, is_prefill, valid_mask=None):
        """``hidden_states`` are the RAW (un-normalised) residual values.

        The block's pre-MoE norm is applied here once, and the normalised states feed the router, the
        routed experts and the shared expert alike. The one exception is the decode kernel, which
        fuses the norm and offers no way to skip it: it therefore takes the raw states plus the norm
        weight as ``gamma``, handed over as ``1 + weight`` to match this model's norm convention.

        The residual arrives in fp32 and is cast to the model dtype FIRST, so the decode kernel's
        internal norm and the norm computed here see the same tensor -- norming one from bf16 and the
        other from fp32 would silently give the two halves of the block different inputs. The cost is
        rounding before the normalisation rather than after, which is below bf16's own resolution.
        """
        hidden_states = hidden_states.to(self.dtype)
        normed = rmsnorm(hidden_states, norm_weight, self.eps).to(self.dtype)
        shared = self._shared_expert(normed)

        if is_prefill:
            routed = self._prefill_routed(normed, valid_mask)
        else:
            routed = self._decode_routed(hidden_states, 1.0 + norm_weight.float())
        output = routed.reshape(shared.shape) + shared
        if self.world_size > 1:
            # >>> PARALLELISM: both halves are computed on a sharded intermediate, so the partials sum
            # across ranks. <<<
            output = self.tp_group.all_reduce(output)
        return output

    def _prefill_routed(self, normed, valid_mask):
        """Both the router and ``NF.moe_cte`` take the already-normalised states here."""
        # No ``gamma``, so the router does not normalise the already-normalised states again.
        expert_affinities = NF.router(
            hidden_states=normed,
            router_weights=self.router_weight.T,
            top_k=self.top_k,
            activation="softmax",
            # <-- MODEL-SPECIFIC: activate ALL logits, then top-k, then L1 renormalise. This is the
            # order Qwen3.5 uses; the default order (top-k then activate) is mathematically the same
            # for softmax but naming the real one keeps the code checkable against the reference.
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
            computation_dtype=torch.float32,
        )
        (expert_affinities_masked, token_position_to_id, block_to_expert,
         conditions) = NF.build_blockwise_mapping(
            # The router returns a tuple when router logits are requested; only the affinities are
            # wanted here, and the call site above has already normalised it.
            expert_affinities=cast(torch.Tensor, expert_affinities),
            num_local_experts=self.num_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.tp_group,
            tp_degree=self.world_size,
            # NOTE the polarity: build_blockwise_mapping documents padding_mask as True = REAL token,
            # not True = padding. `slot_mapping >= 0` is therefore the right sense; inverting it would
            # zero the real tokens' affinities and leave only the shared expert, which still produces
            # coherent-looking output.
            # Zeroes the pad tokens' expert affinities so they contribute nothing to the result. It
            # does NOT shrink the dispatch schedule: that is built from the unmasked affinity pattern
            # and is sized by the bucket width, so a mostly-padded bucket costs the same as a full one.
            # Reducing that cost is a bucketing decision, not something this mask can do.
            padding_mask=valid_mask,
        )
        return NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=normed,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

    def _decode_routed(self, hidden_states, gamma):
        tokens = hidden_states.shape[0]
        # Below the threshold it is cheaper to load only the selected experts, which with 256 experts
        # and a single decode token is always the case here.
        use_all_experts = (tokens * self.top_k / self.num_experts) >= _SELECTIVE_LOADING_THRESHOLD
        rank_id = None
        if use_all_experts:
            rank_id = torch.tensor([[0]], dtype=torch.int32, device=hidden_states.device)
        output = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=gamma.unsqueeze(0),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            # <-- MODEL-SPECIFIC: activate before top-k, then L1 renormalise the selected weights.
            router_pre_norm=True,
            norm_topk_prob=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
            is_all_expert=use_all_experts,
            skip_router_logits=True,
        )
        # The kernel returns a tuple when router logits are included; skip_router_logits=True still
        # returns a 1-tuple on some paths, so normalise here rather than at the call sites. Bound to a
        # new name so neither name has to carry both types.
        tensor_output = output[0] if isinstance(output, tuple) else output
        return cast(torch.Tensor, tensor_output)


# ============================================================
# Gated DeltaNet (stateful): chunked prefill + one-step decode
# ============================================================
class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """The ``linear_attention`` mixer. Conv and recurrent state live in module buffers updated in
    place; the plugin's ``AliasingOutputRewritePass`` rewrites the ``copy_`` into an HLO
    ``input_output_alias`` so the state survives between the runner's per-step graph calls, with no
    runner-side state pool.

    The state carries a leading SLOT axis and every read and write goes through it, so the mixer itself
    does not impose batch=1 -- the runner does. Its ``LayerSpec`` cannot describe a recurrent state, so
    there is no pool to allocate and nothing hands out slots; ``num_state_slots`` is therefore 1 in
    every shipped configuration, which makes the slot arithmetic the identity (see ``ops``). The
    runner-side work is scoped in the project's ``docs/DESIGN-concurrency.md``.

    >>> PARALLELISM: TP (key/value head sharding; the depthwise conv shards with its channels) <<<

    <-- MODEL-SPECIFIC: the whole mixer is new for this architecture. Sharding is clean because the
    value heads are a whole multiple of the key heads: with TP=4 a rank owns key heads
    ``[4r, 4r+4)`` and value heads ``[8r, 8r+8)``, and the reference's ``repeat_interleave(2)`` maps
    key head ``j`` to value heads ``2j, 2j+1`` — so a rank's key heads expand to exactly its own
    value heads and no cross-rank exchange is needed inside the mixer.
    """

    # Registered buffers, annotated because nn.Module.__getattr__ hides their type. The recurrent state
    # is fp32 regardless of the model dtype: it accumulates over the whole prefix.
    recurrent_state: torch.Tensor
    conv_state: torch.Tensor

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_k_heads = config.linear_num_key_heads          # 16
        self.num_v_heads = config.linear_num_value_heads        # 32
        self.head_k_dim = config.linear_key_head_dim            # 128
        self.head_v_dim = config.linear_value_head_dim          # 128
        self.conv_kernel_size = config.linear_conv_kernel_dim   # 4
        self.heads_per_k_head = self.num_v_heads // self.num_k_heads
        # Resolved once: this is a compile-time tiling choice, not a per-step knob.
        self.chunk_size = _gdn_chunk_size()

        for name, count in (("linear_num_key_heads", self.num_k_heads),
                            ("linear_num_value_heads", self.num_v_heads)):
            if count % self.world_size != 0:
                raise ValueError(
                    f"{name}={count} must be divisible by tensor_parallel_size={self.world_size}."
                )
        self.k_heads_pr = self.num_k_heads // self.world_size
        self.v_heads_pr = self.num_v_heads // self.world_size
        self.key_dim_pr = self.k_heads_pr * self.head_k_dim         # 512 at TP=4
        self.value_dim_pr = self.v_heads_pr * self.head_v_dim       # 1024 at TP=4
        self.conv_dim_pr = 2 * self.key_dim_pr + self.value_dim_pr  # 2048 at TP=4

        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.conv_dim_pr, dtype=self.dtype))
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.value_dim_pr, dtype=self.dtype))
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.v_heads_pr, dtype=self.dtype))
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.v_heads_pr, dtype=self.dtype))
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim_pr, 1, self.conv_kernel_size, dtype=self.dtype))
        self.A_log = nn.Parameter(torch.zeros(self.v_heads_pr, dtype=self.dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.v_heads_pr, dtype=self.dtype))
        # The gated norm is per value head (width head_v_dim), so it is replicated across ranks.
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.value_dim_pr, self.hidden_size, dtype=self.dtype))

        # Recurrent state, carried across graphs by in-place update + the aliasing pass. fp32 because
        # the checkpoint declares mamba_ssm_dtype float32 and the recurrence compounds over the whole
        # sequence. Materialised on the device in load_weights (the runner builds on `meta`).
        #
        # The leading axis is the SLOT: one entry per concurrent sequence. It was already there with
        # length one, so widening it is a shape change rather than a redesign -- but the slot has to be
        # selected without a data-dependent index, which is what `_slot_mask` is for. With num_slots == 1
        # every mask is the constant one and the arithmetic reduces to today's, bit for bit.
        self.num_slots = config.num_state_slots
        self.register_buffer(
            "recurrent_state",
            torch.zeros(self.num_slots, self.v_heads_pr, self.head_k_dim, self.head_v_dim,
                        dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "conv_state",
            torch.zeros(self.num_slots, self.conv_dim_pr, self.conv_kernel_size - 1,
                        dtype=self.dtype),
            persistent=False)
        self._setup_weight_loaders()

    def _read_state(self, state: torch.Tensor, slot) -> torch.Tensor:
        """The named slot(s)' state, as ``[requests, ...]``.

        One slot and many slots take different code paths because the single-slot form has to stay the
        identity when there is one slot -- that is what keeps the shipped configuration's graph
        unchanged. The arithmetic is in ``ops`` so the tests can reach it without a TP group.
        """
        if slot is None or torch.as_tensor(slot).numel() == 1:
            return read_state_slot(state, slot, self.num_slots)
        return read_state_slots(state, slot, self.num_slots)

    def _write_state(self, state: torch.Tensor, updated: torch.Tensor, slot) -> None:
        """Write the named slot(s), leaving the others untouched."""
        if slot is None or torch.as_tensor(slot).numel() == 1:
            write_state_slot(state, updated, slot, self.num_slots)
            return
        write_state_slots(state, updated, slot, self.num_slots)

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: slice every projection by head <<<

        Layout arithmetic lives in ``layout.py``; ``gdn_qkv_rows`` documents why a rank's key-head
        slice expands to exactly its own value-head slice.
        """
        world_size = self.world_size
        num_k_heads, num_v_heads = self.num_k_heads, self.num_v_heads
        head_k_dim, head_v_dim = self.head_k_dim, self.head_v_dim

        def _stacked_rows(slices, rank):
            return gdn_qkv_rows(slices[0][:], rank, world_size, num_k_heads, num_v_heads,
                                head_k_dim, head_v_dim)

        set_weight_loader(self.in_proj_qkv_weight, SafetensorsWeightLoader(
            transform=lambda s, r: _stacked_rows(s, r).T.contiguous()))
        set_weight_loader(self.conv1d_weight, SafetensorsWeightLoader(
            transform=lambda s, r: _stacked_rows(s, r).contiguous()))
        for param in (self.in_proj_z_weight, self.in_proj_b_weight, self.in_proj_a_weight):
            set_weight_loader(param, SafetensorsWeightLoader(
                transform=lambda s, r: shard_rows_transposed(s[0][:], r, world_size)))
        for param in (self.A_log, self.dt_bias):
            set_weight_loader(param, SafetensorsWeightLoader(
                transform=lambda s, r: shard_heads(s[0][:], r, world_size)))
        # The gated norm is per value head (width head_v_dim), so it is replicated across ranks.
        set_weight_loader(self.norm_weight,
                          SafetensorsWeightLoader(transform=lambda s, r: s[0][:].contiguous()))
        set_weight_loader(self.out_proj_weight, sharding_weight_loader(
            shard_dim=0, shard_size=self.value_dim_pr,
            num_shards=world_size, is_storage_transposed=True))

    # -- shared pieces of both phases ---------------------------------------
    def _weights(self):
        """The per-rank weights, as the plain mapping ``gdn.py`` consumes.

        The mixer holds parameters and TP sharding; the arithmetic lives in ``gdn.py`` so the CPU test
        can pin the whole mixer against the HF reference without importing any Neuron-only symbol.
        """
        return {
            "in_proj_qkv": self.in_proj_qkv_weight,
            "in_proj_z": self.in_proj_z_weight,
            "in_proj_b": self.in_proj_b_weight,
            "in_proj_a": self.in_proj_a_weight,
            "conv1d": self.conv1d_weight,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "norm": self.norm_weight,
            "out_proj": self.out_proj_weight,
        }

    def _dims(self):
        return {
            "k_heads": self.k_heads_pr,
            "v_heads": self.v_heads_pr,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "kernel": self.conv_kernel_size,
        }

    def _reduce(self, output):
        """>>> PARALLELISM: sum the row-parallel out_proj partials across ranks <<<

        ``out_proj`` is sharded over its INPUT dimension, so each rank computes a partial sum over its
        own value heads and the true output is the sum of those partials. TRAP: dropping this
        collective is invisible in a TP=1 run and in every single-rank test — each of the 30 Gated
        DeltaNet layers would put one rank's partial into the residual instead of the sum, which is
        not a scaled version of the right answer but a different vector, and the model would emit
        fluent-looking nonsense. The result is captured rather than assumed to be mutated in place,
        which is correct under either convention.
        """
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.contiguous()

    def forward_prefill(self, hidden_states, cached_seq_len=None, valid_mask=None, slot=None,
                        packed=None):
        """Chunked gated-delta-rule prefill; the scan and the mask contract are in ``gdn.py``.

        ``cached_seq_len`` is non-None only in segmented / continuation prefill, and is the number of
        tokens already processed. Whether this is the first segment is decided by tensor arithmetic on
        it, not by a Python branch on a runtime value, so the graph stays static either way.
        """
        hidden_states = hidden_states.to(self.dtype)
        if cached_seq_len is None:
            conv_state = None
            is_continuation = None
            initial_state = None
        else:
            is_continuation = (cached_seq_len.reshape(()) > 0).to(self.dtype)
            conv_state = self._read_state(self.conv_state, slot)
            initial_state = (self._read_state(self.recurrent_state, slot)
                             * is_continuation.to(self.recurrent_state.dtype))

        output, state, new_conv_state = gated_delta_net_prefill(
            hidden_states, self._weights(), self._dims(), self.eps,
            chunk_size=self.chunk_size, conv_state=conv_state,
            is_continuation=is_continuation, valid_mask=valid_mask,
            initial_state=initial_state,
        )
        # In place -> aliased by the FX pass, so the next graph call sees this state.
        self._write_state(self.recurrent_state, state, slot)
        self._write_state(self.conv_state, new_conv_state, slot)
        return self._reduce(output)

    def forward_decode(self, hidden_states, is_continuation, slot=None):
        """One-token step. Raises rather than mis-generating if handed more than one token, which is
        what makes speculative decoding fail loudly instead of silently advancing the state once.

        ``is_continuation`` is a runtime {0,1} scalar that is 0 when this token is the first of its
        sequence. A one-token prompt is indistinguishable from a decode step by token count, so this
        is what stops such a request from continuing from the previous request's state.
        """
        tokens = hidden_states.shape[1]
        if tokens != 1:
            raise NotImplementedError(
                "Qwen3.5-MoE Gated DeltaNet decode advances the recurrent state by exactly one "
                f"token; got {tokens}. Speculative decoding is not supported."
            )
        requests = hidden_states.shape[0]
        if slot is not None and torch.as_tensor(slot).numel() != requests:
            raise ValueError(
                f"got {torch.as_tensor(slot).numel()} slot(s) for {requests} request(s); each request "
                "advances its own state, so the counts must agree."
            )
        output, state, new_conv_state = gated_delta_net_decode(
            hidden_states.to(self.dtype), self._weights(), self._dims(), self.eps,
            conv_state=self._read_state(self.conv_state, slot),
            recurrent_state=self._read_state(self.recurrent_state, slot),
            is_continuation=is_continuation,
        )
        self._write_state(self.recurrent_state, state, slot)
        self._write_state(self.conv_state, new_conv_state, slot)
        return self._reduce(output)


# ============================================================
# Decoder layer / model / ForCausalLM
# ============================================================
class Qwen3_5MoeDecoderLayer(nn.Module):
    """One layer: a token mixer chosen by ``layer_types``, then the MoE block. Both are residual.

    <-- MODEL-SPECIFIC: the per-layer mixer dispatch. Unlike the attention-only reference models, the
    mixer is either a Gated DeltaNet or a gated GQA; the MoE half is identical for both.
    """

    # nn.Module.__getattr__ is typed `Tensor | Module`, so without these a type checker cannot tell a
    # submodule from a buffer and every attribute read here becomes ambiguous. They also say what the
    # layer holds, which the dispatch on layer_type otherwise leaves implicit.
    input_layernorm: Qwen3_5MoeRMSNorm
    post_attention_layernorm: Qwen3_5MoeRMSNorm
    mlp: Qwen3_5MoeMoE
    self_attn: Qwen3_5MoeAttention
    linear_attn: Qwen3_5MoeGatedDeltaNet

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        # The MoE kernels fuse this norm, so the layer holds the weight and hands it over rather than
        # applying it: see Qwen3_5MoeMoE.forward.
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == FULL_ATTENTION:
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        else:
            raise ValueError(f"unknown layer type {self.layer_type!r} at index {layer_idx}")
        self.mlp = Qwen3_5MoeMoE(config, layer_idx)


class Qwen3_5MoeModel(nn.Module):
    """Embedding + 40 hybrid layers + final norm."""

    layers: nn.ModuleList
    norm: Qwen3_5MoeRMSNorm
    rotary_cos: torch.Tensor
    rotary_sin: torch.Tensor

    @property
    def decoder_layers(self) -> list[Qwen3_5MoeDecoderLayer]:
        """The layers, typed.

        ``nn.ModuleList`` yields ``Module``, which loses everything the layer declares, so iterating it
        directly defeats the annotations above. This is a view over the same registered modules — not a
        second registration — and it exists so the loops below read the layer's real attributes.
        """
        return cast(list["Qwen3_5MoeDecoderLayer"], list(self.layers))

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=config.hidden_size,
            dtype=config.torch_dtype, tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.first_attention_layer = next(
            i for i, layer in enumerate(self.layers) if layer.layer_type == FULL_ATTENTION)
        self.first_linear_layer = next(
            i for i, layer in enumerate(self.layers) if layer.layer_type == LINEAR_ATTENTION)
        # Registered empty and filled in load_weights: the runner constructs the model under `meta`,
        # where the arange/cos/sin that build these tables would themselves be meta tensors.
        self.register_buffer(
            "rotary_cos",
            torch.zeros(config.max_position_embeddings, config.rotary_dim, dtype=config.torch_dtype),
            persistent=False)
        self.register_buffer(
            "rotary_sin",
            torch.zeros(config.max_position_embeddings, config.rotary_dim, dtype=config.torch_dtype),
            persistent=False)

    def build_rotary_tables(self, device):
        """Fill the partial-rotary cos/sin tables for ``device``.

        Built on the HOST and then copied, not computed on the device: the arange / outer / cos here
        run outside any compiled graph, and eager elementwise math on a Neuron device is not something
        to rely on. The tables are a few tens of megabytes, so the copy is not worth avoiding.
        """
        cos, sin = rotary_tables(self.config.rotary_dim, self.config.max_position_embeddings,
                                 self.config.rope_theta, device="cpu",
                                 dtype=self.config.torch_dtype)
        self.rotary_cos = cos.to(device)
        self.rotary_sin = sin.to(device)

    def _attention_metadata(self, attn_metadata):
        return attn_metadata[attention_metadata_key(self.first_attention_layer)]

    def _is_prefill(self, attn_metadata):
        """Derive the phase from the attention metadata rather than from a runner flag.

        The metadata is authoritative both during compile warmup and at run time; a runner-supplied
        ``is_decode`` can arrive mislabelled for the single-token bucket, which would send a decode
        step down the chunked prefill scan and corrupt the recurrent state.
        """
        metadata = self._attention_metadata(attn_metadata)
        return metadata["max_query_len"] > metadata["decode_token_threshold"]

    def _segmented_cached_len(self, attn_metadata):
        """Tokens already processed, when segmented prefill is enabled; None for single-shot.

        ``kv_segment_size`` is a Python int fixed at trace time (set when
        ``max_num_batched_tokens < max_model_len``), so this branch is graph-static; only the returned
        length is a runtime value.
        """
        metadata = self._attention_metadata(attn_metadata)
        if metadata.get("kv_segment_size"):
            return metadata.get("cached_seq_len")
        return None

    def _packed_prefill_layout(self, attn_metadata, tokens):
        """The chunk-aligned layout for a packed multi-request prefill, or None when there is one request.

        None covers every configuration without the state pool, and also a pooled one carrying a single
        request -- in both cases the row already starts on a chunk boundary and the scan needs no carry
        mask, so the graph is the one that was verified.

        The aligned buffer is sized ``tokens + requests * (chunk - 1)``, the worst case when every
        request needs padding up to a whole chunk. It has to be a compile-time constant because the
        graph is compiled for it, so it is derived from the bucket rather than from the batch.
        """
        entry = attn_metadata.get(state_metadata_key(self.first_linear_layer))
        starts = None if entry is None else entry.get("query_start_loc")
        if starts is None or starts.numel() <= 2:
            return None
        mixer = self.decoder_layers[self.first_linear_layer].linear_attn
        chunk = mixer.chunk_size
        requests = starts.numel() - 1
        aligned = tokens + requests * (chunk - 1)
        aligned += (-aligned) % chunk
        return chunk_aligned_layout(starts, chunk, aligned, mixer.conv_kernel_size)

    def _state_slot(self, attn_metadata, layer_idx):
        """Which slot this layer's recurrent state lives in, or None when there is no pool.

        The runner puts one entry per state layer into the metadata carrying that request's slot, taken
        from the layer's own KV cache group. None means the state is in module buffers and there is a
        single slot, which is what every configuration without the pool gets.

        Read per layer rather than once for the model: the metadata is keyed by layer name, and reading
        one layer's entry for all of them would silently work today (all layers get the same slot) and
        break the moment the runner groups them differently.
        """
        entry = attn_metadata.get(state_metadata_key(layer_idx))
        return None if entry is None else entry["state_slots"]

    def _prefill_valid_mask(self, attn_metadata):
        """Per-token real/pad mask, taken from the attention slot mapping.

        The runner marks pad tokens with ``PAD_SLOT_ID = -1``. The Gated DeltaNet layers have no other
        pad signal, so this is how they keep the bucket padding out of the conv and recurrent state.
        """
        slot_mapping = self._attention_metadata(attn_metadata).get("slot_mapping")
        return None if slot_mapping is None else (slot_mapping >= 0)

    def _rotary(self, positions, rotary_position_ids):
        """cos/sin for this step, from either the single-axis table or the three axes.

        ``rotary_position_ids`` is ``[3, T]`` and is supplied only by the multimodal architecture, where
        the axes genuinely differ. When it is absent the precomputed table is indexed, which is both
        cheaper (a gather instead of an outer product per token, inside the graph) and exact: for a text
        prompt the three axes carry the same position, so the interleave selects the same frequency from
        whichever plane it reads.

        That last sentence is the regression condition, and it is pinned to the bit rather than to a
        tolerance (``test_three_axis_rotary_matches_the_table_when_the_axes_agree``). Without bit
        equality, a change in the text output after wiring vision could not be attributed: it might be
        this path or it might be the vision plumbing.
        """
        if rotary_position_ids is None:
            index = positions.to(torch.long)
            return self.rotary_cos.index_select(0, index), self.rotary_sin.index_select(0, index)
        return mrope_tables(rotary_position_ids, self.config.rotary_dim, self.config.rope_theta,
                            self.config.mrope_section, dtype=self.config.torch_dtype)

    def forward(self, input_ids, positions, attn_metadata,
                vision_embedding_blocks=None, vision_positions=None,
                rotary_position_ids=None):
        is_prefill = self._is_prefill(attn_metadata)
        cached_seq_len = self._segmented_cached_len(attn_metadata) if is_prefill else None
        valid_mask = self._prefill_valid_mask(attn_metadata) if is_prefill else None

        # A one-token prompt has max_query_len == 1 and so cannot be told apart from a decode step by
        # token count; the phase decision below will call it decode. This mask is what makes that
        # harmless: it is 0 when a token sits at absolute position 0, which zeroes the carried recurrent
        # and conv state so a fresh request starts from zero. It is tensor arithmetic, so no Python
        # branch on a runtime value enters the graph.
        #
        # ONE FLAG PER REQUEST at decode, where each position belongs to a different request. With a
        # single request that is a length-one vector rather than a scalar, which multiplies the state
        # identically -- so this is not a change to the shipped configuration's numbers.
        # One layout for the whole forward, not one per layer: it depends only on the request offsets and
        # the chunk width, both of which every Gated DeltaNet layer shares.
        packed = self._packed_prefill_layout(attn_metadata, input_ids.shape[0]) if is_prefill else None
        is_continuation = (positions.reshape(-1) > 0) if not is_prefill else (
            positions.reshape(-1)[:1] > 0).reshape(())
        cos, sin = self._rotary(positions, rotary_position_ids)

        model_dtype = self.config.torch_dtype
        # The residual stream is kept in fp32: see the module docstring for why this deviates from the
        # checkpoint's bf16 accumulation. Each block normalises from fp32 into the model dtype for the
        # weights, and its output is cast back to fp32 for the residual add.
        embedded = self.embed_tokens(input_ids, scatter_tokens=False)
        if is_prefill and vision_embedding_blocks is not None and vision_positions is not None:
            # Scattered in the embedding dtype, before the fp32 cast: the vision embeddings replace token
            # embeddings, so they belong in the same representation the table produced. Casting first
            # would give the same values (the widening is exact) at twice the traffic.
            #
            # rank is 0 because this backbone does not shard the sequence (embed_tokens is called with
            # scatter_tokens=False), so the merge's global and local coordinates coincide. Passing
            # self.tp_group's rank instead would remap positions that were never split.
            embedded, deepstack = merge_vision_embeddings(
                embedded, vision_embedding_blocks, vision_positions, rank=0)
            if deepstack is not None:
                # A wider cache row than the text stream means the checkpoint wants per-layer visual
                # injection. The config class already refuses such a checkpoint; this is the second
                # place it could arrive from, and dropping it silently would serve an image whose
                # intermediate features never reached the decoder.
                raise NotImplementedError(
                    "the encoder cache carries deepstack features, which this implementation does not "
                    "inject into the decoder layers."
                )
        hidden_states = embedded.to(torch.float32)
        for layer_index, layer in enumerate(self.decoder_layers):
            normed = layer.input_layernorm(hidden_states).to(model_dtype)
            if layer.layer_type == FULL_ATTENTION:
                mixed = layer.self_attn(normed, positions, cos, sin, attn_metadata, is_prefill)
            else:
                slot = self._state_slot(attn_metadata, layer_index)
                if is_prefill:
                    # The token axis leads for prefill. With several requests packed into the row the
                    # mixer is additionally given the chunk-aligned layout, which is what keeps one
                    # request's tail out of the next one's head.
                    mixed = layer.linear_attn.forward_prefill(
                        normed.unsqueeze(0), cached_seq_len=cached_seq_len, valid_mask=valid_mask,
                        slot=slot, packed=packed).squeeze(0)
                else:
                    # Decode puts the REQUEST axis first: one token each, and the scan is batch-general
                    # over that axis (measured bit-identical per row against single-request calls). With
                    # one request this is the same shape as the prefill form, so the shipped
                    # configuration traces the same graph.
                    mixed = layer.linear_attn.forward_decode(
                        normed.unsqueeze(1), is_continuation, slot=slot).squeeze(1)
            hidden_states = hidden_states + mixed.to(torch.float32)
            # The MoE block fuses its own pre-norm, so it takes the raw residual plus the norm weight.
            moe_out = layer.mlp(hidden_states, layer.post_attention_layernorm.weight, is_prefill,
                                valid_mask=valid_mask)
            hidden_states = hidden_states + moe_out.to(torch.float32)
        return self.norm(hidden_states).to(model_dtype)


class Qwen3_5MoeForCausalLM(nn.Module):
    """Top level: model + lm_head (+ on-device sampling when configured)."""

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5MoeModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config if config.neuron_config else None)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size, is_storage_transposed=False,
            ),
        )
        # The runner expects the full-vocab logits back alongside the sampled token whenever a
        # request wants logprobs, or when logits are being dumped for debugging. Under on-device
        # sampling the lm_head returns only this rank's vocab shard, so the distribution exists only
        # across ranks and has to be gathered here. TRAP: returning None instead does not refuse
        # `logprobs=N` — it makes the field come back EMPTY, which reads as "this model has no
        # logprobs" at the call site and as a successful request everywhere else.
        neuron = config.neuron_config
        self._gather_logits = bool(neuron is not None and (
            neuron.max_logprobs != 0 or neuron.debug_logits_dir is not None))
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler
            self.sampler = Sampler(self.on_device_sampling_config,
                                   process_group=self.tp_group.device_group)
        # Resolved once, at construction: get_kv_spec and bind_kv_cache have to agree about whether a
        # pool exists, and reading the environment in both would let them disagree if it changed between
        # the two calls.
        from vllm.config import get_current_vllm_config
        self._state_pool = state_pool_requested(get_current_vllm_config())

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, text_neuron_config=None,
                     vision_neuron_config=None, **kwargs):
        # The runner passes text_ and vision_neuron_config for a *ConditionalGeneration architecture
        # whether or not the deployment wants images, so the vision config is accepted and ignored here
        # (see the factory). Image and video input are kept out by declaring no multimodal interface.
        return cls(Qwen3_5MoeConfig.from_configs(
            hf_config, resolve_text_neuron_config(neuron_config, text_neuron_config)))

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """The text-only degenerate case of MRoPE: all three axes carry the same position.

        The runner sets ``uses_mrope`` from the config (which declares ``mrope_section``) and then
        requires this protocol. Implementing it here rather than pretending the model is not MRoPE
        makes the collapse explicit: for a text prompt the t/h/w axes are identical, which is exactly
        what lets the single-axis rotary table be exact (see ``ops.rotary_tables``).

        A request carrying multimodal features breaks that equality, so it is refused rather than
        served with positions that are wrong for the image spans.
        """
        if mm_features:
            raise NotImplementedError(
                "Qwen3.5-MoE on Neuron serves the text backbone only; multimodal features would give "
                "the three MRoPE axes different positions, which the single-axis rotary table here "
                "cannot represent."
            )
        length = len(input_tokens)
        positions = torch.arange(length, dtype=torch.int64).unsqueeze(0).expand(3, length).contiguous()
        return positions, 0

    def state_layer_specs(self):
        """The Gated DeltaNet layers' state, for the runner to pool.

        Two tensors per layer, in this order, and the order is the contract: the runner allocates them
        from one page in the order given and hands them back as a list, so ``bind_state_cache`` checks
        the shapes it receives rather than trusting the position.

        Empty unless the pool was asked for, which keeps the verified single-sequence path unchanged.
        """
        from vllm_neuron.model.kv_cache import StateLayerSpec

        specs = []
        for index, layer in enumerate(self.model.decoder_layers):
            if layer.layer_type != LINEAR_ATTENTION:
                continue
            mixer = layer.linear_attn
            specs.append(StateLayerSpec(
                name=state_metadata_key(index),
                shapes=(
                    (mixer.conv_dim_pr, mixer.conv_kernel_size - 1),
                    (mixer.v_heads_pr, mixer.head_k_dim, mixer.head_v_dim),
                ),
                dtypes=(mixer.dtype, torch.float32),
                # Gated DeltaNet, not Mamba2. Nothing on this platform resolves the name to a backend
                # class, which is exactly why it must not be left at a default that happens to parse.
                state_kind="GDN_ATTN",
                # No speculative blocks: the decode path raises on a multi-token step, so there is never
                # a draft whose state has to be held and rolled back.
                num_speculative_blocks=0,
            ))
        return specs

    def bind_state_cache(self, kv_caches):
        """Point each mixer's state at the runner's pooled tensors.

        Three things are checked, and each was a real defect before it was.

        **Shapes, not order.** Two tensors of different rank arriving the wrong way round fails here;
        two of the same rank would not, and the conv and recurrent states differ in rank, so this catches
        the realistic mistake.

        **Slot counts agree** between the two states, since they index by the same slot.

        **Storage sharing is recorded, not refused.** vLLM hands the same raw tensor to the layer at the
        same POSITION in every group: with groups [0, 4], [1, 5], [2, 6], layers 0, 1 and 2 share one
        tensor and layers 4, 5 and 6 share another. That is the design, not a defect -- each group gets
        its own block id, so those three layers are given different slots (2, 3 and 4 as measured) and
        their writes are disjoint.

        An earlier version of this method REFUSED the sharing. That was wrong, and the reasoning behind
        it is worth keeping: a pointer collision was observed and read as a collision of writes, without
        checking whether the slots differed. The invariant that actually matters is
        ``(storage, slot)`` distinctness, and the slot is not known here -- it arrives per step in the
        metadata. So this logs the sharing and leaves the invariant to be checked where the slot is.
        """
        self._state_storage: dict[int, list[str]] = {}
        for index, layer in enumerate(self.model.decoder_layers):
            if layer.layer_type != LINEAR_ATTENTION:
                continue
            name = state_metadata_key(index)
            if name not in kv_caches:
                raise RuntimeError(f"state cache for {name} not initialized")
            tensors = kv_caches[name]
            if len(tensors) != 2:
                raise RuntimeError(
                    f"{name}: expected the conv and recurrent state, got {len(tensors)} tensor(s)"
                )
            mixer = layer.linear_attn
            conv, recurrent = tensors
            expected = (
                (conv, 3, mixer.conv_state.shape[1:], "conv"),
                (recurrent, 4, mixer.recurrent_state.shape[1:], "recurrent"),
            )
            for tensor, rank, tail, label in expected:
                if tensor.dim() != rank or tuple(tensor.shape[1:]) != tuple(tail):
                    raise RuntimeError(
                        f"{name}: the {label} state arrived as {tuple(tensor.shape)}; expected "
                        f"(slots, *{tuple(tail)}). The two states have different ranks, so this is what "
                        "catches them arriving in the wrong order."
                    )
            if conv.shape[0] != recurrent.shape[0]:
                raise RuntimeError(
                    f"{name}: the two states were allocated {conv.shape[0]} and {recurrent.shape[0]} "
                    "slots; they index by the same slot so the counts must agree."
                )
            self._state_storage[conv.data_ptr()] = self._state_storage.get(
                conv.data_ptr(), []) + [name]
            mixer.conv_state = conv
            mixer.recurrent_state = recurrent
            mixer.num_slots = conv.shape[0]
        for pointer, sharers in self._state_storage.items():
            if len(sharers) > 1:
                # Expected: the layer at the same position in every group. Recorded so a step that hands
                # two of them the same slot can be recognised as a collision rather than as arithmetic.
                logger.info("state storage %s is shared by %s", pointer, sharers)

    def get_kv_spec(self):
        """The full_attention layers' KV, plus the Gated DeltaNet state when a pool was asked for.

        Without the pool the recurrent state stays in module buffers, carried between graph calls by the
        aliasing pass, and the deployment is capped at one sequence."""
        layers = []
        for i, layer in enumerate(self.model.decoder_layers):
            if layer.layer_type != FULL_ATTENTION:
                continue
            attention = layer.self_attn
            layers.append(LayerSpec(
                name=attention_metadata_key(i),
                num_kv_heads=attention.num_key_value_heads_per_rank,
                head_size=attention.head_dim, dtype=attention.dtype,
                sliding_window_size=attention.window_size, chunk_size=None,
            ))
        state_layers = self.state_layer_specs() if self._state_pool else []
        return KVSpec(layers=layers, state_layers=state_layers)

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.decoder_layers):
            if layer.layer_type != FULL_ATTENTION:
                continue
            name = attention_metadata_key(i)
            if name not in kv_caches:
                raise RuntimeError(f"KV cache for {name} not initialized")
            layer.self_attn.bind_caches(kv_caches[name][0], kv_caches[name][1])
        if self._state_pool:
            # Only when the pool was declared. Binding unconditionally would raise for every deployment
            # that never asked for it, since the runner would have allocated nothing.
            self.bind_state_cache(kv_caches)

    def _vision_inputs(self, kwargs):
        """The encoder-cache views to merge, or ``(None, None)`` for the text architecture.

        **The runner supplies these whether or not this architecture wants them.** It sets
        ``supports_mm_inputs`` from the multimodal registry, which is keyed by the config's
        ``model_type``, so overriding ``architectures`` to the text class does not stop vision inputs
        from arriving -- during warmup they arrive as zero blocks sized from the encoder cache.

        Discarding them here rather than testing whether they are present is the whole point. A guard
        that asks "were any supplied?" is always satisfied, so the text model would merge encoder-cache
        blocks it has no encoder for; CPU mode caught exactly that, in the first warmup it ran.
        """
        return None, None

    def _positions(self, positions):
        """Split the runner's positions into the sequential axis and the rotary's position ids.

        Two things are wanted from one argument. The KV cache, the attention mask and the recurrent
        layers' fresh-request test need ONE monotone position per token; the rotary may need three.

        For the text architecture the three agree (see ``get_mrope_input_positions``), so the first axis
        serves as the sequential one and the rotary needs no separate ids — returning None for them
        selects the cheaper table lookup. The multimodal subclass overrides this, because there the axes
        differ once an image is present.

        The returned sequential axis is NOT a monotone per-token index when an image is present; see
        ``ops.temporal_axis`` for why the two consumers in this model are unaffected.
        """
        return temporal_axis(positions), None

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions,
                sampling_params, spec_decode_metadata=None, logit_mask=None, rank=None,
                inputs_embeds=None, is_token_ids=None, **kwargs):
        # Speculative decoding is rejected by the factory, and the Gated DeltaNet decode raises on a
        # multi-token step. Nothing is checked here: the runner can hand over an empty
        # spec_decode_metadata on an ordinary step, so refusing the argument itself would break
        # normal serving while adding no protection the mixer does not already give.
        if inputs_embeds is not None:
            # The factory refuses enable_prompt_embeds, so this should be unreachable; raise rather
            # than embedding the placeholder token IDs and serving unrelated logits.
            raise NotImplementedError(
                "Qwen3.5-MoE on Neuron does not merge prompt embeddings; it embeds input_ids only."
            )
        sequential, rotary_position_ids = self._positions(positions)
        vision_blocks, vision_positions = self._vision_inputs(kwargs)
        hidden_states = self.model(
            input_ids, sequential, attn_metadata,
            vision_embedding_blocks=vision_blocks, vision_positions=vision_positions,
            rotary_position_ids=rotary_position_ids)
        sampled = torch.index_select(hidden_states, 0, sampling_positions)
        logits = self.lm_head(sampled)
        if self.on_device_sampling_config is None:
            # gather_output was True, so these are already the full vocabulary.
            return logits
        gathered_logits = None
        if self._gather_logits:
            # >>> PARALLELISM: gather the vocabulary shards for logprobs <<<
            # The gather uses the lm_head's OWN group and the same primitive the layer uses for
            # gather_output, so it cannot disagree with how the vocabulary was sharded.
            # The layer's group is Optional in its annotation and set by construction here, so the
            # cast records that rather than adding a check that cannot fire.
            gathered_logits = all_gather_tensor(logits, 1, cast(Any, self.lm_head.tp_group))
        return (self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank),
                gathered_logits)

    def checkpoint_mappings(self, source, checkpoint_keys):
        """Every parameter this model needs, mapped to the checkpoint key(s) that fill it.

        Split out of ``load_weights`` so an architecture that adds parameters extends the map instead of
        reimplementing the load. That matters more than it looks: the load ends with two checks — every
        mapped source present, and nothing left on ``meta`` — and both are written over the WHOLE model.
        A subclass that loaded its extra weights in a second pass would run the meta check before its own
        parameters were filled, so it would have to weaken the check that catches the worst failure.
        """
        return checkpoint_mappings(
            [layer.layer_type for layer in self.model.layers],
            source,
            has_lm_head="lm_head.weight" in checkpoint_keys,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )

    def load_weights(self, checkpoint_path, device, cache_dir=None):
        """Map the HF checkpoint onto the per-rank parameters.

        <-- MODEL-SPECIFIC: the key layout below is this checkpoint's.

        The published checkpoint is a multimodal wrapper, so the decoder sits under
        ``model.language_model.*`` while ``lm_head.weight`` is at the top level. A text-only
        checkpoint would place the decoder under ``model.*``; the prefix is probed rather than assumed.
        ``model.visual.*`` and ``mtp.*`` are present and deliberately unmapped — the vision tower and
        the multi-token-prediction head are out of scope.
        """
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        checkpoint_keys = set()
        index_files = glob.glob(os.path.join(checkpoint_path, "*.index.json"))
        if index_files:
            with open(index_files[0]) as handle:
                checkpoint_keys = set(json.load(handle)["weight_map"].keys())
        else:
            # Every shard, not just the first: the completeness check below refuses to load when a
            # mapped source key is absent, so an incomplete key set would reject a valid checkpoint.
            # Reading the headers is cheap (safetensors does not load tensor data here).
            from safetensors import safe_open
            for shard in sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))):
                with safe_open(shard, framework="pt") as opened:
                    checkpoint_keys.update(opened.keys())
        if not checkpoint_keys:
            raise ValueError(
                f"no safetensors index or shards found under {checkpoint_path}; cannot verify that "
                "the checkpoint provides the parameters this implementation needs."
            )
        if any(key.startswith("model.language_model.") for key in checkpoint_keys):
            source = "model.language_model"
        elif any(key.startswith("model.layers.") for key in checkpoint_keys):
            source = "model"
        else:
            raise ValueError(
                f"could not find a Qwen3.5-MoE decoder in {checkpoint_path}: expected keys under "
                "'model.language_model.' (the multimodal checkpoint) or 'model.' (text-only), got "
                f"e.g. {sorted(checkpoint_keys)[:5]}"
            )

        mappings = self.checkpoint_mappings(source, checkpoint_keys)

        missing = sorted(
            key for value in mappings.values()
            for key in ([value] if isinstance(value, str) else value)
            if key not in checkpoint_keys
        )
        if missing:
            raise ValueError(
                f"{len(missing)} parameter(s) this implementation needs are absent from the "
                f"checkpoint, e.g. {missing[:5]}. Loading would leave them on `meta` and produce "
                "garbage rather than fail."
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device, strict=False).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # The runner builds the model on `meta`; the recurrent buffers are runtime state rather than
        # checkpoint entries, so load_state_dict leaves them there. Reassign real zeros so the
        # in-place updates (and the aliasing they rely on) have somewhere to write.
        for layer in self.model.decoder_layers:
            if layer.layer_type != LINEAR_ATTENTION:
                continue
            mixer = layer.linear_attn
            mixer.recurrent_state = torch.zeros(
                mixer.recurrent_state.shape, dtype=torch.float32, device=device)
            mixer.conv_state = torch.zeros(
                mixer.conv_state.shape, dtype=mixer.dtype, device=device)
        self.model.build_rotary_tables(device)

        still_meta = [name for name, param in self.named_parameters() if param.is_meta]
        if still_meta:
            raise ValueError(
                f"{len(still_meta)} parameter(s) are still on `meta` after loading, e.g. "
                f"{sorted(still_meta)[:5]}. Serving them would produce garbage."
            )
