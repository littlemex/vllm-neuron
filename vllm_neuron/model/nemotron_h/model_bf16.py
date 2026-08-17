# SPDX-License-Identifier: Apache-2.0
"""
NemotronH (bf16) — vLLM-Neuron SERVING implementation.
======================================================
Text backbone (NemotronHForCausalLM) of Nemotron-3-Nano-Omni-30B-A3B. Hybrid decoder that
interleaves Mamba2 (SSM), MoE, and Attention layers per `hybrid_override_pattern`
("MEMEM*EMEMEM*..." = 23 Mamba / 23 MoE / 6 Attention over 52 layers).

This is the serving implementation. It follows the proven
vLLM-Neuron plugin patterns:
  - Attention: NF.qkv_proj / NF.flash_attention (prefill) + pure-PyTorch fp32 decode
    (template: verifications/plamo3-vllm-onboard/plamo3/model.py, incl. KNOWN_ISSUES Issue 1 fix).
    NemotronH attention is PLAIN GQA (32 Q / 2 KV / head_dim 128): no attention sinks, no sliding
    window, and NoPE (no rotary embedding; position information is carried by the Mamba2 layers).
  - MoE: DeepSeek-style grouped top-k routing (sigmoid + e_score_correction_bias + group select +
    norm + routed_scaling_factor), 128 routed experts top-6 + 1 shared, relu^2 activation
    (from modeling_nemotron_h.py NemotronHMoE.route_tokens_to_experts). Correctness-first
    per-expert loop; NF.moe_cte fast path is a later optimization.
  - Mamba2: vectorized-SSD prefill + 1-step-recurrence decode, with recurrent (ssm+conv) state
    carried across decode steps via in-place module buffers that the plugin's
    AliasingOutputRewritePass turns into HLO input_output_alias (batch=1). No runner-side state pool.
    This is the NemotronH-specific piece plamo3 does not have.

TP: standard tensor parallelism (head sharding for attention, expert/intermediate sharding for MoE
and Mamba inner dim). 30B-A3B bf16 (~62 GB) needs TP=4 to fit 4 NeuronCores of one trn2 chip.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from .ssd import chunked_ssd_scan

from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
import vllm_neuron.functional as NF
from vllm_neuron.nn import ColumnParallelLinear
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.weight_loader import (
    sharding_weight_loader,
    set_weight_loader,
    SafetensorsWeightLoader,
)

from .config import NemotronHConfig, MAMBA, MOE, ATTENTION

logger = logging.getLogger(__name__)


# ============================================================
# RMSNorm (NemotronH: standard RMSNorm, weight * normed, no offset)
# ============================================================
class NemotronHRMSNorm(nn.Module):
    """Reusable: standard RMSNorm (same shape/math as llama3/gpt_oss). Keep when porting."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(in_dtype))


# ============================================================
# Attention (plain GQA, plamo3-style serving path)
# ============================================================
class NemotronHAttention(nn.Module):
    """GQA (32 Q / 2 KV / head_dim 128), no sinks, no SWA. Template: plamo3 Plamo3Attention.

    >>> PARALLELISM: TP (head sharding, GQA replication) <<<
    <-- MODEL-SPECIFIC: plain GQA, NoPE (no rotary), fp32 pure-PyTorch decode

    HF checkpoint stores separate q_proj/k_proj/v_proj (NOT fused). We fuse into one qkv weight for
    NF.qkv_proj, and the weight loader slices the three HF tensors into the per-rank fused layout.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.window_size = None  # <-- MODEL-SPECIFIC: NemotronH has full attention only, no SWA

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads = config.num_attention_heads       # 32
        self.num_key_value_heads = config.num_key_value_heads       # 2
        # <-- MODEL-SPECIFIC: GQA sharding assumes KV heads either divide, or are divided by, the TP
        # size (so each rank owns a whole number of KV heads, or a whole rank-group shares one).
        # Reject other TP sizes up front rather than silently reading an out-of-range KV-head slice
        # in the weight loader.
        if not (self.world_size % self.num_key_value_heads == 0
                or self.num_key_value_heads % self.world_size == 0):
            raise ValueError(
                f"tensor_parallel_size={self.world_size} is incompatible with "
                f"num_key_value_heads={self.num_key_value_heads}: one must divide the other."
            )
        # >>> PARALLELISM: GQA head-sharding calculation <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        # GQA: KV heads (2) < TP size (4) → replicate KV across ranks
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in = (self.num_attention_heads * self.head_dim) // self.world_size

        self.qkv_proj_weight = nn.Parameter(torch.empty(self.hidden_size, qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(torch.empty(o_proj_in, self.hidden_size, dtype=self.dtype))
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """HF stores separate q/k/v (dims [nh*hd, H], [nkv*hd, H], [nkv*hd, H]).
        Fuse into per-rank [H, q_size+2*kv_size]. Q sharded by head; K/V replicated per rank group.

        >>> PARALLELISM: TP + GQA replication sharding <<<
        """
        head_dim = self.head_dim
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        q_per_rank = self.q_size
        kv_per_rank = self.kv_size
        num_kv_replicas = self.num_kv_replicas
        world_size = self.world_size

        def _qkv_transform(slices, rank):
            local_rank = rank % world_size
            kv_rank = local_rank // num_kv_replicas
            q_w = slices[0][local_rank * q_per_rank:(local_rank + 1) * q_per_rank, :]     # [q_per_rank, H]
            k_w = slices[1][kv_rank * kv_per_rank:(kv_rank + 1) * kv_per_rank, :]         # [kv_per_rank, H]
            v_w = slices[2][kv_rank * kv_per_rank:(kv_rank + 1) * kv_per_rank, :]
            qkv = torch.cat([q_w, k_w, v_w], dim=0)                                       # [qkv_size, H]
            return qkv.T.contiguous()                                                    # [H, qkv_size]

        set_weight_loader(self.qkv_proj_weight, SafetensorsWeightLoader(transform=_qkv_transform))
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(nh * head_dim) // world_size,
                num_shards=world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states, positions, attn_metadata):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        if is_prefill:
            return self._prefill(hidden_states, positions, attn_metadata)
        return self._decode(hidden_states, positions, attn_metadata)

    def _project_qkv(self, hidden_states, tokens):
        # <-- MODEL-SPECIFIC: Plain matmul instead of NF.qkv_proj. The fused NKI kernel internally
        # 2-way shards H1=hidden/128 under LNC=2; for hidden=2688, H1=21 is odd and fails NCC_INKI016.
        # A plain matmul compiles via the general path with no such constraint. qkv_proj_weight is
        # [H, q+2kv], so hidden@W -> [T, q+2kv].
        qkv = hidden_states @ self.qkv_proj_weight
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        return q, k, v

    def _prefill(self, hidden_states, positions, attn_metadata):
        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape
        layer_name = f"layers.{self.layer_idx}.self_attn"
        q, k, v = self._project_qkv(hidden_states, tokens)
        # <-- MODEL-SPECIFIC: NemotronH attention is NoPE: no rotary position embedding (position
        # information is carried by the Mamba layers). Verified against HF NemotronHAttention
        # (cosine 0.9999999); applying RoPE here was a bug that flipped the greedy first token vs
        # the HF reference.
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        self._write_kv_cache(k, v, slot_mapping, block_size)

        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)
        q_flash = q.transpose(1, 2)   # [Nh, Dh, T]
        k_flash = k.transpose(1, 2)
        v_flash = v                    # [Nh, T, Dh]
        attn_output = NF.flash_attention(
            q_flash, k_flash, v_flash, scale=self.scaling, causal_mask=True, tp_q=False, tp_out=True,
        )  # [Nh, Dh, T]
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None).squeeze(0)  # [T, H]
        if self.world_size > 1:
            # >>> PARALLELISM: SP reduce-scatter (prefill) <<<
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def _decode(self, hidden_states, positions, attn_metadata):
        """Pure-PyTorch fp32 decode (KNOWN_ISSUES Issue 1: NF.flash_attention breaks on the
        asymmetric q=1/k=S_ctx decode shape; fp32 also removes bf16 accumulation drift).

        <-- MODEL-SPECIFIC: fp32 pure-PyTorch decode path (plamo3-derived workaround).
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B_local = block_table.shape[0]
        tokens, _ = hidden_states.shape
        S_decode = tokens // B_local
        assert tokens == B_local * S_decode
        if B_local != 1:
            raise NotImplementedError(
                "NemotronH serving currently supports max_num_seqs=1 only: the Mamba2 recurrent "
                "state is a single per-layer buffer (no per-slot pool), so concurrent sequences "
                f"would corrupt each other. Got batch size {B_local}."
            )
        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        q, k, v = self._project_qkv(hidden_states, tokens)
        # <-- MODEL-SPECIFIC: NemotronH attention is NoPE (no rotary) — see the note in _prefill.
        self._write_kv_cache(k, v, slot_mapping, block_size)

        flat_blocks = block_table.reshape(-1)
        k_gathered = self.k_cache[flat_blocks]
        v_gathered = self.v_cache[flat_blocks]
        k_dense = (k_gathered.view(B_local, max_blocks_per_seq, nkh, block_size, self.head_dim)
                   .permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim))
        v_dense = (v_gathered.view(B_local, max_blocks_per_seq, nkh, block_size, self.head_dim)
                   .permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim))
        if B_local == 1:
            k_full = k_dense.squeeze(0).to(self.dtype)
            v_full = v_dense.squeeze(0).to(self.dtype)
        else:
            k_full = k_dense.reshape(B_local * nkh, S_ctx, self.head_dim).to(self.dtype)
            v_full = v_dense.reshape(B_local * nkh, S_ctx, self.head_dim).to(self.dtype)
        k_full = k_full.repeat_interleave(self.num_key_value_groups, dim=0)
        v_full = v_full.repeat_interleave(self.num_key_value_groups, dim=0)

        q_f32 = q.to(torch.float32)
        k_f32 = k_full.to(torch.float32)
        v_f32 = v_full.to(torch.float32)
        scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * self.scaling
        arange_ctx = torch.arange(S_ctx, device=q.device)
        # Per-query causal mask: each of the S_decode query rows attends only up to its own absolute
        # position. (Masking with positions[-1] alone would let earlier decode-bucket rows attend to
        # future keys; identical to the single-value mask when S_decode == 1.)
        valid = arange_ctx.view(1, -1) <= positions.view(-1, 1)     # [S_decode, S_ctx]
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v_f32).to(self.dtype)   # [Nh, S_decode, head_dim]

        attn_output = attn_output.transpose(-2, -1).unsqueeze(0)          # [1, Nh, head_dim, S_decode]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None).squeeze(0)  # [S_decode, H]
        if self.world_size > 1:
            # >>> PARALLELISM: TP all-reduce (decode) <<<
            self.tp_group.all_reduce(attn_output)
        return attn_output.contiguous()

    def _write_kv_cache(self, k, v, slot_mapping, block_size):
        blk_idx = slot_mapping // block_size
        pos_idx = slot_mapping % block_size
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank
        k_f = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_f = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        h_idx = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(num_slots)
        self.k_cache.index_put_((blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh)), k_f)
        self.v_cache.index_put_((blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh)), v_f)


# ============================================================
# MoE (DeepSeek-style grouped top-k routing) — correctness-first
# ============================================================
class NemotronHMoE(nn.Module):
    """128 routed experts (top-6) + 1 shared. relu^2 MLP (up→relu^2→down, no gate).

    Routing (modeling_nemotron_h.py:781): sigmoid(logits) → + e_score_correction_bias →
    group-topk select (n_group / topk_group) → top-k → norm_topk_prob → × routed_scaling_factor.
    TP: experts' intermediate dim is sharded across ranks (each rank holds full expert set but
    intermediate/world_size wide); shared expert likewise. Correctness-first per-expert loop.

    >>> PARALLELISM: TP (expert-intermediate sharding) <<<
    <-- MODEL-SPECIFIC: DeepSeek-style grouped top-k router, DGE-free dense gate, relu^2 experts
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d = config.hidden_size
        self.E = config.n_routed_experts
        self.k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared = config.n_shared_experts
        self.dtype = config.torch_dtype
        # The DGE-free dense router (_gate_dense) is only implemented for n_group=1 (as used by
        # 30B-A3B). The grouped path relies on data-dependent scatter/gather that neuronx-cc
        # miscompiles, so refuse a grouped config rather than fall back to the broken path.
        if self.n_group > 1:
            raise NotImplementedError(
                "NemotronH on Neuron only supports n_group=1 MoE routing (30B-A3B); "
                f"grouped routing (n_group={self.n_group}) is not implemented."
            )
        if config.mlp_hidden_act != "relu2":
            raise ValueError(
                f"NemotronH MoE only implements relu^2 experts; got mlp_hidden_act="
                f"{config.mlp_hidden_act!r}."
            )

        # >>> PARALLELISM: TP group setup + expert-intermediate shard-size calculation <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        mi = config.moe_intermediate_size
        smi = config.moe_shared_expert_intermediate_size
        if mi % self.world_size != 0 or smi % self.world_size != 0:
            raise ValueError(
                f"MoE intermediate sizes (moe_intermediate_size={mi}, "
                f"moe_shared_expert_intermediate_size={smi}) must both be divisible by "
                f"tensor_parallel_size={self.world_size}."
            )
        self.mi_per_rank = mi // self.world_size
        self.smi_per_rank = smi // self.world_size

        # Router (gate): full weight on every rank (small: [E, d]). fp32 compute.
        self.gate_weight = nn.Parameter(torch.empty(self.E, self.d, dtype=torch.float32))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.E, dtype=torch.float32))

        # Routed experts: up [E, mi_per_rank, d], down [E, d, mi_per_rank] (relu^2, no gate).
        self.up = nn.Parameter(torch.empty(self.E, self.mi_per_rank, self.d, dtype=self.dtype).normal_(0, 0.02))
        self.down = nn.Parameter(torch.empty(self.E, self.d, self.mi_per_rank, dtype=self.dtype).normal_(0, 0.02))
        # Shared expert (relu^2).
        self.shared_up = nn.Parameter(torch.empty(self.smi_per_rank, self.d, dtype=self.dtype).normal_(0, 0.02))
        self.shared_down = nn.Parameter(torch.empty(self.d, self.smi_per_rank, dtype=self.dtype).normal_(0, 0.02))
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        ws = self.world_size
        mi_pr = self.mi_per_rank
        smi_pr = self.smi_per_rank

        # gate/bias: no shard (replicate).
        set_weight_loader(self.gate_weight, SafetensorsWeightLoader(transform=lambda s, r: s[0][:, :]))
        set_weight_loader(self.e_score_correction_bias, SafetensorsWeightLoader(transform=lambda s, r: s[0][:]))

        # up: HF experts.{e}.up_proj.weight is [mi, d]; we stack E of them → [E, mi, d], shard mi.
        # The mappings feed E source keys; transform stacks + shards.
        def _up_transform(slices, rank):
            lr = rank % ws
            stacked = torch.stack([sl[lr * mi_pr:(lr + 1) * mi_pr, :] for sl in slices], dim=0)  # [E, mi_pr, d]
            return stacked.contiguous()

        def _down_transform(slices, rank):
            lr = rank % ws
            # HF down_proj.weight is [d, mi]; shard mi (dim 1).
            stacked = torch.stack([sl[:, lr * mi_pr:(lr + 1) * mi_pr] for sl in slices], dim=0)  # [E, d, mi_pr]
            return stacked.contiguous()

        set_weight_loader(self.up, SafetensorsWeightLoader(transform=_up_transform))
        set_weight_loader(self.down, SafetensorsWeightLoader(transform=_down_transform))

        def _shared_up_transform(slices, rank):
            lr = rank % ws
            return slices[0][lr * smi_pr:(lr + 1) * smi_pr, :].contiguous()

        def _shared_down_transform(slices, rank):
            lr = rank % ws
            return slices[0][:, lr * smi_pr:(lr + 1) * smi_pr].contiguous()

        set_weight_loader(self.shared_up, SafetensorsWeightLoader(transform=_shared_up_transform))
        set_weight_loader(self.shared_down, SafetensorsWeightLoader(transform=_shared_down_transform))

    def _gate_dense(self, x):
        """DGE-free routing: build the dense per-expert gate [T, E] directly, using only reductions
        and elementwise comparisons — NO scatter/gather with data-dependent indices.

        <-- MODEL-SPECIFIC: DeepSeek-style grouped top-k router, unique to NemotronH.

        neuronx-cc miscompiles the scatter/gather-based top-k routing (argmax+scatter loop,
        `scores.gather`, `gate.scatter_`) once several MoE layers are stacked: at >=4 MoE layers the
        runtime raises `scatter/gather (indirect memory copy via vector DGE) out-of-bound access`
        during warmup (verified: 3 MoE layers OK, 4+ fail; mamba/attention-only stacks of the same
        depth are fine). Reformulating the router with dense ops removes every data-dependent DGE and
        sidesteps the bug. Math is identical to a scatter-based argmax top-k router (argmax's
        lowest-index tie-break is reproduced exactly), so greedy outputs are unchanged.
        """
        logits = F.linear(x.to(torch.float32), self.gate_weight)   # [T, E]
        scores = logits.sigmoid()
        sfc = scores + self.e_score_correction_bias                # scores_for_choice [T, E]
        E = self.E
        iota = torch.arange(E, device=x.device).view(1, E).to(sfc.dtype)   # [1, E]
        big = torch.full_like(sfc, float(E))
        work = sfc
        sel_mask = torch.zeros_like(sfc)                           # [T, E], 1.0 at chosen experts
        for _ in range(self.k):
            m = work.max(dim=-1, keepdim=True).values              # [T, 1]
            is_max = work >= m                                     # [T, E] (>=1 True on ties)
            # lowest index among the maxima == torch.argmax semantics
            idx_first = torch.where(is_max, iota, big).min(dim=-1, keepdim=True).values  # [T, 1]
            onehot = (iota == idx_first).to(sfc.dtype)             # [T, E], exactly one 1.0
            sel_mask = sel_mask + onehot
            work = work - onehot * 1e30                            # drop the chosen expert (dense)
        gate = sel_mask * scores                                   # weights from RAW sigmoid scores
        if self.norm_topk_prob:
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-20)
        gate = gate * self.routed_scaling_factor
        return gate.to(self.dtype)                                 # [T, E] dense gate

    def _expert_mlp(self, x_e, e):
        # <-- MODEL-SPECIFIC: relu^2 expert activation: down(relu(up(x))^2)
        h = F.linear(x_e, self.up[e])          # [n, mi_per_rank]
        h = F.relu(h).pow(2)
        return F.linear(h, self.down[e])       # [n, d]

    def forward(self, hidden_states, is_prefill=False):
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, orig_shape[-1])   # [T, d]
        # Dense, trace-safe dispatch: build a per-expert weight matrix [T, E] (0 for unselected),
        # then run every expert over all tokens and scale by that column. `_gate_dense` builds the
        # gate with only reductions/elementwise ops (no data-dependent scatter/gather), which the
        # Neuron compiler mishandles once several MoE layers are stacked. No data-dependent Python
        # control flow either — required for torch.compile / Neuron tracing. Heavier than a gather
        # loop but static-shape and correctness-first (fast NF.moe path is future work).
        gate = self._gate_dense(x)                      # [T, E], weight in each selected column
        out = torch.zeros_like(x)
        for e in range(self.E):
            w_e = gate[:, e:e + 1]                       # [T, 1]
            out = out + self._expert_mlp(x, e) * w_e
        # shared expert (relu^2), always on
        sh = F.linear(x, self.shared_up)
        sh = F.relu(sh).pow(2)
        out = out + F.linear(sh, self.shared_down)
        out = out.reshape(orig_shape)
        # >>> PARALLELISM: experts/shared computed on sharded intermediate → sum partials across
        # ranks (reduce-scatter on prefill, all-reduce on decode) <<<
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                self.tp_group.all_reduce(out)
        return out


# ============================================================
# Mamba2 mixer (stateful; ssm+conv state carried via in-place module buffers)
# ============================================================
class NemotronHMamba2Mixer(nn.Module):
    """Vectorized-SSD prefill + 1-step decode. The recurrent (ssm+conv) state lives in module
    buffers updated in place; the plugin's AliasingOutputRewritePass turns the copy_ into an HLO
    input_output_alias so state persists across decode steps (batch=1). Trainium lessons baked in
    (slice not zero-size split; vectorized SSD not chunked-SSD). TP: shard the inner (heads)
    dimension across ranks.

    <-- MODEL-SPECIFIC: entire Mamba2 mixer is new for this architecture (no reference model in the
    plugin has an SSM/recurrent layer); TP head sharding within it is noted per-method below.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.dtype = config.torch_dtype

        self.num_heads = config.mamba_num_heads              # 64
        self.head_dim = config.mamba_head_dim                # 64
        self.ssm_state_size = config.ssm_state_size          # 128
        self.n_groups = config.n_groups                      # 8
        self.conv_kernel_size = config.conv_kernel           # 4
        self.time_step_min = config.time_step_min
        self.intermediate_size = config.mamba_intermediate_size   # 4096
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.norm_eps = config.rms_norm_eps
        if config.mamba_hidden_act != "silu":
            raise ValueError(
                f"NemotronH Mamba2 only implements SiLU activation; got mamba_hidden_act="
                f"{config.mamba_hidden_act!r}."
            )
        self.act = nn.SiLU()

        if self.num_heads % self.world_size != 0:
            raise ValueError(
                f"mamba_num_heads={self.num_heads} must be divisible by "
                f"tensor_parallel_size={self.world_size}."
            )
        if not (self.n_groups % self.world_size == 0 or self.world_size == 1):
            raise ValueError(
                f"n_groups={self.n_groups} must be divisible by tensor_parallel_size="
                f"{self.world_size} (or TP=1)."
            )
        # Per-rank shapes (head sharding).
        self.num_heads_pr = self.num_heads // self.world_size
        self.groups_pr = max(1, self.n_groups // self.world_size)
        self.im_pr = self.num_heads_pr * self.head_dim
        self.conv_dim_pr = self.im_pr + 2 * self.groups_pr * self.ssm_state_size

        # in_proj: column-parallel-like; we keep full weight then slice per rank in loader.
        self.in_proj_weight = nn.Parameter(torch.empty(config.hidden_size, self._proj_out_pr(), dtype=self.dtype))
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim_pr, 1, self.conv_kernel_size, dtype=self.dtype))
        self.conv1d_bias = nn.Parameter(torch.zeros(self.conv_dim_pr, dtype=self.dtype))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads_pr))
        self.D = nn.Parameter(torch.ones(self.num_heads_pr))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads_pr))
        self.norm_weight = nn.Parameter(torch.ones(self.im_pr))
        self.out_proj_weight = nn.Parameter(torch.empty(self.im_pr, config.hidden_size, dtype=self.dtype))
        # Recurrent state carried across decode steps. Held as module buffers and updated in place;
        # the plugin's AliasingOutputRewritePass turns the in-place copy_ into an HLO
        # input_output_alias so the state persists across the runner's decode-step graph calls
        # WITHOUT a runner-side state pool. batch=1 (max_num_seqs=1) bring-up: a single slot.
        # (Materialised to real device tensors in load_weights — the runner builds on `meta`.)
        self.register_buffer(
            "ssm_state", torch.zeros(1, self.num_heads_pr, self.head_dim, self.ssm_state_size,
                                     dtype=torch.float32), persistent=False)
        self.register_buffer(
            "conv_state", torch.zeros(1, self.conv_dim_pr, self.conv_kernel_size - 1,
                                      dtype=self.dtype), persistent=False)
        self._setup_weight_loaders()

    def _proj_out_pr(self):
        # per-rank in_proj output width: gate(im_pr) + conv_dim_pr + dt(num_heads_pr)
        return self.im_pr + self.conv_dim_pr + self.num_heads_pr

    # NOTE: TP weight sharding for Mamba in_proj is intricate (gate/xBC/dt interleave). For TP=1 the
    # loaders are identity; TP>1 sharding is handled in load_weights mapping helpers. Documented in
    # DESIGN-serving-path.md. Here we register identity loaders (TP=1 correct) and refine for TP>1.
    def _setup_weight_loaders(self):
        # in_proj HF: [proj_out, hidden]; param stored [hidden, proj_out_pr]. TP=1 → full transpose.
        def _in_proj_transform(slices, rank):
            w = slices[0]  # [proj_out, hidden] (lazy PySafeSlice)
            if self.world_size == 1:
                return w[:].T.contiguous()   # materialize the slice before .T
            # TP>1: slice gate/xBC/dt segments per rank, then concat and transpose.
            return self._shard_in_proj(w, rank % self.world_size).T.contiguous()
        set_weight_loader(self.in_proj_weight, SafetensorsWeightLoader(transform=_in_proj_transform))

        def _conv_transform(slices, rank):
            w = slices[0]  # [conv_dim, 1, K] (lazy PySafeSlice)
            if self.world_size == 1:
                return w[:].contiguous()
            return self._shard_conv(w, rank % self.world_size).contiguous()
        set_weight_loader(self.conv1d_weight, SafetensorsWeightLoader(transform=_conv_transform))

        def _conv_bias_transform(slices, rank):
            b = slices[0]  # [conv_dim] (lazy PySafeSlice)
            if self.world_size == 1:
                return b[:].contiguous()
            return self._shard_conv_bias(b, rank % self.world_size).contiguous()
        set_weight_loader(self.conv1d_bias, SafetensorsWeightLoader(transform=_conv_bias_transform))

        # head-wise params (A_log/D/dt_bias): shard by head.
        nh_pr = self.num_heads_pr
        for p in (self.A_log, self.D, self.dt_bias):
            set_weight_loader(p, SafetensorsWeightLoader(
                transform=lambda s, r, _n=nh_pr: s[0][(r % self.world_size) * _n:((r % self.world_size) + 1) * _n]))
        # norm_weight over im_pr: shard by inner dim.
        im_pr = self.im_pr
        set_weight_loader(self.norm_weight, SafetensorsWeightLoader(
            transform=lambda s, r: s[0][(r % self.world_size) * im_pr:((r % self.world_size) + 1) * im_pr]))
        # out_proj HF [hidden, im]; param [im_pr, hidden]; shard im (row of param), transpose.
        set_weight_loader(self.out_proj_weight, sharding_weight_loader(
            shard_dim=0, shard_size=im_pr, num_shards=self.world_size, is_storage_transposed=True))

    def _shard_in_proj(self, w, lr):
        # Mixed: PARALLELISM (per-rank slicing) + MODEL-SPECIFIC (gate/x/B/C/dt interleaved layout)
        # w: [proj_out, hidden] = [gate(im) | xBC(conv_dim) | dt(nh)] rows.
        im = self.intermediate_size
        conv_dim = self.conv_dim
        gn = self.n_groups * self.ssm_state_size
        im_pr, gpr, nh_pr = self.im_pr, self.groups_pr, self.num_heads_pr
        # gate rows for this rank
        gate = w[lr * im_pr:(lr + 1) * im_pr, :]
        # xBC = [x(im) | B(gn) | C(gn)]
        x = w[im:im + im, :]
        B = w[im + im:im + im + gn, :]
        C = w[im + im + gn:im + im + 2 * gn, :]
        gpr_n = gpr * self.ssm_state_size
        x_r = x[lr * im_pr:(lr + 1) * im_pr, :]
        B_r = B[lr * gpr_n:(lr + 1) * gpr_n, :]
        C_r = C[lr * gpr_n:(lr + 1) * gpr_n, :]
        dt = w[im + conv_dim:im + conv_dim + self.num_heads, :]
        dt_r = dt[lr * nh_pr:(lr + 1) * nh_pr, :]
        return torch.cat([gate, x_r, B_r, C_r, dt_r], dim=0)

    def _shard_conv(self, w, lr):
        # Mixed: PARALLELISM (per-rank slicing) + MODEL-SPECIFIC (x/B/C interleaved conv layout)
        im = self.intermediate_size
        gn = self.n_groups * self.ssm_state_size
        im_pr, gpr_n = self.im_pr, self.groups_pr * self.ssm_state_size
        x = w[:im]; B = w[im:im + gn]; C = w[im + gn:im + 2 * gn]
        return torch.cat([x[lr * im_pr:(lr + 1) * im_pr],
                          B[lr * gpr_n:(lr + 1) * gpr_n],
                          C[lr * gpr_n:(lr + 1) * gpr_n]], dim=0)

    def _shard_conv_bias(self, b, lr):
        # Mixed: PARALLELISM (per-rank slicing) + MODEL-SPECIFIC (x/B/C interleaved conv layout)
        im = self.intermediate_size
        gn = self.n_groups * self.ssm_state_size
        im_pr, gpr_n = self.im_pr, self.groups_pr * self.ssm_state_size
        x = b[:im]; B = b[im:im + gn]; C = b[im + gn:im + 2 * gn]
        return torch.cat([x[lr * im_pr:(lr + 1) * im_pr],
                          B[lr * gpr_n:(lr + 1) * gpr_n],
                          C[lr * gpr_n:(lr + 1) * gpr_n]], dim=0)

    def _gated_rmsnorm(self, y, gate):
        # NemotronH uses a GROUPED gated RMSNorm (HF MambaRMSNormGated, group_size =
        # mamba_intermediate/n_groups = 512): normalize WITHIN each group of `gsz`, not over the whole
        # inner dim. A plain mean(-1) over the full inner dim is a different function and degrades the
        # real 30B to repetitive output (single-layer CPU parity: grouped -> cosine 1.0 vs HF).
        y = y * self.act(gate)
        gsz = self.intermediate_size // self.n_groups
        yf = y.float().reshape(*y.shape[:-1], -1, gsz)
        yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        yf = yf.reshape(*y.shape)
        return (yf * self.norm_weight.float()).to(gate.dtype)

    def _in_proj(self, hidden_states):
        # hidden_states: [b, l, hidden]; in_proj_weight: [hidden, proj_out_pr]
        return hidden_states @ self.in_proj_weight

    def _split_proj(self, proj):
        off = 0
        gate = proj[..., off:off + self.im_pr]; off += self.im_pr
        xBC = proj[..., off:off + self.conv_dim_pr]; off += self.conv_dim_pr
        dt = proj[..., off:off + self.num_heads_pr]
        return gate, xBC, dt

    def forward_prefill(self, hidden_states, ssm_state0=None):
        b, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        proj = self._in_proj(hidden_states)
        gate, xBC, dt = self._split_proj(proj)
        xBC_t = xBC.transpose(1, 2)
        conv_state = xBC_t[..., -(self.conv_kernel_size - 1):]
        xBC_c = F.conv1d(xBC_t, self.conv1d_weight, self.conv1d_bias,
                         padding=self.conv_kernel_size - 1, groups=self.conv_dim_pr)[..., :seq_len]
        xBC = self.act(xBC_c.transpose(1, 2))

        gn = self.groups_pr * self.ssm_state_size
        im = self.im_pr
        x = xBC[..., :im]; B = xBC[..., im:im + gn]; C = xBC[..., im + gn:im + 2 * gn]
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        # NemotronH config has time_step_limit=(0.0, inf) => NO lower clamp. Clamping dt up to
        # time_step_min (0.001) perturbs small dt and, compounded over the 23 Mamba layers, degrades
        # generation into repetition. With no lower clamp the layer matches HF exactly (cosine 1.0,
        # max_abs_diff 2e-7 fp32); softplus already guarantees dt >= 0.
        H, P, N, G = self.num_heads_pr, self.head_dim, self.ssm_state_size, self.groups_pr
        rep = H // G
        x = x.reshape(b, seq_len, H, P).float()
        B = B.reshape(b, seq_len, G, N).float().repeat_interleave(rep, dim=2)
        C = C.reshape(b, seq_len, G, N).float().repeat_interleave(rep, dim=2)
        if os.environ.get("NEMOTRONH_MAMBA_STUB") == "1":
            # DIAGNOSTIC ONLY (never on by default): skip the SSM recurrence entirely.
            h = (torch.zeros(b, H, P, N, dtype=torch.float32, device=x.device)
                 if ssm_state0 is None else ssm_state0.float())
            y = x * self.D[..., None]
        elif os.environ.get("NEMOTRONH_SCAN") == "sequential":
            # Reference sequential recurrence. Correct and compiles via torch_neuronx.trace
            # (Stage0, cosine 0.9999), but the plugin's neuronx-cc path rejects the unrolled
            # carried-dependency loop with NCC_IFML902 FlattenMacroLoop. Kept as the numeric oracle.
            dA = torch.exp(dt.float() * A)
            h = (torch.zeros(b, H, P, N, dtype=torch.float32, device=x.device)
                 if ssm_state0 is None else ssm_state0.float())
            ys = []
            for t in range(seq_len):
                dBx = (dt[:, t].float()[..., None, None] * B[:, t][:, :, None, :]) * x[:, t][..., None]
                h = h * dA[:, t][..., None, None] + dBx
                ys.append((h * C[:, t][:, :, None, :]).sum(dim=-1))
            y = torch.stack(ys, dim=1) + x * self.D[..., None]
        elif os.environ.get("NEMOTRONH_SCAN") == "chunked":
            # Chunked SSD: O(l*C) so long sequences fit (lifts the quadratic form's short-seq cap).
            # The scan itself lives in module-level chunked_ssd_scan() — the SAME function the CPU
            # equivalence test imports, so implementation and test cannot silently diverge.
            cs = int(os.environ.get("NEMOTRONH_CHUNK", "128"))
            s0 = ssm_state0.float() if ssm_state0 is not None else None
            y, h = chunked_ssd_scan(x, B, C, dt.float(), A, self.D, cs, s0)
        else:
            # DEFAULT: vectorized SSD (quadratic / attention-form) selective scan. No Python time
            # loop and no carried-dependency chain — only matmuls + cumsum + a causal mask + a
            # bounded (exponent <= 0) decay, the same op shapes as attention (which compiles). This
            # is what lets Mamba compile on the plugin's neuronx-cc path WITHOUT native
            # torch_neuronx.trace delegation. Mathematically identical to the sequential recurrence:
            #   y_t[p] = sum_{s<=t} exp(A_h * (csdt_t - csdt_s)) * dt_s * (C_t . B_s) * x_s[p]
            # where csdt = cumsum(dt) and A_h < 0, so the decay exponent is <= 0 for s <= t (fp32
            # stable, no overflow). This closed form starts from a zero SSM state, so it does NOT
            # support a prefix ssm_state0 — guard against silently dropping it (use NEMOTRONH_SCAN=
            # chunked for continuation prefill).
            if ssm_state0 is not None:
                raise NotImplementedError(
                    "prefix ssm_state0 is not supported by the vectorized (quadratic) prefill scan; "
                    "use NEMOTRONH_SCAN=chunked for continuation prefill.")
            dtf = dt.float()                                        # [b,l,H]
            csdt = torch.cumsum(dtf, dim=1)                         # [b,l,H]
            At = (csdt * A).permute(0, 2, 1)                        # [b,H,l]  (A_h * csdt_t)
            ar = torch.arange(seq_len, device=x.device)
            causal_bool = (ar[:, None] >= ar[None, :])              # [l,l]  True where s <= t
            # exponent At_t - At_s: <= 0 for s <= t, but POSITIVE for s > t. On real dt the upper
            # triangle overflows exp() to +inf, and the subsequent causal multiply (inf * 0) would
            # yield NaN -> NaN logits -> argmax 0 (garbage). Mask s>t to -inf BEFORE exp so decay is
            # exactly 0 there. (Tiny random weights had small dt and never overflowed, hiding this.)
            exponent = At.unsqueeze(-1) - At.unsqueeze(-2)          # [b,H,l,l]  At_t - At_s
            exponent = exponent.masked_fill(~causal_bool.view(1, 1, seq_len, seq_len), float("-inf"))
            decay = torch.exp(exponent)                            # [b,H,l,l]  0 for s>t, no inf*0 NaN
            CB = torch.einsum('bthn,bshn->bhts', C, B)              # [b,H,l,l]  C_t . B_s
            dt_s = dtf.permute(0, 2, 1).unsqueeze(-2)               # [b,H,1,l]  weight at source s
            M = CB * decay * dt_s                                   # [b,H,l,l]  (causal baked into decay)
            y = torch.einsum('bhts,bshp->bthp', M, x)              # [b,l,H,P]
            y = y + x * self.D[..., None]
            # Final SSM state for decode carry:
            #   h_L[p,n] = sum_s exp(A_h * (csdt_L - csdt_s)) * dt_s * x_s[p] * B_s[n]
            wL = torch.exp(At[:, :, -1:] - At) * dtf.permute(0, 2, 1)   # [b,H,l]
            h = torch.einsum('bhs,bshp,bshn->bhpn', wL, x, B)      # [b,H,P,N]
        y = y.reshape(b, seq_len, -1)
        out = self._gated_rmsnorm(y, gate).to(dtype) @ self.out_proj_weight
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        # Persist the final SSM state and the last (K-1) conv inputs for the following decode steps
        # (in-place → aliased). conv_state = xBC_t[..., -(K-1):] captured before the conv above.
        self.ssm_state.copy_(h)
        self.conv_state.copy_(conv_state.to(self.conv_state.dtype))
        return out

    def forward_decode(self, hidden_states):
        b = hidden_states.shape[0]
        dtype = hidden_states.dtype
        proj = self._in_proj(hidden_states)
        gate, xBC, dt = self._split_proj(proj)
        xBC = xBC[:, 0]
        conv_in = torch.cat([self.conv_state, xBC[..., None]], dim=-1)   # [b, conv_dim_pr, K]
        w = self.conv1d_weight[:, 0, :]
        conv_o = (conv_in * w[None]).sum(-1)
        if self.conv1d_bias is not None:
            conv_o = conv_o + self.conv1d_bias
        xBC_new = self.act(conv_o)
        conv_state_new = conv_in[..., 1:]
        gn = self.groups_pr * self.ssm_state_size
        im = self.im_pr
        x = xBC_new[..., :im]; B = xBC_new[..., im:im + gn]; C = xBC_new[..., im + gn:im + 2 * gn]
        A = -torch.exp(self.A_log.float())
        dtv = F.softplus(dt[:, 0] + self.dt_bias)
        # No lower clamp: NemotronH time_step_limit=(0.0, inf) (see forward_prefill note).
        H, P, N, G = self.num_heads_pr, self.head_dim, self.ssm_state_size, self.groups_pr
        rep = H // G
        x = x.reshape(b, H, P).float()
        B = B.reshape(b, G, N).float().repeat_interleave(rep, dim=1)
        C = C.reshape(b, G, N).float().repeat_interleave(rep, dim=1)
        dA = torch.exp(dtv.float() * A)
        dBx = (dtv.float()[..., None, None] * B[:, :, None, :]) * x[..., None]
        h = self.ssm_state.float() * dA[..., None, None] + dBx
        y = (h * C[:, :, None, :]).sum(dim=-1) + x * self.D[..., None]
        y = y.reshape(b, 1, -1)
        out = self._gated_rmsnorm(y, gate).to(dtype) @ self.out_proj_weight
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        self.ssm_state.copy_(h)
        self.conv_state.copy_(conv_state_new.to(self.conv_state.dtype))
        return out


# ============================================================
# Decoder layer (heterogeneous by pattern)
# ============================================================
class NemotronHDecoderLayer(nn.Module):
    def __init__(self, config: NemotronHConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.norm = NemotronHRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        # <-- MODEL-SPECIFIC: hybrid layer-type dispatch (Mamba2/Attention/MoE), unique to this
        # hybrid-decoder architecture.
        if self.layer_type == MAMBA:
            self.mixer = NemotronHMamba2Mixer(config, layer_idx)
        elif self.layer_type == ATTENTION:
            self.mixer = NemotronHAttention(config, layer_idx)
        elif self.layer_type == MOE:
            self.mixer = NemotronHMoE(config, layer_idx)
        else:
            raise ValueError(f"unknown layer type {self.layer_type!r} at {layer_idx}")


# ============================================================
# Model (embedding + 52 layers + final norm)
# ============================================================
class NemotronHModel(nn.Module):
    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=config.hidden_size,
            dtype=config.torch_dtype, tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [NemotronHDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = NemotronHRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)

    def _is_prefill(self, attn_metadata):
        # <-- MODEL-SPECIFIC: use the first attention layer's metadata to decide prefill vs decode
        # (this hybrid model may have no attention-only global flag elsewhere).
        for i, layer in enumerate(self.layers):
            if layer.layer_type == ATTENTION:
                m = attn_metadata[f"layers.{i}.self_attn"]
                return m["max_query_len"] > m["decode_token_threshold"]
        return True

    def forward(self, input_ids, positions, attn_metadata):
        # Derive prefill/decode from attn_metadata (authoritative during compile warmup AND real
        # runs); a runner-supplied is_decode flag is unreliable at trace time (the decode bucket
        # compiles with 1 token but can arrive flagged prefill), which would wrongly SP-scatter it.
        is_prefill = self._is_prefill(attn_metadata)
        if is_prefill and self.world_size > 1:
            T = input_ids.shape[0]
            if T <= self.world_size or T % self.world_size != 0:
                raise ValueError(
                    f"Prefill token count ({T}) must be greater than, and a multiple of, "
                    f"tensor_parallel_size ({self.world_size}) for the sequence-parallel scatter at "
                    f"the embedding. Pad the prompt or adjust max_num_batched_tokens."
                )
        # embed (SP scatter on prefill mirrors plamo3). Keep the RESIDUAL STREAM in fp32
        # (config residual_in_fp32=True; HF NemotronH does the same). Over 52 hybrid layers the bf16
        # residual degenerates — real 30B weights produced all-zero tokens with a bf16 residual, and
        # fp32 residual is what the reference model uses. Each block: norm(residual)->bf16 for the
        # mixer weights, mixer out cast back to fp32 for the residual add.
        DT = self.config.torch_dtype
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill).to(torch.float32)
        # <-- MODEL-SPECIFIC: per-layer-type dispatch loop (Mamba2/Attention/MoE), unique to this
        # hybrid-decoder architecture.
        # Mamba runs on the full (un-SP) hidden states; for TP>1 we all_gather at the SSM boundary.
        # Recurrent state lives in each mamba mixer's buffers (batch=1): prefill writes it, decode
        # reads+updates it in place (aliased across steps). No runner-side state pool needed here.
        for i, layer in enumerate(self.layers):
            if layer.layer_type == ATTENTION:
                normed = layer.norm(hidden_states).to(DT)
                hidden_states = hidden_states + layer.mixer(normed, positions, attn_metadata).to(torch.float32)
            elif layer.layer_type == MOE:
                normed = layer.norm(hidden_states).to(DT)
                hidden_states = hidden_states + layer.mixer(normed, is_prefill=is_prefill).to(torch.float32)
            else:  # MAMBA
                h_norm = layer.norm(hidden_states).to(DT)
                if self.world_size > 1 and is_prefill:
                    # >>> PARALLELISM: all-gather to full hidden states at the Mamba/SSM boundary <<<
                    h_norm = self.tp_group.all_gather(h_norm, dim=0)
                # [T, H] -> [1, T, H] (batch=1 bring-up)
                h_b = h_norm.unsqueeze(0)
                if is_prefill:
                    out = layer.mixer.forward_prefill(h_b)
                else:
                    out = layer.mixer.forward_decode(h_b)
                out = out.squeeze(0)
                if self.world_size > 1 and is_prefill:
                    # >>> PARALLELISM: reduce-scatter back to SP shards after the Mamba/SSM boundary <<<
                    out = self.tp_group.reduce_scatter(out, dim=0)
                hidden_states = hidden_states + out.to(torch.float32)
        hidden_states = self.norm(hidden_states).to(DT)
        if is_prefill and self.world_size > 1:
            # >>> PARALLELISM: all-gather the final hidden states back to full sequence <<<
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# ============================================================
# ForCausalLM (top-level: model + lm_head + sampling)
# ============================================================
class NemotronHForCausalLM(nn.Module):
    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.model = NemotronHModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config if config.neuron_config else None
        )
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
        if self.on_device_sampling_config is not None:
            from vllm_neuron.nn.sampler import Sampler
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, text_neuron_config=None,
                     vision_neuron_config=None, **kwargs):
        # The runner passes text_neuron_config/vision_neuron_config for multimodal configs; we are the
        # TEXT backbone only, so accept the text config and ignore the vision one.
        nc = neuron_config if neuron_config is not None else text_neuron_config
        config = NemotronHConfig.from_configs(hf_config, nc)
        return cls(config)

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != ATTENTION:
                continue
            a = layer.mixer
            layers.append(LayerSpec(
                name=f"layers.{i}.self_attn",
                num_kv_heads=a.num_key_value_heads_per_rank,
                head_size=a.head_dim, dtype=a.dtype,
                sliding_window_size=a.window_size, chunk_size=None,
            ))
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != ATTENTION:
                continue
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise RuntimeError(f"KV cache for {name} not initialized")
            layer.mixer.k_cache = kv_caches[name][0]
            layer.mixer.v_cache = kv_caches[name][1]

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions,
                sampling_params, spec_decode_metadata=None, logit_mask=None, rank=None,
                inputs_embeds=None, is_token_ids=None, **kwargs):
        positions = positions.to(torch.int32)
        hidden_states = self.model(input_ids, positions, attn_metadata)
        sampled = torch.index_select(hidden_states, 0, sampling_positions)
        logits = self.lm_head(sampled)
        if self.on_device_sampling_config is None:
            return logits
        sampled_tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        return sampled_tokens, None

    def load_weights(self, checkpoint_path, device, cache_dir=None):
        """Map the HF checkpoint (text-only backbone.*, or the Omni wrapper's language_model.backbone.*) to our params.

        <-- MODEL-SPECIFIC: HF checkpoint key layout and prefix auto-detection are unique to
        NemotronH/Omni.

        HF layout (per layer_id):
          language_model.backbone.embeddings.weight
          language_model.backbone.layers.{i}.norm.weight
          MAMBA:  mixer.{in_proj,out_proj,conv1d.weight,conv1d.bias,A_log,D,dt_bias,norm}.weight
          MOE:    mixer.gate.weight, mixer.gate.e_score_correction_bias,
                  mixer.experts.{e}.{up_proj,down_proj}.weight,
                  mixer.shared_experts.{up_proj,down_proj}.weight
          ATTN:   mixer.{q_proj,k_proj,v_proj,o_proj}.weight
          language_model.backbone.norm_f.weight
          language_model.lm_head.weight
        """
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
        tp_rank, tp_size = self.rank, self.world_size
        # Auto-detect the checkpoint prefix. The Omni-wrapped checkpoint nests the language backbone
        # under `language_model.backbone.*` / `language_model.lm_head.weight`; the text-only NVIDIA
        # checkpoint (NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) uses `backbone.*` / `lm_head.weight`.
        # Probe the actual weight keys so both layouts load unchanged.
        _ckpt_keys = set()
        _idx = glob.glob(os.path.join(checkpoint_path, "*.index.json"))
        if _idx:
            with open(_idx[0]) as _fh:
                _ckpt_keys = set(json.load(_fh)["weight_map"].keys())
        else:
            from safetensors import safe_open as _safe_open
            for _sf in sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))):
                with _safe_open(_sf, framework="pt") as _f:
                    _ckpt_keys.update(_f.keys())
                break  # one shard is enough to detect the prefix
        if any(k.startswith("language_model.") for k in _ckpt_keys):
            SRC = "language_model.backbone"
            LM_HEAD_SRC = "language_model.lm_head.weight"
        else:
            SRC = "backbone"
            LM_HEAD_SRC = "lm_head.weight"
        mappings: dict = {}
        for i, layer in enumerate(self.model.layers):
            src = f"{SRC}.layers.{i}"
            mappings[f"model.layers.{i}.norm.weight"] = f"{src}.norm.weight"
            lt = layer.layer_type
            if lt == MAMBA:
                mappings[f"model.layers.{i}.mixer.in_proj_weight"] = f"{src}.mixer.in_proj.weight"
                mappings[f"model.layers.{i}.mixer.conv1d_weight"] = f"{src}.mixer.conv1d.weight"
                mappings[f"model.layers.{i}.mixer.conv1d_bias"] = f"{src}.mixer.conv1d.bias"
                mappings[f"model.layers.{i}.mixer.A_log"] = f"{src}.mixer.A_log"
                mappings[f"model.layers.{i}.mixer.D"] = f"{src}.mixer.D"
                mappings[f"model.layers.{i}.mixer.dt_bias"] = f"{src}.mixer.dt_bias"
                mappings[f"model.layers.{i}.mixer.norm_weight"] = f"{src}.mixer.norm.weight"
                mappings[f"model.layers.{i}.mixer.out_proj_weight"] = f"{src}.mixer.out_proj.weight"
            elif lt == ATTENTION:
                mappings[f"model.layers.{i}.mixer.qkv_proj_weight"] = [
                    f"{src}.mixer.q_proj.weight", f"{src}.mixer.k_proj.weight", f"{src}.mixer.v_proj.weight"]
                mappings[f"model.layers.{i}.mixer.o_proj_weight"] = f"{src}.mixer.o_proj.weight"
            elif lt == MOE:
                mappings[f"model.layers.{i}.mixer.gate_weight"] = f"{src}.mixer.gate.weight"
                mappings[f"model.layers.{i}.mixer.e_score_correction_bias"] = f"{src}.mixer.gate.e_score_correction_bias"
                mappings[f"model.layers.{i}.mixer.up"] = [
                    f"{src}.mixer.experts.{e}.up_proj.weight" for e in range(self.config.n_routed_experts)]
                mappings[f"model.layers.{i}.mixer.down"] = [
                    f"{src}.mixer.experts.{e}.down_proj.weight" for e in range(self.config.n_routed_experts)]
                mappings[f"model.layers.{i}.mixer.shared_up"] = f"{src}.mixer.shared_experts.up_proj.weight"
                mappings[f"model.layers.{i}.mixer.shared_down"] = f"{src}.mixer.shared_experts.down_proj.weight"
        mappings["model.embed_tokens.weight"] = f"{SRC}.embeddings.weight"
        mappings["model.norm.weight"] = f"{SRC}.norm_f.weight"
        mappings["lm_head.weight"] = LM_HEAD_SRC

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device, strict=False
        ).state_dict
        # load_sharded_pipelined already returns each tensor at its target parameter's dtype
        # (bf16 for weights, fp32 for gate_weight / e_score_correction_bias). No extra cast: on the
        # Neuron device tensor.to(dtype) asserts self.dtype()==dst.dtype() and would crash.
        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Materialise the mamba recurrent-state buffers on the real device. The runner builds the
        # model on `meta`; these buffers are runtime state (not in the checkpoint) so load_state_dict
        # leaves them on meta. Reassign to real zeros so the in-place decode updates work.
        for _layer in self.model.layers:
            _mx = _layer.mixer
            if hasattr(_mx, "ssm_state"):
                _mx.ssm_state = torch.zeros(_mx.ssm_state.shape, dtype=torch.float32, device=device)
                _mx.conv_state = torch.zeros(_mx.conv_state.shape, dtype=_mx.dtype, device=device)

        if os.environ.get("NEMOTRONH_DEBUG_LOAD", "0") == "1":
            model_params = dict(self.named_parameters())
            still_meta = [n for n, p in model_params.items() if getattr(p, "is_meta", False)]
            dst_keys = set(mappings.keys())
            param_keys = set(model_params.keys())
            logger.debug("load_weights: params=%d mappings=%d loaded=%d still_meta=%d",
                         len(param_keys), len(dst_keys), len(rank_sharded), len(still_meta))
            for n in sorted(still_meta)[:40]:
                logger.debug("  META: %s  shape=%s", n, tuple(model_params[n].shape))
            unmapped = sorted(param_keys - dst_keys)
            if unmapped:
                logger.debug("load_weights: param has NO mapping (%d):", len(unmapped))
                for n in unmapped[:40]:
                    logger.debug("  NOMAP: %s", n)
            bad_dst = sorted(dst_keys - param_keys)
            if bad_dst:
                logger.debug("load_weights: mapping dst not a param (%d):", len(bad_dst))
                for n in bad_dst[:40]:
                    logger.debug("  BADDST: %s", n)
