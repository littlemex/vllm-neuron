# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE that runs each expert only on the tokens routed to it (nkilib blockwise MoE).

The dense prefill path runs all 128 experts over every token, 21x the work of the 6 experts a token
uses. nkilib's blockwise kernel (shard_on_I: the two physical cores split the intermediate axis)
groups the routed tokens into fixed-size blocks per expert and runs each block against its expert
only. Its relu^2 / up-only mode (skip_gate_proj + SquaredReLU) is exactly NemotronH's expert MLP.

One mismatch is bridged here: the kernel reads gate and up from one [E, H, 2, I] tensor and, with the
gate skipped, still addresses the up half with the row stride of that two-weight layout. NemotronH has
no gate weight, and padding a zero gate into the layout would double the expert memory (7.3 GB per
rank). The up weight is passed as [E, H, 1, I] instead, and the kernel's weight loader is replaced by
one that takes the row stride and the up index from the tensor's actual shape. The replacement is
installed into the kernel module before it is traced and only changes the addressing.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from nkilib.core.moe.moe_cte import bwmm_shard_on_I as _bwmm
from nkilib.core.moe.moe_cte.bwmm_shard_on_I import blockwise_mm_baseline_shard_intermediate
from nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

BLOCK_SIZE = 256          # tokens per block (the kernel requires a multiple of 256)


def _load_gate_up_weights(gate_or_up, gate_up_proj_weight, block_expert, cfg, num_shards, shard_id,
                          load_dst=None):
    """Drop-in for bwmm_shard_on_I.load_gate_up_proj_weights_shard_intermediate that also accepts an
    up-only [E, H, 1, I] weight: the row stride is n_weights * I and up is index n_weights - 1.
    Returns [h_outer][TILE_SIZE, h_inner, 1, I_TP_per_shard] tiles like the original."""
    _, H, n_weights, _I_TP = gate_up_proj_weight.shape
    if n_weights == 2 or cfg.fuse_gate_and_up_load:
        return _ORIGINAL_LOADER(gate_or_up, gate_up_proj_weight, block_expert, cfg, num_shards, shard_id,
                                load_dst)
    TILE_SIZE, PSUM_SIZE = _bwmm.TILE_SIZE, _bwmm.PSUM_SIZE
    I_TP_per_shard = _I_TP // num_shards
    I_TP_offset = I_TP_per_shard * shard_id
    h_outer = (H + PSUM_SIZE - 1) // PSUM_SIZE
    h_inner = PSUM_SIZE // TILE_SIZE
    index = gate_or_up - (2 - n_weights)          # up (gate_or_up = 1) is index 0 of an up-only tensor
    if load_dst == None:
        load_dst = []
        for h_i in range(h_outer):
            load_dst.append(nl.ndarray((TILE_SIZE, h_inner, 1, I_TP_per_shard), dtype=cfg.weight_dtype,
                                       buffer=nl.sbuf))
    for h_i in range(h_outer):
        for h_j in range(h_inner):
            p0 = PSUM_SIZE * h_i + TILE_SIZE * h_j
            if p0 >= H:
                nisa.memset(dst=load_dst[h_i][0:TILE_SIZE, h_j, 0:1, 0:I_TP_per_shard], value=0.0)
                continue
            n_p = min(TILE_SIZE, H - p0)
            if n_p < TILE_SIZE:
                nisa.memset(dst=load_dst[h_i][0:TILE_SIZE, h_j, 0:1, 0:I_TP_per_shard], value=0.0)
            nisa.dma_copy(
                dst=load_dst[h_i][0:n_p, h_j, 0:1, 0:I_TP_per_shard],
                src=gate_up_proj_weight.ap(
                    pattern=[[n_weights * _I_TP, n_p], [_I_TP, 1], [1, I_TP_per_shard]],
                    offset=p0 * (n_weights * _I_TP) + index * _I_TP + I_TP_offset,
                    scalar_offset=block_expert, indirect_dim=0),
                oob_mode=oob_mode.error)
    return load_dst


_ORIGINAL_LOADER = _bwmm.load_gate_up_proj_weights_shard_intermediate
_bwmm.load_gate_up_proj_weights_shard_intermediate = _load_gate_up_weights


@nki.jit(mode="trace")
def moe_relu2_prefill(hidden_states, expert_affinities_masked, up, down, token_position_to_id,
                      block_to_expert):
    """hidden_states [T, H] bf16, expert_affinities_masked [T*E, 1] (the token's routing weight for
    each expert, 0 if not routed), up [E, H, 1, I] bf16, down [E, I, H] bf16, token_position_to_id
    [N*B] int32 (-1 for an empty slot), block_to_expert [N] int32 -> [T, H] bf16:
    sum over the token's experts of weight * relu(x @ up_e)^2 @ down_e (fp32 accumulation)."""
    # (called by name: the NKI frontend rejects a module attribute lookup inside a kernel)
    return blockwise_mm_baseline_shard_intermediate(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=up,
        down_proj_weight=down,
        block_size=BLOCK_SIZE,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        activation_function=ActFnType.SquaredReLU,
        skip_dma=SkipMode(skip_token=True, skip_weight=False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        accumulation_dtype=nl.float32,
        skip_gate_proj=True,
    )
