# SPDX-License-Identifier: Apache-2.0
"""NemotronH-specific numerical ops (torch-only, no Neuron/plugin deps).

SINGLE SOURCE OF TRUTH for the two non-attention/SSD reformulations: the grouped gated RMSNorm and
the DGE-free MoE top-k gate. The model (model_bf16.py) calls these, and the CPU equivalence test
imports them and compares against the ACTUAL Hugging Face reference (MambaRMSNormGated / rms_norm_ref
and NemotronHTopkRouter) — so the shipped code, the test, and the HF reference cannot silently drift.
"""
import torch
import torch.nn.functional as F


def gated_rmsnorm(y, gate, weight, group_size, eps):
    """Grouped gated RMSNorm — matches HF MambaRMSNormGated (rms_norm_ref, norm_before_gate=False):
    upcast, x = x * silu(gate), RMS-normalize WITHIN each group of `group_size`, then scale by weight.

    SiLU is computed on the fp32 gate (HF upcasts z before F.silu(z)); it is nonlinear, so doing it in
    bf16 vs fp32 moves the rounding. y[..., inner], gate[..., inner], weight[inner]. Returns gate.dtype.
    """
    y = y * F.silu(gate.float())
    yf = y.float().reshape(*y.shape[:-1], -1, group_size)
    yf = yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + eps)
    yf = yf.reshape(*y.shape)
    return (yf * weight.float()).to(gate.dtype)


def dense_moe_gate(scores, e_score_correction_bias, top_k, norm_topk_prob, routed_scaling_factor):
    """DGE-free dense top-k gate [T, E] from post-sigmoid router scores [T, E]. Selection uses the
    bias-corrected scores (lowest-index tie-break == torch.argmax); the WEIGHTS use the raw scores.

    Built from reductions + elementwise comparisons only — NO data-dependent scatter/gather — because
    neuronx-cc miscompiles the scatter/gather top-k ("vector DGE out-of-bound") once several MoE
    layers are stacked. Numerically equal to a scatter-based argmax top-k router (for the DeepSeek-style
    router with n_group == 1, which is NemotronH's config; group-limited routing is not implemented).
    """
    sfc = scores + e_score_correction_bias                         # scores_for_choice [T, E]
    T, E = sfc.shape
    iota = torch.arange(E, device=scores.device).view(1, E).to(sfc.dtype)
    big = torch.full_like(sfc, float(E))
    work = sfc
    sel_mask = torch.zeros_like(sfc)                               # [T, E], 1.0 at chosen experts
    for _ in range(top_k):
        m = work.max(dim=-1, keepdim=True).values
        is_max = work >= m
        idx_first = torch.where(is_max, iota, big).min(dim=-1, keepdim=True).values   # lowest-index max
        onehot = (iota == idx_first).to(sfc.dtype)
        sel_mask = sel_mask + onehot
        work = work - onehot * 1e30                                # drop the chosen expert (dense)
    gate = sel_mask * scores                                       # weights from RAW sigmoid scores
    if norm_topk_prob:
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-20)
    return gate * routed_scaling_factor
