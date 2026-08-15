# SPDX-License-Identifier: Apache-2.0
"""CPU numerical-equivalence tests for the NemotronH-specific kernels.

These validate the two non-trivial numerical reformulations the Neuron serving implementation
relies on, using plain PyTorch on CPU (no Neuron device / no HF weights required):

1. The vectorized SSD (quadratic / attention-form) Mamba2 prefill scan is mathematically
   equivalent to the sequential 1-step recurrence, and its "mask the decay exponent to -inf on
   the strict upper triangle BEFORE exp()" is required to avoid inf*0 = NaN on real-scale dt.
2. The DGE-free dense MoE router selects exactly the same experts (with the same lowest-index
   tie-break) and produces the same weights as a scatter-based argmax top-k router.

Full-model / HF-parity correctness is covered by on-device verification (see the model README);
these tests pin the numerics that a refactor could silently break.
"""
import torch
import torch.nn.functional as F


def _sequential_ssd(x, B, C, dt, A, D):
    """Oracle: the sequential 1-step Mamba2 recurrence. Shapes: x[b,l,H,P], B/C[b,l,H,N], dt[b,l,H]."""
    b, l, H, P = x.shape
    dA = torch.exp(dt * A)                                       # [b,l,H]
    h = torch.zeros(b, H, P, B.shape[-1], dtype=x.dtype)
    ys = []
    for t in range(l):
        dBx = (dt[:, t][..., None, None] * B[:, t][:, :, None, :]) * x[:, t][..., None]
        h = h * dA[:, t][..., None, None] + dBx
        ys.append((h * C[:, t][:, :, None, :]).sum(dim=-1))
    y = torch.stack(ys, dim=1) + x * D[..., None]
    return y, h


def _vectorized_ssd(x, B, C, dt, A, D, mask_before_exp=True):
    """The shipped vectorized SSD form (model_bf16.NemotronHMamba2Mixer.forward_prefill default)."""
    b, l, H, P = x.shape
    csdt = torch.cumsum(dt, dim=1)                              # [b,l,H]
    At = (csdt * A).permute(0, 2, 1)                            # [b,H,l]
    ar = torch.arange(l)
    causal = ar[:, None] >= ar[None, :]                        # [l,l] True where s<=t
    exponent = At.unsqueeze(-1) - At.unsqueeze(-2)             # [b,H,l,l] = At_t - At_s
    if mask_before_exp:
        exponent = exponent.masked_fill(~causal.view(1, 1, l, l), float("-inf"))
        decay = torch.exp(exponent)
    else:
        # The buggy ordering: exp first (upper triangle overflows to +inf on real dt), mask after.
        decay = torch.exp(exponent) * causal.view(1, 1, l, l).to(x.dtype)
    CB = torch.einsum('bthn,bshn->bhts', C, B)
    dt_s = dt.permute(0, 2, 1).unsqueeze(-2)                    # [b,H,1,l]
    M = CB * decay * dt_s
    y = torch.einsum('bhts,bshp->bthp', M, x) + x * D[..., None]
    wL = torch.exp(At[:, :, -1:] - At) * dt.permute(0, 2, 1)
    h = torch.einsum('bhs,bshp,bshn->bhpn', wL, x, B)
    return y, h


def test_vectorized_ssd_matches_sequential():
    torch.manual_seed(0)
    b, l, H, P, N = 1, 40, 4, 8, 6
    x = torch.randn(b, l, H, P, dtype=torch.float64)
    B = torch.randn(b, l, H, N, dtype=torch.float64)
    C = torch.randn(b, l, H, N, dtype=torch.float64)
    # Real-scale dt (softplus, no lower clamp — time_step_limit=(0.0, inf)).
    dt = F.softplus(torch.randn(b, l, H, dtype=torch.float64))
    A = -torch.exp(torch.randn(H, dtype=torch.float64))        # < 0
    D = torch.randn(H, dtype=torch.float64)

    y_seq, h_seq = _sequential_ssd(x, B, C, dt, A, D)
    y_vec, h_vec = _vectorized_ssd(x, B, C, dt, A, D, mask_before_exp=True)

    assert torch.allclose(y_seq, y_vec, atol=1e-9), (y_seq - y_vec).abs().max()
    assert torch.allclose(h_seq, h_vec, atol=1e-9), (h_seq - h_vec).abs().max()


def test_mask_before_exp_is_required_to_avoid_nan():
    """With real-scale dt the strict upper triangle overflows exp() to +inf; masking AFTER exp
    then multiplies inf*0 = NaN. Masking the exponent BEFORE exp keeps it finite."""
    torch.manual_seed(1)
    b, l, H, P, N = 1, 64, 2, 4, 4
    x = torch.randn(b, l, H, P, dtype=torch.float32)
    B = torch.randn(b, l, H, N, dtype=torch.float32)
    C = torch.randn(b, l, H, N, dtype=torch.float32)
    dt = F.softplus(torch.randn(b, l, H, dtype=torch.float32)) + 0.5   # push cumsum up to overflow
    A = -torch.exp(torch.randn(H, dtype=torch.float32))
    D = torch.randn(H, dtype=torch.float32)

    y_good, _ = _vectorized_ssd(x, B, C, dt, A, D, mask_before_exp=True)
    y_bad, _ = _vectorized_ssd(x, B, C, dt, A, D, mask_before_exp=False)
    assert not torch.isnan(y_good).any(), "mask-before-exp must not produce NaN"
    assert torch.isnan(y_bad).any(), "mask-after-exp is expected to produce NaN (documents the bug)"


def _dense_gate(scores, bias, k, norm_topk_prob, rsf):
    """The shipped DGE-free dense router (model_bf16.NemotronHMoE._gate_dense)."""
    sfc = scores + bias
    T, E = sfc.shape
    iota = torch.arange(E).view(1, E).to(sfc.dtype)
    big = torch.full_like(sfc, float(E))
    work = sfc.clone()
    sel_mask = torch.zeros_like(sfc)
    for _ in range(k):
        m = work.max(dim=-1, keepdim=True).values
        is_max = work >= m
        idx_first = torch.where(is_max, iota, big).min(dim=-1, keepdim=True).values
        onehot = (iota == idx_first).to(sfc.dtype)
        sel_mask = sel_mask + onehot
        work = work - onehot * 1e30
    gate = sel_mask * scores
    if norm_topk_prob:
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-20)
    return gate * rsf


def _scatter_gate(scores, bias, k, norm_topk_prob, rsf):
    """Reference: iterative-argmax top-k + scatter (lowest-index tie-break), the router the dense
    form replaces. Uses argmax (not torch.topk) so the tie-break is well-defined and comparable."""
    sfc = scores + bias
    T, E = sfc.shape
    work = sfc.clone()
    idxs = []
    for _ in range(k):
        idx = work.argmax(dim=-1)                              # argmax → lowest index on ties
        idxs.append(idx)
        work = work.scatter(1, idx.unsqueeze(1), float("-inf"))
    topk_idx = torch.stack(idxs, dim=1)
    w = scores.gather(1, topk_idx)
    if norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * rsf
    gate = torch.zeros(T, E, dtype=scores.dtype)
    gate.scatter_(1, topk_idx, w)
    return gate


def test_dense_router_matches_scatter_router():
    torch.manual_seed(2)
    T, E, k = 5, 128, 6
    logits = torch.randn(T, E, dtype=torch.float64)
    scores = logits.sigmoid()
    bias = torch.randn(E, dtype=torch.float64) * 0.1
    g_dense = _dense_gate(scores, bias, k, True, 2.5)
    g_scatter = _scatter_gate(scores, bias, k, True, 2.5)
    assert torch.allclose(g_dense, g_scatter, atol=1e-12), (g_dense - g_scatter).abs().max()


def test_dense_router_tiebreak_prefers_lowest_index():
    """When several experts share the max score, both routers must pick the lowest indices."""
    T, E, k = 1, 8, 3
    scores = torch.full((T, E), 0.5, dtype=torch.float64)      # all tied
    bias = torch.zeros(E, dtype=torch.float64)
    g_dense = _dense_gate(scores, bias, k, False, 1.0)
    g_scatter = _scatter_gate(scores, bias, k, False, 1.0)
    assert torch.allclose(g_dense, g_scatter)
    # exactly the k lowest indices selected
    assert (g_dense[0, :k] > 0).all() and (g_dense[0, k:] == 0).all()


if __name__ == "__main__":
    test_vectorized_ssd_matches_sequential()
    test_mask_before_exp_is_required_to_avoid_nan()
    test_dense_router_matches_scatter_router()
    test_dense_router_tiebreak_prefers_lowest_index()
    print("all NemotronH kernel equivalence tests passed")
