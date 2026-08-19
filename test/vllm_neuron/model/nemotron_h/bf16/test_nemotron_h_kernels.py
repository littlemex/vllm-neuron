# SPDX-License-Identifier: Apache-2.0
"""CPU numerical-equivalence tests for the NemotronH-specific kernels.

These validate the NemotronH-specific numerical reformulations on CPU (no Neuron device / no HF
weights required), using plain PyTorch:

1. chunked SSD == the sequential 1-step recurrence (chunk boundaries, long seq, small chunks, prefix
   state, fp32 stress); mask-before-exp is required to avoid inf*0 = NaN.
2. segmented / continuation prefill == single-shot (SSM + conv1d state carry, first-segment mask,
   segments shorter than the conv kernel), and bucket-padding does not change the state
   (test_prefill_pad_invariance) — with negative tests that a missing mask diverges.
3. The DGE-free dense MoE router and the grouped gated RMSNorm match the ACTUAL Hugging Face
   reference (NemotronHTopkRouter / MambaRMSNormGated rms_norm_ref), not just a self-authored oracle.

Full-model greedy correctness and the segmented long-context path are additionally covered by
on-device verification (see the model README); these tests pin the numerics a refactor could break.
"""
import torch
import torch.nn.functional as F
import importlib.util
import os as _os

# Load the SHIPPED scan directly from ssd.py (single source of truth). Loading the file avoids the
# plugin package __init__ chain (Neuron-only imports), so this stays CPU/pytest-friendly.
_here = _os.path.dirname(_os.path.abspath(__file__))
_ssd_path = _here
for _ in range(8):
    _cand = _os.path.join(_ssd_path, "vllm_neuron", "model", "nemotron_h", "ssd.py")
    if _os.path.exists(_cand):
        break
    _ssd_path = _os.path.dirname(_ssd_path)
_spec = importlib.util.spec_from_file_location("nemotron_ssd", _cand)
_ssd = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ssd)
chunked_ssd_scan = _ssd.chunked_ssd_scan
segmented_causal_conv1d = _ssd.segmented_causal_conv1d
_ops_spec = importlib.util.spec_from_file_location(
    "nemotron_ops", _os.path.join(_os.path.dirname(_cand), "ops.py"))
_ops = importlib.util.module_from_spec(_ops_spec); _ops_spec.loader.exec_module(_ops)
gated_rmsnorm = _ops.gated_rmsnorm
dense_moe_gate = _ops.dense_moe_gate


def _sequential_ssd(x, B, C, dt, A, D, state0=None):
    """Oracle: the sequential 1-step Mamba2 recurrence. Shapes: x[b,l,H,P], B/C[b,l,H,N], dt[b,l,H].
    Optional prefix state0[b,H,P,N] (0 if None)."""
    b, l, H, P = x.shape
    dA = torch.exp(dt * A)                                       # [b,l,H]
    h = torch.zeros(b, H, P, B.shape[-1], dtype=x.dtype) if state0 is None else state0.clone()
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


def test_chunked_ssd_matches_sequential():
    """Chunked SSD == sequential oracle across chunk boundaries, long sequences, and prefix state."""
    # incl. small chunk sizes that force many chunks at short l (cs=16 -> T=2 at l=32; cs=8 -> T=5
    # at l=40), which exercise the cross-chunk closed form at boundaries an on-device short
    # max_model_len run hits.
    for l, cs in [(40, 128), (128, 128), (130, 128), (256, 128), (1024, 128),
                  (1000, 128), (4096, 256), (300, 64), (32, 16), (32, 8), (40, 8), (17, 16)]:
        for use_state0 in (False, True):
            torch.manual_seed(l * 100 + cs + int(use_state0))
            b, H, P, N = 1, 4, 8, 6
            x = torch.randn(b, l, H, P, dtype=torch.float64)
            B = torch.randn(b, l, H, N, dtype=torch.float64)
            C = torch.randn(b, l, H, N, dtype=torch.float64)
            dt = F.softplus(torch.randn(b, l, H, dtype=torch.float64))
            A = -torch.exp(torch.randn(H, dtype=torch.float64))
            D = torch.randn(H, dtype=torch.float64)
            s0 = torch.randn(b, H, P, N, dtype=torch.float64) if use_state0 else None
            y_seq, h_seq = _sequential_ssd(x, B, C, dt, A, D, s0)
            y_ch, h_ch = chunked_ssd_scan(x, B, C, dt, A, D, cs, s0)
            assert torch.allclose(y_seq, y_ch, atol=1e-8), (l, cs, use_state0, (y_seq - y_ch).abs().max())
            assert torch.allclose(h_seq, h_ch, atol=1e-8), (l, cs, use_state0, (h_seq - h_ch).abs().max())


def _mamba_prefill_core(xBC_t, conv_w, conv_b, dt, A, D, K, cs, im, gn, G, ssm_state0, conv_state0):
    """Faithful CPU replica of NemotronHMamba2Mixer.forward_prefill's conv + SSD pipeline, used as a
    segmented-vs-single-shot oracle. Mirrors the shipped ops (depthwise causal conv1d, SiLU, split
    x/B/C, chunked_ssd_scan). conv_state0 is the previous segment's last (K-1) RAW conv inputs (None
    on the first segment => zero left-pad, i.e. single-shot start). Returns
    (y[b,l,H,P], h_final[b,H,P,N], conv_state_out[b,conv_dim,K-1])."""
    b, conv_dim, seq = xBC_t.shape
    # Use the SHIPPED conv-carry helper (single source of truth): is_continuation=1 on a continuation
    # segment, None conv_state0 => single-shot / first segment (fresh zero left-pad).
    is_cont = None if conv_state0 is None else torch.ones((), dtype=xBC_t.dtype)
    xBC_c, conv_state_out = segmented_causal_conv1d(xBC_t, conv_w, conv_b, K, conv_dim,
                                                    conv_state=conv_state0, is_continuation=is_cont)
    xBC = F.silu(xBC_c.transpose(1, 2))                              # [b, seq, conv_dim]
    P = im // (A.shape[0])
    H, N = A.shape[0], gn // G
    rep = H // G
    x = xBC[..., :im].reshape(b, seq, H, P)
    B = xBC[..., im:im + gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    C = xBC[..., im + gn:im + 2 * gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    y, h = chunked_ssd_scan(x, B, C, dt, A, D, cs, ssm_state0)
    return y, h, conv_state_out


def test_segmented_prefill_matches_single_shot():
    """A prefill split into segments (carrying BOTH the SSM state AND the causal-conv1d state across
    the boundary) is numerically identical to a single-shot prefill. This pins the segmented /
    continuation prefill path (NemotronHMamba2Mixer.forward_prefill cont_prefill=True), in particular
    that the conv1d history is carried (not zero-padded) at every segment boundary."""
    # incl. boundary cases where a segment (or the whole prompt) is SHORTER than the conv kernel
    # history K-1=3 — the conv_state carry must still be exactly K-1 wide (no silent broadcast).
    for L, splits, cs in [(40, [24], 16), (40, [16, 28], 16), (256, [128], 128), (300, [128, 200], 128),
                          (5, [2], 16), (5, [1, 3], 16), (2, [1], 16), (1, [], 16)]:
        torch.manual_seed(L * 7 + cs + len(splits))
        b, H, P, N, G, K = 1, 4, 8, 6, 2, 4
        im, gn = H * P, G * N
        conv_dim = im + 2 * gn
        xBC_t = torch.randn(b, conv_dim, L, dtype=torch.float64)
        conv_w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
        conv_b = torch.randn(conv_dim, dtype=torch.float64)
        dt = F.softplus(torch.randn(b, L, H, dtype=torch.float64))
        A = -torch.exp(torch.randn(H, dtype=torch.float64))
        D = torch.randn(H, dtype=torch.float64)

        # single-shot over the whole sequence
        y_full, h_full, _ = _mamba_prefill_core(xBC_t, conv_w, conv_b, dt, A, D, K, cs, im, gn, G,
                                                 ssm_state0=None, conv_state0=None)

        # the conv state carried out of single-shot must always be exactly K-1 wide, even for L<K-1.
        _, _, cs_full = _mamba_prefill_core(xBC_t, conv_w, conv_b, dt, A, D, K, cs, im, gn, G, None, None)
        assert cs_full.shape[-1] == K - 1, (L, cs_full.shape)

        # segmented: walk boundaries carrying ssm_state (h) and conv_state (raw last K-1 inputs)
        bounds = [0] + splits + [L]
        ys, ssm, conv = [], None, None
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            y_seg, ssm, conv = _mamba_prefill_core(
                xBC_t[..., s:e], conv_w, conv_b, dt[:, s:e], A, D, K, cs, im, gn, G,
                ssm_state0=ssm, conv_state0=conv)
            assert conv.shape[-1] == K - 1, (L, splits, i, conv.shape)   # never a too-short broadcast
            ys.append(y_seg)
        y_seg = torch.cat(ys, dim=1)
        assert torch.allclose(y_full, y_seg, atol=1e-8), (L, splits, (y_full - y_seg).abs().max())
        assert torch.allclose(h_full, ssm, atol=1e-8), (L, splits, (h_full - ssm).abs().max())


def _masked_prefill_core(xBC_t, conv_w, conv_b, dt, A, D, K, cs, im, gn, G, valid_len):
    """Mirror of forward_prefill's pad-masked pipeline: conv with valid_len tail, dt zeroed on pad,
    chunked scan. valid_len = number of REAL tokens (rest is bucket padding)."""
    b, conv_dim, seq = xBC_t.shape
    vl = torch.tensor(valid_len)
    is_real = (torch.arange(seq) < valid_len).to(xBC_t.dtype)
    xBC_c, conv_state = segmented_causal_conv1d(xBC_t, conv_w, conv_b, K, conv_dim, valid_len=vl)
    xBC = F.silu(xBC_c.transpose(1, 2))
    P = im // A.shape[0]; H, N = A.shape[0], gn // G; rep = H // G
    x = xBC[..., :im].reshape(b, seq, H, P)
    B = xBC[..., im:im + gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    C = xBC[..., im + gn:im + 2 * gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    dt_m = dt * is_real.reshape(1, seq, 1)
    y, h = chunked_ssd_scan(x, B, C, dt_m, A, D, cs, None)
    return y, h, conv_state


def test_prefill_pad_invariance():
    """Bucket padding must not change the Mamba state. Neuron pads a prefill to a fixed bucket width
    with nonzero pad tokens; without a pad mask the SSM/conv would fold them in and corrupt the state
    handed to decode. Pin that a run over `n` real tokens padded to `N` (dt zeroed on pad, conv tail
    gathered at the real end) yields the SAME y[:n], final ssm state, and conv_state as the unpadded
    n-token run — dt_bias is nonzero so a missing mask would visibly diverge."""
    for n, N, cs in [(40, 128, 16), (85, 512, 128), (2, 128, 16), (300, 512, 128)]:
        torch.manual_seed(n * 31 + N)
        b, H, P, Nst, G, K = 1, 4, 8, 6, 2, 4
        im, gn = H * P, G * Nst
        conv_dim = im + 2 * gn
        conv_w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
        conv_b = torch.randn(conv_dim, dtype=torch.float64)
        A = -torch.exp(torch.randn(H, dtype=torch.float64))
        D = torch.randn(H, dtype=torch.float64)
        dt_bias = torch.randn(H, dtype=torch.float64)                    # nonzero: pad dt != 0 without mask
        xBC_real = torch.randn(b, conv_dim, n, dtype=torch.float64)
        dt_real = torch.randn(b, n, H, dtype=torch.float64)
        # padded: real tokens then N-n arbitrary (nonzero) pad columns
        xBC_pad = torch.cat([xBC_real, torch.randn(b, conv_dim, N - n, dtype=torch.float64)], dim=-1)
        dt_pad = torch.cat([dt_real, torch.randn(b, N - n, H, dtype=torch.float64)], dim=1)
        y_r, h_r, cs_r = _masked_prefill_core(xBC_real, conv_w, conv_b,
                                              F.softplus(dt_real + dt_bias), A, D, K, cs, im, gn, G, n)
        y_p, h_p, cs_p = _masked_prefill_core(xBC_pad, conv_w, conv_b,
                                              F.softplus(dt_pad + dt_bias), A, D, K, cs, im, gn, G, n)
        assert torch.allclose(y_r, y_p[:, :n], atol=1e-8), (n, N, (y_r - y_p[:, :n]).abs().max())
        assert torch.allclose(h_r, h_p, atol=1e-8), (n, N, (h_r - h_p).abs().max())
        assert torch.allclose(cs_r, cs_p, atol=1e-8), (n, N, (cs_r - cs_p).abs().max())


def _seg_step(xBC_t, conv_w, conv_b, dt_soft, A, D, K, cs, im, gn, G, ssm_state0, conv_state0, valid_len):
    """One segmented-prefill step mirroring forward_prefill: conv history carry (valid_len tail),
    dt masked on pad, chunked scan with ssm_state0. valid_len = real tokens in THIS segment."""
    b, conv_dim, seq = xBC_t.shape
    is_cont = None if conv_state0 is None else torch.ones((), dtype=xBC_t.dtype)
    xBC_c, conv_out = segmented_causal_conv1d(xBC_t, conv_w, conv_b, K, conv_dim,
                                              conv_state=conv_state0, is_continuation=is_cont,
                                              valid_len=torch.tensor(valid_len))
    xBC = F.silu(xBC_c.transpose(1, 2))
    P = im // A.shape[0]; H, N = A.shape[0], gn // G; rep = H // G
    x = xBC[..., :im].reshape(b, seq, H, P)
    B = xBC[..., im:im + gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    C = xBC[..., im + gn:im + 2 * gn].reshape(b, seq, G, N).repeat_interleave(rep, dim=2)
    dt_m = dt_soft * (torch.arange(seq) < valid_len).to(xBC_t.dtype).reshape(1, seq, 1)
    y, h = chunked_ssd_scan(x, B, C, dt_m, A, D, cs, ssm_state0)
    return y, h, conv_out


def test_segmented_prefill_pad_invariance():
    """Production path: segmented (continuation carrying nonzero ssm/conv state) AND the last segment
    bucket-padded. Must equal a single-shot prefill over the concatenated REAL tokens — pins that the
    conv carry, the SSM state carry, and the pad mask compose correctly (the combo the earlier tests
    exercised only separately)."""
    for L0, L1r, seg, cs in [(300, 91, 512, 128), (128, 5, 512, 128), (64, 40, 128, 16)]:
        torch.manual_seed(L0 * 17 + L1r + seg)
        b, H, P, N, G, K = 1, 4, 8, 6, 2, 4
        im, gn = H * P, G * N; conv_dim = im + 2 * gn
        conv_w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
        conv_b = torch.randn(conv_dim, dtype=torch.float64)
        A = -torch.exp(torch.randn(H, dtype=torch.float64)); D = torch.randn(H, dtype=torch.float64)
        dt_bias = torch.randn(H, dtype=torch.float64)
        Lr = L0 + L1r                                                   # total real tokens
        xBC_real = torch.randn(b, conv_dim, Lr, dtype=torch.float64)
        dt_real = torch.randn(b, Lr, H, dtype=torch.float64)
        # single-shot over all real tokens (reference); no padding, valid_len=Lr
        y_full, h_full, _ = _masked_prefill_core(xBC_real, conv_w, conv_b,
                                                 F.softplus(dt_real + dt_bias), A, D, K, cs, im, gn, G, Lr)
        # segment 0: first L0 real (full, no pad). segment 1: remaining L1r real + pad to `seg`.
        s0x, s1x = xBC_real[..., :L0], xBC_real[..., L0:]
        s1x = torch.cat([s1x, torch.randn(b, conv_dim, seg - L1r, dtype=torch.float64)], dim=-1)  # pad
        s0dt = F.softplus(dt_real[:, :L0] + dt_bias)
        s1dt = F.softplus(torch.cat([dt_real[:, L0:], torch.randn(b, seg - L1r, H, dtype=torch.float64)], dim=1) + dt_bias)
        y0, ssm, conv = _seg_step(s0x, conv_w, conv_b, s0dt, A, D, K, cs, im, gn, G, None, None, L0)
        y1, ssm, conv = _seg_step(s1x, conv_w, conv_b, s1dt, A, D, K, cs, im, gn, G, ssm, conv, L1r)
        y_seg = torch.cat([y0, y1[:, :L1r]], dim=1)                     # keep only real outputs
        assert torch.allclose(y_full, y_seg, atol=1e-8), (L0, L1r, (y_full - y_seg).abs().max())
        assert torch.allclose(h_full, ssm, atol=1e-8), (L0, L1r, (h_full - ssm).abs().max())


def test_prefill_pad_contamination_without_mask():
    """Guard: if the pad mask / valid_len tail were dropped (process all N as real), the state must
    visibly diverge from the unpadded run — documents why the mask is required."""
    torch.manual_seed(7)
    b, H, P, Nst, G, K, n, N, cs = 1, 4, 8, 6, 2, 4, 85, 512, 128
    im, gn = H * P, G * Nst; conv_dim = im + 2 * gn
    conv_w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
    conv_b = torch.randn(conv_dim, dtype=torch.float64)
    A = -torch.exp(torch.randn(H, dtype=torch.float64)); D = torch.randn(H, dtype=torch.float64)
    dt_bias = torch.randn(H, dtype=torch.float64)
    xBC_real = torch.randn(b, conv_dim, n, dtype=torch.float64)
    dt_real = torch.randn(b, n, H, dtype=torch.float64)
    xBC_pad = torch.cat([xBC_real, torch.randn(b, conv_dim, N - n, dtype=torch.float64)], dim=-1)
    dt_pad = torch.cat([dt_real, torch.randn(b, N - n, H, dtype=torch.float64)], dim=1)
    _, h_real, _ = _masked_prefill_core(xBC_real, conv_w, conv_b, F.softplus(dt_real + dt_bias), A, D, K, cs, im, gn, G, n)
    # BUG path: no mask -> valid_len=N (treat all padded tokens as real)
    _, h_bad, _ = _masked_prefill_core(xBC_pad, conv_w, conv_b, F.softplus(dt_pad + dt_bias), A, D, K, cs, im, gn, G, N)
    assert not torch.allclose(h_real, h_bad, atol=1e-6), "processing pad as real must diverge (documents the pitfall)"


def test_segmented_prefill_needs_conv_carry():
    """Guard the top segmented-Mamba pitfall: dropping the conv1d state carry (zero-padding each
    segment) must visibly diverge from single-shot, so a regression that forgets it is caught."""
    torch.manual_seed(11)
    b, H, P, N, G, K, L = 1, 4, 8, 6, 2, 4, 64
    im, gn = H * P, G * N
    conv_dim = im + 2 * gn
    xBC_t = torch.randn(b, conv_dim, L, dtype=torch.float64)
    conv_w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
    conv_b = torch.randn(conv_dim, dtype=torch.float64)
    dt = F.softplus(torch.randn(b, L, H, dtype=torch.float64))
    A = -torch.exp(torch.randn(H, dtype=torch.float64))
    D = torch.randn(H, dtype=torch.float64)
    y_full, _, _ = _mamba_prefill_core(xBC_t, conv_w, conv_b, dt, A, D, K, 16, im, gn, G, None, None)
    # segment 2 WITHOUT conv carry (conv_state0=None on the continuation) but WITH ssm carry
    y0, ssm, _ = _mamba_prefill_core(xBC_t[..., :32], conv_w, conv_b, dt[:, :32], A, D, K, 16, im, gn, G, None, None)
    y1, _, _ = _mamba_prefill_core(xBC_t[..., 32:], conv_w, conv_b, dt[:, 32:], A, D, K, 16, im, gn, G,
                                   ssm_state0=ssm, conv_state0=None)   # BUG: no conv carry
    y_bad = torch.cat([y0, y1], dim=1)
    assert not torch.allclose(y_full, y_bad, atol=1e-6), "dropping conv carry must diverge (documents the pitfall)"


def test_segmented_conv_first_segment_mask_equals_fresh():
    """The graph-static first-segment path: on-device the FIRST segment still takes the continuation
    form (conv_state != None) but with is_continuation == 0, which must zero the carried history and
    reproduce a fresh single-shot conv EXACTLY. This is the runtime-mask branch the model relies on
    (forward_prefill), and it is otherwise only exercised implicitly — pin it directly."""
    for K, seq, conv_dim in [(4, 40, 8), (4, 2, 8), (4, 1, 8)]:   # incl. seq < K-1
        torch.manual_seed(seq * 13 + conv_dim)
        b = 1
        xBC_t = torch.randn(b, conv_dim, seq, dtype=torch.float64)
        w = torch.randn(conv_dim, 1, K, dtype=torch.float64)
        bias = torch.randn(conv_dim, dtype=torch.float64)
        prev = torch.randn(b, conv_dim, K - 1, dtype=torch.float64)   # arbitrary non-zero history
        out_fresh, cs_fresh = segmented_causal_conv1d(xBC_t, w, bias, K, conv_dim)   # single-shot
        out_first, cs_first = segmented_causal_conv1d(
            xBC_t, w, bias, K, conv_dim, conv_state=prev,
            is_continuation=torch.zeros((), dtype=torch.float64))    # first segment: mask == 0
        assert torch.allclose(out_fresh, out_first, atol=1e-12), (seq, (out_fresh - out_first).abs().max())
        assert torch.allclose(cs_fresh, cs_first, atol=1e-12)
        assert cs_first.shape[-1] == K - 1


def test_chunked_ssd_fp32_long_sequence_stress():
    """fp32 (the production dtype) at long sequences with small AND large dt, vs the fp32 sequential
    oracle. Guards catastrophic cancellation / underflow in the cross-chunk cumsum(log gamma)."""
    for l, cs, dt_scale in [(2048, 128, 1.0), (4096, 256, 0.05), (4096, 256, 3.0), (8192, 256, 1.0)]:
        torch.manual_seed(l + cs + int(dt_scale * 100))
        b, H, P, N = 1, 4, 8, 6
        x = torch.randn(b, l, H, P, dtype=torch.float32)
        B = torch.randn(b, l, H, N, dtype=torch.float32)
        C = torch.randn(b, l, H, N, dtype=torch.float32)
        dt = F.softplus(torch.randn(b, l, H, dtype=torch.float32)) * dt_scale
        A = -torch.exp(torch.randn(H, dtype=torch.float32))
        D = torch.randn(H, dtype=torch.float32)
        y_seq, h_seq = _sequential_ssd(x, B, C, dt, A, D)
        y_ch, h_ch = chunked_ssd_scan(x, B, C, dt, A, D, cs)
        assert torch.isfinite(y_ch).all() and torch.isfinite(h_ch).all(), (l, cs, dt_scale)
        # fp32 accumulation over thousands of steps: compare with a relaxed atol/rtol.
        assert torch.allclose(y_seq, y_ch, atol=2e-3, rtol=2e-3), (l, cs, dt_scale, (y_seq - y_ch).abs().max())


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
    test_chunked_ssd_matches_sequential()
    test_chunked_ssd_fp32_long_sequence_stress()
    test_mask_before_exp_is_required_to_avoid_nan()
    test_dense_router_matches_scatter_router()
    test_dense_router_tiebreak_prefers_lowest_index()
    print("all NemotronH kernel equivalence tests passed")


# ============================================================
# HF-reference parity: compare ops.py against the ACTUAL Hugging Face NemotronH algorithms
# (equivalence-skill Stage-2 rule: check the target against the real reference, not a self-authored
# oracle). The two HF functions below are copied VERBATIM from the reference so the test has no
# mamba_ssm / transformers-internal import dependency:
#   - _hf_rms_norm_ref: mamba_ssm layernorm_gated.rms_norm_ref (what HF MambaRMSNormGated calls),
#     einops.rearrange replaced by an equivalent reshape.
#   - _hf_topk_router_gate: transformers modeling_nemotron_h.NemotronHTopkRouter.get_topk_indices +
#     forward, scattered into a dense [T, E] gate for comparison.
# ============================================================

def _hf_rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    # verbatim from mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref (rearrange -> reshape)
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = x.reshape(*x.shape[:-1], -1, group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).reshape(*x.shape) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def test_gated_rmsnorm_matches_hf_reference():
    """ops.gated_rmsnorm == HF MambaRMSNormGated (rms_norm_ref, norm_before_gate=False) — the actual
    reference, not a re-derivation. fp64 exact; bf16 within a rounding tolerance."""
    for inner, group_size, dt in [(4096, 512, torch.float64), (1024, 512, torch.float64),
                                  (2048, 256, torch.float64), (4096, 512, torch.bfloat16)]:
        torch.manual_seed(inner + group_size + int(dt == torch.bfloat16))
        T = 7
        y = torch.randn(1, T, inner, dtype=dt)
        gate = torch.randn(1, T, inner, dtype=dt)
        weight = torch.randn(inner, dtype=dt)
        eps = 1e-5
        ours = gated_rmsnorm(y, gate, weight, group_size, eps)
        hf = _hf_rms_norm_ref(y, weight, bias=None, z=gate, eps=eps, group_size=group_size,
                              norm_before_gate=False, upcast=True)
        atol = 1e-10 if dt == torch.float64 else 3e-2
        assert torch.allclose(ours, hf, atol=atol), (inner, group_size, dt, (ours.float() - hf.float()).abs().max())


def _hf_topk_router_gate(scores, bias, top_k, n_group, topk_group, norm_topk_prob, rsf):
    # verbatim from transformers modeling_nemotron_h.py::NemotronHTopkRouter (get_topk_indices +
    # forward's weight path), then scattered into a dense [T, E] gate for comparison.
    n_routed = scores.shape[-1]
    scores_for_choice = scores + bias.unsqueeze(0)
    group_scores = (scores_for_choice.view(-1, n_group, n_routed // n_group)
                    .topk(2, dim=-1)[0].sum(dim=-1))
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(-1, n_group, n_routed // n_group)
                  .reshape(-1, n_routed))
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * rsf
    dense = torch.zeros_like(scores)
    dense.scatter_(1, topk_indices, topk_weights)
    return dense


def test_dense_moe_gate_matches_hf_router():
    """ops.dense_moe_gate == HF NemotronHTopkRouter (n_group=1, NemotronH's config). Random logits
    (no exact ties) so the DGE-free lowest-index selection and HF's torch.topk pick the same experts;
    weights/normalization/scaling then match exactly."""
    for E, k, norm, rsf in [(128, 6, True, 2.5), (128, 6, False, 2.5), (32, 4, True, 1.0), (16, 2, True, 3.0)]:
        torch.manual_seed(E * 3 + k)
        T = 5
        logits = torch.randn(T, E, dtype=torch.float64)
        scores = logits.sigmoid()
        bias = torch.randn(E, dtype=torch.float64) * 0.1
        ours = dense_moe_gate(scores, bias, k, norm, rsf)
        hf = _hf_topk_router_gate(scores, bias, top_k=k, n_group=1, topk_group=1,
                                  norm_topk_prob=norm, rsf=rsf)
        assert torch.allclose(ours, hf, atol=1e-12), (E, k, norm, (ours - hf).abs().max())
