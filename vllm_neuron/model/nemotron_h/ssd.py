# SPDX-License-Identifier: Apache-2.0
"""NemotronH Mamba2 SSD prefill scans (torch-only, no Neuron/plugin deps).

Kept in a dedicated, dependency-light module so it is the SINGLE SOURCE OF TRUTH: the model
(model_bf16.py) imports the scan from here, and the CPU equivalence test loads THIS file directly —
so the shipped implementation and the tested one can never silently diverge.
"""
import torch
import torch.nn.functional as F


def segmented_causal_conv1d(xBC_t, conv_weight, conv_bias, kernel_size, groups,
                            conv_state=None, is_continuation=None, valid_len=None):
    """Depthwise causal conv1d for (segmented) Mamba2 prefill. SINGLE SOURCE OF TRUTH shared by the
    model (NemotronHMamba2Mixer.forward_prefill) and the CPU test, so the shipped conv-history carry
    and the tested one cannot silently diverge.

    xBC_t[b, conv_dim, seq] (conv_dim to avoid colliding with the Mamba2 C matrix). Returns
    (conv_out[b, conv_dim, seq], new_conv_state[b, conv_dim, kernel_size-1]) where new_conv_state is
    the last (kernel_size-1) RAW conv inputs carried to the next segment / decode.

    conv_state is None for a single-shot prefill (fresh: a kernel_size-1 zero left-pad). For a
    segment, conv_state is the previous segment's new_conv_state and is_continuation is a runtime
    {0,1} mask (0 on the FIRST segment): the history is `conv_state * is_continuation`, so the first
    segment degenerates to the fresh zero-left-pad and a continuation segment prepends the real
    history. GRAPH-STATIC: no Python branch on a runtime value — is_continuation is tensor arithmetic.

    valid_len (runtime scalar tensor) is the number of REAL tokens; Neuron pads a prefill up to a
    fixed bucket width with a contiguous suffix of pad tokens, so `seq` can exceed the real length.
    new_conv_state must be the last (kernel_size-1) real conv inputs, NOT the padded tail. It is
    gathered from cat(history, xBC_t) at indices `valid_len + [0..K-2]`: for a full (unpadded) segment
    valid_len==seq and this is the last K-1 of xBC_t; for a padded segment it is the real tail; for a
    segment shorter than K-1 the low indices fall into the (zero or history) prefix. valid_len=None
    keeps the simple `last K-1` behaviour (CPU tests that build exact-length segments).
    """
    K = kernel_size
    if conv_state is None:
        hist = torch.zeros(*xBC_t.shape[:-1], K - 1, dtype=xBC_t.dtype, device=xBC_t.device)
        out = F.conv1d(xBC_t, conv_weight, conv_bias, padding=K - 1, groups=groups)[..., :xBC_t.shape[-1]]
    else:
        hist = conv_state.to(xBC_t.dtype) * is_continuation.to(xBC_t.dtype)
        out = F.conv1d(torch.cat([hist, xBC_t], dim=-1), conv_weight, conv_bias, groups=groups)
    full = torch.cat([hist, xBC_t], dim=-1)                            # [b, conv_dim, (K-1)+seq]
    if valid_len is None:
        new_conv_state = full[..., -(K - 1):]                          # always [b, conv_dim, K-1]
    else:
        idx = valid_len.reshape(()).long() + torch.arange(K - 1, device=xBC_t.device)
        new_conv_state = torch.index_select(full, -1, idx)             # real last K-1 (pad-safe)
    return out, new_conv_state


def chunked_ssd_scan(x, B, C, dt, A, D, chunk_size, ssm_state0=None):
    """Chunked Mamba2 SSD prefill. Splits the sequence into T = ceil(l / chunk_size) chunks and
    replaces the single O(l^2) quadratic scan with two BOUNDED attention-form passes:

      intra: each chunk's diagonal block (chunk_size x chunk_size), batched over T chunks
      inter: state passing across chunks, solved in CLOSED FORM on the chunk axis (T x T)

    No l-length Python loop (avoids neuronx-cc NCC_IFML902) and no strided chunk split — the chunk
    axis is a CONTIGUOUS reshape after right-padding l up to a multiple of chunk_size with dt=0
    (avoids NCC_IBCG901). Compute/memory O(l * chunk_size + T^2). Mathematically identical to the
    sequential 1-step recurrence and the quadratic form (test_chunked_ssd_matches_sequential).
    Supports a prefix state ssm_state0[b,H,P,N], which enables continuation prefill.

    Shapes: x[b,l,H,P], B/C[b,l,H,N], dt[b,l,H], A[H] (<=0), D[H]. Returns (y[b,l,H,P], h[b,H,P,N]).
    """
    b, l, H, P = x.shape
    N = B.shape[-1]
    cs = chunk_size
    if l <= 0:   # ValueError (not assert): survives `python -O`, which strips asserts.
        raise ValueError("chunked_ssd_scan requires a non-empty sequence")
    Lp = ((l + cs - 1) // cs) * cs
    T = Lp // cs
    pad = Lp - l

    def _pad(t):
        if pad == 0:
            return t
        shp = list(t.shape)
        shp[1] = pad
        return torch.cat([t, torch.zeros(shp, dtype=t.dtype, device=t.device)], dim=1)

    # Contiguous reshape into [b, T, chunk, ...] (NOT a strided view).
    xc = _pad(x).reshape(b, T, cs, H, P)
    Bc = _pad(B).reshape(b, T, cs, H, N)
    Cc = _pad(C).reshape(b, T, cs, H, N)
    dtc = _pad(dt).reshape(b, T, cs, H)
    dt_src = dtc.permute(0, 1, 3, 2)                                  # [b,T,H,cs] weight at source
    # In-chunk cumulative decay exponent A * cumsum(dt); <= 0 since A <= 0, dt >= 0.
    At = (torch.cumsum(dtc, dim=2) * A).permute(0, 1, 3, 2)           # [b,T,H,cs]

    ar = torch.arange(cs, device=x.device)
    causal = (ar[:, None] >= ar[None, :])                            # [cs,cs] j <= i
    # intra-chunk (diagonal block): same op shape as attention. mask BEFORE exp (inf*0 = NaN trap).
    expo = (At.unsqueeze(-1) - At.unsqueeze(-2)).masked_fill(~causal.view(1, 1, 1, cs, cs), float("-inf"))
    M = torch.einsum('btihn,btjhn->bthij', Cc, Bc) * torch.exp(expo) * dt_src.unsqueeze(-2)
    y_diag = torch.einsum('bthij,btjhp->btihp', M, xc)               # [b,T,cs,H,P]

    # each chunk's own end-state contribution (decay from source i to the chunk end)
    wc = torch.exp(At[..., -1:] - At) * dt_src                       # [b,T,H,cs]
    states = torch.einsum('bthi,btihp,btihn->bthpn', wc, xc, Bc)     # [b,T,H,P,N]

    # inter-chunk state passing in CLOSED FORM. states_{c'} already carries chunk c''s internal decay
    # to its own end, so the coefficient EXCLUDES gamma_{c'}: prod_{c'<k<c} gamma_k = exp(G_exc[c] - G_inc[c']).
    log_gamma = At[..., -1]                                          # [b,T,H] log of per-chunk decay (<=0)
    G_inc = torch.cumsum(log_gamma, dim=1)                           # inclusive prefix
    G_exc = G_inc - log_gamma                                        # exclusive prefix
    arT = torch.arange(T, device=x.device)
    strict = (arT[:, None] > arT[None, :])                           # c' < c
    chunk_decay = (G_exc.permute(0, 2, 1).unsqueeze(-1) - G_inc.permute(0, 2, 1).unsqueeze(-2))  # [b,H,c,c']
    chunk_decay = torch.exp(chunk_decay.masked_fill(~strict.view(1, 1, T, T), float("-inf")))
    state_in = torch.einsum('bhck,bkhpn->bchpn', chunk_decay, states)   # state entering each chunk
    if ssm_state0 is not None:
        state_in = state_in + torch.exp(G_exc).unsqueeze(-1).unsqueeze(-1) * ssm_state0.unsqueeze(1)
    # off-diagonal output: the incoming state decayed to position i within the chunk
    y_off = torch.exp(At).permute(0, 1, 3, 2).unsqueeze(-1) * torch.einsum('btihn,bthpn->btihp', Cc, state_in)

    y = (y_diag + y_off).reshape(b, Lp, H, P)[:, :l] + x * D[..., None]
    # final SSM state for the following decode steps: carry through the last chunk once more
    h = torch.exp(log_gamma[:, -1])[..., None, None] * state_in[:, -1] + states[:, -1]
    return y, h
