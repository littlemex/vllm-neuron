# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for one NemotronH Mamba2 decode step, between in_proj and out_proj.

Per request of the batch (one token each), on this TP rank's heads:

    xBC  = silu(causal_conv1d(conv_state ++ xBC_raw))            # conv state advances by one
    dt   = softplus(dt_raw + dt_bias);  dA = exp(dt * A)
    h    = state * dA + dt * x (outer) B                         # [heads, P, N]
    y    = h . C + D * x                                         # [heads, P]
    y    = rmsnorm_group(y * silu(gate)) * norm_w                # gated RMSNorm per group

Layout: the (head, p) rows go on the partition axis, two heads per 128-partition tile, and the SSM
state's N axis on the free axis, so the recurrence is per-partition scalar times a broadcast row. B
and C (one row per group) are broadcast to all partitions with a matmul against a diagonal.

LNC=2 split: each physical core owns one group, i.e. heads [g*H/G, (g+1)*H/G) with their x, B, C
channels, and that group is exactly one gated-RMSNorm group, so the cores never exchange data.
Requires P == 64, N == 128, groups == number of cores; checked by the caller.

Batch of R requests: everything that is per channel or per head is computed for all requests at
once, with the requests on the free axis ([128 channels, R] tiles; one DMA brings a tile's inputs for
every request). Per-head values (dt, dA, D) are computed on a [heads, R] tile and spread to the
(head, p) partitions with one matmul against a head selector, instead of a broadcast DMA per head.
Only the state update (a [128, N] tile per request) and the B/C row broadcast are per request.
Activations are grouped by function so the activation table is switched as few times as possible.
"""
import nki
import nki.isa as nisa
import nki.language as nl

T_P = 128        # partitions per tile


def _load_cols(src, ch0, stride_r, R, dtype):
    """[128, R] tile: element (p, r) = src[r * stride_r + ch0 + p] (128 channels, R requests)."""
    t = nl.ndarray((T_P, R), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=src.ap(pattern=[[1, T_P], [stride_r, R]], offset=ch0))
    return t


def _conv_tiles(xBC, conv_state, conv_w, conv_b, conv_out, ch0, R, K):
    """silu(causal conv) for channels [ch0, ch0+128) of all R requests as a [128, R] fp32 tile; stores
    the advanced conv state. The history is laid out tap-major ([128, K * R], tap k of request r at
    k * R + r) so each tap is one contiguous [128, R] slice."""
    Km1 = K - 1
    C = conv_state.shape[1]
    cs = nl.ndarray((T_P, K * R), dtype=conv_state.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs[0:T_P, 0:Km1 * R],
                  src=conv_state.ap(pattern=[[Km1, T_P], [1, Km1], [C * Km1, R]], offset=ch0 * Km1))
    nisa.dma_copy(dst=cs[0:T_P, Km1 * R:K * R], src=xBC.ap(pattern=[[1, T_P], [C, R]], offset=ch0))
    w = nl.ndarray((T_P, K), dtype=conv_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=conv_w.ap(pattern=[[K, T_P], [1, K]], offset=ch0 * K))
    wf = nl.ndarray((T_P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wf, src=w)
    b = nl.ndarray((T_P, 1), dtype=conv_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b, src=conv_b.ap(pattern=[[1, T_P], [1, 1]], offset=ch0))
    bf = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bf, src=b)
    acc = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=acc, data=cs[0:T_P, 0:R], op0=nl.multiply, operand0=wf[0:T_P, 0:1])
    for k in range(1, K):
        nisa.scalar_tensor_tensor(dst=acc, data=cs[0:T_P, k * R:(k + 1) * R], op0=nl.multiply,
                                  operand0=wf[0:T_P, k:k + 1], op1=nl.add, operand1=acc)
    out = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=out, op=nl.silu, data=acc, bias=bf)
    nisa.dma_copy(dst=conv_out.ap(pattern=[[Km1, T_P], [1, Km1], [C * Km1, R]], offset=ch0 * Km1),
                  src=cs[0:T_P, R:K * R])
    return out


def _broadcast_row(col, eye_sb, ones):
    """[128, 128] tile whose every partition holds the row col[0:128] (col is a [128, 1] column)."""
    diag = nl.ndarray((T_P, T_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=diag, data=eye_sb, op0=nl.multiply, operand0=col)
    ps = nl.ndarray((T_P, T_P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ones, moving=diag)
    row = nl.ndarray((T_P, T_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row, src=ps)
    return row


def _spread_heads(sel_sb, j, vals, n):
    """[128, n] tile: partition p of x tile j gets vals[head of p, :] (vals is [heads, n], two heads
    per tile), via a matmul against the head selector."""
    ps = nl.ndarray((T_P, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=sel_sb[:, j * T_P:(j + 1) * T_P], moving=vals)
    out = nl.ndarray((T_P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


@nki.jit
def mamba2_decode_step(xBC, gate, dt, conv_state, conv_w, conv_b, dt_bias, A, D, ssm_state,
                       norm_w, eye, head_sel, eps):
    """xBC [R, C_dim] bf16, gate [R, I] bf16, dt [R, H] bf16, conv_state [R, C_dim, K-1] bf16,
    conv_w [C_dim, K] bf16, conv_b [C_dim] bf16, dt_bias/A/D [H] f32, ssm_state [R, H, P, N] f32,
    norm_w [I] f32, eye [128, 128] f32 identity, head_sel [H/G, (H/G/2) * 128] f32 (head_sel[h,
    j * 128 + p] = 1 if h == 2j + p // P), eps [1] f32, for R requests
    -> (y [R, I] bf16, new_conv_state [R, C_dim, K-1] bf16, new_ssm_state [R, H, P, N] f32)."""
    R, C_dim, Km1 = conv_state.shape
    K = Km1 + 1
    _, H, P, N = ssm_state.shape
    I = H * P
    G = nl.num_programs(0)
    g = nl.program_id(0)
    heads_g = H // G                       # heads of this core's group
    tiles_g = heads_g * P // T_P           # x tiles of this group (two heads per tile)
    h_0 = g * heads_g                      # first head of this group
    x_ch0 = h_0 * P                        # first x channel of this group
    b_ch0 = I + g * N                      # this group's B channels
    c_ch0 = I + G * N + g * N              # this group's C channels

    y_out = nl.ndarray((R, I), dtype=gate.dtype, buffer=nl.shared_hbm)
    conv_out = nl.ndarray(conv_state.shape, dtype=conv_state.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray(ssm_state.shape, dtype=nl.float32, buffer=nl.shared_hbm)

    eye_sb = nl.ndarray((T_P, T_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye_sb, src=eye)
    ones = nl.ndarray((T_P, T_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    eps_sb = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eps_sb, src=eps.ap(pattern=[[0, T_P], [1, 1]], offset=0))
    sel_sb = nl.ndarray((heads_g, tiles_g * T_P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sel_sb, src=head_sel)

    # silu: conv of the B, C and x channels, and the gate (all requests per tile)
    Bc = _conv_tiles(xBC, conv_state, conv_w, conv_b, conv_out, b_ch0, R, K)
    Cc = _conv_tiles(xBC, conv_state, conv_w, conv_b, conv_out, c_ch0, R, K)
    xs = []
    gs = []
    for j in range(tiles_g):
        xs.append(_conv_tiles(xBC, conv_state, conv_w, conv_b, conv_out, x_ch0 + j * T_P, R, K))
        gt = _load_cols(gate, x_ch0 + j * T_P, I, R, gate.dtype)
        gsj = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gsj, op=nl.silu, data=gt)
        gs.append(gsj)

    # per head, all requests: dt = softplus(dt_raw + dt_bias), dA = exp(dt * A)  ([heads_g, R])
    dtr = nl.ndarray((heads_g, R), dtype=dt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=dtr, src=dt.ap(pattern=[[1, heads_g], [H, R]], offset=h_0))
    hb = nl.ndarray((heads_g, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hb, src=dt_bias.ap(pattern=[[1, heads_g], [1, 1]], offset=h_0))
    ha = nl.ndarray((heads_g, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ha, src=A.ap(pattern=[[1, heads_g], [1, 1]], offset=h_0))
    hd = nl.ndarray((heads_g, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hd, src=D.ap(pattern=[[1, heads_g], [1, 1]], offset=h_0))
    dts = nl.ndarray((heads_g, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=dts, op=nl.softplus, data=dtr, bias=hb)
    dA = nl.ndarray((heads_g, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=dA, op=nl.exp, data=dts, scale=ha)

    # B and C of each request broadcast to every partition ([128, N] rows)
    Bbs = []
    Cbs = []
    for r in range(R):
        Bbs.append(_broadcast_row(Bc[0:T_P, r:r + 1], eye_sb, ones))
        Cbs.append(_broadcast_row(Cc[0:T_P, r:r + 1], eye_sb, ones))

    # SSM step per (tile, request); y collected as [128, R] per tile
    ys = []
    for j in range(tiles_g):
        row0 = x_ch0 + j * T_P                     # first (head, p) row of this tile
        dt_c = _spread_heads(sel_sb, j, dts, R)    # [128, R]
        dA_c = _spread_heads(sel_sb, j, dA, R)
        D_c = _spread_heads(sel_sb, j, hd, 1)      # [128, 1]
        dtx = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dtx, data1=dt_c, data2=xs[j], op=nl.multiply)
        yj = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
        for r in range(R):
            st = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=st, src=ssm_state.ap(pattern=[[N, T_P], [1, N]], offset=(r * I + row0) * N))
            nisa.tensor_scalar(dst=st, data=st, op0=nl.multiply, operand0=dA_c[0:T_P, r:r + 1])
            hn = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=hn, data=Bbs[r], op0=nl.multiply, operand0=dtx[0:T_P, r:r + 1],
                                      op1=nl.add, operand1=st)
            nisa.dma_copy(dst=state_out.ap(pattern=[[N, T_P], [1, N]], offset=(r * I + row0) * N), src=hn)
            hc = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=hc, data1=hn, data2=Cbs[r], op=nl.multiply)
            nisa.tensor_reduce(dst=yj[0:T_P, r:r + 1], op=nl.add, data=hc, axis=1)
        # y = (h . C + D * x) * silu(gate)
        nisa.scalar_tensor_tensor(dst=yj, data=xs[j], op0=nl.multiply, operand0=D_c, op1=nl.add, operand1=yj)
        nisa.tensor_tensor(dst=yj, data1=yj, data2=gs[j], op=nl.multiply)
        ys.append(yj)

    # gated RMSNorm over the group, per request: every partition accumulates the sums of squares
    sumsq = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.psum)
    for j in range(tiles_g):
        sq = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=ys[j], data2=ys[j], op=nl.multiply)
        nisa.nc_matmul(dst=sumsq, stationary=ones, moving=sq, accumulate=(j > 0))
    rstd = nl.ndarray((T_P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=sumsq, scale=1.0 / (tiles_g * T_P), bias=eps_sb)
    for j in range(tiles_g):
        row0 = x_ch0 + j * T_P
        nw = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=nw, src=norm_w.ap(pattern=[[1, T_P], [1, 1]], offset=row0))
        yo = nl.ndarray((T_P, R), dtype=gate.dtype, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=yo, data=ys[j], op0=nl.multiply, operand0=nw, op1=nl.multiply, operand1=rstd)
        nisa.dma_copy(dst=y_out.ap(pattern=[[1, T_P], [I, R]], offset=row0), src=yo)
    return y_out, conv_out, state_out


def head_selector(heads_g, P, device=None):
    """head_sel input of mamba2_decode_step: [heads_g, (heads_g * P / 128) * 128] f32 with a 1 where
    head h owns partition p of x tile j (two P = 64 heads per 128-partition tile)."""
    import torch
    tiles = heads_g * P // T_P
    col = torch.arange(tiles * T_P, device=device)
    head_of_col = (col // T_P) * (T_P // P) + (col % T_P) // P
    return (torch.arange(heads_g, device=device).view(-1, 1) == head_of_col.view(1, -1)).to(torch.float32)
