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
Requires P == 64, N == 128, groups == number of cores; checked by the caller. The requests of a
batch are processed one after another; B and C, the state and every per-request input are addressed
at the request's row.
"""
import nki
import nki.isa as nisa
import nki.language as nl

T_P = 128        # partitions per tile

def _conv_tile(xBC, conv_state, conv_w, conv_b, conv_out, r, ch0, K):
    """silu(conv) for channels [ch0, ch0+128) of request r as a [128, 1] fp32 column; stores the new
    conv state."""
    Km1 = K - 1
    rc = r * conv_state.shape[1] + ch0                                      # row-major channel of request r
    cs = nl.ndarray((T_P, K), dtype=conv_state.dtype, buffer=nl.sbuf)       # history ++ new input
    nisa.dma_copy(dst=cs[0:T_P, 0:Km1], src=conv_state.ap(pattern=[[Km1, T_P], [1, Km1]], offset=rc * Km1))
    nisa.dma_copy(dst=cs[0:T_P, Km1:K], src=xBC.ap(pattern=[[1, T_P], [1, 1]], offset=rc))
    w = nl.ndarray((T_P, K), dtype=conv_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=conv_w.ap(pattern=[[K, T_P], [1, K]], offset=ch0 * K))
    b = nl.ndarray((T_P, 1), dtype=conv_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b, src=conv_b.ap(pattern=[[1, T_P], [1, 1]], offset=ch0))
    prod = nl.ndarray((T_P, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=cs, data2=w, op=nl.multiply)
    acc = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=acc, op=nl.add, data=prod, axis=1)
    bf = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bf, src=b)
    out = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=out, op=nl.silu, data=acc, bias=bf)
    nisa.dma_copy(dst=conv_out.ap(pattern=[[Km1, T_P], [1, Km1]], offset=rc * Km1), src=cs[0:T_P, 1:K])
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


def _head_scalar(vec, h, P):
    """[128, 1] column: rows [0, P) hold vec[h], rows [P, 2P) hold vec[h + 1] (two heads per tile).
    h indexes vec flat, so a per-request [B, H] input passes r * H + head."""
    col = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=col[0:P, 0:1], src=vec.ap(pattern=[[0, P], [1, 1]], offset=h))
    nisa.dma_copy(dst=col[P:T_P, 0:1], src=vec.ap(pattern=[[0, P], [1, 1]], offset=h + 1))
    return col


@nki.jit
def mamba2_decode_step(xBC, gate, dt, conv_state, conv_w, conv_b, dt_bias, A, D, ssm_state,
                       norm_w, eye, eps):
    """xBC [R, C_dim] bf16, gate [R, I] bf16, dt [R, H] bf16, conv_state [R, C_dim, K-1] bf16,
    conv_w [C_dim, K] bf16, conv_b [C_dim] bf16, dt_bias/A/D [H] f32, ssm_state [R, H, P, N] f32,
    norm_w [I] f32, eye [128, 128] f32 identity, eps [1] f32, for R requests
    -> (y [R, I] bf16, new_conv_state [R, C_dim, K-1] bf16, new_ssm_state [R, H, P, N] f32)."""
    R, C_dim, Km1 = conv_state.shape
    K = Km1 + 1
    _, H, P, N = ssm_state.shape
    I = H * P
    G = nl.num_programs(0)
    g = nl.program_id(0)
    heads_g = H // G                       # heads of this core's group
    tiles_g = heads_g * P // T_P           # x tiles of this group (two heads per tile)
    x_ch0 = g * heads_g * P                # first x channel of this group
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

    for r in range(R):
        Bb = _broadcast_row(_conv_tile(xBC, conv_state, conv_w, conv_b, conv_out, r, b_ch0, K), eye_sb, ones)
        Cb = _broadcast_row(_conv_tile(xBC, conv_state, conv_w, conv_b, conv_out, r, c_ch0, K), eye_sb, ones)

        ys = []
        sumsq = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.psum)
        for j in range(tiles_g):
            row0 = x_ch0 + j * T_P                     # first (head, p) row of this tile
            x = _conv_tile(xBC, conv_state, conv_w, conv_b, conv_out, r, row0, K)
            h0 = g * heads_g + 2 * j
            dtc = _head_scalar(dt, r * H + h0, P)
            dtb = _head_scalar(dt_bias, h0, P)
            dts = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=dts, op=nl.softplus, data=dtc, bias=dtb)
            Ac = _head_scalar(A, h0, P)
            dA = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=dA, op=nl.exp, data=dts, scale=Ac)
            dtx = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dtx, data1=dts, data2=x, op=nl.multiply)

            st = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=st, src=ssm_state.ap(pattern=[[N, T_P], [1, N]], offset=(r * I + row0) * N))
            nisa.tensor_scalar(dst=st, data=st, op0=nl.multiply, operand0=dA)
            hn = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=hn, data=Bb, op0=nl.multiply, operand0=dtx, op1=nl.add, operand1=st)
            nisa.dma_copy(dst=state_out.ap(pattern=[[N, T_P], [1, N]], offset=(r * I + row0) * N), src=hn)

            hc = nl.ndarray((T_P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=hc, data1=hn, data2=Cb, op=nl.multiply)
            y = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=y, op=nl.add, data=hc, axis=1)
            Dc = _head_scalar(D, h0, P)
            nisa.scalar_tensor_tensor(dst=y, data=x, op0=nl.multiply, operand0=Dc, op1=nl.add, operand1=y)

            gt = nl.ndarray((T_P, 1), dtype=gate.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt, src=gate.ap(pattern=[[1, T_P], [1, 1]], offset=r * I + row0))
            gs = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=gs, op=nl.silu, data=gt)
            nisa.tensor_tensor(dst=y, data1=y, data2=gs, op=nl.multiply)
            sq = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sq, data1=y, data2=y, op=nl.multiply)
            # every partition accumulates the group's sum of squares
            nisa.nc_matmul(dst=sumsq, stationary=ones, moving=sq, accumulate=(j > 0))
            ys.append(y)

        rstd = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=sumsq, scale=1.0 / (tiles_g * T_P), bias=eps_sb)
        for j in range(tiles_g):
            row0 = x_ch0 + j * T_P
            nw = nl.ndarray((T_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=nw, src=norm_w.ap(pattern=[[1, T_P], [1, 1]], offset=row0))
            yo = nl.ndarray((T_P, 1), dtype=gate.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=yo, data=ys[j], op0=nl.multiply, operand0=rstd, op1=nl.multiply, operand1=nw)
            nisa.dma_copy(dst=y_out.ap(pattern=[[1, T_P], [1, 1]], offset=r * I + row0), src=yo)
    return y_out, conv_out, state_out
