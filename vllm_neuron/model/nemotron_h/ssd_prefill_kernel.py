# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the NemotronH Mamba2 chunked-SSD prefill scan (same math as ssd.chunked_ssd_scan).

For each chunk of Q tokens and each head, with a = dt * A (<= 0) and cs = cumsum(a) within the chunk:

    y_i   = sum_{j<=i} (C_i . B_j) exp(cs_i - cs_j) dt_j x_j        (intra-chunk)
          + exp(cs_i) C_i . S_in                                    (state entering the chunk)
          + D x_i
    S_out = exp(cs_Q) S_in + sum_j exp(cs_Q - cs_j) dt_j x_j (outer) B_j

Every exponent is <= 0 (the upper triangle of cs_i - cs_j is clamped to 0 before exp and then masked),
so nothing overflows however large |A| * sum(dt) gets within a chunk.

Layout: the chunk's QxQ matrices are built with the source token j on the partition axis, so they
feed the y matmul (contraction over j) without a transpose; the state is kept as [N, P] (N on the
partition axis) across chunks, which is the layout both the off-diagonal term (contraction over N)
and the update (output on N) use.

The causal conv1d (+ bias, SiLU) that produces x, B and C runs inside the kernel, on the token-major
in_proj output: tap k of chunk rows [t0, t0 + Q) is the contiguous row block starting at t0 + k of
the history-prefixed input, so each tap is one DMA and the conv is a multiply-add per tap with the
tap's weights broadcast across partitions. Keeping the activations token-major end to end avoids the
channel-major round trip (and its transposing copies) a separate depthwise conv needs.

LNC=2: each physical core takes one SSM group (its heads share that group's B and C), so the cores
never exchange data. Requires Q == N == 128, P == 64, groups == number of cores; checked by the
caller.
"""
import nki
import nki.isa as nisa
import nki.language as nl

Q = 128


def _conv_rows(xbc, wb, bias_b, t0, ch0, n, K, dtype):
    """silu(causal conv + bias) of channels [ch0, ch0 + n) for the Q tokens starting at t0, as a
    [Q, n] fp32 tile (token on the partition axis). xbc is the history-prefixed input, so token t
    uses rows t .. t + K - 1; wb[k] are the tap weights broadcast to every partition."""
    acc = nl.ndarray((Q, n), dtype=nl.float32, buffer=nl.sbuf)
    for k in range(K):
        tap = nl.ndarray((Q, n), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tap, src=xbc.ap(pattern=[[xbc.shape[1], Q], [1, n]], offset=(t0 + k) * xbc.shape[1] + ch0))
        if k == 0:
            nisa.tensor_tensor(dst=acc, data1=tap, data2=wb[k], op=nl.multiply)
        else:
            prod = nl.ndarray((Q, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=tap, data2=wb[k], op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=prod, op=nl.add)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=bias_b, op=nl.add)
    out = nl.ndarray((Q, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=out, op=nl.silu, data=acc)
    return out


def _bcast_rows(vec, offset, n, stride):
    """[Q, n] fp32 tile whose every partition holds vec[offset + i * stride] for i < n."""
    t = nl.ndarray((Q, n), dtype=vec.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=vec.ap(pattern=[[0, Q], [stride, n]], offset=offset))
    f = nl.ndarray((Q, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=f, src=t)
    return f


def _transpose(src, rows, cols):
    """[cols, rows] fp32 SBUF copy of the [rows, cols] tile src (Tensor Engine transpose)."""
    ps = nl.ndarray((cols, rows), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=src)
    out = nl.ndarray((cols, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=ps)
    return out


@nki.jit
def ssd_prefill(xbc, conv_w, conv_b, dt, A, D, state0, tri):
    """xbc [K-1+L, C_dim] bf16: the conv history (K-1 rows) followed by this segment's raw in_proj
    xBC rows (x | B | C channels), conv_w [C_dim, K] bf16, conv_b [C_dim] bf16, dt [L, H] f32
    (softplus'd, 0 on padding), A [H] f32 (<= 0), D [H] f32, state0 [H, P, N] f32, tri [Q, Q] f32
    with tri[j, i] = 1 for j <= i -> (y [L, H, P] f32, state [H, P, N] f32)."""
    Lk, C_dim = xbc.shape
    K = conv_w.shape[1]
    L = Lk - (K - 1)
    H, P, N = state0.shape
    G = nl.num_programs(0)
    g = nl.program_id(0)
    I = H * P
    heads_g = H // G
    h0 = g * heads_g
    xw = heads_g * P                       # this group's x channels
    x_ch0 = h0 * P
    b_ch0 = I + g * N
    c_ch0 = I + G * N + g * N
    n_chunks = L // Q

    y = nl.ndarray((L, H, P), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out = nl.ndarray((H, P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    tri_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tri_sb, src=tri)
    ones = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    # conv taps and biases broadcast across partitions, once per kernel
    wX, wB, wC = [], [], []
    for k in range(K):
        wX.append(_bcast_rows(conv_w, x_ch0 * K + k, xw, K))
        wB.append(_bcast_rows(conv_w, b_ch0 * K + k, N, K))
        wC.append(_bcast_rows(conv_w, c_ch0 * K + k, N, K))
    bx = _bcast_rows(conv_b, x_ch0, xw, 1)
    bB = _bcast_rows(conv_b, b_ch0, N, 1)
    bC = _bcast_rows(conv_b, c_ch0, N, 1)
    # per-head A and D as [Q, heads_g] rows (column hh is head h0 + hh on every partition)
    Ar = _bcast_rows(A, h0, heads_g, 1)
    Dr = _bcast_rows(D, h0, heads_g, 1)

    # state per head as [N, P]: S[n, p] = state0[h, p, n]
    S = []
    for hh in range(heads_g):
        s_pn = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s_pn, src=state0.ap(pattern=[[N, P], [1, N]], offset=(h0 + hh) * P * N))
        S.append(_transpose(s_pn, P, N))

    for c in range(n_chunks):
        t0 = c * Q
        Bj = _conv_rows(xbc, wB, bB, t0, b_ch0, N, K, xbc.dtype)       # [j, n]
        Cj = _conv_rows(xbc, wC, bC, t0, c_ch0, N, K, xbc.dtype)       # [i, n]
        X = _conv_rows(xbc, wX, bx, t0, x_ch0, xw, K, xbc.dtype)       # [j, (head, p)]
        BT = _transpose(Bj, Q, N)                                       # [n, j]
        CT = _transpose(Cj, Q, N)                                       # [n, i]
        dts = nl.ndarray((Q, heads_g), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dts, src=dt.ap(pattern=[[H, Q], [1, heads_g]], offset=t0 * H + h0))
        # CBt[j, i] = B_j . C_i, shared by the group's heads
        cb_ps = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=cb_ps, stationary=BT, moving=CT)
        CBt = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=CBt, data1=cb_ps, data2=tri_sb, op=nl.multiply)   # keep j <= i

        for hh in range(heads_g):
            h = h0 + hh
            xs = X[0:Q, hh * P:(hh + 1) * P]                                # x[t0 + j, h, :]
            dtc = dts[0:Q, hh:hh + 1]
            a = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a, data1=dtc, data2=Ar[0:Q, hh:hh + 1], op=nl.multiply)
            # cs as a column (cs_j on partition j) and as a row broadcast to every partition
            cs_ps = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=cs_ps, stationary=tri_sb, moving=a)     # sum_{k<=j} a_k
            cs = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cs, src=cs_ps)
            a_b = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=a_b, data=ones, op0=nl.multiply, operand0=a)
            row_ps = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=row_ps, stationary=a_b, moving=tri_sb)  # [*, i] = cs_i
            csr = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=csr, src=row_ps)

            # decay^T[j, i] = exp(cs_i - cs_j) for j <= i: d = cs_j - cs_i >= 0 there; clamp the
            # other triangle (d < 0, would overflow) to 0 before exp, the CB mask zeroes it after.
            d = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=d, data=csr, op0=nl.subtract, operand0=cs, reverse0=True,
                               op1=nl.maximum, operand1=0.0)
            nisa.activation(dst=d, op=nl.exp, data=d, scale=-1.0)
            Mt = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=Mt, data1=d, data2=CBt, op=nl.multiply)

            dtx = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dtx, data=xs, op0=nl.multiply, operand0=dtc)

            # y = Mt^T @ dtx + exp(cs_i) * (C_i . S) + D x
            y_ps = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=y_ps, stationary=Mt, moving=dtx)
            off_ps = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=off_ps, stationary=CT, moving=S[hh])
            ecs = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=ecs, op=nl.exp, data=cs)
            # (the Vector Engine cannot read both operands from PSUM: stage the intra term in SBUF)
            yi = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yi, src=y_ps)
            yo = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=yo, data=off_ps, op0=nl.multiply, operand0=ecs, op1=nl.add,
                                      operand1=yi)
            nisa.scalar_tensor_tensor(dst=yo, data=xs, op0=nl.multiply, operand0=Dr[0:Q, hh:hh + 1], op1=nl.add,
                                      operand1=yo)
            nisa.dma_copy(dst=y.ap(pattern=[[H * P, Q], [1, P]], offset=(t0 * H + h) * P), src=yo)

            # S = exp(cs_Q) S + B^T @ (exp(cs_Q - cs_j) dt_j x_j)
            last = csr[0:Q, Q - 1:Q]                                    # cs_Q on every partition
            wj = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=wj, data1=last, data2=cs, op=nl.subtract)
            nisa.activation(dst=wj, op=nl.exp, data=wj)
            wx = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=wx, data=dtx, op0=nl.multiply, operand0=wj)
            up_ps = nl.ndarray((N, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=up_ps, stationary=Bj, moving=wx)
            el = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=el, op=nl.exp, data=last)
            s_new = nl.ndarray((N, P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=s_new, data=S[hh], op0=nl.multiply, operand0=el, op1=nl.add,
                                      operand1=up_ps)
            S[hh] = s_new

    for hh in range(heads_g):
        nisa.dma_copy(dst=state_out.ap(pattern=[[N, P], [1, N]], offset=(h0 + hh) * P * N),
                      src=_transpose(S[hh], N, P))
    return y, state_out
