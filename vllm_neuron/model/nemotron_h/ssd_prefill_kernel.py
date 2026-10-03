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

LNC=2: each physical core takes one SSM group (its heads share that group's B and C), so the cores
never exchange data. Requires Q == N == 128, groups == number of cores; checked by the caller.
"""
import nki
import nki.isa as nisa
import nki.language as nl

Q = 128


@nki.jit
def ssd_prefill(x, dt, A, B, C, D, state0, tri):
    """x [L, H, P] f32, dt [L, H] f32 (softplus'd, 0 on padding), A [H] f32 (<= 0), B/C [L, G, N] f32,
    D [H] f32, state0 [H, P, N] f32, tri [Q, Q] f32 with tri[j, i] = 1 for j <= i
    -> (y [L, H, P] f32, state [H, P, N] f32)."""
    L, H, P = x.shape
    G = nl.num_programs(0)
    g = nl.program_id(0)
    N = B.shape[2]
    heads_g = H // G
    h0 = g * heads_g
    n_chunks = L // Q

    y = nl.ndarray((L, H, P), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out = nl.ndarray((H, P, N), dtype=nl.float32, buffer=nl.shared_hbm)

    tri_sb = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tri_sb, src=tri)
    ones = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    # state per head as [N, P]: S[n, p] = state0[h, p, n]
    S = []
    for hh in range(heads_g):
        s = nl.ndarray((N, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=state0.ap(pattern=[[1, N], [N, P]], offset=(h0 + hh) * P * N))
        S.append(s)

    for c in range(n_chunks):
        t0 = c * Q
        # this group's B, C for the chunk, in both layouts
        BT = nl.ndarray((N, Q), dtype=nl.float32, buffer=nl.sbuf)       # [n, j]
        nisa.dma_copy(dst=BT, src=B.ap(pattern=[[1, N], [G * N, Q]], offset=(t0 * G + g) * N))
        CT = nl.ndarray((N, Q), dtype=nl.float32, buffer=nl.sbuf)       # [n, i]
        nisa.dma_copy(dst=CT, src=C.ap(pattern=[[1, N], [G * N, Q]], offset=(t0 * G + g) * N))
        Bj = nl.ndarray((Q, N), dtype=nl.float32, buffer=nl.sbuf)       # [j, n]
        nisa.dma_copy(dst=Bj, src=B.ap(pattern=[[G * N, Q], [1, N]], offset=(t0 * G + g) * N))
        # CBt[j, i] = B_j . C_i, shared by the group's heads
        cb_ps = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=cb_ps, stationary=BT, moving=CT)
        CBt = nl.ndarray((Q, Q), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=CBt, data1=cb_ps, data2=tri_sb, op=nl.multiply)   # keep j <= i

        for hh in range(heads_g):
            h = h0 + hh
            xs = nl.ndarray((Q, P), dtype=nl.float32, buffer=nl.sbuf)      # x[t0 + j, h, :]
            nisa.dma_copy(dst=xs, src=x.ap(pattern=[[H * P, Q], [1, P]], offset=(t0 * H + h) * P))
            dtc = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=dtc, src=dt.ap(pattern=[[H, Q], [1, 1]], offset=t0 * H + h))
            Ac = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=Ac, src=A.ap(pattern=[[0, Q], [1, 1]], offset=h))
            a = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a, data1=dtc, data2=Ac, op=nl.multiply)

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
            Dc = nl.ndarray((Q, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=Dc, src=D.ap(pattern=[[0, Q], [1, 1]], offset=h))
            nisa.scalar_tensor_tensor(dst=yo, data=xs, op0=nl.multiply, operand0=Dc, op1=nl.add,
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
        nisa.dma_copy(dst=state_out.ap(pattern=[[1, N], [N, P]], offset=(h0 + hh) * P * N), src=S[hh])
    return y, state_out
