# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the NemotronH MoE decode step: only the selected experts' weights are read.

For each token t and each of its K selected experts e:

    h   = relu(x_t @ up[e]) ** 2 * w[t, k]          # [I]
    out_t += h @ down[e]                             # [H]

The expert index is a runtime value, so each expert's weight slice is fetched from HBM with an
indirect DMA (the index is the scalar offset along the expert axis); no gathered copy of the weights
is materialised. The two physical cores of a logical core (LNC=2) split the intermediate axis I, and
each writes its partial sum to its own row of the output, which the caller adds.

Layout choices for a matrix-vector product on the Tensor Engine:
  - gate/up: the expert's weight tile is the stationary operand and x the moving one, so the result
    comes out as h^T with I on the partition axis, which is the layout the down projection contracts
    over (no transpose).
  - down: h^T is the stationary operand (one column per token) and the weight streams as the moving
    operand in 512-wide chunks of H; the products of all K experts accumulate in PSUM.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partitions
F_MAX = 512      # moving free size per matmul (one fp32 PSUM bank)


@nki.jit
def moe_relu2_decode(x, up, down, expert_index, expert_weight):
    """x [T, H] bf16, up [E, H, I] bf16, down [E, I, H] bf16, expert_index [T, K] int32,
    expert_weight [T, K] fp32 -> partial sums [n_cores, T, H] fp32 (sum over the first axis)."""
    T, H = x.shape
    E, _, I = up.shape
    K = expert_index.shape[1]
    n_cores = nl.num_programs(0)
    core = nl.program_id(0)
    assert H % P == 0, "H must be a multiple of 128"
    assert I % n_cores == 0, "I must split evenly across the cores"
    H1 = H // P
    I_c = I // n_cores
    i0 = core * I_c
    i_tiles = [(s, min(P, I_c - s)) for s in range(0, I_c, P)]
    h_chunks = [(s, min(F_MAX, H - s)) for s in range(0, H, F_MAX)]

    out = nl.ndarray((n_cores, T, H), dtype=nl.float32, buffer=nl.shared_hbm)

    # x^T in SBUF: xT[p, h1, t] = x[t, h1 * P + p]
    xT = nl.ndarray((P, H1, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, P], [P, H1], [H, T]], offset=0))

    for t in nl.static_range(T):
        acc = []
        for (_, hn) in h_chunks:
            acc.append(nl.ndarray((1, hn), dtype=nl.float32, buffer=nl.psum))
        for k in nl.static_range(K):
            e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=e, src=expert_index.ap(pattern=[[1, 1], [1, 1]], offset=t * K + k))
            # router weight broadcast to every partition, to scale h^T
            wk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wk, src=expert_weight.ap(pattern=[[0, P], [1, 1]], offset=t * K + k))

            # up[e][:, i0:i0+I_c] as [P, H1, I_c]: element (p, h1, i) = up[e, h1*P + p, i0 + i]
            up_sb = nl.ndarray((P, H1, I_c), dtype=up.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_sb, src=up.ap(pattern=[[I, P], [P * I, H1], [1, I_c]], offset=i0,
                                               scalar_offset=e, indirect_dim=0))
            for j, (s, m) in enumerate(i_tiles):
                # h^T for intermediate rows [s, s+m): [m, 1] = up_tile^T @ x_t, accumulated over H
                hp = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.psum)
                for h1 in nl.static_range(H1):
                    nisa.nc_matmul(dst=hp, stationary=up_sb[0:P, h1, s:s + m],
                                   moving=xT[0:P, h1, t:t + 1], accumulate=(h1 > 0))
                r = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=r, op=nl.relu, data=hp)
                nisa.activation(dst=r, op=nl.square, data=r)
                hT = nl.ndarray((m, 1), dtype=down.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=hT, data1=r, data2=wk[0:m, 0:1], op=nl.multiply)

                # down[e][i0+s : i0+s+m, :] as [m, H]
                dn = nl.ndarray((m, H), dtype=down.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=dn, src=down.ap(pattern=[[H, m], [1, H]], offset=(i0 + s) * H,
                                                  scalar_offset=e, indirect_dim=0))
                for c, (hs, hn) in enumerate(h_chunks):
                    nisa.nc_matmul(dst=acc[c], stationary=hT, moving=dn[0:m, hs:hs + hn],
                                   accumulate=(k > 0 or j > 0))
        row = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
        for c, (hs, hn) in enumerate(h_chunks):
            nisa.tensor_copy(dst=row[0:1, hs:hs + hn], src=acc[c])
        nisa.dma_copy(dst=out.ap(pattern=[[H, 1], [1, H]], offset=(core * T + t) * H), src=row)
    return out
