# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the NemotronH MoE decode step: only the selected experts' weights are read.

For each token t and each of its K selected experts e:

    h   = relu(x_t @ up[e]) ** 2 * w[t, k]          # [I]
    out_t += h @ down[e]                             # [H]

The expert index is a runtime value, so each expert's weight slice is fetched from HBM with an
indirect DMA (the index is the scalar offset along the expert axis); no gathered copy of the weights
is materialised; all K experts' loads are issued before the compute so they overlap with it. The
two physical cores of a logical core (LNC=2) split the intermediate axis I, and
each writes its partial sum to its own row of the output, which the caller adds.

Layout choices for a matrix-vector product on the Tensor Engine:
  - gate/up: x (one column) is the stationary operand and the weight streams as the moving one,
    giving h as a row; two PE transposes turn it into the h^T columns the down projection contracts
    over. (The weight as the stationary operand would reload the array once per 128x128 tile.)
  - down: h^T is the stationary operand (one column per token) and the weight streams as the moving
    operand in 512-wide chunks of H; the products of all K experts accumulate in PSUM.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partitions
F_MAX = 512      # moving free size per matmul (one fp32 PSUM bank)


def _tile_starts(total, size):
    """Start offsets of `size`-wide tiles covering [0, total) (the last one may be shorter). Built
    outside the kernel body: the NKI tracer does not accept comprehensions there."""
    starts = []
    s = 0
    while s < total:
        starts.append(s)
        s += size
    return starts


@nki.jit
def moe_relu2_decode(x, up, down, expert_index, expert_weight):
    """x [T, H] bf16, up [E, H, I] bf16, down [E, I, H] bf16, expert_index [T, K] int32,
    expert_weight [T, K] fp32 -> partial sums [n_cores, T, H] fp32 (sum over the first axis)."""
    T, H = x.shape
    E, _, I = up.shape
    K = expert_index.shape[1]
    n_cores = nl.num_programs(0)
    core = nl.program_id(0)
    # H % 128 == 0 and I % n_cores == 0 are checked by the caller (_use_moe_decode_kernel); the NKI
    # tracer does not accept assert statements in a kernel body.
    H1 = H // P
    I_c = I // n_cores
    i0 = core * I_c
    i_starts = _tile_starts(I_c, P)
    h_starts = _tile_starts(H, F_MAX)

    out = nl.ndarray((n_cores, T, H), dtype=nl.float32, buffer=nl.shared_hbm)

    # x^T in SBUF: xT[p, h1, t] = x[t, h1 * P + p]
    xT = nl.ndarray((P, H1, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, P], [P, H1], [H, T]], offset=0))

    for t in nl.static_range(T):
        # Issue every selected expert's weight loads first so they overlap with the compute below.
        es, wks, ups, dns = [], [], [], []
        for k in nl.static_range(K):
            e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=e, src=expert_index.ap(pattern=[[1, 1], [1, 1]], offset=t * K + k))
            wk = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wk, src=expert_weight.ap(pattern=[[1, 1], [1, 1]], offset=t * K + k))
            # up[e][:, i0:i0+I_c] as [P, H1, I_c]: element (p, h1, i) = up[e, h1*P + p, i0 + i]
            up_sb = nl.ndarray((P, H1, I_c), dtype=up.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_sb, src=up.ap(pattern=[[I, P], [P * I, H1], [1, I_c]], offset=i0,
                                               scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge)
            dn_k = []
            for j in range(len(i_starts)):
                s = i_starts[j]
                m = min(P, I_c - s)
                # down[e][i0+s : i0+s+m, :] as [m, H]
                dn = nl.ndarray((m, H), dtype=down.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=dn, src=down.ap(pattern=[[H, m], [1, H]], offset=(i0 + s) * H,
                                                  scalar_offset=e, indirect_dim=0),
                              dge_mode=nisa.dge_mode.hwdge)
                dn_k.append(dn)
            es.append(e)
            wks.append(wk)
            ups.append(up_sb)
            dns.append(dn_k)

        acc = []
        for c in range(len(h_starts)):
            hn = min(F_MAX, H - h_starts[c])
            acc.append(nl.ndarray((1, hn), dtype=nl.float32, buffer=nl.psum))
        for k in nl.static_range(K):
            # h = x_t @ up_tile as a row [1, I_c]: x is the (one-column) stationary operand and the
            # weight streams as the moving operand, accumulated over the H tiles.
            hp = nl.ndarray((1, I_c), dtype=nl.float32, buffer=nl.psum)
            for h1 in nl.static_range(H1):
                nisa.nc_matmul(dst=hp, stationary=xT[0:P, h1, t:t + 1], moving=ups[k][0:P, h1, 0:I_c],
                               accumulate=(h1 > 0))
            r = nl.ndarray((1, I_c), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=r, op=nl.relu, data=hp)
            nisa.activation(dst=r, op=nl.square, data=r)
            nisa.tensor_scalar(dst=r, data=r, op0=nl.multiply, operand0=wks[k])
            for j in range(len(i_starts)):
                s = i_starts[j]
                m = min(P, I_c - s)
                # h^T rows [s, s+m) as the stationary column of the down projection
                tp = nl.ndarray((m, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=r[0:1, s:s + m])
                hT = nl.ndarray((m, 1), dtype=down.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=hT, src=tp)
                for c in range(len(h_starts)):
                    hs = h_starts[c]
                    hn = min(F_MAX, H - hs)
                    nisa.nc_matmul(dst=acc[c], stationary=hT, moving=dns[k][j][0:m, hs:hs + hn],
                                   accumulate=(k > 0 or j > 0))
        row = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(len(h_starts)):
            hs = h_starts[c]
            hn = min(F_MAX, H - hs)
            nisa.tensor_copy(dst=row[0:1, hs:hs + hn], src=acc[c])
        nisa.dma_copy(dst=out.ap(pattern=[[H, 1], [1, H]], offset=(core * T + t) * H), src=row)
    return out
