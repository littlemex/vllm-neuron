# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the decode-step projections: y = x @ W for a handful of tokens.

With one token the Tensor Engine is input-starved whichever operand is stationary; what differs is
how often the systolic array is reloaded. With W as the stationary operand it is reloaded for every
128x128 weight tile. Here x (T <= 128 columns) is the stationary operand, loaded once per 128-row
slice of H, and W streams through as the moving operand in 512-wide column chunks.

LNC=2: the two physical cores split the output columns N; each loads only its columns of W, with
one DMA per 128-row slice, issued before the matmuls so the loads overlap with the compute.
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
F_MAX = 512
NB = 2048         # columns of W resident in SBUF at a time (per core)


def _chunk_starts(total, size):
    starts = []
    s = 0
    while s < total:
        starts.append(s)
        s += size
    return starts


@nki.jit
def matvec(x, w):
    """x [T, H] bf16 (T <= 128), w [H, N] bf16 (N % n_cores == 0) -> y [T, N] bf16. H need not be a
    multiple of 128 (the last row slice is shorter). Columns are processed in blocks of NB so the
    resident weight slice fits in SBUF for large N (e.g. the vocabulary projection)."""
    T, H = x.shape
    _, N = w.shape
    n_cores = nl.num_programs(0)
    core = nl.program_id(0)
    N_c = N // n_cores
    n0 = core * N_c
    h_starts = _chunk_starts(H, P)
    blocks = _chunk_starts(N_c, NB)

    y = nl.ndarray((T, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xs = []
    for i in range(len(h_starts)):
        hs = h_starts[i]
        pr = min(P, H - hs)
        xt = nl.ndarray((pr, T), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x.ap(pattern=[[1, pr], [H, T]], offset=hs))
        xs.append(xt)
    for bi in range(len(blocks)):
        b0 = blocks[bi]
        nb = min(NB, N_c - b0)
        ws = []
        for i in range(len(h_starts)):
            hs = h_starts[i]
            pr = min(P, H - hs)
            wt = nl.ndarray((pr, nb), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=w.ap(pattern=[[N, pr], [1, nb]], offset=hs * N + n0 + b0))
            ws.append(wt)
        starts = _chunk_starts(nb, F_MAX)
        for c in range(len(starts)):
            s = starts[c]
            n = min(F_MAX, nb - s)
            ps = nl.ndarray((T, n), dtype=nl.float32, buffer=nl.psum)
            for i in range(len(h_starts)):
                pr = min(P, H - h_starts[i])
                nisa.nc_matmul(dst=ps, stationary=xs[i][0:pr, 0:T], moving=ws[i][0:pr, s:s + n],
                               accumulate=(i > 0))
            o = nl.ndarray((T, n), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o, src=ps)
            nisa.dma_copy(dst=y.ap(pattern=[[N, T], [1, n]], offset=n0 + b0 + s), src=o)
    return y
