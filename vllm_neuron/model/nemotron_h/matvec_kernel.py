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


def _chunk_starts(total, size):
    starts = []
    s = 0
    while s < total:
        starts.append(s)
        s += size
    return starts


@nki.jit
def matvec(x, w):
    """x [T, H] bf16 (T <= 128, H % 128 == 0), w [H, N] bf16 (N % n_cores == 0) -> y [T, N] bf16."""
    T, H = x.shape
    _, N = w.shape
    n_cores = nl.num_programs(0)
    core = nl.program_id(0)
    H1 = H // P
    N_c = N // n_cores
    n0 = core * N_c
    starts = _chunk_starts(N_c, F_MAX)

    y = nl.ndarray((T, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = nl.ndarray((P, H1, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, P], [P, H1], [H, T]], offset=0))
    ws = []
    for h1 in range(H1):
        wt = nl.ndarray((P, N_c), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w.ap(pattern=[[N, P], [1, N_c]], offset=h1 * P * N + n0))
        ws.append(wt)
    for c in range(len(starts)):
        s = starts[c]
        n = min(F_MAX, N_c - s)
        ps = nl.ndarray((T, n), dtype=nl.float32, buffer=nl.psum)
        for h1 in range(H1):
            nisa.nc_matmul(dst=ps, stationary=xT[0:P, h1, 0:T], moving=ws[h1][0:P, s:s + n],
                           accumulate=(h1 > 0))
        o = nl.ndarray((T, n), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=ps)
        nisa.dma_copy(dst=y.ap(pattern=[[N, T], [1, n]], offset=n0 + s), src=o)
    return y
