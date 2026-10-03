# SPDX-License-Identifier: Apache-2.0
"""The NKI MoE decode kernel, run on the NKI CPU simulator against a NumPy reference, at the
real per-rank shapes (H=2688, I=464 at TP=4) with a reduced expert count. LNC=2 is simulated with
a two-program grid; the two partial sums must add up to the reference."""
import importlib.util
import os

import numpy as np
import pytest

nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")

_here = os.path.dirname(os.path.abspath(__file__))
_path = _here
for _ in range(8):
    _cand = os.path.join(_path, "vllm_neuron", "model", "nemotron_h", "moe_decode_kernel.py")
    if os.path.exists(_cand):
        break
    _path = os.path.dirname(_path)
_spec = importlib.util.spec_from_file_location("nemotron_moe_decode_kernel", _cand)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BF16 = ml_dtypes.bfloat16


def _reference(x, up, down, idx, w):
    x, up, down = (a.astype(np.float32) for a in (x, up, down))
    out = np.zeros((x.shape[0], x.shape[1]), np.float32)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = idx[t, k]
            h = np.maximum(x[t] @ up[e], 0.0) ** 2 * w[t, k]
            h = h.astype(BF16).astype(np.float32)          # the kernel feeds bf16 h to the matmul
            out[t] += h @ down[e]
    return out


@pytest.mark.parametrize("T", [1, 2])
def test_moe_relu2_decode_matches_reference(T):
    rng = np.random.default_rng(0)
    E, H, I, K = 8, 2688, 464, 6
    x = rng.standard_normal((T, H)).astype(BF16)
    up = (rng.standard_normal((E, H, I)) * 0.03).astype(BF16)
    down = (rng.standard_normal((E, I, H)) * 0.03).astype(BF16)
    idx = np.stack([rng.permutation(E)[:K] for _ in range(T)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (T, K)).astype(np.float32)
    got = nki.simulate(_mod.moe_relu2_decode[2])(x, up, down, idx, w)
    got = np.asarray(got, dtype=np.float32).sum(axis=0)
    ref = _reference(x, up, down, idx, w)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2 * np.abs(ref).max())
