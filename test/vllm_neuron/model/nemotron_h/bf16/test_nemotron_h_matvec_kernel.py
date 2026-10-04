# SPDX-License-Identifier: Apache-2.0
"""The NKI decode-projection kernel on the NKI CPU simulator, at the per-rank (TP=4) shapes it is
the model can use it for: Mamba in_proj / out_proj, attention qkv / o_proj, the vocabulary
projection, and the shared expert (off by default, NEMOTRONH_MATVEC_OFF), up to a decode batch of 8."""
import importlib.util
import os

import numpy as np
import pytest

nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")

_here = os.path.dirname(os.path.abspath(__file__))
_path = _here
for _ in range(8):
    _cand = os.path.join(_path, "vllm_neuron", "model", "nemotron_h", "matvec_kernel.py")
    if os.path.exists(_cand):
        break
    _path = os.path.dirname(_path)
_spec = importlib.util.spec_from_file_location("nemotron_matvec_kernel", _cand)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BF16 = ml_dtypes.bfloat16


@pytest.mark.parametrize("T,H,N", [(1, 2688, 2576), (1, 1024, 2688), (1, 2688, 1280), (1, 2688, 928),
                                   (1, 928, 2688), (2, 2688, 2576), (8, 2688, 2576), (1, 2688, 32768)])
def test_matvec_matches_numpy(T, H, N):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, H)).astype(BF16)
    w = (rng.standard_normal((H, N)) * 0.03).astype(BF16)
    got = np.asarray(nki.simulate(_mod.matvec[2])(x, w), np.float32)
    ref = x.astype(np.float32) @ w.astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2 * np.abs(ref).max())
