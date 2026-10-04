# SPDX-License-Identifier: Apache-2.0
"""The NKI conv + chunked-SSD prefill kernel, on the NKI CPU simulator, against the shipped PyTorch
path (ssd.segmented_causal_conv1d + SiLU, then ssd.chunked_ssd_scan) at the per-rank TP=4 shape: 16
heads x 64, state 128, 2 groups, conv kernel 4, a 512-token segment (4 chunks). Includes a carried conv history, a large |A| * sum(dt) regime (within-chunk cumulative decay far below
-88, where an exp(-cumsum) formulation overflows), a carried initial state, and padding (dt = 0)."""
import importlib.util
import os

import numpy as np
import pytest

nki = pytest.importorskip("nki")
torch = pytest.importorskip("torch")

_here = os.path.dirname(os.path.abspath(__file__))
_path = _here
for _ in range(8):
    _cand = os.path.join(_path, "vllm_neuron", "model", "nemotron_h", "ssd_prefill_kernel.py")
    if os.path.exists(_cand):
        break
    _path = os.path.dirname(_path)
_dir = os.path.dirname(_cand)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_dir, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kernel = _load("nemotron_ssd_prefill_kernel", "ssd_prefill_kernel.py")
_ssd = _load("nemotron_ssd", "ssd.py")


@pytest.mark.parametrize("dt_scale,with_state,n_pad", [(0.05, False, 0), (1.0, True, 0), (0.3, True, 37)])
def test_ssd_prefill_matches_chunked_scan(dt_scale, with_state, n_pad):
    import ml_dtypes
    bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(0)
    L, H, P, N, G, K = 512, 16, 64, 128, 2, 4
    I = H * P
    C_dim = I + 2 * G * N
    raw = rng.standard_normal((L, C_dim)).astype(bf16)
    hist = (rng.standard_normal((K - 1, C_dim)) if with_state else np.zeros((K - 1, C_dim))).astype(bf16)
    conv_w = (rng.standard_normal((C_dim, K)) * 0.5).astype(bf16)
    conv_b = (rng.standard_normal(C_dim) * 0.1).astype(bf16)
    dt = (rng.uniform(0.0, 1.0, (L, H)) * dt_scale).astype(np.float32)
    if n_pad:
        dt[L - n_pad:] = 0.0
    A = -rng.uniform(1.0, 16.0, H).astype(np.float32)
    D = rng.uniform(0.5, 1.5, H).astype(np.float32)
    s0 = (rng.standard_normal((H, P, N)) if with_state else np.zeros((H, P, N))).astype(np.float32)
    tri = np.triu(np.ones((128, 128), np.float32))            # tri[j, i] = 1 for j <= i
    xbc = np.concatenate([hist, raw], axis=0)
    y, s = nki.simulate(_kernel.ssd_prefill[G])(xbc, conv_w, conv_b, dt, A, D, s0, tri)

    f32 = lambda a: torch.from_numpy(a.astype(np.float32))
    conv, _ = _ssd.segmented_causal_conv1d(
        f32(raw).T[None], f32(conv_w)[:, None, :], f32(conv_b), K, C_dim,
        conv_state=f32(hist).T[None], is_continuation=torch.ones(()))
    act = torch.nn.functional.silu(conv[0].T)                    # [L, C_dim]
    x = act[:, :I].reshape(1, L, H, P)
    rep = H // G
    Bt = act[:, I:I + G * N].reshape(1, L, G, N).repeat_interleave(rep, dim=2)
    Ct = act[:, I + G * N:].reshape(1, L, G, N).repeat_interleave(rep, dim=2)
    ry, rs = _ssd.chunked_ssd_scan(x, Bt, Ct, torch.from_numpy(dt)[None], torch.from_numpy(A),
                                   torch.from_numpy(D), 128, torch.from_numpy(s0)[None] if with_state else None)
    y, s = np.asarray(y, np.float32), np.asarray(s, np.float32)
    assert np.isfinite(y).all() and np.isfinite(s).all()
    np.testing.assert_allclose(y, ry[0].numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(s, rs[0].numpy(), rtol=2e-3, atol=2e-3)
