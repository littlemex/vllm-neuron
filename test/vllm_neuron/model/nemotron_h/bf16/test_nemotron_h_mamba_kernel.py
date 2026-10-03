# SPDX-License-Identifier: Apache-2.0
"""The NKI Mamba2 decode-step kernel, on the NKI CPU simulator, against a NumPy transcription of
NemotronHMamba2Mixer.forward_decode (the part between in_proj and out_proj), at the real per-rank
shapes at TP=4: 16 heads x 64, state 128, 2 groups, conv kernel 4. LNC=2 is simulated with a
two-program grid (one group per core). A batch of requests must equal each request run alone."""
import importlib.util
import os

import numpy as np
import pytest

nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")

_here = os.path.dirname(os.path.abspath(__file__))
_path = _here
for _ in range(8):
    _cand = os.path.join(_path, "vllm_neuron", "model", "nemotron_h", "mamba_decode_kernel.py")
    if os.path.exists(_cand):
        break
    _path = os.path.dirname(_path)
_spec = importlib.util.spec_from_file_location("nemotron_mamba_decode_kernel", _cand)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BF16 = ml_dtypes.bfloat16
f32 = np.float32


def _silu(v):
    return v / (1.0 + np.exp(-v))


def _reference(xBC, gate, dt, conv_state, conv_w, conv_b, dt_bias, A, D, state, norm_w, eps, G):
    """One request: inputs with a leading axis of 1."""
    H, P, N = state.shape[1:]
    I = H * P
    xBC, gate, dt, conv_state, conv_w, conv_b = (a.astype(f32) for a in (xBC, gate, dt, conv_state, conv_w, conv_b))
    conv_in = np.concatenate([conv_state[0], xBC[0][:, None]], axis=-1)      # [C, K]
    xbc = _silu((conv_in * conv_w).sum(-1) + conv_b)
    new_conv = conv_in[:, 1:][None]
    x = xbc[:I].reshape(H, P)
    B = np.repeat(xbc[I:I + G * N].reshape(G, N), H // G, axis=0)
    C = np.repeat(xbc[I + G * N:I + 2 * G * N].reshape(G, N), H // G, axis=0)
    dtv = np.log1p(np.exp(dt[0] + dt_bias))
    dA = np.exp(dtv * A)
    h = state[0] * dA[:, None, None] + (dtv[:, None, None] * B[:, None, :]) * x[..., None]
    y = (h * C[:, None, :]).sum(-1) + x * D[:, None]
    y = y.reshape(-1) * _silu(gate[0])
    yg = y.reshape(G, -1)
    yg = yg / np.sqrt((yg ** 2).mean(-1, keepdims=True) + eps)
    return (yg.reshape(-1) * norm_w)[None], new_conv, h[None]


@pytest.mark.parametrize("R", [1, 3])
def test_mamba2_decode_step_matches_reference(R):
    rng = np.random.default_rng(0)
    H, P, N, G, K = 16, 64, 128, 2, 4
    I = H * P
    C_dim = I + 2 * G * N
    xBC = rng.standard_normal((R, C_dim)).astype(BF16)
    gate = rng.standard_normal((R, I)).astype(BF16)
    dt = rng.standard_normal((R, H)).astype(BF16)
    conv_state = rng.standard_normal((R, C_dim, K - 1)).astype(BF16)
    conv_w = (rng.standard_normal((C_dim, K)) * 0.5).astype(BF16)
    conv_b = (rng.standard_normal(C_dim) * 0.1).astype(BF16)
    dt_bias = rng.standard_normal(H).astype(f32)
    A = -rng.uniform(1.0, 16.0, H).astype(f32)
    D = rng.uniform(0.5, 1.5, H).astype(f32)
    state = rng.standard_normal((R, H, P, N)).astype(f32)
    norm_w = rng.uniform(0.5, 1.5, I).astype(f32)
    eps = np.array([1e-5], f32)
    eye = np.eye(128, dtype=f32)
    y, conv_new, state_new = nki.simulate(_mod.mamba2_decode_step[G])(
        xBC, gate, dt, conv_state, conv_w, conv_b, dt_bias, A, D, state, norm_w, eye, eps)
    for r in range(R):
        one = slice(r, r + 1)
        ry, rconv, rstate = _reference(xBC[one], gate[one], dt[one], conv_state[one], conv_w, conv_b, dt_bias,
                                       A, D, state[one], norm_w, 1e-5, G)
        np.testing.assert_allclose(np.asarray(conv_new, f32)[one], rconv, rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(state_new, f32)[one], rstate, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.asarray(y, f32)[one], ry, rtol=2e-2, atol=2e-2)
