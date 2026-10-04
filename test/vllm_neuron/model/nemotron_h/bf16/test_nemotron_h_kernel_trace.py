# SPDX-License-Identifier: Apache-2.0
"""Every NemotronH NKI kernel through the device compiler's frontend, at the real per-rank shapes
(TP=4, LNC=2), on meta tensors. The NKI CPU simulator accepts constructs the device frontend rejects
(asserts, comprehensions, nested defs, module attribute calls inside a kernel), which otherwise only
show up as a failed warmup on the Neuron host. Needs the Neuron compiler stack, not a device."""
import os

import pytest
import torch

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
try:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from vllm_neuron.model.nemotron_h import (mamba_decode_kernel, matvec_kernel, moe_decode_kernel,
                                              ssd_prefill_kernel)
except Exception as e:  # pragma: no cover
    pytest.skip(f"Neuron compiler stack unavailable: {e}", allow_module_level=True)

BF, F32, I32 = torch.bfloat16, torch.float32, torch.int32
H_MODEL, H_SSM, P, N, G, K = 2688, 16, 64, 128, 2, 4      # per rank at TP=4: 16 Mamba heads, 2 groups


def _meta(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_ssd_prefill_traces():
    L = 512
    y, s = wrap_nki(ssd_prefill_kernel.ssd_prefill)[2](
        _meta(L, H_SSM, P, dtype=F32), _meta(L, H_SSM, dtype=F32), _meta(H_SSM, dtype=F32),
        _meta(L, G, N, dtype=F32), _meta(L, G, N, dtype=F32), _meta(H_SSM, dtype=F32),
        _meta(H_SSM, P, N, dtype=F32), _meta(128, 128, dtype=F32))
    assert y.shape == (L, H_SSM, P) and s.shape == (H_SSM, P, N)


@pytest.mark.parametrize("R", [1, 8])
def test_mamba_decode_traces(R):
    I = H_SSM * P
    C = I + 2 * G * N
    sel = mamba_decode_kernel.head_selector(H_SSM // G, P)
    y, conv, st = wrap_nki(mamba_decode_kernel.mamba2_decode_step)[2](
        _meta(R, C), _meta(R, I), _meta(R, H_SSM), _meta(R, C, K - 1), _meta(C, K), _meta(C),
        _meta(H_SSM, dtype=F32), _meta(H_SSM, dtype=F32), _meta(H_SSM, dtype=F32),
        _meta(R, H_SSM, P, N, dtype=F32), _meta(I, dtype=F32), _meta(128, 128, dtype=F32),
        _meta(*sel.shape, dtype=F32), _meta(1, dtype=F32))
    assert y.shape == (R, I) and st.shape == (R, H_SSM, P, N)


@pytest.mark.parametrize("T", [1, 8])
def test_moe_decode_traces(T):
    E, I, topk = 128, 464, 6
    out = wrap_nki(moe_decode_kernel.moe_relu2_decode)[2](
        _meta(T, H_MODEL), _meta(E, H_MODEL, I), _meta(E, I, H_MODEL), _meta(T, topk, dtype=I32),
        _meta(T, topk, dtype=F32))
    assert out.shape == (2, T, H_MODEL)


@pytest.mark.parametrize("T,h,n", [(1, H_MODEL, 2576),      # Mamba in_proj (gate 1024 + xBC 1536 + dt 16)
                                   (8, 1024, H_MODEL),      # Mamba out_proj
                                   (8, H_MODEL, 32768)])    # vocabulary projection
def test_matvec_traces(T, h, n):
    out = wrap_nki(matvec_kernel.matvec)[2](_meta(T, h), _meta(h, n))
    assert out.shape == (T, n)
