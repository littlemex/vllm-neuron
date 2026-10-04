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
    from vllm_neuron.model.nemotron_h import moe_prefill
except Exception as e:  # pragma: no cover
    pytest.skip(f"Neuron compiler stack unavailable: {e}", allow_module_level=True)


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_moe_prefill_traces():
    T, H, E, I = 512, 2688, 128, 464
    N = 139                                   # blocks build_blockwise_mapping makes for T=512, top-6
    out = wrap_nki(moe_prefill.moe_relu2_prefill)[2](
        _meta(T, H), _meta(T * E, 1, dtype=torch.float32), _meta(E, H, 1, I), _meta(E, I, H),
        _meta(N * moe_prefill.BLOCK_SIZE, dtype=torch.int32), _meta(N, dtype=torch.int32))
    assert out.shape == (T, H)


def test_ssd_prefill_traces():
    from vllm_neuron.model.nemotron_h import ssd_prefill_kernel
    L, H, P, N, G, K = 512, 16, 64, 128, 2, 4
    C_dim = H * P + 2 * G * N
    f32 = torch.float32
    y, s = wrap_nki(ssd_prefill_kernel.ssd_prefill)[2](
        _meta(K - 1 + L, C_dim), _meta(C_dim, K), _meta(C_dim), _meta(L, H, dtype=f32),
        _meta(H, dtype=f32), _meta(H, dtype=f32), _meta(H, P, N, dtype=f32), _meta(128, 128, dtype=f32))
    assert y.shape == (L, H, P) and s.shape == (H, P, N)
