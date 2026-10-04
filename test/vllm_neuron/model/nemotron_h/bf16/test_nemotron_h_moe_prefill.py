# SPDX-License-Identifier: Apache-2.0
"""The blockwise prefill MoE (moe_prefill.py: nkilib shard_on_I in up-only relu^2 mode, with the
up-only weight loader) on the NKI CPU simulator, against the dense all-experts formula the model's
CPU path uses. The routing mapping is built by the plugin's build_blockwise_mapping, as in serving.
Two physical cores (LNC=2) split the intermediate axis."""
import numpy as np
import pytest
import torch

nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")
try:
    from vllm_neuron.functional.moe.moe_blockwise import build_blockwise_mapping
    from vllm_neuron.model.nemotron_h import moe_prefill
    from vllm_neuron.model.nemotron_h.ops import dense_moe_gate
except Exception as e:  # pragma: no cover
    pytest.skip(f"plugin dependencies unavailable: {e}", allow_module_level=True)

BF16 = ml_dtypes.bfloat16


class _OneRank:
    world_size = 1
    rank_in_group = 0


def _bf(t):
    return t.to(torch.bfloat16).float()


@pytest.mark.parametrize("T", [256, 512])
def test_blockwise_prefill_matches_dense(T):
    g = torch.Generator().manual_seed(0)
    H, E, I, K = 512, 8, 32, 2
    x = _bf(torch.randn(T, H, generator=g))
    up = _bf(torch.randn(E, H, I, generator=g) * 0.05)
    down = _bf(torch.randn(E, I, H, generator=g) * 0.05)
    scores = torch.rand(T, E, generator=g)
    gate = dense_moe_gate(scores, torch.zeros(E), K, True, 2.5)          # [T, E], K nonzero per row
    aff, tok, b2e, _ = build_blockwise_mapping(
        expert_affinities=gate, num_local_experts=E, num_experts_per_token=K,
        block_size=moe_prefill.BLOCK_SIZE, moe_group=_OneRank(), tp_degree=1)
    out = nki.simulate(moe_prefill.moe_relu2_prefill[2])(
        x.numpy().astype(BF16), aff.numpy().astype(np.float32), up.unsqueeze(2).numpy().astype(BF16),
        down.numpy().astype(BF16), tok.to(torch.int32).numpy(), b2e.to(torch.int32).numpy())
    h = torch.relu(torch.einsum("td,edi->tei", x, up)).pow(2) * gate.unsqueeze(-1)
    ref = torch.einsum("tei,eid->td", h, down)
    got = torch.from_numpy(np.asarray(out, dtype=np.float32))
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
