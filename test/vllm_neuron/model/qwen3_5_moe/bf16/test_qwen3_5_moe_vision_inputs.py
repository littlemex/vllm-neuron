"""The host-side tensors the vision encoder's forward takes, against the reference helpers.

Six of the encoder's nine arguments are prepared outside the graph, and every one of them is padded to a
fixed block size. That makes the padding value part of the contract rather than an implementation
detail: a padded row carries a position, a bilinear corner and an attention bound like any other, so a
careless choice produces a plausible image embedding instead of an error. These tests pin each choice.

The per-patch quantities are checked by calling ``transformers.vision_utils`` — the reference — and
asserting the blocked tensors reproduce it on the real positions. What is left to test is the blocking
itself, which is this repository's code.
"""
import importlib.util
import os
import sys
import types

import pytest
import torch

# Walk up until the model directory is found, so the test survives being moved and does not depend on a
# fixed depth. The same idiom as the kernel tests next door.
_search = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    _MODEL_DIR = os.path.join(_search, "vllm_neuron", "model", "qwen3_5_moe")
    if os.path.exists(os.path.join(_MODEL_DIR, "vision_inputs.py")):
        break
    _search = os.path.dirname(_search)
else:  # pragma: no cover - only reachable if the test is moved out of the repo
    raise RuntimeError("could not locate vllm_neuron/model/qwen3_5_moe/")


# A stand-in package so the modules' relative imports resolve without importing the plugin (and a
# device with it). The same idiom as the kernel tests next door.
_PACKAGE = "_qwen3_5_moe_vision_under_test"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [_MODEL_DIR]
sys.modules[_PACKAGE] = _package


def _load(name):
    """Load one module by path, as a submodule of the stand-in package."""
    spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE}.{name}", os.path.join(_MODEL_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{_PACKAGE}.{name}"] = module
    spec.loader.exec_module(module)
    return module


_vi = _load("vision_inputs")

try:
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights,
        get_vision_cu_seqlens,
        get_vision_position_ids,
    )
    HAVE_HELPERS = True
except Exception:  # a transformers without the vision helpers
    HAVE_HELPERS = False

requires_helpers = pytest.mark.skipif(
    not HAVE_HELPERS, reason="installed transformers has no vision_utils helpers")


class _VisionConfig:
    """The published Qwen3.6-35B-A3B vision_config, as a plain object."""
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 16
    spatial_merge_size = 2
    num_position_embeddings = 2304
    hidden_size = 1152
    num_heads = 16

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads


def test_patch_dim_matches_the_projection_width():
    """3 channels x 2 temporal x 16 x 16. Derived, not passed, so a config change cannot desync it."""
    assert _vi.patch_dim(_VisionConfig()) == 3 * 2 * 16 * 16


@pytest.mark.parametrize("total,block,expected", [(1, 8, 1), (8, 8, 1), (9, 8, 2), (0, 8, 1)])
def test_block_count_rounds_up_and_never_returns_zero(total, block, expected):
    """An empty request must still be one padded block: a zero-length dimension is not the same thing
    as a padded one to the compiler."""
    assert _vi.block_count(total, block) == expected


def test_blocked_pixel_values_pads_with_zeros_and_keeps_the_real_rows():
    total, block, dim = 10, 8, 4
    flat = torch.arange(total * dim, dtype=torch.float32).reshape(total, dim)
    blocked = _vi.blocked_pixel_values(flat, block)
    assert blocked.shape == (2, block, dim)
    assert torch.equal(blocked.reshape(-1, dim)[:total], flat)
    assert torch.all(blocked.reshape(-1, dim)[total:] == 0)


def test_blocked_bilinear_pads_the_weight_with_zero_so_the_index_cannot_matter():
    """The zero weight is the load-bearing half of that pair.

    A padded position gathers the table's first row and multiplies it by nothing. Padding the INDEX out
    of range instead would be an out-of-bounds gather, which a compiled graph does not reliably report.
    """
    total, block = 5, 4
    indices = torch.arange(4 * total, dtype=torch.int64).reshape(4, total)
    weights = torch.full((4, total), 0.25)
    blocked_indices, blocked_weights = _vi.blocked_bilinear(indices, weights, block)
    assert blocked_indices.shape == (4, 2, block)
    assert blocked_indices.dtype == torch.int32
    flat_weights = blocked_weights.reshape(4, -1)
    assert torch.all(flat_weights[:, total:] == 0)
    assert torch.all(blocked_indices.reshape(4, -1)[:, total:] == 0)


def test_blocked_bounds_confine_each_patch_to_its_own_image():
    """Two images packed into shared blocks must not see each other.

    The bound is per patch, not per block, which is what lets a block boundary fall inside an image.
    """
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    bound_min, bound_max = _vi.blocked_bounds(cu_seqlens, total_patches=7, block_size=4)
    assert bound_min.shape == (2, 4, 1)
    flat_min = bound_min.reshape(-1).tolist()
    flat_max = bound_max.reshape(-1).tolist()
    assert flat_min[:3] == [0, 0, 0] and flat_max[:3] == [3, 3, 3]
    assert flat_min[3:7] == [3, 3, 3, 3] and flat_max[3:7] == [7, 7, 7, 7]
    # The padded tail gets an empty range: whatever it computes, it sees nothing.
    assert flat_min[7] == 0 and flat_max[7] == 0


def test_blocked_bounds_reject_a_cu_seqlens_that_does_not_cover_the_patches():
    """A boundary vector that disagrees with the patch count is a wiring bug upstream, and it would
    otherwise show up as an image attending over the wrong range."""
    with pytest.raises(ValueError, match="covers"):
        _vi.blocked_bounds(torch.tensor([0, 3], dtype=torch.int32), total_patches=7, block_size=4)


def test_vision_rotary_uses_two_axes_and_the_halves_layout():
    """Height and width take half the frequencies each, laid out as non-interleaved halves.

    Checked structurally rather than against a golden: cos of the first and second halves must be equal
    because the table is a concatenation of the same frequency block, and a position of zero must give
    cos one and sin zero for every channel.
    """
    head_dim = 72
    position_ids = torch.tensor([[0, 0], [1, 2], [3, 5]], dtype=torch.int64)
    cos, sin = _vi.vision_rotary_tables(position_ids, head_dim, 10000.0, dtype=torch.float32)
    assert cos.shape == (3, head_dim) and sin.shape == (3, head_dim)
    half = head_dim // 2
    assert torch.allclose(cos[:, :half], cos[:, half:])
    assert torch.allclose(sin[:, :half], sin[:, half:])
    assert torch.allclose(cos[0], torch.ones(head_dim))
    assert torch.allclose(sin[0], torch.zeros(head_dim))


def test_vision_rotary_rejects_a_head_dim_that_cannot_split_two_ways():
    with pytest.raises(ValueError, match="divisible by 4"):
        _vi.vision_rotary_tables(torch.zeros(2, 2, dtype=torch.int64), 70, 10000.0)


@requires_helpers
def test_build_vision_inputs_agrees_with_the_reference_helpers_on_the_real_positions():
    """End to end against the reference, for two images of different shapes in one request.

    Padding is asserted separately above; here what matters is that every REAL position carries exactly
    what the reference computed for it, in the right block slot.
    """
    config = _VisionConfig()
    grid_thw = torch.tensor([[1, 4, 6], [1, 2, 2]], dtype=torch.int64)
    total = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    torch.manual_seed(0)
    pixel_values = torch.randn(total, _vi.patch_dim(config))
    block_size = 16

    built = _vi.build_vision_inputs(pixel_values, grid_thw, config, block_size,
                                    dtype=torch.float32)

    grid_side = int(config.num_position_embeddings ** 0.5)
    indices, weights = get_vision_bilinear_indices_and_weights(
        grid_thw, grid_side, config.spatial_merge_size)
    position_ids = get_vision_position_ids(grid_thw, config.spatial_merge_size)
    cu_seqlens = get_vision_cu_seqlens(grid_thw)

    assert built["pixel_values"].reshape(-1, _vi.patch_dim(config))[:total].equal(pixel_values)
    assert built["pos_emb_idx"].reshape(4, -1)[:, :total].equal(indices.to(torch.int32))
    assert torch.allclose(built["pos_emb_weight"].reshape(4, -1)[:, :total],
                          weights.to(torch.float32))

    expected_cos, expected_sin = _vi.vision_rotary_tables(
        position_ids, config.head_dim, 10000.0, dtype=torch.float32)
    assert torch.allclose(built["cos"].reshape(-1, config.head_dim)[:total], expected_cos)
    assert torch.allclose(built["sin"].reshape(-1, config.head_dim)[:total], expected_sin)

    starts = cu_seqlens.to(torch.int64)[:-1]
    ends = cu_seqlens.to(torch.int64)[1:]
    lengths = ends - starts
    assert built["bound_min"].reshape(-1)[:total].equal(
        torch.repeat_interleave(starts, lengths).to(torch.int32))
    assert built["bound_max"].reshape(-1)[:total].equal(
        torch.repeat_interleave(ends, lengths).to(torch.int32))


@requires_helpers
def test_build_vision_inputs_keys_match_the_encoder_parameter_names():
    """Six positional tensors of similar rank is exactly the mistake type checking will not catch, so
    the builder returns a mapping keyed by the encoder's parameter names."""
    config = _VisionConfig()
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    total = 4
    built = _vi.build_vision_inputs(torch.zeros(total, _vi.patch_dim(config)), grid_thw, config, 8)
    assert set(built) == {"pixel_values", "pos_emb_idx", "pos_emb_weight", "cos", "sin",
                          "bound_min", "bound_max"}


# ---------------------------------------------------------------------------
# The MTP head's weight map, in both directions
# ---------------------------------------------------------------------------
# The first version of this map named the checkpoint's own keys as destinations, which does not work: the
# attention fuses q, k and v into one parameter. That mistake loads nothing and is invisible until the
# device produces garbage, so the destinations are checked against the module's real parameter names —
# the direction that a source-side diff against the checkpoint index cannot see.

_layout = _load("layout")


def test_mtp_sources_cover_the_checkpoint_head_exactly():
    """Both directions on the source side, against a real checkpoint index."""
    index_path = os.environ.get("QWEN3_5_MOE_CHECKPOINT_INDEX")
    if not index_path:
        pytest.skip("set QWEN3_5_MOE_CHECKPOINT_INDEX to a model.safetensors.index.json")
    import json
    with open(index_path) as handle:
        keys = set(json.load(handle)["weight_map"])
    head = {key for key in keys if key.startswith("mtp")}
    if not head:
        pytest.skip("this checkpoint has no MTP head")

    consumed = set()
    for value in _layout.mtp_checkpoint_mappings().values():
        consumed.update(value if isinstance(value, list) else [value])
    assert not consumed - head, sorted(consumed - head)
    assert not head - consumed, sorted(head - consumed)


def test_mtp_destinations_name_parameters_that_exist():
    """The direction the checkpoint diff cannot check.

    A destination that no module declares is silently never written, and the tensor keeps whatever it was
    initialised with. The fused attention parameter is exactly where this goes wrong, because the
    checkpoint's separate q/k/v keys read like destinations and are not.
    """
    expected = {
        "fc.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.qkvg_proj_weight", "self_attn.o_proj_weight",
        "self_attn.q_norm_weight", "self_attn.k_norm_weight",
        "mlp.router_weight", "mlp.gate_up_proj_weight", "mlp.down_proj_weight",
        "mlp.shared_gate_proj_weight", "mlp.shared_up_proj_weight",
        "mlp.shared_down_proj_weight", "mlp.shared_expert_gate_weight",
    }
    assert set(_layout.mtp_checkpoint_mappings()) == expected


def test_mtp_head_fuses_three_attention_sources_into_one_destination():
    """The fusion is the whole reason this map is written by hand rather than derived from key names."""
    mappings = _layout.mtp_checkpoint_mappings()
    fused = mappings["self_attn.qkvg_proj_weight"]
    assert isinstance(fused, list) and len(fused) == 3
    assert [key.rsplit(".", 2)[-2] for key in fused] == ["q_proj", "k_proj", "v_proj"]
