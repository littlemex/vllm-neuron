# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence tests for the Qwen3.5-MoE Gated DeltaNet scans and helper ops.

No Neuron device and no checkpoint required. The scans are loaded from the SHIPPED module
(``vllm_neuron/model/qwen3_5_moe/gdn.py``) so the tested code and the served code cannot diverge.

Two independent oracles are used deliberately:

* the ACTUAL HuggingFace reference (``torch_chunk_gated_delta_rule`` /
  ``torch_recurrent_gated_delta_rule``), not a self-authored re-derivation. Qwen3.5-MoE reuses the
  dense Qwen3.5 GDN core, which is the Qwen3-Next kernel, so ``transformers.models.qwen3_next`` is
  accepted as the same reference when the ``qwen3_5_moe`` module is not present in the installed
  transformers; and
* the token-by-token recurrence, which pins the chunked form against the definition itself rather
  than against another chunked implementation.
"""
import importlib
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

# Load the SHIPPED modules directly, as a synthetic package so their relative imports resolve. Going
# through the file (rather than `import vllm_neuron...`) avoids the plugin package __init__ chain,
# which pulls in Neuron-only dependencies — so these tests run on any CPU box.
_search = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    _model_dir = os.path.join(_search, "vllm_neuron", "model", "qwen3_5_moe")
    if os.path.exists(os.path.join(_model_dir, "gdn.py")):
        break
    _search = os.path.dirname(_search)
else:  # pragma: no cover - only reachable if the test is moved out of the repo
    raise RuntimeError("could not locate vllm_neuron/model/qwen3_5_moe/")

_PACKAGE = "_qwen3_5_moe_under_test"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [_model_dir]
sys.modules[_PACKAGE] = _package
_loaded = {}
for _name in ("ops", "gdn", "config", "layout"):
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE}.{_name}", os.path.join(_model_dir, f"{_name}.py"))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE}.{_name}"] = _module
    _spec.loader.exec_module(_module)
    _loaded[_name] = _module
_gdn, _ops, _layout, _config = (_loaded["gdn"], _loaded["ops"], _loaded["layout"],
                                _loaded["config"])

DEFAULT_CHUNK_SIZE = _gdn.DEFAULT_CHUNK_SIZE
_unit_lower_triangular_inverse = _gdn._unit_lower_triangular_inverse
chunk_gated_delta_rule = _gdn.chunk_gated_delta_rule
recurrent_gated_delta_rule = _gdn.recurrent_gated_delta_rule
segmented_causal_conv1d = _gdn.segmented_causal_conv1d
gated_delta_net_prefill = _gdn.gated_delta_net_prefill
gated_delta_net_decode = _gdn.gated_delta_net_decode

rmsnorm = _ops.rmsnorm
gated_rmsnorm = _ops.gated_rmsnorm
rotary_tables = _ops.rotary_tables
apply_partial_rotary = _ops.apply_partial_rotary
gated_delta_projections = _ops.gated_delta_projections
redirect_padded_slots = _ops.redirect_padded_slots

fuse_attention_qkvg = _layout.fuse_attention_qkvg
gdn_qkv_rows = _layout.gdn_qkv_rows
shard_expert_gate_up = _layout.shard_expert_gate_up
shard_expert_down = _layout.shard_expert_down
shard_rows_transposed = _layout.shard_rows_transposed
shard_columns_transposed = _layout.shard_columns_transposed
shard_heads = _layout.shard_heads
checkpoint_mappings = _layout.checkpoint_mappings
Qwen3_5MoeConfig = _config.Qwen3_5MoeConfig


def _hf_reference():
    """The installed HF GDN reference, or None. See the module docstring for why qwen3_next counts."""
    for module_name in ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                        "transformers.models.qwen3_next.modeling_qwen3_next"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        chunk = getattr(module, "torch_chunk_gated_delta_rule", None)
        recurrent = getattr(module, "torch_recurrent_gated_delta_rule", None)
        if chunk is not None and recurrent is not None:
            return chunk, recurrent
    return None


HF_REFERENCE = _hf_reference()
requires_hf = pytest.mark.skipif(
    HF_REFERENCE is None,
    reason="no installed transformers GDN reference (qwen3_5_moe or qwen3_next)",
)

# Shapes: the real checkpoint's per-head dims, with a head count small enough to stay quick.
DK = 128
DV = 128
HEADS = 4


def _inputs(seq_len, heads=HEADS, batch=1, seed=0, decay_scale=2.0):
    """Random GDN inputs. ``g`` must be <= 0 (it is a log decay) and beta in (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, seq_len, heads, DK, generator=gen)
    key = torch.randn(batch, seq_len, heads, DK, generator=gen)
    value = torch.randn(batch, seq_len, heads, DV, generator=gen)
    g = -torch.rand(batch, seq_len, heads, generator=gen) * decay_scale
    beta = torch.rand(batch, seq_len, heads, generator=gen)
    return query, key, value, g, beta


# --------------------------------------------------------------------------------------
# The exact triangular inverse that replaces solve_triangular / forward substitution
# --------------------------------------------------------------------------------------
def _scan_realistic_system(chunk_size, seed=0, decay_scale=2.0, key_correlation=0.0):
    """Build ``strict_lower`` exactly the way the scan builds it: l2-normalised keys, beta in (0, 1),
    and a causal decay factor <= 1. ``key_correlation`` pushes the keys toward a shared direction and
    ``decay_scale`` weakens the decay; together they raise ``||N||_2``, which is the regime that
    separates a numerically sound inversion from an algebraically-exact but useless one."""
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(chunk_size, DK, generator=gen)
    if key_correlation:
        key = key + key_correlation * torch.randn(1, DK, generator=gen)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    beta = torch.rand(chunk_size, generator=gen)
    cumulative = (-torch.rand(chunk_size, generator=gen) * decay_scale).cumsum(0)
    index = torch.arange(chunk_size)
    decay = (cumulative[:, None] - cumulative[None, :]).masked_fill(
        index[None, :] > index[:, None], float("-inf")).exp()
    return (((key * beta[:, None]) @ key.T) * decay).tril(-1)


@pytest.mark.parametrize("chunk_size", [2, 4, 16, 32, 64, 100, 128])
def test_unit_lower_triangular_inverse_matches_triangular_solver(chunk_size):
    """The inverse must agree with ``solve_triangular`` (what HF uses on the eager path) on
    scan-realistic systems, including a chunk size that is not a power of two."""
    strict_lower = _scan_realistic_system(chunk_size, seed=7)
    system = torch.eye(chunk_size) + strict_lower
    inverse = _unit_lower_triangular_inverse(strict_lower, chunk_size)

    identity = inverse @ system
    assert torch.allclose(identity, torch.eye(chunk_size), atol=1e-5), \
        (identity - torch.eye(chunk_size)).abs().max()
    reference = torch.linalg.solve_triangular(
        system, torch.eye(chunk_size), upper=False, unitriangular=True)
    assert torch.allclose(inverse, reference, atol=1e-5), (inverse - reference).abs().max()


@pytest.mark.parametrize("decay_scale,key_correlation", [(0.1, 1.0), (0.01, 3.0), (0.01, 8.0)])
def test_unit_lower_triangular_inverse_survives_large_operator_norm(decay_scale, key_correlation):
    """The load-bearing property. A weak decay plus correlated keys — ordinary for adjacent tokens —
    drives ``||N||_2`` well above 1 while the true inverse stays well conditioned. A Neumann /
    repeated-squaring form is algebraically exact here but materialises powers up to ``N^(C-1)`` and
    loses everything in fp32; this block form must stay as accurate as the triangular solver.
    """
    chunk_size = 64
    strict_lower = _scan_realistic_system(
        chunk_size, seed=7, decay_scale=decay_scale, key_correlation=key_correlation)
    operator_norm = torch.linalg.matrix_norm(strict_lower, 2)
    assert operator_norm > 1.0, f"this case is meant to exceed ||N||_2 = 1, got {operator_norm}"

    truth = torch.linalg.inv(torch.eye(chunk_size, dtype=torch.float64) + strict_lower.double())
    inverse = _unit_lower_triangular_inverse(strict_lower, chunk_size)
    solver = torch.linalg.solve_triangular(
        torch.eye(chunk_size) + strict_lower, torch.eye(chunk_size),
        upper=False, unitriangular=True)
    block_error = (inverse.double() - truth).abs().max()
    solver_error = (solver.double() - truth).abs().max()
    assert block_error < 1e-5, block_error
    # Not "close to the solver" but "no worse than a small factor of it": this is what fails if the
    # inversion is reformulated into something that cancels large intermediates.
    assert block_error <= 10 * solver_error + 1e-9, (block_error, solver_error)


def test_unit_lower_triangular_inverse_covers_full_dependency_chain():
    """A maximally-deep dependency chain: the sub-diagonal-only system's inverse is non-zero all the
    way to the corner, which only appears if every level of the block recursion ran."""
    chunk_size = 64
    sub_diagonal = torch.diag(torch.ones(chunk_size - 1), -1)
    inverse = _unit_lower_triangular_inverse(sub_diagonal, chunk_size)
    expected = torch.linalg.inv(torch.eye(chunk_size) + sub_diagonal)
    assert torch.allclose(inverse, expected, atol=1e-6)
    assert inverse[chunk_size - 1, 0].abs().item() == pytest.approx(1.0)


def test_coupling_masks_partition_the_strict_lower_triangle():
    """Every strictly-lower entry must be assigned to exactly one level, or the recursion either
    drops a coupling (wrong answer) or applies one twice."""
    for chunk_size in (2, 3, 4, 5, 8, 64, 100):
        masks = _gdn._coupling_masks(chunk_size, torch.device("cpu"), torch.float32)
        total = sum(masks)
        index = torch.arange(chunk_size)
        strict_lower = (index[:, None] > index[None, :]).to(torch.float32)
        assert torch.equal(total, strict_lower), chunk_size
        assert len(masks) == max(1, (chunk_size - 1).bit_length())


# --------------------------------------------------------------------------------------
# Chunked prefill scan vs the HF reference and vs the definition
# --------------------------------------------------------------------------------------
@requires_hf
@pytest.mark.parametrize("seq_len", [8, 37, 64, 100, 128, 192])
def test_chunk_matches_hf_reference(seq_len):
    hf_chunk, _ = HF_REFERENCE
    query, key, value, g, beta = _inputs(seq_len)
    ref_out, ref_state = hf_chunk(
        query, key, value, g=g, beta=beta, chunk_size=DEFAULT_CHUNK_SIZE,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    out, state = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE)
    assert torch.allclose(out, ref_out, atol=1e-5), (out - ref_out).abs().max()
    assert torch.allclose(state, ref_state, atol=1e-5), (state - ref_state).abs().max()


@requires_hf
def test_chunk_matches_hf_reference_with_initial_state():
    hf_chunk, _ = HF_REFERENCE
    query, key, value, g, beta = _inputs(96, seed=3)
    initial = torch.randn(1, HEADS, DK, DV, generator=torch.Generator().manual_seed(4)) * 0.1
    ref_out, ref_state = hf_chunk(
        query, key, value, g=g, beta=beta, chunk_size=DEFAULT_CHUNK_SIZE,
        initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    out, state = chunk_gated_delta_rule(query, key, value, g, beta,
                                        chunk_size=DEFAULT_CHUNK_SIZE, initial_state=initial)
    assert torch.allclose(out, ref_out, atol=1e-5), (out - ref_out).abs().max()
    assert torch.allclose(state, ref_state, atol=1e-5), (state - ref_state).abs().max()


@requires_hf
def test_recurrent_matches_hf_reference():
    _, hf_recurrent = HF_REFERENCE
    query, key, value, g, beta = _inputs(1, seed=5)
    for initial in (None, torch.randn(1, HEADS, DK, DV,
                                      generator=torch.Generator().manual_seed(6)) * 0.1):
        ref_out, ref_state = hf_recurrent(
            query, key, value, g=g, beta=beta, initial_state=initial,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        out, state = recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=initial)
        assert torch.allclose(out, ref_out, atol=1e-5), (out - ref_out).abs().max()
        assert torch.allclose(state, ref_state, atol=1e-5), (state - ref_state).abs().max()


@pytest.mark.parametrize("seq_len", [1, 5, 64, 130])
def test_chunk_matches_token_by_token_recurrence(seq_len):
    """The chunked form against the DEFINITION: apply the one-step recurrence token by token."""
    query, key, value, g, beta = _inputs(seq_len, seed=11)
    out, state = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE)
    running_state = None
    steps = []
    for t in range(seq_len):
        step_out, running_state = recurrent_gated_delta_rule(
            query[:, t:t + 1], key[:, t:t + 1], value[:, t:t + 1],
            g[:, t:t + 1], beta[:, t:t + 1], initial_state=running_state,
        )
        steps.append(step_out)
    sequential = torch.cat(steps, dim=1)
    assert torch.allclose(out, sequential, atol=1e-5), (out - sequential).abs().max()
    assert torch.allclose(state, running_state, atol=1e-5), (state - running_state).abs().max()


@pytest.mark.parametrize("chunk_size", [16, 32, 128])
def test_chunk_size_does_not_change_the_result(chunk_size):
    """The chunk size is a tiling choice, not a numerical one: any size must give the same answer
    (this is what lets the served chunk size be tuned for compile time)."""
    query, key, value, g, beta = _inputs(128, seed=13)
    baseline_out, baseline_state = chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE)
    out, state = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=chunk_size)
    assert torch.allclose(out, baseline_out, atol=1e-5), (out - baseline_out).abs().max()
    assert torch.allclose(state, baseline_state, atol=1e-5), (state - baseline_state).abs().max()


def test_segmented_prefill_matches_single_shot():
    """Splitting a prompt into segments and carrying the recurrent state must equal one pass."""
    query, key, value, g, beta = _inputs(128, seed=17)
    single_out, single_state = chunk_gated_delta_rule(query, key, value, g, beta,
                                                     chunk_size=DEFAULT_CHUNK_SIZE)
    split = 64
    first_out, first_state = chunk_gated_delta_rule(
        query[:, :split], key[:, :split], value[:, :split], g[:, :split], beta[:, :split],
        chunk_size=DEFAULT_CHUNK_SIZE)
    second_out, second_state = chunk_gated_delta_rule(
        query[:, split:], key[:, split:], value[:, split:], g[:, split:], beta[:, split:],
        chunk_size=DEFAULT_CHUNK_SIZE, initial_state=first_state)
    assert torch.allclose(torch.cat([first_out, second_out], dim=1), single_out, atol=1e-5)
    assert torch.allclose(second_state, single_state, atol=1e-5)


def test_bucket_padding_does_not_change_the_state():
    """Neuron right-pads a prefill to a fixed bucket width. Zeroing the log decay and beta on the
    pad positions must make those steps the identity, so the state handed to decode is the state at
    the last REAL token regardless of how much padding the bucket adds."""
    seq_len, pad_len = 100, 28
    query, key, value, g, beta = _inputs(seq_len, seed=19)
    _, real_state = chunk_gated_delta_rule(query, key, value, g, beta,
                                           chunk_size=DEFAULT_CHUNK_SIZE)
    gen = torch.Generator().manual_seed(20)
    def pad(tensor, last):
        return torch.cat([tensor, torch.randn(1, pad_len, HEADS, last, generator=gen)], dim=1)
    padded_out, padded_state = chunk_gated_delta_rule(
        pad(query, DK), pad(key, DK), pad(value, DV),
        torch.cat([g, torch.zeros(1, pad_len, HEADS)], dim=1),
        torch.cat([beta, torch.zeros(1, pad_len, HEADS)], dim=1),
        chunk_size=DEFAULT_CHUNK_SIZE,
    )
    assert torch.allclose(padded_state, real_state, atol=1e-6), \
        (padded_state - real_state).abs().max()
    # And the outputs at the real positions are untouched by the padding.
    real_out, _ = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE)
    assert torch.allclose(padded_out[:, :seq_len], real_out, atol=1e-6)


def test_unmasked_pad_state_diverges():
    """Negative test for the test above: leaving the pad positions' decay and beta at their raw
    values DOES corrupt the state, so the masking is load-bearing rather than decorative."""
    seq_len, pad_len = 100, 28
    query, key, value, g, beta = _inputs(seq_len, seed=19)
    _, real_state = chunk_gated_delta_rule(query, key, value, g, beta,
                                           chunk_size=DEFAULT_CHUNK_SIZE)
    gen = torch.Generator().manual_seed(21)
    def pad(tensor, last):
        return torch.cat([tensor, torch.randn(1, pad_len, HEADS, last, generator=gen)], dim=1)
    _, wrong_state = chunk_gated_delta_rule(
        pad(query, DK), pad(key, DK), pad(value, DV),
        torch.cat([g, -torch.rand(1, pad_len, HEADS, generator=gen) * 2.0], dim=1),
        torch.cat([beta, torch.rand(1, pad_len, HEADS, generator=gen)], dim=1),
        chunk_size=DEFAULT_CHUNK_SIZE,
    )
    assert not torch.allclose(wrong_state, real_state, atol=1e-3)


def test_decay_is_masked_before_exp():
    """The pairwise decay exponent must be masked to -inf BEFORE ``exp``. Masking after ``exp`` lets
    the strictly upper triangle overflow to +inf, and ``inf * 0`` is NaN. This reproduces the trap
    on the same shapes the scan uses, so the ordering inside the scan is not an accident."""
    chunk_size = 64
    # A large negative decay per step: the cumulative exponent difference reaches ~ +2000 for the
    # upper triangle, which overflows fp32's exp.
    cumulative = torch.arange(chunk_size, dtype=torch.float32) * -32.0
    exponent = cumulative[:, None] - cumulative[None, :]
    causal = torch.arange(chunk_size)[:, None] >= torch.arange(chunk_size)[None, :]

    masked_then_exp = exponent.masked_fill(~causal, float("-inf")).exp()
    assert torch.isfinite(masked_then_exp).all()
    assert not torch.isnan(masked_then_exp @ torch.ones(chunk_size, 4)).any()

    exp_then_masked = exponent.exp() * causal.to(torch.float32)
    assert torch.isinf(exponent.exp()).any(), "the trap needs an actually overflowing exponent"
    assert torch.isnan(exp_then_masked).any(), "inf * 0 should have produced NaN"


def test_scan_survives_extreme_decay():
    """End-to-end guard for the trap above on the real scan: a very large negative log decay must
    not produce NaN or Inf anywhere."""
    query, key, value, g, beta = _inputs(128, seed=23)
    g = g * 40.0  # log decay down to ~ -80 per step
    out, state = chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE)
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()


def test_empty_sequence_raises():
    query, key, value, g, beta = _inputs(4)
    with pytest.raises(ValueError):
        chunk_gated_delta_rule(query[:, :0], key[:, :0], value[:, :0], g[:, :0], beta[:, :0])


def test_recurrent_rejects_multi_token():
    """The decode path advances the state by exactly one token; a multi-token step must raise
    rather than silently processing only the first (this is what makes spec-decode fail loudly)."""
    query, key, value, g, beta = _inputs(2)
    with pytest.raises(ValueError):
        recurrent_gated_delta_rule(query, key, value, g, beta)


# --------------------------------------------------------------------------------------
# Depthwise causal conv1d history carry
# --------------------------------------------------------------------------------------
def _conv_setup(conv_dim=16, kernel_size=4, seq_len=32, seed=29):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, conv_dim, seq_len, generator=gen)
    weight = torch.randn(conv_dim, 1, kernel_size, generator=gen)
    return x, weight, kernel_size, conv_dim


def test_segmented_conv1d_matches_single_shot():
    x, weight, kernel_size, conv_dim = _conv_setup()
    single, single_state = segmented_causal_conv1d(x, weight, kernel_size, conv_dim)

    split = 16
    first, first_state = segmented_causal_conv1d(x[..., :split], weight, kernel_size, conv_dim)
    second, second_state = segmented_causal_conv1d(
        x[..., split:], weight, kernel_size, conv_dim,
        conv_state=first_state, is_continuation=torch.ones(()),
    )
    assert torch.allclose(torch.cat([first, second], dim=-1), single, atol=1e-5)
    assert torch.allclose(second_state, single_state, atol=1e-6)


def test_segmented_conv1d_first_segment_mask_zeroes_history():
    """``is_continuation=0`` must make a segment behave exactly like a fresh single-shot prefill,
    so the first segment needs no Python branch on a runtime value."""
    x, weight, kernel_size, conv_dim = _conv_setup()
    fresh, fresh_state = segmented_causal_conv1d(x, weight, kernel_size, conv_dim)
    garbage = torch.randn(1, conv_dim, kernel_size - 1,
                          generator=torch.Generator().manual_seed(31))
    masked, masked_state = segmented_causal_conv1d(
        x, weight, kernel_size, conv_dim,
        conv_state=garbage, is_continuation=torch.zeros(()),
    )
    assert torch.allclose(masked, fresh, atol=1e-6)
    assert torch.allclose(masked_state, fresh_state, atol=1e-6)


def test_segmented_conv1d_valid_len_ignores_bucket_padding():
    """With bucket padding appended, the carried conv state must be the last kernel_size-1 REAL
    inputs, so it equals the state produced from the unpadded segment."""
    x, weight, kernel_size, conv_dim = _conv_setup(seq_len=20)
    _, real_state = segmented_causal_conv1d(x, weight, kernel_size, conv_dim)
    padded = torch.cat(
        [x, torch.randn(1, conv_dim, 12, generator=torch.Generator().manual_seed(37))], dim=-1)
    _, padded_state = segmented_causal_conv1d(
        padded, weight, kernel_size, conv_dim,
        conv_state=torch.zeros(1, conv_dim, kernel_size - 1),
        is_continuation=torch.zeros(()), valid_len=torch.tensor(x.shape[-1]),
    )
    assert torch.allclose(padded_state, real_state, atol=1e-6)


def test_segmented_conv1d_short_segment_reads_into_history():
    """A segment shorter than kernel_size-1 must carry a state that mixes the prior history with the
    new tokens (the low gather indices fall into the history prefix), not just the new tokens."""
    kernel_size, conv_dim = 4, 8
    gen = torch.Generator().manual_seed(41)
    history = torch.randn(1, conv_dim, kernel_size - 1, generator=gen)
    short = torch.randn(1, conv_dim, 1, generator=gen)
    _, state = segmented_causal_conv1d(
        short, torch.randn(conv_dim, 1, kernel_size, generator=gen), kernel_size, conv_dim,
        conv_state=history, is_continuation=torch.ones(()), valid_len=torch.tensor(1),
    )
    expected = torch.cat([history, short], dim=-1)[..., -(kernel_size - 1):]
    assert torch.allclose(state, expected, atol=1e-6)


def test_conv_state_width_is_kernel_minus_one():
    """Regression guard: the carried conv state must always be exactly kernel_size-1 wide, whatever
    the segment length, so an in-place ``copy_`` into the state buffer cannot silently broadcast."""
    for seq_len in (1, 2, 3, 4, 17):
        x, weight, kernel_size, conv_dim = _conv_setup(seq_len=seq_len)
        _, state = segmented_causal_conv1d(x, weight, kernel_size, conv_dim)
        assert state.shape == (1, conv_dim, kernel_size - 1), (seq_len, state.shape)



# --------------------------------------------------------------------------------------
# Parity against the ACTUAL HF reference modules (not a re-derivation)
# --------------------------------------------------------------------------------------
def _hf_modeling():
    try:
        return importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    except ImportError:
        return None


HF_MODELING = _hf_modeling()
requires_hf_modules = pytest.mark.skipif(
    HF_MODELING is None,
    reason="installed transformers has no qwen3_5_moe modeling module",
)

# A small config with the checkpoint's SHAPE (value heads a multiple of key heads, conv kernel 4,
# head dims equal) but tiny widths, so the tests stay fast while exercising the real code paths.
_TEST_DIMS = {"k_heads": 4, "v_heads": 8, "head_k_dim": 16, "head_v_dim": 16, "kernel": 4}
_TEST_HIDDEN = 64
_TEST_EPS = 1e-6


def _hf_text_config():
    config_module = importlib.import_module(
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    return config_module.Qwen3_5MoeTextConfig(
        hidden_size=_TEST_HIDDEN, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=_TEST_DIMS["k_heads"], linear_num_value_heads=_TEST_DIMS["v_heads"],
        linear_key_head_dim=_TEST_DIMS["head_k_dim"], linear_value_head_dim=_TEST_DIMS["head_v_dim"],
        linear_conv_kernel_dim=_TEST_DIMS["kernel"],
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, vocab_size=100, rms_norm_eps=_TEST_EPS,
    )


def _hf_gated_delta_net(seed=0):
    """An HF Gated DeltaNet with randomised but in-distribution parameters.

    ``A_log`` and ``dt_bias`` follow the reference initialisation (``A`` uniform on (0.01, 16), bias
    one); leaving them at the module defaults would make the log decay a constant and hide any error
    in how the decay is applied.
    """
    torch.manual_seed(seed)
    config = _hf_text_config()
    module = HF_MODELING.Qwen3_5MoeGatedDeltaNet(config, 0).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
        module.A_log.copy_(torch.empty(_TEST_DIMS["v_heads"]).uniform_(0.01, 16).log())
        module.dt_bias.fill_(1.0)
        module.norm.weight.fill_(1.0)
    return module, config


def _weights_from_hf(module):
    """The shipped mixer's weight mapping, taken from an HF module.

    The projections are transposed because the served path stores ``[in, out]`` (it right-multiplies
    the hidden states) where ``nn.Linear`` stores ``[out, in]``. Getting this backwards is the classic
    silent port bug, so it is exercised here rather than only on device.
    """
    return {
        "in_proj_qkv": module.in_proj_qkv.weight.T.contiguous(),
        "in_proj_z": module.in_proj_z.weight.T.contiguous(),
        "in_proj_b": module.in_proj_b.weight.T.contiguous(),
        "in_proj_a": module.in_proj_a.weight.T.contiguous(),
        "conv1d": module.conv1d.weight,
        "A_log": module.A_log,
        "dt_bias": module.dt_bias,
        "norm": module.norm.weight,
        "out_proj": module.out_proj.weight.T.contiguous(),
    }


@requires_hf_modules
def test_rmsnorm_matches_hf():
    """The residual-stream norm scales by ``1 + weight``. A ``weight * x`` norm would pass a
    zero-weight smoke test and then collapse the real checkpoint, so pin it against the reference."""
    torch.manual_seed(0)
    reference = HF_MODELING.Qwen3_5MoeRMSNorm(_TEST_HIDDEN, eps=_TEST_EPS)
    with torch.no_grad():
        reference.weight.normal_(0, 0.5)
    x = torch.randn(7, _TEST_HIDDEN)
    assert torch.allclose(rmsnorm(x, reference.weight, _TEST_EPS), reference(x), atol=1e-6)


@requires_hf_modules
def test_rmsnorm_offset_convention_is_load_bearing():
    """With the checkpoint's near-zero norm weights, the plugin's usual ``weight * x`` convention
    would scale the residual stream to nearly nothing. Show the two are not interchangeable."""
    weight = torch.full((_TEST_HIDDEN,), 0.01)
    x = torch.randn(4, _TEST_HIDDEN)
    offset = rmsnorm(x, weight, _TEST_EPS)
    plain = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + _TEST_EPS) * weight
    assert offset.abs().max() > 50 * plain.abs().max()


@requires_hf_modules
def test_gated_rmsnorm_matches_hf():
    """The Gated DeltaNet's own norm uses the OTHER convention (``weight * x``, weight initialised to
    one) and applies the gate through SiLU after normalising."""
    torch.manual_seed(1)
    head_v = _TEST_DIMS["head_v_dim"]
    reference = HF_MODELING.Qwen3_5MoeRMSNormGated(head_v, eps=_TEST_EPS)
    with torch.no_grad():
        reference.weight.normal_(1.0, 0.1)
    x = torch.randn(11, head_v)
    gate = torch.randn(11, head_v)
    assert torch.allclose(
        gated_rmsnorm(x, gate, reference.weight, _TEST_EPS, torch.float32),
        reference(x, gate), atol=1e-6)


@requires_hf_modules
def test_gated_norm_applies_gate_after_norm():
    """Negative test for the ordering: gating before the norm is a different function."""
    torch.manual_seed(2)
    head_v = _TEST_DIMS["head_v_dim"]
    weight = torch.ones(head_v)
    x = torch.randn(5, head_v)
    gate = torch.randn(5, head_v)
    norm_then_gate = gated_rmsnorm(x, gate, weight, _TEST_EPS, torch.float32)
    gated = x * torch.nn.functional.silu(gate)
    gate_then_norm = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + _TEST_EPS)
    assert not torch.allclose(norm_then_gate, gate_then_norm, atol=1e-3)


@requires_hf_modules
def test_rotary_tables_match_hf_for_text_only_positions():
    """Text-only justification for using a single-axis partial rotary table.

    The reference builds three positional axes and interleaves their frequencies. For text every axis
    carries the same position, so the interleave selects from three identical tensors and the result
    must equal the single-axis table. This test is what makes the vision path's exclusion a
    correctness boundary rather than an approximation.
    """
    config = _hf_text_config()
    reference = HF_MODELING.Qwen3_5MoeTextRotaryEmbedding(config)
    positions = torch.arange(23).unsqueeze(0)
    dummy = torch.zeros(1, 23, config.hidden_size)
    ref_cos, ref_sin = reference(dummy, positions)

    rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    cos, sin = rotary_tables(rotary_dim, 32, config.rope_parameters["rope_theta"])
    picked_cos = cos.index_select(0, positions.reshape(-1))
    picked_sin = sin.index_select(0, positions.reshape(-1))
    assert ref_cos.shape[-1] == rotary_dim, (ref_cos.shape, rotary_dim)
    assert torch.allclose(picked_cos, ref_cos.reshape(-1, rotary_dim), atol=1e-6)
    assert torch.allclose(picked_sin, ref_sin.reshape(-1, rotary_dim), atol=1e-6)


@requires_hf_modules
def test_apply_partial_rotary_matches_hf():
    """Rotate the leading channels and pass the rest through, exactly as the reference does."""
    torch.manual_seed(4)
    config = _hf_text_config()
    head_dim = config.head_dim
    rotary_dim = int(head_dim * config.rope_parameters["partial_rotary_factor"])
    tokens, heads = 6, 3
    query = torch.randn(1, heads, tokens, head_dim)
    key = torch.randn(1, heads, tokens, head_dim)
    cos = torch.randn(1, tokens, rotary_dim)
    sin = torch.randn(1, tokens, rotary_dim)
    ref_q, ref_k = HF_MODELING.apply_rotary_pos_emb(query, key, cos, sin)
    mine_q = apply_partial_rotary(query, cos.unsqueeze(1), sin.unsqueeze(1))
    mine_k = apply_partial_rotary(key, cos.unsqueeze(1), sin.unsqueeze(1))
    assert torch.allclose(mine_q, ref_q, atol=1e-6), (mine_q - ref_q).abs().max()
    assert torch.allclose(mine_k, ref_k, atol=1e-6), (mine_k - ref_k).abs().max()
    # And the pass-through channels really are untouched.
    assert torch.equal(mine_q[..., rotary_dim:], query[..., rotary_dim:])


@requires_hf_modules
def test_partial_rotary_is_not_full_rotary():
    """Negative test: rotating the full head_dim is a different function, so the partial width is
    load-bearing rather than an optimisation."""
    torch.manual_seed(5)
    head_dim, rotary_dim, tokens = 32, 8, 4
    query = torch.randn(1, 1, tokens, head_dim)
    cos_full = torch.randn(1, 1, tokens, head_dim)
    sin_full = torch.randn(1, 1, tokens, head_dim)
    partial = apply_partial_rotary(query, cos_full[..., :rotary_dim], sin_full[..., :rotary_dim])
    full = apply_partial_rotary(query, cos_full, sin_full)
    assert not torch.allclose(partial, full, atol=1e-3)


@requires_hf_modules
def test_gated_delta_projections_match_hf():
    """``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``, in the reference's
    dtypes. ``g`` must be computed in fp32: in bf16 a large ``A_log`` overflows to -inf."""
    module, _ = _hf_gated_delta_net(seed=6)
    torch.manual_seed(7)
    a = torch.randn(1, 9, _TEST_DIMS["v_heads"])
    b = torch.randn(1, 9, _TEST_DIMS["v_heads"])
    g, beta = gated_delta_projections(a, b, module.A_log, module.dt_bias)
    expected_beta = b.sigmoid()
    expected_g = (-module.A_log.float().exp()
                  * torch.nn.functional.softplus(a.float() + module.dt_bias.float()))
    assert torch.allclose(beta, expected_beta, atol=1e-7)
    assert torch.allclose(g, expected_g, atol=1e-6)
    assert (g <= 0).all(), "the log decay must be non-positive or the recurrence grows"


@requires_hf_modules
@pytest.mark.parametrize("seq_len", [1, 5, 64, 100])
def test_gated_delta_net_prefill_matches_hf(seq_len):
    """The WHOLE mixer against the HF module: projections, depthwise conv + SiLU, key/value head
    expansion, decay and beta, the chunked scan, the gated norm, and out_proj."""
    module, config = _hf_gated_delta_net(seed=8)
    torch.manual_seed(9)
    hidden = torch.randn(1, seq_len, _TEST_HIDDEN)
    with torch.no_grad():
        reference = module(hidden_states=hidden, cache_params=None, attention_mask=None)
        mine, _, _ = gated_delta_net_prefill(
            hidden, _weights_from_hf(module), _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE)
    assert torch.allclose(mine, reference, atol=1e-4), (mine - reference).abs().max()


@requires_hf_modules
def test_prefill_then_decode_matches_hf_single_shot():
    """Splitting a sequence into a prefill and one decode step, carrying the conv and recurrent
    state, must reproduce the reference's single-shot output for BOTH parts."""
    module, config = _hf_gated_delta_net(seed=10)
    torch.manual_seed(11)
    prefill_len = 32
    hidden = torch.randn(1, prefill_len + 1, _TEST_HIDDEN)
    weights = _weights_from_hf(module)
    with torch.no_grad():
        reference = module(hidden_states=hidden, cache_params=None, attention_mask=None)
        prefill_out, state, conv_state = gated_delta_net_prefill(
            hidden[:, :prefill_len], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE)
        decode_out, _, _ = gated_delta_net_decode(
            hidden[:, prefill_len:], weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=conv_state, recurrent_state=state)
    assert torch.allclose(prefill_out, reference[:, :prefill_len], atol=1e-4)
    assert torch.allclose(decode_out, reference[:, prefill_len:], atol=1e-4)


@requires_hf_modules
def test_multi_step_decode_carry_matches_hf():
    """Eight consecutive decode steps must track the reference. This is what catches a conv-history
    or recurrent-state carry that is right for one step and drifts afterwards."""
    module, config = _hf_gated_delta_net(seed=12)
    torch.manual_seed(13)
    prefill_len, decode_steps = 32, 8
    hidden = torch.randn(1, prefill_len + decode_steps, _TEST_HIDDEN)
    weights = _weights_from_hf(module)
    with torch.no_grad():
        reference = module(hidden_states=hidden, cache_params=None, attention_mask=None)
        _, state, conv_state = gated_delta_net_prefill(
            hidden[:, :prefill_len], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE)
        steps = []
        for step in range(decode_steps):
            token = hidden[:, prefill_len + step:prefill_len + step + 1]
            out, state, conv_state = gated_delta_net_decode(
                token, weights, _TEST_DIMS, config.rms_norm_eps,
                conv_state=conv_state, recurrent_state=state)
            steps.append(out)
    decoded = torch.cat(steps, dim=1)
    assert torch.allclose(decoded, reference[:, prefill_len:], atol=1e-4), \
        (decoded - reference[:, prefill_len:]).abs().max()


@requires_hf_modules
def test_mixer_bucket_padding_does_not_change_carried_state():
    """A padded prefill must hand decode the same conv and recurrent state as the unpadded one.
    Whole-mixer version of the scan-level pad-invariance test: it also covers the conv history, which
    is gathered from the real tail rather than the padded one."""
    module, config = _hf_gated_delta_net(seed=14)
    torch.manual_seed(15)
    real_len, pad_len = 20, 12
    real = torch.randn(1, real_len, _TEST_HIDDEN)
    padded = torch.cat([real, torch.randn(1, pad_len, _TEST_HIDDEN)], dim=1)
    valid_mask = torch.cat([torch.ones(real_len), torch.zeros(pad_len)])
    weights = _weights_from_hf(module)
    with torch.no_grad():
        _, real_state, real_conv = gated_delta_net_prefill(
            real, weights, _TEST_DIMS, config.rms_norm_eps, chunk_size=DEFAULT_CHUNK_SIZE)
        _, padded_state, padded_conv = gated_delta_net_prefill(
            padded, weights, _TEST_DIMS, config.rms_norm_eps, chunk_size=DEFAULT_CHUNK_SIZE,
            valid_mask=valid_mask)
    assert torch.allclose(padded_state, real_state, atol=1e-6), \
        (padded_state - real_state).abs().max()
    assert torch.allclose(padded_conv, real_conv, atol=1e-6)


@requires_hf_modules
def test_mixer_segmented_prefill_matches_single_shot():
    """Continuation prefill: a second segment that carries both states must equal one pass, and the
    first-segment mask must make a carried-state call behave like a fresh one."""
    module, config = _hf_gated_delta_net(seed=16)
    torch.manual_seed(17)
    total, split = 64, 32
    hidden = torch.randn(1, total, _TEST_HIDDEN)
    weights = _weights_from_hf(module)
    with torch.no_grad():
        single, single_state, single_conv = gated_delta_net_prefill(
            hidden, weights, _TEST_DIMS, config.rms_norm_eps, chunk_size=DEFAULT_CHUNK_SIZE)
        # First segment, entered through the continuation path with the mask off.
        garbage_conv = torch.randn(1, 2 * (_TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"])
                                   + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"],
                                   _TEST_DIMS["kernel"] - 1)
        first, first_state, first_conv = gated_delta_net_prefill(
            hidden[:, :split], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE, conv_state=garbage_conv,
            is_continuation=torch.zeros(()), initial_state=None)
        second, second_state, second_conv = gated_delta_net_prefill(
            hidden[:, split:], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE, conv_state=first_conv,
            is_continuation=torch.ones(()), initial_state=first_state)
    assert torch.allclose(torch.cat([first, second], dim=1), single, atol=1e-4)
    assert torch.allclose(second_state, single_state, atol=1e-5)
    assert torch.allclose(second_conv, single_conv, atol=1e-6)


# --------------------------------------------------------------------------------------
# Weight layout: the per-rank shards must reassemble into the reference's own computation
# --------------------------------------------------------------------------------------
@requires_hf_modules
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_expert_shards_reproduce_hf_expert_mlp(world_size):
    """Run one expert through the sharded weights and compare with the HF experts module.

    This is what catches the two silent layout bugs in a fused MoE checkpoint: taking ``up`` where
    ``gate`` lives (the fused tensor stores gate rows FIRST, so it must be split rather than
    reshaped), and forgetting that ``down_proj`` is stored ``[out, in]`` per expert.
    """
    torch.manual_seed(20)
    config = _hf_text_config()
    experts = HF_MODELING.Qwen3_5MoeExperts(config).eval()
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)

    tokens = 5
    hidden = torch.randn(tokens, config.hidden_size)
    expert_index = 3
    # HF's reference expert computation, taken straight from its forward.
    with torch.no_grad():
        gate, up = torch.nn.functional.linear(
            hidden, experts.gate_up_proj[expert_index]).chunk(2, dim=-1)
        reference = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, experts.down_proj[expert_index])

    # The sharded weights, reassembled the way the kernels consume them.
    total = torch.zeros_like(reference)
    for rank in range(world_size):
        gate_up = shard_expert_gate_up(experts.gate_up_proj, rank, world_size)
        down = shard_expert_down(experts.down_proj, rank, world_size)
        assert gate_up.shape[1] == config.hidden_size and gate_up.shape[2] == 2
        gate_shard = hidden @ gate_up[expert_index, :, 0, :]
        up_shard = hidden @ gate_up[expert_index, :, 1, :]
        total = total + (torch.nn.functional.silu(gate_shard) * up_shard) @ down[expert_index]
    assert torch.allclose(total, reference, atol=1e-5), (total - reference).abs().max()


@requires_hf_modules
def test_expert_gate_up_split_is_not_a_reshape():
    """Negative test: reshaping the fused rows into ``[2, intermediate]`` interleaves gate and up, so
    the explicit split in ``shard_expert_gate_up`` is load-bearing."""
    torch.manual_seed(21)
    experts, intermediate, hidden = 2, 8, 4
    fused = torch.randn(experts, 2 * intermediate, hidden)
    # With one rank the two forms coincide; the interleaving only bites once the intermediate
    # dimension is sharded, which is the deployed case.
    assert torch.allclose(shard_expert_gate_up(fused, 0, 1),
                          fused.reshape(experts, 2, intermediate, hidden).permute(0, 3, 1, 2),
                          atol=0)
    world_size = 2
    width = intermediate // world_size
    correct = shard_expert_gate_up(fused, 0, world_size)
    # The naive form: take the rank's slice of the FUSED rows, which for rank 0 is two gate halves.
    wrong = fused[:, :2 * width, :].reshape(experts, 2, width, hidden).permute(0, 3, 1, 2)
    assert not torch.allclose(correct, wrong, atol=1e-4)
    # Rank 0's gate half is the first `width` rows; its up half starts at `intermediate`.
    assert torch.allclose(correct[:, :, 0, :].transpose(1, 2), fused[:, :width, :], atol=0)
    assert torch.allclose(correct[:, :, 1, :].transpose(1, 2),
                          fused[:, intermediate:intermediate + width, :], atol=0)


@requires_hf_modules
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_attention_qkvg_shards_reproduce_hf_projection(world_size):
    """The fused per-rank attention projection must reproduce HF's query, gate, key and value.

    ``q_proj`` is head-major with the query and the output gate adjacent per head, so a naive
    "first half is query, second half is gate" split silently mixes heads. Concatenating the ranks'
    shards must give back exactly what the reference computes.
    """
    torch.manual_seed(22)
    config = _hf_text_config()
    attention = HF_MODELING.Qwen3_5MoeAttention(config, 3).eval()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.normal_(0, 0.1)
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    tokens = 4
    hidden = torch.randn(tokens, config.hidden_size)

    # HF's own split of q_proj into query and gate (from Qwen3_5MoeAttention.forward).
    with torch.no_grad():
        ref_query, ref_gate = torch.chunk(
            attention.q_proj(hidden).view(tokens, -1, head_dim * 2), 2, dim=-1)
        ref_gate = ref_gate.reshape(tokens, -1)
        ref_query = ref_query.reshape(tokens, num_heads * head_dim)
        ref_key = attention.k_proj(hidden)
        ref_value = attention.v_proj(hidden)

    heads_per_rank = num_heads // world_size
    queries, gates = [], []
    for rank in range(world_size):
        fused = fuse_attention_qkvg(
            attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight,
            rank, world_size, num_heads, num_kv_heads, head_dim)
        projected = hidden @ fused
        q_width = heads_per_rank * head_dim
        kv_width = (fused.shape[1] - 2 * q_width) // 2
        query, gate, key, value = torch.tensor_split(
            projected, [q_width, 2 * q_width, 2 * q_width + kv_width], dim=-1)
        queries.append(query)
        gates.append(gate)
        # Each rank's KV slice must match the corresponding slice of the reference projection.
        if world_size >= num_kv_heads:
            kv_rank = rank // (world_size // num_kv_heads)
        else:
            kv_rank = rank
        expected_key = ref_key[:, kv_rank * kv_width:(kv_rank + 1) * kv_width]
        expected_value = ref_value[:, kv_rank * kv_width:(kv_rank + 1) * kv_width]
        assert torch.allclose(key, expected_key, atol=1e-5), (rank, (key - expected_key).abs().max())
        assert torch.allclose(value, expected_value, atol=1e-5)
    assert torch.allclose(torch.cat(queries, dim=-1), ref_query, atol=1e-5)
    assert torch.allclose(torch.cat(gates, dim=-1), ref_gate, atol=1e-5)


@requires_hf_modules
def test_attention_query_gate_split_is_not_a_half_split():
    """Negative test: splitting q_proj's rows in half puts several heads' gates in the query. The
    per-head regroup in ``fuse_attention_qkvg`` is what makes the fused projection correct."""
    torch.manual_seed(23)
    num_heads, head_dim, hidden = 4, 8, 6
    q_proj = torch.randn(num_heads * head_dim * 2, hidden)
    fused = fuse_attention_qkvg(q_proj, torch.zeros(head_dim, hidden),
                                torch.zeros(head_dim, hidden), 0, 1, num_heads, 1, head_dim)
    query = fused[:, :num_heads * head_dim].T
    naive_half = q_proj[:num_heads * head_dim, :]
    assert not torch.allclose(query, naive_half, atol=1e-4)
    # The right answer is the per-head first halves, in head order.
    expected = torch.cat([q_proj[h * 2 * head_dim:h * 2 * head_dim + head_dim, :]
                          for h in range(num_heads)], dim=0)
    assert torch.allclose(query, expected, atol=0)


@requires_hf_modules
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gdn_shards_reproduce_hf_mixer(world_size):
    """Shard the mixer across ranks, run each shard, and sum: the result must equal the unsharded
    mixer. This validates the head sharding as a whole -- that a rank's key heads expand to exactly
    its own value heads, and that ``out_proj`` is row-parallel so the partials are additive.

    The summation here is the invariant the served mixer must satisfy with a collective: each rank
    produces a PARTIAL output, so a mixer that returned it unreduced would contribute 1/TP of the true
    value to the residual at every one of the 30 Gated DeltaNet layers. That failure is invisible to a
    single-rank test and to a TP=1 run, so this test states the requirement even though it cannot
    check that the collective is actually issued."""
    module, config = _hf_gated_delta_net(seed=24)
    torch.manual_seed(25)
    tokens = 40
    hidden = torch.randn(1, tokens, _TEST_HIDDEN)
    with torch.no_grad():
        reference = module(hidden_states=hidden, cache_params=None, attention_mask=None)

    num_k, num_v = _TEST_DIMS["k_heads"], _TEST_DIMS["v_heads"]
    head_k, head_v = _TEST_DIMS["head_k_dim"], _TEST_DIMS["head_v_dim"]
    if num_k % world_size or num_v % world_size:
        pytest.skip(f"world_size {world_size} does not divide the test head counts")

    total = torch.zeros_like(reference)
    for rank in range(world_size):
        weights = {
            "in_proj_qkv": gdn_qkv_rows(module.in_proj_qkv.weight, rank, world_size,
                                        num_k, num_v, head_k, head_v).T.contiguous(),
            "conv1d": gdn_qkv_rows(module.conv1d.weight, rank, world_size,
                                   num_k, num_v, head_k, head_v).contiguous(),
            "in_proj_z": shard_rows_transposed(module.in_proj_z.weight, rank, world_size),
            "in_proj_b": shard_rows_transposed(module.in_proj_b.weight, rank, world_size),
            "in_proj_a": shard_rows_transposed(module.in_proj_a.weight, rank, world_size),
            "A_log": shard_heads(module.A_log, rank, world_size),
            "dt_bias": shard_heads(module.dt_bias, rank, world_size),
            "norm": module.norm.weight,
            "out_proj": shard_columns_transposed(module.out_proj.weight, rank, world_size),
        }
        dims = {"k_heads": num_k // world_size, "v_heads": num_v // world_size,
                "head_k_dim": head_k, "head_v_dim": head_v, "kernel": _TEST_DIMS["kernel"]}
        with torch.no_grad():
            shard_out, _, _ = gated_delta_net_prefill(
                hidden, weights, dims, config.rms_norm_eps, chunk_size=DEFAULT_CHUNK_SIZE)
        total = total + shard_out
    assert torch.allclose(total, reference, atol=1e-4), (total - reference).abs().max()


@requires_hf_modules
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_shared_expert_shards_reproduce_hf_mlp(world_size):
    """The shared expert's shards must sum to the reference MLP (a row-parallel down projection)."""
    torch.manual_seed(26)
    config = _hf_text_config()
    mlp = HF_MODELING.Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.normal_(0, 0.1)
    hidden = torch.randn(6, config.hidden_size)
    with torch.no_grad():
        reference = mlp(hidden)
    total = torch.zeros_like(reference)
    for rank in range(world_size):
        gate_w = shard_rows_transposed(mlp.gate_proj.weight, rank, world_size)
        up_w = shard_rows_transposed(mlp.up_proj.weight, rank, world_size)
        down_w = shard_columns_transposed(mlp.down_proj.weight, rank, world_size)
        total = total + (torch.nn.functional.silu(hidden @ gate_w) * (hidden @ up_w)) @ down_w
    assert torch.allclose(total, reference, atol=1e-5), (total - reference).abs().max()


@requires_hf_modules
def test_router_math_matches_hf_top_k_router():
    """Softmax over ALL experts, then top-k, then L1 renormalise -- the order the kernels are
    configured for. Pinning it here is what makes the kernel configuration checkable: if the reference
    ever renormalised differently, this test fails rather than the device silently mis-routing."""
    torch.manual_seed(27)
    config = _hf_text_config()
    router = HF_MODELING.Qwen3_5MoeTopKRouter(config)
    with torch.no_grad():
        router.weight.normal_(0, 0.5)
    hidden = torch.randn(9, config.hidden_size)
    with torch.no_grad():
        _, scores, indices = router(hidden)

    logits = hidden @ router.weight.T
    probabilities = torch.softmax(logits.float(), dim=-1)
    values, chosen = torch.topk(probabilities, config.num_experts_per_tok, dim=-1)
    renormalised = values / values.sum(dim=-1, keepdim=True)
    assert torch.equal(chosen, indices)
    assert torch.allclose(renormalised.to(scores.dtype), scores, atol=1e-6)
    # And it equals a softmax over the selected logits alone, which is the form the decode kernel's
    # "top-k then activate" order computes -- so both kernel orders are the same function here.
    gathered = torch.gather(logits.float(), -1, chosen)
    assert torch.allclose(torch.softmax(gathered, dim=-1).to(scores.dtype), scores, atol=1e-6)


@requires_hf_modules
def test_decode_first_token_mask_starts_from_zero_state():
    """A one-token prompt cannot be told apart from a decode step by token count, so the decode path
    must be able to start from a zero state. With ``is_continuation = 0`` its output must equal a
    fresh one-token prefill; with ``is_continuation = 1`` it must NOT, or the mask does nothing."""
    module, config = _hf_gated_delta_net(seed=28)
    torch.manual_seed(29)
    weights = _weights_from_hf(module)
    token = torch.randn(1, 1, _TEST_HIDDEN)
    conv_width = (2 * _TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"]
                  + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"])
    stale_conv = torch.randn(1, conv_width, _TEST_DIMS["kernel"] - 1)
    stale_state = torch.randn(1, _TEST_DIMS["v_heads"], _TEST_DIMS["head_k_dim"],
                              _TEST_DIMS["head_v_dim"]) * 0.5

    with torch.no_grad():
        fresh = module(hidden_states=token, cache_params=None, attention_mask=None)
        masked, masked_state, _ = gated_delta_net_decode(
            token, weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=stale_conv, recurrent_state=stale_state,
            is_continuation=torch.zeros(()))
        carried, _, _ = gated_delta_net_decode(
            token, weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=stale_conv, recurrent_state=stale_state,
            is_continuation=torch.ones(()))
    assert torch.allclose(masked, fresh, atol=1e-4), (masked - fresh).abs().max()
    assert not torch.allclose(carried, fresh, atol=1e-3), \
        "the stale state must actually change the result, or the mask is untested"
    # And the state written back starts from zero, not from the stale value.
    with torch.no_grad():
        _, fresh_state, _ = gated_delta_net_decode(
            token, weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=torch.zeros_like(stale_conv), recurrent_state=torch.zeros_like(stale_state),
            is_continuation=None)
    assert torch.allclose(masked_state, fresh_state, atol=1e-6)


# --------------------------------------------------------------------------------------
# The padded paged-cache scatter
# --------------------------------------------------------------------------------------
def _reference_cache_write(cache, slot_mapping, rows, block_size):
    """The scatter as it should end up: real rows land at their slots, pad rows change nothing."""
    expected = cache.clone()
    for token, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            expected[slot // block_size, slot % block_size] = rows[token]
    return expected


def _apply_redirected_write(cache, slot_mapping, rows, block_size):
    safe_slot, safe_rows = redirect_padded_slots(slot_mapping, rows)
    index = (safe_slot // block_size, safe_slot % block_size)
    written = cache.clone()
    written.index_put_(index, safe_rows)
    return written


@pytest.mark.parametrize("pad_len", [0, 1, 7, 31])
def test_padded_cache_write_leaves_padding_out(pad_len):
    """Real rows land at their slots and the pad rows change nothing, whatever the padding length."""
    torch.manual_seed(50)
    block_size, num_blocks, width = 8, 6, 4
    real_len = 9
    cache = torch.randn(num_blocks, block_size, width)
    slot_mapping = torch.cat([torch.arange(real_len),
                              torch.full((pad_len,), -1, dtype=torch.long)])
    rows = torch.randn(real_len + pad_len, width)
    written = _apply_redirected_write(cache, slot_mapping, rows, block_size)
    expected = _reference_cache_write(cache, slot_mapping, rows, block_size)
    assert torch.allclose(written, expected, atol=0), (written - expected).abs().max()


def test_padded_cache_write_survives_a_sequence_in_the_last_block():
    """The case a fixed sink slot cannot handle: the active sequence ends in the FINAL slot of the
    cache. A sink pointed there would collide with a real row and ``index_put_``'s duplicate order
    would decide whether the real K/V survives."""
    torch.manual_seed(51)
    block_size, num_blocks, width = 8, 4, 3
    last_slot = num_blocks * block_size - 1
    real_slots = [last_slot - 2, last_slot - 1, last_slot]
    slot_mapping = torch.tensor(real_slots + [-1] * 13, dtype=torch.long)
    rows = torch.randn(len(slot_mapping), width)
    cache = torch.randn(num_blocks, block_size, width)
    written = _apply_redirected_write(cache, slot_mapping, rows, block_size)
    expected = _reference_cache_write(cache, slot_mapping, rows, block_size)
    assert torch.allclose(written, expected, atol=0)
    # Specifically: the real value at the final slot is intact, not a pad row's.
    assert torch.allclose(written[last_slot // block_size, last_slot % block_size], rows[2], atol=0)


def test_padded_cache_write_rejects_the_naive_sentinel_arithmetic():
    """Negative test for the whole reason this helper exists: scattering the sentinel directly writes
    into the LAST block of the cache, because -1 // block_size is -1."""
    torch.manual_seed(52)
    block_size, num_blocks, width = 8, 4, 3
    slot_mapping = torch.tensor([0, 1, -1, -1], dtype=torch.long)
    rows = torch.randn(4, width)
    cache = torch.zeros(num_blocks, block_size, width)
    naive = cache.clone()
    naive.index_put_((slot_mapping // block_size, slot_mapping % block_size), rows)
    assert naive[num_blocks - 1].abs().sum() > 0, "the trap needs the sentinel to reach a real block"
    safe = _apply_redirected_write(cache, slot_mapping, rows, block_size)
    assert safe[num_blocks - 1].abs().sum() == 0


def test_padded_cache_write_all_padding_is_confined_to_slot_zero():
    """An all-padding batch is only reachable in a warmup trace. It must not touch a real block; the
    clamp confines it to slot 0."""
    block_size, num_blocks, width = 8, 4, 3
    slot_mapping = torch.full((5,), -1, dtype=torch.long)
    rows = torch.randn(5, width, generator=torch.Generator().manual_seed(53))
    cache = torch.zeros(num_blocks, block_size, width)
    written = _apply_redirected_write(cache, slot_mapping, rows, block_size)
    assert written[0, 1:].abs().sum() == 0
    assert written[1:].abs().sum() == 0


def test_padded_cache_write_handles_non_monotonic_slots():
    """Block tables need not be contiguous or ordered: the donor is chosen by ``argmax`` over the
    slots, which is real regardless of their order."""
    torch.manual_seed(54)
    block_size, num_blocks, width = 8, 5, 3
    slot_mapping = torch.tensor([33, 8, 9, 24, -1, -1], dtype=torch.long)
    rows = torch.randn(len(slot_mapping), width)
    cache = torch.randn(num_blocks, block_size, width)
    written = _apply_redirected_write(cache, slot_mapping, rows, block_size)
    expected = _reference_cache_write(cache, slot_mapping, rows, block_size)
    assert torch.allclose(written, expected, atol=0)


# --------------------------------------------------------------------------------------
# Config and checkpoint mapping
# --------------------------------------------------------------------------------------
# The published Qwen/Qwen3.6-35B-A3B config.json, trimmed to the fields this port reads. Kept inline so
# the test pins the parsing of the REAL shape (a nested text_config and nested rope_parameters) rather
# than of a convenient dataclass call.
PUBLISHED_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "attn_output_gate": True, "attention_bias": False, "dtype": "bfloat16",
        "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "max_position_embeddings": 262144, "moe_intermediate_size": 512,
        "num_attention_heads": 16, "num_experts": 256, "num_experts_per_tok": 8,
        "num_hidden_layers": 40, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
        "shared_expert_intermediate_size": 512, "vocab_size": 248320,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                            "rope_type": "default"},
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    },
}


def test_published_config_parses_to_the_expected_shape():
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    assert config.num_hidden_layers == 40
    assert config.layer_type_counts() == {"linear_attention": 30, "full_attention": 10}
    # The rotary width is a derived quantity and the one most easily got wrong: 0.25 of head_dim 256.
    assert config.rotary_dim == 64
    assert config.torch_dtype == torch.bfloat16
    assert config.rope_theta == 10000000
    # The head counts and dims the Gated DeltaNet mixer shards on, and the multiple that ties the
    # query/key heads to the value heads.
    assert (config.linear_num_key_heads, config.linear_key_head_dim) == (16, 128)
    assert (config.linear_num_value_heads, config.linear_value_head_dim) == (32, 128)
    assert config.linear_num_value_heads % config.linear_num_key_heads == 0
    assert config.tie_word_embeddings is False


def test_layer_types_are_reconstructed_from_the_interval():
    """A checkpoint that gives full_attention_interval but no explicit list must produce the published
    pattern, or the layer dispatch silently disagrees with the weights."""
    trimmed = json.loads(json.dumps(PUBLISHED_CONFIG))
    del trimmed["text_config"]["layer_types"]
    config = Qwen3_5MoeConfig.from_configs(trimmed, None)
    assert config.layer_types == PUBLISHED_CONFIG["text_config"]["layer_types"]


@pytest.mark.parametrize("field,value,message", [
    ("attn_output_gate", False, "attn_output_gate"),
    ("attention_bias", True, "attention_bias"),
    ("hidden_act", "gelu", "hidden_act"),
    ("linear_conv_kernel_dim", 1, "linear_conv_kernel_dim"),
    ("linear_num_value_heads", 33, "multiple of"),
    ("dtype", "float32", "bf16 only"),
])
def test_config_refuses_shapes_it_does_not_implement(field, value, message):
    """Each of these would load without complaint and then be wrong (or exhaust HBM), so the config has
    to refuse it rather than carry on."""
    broken = json.loads(json.dumps(PUBLISHED_CONFIG))
    broken["text_config"][field] = value
    with pytest.raises(ValueError, match=message):
        Qwen3_5MoeConfig.from_configs(broken, None)


def test_config_refuses_a_scaled_rope_type():
    broken = json.loads(json.dumps(PUBLISHED_CONFIG))
    broken["text_config"]["rope_parameters"]["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="rope_type"):
        Qwen3_5MoeConfig.from_configs(broken, None)


def test_config_refuses_a_gdn_only_stack():
    """The prefill/decode phase and the real/pad mask both come from an attention layer's metadata, so
    a stack with no full_attention layer has no phase signal."""
    broken = json.loads(json.dumps(PUBLISHED_CONFIG))
    broken["text_config"]["layer_types"] = ["linear_attention"] * 40
    with pytest.raises(ValueError, match="full_attention"):
        Qwen3_5MoeConfig.from_configs(broken, None)


def _mapping_sources(mappings):
    sources = set()
    for value in mappings.values():
        sources.update([value] if isinstance(value, str) else value)
    return sources


def test_checkpoint_mapping_covers_every_declared_parameter():
    """One destination per parameter, no duplicates, and every layer represented. A destination that
    never appears is a parameter left on `meta`; a duplicated one silently overwrites."""
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    mappings = checkpoint_mappings(config.layer_types, "model.language_model",
                                   has_lm_head=True, tie_word_embeddings=False)
    for index, kind in enumerate(config.layer_types):
        prefix = f"model.layers.{index}"
        mixer = "linear_attn" if kind == "linear_attention" else "self_attn"
        assert any(key.startswith(f"{prefix}.{mixer}.") for key in mappings), (index, kind)
        # The other mixer must NOT be mapped for this layer.
        other = "self_attn" if mixer == "linear_attn" else "linear_attn"
        assert not any(key.startswith(f"{prefix}.{other}.") for key in mappings), (index, kind)
        assert f"{prefix}.mlp.router_weight" in mappings
        assert f"{prefix}.input_layernorm.weight" in mappings
    assert "lm_head.weight" in mappings
    assert "model.embed_tokens.weight" in mappings
    assert "model.norm.weight" in mappings


def test_checkpoint_mapping_reads_only_the_decoder():
    """The vision tower and the MTP head are out of scope, so nothing under those prefixes may be
    mapped — a stray mapping would load a tensor of the wrong shape."""
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    sources = _mapping_sources(
        checkpoint_mappings(config.layer_types, "model.language_model",
                            has_lm_head=True, tie_word_embeddings=False))
    assert not [key for key in sources if ".visual." in key or key.startswith("mtp.")]
    assert all(key.startswith("model.language_model.") or key == "lm_head.weight"
               for key in sources), sorted(sources)[:5]


def test_checkpoint_mapping_lm_head_branches():
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    tied = checkpoint_mappings(config.layer_types, "model", has_lm_head=False,
                               tie_word_embeddings=True)
    assert tied["lm_head.weight"] == "model.embed_tokens.weight"
    # An UNTIED checkpoint with no head must be refused, not served with the embedding matrix as an
    # invented tied head.
    with pytest.raises(ValueError, match="lm_head"):
        checkpoint_mappings(config.layer_types, "model", has_lm_head=False,
                            tie_word_embeddings=False)


@pytest.mark.skipif(not os.environ.get("QWEN3_5_MOE_CHECKPOINT_INDEX"),
                    reason="set QWEN3_5_MOE_CHECKPOINT_INDEX to a model.safetensors.index.json")
def test_checkpoint_mapping_matches_a_real_checkpoint():
    """Opt-in: diff the mapping against a real checkpoint's key set. Nothing this port declares may be
    absent from the checkpoint, and nothing in the checkpoint's decoder may go unconsumed."""
    with open(os.environ["QWEN3_5_MOE_CHECKPOINT_INDEX"]) as handle:
        checkpoint_keys = set(json.load(handle)["weight_map"])
    source = ("model.language_model"
              if any(k.startswith("model.language_model.") for k in checkpoint_keys) else "model")
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    mappings = checkpoint_mappings(
        config.layer_types, source,
        has_lm_head="lm_head.weight" in checkpoint_keys,
        tie_word_embeddings=config.tie_word_embeddings)
    sources = _mapping_sources(mappings)
    assert not sorted(sources - checkpoint_keys), sorted(sources - checkpoint_keys)[:5]
    decoder = {k for k in checkpoint_keys if k.startswith(f"{source}.")} | {"lm_head.weight"}
    unconsumed = sorted(k for k in decoder - sources if ".visual." not in k)
    assert not unconsumed, unconsumed[:8]
