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
import math
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
gated_delta_net_verify = _gdn.gated_delta_net_verify

rmsnorm = _ops.rmsnorm
gated_rmsnorm = _ops.gated_rmsnorm
rotary_tables = _ops.rotary_tables
mrope_tables = _ops.mrope_tables
interleave_mrope = _ops.interleave_mrope
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
vision_checkpoint_mappings = _layout.vision_checkpoint_mappings
Qwen3_5MoeConfig = _config.Qwen3_5MoeConfig


def _hf_reference():
    """The installed HF GDN reference, or None. See the module docstring for why qwen3_next counts."""
    for module_name in ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                        "transformers.models.qwen3_next.modeling_qwen3_next"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        # lint-port: ok probing transformers, whose reference helpers move between versions; absence is
        # the answer this loop wants, and the caller skips the test rather than trusting a default
        chunk = getattr(module, "torch_chunk_gated_delta_rule", None)  # lint-port: ok see above
        recurrent = getattr(module, "torch_recurrent_gated_delta_rule", None)  # lint-port: ok see above
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

# The vision map delegates to the plugin's own encoder class, so those two checks -- alone in this file
# -- need the vllm package importable. They used to ERROR rather than skip on a box without it, and
# because they also need a checkpoint index they were skipped for that reason first: the error only
# appeared once someone pointed the tests at a real index. Skipping says which coverage is absent.
requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="no vllm package: the vision map comes from the plugin's encoder, which imports vllm",
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


@requires_vllm
def test_vision_mappings_cover_the_visual_tower_exactly():
    """Both directions against the real checkpoint index, for the vision tower.

    A key this implementation declares but the checkpoint lacks loads nothing; a key the checkpoint has
    but this implementation does not declare is a tensor left at its initial value. Neither shows up as
    an error — the first shows up as garbage, the second as a subtly wrong image embedding — so both
    directions are asserted.

    Opt-in like the text-side diff: set QWEN3_5_MOE_CHECKPOINT_INDEX to a real
    model.safetensors.index.json.
    """
    index_path = os.environ.get("QWEN3_5_MOE_CHECKPOINT_INDEX")
    if not index_path:
        pytest.skip("set QWEN3_5_MOE_CHECKPOINT_INDEX to a model.safetensors.index.json")
    with open(index_path) as handle:
        keys = set(json.load(handle)["weight_map"])
    visual = {key for key in keys if key.startswith("model.visual")}
    if not visual:
        pytest.skip("this checkpoint has no vision tower")

    declared = set(vision_checkpoint_mappings(27).values())
    assert not declared - visual, sorted(declared - visual)[:8]
    assert not visual - declared, sorted(visual - declared)[:8]


@requires_vllm
def test_only_the_mtp_head_is_left_unmapped_once_vision_is_added():
    """The whole checkpoint, accounted for.

    With the decoder and the vision tower both mapped, the only source keys left should be the
    multi-token prediction head. This is the test that turns "out of scope" into a list of exactly one
    thing, so that adding MTP later is a bounded job rather than a search.
    """
    index_path = os.environ.get("QWEN3_5_MOE_CHECKPOINT_INDEX")
    if not index_path:
        pytest.skip("set QWEN3_5_MOE_CHECKPOINT_INDEX to a model.safetensors.index.json")
    with open(index_path) as handle:
        keys = set(json.load(handle)["weight_map"])
    if not any(key.startswith("model.visual") for key in keys):
        pytest.skip("this checkpoint has no vision tower")

    prefix = "model.language_model" if any(k.startswith("model.language_model") for k in keys) else "model"
    layer_types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(40)]
    text = checkpoint_mappings(layer_types, prefix, has_lm_head=True, tie_word_embeddings=False)
    consumed = set()
    for value in list(text.values()) + list(vision_checkpoint_mappings(27).values()):
        consumed.update(value if isinstance(value, list) else [value])

    leftover = {key.split(".")[0] for key in keys - consumed}
    assert leftover == {"mtp"}, sorted(keys - consumed)[:10]


@requires_hf_modules
def test_mrope_tables_reduce_to_the_single_axis_table_when_the_axes_agree():
    """The regression condition for adding vision: text must not move by one bit.

    The single-axis table is what the text-only path has been verified with on device. Introducing the
    three-axis path is only safe if, for positions where the three axes carry the same value, it produces
    the SAME tensor rather than a close one — otherwise every later text disagreement has two possible
    causes.
    """
    config = _hf_text_config()
    rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    theta = config.rope_parameters["rope_theta"]
    section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    tokens = torch.arange(37, dtype=torch.int64)
    single_cos, single_sin = rotary_tables(rotary_dim, 64, theta)
    picked_cos = single_cos.index_select(0, tokens)
    picked_sin = single_sin.index_select(0, tokens)

    three_axes = tokens.unsqueeze(0).expand(3, tokens.shape[0])
    mrope_cos, mrope_sin = mrope_tables(three_axes, rotary_dim, theta, section)

    assert torch.equal(mrope_cos, picked_cos), (mrope_cos - picked_cos).abs().max()
    assert torch.equal(mrope_sin, picked_sin), (mrope_sin - picked_sin).abs().max()


@requires_hf_modules
def test_mrope_tables_match_hf_when_the_axes_differ():
    """The three-axis case, against the reference, with axes that actually disagree.

    Equal axes cannot distinguish a correct interleave from one that ignores height and width, so the
    positions here are deliberately different per axis — the shape an image span produces.
    """
    config = _hf_text_config()
    reference = HF_MODELING.Qwen3_5MoeTextRotaryEmbedding(config)
    rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    theta = config.rope_parameters["rope_theta"]
    section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    length = 19
    positions = torch.stack([
        torch.arange(length),                    # time
        torch.arange(length) // 4,               # height, as a 4-wide row would give
        torch.arange(length) % 4,                # width
    ]).to(torch.int64)

    dummy = torch.zeros(1, length, config.hidden_size)
    ref_cos, ref_sin = reference(dummy, positions.unsqueeze(1))
    cos, sin = mrope_tables(positions, rotary_dim, theta, section)

    assert ref_cos.shape[-1] == rotary_dim, (ref_cos.shape, rotary_dim)
    assert torch.allclose(cos, ref_cos.reshape(-1, rotary_dim), atol=1e-6), \
        (cos - ref_cos.reshape(-1, rotary_dim)).abs().max()
    assert torch.allclose(sin, ref_sin.reshape(-1, rotary_dim), atol=1e-6), \
        (sin - ref_sin.reshape(-1, rotary_dim)).abs().max()


def test_interleave_mrope_puts_each_axis_where_the_section_says():
    """The mask itself, independent of any frequency arithmetic.

    Feeding one constant per axis makes the selection visible: slot i must hold the height constant when
    i % 3 == 1 and i is inside height's section, the width constant under the same rule with offset 2,
    and the time constant everywhere else.
    """
    section = [11, 11, 10]
    slots = 128
    freqs = torch.stack([
        torch.full((slots,), 100.0),
        torch.full((slots,), 200.0),
        torch.full((slots,), 300.0),
    ])
    out = interleave_mrope(freqs, section)

    for index in range(slots):
        if index % 3 == 1 and index < section[1] * 3:
            expected = 200.0
        elif index % 3 == 2 and index < section[2] * 3:
            expected = 300.0
        else:
            expected = 100.0
        assert out[index].item() == expected, (index, out[index].item(), expected)


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
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_verify_commits_the_state_of_exactly_the_accepted_prefix(accepted):
    """Speculative decoding needs a state that matches the ACCEPTED prefix, not the proposed one.

    A verify step advances past tokens that may be rejected, so the committed state has to be the one
    that would have resulted from stepping only the accepted prefix. This walks the same tokens one at a
    time to build that expectation, and asserts the verify step's committed state equals it — including
    accepted=0, which must leave both states untouched.

    Without this, a rejected suffix leaves the recurrent state ahead of the sequence and every later
    token is conditioned on tokens the model never emitted. There is no error; the output is fluent.
    """
    module, config = _hf_gated_delta_net(seed=14)
    torch.manual_seed(15)
    prefill_len, proposed = 32, 3
    hidden = torch.randn(1, prefill_len + proposed, _TEST_HIDDEN)
    weights = _weights_from_hf(module)
    with torch.no_grad():
        _, state, conv_state = gated_delta_net_prefill(
            hidden[:, :prefill_len], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE)

        expected_state, expected_conv = state, conv_state
        for step in range(accepted):
            token = hidden[:, prefill_len + step:prefill_len + step + 1]
            _, expected_state, expected_conv = gated_delta_net_decode(
                token, weights, _TEST_DIMS, config.rms_norm_eps,
                conv_state=expected_conv, recurrent_state=expected_state)

        outputs, committed_state, committed_conv = gated_delta_net_verify(
            hidden[:, prefill_len:], weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=conv_state, recurrent_state=state,
            accepted=torch.tensor(accepted, dtype=torch.int64))

    assert outputs.shape[1] == proposed, outputs.shape
    assert torch.equal(committed_state, expected_state), \
        (committed_state - expected_state).abs().max()
    assert torch.equal(committed_conv, expected_conv), \
        (committed_conv - expected_conv).abs().max()


@requires_hf_modules
def test_verify_outputs_match_the_reference_for_every_proposed_token():
    """Every proposed token needs its own output, because that is what the acceptance test compares.

    Committing the right state is not enough: the verifier decides how many tokens to accept from these
    outputs, so an output that is right only for the first proposed token would make the acceptance
    decision itself wrong.
    """
    module, config = _hf_gated_delta_net(seed=16)
    torch.manual_seed(17)
    prefill_len, proposed = 32, 4
    hidden = torch.randn(1, prefill_len + proposed, _TEST_HIDDEN)
    weights = _weights_from_hf(module)
    with torch.no_grad():
        reference = module(hidden_states=hidden, cache_params=None, attention_mask=None)
        _, state, conv_state = gated_delta_net_prefill(
            hidden[:, :prefill_len], weights, _TEST_DIMS, config.rms_norm_eps,
            chunk_size=DEFAULT_CHUNK_SIZE)
        outputs, _, _ = gated_delta_net_verify(
            hidden[:, prefill_len:], weights, _TEST_DIMS, config.rms_norm_eps,
            conv_state=conv_state, recurrent_state=state,
            accepted=torch.tensor(proposed, dtype=torch.int64))
    assert torch.allclose(outputs, reference[:, prefill_len:], atol=1e-4), \
        (outputs - reference[:, prefill_len:]).abs().max()


def test_verify_selection_stays_graph_static():
    """The acceptance count is a tensor, so the selection must not become a branch or an index.

    ``fullgraph=True`` is the plugin's setting, so a graph break here is a failure rather than a
    fallback. Compiling with two different acceptance counts and checking both results also confirms the
    same traced graph serves every count — which is the point of using masks.
    """
    torch.manual_seed(18)
    candidates = [torch.randn(2, 3) for _ in range(4)]
    select = torch.compile(_gdn._select_candidate, fullgraph=True)
    for index in range(4):
        picked = select(candidates, torch.tensor(index, dtype=torch.int64))
        assert torch.equal(picked, candidates[index]), index


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

# The multimodal half of the same published config: the vision tower's shape and the four special token
# ids sit BESIDE text_config, not inside it. Kept separate so the text tests above keep proving that the
# text architecture ignores all of this.
PUBLISHED_MULTIMODAL_CONFIG = {
    **PUBLISHED_CONFIG,
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "vision_config": {
        "depth": 27, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "in_channels": 3,
        "intermediate_size": 4304, "num_heads": 16, "num_position_embeddings": 2304,
        "out_hidden_size": 2048, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "deepstack_visual_indexes": [],
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


# ---------------------------------------------------------------------------
# The vision activation: why the reference's GELU is kept as-is
# ---------------------------------------------------------------------------


def _gelu_erf(x):
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _gelu_tanh(x):
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def _bf16_grid(lo=-30.0, hi=30.0):
    """Every bf16 value in the range, as fp64. An input the network cannot represent cannot produce a
    difference the network can observe, so the sweep is over the representable grid, not a linspace."""
    dense = torch.arange(lo, hi, 1.0 / 4096, dtype=torch.float32)
    return torch.unique(dense.to(torch.bfloat16)).to(torch.float64)


def test_the_activation_choice_sits_below_the_precision_floor():
    """The checkpoint's config asks for ``gelu_pytorch_tanh``; the reused encoder computes the exact
    erf form. Substituting the tanh form is a change that looks obviously right and buys nothing.

    Two quantities decide it. The approximation difference is what switching would remove. The
    precision floor is what any bf16 realisation of either form already costs. Switching is only worth
    doing if the first is the larger of the two, and it is ~17x smaller.
    """
    grid = _bf16_grid()
    exact_erf = _gelu_erf(grid)
    exact_tanh = _gelu_tanh(grid)

    approximation_gap = (exact_tanh - exact_erf).abs().max().item()
    precision_floor = min(
        (_gelu_erf(grid.to(torch.float32)).to(torch.bfloat16).to(torch.float64)
         - exact_erf).abs().max().item(),
        (_gelu_tanh(grid.to(torch.float32)).to(torch.bfloat16).to(torch.float64)
         - exact_tanh).abs().max().item(),
    )
    assert approximation_gap < 1e-3, approximation_gap
    assert precision_floor > 5e-3, precision_floor
    assert approximation_gap * 10 < precision_floor, (
        f"the approximation gap ({approximation_gap:.3e}) is no longer small against the bf16 "
        f"precision floor ({precision_floor:.3e}); revisit whether the tanh form should be used")


def test_the_tanh_form_cancels_in_bf16_where_the_erf_form_barely_does():
    """A second reason not to substitute it: computed in bf16, ``1 + tanh(...)`` reaches exactly zero
    on the negative side, so the activation returns 0 where the true GELU is still around -3e-3. The
    erf form saturates at a similar count, so this is not an argument for erf either — it is why the
    substitution cannot be justified as 'closer to the checkpoint'."""
    grid = _bf16_grid().to(torch.bfloat16)
    tanh_inner = torch.tanh(math.sqrt(2.0 / math.pi) * (grid + 0.044715 * grid.pow(3)))
    erf_inner = torch.erf(grid / 1.4142135623730951)
    tanh_cancels = int((tanh_inner == -1.0).sum())
    erf_cancels = int((erf_inner == -1.0).sum())
    assert tanh_cancels > 0 and erf_cancels > 0
    assert abs(tanh_cancels - erf_cancels) < 0.05 * tanh_cancels, (
        f"tanh cancels at {tanh_cancels} inputs and erf at {erf_cancels}; if these diverge, the "
        "choice of form starts to matter for a reason other than the approximation")


# ---------------------------------------------------------------------------
# Positions: the sequential axis, and the regression condition for the 3-axis rotary
# ---------------------------------------------------------------------------


def test_temporal_axis_is_taken_and_narrowed():
    positions = torch.stack([torch.arange(6), torch.arange(6) * 2, torch.arange(6) * 3])
    got = _ops.temporal_axis(positions)
    assert got.dtype is torch.int32
    assert got.tolist() == list(range(6)), "the sequential position must come from the temporal axis"
    # A one-dimensional input is text and passes through, so a text request cannot become a shape error.
    assert _ops.temporal_axis(torch.arange(4)).tolist() == [0, 1, 2, 3]


def test_no_mrope_axis_is_a_per_token_index_inside_an_image():
    """Pins the fact that makes ``temporal_axis``'s contract narrow, so the docstring cannot drift.

    For an image the reference builder emits ``np.indices((1, h, w))``, so across a 2x3 image:

        temporal [0,0,0,0,0,0]   constant
        height   [0,0,0,1,1,1]   repeats
        width    [0,1,2,0,1,2]   goes backwards at each row

    **None of the three is a per-token index.** The temporal axis is still the right one to take, because
    it is the only one that is monotone outside the span and because prefill never reads it -- but a
    future consumer that needs a monotone index must not take it from here.
    """
    temporal, height, width = (torch.tensor(a) for a in (
        [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]))
    for axis, name in ((temporal, "temporal"), (height, "height"), (width, "width")):
        assert not bool((axis[1:] > axis[:-1]).all()), f"{name} unexpectedly strictly increasing"
    # The temporal axis is the only one that does not go BACKWARDS, which is why it survives being
    # carried across the span boundary where all three resume together.
    assert bool((temporal[1:] >= temporal[:-1]).all())
    assert bool((height[1:] >= height[:-1]).all())
    assert not bool((width[1:] >= width[:-1]).all())


def test_three_axis_rotary_matches_the_table_when_the_axes_agree():
    """The regression condition for wiring vision: bit equality, not a tolerance.

    Without it, a change in the text output after the vision path lands cannot be attributed — it could
    be the new rotary or the vision plumbing. With it, text moving means the cause is vision.
    """
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    positions = torch.tensor([0, 1, 2, 7, 63, 1024, 16383], dtype=torch.int64)
    table_cos, table_sin = rotary_tables(
        config.rotary_dim, config.max_position_embeddings, config.rope_theta,
        dtype=config.torch_dtype)
    expected_cos = table_cos.index_select(0, positions)
    expected_sin = table_sin.index_select(0, positions)

    three = positions.unsqueeze(0).expand(3, positions.numel()).contiguous()
    got_cos, got_sin = mrope_tables(three, config.rotary_dim, config.rope_theta,
                                        config.mrope_section, dtype=config.torch_dtype)
    assert torch.equal(got_cos, expected_cos), "the three-axis path is not bit-identical for text"
    assert torch.equal(got_sin, expected_sin)


def test_three_axis_rotary_differs_once_the_axes_differ():
    """The negative half: if it agreed here too, the interleave would be doing nothing and the whole
    three-axis path would be decoration."""
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    positions = torch.arange(8, dtype=torch.int64)
    same = positions.unsqueeze(0).expand(3, 8).contiguous()
    varied = torch.stack([positions, positions * 0 + 3, positions * 0 + 5])
    a, _ = mrope_tables(same, config.rotary_dim, config.rope_theta, config.mrope_section,
                            dtype=config.torch_dtype)
    b, _ = mrope_tables(varied, config.rotary_dim, config.rope_theta, config.mrope_section,
                            dtype=config.torch_dtype)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# The multimodal config wrapper
# ---------------------------------------------------------------------------


def test_multimodal_config_reads_the_wrapper_not_the_text_half():
    """The vision shape and the four special token ids live BESIDE text_config, so the text config's
    unwrapping drops them. This is the class that keeps them."""
    config = _config.Qwen3_5MoeMultimodalConfig.from_configs(PUBLISHED_MULTIMODAL_CONFIG, None)
    assert config.vision_config.depth == 27
    assert config.vision_config.out_hidden_size == config.text_config.hidden_size
    for name in ("image_token_id", "video_token_id",
                 "vision_start_token_id", "vision_end_token_id"):
        assert getattr(config, name) > 0, f"{name} was not read from the wrapper"
    assert config.image_token_id != config.video_token_id


def test_multimodal_config_refuses_a_token_id_outside_the_vocabulary():
    """A wrong id marks the wrong prompt span as vision, which places the image at the wrong offsets and
    produces a fluent lie rather than an error."""
    published = dict(PUBLISHED_MULTIMODAL_CONFIG)
    published["image_token_id"] = 10 ** 9
    with pytest.raises(ValueError, match="outside the vocabulary"):
        _config.Qwen3_5MoeMultimodalConfig.from_configs(published, None)


def test_multimodal_config_refuses_a_missing_token_id():
    published = dict(PUBLISHED_MULTIMODAL_CONFIG)
    del published["vision_start_token_id"]
    with pytest.raises(ValueError, match="vision_start_token_id"):
        _config.Qwen3_5MoeMultimodalConfig.from_configs(published, None)


def test_multimodal_config_refuses_a_text_only_checkpoint():
    published = {k: v for k, v in PUBLISHED_MULTIMODAL_CONFIG.items() if k != "vision_config"}
    with pytest.raises(ValueError, match="Qwen3_5MoeForCausalLM"):
        _config.Qwen3_5MoeMultimodalConfig.from_configs(published, None)


def test_no_text_weight_claims_the_vision_namespace():
    """The multimodal load is ONE map: the text map plus the vision map with a ``visual.`` prefix.

    A name claimed by both halves would let one silently win and the other's weights would never be
    loaded. The class raises on a collision, so what is worth checking here is the property that keeps
    the raise from ever firing — and of the two halves, this is the one that can drift quietly. The
    vision half is generated by the encoder itself (``layout.vision_checkpoint_mappings``), so its names
    move only when the encoder's do; the text map is written out by hand and a new parameter could land
    in the vision namespace.
    """
    config = Qwen3_5MoeConfig.from_configs(PUBLISHED_CONFIG, None)
    text = checkpoint_mappings(config.layer_types, "model.language_model", has_lm_head=True,
                               tie_word_embeddings=False)
    intruders = sorted(name for name in text if name.startswith("visual."))
    assert not intruders, f"these text destinations sit in the vision namespace: {intruders}"
    # And the text map must not consume the vision tower's checkpoint keys, which is what makes reading
    # the checkpoint once with one map safe.
    sources = [key for value in text.values()
               for key in ([value] if isinstance(value, str) else value)]
    assert not [key for key in sources if key.startswith("model.visual.")]


def test_the_two_neuron_config_spellings_resolve_to_one_and_refuse_disagreement():
    """Both names reach the model depending on how the architecture was entered. Ranking them would
    serve a model configured by the loser, so a real disagreement is refused instead."""
    model_bf16_source = os.path.join(_model_dir, "model_bf16.py")
    with open(model_bf16_source) as handle:
        source = handle.read()
    # The module cannot be imported without the plugin, so the function is compiled on its own. That is
    # enough: it takes two arguments and returns one, and touches nothing else.
    namespace: dict = {}
    start = source.index("def resolve_text_neuron_config(")
    end = source.index("\nclass ", start)
    exec(compile(source[start:end], model_bf16_source, "exec"), namespace)  # noqa: S102
    resolve = namespace["resolve_text_neuron_config"]

    sentinel = object()
    assert resolve(sentinel, None) is sentinel
    assert resolve(None, sentinel) is sentinel
    assert resolve(None, None) is None
    # The same object under both names is not a disagreement.
    assert resolve(sentinel, sentinel) is sentinel
    with pytest.raises(ValueError, match="both supplied"):
        resolve(sentinel, object())


# ---------------------------------------------------------------------------
# Per-slot recurrent state: the mechanism concurrency needs
# ---------------------------------------------------------------------------


def test_one_slot_is_the_identity_and_the_plain_assignment():
    """At one slot the slot functions must be exactly what they replaced, or generalising the state
    changes the shipped configuration's numbers."""
    state = torch.randn(1, 3, 4, dtype=torch.float32)
    assert _ops.slot_mask(0, state, 1) is None
    assert _ops.read_state_slot(state, 0, 1) is state, "one slot must not copy or reduce"
    updated = torch.randn(1, 3, 4, dtype=torch.float32)
    _ops.write_state_slot(state, updated, 0, 1)
    assert torch.equal(state, updated), "one slot must be the plain assignment, bit for bit"


@pytest.mark.parametrize("num_slots", [2, 3, 8])
def test_reading_a_slot_returns_exactly_that_slot(num_slots):
    state = torch.randn(num_slots, 2, 3, dtype=torch.float32)
    for slot in range(num_slots):
        got = _ops.read_state_slot(state, torch.tensor(slot), num_slots)
        assert got.shape == (1, 2, 3)
        assert torch.equal(got[0], state[slot]), f"slot {slot} read the wrong entry"


@pytest.mark.parametrize("num_slots", [2, 5])
def test_writing_a_slot_leaves_the_others_bit_identical(num_slots):
    """The whole point. A write that perturbs another slot is a request reading someone else's state,
    which produces fluent output with the wrong history rather than an error."""
    state = torch.randn(num_slots, 2, 3, dtype=torch.float32)
    for slot in range(num_slots):
        before = state.clone()
        updated = torch.randn(1, 2, 3, dtype=torch.float32)
        _ops.write_state_slot(state, updated, torch.tensor(slot), num_slots)
        assert torch.equal(state[slot], updated[0]), f"slot {slot} was not written"
        for other in range(num_slots):
            if other != slot:
                assert torch.equal(state[other], before[other]), (
                    f"writing slot {slot} disturbed slot {other}")


def test_the_slot_functions_hold_for_bf16_state_too():
    """The conv state is the model dtype, not fp32, so the mask must not promote it."""
    state = torch.randn(3, 4, dtype=torch.bfloat16).unsqueeze(0).expand(3, 3, 4).contiguous()
    state = state + torch.arange(3, dtype=torch.bfloat16).reshape(3, 1, 1)
    updated = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    before = state.clone()
    _ops.write_state_slot(state, updated, torch.tensor(1), 3)
    assert state.dtype is torch.bfloat16
    assert torch.equal(state[1], updated[0])
    assert torch.equal(state[0], before[0]) and torch.equal(state[2], before[2])


def test_two_slots_carry_two_independent_recurrences():
    """End to end over the real scan: two sequences interleaved in two slots must give the same output
    as each run alone.

    This is the claim concurrency rests on, and the mechanism is checkable without a device even though
    the runner-side pool is not. The decode inputs are pre-generated so both runs consume exactly the
    same tensors -- a shared generator would make an ordering difference look like a state leak.
    """
    module, config = _hf_gated_delta_net(seed=21)
    weights = _weights_from_hf(module)
    eps = config.rms_norm_eps
    torch.manual_seed(31)
    prefill_len, decode_steps, sequences = 32, 3, 2
    prompts = [torch.randn(1, prefill_len, _TEST_HIDDEN) for _ in range(sequences)]
    steps = [[torch.randn(1, 1, _TEST_HIDDEN) for _ in range(decode_steps)]
             for _ in range(sequences)]
    continuing = torch.tensor(1.0)

    def empty(slots):
        recurrent = torch.zeros(slots, _TEST_DIMS["v_heads"], _TEST_DIMS["head_k_dim"],
                                _TEST_DIMS["head_v_dim"], dtype=torch.float32)
        conv = torch.zeros(slots, 2 * _TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"]
                           + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"],
                           _TEST_DIMS["kernel"] - 1)
        return recurrent, conv

    with torch.no_grad():
        # Each sequence alone, in a single-slot state.
        alone = []
        for sequence in range(sequences):
            recurrent, conv = empty(1)
            _, state, conv_out = gated_delta_net_prefill(
                prompts[sequence], weights, _TEST_DIMS, eps, chunk_size=DEFAULT_CHUNK_SIZE)
            _ops.write_state_slot(recurrent, state, 0, 1)
            _ops.write_state_slot(conv, conv_out, 0, 1)
            outputs = []
            for step in steps[sequence]:
                out, state, conv_out = gated_delta_net_decode(
                    step, weights, _TEST_DIMS, eps, conv_state=conv, recurrent_state=recurrent,
                    is_continuation=continuing)
                _ops.write_state_slot(recurrent, state, 0, 1)
                _ops.write_state_slot(conv, conv_out, 0, 1)
                outputs.append(out.clone())
            alone.append(outputs)

        # Both sequences sharing a two-slot state, decode steps alternating between them.
        recurrent, conv = empty(sequences)
        for sequence in range(sequences):
            _, state, conv_out = gated_delta_net_prefill(
                prompts[sequence], weights, _TEST_DIMS, eps, chunk_size=DEFAULT_CHUNK_SIZE)
            _ops.write_state_slot(recurrent, state, torch.tensor(sequence), sequences)
            _ops.write_state_slot(conv, conv_out, torch.tensor(sequence), sequences)
        shared: list[list[torch.Tensor]] = [[] for _ in range(sequences)]
        for step_index in range(decode_steps):
            for sequence in range(sequences):
                slot = torch.tensor(sequence)
                out, state, conv_out = gated_delta_net_decode(
                    steps[sequence][step_index], weights, _TEST_DIMS, eps,
                    conv_state=_ops.read_state_slot(conv, slot, sequences),
                    recurrent_state=_ops.read_state_slot(recurrent, slot, sequences),
                    is_continuation=continuing)
                _ops.write_state_slot(recurrent, state, slot, sequences)
                _ops.write_state_slot(conv, conv_out, slot, sequences)
                shared[sequence].append(out.clone())

    for sequence in range(sequences):
        for step_index, (want, got) in enumerate(zip(alone[sequence], shared[sequence])):
            assert torch.equal(want, got), (
                f"sequence {sequence} step {step_index} differs once the two share the state; "
                "one request is reading the other's history")


# ---------------------------------------------------------------------------
# The shared state-layer spec
# ---------------------------------------------------------------------------


def _kv_cache_module():
    """``vllm_neuron/model/kv_cache.py``, loaded by path: it imports only torch, so the plugin (and a
    device with it) stays out."""
    import importlib.util as _util
    # _model_dir is .../vllm_neuron/model/qwen3_5_moe; kv_cache.py sits one level up.
    path = os.path.join(os.path.dirname(_model_dir), "kv_cache.py")
    spec = _util.spec_from_file_location("_kv_cache_under_test", path)
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_layer_spec_refuses_a_shape_without_a_dtype():
    """Each state tensor needs both; a mismatch allocates the wrong number of bytes, and the symptom is
    a state that overlaps its neighbour rather than an error."""
    kv_cache = _kv_cache_module()
    with pytest.raises(ValueError, match="dtype"):
        kv_cache.StateLayerSpec(name="a", shapes=((4, 3), (2,)), dtypes=(torch.float32,),
                                state_kind="GDN_ATTN")


def test_state_layer_spec_refuses_declaring_nothing():
    kv_cache = _kv_cache_module()
    with pytest.raises(ValueError, match="no state tensors"):
        kv_cache.StateLayerSpec(name="a", shapes=(), dtypes=(), state_kind="GDN_ATTN")


def test_state_layer_spec_refuses_a_degenerate_dimension():
    kv_cache = _kv_cache_module()
    with pytest.raises(ValueError, match="non-positive"):
        kv_cache.StateLayerSpec(name="a", shapes=((4, 0),), dtypes=(torch.float32,),
                                state_kind="GDN_ATTN")


def test_kv_spec_refuses_a_name_claimed_twice():
    """The cache is keyed by name across both kinds of layer, so a collision lets one entry shadow the
    other -- a layer would then read another layer's state with no error."""
    kv_cache = _kv_cache_module()
    attention = kv_cache.LayerSpec(name="layers.0.x", num_kv_heads=2, head_size=8,
                                   dtype=torch.bfloat16)
    state = kv_cache.StateLayerSpec(name="layers.0.x", shapes=((4,),), dtypes=(torch.float32,),
                                    state_kind="GDN_ATTN")
    with pytest.raises(ValueError, match="duplicate layer name"):
        kv_cache.KVSpec(layers=[attention], state_layers=[state])
    # And the two kinds coexist happily under distinct names.
    state = kv_cache.StateLayerSpec(name="layers.0.y", shapes=((4,),), dtypes=(torch.float32,),
                                    state_kind="GDN_ATTN")
    assert kv_cache.KVSpec(layers=[attention], state_layers=[state]).state_layers[0].name


def test_kv_spec_still_works_for_a_model_that_declares_no_state():
    """Every other model in the repository constructs KVSpec positionally with attention layers only."""
    kv_cache = _kv_cache_module()
    spec = kv_cache.KVSpec(layers=[
        kv_cache.LayerSpec(name="layers.0.self_attn", num_kv_heads=2, head_size=8,
                           dtype=torch.bfloat16)])
    assert spec.state_layers == []


def test_a_multi_request_step_is_refused_rather_than_taking_the_first_slot():
    """The runner hands one slot per request. Until the scan is segmented per request, running a
    multi-request step would put every request through one sequence's state -- fluent output with the
    wrong history, which is the failure this whole mechanism exists to prevent."""
    state = torch.zeros(4, 2, 3)
    with pytest.raises(NotImplementedError, match="multi-request step"):
        _ops.slot_mask(torch.tensor([0, 1, 2]), state, 4)
    # One slot supplied as a length-one tensor is the normal case and must work.
    assert _ops.slot_mask(torch.tensor([2]), state, 4) is not None


def test_an_out_of_range_slot_raises_with_index_select_and_is_lost_with_the_one_hot():
    """The two slot forms differ on the case that matters, and the difference is why one is preferred.

    ``index_select`` refuses an out-of-range slot. The one-hot product cannot: the mask is simply all
    zeros, so a read returns zeros and a write is dropped, and the slot value has to be trusted from the
    runner. That silent loss is what an early experiment mistook for a defect in the arithmetic -- capping
    the pool to one slot put the runner's slot (2) out of range, and the resulting zeros looked like a
    storage failure.
    """
    state = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 2, 3)

    previous = os.environ.get("QWEN3_5_MOE_SLOT_INDEX")
    try:
        os.environ["QWEN3_5_MOE_SLOT_INDEX"] = "1"
        with pytest.raises((IndexError, RuntimeError)):
            _ops.read_state_slot(state, torch.tensor(9), 4)

        os.environ["QWEN3_5_MOE_SLOT_INDEX"] = "0"
        before = state.clone()
        mask = _ops.slot_mask(torch.tensor(9), state, 4)
        assert float(mask.abs().sum()) == 0.0
        assert float(_ops.read_state_slot(state, torch.tensor(9), 4).abs().sum()) == 0.0
        _ops.write_state_slot(state, torch.ones(1, 2, 3), torch.tensor(9), 4)
        assert torch.equal(state, before), "an out-of-range write must not land on another slot"
    finally:
        if previous is None:
            os.environ.pop("QWEN3_5_MOE_SLOT_INDEX", None)
        else:
            os.environ["QWEN3_5_MOE_SLOT_INDEX"] = previous


@pytest.mark.parametrize("form", ["1", "0"])
def test_both_slot_forms_read_and_write_the_same_slot(form):
    """Whichever form the device ends up needing, the two must agree on the slot they touch."""
    previous = os.environ.get("QWEN3_5_MOE_SLOT_INDEX")
    os.environ["QWEN3_5_MOE_SLOT_INDEX"] = form
    try:
        state = torch.randn(5, 2, 3)
        for slot in range(5):
            got = _ops.read_state_slot(state, torch.tensor(slot), 5)
            assert torch.equal(got[0], state[slot]), f"{form}: slot {slot} read the wrong row"
        for slot in range(5):
            before = state.clone()
            updated = torch.randn(1, 2, 3)
            _ops.write_state_slot(state, updated, torch.tensor(slot), 5)
            assert torch.equal(state[slot], updated[0]), f"{form}: slot {slot} was not written"
            for other in range(5):
                if other != slot:
                    assert torch.equal(state[other], before[other]), (
                        f"{form}: writing slot {slot} disturbed slot {other}")
    finally:
        if previous is None:
            os.environ.pop("QWEN3_5_MOE_SLOT_INDEX", None)
        else:
            os.environ["QWEN3_5_MOE_SLOT_INDEX"] = previous

def assert_matches_solo(got, want, leaked, what, floor_ulps=64):
    """One row of a batched call against the same request run alone, plus a negative control.

    Bit equality is the natural assertion here and it is NOT portable. The batched call reduces over a
    wider matmul, so the library is free to use a different kernel and a different summation order:
    measured, these rows are bit-identical on arm64 and differ by up to 1.0e-06 at a scale of 2.1 on
    x86_64 with torch 2.11 -- a few ULP of fp32, which is arithmetic, not a leak.

    Relaxing an equality can hide the thing the test exists to catch, so the assertion is two-sided.
    The difference from the SAME request must sit at the rounding floor, and the difference from a
    DIFFERENT request (``leaked``) must be orders of magnitude larger. If the request axis ever stopped
    being independent, the first bound is the one that breaks; if the tolerance were ever loosened far
    enough to admit a leak, the second is what says so.
    """
    difference = (got.float() - want.float()).abs().max().item()
    scale = max(want.float().abs().max().item(), 1.0)
    floor = floor_ulps * torch.finfo(want.dtype).eps * scale
    assert difference <= floor, (
        f"{what}: differs by {difference:.3e}, above the {floor:.3e} rounding floor for "
        f"{want.dtype} at scale {scale:.3e}")

    # The control keeps the bound above honest: it must be nowhere near what a wrong row looks like.
    leak = (leaked.float() - want.float()).abs().max().item()
    assert leak > 100 * floor, (
        f"{what}: the negative control is not discriminating -- a different request differs by only "
        f"{leak:.3e}, which the {floor:.3e} floor cannot be trusted to exclude")


def test_the_decode_step_is_batch_general_over_the_request_axis():
    """Measured, not assumed. The kernel guards the TOKEN axis and says nothing about the request axis,
    and stage A rests entirely on that axis being independent -- so each row of a batched call must be
    bit-identical to the same request called alone."""
    module, config = _hf_gated_delta_net(seed=41)
    weights = _weights_from_hf(module)
    eps = config.rms_norm_eps
    torch.manual_seed(43)
    requests = 3
    tokens = torch.randn(requests, 1, _TEST_HIDDEN)
    recurrent = torch.randn(requests, _TEST_DIMS["v_heads"], _TEST_DIMS["head_k_dim"],
                            _TEST_DIMS["head_v_dim"]) * 0.1
    conv_dim = (2 * _TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"]
                + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"])
    conv = torch.randn(requests, conv_dim, _TEST_DIMS["kernel"] - 1) * 0.1
    continuing = torch.tensor(1.0)

    with torch.no_grad():
        batched = gated_delta_net_decode(
            tokens, weights, _TEST_DIMS, eps, conv_state=conv, recurrent_state=recurrent,
            is_continuation=continuing)
        for request in range(requests):
            alone = gated_delta_net_decode(
                tokens[request:request + 1], weights, _TEST_DIMS, eps,
                conv_state=conv[request:request + 1],
                recurrent_state=recurrent[request:request + 1], is_continuation=continuing)
            neighbour = (request + 1) % requests
            for got, want in zip(batched, alone):
                assert_matches_solo(
                    got[request:request + 1], want, got[neighbour:neighbour + 1],
                    f"request {request} batched with {requests - 1} others")


def test_a_per_request_continuation_flag_zeroes_only_that_request():
    """A fresh request in a batched step must start from zero while its neighbours carry on. Without a
    per-request flag, one request's first token would either reset everyone or continue from state it
    has never seen."""
    module, config = _hf_gated_delta_net(seed=45)
    weights = _weights_from_hf(module)
    eps = config.rms_norm_eps
    torch.manual_seed(47)
    conv_dim = (2 * _TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"]
                + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"])
    tokens = torch.randn(2, 1, _TEST_HIDDEN)
    recurrent = torch.randn(2, _TEST_DIMS["v_heads"], _TEST_DIMS["head_k_dim"],
                            _TEST_DIMS["head_v_dim"]) * 0.1
    conv = torch.randn(2, conv_dim, _TEST_DIMS["kernel"] - 1) * 0.1

    with torch.no_grad():
        # Request 0 is fresh, request 1 continues.
        mixed = gated_delta_net_decode(
            tokens, weights, _TEST_DIMS, eps, conv_state=conv, recurrent_state=recurrent,
            is_continuation=torch.tensor([0.0, 1.0]))
        # The same two calls made separately, with the same flags.
        fresh = gated_delta_net_decode(
            tokens[:1], weights, _TEST_DIMS, eps, conv_state=conv[:1], recurrent_state=recurrent[:1],
            is_continuation=torch.tensor(0.0))
        carried = gated_delta_net_decode(
            tokens[1:], weights, _TEST_DIMS, eps, conv_state=conv[1:], recurrent_state=recurrent[1:],
            is_continuation=torch.tensor(1.0))

    for got, want in zip(mixed, fresh):
        assert_matches_solo(got[:1], want, got[1:], "the fresh request, batched with a continuing one")
    for got, want in zip(mixed, carried):
        assert_matches_solo(got[1:], want, got[:1], "the continuing request, batched with a fresh one")


def test_several_requests_decoding_from_a_shared_pool_match_their_solo_runs():
    """Stage A end to end: a pool of four slots, three requests holding slots 2, 0 and 3, all advancing
    in ONE call, with the gather and scatter written as one-hot products. Each request's output and
    resulting state must match the same request run alone."""
    module, config = _hf_gated_delta_net(seed=51)
    weights = _weights_from_hf(module)
    eps = config.rms_norm_eps
    torch.manual_seed(53)
    slots_total, steps = 4, 3
    holders = torch.tensor([2, 0, 3])
    requests = holders.numel()
    conv_dim = (2 * _TEST_DIMS["k_heads"] * _TEST_DIMS["head_k_dim"]
                + _TEST_DIMS["v_heads"] * _TEST_DIMS["head_v_dim"])
    tokens = [torch.randn(requests, 1, _TEST_HIDDEN) for _ in range(steps)]
    continuing = torch.ones(requests)

    def blank(count):
        return (torch.zeros(count, _TEST_DIMS["v_heads"], _TEST_DIMS["head_k_dim"],
                            _TEST_DIMS["head_v_dim"]),
                torch.zeros(count, conv_dim, _TEST_DIMS["kernel"] - 1))

    with torch.no_grad():
        # Solo: one state each, stepped independently.
        solo = []
        for request in range(requests):
            recurrent, conv = blank(1)
            outputs = []
            for step in range(steps):
                out, recurrent, conv = gated_delta_net_decode(
                    tokens[step][request:request + 1], weights, _TEST_DIMS, eps,
                    conv_state=conv, recurrent_state=recurrent,
                    is_continuation=torch.tensor(1.0))
                outputs.append(out.clone())
            solo.append((outputs, recurrent, conv))

        # Shared: one pool of four slots, three of them in use, all advanced together.
        recurrent, conv = blank(slots_total)
        shared = []
        for step in range(steps):
            out, new_recurrent, new_conv = gated_delta_net_decode(
                tokens[step], weights, _TEST_DIMS, eps,
                conv_state=_ops.read_state_slots(conv, holders, slots_total),
                recurrent_state=_ops.read_state_slots(recurrent, holders, slots_total),
                is_continuation=continuing)
            _ops.write_state_slots(recurrent, new_recurrent, holders, slots_total)
            _ops.write_state_slots(conv, new_conv, holders, slots_total)
            shared.append(out.clone())

    for request in range(requests):
        outputs, want_recurrent, want_conv = solo[request]
        neighbour = (request + 1) % requests
        for step in range(steps):
            assert_matches_solo(
                shared[step][request:request + 1], outputs[step],
                shared[step][neighbour:neighbour + 1],
                f"request {request} at step {step}, sharing a {slots_total}-slot pool")
        slot = int(holders[request])
        other_slot = int(holders[neighbour])
        assert_matches_solo(
            recurrent[slot:slot + 1], want_recurrent, recurrent[other_slot:other_slot + 1],
            f"request {request}'s recurrent state in slot {slot}")
        assert_matches_solo(
            conv[slot:slot + 1], want_conv, conv[other_slot:other_slot + 1],
            f"request {request}'s conv state in slot {slot}")

    # The unused slot must still be zero: nothing may spill into a slot no request holds.
    unused = ({0, 1, 2, 3} - {int(s) for s in holders}).pop()
    assert float(recurrent[unused].abs().sum()) == 0.0
    assert float(conv[unused].abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Stage B: several requests packed into one prefill row
# ---------------------------------------------------------------------------


def _scan_inputs(length, heads=4, dk=16, dv=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, length, heads, dk, generator=generator),
            torch.randn(1, length, heads, dk, generator=generator),
            torch.randn(1, length, heads, dv, generator=generator),
            -torch.rand(1, length, heads, generator=generator) * 0.3,
            torch.rand(1, length, heads, generator=generator))


def test_the_prefill_scan_is_batch_general_over_the_request_axis():
    """The counterpart of the decode measurement, and the reason stage B is a mask rather than a
    rewrite: the scan already carries an independent state per row."""
    rows = [_scan_inputs(48, seed=s) for s in (1, 2, 3)]
    packed = [torch.cat(parts, dim=0) for parts in zip(*rows)]
    with torch.no_grad():
        out, state = chunk_gated_delta_rule(*packed, chunk_size=16)
        for index, row in enumerate(rows):
            out_1, state_1 = chunk_gated_delta_rule(*row, chunk_size=16)
            assert torch.equal(out[index:index + 1], out_1), f"row {index} output differs when batched"
            assert torch.equal(state[index:index + 1], state_1), f"row {index} state differs"


def test_a_chunk_carry_of_all_ones_is_the_identity():
    """The mask must not change anything when nothing is segmented, or every existing agreement with
    the reference is at risk."""
    parts = _scan_inputs(48, seed=5)
    with torch.no_grad():
        plain = chunk_gated_delta_rule(*parts, chunk_size=16)
        masked = chunk_gated_delta_rule(*parts, chunk_size=16, chunk_carry=torch.ones(1, 3))
    for got, want in zip(masked, plain):
        assert torch.equal(got, want)


def test_two_chunk_aligned_requests_packed_in_one_row_match_their_solo_runs():
    """Stage B for the scan. Both requests are multiples of the chunk size, so the boundary lands on a
    chunk edge and a per-chunk carry mask is enough -- a boundary inside a chunk could not be expressed
    this way, because the within-chunk coupling would still mix the two."""
    chunk = 16
    first, second = 32, 16
    a = _scan_inputs(first, seed=7)
    b = _scan_inputs(second, seed=11)
    packed = [torch.cat([x, y], dim=1) for x, y in zip(a, b)]
    chunks = (first + second) // chunk
    carry = torch.ones(1, chunks)
    carry[0, first // chunk] = 0.0          # the second request starts here

    with torch.no_grad():
        out_a, state_a = chunk_gated_delta_rule(*a, chunk_size=chunk)
        out_b, state_b = chunk_gated_delta_rule(*b, chunk_size=chunk)
        out_p, state_p, per_chunk = chunk_gated_delta_rule(
            *packed, chunk_size=chunk, chunk_carry=carry, return_chunk_states=True)

    assert torch.equal(out_p[:, :first], out_a), "the first request's output changed"
    assert torch.equal(out_p[:, first:], out_b), "the second request read the first one's state"
    # Each request's final state is the per-chunk state at its last chunk.
    assert torch.equal(per_chunk[:, first // chunk - 1], state_a)
    assert torch.equal(state_p, state_b)
    assert per_chunk.shape[1] == chunks


def test_the_conv_reads_across_a_packed_boundary_and_the_masked_form_does_not():
    """Both halves of the claim, because only the second is a fix and only the first justifies it.

    F.conv1d over a packed row is wrong on exactly kernel_size - 1 positions of each following request,
    and wrong by order 1 rather than by rounding. The masked tap sum agrees with the solo runs to the
    accumulation-order difference and no more.
    """
    torch.manual_seed(17)
    channels, kernel, first, second = 32, 4, 12, 12
    weight = torch.randn(channels, 1, kernel)
    packed = torch.randn(1, channels, first + second)
    segment = torch.cat([torch.zeros(first), torch.ones(second)])

    with torch.no_grad():
        solo_a, _ = segmented_causal_conv1d(packed[..., :first], weight, kernel, channels)
        solo_b, _ = segmented_causal_conv1d(packed[..., first:], weight, kernel, channels)
        unmasked, _ = segmented_causal_conv1d(packed, weight, kernel, channels)
        masked, _ = segmented_causal_conv1d(packed, weight, kernel, channels, segment_id=segment)

    assert torch.equal(unmasked[..., :first], solo_a), "the first request should already be right"
    damaged = ((unmasked[..., first:] - solo_b).abs().amax(dim=1) > 1e-6).sum().item()
    assert damaged == kernel - 1, (
        f"expected exactly {kernel - 1} damaged positions, got {damaged}; if this changes, the "
        "reasoning about which positions need the mask no longer holds")
    assert (unmasked[..., first:] - solo_b).abs().max() > 1.0, "the damage should be order 1"

    # The masked form agrees with both solo runs. NOT bit-identical: the tap sum accumulates in a
    # different order from F.conv1d, which is why the unsegmented path keeps F.conv1d.
    assert (masked[..., :first] - solo_a).abs().max() < 1e-5
    assert (masked[..., first:] - solo_b).abs().max() < 1e-5


def test_the_masked_conv_is_not_claimed_to_be_bit_identical():
    """Pins the caveat itself. If the tap sum ever became bit-identical the unsegmented path could be
    dropped, and this test failing is how that would be noticed."""
    torch.manual_seed(19)
    channels, kernel, seq = 16, 4, 20
    weight = torch.randn(channels, 1, kernel)
    x = torch.randn(1, channels, seq)
    one_request = torch.zeros(seq)
    with torch.no_grad():
        plain, _ = segmented_causal_conv1d(x, weight, kernel, channels)
        tapped, _ = segmented_causal_conv1d(x, weight, kernel, channels, segment_id=one_request)
    assert (plain - tapped).abs().max() < 1e-5, "the two forms must agree numerically"
    assert not torch.equal(plain, tapped), (
        "the two forms are now bit-identical; the unsegmented F.conv1d path exists only because they "
        "were not, so it can be removed -- and this test should be replaced by that")


def test_the_chunk_aligned_layout_places_each_request_on_a_chunk_boundary():
    """The index math, checked against hand-computed offsets. Three requests of 20, 36 and 8 tokens at
    chunk 16 occupy 32, 48 and 16 aligned positions, so they start at 0, 32 and 80."""
    starts = torch.tensor([0, 20, 56, 64])
    layout = _ops.chunk_aligned_layout(starts, chunk_size=16, aligned_len=112, kernel=4)
    source, valid, carry = layout.source, layout.valid, layout.carry
    assert carry.tolist() == [0, 1, 0, 1, 1, 0, 1], (
        "a chunk that begins a request must carry nothing, and chunk 0 always begins one")
    for aligned_at, length, packed_at in ((0, 20, 0), (32, 36, 20), (80, 8, 56)):
        span = source[aligned_at:aligned_at + length].tolist()
        assert span == list(range(packed_at, packed_at + length))
        assert int(valid[aligned_at:aligned_at + length].sum()) == length
    assert int(valid.sum()) == 20 + 36 + 8, "only real tokens may be marked valid"


def test_the_layout_refuses_an_aligned_length_that_is_not_whole_chunks():
    with pytest.raises(ValueError, match="multiple of chunk_size"):
        _ops.chunk_aligned_layout(torch.tensor([0, 8]), chunk_size=16, aligned_len=100,
                                  kernel=4)


def test_three_unequal_requests_prefill_together_and_match_their_solo_runs():
    """Stage B end to end, and the case the whole design exists for: requests of DIFFERENT lengths,
    packed into one row, re-laid-out onto chunk boundaries, scanned once.

    Each request's output must equal its solo run and its final state must land where its own last chunk
    put it. Bit equality, not a tolerance: the aligned layout moves tokens but must not change any
    arithmetic they take part in.
    """
    chunk = 16
    lengths = [20, 36, 8]
    aligned_len = sum(-(-length // chunk) * chunk for length in lengths)
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)))
    parts = [_scan_inputs(length, seed=100 + index) for index, length in enumerate(lengths)]
    packed = [torch.cat(tensors, dim=1) for tensors in zip(*parts)]

    layout = _ops.chunk_aligned_layout(starts, chunk, aligned_len, kernel=4)
    source, valid, carry = layout.source, layout.valid, layout.carry

    def to_aligned(tensor):
        """Gather onto the aligned layout and zero the padding, which is what `valid` is for."""
        gathered = torch.index_select(tensor, 1, source)
        shape = (1, aligned_len) + (1,) * (tensor.dim() - 2)
        return gathered * valid.reshape(shape).to(gathered.dtype)

    with torch.no_grad():
        solo = [chunk_gated_delta_rule(*part, chunk_size=chunk) for part in parts]
        aligned = [to_aligned(tensor) for tensor in packed]
        out, _final, per_chunk = chunk_gated_delta_rule(
            *aligned, chunk_size=chunk, chunk_carry=carry.unsqueeze(0),
            return_chunk_states=True)

    aligned_starts = [0]
    for length in lengths[:-1]:
        aligned_starts.append(aligned_starts[-1] + -(-length // chunk) * chunk)

    for index, (length, aligned_at) in enumerate(zip(lengths, aligned_starts)):
        want_out, want_state = solo[index]
        got_out = out[:, aligned_at:aligned_at + length]
        assert torch.equal(got_out, want_out), (
            f"request {index} (length {length}) differs when packed and aligned")
        last_chunk = (aligned_at + -(-length // chunk) * chunk) // chunk - 1
        assert torch.equal(per_chunk[:, last_chunk], want_state), (
            f"request {index}'s final state is not at its last chunk")


def test_the_whole_prefill_runs_three_unequal_requests_packed_in_one_row():
    """Stage B through the FULL prefill, not just the scan: projections, the segmented convolution, the
    carry-masked scan, the per-request state extraction and the scatter back.

    Each request's output must match its solo prefill, and each request's returned states must match the
    states its solo prefill ended with. The convolution is the masked tap form here and the solo runs use
    F.conv1d, which are not bit-identical (4.8e-7 by measurement), so this compares numerically -- the
    scan-only test above is the one that pins bit equality.
    """
    module, config = _hf_gated_delta_net(seed=61)
    weights = _weights_from_hf(module)
    eps = config.rms_norm_eps
    torch.manual_seed(67)
    chunk = 16
    lengths = [20, 36, 8]
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    prompts = [torch.randn(1, length, _TEST_HIDDEN) for length in lengths]
    row = torch.cat(prompts, dim=1)

    requests = len(lengths)
    aligned = sum(lengths) + requests * (chunk - 1)
    aligned += (-aligned) % chunk
    layout = _ops.chunk_aligned_layout(starts, chunk, aligned, _TEST_DIMS["kernel"])

    with torch.no_grad():
        solo = [gated_delta_net_prefill(prompt, weights, _TEST_DIMS, eps, chunk_size=chunk)
                for prompt in prompts]
        out, state, conv = gated_delta_net_prefill(
            row, weights, _TEST_DIMS, eps, chunk_size=chunk, packed=layout)

    assert out.shape == row.shape, "the output must come back in the caller's packed layout"
    assert state.shape[0] == requests, f"expected one state per request, got {state.shape[0]}"
    assert conv.shape[0] == requests

    for index, (length, start) in enumerate(zip(lengths, starts[:-1].tolist())):
        want_out, want_state, want_conv = solo[index]
        got_out = out[:, start:start + length]
        assert (got_out - want_out).abs().max() < 2e-5, (
            f"request {index} (length {length}) output differs by "
            f"{(got_out - want_out).abs().max():.3e}")
        assert (state[index] - want_state[0]).abs().max() < 2e-5, (
            f"request {index}'s recurrent state is not the one its solo prefill ended with")
        assert (conv[index] - want_conv[0]).abs().max() < 2e-5, (
            f"request {index}'s conv history is not its own tail")


def test_packed_prefill_of_a_single_request_agrees_with_the_unpacked_path():
    """A pooled deployment carrying one request must not take a different numerical path than one
    without the pool. The two convolutions differ by the accumulation order, so this is a numerical
    bound rather than bit equality -- which is itself the reason the model passes packed=None when there
    is only one request."""
    module, config = _hf_gated_delta_net(seed=71)
    weights = _weights_from_hf(module)
    torch.manual_seed(73)
    chunk, length = 16, 32
    prompt = torch.randn(1, length, _TEST_HIDDEN)
    starts = torch.tensor([0, length], dtype=torch.int32)
    layout = _ops.chunk_aligned_layout(starts, chunk, length, _TEST_DIMS["kernel"])
    with torch.no_grad():
        plain = gated_delta_net_prefill(prompt, weights, _TEST_DIMS, config.rms_norm_eps,
                                        chunk_size=chunk)
        boxed = gated_delta_net_prefill(prompt, weights, _TEST_DIMS, config.rms_norm_eps,
                                        chunk_size=chunk, packed=layout)
    assert (boxed[0] - plain[0]).abs().max() < 2e-5
    assert (boxed[1][0] - plain[1][0]).abs().max() < 2e-5
    assert (boxed[2][0] - plain[2][0]).abs().max() < 2e-5


# ---------------------------------------------------------------------------
# Why the state pool must be allocated contiguously
# ---------------------------------------------------------------------------


def test_a_mutation_of_a_non_contiguous_view_is_lost_across_a_compiled_graph():
    """The invariant behind the state pool's allocation, at the cheapest rung.

    The pool used to be carved out of one raw buffer with ``as_strided``, so a slot's states sat together
    in one page -- the layout upstream's GPU runner uses. Under ``torch.compile`` the mutation of such a
    view does not survive the graph boundary: the model reads a stale state while its writes appear to
    land, and the output degenerates to a repeated token with nothing raising.

    Measured with one difference changed: strided gave [259, 465, 465] and contiguous gave
    [259, 46, 331], which is what the unpooled path gives. Under eager BOTH were correct, which is what
    localised it to the compiled graph rather than to the arithmetic.

    This test is the mechanism on its own, so the reason the allocation is contiguous cannot be lost.
    """
    def step(state, row, value):
        state.index_copy_(0, row, value)
        return state.sum()

    compiled = torch.compile(step, backend="eager", fullgraph=False)

    def survives(state):
        """Does a mutation made inside the compiled callable persist in ``state`` afterwards?"""
        row = torch.tensor([1])
        value = torch.full((1, 4), 7.0)
        compiled(state, row, value)
        return bool(torch.equal(state[1], value[0]))

    contiguous = torch.zeros(3, 4)
    assert contiguous.is_contiguous()
    assert survives(contiguous), "a contiguous state must keep the write"

    # The pool's shape: one page per slot, holding this state plus a second one, so the slot stride is
    # larger than the state's own size and the view is not contiguous.
    page = 4 + 6
    raw = torch.zeros(3 * page)
    strided = torch.as_strided(raw, size=(3, 4), stride=(page, 1), storage_offset=0)
    assert not strided.is_contiguous()
    lost = not survives(strided)
    assert lost or torch.equal(strided[1], torch.full((4,), 7.0)), "the probe itself must be meaningful"
    # The point is not that it ALWAYS loses the write on every backend -- it is that a non-contiguous
    # state is the thing that differed when the pool was wrong, so the allocation must not produce one.
    assert not strided.is_contiguous(), (
        "if as_strided with a larger slot stride now yields a contiguous tensor, the allocation's "
        "contiguity check has become the only thing standing between this port and the defect")
