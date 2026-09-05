# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the multi-token prediction head's checkpoint mapping and its one un-inferrable step.

No Neuron device and no checkpoint required. The head's module itself cannot be imported on CPU (it
builds the attention and the MoE, which reach for the platform), so what is pinned here is the part that
can be: the mapping from the head's parameters to the checkpoint's keys, and the order of the
concatenation that feeds ``fc``.

Those two are also where the failures are silent. A mistyped source key loads nothing and the head
produces plausible tokens from an unweighted layer; the swapped concatenation loads every weight
without complaint and reads the wrong learned columns.
"""
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

_search = os.path.dirname(os.path.abspath(__file__))
for _ in range(8):
    _model_dir = os.path.join(_search, "vllm_neuron", "model", "qwen3_5_moe")
    if os.path.exists(os.path.join(_model_dir, "layout.py")):
        break
    _search = os.path.dirname(_search)
else:  # pragma: no cover - only reachable if the test is moved out of the repo
    raise RuntimeError("could not locate vllm_neuron/model/qwen3_5_moe/")

_PACKAGE = "_qwen3_5_moe_mtp_under_test"
_package = types.ModuleType(_PACKAGE)
_package.__path__ = [_model_dir]
sys.modules[_PACKAGE] = _package
_loaded = {}
for _name in ("ops", "layout"):
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE}.{_name}", os.path.join(_model_dir, f"{_name}.py"))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE}.{_name}"] = _module
    _spec.loader.exec_module(_module)
    _loaded[_name] = _module

mtp_checkpoint_mappings = _loaded["layout"].mtp_checkpoint_mappings
concat_draft_inputs = _loaded["ops"].concat_draft_inputs

MTP_SOURCE_FILE = os.path.join(_model_dir, "mtp.py")

# Every ``mtp.*`` key of the published Qwen3.6-35B-A3B checkpoint, transcribed. The transcription is
# what lets this file run in CI with no checkpoint present; the test below compares it against the real
# index whenever QWEN3_5_MOE_CHECKPOINT_INDEX is set, so a wrong transcription cannot survive a run on a
# machine that has the model. Nineteen keys: two pre-norms, fc, a final norm, and one decoder layer.
CHECKPOINT_KEYS = {
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
}


def sources_of(mappings: dict) -> list:
    """Flatten the mapping's values; a list value means several sources feed one destination."""
    flat = []
    for value in mappings.values():
        flat.extend(value if isinstance(value, list) else [value])
    return flat


# The map, association by association. An earlier version of this file asserted only that the SET of
# source keys equalled the checkpoint's, which permits every permutation: swapping the two pre-norms,
# reordering q/k/v inside the fused destination, or sending `o_proj` to the qkvg parameter all keep the
# set identical and load garbage. Every one of those is silent -- the shapes are the same or the loader
# is handed the wrong three tensors in the right total size. So the expected map is written out.
EXPECTED_MAP = {
    "fc.weight": "mtp.fc.weight",
    "norm.weight": "mtp.norm.weight",
    "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
    "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
    "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
    # Order matters and is q, k, v: the attention fuses them into one parameter in that order, and the
    # q rows also carry the output gate.
    "self_attn.qkvg_proj_weight": [
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
    ],
    "self_attn.o_proj_weight": "mtp.layers.0.self_attn.o_proj.weight",
    "self_attn.q_norm_weight": "mtp.layers.0.self_attn.q_norm.weight",
    "self_attn.k_norm_weight": "mtp.layers.0.self_attn.k_norm.weight",
    "mlp.router_weight": "mtp.layers.0.mlp.gate.weight",
    "mlp.gate_up_proj_weight": "mtp.layers.0.mlp.experts.gate_up_proj",
    "mlp.down_proj_weight": "mtp.layers.0.mlp.experts.down_proj",
    "mlp.shared_gate_proj_weight": "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mlp.shared_up_proj_weight": "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mlp.shared_down_proj_weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate_weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
}


def test_the_map_is_exactly_this_association_of_destinations_to_sources():
    """The whole map, not its key set. A permutation is the defect this is here to refuse."""
    assert mtp_checkpoint_mappings() == EXPECTED_MAP


def test_the_head_consumes_every_mtp_key_and_invents_none():
    """Both directions against the checkpoint. One direction alone lets a real defect through.

    Only "every source I ask for exists" misses a key the checkpoint has and the head never loads —
    a parameter left at its initial value, which for this architecture's norms is zero. Only "every key
    is consumed" misses a source key that is a typo of a real one, which fails at load with a message
    about a missing tensor if you are lucky and about a shape if you are not.
    """
    sources = sources_of(mtp_checkpoint_mappings())
    assert set(sources) == CHECKPOINT_KEYS, {
        "asked for but not in the checkpoint": sorted(set(sources) - CHECKPOINT_KEYS),
        "in the checkpoint but never loaded": sorted(CHECKPOINT_KEYS - set(sources)),
    }
    # No key twice: two destinations reading one source is how a copy-paste of the q/k/v block ends up
    # loading the same projection into two places.
    assert len(sources) == len(set(sources)), sorted(
        key for key in set(sources) if sources.count(key) > 1)


def test_the_transcribed_key_list_matches_the_real_checkpoint():
    """The transcription above, against the actual index. Skipped when the model is not present."""
    index_path = os.environ.get("QWEN3_5_MOE_CHECKPOINT_INDEX")
    if not index_path:
        pytest.skip("set QWEN3_5_MOE_CHECKPOINT_INDEX to a model.safetensors.index.json")
    with open(index_path) as handle:
        keys = set(json.load(handle)["weight_map"])
    actual = {key for key in keys if key.startswith("mtp.")}
    if not actual:
        # Not a skip: a checkpoint pointed at this test is expected to be the one the port targets, and
        # "no MTP keys" is far more likely to be the wrong file than a legitimately headless checkpoint.
        raise AssertionError(
            f"{index_path} has no mtp.* keys, so it is not a checkpoint of this architecture. "
            f"Point QWEN3_5_MOE_CHECKPOINT_INDEX at one, or unset it to skip this check.")
    assert actual == CHECKPOINT_KEYS, {
        "transcribed but absent": sorted(CHECKPOINT_KEYS - actual),
        "present but not transcribed": sorted(actual - CHECKPOINT_KEYS),
    }


def test_the_source_prefix_is_a_parameter_rather_than_a_hardcoded_mtp():
    """A checkpoint that nests the head under another prefix has to be loadable without an edit.

    Asserting the prefix alone would pass an implementation that gets the prefix right and every suffix
    wrong, so the whole map is compared against the default map with the prefix substituted.
    """
    def reprefixed(value):
        if isinstance(value, list):
            return [reprefixed(item) for item in value]
        assert value.startswith("mtp."), value
        return "model." + value

    expected = {destination: reprefixed(source) for destination, source in EXPECTED_MAP.items()}
    assert mtp_checkpoint_mappings(source="model.mtp") == expected


def test_every_destination_names_something_the_head_actually_declares():
    """Guard against renaming a module and leaving the mapping pointing at the old name.

    The loader would then fail at load time on the device, after the compile has been paid for. This is
    a textual check on the shipped source rather than an import, because the head cannot be imported on
    CPU: it asserts the first component of each destination path is assigned in ``__init__``.
    """
    with open(MTP_SOURCE_FILE) as handle:
        source = handle.read()
    for destination in mtp_checkpoint_mappings():
        attribute = destination.split(".")[0]
        assert f"self.{attribute} = " in source, (
            f"the mapping loads into self.{attribute}, which mtp.py does not assign")


def test_the_embedding_is_the_first_half_of_the_concatenation():
    """The one step transcribed from vLLM rather than inferred, pinned by a projection that can tell.

    ``fc`` is built so its first half is the identity and its second half is zero. The reference order
    (embedding first) therefore returns the embedding; the swapped order would return the hidden state.
    Both have the same shape, so nothing but a value check distinguishes them.
    """
    hidden = 8
    torch.manual_seed(0)
    embeddings = torch.randn(2, 3, hidden)
    last_hidden_state = torch.randn(2, 3, hidden)

    fused = concat_draft_inputs(embeddings, last_hidden_state)
    assert fused.shape == (2, 3, 2 * hidden)

    reads_first_half = torch.cat([torch.eye(hidden), torch.zeros(hidden, hidden)], dim=-1)
    projected = fused @ reads_first_half.t()
    torch.testing.assert_close(projected, embeddings)
    assert not torch.allclose(projected, last_hidden_state)


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    """A hidden state of the wrong length must not concatenate into a plausible width."""
    with pytest.raises(ValueError, match="same shape"):
        concat_draft_inputs(torch.zeros(2, 3, 8), torch.zeros(2, 3, 4))


# ---------------------------------------------------------------------------
# The head's own forward, with its framework dependencies stubbed
# ---------------------------------------------------------------------------
#
# Everything above tests the map and one helper. None of it touches `mtp.py`'s forward, so a head that
# calls `torch.cat` directly in the other order, applies the wrong pre-norm to each input, or drops the
# residual passes all of it. The composition is the part transcribed from another implementation, which
# makes it exactly the part worth executing.
#
# The head cannot be imported normally on CPU: it builds the plugin's attention and MoE, which reach for
# the platform. So its four dependencies are replaced with spies and the SHIPPED file is executed as
# written -- no source rewriting, so what runs is what ships.
#
# The stubs go in under the framework's real module names, which is how another file in this directory
# broke the whole suite. Here they are installed and then removed: `sys.modules` is snapshotted and
# restored, and `test_loading_the_head_leaves_sys_modules_as_it_found_it` asserts that it worked.


class RecordingNorm(torch.nn.Module):
    """Stands in for the RMS norm, and is identifiable in the output it produces.

    Each instance multiplies by its own constant, so a swapped pair of norms changes the numbers rather
    than only the call order -- the assertions can then be about values instead of about a call log.
    """

    def __init__(self, hidden, eps, dtype, factor):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((hidden,), float(factor)))
        self.eps = eps
        self.factor = factor
        self.inputs: list = []

    def forward(self, x):
        self.inputs.append(x)
        return x * self.factor


class RecordingLinear(torch.nn.Module):
    """Stands in for `ColumnParallelLinear`, recording what it was handed."""

    def __init__(self, in_features, out_features, bias=False, dtype=None, gather_output=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.out_features_per_rank, self.tp_size = out_features, 1
        self.bias_requested, self.gather_output = bias, gather_output
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.inputs: list = []

    def forward(self, x):
        self.inputs.append(x)
        # A projection whose output is recognisable: the two halves, added, so the assertion can name
        # which half went where.
        return x[..., :self.out_features] + 10.0 * x[..., self.out_features:]


class RecordingAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_heads_per_rank, self.head_dim = 2, 8
        self.dtype, self.window_size = torch.bfloat16, None
        self.inputs: list = []
        self.bound: list = []

    def forward(self, hidden, positions, cos, sin, attn_metadata, is_prefill):
        self.inputs.append(hidden)
        return hidden * 100.0

    def bind_caches(self, k_cache, v_cache):
        self.bound.append((k_cache, v_cache))


class RecordingMoE(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.calls: list = []

    def forward(self, hidden, norm_weight, is_prefill, valid_mask=None):
        self.calls.append({"hidden": hidden, "norm_weight": norm_weight,
                           "is_prefill": is_prefill, "valid_mask": valid_mask})
        return hidden * 1000.0


def load_head_class():
    """Execute the shipped `mtp.py` with its four framework dependencies replaced by the spies above.

    Returns the class, and leaves `sys.modules` exactly as it was found. The stub names have to be the
    framework's real ones because the shipped file imports them by those names, and the shipped file is
    what is worth testing -- so the names are borrowed and given back.
    """
    factors = iter([2.0, 3.0, 5.0, 7.0, 11.0])   # one per norm, in construction order

    nn_stub = types.ModuleType("vllm_neuron.nn")
    nn_stub.ColumnParallelLinear = RecordingLinear
    loader_stub = types.ModuleType("vllm_neuron.utils.weight_loader")
    loader_stub.set_weight_loader = lambda *a, **k: None
    loader_stub.sharding_weight_loader = lambda *a, **k: None

    model_stub = types.ModuleType(f"{_PACKAGE}.model_bf16")
    model_stub.Qwen3_5MoeRMSNorm = lambda hidden, eps, dtype: RecordingNorm(
        hidden, eps, dtype, next(factors))
    model_stub.Qwen3_5MoeAttention = RecordingAttention
    model_stub.Qwen3_5MoeMoE = RecordingMoE

    installed = {
        "vllm_neuron": types.ModuleType("vllm_neuron"),
        "vllm_neuron.nn": nn_stub,
        "vllm_neuron.utils": types.ModuleType("vllm_neuron.utils"),
        "vllm_neuron.utils.weight_loader": loader_stub,
        f"{_PACKAGE}.model_bf16": model_stub,
    }
    snapshot = dict(sys.modules)
    try:
        sys.modules.update(installed)
        spec = importlib.util.spec_from_file_location(
            f"{_PACKAGE}.mtp", os.path.join(_model_dir, "mtp.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PACKAGE}.mtp"] = module
        spec.loader.exec_module(module)
        return module.Qwen3_5MoeMultiTokenPredictor
    finally:
        sys.modules.clear()
        sys.modules.update(snapshot)


class HeadConfig:
    """The fields `mtp.py` reads, and nothing else."""
    hidden_size = 16
    rms_norm_eps = 1e-6
    torch_dtype = torch.float32
    num_hidden_layers = 40


def build_head(layer_idx=40):
    return load_head_class()(HeadConfig(), layer_idx)


def test_loading_the_head_leaves_sys_modules_as_it_found_it():
    """The stubs wear the framework's real names, so giving them back is part of the contract.

    Another file in this directory installed such stubs permanently, and because pytest imports every
    test module during collection, the whole session's real imports resolved to them.
    """
    before = dict(sys.modules)
    build_head()
    assert set(sys.modules) == set(before)
    assert all(sys.modules[name] is module for name, module in before.items())


def test_the_head_normalises_each_input_with_its_own_norm_and_concatenates_embedding_first():
    """The composition, executed. This is the assertion the helper-level test could not make.

    The two pre-norms multiply by different constants and `fc` weights its second half by ten, so the
    fused value names which half held which input. Swapping the pre-norms, swapping the halves, or
    reusing one norm for both inputs all change this number.
    """
    head = build_head()
    embeddings = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
    hidden = torch.ones(2, 3, 16)

    head(embeddings, hidden, positions=None, cos=None, sin=None, attn_metadata=None, is_prefill=True)

    embedding_factor = head.pre_fc_norm_embedding.factor
    hidden_factor = head.pre_fc_norm_hidden.factor
    assert embedding_factor != hidden_factor, "the spies cannot distinguish the two norms"

    # Each pre-norm saw its own input, and only its own.
    assert len(head.pre_fc_norm_embedding.inputs) == 1
    assert torch.equal(head.pre_fc_norm_embedding.inputs[0], embeddings)
    assert torch.equal(head.pre_fc_norm_hidden.inputs[0], hidden)

    # fc was handed [normed embedding | normed hidden], in that order.
    fused_input = head.fc.inputs[0]
    assert fused_input.shape == (2, 3, 32)
    torch.testing.assert_close(fused_input[..., :16], embeddings * embedding_factor)
    torch.testing.assert_close(fused_input[..., 16:], hidden * hidden_factor)


def test_the_residual_is_the_projection_and_the_moe_is_handed_the_post_attention_weight():
    """Residual placement and the fused-norm handover, which no shape would catch."""
    head = build_head()
    embeddings = torch.randn(1, 2, 16)
    hidden = torch.randn(1, 2, 16)

    out = head(embeddings, hidden, positions=None, cos=None, sin=None, attn_metadata=None,
               is_prefill=False, valid_mask="the-mask")

    fused = head.fc(head.fc.inputs[0])              # the spy is deterministic, so this reproduces it
    # The attention was applied to the INPUT norm of the projection, not to the projection itself.
    torch.testing.assert_close(head.self_attn.inputs[0], fused * head.input_layernorm.factor)
    # ...and the residual added to it is the projection.
    after_attention = fused + head.self_attn.inputs[0] * 100.0
    torch.testing.assert_close(head.mlp.calls[0]["hidden"], after_attention)
    # The MoE fuses the post-attention norm, so it must receive that norm's WEIGHT, not a normed tensor.
    assert head.mlp.calls[0]["norm_weight"] is head.post_attention_layernorm.weight
    assert head.mlp.calls[0]["is_prefill"] is False
    assert head.mlp.calls[0]["valid_mask"] == "the-mask"
    # And the output is the final norm of the second residual.
    expected = (after_attention + after_attention * 1000.0) * head.norm.factor
    torch.testing.assert_close(out, expected)


def test_the_draft_layer_index_must_be_past_the_main_model():
    """A draft layer inside the main model's range shares a decoder layer's KV cache, silently."""
    with pytest.raises(ValueError, match="past the main model"):
        build_head(layer_idx=HeadConfig.num_hidden_layers - 1)
    head = build_head(layer_idx=HeadConfig.num_hidden_layers)
    assert head.self_attn.layer_idx == HeadConfig.num_hidden_layers
    assert head.mlp.layer_idx == HeadConfig.num_hidden_layers


def test_the_kv_spec_describes_the_head_s_own_attention_and_the_binding_reaches_it():
    """The tuple the model turns into a real spec, and the cache handover."""
    head = build_head(layer_idx=41)
    assert head.kv_layer_spec() == (41, head.self_attn.num_key_value_heads_per_rank,
                                    head.self_attn.head_dim, head.self_attn.dtype,
                                    head.self_attn.window_size)
    head.bind_kv_cache_entry("k", "v")
    assert head.self_attn.bound == [("k", "v")]


def test_the_projection_is_two_hidden_to_hidden_without_a_bias():
    """`fc` consumes the concatenation and produces one hidden state; a bias would be an extra tensor
    the checkpoint does not carry."""
    head = build_head()
    assert (head.fc.in_features, head.fc.out_features) == (2 * HeadConfig.hidden_size,
                                                           HeadConfig.hidden_size)
    assert head.fc.bias_requested is False
    assert head.fc.gather_output is True
