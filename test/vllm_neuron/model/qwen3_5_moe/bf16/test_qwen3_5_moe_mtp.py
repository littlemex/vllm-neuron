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
    """Stands in for the RMS norm: records what it was built with and what it was handed.

    **Affine, not linear.** An earlier version multiplied by a constant, which made a whole class of defect
    invisible: with a linear stand-in, `norm(a) + norm(b)` equals `norm(a + b)` to the bit, so a head that
    normalises each branch separately instead of normalising the sum passed every assertion. The real RMS
    norm is not additive, and neither is this. The `+ offset` is what restores the distinction.

    The constructor arguments are recorded because nothing else would notice them being swapped: this stub
    uses neither the epsilon nor the dtype, so a head that passes them in the wrong order builds fine.
    """

    def __init__(self, hidden, eps, dtype, factor):
        super().__init__()
        self.built_with = {"hidden": hidden, "eps": eps, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.full((hidden,), float(factor)))
        self.factor = factor
        self.offset = factor / 2.0
        self.inputs: list = []

    def forward(self, x):
        self.inputs.append(x)
        return x * self.factor + self.offset


class RecordingLinear(torch.nn.Module):
    """Stands in for `ColumnParallelLinear`, recording how it was built and what it was handed."""

    def __init__(self, in_features, out_features, bias=False, dtype=None, gather_output=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.out_features_per_rank, self.tp_size = out_features, 1
        self.built_with = {"bias": bias, "dtype": dtype, "gather_output": gather_output}
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.inputs: list = []

    def forward(self, x):
        self.inputs.append(x)
        return self.applied(x)

    def applied(self, x):
        """The projection as a pure function, so a test can predict it without appending to the log."""
        # Recognisable and NOT symmetric in its two halves: the assertion can name which half went where.
        return x[..., :self.out_features] + 10.0 * x[..., self.out_features:]


class RecordingAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_heads_per_rank, self.head_dim = 2, 8
        self.dtype, self.window_size = torch.bfloat16, None
        self.calls: list = []
        self.bound: list = []

    def forward(self, hidden, positions, cos, sin, attn_metadata, is_prefill):
        # Every argument is recorded by NAME. Passing them all as None -- which the first version of these
        # tests did -- makes swapping cos with sin, or positions with the metadata, invisible.
        self.calls.append({"hidden": hidden, "positions": positions, "cos": cos, "sin": sin,
                           "attn_metadata": attn_metadata, "is_prefill": is_prefill})
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

    Returns `(class, recorded)` where `recorded` collects what the stubbed loader was asked to do, and
    leaves `sys.modules` exactly as it was found. The stub names have to be the framework's real ones
    because the shipped file imports them by those names, and the shipped file is what is worth testing --
    so the names are borrowed and given back.

    What runs is the shipped file, unmodified. What it is BOUND to is not: the real imports are never
    resolved here, so nothing in this file says the shipped import lines name modules that exist. Rung 2
    is what settles that.
    """
    factors = iter([2.0, 3.0, 5.0, 7.0, 11.0])   # one per norm, in construction order
    recorded: dict = {"sharding": [], "set_weight_loader": []}

    nn_stub = types.ModuleType("vllm_neuron.nn")
    nn_stub.ColumnParallelLinear = RecordingLinear
    loader_stub = types.ModuleType("vllm_neuron.utils.weight_loader")
    loader_stub.set_weight_loader = lambda *a, **k: recorded["set_weight_loader"].append((a, k))
    # The sharding arguments are recorded because a wrong shard dimension or a transposed-storage flag is
    # invisible until TP > 1, where the output is slightly wrong and gets blamed on precision.
    loader_stub.sharding_weight_loader = lambda *a, **k: recorded["sharding"].append((a, k))

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
    # Only the keys this function touches are restored, and only if they were absent before. Clearing
    # sys.modules and reinstating a snapshot -- the previous approach -- EVICTS anything imported during
    # the window, so the returned class kept a reference to a module the next import would execute a
    # second time, giving two classes with one name and no assertion able to see it.
    #
    # Not thread-safe, and cannot be: the framework's names are globally visible while this runs. Nothing
    # else in this suite imports concurrently.
    previously = {name: sys.modules.get(name) for name in installed}
    added: list = []
    try:
        sys.modules.update(installed)
        target = f"{_PACKAGE}.mtp"
        spec = importlib.util.spec_from_file_location(target, os.path.join(_model_dir, "mtp.py"))
        module = importlib.util.module_from_spec(spec)
        before_exec = set(sys.modules)
        sys.modules[target] = module
        spec.loader.exec_module(module)
        added = [name for name in set(sys.modules) - before_exec if name.startswith(_PACKAGE)]
        return module.Qwen3_5MoeMultiTokenPredictor, recorded
    finally:
        for name, original in previously.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for name in added:
            sys.modules.pop(name, None)


class HeadConfig:
    """The fields `mtp.py` reads, and nothing else."""
    hidden_size = 16
    rms_norm_eps = 1e-6
    torch_dtype = torch.float32
    num_hidden_layers = 40


def build_head(layer_idx=40):
    head_class, recorded = load_head_class()
    head = head_class(HeadConfig(), layer_idx)
    head.recorded = recorded
    return head


def test_loading_the_head_leaves_the_framework_s_names_alone():
    """The stubs wear the framework's real names, so giving them back is part of the contract.

    Asserting "the same as before" is not enough on its own, because pytest imports every test module
    during collection: if something had polluted these names before this test ran, "restored to what I
    found" would bless the pollution. So the names are asserted ABSENT (or real) rather than unchanged.
    """
    stub_names = ("vllm_neuron", "vllm_neuron.nn", "vllm_neuron.utils",
                  "vllm_neuron.utils.weight_loader")
    before = dict(sys.modules)
    build_head()
    assert set(sys.modules) == set(before)
    assert all(sys.modules[name] is module for name, module in before.items())
    for name in stub_names:
        residue = sys.modules.get(name)
        assert residue is None or getattr(residue, "__file__", None), (
            f"{name} is left in sys.modules as a module with no file, i.e. a stub")


def test_the_head_normalises_each_input_with_its_own_norm_and_concatenates_embedding_first():
    """The composition, executed. This is the assertion the helper-level test could not make.

    The two pre-norms are affine with different constants and `fc` weights its second half by ten, so the
    fused value names which half held which input. Swapping the pre-norms, swapping the halves, or reusing
    one norm for both inputs all change this number.
    """
    head = build_head()
    embeddings = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
    hidden = torch.ones(2, 3, 16)

    head(embeddings, hidden, positions=None, cos=None, sin=None, attn_metadata=None, is_prefill=True)

    embedding_norm, hidden_norm = head.pre_fc_norm_embedding, head.pre_fc_norm_hidden
    assert embedding_norm.factor != hidden_norm.factor, "the spies cannot distinguish the two norms"

    # Each pre-norm saw its own input, and only its own.
    assert len(embedding_norm.inputs) == 1
    assert torch.equal(embedding_norm.inputs[0], embeddings)
    assert torch.equal(hidden_norm.inputs[0], hidden)

    # fc was handed [normed embedding | normed hidden], in that order.
    fused_input = head.fc.inputs[0]
    assert fused_input.shape == (2, 3, 32)
    torch.testing.assert_close(fused_input[..., :16], embedding_norm(embeddings.clone()))
    torch.testing.assert_close(fused_input[..., 16:], hidden_norm(hidden.clone()))


def test_every_norm_is_built_with_the_config_s_epsilon_and_dtype():
    """Nothing else would notice these being swapped: the head passes three positional arguments to a
    constructor, and a stand-in that ignores two of them builds happily either way. The real norm reads
    the epsilon."""
    head = build_head()
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm", "input_layernorm",
                 "post_attention_layernorm"):
        built = getattr(head, name).built_with
        assert built == {"hidden": HeadConfig.hidden_size, "eps": HeadConfig.rms_norm_eps,
                         "dtype": HeadConfig.torch_dtype}, f"{name} was built with {built}"


def test_the_projection_s_weight_loader_shards_the_output_dimension():
    """A wrong shard dimension or a transposed-storage flag is invisible until TP > 1, where the output is
    slightly wrong and gets blamed on precision. `fc` maps 2H -> H, so the dimension that may be split
    across ranks is the OUTPUT one, dimension 0, and the checkpoint's storage is not transposed."""
    head = build_head()
    assert len(head.recorded["sharding"]) == 1, head.recorded["sharding"]
    _, kwargs = head.recorded["sharding"][0]
    assert kwargs["shard_dim"] == 0, f"fc is sharded on dimension {kwargs['shard_dim']}"
    assert kwargs["is_storage_transposed"] is False
    assert kwargs["shard_size"] == head.fc.out_features_per_rank
    assert kwargs["num_shards"] == head.fc.tp_size
    # And the loader was actually attached to the parameter, not merely constructed.
    assert len(head.recorded["set_weight_loader"]) == 1
    attached, _ = head.recorded["set_weight_loader"][0]
    assert attached[0] is head.fc.weight


def test_the_residual_is_the_projection_and_the_moe_is_handed_the_post_attention_weight():
    """Residual placement, the fused-norm handover, and the final norm's INPUT.

    The last of those matters because the spies used to be linear: with `norm(x) = kx`, a head that
    normalised each branch separately instead of the sum was bit-identical to the correct one. The norms
    are affine now, and the final norm's recorded input is asserted directly.
    """
    head = build_head()
    embeddings = torch.randn(1, 2, 16)
    hidden = torch.randn(1, 2, 16)
    positions, cos, sin, metadata = "positions", "cos", "sin", "metadata"

    out = head(embeddings, hidden, positions=positions, cos=cos, sin=sin, attn_metadata=metadata,
               is_prefill=False, valid_mask="the-mask")

    # Predicted without calling the spy again, so reading the log does not extend it.
    fused = head.fc.applied(head.fc.inputs[0])
    call = head.self_attn.calls[0]
    # Distinct sentinels, so a swapped pair of arguments is visible. All-None hid this entirely.
    assert (call["positions"], call["cos"], call["sin"], call["attn_metadata"]) == (
        positions, cos, sin, metadata)
    assert call["is_prefill"] is False
    # The attention was applied to the INPUT norm of the projection, not to the projection itself.
    torch.testing.assert_close(call["hidden"], head.input_layernorm(fused.clone()))
    # ...and the residual added to it is the projection.
    after_attention = fused + call["hidden"] * 100.0
    torch.testing.assert_close(head.mlp.calls[0]["hidden"], after_attention)
    # The MoE fuses the post-attention norm, so it must receive that norm's WEIGHT, not a normed tensor.
    assert head.mlp.calls[0]["norm_weight"] is head.post_attention_layernorm.weight
    assert head.mlp.calls[0]["is_prefill"] is False
    assert head.mlp.calls[0]["valid_mask"] == "the-mask"
    # The final norm was applied ONCE, to the second residual -- not to each branch of it.
    second_residual = after_attention + after_attention * 1000.0
    assert len(head.norm.inputs) == 1
    torch.testing.assert_close(head.norm.inputs[0], second_residual)
    torch.testing.assert_close(out, head.norm(second_residual.clone()))


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
    """`fc` consumes the concatenation and produces one hidden state; a bias would be an extra tensor the
    checkpoint does not carry, and an ungathered output would hand the next layer a shard."""
    head = build_head()
    assert (head.fc.in_features, head.fc.out_features) == (2 * HeadConfig.hidden_size,
                                                           HeadConfig.hidden_size)
    assert head.fc.built_with == {"bias": False, "dtype": HeadConfig.torch_dtype,
                                  "gather_output": True}


# The destinations that belong to the SHARED attention and MoE classes rather than to this head. Their
# leaf names are those classes' contract, and a stub cannot check them: this file replaces those classes,
# so asserting their parameter names here would only assert the stub. They are checked where the classes
# live, and for real by the weight load at rung 2.
DELEGATED_PREFIXES = ("self_attn.", "mlp.")


def test_every_mapping_destination_resolves_on_the_built_head():
    """The real version of the textual check: walk each destination path on an instance.

    Grepping the source for `self.<attr> = ` is satisfied by a comment, a docstring, an assignment in
    another method, or `self.attr = None`, and it only ever looked at the FIRST component of the path. Now
    that the head can be built, the components it owns can be resolved for real.
    """
    head = build_head()
    checked = 0
    for destination in mtp_checkpoint_mappings():
        components = destination.split(".")
        if destination.startswith(DELEGATED_PREFIXES):
            # Only the submodule, which is what this head is responsible for owning.
            assert hasattr(head, components[0]), (
                f"the mapping loads into {destination}, and the head has no {components[0]}")
            continue
        target = head
        for component in components:
            assert hasattr(target, component), (
                f"the mapping loads into {destination}, and {component} does not exist on the head")
            target = getattr(target, component)
        assert target is not None, f"{destination} exists but is None, so nothing can be loaded into it"
        checked += 1
    # Guard against the loop degenerating: if every destination became "delegated", this test would pass
    # while resolving nothing. Six destinations are the head's own -- fc, three norms, and the two the
    # decoder layer wraps.
    assert checked == 6, f"{checked} destinations were resolved, not the expected 6"
