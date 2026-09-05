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
rmsnorm = _loaded["ops"].rmsnorm

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


def test_the_head_consumes_every_mtp_key_and_invents_none():
    """Both directions. One direction alone lets a real defect through.

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
        pytest.skip("this checkpoint has no multi-token prediction head")
    assert actual == CHECKPOINT_KEYS, {
        "transcribed but absent": sorted(CHECKPOINT_KEYS - actual),
        "present but not transcribed": sorted(actual - CHECKPOINT_KEYS),
    }


def test_the_source_prefix_is_a_parameter_rather_than_a_hardcoded_mtp():
    """A checkpoint that nests the head under another prefix has to be loadable without an edit."""
    mappings = mtp_checkpoint_mappings(source="model.mtp")
    assert all(key.startswith("model.mtp.") for key in sources_of(mappings))
    # The destinations are the head's own attribute paths and must not move with the source.
    assert set(mappings) == set(mtp_checkpoint_mappings())


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


def test_the_two_pre_norms_are_not_interchangeable():
    """Why the head carries two norm weights for two tensors of the same width.

    If the same weight were used for both, the head would still run. This pins that the norms are
    distinct functions of their weight, so swapping which weight goes with which input changes the
    result — which is what makes the mapping's two separate keys load-order sensitive rather than
    cosmetic.
    """
    hidden = 8
    torch.manual_seed(1)
    x = torch.randn(2, hidden)
    embedding_scale = torch.randn(hidden) * 0.1
    hidden_scale = torch.randn(hidden) * 0.1
    assert not torch.allclose(rmsnorm(x, embedding_scale, 1e-6), rmsnorm(x, hidden_scale, 1e-6))


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    """A hidden state of the wrong length must not concatenate into a plausible width."""
    with pytest.raises(ValueError, match="same shape"):
        concat_draft_inputs(torch.zeros(2, 3, 8), torch.zeros(2, 3, 4))
