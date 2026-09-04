# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE configuration (text backbone).

Language backbone of ``Qwen/Qwen3.6-35B-A3B`` — whose HF ``architectures`` is
``Qwen3_5MoeForConditionalGeneration`` and whose ``model_type`` is ``qwen3_5_moe``. Only the
``text_config`` half of the multimodal wrapper is read: the vision tower and the MTP head are not
implemented. The field defaults below are the published checkpoint's values; the architecture they
describe is laid out in README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from transformers import PretrainedConfig

if TYPE_CHECKING:      # keeps this module free of Neuron imports; it is only an annotation
    from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

# layer_types legend (HF strings, used verbatim so a config change cannot silently remap a layer).
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


@dataclass
class Qwen3_5MoeVisionConfig:
    """Configuration for the vision tower, read from the checkpoint's ``vision_config``.

    Only the fields the tower needs are here. The defaults are the published
    ``Qwen/Qwen3.6-35B-A3B`` values, and ``out_hidden_size`` matching the text ``hidden_size`` is what
    lets the patch merger's output enter the token embedding stream directly, with no extra projection.
    """

    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    # Empty in this checkpoint, which removes the reference implementation's per-layer injection path.
    # A non-empty list means intermediate vision layers feed the text layers and would need that path.
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    torch_dtype: torch.dtype = torch.bfloat16
    # The runner's VisionNeuronConfig: the encoder's block size and token buckets. A deployment choice
    # rather than a checkpoint fact, so it is attached here and not read from the config dict.
    neuron_config: VisionNeuronConfig | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError(
                f"vision hidden_size={self.hidden_size} is not divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.deepstack_visual_indexes:
            raise NotImplementedError(
                "Qwen3.5-MoE on Neuron does not implement deepstack visual injection; this "
                f"checkpoint asks for layers {self.deepstack_visual_indexes}. The published "
                "Qwen3.6-35B-A3B checkpoint has an empty list, so this is a different checkpoint "
                "than the one this implementation was written against."
            )


@dataclass
class Qwen3_5MoeConfig:
    """Configuration for the Qwen3.5-MoE text backbone."""

    # -- Model architecture -------------------------------------------------
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    # Per-layer type; length must equal num_hidden_layers. Defaults are filled from
    # full_attention_interval in __post_init__ when the checkpoint omits the explicit list.
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    # -- Full attention (GQA + output gate + partial rotary) ----------------
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_bias: bool = False
    # q_proj emits query and an output gate side by side; the gate multiplies the attention output.
    attn_output_gate: bool = True
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    # MRoPE metadata. Text-only input makes the three axes identical, which collapses interleaved
    # MRoPE to plain partial rotary; kept here so a future vision path has the real values.
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    mrope_interleaved: bool = True

    # -- Gated DeltaNet (linear attention) ----------------------------------
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # -- MoE ----------------------------------------------------------------
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512

    # -- Sequence / tokens --------------------------------------------------
    max_position_embeddings: int = 262144
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    pad_token_id: int | None = None

    # -- Framework config (not model-specific) ------------------------------
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if not self.layer_types:
            # Reconstruct the published pattern: every full_attention_interval-th layer is full
            # attention (index 3, 7, 11, ... for interval 4), the rest are linear attention.
            interval = self.full_attention_interval
            if interval < 1:
                raise ValueError(f"full_attention_interval must be >= 1, got {interval}")
            self.layer_types = [
                FULL_ATTENTION if (i + 1) % interval == 0 else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(self.layer_types)} != num_hidden_layers "
                f"{self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(
                f"Qwen3.5-MoE on Neuron implements {LINEAR_ATTENTION!r} and {FULL_ATTENTION!r} "
                f"layers only; got unsupported layer types {sorted(unknown)!r}."
            )
        if not any(t == FULL_ATTENTION for t in self.layer_types):
            # The prefill/decode phase and the real/pad token mask both come from an attention layer's
            # metadata, so at least one such layer has to exist.
            raise ValueError("Qwen3.5-MoE on Neuron requires at least one full_attention layer.")
        if not self.attn_output_gate:
            raise ValueError(
                "Qwen3.5-MoE on Neuron assumes attn_output_gate=True (q_proj emits query and gate "
                "side by side). A checkpoint without the gate has a different q_proj width."
            )
        if self.attention_bias:
            raise ValueError("Qwen3.5-MoE on Neuron assumes attention_bias=False.")
        if self.hidden_act != "silu":
            raise ValueError(
                f"Qwen3.5-MoE on Neuron implements SiLU-gated experts and a SiLU conv activation; "
                f"got hidden_act={self.hidden_act!r}."
            )
        if self.torch_dtype != torch.bfloat16:
            # The implementation is named and sized for bf16. A checkpoint declaring fp32 would load
            # fp32 parameters and roughly double the 72 GB footprint, exhausting HBM at TP=4 — with
            # nothing in the quantization field to reveal it.
            raise ValueError(
                f"Qwen3.5-MoE on Neuron implements bf16 only; the checkpoint declares "
                f"{self.torch_dtype}. Convert the checkpoint or wait for the FP8/NVFP4 paths."
            )
        if self.linear_conv_kernel_dim < 2:
            # The conv history carried between graphs is kernel_size - 1 wide; at kernel 1 that is a
            # zero-width buffer and the `-(kernel - 1)` slices used to gather it would select the whole
            # sequence instead of nothing.
            raise ValueError(
                f"Qwen3.5-MoE on Neuron requires linear_conv_kernel_dim >= 2; got "
                f"{self.linear_conv_kernel_dim}."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be a multiple of "
                f"linear_num_key_heads ({self.linear_num_key_heads}): the GDN query/key heads are "
                "repeat-interleaved up to the value head count."
            )
        # Rotary width: partial_rotary_factor of head_dim, and rotate_half needs it even.
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.rotary_dim <= 0 or self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim = head_dim * partial_rotary_factor = {self.rotary_dim} must be a "
                "positive even number (rotate_half splits it in half)."
            )

    def layer_type(self, index: int) -> str:
        return self.layer_types[index]

    def layer_type_counts(self) -> dict:
        return {
            "linear_attention": self.layer_types.count(LINEAR_ATTENTION),
            "full_attention": self.layer_types.count(FULL_ATTENTION),
        }

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None):
        """Build from a HF config (dict, path, or PretrainedConfig) plus a NeuronConfig.

        The published checkpoint is a multimodal wrapper: the decoder fields live under
        ``text_config`` and the rotary fields under ``text_config.rope_parameters``. A text-only
        checkpoint would carry the same fields at the top level, so both are handled.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as handle:
                config_dict = json.load(handle)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = dict(hf_config)

        # Unwrap the multimodal wrapper. Recognise it by a decoder-only field so a top-level
        # text checkpoint is left alone.
        nested = config_dict.get("text_config")
        if isinstance(nested, dict) and "linear_num_value_heads" in nested:
            config_dict = nested

        field_names = {name for name in cls.__dataclass_fields__ if name != "neuron_config"}
        # An explicitly serialised None is treated as absent. transformers emits one for some fields,
        # and a present-but-None entry here shadows the fallbacks below (the dtype, for one, is read
        # from ``dtype`` when ``torch_dtype`` is missing).
        filtered = {k: v for k, v in config_dict.items() if k in field_names and v is not None}

        # Rotary parameters live in a nested dict on this architecture; lift the ones we use. Read
        # them explicitly rather than via the dataclass defaults so a retuned checkpoint is honoured.
        rope = config_dict.get("rope_parameters") or {}
        for source, target in (("rope_theta", "rope_theta"),
                               ("partial_rotary_factor", "partial_rotary_factor"),
                               ("mrope_section", "mrope_section"),
                               ("mrope_interleaved", "mrope_interleaved")):
            if target not in filtered and source in rope:
                filtered[target] = rope[source]
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"Qwen3.5-MoE on Neuron implements the default rotary type only; got "
                f"rope_type={rope_type!r} (scaled/dynamic rotary is not wired up)."
            )

        # transformers may serialise some fields under attribute_map aliases, which would leave them
        # absent here and silently fall back to the dataclass default (the full 40-layer model).
        # Recover them from the config object, which resolves the alias.
        if isinstance(hf_config, PretrainedConfig):
            source_config = getattr(hf_config, "text_config", None) or hf_config  # lint-port: ok HF config shape varies; a text-only checkpoint has no text_config
            for name in field_names:
                if name in filtered:
                    continue
                value = getattr(source_config, name, None)
                if value is not None:
                    filtered[name] = value

        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])
        # The checkpoint spells the parameter dtype ``dtype``; ``torch_dtype`` is the older alias.
        if "torch_dtype" not in filtered and isinstance(config_dict.get("dtype"), str):
            filtered["torch_dtype"] = getattr(torch, config_dict["dtype"])
        if isinstance(filtered.get("eos_token_id"), list):
            filtered["eos_token_id"] = filtered["eos_token_id"][0]

        filtered["neuron_config"] = neuron_config
        return cls(**filtered)


@dataclass
class Qwen3_5MoeMultimodalConfig:
    """The wrapper the published checkpoint actually is: a text backbone plus a vision tower.

    ``Qwen3_5MoeConfig.from_configs`` unwraps ``text_config`` and drops everything beside it, which is
    right for the text-only architecture and wrong for this one: the vision tower's shape and the four
    special token ids live at the TOP level, next to ``text_config`` rather than inside it. Building this
    from the same source dict keeps the unwrapping in one place instead of teaching the text config to
    half-remember its wrapper.

    The token ids have no defaults. A wrong id here does not fail: it makes the position builder mark the
    wrong spans as vision, so the image is placed at the wrong offsets and the output is a fluent lie.
    """

    text_config: Qwen3_5MoeConfig
    vision_config: Qwen3_5MoeVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    def __post_init__(self) -> None:
        text = self.text_config
        vision = self.vision_config
        if vision.out_hidden_size != text.hidden_size:
            raise ValueError(
                f"the vision tower emits {vision.out_hidden_size}-wide embeddings but the text stream "
                f"is {text.hidden_size} wide; this checkpoint family relies on them matching so the "
                "patch merger's output enters the token embedding stream with no extra projection."
            )
        for name in ("image_token_id", "video_token_id",
                     "vision_start_token_id", "vision_end_token_id"):
            value = getattr(self, name)
            if not 0 <= value < text.vocab_size:
                raise ValueError(
                    f"{name}={value} is outside the vocabulary (0, {text.vocab_size}); a wrong id "
                    "marks the wrong prompt spans as vision rather than failing."
                )

    @property
    def neuron_config(self):
        """The text backbone's NeuronConfig, so the top-level model can read it uniformly."""
        return self.text_config.neuron_config

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None,
                     vision_neuron_config=None) -> Qwen3_5MoeMultimodalConfig:
        """Build both halves from one HF config.

        ``vision_neuron_config`` is attached to the vision config rather than consumed here: the
        encoder's block size and token buckets are runner-supplied deployment choices, not checkpoint
        facts, and the preprocessing needs them at call time.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as handle:
                config_dict = json.load(handle)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = dict(hf_config)

        vision_dict = config_dict.get("vision_config") or {}
        if not vision_dict:
            raise ValueError(
                "this architecture requires a vision_config in the checkpoint config; a text-only "
                "checkpoint should be served as Qwen3_5MoeForCausalLM instead."
            )
        vision_fields = set(Qwen3_5MoeVisionConfig.__dataclass_fields__)
        vision = Qwen3_5MoeVisionConfig(
            **{k: v for k, v in vision_dict.items() if k in vision_fields and v is not None})
        vision.neuron_config = vision_neuron_config

        missing = [name for name in ("image_token_id", "video_token_id",
                                     "vision_start_token_id", "vision_end_token_id")
                   if config_dict.get(name) is None]
        if missing:
            raise ValueError(
                f"the checkpoint config is missing {missing}; these mark which prompt spans are "
                "vision, and guessing them would place the image at the wrong offsets."
            )
        return cls(
            text_config=Qwen3_5MoeConfig.from_configs(hf_config, neuron_config),
            vision_config=vision,
            image_token_id=int(config_dict["image_token_id"]),
            video_token_id=int(config_dict["video_token_id"]),
            vision_start_token_id=int(config_dict["vision_start_token_id"]),
            vision_end_token_id=int(config_dict["vision_end_token_id"]),
        )
