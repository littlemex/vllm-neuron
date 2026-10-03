# SPDX-License-Identifier: Apache-2.0
"""
NemotronH Configuration
================================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.
Language backbone (NemotronHForCausalLM) of Nemotron-3-Nano-Omni-30B-A3B. Text-only: the Omni
vision/audio encoders are intentionally out of scope for this onboarding.

Architecture (from nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 config.json):
  - Hybrid decoder: 52 layers interleaving Mamba2 (SSM), MoE, and Attention, selected per-layer
    by `hybrid_override_pattern` (M=Mamba2, E=MoE, *=Attention). Counts: 23 Mamba2 / 23 MoE / 6 Attn.
  - MoE: 128 routed experts, top-6, plus 1 shared expert.
  - Mamba2: mamba_num_heads=64, mamba_head_dim=64, ssm_state_size=128, conv_kernel=4, n_groups=8.
  - Attention: GQA, 32 query heads / 2 KV heads, head_dim 128.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# hybrid_override_pattern legend
MAMBA = "M"
MOE = "E"
ATTENTION = "*"
MLP = "-"


@dataclass
class NemotronHConfig:
    """Configuration for the NemotronH language backbone (text-only).

    <-- MODEL-SPECIFIC: These parameters define the NemotronH hybrid architecture.
    """

    # ── Model architecture (MODEL-SPECIFIC) ──────────────────────────────
    vocab_size: int = 131072
    hidden_size: int = 2688
    num_hidden_layers: int = 52
    # Per-layer type string; length must equal num_hidden_layers.
    hybrid_override_pattern: str = (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    )
    rms_norm_eps: float = 1e-5
    torch_dtype: torch.dtype = torch.bfloat16
    tie_word_embeddings: bool = False

    # ── Attention (GQA) (MODEL-SPECIFIC) ─────────────────────────────────
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128

    # ── MoE (DeepSeek-style grouped top-k routing) (MODEL-SPECIFIC) ──────
    n_routed_experts: int = 128
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 1856
    moe_shared_expert_intermediate_size: int = 3712
    # Router: sigmoid scores + e_score_correction_bias, grouped selection, then top-k.
    # (from modeling_nemotron_h.py NemotronHMoE.route_tokens_to_experts)
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    mlp_hidden_act: str = "relu2"       # expert MLP activation (relu-squared)

    # ── Mamba2 (SSM) (MODEL-SPECIFIC) ────────────────────────────────────
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    ssm_state_size: int = 128           # d_state
    conv_kernel: int = 4                # d_conv
    n_groups: int = 8
    mamba_hidden_act: str = "silu"
    use_conv_bias: bool = True
    use_bias: bool = False
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 0.0001

    # ── Sequence / tokens (MODEL-SPECIFIC) ───────────────────────────────
    max_position_embeddings: int = 262144
    pad_token_id: int | None = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        # <-- MODEL-SPECIFIC: hybrid_override_pattern length/topology validation is unique to this
        # hybrid-decoder architecture; the mamba inner-dim formula is this model's own convention.
        # mamba inner dim = heads * head_dim (NOT expand * hidden for this model)
        self.mamba_intermediate_size = self.mamba_num_heads * self.mamba_head_dim  # 4096
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            raise ValueError(
                f"hybrid_override_pattern length {len(self.hybrid_override_pattern)} "
                f"!= num_hidden_layers {self.num_hidden_layers}"
            )
        # The bf16 impl hardcodes conv bias present and attention/MLP bias absent (as in 30B-A3B).
        # Reject a checkpoint whose config disagrees rather than silently using the wrong topology.
        if not self.use_conv_bias:
            raise ValueError("NemotronH bf16 impl assumes use_conv_bias=True (Mamba conv1d has bias).")
        if self.use_bias:
            raise ValueError("NemotronH bf16 impl assumes use_bias=False (attention/MLP carry no bias).")

    def layer_type(self, i: int) -> str:
        """Return the per-layer type char (M / E / * / -) for layer index i.

        <-- MODEL-SPECIFIC: the hybrid Mamba2/MoE/Attention layer-type dispatch is unique to this
        architecture (attention-only reference models have no such per-layer dispatch).
        """
        return self.hybrid_override_pattern[i]

    def layer_type_counts(self) -> dict:
        p = self.hybrid_override_pattern
        return {"mamba": p.count(MAMBA), "moe": p.count(MOE),
                "attention": p.count(ATTENTION), "mlp": p.count(MLP)}

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Create config from a HuggingFace config (+ NeuronConfig).

        The Omni checkpoint nests the language config under `llm_config`; the text-only
        checkpoint has these fields at top level. Handle both.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        # <-- MODEL-SPECIFIC: the Omni wrapper unwrapping below is unique to this checkpoint family.
        # Unwrap the Omni wrapper: the language backbone lives under llm_config (the Omni-wrapped
        # checkpoint) or under language_model — handle both. A text-only checkpoint has these
        # fields at top level.
        for wrap in ("llm_config", "language_model", "text_config"):
            if isinstance(config_dict.get(wrap), dict) and "hybrid_override_pattern" in config_dict[wrap]:
                config_dict = config_dict[wrap]
                break

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}
        # <-- MODEL-SPECIFIC: recovering attribute_map-aliased fields is needed because this
        # architecture's HF config aliases num_hidden_layers/hybrid_override_pattern etc.
        # transformers PretrainedConfig.to_dict() serialises some fields under attribute_map
        # aliases (e.g. num_hidden_layers, hybrid_override_pattern), so they are absent from
        # config_dict under our field names and would silently fall back to the dataclass
        # defaults (the full 52-layer model). Recover any missing field via getattr on the
        # original config object, which resolves the alias. (For the Omni checkpoint the fields
        # live in the already-unwrapped llm_config dict, so this only fires for the text-only /
        # aliased case.)
        if isinstance(hf_config, PretrainedConfig):
            for fname in field_names:
                if fname not in filtered and fname != "neuron_config" and hasattr(hf_config, fname):
                    filtered[fname] = getattr(hf_config, fname)
        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])
        # HF NemotronH names the RMSNorm epsilon `layer_norm_epsilon` (used by every norm in the
        # reference modeling code); our field is `rms_norm_eps`. Map it explicitly so a checkpoint
        # that changes the eps is honored instead of silently using the dataclass default. (The
        # config's `norm_eps` field is not referenced by the HF modeling code.)
        if "rms_norm_eps" not in filtered:
            eps = config_dict.get("layer_norm_epsilon")
            if eps is None and isinstance(hf_config, PretrainedConfig):
                eps = getattr(hf_config, "layer_norm_epsilon", None)
            if eps is not None:
                filtered["rms_norm_eps"] = eps
        filtered["neuron_config"] = neuron_config
        return cls(**filtered)
