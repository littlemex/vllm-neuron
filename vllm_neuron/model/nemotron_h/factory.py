# SPDX-License-Identifier: Apache-2.0
"""Factory for NemotronH model selection.

Text-only language backbone of Nemotron-3-Nano-Omni. Only bf16 is wired up for now; the FP8 /
NVFP4 paths are future work (mirrors the GptOss factory shape).
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class NemotronHForCausalLM(nn.Module):
    """Factory that validates config and selects the NemotronH implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry. Delegates forward() to the selected impl.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None,
        text_neuron_config: NeuronConfig | None = None, vision_neuron_config=None, **kwargs
    ) -> nn.Module:
        # The multimodal runner passes text_/vision_neuron_config. This is the TEXT backbone only
        # (the Omni vision/audio encoders are out of scope), so fail fast if a vision config is
        # supplied rather than silently running text-only, and fold text_neuron_config in otherwise.
        if vision_neuron_config is not None:
            raise ValueError(
                "NemotronH here is the text backbone only; the Omni vision/audio encoders are out "
                "of scope. Received a vision_neuron_config — point this at the text-only checkpoint."
            )
        nc = neuron_config if neuron_config is not None else text_neuron_config
        return cls._select_implementation(hf_config, nc)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        # bf16 is the only wired implementation today (FP8/NVFP4 are future work).
        from .model_bf16 import NemotronHForCausalLM as Model
        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        # The hybrid pattern length vs num_hidden_layers is asserted in NemotronHConfig.__post_init__.
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"NemotronH onboarding currently supports bf16 only; got quantization="
                f"{quantization!r}. FP8/NVFP4 are future work."
            )
