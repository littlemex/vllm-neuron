# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE text backbone (``Qwen/Qwen3.6-35B-A3B``) for the vLLM Neuron plugin.

See README.md in this directory for the architecture, the supported feature matrix, and the known
limitations (batch=1, no prefix caching, no speculative decoding).

Two architectures are registered: ``Qwen3_5MoeForCausalLM`` serves the text backbone, and
``Qwen3_5MoeForConditionalGeneration`` adds the vision tower. The name decides which
runner path is taken, so it is the choice of what gets served, not a spelling.
"""
from .config import Qwen3_5MoeConfig
from .factory import Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration

__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeForCausalLM",
           "Qwen3_5MoeForConditionalGeneration"]
