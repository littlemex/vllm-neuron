# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE text backbone (``Qwen/Qwen3.6-35B-A3B``) for the vLLM Neuron plugin.

See README.md in this directory for the architecture, the supported feature matrix, and the known
limitations (batch=1, no prefix caching, no speculative decoding, text-only).
"""
from .config import Qwen3_5MoeConfig
from .factory import Qwen3_5MoeForCausalLM

__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeForCausalLM"]
