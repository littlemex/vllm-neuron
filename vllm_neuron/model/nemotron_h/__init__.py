# SPDX-License-Identifier: Apache-2.0
"""NemotronH (text backbone of Nemotron-3-Nano-Omni) for vLLM on Neuron.

Hybrid Mamba2 / MoE / Attention decoder (52 layers). On instances with a single EFA card
(e.g. trn2.3xlarge) run with ``NEURON_SKIP_EFA_AFFINITY=1``: the Neuron EFA-affinity probe expects a
co-located EFA under each NeuronCore's PCI path, which only holds on multi-card instances such as
trn2.48xlarge. This affinity is a CPU-locality optimization, not a correctness requirement.
"""
from .config import NemotronHConfig
from . import model_bf16  # noqa: F401
from .factory import NemotronHForCausalLM

__all__ = ["NemotronHConfig", "NemotronHForCausalLM"]
