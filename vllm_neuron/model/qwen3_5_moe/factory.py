# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5-MoE model selection.

Text backbone of ``Qwen/Qwen3.6-35B-A3B``. bf16 is the only implementation; everything this model
cannot honour is refused here, at construction, rather than left to fail once compilation and device
capacity have been spent. See the model README for the reasons behind each refusal.
"""

from __future__ import annotations

from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5MoeForCausalLM(nn.Module):
    """Validates the config and selects the Qwen3.5-MoE implementation.

    Registered as ``Qwen3_5MoeForCausalLM`` — the TEXT architecture name, matching what this is.

    The published checkpoint declares ``Qwen3_5MoeForConditionalGeneration``, and registering under
    that name is wrong even though it resolves: the runner keys its multimodal path off the
    architecture, so it would build a vision NeuronConfig, validate vision token buckets against the
    prefill buckets, and expect an encoder to feed. Serving the published checkpoint therefore needs
    the architecture overridden to the text name (``hf_overrides``), which is also an honest statement
    that the vision tower is not being served.
    """

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate the runner-facing surface (``get_kv_spec``, ``bind_kv_cache``, ``load_weights``)
        to the selected implementation.

        The runner normally goes through ``from_configs``, which returns the implementation directly
        and never builds this wrapper. But the wrapper is what is registered, so a path that
        instantiates the architecture class would otherwise die on the first ``get_kv_spec`` call.
        """
        # nn.Module.__getattr__ handles parameters/buffers/submodules; this only runs when it misses.
        try:
            model = super().__getattr__("_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(model, name)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None,
                     text_neuron_config: NeuronConfig | None = None,
                     vision_neuron_config=None, **kwargs) -> nn.Module:
        # The checkpoint's architecture name is a *ConditionalGeneration one, so the runner builds
        # both a text and a vision NeuronConfig and hands over both. That is structural, not a
        # statement that the deployment wants images: `vision_neuron_config` is accepted and ignored.
        # What keeps image and video input out is that this model declares no multimodal interface,
        # so the runner has nothing to feed encoder output into.
        neuron = neuron_config if neuron_config is not None else text_neuron_config
        return cls._select_implementation(hf_config, neuron)

    @classmethod
    def _select_implementation(cls, hf_config: PretrainedConfig,
                               neuron_config: NeuronConfig | None) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model_bf16 import Qwen3_5MoeForCausalLM as Model
        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config: PretrainedConfig,
                         neuron_config: NeuronConfig | None) -> None:
        """Reject the serving options this implementation cannot honour.

        Architecture-level checks (layer types, the output gate, the rotary width) live in
        ``Qwen3_5MoeConfig.__post_init__``; this covers the deployment.

        Note where each setting lives: ``NeuronConfig`` owns the quantization, the expert-parallel
        degree and the bucket lists, while the batch size, prefix caching, speculative decoding and
        prompt embeddings are on the vLLM config. Looking for the latter on ``NeuronConfig`` finds
        nothing and the check passes silently.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"Qwen3.5-MoE on Neuron supports bf16 only; got quantization={quantization!r}."
            )

        # Every field below is read DIRECTLY rather than through getattr with a default. That is
        # deliberate: a default turns a renamed or removed field into a guard that silently never
        # fires, which is worse than no guard because the README claims the configuration is refused.
        # Reading directly means a field that moves takes the guard down loudly, at construction.
        #
        # Expert parallelism is expressed as a degree on NeuronConfig and as a flag on the vLLM
        # parallel config; either one being set means the deployment expects experts to be partitioned
        # across ranks, which this implementation does not do.
        if neuron_config is not None and neuron_config.ep_degree > 1:
            raise ValueError(
                f"Qwen3.5-MoE on Neuron does not implement expert parallelism; got ep_degree="
                f"{neuron_config.ep_degree}. Every rank holds all 256 experts and shards the expert "
                "intermediate dimension instead."
            )

        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        if vllm_config is None:
            # Every supported entry point constructs the model inside a vLLM config context. If that
            # ever stops holding, fail loudly rather than skipping the checks below.
            raise RuntimeError(
                "Qwen3.5-MoE on Neuron could not read the vLLM config, so it cannot verify that "
                "max_num_seqs is 1 and prefix caching is off — both of which it requires for "
                "correctness. Refusing to load rather than serving unchecked."
            )
        if vllm_config.parallel_config.enable_expert_parallel:
            raise ValueError(
                "Qwen3.5-MoE on Neuron does not implement expert parallelism; unset "
                "--enable-expert-parallel."
            )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        if max_num_seqs != 1:
            raise ValueError(
                f"Qwen3.5-MoE on Neuron supports max_num_seqs=1 only; got {max_num_seqs}. The Gated "
                "DeltaNet conv and recurrent state are single per-layer buffers with no per-slot "
                "pool, so concurrent sequences would read each other's state."
            )
        if vllm_config.speculative_config is not None:
            # Refuse where the capability is selected, not where it first breaks. The Gated DeltaNet
            # decode advances the recurrent state by exactly one token, so a multi-token verify step
            # cannot be made consistent with the accepted tokens; the mixer does raise, but only after
            # compilation and device capacity have been spent.
            raise ValueError(
                "Qwen3.5-MoE on Neuron does not support speculative decoding: the Gated DeltaNet "
                "decode advances the recurrent state by exactly one token, so a multi-token verify "
                "step would leave the state inconsistent with the accepted tokens."
            )
        if neuron_config is not None and neuron_config.apply_prefill_dcp:
            # With decode-context parallel prefill the runner hands each rank a slice of the prompt's
            # slot_mapping, while the recurrent layers need one mask entry per hidden-state token and a
            # state that spans the whole prefix. Refuse before construction rather than failing during
            # tracing.
            raise ValueError(
                "Qwen3.5-MoE on Neuron does not support decode-context-parallel prefill "
                "(apply_prefill_dcp): the Gated DeltaNet layers need the whole prefix in one pass to "
                "carry their recurrent state, and derive their real/pad mask from an unsliced "
                "slot_mapping."
            )
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            # Decode DCP interleaves the context across ranks, so each rank's KV cache holds only its
            # share. This model's decode attention reads its local cache directly, with no cross-rank
            # gather or log-sum-exp reduction, and would attend over an incomplete context.
            raise ValueError(
                "Qwen3.5-MoE on Neuron does not support decode-context parallelism "
                "(decode_context_parallel_size > 1): its decode attention reads the local KV cache "
                "without the cross-rank gather that DCP requires."
            )
        if vllm_config.model_config.enable_prompt_embeds:
            # The model embeds input_ids itself and never merges prompt embeddings, so rows that are
            # supposed to be embeddings would be read as their placeholder token IDs.
            raise ValueError(
                "Qwen3.5-MoE on Neuron does not support prompt embeddings: it embeds input_ids and "
                "never merges inputs_embeds, so embedding rows would be read as placeholder tokens."
            )
        decode_buckets = neuron_config.decode_batch_buckets if neuron_config else None
        if decode_buckets and max(decode_buckets) > 1:
            # The Gated DeltaNet decode advances the state by exactly one token and raises otherwise.
            # Refuse here so that failure happens at construction with a reason, rather than part-way
            # through compile warmup of a multi-token decode bucket.
            raise ValueError(
                f"Qwen3.5-MoE on Neuron needs single-token decode buckets; got "
                f"decode_batch_buckets={sorted(decode_buckets)}. The Gated DeltaNet decode advances "
                "the recurrent state by exactly one token."
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Qwen3.5-MoE on Neuron cannot use automatic prefix caching. The attention KV is "
                "addressable by block hash and would be reused, but the Gated DeltaNet recurrent "
                "state has no block-hash addressing, so a reused prefix would continue from the "
                "wrong state. Unset --enable-prefix-caching."
            )
