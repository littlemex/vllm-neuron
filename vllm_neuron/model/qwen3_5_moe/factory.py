# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5-MoE model selection.

Text backbone of ``Qwen/Qwen3.6-35B-A3B``. bf16 is the only implementation; everything this model
cannot honour is refused here, at construction, rather than left to fail once compilation and device
capacity have been spent. See the model README for the reasons behind each refusal.
"""

from __future__ import annotations

from typing import cast

import torch
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
        return cast(nn.Module, Model.from_configs(hf_config, neuron_config))

    # ------------------------------------------------------------------
    # vLLM's hybrid-model contract
    # ------------------------------------------------------------------
    # vLLM computes the recurrent state's page size from the REGISTERED class, then raises the attention
    # block size until an attention page is at least as large and records the padding
    # (``Platform._align_hybrid_block_size``). Without these two classmethods that computation cannot
    # run, and the KV cache manager refuses at startup with "The page size of the layer is not
    # divisible by the maximum page size" -- a message that names neither this model nor its state.
    #
    # Declared on the factory because that is the class the registry resolves.

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple[tuple[int, ...], ...]:
        """The conv and recurrent state shapes per layer, WITHOUT the slot axis.

        Derived from the same config the model builds from, and sharded the same way, so the page size
        vLLM computes is the one the model will allocate. A shape that disagreed here would size the
        pages for a state nobody writes.
        """
        from .config import Qwen3_5MoeConfig

        config = Qwen3_5MoeConfig.from_configs(vllm_config.model_config.hf_config, None)
        world_size = vllm_config.parallel_config.tensor_parallel_size
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = (2 * key_dim + value_dim) // world_size
        return (
            (conv_dim, config.linear_conv_kernel_dim - 1),
            (config.linear_num_value_heads // world_size,
             config.linear_key_head_dim, config.linear_value_head_dim),
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config) -> tuple[torch.dtype, ...]:
        """conv in the model dtype, recurrent in fp32.

        The recurrence compounds over the whole prefix and the checkpoint declares fp32 for it, so the
        two states do NOT share a dtype -- which is also why the page size cannot be derived from one
        element size.
        """
        from .config import Qwen3_5MoeConfig

        config = Qwen3_5MoeConfig.from_configs(vllm_config.model_config.hf_config, None)
        return (config.torch_dtype, torch.float32)

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
                f"Qwen3.5-MoE on Neuron supports max_num_seqs=1 only; got {max_num_seqs}. The "
                "remaining blocker is PREFILL, not the state pool: the Gated DeltaNet prefill scan is "
                "a recurrence over one sequence, and a scheduled batch mixing two requests' tokens "
                "would carry one's tail into the other's head. Decode is already batch-general over "
                "the request axis and reads its slot from the pool; see the project's "
                "docs/DESIGN-concurrency.md for what stage B needs."
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


class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForCausalLM):
    """The multimodal architecture: the same text backbone with the vision tower attached.

    Registered under the name the published checkpoint declares. Registering it is what opens the
    runner's multimodal path — the runner keys off the architecture name to build a vision
    NeuronConfig, size the encoder's buckets and route image inputs to ``embed_multimodal``. So this
    class existing is not a convenience; it is the entry point, and without it the model-side vision
    work is unreachable (the same shape of wall the MTP head hit, from the other side).

    Every deployment check the text architecture makes applies unchanged and is inherited. What is
    added is the vision tower's own requirement: the encoder needs a VisionNeuronConfig, and it needs
    the block size and the token buckets in it, because the graph is compiled per bucket.
    """

    def __init__(self, hf_config: PretrainedConfig, text_neuron_config: NeuronConfig | None = None,
                 vision_neuron_config=None) -> None:
        nn.Module.__init__(self)
        self._model = self._select_multimodal(hf_config, text_neuron_config, vision_neuron_config)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None,
                     text_neuron_config: NeuronConfig | None = None,
                     vision_neuron_config=None, **kwargs) -> nn.Module:
        # The runner spells the text NeuronConfig either way depending on how it reached here, so the
        # precedence is resolved HERE and nowhere else: the model's own from_configs would otherwise
        # decide it a second time, and two places that pick between the same two arguments will
        # eventually pick differently.
        neuron = neuron_config if neuron_config is not None else text_neuron_config
        return cls._select_multimodal(hf_config, neuron, vision_neuron_config)

    @classmethod
    def _select_multimodal(cls, hf_config: PretrainedConfig,
                           neuron_config: NeuronConfig | None,
                           vision_neuron_config) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        cls._validate_vision_config(hf_config, vision_neuron_config)
        from .multimodal import Qwen3_5MoeForConditionalGeneration as Model
        return cast(nn.Module, Model.from_configs(
            hf_config, neuron_config, vision_neuron_config=vision_neuron_config))

    @classmethod
    def _validate_vision_config(cls, hf_config: PretrainedConfig,
                                vision_neuron_config) -> None:
        """Reject a vision configuration the encoder cannot be given.

        Read directly rather than through ``getattr`` with a default, for the reason the text checks
        give: a default turns a renamed field into a guard that never fires.
        """
        if vision_neuron_config is None:
            raise ValueError(
                "the multimodal architecture needs a VisionNeuronConfig; without it the encoder's "
                "block size and token buckets are unknown and no encoder graph can be warmed. Serve "
                "the checkpoint as Qwen3_5MoeForCausalLM for text-only."
            )
        # The field is Optional and the platform fills it in check_and_update_config, which runs before
        # the model is built. Checking for None rather than assuming it means a change in that ordering
        # says so here instead of raising TypeError inside list() three frames down.
        if not vision_neuron_config.num_vision_tokens_buckets:
            raise ValueError(
                "num_vision_tokens_buckets is unset or empty, so no encoder graph would be compiled "
                "and the first image would arrive with nothing to run. The platform normally derives "
                "it during config resolution; an empty value here means that step did not run."
            )
        buckets = list(vision_neuron_config.num_vision_tokens_buckets)
        block_size = vision_neuron_config.vision_attention_block_size
        if block_size <= 0:
            raise ValueError(f"vision_attention_block_size must be positive; got {block_size}.")
        # A block has to be expressible as a patch grid whose height and width are both multiples of the
        # spatial merge size, because the merger consumes merge x merge patch groups. Checked here, with
        # the other block-size constraints, so a bad deployment value is reported as a config error
        # rather than surfacing later as a failure to build the encoder's warmup input.
        merge = hf_config.vision_config.spatial_merge_size
        if block_size % (merge ** 2):
            raise ValueError(
                f"vision_attention_block_size={block_size} is not divisible by "
                f"spatial_merge_size**2 ({merge}**2 = {merge ** 2}), so a block cannot be filled by "
                "whole merge groups."
            )
        # The packer refuses an item larger than one block, and the runner budgets in PIXELS. Nothing
        # connects the two, so an image the front end admits can be killed at encode time. Reconcile
        # them here: the factory is where deployment choices meet checkpoint facts.
        max_pixels = cls._configured_max_pixels(hf_config)
        if max_pixels is not None:
            per_item = cls.get_max_pixels_token_count(hf_config, max_pixels)
            if per_item > block_size:
                raise ValueError(
                    f"mm_processor_kwargs max_pixels={max_pixels} admits {per_item} patches per image, "
                    f"but vision_attention_block_size={block_size} is the per-item ceiling (an item "
                    "must fit in one block so its attention is complete). Lower max_pixels or raise "
                    "the block size; leaving them inconsistent fails at the first large image instead."
                )
        if any(bucket % block_size for bucket in buckets):
            raise ValueError(
                f"every vision token bucket must be a multiple of the block size {block_size}; got "
                f"{[b for b in buckets if b % block_size]}. A partial block at the end has no cache "
                "block to be written into."
            )

    @classmethod
    def _configured_max_pixels(cls, hf_config: PretrainedConfig) -> int | None:
        """The deployment's per-image pixel cap, if it set one.

        Read through the vLLM config rather than from ``hf_config``: the cap is a serving choice that
        arrives in ``mm_processor_kwargs``, not a property of the checkpoint.
        """
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        multimodal = getattr(vllm_config.model_config, "multimodal_config", None)  # lint-port: ok absent on a text-only deployment, and its absence means no cap
        kwargs = multimodal.mm_processor_kwargs if multimodal else None
        return kwargs.get("max_pixels") if kwargs else None

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """How many raw patches collapse into one embedding token, for the runner's budgeting."""
        return int(hf_config.vision_config.spatial_merge_size) ** 2

    @classmethod
    def get_max_pixels_token_count(cls, hf_config: PretrainedConfig, max_pixels: int) -> int:
        """Convert a pixel cap into a raw (pre-merge) patch count.

        Patch tiling differs across architectures, which is why the model owns this rather than the
        runner: here a patch is ``patch_size`` square, so the count is the cap divided by its area.
        """
        patch_size = int(hf_config.vision_config.patch_size)
        return max_pixels // (patch_size ** 2)
