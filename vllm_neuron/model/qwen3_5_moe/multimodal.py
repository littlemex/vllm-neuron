# SPDX-License-Identifier: Apache-2.0
"""The multimodal architecture: the text backbone plus the vision tower.

This is a subclass of the text model rather than a sibling, for one reason worth stating: everything the
text model does to load weights, declare its KV, gather logits and sample is identical here, and the two
places that most reliably rot when duplicated are the weight map's completeness check and the
still-on-meta check. Subclassing keeps both watching the vision parameters too — a sibling that wrapped
the text model would report "nothing left on meta" while the encoder sat on meta.

Three contract members are added, and none of them is the encoder's arithmetic (the encoder is reused
from ``model/qwen3_vl/`` unchanged, because this checkpoint's vision keys are identical to it):

    embed_multimodal              run the encoder and write into the encoder cache
    build_vision_synthetic_inputs shape-only inputs so the runner can warm the encoder graph
    get_mrope_input_positions     real 3-axis positions, replacing the text-only degenerate case

The encoder runs as its OWN graph, not as part of the decoder's. That is what makes ``embed_multimodal``
a separate entry point: the runner calls it before prefill, the result lands in the on-device encoder
cache, and the decoder's prefill graph reads the cache through ``vision_embedding_blocks``. So the
decoder never sees a pixel, and the vision tower's compile time and bucket set are independent of the
text side's.
"""

from __future__ import annotations

import torch

from vllm_neuron.model.qwen3_vl.utils.mrope import compute_mrope_positions
from vllm_neuron.model.qwen3_vl.vision_encoder_bf16 import Qwen3VLVisionModel

from .config import Qwen3_5MoeMultimodalConfig
from .layout import vision_checkpoint_mappings
from .model_bf16 import Qwen3_5MoeForCausalLM, resolve_text_neuron_config
from .ops import temporal_axis
from .vision_inputs import (
    blocks_for_bucket,
    build_vision_inputs,
    cache_write_destinations,
    patch_dim,
    vision_blocks,
)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForCausalLM):
    """Text backbone + vision tower, with the runner-facing multimodal contract."""

    def __init__(self, config: Qwen3_5MoeMultimodalConfig):
        super().__init__(config.text_config)
        self.multimodal_config = config
        self.vision_config = config.vision_config
        # bf16 rather than the text config's dtype: the encoder's parameters are bf16 in the checkpoint
        # and its graph is compiled for them. The text side's fp32 residual stream is a deviation local
        # to the decoder (see model_bf16's module docstring) and does not extend here.
        self.visual = Qwen3VLVisionModel(config.vision_config, dtype=torch.bfloat16)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, text_neuron_config=None,
                     vision_neuron_config=None, **kwargs):
        """Build both halves. The text NeuronConfig's two spellings are resolved by the shared helper,
        which refuses a disagreement rather than ranking the two."""
        return cls(Qwen3_5MoeMultimodalConfig.from_configs(
            hf_config, resolve_text_neuron_config(neuron_config, text_neuron_config),
            vision_neuron_config=vision_neuron_config))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Real 3-axis MRoPE positions, delegated to the repository's own builder.

        The text model refuses a request carrying multimodal features, because collapsing the three axes
        to one is exact only while they agree. Here they do not: a vision span advances the height and
        width axes over the image's grid while the temporal axis stands still, so the collapse would
        place the image's tokens at positions that belong to text.

        The builder is shared and reads the four special token ids plus ``spatial_merge_size`` from the
        config, which is why those ids have no defaults: a wrong one marks the wrong span as vision and
        the request is served with the image at the wrong offsets rather than failing.
        """
        return compute_mrope_positions(input_tokens, mm_features, self.multimodal_config)

    # ------------------------------------------------------------------
    # The encoder's own graph
    # ------------------------------------------------------------------

    def build_vision_synthetic_inputs(self, bucket, vision_neuron_config, device):
        """Shape-only inputs for warming the encoder graph, keyed by its parameter names.

        This is what fixes the NEFF's input dtypes, so it is the authority the real call site must agree
        with — not the other way round. It is built from the same helper the real path uses, with zero
        pixels, so the two cannot disagree about shape or dtype: a warmup that declared its own shapes
        would be a second source of truth, and the failure mode is a load-time rejection after the
        compile has already been paid for.
        """
        # The runner's object is the authority and the attached one must agree with it. Picking one
        # silently would let the warmup compile for a block size the serving path never uses, and the
        # mismatch would surface as a load-time rejection at the first image.
        attached = self.vision_config.neuron_config
        if attached is not None and (
                attached.vision_attention_block_size
                != vision_neuron_config.vision_attention_block_size
                or list(attached.num_vision_tokens_buckets or [])
                != list(vision_neuron_config.num_vision_tokens_buckets or [])):
            raise ValueError(
                "the VisionNeuronConfig attached at construction disagrees with the one the runner "
                f"passed for warmup: block size {attached.vision_attention_block_size} vs "
                f"{vision_neuron_config.vision_attention_block_size}, buckets "
                f"{attached.num_vision_tokens_buckets} vs "
                f"{vision_neuron_config.num_vision_tokens_buckets}. The graph would be warmed for one "
                "and fed by the other."
            )
        block_size = vision_neuron_config.vision_attention_block_size
        merge = self.vision_config.spatial_merge_size
        # Synthetic items that fill the bucket, each exactly one block. The bucket is expressed as
        # ceil(bucket / block_size) items of block_size patches rather than one item of `bucket`
        # patches, because an item has to fit inside a block for the packer to accept it.
        #
        # The grid is computed, not searched: height and width must both be multiples of the merge size,
        # so `merge` by `block_size // merge` is a legal grid whenever block_size is divisible by
        # merge**2 — which the factory checks at construction, where the other block-size constraints
        # live. Searching for a legal side here would report a config error as a warmup failure.
        # The bucket is the RUNNER's fact. Converting it is the only step here; re-selecting one from a
        # synthetic token count would let this warm a different graph than the serving path builds for.
        num_blocks = blocks_for_bucket(bucket, block_size, vision_neuron_config.dp_size)
        items = num_blocks
        grid = torch.tensor([[1, merge, block_size // merge]] * items)
        pixels = torch.zeros(items * block_size, patch_dim(self.vision_config), dtype=torch.bfloat16)
        built = build_vision_inputs(pixels, grid, self.vision_config, block_size, num_blocks)
        return {name: tensor.to(device) for name, tensor in built.items()}

    @torch.no_grad()
    def embed_multimodal(self, pixel_values=None, image_grid_thw=None, encoder_cache=None,
                         mm_hashes=None, pixel_values_videos=None, video_grid_thw=None, **kwargs):
        """Encode the images and scatter-write them into the on-device encoder cache.

        Images only. Video is refused rather than folded onto the image path: its frames pack per frame,
        so the block-size floor is per frame rather than per item, and treating a video as one item
        quietly changes what a block means (see ``vision_inputs``).
        """
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "Qwen3.5-MoE on Neuron serves images; video needs the per-frame expansion before "
                "packing, which this implementation does not do."
            )
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("embed_multimodal requires both pixel_values and image_grid_thw.")
        if encoder_cache is None:
            raise ValueError(
                "embed_multimodal writes into the runner's encoder cache and has no fallback; without "
                "it the encoded image would be discarded and the prompt served as text."
            )
        if mm_hashes is None or len(mm_hashes) != image_grid_thw.shape[0]:
            raise ValueError(
                f"expected one mm_hash per item; got {mm_hashes and len(mm_hashes)} for "
                f"{image_grid_thw.shape[0]} item(s). The hash is the cache key, so a mismatch would "
                "store one image under another's identity."
            )

        vision_neuron_config = self.vision_config.neuron_config
        if vision_neuron_config is None:
            raise ValueError(
                "the vision tower has no VisionNeuronConfig, so its block size and token buckets are "
                "unknown; the runner supplies it when the architecture is registered as multimodal."
            )
        block_size = vision_neuron_config.vision_attention_block_size
        merge_factor = self.vision_config.spatial_merge_size ** 2

        # Everything that can be refused is done BEFORE a single cache block is reserved. The order is
        # the point: `allocate` records blocks under the item's mm_hash, which is the cache key, so a
        # later failure would leave a hash present whose blocks were never written — and the next request
        # carrying the same image would find the hash and read back whatever the cache held. Nothing in
        # the two calls below needs the cache, so there is no reason to allocate first.
        num_blocks = vision_blocks(
            image_grid_thw, block_size, list(vision_neuron_config.num_vision_tokens_buckets),
            dp_size=vision_neuron_config.dp_size)
        built = build_vision_inputs(
            pixel_values, image_grid_thw, self.vision_config, block_size, num_blocks)

        # Now reserve, in item order. That order is what makes the encoder-block to cache-block mapping
        # a flattening rather than a lookup.
        #
        # The two block counts are computed in different units: the packer works in raw patches, the
        # cache in merged tokens, and both are divided by the same `block_size`. They agree only because
        # an item is limited to one block, which makes both ceilings 1 — so the equality is checked
        # rather than assumed. If either rule is relaxed, an item could own two encoder blocks and one
        # cache block, and its tail would scatter into a neighbour's entry with no error.
        cache_block_map = []
        for index, row in enumerate(image_grid_thw.tolist()):
            raw = row[0] * row[1] * row[2]
            merged = raw // merge_factor
            blocks = encoder_cache.allocate(
                mm_hashes[index], encoder_cache.dense_tokens_per_block(merged, block_size))
            encoder_blocks = -(-raw // block_size)
            if len(blocks) != encoder_blocks:
                raise ValueError(
                    f"item {index} occupies {encoder_blocks} encoder block(s) but was allocated "
                    f"{len(blocks)} cache block(s) ({raw} raw patches, {merged} merged tokens, "
                    f"block_size {block_size}). The encoder writes block i into cache block i, so the "
                    "two counts must agree per item."
                )
            cache_block_map.append(blocks)

        destinations = cache_write_destinations(
            cache_block_map, num_blocks, encoder_cache.scratch_block_id)

        device = next(self.visual.parameters()).device
        self.visual(
            **{name: tensor.to(device) for name, tensor in built.items()},
            encoder_cache_buffer=encoder_cache.buffer,
            write_block_ids=destinations.to(device),
        )

    # ------------------------------------------------------------------
    # Forward and weights
    # ------------------------------------------------------------------

    def _positions(self, positions):
        """Keep all three axes for the rotary, and use the temporal one as the sequential position.

        This is the whole difference between the two architectures' forward passes, which is why it is a
        two-line override rather than a second forward: everything else — the embedding merge, the
        layers, the head, the sampler — is shared.

        The temporal axis is the sequential one because it is the only axis that is monotone OUTSIDE an
        image span; inside one it is constant while height and width vary. That is safe here only because
        of what reads it — see ``ops.temporal_axis``, which names the three consumers and why none of
        them needs a per-token index.

        A one-dimensional ``positions`` still means text, and it takes the text path: the runner sends
        the collapsed form when no multimodal features are present, and asking for three axes here would
        turn a text request into a shape error.
        """
        if positions.dim() == 1:  # lint-port: ok dim is graph-static, not tensor contents
            # The runner's text convention. Taken as text rather than inferred as such.
            return temporal_axis(positions), None
        if positions.dim() != 2 or positions.shape[0] != 3:  # lint-port: ok dim and shape are graph-static, not tensor contents
            # Refusing beats falling back to text. Treating an unrecognised shape as text is the
            # dangerous direction: MRoPE positions read as a single axis place an image's tokens at
            # positions that belong to the surrounding prose, which is not an error.
            raise ValueError(
                f"positions has shape {tuple(positions.shape)}; expected [T] for text or [3, T] for "
                "MRoPE. An unrecognised shape is refused rather than read as text."
            )
        return temporal_axis(positions), positions

    def checkpoint_mappings(self, source, checkpoint_keys):
        """The text map plus the vision tower's, in one dict.

        One map rather than two passes, so the inherited loader's two checks — every mapped source
        present in the checkpoint, and nothing left on ``meta`` afterwards — cover the encoder as well.
        A second pass would have to run after the meta check had already passed, and the failure it
        would stop catching is the expensive one: an encoder still on ``meta`` produces an image
        embedding of garbage, not an error.

        The vision destinations are prefixed because the encoder names its parameters relative to
        itself, and the names come from its own generator rather than being written out here (a
        hand-written version named submodule paths the encoder does not have, and the source keys
        matched, so a source-side diff could not see it).
        """
        mappings = super().checkpoint_mappings(source, checkpoint_keys)
        for destination, checkpoint_key in vision_checkpoint_mappings(
                self.vision_config.depth).items():
            prefixed = f"visual.{destination}"
            if prefixed in mappings:
                raise ValueError(
                    f"{prefixed} is claimed by both the text map and the vision map; one of them "
                    "would silently win and the other's weights would never be loaded."
                )
            mappings[prefixed] = checkpoint_key
        return mappings
