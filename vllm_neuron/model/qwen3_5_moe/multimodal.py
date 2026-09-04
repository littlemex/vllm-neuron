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
from .model_bf16 import Qwen3_5MoeForCausalLM
from .ops import temporal_axis
from .vision_inputs import (
    build_vision_inputs,
    patch_dim,
    vision_blocks,
    write_block_ids,
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
        neuron = neuron_config if neuron_config is not None else text_neuron_config
        return cls(Qwen3_5MoeMultimodalConfig.from_configs(
            hf_config, neuron, vision_neuron_config=vision_neuron_config))

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
        block_size = vision_neuron_config.vision_attention_block_size
        merge = self.vision_config.spatial_merge_size
        # One synthetic item that fills the bucket. Its grid has to be legal for the packer -- a frame
        # must fit in a block -- so the bucket is expressed as ceil(bucket / block_size) items of
        # block_size patches each, rather than one item of `bucket` patches.
        per_item = block_size
        side = merge
        while (per_item // side) % merge or side * (per_item // side) != per_item:
            side += merge
            if side > per_item:
                raise ValueError(
                    f"vision_attention_block_size={block_size} cannot be written as a grid whose "
                    f"height and width are multiples of spatial_merge_size={merge}."
                )
        items = max(1, -(-bucket // block_size))
        grid = torch.tensor([[1, side, per_item // side]] * items)
        pixels = torch.zeros(items * per_item, patch_dim(self.vision_config), dtype=torch.bfloat16)
        num_blocks = vision_blocks(
            grid, block_size, list(vision_neuron_config.num_vision_tokens_buckets),
            dp_size=getattr(vision_neuron_config, "dp_size", 1))
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

        # Allocate cache blocks per item, in item order. That order is what makes the encoder-block to
        # cache-block mapping a flattening rather than a lookup.
        cache_block_map = []
        for index, row in enumerate(image_grid_thw.tolist()):
            merged = (row[0] * row[1] * row[2]) // merge_factor
            cache_block_map.append(encoder_cache.allocate(
                mm_hashes[index], encoder_cache.dense_tokens_per_block(merged, block_size)))

        num_blocks = vision_blocks(
            image_grid_thw, block_size, list(vision_neuron_config.num_vision_tokens_buckets),
            dp_size=getattr(vision_neuron_config, "dp_size", 1))
        built = build_vision_inputs(
            pixel_values, image_grid_thw, self.vision_config, block_size, num_blocks)
        destinations = write_block_ids(
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

        The temporal axis is the sequential one because it is the only monotone axis. Height and width
        run over an image's grid and go backwards at each new row, so using either for the KV cache or
        for the recurrent layers' fresh-request test would place tokens out of order.

        A one-dimensional ``positions`` still means text, and it takes the text path: the runner sends
        the collapsed form when no multimodal features are present, and asking for three axes here would
        turn a text request into a shape error.
        """
        three_axis = positions.dim() == 2 and positions.shape[0] == 3  # lint-port: ok dim and shape are graph-static, not tensor contents
        return temporal_axis(positions), (positions if three_axis else None)

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
