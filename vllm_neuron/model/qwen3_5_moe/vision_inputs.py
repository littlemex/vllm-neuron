"""Host-side tensors the vision encoder's forward requires.

The encoder itself is reusable as-is — this checkpoint's vision keys are identical to the Qwen3-VL
reference in this repository — but its ``forward`` takes nine arguments and six of them are prepared on
the host, outside the compiled graph.

**Those six are built by this repository's own helpers, not here.** They live under
``model/qwen3_vl/utils/`` and are generic despite the model-specific package they sit in:
``compute_rotary_pos_emb``, ``compute_position_indices_and_weights``, ``ffd_pack_images``,
``scatter_to_blocks``, ``compute_block_bounds``, ``select_vision_bucket``. This module is an adapter over
them, and it exists for one reason the helpers do not cover: it returns a dict keyed by the encoder's
parameter names. Six tensors of similar rank passed positionally is the shape of mistake that neither
the type checker nor a shape assertion catches.

This module used to build the six tensors from ``transformers.vision_utils`` instead. The maths was
right — the position ids and rotary tables agreed with the helpers to the bit in fp32 — and the contract
was wrong in five separate places, every one of which produces a plausible image embedding rather than
an error:

    * ``cos``/``sin`` were cast to bf16; the graph declares them **fp32**
    * one image was allowed to span two blocks, which severs its attention (blocks do not attend to
      each other), where the helper raises
    * two images were allowed to share a block, which the cache write cannot express: the encoder maps
      block i to one cache block, and the cache allocates per item
    * the attention bounds were in **global** coordinates; the encoder reads them **block-local**, so
      any image not starting in block 0 got a window outside its own block
    * one ``dtype=`` argument covered tensors whose declared dtypes differ, which is what produced the
      first item

The lesson, kept here because it is cheap to forget: **testing against the reference proves the
arithmetic, not the contract.** ``transformers`` knows what a rotary table is; only the encoder knows
what dtype it will be handed, which coordinate system its bounds live in, and how many items a block may
hold. Look for the plugin's own preprocessing before writing any.

The shape and dtype contract, for a block size ``B`` and ``N`` blocks, taken from the encoder's warmup
declaration (``build_vision_synthetic_inputs``) rather than inferred:

    pixel_values      [N, B, patch_dim]   bf16     patches, padded with zeros
    pos_emb_idx       [4, N, B]           int32    bilinear corners into the position-embedding table
    pos_emb_weight    [4, N, B]           bf16     interpolation weights, summing to one per patch
    cos / sin         [N, B, head_dim]    fp32     the vision rotary over (h, w) position ids
    bound_min/max     [N, B, 1]           int32    block-local patch range attention may see, per patch

Padding is not neutral by default: a padded row carries a position, a bilinear corner and a bound like
any other. The helpers pad ``pixel_values`` and the rotary with zeros, the bilinear weights with zero
(which is what makes the corner index irrelevant, and why an out-of-range index there would be an
undetectable out-of-bounds gather), and the bounds with an empty range, so a padded row sees nothing.
"""

from __future__ import annotations

import torch

from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
    compute_block_bounds,
    ffd_pack_images,
    scatter_to_blocks,
    select_vision_bucket,
)
from vllm_neuron.model.qwen3_vl.utils.vision_preprocessing import (
    compute_position_indices_and_weights,
    compute_rotary_pos_emb,
)


def patch_dim(vision_config) -> int:
    """Flat size of one patch: channels x temporal patch x patch x patch.

    Derived from the config rather than passed in, because it is the width the checkpoint's
    ``patch_embed.proj`` expects; a mismatch would surface as a shape error deep inside the encoder.
    """
    return int(
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size
        * vision_config.patch_size
    )


def vision_blocks(grid_thw: torch.Tensor, block_size: int, buckets: list[int],
                  dp_size: int = 1) -> int:
    """How many blocks the encoder graph will be handed, via the configured buckets.

    The count is a bucket property, not a token-count property: the graph is compiled per bucket, so a
    request needing three blocks runs the graph warmed for its bucket and pads the rest. Deriving it
    from ``ceil(tokens / block_size)`` instead would produce a shape no graph was warmed for.

    The bucket is selected from the number of blocks the packing NEEDS, expressed back as a token count,
    not from the token count itself. Items do not share blocks, so two images of 64 and 48 patches total
    112, select a 128 bucket, and that is one block where the packing needs two. The reference applies
    this correction for video (where whole-frame packing makes it obvious) and not for images, so a
    request carrying several small images reaches the packer one block short.

    The per-item sum is used rather than ``items * block_size`` because the two agree only while every
    item fits in one block. That rule is enforced by ``ffd_pack_images``, three calls away, and a bucket
    derivation whose correctness depends on a rule enforced elsewhere is the kind that survives the rule
    being relaxed.
    """
    per_item = grid_thw.prod(dim=1).tolist()
    blocks_needed = sum(-(-int(tokens) // block_size) for tokens in per_item)
    bucket, _num_blocks = select_vision_bucket(
        max(int(sum(per_item)), blocks_needed * block_size), buckets, block_size, dp_size=dp_size
    )
    return blocks_for_bucket(bucket, block_size, dp_size)


def blocks_for_bucket(bucket: int, block_size: int, dp_size: int = 1) -> int:
    """Blocks in the graph warmed for ``bucket``.

    One conversion, used by both callers. The warmup is HANDED a bucket by the runner and must not
    re-select one, and the serving path selects a bucket and must convert it the same way -- if the two
    disagree the warmed graph has a shape the real path never produces, and the mismatch surfaces as a
    load-time rejection at the first image, after the compile has been paid for.

    The rounding up to a multiple of ``dp_size`` is the encoder's: it scatters whole blocks across its
    data-parallel ranks and asserts divisibility.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive; got {dp_size}")
    blocks = -(-bucket // block_size)
    return -(-blocks // dp_size) * dp_size


def build_vision_inputs(pixel_values: torch.Tensor, grid_thw: torch.Tensor, vision_config,
                        block_size: int, num_blocks: int) -> dict[str, torch.Tensor]:
    """The six host-side tensors, keyed by the encoder's parameter names.

    ``num_blocks`` is required rather than derived: it has to be the count the graph was warmed for
    (see ``vision_blocks``), and silently choosing a different one would compile a second graph or fail
    at load with a shape the runtime cannot match.

    Items are packed one per block. That is the helper's rule and it is not an efficiency choice: the
    encoder writes block i into one cache block, and the cache allocates blocks per item, so a shared
    block has no valid destination. The helper raises when an item does not fit in a block.

    There is deliberately no ``theta`` argument. This checkpoint's ``vision_config`` does not carry one,
    so ``compute_rotary_pos_emb``'s own default is the single authority; taking it as a parameter would
    let the warmup path and the real path be given different rotary tables, which is not an error and
    not a shape mismatch -- it is a quietly degraded image embedding.
    """
    tokens_per_item = grid_thw.prod(dim=1).tolist()
    total = int(sum(tokens_per_item))
    if pixel_values.shape[0] != total:
        raise ValueError(
            f"pixel_values has {pixel_values.shape[0]} patches but grid_thw describes {total}; "
            "the packing below would place real patches in padded positions."
        )
    if grid_thw.numel() and int(grid_thw[:, 0].max()) > 1:
        # A video's frames are packed per frame, not per item, and the frame-fits-a-block floor is
        # checked against H*W rather than T*H*W. Refusing is better than packing a video as one item,
        # which forces block_size >= the whole video and quietly changes what a block means.
        raise NotImplementedError(
            "build_vision_inputs handles images (T == 1); video needs the per-frame expansion the "
            "reference performs before packing."
        )

    assignment = ffd_pack_images(
        tokens_per_item, block_size, num_blocks, one_item_per_block=True
    )
    head_dim = vision_config.hidden_size // vision_config.num_heads
    merge = vision_config.spatial_merge_size
    grid_side = int(vision_config.num_position_embeddings ** 0.5)

    cos, sin = compute_rotary_pos_emb(grid_thw, head_dim, merge)
    corners, weights = compute_position_indices_and_weights(grid_thw, grid_side, merge)
    bound_min, bound_max = compute_block_bounds(tokens_per_item, assignment, grid_thw)

    return {
        "pixel_values": scatter_to_blocks(
            pixel_values.to(torch.bfloat16), tokens_per_item, assignment),
        # The corner/weight tensors arrive as [total, 4] and the encoder takes [4, N, B], so the
        # transpose happens before the scatter and the permute after it. Writing it as one expression
        # would make the two easy to swap.
        "pos_emb_idx": scatter_to_blocks(
            corners.T, tokens_per_item, assignment).permute(2, 0, 1).contiguous().to(torch.int32),
        "pos_emb_weight": scatter_to_blocks(
            weights.T.to(torch.bfloat16), tokens_per_item, assignment
        ).permute(2, 0, 1).contiguous(),
        # fp32, as the graph declares. Handing bf16 here is the defect this module was rewritten for.
        "cos": scatter_to_blocks(cos.to(torch.float32), tokens_per_item, assignment),
        "sin": scatter_to_blocks(sin.to(torch.float32), tokens_per_item, assignment),
        "bound_min": bound_min,
        "bound_max": bound_max,
    }


def cache_write_destinations(cache_block_map: list[list[int]], num_blocks: int,
                             scratch_block_id: int) -> torch.Tensor:
    """Map each encoder output block to the cache block it writes into.

    Items own contiguous runs of encoder blocks in item order, so flattening the per-item cache block
    ids in that order gives the 1:1 mapping. That is not an assumption about the packer's output order:
    ``_pack_one_item_per_block`` assigns blocks sequentially in item input order and says so, precisely
    so this flattening works. (The First-Fit-*Decreasing* sort in ``ffd_pack_images`` applies only to the
    shared-block path, which this port never takes.)

    Encoder blocks beyond the real ones are aimed at the scratch block: they must land *somewhere*, and a
    scratch destination is what keeps a padded block from overwriting a live cache entry.

    The dtype is int64 because the encoder indexes its cache with it. This builder is the only place the
    tensor is constructed for the real path; the runner builds its own for warmup, so if that one ever
    disagrees the mismatch shows up at load rather than silently.
    """
    flat = [block for blocks in cache_block_map for block in blocks]
    if len(flat) > num_blocks:
        raise ValueError(
            f"the cache allocated {len(flat)} blocks but the graph takes {num_blocks}; the extra "
            "blocks would be dropped and their tokens read back as whatever the cache held."
        )
    if scratch_block_id in flat:
        # A real destination equal to the scratch block would put a padded encoder block and a live one
        # in the same place. Which of the two lands last is not defined, so the image would be encoded
        # correctly and then partly overwritten with zeros -- readable output, wrong content.
        raise ValueError(
            f"the cache allocated block {scratch_block_id}, which is also the scratch block; a padded "
            "encoder block and a real one would write to the same place."
        )
    return torch.tensor(flat + [scratch_block_id] * (num_blocks - len(flat)), dtype=torch.int64)
