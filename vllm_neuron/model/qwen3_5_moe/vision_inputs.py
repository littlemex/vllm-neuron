"""Host-side tensors the vision encoder's forward requires.

The encoder itself is reusable as-is — this checkpoint's vision keys are identical to the Qwen3-VL
reference in this repository — but its ``forward`` takes nine arguments and six of them are prepared on
the host. That preparation is what this module is. It is deliberately separate from the model: it runs
outside the compiled graph, on CPU, so it can be tested against the reference helpers without a device.

The shape contract the encoder expects, for a block size ``B`` and ``N`` blocks:

    pixel_values      [N, B, patch_dim]      patches, padded with zeros
    pos_emb_idx       [4, N, B]  int32       bilinear corners into the position-embedding table
    pos_emb_weight    [4, N, B]              interpolation weights, summing to one per patch
    cos / sin         [N, B, head_dim]       the vision rotary, from (h, w) position ids
    bound_min/max     [N, B, 1] int32        the patch range attention may see, per patch

Everything is padded to a fixed ``B`` because the graph is compiled for one. **Padding is not neutral by
default**: a padded row carries a position, a bilinear corner and a bound like any other, so each of
those has to be given a value that keeps it out of the result. The choices are made explicit below,
one per function, because getting them wrong produces a plausible embedding rather than an error.

The per-patch quantities come from ``transformers.vision_utils``: ``get_vision_cu_seqlens``,
``get_vision_position_ids`` and ``get_vision_bilinear_indices_and_weights``. Those are the reference, so
they are called rather than reimplemented — a reimplementation would have to be tested against them
anyway, and a copy that agrees on the tested cases and diverges elsewhere is the worst outcome.
"""
from __future__ import annotations

import torch


def patch_dim(vision_config) -> int:
    """Flat size of one patch: channels x temporal patch x patch x patch.

    This is the width the checkpoint's ``patch_embed.proj`` expects, so it is derived from the config
    rather than passed in; a mismatch here would show up as a shape error deep inside the encoder.
    """
    return (vision_config.in_channels * vision_config.temporal_patch_size
            * vision_config.patch_size * vision_config.patch_size)


def block_count(total_patches: int, block_size: int) -> int:
    """How many blocks of ``block_size`` are needed for ``total_patches``, rounding up.

    At least one, so an empty request still produces a well-formed (entirely padded) input rather than a
    zero-length dimension, which the compiler treats differently from a padded one.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")
    return max(1, -(-total_patches // block_size))


def _pad_to_blocks(flat: torch.Tensor, block_size: int, fill) -> torch.Tensor:
    """Reshape ``[total, ...]`` into ``[blocks, block_size, ...]``, padding the tail with ``fill``."""
    total = flat.shape[0]
    blocks = block_count(total, block_size)
    padded_len = blocks * block_size
    if padded_len != total:
        tail_shape = (padded_len - total,) + tuple(flat.shape[1:])
        tail = torch.full(tail_shape, fill, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, tail], dim=0)
    return flat.reshape(blocks, block_size, *flat.shape[1:])


def blocked_pixel_values(pixel_values: torch.Tensor, block_size: int) -> torch.Tensor:
    """``[total_patches, patch_dim]`` -> ``[blocks, block_size, patch_dim]``.

    Padded with ZEROS. A zero patch is not inert — ``patch_embed.proj`` has a bias, so a zero patch
    still produces the bias vector — but it does not have to be inert: the bounds keep real patches from
    attending to it, and the merger's output for a padded position is never written to the cache. What
    matters is that it is deterministic, so a padded run and a shorter unpadded one agree on the real
    positions.
    """
    if pixel_values.dim() != 2:
        raise ValueError(f"expected [total_patches, patch_dim]; got {tuple(pixel_values.shape)}")
    return _pad_to_blocks(pixel_values, block_size, 0)


def blocked_bilinear(indices: torch.Tensor, weights: torch.Tensor,
                     block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(4, total)`` corners and weights -> ``(4, blocks, block_size)`` each.

    Corners are padded with index 0 and weights with ZERO. The zero weight is the load-bearing half: a
    padded position then gathers the table's first row and multiplies it by nothing, so the choice of
    index cannot matter. Padding the index with something out of range instead would be an out-of-bounds
    gather, which on a compiled graph is not reliably an error.
    """
    if indices.shape != weights.shape or indices.shape[0] != 4:
        raise ValueError(
            f"expected matching (4, total) shapes; got {tuple(indices.shape)} and "
            f"{tuple(weights.shape)}"
        )
    blocked_indices = torch.stack(
        [_pad_to_blocks(indices[corner], block_size, 0) for corner in range(4)])
    blocked_weights = torch.stack(
        [_pad_to_blocks(weights[corner], block_size, 0) for corner in range(4)])
    return blocked_indices.to(torch.int32), blocked_weights


def vision_rotary_tables(position_ids: torch.Tensor, head_dim: int, theta: float,
                         dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for the vision rotary, from ``(total, 2)`` height/width position ids.

    Two axes, not three: the vision encoder rotates height and width, and the temporal axis is carried
    by repeating the h/w indices per frame rather than by a third rotary axis (which is why
    ``get_vision_position_ids`` returns two columns for this family). The two axes take half the
    frequencies each, so the table is built per axis over ``head_dim // 4`` frequencies and concatenated
    — the same non-interleaved halves layout the text side uses.

    Built in fp32 and cast at the end: the outer product of a position with an inverse frequency is
    where bf16 loses the most.
    """
    if position_ids.dim() != 2 or position_ids.shape[1] != 2:
        raise ValueError(f"expected (total, 2) height/width ids; got {tuple(position_ids.shape)}")
    if head_dim % 4:
        raise ValueError(f"head_dim must be divisible by 4 for a two-axis rotary; got {head_dim}")
    per_axis = head_dim // 4
    inv_freq = 1.0 / (theta ** (torch.arange(0, per_axis, dtype=torch.float32) / per_axis))
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq   # [total, 2, per_axis]
    freqs = freqs.reshape(position_ids.shape[0], -1)                 # [total, head_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)                          # [total, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def blocked_rotary(cos: torch.Tensor, sin: torch.Tensor,
                   block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[total, head_dim]`` -> ``[blocks, block_size, head_dim]``, padded with zeros.

    A zero angle means cos 0 and sin 0 rather than cos 0 and sin... which is to say the padded rows do
    not encode position zero, they encode a degenerate rotation. That is fine because the bounds exclude
    them, and it keeps the padding value the same as everywhere else here — one rule for all the padded
    tensors is worth more than a slightly more principled per-tensor choice.
    """
    return _pad_to_blocks(cos, block_size, 0), _pad_to_blocks(sin, block_size, 0)


def blocked_bounds(cu_seqlens: torch.Tensor, total_patches: int,
                   block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch attention bounds from image boundaries -> ``[blocks, block_size, 1]`` int32 each.

    ``cu_seqlens`` is the cumulative patch count per image, so patch ``i`` belongs to the segment whose
    half-open range contains it, and may attend to exactly that range. This is what stops one image's
    patches from attending to another's when several are packed into the same blocks — and a block
    boundary falling inside an image is fine, because the bound is per patch rather than per block.

    Padded positions get ``bound_min == bound_max == 0``, an empty range: whatever a padded row computes,
    it sees nothing. That is the choice that makes the padding harmless rather than merely deterministic.
    """
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"expected a cumulative boundary vector; got {tuple(cu_seqlens.shape)}")
    boundaries = cu_seqlens.to(torch.int64)
    starts = boundaries[:-1]
    ends = boundaries[1:]
    lengths = ends - starts
    if int(lengths.sum()) != total_patches:
        raise ValueError(
            f"cu_seqlens covers {int(lengths.sum())} patches but total_patches is {total_patches}"
        )
    per_patch_min = torch.repeat_interleave(starts, lengths).to(torch.int32).unsqueeze(-1)
    per_patch_max = torch.repeat_interleave(ends, lengths).to(torch.int32).unsqueeze(-1)
    return (_pad_to_blocks(per_patch_min, block_size, 0),
            _pad_to_blocks(per_patch_max, block_size, 0))


def build_vision_inputs(pixel_values, grid_thw, vision_config, block_size, rope_theta=10000.0,
                        dtype: torch.dtype = torch.bfloat16):
    """Everything the encoder's forward needs except the cache, from the runner's raw inputs.

    Returns a dict keyed by the encoder's argument names, so the call site cannot pair the wrong tensor
    with the wrong parameter — six positional tensors of similar rank is exactly the shape of mistake
    that type checking does not catch.
    """
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights,
        get_vision_cu_seqlens,
        get_vision_position_ids,
    )

    total_patches = pixel_values.shape[0]
    merge = vision_config.spatial_merge_size
    grid_side = int(vision_config.num_position_embeddings ** 0.5)

    cu_seqlens = get_vision_cu_seqlens(grid_thw)
    position_ids = get_vision_position_ids(grid_thw, merge)
    indices, weights = get_vision_bilinear_indices_and_weights(grid_thw, grid_side, merge)

    cos, sin = vision_rotary_tables(position_ids, vision_config.head_dim, rope_theta, dtype=dtype)
    blocked_cos, blocked_sin = blocked_rotary(cos, sin, block_size)
    pos_emb_idx, pos_emb_weight = blocked_bilinear(indices, weights.to(dtype), block_size)
    bound_min, bound_max = blocked_bounds(cu_seqlens, total_patches, block_size)
    return {
        "pixel_values": blocked_pixel_values(pixel_values, block_size),
        "pos_emb_idx": pos_emb_idx,
        "pos_emb_weight": pos_emb_weight,
        "cos": blocked_cos,
        "sin": blocked_sin,
        "bound_min": bound_min,
        "bound_max": bound_max,
    }
