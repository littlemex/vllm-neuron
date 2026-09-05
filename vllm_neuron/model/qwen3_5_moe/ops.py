# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE small ops that the model and the CPU tests must share.

Kept dependency-light (torch only) and treated as the SINGLE SOURCE OF TRUTH so the tests can pin
them against the actual HF reference modules rather than against a re-derivation in the test file.

Two norm conventions coexist in this architecture and mixing them is silent breakage:

* ``rmsnorm`` (HF ``Qwen3_5MoeRMSNorm``) scales by ``1 + weight`` and its weight is initialised to
  ZERO. Using the plugin's usual ``weight * x`` here would multiply the residual stream by zero.
* ``gated_rmsnorm`` (HF ``Qwen3_5MoeRMSNormGated``, used only inside the Gated DeltaNet) scales by
  ``weight`` and its weight is initialised to ONE, and it applies the gate AFTER the norm.
"""
import os
from typing import NamedTuple

import torch
import torch.nn.functional as F


def rmsnorm(x, weight, eps):
    """HF ``Qwen3_5MoeRMSNorm``: ``rms_normalise(x) * (1 + weight)``, computed in fp32.

    Note the ``1 +``: this checkpoint's norm weights are stored as offsets from unity.
    """
    normed = x.float()
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).type_as(x)


def gated_rmsnorm(x, gate, weight, eps, out_dtype):
    """HF ``Qwen3_5MoeRMSNormGated``: norm BEFORE gate, ``weight * normed``, gate through SiLU.

    The dtype order follows the reference: round to the model dtype before applying the weight,
    evaluate the gate's SiLU in fp32. Staying in fp32 throughout would be more precise but would make a
    greedy comparison against HF unattributable.

    ``x``/``gate`` are ``[..., head_v_dim]``: this norm is per value head, not grouped.
    """
    normed = x.float()
    normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + eps)
    normed = weight.to(out_dtype) * normed.to(out_dtype)
    return (normed * F.silu(gate.float())).to(out_dtype)


def rotary_inv_freq(rotary_dim, theta, device=None):
    """The inverse frequencies for the PARTIAL rotary embedding, shape ``[rotary_dim // 2]``.

    ``rotary_dim`` is ``head_dim * partial_rotary_factor`` (64 of 256 here), so only the leading
    ``rotary_dim`` channels of q/k rotate.
    """
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device,
                                         dtype=torch.float32) / rotary_dim))


def rotary_tables(rotary_dim, max_position, theta, device=None, dtype=torch.float32):
    """cos/sin tables for a SINGLE positional axis, shape ``[max_position, rotary_dim]``.

    A single axis is exact for text and only for text: the reference interleaves three positional axes,
    which for a text prompt all carry the same position, so the interleave reduces to this table. Image
    or video input breaks that equality — it is not an approximation there, it is wrong. Use
    ``mrope_tables`` when the three axes can differ.
    """
    inv_freq = rotary_inv_freq(rotary_dim, theta, device=device)
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)                     # [max_position, rotary_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)                      # [max_position, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def interleave_mrope(freqs, mrope_section):
    """Interleave the three positional axes into one frequency vector.

    ``freqs`` is ``[3, ..., rotary_dim // 2]`` (one plane per axis: time, height, width) and the result
    is ``[..., rotary_dim // 2]``. Slot ``i`` takes the height axis when ``i % 3 == 1`` and the width
    axis when ``i % 3 == 2``, in each case only while ``i`` is inside that axis's section
    (``mrope_section[axis] * 3`` slots). Every other slot keeps the time axis, which is why a text
    prompt — where all three axes carry the same position — collapses back to the single-axis table.

    Written with ``torch.where`` rather than slice assignment: an in-place write into a traced tensor is
    a mutation the graph has to model, and the mask form is what the reference uses for the same reason.
    """
    slots = freqs.shape[-1]
    index = torch.arange(slots, device=freqs.device, dtype=torch.int64)
    out = freqs[0]
    for axis, offset in ((1, 1), (2, 2)):
        inside = (index % 3 == offset) & (index < mrope_section[axis] * 3)
        out = torch.where(inside, freqs[axis], out)
    return out


def mrope_tables(positions, rotary_dim, theta, mrope_section, dtype=torch.float32):
    """cos/sin for THREE positional axes, from ``positions`` of shape ``[3, tokens]``.

    Returns ``[tokens, rotary_dim]`` tables, ready for ``apply_partial_rotary``. The frequencies are
    built in fp32 and cast at the end, matching the reference: the outer product of a position index
    with an inverse frequency loses too much in bf16 at long positions.
    """
    inv_freq = rotary_inv_freq(rotary_dim, theta, device=positions.device)
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq   # [3, tokens, rotary_dim // 2]
    interleaved = interleave_mrope(freqs, mrope_section)           # [tokens, rotary_dim // 2]
    emb = torch.cat((interleaved, interleaved), dim=-1)            # [tokens, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def temporal_axis(positions):
    """The temporal axis of the MRoPE positions, whether they arrive as ``[3, T]`` or ``[T]``.

    This is the axis to take when one position per token is wanted, but **it is not a monotone token
    index and must not be used as one.** Within an image span the reference position builder emits
    ``np.indices((1, h, w))``, so the temporal axis is CONSTANT across the whole image while height and
    width vary. Height then repeats per row and width goes backwards at each new row, so no axis is a
    per-token index inside an image; the three only advance together again after the span.

    What consumes this in this model is narrow enough for that to be safe, and the narrowness is the
    reason rather than a coincidence:

    * prefill does not read it at all -- the attention's prefill path takes the rotary tables and the
      attention metadata, and the KV slot comes from ``slot_mapping``, not from a position
    * decode reads it for one token, in the causal mask (``context_index <= position``), and a decode
      step is always past the image, where the three axes have advanced together
    * the recurrent layers' fresh-request test reads only whether the FIRST position is zero

    Anything new that needs a monotone per-token index has to derive it elsewhere. Returned as int32
    because that is what the attention metadata and the cache indexing expect; the rotary takes the
    un-narrowed positions separately.
    """
    if positions.dim() == 2 and positions.shape[0] == 3:  # lint-port: ok dim and shape are graph-static, not tensor contents
        positions = positions[0]
    return positions.to(torch.int32)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rotary(x, cos, sin):
    """Rotate the leading ``cos.shape[-1]`` channels of ``x`` and pass the rest through unchanged.

    ``x`` is ``[..., head_dim]`` and ``cos``/``sin`` are ``[..., rotary_dim]`` broadcastable onto it.
    Matches HF ``apply_rotary_pos_emb`` for this architecture (non-interleaved halves, partial width).
    """
    rotary_dim = cos.shape[-1]
    rotated = x[..., :rotary_dim]
    passthrough = x[..., rotary_dim:]
    rotated = rotated * cos + _rotate_half(rotated) * sin
    if passthrough.shape[-1] == 0:
        return rotated.to(x.dtype)
    return torch.cat([rotated.to(x.dtype), passthrough], dim=-1)


def gated_delta_projections(a, b, a_log, dt_bias):
    """The GDN decay and beta from the ``in_proj_a`` / ``in_proj_b`` outputs.

    ``g = -exp(A_log) * softplus(a + dt_bias)`` (a log decay, so always <= 0) and
    ``beta = sigmoid(b)``.

    ``g`` must be fp32: in bf16 the ``exp`` of a large ``A_log`` overflows to -inf and poisons the whole
    recurrence. ``beta`` deliberately stays in the incoming dtype, as the reference has it — upcasting
    would be more precise but no longer bit-comparable to HF.

    ``a``/``b`` are ``[..., num_v_heads]``; ``a_log``/``dt_bias`` are ``[num_v_heads]``.
    """
    beta = b.sigmoid()
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, beta


def concat_draft_inputs(normed_embeddings, normed_hidden):
    """Concatenate the multi-token-prediction head's two inputs in the reference's order.

    The embedding is the FIRST half and the hidden state the second, as in vLLM's
    ``Qwen3_5MultiTokenPredictor.forward``. This is one line with a name and a test because it is the
    one step of that head that cannot be inferred from the checkpoint: the swapped order has the same
    shape, loads the same ``fc`` weight without complaint, and reads the wrong learned columns of it —
    the output stays fluent and is wrong. A named function means the served path cannot be written in
    the other order without editing something a test is looking at.

    Both are ``[..., hidden]``; the result is ``[..., 2 * hidden]``, which is what ``fc`` consumes.
    """
    if normed_embeddings.shape != normed_hidden.shape:  # lint-port: ok shapes are graph-static
        raise ValueError(
            f"the embedding and the hidden state must have the same shape; got "
            f"{tuple(normed_embeddings.shape)} and {tuple(normed_hidden.shape)}"
        )
    return torch.cat([normed_embeddings, normed_hidden], dim=-1)


def paged_decode_attention(query, key_dense, value_dense, positions, scaling, groups, batch):
    """Decode attention over a densely gathered, padded context, for any number of sequences.

    Shapes, all of which have to be spelled out because the single-sequence form let two of them coincide:

    * ``query``      ``[heads, batch * decode_tokens, head_dim]`` -- the projection's layout
    * ``key_dense``  ``[batch * kv_heads, context_len, head_dim]`` -- the gather's layout
    * ``positions``  ``[batch * decode_tokens]`` -- each row's absolute position
    * returns        ``[batch * decode_tokens, heads * head_dim]``

    The previous implementation multiplied ``query`` (leading axis ``heads``) by the expanded keys
    (leading axis ``batch * heads``) and masked with ``positions`` flattened. At ``batch == 1`` those two
    leading axes are the same number and the mask has one row per token, so it was correct; at any larger
    batch it would have multiplied the wrong pairs. That is why it refused rather than being "probably
    fine": the shapes agree by coincidence, so nothing raises.

    fp32 throughout, as the caller's docstring explains: the asymmetric ``q=1`` / ``k=S_ctx`` shape is not
    what the flash-attention kernel takes, and fp32 removes bf16 accumulation drift over a long context.
    """
    heads, total_tokens, head_dim = query.shape
    if total_tokens % batch:  # lint-port: ok shapes are graph-static; this is a wiring error, not data
        raise ValueError(f"{total_tokens} decode rows do not divide into {batch} sequences")
    decode_tokens = total_tokens // batch
    context_len = key_dense.shape[1]

    # Batch-major everywhere. The keys arrive as [batch * kv_heads, ...] with batch the OUTER axis, so
    # expanding the groups keeps each sequence's heads together.
    key_full = key_dense.repeat_interleave(groups, dim=0).to(torch.float32)
    value_full = value_dense.repeat_interleave(groups, dim=0).to(torch.float32)
    key_full = key_full.view(batch, heads, context_len, head_dim)
    value_full = value_full.view(batch, heads, context_len, head_dim)

    # The query's token axis is batch-major too: row b * decode_tokens + t belongs to sequence b.
    query = (query.to(torch.float32)
             .view(heads, batch, decode_tokens, head_dim)
             .permute(1, 0, 2, 3))                              # [batch, heads, decode_tokens, head_dim]

    scores = torch.matmul(query, key_full.transpose(-2, -1)) * scaling
    # Per-row causal mask over the padded context: each row attends up to its OWN absolute position.
    # Masking with the last position alone would let earlier rows of a decode bucket read future keys.
    context_index = torch.arange(context_len, device=scores.device)
    valid = context_index.view(1, 1, -1) <= positions.view(batch, decode_tokens, 1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(weights, value_full)                     # [batch, heads, decode_tokens, head_dim]
    # Back to the caller's row-major layout: [batch * decode_tokens, heads * head_dim].
    return out.permute(0, 2, 1, 3).reshape(batch * decode_tokens, heads * head_dim)


def redirect_padded_slots(slot_mapping, rows):
    """Make a padded paged-cache scatter collision-proof.

    Neuron right-pads a prefill and marks the pad tokens with ``slot_mapping = -1``.

    TRAP 1: that sentinel must never reach the block arithmetic. ``-1 // block_size`` is ``-1``, which
    resolves to the LAST block of the cache.
    TRAP 2: a fixed "reserved" sink slot does not fix it — nothing reserves a physical block, so the
    allocator may have given the active sequence that one, and a pad-versus-real collision leaves
    ``index_put_``'s duplicate-index order deciding whether the real value survives.

    So pad rows take a slot that is guaranteed REAL together with that row's value: every duplicate
    writes the same thing. The donor is the token holding the largest slot (real, because pads are
    negative), selected with reductions rather than a gather on a runtime index — that op class is
    miscompiled once layers stack. An all-padding batch (warmup only) averages the rows and writes to
    slot 0.

    Args:
        slot_mapping: ``[T]`` integer slots, negative for padding.
        rows: ``[T, width]`` the per-token values to scatter.
    Returns:
        ``(safe_slot[T], safe_rows[T, width])`` with the pad entries replaced.
    """
    real = slot_mapping >= 0
    donor_slot = slot_mapping.max().reshape(1)
    selector = (slot_mapping == donor_slot).to(rows.dtype).unsqueeze(-1)   # [T, 1]
    donor_row = (rows * selector).sum(dim=0, keepdim=True) / selector.sum().clamp(min=1)
    # `where` broadcasts the single donor row over the token axis; no explicit `expand` is created.
    safe_slot = torch.where(real, slot_mapping, donor_slot.clamp(min=0))
    safe_rows = torch.where(real.unsqueeze(-1), rows, donor_row)
    return safe_slot, safe_rows


# ---------------------------------------------------------------------------
# Per-slot recurrent state
# ---------------------------------------------------------------------------
# The Gated DeltaNet's conv and recurrent state carry a leading SLOT axis, one entry per concurrent
# sequence. Selecting a slot with `state[slot]` would be a data-dependent index into a compiled graph,
# which is the thing that does not reliably trace; a one-hot multiply is the same select written as
# arithmetic, and it is the form already used for speculative candidate selection.
#
# `num_slots` is a Python int, so the branch on it is resolved at trace time and no runtime condition
# enters the graph. At one slot the functions are the identity and the assignment they replaced, so the
# single-slot graph is the same graph as before slots existed.


def slot_mask(slot, state: torch.Tensor, num_slots: int):
    """One-hot over the slot axis, shaped to broadcast onto ``state``; None when there is one slot."""
    if num_slots < 1:
        raise ValueError(f"num_slots must be at least 1; got {num_slots}")
    if num_slots == 1:
        return None
    wanted = torch.as_tensor(slot, device=state.device)
    if wanted.numel() != 1:
        # The runner supplies one slot per request in the step. A step carrying several requests needs
        # the scan segmented per request, which is not implemented -- so this refuses rather than taking
        # the first slot, which would run every request through one sequence's state and produce fluent
        # output with the wrong history.
        raise NotImplementedError(
            f"got {wanted.numel()} slots for one step; the Gated DeltaNet scan handles one sequence at "
            "a time, so a multi-request step is not supported yet (see docs/DESIGN-concurrency.md)."
        )
    index = torch.arange(num_slots, device=state.device)
    mask = (index == wanted.reshape(()).to(index.dtype))
    return mask.to(state.dtype).reshape((num_slots,) + (1,) * (state.dim() - 1))


def _use_index_select() -> bool:
    """Whether to select a slot with ``index_select`` instead of a one-hot product.

    The one-hot form was chosen because a data-dependent index was assumed not to trace. That assumption
    is broader than the evidence: this repository already indexes with a runtime value elsewhere (the
    conv history is gathered at a runtime length). Which form is correct on the device is a measurement,
    so both exist and this chooses.

    The one-hot form also has two costs the index form does not. It materialises the WHOLE pool per read
    (at 19539 slots that is hundreds of megabytes per layer per step), and it is not robust to a
    non-finite value in an unused slot, because ``0 * NaN`` is NaN and the sum carries it.
    """
    return os.environ.get("QWEN3_5_MOE_SLOT_INDEX", "1") == "1"


def read_state_slot(state: torch.Tensor, slot, num_slots: int) -> torch.Tensor:
    """This slot's state as ``[1, ...]`` -- the shape the scans take."""
    if num_slots == 1:
        return state
    if _use_index_select():
        # int64 explicitly: index_select accepts int32 but index_copy_ does not, and a form that reads
        # with one dtype and writes with another is the kind of asymmetry that shows up only on the write.
        return torch.index_select(
            state, 0, torch.as_tensor(slot, device=state.device).reshape(1).long())
    mask = slot_mask(slot, state, num_slots)
    if mask is None:
        return state
    return (state * mask).sum(dim=0, keepdim=True)


def write_state_slot(state: torch.Tensor, updated: torch.Tensor, slot, num_slots: int) -> None:
    """Write ``updated`` (``[1, ...]``) into this slot in place, leaving the other slots untouched.

    The ``copy_`` target is the WHOLE buffer, not a slice. That is deliberate: the aliasing pass
    recognises a full-tensor in-place write, and an in-place write into a slice is not known to produce
    the same alias. The other slots are preserved by the arithmetic, not by writing less.
    """
    if num_slots == 1:
        state.copy_(updated.to(state.dtype))
        return
    if _use_index_select():
        # index_copy_ writes only the named row. The whole-buffer copy_ the one-hot form needs exists to
        # keep the aliasing pass looking at a full-tensor write; whether index_copy_ satisfies it is a
        # device question, which is why both forms are kept and this is selectable.
        state.index_copy_(0, torch.as_tensor(slot, device=state.device).reshape(1).long(),
                          updated.to(state.dtype))
        return
    mask = slot_mask(slot, state, num_slots)
    if mask is None:
        state.copy_(updated.to(state.dtype))
        return
    state.copy_(state * (1 - mask) + updated.to(state.dtype) * mask)


def slots_onehot(slots, num_slots: int, dtype: torch.dtype, device) -> torch.Tensor:
    """``[requests, num_slots]``, one row per request with a single 1 at its slot.

    The batched counterpart of ``slot_mask``, and the same reason for existing: a gather by a runtime
    index does not reliably trace, so the gather is written as a matrix product.
    """
    if num_slots < 1:
        raise ValueError(f"num_slots must be at least 1; got {num_slots}")
    wanted = torch.as_tensor(slots, device=device).reshape(-1)
    index = torch.arange(num_slots, device=device)
    return (wanted.unsqueeze(1).to(index.dtype) == index.unsqueeze(0)).to(dtype)


def read_state_slots(state: torch.Tensor, slots, num_slots: int) -> torch.Tensor:
    """The rows of ``state`` named by ``slots``, as ``[requests, *state.shape[1:]]``.

    Written as ``onehot @ flattened`` rather than ``state[slots]``. Same trade as the single-slot form:
    a matrix product against a one-hot costs ``requests x num_slots`` multiplies and traces, where the
    index does not.
    """
    onehot = slots_onehot(slots, num_slots, state.dtype, state.device)
    gathered = onehot @ state.reshape(num_slots, -1)
    return gathered.reshape((onehot.shape[0],) + tuple(state.shape[1:]))


def write_state_slots(state: torch.Tensor, updated: torch.Tensor, slots, num_slots: int) -> None:
    """Write one row of ``updated`` into each slot named by ``slots``, in place.

    Slots not named keep their value, by arithmetic rather than by writing less -- the ``copy_`` target
    stays the whole buffer so the aliasing pass sees what it sees for the single-slot form.

    **The slots must be distinct.** The scatter is a transposed one-hot product, so a slot named twice
    receives the SUM of both rows rather than the last one. The runner allocates one block per request
    from this group, so duplicates cannot arise there; checking it here would need a data-dependent
    comparison, which is the thing this whole construction avoids.
    """
    onehot = slots_onehot(slots, num_slots, state.dtype, state.device)
    written = (onehot.t() @ updated.reshape(onehot.shape[0], -1)).reshape(state.shape)
    touched = onehot.sum(dim=0).reshape((num_slots,) + (1,) * (state.dim() - 1))
    state.copy_(state * (1 - touched) + written)


# ---------------------------------------------------------------------------
# Chunk-aligned packing for multi-request prefill
# ---------------------------------------------------------------------------


class AlignedLayout(NamedTuple):
    """Everything the Gated DeltaNet needs to prefill several requests packed into one row.

    Named rather than positional because six index tensors of similar shape are exactly the kind of
    argument list this port has already been bitten by once, in the vision encoder's inputs.

        source      ``[aligned]``            which packed token each aligned position holds
        valid       ``[aligned]``            1 where the position holds a real token
        carry       ``[aligned // chunk]``   0 where a chunk begins a request
        segment_id  ``[aligned]``            which request owns each aligned position
        to_packed   ``[packed]``             which aligned position each packed token ended up at
        conv_tails  ``[requests, kernel-1]`` the aligned positions holding each request's conv history
    """

    source: torch.Tensor
    valid: torch.Tensor
    carry: torch.Tensor
    segment_id: torch.Tensor
    to_packed: torch.Tensor
    conv_tails: torch.Tensor


def chunk_aligned_layout(query_start_loc, chunk_size: int, aligned_len: int,
                         kernel: int) -> AlignedLayout:
    """Where each request's tokens go when every request must start on a chunk boundary.

    The prefill scan can only be told about request boundaries per CHUNK, so a packed row has to be
    re-laid-out with each request starting at a multiple of ``chunk_size``. This computes that layout.

    ``query_start_loc`` is ``[requests + 1]``: the packed start offset of each request and the total,
    which is how vLLM already describes a packed batch. ``aligned_len`` is the STATIC size of the aligned
    buffer -- it has to be a compile-time constant because the graph is compiled for it, and it must be
    at least ``tokens + requests * (chunk_size - 1)`` for the padding to fit.

    No data-dependent control flow. The request each position belongs to is found by COUNTING how many
    request starts lie at or before it, which is a comparison against a small static-shaped matrix --
    the same trade as the one-hot slot select, and for the same reason.
    """
    if aligned_len % chunk_size:
        raise ValueError(
            f"aligned_len={aligned_len} must be a multiple of chunk_size={chunk_size}; the carry mask "
            "is one entry per whole chunk."
        )
    starts = torch.as_tensor(query_start_loc).reshape(-1)
    if starts.numel() < 2:
        raise ValueError(
            f"query_start_loc must hold at least one request plus the total; got {starts.numel()} entry"
        )
    device = starts.device
    requests = starts.numel() - 1
    lengths = starts[1:] - starts[:-1]                                  # [requests]
    padded = -(-lengths // chunk_size) * chunk_size                     # each rounded up to a chunk
    aligned_starts = torch.cumsum(padded, dim=0) - padded               # [requests]

    position = torch.arange(aligned_len, device=device)
    # How many request starts are at or before this position, minus one: the owning request. Positions
    # past the last request's aligned span clamp to the last request and are excluded by `valid`.
    owner = (position.unsqueeze(1) >= aligned_starts.unsqueeze(0)).sum(dim=1) - 1
    owner = owner.clamp(min=0, max=requests - 1)

    offset = position - aligned_starts[owner]
    valid = ((offset >= 0) & (offset < lengths[owner])).to(torch.int32)
    total = int(starts[-1])
    source = ((starts[:-1][owner] + offset) * valid).clamp(min=0, max=max(total - 1, 0))

    chunk_starts = torch.arange(0, aligned_len, chunk_size, device=device)
    # A chunk continues the previous request unless it begins one. Chunk 0 always begins one.
    carry = (chunk_starts.unsqueeze(1) != aligned_starts.unsqueeze(0)).all(dim=1).to(torch.int32)
    carry[0] = 0

    # The inverse map, for putting the output back where the caller expects it. Built the same way:
    # a packed token's request is found by counting starts, and its aligned position follows.
    packed_position = torch.arange(total, device=device)
    packed_owner = (packed_position.unsqueeze(1) >= starts[:-1].unsqueeze(0)).sum(dim=1) - 1
    packed_owner = packed_owner.clamp(min=0, max=requests - 1)
    to_packed = aligned_starts[packed_owner] + (packed_position - starts[:-1][packed_owner])

    # Each request's conv history: the last kernel-1 REAL positions of its aligned span. A request
    # shorter than the window reaches back into its own padding, which is zero -- the same left pad a
    # fresh single-request prefill gets.
    tail_offsets = torch.arange(-(kernel - 1), 0, device=device)
    conv_tails = (aligned_starts.unsqueeze(1) + lengths.unsqueeze(1) + tail_offsets.unsqueeze(0))
    conv_tails = conv_tails.clamp(min=0, max=aligned_len - 1)
    return AlignedLayout(source=source, valid=valid, carry=carry, segment_id=owner * valid,
                         to_packed=to_packed, conv_tails=conv_tails)
