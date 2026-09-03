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


def rotary_tables(rotary_dim, max_position, theta, device=None, dtype=torch.float32):
    """cos/sin tables for the PARTIAL rotary embedding, shape ``[max_position, rotary_dim]``.

    ``rotary_dim`` is ``head_dim * partial_rotary_factor`` (64 of 256 here), so only the leading
    ``rotary_dim`` channels of q/k rotate.

    A single axis is exact for text and only for text: the reference interleaves three positional axes,
    which for a text prompt all carry the same position, so the interleave reduces to this table. Image
    or video input breaks that equality — it is not an approximation there, it is wrong.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device,
                                            dtype=torch.float32) / rotary_dim))
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)                     # [max_position, rotary_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)                      # [max_position, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


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
