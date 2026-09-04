# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE Gated DeltaNet: the scans and the mixer, in torch only (no Neuron/plugin deps).

Dependency-light on purpose, so this is the single source of truth: the served model imports from
here and the CPU equivalence tests load this file directly.

The recurrence, per head, with ``S`` the ``[head_k_dim, head_v_dim]`` state:

    S_t = S_{t-1} * exp(g_t) + k_t^T ((v_t - S_{t-1}^T k_t) * beta_t)
    o_t = S_t^T q_t

The update subtracts the state's own prediction ``S^T k`` -- the delta rule -- so unlike a Mamba2 SSM
the operator carrying state across a chunk boundary is not diagonal and the inter-chunk pass has no
closed form. Hence a vectorised intra-chunk pass and a sequential pass over the (few) chunks.

Three things here exist because of the compiler and must not be "simplified":
  * the triangular system is inverted by block recursion, not by a solver or a Neumann series (see
    ``_unit_lower_triangular_inverse``);
  * decay exponents are masked BEFORE ``exp``, or the strictly upper triangle overflows to +inf and
    ``inf * 0`` gives NaN;
  * the chunk axis is a contiguous reshape after right-padding, never a strided view or a
    possibly-zero-size ``split``.
"""
import torch
import torch.nn.functional as F

# HF's ``torch_chunk_gated_delta_rule`` default; also the FLA library's default.
DEFAULT_CHUNK_SIZE = 64
# HF ``l2norm`` epsilon (kept identical so the normalisation matches bit-for-bit in fp32).
L2NORM_EPS = 1e-6


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = L2NORM_EPS) -> torch.Tensor:
    """Match HF's ``l2norm`` (which itself matches the FLA library): rsqrt of the SUM of squares."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _coupling_masks(chunk_size: int, device, dtype):
    """The per-level coupling masks used by ``_unit_lower_triangular_inverse``.

    Level ``level`` (block size ``s = 2**level``) keeps exactly the entries ``(i, j)`` that sit in
    the same ``2s``-wide super-block with ``i`` in its upper half and ``j`` in its lower half, i.e.
    ``i // s`` odd and ``j // s == i // s - 1``. Every strictly-lower entry is kept at exactly one
    level (the level of its binary-tree meeting point), so the levels partition ``strict_lower``.
    """
    index = torch.arange(chunk_size, device=device)
    masks = []
    s = 1
    while s < chunk_size:
        block_i = (index // s).unsqueeze(-1)                    # [C, 1]
        block_j = (index // s).unsqueeze(0)                     # [1, C]
        mask = (block_i % 2 == 1) & (block_j == block_i - 1)
        masks.append(mask.to(dtype))
        s *= 2
    return masks


def _unit_lower_triangular_inverse(strict_lower: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Exact inverse of ``I + strict_lower`` for a strictly lower triangular ``strict_lower``.

    Recursive block inversion as a constant mask and two matmuls per level. Since
    ``[[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]]``, holding the size-``s`` diagonal
    blocks' inverses in ``Inv`` and this level's ``(odd, even)`` coupling blocks in ``Lsub`` makes
    ``Inv_2s = Inv_s - Inv_s @ Lsub @ Inv_s`` apply that formula to every pair at once. From
    ``Inv_1 = I``, ``ceil(log2(C))`` levels give the full inverse with static shapes and no
    data-dependent control flow.

    TRAP: this is NOT a Neumann series and must not be simplified into one. The doubling form
    ``(I - N)^-1 = prod (I + N^(2^i))`` is algebraically exact for nilpotent ``N`` and costs the same
    matmuls, but materialises powers up to ``N^(C-1)`` that then cancel, and loses fp32 entirely once
    ``||N||_2 > 1`` — which happens whenever the log decay is weak and a chunk's keys are correlated,
    i.e. routinely.

    Args:
        strict_lower: ``[..., C, C]``, assumed already zero on and above the diagonal.
        chunk_size: C.
    Returns:
        ``[..., C, C]`` the inverse of ``I + strict_lower``.
    """
    inverse = torch.eye(chunk_size, dtype=strict_lower.dtype,
                        device=strict_lower.device).expand_as(strict_lower).contiguous()
    for mask in _coupling_masks(chunk_size, strict_lower.device, strict_lower.dtype):
        coupling = strict_lower * mask
        inverse = inverse - inverse @ coupling @ inverse
    return inverse


def _right_pad(t: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Right-pad ``dim`` with ``pad`` zeros. Zeros are the identity for both padded roles here:
    a zero log-decay means "no decay" and a zero beta means "no state update"."""
    if pad == 0:
        return t
    shape = list(t.shape)
    shape[dim] = pad
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=dim)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=DEFAULT_CHUNK_SIZE,
                           initial_state=None, chunk_carry=None, return_chunk_states=False):
    """Chunked gated delta rule for prefill. Numerically equivalent to
    ``recurrent_gated_delta_rule`` and to HF ``torch_chunk_gated_delta_rule`` (fp32).

    Shapes (query/key already repeat-interleaved to the value head count):
        query, key: ``[b, l, H, Dk]``   value: ``[b, l, H, Dv]``
        g, beta:    ``[b, l, H]``       g is the log decay (<= 0)
        initial_state: ``[b, H, Dk, Dv]`` or None
    Returns ``(out[b, l, H, Dv], final_state[b, H, Dk, Dv])``, both fp32. With
    ``return_chunk_states`` the third element is ``[b, T, H, Dk, Dv]``: the state after each chunk.

    ``chunk_carry`` is ``[b, T]`` in {0, 1} and is what makes several requests packed into one row
    correct: 0 says "this chunk begins a new request, so drop the state carried into it". A request
    boundary that fell INSIDE a chunk could not be expressed this way -- the within-chunk coupling
    (``pairwise_decay``) would still mix the two -- so boundaries must land on chunk edges, and the
    caller is responsible for that. This is why the boundaries are given per chunk rather than per
    token: a per-token form would look as though mid-chunk boundaries were supported.

    ``use_qk_l2norm_in_kernel=True`` is hardcoded: Qwen3.5-MoE always calls the kernels with it.
    """
    b, l, h, dk = key.shape
    dv = value.shape[-1]
    if l <= 0:  # ValueError, not assert: survives `python -O`.
        raise ValueError("chunk_gated_delta_rule requires a non-empty sequence")
    cs = chunk_size

    # [b, l, H, D] -> [b, H, l, D] in fp32 (HF does the same before any arithmetic).
    query, key, value = (x.transpose(1, 2).to(torch.float32) for x in (query, key, value))
    beta, decay = (x.transpose(1, 2).to(torch.float32) for x in (beta, g))

    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query = query * (dk ** -0.5)

    pad = (cs - l % cs) % cs
    total = l + pad
    t_chunks = total // cs
    query, key, value = (_right_pad(x, pad, dim=2) for x in (query, key, value))
    beta, decay = (_right_pad(x, pad, dim=2) for x in (beta, decay))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Contiguous reshape onto the chunk axis: [b, H, T, C, D]. NOT a strided view.
    query, key, k_beta = (x.reshape(b, h, t_chunks, cs, dk) for x in (query, key, k_beta))
    v_beta = v_beta.reshape(b, h, t_chunks, cs, dv)
    decay = decay.reshape(b, h, t_chunks, cs)

    ar = torch.arange(cs, device=key.device)
    strictly_upper = (ar[None, :] > ar[:, None]).view(1, 1, 1, cs, cs)

    cum_decay = decay.cumsum(dim=3)                                        # [b, H, T, C], <= 0
    # pairwise_decay[..., i, j] = exp(cum_i - cum_j) for j <= i, else 0. MASK BEFORE EXP.
    pairwise_decay = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2))
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay          # lower triangular
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # HF solves (I + strict_lower(ut_system)) X = RHS with a unitriangular solver; the diagonal of
    # ut_system is ignored, so take the strictly lower part explicitly and invert it exactly.
    inverse = _unit_lower_triangular_inverse(ut_system.tril(-1), cs)
    new_values = inverse @ v_beta                                          # [b, H, T, C, Dv]
    k_cumdecay = inverse @ decayed_k_beta                                  # [b, H, T, C, Dk]

    # Fold the within-chunk decay into q and k once, outside the chunk loop (as HF does).
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()                                 # [b, H, T]

    if initial_state is None:
        state = torch.zeros(b, h, dk, dv, dtype=torch.float32, device=key.device)
    else:
        state = initial_state.to(torch.float32)

    if chunk_carry is not None:
        # [b, T] -> [b, 1, 1, 1] per chunk when indexed, so it scales the whole carried state.
        carry = chunk_carry.to(torch.float32).reshape(b, t_chunks)
    else:
        carry = None

    # Sequential pass over the T chunks. T = ceil(l / C) is small (l=512, C=64 -> T=8); this is a
    # loop over CHUNKS, not over the sequence, so it unrolls to a bounded graph.
    outputs = []
    chunk_states: list[torch.Tensor] | None = [] if return_chunk_states else None
    for i in range(t_chunks):
        if carry is not None:
            # Dropped BEFORE the chunk reads it, so a chunk that starts a new request neither predicts
            # from the previous request's state nor subtracts it out. Multiplying by a runtime {0,1}
            # keeps this graph-static, the same reason `is_continuation` is arithmetic.
            state = state * carry[:, i].reshape(b, 1, 1, 1)
        # The part of the chunk's target values the incoming state already predicts is subtracted
        # out, so only the correction is written to the state. This is the delta rule.
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state          # [b, H, C, Dv]
        outputs.append(query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i, None, None] + key[:, :, i].transpose(-1, -2) @ v_new
        if chunk_states is not None:
            chunk_states.append(state)

    out = torch.stack(outputs, dim=2).reshape(b, h, total, dv)[:, :, :l]
    if chunk_states is not None:
        return out.transpose(1, 2), state, torch.stack(chunk_states, dim=1)
    return out.transpose(1, 2), state


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    """One-token (decode) step of the gated delta rule, fully vectorised (no loop).

    Shapes: query/key ``[b, 1, H, Dk]``, value ``[b, 1, H, Dv]``, g/beta ``[b, 1, H]``,
    initial_state ``[b, H, Dk, Dv]`` or None. Returns ``(out[b, 1, H, Dv], state[b, H, Dk, Dv])``.

    Equivalent to HF ``torch_recurrent_gated_delta_rule`` with ``sequence_length == 1``.
    """
    b, seq, h, dk = key.shape
    if seq != 1:
        raise ValueError(
            f"recurrent_gated_delta_rule advances the state by exactly one token; got {seq}. "
            "A multi-token step must go through chunk_gated_delta_rule."
        )
    dv = value.shape[-1]
    q = l2norm(query.to(torch.float32).squeeze(1), dim=-1) * (dk ** -0.5)   # [b, H, Dk]
    k = l2norm(key.to(torch.float32).squeeze(1), dim=-1)                    # [b, H, Dk]
    v = value.to(torch.float32).squeeze(1)                                  # [b, H, Dv]
    decay = g.to(torch.float32).squeeze(1)                                  # [b, H]
    bt = beta.to(torch.float32).squeeze(1)                                  # [b, H]

    if initial_state is None:
        state = torch.zeros(b, h, dk, dv, dtype=torch.float32, device=key.device)
    else:
        state = initial_state.to(torch.float32)


    state = state * decay.exp()[..., None, None]
    kv_mem = (state * k[..., None]).sum(dim=-2)                             # [b, H, Dv]
    delta = (v - kv_mem) * bt[..., None]
    state = state + k[..., None] * delta[..., None, :]
    out = (state * q[..., None]).sum(dim=-2)                                # [b, H, Dv]
    return out.unsqueeze(1), state


def _masked_tap_conv1d(x_t, conv_weight, kernel_size, hist, segment_id):
    """Depthwise causal conv as an explicit sum of taps, each masked to the token's own request.

    Used only when several requests share the row. ``F.conv1d`` cannot express it: with a packed row it
    reads across the boundary, and measurement puts the damage at exactly ``kernel_size - 1`` positions
    of each following request -- wrong by order 1, not by rounding.

    **This is NOT bit-identical to the F.conv1d path** (measured 4.8e-7 in fp32, from the accumulation
    order). That is why the unsegmented path keeps ``F.conv1d``: its agreement with the reference is
    pinned to the bit, and there is no such claim to preserve for the multi-request path.

    ``segment_id`` is ``[seq]``, one request index per token. The left history belongs to no request, so
    it is given a sentinel that matches nothing -- which is also what makes the first request's leading
    positions see zeros rather than the carried state when there is no continuation.
    """
    k = kernel_size
    seq = x_t.shape[-1]
    padded = torch.cat([hist, x_t], dim=-1)
    seg = segment_id.to(torch.float32).reshape(-1)
    seg_padded = torch.cat([torch.full((k - 1,), -1.0, device=seg.device, dtype=seg.dtype), seg])
    taps = conv_weight.reshape(-1, k)
    out = torch.zeros_like(x_t)
    for j in range(k):
        window = padded[..., j:j + seq]
        same = (seg_padded[j:j + seq] == seg).to(x_t.dtype)
        out = out + taps[:, j].reshape(1, -1, 1) * window * same
    return out


def segmented_causal_conv1d(x_t, conv_weight, kernel_size, groups,
                            conv_state=None, is_continuation=None, valid_len=None,
                            segment_id=None):
    """Depthwise causal conv1d for (segmented) GDN prefill, with the history carry.

    ``segment_id`` is ``[seq]`` and is a DIFFERENT axis from the "segmented" in this function's name:
    the name means one sequence split across prefill steps, this means several requests packed into one
    row. When it is given the convolution switches to a masked tap sum (see ``_masked_tap_conv1d``),
    because ``F.conv1d`` reads across a request boundary.

    ``x_t`` is ``[b, conv_dim, seq]``; returns ``(out[b, conv_dim, seq], new_conv_state[b, conv_dim,
    kernel_size-1])``, the carried state being the last ``kernel_size-1`` RAW (pre-activation) inputs.

    ``conv_state is None`` is a fresh single-shot prefill (zero left-pad). Otherwise it is the previous
    segment's state and ``is_continuation`` is a runtime {0,1} mask, zero on the first segment, applied
    as tensor arithmetic so no Python branch on a runtime value enters the graph.

    ``valid_len`` is the number of real tokens: with bucket padding ``seq`` exceeds it and the carry
    must come from the real tail. ``valid_len=None`` keeps the plain "last K-1" behaviour.

    No bias argument: Qwen3.5-MoE's ``conv1d`` has ``bias=False``.
    """
    k = kernel_size
    if conv_state is None:
        hist = torch.zeros(*x_t.shape[:-1], k - 1, dtype=x_t.dtype, device=x_t.device)
    else:
        hist = conv_state.to(x_t.dtype) * is_continuation.to(x_t.dtype)
    if segment_id is not None:
        out = _masked_tap_conv1d(x_t, conv_weight, k, hist, segment_id)
    elif conv_state is None:
        out = F.conv1d(x_t, conv_weight, None, padding=k - 1, groups=groups)[..., :x_t.shape[-1]]
    else:
        out = F.conv1d(torch.cat([hist, x_t], dim=-1), conv_weight, None, groups=groups)
    full = torch.cat([hist, x_t], dim=-1)                                   # [b, conv_dim, K-1+seq]
    if valid_len is None:
        new_conv_state = full[..., -(k - 1):]
    else:
        idx = valid_len.reshape(()).long() + torch.arange(k - 1, device=x_t.device)
        new_conv_state = torch.index_select(full, -1, idx)
    return out, new_conv_state


def gated_delta_net_prefill(hidden_states, weights, dims, eps, chunk_size=DEFAULT_CHUNK_SIZE,
                            conv_state=None, is_continuation=None, valid_mask=None,
                            initial_state=None):
    """The whole Gated DeltaNet prefill, from hidden states to the mixer output.

    ``hidden_states``: ``[1, T, hidden]`` in the model dtype.
    ``weights``: mapping with ``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a`` (each
        ``[hidden, out]``), ``conv1d`` ``[conv_dim, 1, K]``, ``A_log``/``dt_bias`` ``[v_heads]``,
        ``norm`` ``[head_v_dim]``, ``out_proj`` ``[value_dim, hidden]``.
    ``dims``: mapping with ``k_heads``, ``v_heads``, ``head_k_dim``, ``head_v_dim``, ``kernel``.
    ``valid_mask``: ``[T]`` {0,1} mask of real tokens, or None when the whole prefill is real. Its
        contract is narrower than the name suggests and the caller must guarantee it: the real tokens
        form a PREFIX and the padding a contiguous suffix, because the conv history is gathered at
        ``valid_mask.sum()``. Neuron's bucket padding of a single sequence, the only producer, is.

    Returns ``(output[1, T, hidden], recurrent_state, conv_state)``.
    """
    from .ops import (  # local: keeps this module import-light
        gated_delta_projections,
        gated_rmsnorm,
    )

    out_dtype = hidden_states.dtype
    tokens = hidden_states.shape[1]
    key_dim = dims["k_heads"] * dims["head_k_dim"]
    value_dim = dims["v_heads"] * dims["head_v_dim"]
    conv_dim = 2 * key_dim + value_dim
    kernel = dims["kernel"]

    mixed = (hidden_states @ weights["in_proj_qkv"]).transpose(1, 2)      # [1, conv_dim, T]
    z = hidden_states @ weights["in_proj_z"]
    b = hidden_states @ weights["in_proj_b"]
    a = hidden_states @ weights["in_proj_a"]

    valid_len = None if valid_mask is None else valid_mask.sum()
    conv_out, new_conv_state = segmented_causal_conv1d(
        mixed, weights["conv1d"], kernel, conv_dim,
        conv_state=conv_state, is_continuation=is_continuation, valid_len=valid_len)
    # The reference applies the conv activation (SiLU) to the conv OUTPUT; the carried state is the
    # raw pre-activation input, which is what segmented_causal_conv1d returns.
    conv_out = F.silu(conv_out).transpose(1, 2)                          # [1, T, conv_dim]

    query, key, value = _split_and_expand(conv_out, tokens, dims, key_dim, value_dim)
    g, beta = gated_delta_projections(a, b, weights["A_log"], weights["dt_bias"])
    if valid_mask is not None:
        keep = valid_mask.reshape(1, tokens, 1)
        g = g * keep.to(g.dtype)
        beta = beta * keep.to(beta.dtype)

    core_attn_out, state = chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state)
    normed = gated_rmsnorm(core_attn_out.reshape(-1, dims["head_v_dim"]),
                           z.reshape(-1, dims["head_v_dim"]),
                           weights["norm"], eps, out_dtype).reshape(*z.shape[:2], value_dim)
    return normed @ weights["out_proj"], state, new_conv_state


def gated_delta_net_decode(hidden_states, weights, dims, eps, conv_state, recurrent_state,
                           is_continuation=None):
    """The one-token Gated DeltaNet step. Same contract as ``gated_delta_net_prefill``.

    ``is_continuation`` is a runtime {0,1} value -- a scalar for a single request, or one entry per
    request for a batched step -- zero when that request's token is the FIRST of its sequence,
    in which case both carried states are zeroed. TRAP: a one-token prompt has ``max_query_len == 1``
    and so cannot be told apart from a decode step by token count, and without this mask it would
    continue from the previous request's state and every layer would return plausible nonsense. It
    also makes a preemption that resumes by recomputing from position 0 safe. Passing None asserts the
    caller has established this really is a continuation.

    The depthwise convolution is a manual multiply-accumulate rather than ``F.conv1d``: a convolution
    with an output length of 1 has been observed to crash neuronx-cc, and over a length-``K`` window
    the taps are a plain elementwise product anyway.
    """
    from .ops import gated_delta_projections, gated_rmsnorm

    out_dtype = hidden_states.dtype
    if hidden_states.shape[1] != 1:
        raise ValueError(
            f"gated_delta_net_decode advances the state by exactly one token; got "
            f"{hidden_states.shape[1]}."
        )
    key_dim = dims["k_heads"] * dims["head_k_dim"]
    value_dim = dims["v_heads"] * dims["head_v_dim"]
    kernel = dims["kernel"]

    mixed = (hidden_states @ weights["in_proj_qkv"]).transpose(1, 2)      # [1, conv_dim, 1]
    z = hidden_states @ weights["in_proj_z"]
    b = hidden_states @ weights["in_proj_b"]
    a = hidden_states @ weights["in_proj_a"]

    if is_continuation is not None:
        # Tensor arithmetic, not a Python branch on a runtime value: this must stay graph-static.
        #
        # Reshaped to lead rather than broadcast from the right. A scalar broadcasts against anything,
        # but a PER-REQUEST flag is [requests] and the states are [requests, ...]: aligning from the
        # right would multiply the last axis by it, or fail, depending on the widths. Leading-axis
        # broadcast is what "one flag per request" means, so it is written out.
        def _per_request(flag, state):
            flag = flag.to(state.dtype)
            if flag.dim() == 0:
                return flag
            return flag.reshape((-1,) + (1,) * (state.dim() - 1))

        conv_state = conv_state * _per_request(is_continuation, conv_state)
        recurrent_state = recurrent_state * _per_request(is_continuation, recurrent_state)
    conv_in = torch.cat([conv_state.to(mixed.dtype), mixed], dim=-1)     # [1, conv_dim, K]
    new_conv_state = conv_in[..., -(kernel - 1):]
    conv_out = (conv_in * weights["conv1d"].squeeze(1)).sum(dim=-1, keepdim=True)
    conv_out = F.silu(conv_out).transpose(1, 2)                          # [1, 1, conv_dim]

    query, key, value = _split_and_expand(conv_out, 1, dims, key_dim, value_dim)
    g, beta = gated_delta_projections(a, b, weights["A_log"], weights["dt_bias"])
    core_attn_out, state = recurrent_gated_delta_rule(
        query, key, value, g, beta, initial_state=recurrent_state)
    normed = gated_rmsnorm(core_attn_out.reshape(-1, dims["head_v_dim"]),
                           z.reshape(-1, dims["head_v_dim"]),
                           weights["norm"], eps, out_dtype).reshape(*z.shape[:2], value_dim)
    return normed @ weights["out_proj"], state, new_conv_state


def gated_delta_net_verify(hidden_states, weights, dims, eps, conv_state, recurrent_state,
                           accepted, is_continuation=None):
    """Advance the state by a VARIABLE number of tokens, chosen at runtime.

    This is what speculative decoding needs and what the one-token step cannot give. A verify step is
    handed ``k`` proposed tokens, computes an output for every one of them, and then learns how many
    were accepted — a number that is only known after the outputs have been compared, i.e. after the
    state has already been advanced past the rejected ones.

    The way out is that the intermediate states are already being computed. Stepping the ``k`` tokens
    produces ``k`` successive states; keeping all of them, together with the incoming state as the
    zero-accepted case, gives ``k + 1`` candidates. ``accepted`` then selects one.

    Two properties make the selection safe in a compiled graph:

    - ``k`` is a Python integer taken from the tensor's static shape, so the loop unrolls at trace time
    - ``accepted`` is a TENSOR, and it is applied as a sum of ``{0, 1}`` masks rather than as an index.
      Indexing by a runtime value would either split the graph or read out of bounds

    Args:
        hidden_states: ``[1, k, hidden]`` — the proposed tokens, in order.
        accepted: scalar tensor in ``[0, k]``. ``0`` leaves both states exactly as they came in.
        is_continuation: as in ``gated_delta_net_decode``; zeroes the carried states for the first
            token of a sequence.

    Returns ``(outputs, recurrent_state, conv_state)`` with outputs ``[1, k, hidden]`` for every
    proposed token — the caller needs all of them to decide what to accept.
    """
    tokens = hidden_states.shape[1]
    if tokens < 1:
        raise ValueError(f"a verify step needs at least one proposed token; got {tokens}.")

    if is_continuation is not None:
        # Once, before the first token. Tensor arithmetic, not a Python branch (see the decode step).
        conv_state = conv_state * is_continuation.to(conv_state.dtype)
        recurrent_state = recurrent_state * is_continuation.to(recurrent_state.dtype)

    # Candidate j is the state after accepting j tokens. Candidate 0 is what came in.
    recurrent_candidates = [recurrent_state]
    conv_candidates = [conv_state]
    outputs = []
    for index in range(tokens):
        step_out, next_recurrent, next_conv = gated_delta_net_decode(
            hidden_states[:, index:index + 1], weights, dims, eps,
            conv_candidates[-1], recurrent_candidates[-1], is_continuation=None)
        outputs.append(step_out)
        recurrent_candidates.append(next_recurrent)
        conv_candidates.append(next_conv)

    committed_recurrent = _select_candidate(recurrent_candidates, accepted)
    committed_conv = _select_candidate(conv_candidates, accepted)
    return torch.cat(outputs, dim=1), committed_recurrent, committed_conv


def _select_candidate(candidates, accepted):
    """Pick ``candidates[accepted]`` with arithmetic, where ``accepted`` is a tensor.

    A weighted sum over one-hot masks. The masks are exclusive and sum to one, so exactly one candidate
    survives; no branch and no data-dependent index appears in the graph. The comparison is done in
    int64 and the mask cast to each candidate's dtype, so a bf16 state is not routed through a float
    comparison.
    """
    total = None
    for index, candidate in enumerate(candidates):
        keep = (accepted == index).to(candidate.dtype)
        term = candidate * keep
        total = term if total is None else total + term
    return total


def _split_and_expand(conv_out, tokens, dims, key_dim, value_dim):
    """Split the post-conv projection into per-head q/k/v and expand q/k to the value head count.

    ``tensor_split`` with explicit static indices, not ``split``: a possibly-zero-size ``split`` is
    miscompiled by neuronx-cc, and static indices fix the shapes at trace time. The batch extent is
    taken from the input rather than assumed to be 1, so a shape that disagrees with the caller fails
    here instead of being reinterpreted.
    """
    batch = conv_out.shape[0]
    query, key, value = torch.tensor_split(conv_out, [key_dim, 2 * key_dim], dim=-1)
    query = query.reshape(batch, tokens, dims["k_heads"], dims["head_k_dim"])
    key = key.reshape(batch, tokens, dims["k_heads"], dims["head_k_dim"])
    value = value.reshape(batch, tokens, dims["v_heads"], dims["head_v_dim"])
    repeats = dims["v_heads"] // dims["k_heads"]
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    return query, key, value
