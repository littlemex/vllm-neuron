# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE checkpoint-to-per-rank weight layout transforms (torch only, no Neuron deps).

Split out from the weight loaders so the layout arithmetic can be pinned on CPU against the HF
reference modules: a transposed projection or a swapped gate/up half loads without error and shows up
only as wrong output.

Checkpoint layouts (confirmed against the published ``Qwen/Qwen3.6-35B-A3B`` safetensors headers):

  ``self_attn.q_proj.weight``          [num_heads * head_dim * 2, hidden]  (head-major: per head,
                                       ``head_dim`` query rows then ``head_dim`` gate rows)
  ``self_attn.{k,v}_proj.weight``      [num_kv_heads * head_dim, hidden]
  ``linear_attn.in_proj_qkv.weight``   [2 * key_dim + value_dim, hidden]   (``[q | k | v]`` rows)
  ``linear_attn.conv1d.weight``        [2 * key_dim + value_dim, 1, kernel]
  ``mlp.experts.gate_up_proj``         [num_experts, 2 * intermediate, hidden]  (gate rows first)
  ``mlp.experts.down_proj``            [num_experts, hidden, intermediate]
"""
import torch

from .config import LINEAR_ATTENTION


def fuse_attention_qkvg(q_proj, k_proj, v_proj, rank, world_size,
                        num_attention_heads, num_key_value_heads, head_dim):
    """Per-rank ``[hidden, query | gate | key | value]`` from the three checkpoint tensors.

    ``q_proj`` packs the query and the output gate per head, so a head-slice keeps them together;
    they are separated here so the forward can treat the fused projection as four flat slices.

    Key and value are sliced per rank GROUP: when the TP size exceeds ``num_key_value_heads`` the KV
    heads are replicated across the ranks that share one, which is the usual GQA arrangement.
    """
    local_rank = rank % world_size
    heads_per_rank = num_attention_heads // world_size
    if world_size >= num_key_value_heads:
        kv_heads_per_rank = 1
        kv_replicas = world_size // num_key_value_heads
    else:
        kv_heads_per_rank = num_key_value_heads // world_size
        kv_replicas = 1
    kv_width = kv_heads_per_rank * head_dim
    kv_rank = local_rank // kv_replicas

    head_block = 2 * head_dim
    q_rows = q_proj[local_rank * heads_per_rank * head_block:
                    (local_rank + 1) * heads_per_rank * head_block, :]
    per_head = q_rows.reshape(heads_per_rank, 2, head_dim, -1)
    query = per_head[:, 0].reshape(heads_per_rank * head_dim, -1)
    gate = per_head[:, 1].reshape(heads_per_rank * head_dim, -1)
    key = k_proj[kv_rank * kv_width:(kv_rank + 1) * kv_width, :]
    value = v_proj[kv_rank * kv_width:(kv_rank + 1) * kv_width, :]
    return torch.cat([query, gate, key, value], dim=0).T.contiguous()


def gdn_qkv_rows(stacked, rank, world_size, num_key_heads, num_value_heads,
                 head_key_dim, head_value_dim):
    """The rank's rows of a ``[q | k | v]``-stacked Gated DeltaNet tensor, concatenated.

    Works for both ``in_proj_qkv.weight`` (2-D) and ``conv1d.weight`` (3-D) because it only slices the
    leading axis. The value heads are a whole multiple of the key heads and the reference expands key
    head ``j`` to value heads ``j * repeats .. j * repeats + repeats - 1``, so a rank's key-head slice
    expands to exactly its own value-head slice — no cross-rank exchange inside the mixer.
    """
    local_rank = rank % world_size
    key_dim = num_key_heads * head_key_dim
    key_width = key_dim // world_size
    value_width = (num_value_heads * head_value_dim) // world_size
    query = stacked[local_rank * key_width:(local_rank + 1) * key_width]
    key = stacked[key_dim + local_rank * key_width:key_dim + (local_rank + 1) * key_width]
    value = stacked[2 * key_dim + local_rank * value_width:
                    2 * key_dim + (local_rank + 1) * value_width]
    return torch.cat([query, key, value], dim=0)


def shard_expert_gate_up(gate_up_proj, rank, world_size):
    """``[E, 2 * intermediate, hidden]`` -> the rank's ``[E, hidden, 2, intermediate_per_rank]``.

    The checkpoint stores one fused tensor whose FIRST ``intermediate`` rows are the gate and whose
    last ``intermediate`` rows are the up projection (the reference chunks the linear's output). The
    MoE kernels want a trailing ``{gate, up}`` axis, so the two halves are sliced per rank and stacked
    rather than reshaped — a reshape of the fused rows would interleave gate and up.
    """
    local_rank = rank % world_size
    twice_intermediate = gate_up_proj.shape[1]
    intermediate = twice_intermediate // 2
    width = intermediate // world_size
    gate = gate_up_proj[:, local_rank * width:(local_rank + 1) * width, :]
    up_start = intermediate + local_rank * width
    up = gate_up_proj[:, up_start:up_start + width, :]
    fused = torch.stack([gate, up], dim=2)                  # [E, width, 2, hidden]
    return fused.permute(0, 3, 2, 1).contiguous()           # [E, hidden, 2, width]


def shard_expert_down(down_proj, rank, world_size):
    """``[E, hidden, intermediate]`` -> the rank's ``[E, intermediate_per_rank, hidden]``.

    The checkpoint stores ``nn.Linear``-style ``[out, in]`` per expert; the kernels want ``[in, out]``.
    """
    local_rank = rank % world_size
    intermediate = down_proj.shape[2]
    width = intermediate // world_size
    sliced = down_proj[:, :, local_rank * width:(local_rank + 1) * width]
    return sliced.transpose(1, 2).contiguous()


def shard_rows_transposed(tensor, rank, world_size):
    """``[out, in]`` -> the rank's ``[in, out_per_rank]`` (a column-parallel projection)."""
    local_rank = rank % world_size
    width = tensor.shape[0] // world_size
    return tensor[local_rank * width:(local_rank + 1) * width, :].T.contiguous()


def shard_columns_transposed(tensor, rank, world_size):
    """``[out, in]`` -> the rank's ``[in_per_rank, out]`` (a row-parallel projection)."""
    local_rank = rank % world_size
    width = tensor.shape[1] // world_size
    return tensor[:, local_rank * width:(local_rank + 1) * width].T.contiguous()


def shard_heads(tensor, rank, world_size):
    """A per-head vector (``A_log`` / ``dt_bias``) sliced to the rank's heads."""
    local_rank = rank % world_size
    width = tensor.shape[0] // world_size
    return tensor[local_rank * width:(local_rank + 1) * width].contiguous()


def vision_checkpoint_mappings(depth, source="model.visual"):
    """Map the vision tower's parameters to their checkpoint keys.

    The key names are identical to the plugin's Qwen3-VL reference implementation, which is why the
    tower can reuse that encoder rather than needing a new one; only the dimensions differ. Kept
    separate from ``checkpoint_mappings`` so a text-only deployment neither declares nor loads these.

    Every tensor here carries a bias, unlike the text side where only the norms have weights: the vision
    blocks come from a ViT lineage that keeps biases on the projections.
    """
    mappings: dict = {}
    mappings["visual.patch_embed.proj.weight"] = f"{source}.patch_embed.proj.weight"
    mappings["visual.patch_embed.proj.bias"] = f"{source}.patch_embed.proj.bias"
    mappings["visual.pos_embed.weight"] = f"{source}.pos_embed.weight"
    for index in range(depth):
        src = f"{source}.blocks.{index}"
        dst = f"visual.blocks.{index}"
        for norm in ("norm1", "norm2"):
            mappings[f"{dst}.{norm}.weight"] = f"{src}.{norm}.weight"
            mappings[f"{dst}.{norm}.bias"] = f"{src}.{norm}.bias"
        for projection in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            mappings[f"{dst}.{projection}.weight"] = f"{src}.{projection}.weight"
            mappings[f"{dst}.{projection}.bias"] = f"{src}.{projection}.bias"
    for part in ("norm", "linear_fc1", "linear_fc2"):
        mappings[f"visual.merger.{part}.weight"] = f"{source}.merger.{part}.weight"
        mappings[f"visual.merger.{part}.bias"] = f"{source}.merger.{part}.bias"
    return mappings


def checkpoint_mappings(layer_types, source, has_lm_head, tie_word_embeddings):
    """Map every parameter this implementation declares to its checkpoint key(s).

    Kept here, dependency-light, so a CPU test can diff it against a real checkpoint's key set: a
    mistyped source key loads nothing and shows up only as garbage output on the device.

    ``source`` is the decoder prefix (``model.language_model`` for the published multimodal
    checkpoint, ``model`` for a text-only one). A value that is a list means one destination is built
    from several sources.

    Deliberately unmapped: ``model.visual.*`` (the vision tower) and ``mtp.*`` (the multi-token
    prediction head), both out of scope.
    """
    mappings: dict = {}
    for index, kind in enumerate(layer_types):
        src = f"{source}.layers.{index}"
        dst = f"model.layers.{index}"
        mappings[f"{dst}.input_layernorm.weight"] = f"{src}.input_layernorm.weight"
        mappings[f"{dst}.post_attention_layernorm.weight"] = f"{src}.post_attention_layernorm.weight"
        if kind == LINEAR_ATTENTION:
            prefix = f"{dst}.linear_attn"
            mappings[f"{prefix}.in_proj_qkv_weight"] = f"{src}.linear_attn.in_proj_qkv.weight"
            mappings[f"{prefix}.in_proj_z_weight"] = f"{src}.linear_attn.in_proj_z.weight"
            mappings[f"{prefix}.in_proj_b_weight"] = f"{src}.linear_attn.in_proj_b.weight"
            mappings[f"{prefix}.in_proj_a_weight"] = f"{src}.linear_attn.in_proj_a.weight"
            mappings[f"{prefix}.conv1d_weight"] = f"{src}.linear_attn.conv1d.weight"
            mappings[f"{prefix}.A_log"] = f"{src}.linear_attn.A_log"
            mappings[f"{prefix}.dt_bias"] = f"{src}.linear_attn.dt_bias"
            mappings[f"{prefix}.norm_weight"] = f"{src}.linear_attn.norm.weight"
            mappings[f"{prefix}.out_proj_weight"] = f"{src}.linear_attn.out_proj.weight"
        else:
            prefix = f"{dst}.self_attn"
            # One fused destination from three sources: the loader's transform slices the rank's heads
            # out of q (which also carries the output gate), k and v.
            mappings[f"{prefix}.qkvg_proj_weight"] = [
                f"{src}.self_attn.q_proj.weight",
                f"{src}.self_attn.k_proj.weight",
                f"{src}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.o_proj_weight"] = f"{src}.self_attn.o_proj.weight"
            mappings[f"{prefix}.q_norm_weight"] = f"{src}.self_attn.q_norm.weight"
            mappings[f"{prefix}.k_norm_weight"] = f"{src}.self_attn.k_norm.weight"
        mlp = f"{dst}.mlp"
        mappings[f"{mlp}.router_weight"] = f"{src}.mlp.gate.weight"
        mappings[f"{mlp}.gate_up_proj_weight"] = f"{src}.mlp.experts.gate_up_proj"
        mappings[f"{mlp}.down_proj_weight"] = f"{src}.mlp.experts.down_proj"
        mappings[f"{mlp}.shared_gate_proj_weight"] = f"{src}.mlp.shared_expert.gate_proj.weight"
        mappings[f"{mlp}.shared_up_proj_weight"] = f"{src}.mlp.shared_expert.up_proj.weight"
        mappings[f"{mlp}.shared_down_proj_weight"] = f"{src}.mlp.shared_expert.down_proj.weight"
        mappings[f"{mlp}.shared_expert_gate_weight"] = f"{src}.mlp.shared_expert_gate.weight"
    mappings["model.embed_tokens.weight"] = f"{source}.embed_tokens.weight"
    mappings["model.norm.weight"] = f"{source}.norm.weight"
    if has_lm_head:
        mappings["lm_head.weight"] = "lm_head.weight"
    elif tie_word_embeddings:
        mappings["lm_head.weight"] = f"{source}.embed_tokens.weight"
    else:
        # Falling back to the embedding matrix for an UNTIED checkpoint would invent a tied head and
        # serve logits from it, which looks like a working model.
        raise ValueError(
            "the checkpoint has no lm_head.weight and the config says tie_word_embeddings=False, so "
            "there is no head to load. Refusing rather than substituting the embedding matrix."
        )
    return mappings
