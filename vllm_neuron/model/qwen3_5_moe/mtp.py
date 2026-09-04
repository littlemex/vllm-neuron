"""The multi-token prediction head.

The checkpoint carries one of these: nineteen keys under ``mtp.``, of which the layer is a single
``mtp.layers.0`` with full attention and a MoE block, plus ``fc``, a final ``norm``, and two norms
applied before the concatenation. There is no ``mtp.embed_tokens``, so the embedding is shared with the
main model, and there is no separate LM head either.

**The composition is transcribed from vLLM's own implementation**
(``vllm/model_executor/models/qwen3_5_mtp.py``, ``Qwen3_5MultiTokenPredictor.forward``) rather than
inferred, because it cannot be inferred:

    inputs_embeds = pre_fc_norm_embedding(embed(next_token_ids))
    hidden_states = pre_fc_norm_hidden(last_hidden_state)
    hidden_states = cat([inputs_embeds, hidden_states], dim=-1)      # embedding FIRST
    hidden_states = fc(hidden_states)                                 # 2H -> H, no bias
    hidden_states = layer(hidden_states)                              # one full-attention + MoE layer
    hidden_states = norm(hidden_states + residual)

The concatenation order is the part that has to be read rather than guessed: ``[hidden, embedding]`` has
the same shape, loads the same weights without complaint, and produces fluent nonsense.

``transformers`` is not a reference here — it drops ``mtp.*`` on load
(``_keys_to_ignore_on_load_unexpected``) — and vLLM's module cannot be instantiated on CPU because its
decoder layer asks the platform for an attention backend implementation. So the differential test that
the rest of this port has against HuggingFace does not exist for this head. What IS tested: every
component in isolation (both norms, the attention, the MoE) already has one, the weight mapping is
diffed against the real checkpoint index in both directions, and the composition is asserted to be
graph-static. The gap is stated in the model README rather than papered over.
"""
from __future__ import annotations

import torch
from torch import nn

from vllm_neuron.nn import ColumnParallelLinear
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .model_bf16 import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeMoE,
    Qwen3_5MoeRMSNorm,
)


class Qwen3_5MoeMultiTokenPredictor(nn.Module):
    """One draft layer that predicts the token after next, given the last hidden state.

    ``layer_idx`` is the index this layer occupies in the runner's view of the model, which must be past
    the main model's layers: the draft layer has full attention, so it needs its own KV cache entry and
    its own attention metadata, and both are keyed by layer index. Passing the main model's layer count
    is what makes it the (N+1)-th layer rather than a second copy of the last one.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        hidden = config.hidden_size

        # Two separate norms, one per input to the concatenation, and a third after the layer. The two
        # pre-norms are not interchangeable: they carry different learned scales for the embedding and
        # the hidden state, which is the whole reason the head can combine two things of the same width.
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, config.rms_norm_eps,
                                                       config.torch_dtype)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, config.rms_norm_eps, config.torch_dtype)
        self.norm = Qwen3_5MoeRMSNorm(hidden, config.rms_norm_eps, config.torch_dtype)

        # 2H -> H, no bias. Column-parallel with the output gathered: the layer that follows wants the
        # full hidden state, and this projection is small enough that the gather is not worth avoiding.
        self.fc = ColumnParallelLinear(2 * hidden, hidden, bias=False, dtype=config.torch_dtype,
                                       gather_output=True)
        set_weight_loader(
            self.fc.weight,
            sharding_weight_loader(shard_dim=0, shard_size=self.fc.out_features_per_rank,
                                   num_shards=self.fc.tp_size, is_storage_transposed=False),
        )

        # The draft layer is FULL attention, so it reuses the main model's attention and MoE blocks
        # unchanged — there is no recurrent state in the draft path at all. What the target needs to
        # verify k tokens lives in the Gated DeltaNet (see gated_delta_net_verify), not here.
        self.input_layernorm = Qwen3_5MoeRMSNorm(hidden, config.rms_norm_eps, config.torch_dtype)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(hidden, config.rms_norm_eps,
                                                          config.torch_dtype)
        self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeMoE(config, layer_idx)

    def forward(self, embeddings, last_hidden_state, positions, cos, sin, attn_metadata,
                is_prefill, valid_mask=None):
        """Predict from ``last_hidden_state`` and the embedding of the token that follows it.

        ``embeddings`` comes from the main model's embedding table — the checkpoint has no separate one
        for the head — and both inputs are ``[batch, tokens, hidden]``.
        """
        if embeddings.shape != last_hidden_state.shape:  # lint-port: ok shapes are graph-static
            raise ValueError(
                f"the embedding and the hidden state must have the same shape; got "
                f"{tuple(embeddings.shape)} and {tuple(last_hidden_state.shape)}"
            )
        # Order transcribed from the reference: the EMBEDDING is the first half of the concatenation.
        # Swapping the halves keeps every shape and silently reads the wrong learned columns of fc.
        normed_embeddings = self.pre_fc_norm_embedding(embeddings)
        normed_hidden = self.pre_fc_norm_hidden(last_hidden_state)
        fused = self.fc(torch.cat([normed_embeddings, normed_hidden], dim=-1))

        residual = fused
        mixed = self.self_attn(self.input_layernorm(fused), positions, cos, sin, attn_metadata,
                               is_prefill)
        hidden_states = residual + mixed
        # The MoE kernels fuse the post-attention norm, so the weight is handed over rather than applied
        # — the same contract as the main model's layers.
        moe_out = self.mlp(hidden_states, self.post_attention_layernorm.weight, is_prefill,
                           valid_mask)
        hidden_states = hidden_states + moe_out.reshape(hidden_states.shape)
        return self.norm(hidden_states)

    def kv_layer_spec(self):
        """What the runner has to allocate for this head, as a plain tuple.

        The draft layer is full attention, so it needs its own KV cache and its own attention metadata,
        both keyed by a layer index that must not collide with the main model's. This is the by-product
        that makes "the model is right but nothing runs": the head itself is four small modules, and the
        contract around it is the work.

        A tuple rather than a ``LayerSpec`` so this module does not import the runner's types and stays
        loadable by a CPU test. The model assembles the real spec.
        """
        attention = self.self_attn
        return (self.layer_idx, attention.num_key_value_heads_per_rank, attention.head_dim,
                attention.dtype, attention.window_size)

    def bind_kv_cache_entry(self, k_cache, v_cache) -> None:
        """Give the draft layer the caches the runner allocated for its index."""
        self.self_attn.bind_caches(k_cache, v_cache)
