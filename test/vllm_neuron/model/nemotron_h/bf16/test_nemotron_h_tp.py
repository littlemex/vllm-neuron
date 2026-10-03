# SPDX-License-Identifier: Apache-2.0
"""CPU tensor-parallel equivalence test for the NemotronH model (TP=1 vs TP=2 vs TP=4).

The kernel tests (test_nemotron_h_kernels.py) pin the numerics of single functions at TP=1. This test
pins the parts that only exist at TP>1: the per-rank weight sharding (in_proj gate/x/B/C/dt slicing,
conv, head-wise params, grouped norm, MoE intermediate split) and the sequence-parallel collectives
around each mixer (all_gather into a mixer, reduce_scatter / all_reduce out of it). A collective
applied twice or a mis-sliced weight still runs and still produces fluent text on short prompts, so
neither shows up at TP=1 or in a smoke test; it shows up here as TP>1 != TP=1.

It builds the real NemotronHForCausalLM from a tiny random checkpoint written in the HF layout, loads
it through load_weights (the shipped sharding loaders), and runs a prefill forward on CPU ranks
connected by gloo. The pattern holds no attention layer (the attention prefill calls an NKI kernel
that has no CPU path). Requires the plugin's Python dependencies (vllm, vllm_neuron); no Neuron
device.
"""
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from safetensors.torch import save_file
    import vllm_neuron.model.nemotron_h.model_bf16 as nh
    import vllm_neuron.nn.embedding as nn_embedding
    from vllm_neuron.model.nemotron_h.config import NemotronHConfig
except Exception as e:  # pragma: no cover - environment without the plugin deps
    pytest.skip(f"plugin dependencies unavailable: {e}", allow_module_level=True)

TINY = dict(
    vocab_size=128, hidden_size=64, num_hidden_layers=5, hybrid_override_pattern="MEMEM",
    torch_dtype=torch.float32,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=64,
    mamba_num_heads=8, mamba_head_dim=8, ssm_state_size=16, conv_kernel=4, n_groups=4,
)
SEQ_LEN = 16


def _write_checkpoint(path, cfg=TINY):
    """A random checkpoint in the text-only HF layout (backbone.* / lm_head.weight)."""
    c = NemotronHConfig(**cfg)
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, std=0.1):
        return (torch.randn(*shape, generator=g) * std).contiguous()

    d, im, nh_, G, N, K = (c.hidden_size, c.mamba_intermediate_size, c.mamba_num_heads,
                           c.n_groups, c.ssm_state_size, c.conv_kernel)
    conv_dim = im + 2 * G * N
    w = {"backbone.embeddings.weight": rnd(c.vocab_size, d, std=1.0),
         "backbone.norm_f.weight": 1 + rnd(d),
         "lm_head.weight": rnd(c.vocab_size, d)}
    for i, t in enumerate(c.hybrid_override_pattern):
        p = f"backbone.layers.{i}"
        w[f"{p}.norm.weight"] = 1 + rnd(d)
        if t == "M":
            w[f"{p}.mixer.in_proj.weight"] = rnd(im + conv_dim + nh_, d)
            w[f"{p}.mixer.conv1d.weight"] = rnd(conv_dim, 1, K, std=0.3)
            w[f"{p}.mixer.conv1d.bias"] = rnd(conv_dim)
            w[f"{p}.mixer.A_log"] = torch.log(torch.rand(nh_, generator=g) * 15 + 1)
            w[f"{p}.mixer.D"] = 1 + rnd(nh_)
            w[f"{p}.mixer.dt_bias"] = rnd(nh_, std=0.5)
            w[f"{p}.mixer.norm.weight"] = 1 + rnd(im)
            w[f"{p}.mixer.out_proj.weight"] = rnd(d, im)
        elif t == "E":
            E, mi, smi = c.n_routed_experts, c.moe_intermediate_size, c.moe_shared_expert_intermediate_size
            w[f"{p}.mixer.gate.weight"] = rnd(E, d, std=0.3)
            w[f"{p}.mixer.gate.e_score_correction_bias"] = rnd(E, std=0.01)
            for e in range(E):
                w[f"{p}.mixer.experts.{e}.up_proj.weight"] = rnd(mi, d)
                w[f"{p}.mixer.experts.{e}.down_proj.weight"] = rnd(d, mi)
            w[f"{p}.mixer.shared_experts.up_proj.weight"] = rnd(smi, d)
            w[f"{p}.mixer.shared_experts.down_proj.weight"] = rnd(d, smi)
        elif t == "*":
            q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
            w[f"{p}.mixer.q_proj.weight"] = rnd(q, d)
            w[f"{p}.mixer.k_proj.weight"] = rnd(kv, d)
            w[f"{p}.mixer.v_proj.weight"] = rnd(kv, d)
            w[f"{p}.mixer.o_proj.weight"] = rnd(d, q)
    save_file(w, os.path.join(path, "model.safetensors"))


class _TPGroup:
    """The slice of vLLM's GroupCoordinator the model uses, with the same semantics as
    DeviceCommunicatorBase: all_reduce sums IN PLACE (and returns the tensor), all_gather and
    reduce_scatter return new tensors. gloo has no reduce_scatter, so it is all_reduce + chunk."""

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.device_group = dist.group.WORLD

    def all_reduce(self, x):
        dist.all_reduce(x)
        return x

    def all_gather(self, x, dim=0):
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=dim)

    def reduce_scatter(self, x, dim=0):
        y = x.clone()
        dist.all_reduce(y)
        return y.chunk(self.world_size, dim=dim)[self.rank_in_group].contiguous()


def _reduce_scatter_tensor(x, reduce_op, scatter_dim, group):
    assert reduce_op == "sum"
    y = x.clone()
    dist.all_reduce(y, group=group)
    return y.chunk(dist.get_world_size(group), dim=scatter_dim)[dist.get_rank(group)].contiguous()


def _build(cfg, ckpt):
    """NemotronHForCausalLM loaded through load_weights. Every parameter is NaN-filled first and
    checked after, so a parameter the checkpoint or the mapping misses fails here instead of
    silently computing on uninitialised memory."""
    model = nh.NemotronHForCausalLM(NemotronHConfig(**cfg))
    with torch.no_grad():
        for prm in model.parameters():
            prm.fill_(float("nan"))
    model.load_weights(ckpt, torch.device("cpu"))
    unloaded = [n for n, prm in model.named_parameters() if torch.isnan(prm).any()]
    assert not unloaded, f"parameters not loaded from the checkpoint: {unloaded}"
    return model


def _worker(rank, world_size, ckpt, init_file, out_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        nh.get_tp_group = lambda: _TPGroup()
        nn_embedding.reduce_scatter_tensor = _reduce_scatter_tensor
        model = _build(TINY, ckpt)
        input_ids = torch.randint(0, TINY["vocab_size"], (SEQ_LEN,),
                                  generator=torch.Generator().manual_seed(1))
        positions = torch.arange(SEQ_LEN, dtype=torch.int32)
        with torch.no_grad():
            hidden = model.model(input_ids, positions, attn_metadata={})
            states = [m.mixer.ssm_state.clone() for m in model.model.layers if m.layer_type == "M"]
        if rank == 0:
            torch.save({"hidden": hidden, "ssm_state_rank0": states}, out_file)
    finally:
        dist.destroy_process_group()


def _run(world_size, ckpt, tmp):
    init_file = os.path.join(tmp, f"init_{world_size}")
    out_file = os.path.join(tmp, f"out_{world_size}.pt")
    mp.spawn(_worker, args=(world_size, ckpt, init_file, out_file), nprocs=world_size, join=True)
    return torch.load(out_file)


HYBRID = dict(TINY, num_hidden_layers=6, hybrid_override_pattern="MEM*EM")
BLOCK_SIZE, NUM_BLOCKS = 8, 4                     # 32 KV slots per request, enough for SEQ_LEN + 1
# Block 0 is the reserved null block (NULL_BLOCK_ID): the KV cache manager never gives it to a
# request, and the runner points padding writes at it. The request owns blocks 1..NUM_BLOCKS.
NULL_BLOCK = 0


def _attn_metadata(model, slot_mapping, query_len):
    """The per-attention-layer metadata the runner hands the model (block_table = blocks 0..N-1)."""
    md = {}
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type == "*":
            md[f"layers.{i}.self_attn"] = {
                "max_query_len": query_len, "decode_token_threshold": 1,
                "slot_mapping": slot_mapping, "block_size": BLOCK_SIZE,
                "block_table_tensor": torch.arange(1, NUM_BLOCKS + 1).view(1, -1),
                "max_blocks_per_seq": NUM_BLOCKS, "kv_segment_size": None,
            }
    return md


def _bind_empty_kv(model):
    kv = {}
    for i, layer in enumerate(model.model.layers):
        if layer.layer_type == "*":
            a = layer.mixer
            shape = (NUM_BLOCKS + 1, a.num_key_value_heads_per_rank, BLOCK_SIZE, a.head_dim)
            kv[f"layers.{i}.self_attn"] = (torch.zeros(shape), torch.zeros(shape))
    model.bind_kv_cache(kv)


def _slot(pos):
    """KV slot of token position `pos` of the request (its blocks start at block 1)."""
    return BLOCK_SIZE + pos


def _decode_worker(rank, world_size, ckpt, init_file, out_file, n_real, pad_slot):
    """Prefill tokens[:SEQ_LEN] (the last SEQ_LEN - n_real of them bucket padding whose KV write goes
    to `pad_slot`), then decode tokens[n_real]. Saves the decode step's final hidden state."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        nh.get_tp_group = lambda: _TPGroup()
        nn_embedding.reduce_scatter_tensor = _reduce_scatter_tensor
        model = _build(HYBRID, ckpt)
        _bind_empty_kv(model)
        tokens = torch.randint(0, HYBRID["vocab_size"], (SEQ_LEN + 1,),
                               generator=torch.Generator().manual_seed(2))
        prompt = tokens[:SEQ_LEN].clone()
        prompt[n_real:] = 0                                   # pad token id, as the runner pads
        slots = _slot(torch.arange(SEQ_LEN))
        if n_real < SEQ_LEN:
            slots[n_real:] = pad_slot
        with torch.no_grad():
            model.model(prompt, torch.arange(SEQ_LEN, dtype=torch.int32),
                        _attn_metadata(model, slots, SEQ_LEN))
            hidden = model.model(tokens[n_real:n_real + 1], torch.tensor([n_real], dtype=torch.int32),
                                 _attn_metadata(model, _slot(torch.tensor([n_real])), 1))
        if rank == 0:
            torch.save({"hidden": hidden, "tokens": tokens}, out_file)
    finally:
        dist.destroy_process_group()


def _full_prefill_last(ckpt, tokens):
    """TP=1 single-shot prefill over tokens; returns the last position's final hidden state."""
    init = tempfile.mktemp()
    dist.init_process_group("gloo", init_method=f"file://{init}", rank=0, world_size=1)
    try:
        nh.get_tp_group = lambda: _TPGroup()
        model = _build(HYBRID, ckpt)
        _bind_empty_kv(model)
        n = tokens.numel()
        with torch.no_grad():
            hidden = model.model(tokens, torch.arange(n, dtype=torch.int32),
                                 _attn_metadata(model, _slot(torch.arange(n)), n))
        return hidden[-1:]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("n_real,pad_slot", [
    (SEQ_LEN, None),
    (SEQ_LEN - 5, -1),                       # PAD_SLOT_ID
    (SEQ_LEN - 5, NULL_BLOCK * BLOCK_SIZE),  # null block, as the 0.24 runner pads
])
def test_decode_after_prefill_matches_single_shot(world_size, n_real, pad_slot):
    """Prefill (optionally bucket-padded) + one decode step at TP=world_size must give the same
    hidden state as a TP=1 single-shot prefill over the real tokens plus the decoded one. This runs
    the attention prefill and decode paths, the KV-cache write/read, and the Mamba state handed from
    prefill to decode, across the TP collectives, under both padding conventions."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(tmp, HYBRID)
        out_file = os.path.join(tmp, "decode.pt")
        mp.spawn(_decode_worker,
                 args=(world_size, tmp, os.path.join(tmp, "init"), out_file, n_real, pad_slot),
                 nprocs=world_size, join=True)
        got = torch.load(out_file)
        ref = _full_prefill_last(tmp, got["tokens"][:n_real + 1])
        torch.testing.assert_close(got["hidden"], ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("world_size", [2, 4])
def test_prefill_tp_matches_tp1(world_size):
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(tmp)
        ref = _run(1, tmp, tmp)
        got = _run(world_size, tmp, tmp)
        assert ref["hidden"].shape == got["hidden"].shape == (SEQ_LEN, TINY["hidden_size"])
        torch.testing.assert_close(got["hidden"], ref["hidden"], rtol=1e-4, atol=1e-4)
        # Rank 0 owns the first 1/world_size of the Mamba heads; its carried SSM state (what decode
        # starts from) must equal that slice of the TP=1 state.
        h_pr = TINY["mamba_num_heads"] // world_size
        for s_ref, s_got in zip(ref["ssm_state_rank0"], got["ssm_state_rank0"]):
            torch.testing.assert_close(s_got, s_ref[:, :h_pr], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("is_prefill,expert_group", [(True, 16), (True, 3), (False, 16)])
def test_moe_matches_per_expert_reference(is_prefill, expert_group, monkeypatch):
    """The concatenated two-GEMM MoE equals the per-expert sum over the checkpoint's own expert
    tensors: sum_e gate[:, e] * down_e(relu(up_e(x))^2) + shared(x). Guards the expert-major layout
    of the up/down loaders (an expert/column mix-up would still run and stay fluent)."""
    from safetensors.torch import load_file
    monkeypatch.setattr(nh, "_MOE_PREFILL_EXPERT_GROUP", expert_group)   # 3 does not divide E=8
    cfg = dict(TINY, num_hidden_layers=1, hybrid_override_pattern="E")
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(tmp, cfg)
        dist.init_process_group("gloo", init_method=f"file://{os.path.join(tmp, 'init')}",
                                rank=0, world_size=1)
        try:
            nh.get_tp_group = lambda: _TPGroup()
            moe = _build(cfg, tmp).model.layers[0].mixer
            w = load_file(os.path.join(tmp, "model.safetensors"))
            x = torch.randn(SEQ_LEN, cfg["hidden_size"], generator=torch.Generator().manual_seed(3))
            with torch.no_grad():
                got = moe(x, is_prefill=is_prefill)
                gate = moe._gate_dense(x)
                p = "backbone.layers.0.mixer"
                ref = torch.zeros_like(x)
                for e in range(cfg["n_routed_experts"]):
                    h = F.relu(x @ w[f"{p}.experts.{e}.up_proj.weight"].T).pow(2)
                    ref += gate[:, e:e + 1] * (h @ w[f"{p}.experts.{e}.down_proj.weight"].T)
                sh = F.relu(x @ w[f"{p}.shared_experts.up_proj.weight"].T).pow(2)
                ref += sh @ w[f"{p}.shared_experts.down_proj.weight"].T
            assert (gate > 0).sum(1).eq(cfg["num_experts_per_tok"]).all()
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
        finally:
            dist.destroy_process_group()
