# SPDX-License-Identifier: Apache-2.0
"""Which row of the Mamba state pool each request of a step reads and writes.

The Mamba2 layers carry a recurrent state (SSM + conv history) per request. It lives in a pool of
`S` rows per layer on the device; row S - 1 is a scratch row that padding rows of a batch read and
write. The runner does not hand the model a per-request index, and a request's position in the batch
changes from step to step (finished requests are removed and the batch is condensed), so the pool row
is found from something the runner does pass and that stays fixed for a request's lifetime: the id of
its first KV block. No two live requests share a first block (prefix caching is off), and a block id
is only reused after its request has been freed.

The model keeps two small tensors next to the pool, updated in place each step like the state itself:

    owner[s]  first-block id of the request that holds row s (-1: none)
    live[s]   1 while that request is still running

Steps are either one prefill request (possibly one segment of a longer prompt) or a decode batch of
every running request (the Neuron scheduler never mixes the two). That gives the two rules:

  - decode: each row of the batch reads and writes the live row its first block owns. A row that no
    request in this decode batch owns is released (live = 0): its request has finished.
  - prefill, first segment: take the lowest free row, and drop any older row recorded for the same
    first block (its request has finished and the block was reused). Later segments find the row by
    owner, like decode.

A request admitted between two decode steps can see rows of requests that finished in the last
decode step still marked live, so the pool holds 2 * max_num_seqs rows plus the scratch row.

Everything is static-shaped tensor arithmetic (comparisons, max-reductions, where), so one compiled
graph serves every step. Indices are found with max over (mask * weight) rather than argmax.
"""
import torch


def pool_size(max_num_seqs: int) -> int:
    """Rows of the state pool for `max_num_seqs` concurrent requests (the last row is scratch)."""
    return 2 * max_num_seqs + 1


def decode_slots(owner, live, keys, real):
    """owner/live [S] int32, keys [B] first-block id per batch row, real [B] bool (False for
    padding rows) -> (slots [B] int64, new live [S] int32)."""
    S = owner.shape[0]
    ar = torch.arange(S, device=owner.device)
    usable = (live > 0) & (ar < S - 1)
    match = (owner.view(1, S) == keys.view(-1, 1).to(owner.dtype)) & usable.view(1, S) & real.view(-1, 1)
    idx = (match.to(torch.float32) * (ar + 1).to(torch.float32).view(1, S)).amax(dim=1) - 1
    slots = torch.where(match.any(dim=1), idx.to(torch.int64), torch.full_like(idx, S - 1, dtype=torch.int64))
    return slots, match.any(dim=0).to(live.dtype)


def prefill_slot(owner, live, key, is_first):
    """owner/live [S] int32, key [1] first-block id of the prefill request, is_first [1] bool (first
    segment of its prompt) -> (slot [1] int64, new owner [S], new live [S]). Every intermediate keeps
    a dimension of size 1: on the Neuron graph path a reduction to a 0-d tensor comes back with the
    wrong shape."""
    S = owner.shape[0]
    ar = torch.arange(S, device=owner.device)
    not_scratch = ar < S - 1
    key = key.reshape(1).to(owner.dtype)
    is_first = is_first.reshape(1)
    same = owner == key
    # later segment: the live row this request already holds
    held = same & (live > 0) & not_scratch
    held_idx = (held.to(torch.float32) * (ar + 1).to(torch.float32)).amax(dim=0, keepdim=True) - 1
    # first segment: the lowest free row (weight S - s is largest for the lowest index)
    free = (live == 0) & not_scratch
    free_idx = S - (free.to(torch.float32) * (S - ar).to(torch.float32)).amax(dim=0, keepdim=True)
    scratch = torch.full((1,), S - 1, dtype=torch.int64, device=owner.device)
    new_row = torch.where(free.any(dim=0, keepdim=True), free_idx.to(torch.int64), scratch)
    old_row = torch.where(held.any(dim=0, keepdim=True), held_idx.to(torch.int64), scratch)
    slot = torch.where(is_first, new_row, old_row)
    take = (ar == slot) & is_first
    drop = same & is_first
    owner_new = torch.where(take, key, torch.where(drop, torch.full_like(owner, -1), owner))
    live_new = torch.where(take, torch.ones_like(live), torch.where(drop, torch.zeros_like(live), live))
    return slot, owner_new, live_new
