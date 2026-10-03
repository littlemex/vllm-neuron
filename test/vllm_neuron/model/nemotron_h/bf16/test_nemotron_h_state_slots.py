# SPDX-License-Identifier: Apache-2.0
"""The Mamba state-pool row assignment (state_slots.py) under a simulated Neuron scheduler: requests
are admitted one prefill at a time (some over several segments), decode steps run every running
request in a shuffled order with padding rows, requests finish at random, and freed first-block ids
are handed to new requests. Every running request must keep the row it got at its first segment,
and no two running requests may share a row or use the scratch row."""
import random

import pytest
import torch

from vllm_neuron.model.nemotron_h.state_slots import decode_slots, pool_size, prefill_slot


@pytest.mark.parametrize("max_num_seqs,seed", [(1, 0), (2, 1), (4, 2), (8, 3)])
def test_rows_are_stable_and_distinct(max_num_seqs, seed):
    rng = random.Random(seed)
    S = pool_size(max_num_seqs)
    owner = torch.full((S,), -1, dtype=torch.int32)
    live = torch.zeros(S, dtype=torch.int32)
    free_blocks = list(range(1, 4 * max_num_seqs + 1))      # block 0 is the null block
    running = {}                                             # first block -> row
    for _ in range(400):
        if len(running) < max_num_seqs and rng.random() < 0.5:
            free_blocks.sort()
            key = free_blocks.pop(0)                         # lowest free block, so ids get reused
            row = None
            for seg in range(rng.randint(1, 3)):
                slot, owner, live = prefill_slot(owner, live, torch.tensor(key), torch.tensor(seg == 0))
                row = int(slot) if row is None else row
                assert int(slot) == row, "a later segment moved to another row"
            assert row != S - 1, "a request got the scratch row"
            running[key] = row
        elif running:
            keys = list(running)
            rng.shuffle(keys)
            pad = rng.randint(0, 2)
            k = torch.tensor(keys + [-1] * pad)
            real = torch.tensor([True] * len(keys) + [False] * pad)
            slots, live = decode_slots(owner, live, k, real)
            got = [int(x) for x in slots[:len(keys)]]
            assert got == [running[kk] for kk in keys]
            assert all(int(x) == S - 1 for x in slots[len(keys):])
            assert len(set(got)) == len(got)
            for kk in keys:                                  # some requests finish after this step
                if rng.random() < 0.3:
                    del running[kk]
                    free_blocks.append(kk)


def test_prefill_drops_stale_row_of_reused_block():
    """A request that finished in the last decode step still looks live; a new request reusing its
    first block must get a new row and the stale row must be released, so a later segment of the new
    request cannot land on it."""
    S = pool_size(2)
    owner = torch.full((S,), -1, dtype=torch.int32)
    live = torch.zeros(S, dtype=torch.int32)
    slot_a, owner, live = prefill_slot(owner, live, torch.tensor(7), torch.tensor(True))
    _, live = decode_slots(owner, live, torch.tensor([7]), torch.tensor([True]))
    # request on block 7 finishes; a new one gets block 7 before the next decode step
    slot_b, owner, live = prefill_slot(owner, live, torch.tensor(7), torch.tensor(True))
    assert int(slot_b) != int(slot_a)
    assert int(live[slot_a]) == 0 and int(owner[slot_a]) == -1
    slot_c, owner, live = prefill_slot(owner, live, torch.tensor(7), torch.tensor(False))
    assert int(slot_c) == int(slot_b)
