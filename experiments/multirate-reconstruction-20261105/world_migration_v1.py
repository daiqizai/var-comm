"""Explicit 4-to-8 DDP migration, preserving the global training problem."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from runtime import write


def positions(cursor, global_batch, world, microbatch):
    assert global_batch % (world * microbatch) == 0
    return [[cursor + micro * world * microbatch + rank * microbatch + j
             for micro in range(global_batch // (world * microbatch))
             for j in range(microbatch)] for rank in range(world)]


def new_rank_seed(checkpoint_sha, rank):
    assert len(checkpoint_sha) == 64 and 4 <= rank < 8
    value = f'multirate-world4-to8-v1:{checkpoint_sha}:{rank}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % (2**32)


def rng_digest(state):
    h = hashlib.sha256()
    h.update(repr(state['python']).encode())
    n = state['numpy']; h.update(n[0].encode()); h.update(n[1].tobytes()); h.update(repr(n[2:]).encode())
    for name in ['torch', 'cuda']:
        h.update(state[name].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def restore_rank(state, request, rank, world, restore, capture):
    old_world = state['world_size']
    if old_world == world:
        restore(state['rng_by_rank'][rank])
        return dict(migrated=False, rank=rank, old_world=old_world, world=world, policy='exact saved rank RNG', rng_sha256=rng_digest(capture()))
    config = request['world_migration']
    assert old_world == config['from_world'] == 4
    assert world == config['to_world'] == 8
    assert request['resume']['sha256'] == config['checkpoint_sha256']
    assert state['global_batch'] == request['global_batch'] == 32
    assert len(state['rng_by_rank']) == 4
    if rank < 4:
        restore(state['rng_by_rank'][rank])
        seed = None; policy = 'retain saved logical rank 0..3 stream'
    else:
        seed = new_rank_seed(config['checkpoint_sha256'], rank)
        assert seed == config['new_rank_seeds'][str(rank)]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        policy = 'new logical rank 4..7 deterministic checkpoint-derived stream'
    return dict(migrated=True, rank=rank, old_world=old_world, world=world, policy=policy,
                seed=seed, rng_sha256=rng_digest(capture()), bitwise_old_trajectory_claim=False)


def _hash_tree(value, digest):
    if torch.is_tensor(value):
        digest.update(str(value.dtype).encode()); digest.update(repr(tuple(value.shape)).encode())
        tensor = value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8)
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: (type(x).__name__, repr(x))):
            digest.update(repr(key).encode()); _hash_tree(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:_hash_tree(item, digest)
    else:
        digest.update(repr(value).encode())


def training_state_digests(model, optimizer):
    model_hash = hashlib.sha256(); optimizer_hash = hashlib.sha256()
    _hash_tree(dict(model.named_parameters()), model_hash)
    _hash_tree(optimizer.state_dict(), optimizer_hash)
    return dict(parameters_sha256=model_hash.hexdigest(), optimizer_sha256=optimizer_hash.hexdigest())


def audit_first_update(model, optimizer, examples, request, step, cursor, out, rank, world):
    """The accepted update is retained as main training; no replay or discard."""
    local = dict(rank=rank, **training_state_digests(model, optimizer), examples=examples)
    all_ranks = [None] * world
    dist.all_gather_object(all_ranks, local)
    assert len({r['parameters_sha256'] for r in all_ranks}) == 1, 'DDP parameters differ across ranks'
    assert len({r['optimizer_sha256'] for r in all_ranks}) == 1, 'DDP optimizer states differ across ranks'
    actual = sorted([e for r in all_ranks for e in r['examples']], key=lambda e: e['position'])
    assert [r['position'] for r in actual] == list(range(cursor - request['global_batch'], cursor))
    expected = json.loads(Path(request['world_migration']['preflight_path']).read_text())['expected_first_batch']
    assert actual == expected, 'Migrated first global batch differs from checkpoint-cursor audit'
    if rank == 0:
        write(out / 'first_update_audit.json', dict(passed=True, step=step, world=world,
              all_parameter_and_optimizer_hashes_equal=True, global_batch=32,
              samples_and_augmentations_match=True, ranks=all_ranks,
              update_is_retained_main_training=True, original_four_card_trajectory_bitwise=False))
    dist.barrier()
