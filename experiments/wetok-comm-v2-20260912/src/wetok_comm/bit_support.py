"""Frozen-parent, single-loss-weight continuation specification and state cloning."""

import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from .common import EXPERIMENT, PROJECT, settings, sha256, verify_sources
from .training import new_network


def load_repair(path=None):
    path = Path(path or EXPERIMENT / 'configs/bit_support.yaml')
    repair = yaml.safe_load(path.read_text())
    base = settings(EXPERIMENT / repair['base_config'])
    original, changed = repair['recipes']['joint_original'], repair['recipes']['bit_support']
    if original != base['training']['joint_weights']:
        raise ValueError('the continuation control must retain the registered original objective')
    if {key for key in original if original[key] != changed[key]} != {'bits'} or changed['bits'] != 1.:
        raise ValueError('only the bit BCE coefficient may change in this comparison')
    if repair['variants'] != base['arms'] or repair['parent_step'] != 5000:
        raise ValueError('registered source variants and parent history changed')
    if repair['training']['effective_batch_size'] != 4 or repair['training']['micro_batch_size'] != 1:
        raise ValueError('complete effective/microbatch settings must remain matched')
    parent = PROJECT / 'outputs' / repair['parent_training']
    receipt_path = parent / repair['parent_milestone']
    checkpoint = parent / repair['parent_optimizer_checkpoint']
    if sha256(receipt_path) != repair['parent_milestone_sha256'] or sha256(checkpoint) != repair['parent_optimizer_checkpoint_sha256']:
        raise RuntimeError('frozen model/Adam parent changed')
    receipt = json.loads(receipt_path.read_text())
    verify_sources(receipt['source_hashes'])
    if receipt['updates_per_arm'] != repair['parent_step'] or receipt['checkpoint_sha256'] != sha256(checkpoint):
        raise RuntimeError('parent checkpoint is not the complete registered milestone')
    return repair, base, parent, receipt


def arm_definitions(repair):
    return {f'{recipe}__{variant}': {'variant': variant, 'recipe': recipe, 'weights': weights}
            for variant in repair['variants'] for recipe, weights in repair['recipes'].items()}


def optimizer_digest(state):
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def clone_optimizer(network, saved_state, rate):
    group = saved_state['param_groups'][0]
    optimizer = torch.optim.AdamW(network.parameters(), lr=rate, betas=tuple(group['betas']),
                                 eps=group['eps'], weight_decay=group['weight_decay'], amsgrad=group['amsgrad'])
    optimizer.load_state_dict(copy.deepcopy(saved_state))
    for group in optimizer.param_groups:
        group['lr'] = rate
    return optimizer


def cloned_arm(base, variant, saved, rate, device):
    network = new_network(base, variant, device)
    network.load_state_dict(saved['models'][variant], strict=True)
    return network, clone_optimizer(network, saved['optimizers'][variant], rate)


def repair_output(repair, kind):
    path = (PROJECT / 'outputs' / repair['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('repair output escaped the project')
    return path
