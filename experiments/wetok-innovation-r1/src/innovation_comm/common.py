"""Pinned parent/model/data bindings for fixed-transmitter receiver training."""

import json
from pathlib import Path

import torch
import yaml

from wetok_comm.common import PROJECT, settings as base_settings, sha256, verify_sources
from wetok_comm.geometry_candidate import GeometryJSCC
from .model import InnovationSystem, VARIANTS


EXPERIMENT = Path(__file__).resolve().parents[2]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'


def settings():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    base = base_settings(REFERENCE / 'configs/study.yaml')
    if tuple(config['variants']) != VARIANTS or config['geometry'] != [204, 30]:
        raise ValueError('registered receiver matrix or common geometry changed')
    if config['training']['weights'] != base['training']['joint_weights']:
        raise ValueError('receiver innovation study does not change the image/source objective')
    if not config['fusion']['detach_ratio_for_gate'] or config['training']['global_data_start'] != 7000:
        raise ValueError('registered training path or continuation exposure changed')
    parent = PROJECT / 'outputs' / config['parent_training']
    milestone_path = parent / config['parent_milestone']
    model_path = parent / config['parent_model']
    if sha256(milestone_path) != config['parent_milestone_sha256'] or sha256(model_path) != config['parent_model_sha256']:
        raise RuntimeError('frozen common transmitter/receiver parent changed')
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    choice = milestone['selected']['single_pass']
    if choice['checkpoint_sha256'] != config['parent_model_sha256'] or choice['step'] != 5000:
        raise RuntimeError('common parent is not the registered full-calibration single-pass choice')
    return config, base, milestone


def load_parent(config, base, device):
    model_config = {**base['model'], 'channel_positions': 204, 'channel_features': 30}
    torch.manual_seed(base['training']['initialization_seed'])
    parent = GeometryJSCC('single_pass', model_config)
    stored = torch.load(PROJECT / 'outputs' / config['parent_training'] / config['parent_model'], map_location='cpu', weights_only=True)
    if stored['variant'] != 'single_pass' or stored['geometry'] != [204, 30] or stored['image_step'] != 5000:
        raise RuntimeError('parent geometry or selected training state changed')
    parent.load_state_dict(stored['model'], strict=True)
    return parent.to(device).eval().requires_grad_(False)


def new_system(parent, variant, config, device):
    torch.manual_seed(config['training']['new_parameter_seed'])
    return InnovationSystem(parent, variant, config['fusion']).to(device)


def output_path(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('innovation outputs escaped the project')
    return path
