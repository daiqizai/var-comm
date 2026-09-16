"""Frozen-history bindings for the global symbol-geometry comparison."""

import json
from pathlib import Path

import torch
import yaml

from .common import EXPERIMENT, PROJECT, sha256, verify_sources
from .geometry_candidate import matched_geometry_initializations
from .interface_study import load_interface_study


def load_geometry_study():
    config = yaml.safe_load((EXPERIMENT / 'configs/geometry_study.yaml').read_text())
    interface, base, prefix = load_interface_study(EXPERIMENT / config['interface_config'])
    if config['candidate_geometry'] != [204, 30] or config['control_geometry'] != [153, 40]:
        raise ValueError('unregistered geometry change')
    if config['variants'] != base['arms'] or config['training']['image_weights'] != interface['weights']:
        raise ValueError('this geometry study cannot change the receiver comparison or objective')
    if config['training']['representation_weights'] != base['training']['representation_weights']:
        raise ValueError('representation supervision differs from the frozen controls')
    qualification_path = PROJECT / 'outputs' / config['control_qualification']
    if sha256(qualification_path) != config['control_qualification_sha256']:
        raise RuntimeError('historical control qualification changed')
    qualification = json.loads(qualification_path.read_text())
    verify_sources(qualification['source_hashes'])
    control = PROJECT / 'outputs' / config['control_training']
    if sha256(control / config['control_milestone']) != config['control_milestone_sha256']:
        raise RuntimeError('frozen control milestone changed')
    evaluation = PROJECT / 'outputs' / config['control_evaluation']
    if sha256(evaluation / 'completion.json') != config['control_evaluation_receipt_sha256']:
        raise RuntimeError('frozen control evaluation changed')
    if qualification['initialization_cpu_threads'] != config['training']['initialization_cpu_threads']:
        raise RuntimeError('initialization environment differs from the qualified controls')
    return config, base, qualification


def geometry_output(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('geometry outputs escaped the project')
    return path


def geometry_network(config, base, qualification, variant, device):
    from .training import module_sha256

    if torch.get_num_threads() != config['training']['initialization_cpu_threads']:
        raise RuntimeError('use the original 8-thread QR initialization environment')
    control, candidate, match = matched_geometry_initializations(base, variant)
    expected = next(row for row in qualification['control_variants'] if row['variant'] == variant)
    if module_sha256(control) != expected['control_initial_sha256']:
        raise RuntimeError('real original initialization is not reproduced')
    return candidate.to(device), match
