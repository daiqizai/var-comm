"""Pinned same-start receiver-control provenance; preparation does not authorize a live launch."""

from pathlib import Path

import torch
import yaml

from innovation_comm.common import settings as reference_settings, load_parent
from wetok_comm.common import PROJECT, sha256
from .model import JointSenderSystem, VARIANTS


EXPERIMENT = Path(__file__).resolve().parents[2]
REFERENCE = EXPERIMENT.parent / 'wetok-innovation-r1'


def settings():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    if sha256(REFERENCE / 'configs/study.yaml') != config['reference_config_sha256']:
        raise RuntimeError('the matched receiver-only reference recipe changed')
    reference, base, parent = reference_settings()
    if (tuple(config['variants']) != VARIANTS or config['unchanged_loss'] != reference['training']['weights'] or
        config['learning_rate'] != reference['training']['learning_rate'] or config['global_data_start'] != 7000 or
        config['effective_batch_size'] != 4 or config['micro_batch_size'] != 1 or config['required_control_updates'] != 5000 or
        config['planned_updates'] != 5000 or config['first_updates'] != 1000 or
        config['gradient_clip'] != reference['training']['gradient_clip'] or config['monitor_every'] != reference['calibration']['monitor_every'] or
        config['decoder_interface'] != 'continuous_mean' or config['full_calibration_steps'] != reference['calibration']['full_steps'] or
        config['channel'] != {'complex_uses': 3060, 'header_uses': 0, 'data_uses': 3060, 'total_energy': 6120, 'average_complex_energy': 2.0}):
        raise RuntimeError('joint/frozen comparison changed an additional training variable')
    return config, reference, base, parent


def initial_system(parent, variant, reference, device, update_sender=True):
    torch.manual_seed(reference['training']['new_parameter_seed'])
    system = JointSenderSystem(parent, variant, reference['fusion'], update_sender=update_sender).to(device)
    return system


def output_path(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to((PROJECT / 'outputs').resolve()):
        raise ValueError('joint sender outputs escaped the project')
    return path


def complete_control(config, reference):
    from innovation_comm.evaluation import audited_milestone

    milestone, path, review = audited_milestone(reference, config['required_control_updates'])
    if any(name not in milestone['selected'] for name in config['variants']):
        raise RuntimeError('required frozen-sender control is missing')
    return milestone, path, review
