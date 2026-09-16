"""Original-parent initialization and matched training opportunity for Joint grid controls."""

import json
from pathlib import Path

import torch
import yaml

from joint_sender.common import settings as joint_settings
from joint_sender.evaluation import matched_milestone
from innovation_comm.common import load_parent
from wetok_comm.common import PROJECT, sha256, verify_sources
from .model import GridJointSystem, VARIANTS


EXPERIMENT = Path(__file__).resolve().parents[2]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'


def settings():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    if sha256(JOINT / 'configs/study.yaml') != config['reference_joint_config_sha256']:
        raise RuntimeError('original Joint recipe changed')
    original, reference, base, parent = joint_settings()
    shared = ('unchanged_loss', 'learning_rate', 'gradient_clip', 'global_data_start', 'effective_batch_size',
        'micro_batch_size', 'first_updates', 'planned_updates', 'full_calibration_steps', 'monitor_every',
        'select_by', 'visual_models_frozen', 'decoder_interface', 'channel', 'comparison_information')
    if (tuple(config['variants']) != VARIANTS or config['geometry'] != reference['geometry'] or
        any(config[key] != original[key] for key in shared)):
        raise RuntimeError('grid controls changed another registered training or physical variable')
    return config, original, reference, base, parent


def initial_system(parent, variant, reference, device, update_sender=True):
    torch.manual_seed(reference['training']['new_parameter_seed'])
    return GridJointSystem(parent, variant, reference['fusion'], update_sender=update_sender).to(device)


def output_path(config, kind):
    output = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not output.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('grid outputs escaped the project')
    return output


def completed_joint_reference(config, original, reference):
    joint, joint_path, joint_review, frozen, frozen_path, frozen_review = matched_milestone(original, reference, 5000)
    if sha256(joint_path) != config['required_joint_milestone_sha256']:
        raise RuntimeError('the matched original Joint endpoint changed')
    evaluation_path = PROJECT / 'outputs' / original['outputs']['evaluation'] / 'step_0005000/completion.json'
    analysis_path = PROJECT / 'outputs' / original['outputs']['analysis'] / 'development_0005000/completion.json'
    evaluation, analysis = json.loads(evaluation_path.read_text()), json.loads(analysis_path.read_text())
    if (evaluation['status'] != 'JOINT_EVALUATION_COMPLETE' or evaluation['rows'] != 33600 or
        analysis['status'] != 'JOINT_DEVELOPMENT_ANALYSIS_COMPLETE_NOT_RESEARCH_COMPLETE' or
        analysis['main_rows'] != 33600 or analysis['evaluation_receipt_sha256'] != sha256(evaluation_path) or
        any(record['milestone_sha256'] != sha256(joint_path) for record in (evaluation, analysis))):
        raise RuntimeError('the original Joint trial has not completed its full registered analysis')
    for record in (evaluation, analysis):
        verify_sources(record['source_hashes'])
    for relative, expected in analysis['output_hashes'].items():
        if sha256(analysis_path.parent / relative) != expected:
            raise RuntimeError('the original Joint analysis artifact changed')
    paths = {'joint_milestone': joint_path, 'joint_review': joint_review, 'frozen_milestone': frozen_path,
        'frozen_review': frozen_review, 'joint_evaluation': evaluation_path, 'joint_analysis': analysis_path}
    return {'paths': {key: str(path) for key, path in paths.items()},
            'hashes': {key: sha256(path) for key, path in paths.items()}, 'joint': joint, 'frozen': frozen}
