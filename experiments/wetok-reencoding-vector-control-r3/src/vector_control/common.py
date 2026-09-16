"""Frozen residual lineage and exact original-parent initialization for the new vector control."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from grid_controls.common import initial_system as residual_system
from innovation_comm.common import load_parent
from sufficiency.common import settings as r2_settings, tree_digest, validate_optimizer
from sufficiency.evaluation import audited_endpoint
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.training import module_sha256
from .model import PredictionVectorSystem, VARIANT


EXPERIMENT = Path(__file__).resolve().parents[2]


def settings():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    r2, original, grid, reference, base, parent = r2_settings()
    shared = ('learning_rate', 'gradient_clip', 'effective_batch_size', 'micro_batch_size', 'monitor_every',
        'unchanged_loss', 'decoder_interface', 'visual_models_frozen', 'select_by', 'channel')
    if (config['variant'] != VARIANT or config['control_variant'] != 'full_grid_innovation' or
        config['planned_total_updates'] != 10000 or config['global_data_offset'] != 7000 or
        config['full_calibration_steps'] != [0, 1000, 2500, 5000, 7500, 10000] or config['geometry'] != reference['geometry'] or
        config['new_parameter_seed'] != reference['training']['new_parameter_seed'] or any(config[key] != r2[key] for key in shared)):
        raise RuntimeError('R3 changed a factor other than the registered fused vector')
    return config, r2, original, grid, reference, base, parent


def output_path(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('R3 output escaped the project')
    return path


def initial_system(parent, reference, device):
    torch.manual_seed(reference['training']['new_parameter_seed'])
    return PredictionVectorSystem(parent, reference['fusion']).to(device)


def calibration_summary(rows, step, variant, scope):
    metrics = [key for key in rows[0] if key not in ('image_id', 'snr_db') and rows[0][key] != '']
    means = {key: float(np.mean([float(row[key]) for row in rows])) for key in metrics}
    summary = []
    for snr in sorted({float(row['snr_db']) for row in rows}):
        subset = [row for row in rows if float(row['snr_db']) == snr]
        summary.append({'variant': variant, 'step': step, 'scope': scope, 'snr_db': snr, 'source_images': len(subset),
            **{key: float(np.mean([float(row[key]) for row in subset])) if key in metrics else ''
                for key in rows[0] if key not in ('image_id', 'snr_db')}})
    return means, summary


def qualified_reference(config, r2, grid, scan_outputs=False):
    bindings = {}
    for name, requirement in config['reference_results'].items():
        path = PROJECT / 'outputs' / requirement['path']
        if sha256(path) != requirement['sha256']:
            raise RuntimeError('completed R2 reference receipt changed')
        record = json.loads(path.read_text())
        if record['status'] != requirement['status']:
            raise RuntimeError('required R2 result is not complete')
        verify_sources(record['source_hashes'])
        if scan_outputs:
            for relative, expected in record['output_hashes'].items():
                if sha256(path.parent / relative) != expected:
                    raise RuntimeError('a sealed R2 result artifact changed')
        bindings[name] = sha256(path)
    milestone, milestone_path, review_path, origin = audited_endpoint(r2, 10000)
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    variant = config['control_variant']
    trace = saved['histories'][variant]
    if len(trace) != 10000 or [row['global_data_step'] for row in trace] != list(range(7001, 17001)):
        raise RuntimeError('reference residual does not have the complete required data history')
    zero_calibration = saved['calibration_files'][variant]['0']
    if sha256(zero_calibration['path']) != zero_calibration['sha256']:
        raise RuntimeError('original zero-update calibration changed')
    zero_path = Path(zero_calibration['path']).parent / 'checkpoints/step_0000000.pt'
    zero = torch.load(zero_path, map_location='cpu', weights_only=True)
    if zero['variant'] != variant or zero['step'] != 0:
        raise RuntimeError('R3 must match the original residual initialization, not a trained checkpoint')
    old_root = PROJECT / 'outputs' / grid['outputs']['training']
    old_metadata = json.loads((old_root / 'metadata.json').read_text())
    verify_sources(old_metadata['source_hashes'])
    profile_path = PROJECT / 'outputs' / grid['outputs']['profile'] / 'profile.json'
    if sha256(profile_path) != old_metadata['profile_sha256']:
        raise RuntimeError('original initialization profile changed')
    profile = json.loads(profile_path.read_text())
    if profile['frozen'] != saved['frozen']:
        raise RuntimeError('original residual and completed R2 have different frozen visual identities')
    with Path(zero_calibration['path']).open() as handle:
        zero_rows = list(csv.DictReader(handle))
    if len(zero_rows) != 5000 or len({(row['image_id'], float(row['snr_db'])) for row in zero_rows}) != 5000:
        raise RuntimeError('original full calibration source/SNR matrix is incomplete')
    bindings.update({'R2_milestone': sha256(milestone_path), 'R2_review': sha256(review_path),
        'R2_optimizer': milestone['checkpoint_sha256'], 'original_zero_checkpoint': sha256(zero_path),
        'original_zero_calibration': zero_calibration['sha256'], 'original_initialization_profile': sha256(profile_path),
        'residual_full10000_history': tree_digest(trace)})
    return {'bindings': bindings, 'trace': trace, 'zero_state': zero['model'], 'zero_rows': zero_rows,
        'zero_checkpoint': str(zero_path), 'zero_calibration': zero_calibration,
        'initial_model_sha256': profile['model_hashes_before'][variant], 'frozen': saved['frozen'],
        'reference_optimizer': saved['optimizers'][variant], 'reference_selected': milestone['selected'][variant]}


def check_initial_models(parent, model, reference, qualified, config):
    control = residual_system(parent, config['control_variant'], reference, next(model.parameters()).device).eval()
    if (tree_digest(model.state_dict()) != tree_digest(qualified['zero_state']) or
        tree_digest(control.state_dict()) != tree_digest(qualified['zero_state']) or
        module_sha256(model) != qualified['initial_model_sha256'] or module_sha256(parent) != qualified['frozen']['parent']):
        raise RuntimeError('new control did not match the actual original residual initialization')
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != config['expected_communication_parameters'] or any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('R3 changed the optimized parameter scope')
    return control
