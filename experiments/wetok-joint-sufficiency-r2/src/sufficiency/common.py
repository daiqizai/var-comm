"""Frozen four-arm lineage, optimizer-preserving reconstruction and inherited calibration."""

import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from grid_controls.common import initial_system as grid_system, settings as grid_settings
from innovation_comm.common import load_parent
from joint_sender.common import initial_system as basic_system
from wetok_comm.common import PROJECT, sha256, verify_sources
from wetok_comm.training import module_sha256


EXPERIMENT = Path(__file__).resolve().parents[2]
VARIANTS = ('single_pass', 'multiscale_state_history', 'full_grid_state_history', 'full_grid_innovation')
ORIGINS = dict(zip(VARIANTS, ('joint', 'joint', 'grid', 'grid')))


def tree_digest(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        data = str((str(tensor.dtype), tuple(tensor.shape))).encode() + tensor.numpy().tobytes()
    elif isinstance(value, dict):
        data = json.dumps([(str(key), tree_digest(value[key])) for key in sorted(value, key=str)], separators=(',', ':')).encode()
    elif isinstance(value, (list, tuple)):
        data = json.dumps([tree_digest(item) for item in value], separators=(',', ':')).encode()
    else:
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def settings():
    config = yaml.safe_load((EXPERIMENT / 'configs/study.yaml').read_text())
    grid, original, reference, base, parent = grid_settings()
    shared = ('learning_rate', 'gradient_clip', 'effective_batch_size', 'micro_batch_size', 'monitor_every',
              'unchanged_loss', 'decoder_interface', 'visual_models_frozen', 'select_by', 'channel')
    if (tuple(config['variants']) != VARIANTS or config['origins'] != ORIGINS or config['optimizer_reset'] or
        config['initial_total_updates'] != 5000 or config['planned_total_updates'] != 10000 or
        config['new_full_steps'] != [7500, 10000] or config['inherited_full_steps'] != original['full_calibration_steps'] or
        config['global_data_offset'] != 7000 or config['initial_global_data_step'] != 12000 or
        any(config[key] != grid[key] or config[key] != original[key] for key in shared)):
        raise RuntimeError('R2 changed a registered architecture, data, optimizer, loss or physical factor')
    return config, original, grid, reference, base, parent


def output_path(config, kind):
    path = (PROJECT / 'outputs' / config['outputs'][kind]).resolve()
    if not path.is_relative_to(PROJECT / 'outputs'):
        raise ValueError('R2 output escaped the project')
    return path


def load_sources(config):
    audit_path = PROJECT / 'outputs' / config['endpoint_audit']
    if sha256(audit_path) != config['endpoint_audit_sha256']:
        raise RuntimeError('the four-endpoint qualification changed')
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'FOUR_ACTUAL5000_ENDPOINTS_MATCHED_FOR_POSSIBLE_CONTINUATION_NOT_STARTED':
        raise RuntimeError('actual5000 optimizer endpoints were not qualified')
    for required in config['required_completed_results'].values():
        path = PROJECT / 'outputs' / required['path']
        if sha256(path) != required['sha256']:
            raise RuntimeError('a completed source trial changed')
        verify_sources(json.loads(path.read_text())['source_hashes'])
    identities = {row['variant']: row for row in audit['models']}
    if set(identities) != set(config['variants']):
        raise RuntimeError('a continuation arm was omitted or replaced')
    states, milestones, roots = {}, {}, {}
    hashes = {'endpoint_audit': sha256(audit_path)}
    for variant in config['variants']:
        row = identities[variant]
        origin = config['origins'][variant]
        if row['origin'] != origin or row['actual_steps'] != 5000 or row['global_data_step'] != 12000:
            raise RuntimeError('wrong source endpoint or data position')
        if origin not in states:
            milestone_path, checkpoint = Path(row['model_endpoint']), Path(row['optimizer_checkpoint'])
            if sha256(milestone_path) != row['endpoint_sha256'] or sha256(checkpoint) != row['optimizer_checkpoint_sha256']:
                raise RuntimeError('a source model or optimizer checkpoint changed')
            milestone = json.loads(milestone_path.read_text())
            verify_sources(milestone['source_hashes'])
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            count_key = 'completed_joint_updates' if origin == 'joint' else 'completed_grid_updates'
            if state[count_key] != 5000 or state['global_data_step'] != 12000 or state['frozen'] != audit['frozen']:
                raise RuntimeError('source endpoints have unequal actual histories or visual identity')
            states[origin], milestones[origin], roots[origin] = state, milestone, checkpoint.parent.parent
            hashes[origin + '_milestone'] = row['endpoint_sha256']
            hashes[origin + '_optimizer'] = row['optimizer_checkpoint_sha256']
    histories, selected, summaries, calibration_files, prior_timings = {}, {}, [], {}, {}
    calibrated_sets = []
    for variant in config['variants']:
        origin = config['origins'][variant]
        state = states[origin]
        history = state['histories'][variant]
        if len(history) != 5000 or history[-1]['global_data_step'] != 12000:
            raise RuntimeError('source history is not at the actual5000 endpoint')
        histories[variant] = copy.deepcopy(history)
        choice = copy.deepcopy(state['selected'][variant])
        path = roots[origin] / choice['checkpoint']
        if sha256(path) != choice['checkpoint_sha256']:
            raise RuntimeError('inherited legal selection changed')
        choice['checkpoint'] = str(path)
        choice['checkpoint_origin'] = origin
        selected[variant] = choice
        summaries.extend(copy.deepcopy([row for row in state['summaries'] if row['variant'] == variant]))
        calibrated_sets.append(set(state['calibrated']))
        calibration_files[variant] = {}
        for step in config['inherited_full_steps']:
            path = roots[origin] / variant / f'full_{step:07d}.csv'
            calibration_files[variant][str(step)] = {'path': str(path), 'sha256': sha256(path)}
        prior_timings[variant] = {phase: state['timings'][phase][variant] for phase in ('training', 'calibration')}
    if any(value != calibrated_sets[0] for value in calibrated_sets) or 5000 not in calibrated_sets[0]:
        raise RuntimeError('inherited calibration/monitor opportunities differ')
    if len({history[-1]['batch_sha256'] for history in histories.values()}) != 1 or len({history[-1]['paired_standard_noise_sha256'] for history in histories.values()}) != 1:
        raise RuntimeError('the actual5000 data/noise endpoints are not paired')
    return {'states': states, 'identities': identities, 'hashes': hashes, 'frozen': audit['frozen'],
        'histories': histories, 'selected': selected, 'summaries': summaries, 'calibrated': sorted(calibrated_sets[0]),
        'calibration_files': calibration_files, 'prior_timings': prior_timings}


def validate_optimizer(saved, parameters, config, base, total_steps):
    if len(saved['param_groups']) != 1:
        raise RuntimeError('unexpected optimizer groups')
    group = saved['param_groups'][0]
    if (group['lr'] != config['learning_rate'] or tuple(group['betas']) != tuple(base['training']['betas']) or
        group['weight_decay'] != base['training']['weight_decay'] or len(group['params']) != len(parameters) or
        set(saved['state']) != set(group['params'])):
        raise RuntimeError('optimizer recipe or parameter coverage changed')
    for identifier, parameter in zip(group['params'], parameters):
        state = saved['state'][identifier]
        if int(state['step']) != total_steps:
            raise RuntimeError('optimizer state was reset or has a different update count')
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != parameter.shape or state[key].dtype != parameter.dtype or not bool(torch.isfinite(state[key]).all()):
                raise RuntimeError('optimizer moment shape, dtype or finiteness differs')


def restore_systems(config, reference, base, sources, parent, device, resumed=None):
    systems, optimizers, audit = {}, {}, {}
    for variant in config['variants']:
        factory = basic_system if config['origins'][variant] == 'joint' else grid_system
        model = factory(parent, variant, reference, device)
        origin = sources['states'][config['origins'][variant]] if resumed is None else resumed
        model_state, optimizer_state = origin['models'][variant], origin['optimizers'][variant]
        count = config['initial_total_updates'] if resumed is None else resumed['completed_total_updates']
        model.load_state_dict(model_state, strict=True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        validate_optimizer(optimizer_state, parameters, config, base, count)
        optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=tuple(base['training']['betas']),
                                    weight_decay=base['training']['weight_decay'])
        optimizer.load_state_dict(optimizer_state)
        if tree_digest(model.state_dict()) != tree_digest(model_state) or tree_digest(optimizer.state_dict()) != tree_digest(optimizer_state):
            raise RuntimeError('restored model/Adam values differ from the actual source endpoint')
        systems[variant], optimizers[variant] = model, optimizer
        audit[variant] = {'model_sha256': module_sha256(model), 'model_tree_sha256': tree_digest(model_state),
            'optimizer_tree_sha256': tree_digest(optimizer_state), 'optimizer_steps': count,
            'parameter_tensors': len(parameters), 'communication_parameters': sum(parameter.numel() for parameter in parameters),
            'parameter_names_sha256': tree_digest([name for name, parameter in model.named_parameters() if parameter.requires_grad])}
    return systems, optimizers, audit
