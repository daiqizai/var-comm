"""Audit an actual R3 model/Adam/data endpoint and compare calibration at equal update opportunities."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT.parent / 'wetok-comm-v2-20260912/scripts')]

import numpy as np
import torch

from train_milestone import choose, write_csv
from vector_control.common import calibration_summary, initial_system, load_parent, output_path, qualified_reference, settings, tree_digest, validate_optimizer
from wetok_comm.common import PROJECT, artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, read_population


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, r2, original, grid, reference, base, parent_record = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step <= 0:
        raise ValueError('review only a registered completed milestone')
    if not arguments.execute:
        print('PLAN ONLY: actual R3 model/Adam/history/calibration audit; no inference or future-budget reference comparison')
        return
    configure_torch()
    qualified = qualified_reference(config, r2, grid)
    root = output_path(config, 'training')
    metadata = json.loads((root / 'metadata.json').read_text())
    verify_sources(metadata['source_hashes'])
    profile_path = output_path(config, 'profile') / 'profile.json'
    qualification_path = output_path(config, 'preparation') / 'initialization.json'
    if metadata['profile_sha256'] != sha256(profile_path) or metadata['qualification_sha256'] != sha256(qualification_path):
        raise RuntimeError('R3 qualification/profile changed after activation')
    milestone_path = root / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if (milestone['status'] != 'R3_VECTOR_CONTROL_MILESTONE_COMPLETE' or milestone['completed_updates'] != arguments.step or
        milestone['reference_bindings'] != qualified['bindings'] or sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('R3 milestone identity, budget or checkpoint changed')
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    if (saved['completed_updates'] != arguments.step or saved['global_data_step'] != 7000 + arguments.step or
        saved['frozen'] != qualified['frozen'] or saved['reference_bindings'] != qualified['bindings']):
        raise RuntimeError('R3 actual model or history lineage differs from the registered endpoint')
    parent = load_parent(reference, base, 'cpu')
    model = initial_system(parent, reference, 'cpu')
    if tree_digest(model.state_dict()) != tree_digest(qualified['zero_state']):
        raise RuntimeError('R3 original initialization changed')
    initial_sha = module_sha256(model)
    model.load_state_dict(saved['model'], strict=True)
    if module_sha256(model) == initial_sha:
        raise RuntimeError('R3 endpoint did not change the initial communication model')
    validate_optimizer(saved['optimizer'], [parameter for parameter in model.parameters() if parameter.requires_grad], config, base, arguments.step)
    history = saved['history']
    if len(history) != arguments.step:
        raise RuntimeError('R3 update history is incomplete')
    for row, expected in zip(history, qualified['trace']):
        if any(row[key] != expected[key] for key in ('step', 'global_data_step', 'batch_sha256', 'paired_standard_noise_sha256')):
            raise RuntimeError('R3 did not match the complete residual data/noise history')
        if row['power_max_error'] > 1e-5 or any(not np.isfinite(row[key]) for key in ('loss', 'gradient_norm',
            'encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
            raise RuntimeError('R3 has an invalid power or gradient record')
    if saved['calibration_files']['0'] != qualified['zero_calibration']:
        raise RuntimeError('R3 replaced the shared qualified initial calibration')
    peer_path = PROJECT / 'outputs' / r2['outputs']['training'] / 'milestones/step_0010000.json'
    if sha256(peer_path) != qualified['bindings']['R2_milestone']:
        raise RuntimeError('completed residual lineage changed')
    peer_milestone = json.loads(peer_path.read_text())
    peer = torch.load(peer_milestone['checkpoint'], map_location='cpu', weights_only=True)
    identifiers = read_population(base, 'calibration')[2]
    expected_grid = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    if len(identifiers) != 1000 or len(expected_grid) != 5000:
        raise RuntimeError('R3 calibration population changed')
    choices, curves, chosen_rows = {}, [], {}
    for name, bindings in ((config['variant'], saved['calibration_files']),
        (config['control_variant'], peer['calibration_files'][config['control_variant']])):
        selected = None
        for step in [point for point in config['full_calibration_steps'] if point <= arguments.step]:
            binding = bindings[str(step)]
            if sha256(binding['path']) != binding['sha256']:
                raise RuntimeError('a complete-calibration artifact changed')
            rows = read_rows(binding['path'])
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected_grid:
                raise RuntimeError('calibration omitted source images or conditions')
            means, summary = calibration_summary(rows, step, name, 'full')
            curves.extend(summary)
            candidate = {'step': step, 'scope': 'full', 'source_images': 1000, **means}
            updated = choose(selected, candidate)
            if selected is None or updated['step'] != selected['step']:
                chosen_rows[name] = rows
            selected = updated
        choices[name] = selected
    if saved['selected']['step'] != choices[config['variant']]['step'] or sha256(saved['selected']['checkpoint']) != saved['selected']['checkpoint_sha256']:
        raise RuntimeError('R3 selected a different model from the registered full-calibration rule')
    metrics = ('psnr_db', 'lpips', 'ssim', 'bit_error_rate', 'state_error')
    summaries = []
    for snrs in (base['channel']['snrs_db'], base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]):
        for name, rows in chosen_rows.items():
            subset = [row for row in rows if float(row['snr_db']) in snrs]
            summaries.append({'variant': name, 'available_updates': arguments.step, 'selected_step': choices[name]['step'],
                'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                **{key: float(np.mean([float(row[key]) for row in subset])) for key in metrics}})
    output = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    write_csv(output / 'curves.csv', curves)
    write_csv(output / 'summary.csv', summaries)
    write_json(output / 'completion.json', {'status': 'R3_CALIBRATION_AND_HISTORY_AUDIT_PASS_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'completed_updates': arguments.step, 'global_data_step': saved['global_data_step'],
        'milestone_sha256': sha256(milestone_path), 'reference_bindings': qualified['bindings'],
        'fresh_origin_actual_Adam_new_data_power_full_selection_audit': 'PASS',
        'selected_steps_at_equal_opportunities': {name: choice['step'] for name, choice in choices.items()},
        'source_hashes': sources, 'new_development_inference': False, 'research_goal_complete': False,
        'output_hashes': artifact_hashes(output)})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
