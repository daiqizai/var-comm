"""Audit shared observations, frozen transmitter and complete-calibration model selection."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from innovation_comm.common import load_parent, new_system, output_path, settings
from wetok_comm.common import artifact_hashes, configure_torch, now, output_path as base_output, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, paired_batches


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, parent_milestone = settings()
    if arguments.step not in config['calibration']['full_steps'] or arguments.step <= 0:
        raise ValueError('review requires a registered full-calibration milestone')
    if not arguments.execute:
        print('PLAN ONLY: shared-y/frozen-TX/Adam/full-calibration audit, no GPU or development')
        return
    configure_torch()
    training = output_path(config, 'training')
    milestone_path = training / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if milestone['status'] != 'INNOVATION_MILESTONE_COMPLETE' or milestone['receiver_updates_per_arm'] != arguments.step:
        raise RuntimeError('receiver milestone incomplete')
    verify_sources(milestone['source_hashes'])
    if sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']:
        raise RuntimeError('frozen milestone checkpoint changed')
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    parent = load_parent(config, base, 'cpu')
    initial = json.loads((training / 'initialization.json').read_text())
    for name in config['variants']:
        system = new_system(parent, name, config, 'cpu')
        if module_sha256(system) != initial['model_hashes'][name]:
            raise RuntimeError('receiver initialization no longer matches the common parent')
        system.load_state_dict(saved['models'][name], strict=True)
        if module_sha256(system.encoder) != saved['frozen']['encoder']:
            raise RuntimeError('frozen transmitter changed in a receiver arm')
        if len(saved['histories'][name]) != arguments.step or any(int(state['step']) != arguments.step for state in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('unequal receiver training/Adam budgets')
        expected_parameters = len([value for value in system.parameters() if value.requires_grad])
        if len(saved['optimizers'][name]['param_groups'][0]['params']) != expected_parameters:
            raise RuntimeError('optimizer contains a different trainable parameter set')
        del system
    for index, batch in enumerate(paired_batches(base, 20000, 7000, 7000 + arguments.step)):
        observations = set()
        for name in config['variants']:
            row = saved['histories'][name][index]
            if row['step'] != index + 1 or row['global_data_step'] != batch['step'] + 1 or row['batch_sha256'] != batch['fingerprint']:
                raise RuntimeError('source/augmentation/SNR/noise training sequence differs')
            observations.add(row['shared_received_sha256'])
            if not np.isfinite(row['loss']):
                raise RuntimeError('nonfinite retained objective')
        if len(observations) != 1:
            raise RuntimeError('receivers did not share exactly the same observed waveform')
    identifiers = json.loads((base_output(base, 'cache') / 'calibration_ids.json').read_text())
    expected_grid = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    summary, curves, selected_rows = [], [], {}
    for name in config['variants']:
        pool = []
        for step in config['calibration']['full_steps']:
            if step > arguments.step:
                continue
            rows = read_rows(training / name / f'full_{step:07d}.csv')
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected_grid:
                raise RuntimeError('complete receiver calibration grid differs')
            means = {key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'lpips', 'bit_error_rate')}
            pool.append((means['lpips'], step, rows))
            curves.append({'variant': name, 'step': step, **means})
        choice = min(pool, key=lambda item: (item[0], item[1]))
        recorded = milestone['selected'][name]
        if choice[1] != recorded['step'] or abs(choice[0] - recorded['lpips']) > 1e-12:
            raise RuntimeError('receiver model selection used a different rule or monitoring subset')
        if sha256(training / recorded['checkpoint']) != recorded['checkpoint_sha256']:
            raise RuntimeError('selected receiver weights changed')
        selected_rows[name] = {(row['image_id'], float(row['snr_db'])): row for row in choice[2]}
        for snrs in [base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]]:
            rows = [row for row in choice[2] if float(row['snr_db']) in snrs]
            summary.append({'variant': name, 'snrs_db': '+'.join(map(str, snrs)), 'selected_step': choice[1], 'source_images': 1000,
                **{key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'ssim', 'lpips', 'bit_error_rate', 'state_error')}})
    comparisons = [('multiscale_innovation', control) for control in ('single_pass', 'multiscale_state_history', 'multiscale_prediction_features', 'full_grid_innovation')]
    comparisons.append(('multiscale_state_history', 'multiscale_no_history'))
    deltas = []
    for method, control in comparisons:
        for metric in ('psnr_db', 'lpips', 'bit_error_rate'):
            differences = [float(selected_rows[method][key][metric]) - float(selected_rows[control][key][metric]) for key in expected_grid]
            deltas.append({'method': method, 'control': control, 'metric': metric, 'delta': float(np.mean(differences)),
                           'role': 'calibration_used_for_selection_not_independent_confirmation'})
    output = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'curves.csv', curves)
    write_csv(output / 'paired_calibration_deltas.csv', deltas)
    write_json(output / 'completion.json', {'status': 'INNOVATION_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'shared_y_source_noise_frozen_encoder_Adam_selection_audit': 'PASS',
        'receiver_updates': arguments.step, 'milestone_sha256': sha256(milestone_path), 'source_hashes': sources,
        'selected_steps': {name: value['step'] for name, value in milestone['selected'].items()},
        'output_hashes': artifact_hashes(output), 'new_development_access': False, 'research_goal_complete': False})
    print(json.dumps(deltas, indent=2))


if __name__ == '__main__':
    main()
