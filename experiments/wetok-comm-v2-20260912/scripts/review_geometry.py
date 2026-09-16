"""Audit geometry training and compare only equal-budget full-calibration candidates."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from wetok_comm.common import PROJECT, artifact_hashes, configure_torch, now, output_path, sha256, snapshot, verify_sources, write_json
from wetok_comm.geometry_study import geometry_network, geometry_output, load_geometry_study
from wetok_comm.training import module_sha256, paired_batches


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, qualification = load_geometry_study()
    image_budget = arguments.total - 2000
    if image_budget not in config['calibration']['full_image_steps']:
        raise ValueError('no qualified historical control checkpoint exists at this image budget')
    if not arguments.execute:
        print('PLAN ONLY: equal-budget calibration and full sampler/optimizer audit; no development or GPU')
        return
    configure_torch()
    training = geometry_output(config, 'training')
    milestone_path = training / 'milestones' / f'total_{arguments.total:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if milestone['status'] != 'GEOMETRY_MILESTONE_COMPLETE' or milestone['total_updates_per_new_arm'] != arguments.total:
        raise RuntimeError('geometry milestone is not complete')
    verify_sources(milestone['source_hashes'])
    if sha256(milestone['checkpoint']) != milestone['optimizer_checkpoint_sha256']:
        raise RuntimeError('geometry optimizer checkpoint changed')
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    initial = json.loads((training / 'initialization.json').read_text())
    for variant in config['variants']:
        network, unused = geometry_network(config, base, qualification, variant, 'cpu')
        if module_sha256(network) != initial['candidate_initial_hashes'][variant]:
            raise RuntimeError('actual candidate initialization differs from the matched construction')
        if len(saved['histories'][variant]) != arguments.total:
            raise RuntimeError('unequal candidate update budgets')
        if any(int(value['step']) != image_budget for value in saved['optimizers'][variant]['state'].values()):
            raise RuntimeError('geometry fresh image-Adam counters differ from the common schedule')
        del network
    if not saved['image_optimizer_reset']:
        raise RuntimeError('fresh Adam at representation boundary was skipped')
    maximum_power_error = 0.
    for index, batch in enumerate(paired_batches(base, 20000, 0, arguments.total)):
        phase = 'representation' if index < 2000 else 'image'
        step = index + 1 if phase == 'representation' else index + 1 - 2000
        for variant in config['variants']:
            row = saved['histories'][variant][index]
            if (row['total_step'] != index + 1 or row['phase'] != phase or row['optimizer_step'] != step or
                row['learning_rate'] != config['training'][phase + '_learning_rate'] or row['batch_sha256'] != batch['fingerprint']):
                raise RuntimeError('candidate data/noise/phase/reset schedule differs from the qualified controls')
            if not np.isfinite(row['loss']):
                raise RuntimeError('nonfinite loss in retained geometry history')
            maximum_power_error = max(maximum_power_error, row['power_max_error'])
    if maximum_power_error > 1e-5:
        raise RuntimeError('candidate physical energy drift')
    identifiers = json.loads((output_path(base, 'cache') / 'calibration_ids.json').read_text())
    expected_grid = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    control = PROJECT / 'outputs' / config['control_training']
    summary, deltas, curves = [], [], []
    for variant in config['variants']:
        pools = {'153x40': [], '204x30': []}
        for step in config['calibration']['full_image_steps']:
            if step > image_budget:
                continue
            files = {'153x40': control / ('continuous_mean__' + variant) / f'full_{step:07d}.csv',
                     '204x30': training / variant / f'full_total_{step + 2000:07d}.csv'}
            for geometry, path in files.items():
                rows = read_rows(path)
                if len(rows) != len(expected_grid) or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected_grid:
                    raise RuntimeError('equal-budget full-calibration population mismatch')
                lpips = float(np.mean([float(row['lpips']) for row in rows]))
                pools[geometry].append((lpips, step, rows))
                curves.append({'variant': variant, 'geometry': geometry, 'image_step': step, 'lpips': lpips})
        best = {name: min(pool, key=lambda item: (item[0], item[1])) for name, pool in pools.items()}
        chosen = milestone['selected'][variant]
        if best['204x30'][1] != chosen['step'] or abs(best['204x30'][0] - chosen['lpips']) > 1e-12:
            raise RuntimeError('candidate selection used a different budget or objective')
        if sha256(training / chosen['checkpoint']) != chosen['checkpoint_sha256']:
            raise RuntimeError('selected candidate model changed')
        selected_rows = {}
        for geometry, (lpips, step, rows) in best.items():
            selected_rows[geometry] = {(row['image_id'], float(row['snr_db'])): row for row in rows}
            for snrs in [base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]]:
                subset = [row for row in rows if float(row['snr_db']) in snrs]
                summary.append({'variant': variant, 'geometry': geometry, 'available_image_budget': image_budget,
                    'selected_image_step': step, 'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                    **{metric: float(np.mean([float(row[metric]) for row in subset])) for metric in
                        ('psnr_db', 'ssim', 'lpips', 'bit_error_rate', 'feature_mse', 'state_error')}})
        for metric in ('psnr_db', 'ssim', 'lpips', 'bit_error_rate', 'feature_mse', 'state_error'):
            differences = [float(selected_rows['204x30'][key][metric]) - float(selected_rows['153x40'][key][metric]) for key in expected_grid]
            deltas.append({'variant': variant, 'metric': metric, 'candidate_minus_control': float(np.mean(differences)),
                'population': 'calibration_used_for_selection_not_independent_confirmation', 'available_image_budget': image_budget})
    output = geometry_output(config, 'analysis') / f'calibration_total_{arguments.total:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/geometry_study.yaml'])
    write_csv(output / 'matched_calibration_summary.csv', summary)
    write_csv(output / 'matched_calibration_deltas.csv', deltas)
    write_csv(output / 'matched_calibration_curve.csv', curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, variant in zip(axes, config['variants']):
        for geometry in ('153x40', '204x30'):
            rows = sorted((row for row in curves if row['variant'] == variant and row['geometry'] == geometry), key=lambda row: row['image_step'])
            axis.plot([row['image_step'] for row in rows], [row['lpips'] for row in rows], marker='o', label=geometry)
        axis.set(title=variant, xlabel='Image-phase updates', ylabel='Full calibration LPIPS')
        axis.grid(alpha=.2)
    axes[0].legend()
    figure.savefig(output / 'matched_calibration.png', dpi=170)
    plt.close(figure)
    write_json(output / 'completion.json', {'status': 'GEOMETRY_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'milestone_sha256': sha256(milestone_path), 'new_total_updates': arguments.total,
        'matched_image_budget': image_budget, 'data_noise_phase_Adam_power_selection_audit': 'PASS',
        'maximum_power_error': maximum_power_error, 'source_hashes': sources, 'output_hashes': artifact_hashes(output),
        'new_development_access': False, 'new_holdout_access': False, 'research_goal_complete': False})
    print(json.dumps(deltas, indent=2))


if __name__ == '__main__':
    main()
