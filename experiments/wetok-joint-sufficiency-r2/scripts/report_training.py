"""Read only a closed R2 milestone to plot full calibration and account for continuation cost."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from joint_sender.evaluation_io import write_rows
from sufficiency.common import output_path, settings
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json


LABELS = {'single_pass': 'Single pass', 'multiscale_state_history': 'Multiscale state',
    'full_grid_state_history': 'Full-grid state', 'full_grid_innovation': 'Full-grid innovation'}
COLORS = dict(zip(LABELS, ('#4C78A8', '#F58518', '#54A24B', '#B279A2')))
TRAIN_METRICS = ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate')


def training_bins(saved, config, width=100):
    if width <= 0:
        raise ValueError('training display bin width must be positive')
    rows = []
    for name in config['variants']:
        history = saved['histories'][name]
        if len(history) != saved['completed_total_updates']:
            raise RuntimeError('closed training history is incomplete')
        for expected_step, row in enumerate(history, 1):
            expected_loss = sum(config['unchanged_loss'][metric] * float(row[metric]) for metric in ('mse', 'lpips', 'bits', 'state'))
            if row['step'] != expected_step or abs(float(row['loss']) - expected_loss) > 1e-7:
                raise RuntimeError('training history order or registered loss decomposition changed')
            if any(not np.isfinite(float(row[metric])) for metric in TRAIN_METRICS):
                raise RuntimeError('a closed training history contains a nonfinite metric')
        for start in range(0, len(history), width):
            subset = history[start:start + width]
            rows.append({'variant': name, 'first_step': start + 1, 'last_step': start + len(subset), 'updates': len(subset),
                **{metric: float(np.mean([row[metric] for row in subset])) for metric in TRAIN_METRICS}})
    return rows


def continuation_cost(saved, metadata, config):
    initial, completed = config['initial_total_updates'], saved['completed_total_updates']
    if completed <= initial:
        raise RuntimeError('R2 cost report requires actual new updates')
    rows = []
    for name in config['variants']:
        history = saved['histories'][name]
        if len(history) != completed:
            raise RuntimeError('R2 cost history has the wrong update count')
        new_history = history[initial:]
        seconds = np.array([float(row['step_seconds']) for row in new_history])
        allocated = np.array([int(row['peak_GPU_allocated_bytes']) for row in new_history])
        measured_training = float(saved['timings']['training'][name])
        measured_calibration = float(saved['timings']['calibration'][name])
        prior = metadata['prior_training_and_calibration_seconds_by_arm'][name]
        if (not np.all(np.isfinite(seconds)) or np.any(seconds <= 0) or np.any(allocated < 0) or
            not np.isfinite(measured_training) or not np.isfinite(measured_calibration) or measured_calibration < 0 or
            abs(seconds.sum() - measured_training) > 1e-5):
            raise RuntimeError('R2 per-update timing does not reconcile with continuation accounting')
        rows.append({'variant': name, 'inherited_updates': initial, 'new_updates': completed - initial,
            'available_total_updates': completed, 'selected_step': saved['selected'][name]['step'],
            'historical_branch_training_hours': float(prior['training']) / 3600,
            'historical_branch_calibration_hours': float(prior['calibration']) / 3600,
            'R2_training_hours': measured_training / 3600, 'R2_calibration_hours': measured_calibration / 3600,
            'R2_recorded_component_hours': (measured_training + measured_calibration) / 3600,
            'R2_update_seconds_mean': float(seconds.mean()), 'R2_update_seconds_median': float(np.median(seconds)),
            'R2_update_seconds_p95': float(np.percentile(seconds, 95)),
            'R2_peak_allocated_GiB_in_shared_four_model_process': float(allocated.max()) / 1024 ** 3})
    components = sum(row['R2_recorded_component_hours'] for row in rows)
    elapsed = float(saved['elapsed_seconds']) / 3600
    if not np.isfinite(elapsed) or elapsed + 1e-6 < components:
        raise RuntimeError('R2 component timings exceed observed process time')
    return rows, {'R2_observed_process_hours': elapsed, 'R2_recorded_component_hours': components,
        'R2_unattributed_process_hours': max(0., elapsed - components),
        'historical_shared_parent7000_training_excluded_not_free': True,
        'process_time_is_not_GPU_utilization_integral': True,
        'peak_allocated_is_not_standalone_deployment_memory': True}


def complete_calibration(saved, config, base):
    steps = config['inherited_full_steps'] + [step for step in config['new_full_steps'] if step <= saved['completed_total_updates']]
    expected = {(name, step, float(snr)) for name in config['variants'] for step in steps for snr in base['channel']['snrs_db']}
    rows = [dict(row) for row in saved['summaries'] if row['scope'] == 'full']
    keys = {(row['variant'], int(row['step']), float(row['snr_db'])) for row in rows}
    if len(rows) != len(expected) or keys != expected or any(int(row['source_images']) != 1000 for row in rows):
        raise RuntimeError('full-calibration plot substituted monitor results or lost conditions')
    return rows


def save_figure(figure, output, stem):
    figure.savefig(output / (stem + '.png'), dpi=170, bbox_inches='tight')
    figure.savefig(output / (stem + '.pdf'), bbox_inches='tight')
    plt.close(figure)


def draw_figures(output, training, calibration, costs, config, base, step):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric in zip(axes.flat, TRAIN_METRICS):
        for name in config['variants']:
            subset = [row for row in training if row['variant'] == name]
            axis.plot([row['last_step'] for row in subset], [row[metric] for row in subset], color=COLORS[name], label=LABELS[name])
        axis.axvline(config['initial_total_updates'], color='black', linestyle='--', linewidth=1)
        axis.set_title(metric)
        axis.set_xlabel('Total branch updates after shared 7000 parent')
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f'Training: fixed 100-update means through {step}; dashed line = R2 start\nSame paired batches; loss components are not evidence of image-quality dominance')
    figure.tight_layout(rect=(0, 0, 1, .92))
    save_figure(figure, output, 'training_components')
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, metric, title in zip(axes.flat, ('lpips', 'psnr_db', 'bit_error_rate', 'state_error'), ('LPIPS (lower)', 'PSNR dB (higher)', 'Bit error rate (lower)', 'State error (lower)')):
        for name in config['variants']:
            subset = [row for row in calibration if row['variant'] == name]
            points = sorted({int(row['step']) for row in subset})
            means = [np.mean([float(row[metric]) for row in subset if int(row['step']) == point]) for point in points]
            axis.plot(points, means, marker='o', color=COLORS[name], label=LABELS[name])
        axis.axvline(config['initial_total_updates'], color='black', linestyle='--', linewidth=1)
        axis.set_title(title)
        axis.set_xlabel('Total branch updates')
    axes[0, 0].legend(fontsize=9)
    figure.suptitle('Complete calibration only: 1000 sources x five SNRs (1/4/7/13/19 dB)\nNot development; 100-image monitoring is excluded; DINO does not select checkpoints')
    figure.tight_layout(rect=(0, 0, 1, .91))
    save_figure(figure, output, 'full_calibration')
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for column, snr in enumerate(base['channel']['snrs_db']):
        for row_index, metric in enumerate(('lpips', 'psnr_db')):
            axis = axes[row_index, column]
            for name in config['variants']:
                subset = sorted([row for row in calibration if row['variant'] == name and float(row['snr_db']) == float(snr)], key=lambda row: int(row['step']))
                axis.plot([int(row['step']) for row in subset], [float(row[metric]) for row in subset], marker='o', color=COLORS[name], label=LABELS[name])
            axis.axvline(config['initial_total_updates'], color='black', linestyle='--', linewidth=1)
            axis.set_title(f'{snr:g} dB: {metric}')
            if row_index == 1:
                axis.set_xlabel('Total updates')
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Fixed full-calibration population and noise; scales differ between panels\nNo pooled development or noiseless reference is mixed into these curves')
    figure.tight_layout(rect=(0, 0, 1, .91))
    save_figure(figure, output, 'full_calibration_by_snr')
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(costs))
    training_hours = [row['R2_training_hours'] for row in costs]
    axis.bar(positions, training_hours, label='R2 training', color='#4C78A8')
    axis.bar(positions, [row['R2_calibration_hours'] for row in costs], bottom=training_hours, label='R2 monitor + full calibration', color='#F58518')
    axis.set_xticks(positions, [LABELS[row['variant']] for row in costs])
    axis.set_ylabel('Recorded component hours (shared process)')
    axis.set_title(f'Additional R2 cost: {step - config["initial_total_updates"]} updates per arm\nExcludes historical parent/branch training and process overhead')
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output, 'continuation_cost')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, original, grid, reference, base, parent = settings()
    if arguments.step not in config['new_full_steps']:
        raise ValueError('report only a registered closed7500/10000 milestone')
    if not arguments.execute:
        print('PLAN ONLY: closed milestone -> full-calibration figures and reconciled training-cost tables; no GPU or new inference')
        return
    root = output_path(config, 'training')
    milestone_path = root / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    review_root = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    review_path = review_root / 'completion.json'
    review = json.loads(review_path.read_text())
    if (milestone['status'] != 'SUFFICIENCY_MILESTONE_COMPLETE' or milestone['total_updates_per_arm'] != arguments.step or
        review['model_Adam_inheritance_new_data_power_selection_audit'] != 'PASS' or review['milestone_sha256'] != sha256(milestone_path) or
        sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('training report requires an audited immutable milestone')
    verify_sources(milestone['source_hashes'])
    verify_sources(review['source_hashes'])
    for relative, expected in review['output_hashes'].items():
        if sha256(review_root / relative) != expected:
            raise RuntimeError('completed calibration audit output changed')
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    metadata = json.loads((root / 'metadata.json').read_text())
    if saved['completed_total_updates'] != arguments.step:
        raise RuntimeError('report accidentally loaded a live or different endpoint')
    training = training_bins(saved, config)
    calibration = complete_calibration(saved, config, base)
    costs, total = continuation_cost(saved, metadata, config)
    output = output_path(config, 'analysis') / f'figures_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__)])
    for name, rows in (('training_bins.csv', training), ('full_calibration_by_snr.csv', calibration), ('training_cost.csv', costs)):
        write_rows(output / name, rows)
    draw_figures(output, training, calibration, costs, config, base, arguments.step)
    write_json(output / 'cost_scope.json', total)
    write_json(output / 'completion.json', {'status': 'CLOSED_R2_TRAINING_FIGURES_AND_COST_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'total_updates': arguments.step, 'milestone_sha256': sha256(milestone_path),
        'review_sha256': sha256(review_path), 'source_hashes': sources, 'GPU_used': False, 'new_inference': False,
        'monitor_used_for_selection': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps({'output': str(output), **total, 'per_arm': costs}, indent=2))


if __name__ == '__main__':
    main()
