"""Plot sealed Joint/R-only milestones without inference or checkpoint reselection."""

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / 'experiments/wetok-joint-sender-r1'
INNOVATION = PROJECT / 'experiments/wetok-innovation-r1'
BASE = PROJECT / 'experiments/wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'scripts'), str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
import torch

from finish_registered_trial import validate_receipt
from innovation_comm.evaluation import audited_milestone
from joint_sender.common import output_path, settings
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json


LABELS = {'single_pass': 'Single pass', 'multiscale_no_history': 'Multiscale / no history',
          'multiscale_state_history': 'Multiscale / state history'}
STYLES = {'Joint': '-', 'R-only': '--'}


def smooth(values, width=100):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('cannot hide nonfinite training observations in smoothing')
    sums = np.r_[0., np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions + 1 - width)
    return (sums[positions + 1] - sums[starts]) / (positions + 1 - starts)


def save_figure(figure, output, stem):
    figure.savefig(output / f'{stem}.png', dpi=160)
    figure.savefig(output / f'{stem}.pdf')
    pyplot.close(figure)


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: sealed matching calibration/monitor/training plots; no GPU, inference, or new selection')
        return
    config, reference, base, parent = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step == 0:
        raise ValueError('plots require a completed audited Joint milestone')
    control, control_path, control_review = audited_milestone(reference, arguments.step)
    joint_path = output_path(config, 'training') / 'milestones' / f'step_{arguments.step:07d}.json'
    joint = json.loads(joint_path.read_text())
    if joint['status'] != 'JOINT_SENDER_MILESTONE_COMPLETE' or joint['joint_updates_per_arm'] != arguments.step:
        raise RuntimeError('Joint opportunity is incomplete')
    verify_sources(joint['source_hashes'])
    if sha256(joint['checkpoint']) != joint['checkpoint_sha256']:
        raise RuntimeError('Joint endpoint changed')
    joint_review = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}/completion.json'
    validate_receipt(joint_review, 'JOINT_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        {'milestone_sha256': sha256(joint_path), 'joint_updates': arguments.step,
         'actual_E_R_Adam_data_noise_power_selection_audit': 'PASS',
         'matched_opportunity_control_milestone_sha256': sha256(control_path)})
    milestones = {'Joint': joint, 'R-only': control}
    states = {policy: torch.load(record['checkpoint'], map_location='cpu', weights_only=True)
              for policy, record in milestones.items()}
    variants = list(config['variants'])
    snrs = list(base['channel']['snrs_db'])
    summaries, choices = [], []
    for policy, state in states.items():
        for variant in variants:
            if len(state['histories'][variant]) != arguments.step:
                raise RuntimeError('plotted training opportunities differ')
            selected = milestones[policy]['selected'][variant]
            choices.append({'policy': policy, 'variant': variant, 'available_updates': arguments.step,
                'selected_step': selected['step'], 'lpips': selected['lpips'], 'psnr_db': selected['psnr_db']})
        for row in state['summaries']:
            if row['variant'] not in variants:
                continue
            if row['step'] > arguments.step:
                raise RuntimeError('future calibration cannot enter an earlier milestone plot')
            summaries.append({'policy': policy, **{key: row[key] for key in
                ('variant', 'scope', 'step', 'snr_db', 'source_images', 'lpips', 'psnr_db', 'bit_error_rate', 'state_error')}})
    output = PROJECT / 'outputs' / f'WETOK-JOINT-SENDER-R1-CALIBRATION-FIGURES-step{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml', INNOVATION / 'configs/study.yaml'])
    write_rows(output / 'quality_points.csv', summaries)
    write_rows(output / 'selected_points.csv', choices)
    means = []
    figure, axes = pyplot.subplots(2, 3, figsize=(13, 7))
    for column, variant in enumerate(variants):
        for policy_position, policy in enumerate(STYLES):
            rows = [row for row in summaries if row['variant'] == variant and row['policy'] == policy and row['scope'] == 'full']
            pooled = []
            for step in sorted({row['step'] for row in rows}):
                block = [row for row in rows if row['step'] == step]
                if len(block) != len(snrs) or {row['snr_db'] for row in block} != set(snrs) or any(row['source_images'] != 1000 for row in block):
                    raise RuntimeError('full-calibration population or SNR grid is incomplete')
                point = {'policy': policy, 'variant': variant, 'step': step,
                    **{metric: float(np.mean([row[metric] for row in block])) for metric in ('lpips', 'psnr_db')}}
                pooled.append(point)
                means.append(point)
            selected = milestones[policy]['selected'][variant]
            for axis, metric in zip(axes[:, column], ('lpips', 'psnr_db')):
                axis.plot([row['step'] for row in pooled], [row[metric] for row in pooled],
                    marker='o', linestyle=STYLES[policy], color=f'C{policy_position}', label=policy)
                axis.scatter([selected['step']], [selected[metric]], marker='*', s=155,
                    facecolors='none', edgecolors=f'C{policy_position}', zorder=5)
                axis.grid(alpha=.25)
                axis.set_ylabel(metric)
            axes[0, column].set_title(LABELS[variant])
            axes[1, column].set_xlabel('Additional updates (same opportunity)')
        axes[0, column].text(.04, .96, '\n'.join(f'{policy}: selected step {milestones[policy][variant_key][variant]["step"]}'
            for policy in STYLES for variant_key in ['selected']), transform=axes[0, column].transAxes,
            va='top', fontsize=9, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    figure.suptitle('Full calibration: 1000 images x five SNRs; stars = recorded LPIPS selections\n'
                   'All initial points retained; panel y-axis ranges differ; not independent validation')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=2)
    figure.tight_layout(rect=(0, .055, 1, .90))
    save_figure(figure, output, 'full_mean_quality')
    write_rows(output / 'full_mean_points.csv', means)
    for scope in ('full', 'monitor'):
        scope_rows = [row for row in summaries if row['scope'] == scope]
        if not scope_rows:
            continue
        figure, axes = pyplot.subplots(2, 5, figsize=(18, 7), sharex=True)
        for column, snr in enumerate(snrs):
            for position, variant in enumerate(variants):
                for policy in STYLES:
                    rows = sorted((row for row in scope_rows if row['policy'] == policy and
                        row['variant'] == variant and row['snr_db'] == snr), key=lambda row: row['step'])
                    for axis, metric in zip(axes[:, column], ('lpips', 'psnr_db')):
                        axis.plot([row['step'] for row in rows], [row[metric] for row in rows],
                            color=f'C{position}', linestyle=STYLES[policy], marker='o', markersize=3,
                            label=f'{policy}: {LABELS[variant]}', linewidth=1.2)
                        axis.grid(alpha=.25)
                        axis.set_ylabel(metric)
            axes[0, column].set_title(f'{snr:g} dB')
            axes[1, column].set_xlabel('Additional updates')
        figure.suptitle('Full calibration (1000 images): model-selection population' if scope == 'full'
            else 'Fixed monitoring subset only: never used for checkpoint selection')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=3)
        figure.tight_layout(rect=(0, .1, 1, .94))
        save_figure(figure, output, scope + '_quality_by_snr')
    figure, axes = pyplot.subplots(2, 4, figsize=(17, 7))
    metrics = ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate', 'gradient_norm', 'step_seconds')
    for axis, metric in zip(axes.flat, metrics):
        for position, variant in enumerate(variants):
            for policy, state in states.items():
                rows = state['histories'][variant]
                axis.plot([row['step'] for row in rows], smooth([row[metric] for row in rows]),
                    color=f'C{position}', linestyle=STYLES[policy], linewidth=1., label=f'{policy}: {LABELS[variant]}')
        axis.set_title(metric)
        axis.set_xlabel('Additional updates')
        axis.grid(alpha=.25)
    figure.suptitle('Training only: trailing 100-update mean; time is observed, not guaranteed exclusive-GPU cost')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3)
    figure.tight_layout(rect=(0, .1, 1, .94))
    save_figure(figure, output, 'training_components')
    write_json(output / 'completion.json', {'status': 'MATCHED_JOINT_CALIBRATION_FIGURES_COMPLETE', 'completed_local': now(),
        'available_updates_each': arguments.step, 'joint_milestone_sha256': sha256(joint_path),
        'frozen_milestone_sha256': sha256(control_path), 'joint_review_sha256': sha256(joint_review),
        'frozen_review_sha256': sha256(control_review), 'source_hashes': sources, 'quality_points': len(summaries),
        'selected_points': len(choices), 'GPU_used': False, 'new_inference': False, 'reselected_checkpoints': False,
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps({'output': str(output), 'updates_each': arguments.step, 'quality_points': len(summaries)}, indent=2))


if __name__ == '__main__':
    main()
