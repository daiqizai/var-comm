"""Plot only audited training/calibration histories; no new inference or model selection."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
import torch

from innovation_comm.common import output_path, settings
from innovation_comm.evaluation import audited_milestone
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, write_json


LABELS = {'single_pass': 'Single pass', 'multiscale_no_history': 'Multiscale / no history',
    'multiscale_state_history': 'Multiscale / state history', 'multiscale_prediction_features': 'Multiscale / prediction',
    'multiscale_innovation': 'Multiscale / innovation', 'full_grid_innovation': 'Full-grid / innovation'}


def smooth(values, width=100):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite observations may not be hidden in a moving average')
    sums = np.r_[0., np.cumsum(values)]
    positions = np.arange(len(values))
    starts = np.maximum(0, positions + 1 - width)
    return (sums[positions + 1] - sums[starts]) / (positions + 1 - starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, base, parent = settings()
    if not arguments.execute:
        print('PLAN ONLY: immutable milestone training curves and complete-calibration plots, no GPU')
        return
    milestone, milestone_path, review_path = audited_milestone(config, arguments.step)
    state = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    output = output_path(config, 'analysis') / f'calibration_plots_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    for scope in ('full', 'monitor'):
        rows = [row for row in state['summaries'] if row['scope'] == scope]
        if not rows:
            continue
        figure, axes = pyplot.subplots(2, 5, figsize=(18, 7), sharex=True)
        for column, snr in enumerate(base['channel']['snrs_db']):
            for position, variant in enumerate(config['variants']):
                values = sorted((row for row in rows if row['variant'] == variant and row['snr_db'] == snr), key=lambda row: row['step'])
                for axis, metric in zip(axes[:, column], ('lpips', 'psnr_db')):
                    axis.plot([row['step'] for row in values], [row[metric] for row in values],
                        marker='o', color=f'C{position}', label=LABELS[variant], linewidth=1.3)
                    axis.grid(alpha=.25)
                    axis.set_ylabel(metric)
            axes[0, column].set_title(f'{snr:g} dB')
            axes[1, column].set_xlabel('Additional receiver updates')
        figure.suptitle('Full calibration (1000 images): model-selection population' if scope == 'full'
                       else 'Fixed monitoring subset only: never used for model selection')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=3)
        figure.tight_layout(rect=(0, .1, 1, .94))
        figure.savefig(output / f'{scope}_quality_by_snr.png', dpi=160)
        figure.savefig(output / f'{scope}_quality_by_snr.pdf')
        pyplot.close(figure)
    figure, axes = pyplot.subplots(2, 4, figsize=(16, 7))
    metrics = ('loss', 'mse', 'lpips', 'bits', 'state', 'bit_error_rate', 'gradient_norm', 'step_seconds')
    for axis, metric in zip(axes.flat, metrics):
        for position, variant in enumerate(config['variants']):
            rows = state['histories'][variant]
            if len(rows) != arguments.step:
                raise RuntimeError('plot histories do not cover the audited common update budget')
            axis.plot([row['step'] for row in rows], smooth([row[metric] for row in rows]),
                      label=LABELS[variant], color=f'C{position}', linewidth=1.)
        axis.set_title(metric)
        axis.set_xlabel('Additional receiver updates')
        axis.grid(alpha=.25)
    figure.suptitle('Paired training histories: trailing 100-update mean, not calibration performance')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3)
    figure.tight_layout(rect=(0, .1, 1, .94))
    figure.savefig(output / 'training_components.png', dpi=160)
    figure.savefig(output / 'training_components.pdf')
    pyplot.close(figure)
    write_json(output / 'completion.json', {'status': 'AUDITED_CALIBRATION_PLOTS_COMPLETE', 'completed_local': now(),
        'milestone_sha256': sha256(milestone_path), 'calibration_review_sha256': sha256(review_path),
        'source_hashes': sources, 'development_accessed': False, 'GPU_used': False,
        'monitoring_subset_used_for_selection': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps({'output': str(output), 'plotted_receiver_updates': arguments.step}))


if __name__ == '__main__':
    main()
