#!/usr/bin/env python3
"""Summarize completed training, quality curves and source-message dependence."""

import csv
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
import yaml

from var_comm.study import artifact_hashes, create_output, paired_interval, sha256, snapshot, verify_artifacts, write_csv, write_json


def read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    training, evaluation = ROOT / config['outputs']['training'], ROOT / config['outputs']['evaluation']
    verify_artifacts(training, 'completion.json')
    verify_artifacts(evaluation, 'completion.json')
    output = create_output(ROOT / 'outputs/VAR-PREFIX-JSCC-SUMMARY-001')
    sources = snapshot(output, [Path(__file__)])
    rows = read(evaluation / 'per_frame.csv')
    metrics = ('psnr_db', 'lpips_alex', 'dino_cosine')
    names = ('prefix_parallel', 'prefix_next_scale', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc')
    summaries = []
    for name in names:
        selected = [row for row in rows if row['arm'] == name and float(row['snr_db']) in config['evaluation']['primary_snr_db']]
        summaries.append({'arm': name, 'source_images': 100, 'transmissions': len(selected),
                          **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in metrics}})
    write_csv(output / 'primary_summary.csv', summaries)
    ablations = read(evaluation / 'shuffled_body.csv')
    message = []
    for name in ('prefix_parallel', 'prefix_next_scale'):
        original = {int(row['image_index']): row for row in rows if row['arm'] == name and float(row['snr_db']) == 7 and int(row['seed']) == 2001}
        altered = {int(row['image_index']): row for row in ablations if row['arm'] == name}
        for metric in metrics:
            values = [float(altered[index][metric]) - float(original[index][metric]) for index in range(100)]
            interval = paired_interval(values, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
            message.append({'arm': name, 'metric': metric, 'source_images': 100, 'shuffled_minus_true_body': interval['gain'],
                            'ci_low': interval['ci_low'], 'ci_high': interval['ci_high']})
    write_csv(output / 'message_dependence.csv', message)
    figure, axes = pyplot.subplots(2, 2, figsize=(12, 8))
    for name, color in (('parallel', '#377eb8'), ('next_scale', '#e41a1c')):
        history = read(training / name / 'training.csv')
        for axis, metric in ((axes[0, 0], 'mse'), (axes[0, 1], 'lpips')):
            values = np.array([float(row[metric]) for row in history])
            axis.plot(np.arange(250, len(values) + 1), np.convolve(values, np.ones(250) / 250, mode='valid'), label=name, color=color)
            axis.axvline(4000, color='#777777', linestyle=':', label='Teacher probability reaches zero' if name == 'parallel' else None)
            axis.set_xlabel('Optimizer updates per variant')
            axis.set_title('Training ' + metric + ' (250-update mean)')
            axis.grid(alpha=0.2)
    summary = read(evaluation / 'summary.csv')
    colors = ('#377eb8', '#e41a1c', '#999999', '#4daf4a', '#984ea3')
    for axis, metric in ((axes[1, 0], 'lpips_alex'), (axes[1, 1], 'dino_cosine')):
        for name, color in zip(names, colors):
            values = sorted((float(row['snr_db']), float(row[metric])) for row in summary if row['arm'] == name)
            axis.plot([value[0] for value in values], [value[1] for value in values], marker='o', label=name, color=color)
        axis.set_title('Development ' + metric + (' (not optimized)' if metric == 'dino_cosine' else ''))
        axis.set_xlabel('SNR (dB)')
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle('Global m8 prefix JSCC: matched communication parameters, 3060 total complex uses')
    figure.tight_layout()
    figure.savefig(output / 'training_and_quality.png', dpi=180)
    pyplot.close(figure)
    write_json(output / 'completion.json', {'status': 'TRAINED_PREFIX_SUMMARY_COMPLETE', 'source_hashes': sources,
                                            'training_receipt_sha256': sha256(training / 'completion.json'),
                                            'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'), 'output_hashes': artifact_hashes(output)})
    print(summaries, flush=True)
    print(message, flush=True)


if __name__ == '__main__':
    main()
