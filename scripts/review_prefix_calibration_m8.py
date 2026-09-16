#!/usr/bin/env python3
"""Review frozen calibration endpoints and development fixed-m8 comparisons."""

import argparse
import csv
from datetime import datetime, timezone
import json
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

from var_comm.study import artifact_hashes, create_output, paired_interval, sha256, snapshot, verify_snapshot, write_csv, write_json


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def verify_inputs(directory, filenames, records):
    receipt_path = directory / 'completion.json'
    records[str(receipt_path.relative_to(ROOT))] = sha256(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    verify_snapshot(receipt['source_hashes'])
    records.update(receipt['source_hashes'])
    for filename in filenames:
        path = directory / filename
        actual = sha256(path)
        if actual != receipt['output_hashes'][filename]:
            raise RuntimeError(f'frozen artifact mismatch: {path}')
        records[str(path.relative_to(ROOT))] = actual
    return receipt


def calibration_review(training, config):
    summaries, changes = [], []
    populations = set()
    metrics = ('mse', 'lpips', 'objective', 'token_accuracy')
    snrs = config['selection']['snrs_db']
    for variant in config['variants']:
        epochs = {}
        for epoch in (1, 2):
            rows = read_csv(training / variant / f'calibration_epoch_{epoch:02d}.csv')
            images = sorted({row['image_id'] for row in rows})
            if len(images) != config['training']['calibration_images']:
                raise RuntimeError('unexpected calibration population')
            if populations and populations != set(images):
                raise RuntimeError('calibration population changed between runs')
            populations = set(images)
            indexed = {(row['image_id'], float(row['snr_db'])): row for row in rows}
            expected = {(image, snr) for image in images for snr in snrs}
            if len(indexed) != len(rows) or set(indexed) != expected:
                raise RuntimeError('incomplete or duplicated calibration grid')
            for row in rows:
                np.testing.assert_allclose(float(row['objective']), float(row['mse']) + 0.01 * float(row['lpips']), rtol=0, atol=1e-8)
            epochs[epoch] = indexed
            for group, subset in [('all_5_snr', snrs)] + [(str(snr), [snr]) for snr in snrs]:
                selected = [indexed[image, snr] for image in images for snr in subset]
                summaries.append({'variant': variant, 'epoch': epoch, 'snr_db': group,
                                  'source_images': len(images), 'observations': len(selected),
                                  **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in metrics},
                                  'header_failures': sum(1 - int(row['header_usable']) for row in selected),
                                  'header_false_acceptances': sum(int(row['header_false_acceptance']) for row in selected)})
        selection = json.loads((training / variant / 'selected.json').read_text())
        objectives = {epoch: float(np.mean([float(row['objective']) for row in indexed.values()])) for epoch, indexed in epochs.items()}
        if selection['epoch'] != min(objectives, key=objectives.get):
            raise RuntimeError('frozen checkpoint selection does not match calibration objective')
        np.testing.assert_allclose(selection['objective'], objectives[selection['epoch']], rtol=0, atol=1e-12)
        for group, subset in [('all_5_snr', snrs)] + [(str(snr), [snr]) for snr in snrs]:
            for metric in metrics:
                values = {epoch: np.array([np.mean([float(indexed[image, snr][metric]) for snr in subset]) for image in images])
                          for epoch, indexed in epochs.items()}
                interval = paired_interval(values[2] - values[1], config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
                changes.append({'variant': variant, 'snr_db': group, 'metric': metric, 'source_images': len(images),
                                'epoch_1': float(values[1].mean()), 'epoch_2': float(values[2].mean()),
                                'epoch_2_minus_1': interval['gain'], 'relative_change_percent': interval['gain'] / float(values[1].mean()) * 100,
                                'ci_low': interval['ci_low'], 'ci_high': interval['ci_high']})
    return summaries, changes, populations


def development_review(evaluation, config, calibration_images):
    arms = ('prefix_parallel', 'prefix_next_scale', 'digital_m8')
    metrics = ('psnr_db', 'lpips_alex', 'dino_cosine')
    rows = [row for row in read_csv(evaluation / 'per_frame.csv') if row['arm'] in arms]
    indexed = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    images = sorted({int(row['image_index']) for row in rows})
    identifiers = {row['image_id'] for row in rows}
    if len(images) != 100 or len(identifiers) != 100 or identifiers & calibration_images:
        raise RuntimeError('unexpected or overlapping development population')
    snrs, seeds = config['evaluation']['snrs_db'], config['evaluation']['noise_seeds']
    expected = {(image, snr, seed, arm) for image in images for snr in snrs for seed in seeds for arm in arms}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise RuntimeError('incomplete or duplicated development grid')
    for image in images:
        for snr in snrs:
            for seed in seeds:
                paired = [indexed[image, snr, seed, arm] for arm in arms]
                if len({row['noise_sha256'] for row in paired}) != 1:
                    raise RuntimeError('development noise pairing mismatch')
                for row in paired:
                    for field, expected_uses in [('total_complex_uses', 3060), ('header_uses', 68), ('data_uses', 2992)]:
                        if int(row[field]) != expected_uses:
                            raise RuntimeError(f'unexpected communication budget: {field}')
    summaries = [row for row in read_csv(evaluation / 'summary.csv') if row['arm'] in arms]
    for row in summaries:
        selected = [indexed[image, float(row['snr_db']), seed, row['arm']] for image in images for seed in seeds]
        for metric in metrics:
            np.testing.assert_allclose(float(row[metric]), np.mean([float(item[metric]) for item in selected]), rtol=0, atol=1e-12)
    contrasts = [row for row in read_csv(evaluation / 'paired_quality.csv') if row['method'] in arms and row['control'] in arms]
    maximum_error = 0.0
    for row in contrasts:
        subset = [float(value) for value in row['snr_db'].split('+')]
        differences = [np.mean([float(indexed[image, snr, seed, row['method']][row['metric']]) -
                                float(indexed[image, snr, seed, row['control']][row['metric']]) for snr in subset for seed in seeds]) for image in images]
        interval = paired_interval(differences, config['evaluation']['bootstrap_seed'], config['evaluation']['bootstrap_resamples'])
        for field, result_field in [('delta', 'gain'), ('ci_low', 'ci_low'), ('ci_high', 'ci_high')]:
            maximum_error = max(maximum_error, abs(float(row[field]) - interval[result_field]))
            np.testing.assert_allclose(float(row[field]), interval[result_field], rtol=0, atol=1e-12)
    return summaries, contrasts, {'development_rows_checked': len(rows), 'paired_rows_recomputed': len(contrasts), 'maximum_paired_error': maximum_error}


def draw_plots(output, calibration, changes, development, contrasts, config):
    variants = [('parallel', '#377eb8'), ('next_scale', '#e41a1c')]
    metrics = [('mse', 'MSE'), ('lpips', 'LPIPS'), ('objective', 'MSE + 0.01 LPIPS')]
    figure, axes = pyplot.subplots(2, 3, figsize=(13, 7))
    snrs = config['selection']['snrs_db']
    for column, (metric, title) in enumerate(metrics):
        for offset, (variant, color) in enumerate(variants):
            endpoints = sorted([row for row in calibration if row['variant'] == variant and row['snr_db'] == 'all_5_snr'], key=lambda row: row['epoch'])
            axes[0, column].plot([row['epoch'] for row in endpoints], [row[metric] for row in endpoints], 'o--', color=color, label=variant)
            selected = [next(row for row in changes if row['variant'] == variant and row['metric'] == metric and row['snr_db'] == str(snr)) for snr in snrs]
            values = np.array([row['epoch_2_minus_1'] for row in selected])
            errors = np.array([[row['epoch_2_minus_1'] - row['ci_low'] for row in selected], [row['ci_high'] - row['epoch_2_minus_1'] for row in selected]])
            axes[1, column].bar(np.arange(len(snrs)) + (offset - 0.5) * 0.36, values, width=0.36, yerr=errors, capsize=2, color=color, label=variant)
        axes[0, column].set(title=title + ' (five-SNR mean)', xlabel='Calibration checkpoint', xticks=[1, 2])
        axes[1, column].set(title=title + ': epoch 2 minus epoch 1', xlabel='Calibration SNR (dB)', xticks=np.arange(len(snrs)), xticklabels=[f'{snr:g}' for snr in snrs])
        axes[1, column].axhline(0, color='black', linewidth=0.8)
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle('Calibration: 1000 separate images, fixed noise, own-history only\nOnly TWO checkpoints; lower panels: paired pointwise 95% bootstrap CIs (no multiplicity correction)')
    figure.tight_layout()
    figure.savefig(output / 'calibration_checkpoints.png', dpi=170)
    pyplot.close(figure)

    figure, axes = pyplot.subplots(2, 3, figsize=(13, 7))
    for column, (metric, title) in enumerate([('psnr_db', 'PSNR (higher better)'), ('lpips_alex', 'LPIPS (lower better)'), ('dino_cosine', 'DINO (higher better)')]):
        for arm, color in [('prefix_parallel', '#377eb8'), ('prefix_next_scale', '#e41a1c'), ('digital_m8', '#222222')]:
            selected = sorted([row for row in development if row['arm'] == arm], key=lambda row: float(row['snr_db']))
            axes[0, column].plot([float(row['snr_db']) for row in selected], [float(row[metric]) for row in selected], 'o-', color=color, label=arm)
        for variant, color in variants:
            selected = sorted([row for row in contrasts if row['method'] == 'prefix_' + variant and row['control'] == 'digital_m8' and row['metric'] == metric and '+' not in row['snr_db']], key=lambda row: float(row['snr_db']))
            values = np.array([float(row['delta']) for row in selected])
            errors = np.array([[float(row['delta']) - float(row['ci_low']) for row in selected], [float(row['ci_high']) - float(row['delta']) for row in selected]])
            axes[1, column].errorbar([float(row['snr_db']) for row in selected], values, yerr=errors, fmt='o-', color=color, capsize=3, label=variant)
        axes[0, column].set_title(title)
        axes[1, column].set_title('Learned minus fixed digital m8')
        axes[1, column].axhline(0, color='black', linewidth=0.8)
        for axis in axes[:, column]:
            axis.set(xlabel='Physical SNR (dB)', xticks=config['evaluation']['snrs_db'])
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle('Development: 100 images x 3 original noises; all arms fixed m8 / 3060 complex uses\nEpoch-2 frozen selections; average noises within image, then source-image paired 95% bootstrap')
    figure.tight_layout()
    figure.savefig(output / 'development_fixed_m8.png', dpi=170)
    pyplot.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/VAR-PREFIX-CALIBRATION-M8-REVIEW-001')
    arguments = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    training, evaluation = ROOT / config['outputs']['training'], ROOT / config['outputs']['evaluation']
    input_hashes = {}
    verify_inputs(training, [f'{variant}/{filename}' for variant in config['variants'] for filename in ['selected.json', 'training.csv', 'calibration_epoch_01.csv', 'calibration_epoch_02.csv']], input_hashes)
    evaluation_receipt = verify_inputs(evaluation, ['summary.csv', 'per_frame.csv', 'paired_quality.csv'], input_hashes)
    if evaluation_receipt['training_completion_sha256'] != sha256(training / 'completion.json'):
        raise RuntimeError('evaluation is not bound to these training results')
    calibration, changes, populations = calibration_review(training, config)
    development, contrasts, checks = development_review(evaluation, config, populations)
    windows = []
    for variant in config['variants']:
        history = read_csv(training / variant / 'training.csv')
        if [int(row['step']) for row in history] != list(range(1, 10001)):
            raise RuntimeError('unexpected training updates')
        if any(float(row['teacher_samples']) != 0 for row in history[4000:]):
            raise RuntimeError('teacher samples found in final 6000 updates')
        for start in range(0, len(history), 1000):
            window = history[start:start + 1000]
            windows.append({'variant': variant, 'first_update': start + 1, 'last_update': start + len(window),
                            **{metric: float(np.mean([float(row[metric]) for row in window])) for metric in ['mse', 'lpips', 'token_ce', 'token_accuracy', 'teacher_probability', 'learning_rate']}})
    output = create_output(arguments.output_dir)
    sources = snapshot(output, [Path(__file__), ROOT / 'src/var_comm/study.py'])
    for filename, rows in [('calibration_summary.csv', calibration), ('calibration_epoch_delta.csv', changes), ('training_windows_descriptive.csv', windows), ('development_m8_summary.csv', development), ('development_m8_paired.csv', contrasts)]:
        write_csv(output / filename, rows)
    draw_plots(output, calibration, changes, development, contrasts, config)
    verify_snapshot(input_hashes)
    verify_snapshot(sources)
    write_json(output / 'completion.json', {'status': 'EXISTING_CALIBRATION_FIXED_M8_REVIEW_COMPLETE',
               'completed_utc': datetime.now(timezone.utc).isoformat(), 'training_or_inference': False,
               'checkpoint_selection_changed': False, 'input_hashes': input_hashes, 'source_hashes': sources,
               'calibration_bootstrap': {'source_images': 1000, 'noise_realisations_per_snr': 1, 'aggregate': 'average all five SNRs within image before pairing', 'seed': config['evaluation']['bootstrap_seed'], 'resamples': config['evaluation']['bootstrap_resamples'], 'interpretation': 'post-hoc diagnostic pointwise intervals; no multiple-comparison correction; not a new selection criterion'},
               'checks': checks, 'output_hashes': artifact_hashes(output)})
    print(json.dumps({'output': str(output), **checks}, indent=2))


if __name__ == '__main__':
    main()
