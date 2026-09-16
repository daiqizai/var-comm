"""Analyze only retained early-stop history and the same fixed calibration sources."""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from wetok_comm.bit_support import arm_definitions, load_repair, repair_output
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, write_json


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    torch.set_num_threads(2)
    repair, base, parent, receipt = load_repair()
    training = repair_output(repair, 'training')
    termination = json.loads((training / 'exploratory_early_stop/termination.json').read_text())
    checkpoint = Path(termination['checkpoint'])
    if termination['status'] != 'CANDIDATE_STOPPED_EARLY_NOT_COMPLETE' or sha256(checkpoint) != termination['checkpoint_sha256']:
        raise RuntimeError('stopped paired checkpoint has not been preserved')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['completed_additional_updates'] != 1000:
        raise RuntimeError('this report requires the actually retained 1000-update endpoint')
    rows, trajectories = [], []
    for name, definition in arm_definitions(repair).items():
        monitored = read_rows(training / name / 'monitor_0001000.csv')
        parent_rows = {(row['image_id'], row['snr_db']): row for row in read_rows(parent / definition['variant'] / 'full_0005000.csv')}
        best_rows = {(row['image_id'], row['snr_db']): row for row in read_rows(parent / definition['variant'] / 'full_0002000.csv')}
        keys = [(row['image_id'], row['snr_db']) for row in monitored]
        if len(keys) != 500 or len(set(keys)) != 500 or len({key[0] for key in keys}) != 100:
            raise RuntimeError('fixed five-SNR calibration monitor is incomplete')
        record = {'arm': name, **{key: value for key, value in definition.items() if key != 'weights'},
                  'source_images': 100, 'snrs_per_source': 5, 'retained_additional_updates': 1000}
        for metric in ('lpips', 'psnr_db', 'bit_error_rate', 'bits_BCE'):
            current = np.array([float(row[metric]) for row in monitored])
            baseline = np.array([float(parent_rows[key][metric]) for key in keys])
            historical = np.array([float(best_rows[key][metric]) for key in keys])
            record.update({f'{metric}_parent5000_same_sources': float(baseline.mean()),
                           f'{metric}_historical2000_same_sources': float(historical.mean()),
                           f'{metric}_additional1000': float(current.mean()),
                           f'{metric}_delta_from_parent': float((current - baseline).mean())})
        rows.append(record)
        history = saved['histories'][name]
        if len(history) != 1000:
            raise RuntimeError('retained training histories differ')
        for start in range(0, 1000, 100):
            window = history[start:start + 100]
            trajectories.append({'arm': name, 'window_last_update': start + 100,
                **{key: float(np.mean([item[key] for item in window])) for key in ('loss', 'bits', 'lpips', 'mse', 'gradient_norm', 'bit_error_rate')}})
    output = repair_output(repair, 'analysis') / 'exploratory_early_stop'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'docs/bit_support_early_stop.md'])
    write_csv(output / 'same_source_calibration_deterioration.csv', rows)
    write_csv(output / 'retained_training_windows.csv', trajectories)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for column, variant in enumerate(repair['variants']):
        for recipe in repair['recipes']:
            selected = [row for row in trajectories if row['arm'] == f'{recipe}__{variant}']
            for axis, metric in zip(axes[:, column], ('bits', 'gradient_norm')):
                axis.plot([row['window_last_update'] for row in selected], [row[metric] for row in selected], marker='o', label=recipe)
                axis.set(xlabel='Retained additional updates', ylabel=f'{metric}: 100-update mean', yscale='log', title=variant)
                axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(output / 'retained_training_deterioration.png', dpi=150)
    figure.savefig(output / 'retained_training_deterioration.pdf')
    plt.close(figure)
    metadata = json.loads((training / 'metadata.json').read_text())
    wall_seconds = (datetime.fromisoformat(termination['stopped_local']) - datetime.fromisoformat(metadata['created_local'])).total_seconds()
    write_json(output / 'summary.json', {'status': 'EXPLORATORY_STOP_DIAGNOSTIC_NOT_COMPLETED_EFFICACY_STUDY',
        'completed_local': now(), 'retained_additional_updates': 1000, 'trainer_completed_before_interrupt': 1113,
        'planned_additional_updates': 5000, 'allocated_wall_hours_including_discarded_updates': wall_seconds / 3600,
        'calibration_monitor_only': True, 'development_used_for_stop': False,
        'source_hashes': sources, 'checkpoint_sha256': sha256(checkpoint), 'rows': rows,
        'output_hashes': artifact_hashes(output),
        'limitations': 'Outcome-observed exploratory stop; no claim about all bit weights or fully converged training; no independent efficacy interval.'})
    for row in rows:
        print(row['arm'], 'LPIPS historical/parent/stop', *(round(row[field], 6) for field in
            ('lpips_historical2000_same_sources', 'lpips_parent5000_same_sources', 'lpips_additional1000')))
    print('allocated_wall_GPU_hours', wall_seconds / 3600)


if __name__ == '__main__':
    main()
