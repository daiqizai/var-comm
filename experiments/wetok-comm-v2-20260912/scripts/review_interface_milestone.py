"""Audit a paired interface milestone and plot calibration without accessing development."""

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
from wetok_comm.common import artifact_hashes, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.interface_study import interface_definitions, interface_output, load_interface_study
from wetok_comm.training import paired_batches


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    study, base, parent = load_interface_study()
    if not arguments.execute:
        print('PLAN ONLY: CPU paired checkpoint/Adam/data audit and calibration review, no development or model selection changes')
        return
    torch.set_num_threads(2)
    training = interface_output(study, 'training')
    receipt_path = training / 'milestones' / f'additional_{arguments.milestone:07d}.json'
    receipt = json.loads(receipt_path.read_text())
    verify_sources(receipt['source_hashes'])
    if receipt['status'] != 'INTERFACE_MILESTONE_COMPLETE' or receipt['additional_updates_per_arm'] != arguments.milestone:
        raise RuntimeError('requested paired interface milestone is incomplete')
    checkpoint = Path(receipt['checkpoint'])
    if sha256(checkpoint) != receipt['optimizer_checkpoint_sha256']:
        raise RuntimeError('immutable paired checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    definitions = interface_definitions(study)
    for name in definitions:
        if len(saved['histories'][name]) != arguments.milestone:
            raise RuntimeError('unequal retained arm histories')
        if any(int(value['step']) != arguments.milestone for value in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('fresh Adam counters differ from retained updates')
    for local, batch in enumerate(paired_batches(base, 20000, study['parent_step'], study['parent_step'] + arguments.milestone)):
        for name in definitions:
            row = saved['histories'][name][local]
            if row['batch_sha256'] != batch['fingerprint'] or row['global_data_step'] != batch['step'] + 1:
                raise RuntimeError('paired source/augmentation/SNR/noise history differs')
            if not np.isfinite(row['loss']) or row['power_max_error'] > 1e-5:
                raise RuntimeError('nonfinite training loss or energy violation in saved history')
    calibration_ids = json.loads((interface_output({'outputs': {'cache': base['outputs']['cache']}}, 'cache') / 'calibration_ids.json').read_text())
    expected_grid = {(identifier, float(snr)) for identifier in calibration_ids for snr in base['channel']['snrs_db']}
    latest, selected_rows, summary = {}, {}, []
    for name, definition in definitions.items():
        candidates = []
        for step in saved['calibrated']:
            path = training / name / f'full_{step:07d}.csv'
            if not path.exists():
                continue
            rows = read_csv(path)
            if len(rows) != len(expected_grid) or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected_grid:
                raise RuntimeError('full calibration source/SNR grid incomplete or duplicated')
            metric = float(np.mean([float(row['lpips']) for row in rows]))
            candidates.append((metric, step, rows))
        best = min(candidates, key=lambda value: (value[0], value[1]))
        selected = receipt['selected'][name]
        if best[1] != selected['step'] or abs(best[0] - selected['lpips']) > 1e-10:
            raise RuntimeError('model selection differs from registered full-calibration LPIPS')
        if sha256(training / selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('selected model bytes changed')
        selected_rows[name] = best[2]
        latest[name] = read_csv(training / name / f'full_{arguments.milestone:07d}.csv')
        for kind, rows in (('selected_full', best[2]), ('latest_full', latest[name])):
            for snrs in (base['channel']['snrs_db'], *[[value] for value in base['channel']['snrs_db']]):
                subset = [row for row in rows if float(row['snr_db']) in snrs]
                summary.append({'arm': name, **definition, 'kind': kind, 'selected_additional_step': best[1],
                    'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                    **{key: float(np.mean([float(row[key]) for row in subset])) for key in
                        ('psnr_db', 'lpips', 'ssim', 'bits_BCE', 'bit_error_rate', 'state_error', 'feature_mse', 'feature_abs_mean', 'logit_abs_mean')}})
    pairs = [(f'hard_bounded__{variant}', f'hard_identity__{variant}') for variant in study['variants']]
    pairs += [(f'continuous_mean__{variant}', f'hard_bounded__{variant}') for variant in study['variants']]
    pairs += [(f'{interface}__multiscale_conditioned', f'{interface}__multiscale_no_history') for interface in study['interfaces']]
    differences = []
    for method, control in pairs:
        method_rows = {(row['image_id'], row['snr_db']): row for row in selected_rows[method]}
        control_rows = {(row['image_id'], row['snr_db']): row for row in selected_rows[control]}
        if set(method_rows) != set(control_rows):
            raise RuntimeError('selected calibration source pairing differs')
        for metric in ('lpips', 'psnr_db', 'ssim', 'bit_error_rate'):
            delta = np.array([float(method_rows[key][metric]) - float(control_rows[key][metric]) for key in method_rows])
            differences.append({'method': method, 'control': control, 'metric': metric, 'delta': float(delta.mean()),
                'population': 'calibration_used_for_selection_not_independent_confirmation'})
    output = interface_output(study, 'analysis') / f'calibration_{arguments.milestone:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/interface_study.yaml'])
    write_csv(output / 'calibration_summary.csv', summary)
    write_csv(output / 'calibration_paired_deltas.csv', differences)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, variant in zip(axes, study['variants']):
        for interface in study['interfaces']:
            name = f'{interface}__{variant}'
            rows = [row for row in saved['summaries'] if row['arm'] == name and row['scope'] == 'full']
            steps = sorted({row['additional_step'] for row in rows})
            values = [np.mean([row['lpips'] for row in rows if row['additional_step'] == step]) for step in steps]
            axis.plot(steps, values, marker='o', label=interface)
        axis.set(title=variant, xlabel='Additional paired updates', ylabel='Full calibration LPIPS')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.savefig(output / 'full_calibration_lpips.png', dpi=160)
    figure.savefig(output / 'full_calibration_lpips.pdf')
    plt.close(figure)
    audit = {'status': 'INTERFACE_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE', 'completed_local': now(),
        'additional_updates_per_arm': arguments.milestone, 'arms': len(definitions), 'selected_steps':
            {name: receipt['selected'][name]['step'] for name in definitions},
        'paired_data_Adam_energy_and_selection_audit': 'PASS', 'new_development_access': False,
        'independent_holdout_access': False, 'source_hashes': sources, 'milestone_sha256': sha256(receipt_path),
        'output_hashes': artifact_hashes(output), 'research_goal_complete': False,
        'next_action': 'Review full calibration and gradient/feature trajectories; decide matched continuation or a registered revision before final development evaluation.'}
    write_json(output / 'completion.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
