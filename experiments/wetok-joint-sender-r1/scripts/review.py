"""Audit actual Joint updates, paired control opportunities and complete-calibration selection."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from innovation_comm.evaluation import audited_milestone as reviewed_control
from joint_sender.common import complete_control, initial_system, load_parent, output_path, settings
from joint_sender.runtime import tensor_sha256
from wetok_comm.common import PROJECT, artifact_hashes, configure_torch, now, output_path as base_output, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, paired_batches


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, reference, base, parent_milestone = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step <= 0:
        raise ValueError('Joint review requires a complete-calibration milestone')
    if not arguments.execute:
        print('PLAN ONLY: actual Joint E/R/Adam history, matched frozen controls, full-calibration choices; CPU only')
        return
    configure_torch()
    control, control_path, control_review = complete_control(config, reference)
    matched_control, matched_control_path, matched_control_review = reviewed_control(reference, arguments.step)
    training = output_path(config, 'training')
    metadata = json.loads((training / 'metadata.json').read_text())
    for path, expected_hash in metadata['control_input_hashes'].items():
        if sha256(path) != expected_hash:
            raise RuntimeError('sealed frozen-control history or calibration input changed')
    milestone_path = training / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if (milestone['status'] != 'JOINT_SENDER_MILESTONE_COMPLETE' or milestone['joint_updates_per_arm'] != arguments.step or
        milestone['control_milestone_sha256'] != sha256(control_path) or milestone['control_review_sha256'] != sha256(control_review)):
        raise RuntimeError('Joint milestone or reference control qualification differs')
    verify_sources(milestone['source_hashes'])
    if sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']:
        raise RuntimeError('actual Joint endpoint/optimizer checkpoint changed')
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    parent = load_parent(reference, base, 'cpu')
    encoder_start = module_sha256(parent.encoder)
    initial = json.loads((training / 'initialization.json').read_text())
    if saved['frozen'] != milestone['frozen'] or initial['frozen'] != milestone['frozen'] or module_sha256(parent) != milestone['frozen']['parent']:
        raise RuntimeError('frozen parent/visual identity differs across the Joint training record')
    frozen_histories = {}
    for name in config['variants']:
        system = initial_system(parent, name, reference, 'cpu')
        if module_sha256(system) != initial['model_hashes'][name]:
            raise RuntimeError('Joint initial model no longer matches the common parent')
        system.load_state_dict(saved['models'][name], strict=True)
        if module_sha256(system.encoder) == encoder_start:
            raise RuntimeError('the supposedly joint-trained sender never changed')
        if len(saved['histories'][name]) != arguments.step or any(int(state['step']) != arguments.step for state in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('Joint optimizer budgets differ')
        if len(saved['optimizers'][name]['param_groups'][0]['params']) != len([value for value in system.parameters() if value.requires_grad]):
            raise RuntimeError('the optimizer does not cover the intended E/R parameters')
        frozen_histories[name] = read_rows(PROJECT / 'outputs' / config['reference_training'] / name / 'training.csv')
        if len(frozen_histories[name]) < arguments.step:
            raise RuntimeError('frozen control lacks the corresponding training opportunity')
        del system
    for index, batch in enumerate(paired_batches(base, 20000, 7000, 7000 + arguments.step)):
        noise_hash = tensor_sha256(batch['noise'])
        for name in config['variants']:
            row = saved['histories'][name][index]
            previous = frozen_histories[name][index]
            if (row['step'] != index + 1 or row['global_data_step'] != batch['step'] + 1 or
                row['batch_sha256'] != batch['fingerprint'] or previous['batch_sha256'] != batch['fingerprint'] or
                row['paired_standard_noise_sha256'] != noise_hash):
                raise RuntimeError('Joint/control data, augmentation, SNR or actual noise sequence differs')
            if row['power_max_error'] > 1e-5 or not all(np.isfinite(row[key]) for key in
                ('loss', 'encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm', 'gradient_norm')):
                raise RuntimeError('Joint physical/gradient outcome is nonfinite or invalid')
    identifiers = json.loads((base_output(base, 'cache') / 'calibration_ids.json').read_text())
    expected = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    summary, curves, selected_rows, paired = [], [], {}, []
    for name in config['variants']:
        pool = []
        for step in config['full_calibration_steps']:
            if step > arguments.step:
                continue
            rows = read_rows(training / name / f'full_{step:07d}.csv')
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected:
                raise RuntimeError('Joint complete calibration grid differs')
            means = {key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'lpips', 'bit_error_rate')}
            if not all(np.isfinite(value) for value in means.values()):
                raise RuntimeError('nonfinite Joint calibration cannot be excluded from selection')
            pool.append((means['lpips'], step, rows))
            curves.append({'variant': name, 'step': step, **means})
        choice = min(pool, key=lambda item: (item[0], item[1]))
        recorded = milestone['selected'][name]
        if recorded['step'] != choice[1] or abs(recorded['lpips'] - choice[0]) > 1e-12 or sha256(training / recorded['checkpoint']) != recorded['checkpoint_sha256']:
            raise RuntimeError('Joint selected checkpoint differs from full-calibration LPIPS rule')
        selected_rows[name] = {(row['image_id'], float(row['snr_db'])): row for row in choice[2]}
        control_choice = matched_control['selected'][name]
        control_rows = read_rows(PROJECT / 'outputs' / config['reference_training'] / name / f'full_{control_choice["step"]:07d}.csv')
        if len(control_rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in control_rows} != expected:
            raise RuntimeError('matched-opportunity frozen control calibration is incomplete')
        control_lookup = {(row['image_id'], float(row['snr_db'])): row for row in control_rows}
        for snrs in (base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]):
            subset = [row for row in choice[2] if float(row['snr_db']) in snrs]
            summary.append({'variant': name, 'snrs_db': '+'.join(map(str, snrs)), 'selected_step': choice[1], 'source_images': 1000,
                **{key: float(np.mean([float(row[key]) for row in subset])) for key in ('psnr_db', 'ssim', 'lpips', 'bit_error_rate', 'state_error')}})
            for metric in ('psnr_db', 'ssim', 'lpips', 'bit_error_rate', 'state_error'):
                differences = [np.mean([float(selected_rows[name][identifier, snr][metric]) -
                    float(control_lookup[identifier, snr][metric]) for snr in snrs]) for identifier in identifiers]
                paired.append({'variant': name, 'available_updates_each': arguments.step, 'joint_selected_step': choice[1],
                    'frozen_selected_step': control_choice['step'], 'snrs_db': '+'.join(map(str, snrs)), 'metric': metric,
                    'joint_minus_frozen': float(np.mean(differences)), 'role': 'calibration_selection_not_independent_confirmation'})
    output = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'curves.csv', curves)
    write_csv(output / 'paired_same_opportunity_calibration.csv', paired)
    write_json(output / 'completion.json', {'status': 'JOINT_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'actual_E_R_Adam_data_noise_power_selection_audit': 'PASS', 'joint_updates': arguments.step,
        'milestone_sha256': sha256(milestone_path), 'control_milestone_sha256': sha256(control_path), 'source_hashes': sources,
        'matched_opportunity_control_milestone_sha256': sha256(matched_control_path),
        'matched_opportunity_control_review_sha256': sha256(matched_control_review),
        'selected_steps': {name: value['step'] for name, value in milestone['selected'].items()},
        'output_hashes': artifact_hashes(output), 'new_development_access': False, 'research_goal_complete': False})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
