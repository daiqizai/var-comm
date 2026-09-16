"""Audit real grid-control updates, paired data, Adam state and full-calibration selection."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from grid_controls.common import completed_joint_reference, initial_system, load_parent, output_path, settings
from joint_sender.runtime import tensor_sha256
from train_milestone import write_csv
from wetok_comm.common import PROJECT, artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, paired_batches, read_population


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, original, reference, base, parent_record = settings()
    if arguments.step not in config['full_calibration_steps'] or arguments.step <= 0:
        raise ValueError('review a registered complete milestone')
    if not arguments.execute:
        print('PLAN ONLY: real optimizer/data/energy/selection audit; CPU only, no new inference')
        return
    configure_torch()
    references = completed_joint_reference(config, original, reference)
    training = output_path(config, 'training')
    metadata = json.loads((training / 'metadata.json').read_text())
    if metadata['reference_hashes'] != references['hashes'] or sha256(metadata['paired_trace_path']) != metadata['paired_trace_sha256']:
        raise RuntimeError('training references or paired trace changed')
    path = training / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(path.read_text())
    if (milestone['status'] != 'JOINT_GRID_MILESTONE_COMPLETE' or milestone['grid_updates_per_arm'] != arguments.step or
        milestone['reference_hashes'] != references['hashes'] or sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('grid milestone is incomplete or changed')
    verify_sources(milestone['source_hashes'])
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    parent = load_parent(reference, base, 'cpu')
    initial = json.loads((training / 'initialization.json').read_text())
    if (saved['completed_grid_updates'] != arguments.step or saved['global_data_step'] != 7000 + arguments.step or
        saved['frozen'] != milestone['frozen'] or initial['frozen'] != milestone['frozen'] or
        module_sha256(parent) != milestone['frozen']['parent']):
        raise RuntimeError('grid frozen identity or global position changed')
    trace = read_rows(metadata['paired_trace_path'])
    for name in config['variants']:
        model = initial_system(parent, name, reference, 'cpu')
        if module_sha256(model) != initial['model_hashes'][name]:
            raise RuntimeError('initial grid model is not the registered original-parent model')
        encoder_start, receiver_start = module_sha256(model.encoder), module_sha256(model.receiver)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = saved['optimizers'][name]
        if (len(optimizer['param_groups'][0]['params']) != len(parameters) or len(optimizer['state']) != len(parameters) or
            any(int(state['step']) != arguments.step for state in optimizer['state'].values()) or
            len(saved['histories'][name]) != arguments.step):
            raise RuntimeError('Adam does not cover the intended equal-budget E/R parameters')
        model.load_state_dict(saved['models'][name], strict=True)
        if module_sha256(model.encoder) == encoder_start or module_sha256(model.receiver) == receiver_start:
            raise RuntimeError('a supposedly trained communication side did not change')
        del model
    for index, batch in enumerate(paired_batches(base, 20000, 7000, 7000 + arguments.step)):
        noise_hash = tensor_sha256(batch['noise'])
        expected = trace[index]
        for name in config['variants']:
            row = saved['histories'][name][index]
            if (row['step'] != index + 1 or row['global_data_step'] != batch['step'] + 1 or
                row['batch_sha256'] != batch['fingerprint'] or expected['batch_sha256'] != batch['fingerprint'] or
                row['paired_standard_noise_sha256'] != noise_hash or expected['paired_standard_noise_sha256'] != noise_hash):
                raise RuntimeError('grid data/augmentation/SNR/noise differ from the same-opportunity original trace')
            if row['power_max_error'] > 1e-5 or not all(np.isfinite(row[key]) for key in
                ('loss', 'gradient_norm', 'encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
                raise RuntimeError('invalid physical or gradient history')
    identifiers = read_population(base, 'calibration')[2]
    expected_grid = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    if len(identifiers) != 1000 or len(expected_grid) != 5000:
        raise RuntimeError('full calibration population changed')
    curves, summaries = [], []
    for name in config['variants']:
        candidates = []
        for step in config['full_calibration_steps']:
            if step > arguments.step:
                continue
            rows = read_rows(training / name / f'full_{step:07d}.csv')
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected_grid:
                raise RuntimeError('a full-calibration source or SNR was omitted')
            means = {key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'lpips', 'bit_error_rate', 'state_error')}
            if not all(np.isfinite(value) for value in means.values()):
                raise RuntimeError('nonfinite full-calibration candidate')
            curves.append({'variant': name, 'step': step, **means})
            candidates.append((means['lpips'], step, rows))
        choice = min(candidates, key=lambda value: (value[0], value[1]))
        selected = milestone['selected'][name]
        if (selected['step'] != choice[1] or abs(selected['lpips'] - choice[0]) > 1e-12 or
            sha256(training / selected['checkpoint']) != selected['checkpoint_sha256']):
            raise RuntimeError('selected grid model differs from the recorded LPIPS-only rule')
        for snrs in (base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]):
            rows = [row for row in choice[2] if float(row['snr_db']) in snrs]
            summaries.append({'variant': name, 'available_updates': arguments.step, 'selected_step': choice[1],
                'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                **{key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'lpips', 'ssim', 'bit_error_rate', 'state_error')}})
    output = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    write_csv(output / 'curves.csv', curves)
    write_csv(output / 'summary.csv', summaries)
    write_json(output / 'completion.json', {'status': 'GRID_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'grid_updates': arguments.step, 'actual_E_R_Adam_data_noise_power_selection_audit': 'PASS',
        'milestone_sha256': sha256(path), 'reference_hashes': references['hashes'], 'source_hashes': sources,
        'selected_steps': {name: selected['step'] for name, selected in milestone['selected'].items()},
        'output_hashes': artifact_hashes(output), 'new_development_access': False, 'research_goal_complete': False})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
