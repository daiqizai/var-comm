"""Qualify existing continuous-control histories for a future geometry comparison."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / 'src'), str(EXPERIMENT / 'scripts')]

import numpy as np
import torch

from wetok_comm.common import configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.geometry_candidate import matched_geometry_initializations
from wetok_comm.interface_study import interface_output, load_interface_study
from wetok_comm.training import module_sha256, paired_batches


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: check historical 2000 representation + 5000 continuous-image controls, no updates or GPU')
        return
    configure_torch()
    study, base, parent = load_interface_study()
    training = interface_output(study, 'training')
    milestone_path = training / 'milestones/additional_0005000.json'
    milestone = json.loads(milestone_path.read_text())
    verify_sources(milestone['source_hashes'])
    if milestone['status'] != 'INTERFACE_MILESTONE_COMPLETE' or milestone['additional_updates_per_arm'] != 5000:
        raise RuntimeError('complete continuous-control training history is not available')
    checkpoint = Path(milestone['checkpoint'])
    if sha256(checkpoint) != milestone['optimizer_checkpoint_sha256']:
        raise RuntimeError('frozen 5000 control checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    original_initial = json.loads((parent / 'initialization.json').read_text())['initial_model_hashes']
    current_initial = json.loads((training / 'initialization.json').read_text())
    if not current_initial['all_optimizers_fresh_and_equal'] or current_initial['inherited_Adam_moments']:
        raise RuntimeError('image-phase optimizer reset is not the recorded common fresh reset')
    prefix_histories, records = {}, []
    for variant in study['variants']:
        name = 'continuous_mean__' + variant
        control, candidate, match = matched_geometry_initializations(base, variant)
        if module_sha256(control) != original_initial[variant]:
            raise RuntimeError('prepared geometry control does not match the real original initialization')
        prefix_path = parent / variant / 'checkpoints/step_0002000.pt'
        if sha256(prefix_path) != study['parent_checkpoints'][variant]:
            raise RuntimeError('original representation checkpoint changed')
        control.load_state_dict(torch.load(prefix_path, map_location='cpu', weights_only=True)['model'], strict=True)
        if module_sha256(control) != current_initial['model_hashes'][name]:
            raise RuntimeError('continuous control did not actually start from this representation checkpoint')
        prefix_histories[variant] = read_rows(parent / variant / 'training.csv')[:2000]
        if len(prefix_histories[variant]) != 2000 or len(saved['histories'][name]) != 5000:
            raise RuntimeError('control update counts do not cover both full phases')
        if any(int(value['step']) != 5000 for value in saved['optimizers'][name]['state'].values()):
            raise RuntimeError('image control optimizer steps differ from the reset image history')
        candidates = []
        for image_step in (0, 1000, 2500, 5000):
            path = training / name / f'full_{image_step:07d}.csv'
            rows = read_rows(path)
            if len(rows) != 5000 or len({(row['image_id'], row['snr_db']) for row in rows}) != 5000:
                raise RuntimeError('control full-calibration candidate grid is incomplete')
            candidates.append((float(np.mean([float(row['lpips']) for row in rows])), image_step))
        chosen = min(candidates)
        if chosen[1] != milestone['selected'][name]['step'] or abs(chosen[0] - milestone['selected'][name]['lpips']) > 1e-12:
            raise RuntimeError('control candidate pool or LPIPS selection differs')
        records.append({'variant': variant, 'control_initial_sha256': original_initial[variant],
            'prefix_checkpoint_sha256': sha256(prefix_path), 'continuous_phase_initial_sha256': module_sha256(control),
            'selected_image_step': chosen[1], 'calibration_candidates': [item[1] for item in candidates], **match})
        del control, candidate
    for index, batch in enumerate(paired_batches(base, 20000, 0, 2000)):
        for variant in study['variants']:
            row = prefix_histories[variant][index]
            if (int(row['step']) != index + 1 or int(row['optimizer_step']) != index + 1 or row['phase'] != 'representation' or
                row['batch_sha256'] != batch['fingerprint'] or float(row['learning_rate']) != .0003):
                raise RuntimeError('original representation data/noise/optimizer schedule differs')
    for index, batch in enumerate(paired_batches(base, 20000, 2000, 7000)):
        for variant in study['variants']:
            row = saved['histories']['continuous_mean__' + variant][index]
            if (row['additional_step'] != index + 1 or row['global_data_step'] != batch['step'] + 1 or
                row['Adam_step'] != index + 1 or row['batch_sha256'] != batch['fingerprint']):
                raise RuntimeError('recorded continuous data/noise/reset-Adam sequence differs')
    output = training.parent / 'WETOK-GEOMETRY-CONTROL-QUALIFICATION-20260913'
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(output, [Path(__file__), EXPERIMENT / 'src/wetok_comm/geometry_candidate.py',
        EXPERIMENT / 'tests/test_geometry_candidate.py', EXPERIMENT / 'configs/study.yaml',
        EXPERIMENT / 'configs/interface_study.yaml'])
    result = {'status': 'HISTORICAL_GEOMETRY_CONTROLS_QUALIFIED_FOR_MATCHED_FUTURE_STUDY', 'completed_local': now(),
        'source_hashes': sources, 'control_training_milestone_sha256': sha256(milestone_path),
        'control_optimizer_checkpoint_sha256': sha256(checkpoint), 'control_variants': records,
        'representation_updates': 2000, 'image_updates': 5000, 'effective_batch': 4, 'microbatch': 1,
        'initialization_cpu_threads': torch.get_num_threads(),
        'registered_image_weights': study['weights'], 'representation_weights': base['training']['representation_weights'],
        'new_training_updates': 0, 'GPU_used': False,
        'scope': 'Validates real historical controls and code/initialization/policy equivalence; does not train the new geometry or prove its quality. Candidate must follow this same history and selection opportunity.'}
    write_json(output / 'qualification.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_hashes', 'control_variants')}, indent=2))


if __name__ == '__main__':
    main()
