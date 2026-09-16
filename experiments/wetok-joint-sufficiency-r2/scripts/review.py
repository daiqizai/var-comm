"""Audit inherited5000 state, new paired updates, complete-calibration choice and actual Adam continuation."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
GRID = EXPERIMENT.parent / 'wetok-joint-grid-controls-r1'
JOINT = EXPERIMENT.parent / 'wetok-joint-sender-r1'
INNOVATION = EXPERIMENT.parent / 'wetok-innovation-r1'
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(GRID / 'src'), str(JOINT / 'src'), str(INNOVATION / 'src'), str(BASE / 'src'), str(BASE / 'scripts')]

import numpy as np
import torch

from joint_sender.runtime import tensor_sha256
from sufficiency.common import load_parent, load_sources, output_path, restore_systems, settings, tree_digest
from train_milestone import write_csv
from wetok_comm.common import artifact_hashes, configure_torch, now, sha256, snapshot, verify_sources, write_json
from wetok_comm.training import module_sha256, paired_batches, read_population


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    config, original, grid, reference, base, parent_record = settings()
    if arguments.step not in config['new_full_steps']:
        raise ValueError('review only registered7500/10000 milestones')
    if not arguments.execute:
        print('PLAN ONLY: inherited model/Adam history and new paired data/full-calibration audit; no GPU inference')
        return
    configure_torch()
    sources = load_sources(config)
    root = output_path(config, 'training')
    metadata = json.loads((root / 'metadata.json').read_text())
    initialization = json.loads((root / 'initialization.json').read_text())
    preparation_path = output_path(config, 'preparation') / 'resume_initialization.json'
    preparation = json.loads(preparation_path.read_text())
    if (metadata['preparation_sha256'] != sha256(preparation_path) or metadata['source_endpoint_hashes'] != sources['hashes'] or
        initialization['restored'] != preparation['restored'] or initialization['starting_optimizer_steps'] != 5000):
        raise RuntimeError('R2 original model/Adam restoration record changed')
    milestone_path = root / 'milestones' / f'step_{arguments.step:07d}.json'
    milestone = json.loads(milestone_path.read_text())
    if (milestone['status'] != 'SUFFICIENCY_MILESTONE_COMPLETE' or milestone['total_updates_per_arm'] != arguments.step or
        milestone['new_updates_per_arm'] != arguments.step - 5000 or milestone['global_data_step'] != 7000 + arguments.step or
        milestone['source_endpoint_hashes'] != sources['hashes'] or sha256(milestone['checkpoint']) != milestone['checkpoint_sha256']):
        raise RuntimeError('R2 milestone identity, count or frozen endpoint changed')
    verify_sources(milestone['source_hashes'])
    saved = torch.load(milestone['checkpoint'], map_location='cpu', weights_only=True)
    parent = load_parent(reference, base, 'cpu')
    if (saved['completed_total_updates'] != arguments.step or saved['global_data_step'] != 7000 + arguments.step or
        saved['frozen'] != sources['frozen'] or module_sha256(parent) != sources['frozen']['parent']):
        raise RuntimeError('R2 saved state or visual identity differs')
    models, optimizers, restored = restore_systems(config, reference, base, sources, parent, 'cpu', saved)
    initial = config['initial_total_updates']
    for name in config['variants']:
        history = saved['histories'][name]
        if len(history) != arguments.step or tree_digest(history[:initial]) != metadata['inherited_history_sha256'][name]:
            raise RuntimeError('inherited training history was replaced or duplicated')
        if restored[name]['model_tree_sha256'] == preparation['restored'][name]['model_tree_sha256']:
            raise RuntimeError('R2 did not change the actual5000 communication model')
        for step, binding in sources['calibration_files'][name].items():
            if saved['calibration_files'][name].get(step) != binding or sha256(binding['path']) != binding['sha256']:
                raise RuntimeError('an inherited full-calibration artifact changed')
    for offset, batch in enumerate(paired_batches(base, 20000, 12000, 7000 + arguments.step)):
        noise_hash = tensor_sha256(batch['noise'])
        for name in config['variants']:
            row = saved['histories'][name][initial + offset]
            if (row['step'] != initial + offset + 1 or row['global_data_step'] != batch['step'] + 1 or
                row['batch_sha256'] != batch['fingerprint'] or row['paired_standard_noise_sha256'] != noise_hash):
                raise RuntimeError('R2 data/augmentation/SNR/noise did not continue from global12000')
            if row['power_max_error'] > 1e-5 or any(not np.isfinite(row[key]) for key in
                ('loss', 'gradient_norm', 'encoder_gradient_norm', 'receiver_gradient_norm', 'waveform_gradient_norm')):
                raise RuntimeError('R2 has invalid energy or gradient history')
    identifiers = read_population(base, 'calibration')[2]
    expected = {(identifier, float(snr)) for identifier in identifiers for snr in base['channel']['snrs_db']}
    if len(identifiers) != 1000 or len(expected) != 5000:
        raise RuntimeError('R2 full calibration population changed')
    curves, summaries = [], []
    steps = config['inherited_full_steps'] + [step for step in config['new_full_steps'] if step <= arguments.step]
    for name in config['variants']:
        candidates = []
        for step in steps:
            binding = saved['calibration_files'][name][str(step)]
            if sha256(binding['path']) != binding['sha256']:
                raise RuntimeError('R2 full-calibration scores changed')
            rows = read_rows(binding['path'])
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected:
                raise RuntimeError('R2 full-calibration grid is missing sources or SNRs')
            means = {key: float(np.mean([float(row[key]) for row in rows])) for key in ('psnr_db', 'lpips', 'bit_error_rate', 'state_error')}
            if not all(np.isfinite(value) for value in means.values()):
                raise RuntimeError('nonfinite calibration must not be discarded')
            curves.append({'variant': name, 'step': step, 'inherited_from_previous_trial': step <= initial, **means})
            candidates.append((means['lpips'], step, rows))
        chosen = min(candidates, key=lambda value: (value[0], value[1]))
        selected = milestone['selected'][name]
        if selected['step'] != chosen[1] or abs(selected['lpips'] - chosen[0]) > 1e-12 or sha256(selected['checkpoint']) != selected['checkpoint_sha256']:
            raise RuntimeError('R2 selected checkpoint violates the inherited/full-calibration LPIPS rule')
        if selected['step'] <= initial and selected != sources['selected'][name]:
            raise RuntimeError('R2 altered the inherited best checkpoint instead of preserving it')
        for snrs in (base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]):
            subset = [row for row in chosen[2] if float(row['snr_db']) in snrs]
            summaries.append({'variant': name, 'available_total_updates': arguments.step, 'new_updates': arguments.step - initial,
                'selected_step': chosen[1], 'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                **{key: float(np.mean([float(row[key]) for row in subset])) for key in ('psnr_db', 'lpips', 'ssim', 'bit_error_rate', 'state_error')}})
    output = output_path(config, 'analysis') / f'calibration_{arguments.step:07d}'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), EXPERIMENT / 'configs/study.yaml'])
    write_csv(output / 'curves.csv', curves)
    write_csv(output / 'summary.csv', summaries)
    write_json(output / 'completion.json', {'status': 'R2_CALIBRATION_REVIEW_READY_NOT_RESEARCH_COMPLETE',
        'completed_local': now(), 'total_updates': arguments.step, 'new_updates': arguments.step - initial,
        'model_Adam_inheritance_new_data_power_selection_audit': 'PASS', 'milestone_sha256': sha256(milestone_path),
        'source_endpoint_hashes': sources['hashes'], 'source_hashes': source_hashes,
        'selected_steps': {name: choice['step'] for name, choice in milestone['selected'].items()},
        'new_development_access': False, 'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
