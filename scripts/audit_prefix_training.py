#!/usr/bin/env python3
"""Check completed update budgets, curriculum, calibration choices and fixed backbones."""

import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, verify_snapshot, write_json


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    config = yaml.safe_load((ROOT / 'configs/learned_prefix_jscc.yaml').read_text())
    run = ROOT / config['outputs']['training']
    receipt = verify_artifacts(run, 'completion.json')
    verify_snapshot(receipt['source_hashes'])
    output = create_output(ROOT / 'outputs/VAR-PREFIX-JSCC-TRAIN-AUDIT-001')
    sources = snapshot(output, [Path(__file__)])
    assert receipt['optimizer_updates_per_variant'] == 10000 and receipt['training_images_seen_per_variant'] == 40000
    assert receipt['frozen_before'] == receipt['frozen_after'] and not receipt['DINO_training_or_selection']
    initialization = json.loads((run / 'initialization.json').read_text())
    assert len(set(initialization['state_sha256'].values())) == 1
    assert set(initialization['trainable_counts'].values()) == {1508750}
    assert initialization['fresh_initialization_no_selfcheck_weights_reused']
    histories, checks = {}, []
    for variant in config['variants']:
        history = read_csv(run / variant / 'training.csv')
        assert len(history) == 10000
        assert [int(row['step']) for row in history] == list(range(1, 10001))
        assert [int(row['samples_seen']) for row in history] == list(range(4, 40001, 4))
        assert all(np.isfinite(float(row[name])) for row in history for name in ('mse', 'lpips', 'token_ce', 'gradient_norm', 'step_seconds'))
        assert all(float(row['power_max_error']) < 1e-5 for row in history)
        assert not any('dino' in name.lower() for name in history[0])
        if variant == 'next_scale':
            for row in history:
                expected = max(0.0, 1 - (int(row['step']) - 1) / 4000)
                assert abs(float(row['teacher_probability']) - expected) < 1e-12
                if int(row['step']) >= 4001:
                    assert float(row['teacher_samples']) == 0
        else:
            assert all(float(row['teacher_samples']) == 0 for row in history)
        candidates = []
        for epoch in (1, 2):
            calibration = read_csv(run / variant / f'calibration_epoch_{epoch:02d}.csv')
            assert len(calibration) == 5000 and len({row['image_id'] for row in calibration}) == 1000
            assert {float(row['snr_db']) for row in calibration} == set(config['selection']['snrs_db'])
            assert not any('dino' in name.lower() for name in calibration[0])
            for row in calibration:
                assert abs(float(row['objective']) - float(row['mse']) - 0.01 * float(row['lpips'])) < 1e-7
            candidates.append((float(np.mean([float(row['objective']) for row in calibration])), epoch))
        expected_epoch = min(candidates)[1]
        selected = json.loads((run / variant / 'selected.json').read_text())
        assert selected['epoch'] == expected_epoch == receipt['selected'][variant]['epoch']
        assert sha256(run / selected['checkpoint']) == selected['checkpoint_sha256']
        checkpoint = torch.load(run / selected['checkpoint'], map_location='cpu', weights_only=False)
        assert checkpoint['global_step'] == 5000 * expected_epoch and checkpoint['initial_state_sha256'] == initialization['state_sha256'][variant]
        assert checkpoint['variant'] == variant
        checks.append({'variant': variant, 'updates': len(history), 'selected_epoch': expected_epoch,
                       'selected_image_objective': min(candidates)[0], 'teacher_samples_total': sum(float(row['teacher_samples']) * 4 for row in history)})
        histories[variant] = history
    for before, after in zip(histories['parallel'], histories['next_scale']):
        for field in ('epoch', 'step', 'samples_seen', 'header_usable_fraction', 'header_false_acceptances', 'learning_rate'):
            assert before[field] == after[field], field
    train_data = json.loads((ROOT / config['data_cache'] / 'train_manifest.json').read_text())
    calibration_data = json.loads((ROOT / config['data_cache'] / 'calibration_manifest.json').read_text())
    train_hashes = {entry[field] for entry in train_data for field in ('pixel_sha256', 'flip_pixel_sha256')}
    calibration_hashes = {entry[field] for entry in calibration_data for field in ('pixel_sha256', 'flip_pixel_sha256')}
    assert not train_hashes & calibration_hashes
    result = {'status': 'TRAINING_BUDGET_AND_SELECTION_AUDIT_PASS', 'checks': checks, 'frozen_backbones': receipt['frozen_before'],
              'image_objective_calibration_only_no_DINO': True, 'matched_initialization_and_update_budget': True,
              'final_6000_updates_self_history_only': True, 'training_receipt_sha256': sha256(run / 'completion.json'),
              'source_hashes': sources, 'output_hashes': artifact_hashes(output)}
    write_json(output / 'audit.json', result)
    print({name: value for name, value in result.items() if name not in ('source_hashes', 'output_hashes')}, flush=True)


if __name__ == '__main__':
    main()
