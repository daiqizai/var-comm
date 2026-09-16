"""Audit actual sender exposure and existing calibration curves without new inference."""

import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src'), str(REFERENCE / 'scripts')]

import numpy as np
import torch

from train_milestone import write_csv
from innovation_comm.common import settings
from wetok_comm.common import PROJECT, WORKSPACE, artifact_hashes, configure_torch, now, output_path, sha256, snapshot, verify_sources, write_json
from wetok_comm.deep_support import FrozenDeepSupport, load_deep_support
from wetok_comm.training import paired_batches


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    configure_torch()
    config, base, milestone = settings()
    root = PROJECT / 'outputs' / config['parent_training']
    milestone_path = root / config['parent_milestone']
    verify_sources(milestone['source_hashes'])
    identifiers_path = output_path(base, 'cache') / 'train_ids.json'
    cache = json.loads((identifiers_path.parent / 'completion.json').read_text())
    if sha256(identifiers_path) != cache['output_hashes']['train_ids.json']:
        raise RuntimeError('frozen training source identifiers changed')
    identifiers = json.loads(identifiers_path.read_text())
    if len(identifiers) != 20000 or len(set(identifiers)) != 20000:
        raise RuntimeError('training population is not the fixed 20000 distinct sources')
    variants = ('single_pass', 'multiscale_no_history', 'multiscale_conditioned')
    histories = {name: read_rows(root / name / 'training.csv') for name in variants}
    if any(len(rows) != 7000 for rows in histories.values()):
        raise RuntimeError('frozen joint sender/receiver history must contain 7000 updates')
    exposure = {name: np.zeros((20000, 2), dtype=np.int64) for name in ('representation', 'image', 'total')}
    for batch in paired_batches(base, 20000, 0, 7000):
        phase = 'representation' if batch['step'] < 2000 else 'image'
        for name in variants:
            row = histories[name][batch['step']]
            if int(row['total_step']) != batch['step'] + 1 or row['phase'] != phase or row['batch_sha256'] != batch['fingerprint']:
                raise RuntimeError('actual parent history does not match the registered source/augmentation/noise sampler')
        for index, flip in zip(batch['indices'].tolist(), batch['flip'].tolist()):
            exposure[phase][index, int(flip)] += 1
            exposure['total'][index, int(flip)] += 1
    summaries, source_rows = {}, []
    for name, counts in exposure.items():
        totals = counts.sum(1)
        values, frequencies = np.unique(totals, return_counts=True)
        summaries[name] = {'source_exposures': int(counts.sum()), 'distinct_sources': int(np.count_nonzero(totals)),
            'distinct_source_views': int(np.count_nonzero(counts)), 'sources_with_no_exposure': int(np.count_nonzero(totals == 0)),
            'dataset_pass_equivalents': float(counts.sum() / 20000),
            'source_exposure_histogram': dict(zip(map(str, values.tolist()), frequencies.tolist()))}
    for index, identifier in enumerate(identifiers):
        source_rows.append({'source_index': index, 'image_id': identifier,
            **{f'{phase}_view_{view}': int(counts[index, view]) for phase, counts in exposure.items() for view in (0, 1)}})
    curves, changes = [], []
    metrics = ('psnr_db', 'lpips', 'bit_error_rate', 'feature_mse')
    calibration_ids_path = identifiers_path.parent / 'calibration_ids.json'
    if sha256(calibration_ids_path) != cache['output_hashes']['calibration_ids.json']:
        raise RuntimeError('calibration source binding changed')
    calibration_ids = json.loads(calibration_ids_path.read_text())
    expected = {(identifier, float(snr)) for identifier in calibration_ids for snr in base['channel']['snrs_db']}
    for name in variants:
        endpoints = {}
        for total in (2000, 3000, 4500, 7000):
            path = root / name / f'full_total_{total:07d}.csv'
            rows = read_rows(path)
            if len(rows) != 5000 or {(row['image_id'], float(row['snr_db'])) for row in rows} != expected:
                raise RuntimeError('complete parent calibration grid differs')
            endpoints[total] = {(row['image_id'], float(row['snr_db'])): row for row in rows}
            for snrs in (base['channel']['snrs_db'], *[[snr] for snr in base['channel']['snrs_db']]):
                subset = [row for row in rows if float(row['snr_db']) in snrs]
                curves.append({'variant': name, 'total_updates': total, 'image_updates': total - 2000,
                    'snrs_db': '+'.join(map(str, snrs)), 'source_images': 1000,
                    **{metric: float(np.mean([float(row[metric]) for row in subset])) for metric in metrics}})
        for metric in metrics:
            differences = np.array([np.mean([float(endpoints[7000][identifier, snr][metric]) -
                float(endpoints[4500][identifier, snr][metric]) for snr in base['channel']['snrs_db']]) for identifier in calibration_ids])
            changes.append({'variant': name, 'metric': metric, 'from_total': 4500, 'to_total': 7000,
                'mean_delta': float(differences.mean()), 'source_fraction_lower': float(np.mean(differences < 0)),
                'source_fraction_higher': float(np.mean(differences > 0)),
                'role': 'calibration_trajectory_not_independent_confirmation_or_convergence_proof'})
    deep = load_deep_support()
    initial = torch.load(WORKSPACE / deep['initializer'], map_location='cpu', weights_only=True)
    selected = torch.load(WORKSPACE / deep['checkpoint'], map_location='cpu', weights_only=True)
    deep_root = (WORKSPACE / deep['checkpoint']).parent.parent
    history = read_rows(deep_root / 'history.csv')
    if sum(int(row['optimizer_steps']) for row in history) != selected['global_step'] or selected['epoch'] != len(history):
        raise RuntimeError('Deep selected checkpoint and recorded training opportunity differ')
    network = FrozenDeepSupport(deep, 'cpu')
    deep_evidence = {'source_checkpoint_sha256': deep['checkpoint_sha256'], 'initializer_sha256': deep['initializer_sha256'],
        'actual_communication_parameters': sum(value.numel() for value in network.model.parameters()),
        'ImageNet_finetune_source_exposures': sum(int(row['train_images']) for row in history),
        'ImageNet_finetune_optimizer_steps': selected['global_step'], 'ImageNet_finetune_max_batch_size': selected['config']['training']['batch_size'],
        'partial_shard_batches_are_not_full_batch_exposures': True,
        'selected_COCO_initializer_epoch_field': initial['epoch'], 'selected_COCO_initializer_optimizer_steps': initial['global_optimizer_steps'],
        'still_earlier_initialization': initial['config']['initialization'],
        'legacy_initial_training_channel_caveat': selected['config']['initialization']['source_channel_caveat'],
        'kept_as_strong_physical_budget_reference_not_equal_parameter_or_equal_history_control': True}
    output = PROJECT / 'outputs/WETOK-TRAINING-SUFFICIENCY-20260913'
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output, [Path(__file__), REFERENCE / 'src/wetok_comm/training.py',
        REFERENCE / 'configs/study.yaml', REFERENCE / 'configs/geometry_study.yaml', REFERENCE / 'configs/deep_support.yaml'])
    write_json(output / 'source_exposure_summary.json', summaries)
    write_csv(output / 'per_source_exposures.csv', source_rows)
    write_csv(output / 'parent_calibration_curves.csv', curves)
    write_csv(output / 'parent_last_interval_deltas.csv', changes)
    write_json(output / 'strong_Deep_training_context.json', deep_evidence)
    inputs = [milestone_path, identifiers_path, calibration_ids_path, deep_root / 'history.csv',
        deep_root / 'train_deepjscc3060_perceptual_sweep.py',
        *[root / name / 'training.csv' for name in variants],
        *[root / name / f'full_total_{total:07d}.csv' for name in variants for total in (2000, 3000, 4500, 7000)]]
    write_json(output / 'completion.json', {'status': 'PARENT_TRAINING_EXPOSURE_CALIBRATION_AUDIT_COMPLETE',
        'completed_local': now(), 'source_hashes': source_hashes, 'input_hashes': {str(path): sha256(path) for path in inputs},
        'GPU_used': False, 'new_training_updates': 0, 'new_model_inference': False, 'new_development_access': False,
        'research_goal_complete': False, 'output_hashes': artifact_hashes(output)})
    print(json.dumps({'exposures': summaries, 'last_interval_LPIPS': [row for row in changes if row['metric'] == 'lpips'],
                      'strong_Deep': deep_evidence}, indent=2))


if __name__ == '__main__':
    main()
