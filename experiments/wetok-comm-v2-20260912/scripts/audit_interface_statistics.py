"""Independent CSV/group-count bootstrap audit; not a second neural image evaluation."""

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np

from wetok_comm.common import now, sha256, write_json
from wetok_comm.interface_study import interface_output, load_interface_study


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        print('PLAN ONLY: independent group-count bootstrap and unchanged-native-image checks')
        return
    study, base, parent = load_interface_study()
    evaluation = interface_output(study, 'evaluation') / f'additional_{arguments.milestone:07d}'
    analysis = interface_output(study, 'analysis') / f'development_{arguments.milestone:07d}'
    receipt = json.loads((evaluation / 'completion.json').read_text())
    analysis_receipt = json.loads((analysis / 'completion.json').read_text())
    for root, proof, filename in ((evaluation, receipt, 'per_frame.csv'), (analysis, analysis_receipt, 'paired.csv')):
        if sha256(root / filename) != proof['output_hashes'][filename]:
            raise RuntimeError('audited result rows changed')
    rows = read_rows(evaluation / 'per_frame.csv')
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    if len(rows) != 33600 or len(lookup) != len(rows):
        raise RuntimeError('main comparison count or uniqueness changed')
    native_replays = 0
    for variant in study['variants']:
        for index in range(100):
            for snr in base['evaluation']['snrs_db']:
                for seed in base['evaluation']['noise_seeds']:
                    selected = [lookup[index, snr, seed, prefix + variant] for prefix in
                        ('hard_identity__', 'hard_bounded__', 'old_selected__')]
                    if len({row['image_sha256'] for row in selected}) != 1:
                        raise RuntimeError('selected hard parent outputs do not match the frozen reference')
                    native_replays += 2
    generator = np.random.default_rng(base['evaluation']['bootstrap_seed'])
    draws = generator.integers(0, 100, size=(base['evaluation']['bootstrap_resamples'], 100))
    offsets = np.arange(len(draws))[:, None] * 100
    counts = np.bincount((draws + offsets).ravel(), minlength=len(draws) * 100).reshape(len(draws), 100).astype(np.float64)
    cache = {}
    def source_means(name, snrs, metric):
        key = name, snrs, metric
        if key not in cache:
            values = np.empty((100, len(snrs), len(base['evaluation']['noise_seeds'])), dtype=np.float64)
            for index in range(100):
                for position, snr in enumerate(snrs):
                    for repetition, seed in enumerate(base['evaluation']['noise_seeds']):
                        values[index, position, repetition] = float(lookup[index, snr, seed, name][metric])
            cache[key] = values.mean(axis=2).mean(axis=1)
        return cache[key]
    maximum_delta_error, maximum_interval_error, checked = 0., 0., 0
    for row in read_rows(analysis / 'paired.csv'):
        snrs = tuple(float(value) for value in row['snrs_db'].split('+'))
        differences = source_means(row['method'], snrs, row['metric']) - source_means(row['control'], snrs, row['metric'])
        distribution = (counts * differences[None]).sum(axis=1) / 100
        low, high = np.quantile(distribution, (.025, .975))
        maximum_delta_error = max(maximum_delta_error, abs(float(differences.mean()) - float(row['delta'])))
        maximum_interval_error = max(maximum_interval_error, abs(float(low) - float(row['ci_low'])), abs(float(high) - float(row['ci_high'])))
        checked += 1
    if checked != 3024 or max(maximum_delta_error, maximum_interval_error) > 1e-10:
        raise RuntimeError('independent source-image resampling did not reproduce the reported comparisons')
    output = EXPERIMENT / 'docs' / f'interface_statistics_audit_{arguments.milestone:07d}.json'
    if output.exists():
        raise FileExistsError('prior audit evidence must not be replaced')
    record = {'status': 'INDEPENDENT_INTERFACE_STATISTICS_AUDIT_PASS', 'completed_local': now(),
        'main_wireless_rows': len(rows), 'unchanged_hard_reference_image_hash_comparisons': native_replays,
        'paired_intervals_recomputed': checked, 'maximum_delta_error': maximum_delta_error,
        'maximum_interval_error': maximum_interval_error, 'source_script_sha256': sha256(Path(__file__)),
        'evaluation_receipt_sha256': sha256(evaluation / 'completion.json'),
        'method': 'source means average noise then SNR; bootstrap implemented via source occurrence counts',
        'scope': 'Independent numerical/statistical path and frozen-image hash equality, not new neural/metric inference or independent holdout.'}
    write_json(output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
