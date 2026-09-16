"""Training-source-only low-order LFQ statistics, not an achieved coding rate."""

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np

from wetok_comm.common import PROJECT, now, output_path, settings, sha256, write_json


def entropy(counts):
    values = np.asarray(counts, dtype=np.float64)
    values = values[values > 0]
    probability = values / values.sum()
    return float(-np.sum(probability * np.log2(probability)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: count training-only native statistics; no models, no bitstream claims')
        return
    config = settings()
    cache = output_path(config, 'cache')
    receipt = json.loads((cache / 'completion.json').read_text())
    path = cache / 'train_codes.npy'
    if sha256(path) != receipt['output_hashes']['train_codes.npy']:
        raise RuntimeError('training native codes changed')
    codes = np.load(path, mmap_mode='r', allow_pickle=False)[:, 0]
    marginal = np.zeros((4, 256), dtype=np.int64)
    horizontal = np.zeros((4, 256, 256), dtype=np.int64)
    vertical = np.zeros_like(horizontal)
    positive = np.zeros(32, dtype=np.float64)
    pooled_squares = {4: 0., 8: 0.}
    pooled_counts = {4: 0, 8: 0}
    for start in range(0, len(codes), 256):
        batch = np.array(codes[start:start + 256], copy=True)
        for group in range(4):
            values = batch[..., group]
            marginal[group] += np.bincount(values.reshape(-1), minlength=256)
            pairs = values[:, :, :-1].astype(np.int32) * 256 + values[:, :, 1:]
            horizontal[group] += np.bincount(pairs.reshape(-1), minlength=65536).reshape(256, 256)
            pairs = values[:, :-1, :].astype(np.int32) * 256 + values[:, 1:, :]
            vertical[group] += np.bincount(pairs.reshape(-1), minlength=65536).reshape(256, 256)
        bits = np.unpackbits(batch, axis=-1, bitorder='little')
        positive += bits.sum(axis=(0, 1, 2))
        signs = bits.astype(np.float32) * 2 - 1
        for size in (4, 8):
            stride = 16 // size
            pooled = signs.reshape(len(batch), size, stride, size, stride, 32).mean(axis=(2, 4))
            pooled_squares[size] += float(np.square(pooled, dtype=np.float64).sum())
            pooled_counts[size] += pooled.size
    groups = []
    for group in range(4):
        groups.append({'group': group, 'marginal_entropy_bits_per_index': entropy(marginal[group]),
                       'horizontal_conditional_entropy_bits': entropy(horizontal[group]) - entropy(horizontal[group].sum(1)),
                       'vertical_conditional_entropy_bits': entropy(vertical[group]) - entropy(vertical[group].sum(1))})
    probability = positive / (len(codes) * 256)
    pooled = {str(size): {'mean_squared_area_state': pooled_squares[size] / pooled_counts[size],
                         'independent_balanced_bits_reference': 1 / (16 // size) ** 2} for size in (4, 8)}
    output = PROJECT / 'outputs/WETOK-NATIVE-SOURCE-STATS-20260912'
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'counts.npz', group_marginal=marginal, horizontal=horizontal, vertical=vertical, bit_positive_counts=positive)
    record = {'status': 'TRAINING_NATIVE_STATISTICS_COMPLETE', 'completed_local': now(), 'source_images': len(codes),
              'views_per_image': 1, 'training_only': True, 'source_sha256': sha256(path), 'script_sha256': sha256(__file__),
              'groups': groups, 'marginal_plugin_bits_per_image': sum(item['marginal_entropy_bits_per_index'] for item in groups) * 256,
              'bit_positive_probability_range': [float(probability.min()), float(probability.max())], 'area_states': pooled,
              'limitations': ['Plugin entropies are not an implemented bitstream or measured FEC gain.',
                             'Adjacent-code dependence is not evidence that LFQ groups are semantic scales.',
                             'Do not replace true total-channel-use/energy accounting with source entropy.']}
    write_json(output / 'summary.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
