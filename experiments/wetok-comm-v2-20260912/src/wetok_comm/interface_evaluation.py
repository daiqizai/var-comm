"""Frozen image references and strict source-paired statistics for receiver interfaces."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, PROJECT, sha256, verify_sources
from .interface_study import INTERFACES, interface_definitions, load_interface_study


METRICS = ('psnr_db', 'ssim', 'lpips', 'dino', 'LPIPS_excess_from_native', 'severe_distortion')
SYSTEM_REFERENCES = ('wetok_8PSK_FEC', 'digital_m8', 'digital_adaptive', 'perceptual_deepjscc')


def load_evaluation_config():
    config = yaml.safe_load((EXPERIMENT / 'configs/interface_evaluation.yaml').read_text())
    study, base, parent = load_interface_study(EXPERIMENT / config['study_config'])
    if config['population'] != 'original_100_development_only' or config['source_images'] != 100:
        raise ValueError('this registered evaluation does not permit a new test population')
    return config, study, base, parent


def reference_names(study):
    return ['old_selected__' + variant for variant in study['variants']] + list(SYSTEM_REFERENCES)


def comparisons(study):
    pairs = []
    for variant in study['variants']:
        pairs.extend([(f'hard_bounded__{variant}', f'hard_identity__{variant}'),
                      (f'continuous_mean__{variant}', f'hard_bounded__{variant}'),
                      (f'continuous_mean__{variant}', f'hard_identity__{variant}')])
    for interface in study['interfaces']:
        pairs.extend([(f'{interface}__multiscale_conditioned', f'{interface}__multiscale_no_history'),
                      (f'{interface}__multiscale_conditioned', f'{interface}__single_pass'),
                      (f'{interface}__multiscale_no_history', f'{interface}__single_pass')])
    for name, definition in interface_definitions(study).items():
        pairs.extend((name, reference) for reference in SYSTEM_REFERENCES)
        pairs.append((name, 'old_selected__' + definition['variant']))
    if len(pairs) != len(set(pairs)):
        raise ValueError('duplicated causal comparisons')
    return pairs


class FrozenImageReferences:
    def __init__(self, config, study):
        self.root = PROJECT / 'outputs' / config['reference_evaluation']
        receipt_path = self.root / 'completion.json'
        if sha256(receipt_path) != config['reference_receipt_sha256']:
            raise RuntimeError('frozen reference evaluation receipt changed')
        self.receipt = json.loads(receipt_path.read_text())
        if self.receipt['status'] != 'EVALUATION_COMPLETE' or self.receipt['milestone_sha256'] != study['parent_milestone_sha256']:
            raise RuntimeError('reference models do not belong to the registered original history')
        verify_sources(self.receipt['source_hashes'])
        if sha256(self.root / 'per_frame.csv') != self.receipt['output_hashes']['per_frame.csv']:
            raise RuntimeError('reference quality/physical ledger changed')
        with (self.root / 'per_frame.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
        if len(self.rows) != 14700 or len(self.rows) != len(rows):
            raise RuntimeError('reference comparison grid incomplete or duplicated')
        self.names = reference_names(study)
        self.loaded_index, self.images, self.archive = None, None, None

    def image(self, index, identifier, snr, seed, name, noise_hash):
        if name not in self.names:
            raise ValueError('unknown frozen reference')
        original = name.split('__', 1)[1] if name.startswith('old_selected__') else name
        row = self.rows[index, float(snr), int(seed), original]
        if row['image_id'] != identifier or row['noise_sha256'] != noise_hash:
            raise RuntimeError('reference source/noise pairing changed')
        header = 68 if original in ('digital_m8', 'digital_adaptive') else 0
        if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header):
            raise RuntimeError('reference N/header/data ledger changed')
        if abs(float(row['total_energy']) - 6120) > 1e-5:
            raise RuntimeError('reference energy differs from the paid comparison')
        if self.loaded_index != index:
            relative = f'images/{index:03d}/reconstructions.npz'
            self.archive = self.root / relative
            if sha256(self.archive) != self.receipt['output_hashes'][relative]:
                raise RuntimeError('reference image archive changed')
            with np.load(self.archive, allow_pickle=False) as data:
                self.images = data['images'].copy()
            self.loaded_index = index
        values = self.images[int(row['image_ref'])]
        if values.dtype != np.float32 or values.shape != (3, 256, 256):
            raise RuntimeError('reference images must retain their actual float32 output')
        if hashlib.sha256(values.tobytes()).hexdigest() != row['image_sha256']:
            raise RuntimeError('referenced image does not match the saved wireless row')
        return torch.from_numpy(values.copy()), row, str(self.archive)


def feature_diagnostics(result, source, interface):
    if interface not in INTERFACES:
        raise ValueError('unknown deployed receiver interface')
    features = result['receiver_features'].detach()
    if not bool(torch.isfinite(features).all()) or bool((features.abs() > 1).any()):
        raise RuntimeError('receiver output violates its finite bounded interface')
    hard = interface != 'continuous_mean'
    if hard and not bool(((features == 1) | (features == -1)).all()):
        raise RuntimeError('hard receiver supplied non-native features')
    expected_hard = torch.where(result['logits'].detach() > 0, 1., -1.)
    if not torch.equal(result['native_fq'].detach(), expected_hard) or (hard and not torch.equal(features, expected_hard)):
        raise RuntimeError('hard image input or diagnostic bits differ from the receiver logits')
    if not hard and not torch.equal(features, torch.tanh(result['logits'].detach() / 2)):
        raise RuntimeError('continuous receiver was projected or changed before decoding')
    return {'decoder_interface': interface, 'decoded_feature_kind': 'native_signs' if hard else 'continuous_bounded_features',
            'bit_metric_role': 'estimated_native_signs' if hard else 'diagnostic_thresholded_logits_not_delivered_bits',
            'bit_error_rate': float(result['native_fq'].detach().ne(source).float().mean()),
            'feature_mse': float((features - source).square().mean()), 'feature_abs_mean': float(features.abs().mean()),
            'feature_saturation_fraction': float(features.abs().ge(.99).float().mean())}


def validate_grid(rows, study, base):
    definitions = interface_definitions(study)
    names = list(definitions) + reference_names(study)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, float(snr), int(seed), name) for index in range(100) for snr in snrs for seed in seeds for name in names}
    if len(rows) != len(lookup) or set(lookup) != expected:
        raise RuntimeError('interface wireless grid is incomplete, duplicated or contains unregistered methods')
    identifiers = []
    for index in range(100):
        source_ids = set()
        for seed in seeds:
            hashes = set()
            for snr in snrs:
                for name in names:
                    row = lookup[index, float(snr), int(seed), name]
                    source_ids.add(row['image_id'])
                    hashes.add(row['noise_sha256'])
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header):
                        raise RuntimeError('comparison physical use/header ledger drift')
                    if abs(float(row['total_energy']) - 6120) > 1e-5:
                        raise RuntimeError('comparison total energy drift')
                    if name in definitions:
                        interface = definitions[name]['interface']
                        kind = 'continuous_bounded_features' if interface == 'continuous_mean' else 'native_signs'
                        if row['decoder_interface'] != interface or row['decoded_feature_kind'] != kind:
                            raise RuntimeError('receiver interface mislabeled in a wireless row')
                        if int(row['raw_bits']) != 8192 or row['coded_bits'] != '':
                            raise RuntimeError('native source bits and learned physical symbols were conflated')
                    if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                        raise RuntimeError('nonfinite image outcome; difficult samples must not be dropped')
                    if not (-1.00001 <= float(row['dino']) <= 1.00001 and -1.00001 <= float(row['ssim']) <= 1.00001):
                        raise RuntimeError('bounded image similarity metric is outside its range')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('severe-distortion flag does not use the registered threshold')
            if len(hashes) != 1:
                raise RuntimeError('source/seed noise is not paired across all SNRs and methods')
        if len(source_ids) != 1:
            raise RuntimeError('source IDs differ across a paired image group')
        identifiers.extend(source_ids)
    if len(set(identifiers)) != 100:
        raise RuntimeError('source-image bootstrap population contains duplicated images')
    return lookup, names


def statistics(rows, study, base):
    lookup, names = validate_grid(rows, study, base)
    seeds = base['evaluation']['noise_seeds']
    resampled = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100,
        size=(base['evaluation']['bootstrap_resamples'], 100))
    summary, paired = [], []
    for snrs in [base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]]:
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, float(snr), int(seed), name][metric])
            for snr in snrs for seed in seeds]) for index in range(100)]) for metric in METRICS} for name in names}
        for name in names:
            row = {'snrs_db': label, 'arm': name, 'source_images': 100, 'transmissions': 100 * len(snrs) * len(seeds),
                   **{metric: float(value.mean()) for metric, value in values[name].items()}}
            frames = [lookup[index, float(snr), int(seed), name] for index in range(100) for snr in snrs for seed in seeds]
            for field in ('online_TX_seconds', 'receiver_seconds'):
                times = [float(frame[field]) for frame in frames if frame[field] != '']
                row[field + '_mean'] = float(np.mean(times)) if times else ''
                row[field + '_p95'] = float(np.percentile(times, 95)) if times else ''
            summary.append(row)
        for method, control in comparisons(study):
            for metric in METRICS:
                delta = values[method][metric] - values[control][metric]
                low, high = np.percentile(delta[resampled].mean(1), [2.5, 97.5])
                paired.append({'snrs_db': label, 'method': method, 'control': control, 'metric': metric,
                               'delta': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summary, paired
