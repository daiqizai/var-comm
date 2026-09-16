"""Matched geometry model registry, frozen controls and source-image statistics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, PROJECT, sha256, verify_sources
from .geometry_candidate import GeometryJSCC
from .geometry_study import geometry_network, load_geometry_study
from .interface_evaluation import FrozenImageReferences, METRICS, SYSTEM_REFERENCES, load_evaluation_config


def load_geometry_evaluation():
    config = yaml.safe_load((EXPERIMENT / 'configs/geometry_evaluation.yaml').read_text())
    study, base, qualification = load_geometry_study()
    if config['matched_image_updates'] != 5000 or config['matched_total_updates'] != 7000:
        raise ValueError('the frozen image controls cover exactly this full training opportunity')
    return config, {**study, 'parent_step': 2000}, base, qualification


def system_reference_names(unused_config):
    return list(SYSTEM_REFERENCES)


def geometry_supplement_statistics(main_rows, support_rows, config, base):
    from .deep_support import supplement_statistics

    labels = {'variants': config['variants'], 'interfaces': ['geometry153x40', 'geometry204x30']}
    return supplement_statistics(main_rows, support_rows, labels, base)


def method_definitions(config):
    return {f'geometry{geometry}__{variant}': {'geometry': geometry, 'variant': variant, 'interface': 'continuous_mean'}
            for geometry in ('153x40', '204x30') for variant in config['variants']}


def model_registry(config, base, qualification, milestone, candidate_training, device):
    if milestone['image_updates_per_new_arm'] != 5000:
        raise ValueError('final geometry evaluation requires the matched 5000-image-update control budget')
    control_training = PROJECT / 'outputs' / config['control_training']
    control = json.loads((control_training / config['control_milestone']).read_text())
    verify_sources(control['source_hashes'])
    networks, selections = {}, {}
    for name, definition in method_definitions(config).items():
        variant = definition['variant']
        if definition['geometry'] == '153x40':
            choice = control['selected']['continuous_mean__' + variant]
            path = control_training / choice['checkpoint']
            network = GeometryJSCC(variant, base['model']).to(device)
        else:
            choice = milestone['selected'][variant]
            path = candidate_training / choice['checkpoint']
            network, unused = geometry_network(config, base, qualification, variant, device)
        if sha256(path) != choice['checkpoint_sha256']:
            raise RuntimeError('selected matched geometry checkpoint changed')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if stored['variant'] != variant:
            raise RuntimeError('receiver variant was swapped in the selected registry')
        if definition['geometry'] == '153x40':
            valid = stored['interface'] == 'continuous_mean' and stored['additional_step'] == choice['step']
        else:
            valid = stored['geometry'] == [204, 30] and stored['image_step'] == choice['step']
        if not valid:
            raise RuntimeError('checkpoint geometry/interface or image step differs from calibration selection')
        network.load_state_dict(stored['model'], strict=True)
        networks[name] = network.eval().requires_grad_(False)
        selections[name] = {**choice, 'absolute_checkpoint': str(path),
                            'available_representation_updates': 2000, 'available_image_updates': 5000}
    return networks, selections


class GeometryReferences:
    def __init__(self, config):
        legacy_config, interface, unused_base, unused_parent = load_evaluation_config()
        self.legacy = FrozenImageReferences(legacy_config, interface)
        self.root = PROJECT / 'outputs' / config['control_evaluation']
        if sha256(self.root / 'completion.json') != config['control_evaluation_receipt_sha256']:
            raise RuntimeError('qualified old-geometry evaluation receipt changed')
        self.receipt = json.loads((self.root / 'completion.json').read_text())
        if self.receipt['status'] != 'INTERFACE_EVALUATION_COMPLETE':
            raise RuntimeError('old geometry evaluation is incomplete')
        verify_sources(self.receipt['source_hashes'])
        if sha256(self.root / 'per_frame.csv') != self.receipt['output_hashes']['per_frame.csv']:
            raise RuntimeError('old geometry source/metric rows changed')
        with (self.root / 'per_frame.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
        if len(self.rows) != 33600 or len(rows) != len(self.rows):
            raise RuntimeError('old geometry reference grid is not complete')
        self.index, self.images = None, None

    def control_image(self, index, identifier, snr, seed, variant, noise_hash):
        row = self.rows[index, float(snr), int(seed), 'continuous_mean__' + variant]
        if row['image_id'] != identifier or row['noise_sha256'] != noise_hash:
            raise RuntimeError('old geometry source/noise pairing changed')
        if row['decoder_interface'] != 'continuous_mean' or row['decoded_feature_kind'] != 'continuous_bounded_features':
            raise RuntimeError('old geometry reference was not the actual continuous receiver')
        if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, 0, 3060):
            raise RuntimeError('old geometry resource ledger differs')
        if abs(float(row['total_energy']) - 6120) > 1e-5:
            raise RuntimeError('old geometry energy differs')
        if self.index != index:
            relative = f'images/{index:03d}/reconstructions.npz'
            path = self.root / relative
            if sha256(path) != self.receipt['output_hashes'][relative]:
                raise RuntimeError('old geometry float image archive changed')
            with np.load(path, allow_pickle=False) as data:
                self.images = data['images'].copy()
            self.index = index
        image = self.images[int(row['image_ref'])]
        if image.dtype != np.float32 or image.shape != (3, 256, 256) or hashlib.sha256(image.tobytes()).hexdigest() != row['image_sha256']:
            raise RuntimeError('old geometry image no longer matches its archived row')
        return torch.from_numpy(image.copy()), row

    def image(self, index, identifier, snr, seed, name, noise_hash):
        if name not in SYSTEM_REFERENCES:
            raise ValueError('unknown geometry system reference')
        return self.legacy.image(index, identifier, snr, seed, name, noise_hash)


def comparison_pairs(config):
    pairs = [(f'geometry204x30__{variant}', f'geometry153x40__{variant}') for variant in config['variants']]
    for geometry in ('153x40', '204x30'):
        pairs.extend([(f'geometry{geometry}__multiscale_conditioned', f'geometry{geometry}__multiscale_no_history'),
                      (f'geometry{geometry}__multiscale_conditioned', f'geometry{geometry}__single_pass'),
                      (f'geometry{geometry}__multiscale_no_history', f'geometry{geometry}__single_pass')])
    pairs.extend((f'geometry204x30__{variant}', reference) for variant in config['variants'] for reference in SYSTEM_REFERENCES)
    return pairs


def validate_geometry_rows(rows, config, base):
    definitions = method_definitions(config)
    names = list(definitions) + list(SYSTEM_REFERENCES)
    snrs, seeds = base['evaluation']['snrs_db'], base['evaluation']['noise_seeds']
    lookup = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in rows}
    expected = {(index, snr, seed, name) for index in range(100) for snr in snrs for seed in seeds for name in names}
    if len(rows) != len(lookup) or set(lookup) != expected:
        raise RuntimeError('matched geometry grid must contain all 21000 unique wireless rows')
    identifiers = []
    for index in range(100):
        source_ids = set()
        for seed in seeds:
            hashes = set()
            for snr in snrs:
                for name in names:
                    row = lookup[index, snr, seed, name]
                    source_ids.add(row['image_id'])
                    hashes.add(row['noise_sha256'])
                    header = 68 if name in ('digital_m8', 'digital_adaptive') else 0
                    if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, 3060 - header):
                        raise RuntimeError('geometry/system N/header/data mismatch')
                    if abs(float(row['total_energy']) - 6120) > 1e-5:
                        raise RuntimeError('geometry/system energy mismatch')
                    if name in definitions:
                        definition = definitions[name]
                        if (row['geometry'] != definition['geometry'] or row['variant'] != definition['variant'] or
                            row['decoder_interface'] != 'continuous_mean' or row['decoded_feature_kind'] != 'continuous_bounded_features'):
                            raise RuntimeError('geometry or continuous interface was relabeled')
                        if int(row['available_image_updates']) != 5000 or int(row['representation_updates']) != 2000:
                            raise RuntimeError('geometry controls have unequal training opportunities')
                        if int(row['raw_bits']) != 8192 or row['coded_bits'] != '':
                            raise RuntimeError('source bits were conflated with the learned physical channel coordinates')
                    if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                        raise RuntimeError('invalid image outcome in geometry comparison')
                    if float(row['severe_distortion']) != int(float(row['LPIPS_excess_from_native']) >= .15):
                        raise RuntimeError('geometry severe-distortion threshold changed')
            if len(hashes) != 1:
                raise RuntimeError('geometry noise realizations are not paired across methods/SNRs')
        if len(source_ids) != 1:
            raise RuntimeError('geometry source-image grouping changed')
        identifiers.extend(source_ids)
    if len(set(identifiers)) != 100:
        raise RuntimeError('geometry source bootstrap population contains duplicate IDs')
    return lookup, names


def geometry_statistics(rows, config, base):
    lookup, names = validate_geometry_rows(rows, config, base)
    draws = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100,
        size=(base['evaluation']['bootstrap_resamples'], 100))
    summary, paired, interactions = [], [], []
    for snrs in [base['evaluation']['primary_snrs_db'], *[[snr] for snr in base['evaluation']['snrs_db']]]:
        label = '+'.join(map(str, snrs))
        values = {name: {metric: np.array([np.mean([float(lookup[index, snr, seed, name][metric])
            for snr in snrs for seed in base['evaluation']['noise_seeds']]) for index in range(100)]) for metric in METRICS} for name in names}
        for name in names:
            frames = [lookup[index, snr, seed, name] for index in range(100) for snr in snrs for seed in base['evaluation']['noise_seeds']]
            row = {'arm': name, 'snrs_db': label, 'source_images': 100,
                   **{metric: float(value.mean()) for metric, value in values[name].items()}}
            for field in ('online_TX_seconds', 'receiver_seconds'):
                times = [float(frame[field]) for frame in frames if frame[field] != '']
                row[field + '_mean'] = float(np.mean(times)) if times else ''
                row[field + '_p95'] = float(np.percentile(times, 95)) if times else ''
            summary.append(row)
        for method, reference in comparison_pairs(config):
            for metric in METRICS:
                delta = values[method][metric] - values[reference][metric]
                low, high = np.percentile(delta[draws].mean(1), (2.5, 97.5))
                paired.append({'method': method, 'control': reference, 'snrs_db': label, 'metric': metric,
                    'delta': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)})
        for metric in METRICS:
            new = values['geometry204x30__multiscale_conditioned'][metric] - values['geometry204x30__multiscale_no_history'][metric]
            old = values['geometry153x40__multiscale_conditioned'][metric] - values['geometry153x40__multiscale_no_history'][metric]
            delta = new - old
            low, high = np.percentile(delta[draws].mean(1), (2.5, 97.5))
            interactions.append({'contrast': '(conditioned-no_history)_204x30_minus_153x40', 'snrs_db': label, 'metric': metric,
                'delta': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summary, paired, interactions
