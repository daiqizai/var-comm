"""Frozen, physically matched DeepJSCC at its predeclared nearest NN support points."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from .common import EXPERIMENT, PROJECT, WORKSPACE, sha256, verify_sources

SUPPORT_NAME = 'perceptual_deepjscc_fixed_support'


def add_actual_noise(signal, standard_noise, actual_snr):
    if float(actual_snr) not in (5., 6.) or tuple(signal.shape[1:]) != (3060, 2) or standard_noise.shape != signal.shape:
        raise ValueError('support AWGN needs actual SNR 5/6 and the complete paid waveform')
    snrs = signal.new_full((len(signal),), float(actual_snr))
    return signal + standard_noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]


def load_deep_support():
    config = yaml.safe_load((EXPERIMENT / 'configs/deep_support.yaml').read_text())
    if config['actual_to_condition_snr'] != {5.: 4., 6.: 7.} or config['physical_complex_uses'] != 3060:
        raise ValueError('fixed support policy or physical budget changed')
    
    for relative, expected in config['source_hashes'].items():
        local = PROJECT / 'src' / 'cadsd_jscc' / Path(relative).name
        if sha256(local) != expected:
            raise RuntimeError(f'frozen local dependency changed: {local}')
    for name in ('initializer', 'checkpoint'):
        if sha256(WORKSPACE / config[name]) != config[name + '_sha256']:
            raise RuntimeError('frozen Deep weights changed')
    return config


def legacy_model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenDeepSupport:
    def __init__(self, config, device):
        source = (PROJECT / 'src').resolve()
        loaded = sys.modules.get('cadsd_jscc')
        if loaded is not None and Path(loaded.__file__).resolve() != source / 'cadsd_jscc/__init__.py':
            raise RuntimeError('another cadsd_jscc package would shadow the frozen baseline')
        sys.path.insert(0, str(source))
        from cadsd_jscc.exact_budget_strong_jscc import build_exact_budget_model

        initial = torch.load(WORKSPACE / config['initializer'], map_location='cpu', weights_only=True)
        checkpoint = torch.load(WORKSPACE / config['checkpoint'], map_location='cpu', weights_only=True)
        self.model = build_exact_budget_model(initial).eval().requires_grad_(False)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if legacy_model_hash(self.model) != config['legacy_model_sha256']:
            raise RuntimeError('Deep architecture/state differs from the frozen strong control')
        image_size = int(initial['config']['image_size'])
        channels = int(initial['config']['model']['latent_channels'])
        self.layout = (channels, image_size // 16, image_size // 16)
        if int(np.prod(self.layout)) != self.model.native_real_symbols or self.model.active_real_indices.numel() != 6120:
            raise RuntimeError('public fixed latent layout does not implement 3060 complex uses')
        self.config, self.device = config, torch.device(device)
        self.model.to(self.device)

    def condition(self, actual_snr, batch):
        if float(actual_snr) not in self.config['actual_to_condition_snr']:
            raise ValueError('supplement only defines actual SNR 5 and 6')
        return torch.full((batch,), self.config['actual_to_condition_snr'][float(actual_snr)], device=self.device)

    @torch.no_grad()
    def transmit(self, images, actual_snr):
        condition = self.condition(actual_snr, len(images))
        encoded = self.model.encode(images, condition)
        if tuple(encoded.shape[1:]) != self.layout:
            raise RuntimeError('source-dependent latent layout would need additional signalling')
        normalized, unused = self.model.normalize_channel_input(encoded)
        active = normalized.flatten(1).index_select(1, self.model.active_real_indices)
        signal = active.reshape(len(images), 3060, 2)
        if float((signal.square().sum(-1).mean(-1) - 2).abs().max()) > 1e-5:
            raise RuntimeError('Deep supplement violates the actual energy budget')
        return signal

    @torch.no_grad()
    def receive(self, received, actual_snr):
        if tuple(received.shape[1:]) != (3060, 2):
            raise ValueError('Deep receiver requires exactly the paid complex waveform')
        condition = self.condition(actual_snr, len(received))
        flat = received.new_zeros((len(received), self.model.native_real_symbols))
        latent = flat.index_copy(1, self.model.active_real_indices, received.reshape(len(received), 6120))
        return self.model.decode(latent.reshape(len(received), *self.layout), condition).clamp(0, 1)


class DeepSupportReferences:
    def __init__(self, config, base):
        self.config = config
        self.root = PROJECT / 'outputs' / config['reference']
        receipt_path = self.root / 'completion.json'
        if sha256(receipt_path) != config['reference_receipt_sha256']:
            raise RuntimeError('Deep support reference receipt changed')
        self.receipt = json.loads(receipt_path.read_text())
        if (self.receipt['status'] != 'DEEP_SNR_SUPPORT_SANITY_COMPLETE' or
            self.receipt['main_evaluation_receipt_sha256'] != config['original_evaluation_receipt_sha256'] or
            self.receipt['frozen_deep_state'] != config['legacy_model_sha256']):
            raise RuntimeError('Deep supplement belongs to a different frozen model/protocol')
        for relative, expected in self.receipt['source_hashes'].items():
            if sha256(PROJECT / relative) != expected:
                raise RuntimeError('historical fixed-support driver changed')
        for relative in ('protocol.json', 'per_frame.csv'):
            if sha256(self.root / relative) != self.receipt['output_hashes'][relative]:
                raise RuntimeError('Deep support protocol or rows changed')
        with (self.root / 'per_frame.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.rows, self.positions = {}, {}
        positions = {}
        for row in rows:
            index, snr, seed = int(row['image_index']), float(row['snr_db']), int(row['seed'])
            key = index, snr, seed
            if key in self.rows:
                raise RuntimeError('duplicated frozen support row')
            self.rows[key] = row
            self.positions[key] = positions.get(index, 0)
            positions[index] = self.positions[key] + 1
        expected = {(index, snr, seed) for index in range(100) for snr in (5., 6.) for seed in base['evaluation']['noise_seeds']}
        if set(self.rows) != expected or any(value != 6 for value in positions.values()):
            raise RuntimeError('incomplete 600-row Deep support grid')
        self.index, self.images = None, None

    def image(self, index, identifier, actual_snr, seed):
        key = index, float(actual_snr), int(seed)
        row = self.rows[key]
        if row['image_id'] != identifier or float(row['condition_snr_db']) != self.config['actual_to_condition_snr'][float(actual_snr)]:
            raise RuntimeError('source or NN support condition changed')
        if int(row['total_complex_uses']) != 3060 or abs(float(row['data_power']) - 2) > 1e-5:
            raise RuntimeError('support reference changed actual N/E')
        if self.index != index:
            path = self.root / f'images_{index:03d}.npz'
            if sha256(path) != self.receipt['output_hashes'][path.name]:
                raise RuntimeError('Deep support reference pixels changed')
            with np.load(path, allow_pickle=False) as data:
                self.images = data['images'].copy()
            if self.images.shape != (6, 3, 256, 256) or self.images.dtype != np.float32:
                raise RuntimeError('Deep support must retain six actual float32 images per source')
            self.index = index
        return torch.from_numpy(self.images[self.positions[key]].copy()), row


def supplement_statistics(main_rows, support_rows, study, base):
    from .interface_evaluation import METRICS
    from .interface_study import interface_definitions

    main = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in main_rows}
    support = {(int(row['image_index']), float(row['snr_db']), int(row['seed'])): row for row in support_rows}
    seeds = base['evaluation']['noise_seeds']
    expected = {(index, snr, seed) for index in range(100) for snr in (5., 6.) for seed in seeds}
    if len(support) != len(support_rows) or set(support) != expected:
        raise RuntimeError('fixed-support supplement must retain all 600 paired outcomes')
    methods = list(interface_definitions(study))
    for key, row in support.items():
        index, snr, seed = key
        if (row['arm'] != SUPPORT_NAME or float(row['condition_snr_db']) != {5.: 4., 6.: 7.}[snr] or
            int(row['total_complex_uses']) != 3060 or int(row['header_uses']) != 0 or int(row['data_uses']) != 3060 or
            abs(float(row['total_energy']) - 6120) > 1e-5):
            raise RuntimeError('fixed-support rule or physical ledger changed')
        for name in methods:
            candidate = main[index, snr, seed, name]
            if row['image_id'] != candidate['image_id'] or row['noise_sha256'] != candidate['noise_sha256']:
                raise RuntimeError('fixed-support observations are not source/noise paired')
        if not all(np.isfinite(float(row[metric])) for metric in METRICS):
            raise RuntimeError('supplement may not drop nonfinite quality outcomes')
    sampled = np.random.default_rng(base['evaluation']['bootstrap_seed']).integers(100,
        size=(base['evaluation']['bootstrap_resamples'], 100))
    summary, paired = [], []
    for snrs in ([5.], [6.], [5., 6.]):
        label = '+'.join(map(str, snrs))
        values = {}
        for name in methods + [SUPPORT_NAME]:
            values[name] = {}
            for metric in METRICS:
                values[name][metric] = np.array([np.mean([float((support[index, snr, seed] if name == SUPPORT_NAME
                    else main[index, snr, seed, name])[metric]) for snr in snrs for seed in seeds]) for index in range(100)])
            summary.append({'snrs_db': label, 'arm': name, 'source_images': 100,
                **{metric: float(value.mean()) for metric, value in values[name].items()}})
        for name in methods:
            for metric in METRICS:
                delta = values[name][metric] - values[SUPPORT_NAME][metric]
                low, high = np.percentile(delta[sampled].mean(1), [2.5, 97.5])
                paired.append({'snrs_db': label, 'method': name, 'control': SUPPORT_NAME, 'metric': metric,
                    'delta': float(delta.mean()), 'ci_low': float(low), 'ci_high': float(high)})
    return summary, paired
