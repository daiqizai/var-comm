"""Small CPU fixtures for actual R3 evaluation drivers; not alternate production implementations."""

import importlib.util
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
for directory in ('wetok-comm-v2-20260912', 'wetok-innovation-r1', 'wetok-joint-sender-r1', 'wetok-joint-grid-controls-r1', 'wetok-joint-sufficiency-r2'):
    sys.path.insert(0, str(EXPERIMENT.parent / directory / 'src'))
sys.path.insert(0, str(EXPERIMENT / 'src'))

import numpy as np
import torch
from torch import nn
from torch.nn import functional
import yaml

from vector_control.evaluation import all_names
from joint_sender.evaluation_io import digest
from wetok_comm.evaluation import raw_noise
from wetok_comm.native import hard_native_st


def driver(script):
    spec = importlib.util.spec_from_file_location('r3_fixture_' + script, EXPERIMENT / 'scripts' / (script + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    load = lambda path: yaml.safe_load(path.read_text())
    evaluation = load(EXPERIMENT / 'configs/evaluation.yaml')
    config = load(EXPERIMENT / 'configs/study.yaml')
    r2 = load(EXPERIMENT.parent / 'wetok-joint-sufficiency-r2/configs/study.yaml')
    original = load(EXPERIMENT.parent / 'wetok-joint-sender-r1/configs/study.yaml')
    grid = load(EXPERIMENT.parent / 'wetok-joint-grid-controls-r1/configs/study.yaml')
    reference = load(EXPERIMENT.parent / 'wetok-innovation-r1/configs/study.yaml')
    base = load(EXPERIMENT.parent / 'wetok-comm-v2-20260912/configs/study.yaml')
    base['evaluation']['bootstrap_resamples'] = 20
    rows = []
    for index in range(2):
        identifier = f'synthetic_source_{index}'
        for snr in base['evaluation']['snrs_db']:
            for seed in base['evaluation']['noise_seeds']:
                for name in all_names(config, r2, original, grid, reference):
                    fresh = name.startswith('r3__')
                    score = .2 + .01 * index - (.005 if fresh else 0.)
                    rows.append({'image_index': index, 'image_id': identifier, 'snr_db': snr, 'seed': seed, 'arm': name,
                        'total_complex_uses': 3060, 'total_energy': 6120., 'noise_sha256': digest(raw_noise(identifier, seed)),
                        'psnr_db': 20., 'ssim': .8, 'lpips': score, 'dino': .7, 'LPIPS_excess_from_native': score - .1,
                        'severe_distortion': 0, 'online_TX_seconds': '', 'receiver_seconds': '',
                        'r3_quality_origin': 'new_r3_measurement' if fresh else 'sealed_R2_reference',
                        'available_updates': 10000 if fresh or name.startswith('r2__') else 5000,
                        'new_update_opportunity': 10000 if fresh else 5000, 'parent_updates': 7000,
                        'header_uses': 0, 'data_uses': 3060, 'raw_bits': 8192, 'decoder_interface': 'continuous_mean',
                        'communication_parameters': config['expected_communication_parameters'],
                        'transmitted_sha256': f'{name}_{index}_{snr}', 'selected_step': 10000,
                        'selected_global_data_step': 17000, 'checkpoint_sha256': 'selected_checkpoint'})
    return evaluation, config, r2, original, grid, reference, base, rows


class CPUTorch:
    class cuda:
        @staticmethod
        def synchronize():
            return None

    @staticmethod
    def device(unused):
        return torch.device('cpu')

    def __getattr__(self, name):
        return getattr(torch, name)


class NativeFixture:
    def __init__(self, fail_on_second=False):
        self.codec = nn.Identity()
        self.calls = 0
        self.fail_on_second = fail_on_second

    def encode(self, source):
        self.calls += 1
        if self.calls == 2 and self.fail_on_second:
            raise RuntimeError('synthetic interruption')
        return torch.full((len(source), 16, 16, 4), 255, dtype=torch.uint8)

    def decode(self, features):
        return functional.interpolate((features[:, :3] + 1) / 4 + .2, (256, 256))


class ModelFixture(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.encoder = nn.Identity()
        self.weight = nn.Parameter(torch.tensor(.5))

    def transmit(self, source, snrs):
        return torch.ones(len(source), 3060, 2)

    def receive(self, received, snrs):
        logits = (received[:, :1, :1] * self.weight).reshape(-1, 1, 1, 1).expand(-1, 32, 16, 16)
        mean = torch.tanh(logits / 2)
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'receiver_features': mean,
            'states': [functional.adaptive_avg_pool2d(mean, size) for size in (4, 8)],
            'feature_gates': [], 'residual_noise_ratios': []}


def fixture_metrics(source, reconstructions, unused_perceptual=None, unused_dino=None):
    rows = []
    for image in reconstructions:
        mse = float((image[None] - source).square().mean())
        rows.append({'psnr_db': float(-10 * np.log10(max(mse, 1e-12))), 'ssim': .8, 'lpips': mse, 'dino': .7})
    return rows
