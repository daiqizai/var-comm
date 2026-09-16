"""Small CPU-only stand-ins for the visual and communication models, not alternative production paths."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from test_quality_contracts import EXPERIMENT
from wetok_comm.native import hard_native_st


def driver(script):
    spec = importlib.util.spec_from_file_location('r2_fixture_' + script, EXPERIMENT / 'scripts' / (script + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
