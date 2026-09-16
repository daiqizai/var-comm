"""Unified image metrics and read-only, waveform-matched historical references."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from .common import PROJECT, WORKSPACE, assets, sha256
from .training import load_lpips


def raw_noise(identifier, seed):
    key = int.from_bytes(hashlib.sha256(f'{identifier}|{seed}'.encode()).digest()[:8], 'big')
    return np.random.default_rng(key).standard_normal((3060, 2))


def metric_models(device):
    binding = assets()
    if sha256(binding['dino_checkpoint']) != 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9':
        raise RuntimeError('DINO metric weights changed')
    model = torch.hub.load(binding['dino_source'], 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(binding['dino_checkpoint'], map_location='cpu', weights_only=True), strict=True)
    return load_lpips(device), model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def dino_features(model, images):
    images = functional.interpolate(images.float(), (224, 224), mode='bicubic', align_corners=False)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return model.forward_features((images - mean) / std)['x_norm_clstoken'].float()


@torch.no_grad()
def metrics(source, reconstructions, perceptual, dino):
    from pytorch_msssim import ssim

    reference_feature = dino_features(dino, source)
    rows = []
    for start in range(0, len(reconstructions), 4):
        images = torch.stack(reconstructions[start:start + 4]).to(source.device)
        targets = source.expand_as(images)
        mse = (images - targets).square().flatten(1).mean(1)
        psnr = -10 * mse.clamp_min(1e-12).log10()
        structural = ssim(images, targets, data_range=1., size_average=False)
        perceptual_values = perceptual(images * 2 - 1, targets * 2 - 1).flatten()
        features = dino_features(dino, images)
        similarity = functional.cosine_similarity(features, reference_feature.expand_as(features), dim=-1)
        for index in range(len(images)):
            rows.append({'psnr_db': float(psnr[index]), 'ssim': float(structural[index]),
                         'lpips': float(perceptual_values[index]), 'dino': float(similarity[index])})
    return rows


class LegacyReferences:
    def __init__(self):
        binding = assets()
        self.root = Path(binding['reference_evaluation'])
        self.digital = Path(binding['digital_reference'])
        self.transition = PROJECT / 'outputs/VAR-WHOLE-FRAME-PRIOR-001'
        expected = '4b21e10d28a1bd9c730081b5879ca8501f234c58e66395f34d8c14e9b7520d68'
        if sha256(self.root / 'completion.json') != expected:
            raise RuntimeError('legacy same-budget evaluation receipt changed')
        self.receipt = json.loads((self.root / 'completion.json').read_text())
        if sha256(self.root / 'per_frame.csv') != self.receipt['output_hashes']['per_frame.csv']:
            raise RuntimeError('legacy metrics changed')
        with (self.root / 'per_frame.csv').open() as handle:
            self.rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in csv.DictReader(handle)}
        for root, key in ((self.digital, 'old_digital_receipt_sha256'), (self.transition, 'transition_receipt_sha256')):
            if sha256(root / 'completion.json') != self.receipt[key]:
                raise RuntimeError('legacy source-image reference changed')
        self.receipts = {str(root): json.loads((root / 'completion.json').read_text()) for root in (self.root, self.digital, self.transition)}
        self.loaded_index, self.arrays = None, {}

    def image(self, image_index, image_id, snr, seed, arm, noise_sha256):
        row = self.rows[image_index, float(snr), int(seed), arm]
        header, data = (0, 3060) if arm == 'perceptual_deepjscc' else (68, 2992)
        if row['image_id'] != image_id or row['noise_sha256'] != noise_sha256:
            raise RuntimeError('legacy source/noise pairing mismatch')
        if (int(row['total_complex_uses']), int(row['header_uses']), int(row['data_uses'])) != (3060, header, data):
            raise RuntimeError('legacy physical ledger mismatch')
        if self.loaded_index != image_index:
            self.loaded_index, self.arrays = image_index, {}
        kind, position = row['image_ref'].split(':')
        root = {'new': self.root, 'old': self.digital, 'transition': self.transition}[kind]
        if kind not in self.arrays:
            relative = f'images/{image_index:03d}/reconstructions.npz'
            if sha256(root / relative) != self.receipts[str(root)]['output_hashes'][relative]:
                raise RuntimeError('frozen legacy reconstruction bytes changed')
            with np.load(root / relative, allow_pickle=False) as content:
                self.arrays[kind] = content['images'].copy()
        return torch.from_numpy(self.arrays[kind][int(position)].copy()), row
