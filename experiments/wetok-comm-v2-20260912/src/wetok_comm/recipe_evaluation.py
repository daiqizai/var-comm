"""Recipe-study views and frozen best-before-repair references."""

import copy
import csv
import json

import numpy as np
import torch

from .bit_support import arm_definitions
from .common import PROJECT, sha256, verify_sources
from .evaluation import LegacyReferences


def evaluation_config(repair, base):
    config = copy.deepcopy(base)
    config['arms'] = list(arm_definitions(repair))
    config['outputs'].update(repair['outputs'])
    return config


class RecipeReferences(LegacyReferences):
    def __init__(self, base, repair):
        super().__init__()
        self.previous = PROJECT / 'outputs' / base['outputs']['evaluation'] / 'step_0005000'
        self.previous_receipt = json.loads((self.previous / 'completion.json').read_text())
        if self.previous_receipt['milestone_sha256'] != repair['parent_milestone_sha256']:
            raise RuntimeError('historical best reference belongs to a different training milestone')
        verify_sources(self.previous_receipt['source_hashes'])
        relative = 'per_frame.csv'
        if sha256(self.previous / relative) != self.previous_receipt['output_hashes'][relative]:
            raise RuntimeError('historical best quality rows changed')
        with (self.previous / relative).open() as handle:
            self.best_rows = {(int(row['image_index']), float(row['snr_db']), int(row['seed']), row['arm']): row for row in csv.DictReader(handle)}
        self.best_index, self.best_images = None, None

    def image(self, image_index, image_id, snr, seed, arm, noise_sha256):
        if not arm.startswith('old_selected__'):
            return super().image(image_index, image_id, snr, seed, arm, noise_sha256)
        variant = arm.split('__', 1)[1]
        row = self.best_rows[image_index, float(snr), int(seed), variant]
        if row['image_id'] != image_id or row['noise_sha256'] != noise_sha256:
            raise RuntimeError('historical best source/noise pairing changed')
        if int(row['total_complex_uses']) != 3060 or int(row['header_uses']) != 0:
            raise RuntimeError('historical WeTok physical ledger changed')
        if self.best_index != image_index:
            path = f'images/{image_index:03d}/reconstructions.npz'
            if sha256(self.previous / path) != self.previous_receipt['output_hashes'][path]:
                raise RuntimeError('historical best image bytes changed')
            with np.load(self.previous / path, allow_pickle=False) as data:
                self.best_images = data['images'].copy()
            self.best_index = image_index
        adapted = {**row, 'lpips_alex': row['lpips'], 'dino_cosine': row['dino'], 'raw_bits': 8192}
        return torch.from_numpy(self.best_images[int(row['image_ref'])].copy()), adapted
