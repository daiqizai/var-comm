"""Read-only ImageNet shards and fixed-header support for learned prefix communication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .progressive import decode_header, header_bits
from .scale_channel import encode_packet
from .study import sha256

LEGACY = Path('/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc')
IMAGE_CACHE = LEGACY / 'outputs/cache/CACHE-VAR-DECODER-FT-IMAGENET20K-CAL1K-001'
MANIFEST_SHA = '49f344e8cf72960b7c164af96e05c203e24f4ae1cbfb8be9f3f02f8842b8e8b8'


def read_image_population(population, verify=True):
    if sha256(IMAGE_CACHE / 'manifest.json') != MANIFEST_SHA:
        raise RuntimeError('historical image manifest changed')
    manifest = json.loads((IMAGE_CACHE / 'manifest.json').read_text())
    selected = next(entry for entry in manifest['populations'] if entry['name'] == population)
    images = torch.empty((selected['count'], 3, 256, 256), dtype=torch.uint8)
    labels = torch.empty(selected['count'], dtype=torch.long)
    identifiers, bindings, offset = [], [], 0
    for descriptor in selected['shards']:
        path = IMAGE_CACHE / descriptor['path']
        if verify and sha256(path) != descriptor['sha256']:
            raise RuntimeError('historical image shard changed')
        shard = torch.load(path, map_location='cpu', weights_only=True)
        count = len(shard['targets_u8'])
        images[offset:offset + count].copy_(shard['targets_u8'])
        labels[offset:offset + count].copy_(shard['labels'])
        identifiers.extend(shard['image_ids'])
        bindings.extend({'shard': descriptor['path'], 'index_in_shard': index, 'image_id': shard['image_ids'][index],
                         'class_index': int(shard['labels'][index])} for index in range(count))
        offset += count
    if offset != selected['count']:
        raise RuntimeError('image population incomplete')
    return images, labels, identifiers, bindings


class HeaderProtocol:
    def __init__(self):
        self.transmitted = np.stack([encode_packet(header_bits(label, 8), 68)['symbols'] for label in range(1000)])

    def decode(self, labels, snrs, standard_noise):
        labels = np.asarray(labels, dtype=np.int64)
        snrs = np.asarray(snrs, dtype=np.float64)
        observations = self.transmitted[labels] + np.asarray(standard_noise, dtype=np.float64) / np.sqrt(10 ** (snrs[:, None, None] / 10))
        decoded, usable, false_acceptance = [], [], []
        for label, snr, observed in zip(labels, snrs, observations):
            header = decode_header(observed, float(snr))
            valid = header['accepted'] and header['mode'] == 8
            decoded.append(header['label'] if valid else 0)
            usable.append(valid)
            false_acceptance.append(valid and header['label'] != int(label))
        return np.asarray(decoded), np.asarray(usable), np.asarray(false_acceptance)
