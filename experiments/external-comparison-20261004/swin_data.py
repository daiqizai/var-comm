"""Read only the original ImageNet RGB cache used to train P1024."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = '49f344e8cf72960b7c164af96e05c203e24f4ae1cbfb8be9f3f02f8842b8e8b8'
COUNTS = {'train': 20000, 'calibration': 1000}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_population(role, ids, source_rows, reference):
    if role not in COUNTS or len(ids) != COUNTS[role] or len(set(ids)) != len(ids):
        raise RuntimeError('Only the complete original train/calibration populations are admitted')
    id_key = 'train_ids' if role == 'train' else 'calibration_ids'
    if ids != reference[id_key]:
        raise RuntimeError('Swin source order differs from the original P1024 population')
    if source_rows != reference['source_image_bindings'][role]:
        raise RuntimeError('Swin source shard/index/class mapping differs from P1024')
    if set(reference['train_ids']) & set(reference['calibration_ids']):
        raise RuntimeError('Original train/calibration identities overlap')


class RGBPopulation:
    def __init__(self, cache_root, p1024_registration, role):
        import torch
        if role not in COUNTS:
            raise ValueError('Training loader never opens development or holdout')
        cache_root = Path(cache_root).resolve()
        registration = Path(p1024_registration).resolve()
        manifest_path = cache_root / 'manifest.json'
        if sha(manifest_path) != MANIFEST_SHA256:
            raise RuntimeError('Original RGB manifest changed')
        reference, manifest = read(registration), read(manifest_path)
        populations = [p for p in manifest['populations'] if p['name'] == role]
        if len(populations) != 1 or populations[0]['count'] != COUNTS[role]:
            raise RuntimeError('Original full population is missing')
        self.images = torch.empty((COUNTS[role], 3, 256, 256), dtype=torch.uint8)
        self.ids, self.source_rows, self.labels = [], [], []
        self.bindings = {str(manifest_path): MANIFEST_SHA256, str(registration): sha(registration)}
        offset = 0
        for descriptor in populations[0]['shards']:
            path = (cache_root / descriptor['path']).resolve()
            if not path.is_relative_to(cache_root) or sha(path) != descriptor['sha256']:
                raise RuntimeError('Original RGB shard path/hash changed')
            # These are explicitly hash-bound trusted local historical tensors.
            shard = torch.load(path, map_location='cpu')
            pixels, labels, ids = shard['targets_u8'], shard['labels'], shard['image_ids']
            count = len(ids)
            if pixels.dtype != torch.uint8 or tuple(pixels.shape) != (count, 3, 256, 256):
                raise RuntimeError('Original RGB dtype/layout differs')
            if len(labels) != count or offset + count > COUNTS[role]:
                raise RuntimeError('Original RGB shard count differs')
            self.images[offset:offset + count].copy_(pixels)
            self.ids.extend(ids)
            self.labels.extend(int(v) for v in labels)
            self.source_rows.extend(dict(shard=descriptor['path'], index_in_shard=i,
                image_id=ids[i], class_index=int(labels[i])) for i in range(count))
            self.bindings[str(path)] = descriptor['sha256']
            offset += count
            del shard, pixels, labels
        validate_population(role, self.ids, self.source_rows, reference)
        if offset != COUNTS[role]:
            raise RuntimeError('Incomplete original RGB population')
        self.role = role

    def __len__(self):
        return len(self.ids)

    def batch(self, indices, device):
        return self.images[indices].to(device=device, dtype=__import__('torch').float32).div(255)

