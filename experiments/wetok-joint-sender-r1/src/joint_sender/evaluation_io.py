"""Small auditable IO helpers; identical frozen replays reuse existing archives rather than copying them."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch


def digest(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class SourceImages:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metrics_images, self.metrics_index = [], {}
        self.new_images, self.new_index = [], {}
        self.reused_frozen = 0

    def add(self, image, previous=None):
        image = image.detach().cpu().contiguous()
        image_hash = digest(image)
        if image_hash not in self.metrics_index:
            self.metrics_index[image_hash] = len(self.metrics_images)
            self.metrics_images.append(image)
        if previous is not None and image_hash == previous['image_sha256']:
            self.reused_frozen += 1
            return {'image_store': 'verified_existing_reference', 'image_archive': previous['image_archive'],
                    'image_ref': int(previous['image_ref']), 'image_sha256': image_hash}
        if image_hash not in self.new_index:
            self.new_index[image_hash] = len(self.new_images)
            self.new_images.append(image)
        return {'image_store': 'new_measured_output', 'image_archive': str(self.directory / 'reconstructions.npz'),
                'image_ref': self.new_index[image_hash], 'image_sha256': image_hash}

    def save(self):
        if not self.new_images:
            raise RuntimeError('Joint outputs are missing from the source-image archive')
        np.savez_compressed(self.directory / 'reconstructions.npz', images=torch.stack(self.new_images).numpy())
