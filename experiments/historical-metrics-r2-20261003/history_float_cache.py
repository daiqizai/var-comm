"""Preserve exact float reconstruction pixels for future additive metrics."""
import hashlib
import os
from pathlib import Path
import numpy as np
from history_common import sha


def fingerprint(image):
    value = np.asarray(image)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 3:
        raise RuntimeError('Float cache requires CHW float32 RGB')
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise RuntimeError('Float cache rejects invalid RGB')
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class SourceImages:
    def __init__(self, target):
        self.target = np.ascontiguousarray(target).copy()
        self.target_sha = fingerprint(target)
        self.images, self.lookup, self.rows = [], {}, {}

    def add(self, row_id, image):
        if row_id in self.rows:
            raise RuntimeError('Duplicate reconstruction row identity')
        digest = fingerprint(image)
        if digest not in self.lookup:
            self.lookup[digest] = len(self.images)
            self.images.append(np.ascontiguousarray(image).copy())
        self.rows[row_id] = self.lookup[digest]

    def save(self, path, ordered_ids):
        if len(ordered_ids) != len(self.rows) or set(ordered_ids) != set(self.rows):
            raise RuntimeError('Incomplete reconstruction cache rows')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        row_slots = [self.rows[key] for key in ordered_ids]
        image_shas = [fingerprint(x) for x in self.images]
        payload = dict(images=np.stack(self.images), source_rgb=self.target,
                       row_ids=np.asarray(ordered_ids), image_slots=np.asarray(row_slots, dtype=np.int64))
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != set(payload) or any(not np.array_equal(data[k], v) for k, v in payload.items()):
                    raise RuntimeError('Existing float reconstruction cache differs')
        else:
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temp, path)
        return dict(path=str(path), sha256=sha(path), dtype='float32', layout='CHW',
                    reference_sha256=self.target_sha, rows=len(ordered_ids), unique_images=len(self.images),
                    row_ids=list(ordered_ids), image_slots=row_slots, image_sha256=image_shas)


def verify_saved(proof, rows):
    if (sha(proof['path']) != proof['sha256'] or proof.get('dtype') != 'float32'
            or proof.get('layout') != 'CHW' or proof.get('rows') != len(rows)
            or proof.get('row_ids') != [r['history_row_id'] for r in rows]):
        raise RuntimeError('Saved reconstruction cache identity differs')
    with np.load(proof['path'], allow_pickle=False) as data:
        if (data['row_ids'].tolist() != proof['row_ids'] or data['image_slots'].tolist() != proof['image_slots']
                or len(data['images']) != proof['unique_images']
                or fingerprint(data['source_rgb']) != proof['reference_sha256']):
            raise RuntimeError('Saved reconstruction row/target identity differs')
        actual = [fingerprint(v) for v in data['images']]
        if actual != proof['image_sha256']:
            raise RuntimeError('Saved reconstruction pixels differ')
        for row, slot in zip(rows, proof['image_slots']):
            if (actual[slot] != row['history_image_sha256']
                    or proof['reference_sha256'] != row['history_reference_sha256']):
                raise RuntimeError('Saved RGB differs from scored RGB')
    return proof
