"""Preserve exact float reconstruction pixels for future additive metrics."""
import hashlib
import os
from pathlib import Path
import numpy as np
from history_common import sha


SCORING_DOMAIN = b'float32:3,256,256:RGB\0'


def _pixels(image):
    value = np.asarray(image)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 3:
        raise RuntimeError('Float cache requires CHW float32 RGB')
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise RuntimeError('Float cache rejects invalid RGB')
    return np.ascontiguousarray(value)


def fingerprint(image):
    """Original cache-receipt hash; preserve raw float-byte identity."""
    return hashlib.sha256(_pixels(image).tobytes()).hexdigest()


def scored_fingerprint(image):
    """Exact unified replay RGB hash, distinct from the cache receipt hash.

    Kept local so CPU-only inherited-cache validation never imports the replay
    engine or constructs a model. The domain is from replay.rgb_fingerprint.
    """
    value = _pixels(image)
    if value.shape != (3, 256, 256):
        raise RuntimeError('Scored RGB requires registered [3,256,256] pixels')
    return hashlib.sha256(SCORING_DOMAIN + value.tobytes()).hexdigest()


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
        slots = proof['image_slots']
        if (len(slots) != len(rows) or type(proof['unique_images']) is not int
                or proof['unique_images'] < 1
                or any(type(slot) is not int or not 0 <= slot < proof['unique_images'] for slot in slots)):
            raise RuntimeError('Saved reconstruction slot coverage differs')
        # Each NpzFile lookup decompresses its member again. Keep the actual
        # pixels once for both historical receipt and scientific row checks.
        images, source = data['images'], data['source_rgb']
        if (data['row_ids'].tolist() != proof['row_ids'] or data['image_slots'].tolist() != proof['image_slots']
                or len(images) != proof['unique_images']
                or fingerprint(source) != proof['reference_sha256']):
            raise RuntimeError('Saved reconstruction row/target identity differs')
        actual = [fingerprint(v) for v in images]
        if actual != proof['image_sha256']:
            raise RuntimeError('Saved reconstruction pixels differ')
        # R2 receipts used raw bytes, while scientific rows used the unified
        # replay domain above. Verify both definitions from actual pixels;
        # never rewrite either historical hash or accept one as the other.
        scored = [scored_fingerprint(v) for v in images]
        reference = scored_fingerprint(source)
        for row, slot in zip(rows, slots):
            if (scored[slot] != row['history_image_sha256']
                    or reference != row['history_reference_sha256']):
                raise RuntimeError('Saved RGB differs from scored RGB')
    return proof
