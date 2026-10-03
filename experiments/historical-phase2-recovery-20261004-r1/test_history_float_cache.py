"""Synthetic CPU engineering tests, not image quality evidence."""
from pathlib import Path
from copy import deepcopy
import hashlib
import tempfile
import unittest
import numpy as np
from history_float_cache import SourceImages, fingerprint, scored_fingerprint, verify_saved
from history_common import sha


class Tests(unittest.TestCase):
    def test_dedup_exact_rows_target_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            target = np.full((3,256,256), .7, np.float32)
            rgb = np.full((3,256,256), .2, np.float32)
            cache = SourceImages(target)
            cache.add('a', rgb); cache.add('b', rgb)
            path = Path(folder)/'source.npz'
            receipt = cache.save(path, ['b','a'])
            rows = [dict(history_row_id=k,history_image_sha256=scored_fingerprint(rgb),
                         history_reference_sha256=scored_fingerprint(target)) for k in ['b','a']]
            before = deepcopy((receipt, rows))
            self.assertEqual(verify_saved(receipt, rows), receipt)
            self.assertEqual((receipt, rows), before)
            self.assertEqual(receipt['reference_sha256'], hashlib.sha256(target.tobytes()).hexdigest())
            self.assertEqual(receipt['image_sha256'], [hashlib.sha256(rgb.tobytes()).hexdigest()])
            self.assertNotEqual(receipt['image_sha256'][0], rows[0]['history_image_sha256'])
            self.assertEqual(receipt['unique_images'], 1)
            self.assertEqual(cache.save(path, ['b','a']), receipt)
            with self.assertRaises(RuntimeError): cache.save(path, ['a','b'])
            with path.open('ab') as stream: stream.write(b'corruption')
            with self.assertRaises(RuntimeError): verify_saved(receipt, rows)

    def fixture(self, folder):
        target = np.full((3,256,256), .7, np.float32)
        rgb = np.full((3,256,256), .2, np.float32)
        cache = SourceImages(target); cache.add('a', rgb)
        path = Path(folder)/'source.npz'
        proof = cache.save(path, ['a'])
        # Literal frozen replay contract, not the helper under test.
        digest = lambda x: hashlib.sha256(b'float32:3,256,256:RGB\0'+x.tobytes()).hexdigest()
        rows = [dict(history_row_id='a',history_image_sha256=digest(rgb),
                     history_reference_sha256=digest(target))]
        return proof, rows

    def test_exact_scoring_domain_and_registered_shape(self):
        rgb = np.zeros((3,256,256), np.float32)
        self.assertEqual(scored_fingerprint(rgb),hashlib.sha256(b'float32:3,256,256:RGB\0'+rgb.tobytes()).hexdigest())
        with self.assertRaises(RuntimeError): scored_fingerprint(np.zeros((3,4,4),np.float32))

    def test_row_hash_tampering_and_raw_hash_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            proof, rows = self.fixture(folder)
            verify_saved(proof, rows)
            for field, replacement in [('history_image_sha256','0'*64),
                    ('history_reference_sha256','0'*64),
                    ('history_image_sha256',proof['image_sha256'][0]),
                    ('history_reference_sha256',proof['reference_sha256'])]:
                altered = deepcopy(rows); altered[0][field] = replacement
                with self.assertRaises(RuntimeError): verify_saved(proof, altered)

    def test_pixels_cannot_change_even_after_container_hash_is_updated(self):
        for key in ('images','source_rgb'):
            for update_raw_receipt in (False,True):
                with self.subTest(key=key,update_raw_receipt=update_raw_receipt), tempfile.TemporaryDirectory() as folder:
                    proof, rows = self.fixture(folder)
                    with np.load(proof['path'], allow_pickle=False) as archive:
                        payload = {name: archive[name].copy() for name in archive.files}
                    payload[key].flat[0] += np.float32(.01)
                    np.savez_compressed(proof['path'], **payload)
                    proof['sha256'] = sha(proof['path'])
                    if update_raw_receipt:
                        proof['image_sha256'] = [fingerprint(x) for x in payload['images']]
                        proof['reference_sha256'] = fingerprint(payload['source_rgb'])
                    with self.assertRaises(RuntimeError): verify_saved(proof, rows)

    def test_slot_coverage_and_bounds_rejected(self):
        for slots in ([], [-1], [1], [True]):
            with self.subTest(slots=slots), tempfile.TemporaryDirectory() as folder:
                proof, rows = self.fixture(folder)
                proof['image_slots'] = slots
                with self.assertRaises(RuntimeError): verify_saved(proof, rows)

    def test_invalid_rows_and_pixels_rejected(self):
        rgb = np.zeros((3,4,4), np.float32)
        cache = SourceImages(rgb); cache.add('a',rgb)
        with self.assertRaises(RuntimeError): cache.add('a',rgb)
        with self.assertRaises(RuntimeError): cache.save('unused.npz',['b'])
        with self.assertRaises(RuntimeError): SourceImages(rgb.astype(np.uint8))


if __name__ == '__main__': unittest.main()
