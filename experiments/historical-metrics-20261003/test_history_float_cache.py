"""Synthetic CPU engineering tests, not image quality evidence."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from history_float_cache import SourceImages, fingerprint, verify_saved


class Tests(unittest.TestCase):
    def test_dedup_exact_rows_target_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            target = np.full((3,4,4), .7, np.float32)
            rgb = np.full((3,4,4), .2, np.float32)
            cache = SourceImages(target)
            cache.add('a', rgb); cache.add('b', rgb)
            path = Path(folder)/'source.npz'
            receipt = cache.save(path, ['b','a'])
            rows = [dict(history_row_id=k,history_image_sha256=fingerprint(rgb),
                         history_reference_sha256=fingerprint(target)) for k in ['b','a']]
            self.assertEqual(verify_saved(receipt, rows), receipt)
            self.assertEqual(receipt['unique_images'], 1)
            self.assertEqual(cache.save(path, ['b','a']), receipt)
            with self.assertRaises(RuntimeError): cache.save(path, ['a','b'])
            with path.open('ab') as stream: stream.write(b'corruption')
            with self.assertRaises(RuntimeError): verify_saved(receipt, rows)

    def test_invalid_rows_and_pixels_rejected(self):
        rgb = np.zeros((3,4,4), np.float32)
        cache = SourceImages(rgb); cache.add('a',rgb)
        with self.assertRaises(RuntimeError): cache.add('a',rgb)
        with self.assertRaises(RuntimeError): cache.save('unused.npz',['b'])
        with self.assertRaises(RuntimeError): SourceImages(rgb.astype(np.uint8))


if __name__ == '__main__': unittest.main()
