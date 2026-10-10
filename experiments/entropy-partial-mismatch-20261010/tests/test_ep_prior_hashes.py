"""Small synthetic tests; never decode an actual JPEG or import a neural runtime."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ep_prior_hashes', HERE/'scripts/ep_prior_hashes.py')
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


class Tensor:
    device = 'cpu'
    def __init__(self, array): self.array = array
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


def audit_rows():
    rows = []
    for i in range(1900):
        sid = f'n00000001/ILSVRC2012_val_{i:08d}_n00000001'
        rows.append(dict(source_id=sid, canonical_source_id=f'imagenet-val:{i:08d}',
                         path=ep.IMAGE_ROOT+'/'+sid+'.JPEG', original_file_sha256='a'*64))
    return dict(schema='EP_SOURCE_POPULATION_READONLY_AUDIT_V1', pixel_hash_missing_count=1900,
                pixel_hash_missing_records=rows, original_preprocess=dict(sha256=ep.EXPECTED_PREPROCESS_SHA))


class PriorHashTests(unittest.TestCase):
    def test_exact_1900_identity_and_pool_guard(self):
        a = audit_rows()
        self.assertEqual(len(ep.checked_rows(a)), 1900)
        a['pixel_hash_missing_records'][-1] = dict(a['pixel_hash_missing_records'][0])
        with self.assertRaises(ValueError): ep.checked_rows(a)
        a = audit_rows()
        a['pixel_hash_missing_records'][0]['path'] = '/different-pool/image.JPEG'
        with self.assertRaises(ValueError): ep.checked_rows(a)

    def test_stop_and_explicit_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp)/'STOP'
            request = dict(deadline_unix=20, stop_files=[str(stop)])
            ep.guard(request, now=lambda: 19)
            with self.assertRaises(ValueError): ep.guard(request, now=lambda: 20)
            stop.touch()
            with self.assertRaises(ValueError): ep.guard(request, now=lambda: 19)

    def test_original_preprocess_forward_and_flip_exact_hash(self):
        pixels = np.arange(3*256*256, dtype=np.uint8).reshape(3, 256, 256)
        forward = hashlib.sha256(pixels.tobytes()).hexdigest()
        tensor = Tensor(pixels.astype(np.float32)/127.5-1)
        calls = []
        def checker(row):
            calls.append(row['source_id'])
            return Path('/synthetic.JPEG')
        row = audit_rows()['pixel_hash_missing_records'][0]
        result = ep.process_one(row, lambda path: (tensor, forward), np, file_checker=checker)
        self.assertEqual(result['original_preprocess_pixel_sha256'], forward)
        self.assertEqual(result['horizontal_flip_pixel_sha256'],
                         hashlib.sha256(np.ascontiguousarray(pixels[:, :, ::-1]).tobytes()).hexdigest())
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            ep.process_one(row, lambda path: (tensor, '0'*64), np, file_checker=checker)
        tensor.device = 'cuda:0'
        with self.assertRaises(ValueError):
            ep.process_one(row, lambda path: (tensor, forward), np, file_checker=checker)

    def test_changed_original_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'class/test.JPEG'
            p.parent.mkdir()
            p.write_bytes(b'synthetic non-image bytes')
            row = dict(path=str(p), original_file_sha256=ep.sha(p))
            with patch.object(ep, 'IMAGE_ROOT', temp):
                self.assertEqual(ep.image_file_matches(row), p)
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError): ep.image_file_matches(row)

    def test_write_once_and_bound_json(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'x.json'
            d = ep.save_new(p, dict(synthetic=True))
            self.assertEqual(ep.load_bound(d), dict(synthetic=True))
            with self.assertRaises(FileExistsError): ep.save_new(p, dict(changed=True))
            p.write_text('{}')
            with self.assertRaises(ValueError): ep.load_bound(d)

    def test_prepare_reads_no_declared_images(self):
        # A synthetic module with the exact expected digest substituted only in
        # this unit test; production admission still requires the frozen SHA.
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            module = temp/'preprocess.py'
            module.write_text('raise RuntimeError("must not import during prepare")\n')
            expected = ep.sha(module)
            audit = audit_rows()
            audit['original_preprocess']['sha256'] = expected
            ap = temp/'audit.json'
            ap.write_text(json.dumps(audit))
            with patch.object(ep, 'EXPECTED_PREPROCESS_SHA', expected):
                result = ep.prepare(ap, module, temp/'out', sys.executable, [temp/'STOP'], time.time()+60)
                self.assertEqual(result['image_reads'], 0)
                request = ep.load_bound(result['request'])
                self.assertEqual(request['source_count'], 1900)
                self.assertFalse(request['automatic_retry'])
                with self.assertRaises(ValueError):
                    ep.prepare(ap, module, temp/'out', sys.executable, [temp/'STOP'], time.time()+60)


if __name__ == '__main__':
    unittest.main()
