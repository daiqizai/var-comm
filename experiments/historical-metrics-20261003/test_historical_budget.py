"""Synthetic CPU engineering checks; no real-weight quality claims."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import historical_budget as h


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def fixture(root, c=False):
    folder = root/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'/('C_priority_selected_v1/selected_grid' if c else 'continuous_grid_v1')
    pixels = np.full((3, 256, 256), 127, np.uint8)
    pre = h.hashlib.sha256(pixels.tobytes()).hexdigest()
    methods = [f'H6-V_N4084_seed{i}' for i in range(12)] + ['P4084_seed1', 'P4084_seed2'] if c else ['P2048', 'P3060', 'P4084']
    metas = {}
    for i, method in enumerate(methods):
        n = 4084 if c else int(method[1:])
        selected = {'step': 10000+i, 'checkpoint_sha256': str(i)*64}
        metas[method] = {'N': n, 'selected': selected if c else 'path', 'step': 10000+i,
                         'checkpoint_sha256': str(i)*64, 'training_seed': i,
                         'kind': 'hybrid' if c and i < 12 else 'continuous'}
    context = {'models': metas, 'selected': {m[1:]: v for m, v in metas.items()},
               'decoder_sha256': 'd'*64, 'bindings': {}, 'precision': {}}
    reg = {'synthetic': False, 'context': context, 'context_sha256': h.identity(context),
           'sources': {f'image{i}': pre for i in range(100)}, 'snrs': list(h.SNRS), 'seeds': list(h.SEEDS)}
    write(folder/'registration.json', reg)
    regsha = h.sha256(folder/'registration.json')
    write(folder/'qualification.json', {'status': 'REAL_C_SELECTED_CACHE_ONLINE_REPLAY_PASS' if c else 'REAL_SELECTED_CONTINUOUS_REPLAY_PASS',
          'synthetic': False, 'registration_sha256': regsha})
    all_rows, seals = [], {}
    for i in range(100):
        for mi, method in enumerate(methods):
            meta = metas[method]; rows, bases = [], []
            for snr in h.SNRS:
                for seed in h.SEEDS:
                    row = {'method': method, 'source_id': f'image{i}', 'source_index': i,
                           'preprocessing_id': pre, 'population': 'development',
                           'context_sha256': reg['context_sha256'], 'snr_db': snr, 'noise_seed': seed,
                           'psnr_db': 30., 'lpips_alex': .1, 'dino_cosine': .9, 'mse': .001,
                           'header_ok': False if meta['kind'] == 'hybrid' else 'not_applicable',
                           'body_crc_ok': False if meta['kind'] == 'hybrid' else 'not_applicable',
                           'N': meta['N'], 'waveform_sha256': 'w', 'observation_sha256': 'o'}
                    rows.append(row)
                    if meta['kind'] == 'hybrid':
                        for condition in ('B_RX', 'C_RX'):
                            bases.append({k: row[k] for k in ('method', 'source_id', 'snr_db', 'noise_seed',
                                          'psnr_db', 'lpips_alex', 'dino_cosine', 'mse', 'header_ok', 'body_crc_ok')})
                            bases[-1].update(condition=condition, enhancement_removed=True, N_paid=meta['N'])
            suffix = f'{mi:02d}' if c else method[1:]
            name = f'{i:04d}_{suffix}.json'; path = folder/'cells'/name
            write(path, {'registration_sha256': regsha, 'source_id': f'image{i}', 'method': method,
                         'rows': rows, 'base_rows': bases})
            seals[name] = h.sha256(path); all_rows.extend(rows)
    with (folder/'per_frame.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    done = {'status': 'REAL_C_SELECTED_GRID_COMPLETE' if c else 'REAL_CONTINUOUS_GRID_COMPLETE',
            'synthetic': False, 'registration_sha256': regsha, 'frame_rows': len(all_rows),
            'files': {'per_frame.csv': h.sha256(folder/'per_frame.csv')}, 'cell_sha256': seals}
    write(folder/'completion.json', done)
    return folder, pixels


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.cont, cls.pixels = fixture(cls.root)
        cls.cfolder, _ = fixture(cls.root, c=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_scopes_and_model_identity(self):
        a = h.create_adapter(self.root, 'CONTINUOUS_GRID')
        b = h.create_adapter(self.root, 'C_SELECTED_GRID')
        c = h.create_adapter(self.root, 'C_SELECTED_BASELINES')
        self.assertEqual([len(x.rows) for x in (a, b, c)], [4500, 21000, 36000])
        self.assertEqual([len(x.expected_rows(0)) for x in (a, b, c)], [45, 210, 360])
        self.assertEqual(a.metadata(a.rows[0])['selected_step'], 10000)
        self.assertNotEqual(b.metadata(b.rows[180])['checkpoint_sha256'], b.metadata(b.rows[195])['checkpoint_sha256'])
        self.assertFalse(b.metadata(b.rows[0])['classification_main_eligible'])
        self.assertTrue(b.metadata(b.rows[180])['classification_main_eligible'])
        self.assertTrue(c.metadata(c.rows[0])['reference_only'])
        self.assertIn('/base_rows/0', c.metadata(c.rows[0])['original_location']['pointer'])
        self.assertNotEqual(c.metadata(c.rows[0])['row_id'], c.metadata(c.rows[1])['row_id'])
        original = a.expected_rows(0); original[0]['method'] = 'changed'
        self.assertEqual(a.expected_rows(0)[0]['method'], 'P2048')

    def test_original_row_parity_rejects_missing_and_changed_fields(self):
        row = {'psnr_db': 30., 'lpips_alex': .1, 'dino_cosine': .9, 'mse': .001,
               'waveform_sha256': 'a', 'header_ok': False}
        self.assertTrue(h.parity_check(row, dict(row))['passed'])
        for key, value in [('waveform_sha256', 'b'), ('header_ok', True), ('lpips_alex', .2), ('mse', float('nan'))]:
            changed = dict(row, **{key: value})
            with self.assertRaises(h.ReplayMismatch):
                h.parity_check(row, changed)
        missing = dict(row); del missing['dino_cosine']
        with self.assertRaises(h.ReplayMismatch):
            h.parity_check(row, missing)

    def test_hash_and_registration_fail_closed(self):
        path = self.cont/'cells/0000_2048.json'; old = path.read_bytes()
        try:
            path.write_bytes(old+b' ')
            with self.assertRaisesRegex(h.ReplayMismatch, 'hash differs'):
                h.create_adapter(self.root, 'CONTINUOUS_GRID')
        finally:
            path.write_bytes(old)
        path = self.cont/'completion.json'; old = path.read_bytes()
        try:
            bad = h.read(path); bad['status'] = 'RUNNING'; write(path, bad)
            with self.assertRaisesRegex(h.ReplayMismatch, 'completed real'):
                h.create_adapter(self.root, 'CONTINUOUS_GRID')
        finally:
            path.write_bytes(old)

    def test_baseline_order_cannot_be_changed_even_if_cell_resealed(self):
        path = self.cfolder/'cells/0000_00.json'; donepath = self.cfolder/'completion.json'
        old, olddone = path.read_bytes(), donepath.read_bytes()
        try:
            cell = h.read(path); cell['base_rows'][0], cell['base_rows'][1] = cell['base_rows'][1], cell['base_rows'][0]
            write(path, cell); done = h.read(donepath); done['cell_sha256'][path.name] = h.sha256(path); write(donepath, done)
            with self.assertRaisesRegex(h.ReplayMismatch, 'baseline grid'):
                h.create_adapter(self.root, 'C_SELECTED_BASELINES')
        finally:
            path.write_bytes(old); donepath.write_bytes(olddone)

    def test_float_range_and_no_unqualified_model_shortcut(self):
        h.rgb(np.zeros((3, 256, 256), np.float32))
        for a in (np.zeros((3, 256, 256), np.uint8), np.zeros((256, 256, 3), np.float32), np.full((3, 256, 256), 2., np.float32)):
            with self.assertRaises(h.ReplayMismatch):
                h.rgb(a)
        a = h.create_adapter(self.root, 'CONTINUOUS_GRID', loaded={})
        with self.assertRaisesRegex(h.ReplayMismatch, 'Unqualified'):
            a.setup()

    def test_original_batching_and_iterator_order_with_fake_native(self):
        a = h.create_adapter(self.root, 'CONTINUOUS_GRID')
        calls, batches = [], []
        image = np.full((3, 256, 256), .5, np.float32)
        target = self.pixels.astype(np.float32)/255
        mse = float(np.mean(np.square(image-target), dtype=np.float64))
        def execute(record, cell, snr, seed, *args):
            calls.append((cell.N, snr, seed))
            return image, {'ledger': {'N': cell.N}, 'rx': {'header_ok': 'not_applicable', 'body_crc_ok': 'not_applicable'},
                           'waveform_sha256': 'w', 'observation_sha256': 'o'}
        def quality(reference, images, *args):
            batches.append(len(images))
            return ([{'psnr_db': 30., 'lpips_alex': .1, 'dino_cosine': .9} for _ in images], None, None)
        # Synthetic native fixture explicitly supplies its own expected MSE; no
        # real archived values or bound files are modified.
        for row in a.by_source[0]:
            oldid = h.identity(row); location = a._locations.pop(oldid)
            row['mse'] = str(mse); a._locations[h.identity(row)] = location
        a.records[0].update(pixels=self.pixels, class_index=0)
        a.native = SimpleNamespace(execute=execute, quality_metrics=quality, Cell=lambda family, n: SimpleNamespace(N=n))
        a.torch = SimpleNamespace(no_grad=nullcontext); a.safe = SimpleNamespace(check=lambda: None)
        a.models = {m: object() for m in a.methods}; a.ready = True
        a.vae = a.var = a.decoder = a.device = a.lp = a.dino = None
        actual = list(a.iterate_source(0))
        self.assertEqual([x[0] for x in actual], a.expected_rows(0))
        self.assertEqual(batches, [15, 15, 15])
        self.assertEqual(len(calls), 45)
        self.assertTrue(all(x[3]['passed'] for x in actual))
        np.testing.assert_array_equal(actual[0][1], image)
        np.testing.assert_array_equal(actual[0][2], target)


if __name__ == '__main__':
    unittest.main()
