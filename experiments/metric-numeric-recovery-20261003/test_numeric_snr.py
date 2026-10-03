"""CPU engineering checks using the pinned frozen replay; no model inference."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numeric_snr as n


def load_replay():
    candidates = [parent / relative for parent in Path(__file__).resolve().parents
        for relative in ('experiments/unified-metrics-20261002/replay.py',
                         '.research/perf_r5_inputs/replay.py',
                         '.research/metrics_20261002/source/replay.py')]
    path = next(p for p in candidates if p.is_file())
    name = '_numeric_snr_original_replay_test'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def records():
    return [dict(image_id=f'ENGINEERING_SOURCE_{i:03}', preprocessing_id='ENGINEERING_PREPROCESS', class_index=i)
            for i in range(100)]


def actual_grid(replay):
    selected = [dict(status='SELECTED', N=N, phy_family=phy, snr_db=float(snr), projection='g4_c8')
                for N in (512, 1024) for phy in ('QPSK', '16QAM') for snr in n.SNR_VALUES]
    rows = [dict(source_id=records()[i]['image_id'], source_index=str(i),
                 preprocessing_id='ENGINEERING_PREPROCESS', N=str(choice['N']),
                 phy_family=choice['phy_family'], snr_db=str(choice['snr_db']),
                 noise_seed=str(seed), projection=choice['projection'], control=control,
                 stage='actual_link', oracle_side_information='False', decoder_id='Dc',
                 psnr_db='20.25', lpips_alex='0.2', dino_cosine='0.75',
                 measurement_wave_sha256='a' * 64)
            for i in range(100) for choice in selected for seed in replay.DEV_SEEDS
            for control in replay.M2_CONTROLS]
    return rows, selected


class NumericSNRTests(unittest.TestCase):
    def test_only_ten_registered_literals_and_no_coercion(self):
        for value in n.SNR_VALUES:
            for raw in (str(value), str(value) + '.0'):
                parsed = n.IntSNR(raw)
                self.assertIsInstance(parsed, str)
                self.assertEqual((int(parsed), str(parsed), float(parsed)), (value, raw, float(value)))
                self.assertEqual(parsed, raw)
                self.assertEqual(hash(parsed), hash(raw))
                self.assertEqual(json.dumps(parsed), json.dumps(raw))
        for value in ('1.5', 'NaN', 'inf', '-inf', 'clean', '', '0', '2', '1.00', '1e0',
                      ' 1.0', '1.0 ', '+1', '-1', '01', '1_0', 1, 1., True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                n.IntSNR(value)

    def test_hashes_ids_csv_and_copy_keep_original_text(self):
        r = load_replay()
        raw = dict(source_id='source', source_index='0', N='512', snr_db='1.0', noise_seed='2001',
                   control='VAR_GUIDED', projection='g4_c8', psnr_db='20.25')
        prepared = n.adapt_prepare_rows(lambda self, study, rows: rows)(None, 'M2_ACTUAL', [raw])[0]
        self.assertIs(type(raw['snr_db']), str)
        self.assertIs(type(prepared['snr_db']), n.IntSNR)
        self.assertEqual(r.source_row_hash(raw), r.source_row_hash(prepared))
        self.assertEqual(r.row_id('M2_ACTUAL', raw), r.row_id('M2_ACTUAL', prepared))
        self.assertEqual(r.row_key('M2_ACTUAL', raw), r.row_key('M2_ACTUAL', prepared))
        self.assertEqual(json.dumps(raw), json.dumps(prepared))
        for row in (copy.copy(prepared), copy.deepcopy(prepared)):
            self.assertEqual(int(row['snr_db']), 1)
            self.assertEqual(r.source_row_hash(raw), r.source_row_hash(row))
        def csv_bytes(row):
            handle = io.StringIO(newline='')
            writer = csv.DictWriter(handle, list(row)); writer.writeheader(); writer.writerow(row)
            return handle.getvalue()
        self.assertEqual(csv_bytes(raw), csv_bytes(prepared))

    def test_complete_actual_grid_original_validators_all_fields_unchanged(self):
        r = load_replay(); rows, selected = actual_grid(r)
        self.assertEqual(len(rows), 30000)
        engine = r.ReplayEngine(Path('.'), {}, {'records': records()}, synthetic=False)
        with self.assertRaises(ValueError):
            engine._validate_rows('M2_ACTUAL', rows)
        untouched = (r.parity_check, r.row_id, r.source_row_hash, r.norm,
                     r.ReplayEngine._validate_rows, r._M2Adapter.validate_rows, r._M2Adapter.iterate_source)
        policy = dict(choices=selected)
        policy_json = json.dumps(policy, sort_keys=True)
        before = [(r.source_row_hash(row), r.row_id('M2_ACTUAL', row)) for row in rows]
        proof = n.install(r)
        prepared = engine._prepare_rows('M2_ACTUAL', rows)
        engine._validate_rows('M2_ACTUAL', prepared)
        adapter = object.__new__(r._M2Adapter)
        adapter.study = 'M2_ACTUAL'; adapter.actual_policy = policy
        adapter.actual_receipt = dict(branch='ACTUAL_LINK_EVALUATED')
        adapter.validate_rows(prepared)
        self.assertEqual(untouched, (r.parity_check, r.row_id, r.source_row_hash, r.norm,
            r.ReplayEngine._validate_rows, r._M2Adapter.validate_rows, r._M2Adapter.iterate_source))
        self.assertEqual(json.dumps(policy, sort_keys=True), policy_json)
        self.assertEqual(before, [(r.source_row_hash(row), r.row_id('M2_ACTUAL', row)) for row in prepared])
        self.assertEqual(json.dumps(rows), json.dumps(prepared))
        self.assertTrue(all(type(row['snr_db']) is str for row in rows))
        self.assertEqual(proof['scope'], ['M2_ACTUAL'])
        # Exercise original CSV reading, preparation, validation, grouping, and
        # metadata without loading receiver models. The adapter's real original
        # grid validator above is retained; only its GPU constructor is skipped.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); table = folder / 'actual.csv'
            with table.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
            table_hash = n.sha(table)
            engine.layouts['M2_ACTUAL'] = r.Layout(folder, folder, folder, table)
            engine.adapters['M2_ACTUAL'] = adapter
            engine.setup(['M2_ACTUAL'])
            self.assertEqual(n.sha(table), table_hash)
            self.assertEqual(sorted(engine.by_source['M2_ACTUAL']), list(range(100)))
            self.assertEqual([len(engine.by_source['M2_ACTUAL'][i]) for i in range(100)], [300] * 100)
            self.assertEqual(before, [(r.source_row_hash(row), r.row_id('M2_ACTUAL', row))
                                     for row in engine.rows['M2_ACTUAL']])
            for row in engine.by_source['M2_ACTUAL'][59]:
                self.assertEqual(engine.metadata(row, 'M2_ACTUAL')['replay_row_id'], r.row_id('M2_ACTUAL', row))
        with self.assertRaises(r.ReplayMismatch): adapter.validate_rows(prepared[:-1])
        with self.assertRaises(r.ReplayMismatch): engine._validate_rows('M2_ACTUAL', prepared + [prepared[0]])
        for field, value in (('source_index', '100'), ('noise_seed', '9999'),
                             ('stage', 'wrong'), ('oracle_side_information', 'True')):
            changed = [dict(prepared[0], **{field: value})]
            with self.assertRaises(r.ReplayMismatch): engine._validate_rows('M2_ACTUAL', changed)

    def test_old_studies_and_original_preparation_are_untouched(self):
        calls = []
        def original(self, study, rows):
            calls.append(study)
            return rows
        wrapped = n.adapt_prepare_rows(original)
        raw = [dict(snr_db='clean')]
        for study in ('N512', 'N1024', 'M1', 'M1_RATE', 'M2_ORACLE'):
            self.assertIs(wrapped(None, study, raw), raw)
        self.assertEqual(calls, ['N512', 'N1024', 'M1', 'M1_RATE', 'M2_ORACLE'])
        r = load_replay(); engine = r.ReplayEngine(Path('.'), {}, {'records': records()}, synthetic=False)
        rate = [dict(source_index='0', m='5', q='0', order='entropy')]
        expected = engine._prepare_rows('M1_RATE', rate)
        n.install(r)
        self.assertEqual(engine._prepare_rows('M1_RATE', rate), expected)

    def test_original_parity_stays_strict(self):
        r = load_replay(); n.install(r)
        original = dict(psnr_db='20.25', lpips_alex='0.2', dino_cosine='0.75',
                        observation_sha256='a' * 64)
        computed = dict(original)
        self.assertEqual(r.parity_check(original, computed)['status'], 'PASS')
        for change in (dict(psnr_db=21.), dict(observation_sha256='b' * 64)):
            with self.assertRaises(r.ReplayMismatch): r.parity_check(original, dict(computed, **change))

    def test_duplicate_install_or_unexpected_original_refused(self):
        r = load_replay(); n.install(r)
        with self.assertRaisesRegex(RuntimeError, 'already installed'): n.install(r)
        r = load_replay(); r.ReplayEngine._prepare_rows = lambda self, study, rows: rows
        with self.assertRaisesRegex(RuntimeError, 'Unexpected original'): n.install(r)
        r = load_replay()
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / 'replay.py'; altered.write_text('different source')
            r.__file__ = str(altered)
            with self.assertRaisesRegex(RuntimeError, 'pinned frozen'): n.install(r)


if __name__ == '__main__':
    unittest.main()
