"""CPU engineering checks; these do not qualify native GPU reconstruction."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from history_common import identity, sha, write
from historical_selected_references import (METHODS, MODEL_STATES, STUDY, SelectedReferences,
    ReferenceMismatch, deterministic_parity, output_spec, scalar_parity, rgb_sha)


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def fixture(root):
    source = root/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/source/development'
    digital = root/'outputs/EXTREME-BW-20261001-R1-N1024/digital'
    refs = digital/'references/development'
    ids, prep = ['source_'+str(i) for i in range(100)], ['pixels_'+str(i) for i in range(100)]
    write(source/'registration.json', dict(decoder_sha256=MODEL_STATES['decoder'], bindings={}))
    write(refs/'identity.json', dict(assets={'models': MODEL_STATES}, role='development',
        sources=ids, preprocessing_ids=prep, source_bindings={}))
    source_reg, ref_reg = sha(source/'registration.json'), sha(refs/'identity.json')
    source_rows, reference_rows, receipts = [], [], {}
    for i in range(100):
        base = dict(population='development', source_index=i, source_id=ids[i], preprocessing_id=prep[i],
                    N='not_applicable_source_reference', E='not_applicable_source_reference')
        metrics = dict(psnr_db=20., lpips_alex=.2, dino_cosine=.7)
        a = [dict(base, method='codec_m'+str(m), source_bits=100) for m in range(6, 11)]
        a += [dict(base, method=d+'_'+kind, **metrics) for kind in ('F', 'Fq') for d in ('D0', 'Dc')]
        a += [dict(base, method=f'{d}_{kind}_m{m}', **metrics)
              for m in range(6, 10) for kind in ('prefix', 'VAR') for d in ('D0', 'Dc')]
        r = [dict(base, method=f'{d}_{kind}_m{m}', action_m=m, decoder_id=d, class_condition=kind,
                  reference_origin='compatible_Source_A' if m >= 6 and kind != 'U' else 'new_noiseless_reference',
                  **metrics) for m in range(4, 8) for d in ('Dc', 'D0') for kind in ('C', 'U', 'prefix')]
        ap = source/'images'/f'{i:04d}.json'
        write(ap, dict(source_id=ids[i], registration_sha256=source_reg, rows=a))
        receipts[ap.name] = sha(ap)
        write(refs/f'{i:04d}.json', dict(identity_sha256=ref_reg, rows=r))
        source_rows.extend(a); reference_rows.extend(r)
    atable, rtable = source/'source_codec_per_image.csv', digital/'source_references_development.csv'
    csv_file(atable, source_rows); csv_file(rtable, reference_rows)
    write(source/'completion.json', dict(status='REAL_SOURCE_CODEC_COMPLETE', sources=100,
        source_rows=2500, synthetic=False, registration_sha256=source_reg,
        decoder_sha256=MODEL_STATES['decoder'], files={atable.name:sha(atable)}, source_receipts=receipts))
    write(refs/'completion.json', dict(status='SOURCE_REFERENCE_COMPLETE', sources=100, rows=2400,
        reference_only=True, sha256=sha(rtable)))


class SelectedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        fixture(cls.root)
        cls.adapter = SelectedReferences(cls.root, STUDY)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_scope_has_no_fake_noise_repetitions(self):
        desc = self.adapter.describe()
        self.assertEqual((desc['sources'], desc['rows'], desc['original_scalar_rows'], desc['new_deterministic_rows']),
                         (100, 1800, 1700, 100))
        self.assertEqual(desc['noise_seeds'], [])
        self.assertEqual(desc['snrs_db'], [])
        self.assertEqual(len(METHODS), 18)
        for i in range(100):
            rows = self.adapter.expected_rows(i)
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(set(identity(r) for r in rows)), 18)
            self.assertTrue(all(r.get('noise_seed', '') == '' and r.get('snr_db', '') == '' for r in rows))
            self.assertTrue(all(self.adapter.metadata(r)['N'] == '' for r in rows))

    def test_decoder_and_class_mappings(self):
        self.assertEqual([m for m in METHODS if output_spec(m)['decoder'] == 'D0'], ['D0_Fq'])
        for row in self.adapter.expected_rows(0):
            m = self.adapter.metadata(row)
            self.assertTrue(m['reference_only'])
            self.assertFalse(m['oracle'])
            self.assertFalse(m['classification_main_eligible'])
            self.assertEqual(m['label_conditioned'], '_C_' in m['method_id'])
            if '_U_' in m['method_id']:
                self.assertEqual(m['class_embedding_index'], 1000)
            if '_prefix_' in m['method_id']:
                self.assertEqual(m['class_condition'], 'none')
        for invalid in ('Dc_U_m9', 'Dc_F', 'D0_C_m4'):
            if invalid == 'Dc_F':
                self.assertEqual(output_spec(invalid)['kind'], 'F')
            else:
                with self.assertRaises(ReferenceMismatch):
                    output_spec(invalid)

    def test_original_m8_conditional_row_is_preserved(self):
        row = next(r for r in self.adapter.expected_rows(3) if r['method'] == 'Dc_VAR_m8')
        self.assertEqual(row['psnr_db'], '20.0')
        metadata = self.adapter.metadata(row)
        self.assertEqual(metadata['method_id'], 'Dc_C_m8')
        self.assertEqual(metadata['original_location']['original_method'], 'Dc_VAR_m8')
        self.assertTrue(metadata['original_scalar_parity_available'])
        copy = self.adapter.expected_rows(3)
        copy[0]['method'] = 'tampered'
        self.assertNotEqual(self.adapter.expected_rows(3)[0]['method'], 'tampered')

    def test_new_m8_has_no_invented_scalar_or_old_parity(self):
        row = next(r for r in self.adapter.expected_rows(1) if r['method'] == 'Dc_U_m8')
        self.assertTrue(all(k not in row for k in ('psnr_db', 'lpips_alex', 'dino_cosine')))
        self.assertFalse(row['historical_scalar_reference'])
        m = self.adapter.metadata(row)
        self.assertFalse(m['original_scalar_parity_available'])
        self.assertEqual(m['original_metric_columns']['psnr_db'], 'new_native_psnr_db')
        image = np.zeros((3, 256, 256), np.float32)
        p = deterministic_parity(image, image.copy(), dict(psnr_db=20., lpips_alex=.2, dino_cosine=.7))
        self.assertTrue(p['replay_parity_passed'])
        self.assertFalse(p['original_scalar_parity_available'])
        self.assertEqual(p['historical_scalar_comparison'], 'NOT_AVAILABLE')
        changed = image.copy(); changed[0, 0, 0] = .1
        with self.assertRaises(ReferenceMismatch):
            deterministic_parity(image, changed, p['native_metrics'])

    def test_old_scalar_parity_rejects_mismatch_missing_and_nan(self):
        original = dict(psnr_db='20.0', lpips_alex='.2', dino_cosine='.7')
        actual = dict(psnr_db=20., lpips_alex=.2, dino_cosine=.7)
        self.assertTrue(scalar_parity(original, actual)['original_scalar_parity_available'])
        for changed in (dict(actual, psnr_db=20.1), dict(actual, dino_cosine=float('nan')), {'psnr_db':20.}):
            with self.assertRaises(ReferenceMismatch):
                scalar_parity(original, changed)

    def test_rejects_unknown_scope_and_injected_models(self):
        with self.assertRaises(ReferenceMismatch):
            SelectedReferences(self.root, 'ALL_REFERENCES')
        with self.assertRaises(ReferenceMismatch):
            SelectedReferences(self.root, STUDY, loaded={})
        for bad in (-1, 100, True, 1.0):
            with self.assertRaises(ReferenceMismatch):
                self.adapter.expected_rows(bad)

    def test_receipt_tamper_is_detected_before_gpu(self):
        path = self.adapter.ref_folder/'0000.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['rows'][0]['psnr_db'] = 25.
            write(path, changed)
            with self.assertRaises(ReferenceMismatch):
                SelectedReferences(self.root, STUDY)
        finally:
            path.write_bytes(original)

    def test_float_rgb_rejects_png_like_quantization_and_bad_dimensions(self):
        self.assertEqual(len(rgb_sha(np.zeros((3, 256, 256), np.float32))), 64)
        for image in (np.zeros((3, 256, 256), np.uint8), np.zeros((256, 256, 3), np.float32),
                      np.ones((3, 256, 256), np.float32)*1.1):
            with self.assertRaises(ReferenceMismatch):
                rgb_sha(image)


if __name__ == '__main__':
    unittest.main()
