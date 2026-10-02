"""ENGINEERING/SYNTHETIC replay contracts; no scientific image-quality claim."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('metrics_replay_test_target', HERE / 'replay.py')
r = importlib.util.module_from_spec(spec); sys.modules[spec.name] = r; spec.loader.exec_module(r)


def rgb(value=.5):
    return np.full((3, 256, 256), value, np.float32)


def records():
    pixels = np.zeros((3, 256, 256), np.uint8)
    return [dict(image_id=f'ENGINEERING_SOURCE_{i:03d}', preprocessing_id='ENGINEERING_PREPROCESS',
                 class_index=i, pixels=pixels) for i in range(100)]


def row(**changes):
    x = dict(source_id='ENGINEERING_SOURCE_000', source_index='0',
        preprocessing_id='ENGINEERING_PREPROCESS', N='512', phy_family='continuous',
        snr_db='1', noise_seed='2001', method='P512', decoder_id='Dc',
        psnr_db='20.0', lpips_alex='.2', dino_cosine='.7', dino_mismatched='.1',
        latent_valid='True', decoder_applied='True', latent_sq_err_final='100.0',
        waveform_sha256='a' * 64, observation_sha256='b' * 64)
    x.update(changes)
    return x


class FakeAdapter:
    def validate_rows(self, rows):
        self.validated = list(rows)

    def iterate_source(self, index, rows):
        for old in rows:
            yield old, rgb(), dict(old)


class ReplayEngineering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def engine(self, rows, study='N512', adapter=None, legacy_check=True):
        path = self.root / (study + '.csv')
        names = list(dict.fromkeys(k for x in rows for k in x)) or ['source_index', 'snr_db', 'noise_seed']
        with path.open('w', newline='') as h:
            writer = csv.DictWriter(h, names); writer.writeheader(); writer.writerows(rows)
        layout = r.Layout(self.root, self.root, self.root, path)
        e = r.ReplayEngine(self.root, {'identity': {'synthetic': True}}, {'records': records()},
            {study: layout}, adapters={study: adapter or FakeAdapter()}, synthetic=True,
            legacy_check=legacy_check)
        return e.setup([study])

    def test_complete_source_iterator_preserves_duplicate_rgb_rows(self):
        original = [row(), row(method='V_policy', policy_action='BYPASS')]
        e = self.engine(original)
        outputs = list(e.iterate_source('N512', 0))
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][2]['rgb_sha256'], outputs[1][2]['rgb_sha256'])
        self.assertNotEqual(outputs[0][0]['replay_row_id'], outputs[1][0]['replay_row_id'])
        for old, image, parity in outputs:
            self.assertTrue(parity['replay_parity_passed'])
            self.assertTrue(parity['synthetic'])
            self.assertEqual(image.dtype, np.float32)
            self.assertFalse(parity['original_png_used'])
        self.assertEqual(e.manifest()['rows']['N512'], 2)
        self.assertTrue(e.setup_bound_files())
        e.verify_frozen()

    def test_input_table_mutation_rejected_at_final_verification(self):
        e = self.engine([row()])
        e.layouts['N512'].rows.write_text('changed', encoding='utf-8')
        with self.assertRaises(r.ReplayMismatch):
            e.verify_frozen()

    def test_source_and_preprocessing_identity_rejected(self):
        for change in ({'source_id': 'WRONG'}, {'preprocessing_id': 'WRONG'}, {'source_index': '100'}):
            with self.subTest(change=change), self.assertRaises(r.ReplayMismatch):
                self.engine([row(**change)])

    def test_duplicate_original_row_rejected(self):
        with self.assertRaises(r.ReplayMismatch):
            self.engine([row(), row()])

    def test_omitted_or_repeated_output_rejected(self):
        class Omit(FakeAdapter):
            def iterate_source(self, i, rr):
                return iter(())
        class Repeat(FakeAdapter):
            def iterate_source(self, i, rr):
                yield rr[0], rgb(), dict(rr[0])
                yield rr[0], rgb(), dict(rr[0])
        for adapter in (Omit(), Repeat()):
            with self.subTest(adapter=type(adapter).__name__), self.assertRaises(r.ReplayMismatch):
                list(self.engine([row()], adapter=adapter).iterate_source('N512', 0))

    def test_changed_row_or_metric_rejected(self):
        class Change(FakeAdapter):
            def iterate_source(self, i, rr):
                changed = dict(rr[0], method='CHANGED')
                yield changed, rgb(), dict(changed)
        with self.assertRaises(r.ReplayMismatch):
            list(self.engine([row()], adapter=Change()).iterate_source('N512', 0))
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(row(), dict(row(), psnr_db=20.001))

    def test_hash_exactness_and_all_available_parity_fields_required(self):
        original = row()
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(original, dict(original, waveform_sha256='c' * 64))
        missing = dict(original); missing.pop('observation_sha256')
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(original, missing)

    def test_numeric_policy_fields_compare_value_with_exact_float(self):
        original = row(**{'lambda': '1.0', 'decoded_mode': '4', 'action_m': '4'})
        computed = dict(original, **{'lambda': 1., 'decoded_mode': 4, 'action_m': 4})
        self.assertEqual(r.parity_check(original, computed)['status'], 'PASS')
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(original, dict(computed, **{'lambda': 1.0000001}))
        missing = dict(original); missing.pop('latent_sq_err_final')
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(original, missing)

    def test_erasure_has_gray_pixels_and_blank_latent_error(self):
        original = row(latent_valid='False', decoder_applied='False', latent_sq_err_final='',
                       zero_erasure_proxy_sq_error='33.0', header_ok='False')
        e = self.engine([original])
        _, image, diag = next(e.iterate_source('N512', 0))
        self.assertTrue(np.all(image == np.float32(.5)))
        self.assertTrue(diag['replay_parity_passed'])
        with self.assertRaises(r.ReplayMismatch):
            r.parity_check(original, dict(original, latent_sq_err_final=0.))

    def test_d0_and_paid_class_semantics(self):
        c = r.metadata(row(method='D_C_QPSK_D0', decoder_id='D0', class_condition='C'), 'N512')
        self.assertTrue(c['replay_reference_only'])
        self.assertTrue(c['replay_semantic_side_information'])
        u = r.metadata(row(method='D_U_QPSK', class_condition='U'), 'N512')
        self.assertTrue(u['replay_true_class_sent'])
        self.assertFalse(u['replay_received_class_used_by_generator'])
        prefix = r.metadata(row(method='D_prefix_C_QPSK', class_condition='prefix'), 'N512')
        self.assertTrue(prefix['replay_paid_class_header'])
        self.assertFalse(prefix['replay_received_class_used_by_generator'])
        m1 = r.metadata(row(method='oracle_policy', order='oracle', historical_reference=False), 'M1')
        self.assertTrue(m1['replay_paid_oracle_mask'])
        self.assertFalse(m1['replay_true_class_sent'])

    def test_m2_clean_scope_not_replicated_as_three_noise_seeds(self):
        old = row(N='', phy_family='', method='', stage='full_noiseless',
                  projection='g8_c32', control='DIRECT', snr_db='clean', noise_seed='0')
        e = self.engine([old], 'M2_ORACLE')
        outputs = list(e.iterate_source('M2_ORACLE', 0))
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0][0]['replay_clean_oracle'])
        self.assertEqual(outputs[0][0]['noise_seed'], '0')
        for study, seed in (('M2_ORACLE', '2001'), ('N512', '0')):
            with self.subTest(study=study), self.assertRaises(r.ReplayMismatch):
                self.engine([dict(old, noise_seed=seed)], study)

    def test_rgb_has_no_uint8_or_resize_conversion(self):
        for value in (np.zeros((3, 256, 256), np.uint8),
                      np.zeros((256, 256, 3), np.float32),
                      np.zeros((3, 256, 256), np.float64), rgb(float('nan')), rgb(1.01)):
            with self.assertRaises(r.ReplayMismatch):
                r.rgb_array(value)
        image = rgb(.12345678)
        self.assertTrue(np.array_equal(image, r.rgb_array(image)))

    def test_cache_key_binds_target_and_metric_models(self):
        a = r.metric_cache_key(rgb(), rgb(0.), {'clip': 'FROZEN_A'})
        self.assertEqual(a, r.metric_cache_key(rgb(), rgb(0.), {'clip': 'FROZEN_A'}))
        self.assertNotEqual(a, r.metric_cache_key(rgb(), rgb(.1), {'clip': 'FROZEN_A'}))
        self.assertNotEqual(a, r.metric_cache_key(rgb(), rgb(0.), {'clip': 'FROZEN_B'}))
        x = rgb(); x[0, 0, 0] = np.nextafter(x[0, 0, 0], np.float32(1))
        self.assertNotEqual(r.rgb_fingerprint(x), r.rgb_fingerprint(rgb()))

    def test_import_aliases_restore_and_collision_is_rejected(self):
        path = self.root / 'module.py'; path.write_text('import engineering_alias\nvalue=engineering_alias.value\n')
        before = SimpleNamespace(value='BEFORE'); replacement = SimpleNamespace(value='DURING')
        with mock.patch.dict(sys.modules, {'engineering_alias': before}):
            mod = r.load_file_module('metrics_engineering_module', path, {'engineering_alias': replacement})
            self.assertEqual(mod.value, 'DURING')
            self.assertIs(sys.modules['engineering_alias'], before)
            wrong = self.root / 'wrong.py'; wrong.write_text('pass')
            with self.assertRaises(r.ReplayMismatch):
                r.load_file_module('metrics_engineering_module', wrong)
        sys.modules.pop('metrics_engineering_module', None)

    def test_formal_injected_adapter_and_partial_population_forbidden(self):
        with self.assertRaises(r.ReplayMismatch):
            r.ReplayEngine(self.root, {}, {'records': records()}, adapters={'M1': FakeAdapter()})
        with self.assertRaises(r.ReplayMismatch):
            r.ReplayEngine(self.root, {}, {'records': records()[:2]})
        with self.assertRaises(r.ReplayMismatch):
            r.ReplayEngine(self.root, {}, {'records': records()}, legacy_check=False)

    def test_old_method_set_is_exact30_with_four_d0_references(self):
        for N in (512, 1024):
            methods = r.old_methods(N)
            self.assertEqual(len(methods), 30)
            self.assertEqual(sum(m.endswith('_D0') for m in methods), 4)
            self.assertIn(f'P{N}', methods)
            self.assertIn('D_U_16QAM_at_C_action', methods)

    def test_specificity_mapping_checked_against_original_derangement(self):
        wrong = row(mismatch_source_id='ENGINEERING_SOURCE_000')
        with self.assertRaises(r.ReplayMismatch):
            self.engine([wrong])
        expected = records()[r.ReplayEngine._derangement()[0]]['image_id']
        e = self.engine([row(mismatch_source_id=expected)])
        self.assertEqual(next(e.iterate_source('N512', 0))[0]['replay_mismatch_source_id'], expected)

    def test_empty_actual_gate_skip_has_no_fabricated_rows(self):
        e = self.engine([], 'M2_ACTUAL')
        self.assertEqual(list(e.iterate_source('M2_ACTUAL', 0)), [])
        self.assertEqual(e.manifest()['rows']['M2_ACTUAL'], 0)

    def test_rate_curve_adds_registered_identity_without_changing_original(self):
        original = dict(source_id='ENGINEERING_SOURCE_000', source_index='0', m='5', q='4',
            order='oracle', source_bits='100', mask_bits='36', wireless_claim='False',
            psnr_db='20', lpips_alex='.2', dino_cosine='.7', dino_mismatched='.1',
            latent_valid='True', decoder_applied='True', latent_sq_err_final='100')
        e = self.engine([original], 'M1_RATE')
        added, image, parity = next(e.iterate_source('M1_RATE', 0))
        for name, value in original.items():
            self.assertEqual(added[name], value)
        self.assertEqual(added['preprocessing_id'], 'ENGINEERING_PREPROCESS')
        self.assertEqual(added['snr_db'], 'clean')
        self.assertEqual(added['noise_seed'], '0')
        self.assertEqual(added['N'], '')
        self.assertEqual(added['phy_family'], 'source_only')
        self.assertEqual(added['method'], 'rate_curve_oracle')
        self.assertEqual(added['projection'], 'm5_K4')
        self.assertEqual(added['replay_source_row_sha256'], r.source_row_hash(original))
        self.assertTrue(added['replay_source_only'])
        self.assertFalse(added['replay_paid_oracle_mask'])
        self.assertTrue(parity['replay_parity_passed'])

    def test_rate_dense_grid_requires_exact10800_source_points(self):
        self.assertEqual(len(r.rate_scopes()), 108)
        e = r.ReplayEngine(self.root, {'identity': {}}, {'records': records()})
        raw = [dict(source_id=f'ENGINEERING_SOURCE_{i:03d}', source_index=str(i),
            m=str(m), q=str(q), order=order, wireless_claim='False')
            for i in range(100) for m, q, order in r.rate_scopes()]
        prepared = e._prepare_rows('M1_RATE', raw)
        e._validate_rows('M1_RATE', prepared)
        self.assertEqual(len({r.row_id('M1_RATE', x) for x in prepared}), 10800)
        with self.assertRaises(r.ReplayMismatch):
            e._validate_rows('M1_RATE', prepared[:-1])
        changed = [dict(x) for x in prepared]
        changed[0]['wireless_claim'] = 'True'
        with self.assertRaises(r.ReplayMismatch):
            e._validate_rows('M1_RATE', changed)

    def test_factory_resolves_plain_module_imports_for_caller(self):
        loaded = {'device': 'ENGINEERING_CPU', 'identity': {'synthetic': True}}
        data = {'records': records()}
        common = SimpleNamespace(setup=mock.Mock(return_value=loaded),
                                 data=mock.Mock(return_value=data))
        modules = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), common]
        dummy = SimpleNamespace(setup=mock.Mock(return_value='SET_UP_ENGINE'))
        with mock.patch.object(r, 'load_file_module', side_effect=modules) as loader, \
             mock.patch.object(r, 'ReplayEngine', return_value=dummy) as constructor:
            result = r.create_engine(self.root, ['N512'])
        self.assertEqual(result, 'SET_UP_ENGINE')
        common.setup.assert_called_once_with()
        common.data.assert_called_once_with('development', loaded)
        constructor.assert_called_once()
        self.assertIn('assets', loader.call_args_list[2].args[2])
        self.assertIn('digital', loader.call_args_list[3].args[2])

    def test_original_three_noise_batch_and_alias_order_preserved(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Native Torch unavailable locally; remote CPU qualification executes this test')
        d = dict(records=records(), F=torch.zeros(100, 32, 16, 16),
                 T=torch.zeros(100, 680, dtype=torch.long), reference=torch.zeros(100, 3))
        e = r.ReplayEngine(self.root, dict(device=torch.device('cpu'), vae=object(),
            var=object(), static=[], identity={}), d, synthetic=True)
        batches = []; calls = []
        pmodel = SimpleNamespace(transmit=lambda F: torch.zeros(1, 4, 2),
            receive=lambda y, snr: torch.zeros(1, 32, 16, 16))
        def infer(vae, var, Z, prior, variance, static, lam, **kw):
            calls.append((prior, lam, Z.shape[0], kw.get('details')))
            return {'fhat': Z + float(lam) * .1}
        def render(Z, ii, records, models, refs, mismatch, save_images):
            batches.append(Z.shape[0])
            return ([dict(psnr_db=20., lpips_alex=.2, dino_cosine=.7, dino_mismatched=.1)] * len(Z),
                    [torch.full((3, 256, 256), .5, dtype=torch.float32)] * len(Z))
        plugin = SimpleNamespace(waveform_sha=r.arraysha,
            apply_channel=lambda wave, *args: wave,
            CELL=object(), candidate_key=lambda prior, lam: 'A1' if lam == 0 or prior == 'A1' else prior + '_lambda' + str(lam),
            rx=SimpleNamespace(infer=infer, fusion=lambda Z, Q, alpha: Z * (1 - alpha) + Q * alpha),
            old=SimpleNamespace(b=SimpleNamespace(split=lambda T: [T])), images_metrics=render)
        adapter = r._OldAdapter.__new__(r._OldAdapter)
        adapter.e, adapter.p, adapter.plugin, adapter.N, adapter.torch = e, pmodel, plugin, 512, torch
        adapter.models = {}; adapter.obs_index = {(0, 1, seed): j for j, seed in enumerate(r.DEV_SEEDS)}
        wave_hash = r.arraysha(np.zeros((4, 2), np.float32))
        adapter.obs = dict(Z=torch.zeros(3, 32, 16, 16), source_indices=torch.zeros(3, dtype=torch.long),
            rows=[dict(waveform_sha256=wave_hash, observation_sha256=wave_hash) for _ in range(3)])
        selected = dict(raw_selected_lambda=0., alpha=[.25] * 32, policy_action='BYPASS')
        adapter.ppolicy = dict(levels={'1': dict(variance_by_scale=[1.] * 10,
            methods={p: selected for p in ('A1', 'A2', 'V')}, alpha_common=[.25] * 32,
            V_lambda1=dict(alpha=[.25] * 32))})
        oldrows = [row(method=name, noise_seed=str(seed))
                   for name in r.old_methods(512) if not name.startswith('D_') for seed in r.DEV_SEEDS]
        outputs = list(adapter._plugin_source(0, oldrows))
        self.assertEqual(len(outputs), 42)
        self.assertTrue(batches and all(x == 3 for x in batches))
        self.assertEqual(calls, [('A1', 0., 3, True), ('V', 1., 3, False)])
        self.assertTrue(all(image.dtype == torch.float32 for _, image, _ in outputs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
