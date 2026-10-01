"""Scientific gate and missing-latent checks; no GPU or model inference."""
import importlib.util
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('_extreme_bw_analysis_tested', Path(__file__).with_name('analysis.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def metric(delta, low, high):
    return dict(delta=delta, ci_low=low, ci_high=high)


def fixture(snr=1, action='CORRECT', branches=None, psnr=0, calibration_psnr=0):
    branches = branches or {c: 'LPIPS' for c in ('P1024', 'A1_policy', 'A2_policy')}
    lookup = {}
    for control, branch in branches.items():
        lookup[(snr, 'V_policy', control, 'lpips_alex')] = metric(-.03, -.04, -.02) if branch == 'LPIPS' else metric(0, -.001, .001)
        lookup[(snr, 'V_policy', control, 'dino_cosine')] = metric(.03, .02, .04) if branch == 'DINO' else metric(0, -.001, .001)
        lookup[(snr, 'V_policy', control, 'delta_specific')] = metric(.02, .01, .03)
    lookup[(snr, 'V_policy', 'P1024', 'psnr_db')] = metric(psnr, psnr-.01, psnr+.01)
    for method in analysis.DIGITAL:
        for name in ('lpips_alex', 'dino_cosine', 'delta_specific', 'psnr_db', 'latent_sq_err_final'):
            lookup[(snr, method, 'P1024', name)] = metric(0, -.001, .001)
    cap = .2 if snr == 13 else .3
    policy = dict(levels={str(snr): dict(psnr_drop_cap_db=cap, P1024=dict(psnr_db=20.),
        methods={'V': dict(policy_action=action, raw_feasible=calibration_psnr >= -cap,
            calibration=dict(psnr_db=20.+calibration_psnr), calibration_psnr_delta_db=calibration_psnr,
            calibration_psnr_drop_cap_db=cap)})})
    return lookup, policy


class ScientificAnalysisTests(unittest.TestCase):
    def gate(self, snr=1, **kwargs):
        lookup, policy = fixture(snr, **kwargs)
        return next(r for r in analysis.decisions([snr], lookup, policy) if r['hypothesis'] == 'H_R')

    def test_same_metric_branch_is_required(self):
        gate = self.gate(branches={'P1024': 'LPIPS', 'A1_policy': 'DINO', 'A2_policy': 'LPIPS'})
        self.assertEqual(gate['label'], 'PARTIAL_GAIN')
        self.assertEqual(gate['same_branch_supported'], [])

    def test_same_branch_clear_gain_and_bypass(self):
        self.assertEqual(self.gate()['label'], 'CLEAR_GAIN')
        self.assertEqual(self.gate(action='BYPASS')['label'], 'BYPASS_SELECTED')

    def test_dino_requires_specificity_for_every_control(self):
        lookup, policy = fixture(branches={c: 'DINO' for c in ('P1024', 'A1_policy', 'A2_policy')})
        lookup[(1, 'V_policy', 'A2_policy', 'delta_specific')] = metric(.02, -.01, .03)
        gate = next(r for r in analysis.decisions([1], lookup, policy) if r['hypothesis'] == 'H_R')
        self.assertEqual(gate['label'], 'PARTIAL_GAIN')

    def test_13db_uses_registered_tighter_cap(self):
        diagnostic = self.gate(13, psnr=-.25)
        self.assertEqual(diagnostic['label'], 'CLEAR_GAIN')
        self.assertFalse(diagnostic['observed_psnr_constraint_pass'])
        self.assertEqual(diagnostic['delta_psnr']['delta'], -.25)
        self.assertTrue(diagnostic['observed_psnr_is_diagnostic_only'])
        self.assertEqual(self.gate(13, calibration_psnr=-.25)['label'], 'NO_GAIN_ESTABLISHED')
        self.assertEqual(self.gate(7, calibration_psnr=-.25)['label'], 'CLEAR_GAIN')

    def test_calibration_cap_receipt_cannot_change(self):
        lookup, policy = fixture(13)
        policy['levels']['13']['methods']['V']['calibration_psnr_drop_cap_db'] = .3
        with self.assertRaises(ValueError):
            analysis.decisions([13], lookup, policy)

    def test_missing_latent_has_conditional_scope(self):
        values = np.array([np.nan, 4., 8.])
        indices = np.random.default_rng(2).integers(3, size=(100, 3))
        result = analysis.conditional_interval(values, indices)
        self.assertEqual(result['mean'], 6.)
        self.assertEqual(result['valid_sources'], 2)
        empty = analysis.conditional_interval(np.full(3, np.nan), indices)
        self.assertIsNone(empty['mean'])
        self.assertEqual(empty['valid_sources'], 0)

    def test_header_failure_does_not_get_a_fabricated_latent(self):
        row = dict(source_id='x', source_index=0, preprocessing_id='p', snr_db=1, noise_seed=2001,
                   method='D_U_QPSK', psnr_db=10., lpips_alex=.5, dino_cosine=.2, dino_mismatched=.1,
                   decoder_id='Dc', class_condition='unconditional', phy_family='QPSK',
                   energy_constraint='per_frame_2N', N=1024, E=2048., latent_valid=False,
                   latent_sq_error=float('nan'), header_ok=False)
        normalized = analysis.normalize_rows([row])[0]
        self.assertEqual(normalized['latent_sq_err_final'], '')
        self.assertFalse(normalized['latent_valid'])
        row['latent_valid'] = True
        with self.assertRaises(ValueError):
            analysis.normalize_rows([row])

    def test_paired_latent_uses_frame_intersection(self):
        methods = list(analysis.POLICIES) + list(analysis.DIGITAL)
        methods += [p + '_' + kind for kind in ('common', 'tok') for p in ('A1', 'A2', 'V')]
        for family in ('QPSK', '16QAM'):
            methods += ['D_C_' + family + '_at_U_action', 'D_U_' + family + '_at_C_action']
            methods += ['D_prefix_' + c + '_' + family for c in ('U', 'C')]
            methods += ['D_' + c + '_' + family + '_D0' for c in ('U', 'C')]
        ids = ['a', 'b']; raw = []; means = []; matrix = {}
        for method in methods:
            matrix[(1., method)] = np.zeros((2, len(analysis.METRICS)))
            for sid in ids:
                for seed, value in zip((2001, 2002, 2003), (1., 2., 9.)):
                    valid = method != 'D_U_QPSK' or seed == 2002
                    raw.append(dict(source_id=sid, method=method, snr_db=1., noise_seed=seed,
                                    latent_valid=valid, latent_sq_err_final=10. if method == 'D_U_QPSK' else value))
                means.append(dict(source_id=sid, snr_db=1., method=method,
                    decoder_id='Dc', class_condition='unconditional', phy_family='QPSK',
                    energy_constraint='per_frame_2N', action_m='', lambda_='', policy_action='',
                    latent_valid_noise_count=1 if method == 'D_U_QPSK' else 3))
        indices = np.zeros((20, 2), dtype=np.int64)
        _, _, lookup = analysis.analyze(matrix, [1.], methods, indices, means, raw, ids)
        result = lookup[(1., 'D_U_QPSK', 'P1024', 'latent_sq_err_final')]
        self.assertEqual(result['delta'], 8.)
        self.assertEqual(result['valid_paired_frames'], 2)
        self.assertEqual(result['valid_paired_sources'], 2)

    def test_difference_in_differences_uses_same_source_indices(self):
        ids = ['a', 'b']; methods = ['P1024', *analysis.DIGITAL]
        matrices, raws = [], []
        for budget in (512, 1024):
            matrix, raw = {}, []
            for snr in analysis.SNRS:
                for method in methods:
                    actual_method = 'P512' if budget == 512 and method == 'P1024' else method
                    digital = actual_method.startswith('D_')
                    psnr = (18. if digital else 20.) if budget == 512 else (22. if digital else 23.)
                    lpips = (.4 if digital else .3) if budget == 512 else (.25 if digital else .2)
                    dino = (.5 if digital else .2) if budget == 512 else (.6 if digital else .4)
                    latent = (140. if digital else 100.) if budget == 512 else (90. if digital else 70.)
                    values = np.zeros((2, len(analysis.METRICS)))
                    for metric, value in (('psnr_db', psnr), ('lpips_alex', lpips), ('dino_cosine', dino), ('latent_sq_err_final', latent)):
                        values[:, analysis.METRICS.index(metric)] = value
                    matrix[(float(snr), actual_method)] = values
                    for index, sid in enumerate(ids):
                        for seed in sorted(analysis.NOISE_SEEDS):
                            raw.append(dict(source_id=sid, source_index=index, preprocessing_id='prep' + sid,
                                snr_db=snr, noise_seed=seed, method=actual_method, N=budget, E=2*budget,
                                decoder_id='Dc', class_condition='unconditional', phy_family='QPSK' if digital else 'continuous',
                                energy_constraint='per_frame_2N', action_m=4 if digital else '',
                                waveform_sha256=str(budget) + sid, observation_sha256=str(budget) + sid + str(seed),
                                latent_valid=True, latent_sq_err_final=latent))
            matrices.append(matrix); raws.append(raw)
        indices = np.tile(np.arange(2), (32, 1))
        paired, changed, sources, ledger = analysis.cross_budget_analysis(raws[1], matrices[1], raws[0], matrices[0], methods, ids, indices)
        keyed = {(r['snr_db'], r['method'], r['metric']): r for r in changed}
        self.assertAlmostEqual(keyed[(1, 'D_U_QPSK', 'psnr_db')]['delta'], 1.)
        self.assertAlmostEqual(keyed[(1, 'D_U_QPSK', 'lpips_alex')]['delta'], -.05)
        self.assertAlmostEqual(keyed[(1, 'D_U_QPSK', 'dino_cosine')]['delta'], -.1)
        self.assertAlmostEqual(keyed[(1, 'D_U_QPSK', 'latent_sq_err_final')]['delta'], -20.)
        self.assertTrue(all(r['source_reuse_correlated'] for r in changed))
        self.assertTrue(all(not r['same_waveform'] and not r['same_observation'] for r in ledger))
        self.assertEqual({r['bootstrap_indices_sha256'] for r in changed}, {hashlib.sha256(indices.tobytes()).hexdigest()})

    def test_four_way_latent_intersection_excludes_disjoint_failures(self):
        ids = ['a']; first = {}; second = {}
        for seed in sorted(analysis.NOISE_SEEDS):
            for method in ('P', 'D'):
                first[(1., method, 'a', seed)] = dict(latent_valid=method == 'P' or seed == 2001, latent_sq_err_final=100.)
                second[(1., method, 'a', seed)] = dict(latent_valid=method == 'P' or seed == 2002, latent_sq_err_final=90.)
        values, frames = analysis.latent_term_vector([(first, 'D', 1), (first, 'P', -1), (second, 'D', -1), (second, 'P', 1)], ids, 1)
        self.assertEqual(frames, 0)
        self.assertTrue(np.isnan(values[0]))


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / 'results/extreme_bandwidth_20260930_R1'
if not (FIXTURE_ROOT / 'per_frame.csv').exists():
    FIXTURE_ROOT = Path(__file__).resolve().parent.parent / 'extreme_bw_results'
FULL_TEST_AVAILABLE = (importlib.util.find_spec('matplotlib') is not None and importlib.util.find_spec('PIL') is not None
                       and (FIXTURE_ROOT / 'per_frame.csv').exists())


@unittest.skipUnless(FULL_TEST_AVAILABLE, 'Full engineering fixture needs matplotlib/PIL and immutable N512 measured table')
class FullEngineeringAnalysisTests(unittest.TestCase):
    def test_complete_main_report_figures_and_synthetic_receipt(self):
        """Adapt N512 copies only inside a clearly named disposable temp directory."""
        from PIL import Image, ImageDraw
        raw_original = analysis.csv_rows(FIXTURE_ROOT / 'per_frame.csv')
        prior_config = analysis.read_json(FIXTURE_ROOT / 'config.json')
        prior_selected = prior_config['plugin']['identity']['selected']
        prior_done_path = FIXTURE_ROOT / 'training_completion.json'
        if not prior_done_path.exists():
            prior_done_path = Path(prior_config['plugin']['identity']['base']['training']) / 'completion.json'
        prior_done = analysis.read_json(prior_done_path)
        previous_budget = analysis.training_budget(prior_done, prior_selected, 512)
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_SYNTHETIC_N1024_ANALYSIS_') as directory:
            temporary = Path(directory); out = temporary / 'out'; results = temporary / 'results'
            out.mkdir(); results.mkdir()
            config = copy.deepcopy(prior_config['plugin']); config['engineering_fixture'] = True
            config.update(N=1024, E=2048, scope='ENGINEERING_SYNTHETIC_INTERFACE_TEST_ONLY')
            selected = copy.deepcopy(prior_selected)
            selected.update(N=1024, arm_key='P1024', checkpoint_sha256='0'*64)
            training_path = temporary / 'synthetic_training'; training_path.mkdir()
            config['identity']['selected'] = selected
            config['identity']['base'].update(N=1024, method='P1024', arm='P1024', training=str(training_path), selected=selected)
            config['identity']['models']['P1024'] = config['identity']['models'].pop('P512')
            done = copy.deepcopy(prior_done); done['selected'] = {'P1024': selected}
            analysis.write_json(training_path / 'completion.json', done)
            policy = copy.deepcopy(analysis.read_json(FIXTURE_ROOT / 'selected_policy.json'))
            for level in policy['plugin']['levels'].values():
                level['P1024'] = level.pop('P512')
            analysis.write_json(results / 'plugin_config.json', config)
            analysis.write_json(results / 'digital_config.json', prior_config['digital'])
            analysis.write_json(results / 'plugin_selected_policy.json', policy['plugin'])
            analysis.write_json(results / 'digital_selected_policy.json', policy['digital'])
            plugin_rows, digital_rows = [], []
            for old in raw_original:
                row = dict(old); row.update(N=1024, E=2*float(old['E']), energy=2*float(old['E']))
                row['method'] = 'P1024' if row['method'] == 'P512' else row['method']
                row['waveform_sha256'] = hashlib.sha256(('SYNTHETIC1024|' + old['waveform_sha256']).encode()).hexdigest()
                row['observation_sha256'] = hashlib.sha256(('SYNTHETIC1024|' + old['observation_sha256']).encode()).hexdigest()
                (plugin_rows if row['system'] == 'continuous_receiver_plugin' else digital_rows).append(row)
            analysis.write_csv(results / 'plugin_per_frame.csv', plugin_rows)
            analysis.write_csv(results / 'digital_per_frame.csv', digital_rows)
            token_rows = []
            counts = (1, 4, 9, 16, 25, 36, 64, 100, 169, 256)
            for sid in analysis.read_json(FIXTURE_ROOT / 'analysis_completion.json')['source_order']:
                for snr in analysis.SNRS:
                    for prior in ('A1', 'A2', 'V'):
                        for mode in ('CL', 'TF'):
                            for seed in sorted(analysis.NOISE_SEEDS):
                                for scale, count in enumerate(counts, 1):
                                    token_rows.append(dict(source_id=sid, snr_db=snr, method=prior, mode=mode, noise_seed=seed,
                                        scale=scale, token_count=count, acc=.5, path_accuracy=.5, entropy=1., logp_true=-1.))
            analysis.write_csv(results / 'plugin_token_accuracy.csv', token_rows)
            source_ids = analysis.read_json(FIXTURE_ROOT / 'analysis_completion.json')['source_order']
            times = []
            for method in ('P1024', 'V_raw', 'V_policy', 'V_lambda1', *analysis.DIGITAL):
                for snr in analysis.SNRS:
                    for index in (0, 11, 22, 33, 44, 55, 66, 77, 88, 99):
                        for repeat in (0, 1):
                            times.append(dict(source_id=source_ids[index], source_index=index, snr_db=snr, method=method,
                                repeat=repeat, rx_ms=1., ran_VAR=method == 'V_lambda1', tx_ms=2., policy_action='ENGINEERING_SYNTHETIC'))
            analysis.write_csv(results / 'plugin_timing.csv', times[:400])
            analysis.write_csv(results / 'digital_timing.csv', times[400:])
            files = [FIXTURE_ROOT / name for name in ('per_frame.csv', 'config.json', 'selected_policy.json', 'analysis_completion.json', 'summary.csv', 'decisions.json')]
            normalized = analysis.normalize_rows(raw_original, 512)
            preprocess = {r['source_id']: r['preprocessing_id'] for r in normalized}
            receipt = dict(status='IMMUTABLE_N512_COMPARISON_IMPORTED', N=512, engineering_fixture=True,
                source_bindings={str(path): analysis.sha(path) for path in files}, original_result=str(FIXTURE_ROOT),
                source_ids=source_ids, preprocessing_ids=[preprocess[sid] for sid in source_ids], budget=previous_budget)
            analysis.write_json(out / 'prior_budget_import.json', receipt)
            for index in (0, 25, 50, 75):
                for snr in (4, 13):
                    for method in ('Source', 'P1024', 'V_policy', *analysis.DIGITAL):
                        origin = 'plugin' if method in ('Source', 'P1024', 'V_policy') else 'digital'
                        path = results / 'examples' / origin / f'source_{index:03d}_snr_{snr}_{method}_seed2001.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        image = Image.new('RGB', (256, 256), 'white')
                        ImageDraw.Draw(image).text((8, 8), 'ENGINEERING / SYNTHETIC', fill='black'); image.save(path)
            # The source/bootstrap/report code is unchanged; reduce only test resamples.
            with patch.object(analysis, 'RESAMPLES', 128):
                completed = analysis.main(out, results)
            self.assertEqual(completed['status'], 'ENGINEERING_SYNTHETIC_N1024_ANALYSIS_COMPLETE')
            self.assertTrue(completed['synthetic'])
            self.assertFalse(completed['automatic_further_experiments'])
            self.assertEqual(completed['metric_rows'], 45000)
            self.assertEqual(completed['plugin_metric_rows'], 21000)
            self.assertEqual(completed['digital_metric_rows'], 24000)
            self.assertEqual(completed['cross_budget_source_pairing_rows'], 45000)
            self.assertEqual(len(completed['examples']), 8)
            self.assertTrue(completed['figures'])
            self.assertTrue(completed['training_budgets']['N512']['budget_truncated'])
            report = (results / 'report.md').read_text(encoding='utf-8')
            self.assertIn('ENGINEERING / SYNTHETIC', report)
            self.assertIn('(D1024−P1024)−(D512−P512)', report)
            self.assertIn('预算截断', report)
            self.assertIn('16QAM 与 QPSK', report)
            self.assertNotIn('第一阶段 N1024', report)
            self.assertNotIn('是否需要 N1024', report)
            for name, expected in completed['files'].items():
                self.assertEqual(analysis.sha(results / name), expected)


if __name__ == '__main__':
    unittest.main()
