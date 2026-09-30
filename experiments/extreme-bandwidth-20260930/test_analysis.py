"""Scientific gate and missing-latent checks; no GPU or model inference."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('_extreme_bw_analysis_tested', Path(__file__).with_name('analysis.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def metric(delta, low, high):
    return dict(delta=delta, ci_low=low, ci_high=high)


def fixture(snr=1, action='CORRECT', branches=None, psnr=0, calibration_psnr=0):
    branches = branches or {c: 'LPIPS' for c in ('P512', 'A1_policy', 'A2_policy')}
    lookup = {}
    for control, branch in branches.items():
        lookup[(snr, 'V_policy', control, 'lpips_alex')] = metric(-.03, -.04, -.02) if branch == 'LPIPS' else metric(0, -.001, .001)
        lookup[(snr, 'V_policy', control, 'dino_cosine')] = metric(.03, .02, .04) if branch == 'DINO' else metric(0, -.001, .001)
        lookup[(snr, 'V_policy', control, 'delta_specific')] = metric(.02, .01, .03)
    lookup[(snr, 'V_policy', 'P512', 'psnr_db')] = metric(psnr, psnr-.01, psnr+.01)
    for method in analysis.DIGITAL:
        for name in ('lpips_alex', 'dino_cosine', 'delta_specific', 'psnr_db', 'latent_sq_err_final'):
            lookup[(snr, method, 'P512', name)] = metric(0, -.001, .001)
    cap = .2 if snr == 13 else .3
    policy = dict(levels={str(snr): dict(psnr_drop_cap_db=cap, P512=dict(psnr_db=20.),
        methods={'V': dict(policy_action=action, raw_feasible=calibration_psnr >= -cap,
            calibration=dict(psnr_db=20.+calibration_psnr), calibration_psnr_delta_db=calibration_psnr,
            calibration_psnr_drop_cap_db=cap)})})
    return lookup, policy


class ScientificAnalysisTests(unittest.TestCase):
    def gate(self, snr=1, **kwargs):
        lookup, policy = fixture(snr, **kwargs)
        return next(r for r in analysis.decisions([snr], lookup, policy) if r['hypothesis'] == 'H_R')

    def test_same_metric_branch_is_required(self):
        gate = self.gate(branches={'P512': 'LPIPS', 'A1_policy': 'DINO', 'A2_policy': 'LPIPS'})
        self.assertEqual(gate['label'], 'PARTIAL_GAIN')
        self.assertEqual(gate['same_branch_supported'], [])

    def test_same_branch_clear_gain_and_bypass(self):
        self.assertEqual(self.gate()['label'], 'CLEAR_GAIN')
        self.assertEqual(self.gate(action='BYPASS')['label'], 'BYPASS_SELECTED')

    def test_dino_requires_specificity_for_every_control(self):
        lookup, policy = fixture(branches={c: 'DINO' for c in ('P512', 'A1_policy', 'A2_policy')})
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
                   energy_constraint='per_frame_2N', N=512, E=1024., latent_valid=False,
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
        result = lookup[(1., 'D_U_QPSK', 'P512', 'latent_sq_err_final')]
        self.assertEqual(result['delta'], 8.)
        self.assertEqual(result['valid_paired_frames'], 2)
        self.assertEqual(result['valid_paired_sources'], 2)


if __name__ == '__main__':
    unittest.main()
