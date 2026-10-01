"""CPU engineering checks; no image-quality or real-weight acceptance claims."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('methods_20261002_residual_receiver', HERE / 'residual_receiver.py')
rx = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rx
spec.loader.exec_module(rx)


class AffinePhi(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        with torch.no_grad():
            self.conv.weight.fill_(.03)
            self.conv.bias.fill_(.17)

    def forward(self, x):
        return .5 * x + .5 * self.conv(x)


class Shared(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.phi = AffinePhi(channels)

    def __getitem__(self, index):
        return self.phi


class Quant(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 2)
        self.quant_resi = Shared(2)
        self.v_patch_nums = rx.PATCH_NUMS
        self.using_znorm = False
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[-1., -1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., -1.], [-2., 1.]]))

    def get_next_autoregressive_input(self, k, n, fhat, h):
        fhat = fhat + rx.contribution_map(self, h, k)
        return fhat, fn.interpolate(fhat, size=(rx.PATCH_NUMS[min(k + 1, 9)],) * 2, mode='area')


class ResidualEngineeringTests(unittest.TestCase):
    def test_area_right_inverse_all_registered_grids(self):
        for g, c in rx.PROJECTIONS:
            p = rx.Projection(g, c, torch.eye(32))
            z = torch.randn(2, p.dimension, generator=torch.Generator().manual_seed(8))
            self.assertTrue(torch.allclose(p.forward(p.lift(z)), z, atol=5e-6, rtol=5e-6))

    def test_noninteger_area_composition_is_not_original_area(self):
        x = torch.randn(1, 2, 16, 16, generator=torch.Generator().manual_seed(6))
        nested = fn.interpolate(fn.interpolate(x, size=(8, 8), mode='area'), size=(6, 6), mode='area')
        direct = fn.interpolate(x, size=(6, 6), mode='area')
        self.assertGreater(float((nested - direct).abs().max()), .01)

    def test_pca_shared_axes_and_no_centering_of_measurement(self):
        x = torch.randn(4, 32, 16, 16, generator=torch.Generator().manual_seed(9)) + 1.7
        pca = rx.fit_channel_pca(x)
        self.assertTrue(torch.allclose(pca['axes'] @ pca['axes'].T, torch.eye(32), atol=2e-5, rtol=2e-5))
        self.assertEqual(pca['samples'], 4 * 256)
        p = rx.Projection(4, 8, pca['axes'])
        self.assertTrue(torch.allclose(p.forward(2 * x), 2 * p.forward(x), atol=2e-5))

    def test_affine_bias_difference_and_quadratic_score_matches_forward(self):
        q = Quant().eval()
        p = rx.Projection(4, 2, torch.eye(2), feature_channels=2)
        op = rx.ScaleOperator(q, p, 1)
        tokens = torch.tensor([[0, 1, 2, 3]])
        target = torch.randn(p.dimension, generator=torch.Generator().manual_seed(7))
        error = target - op.measured(tokens)[0]
        position, variance = 1, .37
        scores = rx.candidate_scores(op, position, int(tokens[0, position]), error, variance)
        exact = []
        for v in range(7):
            candidate = tokens.clone(); candidate[0, position] = v
            exact.append(-.5 * (target - op.measured(candidate)[0]).square().sum() / variance)
        exact = torch.stack(exact)
        self.assertTrue(torch.allclose(scores - scores[0], exact - exact[0], atol=2e-4, rtol=2e-4))
        self.assertGreater(float(op.bias.abs().sum()), 0.)

    def test_phi_observation_couples_different_tokens(self):
        q = Quant().eval()
        p = rx.Projection(4, 2, torch.eye(2), feature_channels=2)
        op = rx.ScaleOperator(q, p, 1)
        cross = op.columns[0].T @ op.columns[1]
        self.assertGreater(float(cross.abs().max()), .001)

    def test_resource_exclusions_and_paid_padding(self):
        self.assertFalse(rx.resource_record(1024, 8, 32)['feasible'])
        self.assertFalse(rx.resource_record(512, 4, 32)['feasible'])
        self.assertTrue(rx.resource_record(512, 4, 32, '16QAM')['feasible'])
        r = rx.resource_record(512, 6, 8)
        self.assertTrue(r['feasible'])
        self.assertEqual(r['N_header'] + r['N_gain'] + r['N_measurement'] + r['N_padding'] + r['N_digital'], 512)
        self.assertLessEqual(r['effective_prefix_rate'], .9)

    def test_gain_float32_paid_and_analog_energy_exact(self):
        for z in (np.zeros(128), np.random.default_rng(2).normal(size=512), np.ones(72) * 100):
            wave, gain = rx.analog_transmit(z)
            self.assertEqual(gain, float(np.float32(gain)))
            self.assertGreater(gain, 0)
            self.assertTrue(np.isclose(np.square(wave).sum(), 2 * len(wave), atol=1e-8, rtol=1e-12))
            # RX discards final padding; gain must arrive from the paid packet.
            self.assertTrue(np.allclose(wave[:-1].ravel() / gain, z, atol=1e-10))

    def test_invalid_rx_variance_rejected(self):
        q = Quant().eval(); p = rx.Projection(4, 2, torch.eye(2), feature_channels=2)
        op = rx.ScaleOperator(q, p, 0)
        with self.assertRaises(ValueError):
            rx.candidate_scores(op, 0, 0, torch.zeros(p.dimension), 0.)


class RunnerEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common only imports adapters; model loading occurs exclusively in
        # setup(), which these CPU engineering tests never call.
        spec = importlib.util.spec_from_file_location('methods_20261002_m2_runner', HERE / 'm2_runner.py')
        cls.runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.runner
        spec.loader.exec_module(cls.runner)

    def profile(self):
        p = rx.Projection(4, 8, torch.eye(32))
        scales = [np.arange(pn * pn, dtype=np.int64) + 17 for pn in rx.PATCH_NUMS]
        z = np.random.default_rng(21).normal(size=p.dimension)
        return p, scales, z

    def test_actual_gain_header_prefix_roundtrip_both_budgets_phys(self):
        m = self.runner; p, scales, z = self.profile()
        for N in (512, 1024):
            for phy in ('QPSK', '16QAM'):
                wave, ledger = m.actual_transmit(scales, z, N, p, phy)
                event = m.actual_receive(wave, 19, N, p, phy)
                self.assertTrue(event['header_ok'] and event['prefix_crc_ok'] and event['gain_crc_ok'])
                self.assertTrue(event['analog_used'])
                self.assertTrue(np.allclose(event['observation'], z, atol=1e-10))
                self.assertEqual(wave.shape, (N, 2))
                if phy == 'QPSK': self.assertTrue(np.isclose(ledger['E'], 2 * N, atol=1e-8))
                for a, b in zip(event['prefix'], scales[:4]): self.assertTrue(np.array_equal(a, b))

    def test_padding_is_discarded_and_never_gain_side_information(self):
        m = self.runner; p, scales, z = self.profile()
        wave, _ = m.actual_transmit(scales, z, 512, p, 'QPSK')
        a = m.actual_receive(wave, 19, 512, p, 'QPSK')
        wave[-1] = [1e6, -1e6]
        b = m.actual_receive(wave, 19, 512, p, 'QPSK')
        self.assertEqual(a['gain'], b['gain'])
        self.assertTrue(np.array_equal(a['observation'], b['observation']))

    def test_correct_crc_nan_gain_is_rejected(self):
        m = self.runner; p, scales, z = self.profile()
        wave, ledger = m.actual_transmit(scales, z, 512, p, 'QPSK')
        end = 68 + ledger['N_digital']
        wave[end:end + 32] = m.encode_packet(m.gain_bits(float('nan')), 32)['symbols']
        event = m.actual_receive(wave, 19, 512, p, 'QPSK')
        self.assertTrue(event['gain_crc_ok'])
        self.assertFalse(event['gain_fields_valid'])
        self.assertFalse(event['analog_used'])
        self.assertIsNone(event['observation'])

    def test_prefix_failure_discards_analog_preserves_hard_prefix(self):
        m = self.runner; p, scales, z = self.profile()
        wave, ledger = m.actual_transmit(scales, z, 512, p, 'QPSK')
        wave[68:68 + ledger['N_digital']] = 0
        event = m.actual_receive(wave, 19, 512, p, 'QPSK')
        self.assertTrue(event['header_ok'])
        self.assertFalse(event['prefix_crc_ok'])
        self.assertFalse(event['analog_used'])
        self.assertEqual(len(event['prefix']), 4)

    def test_accepted_extreme_gain_discards_nonrepresentable_observation_or_variance(self):
        m = self.runner; p, scales, z = self.profile()
        for gain in (float(np.nextafter(np.float32(0), np.float32(1))), float(np.finfo(np.float32).max)):
            wave, ledger = m.actual_transmit(scales, z, 512, p, 'QPSK')
            end = 68 + ledger['N_digital']
            wave[end:end + 32] = m.encode_packet(m.gain_bits(gain), 32)['symbols']
            event = m.actual_receive(wave, 19, 512, p, 'QPSK')
            self.assertTrue(event['gain_crc_ok'] and event['gain_fields_valid'])
            self.assertFalse(event['gain_numeric_valid'])
            self.assertFalse(event['analog_used'])
            self.assertIsNone(event['observation'])

    def test_header_has_no_class_and_wrong_public_projection_is_rejected(self):
        m = self.runner; p, scales, z = self.profile()
        wave, ledger = m.actual_transmit(scales, z, 512, p, 'QPSK')
        event = m.actual_receive(wave, 19, 512, p, 'QPSK')
        self.assertFalse(event['true_class_sent'])
        self.assertEqual(event['decoded_header_code'], m.header_code(p))
        wrong = rx.Projection(4, 16, torch.eye(32))
        mismatch = m.actual_receive(wave, 19, 512, wrong, 'QPSK')
        self.assertFalse(mismatch['header_ok'])

    def test_method1_status_alone_cannot_authorize_method2(self):
        m = self.runner
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_SYNTHETIC_M2_') as tmp:
            with mock.patch.object(m.c, 'OUT', Path(tmp)):
                m.c.write(Path(tmp) / 'm1_complete.json', dict(status='M1_COMPLETE', synthetic=False, training_updates=0))
                with self.assertRaises(RuntimeError): m.require_m1()

    def test_gate_joint_conditions_and_full_calibration_scope(self):
        m = self.runner
        rows = [dict(projection='g4_c8', metric=metric, mean=mean,
                     lower95=lo, upper95=hi, paired_sources=1000)
                for metric, mean, lo, hi in [('psnr_db', .6, .2, .8),
                    ('lpips_alex', -.03, -.04, -.02), ('dino_cosine', -.009, -.04, .02)]]
        self.assertTrue(m.gate_from_pairs(rows)['passed'])
        rows[0]['lower95'] = -.001
        self.assertFalse(m.gate_from_pairs(rows)['passed'])
        rows[0]['lower95'] = .2; rows[2]['mean'] = -.011
        self.assertFalse(m.gate_from_pairs(rows)['passed'])

    def test_var_static_lambda_selected_independently(self):
        m = self.runner
        rows = [dict(projection='g4_c8', control='UNGUIDED', source_id='SYNTHETIC',
                     psnr_db=20., lpips_alex=.3, dino_cosine=.8)]
        for control, candidates in [('VAR_GUIDED', [(.25, 20.7, .20, .795), (1., 20.3, .19, .8)]),
                                    ('STATIC', [(.25, 20.7, .22, .795), (1., 20.7, .21, .8)])]:
            for lam, psnr, lpips, dino in candidates:
                rows.append(dict(projection='g4_c8', control=control, **{'lambda': lam},
                    psnr_db=psnr, lpips_alex=lpips, dino_cosine=dino))
        self.assertEqual(m.choose_lambda(rows, 'VAR_GUIDED')['g4_c8']['selected_lambda'], .25)
        self.assertEqual(m.choose_lambda(rows, 'STATIC')['g4_c8']['selected_lambda'], 1.)

    def test_method_order_blocks_model_loading_before_m1_receipt(self):
        m = self.runner
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_SYNTHETIC_M2_') as tmp:
            with mock.patch.object(m.c, 'OUT', Path(tmp)), mock.patch.object(m.c, 'setup') as setup:
                with self.assertRaises(RuntimeError): m.main('calibration')
                setup.assert_not_called()

    def test_all_stage_dispatch_and_names_exist_after_import(self):
        m = self.runner; seen = []
        with tempfile.TemporaryDirectory(prefix='ENGINEERING_SYNTHETIC_M2_') as tmp:
            with mock.patch.object(m.c, 'OUT', Path(tmp)), mock.patch.object(m, 'require_m1', return_value='SYNTHETIC'), \
                 mock.patch.object(m.c, 'setup', return_value={}), \
                 mock.patch.object(m, 'calibration', side_effect=lambda loaded: seen.append('calibration')), \
                 mock.patch.object(m, 'evaluation', side_effect=lambda loaded: seen.append('evaluation')), \
                 mock.patch.object(m, 'actual', side_effect=lambda loaded: seen.append('actual')), \
                 mock.patch.object(m, 'timing', side_effect=lambda loaded: seen.append('timing')):
                m.main('all')
        self.assertEqual(seen, ['calibration', 'evaluation', 'actual', 'timing'])


if __name__ == '__main__':
    unittest.main()

