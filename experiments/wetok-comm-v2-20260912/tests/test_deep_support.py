import inspect
from pathlib import Path
import sys
import unittest

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'src'))

from wetok_comm.deep_support import SUPPORT_NAME, FrozenDeepSupport, add_actual_noise, supplement_statistics
import test_interface_evaluation as fixture_module


class DeepSupportTests(unittest.TestCase):
    def test_channel_uses_actual_not_nearest_condition_snr(self):
        signal = torch.ones(1, 3060, 2)
        noise = torch.ones_like(signal)
        for snr, condition in ((5., 4.), (6., 7.)):
            received = add_actual_noise(signal, noise, snr)
            torch.testing.assert_close(received - signal, noise * 10 ** (-snr / 20), atol=1e-7, rtol=0)
            self.assertGreater(float((received - signal - noise * 10 ** (-condition / 20)).abs().max()), .01)
        for wrong in (4., 7.):
            with self.assertRaises(ValueError):
                add_actual_noise(signal, noise, wrong)
        self.assertEqual(list(inspect.signature(FrozenDeepSupport.receive).parameters), ['self', 'received', 'actual_snr'])

    def test_supplement_is_separate_and_all_600_observations_are_paired(self):
        fixture_module.InterfaceEvaluationTests.setUpClass()
        fixtures = fixture_module.InterfaceEvaluationTests
        support = []
        for row in fixtures.rows:
            if row['arm'] == 'perceptual_deepjscc' and row['snr_db'] in (5., 6.):
                support.append({**row, 'arm': SUPPORT_NAME, 'condition_snr_db': {5.: 4., 6.: 7.}[row['snr_db']]})
        self.assertEqual(len(support), 600)
        summary, paired = supplement_statistics(fixtures.rows, support, fixtures.study, fixtures.base)
        self.assertEqual(len(summary), 30)
        self.assertEqual(len(paired), 162)
        with self.assertRaises(RuntimeError):
            supplement_statistics(fixtures.rows, support[:-1], fixtures.study, fixtures.base)
        wrong = [dict(support[0], noise_sha256='wrong_noise'), *support[1:]]
        with self.assertRaises(RuntimeError):
            supplement_statistics(fixtures.rows, wrong, fixtures.study, fixtures.base)


if __name__ == '__main__':
    unittest.main()
