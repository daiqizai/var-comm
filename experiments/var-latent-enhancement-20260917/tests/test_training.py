import unittest

import numpy as np
import torch

from latent_enhancement.training import PairedOrder, calibration_summary, draw_mixture, mixed_input, monitor, paired_interval


class TrainingTests(unittest.TestCase):
    def test_resume_source_order_across_epochs(self):
        original = PairedOrder(19, 51)
        original.next(16)
        state = original.state_dict()
        expected = [original.next(16) for _ in range(4)]
        restored = PairedOrder(19, 0)
        restored.load_state_dict(state)
        for indices in expected:
            self.assertTrue(torch.equal(indices, restored.next(16)))

    def test_mixture_exact_and_resume(self):
        generator = torch.Generator().manual_seed(8)
        state = generator.get_state()
        branches, alpha = draw_mixture(16, generator)
        self.assertEqual(torch.bincount(branches).tolist(), [8, 4, 4])
        generator.set_state(state)
        repeated_branches, repeated_alpha = draw_mixture(16, generator)
        self.assertTrue(torch.equal(branches, repeated_branches))
        self.assertTrue(torch.equal(alpha, repeated_alpha))
        batch = {"F": torch.ones(16, 3, 2, 2), "Fq": torch.full((16, 3, 2, 2), 0.2), "Fb_TX": torch.zeros(16, 3, 2, 2)}
        values = mixed_input(batch, branches, alpha)[:, 0, 0, 0]
        torch.testing.assert_close(values[branches == 2], alpha[branches == 2])

    def test_monitor_requires_all_relevant_curves_to_plateau(self):
        summary = calibration_summary({name: np.array([[0.1, 0.2], [0.2, 0.1]]) for name in ("F", "Fq", "Fb_TX", "interpolation")})
        tolerances = {"relative_utility_improvement": 0.001, "PSNR_improvement_db": 0.01, "LPIPS_improvement": 0.0001}
        best, improved = monitor(summary, {}, tolerances)
        self.assertTrue(improved)
        _, improved = monitor(summary, best, tolerances)
        self.assertFalse(improved)
        summary["Fb_TX"]["lpips"] -= 0.001
        _, improved = monitor(summary, best, tolerances)
        self.assertTrue(improved)

    def test_paired_constant_difference(self):
        interval = paired_interval(np.ones(12) * 0.4, resamples=1000)
        self.assertAlmostEqual(interval["low95"], 0.4)
        self.assertAlmostEqual(interval["high95"], 0.4)


if __name__ == "__main__":
    unittest.main()
