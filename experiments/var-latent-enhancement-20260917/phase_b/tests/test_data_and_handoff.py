import unittest

import numpy as np
import torch

from latent_enhancement_b.data import evaluation_grid, metric_summary, paired_stop, update_monitor
from latent_enhancement_b.pipeline import handoff_decision


class DataAndHandoffTests(unittest.TestCase):
    def test_full_grid_contains_every_source_snr_noise(self):
        sources, snrs, noises = evaluation_grid([3, 7, 10], 5, 3)
        self.assertEqual(len(sources), 45)
        self.assertEqual(len(set(zip(sources.tolist(), snrs.tolist(), noises.tolist()))), 45)
        self.assertEqual(torch.bincount(snrs).tolist(), [9] * 5)

    def test_source_level_mean_not_frame_weighted(self):
        values = np.array([[0.01, 0.1, 0.2], [0.01, 0.1, 0.2], [0.04, 0.3, 0.4]])
        summary = metric_summary(values, np.array([0, 0, 1]), np.array([0, 0, 0]))
        self.assertAlmostEqual(summary['all']['lpips'], 0.2)
        self.assertAlmostEqual(summary['all']['utility'], 0.025 + 0.1 * 0.2 + 0.01 * 0.3)

    def test_paired_stop_waits_for_all_arms_and_minimum(self):
        recipe = {'minimum_updates': 10000}
        state = {'step': 9000, 'plateau_checks': {'first': 3, 'second': 3, 'control': 3}}
        self.assertFalse(paired_stop(state, recipe, 3))
        state['step'] = 15000
        state['plateau_checks']['second'] = 2
        self.assertFalse(paired_stop(state, recipe, 3))
        state['plateau_checks']['second'] = 3
        self.assertTrue(paired_stop(state, recipe, 3))

    def test_any_snr_curve_improvement_prevents_plateau(self):
        summary = metric_summary(np.array([[0.01, 0.1, 0.2], [0.04, 0.3, 0.4]]), np.array([0, 1]), np.array([0, 0]))
        tolerance = {'PSNR_improvement_db': 0.01, 'LPIPS_improvement': 0.0001, 'relative_utility_improvement': 0.001}
        best, improved = update_monitor(summary, {}, tolerance)
        self.assertTrue(improved)
        self.assertFalse(update_monitor(summary, best, tolerance)[1])
        summary['per_snr']['0']['lpips'] -= 0.001
        self.assertTrue(update_monitor(summary, best, tolerance)[1])

    def test_no_handoff_while_A_is_running(self):
        self.assertEqual(handoff_decision({}, True, 'STAGE_A_TRAINING'), 'WAIT_FOR_A')
        self.assertEqual(handoff_decision({}, True, 'WAITING_FOR_AUTHORIZED_GPU0/YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK'), 'WAIT_FOR_A')
        self.assertEqual(handoff_decision({}, False, 'STAGE_A_TRAINING'), 'STOP_A_MISSING_WITHOUT_COMPLETION')
        self.assertEqual(handoff_decision({}, True, 'PAUSED_EXPLICIT_INTERRUPT'), 'STOP_A_FAILED_OR_PAUSED')
        self.assertEqual(handoff_decision({'stage_B_eligible': True}, False, ''), 'CHECK_FINAL_A_QUALIFICATION')


if __name__ == '__main__':
    unittest.main()
