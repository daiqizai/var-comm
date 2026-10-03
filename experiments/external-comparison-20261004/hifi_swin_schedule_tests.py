"""CPU-only schedule tests; require numpy, not torch or model checkpoints."""
import math
import unittest
import numpy as np
from hifi_swin_schedule import sampling_schedule


class HiFiScheduleTests(unittest.TestCase):
    def setUp(self):
        betas = np.linspace(.0001, .02, 1000, dtype=np.float32)
        alphas = np.cumprod(1 - betas, axis=0)
        self.levels = np.log10(alphas / (1 - alphas))

    def test_data_4096_reproduces_author_start_and_full_sequence(self):
        # Original author expressions, evaluated independently over a broad SNR
        # range. Exact discrete equality matters; DSNR floats may differ by ulps.
        for snr in [-5, -2, 0, 1, 4, 7, 10, 13, 19, 30]:
            author_dsnr = np.log10((1 + 10 ** (snr / 10)) ** (1 / 48)) * 5
            author_start = int(1000 - np.searchsorted(self.levels[::-1], author_dsnr))
            result = sampling_schedule(self.levels, snr, 4096)
            self.assertEqual(result["t_start"], author_start)
            self.assertEqual(result["seq"], list(range(author_start))[::-1])

    def test_actual_data_budgets_follow_registered_formula(self):
        for data_n in (768, 1664):
            for snr in (1, 4, 7, 10, 13):
                result = sampling_schedule(self.levels, snr, data_n)
                expected_dsnr = 5 * data_n / (3 * 256 * 256) * math.log10(1 + 10 ** (snr / 10))
                expected_start = int(1000 - np.searchsorted(self.levels[::-1], expected_dsnr))
                self.assertAlmostEqual(result["schedule_dsnr"], expected_dsnr, places=15)
                self.assertEqual(result["t_start"], expected_start)
                self.assertEqual(result["seq"], list(range(expected_start))[::-1])
                self.assertEqual(result["schedule_data_N"], data_n)
                self.assertEqual(result["schedule_multiplier"], 1.0)

    def test_fixed_author_mode_is_reference_only_and_labeled(self):
        author = sampling_schedule(self.levels, 7, 768, mode="author_fixed_1_over_48")
        matching_data = sampling_schedule(self.levels, 7, 4096)
        self.assertEqual(author["t_start"], matching_data["t_start"])
        self.assertEqual(author["seq"], matching_data["seq"])
        self.assertEqual(author["schedule_effective_cbr"], 1 / 48)
        self.assertEqual(author["schedule_actual_data_cbr"], 768 / (3 * 256 * 256))

    def test_dimensionless_multiplier_and_iteration_skip_remain_author_settings(self):
        default = sampling_schedule(self.levels, 7, 768)
        result = sampling_schedule(self.levels, 7, 768, multiplier=.5, iter_num=250)
        self.assertEqual(result["t_start"], int(default["t_start"] * .5))
        self.assertEqual(result["seq"], list(range(0, (result["t_start"] // 4) * 4, 4))[::-1])


if __name__ == "__main__":
    unittest.main()
