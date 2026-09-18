import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from reproduce_latent_results import RESULTS, SNRS, aggregate, load_cube, paired, read_rows, verify_results


class LatentResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = load_cube()

    def test_complete_source_noise_averaged_grid(self):
        self.assertEqual(len(self.arrays), 41)
        self.assertTrue(all(values.shape == (100, 5, 3) for values in self.arrays.values()))

    def test_pairing_is_by_source_id_not_csv_order(self):
        reordered = load_cube(list(reversed(read_rows(RESULTS / "per_source_snr.csv"))))
        for method in self.arrays:
            np.testing.assert_array_equal(self.arrays[method], reordered[method])

    def test_duplicate_source_cell_is_rejected(self):
        rows = read_rows(RESULTS / "per_source_snr.csv")
        rows[1] = dict(rows[0])
        with self.assertRaises(ValueError):
            load_cube(rows)

    def test_means_and_paired_intervals_reproduce(self):
        overall, _ = aggregate(self.arrays)
        self.assertLess(verify_results(overall, paired(self.arrays)), 1e-10)

    def test_plot_table_averages_all_sources(self):
        _, per_snr = aggregate(self.arrays)
        for row in per_snr:
            expected = self.arrays[row["method"]][:, SNRS.index(row["snr_db"]), 0].mean()
            self.assertAlmostEqual(row["psnr_db"], expected, places=12)

    def test_timing_has_explicit_diagnostic_status(self):
        status = json.loads((RESULTS / "publication_review.json").read_text())
        self.assertFalse(status["timing_valid_for_method_ranking"])
        self.assertFalse(status["adaptive_and_full_scale_digital_comparison_complete"])


if __name__ == "__main__":
    unittest.main()
