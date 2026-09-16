import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from external_positioning.analysis import SEEDS, SNRS, source_values, summarize, validate_groups


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.sources = {index: {"image_id": str(index), "source_pixels_sha256": str(index)} for index in range(2)}
        self.rows = [{"method": method, "protocol": "common_paid_information", "complex_uses": budget,
                     "total_energy": 2 * budget, "image_id": str(index), "image_index": index,
                     "source_pixels_sha256": str(index), "snr_db": snr, "seed": seed,
                     "psnr_db": 20 + offset + index, "lpips": .2 - offset / 100,
                     "ssim": .5, "dino": .7}
                    for method, budget, offset in (("raw_adaptive", 3060, 0), ("swin_ra32", 4498, 1))
                    for index in self.sources for snr in SNRS for seed in SEEDS]

    def test_requires_complete_population(self):
        with self.assertRaises(ValueError):
            validate_groups(self.rows[:-1], self.sources)

    def test_pairing_uses_source_and_labels_unequal_resources(self):
        summary, paired, sources = summarize(self.rows, self.sources, resamples=20)
        selected = next(row for row in paired if row["metric"] == "psnr_db" and row["scope"] == "primary_1_4_7")
        self.assertEqual(selected["mean_method_minus_control"], 1.)
        self.assertEqual(selected["ci_low"], 1.)
        self.assertFalse(selected["equal_resource"])
        self.assertEqual(selected["source_images"], 2)
        self.assertEqual(len(summary), 14)
        self.assertEqual(len(sources), 28)

    def test_rejects_source_leak_and_energy_mismatch(self):
        self.rows[0]["source_pixels_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            validate_groups(self.rows, self.sources)
        self.rows[0]["source_pixels_sha256"] = "0"
        self.rows[0]["total_energy"] = 3060
        with self.assertRaises(ValueError):
            validate_groups(self.rows, self.sources)


if __name__ == "__main__":
    unittest.main()
