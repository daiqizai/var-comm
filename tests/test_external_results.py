from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from reproduce_external_results import RESULTS, check_calibration, load_public_rows, read_csv


class ExternalResultsTests(unittest.TestCase):
    def test_complete_populations_not_partial_hifi(self):
        rows, sources = load_public_rows()
        self.assertEqual(len(rows), 25500)
        self.assertEqual(len(sources), 100)
        self.assertFalse(any("hifi" in row["method"] for row in rows))

    def test_calibration_reproduces_frozen_modes(self):
        self.assertEqual(check_calibration(), 18000)

    def test_unequal_resources_not_labelled_equal(self):
        for row in read_csv(RESULTS / "analysis/paired.csv"):
            self.assertEqual(row["equal_resource"] == "True", int(row["method_complex_uses"]) == int(row["control_complex_uses"]))

    def test_curated_comparisons_not_raw_dataset(self):
        directory = RESULTS / "figures/preselected_images"
        self.assertEqual(len(list(directory.glob("equal_N4498_*.png"))), 12)
        self.assertEqual(len(list(directory.glob("comparison_*.png"))), 12)
        self.assertEqual(len(list(directory.glob("*.png"))), 120)
        self.assertFalse(list(directory.glob("source_*.png")))
        self.assertFalse(list(RESULTS.rglob("*.npz")))
        self.assertFalse(list(RESULTS.rglob("*.npy")))


if __name__ == "__main__":
    unittest.main()
