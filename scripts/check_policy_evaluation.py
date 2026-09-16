#!/usr/bin/env python3
"""Quality aggregation rejects unpaired sources/resources and does not refit mode rules."""

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_communication_policies import cube


class PolicyEvaluationTests(unittest.TestCase):
    def example(self):
        return {"image_index": 0, "image_id": "source", "snr_db": 1., "seed": 2001,
                "complex_uses": 3060, "total_energy": 6120., "psnr_db": 20., "lpips": .2, "dino": .8}

    def test_valid_single_source_cube(self):
        result = cube([self.example()], ["source"], [1.], [2001], "fixed")
        self.assertEqual(result.shape, (1, 1, 1, 3))
        self.assertEqual(float(result[0, 0, 0, 1]), .2)

    def test_duplicate_or_missing_noise_is_rejected(self):
        row = self.example()
        with self.assertRaises(RuntimeError):
            cube([row, row], ["source"], [1.], [2001], "fixed")
        with self.assertRaises(RuntimeError):
            cube([row], ["source"], [1.], [2001, 2002], "fixed")

    def test_changed_source_is_rejected(self):
        with self.assertRaises(RuntimeError):
            cube([self.example()], ["different-source"], [1.], [2001], "fixed")

    def test_unequal_budget_is_rejected(self):
        row = self.example()
        row["complex_uses"] = 9856
        with self.assertRaises(RuntimeError):
            cube([row], ["source"], [1.], [2001], "fixed")
        row["complex_uses"], row["total_energy"] = 3060, 8000.
        with self.assertRaises(RuntimeError):
            cube([row], ["source"], [1.], [2001], "fixed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
