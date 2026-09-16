#!/usr/bin/env python3
"""Decision rules are fixed functions of calibration summaries, never a per-image oracle."""

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from var_comm.mode_policies import choose_modes, selected_mode, summarize_candidates


def candidates():
    return [{"mode": mode, "source_packet_BLER": 1 - success, "accepted_correct_probability": success,
             "source_index_goodput": bits * success, "psnr_db": psnr, "lpips": lpips}
            for mode, success, bits, psnr, lpips in ((7, 1., 1860, 18., .24), (8, .98, 3060, 20., .18), (9, .70, 5088, 21., .14))]


class ModePoliciesTests(unittest.TestCase):
    def test_conventional_control_is_not_always_smallest_prefix(self):
        modes = choose_modes(candidates())
        self.assertEqual(modes["reliability"], 8)
        self.assertEqual(modes["quality"], 9)
        self.assertEqual(modes["goodput"], 9)

    def test_all_reliable_selects_largest_prefix(self):
        values = candidates()
        values[2]["source_packet_BLER"] = .01
        values[2]["accepted_correct_probability"] = .99
        self.assertEqual(choose_modes(values)["reliability"], 9)

    def test_PSNR_guard_is_enforced(self):
        values = candidates()
        values[2]["psnr_db"] = 19.74
        self.assertEqual(choose_modes(values)["quality"], 8)

    def test_no_qualifying_mode_uses_maximum_success(self):
        values = candidates()
        for row, success in zip(values, (.8, .3, .1)):
            row["source_packet_BLER"], row["accepted_correct_probability"] = 1 - success, success
        self.assertEqual(choose_modes(values)["reliability"], 7)

    def test_ties_choose_smaller_mode(self):
        values = candidates()
        for row in values:
            row["lpips"], row["psnr_db"] = .2, 20.
        self.assertEqual(choose_modes(values)["quality"], 7)

    def test_lookup_uses_only_SNR_and_fixed_midpoints(self):
        actions = {"raw": {"quality": {"1.0": 7, "4.0": 8, "7.0": 9}}}
        self.assertEqual(selected_mode(actions, "raw", "quality", 2.5), 7)
        self.assertEqual(selected_mode(actions, "raw", "quality", 2.51), 8)

    def test_no_development_fitting(self):
        with self.assertRaises(ValueError):
            summarize_candidates([{"population": "development"}])

    def test_candidate_set_cannot_expand(self):
        with self.assertRaises(ValueError):
            choose_modes(candidates() + [dict(candidates()[0], mode=10)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
