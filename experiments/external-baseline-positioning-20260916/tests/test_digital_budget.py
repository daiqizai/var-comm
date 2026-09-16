from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src")]
from external_positioning.digital_budget import receive, transmit
from var_comm.next_scale_prior import PATCH_NUMS
from var_comm.progressive import transmit_whole
from var_comm.scale_channel import indices_to_bits
from var_comm.whole_entropy import transmit as original_entropy_transmit


class ExistingCodecBudgetTests(unittest.TestCase):
    def test_original_N3060_waveforms_remain_identical(self):
        generator = np.random.default_rng(333)
        source = [generator.integers(0, 4096, size ** 2) for size in PATCH_NUMS]
        for mode in (7, 8, 9):
            raw = indices_to_bits(np.concatenate(source[:mode]))
            signal, ledger = transmit(source, raw, 17, mode, "raw", 3060)
            np.testing.assert_array_equal(signal, transmit_whole(source, 17, mode))
            entropy, _original_ledger = original_entropy_transmit({"payload": raw, "length_field": 0, "attempted_arithmetic_bits": len(raw)}, 17, mode)
            adapted, _ledger = transmit(source, raw, 17, mode, "arithmetic", 3060)
            np.testing.assert_array_equal(entropy, adapted)
            self.assertEqual(ledger["total_energy"], 6120)

    def test_common_budgets_keep_packet_and_energy_invariants(self):
        generator = np.random.default_rng(334)
        source = [generator.integers(0, 4096, size ** 2) for size in PATCH_NUMS]
        for budget in (4204, 4498):
            for family in ("raw", "arithmetic"):
                for mode in (7, 8, 9):
                    payload = indices_to_bits(np.concatenate(source[:mode]))
                    signal, ledger = transmit(source, payload, 25, mode, family, budget)
                    physical = receive(signal, 19., family)
                    self.assertTrue(physical["header"]["accepted"])
                    self.assertTrue(physical["body_crc_accepted"])
                    np.testing.assert_array_equal(physical["payload"], payload)
                    self.assertEqual(signal.shape, (budget, 2))
                    self.assertEqual(ledger["total_energy"], 2 * budget)


if __name__ == "__main__":
    unittest.main()
