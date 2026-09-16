#!/usr/bin/env python3
"""CPU checks of measurement boundaries, paired repetition accounting and drift rejection."""

import json
from pathlib import Path
import sys
import types
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from var_comm.online_timing import ARMS, SNRS, compare_arrays, measurement_order, single_waveform, summarize, timed_call, validate_config, validate_rows


def artificial_rows():
    rows = []
    for position, index in enumerate((0, 99)):
        for snr_index, snr in enumerate(SNRS):
            for repeat in range(3):
                for order_index, arm in enumerate(measurement_order(position * 5 + snr_index, repeat)):
                    tx = .01 + ARMS.index(arm) * .001
                    rx = .02 + position * .002
                    rows.append({"image_index": index, "image_id": f"source-{index}", "snr_db": snr, "noise_seed": 2001,
                        "arm": arm, "repeat": repeat, "order_index": order_index, "total_complex_uses": 3060,
                        "total_energy": 6120., "TX_seconds": tx, "RX_seconds": rx, "processing_sum_seconds": tx + rx,
                        "timing_scope": "CPU_to_CPU_contiguous_TX_and_RX"})
    return rows


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/frozen_system_online_timing.json").read_text())

    def test_contract_excludes_training(self):
        validate_config(self.config)
        self.config["new_training"] = True
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_dynamic_module_paths_are_not_files(self):
        from var_comm.frozen_timing_adapters import loaded_local_sources
        module = types.ModuleType("timing_test_dynamic_module")
        module.__file__ = str(ROOT.parent / "_nonexistent_dynamic_timing_module.py")
        sys.modules[module.__name__] = module
        try:
            paths = loaded_local_sources()
            self.assertNotIn(Path(module.__file__), paths)
            self.assertTrue(all(path.is_file() for path in paths))
        finally:
            del sys.modules[module.__name__]

    def test_corrected_paired_digital_archive_is_used(self):
        from var_comm.frozen_timing_adapters import WETOK_DIGITAL
        path = ROOT / "experiments/wetok-comm-v2-20260912/docs/digital_noise_pairing_correction.json"
        correction = json.loads(path.read_text())
        self.assertEqual(WETOK_DIGITAL, ROOT.parent / correction["corrected_output"])
        self.assertNotEqual(WETOK_DIGITAL, ROOT.parent / correction["preserved_output"])

    def test_single_batch_archive_preserves_observation_bytes(self):
        batched = np.arange(6120, dtype=np.float32).reshape(1, 3060, 2)
        frame = single_waveform(batched)
        self.assertEqual(frame.shape, (3060, 2))
        self.assertEqual(frame.tobytes(), batched.tobytes())
        with self.assertRaises(ValueError):
            single_waveform(np.zeros((2, 3060, 2)))

    def test_synchronization_encloses_callback(self):
        events = []
        ticks = iter((10., 10.5))
        def clock():
            events.append("clock")
            return next(ticks)
        def callback():
            events.append("callback")
            return np.ones(1)
        value, elapsed = timed_call(callback, lambda: events.append("sync"), clock)
        self.assertEqual(events, ["sync", "clock", "callback", "sync", "clock"])
        self.assertEqual(elapsed, .5)
        self.assertEqual(value.shape, (1,))

    def test_no_repeated_position_bias(self):
        orders = [measurement_order(frame, 0) for frame in range(7)]
        self.assertEqual(len({order[0] for order in orders}), 7)
        self.assertTrue(all(set(order) == set(ARMS) for order in orders))

    def test_exact_and_tolerant_array_checks(self):
        original = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(compare_arrays(original, original, 0., "RGB"), (0., True))
        changed = original.copy()
        changed[0, 0] = 5e-7
        self.assertFalse(compare_arrays(changed, original, 1e-6, "RGB")[1])
        with self.assertRaises(RuntimeError):
            compare_arrays(changed, original, 0., "digital TX")
        with self.assertRaises(RuntimeError):
            compare_arrays(original.astype(np.float64), original, 1e-6, "precision")

    def test_reject_nonfinite_or_wrong_shape(self):
        with self.assertRaises(RuntimeError):
            compare_arrays(np.array([np.nan]), np.ones(1), 1., "finite")
        with self.assertRaises(RuntimeError):
            compare_arrays(np.ones(2), np.ones(1), 1., "shape")

    def test_matrix_retains_every_repeat(self):
        rows = artificial_rows()
        validate_rows(rows, [0, 99], 3)
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1], [0, 99], 3)
        with self.assertRaises(ValueError):
            validate_rows(rows + rows[:1], [0, 99], 3)

    def test_changed_endpoint_or_energy_rejected(self):
        rows = artificial_rows()
        rows[0]["timing_scope"] = "GPU_only"
        with self.assertRaises(ValueError):
            validate_rows(rows, [0, 99], 3)
        rows = artificial_rows()
        rows[0]["total_energy"] = 6200
        with self.assertRaises(ValueError):
            validate_rows(rows, [0, 99], 3)

    def test_statistics_count_sources_not_repeats(self):
        self.config["bootstrap_resamples"] = 100
        summary, paired = summarize(artificial_rows(), self.config, [0, 99], 3)
        selected = next(row for row in summary if row["scope"] == "all_5_snrs" and row["arm"] == "whole_adaptive")
        self.assertEqual(selected["source_images"], 2)
        self.assertEqual(selected["timed_calls"], 30)
        self.assertAlmostEqual(selected["TX_mean_ms"], 13.)
        contrast = next(row for row in paired if row["scope"] == "primary_1_4_7" and row["control"] == "whole_m8" and row["metric"] == "TX_seconds")
        self.assertAlmostEqual(contrast["gain"], 2.)
        self.assertEqual(contrast["source_images"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
