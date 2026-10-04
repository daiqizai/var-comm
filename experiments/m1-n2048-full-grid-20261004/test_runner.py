"""CPU protocol/receipt checks; no real-image quality claims are made here."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import m1_common as u
from m1_run import Runner


def actions():
    result = []
    for family in u.FAMILIES:
        for m in range(4, 9):
            result.append(SimpleNamespace(N=2048, phy=family, m=m, q=0, order="whole"))
        for order in ("raster", "random", "entropy", "oracle"):
            for m, counts in ((4, [6, 12, 18, 24]), (5, [9, 18, 27, 35]),
                              (6, [16, 32, 48, 63]), (7, [25, 50, 75, 99])):
                for q in counts:
                    result.append(SimpleNamespace(N=2048, phy=family, m=m, q=q, order=order))
    return result


class RunnerTests(unittest.TestCase):
    def test_complete_task_grid_has_every_physical_event_once(self):
        grid = Runner.task_grid(actions(), u.CAL_SEEDS)
        ids = [(u.action_id(a), snr, seed) for a, snr, seed in grid]
        self.assertEqual(len(ids), 2070)
        self.assertEqual(len(set(ids)), 2070)
        self.assertEqual(sum(a.phy == "QPSK" for a, _, _ in grid), 1035)
        self.assertEqual(sum(a.phy == "16QAM" for a, _, _ in grid), 1035)

    def test_validation_rejects_missing_duplicate_or_wrong_source(self):
        runner = Runner.__new__(Runner)
        aa = actions()
        rows = [dict(phy_family=a.phy, snr_db=snr, action_id=u.action_id(a), noise_seed=seed,
                     source_index=0, source_id="source0", preprocessing_id="pixels0", N=2048)
                for a, snr, seed in Runner.task_grid(aa, u.CAL_SEEDS)]
        data = dict(records=[dict(image_id="source0", preprocessing_id="pixels0")])
        runner.validate_calibration_rows(rows, data, 0, aa)
        with self.assertRaises(RuntimeError):
            runner.validate_calibration_rows(rows[:-1], data, 0, aa)
        with self.assertRaises(RuntimeError):
            runner.validate_calibration_rows(rows[:-1]+[rows[0]], data, 0, aa)
        changed = [dict(row) for row in rows]
        changed[0]["source_id"] = "different"
        with self.assertRaises(RuntimeError):
            runner.validate_calibration_rows(changed, data, 0, aa)

    def test_streaming_calibration_means_keep_failure_definition(self):
        rows = [dict(phy_family="QPSK", snr_db=7, action_id="a", psnr_db=10+i,
                     lpips_alex=0.2+i/10, dino_cosine=0.5, E=4096,
                     header_ok=(i != 0), body_crc_ok=(i != 1)) for i in range(3)]
        accumulated, first = {}, {}
        Runner.aggregate_rows(accumulated, first, rows)
        values = accumulated[("QPSK", 7, "a")]
        self.assertEqual(values["n"], 3)
        self.assertEqual(values["psnr_db"]/3, 11)
        self.assertEqual(values["failure_fraction"]/3, 2/3)
        self.assertEqual(values["E"]/3, 4096)

    def test_source_checkpoint_seal_detects_modified_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"0000.json"
            payload = dict(binding="registered", source_index=0, rows=[dict(score=1.0)])
            u.sealed_checkpoint(path, payload)
            self.assertEqual(u.checkpoint(path, "registered", 0), payload)
            changed = u.read(path)
            changed["rows"][0]["score"] = 2.0
            u.write(path, changed)
            with self.assertRaises(RuntimeError):
                u.checkpoint(path, "registered", 0)

    def test_image_hash_uses_published_float_domain(self):
        image = np.zeros((3, 256, 256), dtype=np.float32)
        expected = hashlib.sha256(b"float32:3,256,256:RGB\0"+image.tobytes()).hexdigest()
        self.assertEqual(u.rgb_sha(image), expected)
        with self.assertRaises(RuntimeError):
            u.rgb_sha(image[:, :128])

    def test_zero_k_same_k_labels_remain_whole(self):
        runner = Runner.__new__(Runner)
        def action(record):
            return SimpleNamespace(**record)
        def construct(N, family, m, q, order):
            return SimpleNamespace(N=N, phy=family, m=m, q=q, order="whole" if q == 0 else order)
        runner.native = SimpleNamespace(action=action, phy=SimpleNamespace(Action=construct))
        runner.load = lambda: runner.native
        cells = {(family, snr, method): dict(action=dict(N=2048, phy=family, m=8, q=0, order="whole"))
                 for family in u.FAMILIES for snr in u.SNRS for method in u.METHODS}
        choices = runner.development_choices(cells)
        self.assertEqual(sum(len(v) for v in choices.values()), 80)
        for methods in choices.values():
            for name in ("raster_at_entropy", "random_at_entropy", "oracle_at_entropy"):
                self.assertEqual(methods[name].order, "whole")
                self.assertEqual(methods[name].q, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
