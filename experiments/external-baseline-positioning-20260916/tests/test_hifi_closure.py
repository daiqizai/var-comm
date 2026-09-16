import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/finish_hifi_positioning.py"
SPEC = importlib.util.spec_from_file_location("hifi_closure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HiFiClosureTests(unittest.TestCase):
    def test_partial_results_cannot_create_final_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(RuntimeError, "full inference"):
                MODULE.finish(argparse.Namespace(root=root, output=output))
            self.assertFalse(output.exists())

    def test_paired_statistics_use_sources_and_fixed_scopes(self):
        controls = [{"image_index": index, "snr_db": snr, "seed": seed, "method": "control",
                     "complex_uses": 4204, "psnr_db": 20. + index / 100, "ssim": .5, "lpips": .2, "dino": .7}
                    for index in range(100) for snr in (1., 4., 7., 13., 19.) for seed in (2001, 2002, 2003)]
        methods = [{**row, "method": "hifi", "psnr_db": row["psnr_db"] + 1, "lpips": .1} for row in controls]
        rows = MODULE.paired_rows(methods, controls, "test", "same_N4204")
        self.assertEqual(len(rows), 28)
        for row in rows:
            if row["metric"] == "psnr_db":
                self.assertAlmostEqual(row["mean_difference"], 1.)
                self.assertAlmostEqual(row["ci_low"], 1.)
                self.assertAlmostEqual(row["ci_high"], 1.)
            if row["metric"] == "lpips":
                self.assertAlmostEqual(row["mean_difference"], -.1)
            self.assertEqual(row["source_images"], 100)

    def test_attribution_requires_full_coverage(self):
        with self.assertRaisesRegex(RuntimeError, "full1500"):
            MODULE.verify_observation_pairing([], [], [])


if __name__ == "__main__":
    unittest.main()
