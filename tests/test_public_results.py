import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_release import check_manifest
from reproduce_results import RESULTS, aggregate, assemble_rows, expected_value, read_rows


class PublicResultsTests(unittest.TestCase):
    def test_manifest_keeps_original_and_published_hashes(self):
        manifest = check_manifest()
        self.assertEqual(manifest["management_mode"], "single_project_worktree")
        self.assertTrue(manifest["source_artifacts_modified"])
        self.assertFalse(manifest["model_weights_dataset_pixels_paper_PDFs_and_credentials_included"])
        self.assertTrue(all(row["source_sha256"] and row["published_sha256"] for row in manifest["files"]))

    def test_expected_metric_is_not_two_image_blending(self):
        self.assertEqual(expected_value(18., 26., .25), 20.)
        self.assertEqual(expected_value(18., 26., 0.), 18.)
        self.assertEqual(expected_value(18., 26., 1.), 26.)
        with self.assertRaises(ValueError):
            expected_value(18., 26., 1.1)

    def test_population_and_reference_means_reproduce(self):
        records, identities, error = assemble_rows()
        self.assertEqual(len(records), 27000)
        self.assertEqual(len(identities), 100)
        self.assertEqual(error, 0.)
        summary, _reduced = aggregate(records, identities)
        previous = {(row["region"], row["arm"]): row for row in read_rows(RESULTS / "quality_summary.csv")}
        for row in summary:
            for metric in ("psnr_db", "lpips", "dino", "mse"):
                self.assertAlmostEqual(row[metric], float(previous[row["region"], row["arm"]][metric]), places=10)

    def test_checkpoint_selection_uses_full_calibration_only(self):
        frozen = json.loads((RESULTS / "frozen_comparisons.json").read_text())
        self.assertFalse(frozen["development_used_for_selection_matching_ratio"])
        self.assertFalse(frozen["DINO_used_for_selection_matching_ratio"])
        curves = read_rows(RESULTS / "full_calibration_all_metrics.csv")
        for arm, specification in frozen["models"].items():
            candidates = [row for row in curves if row["arm"] == arm and row["region"] == "primary" and int(row["new_step"]) in (5000, 10000, 15000, 20000)]
            selected = min(candidates, key=lambda row: (float(row["mse"]) + specification["weight"] * float(row["lpips"]), int(row["new_step"])))
            self.assertEqual(int(selected["new_step"]), specification["step"])

    def test_all_failure_events_are_paired_and_retained(self):
        expected = {1.: 300, 4.: 1, 7.: 0, 13.: 0, 19.: 0}
        rows = read_rows(RESULTS / "failure_coverage.csv")
        self.assertEqual(len(rows), 30)
        for row in rows:
            self.assertEqual(int(row["transmissions"]), 300)
            self.assertEqual(int(row["body_crc_failures_with_usable_header"]), expected[float(row["snr_db"])])


if __name__ == "__main__":
    unittest.main()
