"""CPU checks for missing-data handling and paired source statistics."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import render_comparison as render


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.png = self.base / "fixture.png"
        Image.new("RGB", (256, 256), (40, 80, 120)).save(self.png)
        self.csv = self.base / "frames.csv"
        self.manifest = self.base / "manifest.json"
        self.value = dict(schema="existing_reconstruction_comparison_v1", source_indices=list(render.FIXED), noise_seed=2001,
                          conditions=[dict(N=1024, snr_db=7)], assets=[], metrics={})
        self.rows = []
        for index in render.FIXED:
            for position, method in enumerate(render.METHODS):
                self.value["assets"].append(dict(N=1024, snr_db=7, source_index=index, method=method, status="AVAILABLE",
                                                path=self.png.name, sha256=render.sha(self.png)))
                if method != "original":
                    row = dict(N=1024, snr_db=7, source_index=index, method=method, noise_seed=2001)
                    row.update({metric: index / 100 + position for metric in render.METRICS})
                    self.rows.append(row)
        self.save()

    def save(self):
        render.write_csv(self.csv, self.rows)
        self.value["metrics"] = dict(path=self.csv.name, sha256=render.sha(self.csv))
        render.write_json(self.manifest, self.value)

    def test_complete_population_and_exact_paired_delta(self):
        _, conditions, _, indexed = render.load_bundle(self.manifest)
        summaries, pairs = render.bootstrap_tables(conditions, indexed)
        self.assertEqual(len(summaries), 4 * 13)
        self.assertEqual(len(pairs), 6 * 13)
        paired = next(row for row in pairs if row["method_A"] == "M1" and row["method_B"] == "P")
        self.assertAlmostEqual(paired["mean"], 1.0)
        self.assertAlmostEqual(paired["ci_low"], 1.0)
        self.assertAlmostEqual(paired["ci_high"], 1.0)
        self.assertEqual(paired["n_sources"], 16)

    def test_missing_method_never_gets_fabricated_statistics(self):
        for asset in self.value["assets"]:
            if asset["method"] == "M1":
                asset.update(status="NOT_RUN", reason="This bandwidth was not run.")
                asset.pop("path"); asset.pop("sha256")
        self.rows = [row for row in self.rows if row["method"] != "M1"]
        self.save()
        _, conditions, _, indexed = render.load_bundle(self.manifest)
        summaries, pairs = render.bootstrap_tables(conditions, indexed)
        self.assertEqual(len(summaries), 3 * 13)
        self.assertEqual(len(pairs), 3 * 13)
        self.assertTrue(all(row["method"] != "M1" for row in summaries))

    def test_one_missing_metric_does_not_silently_change_denominator(self):
        self.rows[0]["dreamsim"] = ""
        self.save()
        _, conditions, _, indexed = render.load_bundle(self.manifest)
        summaries, pairs = render.bootstrap_tables(conditions, indexed)
        self.assertFalse(any(row["method"] == "swin" and row["metric"] == "dreamsim" for row in summaries))
        self.assertFalse(any(row["method_B"] == "swin" and row["metric"] == "dreamsim" for row in pairs))

    def test_corrupt_image_rejected(self):
        Image.new("RGB", (256, 256), (0, 0, 0)).save(self.png)
        with self.assertRaisesRegex(RuntimeError, "PNG differs"):
            render.load_bundle(self.manifest)

    def test_missing_row_rejected(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(RuntimeError, "Every measured reconstruction"):
            render.load_bundle(self.manifest)

    def test_image_metric_sha_disagreement_rejected(self):
        self.rows[0]["asset_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(RuntimeError, "different PNG"):
            render.load_bundle(self.manifest)

    def test_invalid_noise_seed_rejected(self):
        self.rows[0]["noise_seed"] = 2002
        self.save()
        with self.assertRaisesRegex(RuntimeError, "noise seed differs"):
            render.load_bundle(self.manifest)

    def test_sources_saved_once_without_resizing(self):
        _, _, assets, _ = render.load_bundle(self.manifest)
        output = self.base / "output"
        output.mkdir()
        self.assertEqual(render.export_assets(output, assets), 80)
        path = output / "images/original/source_00.png"
        self.assertEqual(render.sha(path), render.sha(self.png))
        with Image.open(path) as image:
            self.assertEqual(image.size, (256, 256))

    def test_cache_unavailable_retains_completed_metric_population(self):
        self.value["assets"][-1].update(status="UNAVAILABLE_CACHE", reason="Metrics retained; reconstructed PNG was not saved.")
        self.value["assets"][-1].pop("path"); self.value["assets"][-1].pop("sha256")
        self.save()
        _, conditions, _, indexed = render.load_bundle(self.manifest)
        summaries, pairs = render.bootstrap_tables(conditions, indexed)
        self.assertEqual(len(summaries), 4 * 13)
        self.assertEqual(len(pairs), 6 * 13)


if __name__ == "__main__":
    unittest.main()
