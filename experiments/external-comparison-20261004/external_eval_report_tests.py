"""CPU engineering tests. Synthetic fixtures never enter scientific outputs."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import external_eval_report as report
from external_eval_common import atomic_npz, rgb_sha, write


def fixture_rows():
    result = []
    for i in range(100):
        for n in report.BUDGETS:
            for snr in report.SNRS:
                for seed in report.SEEDS:
                    for method in report.METHODS:
                        result.append(dict(source_index=i, N=n, snr_db=snr, noise_seed=seed, method=method,
                            synthetic=False, label_conditioned=False, audit_used_to_control_receiver=False,
                            E=2*n, actual_energy=2*n, observed_sha256="same_measured_waveform",
                            header_accepted=True, complete_posterior_schedule=method==report.METHODS[1],
                            NFE=250 if method==report.METHODS[1] else 0, t_start=250, fallback="",
                            TX_seconds=.01, RX_seconds=10 if method==report.METHODS[1] else .1,
                            offline_false_accept=False, **{name: .5 for name in report.METRIC_LABELS}))
    return result


class ExternalReportTests(unittest.TestCase):
    def test_full_inventory_rejects_missing_changed_and_truncated_pairs(self):
        rows = fixture_rows()
        self.assertEqual(len(report.validate_inventory(rows)), 3600)
        with self.assertRaises(RuntimeError): report.validate_inventory(rows[:-1])
        for field, value in (("observed_sha256", "different"), ("NFE", 2), ("complete_posterior_schedule", False)):
            changed = copy.deepcopy(rows); changed[1][field] = value
            with self.assertRaises(RuntimeError): report.validate_inventory(changed)

    def test_cost_keeps_gray_and_generation_cost_separate(self):
        rows = fixture_rows()
        for row in rows:
            if row["source_index"] == 0:
                row.update(header_accepted=False, RX_seconds=.001, NFE=0, fallback="fixed_gray_0.5")
        values = report.timing_table(rows)
        self.assertEqual(len(values), 12)
        for row in values:
            self.assertEqual(row["header_rejected_frames"], 3)
            self.assertEqual(row["accepted_frames"], 297)
            self.assertAlmostEqual(row["header_reject_rate"], .01)
            if row["method"] == report.METHODS[1]:
                self.assertAlmostEqual(row["RX_mean_accepted_seconds"], 10)
                self.assertLess(row["RX_mean_seconds"], 10)

    def test_fixed_source_selection_checks_original_identity(self):
        records = [dict(source_index=i, image_id=str(i), class_index=i, preprocessing_id=str(i)*3) for i in range(100)]
        fixed = dict(status="FROZEN_FIXED_EXAMPLES", source_indices=list(report.FIXED),
                     records=[records[i].copy() for i in report.FIXED])
        self.assertEqual(report.fixed_selection(fixed, records), report.FIXED)
        fixed["records"][0]["preprocessing_id"] = "changed"
        with self.assertRaises(RuntimeError): report.fixed_selection(fixed, records)

    def test_summary_requires_all_metrics_and_paired_population(self):
        rows = [dict(N=n, snr_db=snr, method=m, metric=metric, n_sources=100, n_frames=300,
                     mean=.5, ci_low=.4, ci_high=.6)
                for n in report.BUDGETS for snr in report.SNRS for m in report.METHODS for metric in report.METRIC_LABELS]
        self.assertEqual(len(report.summary_index(rows)), 156)
        with self.assertRaises(RuntimeError): report.summary_index(rows[:-1])
        rows[0]["n_sources"] = 99
        with self.assertRaises(RuntimeError): report.summary_index(rows)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required for the CPU rendering integration test")
    def test_engineering_figure_pages_keep_all_fixed_sources(self):
        # Deliberately scoped to one N/SNR and temporary synthetic caches.
        # Real CLI has no corresponding reduced-coverage option.
        with tempfile.TemporaryDirectory(prefix="ENGINEERING_REPORT_") as temporary:
            root = Path(temporary)
            output = root/"figures"; output.mkdir()
            indexed = {}
            axis = np.linspace(0, 1, 256, dtype=np.float32)
            x, y = np.meshgrid(axis, axis)
            images = np.stack((np.stack((x,y,x*.5)),np.stack((y,x,x*.2)),np.stack((x*.2,y,x))))
            for index in report.FIXED:
                archive = root/f"{index}.npz"
                atomic_npz(archive, images=images[1:], source_rgb=images[0], image_slots=np.array([0,1]))
                rows = []
                for j, method in enumerate(report.METHODS):
                    row = dict(source_index=index, N=1024, snr_db=7, noise_seed=2001, method=method,
                        psnr_db=22.22, lpips_alex=.222, dinov2_vitl14_cosine=.777,
                        resnet50_top1_source_prediction=j==0, resnet50_prediction=100+j,
                        resnet50_source_prediction=100, true_class_index=100, resnet50_top1_probability=.876,
                        header_accepted=True, image_sha256=rgb_sha(images[1+j]), reference_sha256=rgb_sha(images[0]),
                        observed_sha256="ENGINEERING", replay_row_id=f"ENGINEERING_{index}_{method}")
                    rows.append(row); indexed[report.frame_identity(row)] = row
                write(root/"source_checkpoints"/f"{index:04d}.json", dict(rows=rows, float_reconstructions=dict(path=str(archive))))
            with patch.object(report, "BUDGETS", (1024,)), patch.object(report, "SNRS", (7,)), patch.object(report, "validate_source", side_effect=lambda value,*_:value):
                figures, cells = report.plot_figures(dict(output=str(root)), output, indexed, {"binding":"ENGINEERING"}, {},
                    ["engineering long class label example"]*1000, lambda:False, lambda *a,**k:None)
            self.assertEqual(len(figures),1)
            self.assertEqual(len(figures[0]["png_pages"]),4)
            self.assertEqual(len(cells),32)
            self.assertEqual({cell["source_index"] for cell in cells}, set(report.FIXED))
            self.assertGreater(Path(figures[0]["pdf"]).stat().st_size,1000)


if __name__ == "__main__":
    unittest.main()
