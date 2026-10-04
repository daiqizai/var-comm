"""Static publication checks using synthetic scalars; no models or GPU."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_report as r


def policies(k=32):
    cells = []
    for phy in r.PHYS:
        for snr in r.SNRS:
            for method in ("whole", "raster", "random", "entropy", "oracle"):
                q = 0 if method == "whole" else k
                cells.append(dict(phy_family=phy, snr_db=snr, method=method,
                    action=dict(N=2048, phy=phy, m=6, q=q, order="whole" if q == 0 else method)))
    return dict(sources=1000, noise_seeds=[4101, 4102, 4103], full_grid=True, development_read=False, cells=cells)


def summary_row(n, phy, snr, method, metric, value=0.5):
    return dict(N=n, phy_family=phy, snr_db=snr, method=method, metric=metric,
                mean=value, ci_low=value-.01, ci_high=value+.01, n_sources=100, n_frames=300)


class ReportTests(unittest.TestCase):
    def test_missing_resource_point_stays_a_gap(self):
        index = r.summary_index([summary_row(n, "QPSK", 7, "entropy_policy", "lpips_alex", n/10000)
                                 for n in (512, 2048)])
        x, y, lo, hi = r.measured_curve(index, "QPSK", 7, "entropy_policy", "lpips_alex")
        self.assertEqual(x.tolist(), [512, 1024, 2048])
        self.assertTrue(np.isnan(y[1]) and np.isnan(lo[1]) and np.isnan(hi[1]))
        self.assertEqual(y[0], .0512)
        self.assertEqual(y[2], .2048)

    def test_duplicate_or_incomplete_metric_group_is_rejected(self):
        row = summary_row(2048, "QPSK", 7, "entropy_policy", "psnr_db")
        with self.assertRaises(RuntimeError):
            r.summary_index([row, row])
        with self.assertRaises(RuntimeError):
            r.summary_index([dict(row, n_sources=99)])

    def test_kzero_same_k_order_is_canonical_whole(self):
        p = r.policy_index(policies(k=0))
        for method in ("raster_at_entropy", "random_at_entropy", "oracle_at_entropy"):
            action = r.selected_action(p, "QPSK", 7, method)
            self.assertEqual(action["order"], "whole")
            self.assertEqual(action["q"], 0)

    def test_main_table_preserves_all_methods_and_missing_old_metrics(self):
        rows = [summary_row(2048, phy, snr, method, metric, 4096 if metric == "E" else .5)
                for phy in r.PHYS for snr in r.SNRS for method in r.METHODS for metric in r.METRICS+("E",)]
        controls = [summary_row(2048, "continuous", snr, "P2048", metric, 4096 if metric == "E" else .5)
                    for snr in r.SNRS for metric in ("psnr_db", "lpips_alex", "dinov2_vitl14_cosine", "E")]
        result = r.main_rows(r.summary_index(rows), r.summary_index(controls), r.policy_index(policies(k=0)))
        self.assertEqual(len(result), 85)
        self.assertTrue(all(x["same_K_degenerate"] for x in result if x["method"].endswith("_at_entropy")))
        self.assertTrue(all("confidently_wrong_mean" not in x for x in result if x["method"] == "P2048"))
        rows[0] = dict(rows[0], mean=0.6)
        self.assertEqual(len(result), 85)

    def test_qpsk_energy_mismatch_is_rejected(self):
        rows = [summary_row(2048, phy, snr, method, metric, 3000 if metric == "E" else .5)
                for phy in r.PHYS for snr in r.SNRS for method in r.METHODS for metric in r.METRICS+("E",)]
        with self.assertRaises(RuntimeError):
            r.main_rows(r.summary_index(rows), {}, r.policy_index(policies()))

    def test_cost_summary_does_not_report_offline_rows_as_online_latency(self):
        benchmark = dict(sources=2, physical_frames=4140, mean_source_seconds=180)
        checkpoints = [dict(rows=[{}]*240, seconds=60, unique_receiver_events=20, unique_quality_images=18)]*100
        cost = r.cost_summary(benchmark, checkpoints)
        self.assertEqual(cost["development_saved_source_grid_seconds_sum"], 6000)
        self.assertEqual(cost["development_saved_source_grid_seconds_mean"], 60)
        self.assertEqual(cost["development_method_rows"], 24000)
        self.assertEqual(cost["online_single_frame_latency"], "NOT_MEASURED")
        self.assertFalse(cost["latency_division_by_method_rows_permitted"])

    def test_tradeoff_text_keeps_interval_overlap_distinct(self):
        rows = []
        for phy in r.PHYS:
            for snr in r.SNRS:
                for metric in ("dinov2_vitl14_cosine", "psnr_db", "lpips_alex"):
                    rows.append(dict(N=2048, phy_family=phy, snr_db=snr,
                        method_A="entropy_policy", method_B="P2048", metric=metric,
                        mean=.1 if snr == 7 else 0, ci_low=.05 if snr == 7 else -.05,
                        ci_high=.15 if snr == 7 else .05))
        text = r.tradeoff_sentences(rows)
        self.assertEqual(len(text), 2)
        self.assertTrue(all("为7 dB" in line and "其余 4 个区间" in line for line in text))

    def test_publication_file_limit_and_rgb_hash_domain(self):
        rgb = np.zeros((3, 256, 256), dtype=np.float32)
        self.assertEqual(r.rgb_sha(rgb), hashlib.sha256(b"float32:3,256,256:RGB\0"+rgb.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            builder = r.Builder(base, base/"out", base/"result")
            output = base/"result/file.bin"
            with output.open("wb") as f:
                f.truncate(r.LIMIT)
            with self.assertRaises(RuntimeError):
                builder.output(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
