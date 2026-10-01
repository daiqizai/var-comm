"""Scientific-unit and fail-closed completeness tests using synthetic CPU data."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("october2_analysis_under_test", Path(__file__).with_name("analysis.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def observation(i, seed, method="entropy_policy", N=512, phy="QPSK", snr=7, **changes):
    row = dict(source_id=f"source_{i}", source_index=i, preprocessing_id=f"preprocess_{i}",
               N=N, phy_family=phy, snr_db=snr, noise_seed=seed, method=method,
               action_id=method, m=5, q=8, order=method, psnr_db=20 + i,
               lpips_alex=.4 - .01 * i, dino_cosine=.7 + .01 * i, dino_mismatched=.2,
               latent_valid=True, latent_sq_err_final=100 + i, E=2 * N,
               header_ok=True, prefix_crc_ok=True, partial_crc_ok=True,
               partial_used=True, prefix_false_accept=False)
    row.update(changes)
    return row


def population(n=3, method="entropy_policy", **changes):
    return [observation(i, seed, method, **changes) for i in range(n) for seed in a.DEV_SEEDS]


def m1_grid(n=3):
    rows = []
    for method in (*a.M1_REQUIRED, "P512"):
        rr = population(n, method)
        for row in rr:
            if method.endswith("at_entropy"):
                row["order"] = method.removesuffix("_at_entropy")
            if method in ("legacy_policy", "P512"):
                row["historical_reference"] = True
            if method == "P512":
                for key in ("m", "q", "order", "header_ok", "prefix_crc_ok", "partial_crc_ok", "partial_used"):
                    row[key] = ""
        rows.extend(rr)
    return rows


class StatisticalTests(unittest.TestCase):
    def test_noise_mean_precedes_source_mean_with_conditional_F(self):
        rows = population(2)
        for row in rows:
            row["latent_sq_err_final"] = 0 if row["source_index"] == 0 else 10
            if row["source_index"] == 0 and row["noise_seed"] != 2001:
                row["latent_valid"] = False
                row["latent_sq_err_final"] = ""
        values, frames = a.source_values(rows, "F_sq_error_valid", ["source_0", "source_1"])
        self.assertEqual(frames, 4)
        self.assertEqual(np.mean(list(values.values())), 5)
        self.assertNotEqual(np.mean(list(values.values())), 7.5)

    def test_paired_F_uses_common_valid_noise_and_reports_denominator(self):
        A, B = population(2), population(2, "whole_policy")
        A[0]["latent_sq_err_final"] = 120
        B[0]["latent_sq_err_final"] = 100
        A[1]["latent_valid"] = False
        A[1]["latent_sq_err_final"] = ""
        B[2]["latent_valid"] = False
        B[2]["latent_sq_err_final"] = ""
        result = a.paired(A, B, ["source_0", "source_1"], a.Bootstrap(1000), {})
        value = next(r for r in result if r["metric"] == "F_sq_error_valid")
        self.assertEqual(value["n_frames_paired"], 4)
        self.assertEqual(value["n_sources_paired"], 2)
        self.assertEqual(value["delta_mean"], 10)
        quality = next(r for r in result if r["metric"] == "psnr_db")
        self.assertEqual(quality["n_frames_paired"], 6)

    def test_specificity_is_paired_difference_in_difference(self):
        A, B = population(1), population(1)
        for row in A:
            row.update(dino_cosine=.8, dino_mismatched=.5)
        for row in B:
            row.update(dino_cosine=.7, dino_mismatched=.2)
        result = a.paired(A, B, ["source_0"], a.Bootstrap(100), {})
        self.assertAlmostEqual(next(r["delta_mean"] for r in result if r["metric"] == "dino_specific"), -.2)

    def test_endpoint_zero_rule_applies_to_CI_not_raw_mean(self):
        boot = a.Bootstrap(1000)
        self.assertEqual(boot.interval([1e-13, -1e-13], True), (0., 0.))
        self.assertNotEqual(boot.interval([1e-10, 1e-10], True), (0., 0.))

    def test_shared_resamples_and_seed_are_reproducible(self):
        values = [0, 1, 2, 8]
        b1, b2 = a.Bootstrap(2000), a.Bootstrap(2000)
        self.assertEqual(b1.interval(values), b2.interval(values))
        self.assertEqual(b1.interval(values), b1.interval(values))

    def test_blank_history_event_is_not_zero_failure(self):
        row = observation(0, 2001, header_ok="", prefix_crc_ok="", partial_crc_ok="")
        self.assertIsNone(a.metric(row, "header_fail_rate"))
        self.assertIsNone(a.metric(row, "prefix_crc_fail_given_header"))
        row["header_ok"] = False
        self.assertEqual(a.metric(row, "header_fail_rate"), 1)
        self.assertIsNone(a.metric(row, "prefix_crc_fail_given_header"))

    def test_invalid_F_is_not_zero_erasure_proxy(self):
        row = observation(0, 2001, latent_valid=False, latent_sq_err_final=999)
        self.assertIsNone(a.metric(row, "F_sq_error_valid"))
        self.assertEqual(a.metric(row, "latent_valid_rate"), 0)


class CompletenessTests(unittest.TestCase):
    def validate(self, rows):
        return a.validate_m1(rows, 3, (512,), ("QPSK",), (7,))

    def test_complete_grid_and_optional_oracle_group(self):
        groups, sources = self.validate(m1_grid())
        self.assertEqual(len(sources), 3)
        self.assertEqual(len(groups), 9)

    def test_missing_noise_duplicate_and_preprocessing_fail_closed(self):
        base = m1_grid()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate(base[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate(base + [base[0]])
        changed = copy.deepcopy(base)
        changed[0]["preprocessing_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.validate(changed)

    def test_same_K_control_change_is_rejected(self):
        rows = m1_grid()
        row = next(r for r in rows if r["method"] == "raster_at_entropy")
        row["q"] += 1
        with self.assertRaisesRegex(ValueError, "Same-K"):
            self.validate(rows)

    def test_QPSK_energy_rejected_but_QAM_variation_retained(self):
        rows = m1_grid()
        rows[0]["E"] += 1
        with self.assertRaisesRegex(ValueError, "energy"):
            self.validate(rows)
        for row in rows:
            row["phy_family"] = "16QAM"
        a.validate_m1(rows, 3, (512,), ("16QAM",), (7,))

    def test_nonfinite_quality_is_rejected(self):
        rows = m1_grid()
        rows[0]["psnr_db"] = "inf"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.validate(rows)

    def test_pairing_labels_distinguish_historical_noise(self):
        groups, sources = self.validate(m1_grid())
        rows = a.m1_contrasts(groups, sources, a.Bootstrap(100))
        new = next(r for r in rows if r["method_B"] == "raster_at_entropy")
        old = next(r for r in rows if r["method_B"] == "P512")
        self.assertEqual(new["pairing"], "same_source_shared_physical_noise")
        self.assertEqual(old["pairing"], "same_source_same_nominal_seed_historical_distinct_noise")

    def test_full_cpu_output_in_explicit_synthetic_temp(self):
        original_validate = a.validate_m1
        with tempfile.TemporaryDirectory(prefix="synthetic_oct2_analysis_") as td:
            root = Path(td)
            results, out = root / "synthetic_results", root / "synthetic_outputs"
            results.mkdir()
            a.write_csv(results / "m1_per_frame.csv", m1_grid())
            a.write_json(results / "m1_policy.json", dict(development_read=False, synthetic=True, cells=[]))
            with patch.object(a, "validate_m1", lambda rows: original_validate(rows, 3, (512,), ("QPSK",), (7,))), patch.object(a, "REPLICATES", 200):
                receipt = a.analysis_m1(out, results, allow_synthetic=True)
            self.assertTrue(receipt["synthetic"])
            self.assertFalse(receipt["whole_study_complete"])
            self.assertTrue(receipt["exact_three_noise_pairs_verified"])
            self.assertTrue((results / "m1_report.md").exists())
            self.assertTrue((results / "figures/m1_quality_N512.png").exists())
            self.assertEqual(a.read_json(out / "m1_analysis_completion.json"), receipt)
            for path, digest in receipt["outputs"].items():
                self.assertEqual(a.sha(path), digest)
            self.assertEqual(len(a.read_csv(results / "m1_source_means.csv")) > 0, True)

    def test_formal_analysis_rejects_unregistered_fixture(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_registration_boundary_") as td:
            root = Path(td)
            rows = m1_grid()
            with self.assertRaisesRegex(ValueError, "Actual registered"):
                a.verify_m1_registration(root, root, rows, ["source_0", "source_1", "source_2"])


def m2_grid(n=3, projections=("g8_c32", "g8_c8"), snrs=(7,)):
    rows = []
    for projection in projections:
        g, channels = (int(x[1:]) for x in projection.split("_"))
        for snr in ("clean", *snrs):
            stage = ("full_noiseless" if projection == "g8_c32" else "reduced_noiseless") if snr == "clean" else (
                "full_noisy_infeasible_oracle" if projection == "g8_c32" else "reduced_noisy")
            for control in a.CONTROLS:
                for i in range(n):
                    for seed in (0,) if snr == "clean" else a.DEV_SEEDS:
                        row = observation(i, seed, snr=snr)
                        row.update(stage=stage, projection=projection, g=g, channels=channels, control=control,
                                   N="", E="", oracle_side_information=True, exact_posterior_claim=False,
                                   assumed_prefix_correct=True, assumed_gain_correct=True,
                                   measurement_y_sha256=f"{i}/{projection}/{snr}/{seed}", **{"lambda": .5})
                        row.pop("phy_family")
                        row["psnr_db"] += .6 if control == "VAR_GUIDED" else 0
                        row["lpips_alex"] -= .03 if control == "VAR_GUIDED" else 0
                        rows.append(row)
    return rows


class ResidualTests(unittest.TestCase):
    def validate(self, rows):
        return a.validate_m2_oracle(rows, 3, (7,), ("g8_c32", "g8_c8"))

    def test_oracle_clean_once_and_noisy_three_complete(self):
        rows = m2_grid()
        groups, sources = self.validate(rows)
        self.assertEqual(len(groups), 20)
        self.assertEqual(len(sources), 3)
        self.assertEqual(sum(str(r["snr_db"]) == "clean" for r in rows), 30)
        self.assertEqual(sum(str(r["snr_db"]) != "clean" for r in rows), 90)

    def test_shared_measurement_and_oracle_budget_boundary(self):
        rows = m2_grid()
        changed = copy.deepcopy(rows)
        next(r for r in changed if r["snr_db"] != "clean")["measurement_y_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different noisy"):
            self.validate(changed)
        rows[0]["N"] = 1024
        with self.assertRaises(ValueError):
            self.validate(rows)

    def test_clean_noisy_identity_and_dimensions_rejected(self):
        rows = m2_grid()
        next(r for r in rows if r["snr_db"] != "clean")["preprocessing_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.validate(rows)
        rows = m2_grid()
        rows[0]["channels"] = 16
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.validate(rows)

    def test_all_controls_have_same_measurement_pairing(self):
        groups, sources = self.validate(m2_grid())
        contrasts = a.m2_control_contrasts(groups, sources, a.Bootstrap(200))
        ps = next(r for r in contrasts if r["metric"] == "psnr_db")
        self.assertAlmostEqual(ps["delta_mean"], .6)
        self.assertFalse(ps["actual_paid_link"])
        self.assertTrue(ps["oracle_side_information"])

    def test_full_calibration_gate_independent_hand_calculation(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_gate_audit_") as td:
            root = Path(td)
            results, out = root / "synthetic_results", root / "synthetic_outputs"
            results.mkdir()
            rows = []
            for i in range(1000):
                for seed in (4101, 4102, 4103):
                    for control in ("VAR_GUIDED", "UNGUIDED"):
                        row = observation(i, seed)
                        row.update(projection="g8_c8", control=control, psnr_db=20., lpips_alex=.4, dino_cosine=.7)
                        if control == "VAR_GUIDED":
                            row.update(psnr_db=20.6, lpips_alex=.37, dino_cosine=.695)
                        rows.append(row)
            a.write_csv(out / "m2/calibration/gate_per_frame.csv", rows)
            decisions = [dict(projection="g8_c8", passed=True,
                              delta_psnr=dict(mean=.6, lower95=.6, upper95=.6),
                              delta_lpips=dict(mean=-.03, lower95=-.03, upper95=-.03),
                              delta_dino=dict(mean=-.005, lower95=-.005, upper95=-.005))]
            gate = dict(passed=True, status="PASSED", eligible_projections=["g8_c8"], decisions=decisions)
            a.write_json(results / "m2_gate.json", gate)
            a.write_csv(results / "m2_resource_ledger.csv", [dict(N=1024, phy_family="QPSK", g=8, channels=8, feasible=True)])
            audit = a.audit_m2_gate(out, results, allow_synthetic=True)
            self.assertTrue(audit["independently_recomputed"])
            self.assertEqual(audit["sources"], 1000)
            self.assertEqual(audit["status"], "PASSED")
            gate["decisions"][0]["passed"] = False
            a.write_json(results / "m2_gate.json", gate)
            with self.assertRaisesRegex(ValueError, "decision differs"):
                a.audit_m2_gate(out, results, allow_synthetic=True)

    def test_full_m2_cpu_report_and_oracle_receipt_synthetic(self):
        original_validate = a.validate_m2_oracle
        with tempfile.TemporaryDirectory(prefix="synthetic_oct2_m2_analysis_") as td:
            root = Path(td)
            results, out = root / "synthetic_results", root / "synthetic_outputs"
            results.mkdir()
            a.write_csv(results / "m2_oracle_per_frame.csv", m2_grid())
            a.write_json(results / "m2_gate.json", dict(passed=False, status="SKIPPED_GATE_NOT_MET", synthetic=True))
            a.write_json(results / "m2_oracle_policy.json", dict(synthetic=True))
            with patch.object(a, "validate_m2_oracle", lambda rows: original_validate(rows, 3, (7,), ("g8_c32", "g8_c8"))):
                receipt = a.analysis_m2(out, results, allow_synthetic=True)
            self.assertTrue(receipt["synthetic"])
            self.assertFalse(receipt["whole_study_complete"])
            self.assertEqual(receipt["actual_link_status"], "SKIPPED_GATE_NOT_MET")
            self.assertTrue((results / "m2_report.md").exists())
            self.assertTrue((results / "figures/m2_oracle_damage_ladder.png").exists())

    def test_passed_gate_cannot_complete_without_actual_link(self):
        original_validate = a.validate_m2_oracle
        with tempfile.TemporaryDirectory(prefix="synthetic_gate_waits_real_link_") as td:
            root = Path(td)
            a.write_csv(root / "m2_oracle_per_frame.csv", m2_grid())
            a.write_json(root / "m2_gate.json", dict(passed=True, status="PASSED", synthetic=True))
            with patch.object(a, "validate_m2_oracle", lambda rows: original_validate(rows, 3, (7,), ("g8_c32", "g8_c8"))):
                with self.assertRaisesRegex(ValueError, "requires actual-link"):
                    a.analysis_m2(root, root, allow_synthetic=True)


def actual_fixture(n=3):
    rows = []
    policy = dict(development_read=False, choices=[dict(status="SELECTED", N=512, phy_family="QPSK", snr_db=7, projection="g4_c8")])
    ledger = [dict(N=512, phy_family="QPSK", g=4, channels=8, feasible=True)]
    for i in range(n):
        for seed in a.DEV_SEEDS:
            for control in a.CONTROLS:
                row = observation(i, seed)
                row.update(stage="actual_link", projection="g4_c8", g=4, channels=8, control=control,
                           oracle_side_information=False, offline_truth_never_controls_rx=True,
                           N_header=68, N_gain=32, N_digital=347, N_measurement=64, N_padding=1,
                           effective_prefix_rate=382/694, E_header=136., E_gain=64., E_digital=694., E_analog=130.,
                           waveform_sha256=f"wave_{i}", observation_sha256=f"obs_{i}_{seed}",
                           gain_crc_ok=True, gain_fields_valid=True, analog_used=True, padding_ignored=True)
                rows.append(row)
    return rows, policy, ledger


class FinalEvidenceTests(unittest.TestCase):
    def test_actual_paid_shape_and_five_control_waveform(self):
        rows, policy, ledger = actual_fixture()
        groups = a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)
        self.assertEqual(len(groups), 5)
        rows[0]["observation_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "share waveform"):
            a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)

    def test_actual_paid_resource_and_gate_control_failures(self):
        rows, policy, ledger = actual_fixture()
        rows[0]["N_padding"] = 2
        with self.assertRaisesRegex(ValueError, "resource ledger"):
            a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)
        rows, policy, ledger = actual_fixture()
        rows[0]["gain_crc_ok"] = False
        with self.assertRaisesRegex(ValueError, "Analog used"):
            a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)

    def test_actual_failed_packets_have_identical_outputs(self):
        rows, policy, ledger = actual_fixture()
        for r in rows:
            r["analog_used"] = False
            r["gain_crc_ok"] = False
        a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)
        rows[0]["psnr_db"] += 1
        with self.assertRaisesRegex(ValueError, "same RX output"):
            a.validate_m2_actual(rows, policy, ledger, [f"source_{i}" for i in range(3)], 3)

    def test_wrong_image_mapping_and_pixel_binding_fail_closed(self):
        rows = population()
        for r in rows:
            r["mismatch_source_id"] = f"source_{(r['source_index']+1)%3}"
        a.verify_mismatch_identity(rows, [f"source_{i}" for i in range(3)], True)
        rows[0]["mismatch_source_id"] = rows[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "equals true"):
            a.verify_mismatch_identity(rows, [f"source_{i}" for i in range(3)], True)
        reg = dict(data_bindings=[dict(rgb_sha256=f"preprocess_{i}", source_npz_sha256="a"*64) for i in range(3)])
        a.verify_registered_inputs(reg, rows, [f"source_{i}" for i in range(3)])
        reg["data_bindings"][0]["rgb_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "pixel/preprocessing"):
            a.verify_registered_inputs(reg, rows, [f"source_{i}" for i in range(3)])

    def test_formal_failed_gate_requires_real_skip_receipt(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_skip_evidence_") as td:
            p = Path(td)
            a.write_csv(p / "m2_actual_per_frame.csv", [])
            a.write_json(p / "m2_actual_policy.json", dict(development_read=False, choices=[]))
            a.write_csv(p / "m2_resource_ledger.csv", [])
            with self.assertRaises(FileNotFoundError):
                a.verify_m2_actual_evidence(p, p, dict(passed=False), [], [], False)

    def test_timing_exact_repeats_and_oracle_scope(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_timing_evidence_") as td:
            p = Path(td)
            rows = [dict(scope="oracle_only_ideal_prefix_gain", N="", phy_family="ideal", projection="g4_c8",
                         snr_db=7, control="VAR_GUIDED", source_id=f"s{i}", repeat=rep,
                         no_image_output_cache=True, deployment_endpoint=False, tx_ms=2+rep, rx_ms=4, total_ms=6+rep)
                    for i in range(10) for rep in (0, 1)]
            a.write_csv(p / "m2_timing.csv", rows)
            summary, _ = a.m2_timing(p, p, True)
            self.assertEqual(next(r for r in summary if r["metric"] == "total_ms")["mean"], 6.5)
            rows.pop()
            a.write_csv(p / "m2_timing.csv", rows)
            with self.assertRaisesRegex(ValueError, "paired repeats"):
                a.m2_timing(p, p, True)

    def test_report_copy_relocates_evidence_links(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_report_links_") as td:
            p = Path(td)
            original = p / "results" / "report.md"
            original.parent.mkdir()
            original.write_text("[CSV](table.csv) [Web](https://example.com/p)\n", encoding="utf-8")
            a.copy_report(original, p / "reports", "report.md")
            copied = (p / "reports/report.md").read_text(encoding="utf-8")
            self.assertIn("../results/table.csv", copied)
            self.assertIn("https://example.com/p", copied)


if __name__ == "__main__":
    unittest.main(verbosity=2)
