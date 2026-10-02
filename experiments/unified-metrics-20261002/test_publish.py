"""Synthetic CPU publication checks. No repository commit or push is performed."""
from __future__ import annotations

import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import publish as p
import batch_speed as b


class PublicationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.result = self.root / "results"
        self.out = self.root / "outputs"
        self.source = self.root / "source"
        for folder in (self.result, self.out, self.source):
            folder.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def text(self, folder, name, content="evidence\n"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def parent(self):
        parent = self.root / "parent"
        parent.mkdir()
        record = {"status": "PUSHED", "checks": "PASS", "commit": "a"*40, "remote_commit": "a"*40}
        p.write(parent / "completion.json", {"status": "AUTHORIZED_TWO_METHODS_COMPLETE", "synthetic": False, "training_updates": 0, "stop": True, "publication": record})
        p.write(parent / "supervisor_status.json", {"status": "COMPLETE", "stopped_after_authorized_scope": True, "pid": 100})
        p.write(parent / "supervisor_launch.json", {"pid": 100, "start_ticks": "1234"})
        return parent

    def evidence(self):
        model = self.text(self.out, "modelmanifest.json", "{}\n")
        reg = self.result / "metrics_registration.json"
        p.write(reg, {"modelmanifest_sha256": p.sha(model), "original_pipeline_complete": True})
        columns = ["source", *sorted(p.REQUIRED_FRAME_COLUMNS)]
        frame = self.text(self.result, "metrics_per_frame.csv", ",".join(columns) + "\n" + ",".join(["x", *(["1"] * (len(columns)-1))]) + "\n")
        summary = self.text(self.result, "metrics_summary.csv", "group,value\nx,1\n")
        code = [self.text(self.source, n) for n in ("metric_models.py", "qualify_models.py", "runner.py", "batch_speed.py")]
        source_bindings = p.bindings([code[0], code[3]])
        identity = {"metric_models": "fixture"}
        binding = {"metric_evaluator_identity": identity, "modelmanifest_sha256": p.sha(model),
            "source_bindings": source_bindings, "device": {"total_memory_bytes": 10000}}
        batch = {"status": b.STATUS, "binding": binding, "protocol": b.PROTOCOL, "synthetic_images": True,
            "scientific_result": False, "real_model_weights": True, "source_or_development_images_used": False,
            "rng_state_preserved": True, "metric_evaluator_identity": identity, "modelmanifest_sha256": p.sha(model),
            "source_bindings": source_bindings, "training_updates": 0, "policy_selection_updates": 0,
            "chosen_batch_size": 1, "qualified_batch_sizes": [1], "candidates": [
                {"batch_size": 1, "eligible": True, "status": "PASS", "comparison": {"passed": True,
                 "classification_exact": True, "max_absolute_deltas": {name: 0. for name in b.FLOAT_FIELDS}},
                 "peak_reserved_bytes": 1000, "repeat_seconds_per_image": [1., 1., 1.], "median_seconds_per_image": 1.},
                *[{"batch_size": size, "eligible": False, "status": "REJECT_CUDA_OOM"} for size in (2,4,8,16)]]}
        batch["payload_sha256"] = b.digest(batch)
        batch_path = self.out / b.RECEIPT_NAME
        batch_copy = self.result / b.RECEIPT_NAME
        p.write(batch_path, batch); p.write(batch_copy, batch)
        batch_fields = {"metric_batch_size": 1, "metric_qualified_batch_sizes": [1],
            "metric_batch_qualification_sha256": p.sha(batch_path), "metric_evaluator_identity": identity}
        registration = p.read(reg)
        registration.update(batch_fields, metric_batch_qualification_path=str(batch_path.resolve()))
        p.write(reg, registration)
        analysis = {"status": "COMPLETE", "synthetic": False, "training_updates": 0, "policy_selection_updates": 0,
            "sources": 100, "frames": 300, "bootstrap_replicates": 10000, "statistical_unit": "source_image",
            "evaluated_metrics": sorted(p.NEEDED_METRICS), "inputs": p.bindings([reg, frame, model]), "outputs": p.bindings([summary])}
        scoring = {"status": "COMPLETE", "synthetic": False, "training_updates": 0, "policy_selection_updates": 0,
            "parity_passed": True, "frames": 300, "modelmanifest_sha256": p.sha(model), "inputs": p.bindings([reg, model,batch_path]),
            "outputs": p.bindings([frame,batch_copy]), "source_bindings": p.bindings(code), **batch_fields}
        qualification = {"status": "REAL_MODEL_WEIGHTS_QUALIFICATION_PASS", "real_model_weights": True, "scientific_result": False,
            "synthetic_images": True, "source_bindings": p.bindings(code[:2]), "modelmanifest_sha256": p.sha(model),
            "metadata": {"metrics": {name: {"status": "READY"} for name in p.QUALIFIED_MODELS}}}
        paths = [self.result / "metrics_analysis_completion.json", self.out / "scoring_completion.json", self.out / "models_qualification.json"]
        for path, rec in zip(paths, (analysis, scoring, qualification)):
            p.write(path, rec)
        return paths

    def test_parent_requires_completed_verified_delivery(self):
        parent = self.parent()
        gate = p.verify_parent_gate(parent, processes={}, compute_pids=set())
        self.assertEqual(gate["status"], "PARENT_DELIVERED_AND_EXITED")
        rec = p.read(parent / "completion.json")
        rec["publication"]["status"] = "COMMITTED"
        p.write(parent / "completion.json", rec)
        with self.assertRaisesRegex(RuntimeError, "push"):
            p.verify_parent_gate(parent, processes={}, compute_pids=set())

    def test_parent_running_same_pid_and_worker_rejected(self):
        parent = self.parent()
        for processes in ({100: {"state": "S", "start_ticks": "1234", "command": "python supervisor.py"}},
                          {200: {"state": "S", "start_ticks": "5678", "command": "python /repo/experiments/scale-causal-partial-residual-20261002/m1_runner.py"}}):
            with self.subTest(processes=processes), self.assertRaises(RuntimeError):
                p.verify_parent_gate(parent, processes=processes, compute_pids=set())

    def test_parent_recycled_pid_allowed_and_original_gpu_rejected(self):
        parent = self.parent()
        p.verify_parent_gate(parent, processes={100: {"state": "S", "start_ticks": "new", "command": "unrelated"}}, compute_pids={100})
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            p.verify_parent_gate(parent, processes={}, compute_pids={100})

    def test_parent_supervisor_incomplete_rejected(self):
        parent = self.parent()
        rec = p.read(parent / "supervisor_status.json")
        rec["status"] = "RUNNING"
        p.write(parent / "supervisor_status.json", rec)
        with self.assertRaises(RuntimeError):
            p.verify_parent_gate(parent, processes={}, compute_pids=set())

    def test_results_complete_actual_evidence(self):
        self.evidence()
        rec = p.verify_results(self.out, self.result, self.source)
        self.assertEqual(rec["analysis"]["frames"], 300)
        self.assertEqual(len(rec["verified_inputs"]), 6)

    def test_results_reject_unqualified_batch_size_or_identity(self):
        _, scoring, _ = self.evidence()
        original = p.read(scoring)
        for key, value in (("metric_batch_size", 4), ("metric_qualified_batch_sizes", [1,4]),
                           ("metric_evaluator_identity", {"changed": True}),
                           ("metric_batch_qualification_sha256", "0"*64)):
            changed = copy.deepcopy(original); changed[key] = value; p.write(scoring, changed)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "batch"):
                p.verify_results(self.out, self.result, self.source)

    def test_results_reject_batch_receipt_tamper(self):
        self.evidence()
        path = self.out / b.RECEIPT_NAME
        changed = p.read(path); changed["chosen_batch_size"] = 8; p.write(path, changed)
        with self.assertRaises(RuntimeError):
            p.verify_results(self.out, self.result, self.source)

    def test_results_require_frame_parity(self):
        _, scoring, _ = self.evidence()
        rec = p.read(scoring)
        rec["parity_passed"] = False
        p.write(scoring, rec)
        with self.assertRaisesRegex(RuntimeError, "parity"):
            p.verify_results(self.out, self.result, self.source)

    def test_results_require_all_metric_weights_and_current_sources(self):
        _, _, qualification = self.evidence()
        rec = p.read(qualification)
        rec["metadata"]["metrics"]["dreamsim"]["status"] = "UNAVAILABLE"
        p.write(qualification, rec)
        with self.assertRaisesRegex(RuntimeError, "dreamsim"):
            p.verify_results(self.out, self.result, self.source)
        rec["metadata"]["metrics"]["dreamsim"]["status"] = "READY"
        p.write(qualification, rec)
        (self.source / "metric_models.py").write_text("changed\n")
        with self.assertRaisesRegex(RuntimeError, "source"):
            p.verify_results(self.out, self.result, self.source)

    def test_results_require_large_dino_model_ready(self):
        _, _, qualification = self.evidence()
        rec = p.read(qualification)
        rec["metadata"]["metrics"].pop("dinov2_vitl14")
        p.write(qualification, rec)
        with self.assertRaisesRegex(RuntimeError, "dinov2_vitl14"):
            p.verify_results(self.out, self.result, self.source)
        rec["metadata"]["metrics"]["dinov2_vitl14"] = {"status": "UNAVAILABLE"}
        p.write(qualification, rec)
        with self.assertRaisesRegex(RuntimeError, "dinov2_vitl14"):
            p.verify_results(self.out, self.result, self.source)

    def test_results_require_large_dino_in_evaluated_analysis(self):
        analysis, _, _ = self.evidence()
        rec = p.read(analysis)
        rec["evaluated_metrics"].remove("dinov2_vitl14_cosine")
        p.write(analysis, rec)
        with self.assertRaisesRegex(RuntimeError, "analysis is incomplete"):
            p.verify_results(self.out, self.result, self.source)

    def test_results_require_large_dino_column_and_preserved_small_dino(self):
        analysis, scoring, _ = self.evidence()
        frame = self.result / "metrics_per_frame.csv"
        original = frame.read_text()
        for missing in ("dinov2_vitl14_cosine", "dino_cosine"):
            with self.subTest(missing=missing):
                rows = list(csv.reader(io.StringIO(original)))
                index = rows[0].index(missing)
                for row in rows:
                    row.pop(index)
                buffer = io.StringIO(); csv.writer(buffer).writerows(rows)
                frame.write_text(buffer.getvalue())
                for path, field in ((analysis, "inputs"), (scoring, "outputs")):
                    rec = p.read(path)
                    rec[field][str(frame.resolve())] = p.sha(frame)
                    p.write(path, rec)
                with self.assertRaisesRegex(RuntimeError, missing):
                    p.verify_results(self.out, self.result, self.source)

    def test_results_reject_synthetic_analysis_and_model_mismatch(self):
        analysis, scoring, _ = self.evidence()
        rec = p.read(analysis)
        rec["synthetic"] = True
        p.write(analysis, rec)
        with self.assertRaises(RuntimeError):
            p.verify_results(self.out, self.result, self.source)
        rec["synthetic"] = False
        p.write(analysis, rec)
        rec = p.read(scoring)
        rec["modelmanifest_sha256"] = "0"*64
        p.write(scoring, rec)
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            p.verify_results(self.out, self.result, self.source)

    def make_table(self):
        path = self.result / "nested" / "measurements.csv"
        path.parent.mkdir()
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\r\n")
            writer.writerow(["source", "quoted, value", "unicode"])
            for i in range(20):
                writer.writerow([i, f'field "{i}"\nwith a second line', "中文"])
        return path

    def test_archive_exact_bytes_multiline_headers_and_records(self):
        path = self.make_table()
        original = path.read_bytes()
        archive = p.archive_tables(self.result, limit=170)
        entry = archive["tables"]["nested/measurements.csv"]
        self.assertEqual(entry["rows"], 20)
        self.assertGreater(len(entry["parts"]), 1)
        self.assertTrue(all(part["bytes"] <= 170 for part in entry["parts"]))
        for part in entry["parts"]:
            records = list(p.csv_records(self.result / part["path"]))
            self.assertEqual(records[0][0], ["source", "quoted, value", "unicode"])
            self.assertTrue(all(len(fields) == 3 for fields, raw in records))
        path.unlink()
        p.restore_tables(self.result)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(p.archive_tables(self.result, limit=170), archive)

    def test_archive_tamper_and_restore_overwrite_rejected(self):
        path = self.make_table()
        archive = p.archive_tables(self.result, limit=170)
        entry = archive["tables"]["nested/measurements.csv"]
        path.write_text("different\n")
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            p.restore_tables(self.result)
        path.unlink()
        part = self.result / entry["parts"][0]["path"]
        part.write_bytes(part.read_bytes() + b"changed")
        with self.assertRaisesRegex(RuntimeError, "part changed"):
            p.restore_tables(self.result)
        self.assertFalse(path.exists())

    def test_archive_metadata_rows_and_escape_rejected(self):
        self.make_table()
        entry = next(iter(p.archive_tables(self.result, limit=170)["tables"].values()))
        bad = copy.deepcopy(entry)
        bad["parts"][0]["rows"] += 1
        with self.assertRaisesRegex(RuntimeError, "row count"):
            p.verify_archive_entry(self.result, bad)
        bad = copy.deepcopy(entry)
        bad["parts"][0]["path"] = "../../outside.csv"
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            p.verify_archive_entry(self.result, bad)

    def test_archive_oversized_single_record_rejected(self):
        self.text(self.result, "one.csv", "a,b\n" + "x"*100 + ",y\n")
        with self.assertRaisesRegex(RuntimeError, "single CSV record"):
            p.archive_tables(self.result, limit=50)

    def test_explicit_file_whitelists_and_staging_scope(self):
        code = self.text(self.source, "valid.py")
        self.text(self.source, "__pycache__/valid.pyc", "not tracked")
        self.assertEqual(p.source_files(self.source), [code])
        self.text(self.source, "weights.pt", "forbidden")
        with self.assertRaises(RuntimeError):
            p.source_files(self.source)
        self.text(self.result, "weights.pt", "forbidden")
        with self.assertRaises(RuntimeError):
            p.result_files(self.result)
        p.assert_scope({"our.py"}, {"our.py", "release_manifest.json"})
        with self.assertRaisesRegex(RuntimeError, "Unrelated"):
            p.assert_scope({"their.py"}, {"our.py"})

    def test_result_list_omits_original_large_csv(self):
        path = self.make_table()
        archive = p.archive_tables(self.result, limit=170)
        files = p.result_files(self.result, archive)
        self.assertNotIn(path, files)
        self.assertIn(self.result / "table_shards/manifest.json", files)
        self.assertTrue(any(".part" in item.name for item in files))

    def test_required_checks_order_without_executing_git(self):
        self.text(self.source, "test_a.py")
        calls = []
        with mock.patch.object(p, "HERE", self.source), mock.patch.object(p, "command", side_effect=lambda args, **kw: calls.append((args, kw)) or b""):
            p.run_checks(io.StringIO())
        self.assertEqual(calls[0][0][1], "tools/update_repository_manifest.py")
        self.assertEqual(calls[1][0], ["git", "add", "--", "release_manifest.json"])
        self.assertEqual(calls[2][0][1], "tools/verify_repository.py")
        self.assertEqual(calls[3][0][1], "tools/run_cpu_checks.py")
        self.assertTrue(calls[4][1]["cpu"])
        self.assertEqual(Path(calls[4][0][1]).name, "test_a.py")

    def test_push_verification_and_completion_without_real_push(self):
        commit = "b"*40
        record = {"status": "COMMITTED", "phase": "results", "commit": commit, "checks": "PASS"}
        calls = []
        def fake_command(args, **kwargs):
            calls.append(args)
            if args[1] == "ls-remote":
                return f"{commit}\trefs/heads/main\n".encode()
            return b""
        with mock.patch.object(p, "OUT", self.out), mock.patch.object(p, "command", side_effect=fake_command):
            rec = p.push_record(record, self.out / "results_publication.json", io.StringIO())
        self.assertEqual(calls[0], ["git", "push", "origin", "main"])
        self.assertEqual(rec["status"], "PUSHED")
        self.assertEqual(p.read(self.out / "completion.json")["status"], "UNIFIED_METRICS_COMPLETE")

    def test_wrong_remote_sha_never_marks_pushed(self):
        record = {"status": "COMMITTED", "phase": "source", "commit": "b"*40, "checks": "PASS"}
        def fake_command(args, **kwargs):
            return ("c"*40 + "\trefs/heads/main\n").encode() if args[1] == "ls-remote" else b""
        with mock.patch.object(p, "command", side_effect=fake_command), self.assertRaisesRegex(RuntimeError, "SHA"):
            p.push_record(record, self.out / "source_publication.json", io.StringIO())
        self.assertEqual(record["status"], "COMMITTED")
        self.assertFalse((self.out / "source_publication.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
