"""Complete N2048 M1 grid with calibration-only selection and cached RX reuse.

Stages are explicit: qualify, benchmark, calibrate, development. No original
main, historical output writer, training task or old queue is launched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import signal
import time

import numpy as np
import m1_common as u


class PauseRequested(RuntimeError):
    pass


class Runner:
    def __init__(self, root, protocol, out):
        self.root, self.out = Path(root).resolve(), Path(out).resolve()
        self.protocol_path = Path(protocol).resolve()
        self.protocol = u.validate_protocol(u.read(self.protocol_path))
        self.registration_path = self.out/"registration.json"
        self.registration = u.read(self.registration_path)
        u.require(self.registration["status"] == "REGISTERED" and self.registration["protocol_sha256"] == u.sha(self.protocol_path), "Frozen full-grid registration required")
        for key in ("source_bindings", "input_bindings"):
            u.verify(self.registration[key], self.root)
        self.registration_sha256 = u.sha(self.registration_path)
        self.native = None
        self.stop = False
        self.stage = "controller"

    def stopped(self, *_):
        self.stop = True

    def guard(self):
        if self.stop or (self.out/"STOP").exists():
            raise PauseRequested("Requested source-safe pause; completed source checkpoints retained")

    def status(self, status, **fields):
        u.write(self.out/(self.stage+"_status.json"), dict(status=status, stage=self.stage,
                pid=os.getpid(), time=time.time(), **fields))

    def load(self):
        self.guard()
        if self.native is None:
            self.status("LOADING_FROZEN_MODELS")
            from m1_native import Native
            self.native = Native(self.root, self.stopped)
        return self.native

    def completed(self, stage):
        path = self.out/(stage+"_completion.json")
        if not path.exists():
            return None
        receipt = u.read(path)
        u.require(receipt["status"] == "COMPLETE" and receipt["registration_sha256"] == self.registration_sha256,
                  "Completed stage has a different registration")
        u.verify(receipt["outputs"])
        return receipt

    def require_stage(self, stage):
        receipt = self.completed(stage)
        u.require(receipt is not None, "Required stage is incomplete: " + stage)
        return receipt

    def finish(self, stage, paths, **fields):
        u.verify(self.registration["source_bindings"], self.root)
        value = dict(status="COMPLETE", stage=stage, registration_sha256=self.registration_sha256,
                     outputs={str(p): u.sha(p) for p in paths}, synthetic=False, training_updates=0,
                     **fields)
        u.seal(self.out/(stage+"_completion.json"), value)
        self.status("COMPLETE", **fields)
        return value

    def stage_registration(self, stage, data, extra=None):
        native = self.load()
        role = "development" if stage == "development" else "calibration"
        value = dict(stage=stage, registration_sha256=self.registration_sha256,
            protocol_sha256=u.sha(self.protocol_path), source_bindings=self.registration["source_bindings"],
            frozen_identity=native.loaded["identity"], numerical_runtime=native.flags,
            source_records=[{k: r[k] for k in ("image_id", "preprocessing_id", "class_index")} for r in data["records"]],
            data_bindings=data.get("bindings", {}), original_population=native.proofs[role],
            image_hash_domain="sha256(b'float32:3,256,256:RGB\\0' + contiguous CHW float32 bytes)",
            training_updates=0, holdout_access=False, development_used_for_selection=False,
            **(extra or {}))
        path = self.out/(stage+"_registration.json")
        u.seal(path, value)
        return u.identity(value), path

    def qualify(self):
        self.stage = "qualify"
        if self.completed("qualify"):
            return
        native = self.load()
        result = native.qualify()
        path = self.out/"native_qualification.json"
        u.seal(path, dict(result, registration_sha256=self.registration_sha256))
        self.finish("qualify", [path], sources=1, physical_frames=138,
                    development_read=False, engineering_only=True)

    @staticmethod
    def task_grid(actions, seeds):
        # Preserve deterministic source -> family -> SNR -> action -> seed order.
        return [(action, snr, seed) for family in u.FAMILIES for snr in u.SNRS
                for action in actions if action.phy == family for seed in seeds]

    def validate_calibration_rows(self, rows, data, index, actions):
        expected = {(a.phy, s, u.action_id(a), n) for a, s, n in self.task_grid(actions, u.CAL_SEEDS)}
        actual = [(r["phy_family"], r["snr_db"], r["action_id"], r["noise_seed"]) for r in rows]
        source = data["records"][index]
        u.require(len(rows) == 2070 and len(set(actual)) == 2070 and set(actual) == expected,
                  "Full calibration source grid has missing or duplicate physical events")
        u.require(all(r["source_index"] == index and r["source_id"] == source["image_id"]
                      and r["preprocessing_id"] == source["preprocessing_id"] and r["N"] == 2048 for r in rows),
                  "Source grid identity mismatch")

    def calibration_source(self, stage, binding, data, index, actions):
        path = self.out/stage/"source_checkpoints"/("%04d.json" % index)
        if path.exists():
            saved = u.checkpoint(path, binding, index)
        else:
            saved = None
            if stage == "calibrate":
                benchmark_path = self.out/"benchmark/source_checkpoints"/("%04d.json" % index)
                if benchmark_path.exists():
                    benchmark_registration = u.read(self.out/"benchmark_registration.json")
                    benchmark = u.checkpoint(benchmark_path, u.identity(benchmark_registration), index)
                    u.require(benchmark_registration["registration_sha256"] == self.registration_sha256,
                              "Benchmark is from a different complete-grid registration")
                    current = self.load()
                    u.require(benchmark_registration["frozen_identity"] == current.loaded["identity"]
                              and benchmark_registration["numerical_runtime"] == current.flags
                              and benchmark_registration["original_population"] == current.proofs["calibration"],
                              "Benchmark reuse changed models, numeric runtime or source population")
                    saved = dict(benchmark, binding=binding,
                                 reused_benchmark_checkpoint=str(benchmark_path),
                                 reused_benchmark_sha256=u.sha(benchmark_path))
            if saved is None:
                native = self.load()
                memo = native.memo()
                started = time.monotonic()
                timing_start = {k: dict(v) for k, v in native.timings.items()}
                rows = []
                for action, snr, seed in self.task_grid(actions, u.CAL_SEEDS):
                    row, _image = native.score(data, index, action, snr, seed, memo)
                    rows.append(row)
                timing = {k: {field: v[field]-timing_start[k][field] for field in ("calls", "seconds")}
                          for k, v in native.timings.items()}
                saved = dict(binding=binding, source_index=index, rows=rows,
                    seconds=time.monotonic()-started, unique_receiver_events=len(memo["render"]),
                    unique_quality_images=len(memo["metric"]), unique_tx_waveforms=len(memo["tx"]["waves"]),
                    unique_tx_prefix_logits=len(memo["tx"]["logits"]), timing=timing,
                    synthetic=False, development_read=False)
                del memo
            self.validate_calibration_rows(saved["rows"], data, index, actions)
            u.sealed_checkpoint(path, saved)
        self.validate_calibration_rows(saved["rows"], data, index, actions)
        return saved, path

    def benchmark(self):
        self.stage = "benchmark"
        if self.completed("benchmark"):
            return
        self.require_stage("qualify")
        native = self.load()
        data = native.data("calibration")
        actions = native.actions()
        binding, registration = self.stage_registration("benchmark", data,
            dict(action_grid=[u.action_dict(a) for a in actions], noise_seeds=u.CAL_SEEDS,
                 measured_sources=2, full_grid=True, policy_selection=False))
        paths = [registration]
        summaries = []
        for index in range(2):
            self.guard()
            saved, path = self.calibration_source("benchmark", binding, data, index, actions)
            paths.append(path)
            summaries.append({k: saved[k] for k in ("source_index", "seconds", "unique_receiver_events", "unique_quality_images", "unique_tx_waveforms", "unique_tx_prefix_logits", "timing")})
            self.status("BENCHMARK", sources=index+1, total_sources=2, source_seconds=saved["seconds"])
        mean = sum(s["seconds"] for s in summaries)/2
        report = dict(status="MEASURED", sources=2, physical_frames=4140, per_source=summaries,
            mean_source_seconds=mean, projected_full_calibration_seconds=mean*1000,
            projected_remaining_after_reuse_seconds=mean*998,
            estimate_caveat="two calibration sources; image-dependent unique receiver states and contention can change throughput",
            current_cuda_reserved_bytes=int(native.torch.cuda.memory_reserved()),
            peak_cuda_reserved_bytes=int(native.torch.cuda.max_memory_reserved()),
            numerical_runtime=native.flags, policy_selection=False, development_read=False)
        path = self.out/"benchmark_report.json"
        u.seal(path, report)
        paths.append(path)
        native.frozen()
        self.finish("benchmark", paths, sources=2, physical_frames=4140,
                    mean_source_seconds=mean, development_read=False)

    @staticmethod
    def aggregate_rows(accumulated, first, rows):
        for row in rows:
            key = (row["phy_family"], row["snr_db"], row["action_id"])
            first[key] = row
            values = accumulated.setdefault(key, dict(n=0, psnr_db=0.0, lpips_alex=0.0,
                                                     dino_cosine=0.0, E=0.0, failure_fraction=0.0))
            values["n"] += 1
            for metric in ("psnr_db", "lpips_alex", "dino_cosine", "E"):
                values[metric] += float(row[metric])
            values["failure_fraction"] += float(not row["header_ok"] or not row["body_crc_ok"])

    def calibrate(self):
        self.stage = "calibrate"
        if self.completed("calibrate"):
            return
        self.require_stage("benchmark")
        native = self.load()
        data = native.data("calibration")
        actions = native.actions()
        mapping = {u.action_id(a): a for a in actions}
        binding, registration = self.stage_registration("calibrate", data,
            dict(action_grid=[u.action_dict(a) for a in actions], noise_seeds=u.CAL_SEEDS,
                 source_count=1000, full_grid=True, shortlist_used=False))
        paths = [registration]
        accumulated, first = {}, {}
        elapsed_sources = unique = rows_count = 0
        started = time.monotonic()
        for index in range(1000):
            self.guard()
            saved, path = self.calibration_source("calibrate", binding, data, index, actions)
            paths.append(path)
            self.aggregate_rows(accumulated, first, saved["rows"])
            elapsed_sources += saved["seconds"]
            unique += saved["unique_receiver_events"]
            rows_count += len(saved["rows"])
            self.status("CALIBRATING", sources=index+1, total_sources=1000, physical_frames=rows_count,
                unique_receiver_events=unique, source_seconds_mean=elapsed_sources/(index+1),
                estimated_remaining_seconds=(999-index)*elapsed_sources/(index+1),
                wall_seconds=time.monotonic()-started)
        summary = []
        for key, values in sorted(accumulated.items()):
            row = first[key]
            entry = {k: row[k] for k in ("N", "phy_family", "snr_db", "action_id", "m", "q", "order")}
            entry.update({k: v/values["n"] for k, v in values.items() if k != "n"})
            entry["rows"] = values["n"]
            summary.append(entry)
        u.require(len(summary) == 690 and all(r["rows"] == 3000 for r in summary), "Incomplete full calibration means")
        cells = []
        for family in u.FAMILIES:
            for snr in u.SNRS:
                for method in u.METHODS:
                    group = [r for r in summary if r["phy_family"] == family and r["snr_db"] == snr
                             and (r["q"] == 0 if method == "whole" else r["q"] == 0 or r["order"] == method)]
                    ranked, reference = native.m1.rank(group)
                    winner = ranked[0]
                    cells.append(dict(N=2048, phy_family=family, snr_db=snr, method=method,
                                      action=u.action_dict(mapping[winner["action_id"]]),
                                      calibration=winner, reliability_reference=reference))
        summary_path = self.out/"calibration_summary.csv"
        u.write_csv(summary_path, summary)
        policy_path = self.out/"m1_policy.json"
        u.seal(policy_path, dict(N=2048, cells=cells, sources=1000, noise_seeds=u.CAL_SEEDS,
            full_grid=True, physical_frames=2070000, registration_sha256=self.registration_sha256,
            calibration_binding=binding, development_read=False, training_updates=0,
            selection_rule="unchanged original m1_runner.rank", shortlist_used=False))
        paths += [summary_path, policy_path]
        native.frozen()
        self.finish("calibrate", paths, sources=1000, physical_frames=rows_count,
                    policies=50, policy_sha256=u.sha(policy_path), development_read=False,
                    unique_receiver_events=unique)

    def development_choices(self, cells):
        native = self.load()
        result = {}
        for family in u.FAMILIES:
            for snr in u.SNRS:
                choices = {method+"_policy": native.action(cells[(family, snr, method)]["action"]) for method in u.METHODS}
                entropy = choices["entropy_policy"]
                for order in ("raster", "random", "oracle"):
                    # q0 canonicalizes to whole in the original Action class.
                    choices[order+"_at_entropy"] = native.phy.Action(2048, family, entropy.m, entropy.q, order)
                result[(family, snr)] = choices
        return result

    def development(self):
        self.stage = "development"
        if self.completed("development"):
            return
        self.require_stage("calibrate")
        policy_path = self.out/"m1_policy.json"
        policy = u.read(policy_path)
        u.require(policy["sources"] == 1000 and policy["noise_seeds"] == u.CAL_SEEDS
                  and policy["full_grid"] and not policy["development_read"], "Complete frozen calibration required")
        native = self.load()
        data = native.data("development")
        cells = {(r["phy_family"], r["snr_db"], r["method"]): r for r in policy["cells"]}
        u.require(set(cells) == {(f, s, m) for f in u.FAMILIES for s in u.SNRS for m in u.METHODS}, "Frozen policy scope incomplete")
        choices = self.development_choices(cells)
        binding, registration = self.stage_registration("development", data,
            dict(policy_sha256=u.sha(policy_path), noise_seeds=u.DEV_SEEDS, source_count=100,
                 choices=[dict(phy_family=f, snr_db=s, methods={m: u.action_dict(a) for m, a in values.items()})
                          for (f, s), values in choices.items()]))
        paths = [registration]
        allrows = []
        for index, source in enumerate(data["records"]):
            self.guard()
            path = self.out/"development/source_checkpoints"/("%04d.json" % index)
            if path.exists():
                saved = u.checkpoint(path, binding, index)
                proof = saved["float_reconstructions"]
                u.require(u.sha(proof["path"]) == proof["sha256"], "Saved development float image cache changed")
            else:
                source_rgb = np.ascontiguousarray(source["pixels"], dtype=np.float32)/np.float32(255)
                reference_sha = u.rgb_sha(source_rgb)
                images, slots, rows, row_ids, image_slots = [], {}, [], [], []
                memo = native.memo()
                started = time.monotonic()
                for (family, snr), methods in choices.items():
                    for seed in u.DEV_SEEDS:
                        for method, action in methods.items():
                            row, image = native.score(data, index, action, snr, seed, memo, images=True)
                            image = np.ascontiguousarray(image, dtype=np.float32)
                            image_hash = u.rgb_sha(image)
                            if image_hash not in slots:
                                slots[image_hash] = len(images)
                                images.append(image)
                            row_id = u.identity(dict(study="M1_N2048_FULL_GRID", source_id=source["image_id"],
                                                     phy_family=family, snr_db=snr, noise_seed=seed, method=method))
                            row.update(method=method, replay_study="M1_N2048_FULL_GRID", replay_row_id=row_id,
                                       image_sha256=image_hash, reference_sha256=reference_sha,
                                       degenerate_order_no_partial=bool(method.endswith("_at_entropy") and action.q == 0))
                            rows.append(row)
                            row_ids.append(row_id)
                            image_slots.append(slots[image_hash])
                            if index in self.protocol["fixed_source_indices"] and seed == self.protocol["fixed_noise_seed"]:
                                native.common.save_image(self.out/"development/fixed_png"/
                                    ("src%03d_%s_snr%d_%s.png" % (index, family, snr, method)), image)
                u.require(len(rows) == 240 and len(set(row_ids)) == 240, "Development source labels incomplete")
                image_path = self.out/"development/reconstructions"/("%04d.npz" % index)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = image_path.with_name(image_path.name+".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, images=np.stack(images), source_rgb=source_rgb,
                             row_ids=np.asarray(row_ids), image_slots=np.asarray(image_slots, dtype=np.int64))
                os.replace(temporary, image_path)
                saved = dict(binding=binding, source_index=index,
                    record={k: source[k] for k in ("image_id", "preprocessing_id", "class_index")}, rows=rows,
                    float_reconstructions=dict(path=str(image_path), sha256=u.sha(image_path),
                                               rows=240, unique_images=len(images), dtype="float32", layout="CHW"),
                    seconds=time.monotonic()-started, unique_receiver_events=len(memo["render"]),
                    unique_quality_images=len(memo["metric"]), synthetic=False,
                    development_used_for_selection=False)
                u.sealed_checkpoint(path, saved)
                del memo, images
            u.require(len(saved["rows"]) == 240, "Incomplete saved development source")
            allrows.extend(saved["rows"])
            paths += [path, Path(saved["float_reconstructions"]["path"])]
            self.status("EVALUATING", sources=index+1, total_sources=100, method_rows=len(allrows),
                        last_source_seconds=saved["seconds"])
        table = self.out/"development/per_frame.csv"
        u.write_csv(table, allrows)
        paths.append(table)
        native.frozen()
        self.finish("development", paths, sources=100, method_rows=len(allrows),
                    policy_sha256=u.sha(policy_path), development_used_for_selection=False,
                    saved_float_rgb=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=("qualify", "benchmark", "calibrate", "development"), required=True)
    args = parser.parse_args()
    out = args.output or str(Path(args.root)/"outputs/M1-N2048-FULL-GRID-20261004")
    # These stages execute on the original Linux GPU host. One owner prevents
    # accidental duplicate calibration or concurrent writes to the same cache.
    import fcntl
    Path(out).mkdir(parents=True, exist_ok=True)
    lock = (Path(out)/"execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = None
    try:
        runner = Runner(args.root, args.protocol, out)
        signal.signal(signal.SIGTERM, runner.stopped)
        signal.signal(signal.SIGINT, runner.stopped)
        getattr(runner, args.stage)()
    except PauseRequested as error:
        if runner.native is not None:
            runner.native.frozen()
        runner.status("PAUSED_AT_SOURCE_BOUNDARY", reason=str(error))
        raise SystemExit(75)
    except Exception as error:
        if runner is not None:
            runner.status("FAILED", error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
