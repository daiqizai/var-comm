#!/usr/bin/env python3
"""Independently check timing artifacts and source-paired statistics without model inference."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run_dir.resolve()
    output = arguments.output_dir.resolve()
    parent = ROOT / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915"
    if not run.is_relative_to(parent) or not output.is_relative_to(parent):
        raise ValueError("audit paths must stay in the timing study")
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt_path = run / "completion.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["status"] != "FROZEN_SYSTEM_CPU_ENDPOINT_TIMING_COMPLETE":
            raise RuntimeError("full timing is not complete")
        for relative, expected in receipt["output_hashes"].items():
            path = (run / relative).resolve()
            if not path.is_relative_to(run) or sha256(path) != expected:
                raise RuntimeError(f"timing artifact changed: {path}")
        for path, expected in {**receipt["source_hashes"], **receipt["input_sha256"]}.items():
            if sha256(path) != expected:
                raise RuntimeError(f"frozen source/input changed: {path}")
        if receipt["frozen_models_before"] != receipt["frozen_models_after"]:
            raise RuntimeError("timing altered a model")
        config_path = ROOT / "configs/frozen_system_online_timing.json"
        if sha256(config_path) != receipt["config_sha256"]:
            raise RuntimeError("timing configuration changed")
        config = json.loads(config_path.read_text())
        rows = read_rows(run / "per_call.csv")
        lookup = {(int(row["image_index"]), float(row["snr_db"]), row["arm"], int(row["repeat"])): row for row in rows}
        indices = np.rint(np.linspace(0, 99, 32)).astype(int).tolist()
        keys = {(index, snr, arm, repeat) for index in indices for snr in config["snrs_db"] for arm in config["arms"] for repeat in range(3)}
        if len(rows) != 3360 or set(lookup) != keys or len(lookup) != len(rows):
            raise RuntimeError("timing population or repetition matrix changed")
        prepared = read_rows(ROOT / "outputs/DIGITAL-ONLINE-TIMING-PREPARATION-20260915/input_audit/archived_frame_inputs.csv")
        digital = {(int(row["image_index"]), float(row["snr_db"]), row["arm"]): row for row in prepared}
        original = read_rows(ROOT / "outputs/WETOK-JOINT-SUFFICIENCY-R2-EVALUATION/quality_0010000/per_frame.csv")
        controls = {(int(row["image_index"]), float(row["snr_db"]), row["arm"]): row for row in original if int(row["seed"]) == 2001}
        exact_images = 0
        for row in rows:
            key = int(row["image_index"]), float(row["snr_db"]), row["arm"]
            if row["arm"].startswith("whole_"):
                expected = digital[key]
                image_hash = expected["archived_image_sha256"]
                if row["received_sha256"] != expected["received_sha256"] or row["reference_transmitted_sha256"] != expected["transmitted_sha256"]:
                    raise RuntimeError("digital observation differs from prepared original transmission")
                if row["failure_stratum"] != expected["failure_stratum"]:
                    raise RuntimeError("digital failure decision changed or failed sample was relabeled")
            else:
                expected = controls[key]
                image_hash = expected["image_sha256"]
                if row["arm"].startswith("r2__") and row["received_sha256"] != expected["received_sha256"]:
                    raise RuntimeError("R2 observation changed")
            if row["image_id"] != expected["image_id"] or row["reference_image_sha256"] != image_hash:
                raise RuntimeError("reference image/source changed")
            same = row["actual_image_sha256"] == image_hash
            if same != (row["image_exact"] == "True"):
                raise RuntimeError("exact-image flag disagrees with its digest")
            exact_images += int(same)
            if float(row["image_max_error"]) > config["image_max_error"]:
                raise RuntimeError("reported pixel error exceeds the fixed tolerance")
            if row["signal_exact"] == "True" and row["actual_transmitted_sha256"] != row["reference_transmitted_sha256"]:
                raise RuntimeError("exact-waveform flag disagrees with its digest")
            if (int(row["total_complex_uses"]) != 3060 or abs(float(row["total_energy"]) - 6120) > .02 or
                    row["timing_scope"] != "CPU_to_CPU_contiguous_TX_and_RX" or
                    abs(float(row["processing_sum_seconds"]) - float(row["TX_seconds"]) - float(row["RX_seconds"])) > 1e-9):
                raise RuntimeError("resource, endpoint or timing sum changed")
            if any(not np.isfinite(float(row[metric])) or float(row[metric]) <= 0
                   for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds")):
                raise RuntimeError("nonfinite or nonpositive latency")
        scopes = {str(int(snr)): [snr] for snr in config["snrs_db"]}
        scopes.update(primary_1_4_7=[1., 4., 7.], all_5_snrs=config["snrs_db"])
        per_source, percentiles = {}, {}
        for scope, snrs in scopes.items():
            for arm in config["arms"]:
                for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds"):
                    values = np.asarray([[float(lookup[index, snr, arm, repeat][metric]) for snr in snrs for repeat in range(3)] for index in indices]) * 1000
                    per_source[scope, arm, metric] = values.mean(axis=1)
                    percentiles[scope, arm, metric] = float(np.percentile(values, 95))
        summary_error = 0.
        for row in read_rows(run / "summary.csv"):
            for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds"):
                average = float(per_source[row["scope"], row["arm"], metric].mean())
                summary_error = max(summary_error, abs(average - float(row[metric.replace("seconds", "mean_ms")])))
                summary_error = max(summary_error, abs(percentiles[row["scope"], row["arm"], metric] -
                                                       float(row[metric.replace("seconds", "raw_call_p95_ms")])))
        bootstrap = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(indices), (config["bootstrap_resamples"], len(indices)))
        interval_error = 0.
        for row in read_rows(run / "paired.csv"):
            delta = per_source[row["scope"], row["method"], row["metric"]] - per_source[row["scope"], row["control"], row["metric"]]
            low, high = np.percentile(delta[bootstrap].mean(axis=1), [2.5, 97.5])
            interval_error = max(interval_error, abs(delta.mean() - float(row["gain"])), abs(low - float(row["ci_low"])), abs(high - float(row["ci_high"])))
            if int(row["source_images"]) != 32:
                raise RuntimeError("timing repetitions inflated the independent source count")
        if max(summary_error, interval_error) > 1e-8:
            raise RuntimeError("independent timing statistics differ")
        result = {"status": "CPU_ARTIFACT_AND_STATISTICS_AUDIT_PASS_NO_SECOND_MODEL_RENDER",
                  "completed_local": datetime.now().astimezone().isoformat(), "timing_receipt_sha256": sha256(receipt_path),
                  "timing_rows": len(rows), "source_images": 32, "pixel_exact_hash_matches": exact_images,
                  "nonexact_pixels_independently_rerendered": False, "max_summary_difference_ms": summary_error,
                  "max_bootstrap_difference_ms": interval_error, "source_script_sha256": sha256(Path(__file__)),
                  "new_model_inference": False, "new_training": False, "research_goal_complete": False}
        write_json(output / "completion.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "failed_local": datetime.now().astimezone().isoformat()})
        raise


if __name__ == "__main__":
    main()
