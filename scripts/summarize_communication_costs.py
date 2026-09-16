#!/usr/bin/env python3
"""Reuse sealed system timings and append only R3/whole-entropy measurements."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from var_comm.mode_policies import RULES, selected_mode
from var_comm.online_timing import SOURCE_INDICES
from var_comm.study import artifact_hashes, create_output, paired_interval, sha256, write_csv, write_json


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    policies = json.loads(arguments.policies.read_text())
    if policies["status"] != "CALIBRATION_POLICIES_FROZEN":
        raise RuntimeError("cost selection requires the frozen calibration rules")
    runs = {"old": ROOT / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/full_001",
            "r3": ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/R3_CPU_TIMING_001",
            "arithmetic": ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/ENTROPY_CPU_TIMING_001"}
    tables, receipts = {}, {}
    for name, run in runs.items():
        receipt = json.loads((run / "completion.json").read_text())
        if sha256(run / "per_call.csv") != receipt["output_hashes"]["per_call.csv"]:
            raise RuntimeError("sealed timing table changed")
        tables[name] = read_rows(run / "per_call.csv")
        receipts[name] = sha256(run / "completion.json")
    old = {(int(row["image_index"]), float(row["snr_db"]), row["arm"], int(row["repeat"])): row for row in tables["old"]}
    arithmetic = {(int(row["image_index"]), float(row["snr_db"]), int(row["mode"]), int(row["repeat"])): row for row in tables["arithmetic"]}
    methods = {}
    common_snrs = (1., 4., 7., 13., 19.)
    for family in ("raw", "arithmetic"):
        for rule in RULES:
            rows = []
            for index in SOURCE_INDICES:
                for snr in common_snrs:
                    mode = selected_mode(policies["actions"], family, rule, snr)
                    for repeat in range(3):
                        rows.append(old[index, snr, f"whole_m{mode}", repeat] if family == "raw" else arithmetic[index, snr, mode, repeat])
            methods[family + "_" + rule] = rows
    for name in ("r2__full_grid_innovation", "perceptual_deepjscc", "wetok_8PSK_FEC", "whole_m7", "whole_m8", "whole_m9", "whole_adaptive"):
        methods[name] = [row for row in tables["old"] if row["arm"] == name]
    methods["r3__full_grid_prediction_features"] = tables["r3"]
    scopes = [(str(snr), [snr]) for snr in common_snrs] + [("primary_1_4_7", [1., 4., 7.]), ("high_13_19", [13., 19.]), ("all_5_common_snrs", list(common_snrs))]
    summary, paired = [], []
    for scope, snrs in scopes:
        per_source = {}
        for name, all_rows in methods.items():
            selected = [row for row in all_rows if float(row["snr_db"]) in snrs]
            if len(selected) != len(SOURCE_INDICES) * len(snrs) * 3:
                raise RuntimeError("cost comparison lost the matched source/repeat/SNR set")
            result = {"scope": scope, "method": name, "source_images": 32, "timing_rows": len(selected),
                      "endpoint": "CPU_to_CPU", "session_relation": "sealed_old_plus_separate_R3_and_entropy_sessions"}
            for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds"):
                values = np.array([np.mean([float(row[metric]) for row in selected if int(row["image_index"]) == index]) * 1000 for index in SOURCE_INDICES])
                per_source[name, metric] = values
                result[metric.replace("seconds", "mean_ms")] = float(values.mean())
                result[metric.replace("seconds", "p95_ms")] = float(np.percentile([float(row[metric]) * 1000 for row in selected], 95))
            summary.append(result)
        for control in ("raw_quality", "arithmetic_reliability", "r2__full_grid_innovation", "r3__full_grid_prediction_features", "perceptual_deepjscc", "wetok_8PSK_FEC"):
            for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds"):
                delta = per_source["arithmetic_quality", metric] - per_source[control, metric]
                paired.append({"scope": scope, "method": "arithmetic_quality", "control": control, "metric": metric,
                    "units": "ms_method_minus_control", "source_images": 32,
                    "caveat": "different_sessions_and_model_residency_not_a_pure_mechanism_speed_effect",
                    **paired_interval(delta, 9142026, 10000)})
    output = create_output(arguments.output_dir)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    write_json(output / "completion.json", {"status": "FROZEN_COMMUNICATION_COST_SUMMARY_COMPLETE", "completed_local": datetime.now().astimezone().isoformat(),
        "policy_sha256": sha256(arguments.policies), "input_timing_receipts": receipts, "existing_systems_retimed": False,
        "no_airtime_or_queueing_claim": True, "source_script_sha256": sha256(Path(__file__)),
        "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
    print(json.dumps([row for row in summary if row["scope"] == "primary_1_4_7"], indent=2))


if __name__ == "__main__":
    main()
