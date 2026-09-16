#!/usr/bin/env python3
"""Evaluate already frozen calibration policies; never select a mode from evaluation quality."""

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
from var_comm.study import artifact_hashes, create_output, paired_interval, sha256, verify_snapshot, write_csv, write_json

METRICS = ("psnr_db", "lpips", "dino")
REFERENCES = ("r2__full_grid_innovation", "r3__full_grid_prediction_features", "perceptual_deepjscc", "wetok_8PSK_FEC")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frame_key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def cube(rows, identifiers, snrs, seeds, method):
    lookup = {frame_key(row): row for row in rows}
    expected = {(index, snr, seed) for index in range(len(identifiers)) for snr in snrs for seed in seeds}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError(f"incomplete or duplicated method matrix: {method}")
    values = np.empty((len(identifiers), len(snrs), len(seeds), len(METRICS)), dtype=np.float64)
    for index, identifier in enumerate(identifiers):
        for snr_index, snr in enumerate(snrs):
            for seed_index, seed in enumerate(seeds):
                row = lookup[index, snr, seed]
                if row["image_id"] != identifier:
                    raise RuntimeError("a paired method changed source identity")
                uses = row.get("complex_uses", row.get("total_complex_uses"))
                if int(uses) != 3060 or abs(float(row["total_energy"]) - 6120) > .02:
                    raise RuntimeError("a compared system does not have the same actual N/E budget")
                values[index, snr_index, seed_index] = [float(row[metric]) for metric in METRICS]
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite image quality entered the comparison")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--matrix-audit", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.matrix_run.resolve()
    receipt = json.loads((run / "completion.json").read_text())
    audit = json.loads(arguments.matrix_audit.read_text())
    policies = json.loads(arguments.policies.read_text())
    config_path = ROOT / "configs/communication_decision_study.json"
    config = json.loads(config_path.read_text())
    role = receipt["population"]
    if (role not in ("development", "holdout") or receipt["mode"] != "full" or receipt["status"] != "FIXED_MODE_MATRIX_COMPLETE" or
            policies["status"] != "CALIBRATION_POLICIES_FROZEN" or policies["config_sha256"] != sha256(config_path) or
            receipt["policy_sha256"] != sha256(arguments.policies) or audit["status"] != "COMMUNICATION_MATRIX_CPU_AUDIT_PASS" or
            audit["input_receipt_sha256"] != sha256(run / "completion.json")):
        raise RuntimeError("evaluation is incomplete, unverified or not bound to frozen calibration decisions")
    verify_snapshot(policies["source_hashes"])
    for name in ("per_frame.csv", "population.json"):
        if sha256(run / name) != receipt["output_hashes"][name]:
            raise RuntimeError("evaluated source/mode matrix changed")
    rows = read_rows(run / "per_frame.csv")
    population = json.loads((run / "population.json").read_text())
    identifiers = [target["image_id"] for target in population]
    snrs, seeds = config["snrs_db"], config[role + "_seeds"]
    fixed = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["family"], int(row["mode"])): row for row in rows}
    if len(fixed) != len(rows) or len(rows) != len(identifiers) * len(snrs) * len(seeds) * 6:
        raise RuntimeError("fixed candidate grid is incomplete")
    methods = {f"{family}_m{mode}": [row for row in rows if row["family"] == family and int(row["mode"]) == mode]
               for family in config["coding_families"] for mode in config["prefix_modes"]}
    policy_rows, selections = [], []
    for family in config["coding_families"]:
        for rule in RULES:
            name = family + "_" + rule
            selected = []
            for index in range(len(identifiers)):
                for snr in snrs:
                    mode = selected_mode(policies["actions"], family, rule, snr)
                    if index == 0:
                        selections.append({"method": name, "snr_db": snr, "mode": mode, "selected_using": "calibration_only"})
                    for seed in seeds:
                        row = fixed[index, snr, seed, family, mode]
                        selected.append(row)
                        policy_rows.append({"image_index": index, "image_id": identifiers[index], "snr_db": snr, "seed": seed,
                            "method": name, "family": family, "mode": mode, "header_uses": int(row["header_uses"]),
                            "data_uses": int(row["data_uses"]), "complex_uses": int(row["complex_uses"]), "total_energy": float(row["total_energy"]),
                            "accepted_correct": int(row["accepted_correct"]), "source_correct": int(row["source_correct"]),
                            **{metric: float(row[metric]) for metric in METRICS}})
            methods[name] = selected
    methods["old_raw_adaptive"] = [fixed[index, snr, seed, "raw", 7 if snr < 2.5 else (8 if snr < 5.5 else 9)]
        for index in range(len(identifiers)) for snr in snrs for seed in seeds]
    reference_root = arguments.reference_run.resolve()
    reference_receipt = json.loads((reference_root / "completion.json").read_text())
    reference_path = reference_root / "per_frame.csv"
    if sha256(reference_path) != reference_receipt["output_hashes"]["per_frame.csv"]:
        raise RuntimeError("frozen strong-reference quality rows changed")
    reference_rows = read_rows(reference_path)
    for name in REFERENCES:
        methods[name] = [row for row in reference_rows if row["arm"] == name and float(row["snr_db"]) in snrs and int(row["seed"]) in seeds]
    support_path = reference_root / "deep_support_supplement.csv"
    support_used = reference_receipt.get("Deep_support_mapping") == {"5.0": 4.0, "6.0": 7.0}
    if support_path.exists():
        if sha256(support_path) != reference_receipt["output_hashes"]["deep_support_supplement.csv"]:
            raise RuntimeError("frozen Deep support correction changed")
        support = read_rows(support_path)
        methods["legacy_Deep_nominal_condition"] = methods["perceptual_deepjscc"]
        methods["perceptual_deepjscc"] = [row for row in methods["perceptual_deepjscc"] if float(row["snr_db"]) not in (5., 6.)] + support
        support_used = True
    arrays = {name: cube(values, identifiers, snrs, seeds, name) for name, values in methods.items()}
    if role == "development":
        old_lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in reference_rows}
        maximum_replay_error = {metric: 0. for metric in METRICS}
        for name, old_name in (("raw_m8", "digital_m8"), ("old_raw_adaptive", "digital_adaptive")):
            for row in methods[name]:
                old = old_lookup[(*frame_key(row), old_name)]
                for metric in METRICS:
                    maximum_replay_error[metric] = max(maximum_replay_error[metric], abs(float(row[metric]) - float(old[metric])))
        if maximum_replay_error["psnr_db"] > 1e-4 or max(maximum_replay_error["lpips"], maximum_replay_error["dino"]) > 1e-5:
            raise RuntimeError("raw digital replay does not match the frozen strong reference")
    else:
        maximum_replay_error = None
    scopes = [(str(snr), [snr]) for snr in snrs] + [("primary_1_4_7", config["primary_snrs_db"]),
        ("transition_5_6", [5., 6.]), ("high_13_19", config["high_snrs_db"]), ("all_snrs", snrs)]
    selected_method = policies["global_selected_family"] + "_quality"
    contrasts = [(family + "_quality", family + "_" + rule) for family in config["coding_families"] for rule in ("reliability", "goodput")]
    contrasts += [("raw_quality", "arithmetic_quality")]
    contrasts += [(selected_method, control) for control in methods if control not in (selected_method, "legacy_Deep_nominal_condition")]
    contrasts = list(dict.fromkeys(contrasts))
    summary, paired, per_source = [], [], []
    for scope, selected_snrs in scopes:
        positions = [snrs.index(snr) for snr in selected_snrs]
        reduced = {name: values[:, positions].mean(axis=(1, 2)) for name, values in arrays.items()}
        for name, values in reduced.items():
            summary.append({"population": role, "scope": scope, "method": name, "source_images": len(identifiers),
                            **{metric: float(values[:, position].mean()) for position, metric in enumerate(METRICS)}})
            for index, identifier in enumerate(identifiers):
                per_source.append({"scope": scope, "method": name, "image_index": index, "image_id": identifier,
                                   **{metric: float(values[index, position]) for position, metric in enumerate(METRICS)}})
        for method, control in contrasts:
            for position, metric in enumerate(METRICS):
                interval = paired_interval(reduced[method][:, position] - reduced[control][:, position], config["bootstrap_seed"], config["bootstrap_resamples"])
                paired.append({"population": role, "scope": scope, "method": method, "control": control, "metric": metric,
                    "source_images": len(identifiers), "selection_data": "calibration_only", **interval})
    output = create_output(arguments.output_dir)
    write_csv(output / "policy_frames.csv", policy_rows)
    write_csv(output / "policy_modes.csv", selections)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "per_source.csv", per_source)
    write_json(output / "completion.json", {"status": "FROZEN_COMMUNICATION_POLICY_EVALUATION_COMPLETE", "completed_local": datetime.now().astimezone().isoformat(),
        "population": role, "source_images": len(identifiers), "selected_method_from_calibration": selected_method,
        "policy_sha256": sha256(arguments.policies), "matrix_receipt_sha256": sha256(run / "completion.json"),
        "matrix_audit_sha256": sha256(arguments.matrix_audit), "reference_receipt_sha256": sha256(reference_root / "completion.json"),
        "Deep_fixed_support_used_for_5_6": support_used, "maximum_old_raw_replay_metric_error": maximum_replay_error,
        "source_script_sha256": sha256(Path(__file__)), "no_evaluation_based_mode_selection": True,
        "new_model_inference": False, "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
    print(json.dumps([row for row in summary if row["scope"] == "primary_1_4_7"], indent=2))


if __name__ == "__main__":
    main()
