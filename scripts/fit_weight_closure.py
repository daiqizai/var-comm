#!/usr/bin/env python3
"""Freeze only calibration-chosen actual points, matches, and selector probabilities."""

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

from var_comm.study import sha256, write_csv, write_json
from var_comm.weight_closure import fit_time_sharing, load_configs, select_matched_controls, specifications


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    closure, base, _runtime = load_configs()
    selection = json.loads((arguments.training / "selection.json").read_text())["selected"]
    if not (arguments.training / "completion.json").exists():
        raise RuntimeError("freeze must follow all registered weight training")
    metrics, models, bindings = {}, {}, {}
    specs = specifications(closure)
    for arm, chosen in selection.items():
        summary_path = arguments.training / "calibration" / f"step_{chosen['step']:07d}_full" / "summary.json"
        metrics[arm] = json.loads(summary_path.read_text())[arm]
        directory = ROOT / closure["base_run"] / "training_001" if specs[arm]["reused_training"] else arguments.training
        path = directory / "checkpoints" / f"step_{chosen['step']:07d}.pt"
        models[arm] = {**chosen, "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
                       "state_key": chosen["variant"] if specs[arm]["reused_training"] else arm}
        bindings[str(summary_path.resolve())] = sha256(summary_path)
    history = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"
    policy_path = history / "POLICIES_001/policies.json"
    policy = json.loads(policy_path.read_text())["actions"]
    raw_path = history / "CALIBRATION_001/per_frame.csv"
    refs = []
    for row in read_rows(raw_path):
        snr = float(row["snr_db"])
        family = row["family"]
        if snr in base["snrs_db"] and int(row["mode"]) == policy[family]["quality"][row["snr_db"]]:
            refs.append({"arm": family + "_adaptive", "image_id": row["image_id"], "image_index": int(row["image_index"]),
                         "snr_db": snr, "seed": int(row["seed"]), "mse": 10 ** (-float(row["psnr_db"]) / 10),
                         **{name: float(row[name]) for name in ("psnr_db", "lpips", "dino")},
                         "source_pixels_sha256": row["source_pixels_sha256"]})
    deep_path = arguments.deep_calibration / "per_frame.csv"
    deep_receipt = json.loads((arguments.deep_calibration / "completion.json").read_text())
    if sha256(deep_path) != deep_receipt["per_frame_sha256"]:
        raise RuntimeError("frozen Deep calibration changed")
    for row in read_rows(deep_path):
        refs.append({"arm": "perceptual_deepjscc", "image_id": row["image_id"], "image_index": int(row["image_index"]),
                     "snr_db": float(row["snr_db"]), "seed": int(row["seed"]),
                     **{name: float(row[name]) for name in ("psnr_db", "lpips", "dino", "mse")},
                     "source_pixels_sha256": row["source_pixels_sha256"]})
    source_lookup = {(row["image_id"], row["snr_db"], row["seed"]): row["source_pixels_sha256"] for row in refs if row["arm"] == "raw_adaptive"}
    for method in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc"):
        chosen = [row for row in refs if row["arm"] == method]
        if len(chosen) != 15000 or len({(row['image_id'],row['snr_db'],row['seed']) for row in chosen}) != 15000:
            raise RuntimeError("calibration references lack complete paired source/noise coverage")
        if any(source_lookup[row["image_id"],row["snr_db"],row["seed"]] != row["source_pixels_sha256"] for row in chosen):
            raise RuntimeError("calibration references do not use identical original pixels")
    reference_means = {method: {name: float(np.mean([row[name] for row in refs if row["arm"] == method and row["snr_db"] in base["primary_snrs_db"]]))
                               for name in ("psnr_db", "lpips", "mse", "dino")}
                       for method in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc")}
    matches = select_matched_controls(metrics, closure)
    mixtures = fit_time_sharing(metrics, reference_means, closure)
    bindings.update({str(path.resolve()): sha256(path) for path in (raw_path, deep_path, policy_path, arguments.training / "selection.json")})
    record = {"status": "CALIBRATION_ONLY_WORKPOINTS_MATCHES_AND_PROBABILITIES_FROZEN",
              "frozen_at": datetime.now().astimezone().isoformat(), "models": models, "calibration_metrics": metrics,
              "reference_calibration_means": reference_means, "matches": matches, "time_sharing": mixtures,
              "source_bindings": bindings, "DINO_used_for_selection_matching_ratio": False,
              "development_used_for_selection_matching_ratio": False, "new_holdout": False,
              "mixture_reference_is_expectation_not_deployment": True}
    write_csv(output / "reference_calibration_rows.csv", refs)
    write_json(output / "frozen_comparisons.json", record)
    write_json(output / "completion.json", {"status": "CALIBRATION_FREEZE_COMPLETE", "frozen_sha256": sha256(output / "frozen_comparisons.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--deep-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
