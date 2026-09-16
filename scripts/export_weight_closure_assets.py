#!/usr/bin/env python3
"""Export all measured closure curves and tables without selecting new points."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from var_comm.study import sha256, write_csv, write_json


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure, output, name):
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.pdf")
    plt.close(figure)


def run(arguments):
    root = arguments.run.resolve()
    if not (root / "independent_audit_001/completion.json").exists():
        raise RuntimeError("only export audited, complete experiments")
    output = root / "report_assets_001"
    output.mkdir(exist_ok=False)
    closure = load(ROOT / "configs/hybrid_weight_closure.json")
    selected = load(root / "CALIBRATION_FREEZE_001/frozen_comparisons.json")["models"]
    full = []
    for step in closure["calibration_full_steps"]:
        summary = load(root / "training_001/calibration" / f"step_{step:07d}_full/summary.json")
        for arm, regions in summary.items():
            for region, values in regions.items():
                full.append({"new_step": step, "arm": arm, "region": region, "weight": selected[arm]["weight"],
                             **values, "registered_selection_score": values["mse"] + selected[arm]["weight"] * values["lpips"]})
    write_csv(output / "full_calibration_all_metrics.csv", full)
    trends = []
    for arm, point in selected.items():
        for region in ("primary", "mechanism"):
            first = next(row for row in full if row["arm"] == arm and row["region"] == region and row["new_step"] == 15000)
            last = next(row for row in full if row["arm"] == arm and row["region"] == region and row["new_step"] == 20000)
            trends.append({"arm": arm, "region": region, "selected_step": point["step"],
                           **{f"final_5000_delta_{metric}": last[metric] - first[metric] for metric in ("mse", "psnr_db", "lpips", "dino", "registered_selection_score")},
                           "budget_cap_not_convergence_proof": True})
    write_csv(output / "final_calibration_trends.csv", trends)
    colors = {.01: "#3267a8", .03: "#d87a20", .1: "#3e9852"}
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for row_index, region in enumerate(("primary", "mechanism")):
        for column, metric in enumerate(("psnr_db", "lpips", "dino")):
            axis = axes[row_index, column]
            for arm, point in selected.items():
                data = [row for row in full if row["arm"] == arm and row["region"] == region]
                axis.plot([row["new_step"] for row in data], [row[metric] for row in data],
                          color=colors[point["weight"]], linestyle="-" if point["variant"] == "conditioned" else "--",
                          marker="s" if point["variant"] == "conditioned" else "o", markersize=4,
                          label=f"{point['variant']}, lambda={point['weight']}")
            axis.set(title=f"{region}: {metric}", xlabel="Additional updates / arm", ylabel=metric)
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Fixed 1000-source calibration; DINO is report-only; no development checkpoint selection", fontsize=11)
    save(figure, output, "full_calibration_curves")
    history = []
    for directory, reused in ((ROOT / closure["base_run"] / "training_001", True), (root / "training_001", False)):
        for path in sorted(directory.glob("updates_from_*.jsonl")):
            with path.open() as handle:
                for line in handle:
                    event = json.loads(line)
                    for name, scores in event["arms"].items():
                        arm = f"lpips_0p01__{name}" if reused else name
                        history.append({"arm": arm, "new_step": event["new_step"],
                                        **{metric: scores[metric] for metric in ("mse", "lpips", "loss", "gradient_norm")}})
    smoothed = []
    for arm in selected:
        ordered = sorted([row for row in history if row["arm"] == arm], key=lambda row: row["new_step"])
        for offset in range(0, len(ordered), 100):
            batch = ordered[offset:offset + 100]
            smoothed.append({"arm": arm, "last_step": batch[-1]["new_step"], "updates_in_window": len(batch),
                             **{metric: float(np.mean([row[metric] for row in batch])) for metric in ("mse", "lpips", "loss", "gradient_norm")}})
    write_csv(output / "training_curves_100update_windows.csv", smoothed)
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for column, weight in enumerate(closure["lambdas"]):
        for row_index, metric in enumerate(("mse", "lpips")):
            axis = axes[row_index, column]
            for arm, point in selected.items():
                if point["weight"] != weight:
                    continue
                data = [row for row in smoothed if row["arm"] == arm]
                axis.plot([row["last_step"] for row in data], [row[metric] for row in data], label=point["variant"],
                          linestyle="-" if point["variant"] == "conditioned" else "--")
            axis.set(title=f"lambda={weight}: training {metric}", xlabel="Additional updates / arm", ylabel=metric)
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Paired draws / noise; 100-update windows; raw per-update observations are retained", fontsize=11)
    save(figure, output, "training_mse_lpips_curves")
    quality = rows(root / "analysis_001/quality_summary.csv")
    actual = [row for row in quality if row["arm"] in selected or row["arm"] in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc")]
    write_csv(output / "actual_points_and_strong_systems.csv", actual)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, metric in zip(axes, ("psnr_db", "lpips", "dino")):
        for arm in selected:
            point = selected[arm]
            data = sorted([row for row in actual if row["arm"] == arm and row["region"].startswith("snr_")], key=lambda row: float(row["region"][4:]))
            axis.plot([float(row["region"][4:]) for row in data], [float(row[metric]) for row in data],
                      color=colors[point["weight"]], linestyle="-" if point["variant"] == "conditioned" else "--",
                      marker="s" if point["variant"] == "conditioned" else "o", markersize=4,
                      label=f"{point['variant']}, lambda={point['weight']}")
        for arm, color, marker in (("raw_adaptive", "black", "^"), ("arithmetic_adaptive", "#8a3ca2", "v"), ("perceptual_deepjscc", "#aa3434", "*")):
            data = sorted([row for row in actual if row["arm"] == arm and row["region"].startswith("snr_")], key=lambda row: float(row["region"][4:]))
            axis.plot([float(row["region"][4:]) for row in data], [float(row[metric]) for row in data], color=color, marker=marker, label=arm)
        axis.set(xlabel="SNR (dB)", ylabel=metric, title=metric, xticks=closure["system"]["all_snrs_db"])
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=6, ncol=1)
    figure.suptitle("Actual development points: 100 sources x 3 noises / SNR; all failures retained; N3060/E6120", fontsize=11)
    save(figure, output, "development_per_snr")
    write_json(output / "completion.json", {"status": "AUDITED_TABLES_AND_CURVES_EXPORTED", "actual_points": 6,
               "full_calibration_rows": len(full), "training_window_rows": len(smoothed), "new_checkpoint_selection": False,
               "new_training_or_holdout": False, "script_sha256": sha256(Path(__file__)), "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args())
