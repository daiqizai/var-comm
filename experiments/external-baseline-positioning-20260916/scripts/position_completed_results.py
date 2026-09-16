#!/usr/bin/env python3
"""Position only completed systems; never read unfinished diffusion quality."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from external_positioning.analysis import METRICS, SCOPES, source_values
from var_comm.study import sha256, write_csv, write_json


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def name(method):
    values = {"raw_adaptive": "VAR raw", "arithmetic_adaptive": "VAR arithmetic", "wetok_r3": "WeTok R3",
              "perceptual_deepjscc": "Perceptual DeepJSCC", "swin_ra32": "Swin RA32"}
    if "_N" in method:
        method = method.split("_N")[0]
    return values.get(method, method.replace("_", " "))


def mean_or_blank(values):
    return float(np.mean(values)) if len(values) else ""


def dominate(left, right):
    left_values = np.array([left["complex_uses"], left["lpips"], -left["psnr_db"], left["RX_mean_ms"]], dtype=float)
    right_values = np.array([right["complex_uses"], right["lpips"], -right["psnr_db"], right["RX_mean_ms"]], dtype=float)
    return bool(np.all(left_values <= right_values) and np.any(left_values < right_values))


def run(arguments):
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    root = arguments.root
    analysis = root / "non_diffusion_analysis_001"
    files = [analysis / name for name in ("completion.json", "summary.csv", "online_costs.csv", "per_source.csv", "paired.csv")]
    files.append(root / "digital_development_001/per_frame.csv")
    bindings = {str(path): sha256(path) for path in files}
    if not json.loads(files[0].read_text())["equal_budget_digital_included"]:
        raise RuntimeError("completed equal-resource digital evidence is required")
    summary = [row for row in read_csv(analysis / "summary.csv") if row["protocol"] == "common_paid_information"]
    costs = {(row["method"], row["scope"]): row for row in read_csv(analysis / "online_costs.csv")}
    points = []
    for row in summary:
        timing = costs.get((row["method"], row["scope"]))
        if timing is None or row["scope"] not in ("primary_1_4_7", "high_13_19"):
            continue
        points.append({"scope": row["scope"], "method": row["method"], "complex_uses": int(row["complex_uses"]),
                       "total_energy": int(row["complex_uses"]) * 2, **{metric: float(row[metric]) for metric in METRICS},
                       "TX_mean_ms": float(timing["TX_mean_ms"]), "RX_mean_ms": float(timing["RX_mean_ms"]),
                       "RX_p95_ms": float(timing["RX_p95_ms"]), "timing_source_images": int(timing["source_images"]),
                       "timing_scope_note": timing["scope_note"], "quality_source_images": int(row["source_images"]),
                       "empirical_frontier_only_not_statistical_or_application_success": True,
                       "source_class_assumptions_matched": False})
    for point in points:
        dominators = [other["method"] for other in points if other["scope"] == point["scope"] and dominate(other, point)]
        point["observed_nondominated_N_LPIPS_PSNR_RX"] = not dominators
        point["observed_dominators"] = ";".join(dominators)
    write_csv(output / "quality_resource_compute_points.csv", points)

    snrs = (1., 4., 7., 13., 19.)
    figure, axes = plt.subplots(3, 2, figsize=(13, 13))
    for row_index, budget in enumerate((3060, 4204, 4498)):
        methods = sorted({row["method"] for row in summary if int(row["complex_uses"]) == budget})
        for column, metric in enumerate(("psnr_db", "lpips")):
            axis = axes[row_index, column]
            for method in methods:
                selected = [next(row for row in summary if row["method"] == method and row["scope"] == f"snr_{snr:g}") for snr in snrs]
                means = np.array([float(row[metric]) for row in selected])
                lower = means - np.array([float(row[metric + "_ci_low"]) for row in selected])
                upper = np.array([float(row[metric + "_ci_high"]) for row in selected]) - means
                axis.errorbar(snrs, means, yerr=np.stack((lower, upper)), marker="o", capsize=3, label=name(method))
            suffix = "; HiFi pending" if budget == 4204 else ""
            axis.set(title=f"Equal N={budget}, E={2 * budget}{suffix}", xlabel="SNR (dB)", ylabel=metric)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    figure.suptitle("Completed development only: equal-resource quality with source-image95% intervals")
    figure.tight_layout(rect=(0, 0, 1, .97))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"equal_resource_quality.{extension}", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(17, 10))
    for column, scope in enumerate(("primary_1_4_7", "high_13_19")):
        selected = [row for row in points if row["scope"] == scope]
        for row_index, horizontal in enumerate(("complex_uses", "RX_mean_ms")):
            axis = axes[row_index, column]
            for ordinal, point in enumerate(selected):
                axis.scatter(point[horizontal], point["lpips"], marker="o", s=45,
                             label=f"{name(point['method'])}, N={point['complex_uses']}", color=plt.cm.tab20(ordinal % 20))
            axis.set(xscale="log", xlabel="Complex channel uses" if horizontal == "complex_uses" else "Complete RX processing mean (ms)",
                     ylabel="LPIPS", title=scope)
            axis.grid(alpha=.25)
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Measured workpoints, not an equal-budget ranking; RX excludes airtime/queueing")
    figure.tight_layout(rect=(0, .18, 1, .96))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"quality_resource_and_RX.{extension}", dpi=170)
    plt.close(figure)

    frames = read_csv(root / "digital_development_001/per_frame.csv")
    consequences, paired = [], []
    draws = np.random.default_rng(20260916).integers(100, size=(10000, 100))
    for budget in (4204, 4498):
        for scope, support in SCOPES:
            matrices = {}
            for family in ("raw", "arithmetic"):
                selected = [row for row in frames if int(row["complex_uses"]) == budget and row["family"] == family and float(row["snr_db"]) in support]
                failed = np.array([not int(row["accepted_correct"]) for row in selected])
                lpips = np.array([float(row["lpips"]) for row in selected])
                failed_ber = [float(row["source_bit_error_rate"]) for row, failure in zip(selected, failed) if failure and row["source_bit_error_rate"] != ""]
                matrices[family] = source_values(selected, support, list(range(100)))
                consequences.append({"complex_uses": budget, "scope": scope, "family": family, "transmissions": len(selected),
                                     "reliably_accepted_correct_rate": float(1 - failed.mean()),
                                     "actual_source_correct_rate": float(np.mean([int(row["source_correct"]) for row in selected])),
                                     "all_transmission_lpips": float(lpips.mean()), "not_reliably_correct_count": int(failed.sum()),
                                     "not_reliably_correct_mean_lpips": mean_or_blank(lpips[failed]),
                                     "reliably_correct_mean_lpips": mean_or_blank(lpips[~failed]),
                                     "not_reliably_correct_contribution_to_total_lpips": float(np.where(failed, lpips, 0).mean()),
                                     "not_reliably_correct_mean_source_BER": mean_or_blank(failed_ber),
                                     "source_BER_measurable_frames_in_failure_group": len(failed_ber),
                                     "header_failed_frames": sum(not int(row["header_accepted"]) for row in selected),
                                     "truth_used_for_offline_explanation_only": True})
            difference = matrices["raw"] - matrices["arithmetic"]
            interval = np.quantile(difference[draws].mean(axis=1), (.025, .975), axis=0)
            for position, metric in enumerate(METRICS):
                paired.append({"complex_uses": budget, "scope": scope, "method": "raw", "control": "arithmetic", "metric": metric,
                               "mean_difference": float(difference[:, position].mean()), "ci_low": float(interval[0, position]),
                               "ci_high": float(interval[1, position]), "source_images": 100, "bootstrap_resamples": 10000})
    write_csv(output / "failure_consequences.csv", consequences)
    write_csv(output / "raw_minus_arithmetic_paired.csv", paired)
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("completed evidence changed during CPU-only review")
    write_json(output / "completion.json", {"status": "COMPLETED_SYSTEM_POSITIONING_REVIEW", "bindings": bindings,
               "GPU_used": False, "new_model_inference": False, "new_training": False, "holdout_accessed": False,
               "unfinished_HiFi_quality_used": False, "frontier_uses_DINO": False,
               "scientific_boundaries": "measured workpoints only; cross-model information/training differs; m9 is not the full digital frontier",
               "finished_at": datetime.now().astimezone().isoformat()})
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
