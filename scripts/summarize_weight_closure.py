#!/usr/bin/env python3
"""Measured operating points and calibration-fixed expectation comparisons, not an interpolated learned frontier."""

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

from var_comm.study import paired_interval, sha256, write_csv, write_json
from var_comm.weight_closure import expected_metric, load_configs, specifications


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    closure, base, _runtime = load_configs()
    specs = specifications(closure)
    actual = read_rows(arguments.quality / "per_frame.csv")
    references = read_rows(arguments.quality / "frozen_reference_rows.csv")
    engineering = json.loads((arguments.quality / "completion.json").read_text())["status"] != "WEIGHT_CLOSURE_DEVELOPMENT_COMPLETE"
    frozen = json.loads((arguments.quality / "frozen_before_development.json").read_text())
    pure = actual + references
    identities = sorted({row["image_id"] for row in pure})
    index = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in pure}
    if len(index) != len(pure):
        raise RuntimeError("duplicated measured frame")
    mixtures = frozen.get("time_sharing", [])
    expectation_rows = []
    metrics = ("psnr_db", "lpips", "dino", "mse")
    for mixture in mixtures:
        if not mixture["coverage"]:
            continue
        for identifier in identities:
            for snr in base["snrs_db"]:
                for seed in base["development_seeds"]:
                    digital = index[mixture["digital"], identifier, snr, seed]
                    deep = index["perceptual_deepjscc", identifier, snr, seed]
                    expectation_rows.append({"arm": mixture["reference_id"], "image_id": identifier, "snr_db": snr, "seed": seed,
                        **{metric: float(expected_metric(float(digital[metric]), float(deep[metric]), mixture["p_deep"])) for metric in metrics},
                        "p_deep": mixture["p_deep"], "digital": mixture["digital"], "actual_deployment": False,
                        "complex_uses_each_selected_frame": 3060, "energy_each_selected_frame": 6120})
    if expectation_rows:
        write_csv(output / "expected_time_sharing_per_frame.csv", expectation_rows)
    all_rows = pure + expectation_rows
    all_index = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in all_rows}
    methods = sorted({row["arm"] for row in all_rows})
    regions = {"primary": base["primary_snrs_db"], "mechanism": base["mechanism_snrs_db"], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in base["snrs_db"]})
    summary, reduced = [], {}
    for region, snrs in regions.items():
        for arm in methods:
            values = {metric: [] for metric in metrics}
            for identifier in identities:
                records = [all_index[arm, identifier, snr, seed] for snr in snrs for seed in base["development_seeds"]]
                for metric in metrics:
                    values[metric].append(float(np.mean([float(record[metric]) for record in records])))
            reduced[region, arm] = {metric: np.asarray(entries) for metric, entries in values.items()}
            summary.append({"region": region, "arm": arm, "source_images": len(identities), "expected_not_deployed": arm.startswith("expected_mix_"),
                            **{metric: float(np.mean(entries)) for metric, entries in values.items()}})
    write_csv(output / "quality_summary.csv", summary)
    comparisons = set()
    for arm in specs:
        for control in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "raw_m7", "raw_m8", "raw_m9"):
            comparisons.add((arm, control))
    conditions = [arm for arm, item in specs.items() if item["variant"] == "conditioned"]
    controls = [arm for arm, item in specs.items() if item["variant"] == "unconditioned"]
    comparisons.update((arm, control) for arm in conditions for control in controls)
    for mixture in mixtures:
        if mixture["coverage"]:
            if mixture["target_arm"] in specs:
                comparisons.add((mixture["target_arm"], mixture["reference_id"]))
            else:
                comparisons.update((arm, mixture["reference_id"]) for arm in specs)
                comparisons.update((mixture["reference_id"], name) for name in ("raw_adaptive", "arithmetic_adaptive"))
    intervals = []
    for region in regions:
        for arm, control in sorted(comparisons):
            for metric in metrics:
                estimate = paired_interval(reduced[region, arm][metric] - reduced[region, control][metric],
                                           closure["system"]["bootstrap_seed"], closure["system"]["source_paired_bootstrap_resamples"])
                intervals.append({"region": region, "arm": arm, "control": control, "metric": metric,
                                  "difference": estimate["gain"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
                                  "source_images": len(identities), "involves_expected_reference": control.startswith("expected_mix_") or arm.startswith("expected_mix_")})
    write_csv(output / "source_paired_intervals.csv", intervals)
    interval_lookup = {(row["region"], row["arm"], row["control"], row["metric"]): row for row in intervals}
    matching_results = []
    for match in frozen.get("matches", []):
        record = dict(match)
        record.update(development_close=False, PSNR_direction_supported=False, LPIPS_direction_supported=False, practical_matched_advantage=False)
        if match["coverage"]:
            arm, control, region = match["conditioned_arm"], match["control_arm"], match["region"]
            psnr = interval_lookup[region, arm, control, "psnr_db"]
            lpips = interval_lookup[region, arm, control, "lpips"]
            record.update(PSNR_difference=psnr["difference"], LPIPS_difference=lpips["difference"],
                          PSNR_direction_supported=psnr["ci_low"] > 0, LPIPS_direction_supported=lpips["ci_high"] < 0)
            if match["matching"] == "matched_LPIPS":
                close = abs(lpips["difference"]) <= closure["matched_operating_points"]["LPIPS_absolute_tolerance"]
                succeeds = close and lpips["ci_high"] <= .005 and psnr["ci_low"] > 0 and psnr["difference"] >= .1
            else:
                close = abs(psnr["difference"]) <= closure["matched_operating_points"]["PSNR_absolute_tolerance_db"]
                succeeds = close and psnr["ci_low"] >= -.1 and lpips["ci_high"] < 0 and lpips["difference"] <= -.002
            record.update(development_close=close, practical_matched_advantage=bool(succeeds))
        matching_results.append(record)
    write_json(output / "matched_actual_points.json", {"matches": matching_results, "interpolated_learned_points": False,
               "selected_on_calibration_only": True, "engineering_only": engineering})
    if matching_results:
        fields = sorted({field for row in matching_results for field in row})
        write_csv(output / "matched_actual_points.csv", [{field: row.get(field, "") for field in fields} for row in matching_results])
    dominance = []
    fixed_expectations = [item["reference_id"] for item in mixtures if item["coverage"]]
    for region in ("primary", "mechanism"):
        for arm in specs:
            for reference in ["raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", *fixed_expectations]:
                point = reduced[region, arm]
                comparator = reduced[region, reference]
                delta_psnr = float(np.mean(point["psnr_db"] - comparator["psnr_db"]))
                delta_lpips = float(np.mean(point["lpips"] - comparator["lpips"]))
                delta_dino = float(np.mean(point["dino"] - comparator["dino"]))
                dominance.append({"region": region, "arm": arm, "reference": reference,
                                  "PSNR_minus_reference": delta_psnr, "LPIPS_minus_reference": delta_lpips, "DINO_minus_reference": delta_dino,
                                  "mean_PSNR_LPIPS_covered": delta_psnr <= 0 and delta_lpips >= 0,
                                  "mean_three_metrics_covered": delta_psnr <= 0 and delta_lpips >= 0 and delta_dino <= 0,
                                  "reference_is_expectation_not_deployment": reference.startswith("expected_mix_")})
    write_csv(output / "fixed_reference_coverage.csv", dominance)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, region in zip(axes, ("primary", "mechanism")):
        for variant, marker in (("unconditioned", "o"), ("conditioned", "s")):
            arms = [arm for arm, item in specs.items() if item["variant"] == variant]
            values = [(float(np.mean(reduced[region, arm]["psnr_db"])), float(np.mean(reduced[region, arm]["lpips"]))) for arm in arms]
            axis.scatter([value[0] for value in values], [value[1] for value in values], marker=marker, s=55, label=variant)
            for arm, value in zip(arms, values):
                axis.annotate(str(specs[arm]["weight"]), value, xytext=(4, 4), textcoords="offset points", fontsize=8)
        deep = [float(np.mean(reduced[region, "perceptual_deepjscc"][metric])) for metric in ("psnr_db", "lpips")]
        for digital in ("raw_adaptive", "arithmetic_adaptive"):
            anchor = [float(np.mean(reduced[region, digital][metric])) for metric in ("psnr_db", "lpips")]
            axis.plot([anchor[0], deep[0]], [anchor[1], deep[1]], "--", alpha=.65, label=digital + " / Deep expectation")
            axis.scatter(*anchor, marker="^", s=45)
        axis.scatter(*deep, marker="*", s=95, label="frozen Deep")
        axis.set(xlabel="PSNR (dB) ↑", ylabel="LPIPS ↓", title=region)
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=7)
    figure.suptitle("Dots: measured models; dashed segments: selector expectations, NOT deployed mixtures")
    figure.tight_layout()
    figure.savefig(output / "measured_tradeoffs_and_expectations.png", dpi=190)
    figure.savefig(output / "measured_tradeoffs_and_expectations.pdf")
    plt.close(figure)
    curves = []
    for directory in sorted((arguments.training / "calibration").iterdir()):
        if not (directory / "completion.json").exists():
            continue
        step = int(directory.name.split("_")[1])
        records = read_rows(directory / "per_frame.csv")
        selected_indices = set(np.linspace(0, 999, 100, dtype=int).tolist())
        for arm in specs:
            for region in ("primary", "mechanism"):
                selected = [row for row in records if row["arm"] == arm and (engineering or int(row["image_index"]) in selected_indices)
                            and float(row["snr_db"]) in regions[region]]
                if selected:
                    curves.append({"new_step": step, "arm": arm, "region": region,
                                   **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in ("mse", "psnr_db", "lpips")}})
    if curves:
        write_csv(output / "matched_calibration_curves.csv", curves)
    write_json(output / "review_inputs.json", {"status": "NUMERICAL_CLOSURE_READY_FOR_EVIDENCE_REVIEW", "engineering_only": engineering,
               "actual_workpoints": frozen["models"], "time_sharing_calibration_fixed": mixtures,
               "structure_matching": matching_results, "new_training_or_search_authorized_after_this": False,
               "mean_coverage_is_not_theoretical_or_deployment_dominance": True, "no_new_holdout": True})
    write_json(output / "completion.json", {"status": "WEIGHT_CLOSURE_NUMERICS_COMPLETE", "sources": len(identities),
               "measured_rows": len(actual), "expected_rows": len(expectation_rows), "new_holdout": False,
               "all_choices_frozen_on_calibration": not engineering, "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
