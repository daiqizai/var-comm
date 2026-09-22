"""Frozen three-weight closure: actual-point selection and expectation references."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def weight_name(value):
    return {0.01: "lpips_0p01", 0.03: "lpips_0p03", 0.1: "lpips_0p1"}[float(value)]


def specifications(closure):
    return {f"{weight_name(weight)}__{variant}": {"weight": weight, "variant": variant,
             "reused_training": weight == .01} for weight in closure["lambdas"] for variant in closure["variants"]}


def load_configs():
    closure = json.loads((ROOT / "configs/hybrid_weight_closure.json").read_text())
    base = json.loads((ROOT / closure["base_config"]).read_text())
    runtime = copy.deepcopy(base)
    runtime["arms"] = [name for name, item in specifications(closure).items() if not item["reused_training"]]
    runtime["training"]["minimum_new_updates"] = closure["new_updates_each_operating_arm"]
    runtime["training"]["maximum_new_updates"] = closure["new_updates_each_operating_arm"]
    return closure, base, runtime


def select_actual_points(full_summaries, closure):
    chosen = {}
    for arm, item in specifications(closure).items():
        candidates = []
        for step, summary in full_summaries.items():
            if int(step) not in closure["checkpoint_selection"]["eligible_steps"]:
                continue
            metrics = summary[arm]["primary"]
            score = metrics["mse"] + item["weight"] * metrics["lpips"]
            candidates.append({"step": int(step), "score": float(score), "primary_calibration": metrics, **item})
        if not candidates:
            raise RuntimeError("each registered weight/structure needs actual completed calibration checkpoints")
        chosen[arm] = min(candidates, key=lambda candidate: (candidate["score"], candidate["step"]))
    return chosen


def select_matched_controls(calibration, closure):
    specification = specifications(closure)
    rules = closure["matched_operating_points"]
    records = []
    controls = [arm for arm, item in specification.items() if item["variant"] == "unconditioned"]
    conditions = [arm for arm, item in specification.items() if item["variant"] == "conditioned"]
    for region in rules["regions"]:
        for arm in conditions:
            target = calibration[arm][region]
            for kind, metric, tolerance in (("matched_LPIPS", "lpips", rules["LPIPS_absolute_tolerance"]),
                                             ("matched_PSNR", "psnr_db", rules["PSNR_absolute_tolerance_db"])):
                eligible = [control for control in controls if abs(calibration[control][region][metric] - target[metric]) <= tolerance]
                record = {"region": region, "conditioned_arm": arm, "matching": kind,
                          "metric_tolerance": tolerance, "coverage": bool(eligible), "control_arm": None}
                if eligible:
                    if kind == "matched_LPIPS":
                        control = min(eligible, key=lambda name: (-calibration[name][region]["psnr_db"],
                                      abs(calibration[name][region][metric] - target[metric]), specification[name]["weight"]))
                    else:
                        control = min(eligible, key=lambda name: (calibration[name][region]["lpips"],
                                      abs(calibration[name][region][metric] - target[metric]), specification[name]["weight"]))
                    record.update(control_arm=control, calibration_delta=target[metric] - calibration[control][region][metric])
                records.append(record)
    return records


def expected_metric(digital, deep, probability):
    if not 0 <= probability <= 1:
        raise ValueError("selection probability must lie in [0,1]")
    return (1 - probability) * np.asarray(digital) + probability * np.asarray(deep)


def fit_time_sharing(calibration, references, closure):
    deep = references["perceptual_deepjscc"]
    digitals = ("raw_adaptive", "arithmetic_adaptive")
    result = []
    for arm, regions in calibration.items():
        target = regions["primary"]
        for constraint, metric in (("same_LPIPS", "lpips"), ("same_PSNR", "psnr_db")):
            candidates = []
            for digital in digitals:
                start = references[digital]
                denominator = deep[metric] - start[metric]
                if abs(denominator) < 1e-12:
                    if abs(target[metric] - start[metric]) > 1e-12:
                        continue
                    probability = 1.0 if (deep["psnr_db"] >= start["psnr_db"] if metric == "lpips" else deep["lpips"] <= start["lpips"]) else 0.0
                else:
                    probability = (target[metric] - start[metric]) / denominator
                if not -1e-12 <= probability <= 1 + 1e-12:
                    continue
                probability = float(np.clip(probability, 0, 1))
                metrics = {name: float(expected_metric(start[name], deep[name], probability)) for name in ("psnr_db", "lpips", "mse", "dino")}
                candidates.append({"digital": digital, "p_deep": probability, "calibration_expected": metrics})
            record = {"target_arm": arm, "criterion": constraint, "coverage": bool(candidates), "digital": None, "p_deep": None}
            if candidates:
                if metric == "lpips":
                    selected = min(candidates, key=lambda item: (-item["calibration_expected"]["psnr_db"], item["digital"] != "raw_adaptive"))
                else:
                    selected = min(candidates, key=lambda item: (item["calibration_expected"]["lpips"], item["digital"] != "raw_adaptive"))
                record.update(selected)
            result.append(record)
    limit = min(references[name]["lpips"] for name in digitals) + closure["system"]["LPIPS_noninferiority_margin"]
    candidates = []
    for digital in digitals:
        start = references[digital]
        if deep["lpips"] <= limit:
            probability = 1.0
        elif start["lpips"] <= limit and deep["lpips"] > start["lpips"]:
            probability = float(np.clip((limit - start["lpips"]) / (deep["lpips"] - start["lpips"]), 0, 1))
        else:
            continue
        metrics = {name: float(expected_metric(start[name], deep[name], probability)) for name in ("psnr_db", "lpips", "mse", "dino")}
        candidates.append({"digital": digital, "p_deep": probability, "calibration_expected": metrics})
    result.append({"target_arm": "original_joint_target", "criterion": "LPIPS_budget", "coverage": bool(candidates),
                   **(min(candidates, key=lambda item: (-item["calibration_expected"]["psnr_db"], item["digital"] != "raw_adaptive")) if candidates else {"digital": None, "p_deep": None})})
    for index, record in enumerate(result):
        record["reference_id"] = f"expected_mix_{index:02d}"
        record["ratio_fitted_on"] = "calibration_only"
        record["actual_deployment_tested"] = False
    return result
