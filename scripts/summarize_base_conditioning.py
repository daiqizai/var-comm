#!/usr/bin/env python3
"""Separate receiver-mechanism evidence from the unchanged system requirement."""

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


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
    quality_receipt = json.loads((arguments.quality / "completion.json").read_text())
    engineering = quality_receipt["status"] != "DEVELOPMENT_COMPLETE"
    trained = read_rows(arguments.quality / "per_frame.csv")
    references = read_rows(arguments.quality / "frozen_reference_rows.csv")
    rows = trained + references
    methods = sorted({row["arm"] for row in rows})
    identities = sorted({row["image_id"] for row in rows})
    keys = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate frame; do not inflate source-level uncertainty")
    metrics = ("psnr_db", "lpips", "dino", "mse")
    regions = {"mechanism": config["mechanism_snrs_db"], "primary": config["primary_snrs_db"], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in config["snrs_db"]})
    reduced, summaries, intervals = {}, [], []
    for region, snrs in regions.items():
        for arm in methods:
            values = {metric: [] for metric in metrics}
            for identifier in identities:
                frames = [keys[arm, identifier, snr, seed] for snr in snrs for seed in config["development_seeds"]]
                for metric in metrics:
                    values[metric].append(float(np.mean([float(row[metric]) for row in frames])))
            reduced[region, arm] = {metric: np.asarray(entries) for metric, entries in values.items()}
            summaries.append({"region": region, "arm": arm, "source_images": len(identities),
                              "context_ablation_not_a_ranked_system": arm == "conditioned_spatial_shuffle",
                              **{metric: float(np.mean(entries)) for metric, entries in values.items()}})
        comparisons = [("conditioned", reference) for reference in ("unconditioned", "raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "raw_m7", "raw_m8", "raw_m9")]
        comparisons += [("unconditioned", reference) for reference in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc")]
        comparisons += [("conditioned_spatial_shuffle", "conditioned")]
        for arm, control in comparisons:
            for metric in metrics:
                estimate = paired_interval(reduced[region, arm][metric] - reduced[region, control][metric],
                                           config["statistics"]["bootstrap_seed"], config["statistics"]["bootstrap_resamples"])
                intervals.append({"region": region, "arm": arm, "control": control, "metric": metric,
                                  "difference": estimate["gain"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
                                  "source_images": len(identities)})
    write_csv(output / "quality_summary.csv", summaries)
    write_csv(output / "source_paired_intervals.csv", intervals)
    failures = []
    for arm in config["arms"]:
        for snr in config["snrs_db"]:
            selected = [row for row in trained if row["arm"] == arm and float(row["snr_db"]) == snr]
            for category in ("header_failure", "body_crc_failure", "false_acceptance", "accepted_correct"):
                members = []
                for row in selected:
                    label = ("header_failure" if row["header_usable"] == "False" else "body_crc_failure" if row["body_crc_accepted"] == "False"
                             else "accepted_correct" if row["accepted_correct"] == "True" else "false_acceptance")
                    if label == category:
                        members.append(row)
                failures.append({"arm": arm, "snr_db": snr, "stratum": category, "frames": len(members),
                                 **{metric: float(np.mean([float(row[metric]) for row in members])) if members else "" for metric in metrics}})
    write_csv(output / "failure_strata.csv", failures)
    selection_record = json.loads((arguments.quality / "selection_frozen_before_development.json").read_text())
    choices = selection_record["selected"]
    calibration = []
    full = {}
    subset = set(np.linspace(0, 999, 100, dtype=int).tolist())
    for directory in sorted((arguments.training / "calibration").iterdir()):
        if not (directory / "completion.json").exists():
            continue
        step = int(directory.name.split("_")[1])
        records = read_rows(directory / "per_frame.csv")
        if directory.name.endswith("_full"):
            full[step] = json.loads((directory / "summary.json").read_text())
        for arm in config["arms"]:
            for region in ("mechanism", "primary"):
                selected = [row for row in records if row["arm"] == arm and (engineering or int(row["image_index"]) in subset)
                            and float(row["snr_db"]) in regions[region]]
                calibration.append({"new_step": step, "arm": arm, "region": region,
                                    **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in ("psnr_db", "lpips", "mse")}})
    write_csv(output / "matched_calibration_curves.csv", calibration)
    common_points = sorted(step for step in full if step >= 5000)[-2:]
    stability = []
    for step in common_points:
        condition = full[step]["conditioned"]["mechanism"]
        control = full[step]["unconditioned"]["mechanism"]
        lpips_gain, psnr_delta = control["lpips"] - condition["lpips"], condition["psnr_db"] - control["psnr_db"]
        stability.append({"new_step": step, "LPIPS_gain": lpips_gain, "PSNR_difference": psnr_delta,
                          "passes": lpips_gain >= config["mechanism"]["minimum_LPIPS_gain"] and psnr_delta >= -config["mechanism"]["stability_maximum_PSNR_drop_db"]})
    stable = len(stability) == 2 and all(row["passes"] for row in stability)
    contrast = {row["metric"]: row for row in intervals if row["region"] == "mechanism" and row["arm"] == "conditioned" and row["control"] == "unconditioned"}
    control_qualified = choices["unconditioned"]["training_reference_qualified"] and choices["conditioned"]["selection_guard_passed"]
    practical_lpips = contrast["lpips"]["difference"] <= -config["mechanism"]["minimum_LPIPS_gain"] and contrast["lpips"]["ci_high"] < 0
    strict = bool(not engineering and control_qualified and stable and practical_lpips and contrast["psnr_db"]["ci_low"] > 0 and contrast["dino"]["ci_low"] > 0)
    partial = bool(not engineering and control_qualified and stable and practical_lpips and not strict)
    system = {}
    for reference in ("raw_adaptive", "arithmetic_adaptive"):
        estimates = {row["metric"]: row for row in intervals if row["region"] == "primary" and row["arm"] == "conditioned" and row["control"] == reference}
        system[reference] = {"PSNR_positive_CI": estimates["psnr_db"]["ci_low"] > 0,
                             "PSNR_original_0p5_target": estimates["psnr_db"]["difference"] >= config["selection"]["system_desired_PSNR_gain_db"] and estimates["psnr_db"]["ci_low"] > 0,
                             "LPIPS_noninferiority": estimates["lpips"]["ci_high"] <= config["selection"]["system_LPIPS_margin"],
                             "contrasts": estimates}
    system_target = bool(not engineering and all(value["PSNR_original_0p5_target"] and value["LPIPS_noninferiority"] for value in system.values()))
    if engineering:
        decision = "ENGINEERING_ONLY_NO_SCIENTIFIC_CONCLUSION"
    elif not control_qualified:
        decision = "CONTROL_TRAINING_QUALIFICATION_NOT_MET_NO_MECHANISM_CLAIM"
    elif strict and system_target:
        decision = "DEVELOPMENT_MECHANISM_AND_SYSTEM_TARGET_SUPPORTED_NOT_INDEPENDENT_VALIDATION"
    elif strict:
        decision = "STABLE_THREE_METRIC_MECHANISM_ONLY_SYSTEM_TARGET_NOT_MET"
    elif partial:
        decision = "STABLE_LIMITED_METRIC_MECHANISM_EVIDENCE_NOT_THREE_METRIC_SUCCESS"
    else:
        decision = "NO_STABLE_PRACTICAL_CONDITIONING_INCREMENT_END_THIS_RECEIVER_CANDIDATE"
    report = {"decision": decision, "engineering_only": engineering, "selection": choices,
              "stable_calibration_increment": stable, "stability_common_checkpoints": stability,
              "mechanism_three_metrics_supported": strict, "mechanism_partial_supported": partial,
              "mechanism_contrasts": contrast, "system_original_target_met": system_target, "system": system,
              "old_1000_holdout_used_for_new_testing": False, "independent_validation": False,
              "automatic_resource_or_architecture_extensions": False}
    write_json(output / "decision.json", report)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric in zip(axes, ("psnr_db", "lpips", "dino")):
        for arm in ("unconditioned", "conditioned", "raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc"):
            axis.plot(config["snrs_db"], [float(np.mean(reduced[f"snr_{snr:g}", arm][metric])) for snr in config["snrs_db"]], marker="o", label=arm)
        axis.set(xlabel="SNR (dB)", ylabel=metric)
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=7)
    figure.suptitle("Development · all failures included · N3060/E6120 · mechanism region does not replace system region")
    figure.tight_layout()
    figure.savefig(output / "quality_by_snr.png", dpi=180)
    figure.savefig(output / "quality_by_snr.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row_index, region in enumerate(("mechanism", "primary")):
        for column, metric in enumerate(("psnr_db", "lpips")):
            for arm in config["arms"]:
                selected = sorted([row for row in calibration if row["region"] == region and row["arm"] == arm], key=lambda row: row["new_step"])
                axes[row_index, column].plot([row["new_step"] for row in selected], [row[metric] for row in selected], marker="o", label=arm)
            axes[row_index,column].set(title=region, xlabel="New paired updates", ylabel=metric)
            axes[row_index,column].grid(alpha=.25)
    axes[0,1].legend()
    figure.tight_layout()
    figure.savefig(output / "matched_calibration_curves.png", dpi=180)
    plt.close(figure)
    timings = read_rows(arguments.timing / "per_call.csv")
    timing_summary = []
    for region in ("mechanism", "primary", "high"):
        for arm in config["arms"]:
            selected = [row for row in timings if row["arm"] == arm and float(row["snr_db"]) in regions[region]]
            timing_summary.append({"region": region, "arm": arm, **{name.replace("_seconds", "_ms"): 1000 * float(np.mean([float(row[name]) for row in selected]))
                                                                      for name in ("tx_seconds", "rx_seconds", "processing_seconds")}})
    write_csv(output / "new_system_timing_ms.csv", timing_summary)
    lines = ["# 实际基图条件读取：机制与系统分开报告", "", f"决定：`{decision}`。", "", "## 改了什么",
             "仅给连续Decoder中间32/64空间恢复加入基图条件；两臂同参数同初始功能、原loss/PHY/分配不变。",
             "共享E/D也在两臂中同机会联合训练，因此结果不冒充编码器固定、波形完全相同的纯RX比较。",
             "", "## 7/13/19机制诊断（所有传输）", "", "| 方法 | PSNR | LPIPS | DINO |", "|---|---:|---:|---:|"]
    for row in summaries:
        if row["region"] == "mechanism" and row["arm"] in ["unconditioned", "conditioned", "conditioned_spatial_shuffle"]:
            lines.append(f"| {row['arm']} | {row['psnr_db']:.5f} | {row['lpips']:.6f} | {row['dino']:.6f} |")
    lines += ["", f"相同校准里程碑的稳定增量：{stable}；三指标区间支持：{strict}。空间打乱只作诊断，不进入正常系统排名。",
              "", "## 原1/4/7系统区间", "", "| 方法 | PSNR | LPIPS | DINO |", "|---|---:|---:|---:|"]
    for row in summaries:
        if row["region"] == "primary" and row["arm"] in ["unconditioned", "conditioned", "raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc"]:
            lines.append(f"| {row['arm']} | {row['psnr_db']:.5f} | {row['lpips']:.6f} | {row['dino']:.6f} |")
    lines += ["", f"原联合系统目标是否达到：{system_target}。不能用机制区间替换系统目标。",
              "", "## 成本与收口", "新增特征提取、下采样和投影已计入完整CPU端点RX时间，TX不使用免费缓存；详见timing CSV。",
              "停止条件、训练机会、选择点与资格见selection/decision JSON。达到上限不证明已经收敛；不合格训练控制不冒充强基线。",
              "未访问新holdout；旧数据/失败全部保留，不自动改资源或加模块。DINO本轮不训练/选模，但参与过历史数字模式开发。"]
    if engineering:
        lines.insert(1, "\n仅工程回归，不据少量更新/图像评价假设。")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "completion.json", {"status": "CONDITIONING_ANALYSIS_COMPLETE", "decision": decision,
               "sources": len(identities), "new_holdout": False, "report_sha256": sha256(output / "report.md"),
               "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
