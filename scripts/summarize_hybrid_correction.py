#!/usr/bin/env python3
"""Source-level paired quality, failure stratification, timing, and fixed plots."""

import argparse
import csv
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

from var_comm.study import paired_interval, write_csv, write_json


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
    quality_receipt = json.loads((arguments.quality / "completion.json").read_text())
    engineering_only = quality_receipt["status"] != "DEVELOPMENT_COMPLETE"
    rows = read_rows(arguments.quality / "per_frame.csv") + read_rows(arguments.quality / "frozen_reference_rows.csv")
    methods = sorted({row["arm"] for row in rows})
    metrics = ("psnr_db", "lpips", "dino", "mse")
    regions = {"primary": config["primary_snrs_db"], "high": [13., 19.]}
    regions.update({f"snr_{snr:g}": [snr] for snr in config["snrs_db"]})
    index = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in rows}
    if len(index) != len(rows):
        raise RuntimeError("duplicated physical frames")
    identities = sorted({row["image_id"] for row in rows})
    summary, pairs, stratified = [], [], []
    reduced = {}
    for region, snrs in regions.items():
        for arm in methods:
            values = {metric: [] for metric in metrics}
            for identifier in identities:
                frames = [index[arm, identifier, snr, seed] for snr in snrs for seed in config["development_seeds"]]
                for metric in metrics:
                    values[metric].append(np.mean([float(frame[metric]) for frame in frames]))
            reduced[region, arm] = {metric: np.asarray(array) for metric, array in values.items()}
            summary.append({"region": region, "arm": arm, "sources": len(identities),
                            **{metric: float(np.mean(array)) for metric, array in values.items()}})
        for arm in config["arms"]:
            for control in ["raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "raw_m7", "raw_m8", "raw_m9", "add", "snr_gain"]:
                if arm == control:
                    continue
                for metric in metrics:
                    interval = paired_interval(reduced[region, arm][metric] - reduced[region, control][metric],
                                               config["selection"]["bootstrap_seed"], config["selection"]["bootstrap_resamples"])
                    pairs.append({"region": region, "arm": arm, "control": control, "metric": metric,
                                  "difference": interval["gain"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                                  "source_images": len(identities)})
    for arm in config["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        for snr in config["snrs_db"]:
            for stratum in ("header_failure", "body_crc_failure", "false_acceptance", "accepted_correct"):
                filtered = []
                for row in selected:
                    actual = ("header_failure" if row["header_usable"] == "False" else
                              "body_crc_failure" if row["body_crc_accepted"] == "False" else
                              "accepted_correct" if row["accepted_correct"] == "True" else "false_acceptance")
                    if float(row["snr_db"]) == snr and actual == stratum:
                        filtered.append(row)
                stratified.append({"arm": arm, "snr_db": snr, "stratum": stratum, "frames": len(filtered),
                                   **{metric: float(np.mean([float(row[metric]) for row in filtered])) if filtered else "" for metric in metrics}})
    write_csv(output / "quality_summary.csv", summary)
    write_csv(output / "source_paired_intervals.csv", pairs)
    write_csv(output / "failure_strata.csv", stratified)
    focus = ["raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "add", "snr_gain", "reliability_gain"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric, label in zip(axes, metrics[:3], ("PSNR (dB) ↑", "LPIPS ↓", "DINO cosine ↑")):
        for arm in focus:
            values = [np.mean(reduced[f"snr_{snr:g}", arm][metric]) for snr in config["snrs_db"]]
            axis.plot(config["snrs_db"], values, marker="o", label=arm)
        axis.set(xlabel="SNR (dB)", ylabel=label)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    figure.suptitle("Development only · N=3060, E=6120 · all failed transmissions retained")
    figure.tight_layout()
    figure.savefig(output / "quality_by_snr.png", dpi=200)
    figure.savefig(output / "quality_by_snr.pdf")
    plt.close(figure)
    verdict = {}
    for control in ("raw_adaptive", "arithmetic_adaptive"):
        tests = {row["metric"]: row for row in pairs if row["region"] == "primary" and row["arm"] == "reliability_gain" and row["control"] == control}
        verdict[control] = {"PSNR_target_met": tests["psnr_db"]["difference"] >= config["selection"]["desired_psnr_gain_db"] and tests["psnr_db"]["ci_low"] > 0,
                            "LPIPS_noninferiority_met": tests["lpips"]["ci_high"] <= config["selection"]["lpips_noninferiority_margin_absolute"],
                            "PSNR": tests["psnr_db"], "LPIPS": tests["lpips"]}
    timing_summary = []
    if arguments.timing:
        timings = read_rows(arguments.timing / "per_frame.csv")
        for arm in config["arms"]:
            selected = [row for row in timings if row["arm"] == arm and float(row["snr_db"]) in config["primary_snrs_db"]]
            timing_summary.append({"arm": arm, **{name: 1000 * float(np.mean([float(row[name]) for row in selected]))
                                                  for name in ("tx_seconds", "rx_seconds", "processing_seconds")}, "unit": "ms"})
        write_csv(output / "new_system_timing_ms.csv", timing_summary)
    training_receipt = json.loads((arguments.training / "completion.json").read_text())
    qualified = not engineering_only and all(item["PSNR_target_met"] and item["LPIPS_noninferiority_met"] for item in verdict.values())
    write_json(output / "decision.json", {"registered_primary_target_met_on_development": qualified,
               "engineering_only": engineering_only,
               "comparisons": verdict, "independent_validation": False,
               "automatic_new_architecture_or_budget_search": False,
               "training_completion": training_receipt})
    lines = ["# 固定分配源残差混合通信：development收敛结果", "", "## 主1/4/7 dB", "",
             "| 方法 | PSNR↑ | LPIPS↓ | DINO↑ |", "|---|---:|---:|---:|"]
    for row in summary:
        if row["region"] == "primary":
            lines.append(f"| {row['arm']} | {row['psnr_db']:.5f} | {row['lpips']:.6f} | {row['dino']:.6f} |")
    lines += ["", f"预先登记的主目标在development上{'满足' if qualified else '未满足'}。这不是新holdout验证，也不等于已证明可靠度融合的新颖性。",
              "可靠度本身的作用另看同参数snr_gain配对差；强对照、固定m7更多FEC、固定m8/m9更多真实token全部保留。",
              "所有失败进入分母；先按源图平均原三噪声再bootstrap。完整逐SNR、失败分层和配对区间在同目录CSV。",
              "训练仅为本次固定10000更新里程碑，训练充分性应结合校准曲线，不据单一分配或短训否定混合通信架构。",
              "处理时间包含真实TX VAR补全，不使用训练缓存，且不含空口/排队/模型加载/类别获取/CSI估计。",
              "停止自动扩展；下一步如需独立验证，必须先冻结最终方法并另建未使用数据名单。"]
    if engineering_only:
        lines.insert(1, "\n**仅工程回归，不是方案的科学性能结果；不能据本表判断训练方案优劣。**")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "completion.json", {"status": "BOUNDED_DEVELOPMENT_ANALYSIS_COMPLETE", "sources": len(identities),
               "primary_target_met": qualified, "independent_validation": False, "new_holdout_accessed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args())
