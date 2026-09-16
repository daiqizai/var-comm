#!/usr/bin/env python3
"""CPU-only closure of the already-running author measurement queue."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src"), str(EXPERIMENT / "scripts")]

import numpy as np

from external_positioning.analysis import METRICS, SCOPES, source_values, summarize, validate_groups
from summarize_available import cost_figure, figures, load_rows, read_csv, timing_rows
from var_comm.study import sha256, write_csv, write_json


def frame_key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def paired_rows(method_rows, control_rows, label, relation):
    if {frame_key(row) for row in method_rows} != {frame_key(row) for row in control_rows}:
        raise RuntimeError("paired transmissions do not match")
    draws = np.random.default_rng(20260916).integers(100, size=(10000, 100))
    result = []
    for scope, support in SCOPES:
        difference = source_values(method_rows, support, list(range(100))) - source_values(control_rows, support, list(range(100)))
        intervals = np.quantile(difference[draws].mean(axis=1), (.025, .975), axis=0)
        for index, metric in enumerate(METRICS):
            result.append({"contrast": label, "scope": scope, "metric": metric, "method": "hifi_diffcom_c2",
                           "control": control_rows[0]["method"], "mean_difference": float(difference[:, index].mean()),
                           "ci_low": float(intervals[0, index]), "ci_high": float(intervals[1, index]),
                           "source_images": 100, "resource_relation": relation,
                           "method_complex_uses": 4204, "standalone_control_complex_uses": int(control_rows[0]["complex_uses"])})
    return result


def verify_observation_pairing(hifi, companion, standalone):
    indexed = {frame_key(row): row for row in standalone}
    companion_index = {frame_key(row): row for row in companion}
    if len(hifi) != 1500 or len(companion) != 1500 or len(indexed) != 1500:
        raise RuntimeError("full1500 matched transmissions are required")
    keys = {frame_key(row) for row in hifi}
    if keys != set(indexed) or keys != set(companion_index):
        raise RuntimeError("same-observation populations differ")
    maximum, exact, fallbacks = 0., 0, 0
    for row in hifi:
        base, original = companion_index[frame_key(row)], indexed[frame_key(row)]
        for field in ("source_pixels_sha256", "checkpoint_sha256", "data_transmitted_sha256", "data_observed_sha256", "standard_data_noise_sha256"):
            if row[field] != base[field] or base[field] != original[field]:
                raise RuntimeError(f"same-observation contract changed: {field}, {frame_key(row)}")
        with np.load(base["image_archive"], allow_pickle=False) as archive:
            images = archive["images"]
            paired_image = images[int(base["image_slot"])]
            hifi_image = images[int(row["image_slot"])]
        with np.load(original["image_archive"], allow_pickle=False) as archive:
            standalone_image = archive["images"][int(original["image_slot"])]
        for image, record in ((paired_image, base), (hifi_image, row), (standalone_image, original)):
            if image.shape != (3, 256, 256) or image.dtype != np.float32 or not np.isfinite(image).all():
                raise RuntimeError("invalid stored receiver RGB")
            if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != record["image_sha256"]:
                raise RuntimeError("stored receiver pixels changed after metric evaluation")
        error = float(np.max(np.abs(paired_image.astype(np.float64) - standalone_image)))
        maximum = max(maximum, error)
        exact += int(np.array_equal(paired_image, standalone_image))
        if error > 2e-4:
            raise RuntimeError("ADJSCC exceeds the existing author functional-qualification RGB tolerance")
        if row["fallback"]:
            if row["fallback"] != "invalid_metadata_same_observation_ADJSCC" or row["image_sha256"] != base["image_sha256"]:
                raise RuntimeError("metadata failure did not preserve the registered actual fallback")
            fallbacks += 1
        elif int(row["NFE"]) <= 0 or row.get("complete_author_schedule") not in ("True", True):
            raise RuntimeError("a nonfallback HiFi output did not use the full registered sampling schedule")
    return {"paired_transmissions": len(hifi), "same_data_waveform_observation_and_noise_hashes": True,
            "ADJSCC_bit_exact_images": exact, "ADJSCC_max_abs_RGB_difference": maximum,
            "RGB_tolerance_from_existing_qualification": 2e-4, "mechanism_control": "actual_same_session_ADJSCC_output",
            "standalone_ADJSCC_kept_at_N4096": True, "metadata_fallbacks": fallbacks,
            "not_a_claim_of_bit_exact_RGB": exact != 1500}


def make_report(summary, contrasts, costs, audit, output):
    by_quality = {(row["method"], row["scope"]): row for row in summary if row["protocol"] == "common_paid_information"}
    by_cost = {(row["method"], row["scope"]): row for row in costs}
    lines = ["# 外部方法研究定位：全量评测完成", "",
             f"生成时间：{datetime.now().astimezone().isoformat()}。原100 development、五SNR、三噪声；未训练、未访问新holdout。", "",
             "## 1. 同观测机制与等资源系统，分别判断", "",
             "ADJSCC→HiFi严格共享4096个数据符号及实际数据观测；HiFi另付108次元数据，总N4204/E8408。机制对照用HiFi会话内实际ADJSCC输出。独立ADJSCC系统仍N4096，不被强加header，也不重复计入系统统计。", "",
             "| 比较：HiFi−对照 | 区间 | ΔPSNR及95%CI | ΔLPIPS及95%CI |", "|---|---|---|---|"]
    for contrast in ("same_observation_ADJSCC_to_HiFi", "same_N4204_raw", "same_N4204_arithmetic"):
        for scope in ("primary_1_4_7", "high_13_19"):
            selected = {row["metric"]: row for row in contrasts if row["contrast"] == contrast and row["scope"] == scope}
            formatted = [f"{selected[metric]['mean_difference']:+.6f} [{selected[metric]['ci_low']:+.6f},{selected[metric]['ci_high']:+.6f}]" for metric in ("psnr_db", "lpips")]
            lines.append(f"| {contrast} | {scope} | {formatted[0]} | {formatted[1]} |")
    lines.extend(["", "## 2. 完整质量与RX处理代价", "",
                  "下表跨N展示工作点而非排名。RX是全区间实测均值/p95，不以单帧代表HiFi；不含空口、排队、模型加载与指标。旧系统与作者环境/会话、计时源数不同，不能据微小差别作纯架构归因。", "",
                  "| 系统 | 区间 | N | PSNR | LPIPS | DINO | RX均值/p95 ms |", "|---|---|---:|---:|---:|---:|---:|"])
    methods = ("raw_adaptive", "arithmetic_adaptive", "wetok_r3", "perceptual_deepjscc", "raw_adaptive_N4204",
               "arithmetic_adaptive_N4204", "swin_ra32", "adjscc_c2", "hifi_diffcom_c2")
    for scope in ("primary_1_4_7", "high_13_19"):
        for method in methods:
            row, timing = by_quality[method, scope], by_cost[method, scope]
            lines.append(f"| {method} | {scope} | {row['complex_uses']} | {row['psnr_db']:.4f} | {row['lpips']:.5f} | {row['dino']:.5f} | {float(timing['RX_mean_ms']):.2f}/{float(timing['RX_p95_ms']):.2f} |")
    lines.extend(["", "## 3. 定位与边界", "",
                  "服务问题是单次固定带宽/总能量、不可重传的图像传输：同时考虑像素、感知、失败输出和RX处理预算。应用容差尚未冻结，不能把数学非支配自动称为实用成功。", ""])
    for scope in ("primary_1_4_7", "high_13_19"):
        gain = next(row for row in contrasts if row["contrast"] == "same_observation_ADJSCC_to_HiFi" and row["scope"] == scope and row["metric"] == "lpips")
        conclusion = "支持HiFi的感知恢复增量" if gain["ci_high"] < 0 else "HiFi的LPIPS明确更差" if gain["ci_low"] > 0 else "尚不能确认HiFi的LPIPS增量"
        lines.append(f"- {scope}：{conclusion}；这不是纯先验容量因果实验，也不自动等于系统收益。")
    lines.extend(["- 数字VAR、R3、Swin和感知Deep保留其已测工作点。若服务要求低处理代价/像素保真，以Swin或Deep为起点；若允许像素偏移而重视低带宽感知，依据上表保留数字VAR/R3；HiFi仅在实测恢复收益值得其处理代价的需求下保留。不是要求保住VAR。",
                  "- ARPC已有next-scale渐进压缩/补全，Ada-TokenCom已有AR算术码/前缀与MCS适配。当前差异是单次固定N/E、无ARQ、失败质量计分和计算口径，不是自动成立的新机制。参见external_related_work_difference_20260916.md；旧算术quality=BLER负结果不改写。",
                  "- 最小证据缺口：现有失败后果取舍是否仍不能被普通误差韧性/可重同步熵码对照覆盖。这里不启动此对照或新holdout；若被普通方法覆盖，则更换研究起点，不增加VAR模块。",
                  "- m9饱和仅属于当前表示和候选集，不是数字VAR整体上限。TX真类别已知、理想SNR共享、训练域/先验规模不一致继续披露；DINO有既往开发暴露。",
                  f"- 配对审计：{audit['paired_transmissions']}次同波形/同数据观测；ADJSCC两会话最大RGB差{audit['ADJSCC_max_abs_RGB_difference']:.3g}，不虚称全部像素逐字节相同。所有失败保留。", "",
                  f"全部表/图/逐图数据：`{output.relative_to(PROJECT)}`。采样步数和完整RX逐SNR分布见`hifi_cost_by_snr.csv`；原表NFE记录的是反向采样步数，不冒充精确FLOPs/包括反向重计算的模型调用数。"])
    return "\n".join(lines) + "\n"


def one_page(summary, contrasts, costs):
    quality = {(row["method"], row["scope"]): row for row in summary if row["protocol"] == "common_paid_information"}
    timing = {(row["method"], row["scope"]): row for row in costs}
    lines = ["# 一页研究定位：HiFi全量完成后的工作点", "",
             "**只用原development；不是新的最终测试。** 需求：每帧固定带宽/能量、无重传的低码率图像传输，同时约束失败输出、感知/像素失真及完整RX处理代价。", "",
             "## 该保留什么", "", "区间/N不同，不能跨行当等资源排名；下列时间不是含空口/排队的端到端时延。", "",
             "| 系统/用途候选 | 区间 | N | PSNR / LPIPS | RX均值ms |", "|---|---|---:|---:|---:|"]
    for method, scope, role in (("raw_adaptive_N4204", "primary_1_4_7", "raw VAR：感知优先预览"),
                               ("hifi_diffcom_c2", "primary_1_4_7", "HiFi：较宽松计算预算恢复"),
                               ("wetok_r3", "high_13_19", "R3：高SNR学习参考"),
                               ("perceptual_deepjscc", "primary_1_4_7", "Deep：轻计算/像素"),
                               ("swin_ra32", "primary_1_4_7", "Swin：像素/速度")):
        row, cost = quality[method, scope], timing[method, scope]
        lines.append(f"| {role} | {scope} | {row['complex_uses']} | {row['psnr_db']:.2f} / {row['lpips']:.4f} | {float(cost['RX_mean_ms']):.2f} |")
    lines.extend(["", "## 机制与系统分开", ""])
    for contrast, label in (("same_observation_ADJSCC_to_HiFi", "同4096数据观测，HiFi−ADJSCC"),
                            ("same_N4204_raw", "同N4204，HiFi−raw VAR"),
                            ("same_N4204_arithmetic", "同N4204，HiFi−算术VAR")):
        row = next(row for row in contrasts if row["contrast"] == contrast and row["scope"] == "primary_1_4_7" and row["metric"] == "lpips")
        lines.append(f"- {label}：ΔLPIPS{row['mean_difference']:+.5f}，95%CI[{row['ci_low']:+.5f},{row['ci_high']:+.5f}]。")
    lines.extend(["HiFi另付108次元数据；裸ADJSCC仍为4096，不被加header削弱。全五SNR、失败和采样步数分布均报告，不用单帧推断整体速度。", "",
                  "## 最近工作差异与最小缺口", "",
                  "ARPC已有next-scale渐进压缩/补全，Ada-TokenCom已有AR熵编码与源率/MCS适配。我们的具体边界是单次固定N/E、无ARQ、失败质量及完整计算账本；这不是自动成立的新增算法。最小证据缺口是该失败风险取舍能否胜过常规误差韧性/可重同步熵码强对照；当前不启动新对照或holdout。", "",
                  "**决定：按表中真实取舍保留系统，不以保住VAR为目标。** 感知优先且能承受其像素偏移时才保留数字VAR；高SNR保留R3；像素/快速接收需求以Swin或Deep为起点；只有恢复收益值得计算代价时才保留HiFi。若普通强对照覆盖所谓机制，就更换研究起点，不增加模块。", "",
                  "m9饱和不是数字VAR整体上限。TX真类别、理想SNR、不同训练域/先验容量及DINO既往暴露继续披露；应用容差尚未冻结，不把样本均值冒称实时保证或数学非支配冒称实用成功。详表与原文差异见external_baseline_positioning_result.md及external_related_work_difference_20260916.md。"])
    return "\n".join(lines) + "\n"


def finish(arguments):
    root, output = arguments.root, arguments.output
    for directory in ("author_queue_001", "development_hifi_001", "hifi_metrics_001"):
        if not (root / directory / "completion.json").exists():
            raise RuntimeError("full inference, scoring and original queue receipts are required before reporting")
    metric_path = root / "hifi_metrics_001/per_frame.csv"
    receipt = json.loads((metric_path.parent / "completion.json").read_text())
    if receipt["rows"] != 4500 or sha256(metric_path) != receipt["per_frame_sha256"]:
        raise RuntimeError("HiFi metric coverage or artifact binding changed")
    base_rows, sources, bindings = load_rows(root, [root / "digital_development_001"])
    new = read_csv(metric_path)
    hifi = [row for row in new if row["method"] == "hifi_diffcom_c2" and row["protocol"] == "common_paid_information"]
    companion = [row for row in new if row["method"] == "adjscc_c2"]
    standalone = [row for row in base_rows if row["method"] == "adjscc_c2"]
    audit = verify_observation_pairing(hifi, companion, standalone)
    combined = base_rows + [row for row in new if row["method"] == "hifi_diffcom_c2"]
    for row in combined:
        row["total_energy"] = row.get("total_energy", row.get("actual_total_energy"))
    validate_groups(combined, sources)
    if len(combined) != 28500:
        raise RuntimeError("a partial point or duplicate ADJSCC baseline entered the system table")
    output.mkdir(parents=True, exist_ok=False)
    summary, general_paired, per_source = summarize(combined, sources)
    contrasts = paired_rows(hifi, companion, "same_observation_ADJSCC_to_HiFi", "same4096_data_observation;HiFi108_paid_metadata_extra;standalone_ADJSCC_not_penalized")
    for family in ("raw", "arithmetic"):
        selected = [row for row in base_rows if row["method"] == f"{family}_adaptive_N4204"]
        contrasts.extend(paired_rows(hifi, selected, f"same_N4204_{family}", "same_total_N4204_and_energy8408"))
    costs = timing_rows(combined, root / "digital_timing_001")
    fields = sorted({key for row in combined for key in row})
    write_csv(output / "per_frame.csv", [{key: row.get(key, "") for key in fields} for row in combined])
    for filename, rows in (("summary", summary), ("paired", general_paired), ("per_source", per_source),
                           ("hifi_mechanism_and_equal_budget", contrasts), ("online_costs", costs)):
        write_csv(output / f"{filename}.csv", rows)
    hifi_cost = []
    for scope, support in SCOPES + [("all_5_snrs", (1., 4., 7., 13., 19.))]:
        selected = [row for row in hifi if float(row["snr_db"]) in support]
        elapsed = np.array([float(row["RX_seconds"]) for row in selected])
        steps = np.array([int(row["NFE"]) for row in selected])
        hifi_cost.append({"scope": scope, "frames": len(selected), "sources": 100, "RX_mean_seconds": float(elapsed.mean()),
                          "RX_median_seconds": float(np.median(elapsed)), "RX_p95_seconds": float(np.quantile(elapsed, .95)),
                          "RX_max_seconds": float(elapsed.max()), "sampler_steps_mean": float(steps.mean()),
                          "sampler_steps_min": int(steps.min()), "sampler_steps_max": int(steps.max()),
                          "metadata_fallbacks": sum(bool(row["fallback"]) for row in selected),
                          "GPU_peak_allocated_bytes": max(int(row["GPU_peak_allocated_bytes"]) for row in selected)})
    write_csv(output / "hifi_cost_by_snr.csv", hifi_cost)
    figures(output, summary)
    cost_figure(output, summary, costs)
    report = make_report(summary, contrasts, costs, audit, output)
    (output / "positioning_result.md").write_text(report)
    report_path = PROJECT / "reports/external_baseline_positioning_result.md"
    one_page_path = PROJECT / "reports/research_positioning_one_page_final.md"
    if report_path.exists() or one_page_path.exists():
        raise RuntimeError("do not overwrite an existing scientific conclusion")
    bindings[str(metric_path)] = sha256(metric_path)
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("frozen results changed during closure")
    report_path.write_text(report)
    one_page_path.write_text(one_page(summary, contrasts, costs))
    write_json(output / "pairing_audit.json", audit)
    write_json(output / "completion.json", {"status": "FROZEN_HIFI_SYSTEM_COMPARISON_COMPLETE", "rows": len(combined),
               "matched_HiFi_transmissions": len(hifi), "contrasts": len(contrasts), "bindings": bindings,
               "GPU_used_for_closure": False, "new_training_or_holdout": False, "automatic_new_research_allowed": False,
               "report": str(report_path), "one_page_report": str(one_page_path), "finished_at": datetime.now().astimezone().isoformat()})


def main(arguments):
    queue = arguments.root / "closure_queue_001"
    queue.mkdir(exist_ok=True)
    lock = (queue / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (arguments.output / "completion.json").exists():
        raise RuntimeError("completed closure must not be repeated")
    bindings = json.loads((arguments.root / "author_queue_001/bindings.json").read_text())
    bindings.update(json.loads((arguments.root / "non_diffusion_analysis_001/completion.json").read_text())["bindings"])
    bindings[str(Path(__file__))] = sha256(Path(__file__))
    write_json(queue / "bindings.json", bindings)
    try:
        while not all((arguments.root / name / "completion.json").exists() for name in ("author_queue_001", "development_hifi_001", "hifi_metrics_001")):
            if not arguments.wait:
                raise RuntimeError("full HiFi results are not ready; no partial report produced")
            author = json.loads((arguments.root / "author_queue_001/status.json").read_text())
            if author.get("status", "").startswith("STOPPED"):
                raise RuntimeError("author queue stopped; preserve records and request implementation review")
            if any(sha256(path) != expected for path, expected in bindings.items()):
                raise RuntimeError("registered inference or closure code changed")
            write_json(queue / "status.json", {"status": "WAITING_FOR_EXISTING_HIFI_AND_METRICS_NO_GPU", "pid": os.getpid(),
                       "author_stage": author.get("stage"), "new_training": False, "updated_at": datetime.now().astimezone().isoformat()})
            time.sleep(60)
        write_json(queue / "status.json", {"status": "CPU_CLOSURE_RUNNING", "pid": os.getpid()})
        finish(arguments)
        write_json(queue / "status.json", {"status": "COMPLETE_NO_NEW_TASKS", "at": datetime.now().astimezone().isoformat()})
    except Exception:
        failure = {"status": "STOPPED_FOR_REVIEW_NO_NEW_TASKS", "traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()}
        write_json(queue / f"failure_{time.time_ns()}.json", failure)
        write_json(queue / "status.json", failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    main(parser.parse_args())
