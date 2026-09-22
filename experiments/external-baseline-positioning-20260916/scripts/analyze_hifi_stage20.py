#!/usr/bin/env python3
"""All systems on the same frozen exploratory cohort; not the original1500-frame result."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src"), str(EXPERIMENT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from external_positioning.analysis import METRICS, SCOPES
from prepare_hifi_stage20 import account_frame, summarize_cost
from summarize_available import color, label, read_csv
from var_comm.online_timing import SOURCE_INDICES
from var_comm.study import sha256, write_csv, write_json


def key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def source_matrix(rows, support, indices):
    grouped = defaultdict(list)
    for row in rows:
        if float(row["snr_db"]) in support:
            grouped[int(row["image_index"])].append([float(row[name]) for name in METRICS])
    if any(len(grouped[index]) != len(support) for index in indices):
        raise RuntimeError("exactly one registered noise per image and SNR is required")
    return np.array([np.mean(grouped[index], axis=0) for index in indices])


def quality_tables(rows, indices, sources):
    expected = {(index, snr, 2001) for index in indices for snr in (1., 4., 7., 13., 19.)}
    groups = defaultdict(list)
    for row in rows:
        if key(row) not in expected or row["image_id"] != sources[int(row["image_index"])]["image_id"]:
            raise RuntimeError("another population entered the exploratory study")
        if row["source_pixels_sha256"] != sources[int(row["image_index"])]["source_pixels_sha256"]:
            raise RuntimeError("source preprocessing differs")
        if not np.isfinite([float(row[name]) for name in METRICS]).all():
            raise RuntimeError("failed outputs must not disappear as missing quality")
        if abs(float(row["total_energy"]) - 2 * int(row["complex_uses"])) > .02:
            raise RuntimeError("physical resource mismatch")
        groups[row["method"], row["protocol"], int(row["complex_uses"])].append(row)
    draws = np.random.default_rng(20260916).integers(len(indices), size=(10000, len(indices)))
    summaries, by_source = [], []
    for group, selected in sorted(groups.items()):
        if len(selected) != 100 or {key(row) for row in selected} != expected:
            raise RuntimeError(f"not all systems have the same100 transmissions: {group}")
        for scope, support in SCOPES:
            matrix = source_matrix(selected, support, indices)
            intervals = np.quantile(matrix[draws].mean(axis=1), (.025, .975), axis=0)
            row = {"method": group[0], "protocol": group[1], "complex_uses": group[2], "total_energy": 2 * group[2],
                   "scope": scope, "sources": len(indices), "frames": len(indices) * len(support), "noise_seeds_per_source_SNR": 1}
            for position, metric in enumerate(METRICS):
                row[metric] = float(matrix[:, position].mean())
                row[metric + "_ci_low"] = float(intervals[0, position])
                row[metric + "_ci_high"] = float(intervals[1, position])
            summaries.append(row)
            for position, index in enumerate(indices):
                by_source.append({"method": group[0], "protocol": group[1], "complex_uses": group[2], "scope": scope,
                                  "image_index": index, **dict(zip(METRICS, matrix[position].tolist()))})
    return groups, summaries, by_source, draws


def pair(method, control, indices, draws, name, relation):
    if {key(row) for row in method} != {key(row) for row in control}:
        raise RuntimeError("paired source/noise/SNR keys differ")
    result = []
    for scope, support in SCOPES:
        difference = source_matrix(method, support, indices) - source_matrix(control, support, indices)
        interval = np.quantile(difference[draws].mean(axis=1), (.025, .975), axis=0)
        for position, metric in enumerate(METRICS):
            result.append({"contrast": name, "method": method[0]["method"], "control": control[0]["method"],
                           "scope": scope, "metric": metric, "source_images": len(indices), "bootstrap_resamples": 10000,
                           "method_complex_uses": int(method[0]["complex_uses"]), "control_complex_uses": int(control[0]["complex_uses"]),
                           "resource_relation": relation, "mean_difference": float(difference[:, position].mean()),
                           "ci_low": float(interval[0, position]), "ci_high": float(interval[1, position])})
    return result


def reuse_timing(root, current_rows, indices, support):
    common_indices = sorted(set(indices).intersection(SOURCE_INDICES))
    selected_rows = []
    history = PROJECT / "outputs/COMMUNICATION-CONVERGENCE-20260915"
    policies = json.loads((history / "POLICIES_001/policies.json").read_text())["actions"]
    old = read_csv(PROJECT / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/full_001/per_call.csv")
    entropy = read_csv(history / "ENTROPY_CPU_TIMING_001/per_call.csv")
    r3 = read_csv(history / "R3_CPU_TIMING_001/per_call.csv")
    newer = read_csv(root / "digital_timing_001/per_call.csv")
    for source, kind in ((old, "old"), (entropy, "entropy"), (r3, "r3"), (newer, "new")):
        for row in source:
            index, snr = int(row["image_index"]), float(row["snr_db"])
            if index not in common_indices or snr not in support:
                continue
            method = None
            if kind == "old":
                mode = policies["raw"]["quality"][str(snr)]
                if row["arm"] == f"whole_m{mode}":
                    method = "raw_adaptive"
                elif row["arm"] == "perceptual_deepjscc":
                    method = "perceptual_deepjscc"
            elif kind == "entropy" and int(row["mode"]) == policies["arithmetic"]["quality"][str(snr)]:
                method = "arithmetic_adaptive"
            elif kind == "r3":
                method = "wetok_r3"
            elif kind == "new":
                method = row["method"]
            if method:
                selected_rows.append({"method": method, "image_index": index, "snr_db": snr,
                    "TX_seconds": float(row["TX_seconds"]), "RX_seconds": float(row["RX_seconds"]),
                    "repeat": int(row["repeat"]), "scope": "same_archived_source_intersection_and_seed2001"})
    for row in current_rows:
        if row["protocol"] != "common_paid_information" or not row.get("RX_seconds") or int(row["image_index"]) not in common_indices:
            continue
        selected_rows.append({"method": row["method"], "image_index": int(row["image_index"]), "snr_db": float(row["snr_db"]),
                             "TX_seconds": float(row["TX_seconds"]), "RX_seconds": float(row["RX_seconds"]), "repeat": 0,
                             "scope": "same_archived_source_intersection_and_seed2001"})
    summaries = []
    for method in sorted({row["method"] for row in selected_rows}):
        for scope, snrs in SCOPES:
            rows = [row for row in selected_rows if row["method"] == method and row["snr_db"] in snrs]
            keys = {(row["image_index"], row["snr_db"]) for row in rows}
            if keys != {(index, snr) for index in common_indices for snr in snrs}:
                raise RuntimeError(f"incomplete matched timing cohort for {method}")
            rx = [np.mean([row["RX_seconds"] for row in rows if row["image_index"] == index]) for index in common_indices]
            tx = [np.mean([row["TX_seconds"] for row in rows if row["image_index"] == index]) for index in common_indices]
            summaries.append({"method": method, "scope": scope, "timing_sources": len(common_indices), "timing_calls": len(rows),
                              "TX_mean_ms": float(np.mean(tx) * 1000), "RX_mean_ms": float(np.mean(rx) * 1000),
                              "RX_p95_ms": float(np.quantile([row["RX_seconds"] for row in rows], .95) * 1000),
                              "noise_seed": 2001, "precision": "FP32_TF32_disabled", "batch": 1,
                              "endpoint": "CPU_to_CPU;airtime_queueing_IO_loading_metrics_excluded",
                              "framework_and_session_caveat": "author_Torch1.12_vs_existing_Torch2.11;repeat_counts_differ"})
    return common_indices, selected_rows, summaries


def figures(output, summary, times):
    common = [row for row in summary if row["protocol"] == "common_paid_information"]
    snrs = (1., 4., 7., 13., 19.)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12))
    for position, budget in enumerate((3060, 4204, 4498)):
        for column, metric in enumerate(("psnr_db", "lpips")):
            axis = axes[position, column]
            methods = sorted({row["method"] for row in common if row["complex_uses"] == budget})
            for method in methods:
                selected = [next(row for row in common if row["method"] == method and row["scope"] == f"snr_{snr:g}") for snr in snrs]
                axis.plot(snrs, [row[metric] for row in selected], marker="o", label=label(method))
            axis.set(title=f"Equal N={budget}, E={budget * 2}", xlabel="SNR (dB)", ylabel=metric)
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    figure.suptitle("Exploratory20 original sources x five SNR x seed2001; not the1500-frame study")
    figure.tight_layout(rect=(0, 0, 1, .97))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"equal_resource_quality.{extension}", dpi=160)
    plt.close(figure)
    time_lookup = {(row["method"], row["scope"]): row for row in times}
    figure, axes = plt.subplots(2, 2, figsize=(17, 10))
    for column, scope in enumerate(("primary_1_4_7", "high_13_19")):
        for row in (row for row in common if row["scope"] == scope):
            name = f"{label(row['method'])} N={row['complex_uses']}"
            axes[0, column].scatter(row["complex_uses"], row["lpips"], label=name, color=color(row["method"]))
            cost = time_lookup[row["method"], scope]
            axes[1, column].scatter(cost["RX_mean_ms"], row["lpips"], label=name, color=color(row["method"]))
        axes[0, column].set(xscale="log", xlabel="Complex channel uses", ylabel="LPIPS", title=scope)
        axes[1, column].set(xscale="log", xlabel="Complete RX processing mean ms (matched timing cohort)", ylabel="LPIPS", title=scope)
        for axis in axes[:, column]:
            axis.grid(alpha=.25)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Same20-source quality; timings restricted to common archived sources; unequal N labelled")
    figure.tight_layout(rect=(0, .19, 1, .96))
    for extension in ("png", "pdf"):
        figure.savefig(output / f"quality_resources_RX.{extension}", dpi=160)
    plt.close(figure)


def previews(output, rows, sources, indices):
    directory = output / "preselected_images"
    directory.mkdir()
    lookup = {(row["method"], *key(row)): row for row in rows if row["protocol"] == "common_paid_information"}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    manifests = []
    for index in indices[::5]:
        pixels = np.load(sources[index]["path"], allow_pickle=False)
        source = Image.fromarray(pixels.transpose(1, 2, 0))
        for snr in (1., 7., 19.):
            methods = ("raw_adaptive_N4204", "arithmetic_adaptive_N4204", "hifi_diffcom_c2")
            board = Image.new("RGB", (1024, 325), "white")
            draw = ImageDraw.Draw(board)
            board.paste(source, (0, 65))
            draw.text((5, 4), f"Source{index:04d}\nEqual N4204/E8408\n{snr:g}dB, seed2001", font=font, fill="black")
            for column, method in enumerate(methods, 1):
                row = lookup[method, index, snr, 2001]
                with np.load(row["image_archive"], allow_pickle=False) as archive:
                    image = archive["images"][int(row["image_slot"])].copy()
                picture = Image.fromarray(np.rint(np.clip(image, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
                board.paste(picture, (column * 256, 65))
                draw.text((column * 256 + 5, 4), f"{label(method)}\nPSNR {float(row['psnr_db']):.2f}\nLPIPS {float(row['lpips']):.4f}", font=font, fill="black")
            filename = f"equal_N4204_source{index:04d}_snr{snr:g}.png"
            board.save(directory / filename)
            manifests.append({"file": filename, "image_index": index, "snr_db": snr, "seed": 2001,
                              "choice": "every_fifth_registered_source;not_quality_selected"})
    write_csv(directory / "manifest.csv", manifests)


def run(arguments):
    root, stage, output = arguments.root, arguments.stage, arguments.output
    protocol = json.loads((stage / "protocol.json").read_text())
    metric_receipt = json.loads((stage / "metrics_001/completion.json").read_text())
    if metric_receipt["rows"] != 300 or sha256(stage / "metrics_001/per_frame.csv") != metric_receipt["per_frame_sha256"]:
        raise RuntimeError("complete100-frame/300-view stage metrics are required")
    indices = protocol["source_indices"]
    sources = {int(row["image_index"]): row for row in json.loads((stage / "development_inputs.json").read_text())}
    expected = {(index, snr, 2001) for index in indices for snr in protocol["snrs_db"]}
    rows = []
    paths = [root / "inputs_001/reference_per_frame.csv", root / "fast_author_metrics_001/per_frame.csv",
             root / "digital_development_001/per_frame.csv", stage / "metrics_001/per_frame.csv"]
    bindings = {str(path): sha256(path) for path in paths}
    for path in paths[:-1]:
        for row in read_csv(path):
            if key(row) not in expected:
                continue
            row.setdefault("protocol", "common_paid_information")
            row["total_energy"] = row.get("total_energy", row.get("actual_total_energy"))
            rows.append(row)
    new = read_csv(paths[-1])
    companion = [row for row in new if row["method"] == "adjscc_c2"]
    hifi = [row for row in new if row["method"] == "hifi_diffcom_c2" and row["protocol"] == "common_paid_information"]
    standalone = {key(row): row for row in rows if row["method"] == "adjscc_c2"}
    if len(companion) != 100 or len(hifi) != 100:
        raise RuntimeError("missing same-observation attribution view")
    for row in companion + hifi:
        reference = standalone[key(row)]
        for field in ("data_transmitted_sha256", "data_observed_sha256", "standard_data_noise_sha256", "source_pixels_sha256", "checkpoint_sha256"):
            if row[field] != reference[field]:
                raise RuntimeError(f"same-observation match failed: {field}")
    rows.extend({**row, "total_energy": row["actual_total_energy"]} for row in new if row["method"] == "hifi_diffcom_c2")
    if len(rows) != 1900:
        raise RuntimeError("incomplete subset or a duplicate ADJSCC baseline")
    groups, summary, by_source, draws = quality_tables(rows, indices, sources)
    common = {group[0]: selected for group, selected in groups.items() if group[1] == "common_paid_information"}
    contrasts = pair(hifi, companion, indices, draws, "HiFi_minus_same_session_ADJSCC", "same4096_data_observation;HiFi108_paid_metadata_extra")
    for method, control in (("hifi_diffcom_c2", "raw_adaptive_N4204"), ("hifi_diffcom_c2", "arithmetic_adaptive_N4204"),
                            ("swin_ra32", "raw_adaptive_N4498"), ("swin_ra32", "arithmetic_adaptive_N4498"),
                            ("wetok_r3", "raw_adaptive"), ("wetok_r3", "arithmetic_adaptive"), ("wetok_r3", "perceptual_deepjscc")):
        contrasts.extend(pair(common[method], common[control], indices, draws, method + "_minus_" + control, "same_total_N_and_energy"))
    timing_indices, timing_rows, costs = reuse_timing(root, rows, indices, protocol["snrs_db"])
    frozen_timing = json.loads((stage / "timing_scope.json").read_text())
    if timing_indices != frozen_timing["common_timing_source_indices"]:
        raise RuntimeError("common timing cohort differs from predeclared archived coverage")
    output.mkdir(parents=True, exist_ok=False)
    fields = sorted({field for row in rows for field in row})
    write_csv(output / "per_frame.csv", [{field: row.get(field, "") for field in fields} for row in rows])
    for filename, values in (("summary", summary), ("paired", contrasts), ("per_source", by_source),
                             ("matched_timing_calls", timing_rows), ("matched_timing_summary", costs)):
        write_csv(output / f"{filename}.csv", values)
    work = [account_frame(json.loads(path.read_text())) for path in sorted((stage / "inference_001/frames").glob("*/frame.json"))]
    if len(work) != 100:
        raise RuntimeError("exploratory work accounting is incomplete")
    contention_path = stage / "gpu_contention_note.json"
    contention = json.loads(contention_path.read_text()) if contention_path.exists() else {}
    flagged = set(contention.get("conservative_timing_review_frame_keys", []))
    for row in work:
        row["timing_contention_window_flagged"] = row["frame_key"] in flagged
    event_paths = sorted((stage / "resource_resume_001").glob("contention_*.json"))
    write_json(output / "timing_resource_caveat.json", {"initial_event": contention,
               "later_resource_events": [json.loads(path.read_text()) for path in event_paths],
               "raw_times_preserved": True, "quality_outputs_not_removed_or_reselected": True,
               "known_initial_flagged_sources_not_in_matched10": not any(int(name.split("source")[-1]) in timing_indices for name in flagged),
               "additional_events_require_review_before_exclusive_GPU_latency_claims": bool(event_paths)})
    write_csv(output / "hifi_actual_work.csv", work)
    raw_costs = summarize_cost(work)
    for row in raw_costs:
        row["timing_contention_window_flagged_frames"] = sum(item["timing_contention_window_flagged"] for item in work if item["snr_db"] == row["snr_db"])
        row["raw_observed_time_not_unqualified_exclusive_GPU_time"] = bool(row["timing_contention_window_flagged_frames"] or event_paths)
    write_csv(output / "hifi_full_cohort_cost_by_snr.csv", raw_costs)
    figures(output, summary, costs)
    previews(output, rows, sources, indices)
    lines = ["# HiFi20源图探索性定位", "", f"完成时间：{datetime.now().astimezone().isoformat()}。",
             "**固定20原development源×五SNR×原noise2001，每方法100次；不是原1500帧完成，也不是新holdout。** 名单与质量无关；所有失败均计入。", "",
             "原checkpoint、FP32、完整作者DDPM/adaptive start、sampling seed23、元数据/失败规则不变。无checkpoint分支只用于校准性能诊断，没有用于本阶段推理。", "",
             "## 分开看机制与同预算系统", "", "HiFi与同会话ADJSCC共享4096个数据符号和观测；HiFi额外计108次元数据。独立ADJSCC保留N4096，未被加header削弱。", "",
             "| 比较：HiFi−对照 | 区间 | ΔPSNR及95%CI | ΔLPIPS及95%CI |", "|---|---|---|---|"]
    for name in ("HiFi_minus_same_session_ADJSCC", "hifi_diffcom_c2_minus_raw_adaptive_N4204", "hifi_diffcom_c2_minus_arithmetic_adaptive_N4204"):
        for scope in ("primary_1_4_7", "high_13_19"):
            selected = {row["metric"]: row for row in contrasts if row["contrast"] == name and row["scope"] == scope}
            text = [f"{selected[metric]['mean_difference']:+.5f} [{selected[metric]['ci_low']:+.5f},{selected[metric]['ci_high']:+.5f}]" for metric in ("psnr_db", "lpips")]
            lines.append(f"| {name} | {scope} | {text[0]} | {text[1]} |")
    lines.extend(["", "## 全系统同子集质量", "", "下面N不同，不能跨行当等预算排名。逐SNR及SSIM/区间在CSV；三种记录视图不是三次扩散。", "",
                  "| 方法 | 区间 | N | PSNR | LPIPS | DINO |", "|---|---|---:|---:|---:|---:|"])
    for scope in ("primary_1_4_7", "high_13_19"):
        for row in summary:
            if row["scope"] == scope and row["protocol"] == "common_paid_information":
                lines.append(f"| {row['method']} | {scope} | {row['complex_uses']} | {row['psnr_db']:.3f} | {row['lpips']:.5f} | {row['dino']:.5f} |")
    lines.extend(["", "## 计算、范围和研究边界", "",
                  f"HiFi完整100帧的NFE/真实采样次数/完整RX分布单独报告，不用单帧代表全体。与旧系统的主时间图统一用已有计时共同覆盖的{len(timing_indices)}个源{timing_indices}和noise2001；所有系统相同时间子集，质量仍为全部20源。不同框架/会话及原计时重复数披露，不推断微小纯架构速度优势。", "",
                  "计时为CPU观测到CPU RGB，排除加载、指标、文件I/O、空口与排队。原1500计划和已完成帧均保留；本阶段没有重跑旧强系统或改变它们校准选定的checkpoint/模式。", "",
                  "资源说明：若出现外部GPU任务，队列让卡并原参数续跑。已知中断窗口的19dB源47/52原始耗时保留且单独标注，不能无条件当成独占GPU时延；它们不在共同10源时间图中。后续资源事件也记录在timing_resource_caveat.json，若涉及时间子集则须先核查，不据污染时间宣判速度优劣。质量结果不因此丢弃或重选，详见hifi_stage20_gpu_contention_20260917.md。", "",
                  "20源/一个噪声的区间较不稳定，不替代原100源/三噪声或独立验证。保留数字VAR、R3、Swin、感知Deep及实际算术链各自工作点；不要求学习方法必须胜数字VAR。m9饱和不是数字整体上限。真类别TX已知、理想SNR、不同训练域/先验容量和DINO既往暴露继续披露。", "",
                  "这一阶段只据同观测增量、同N4204质量及处理成本定位；没有新颖性证据时不称可发表创新，也不自动开启混合、CSI或新骨干。原文差异和最小证据缺口仍见一页定位/相关工作差异表。", "",
                  f"产物：`{output.relative_to(PROJECT)}`。完整表与源图级paired bootstrap10000次（20个源为重采样单位）均保留。"])
    report = "\n".join(lines) + "\n"
    (output / "stage20_positioning.md").write_text(report)
    destination = PROJECT / "reports/hifi_stage20_positioning_20260916.md"
    if destination.exists():
        raise RuntimeError("do not overwrite a completed scientific report")
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("frozen input changed")
    destination.write_text(report)
    write_json(output / "completion.json", {"status": "EXPLORATORY_STAGE20_POSITIONING_COMPLETE", "source_images": 20,
               "transmissions_per_method": 100, "system_view_rows": len(rows), "paired_metric_rows": len(contrasts),
               "timing_source_intersection": timing_indices, "original1500_complete": False, "all_failures_retained": True,
               "sampler_implementation": "original_forced_attention_checkpoint", "new_training_or_holdout": False,
               "bindings": bindings, "report": str(destination), "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
