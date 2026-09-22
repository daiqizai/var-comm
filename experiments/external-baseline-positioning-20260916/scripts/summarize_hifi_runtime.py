#!/usr/bin/env python3
"""Report completed runtime evidence separately from still-running stage20 quality."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(EXPERIMENT / "scripts")]

import numpy as np

from prepare_hifi_stage20 import account_frame, summarize_cost
from summarize_available import read_csv
from var_comm.study import sha256, write_csv, write_json


def run(arguments):
    root, stage = arguments.root, arguments.stage
    profile, repeated = stage / "attention_profile_001", stage / "repeatability_001"
    if not (profile / "completion.json").exists() or not (repeated / "completion.json").exists():
        raise RuntimeError("paired full profiling and bounded repeatability must finish before reporting")
    output = stage / "runtime_analysis_001"
    output.mkdir(exist_ok=False)
    metadata = json.loads((profile / "metadata.json").read_text())
    source_frames = sorted((root / "development_hifi_001/frames").glob("*/frame.json"))
    work = [account_frame(json.loads(path.read_text())) for path in source_frames]
    costs = summarize_cost(work)
    write_csv(output / "original_committed_work.csv", work)
    write_csv(output / "original_cost_by_snr.csv", costs)
    speed = read_csv(profile / "uninstrumented_speed_comparison.csv")
    gradients = read_csv(profile / "matched_input_gradients.csv")
    outputs = read_csv(profile / "full_output_comparisons.csv")
    repeated_rows = read_csv(repeated / "original_repeat_comparisons.csv")
    details = []
    for case in range(3):
        for variant in ("original", "attention_direct"):
            record = json.loads((profile / f"case_{case:02d}/{variant}_instrumentation.json").read_text())
            measured = record["GPU_seconds_inclusive"]
            counts = record["calls"]
            details.append({"case": case, "variant": variant, "top_level_UNet_calls": counts["unet_forward"],
                            "input_gradient_calls": counts["input_gradient_inclusive"],
                            "attention_forward_calls": counts["attention_initial_forward"],
                            "attention_backward_recompute_calls": counts.get("attention_backward_recompute", 0),
                            "checkpoint_inner_grad_calls": counts.get("checkpoint_inner_autograd_calls", 0),
                            "checkpoint_parameter_tensor_gradient_requests": counts.get("checkpoint_requested_parameter_tensors", 0),
                            "GPU_unet_forward_seconds": measured["unet_forward"],
                            "GPU_consistency_forward_seconds": measured["consistency_loss_forward"],
                            "GPU_input_gradient_seconds_inclusive": measured["input_gradient_inclusive"],
                            "GPU_attention_recompute_seconds_nested_in_gradient": measured.get("attention_backward_recompute", 0.),
                            "GPU_posterior_steps_seconds_inclusive": measured["posterior_step_inclusive"],
                            "nested_terms_not_additive": True})
    write_csv(output / "profile_breakdown.csv", details)
    lines = ["# HiFi实际耗时与独立无检查点分支结果", "", f"汇总时间：{datetime.now().astimezone().isoformat()}。",
             "**结论：三种记录视图没有造成三次扩散；仅关闭attention checkpoint未见足够的完整RX提速，不替换原实现。** 权重、精度、完整步数、seed、观测不变，无训练/新架构/共享环境修改。", "",
             "## 1. 原队列已提交工作：按SNR计，而非按视图计", "",
             "| SNR | 完成帧 | 记录视图 | 真实采样调用 | 元数据额外恢复 | NFE/反向步数 | 完整付费RX均值/中位数/p95秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in costs:
        if not row["completed_transmissions"]:
            lines.append(f"| {row['snr_db']:g} | 0 | 0 | 0 | 0 | 未覆盖 | 未覆盖 |")
        else:
            lines.append(f"| {row['snr_db']:g} | {row['completed_transmissions']} | {row['record_views']} | {row['actual_sampler_calls']} | {row['metadata_extra_author_sampler_calls']} | {row['common_NFE_min']}–{row['common_NFE_max']} | {row['RX_mean_seconds']:.3f}/{row['RX_median_seconds']:.3f}/{row['RX_p95_seconds']:.3f} |")
    lines.extend(["", f"原1500计划目前保留{len(work)}个已提交帧。三视图为实际付费HiFi、同观测ADJSCC、作者信息假设视图；后者在接收参数相同时直接复用图像，ADJSCC不是扩散。当前元数据额外恢复总数{sum(row['metadata_extra_author_sampler_calls'] for row in work)}。计数来自完成回执和已冻结分支逻辑，不把视图数乘三。",
                  "未提交的中断/在途工作、工程热身和校准profile不冒充已完成传输；额外作者恢复若未独立计时则记缺失，不从平均RX猜时间。原NFE记录的是采样迭代，attention反向重计算不是额外整次采样。", "",
                  "## 2. 强制checkpoint的实际调用", "",
                  f"作者U-Net有{metadata['attention_blocks']}个AttentionBlock，构造属性use_checkpoint=False但forward仍硬编码True；ResBlock开启数为{metadata['ResBlocks_checkpointed']}。仅改配置属性不能关掉它。独立分支只将该实例的forward指向原_forward，没有编辑vendor或覆盖原队列。", "",
                  "原自定义CheckpointFunction.backward会重算attention，并对输入及参数调用autograd.grad。校准1dB的251步实际调用顶层U-Net251次、根输入梯度250次、attention首算4016次、反向重算4000次。这里4000次是内部模块调用，绝不是4000次完整扩散。", "",
                  "## 3. 配对的完整RX与显存（无仪器结果）", "",
                  "三个预登记校准工作点，每分支一次无仪器完整推理；仪器轨迹不混入速度比。范围仅这三个点，没有据此声称所有SNR/图像加速。", "",
                  "| 校准源/SNR | 原版秒 | 直接attention秒 | 原/直接速度比 | 原/直接allocated峰值GiB |", "|---|---:|---:|---:|---:|"])
    for row in speed:
        lines.append(f"| {row['image_index']}/{float(row['snr_db']):g}dB | {float(row['original_RX_seconds']):.3f} | {float(row['direct_RX_seconds']):.3f} | {float(row['speed_ratio_original_over_direct']):.4f} | {int(row['original_peak_allocated_bytes'])/2**30:.3f}/{int(row['direct_peak_allocated_bytes'])/2**30:.3f} |")
    baseline = next(row for row in details if row["case"] == 0 and row["variant"] == "original")
    lines.extend(["", f"以原版1dB仪器轨迹举例：GPU U-Net正向{baseline['GPU_unet_forward_seconds']:.3f}秒，一致性loss正向{baseline['GPU_consistency_forward_seconds']:.3f}秒，根输入反传{baseline['GPU_input_gradient_seconds_inclusive']:.3f}秒；attention重算仅约{baseline['GPU_attention_recompute_seconds_nested_in_gradient']:.3f}秒，已包含在反传内，不再次相加。主要代价是完整网络反向/正向，并非三视图或单独的attention重算。",
                  "CPU算子trace仅帮助辨认调用，不能把其inclusive时间当GPU时间或叠加到CUDA事件。当前没有继续改变参数梯度开关/精度/采样，也没有做自动优化搜索。", "",
                  "## 4. 梯度、完整输出与原版自身重复性", "",
                  f"9个匹配原版状态/RNG/观测的输入梯度均非零且有限；其中{sum(row['input_gradient_passed']=='True' for row in gradients)}/9通过全部预登记阈值，最大相对L2误差{max(float(row['relative_L2_error']) for row in gradients):.6g}、最大绝对差{max(float(row['max_abs_gradient_error']) for row in gradients):.6g}。不能写成梯度全部精确一致。"])
    cross = [row for row in outputs if row["comparison"] == "attention_direct_vs_original_full_output"]
    lines.append("三点完整RGB最大差分别为" + ", ".join(f"{float(row['max_abs_RGB_error']):.6g}" for row in cross) + "，均未通过原2e−4严格阈值；没有事后放宽阈值。")
    lines.append("同一固定校准观测额外两次原版完整重复的最大差为" + f"{float(repeated_rows[0]['max_abs_RGB_error']):.6g}" + "；与此前原版输出也存在同量级差异。原版本身存在数值重复性边界，因此既不能把A/B差异全归优化，也不能宣称优化完全等价。没有将这种工程数值差异当成模型质量负结果。")
    lines.extend(["", "**决定：不部署无checkpoint分支。** 输入梯度仍保留，全路径没有no_grad/inference_mode；权重/状态未更新。既无明显加速，又未建立严格输出资格，不值得覆盖原队列。", "",
                  "## 5. 缩小第一阶段，不伪称1500完成", "",
                  "固定20源的名单由原100序号等距选取，与质量无关，五SNR和原noise2001不变，每系统100次。旧完整计划/全部完成帧保留；已完成合法结果按原归档路径复用，不复制或链接大图像归档。缺帧仍用原checkpoint、完整采样和原元数据/失败规则，未使用性能分支。",
                  "质量对照全部用同20源/5SNR/一个噪声。时间图为避免补跑旧benchmark，所有系统都限制到该20源内原计时共同覆盖的10源：0/10/16/26/42/57/73/83/89/99；同时另报HiFi整个100帧的时延和NFE分布。单次观察和小样本区间仅探索性，不替代原100源/三噪声或新holdout。",
                  f"原始性能与分层产物：`{stage.relative_to(PROJECT)}`。100帧阶段完成后另出`reports/hifi_stage20_positioning_20260916.md`，不是原1500的完成报告。"])
    destination = PROJECT / "reports/hifi_runtime_profile_result_20260916.md"
    if destination.exists():
        raise RuntimeError("do not overwrite a completed profiling result")
    destination.write_text("\n".join(lines) + "\n")
    write_json(output / "completion.json", {"status": "ACTUAL_WORK_AND_ISOLATED_PROFILE_REPORTED", "original_committed_frames": len(work),
               "original_plan_frames": 1500, "original_plan_complete": False, "direct_variant_deployed": False,
               "profile_receipt_sha256": sha256(profile / "completion.json"), "repeatability_receipt_sha256": sha256(repeated / "completion.json"),
               "report": str(destination), "stage20_quality_complete_at_this_report": (stage / "analysis_001/completion.json").exists(),
               "new_training": False, "finished_at": datetime.now().astimezone().isoformat()})
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    run(parser.parse_args())
