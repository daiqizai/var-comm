"""Source-paired CPU analysis for the frozen P4084 receiver study.

No model inference, parameter choice, or training occurs in this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260930
RESAMPLES = 10000
STATISTICAL_ZERO_TOLERANCE = 1e-12
MAIN_SNRS = (1, 4, 7)
NOISE_SEEDS = {2001, 2002, 2003}
POLICIES = ("B2", "A1_policy", "A2_policy", "V_policy")
METRICS = (
    "psnr_db", "lpips_alex", "dino_cosine", "dino_mismatched",
    "latent_sq_err_base", "latent_sq_err_post", "latent_sq_err_final",
    "dino_lt_0_6", "lpips_gt_0_35", "dino_match_specificity",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty analysis table: " + str(path))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_means(rows, expected_sources=100):
    grouped = defaultdict(list)
    identity = {}
    for row in rows:
        sid = row["source_id"]
        index = int(row["source_index"])
        if sid in identity and identity[sid] != index:
            raise ValueError("Source identity/index changed")
        identity[sid] = index
        key = (float(row["snr_db"]), row["method"], sid)
        vals = {k: float(row[k]) for k in METRICS[:7]}
        if not np.isfinite(list(vals.values())).all():
            raise ValueError("Nonfinite measured metric")
        vals["dino_lt_0_6"] = float(vals["dino_cosine"] < 0.6)
        vals["lpips_gt_0_35"] = float(vals["lpips_alex"] > 0.35)
        vals["dino_match_specificity"] = vals["dino_cosine"] - vals["dino_mismatched"]
        grouped[key].append((int(row["noise_seed"]), vals, row))
    source_order = sorted(identity, key=lambda sid: identity[sid])
    if len(source_order) != expected_sources or len(set(identity.values())) != expected_sources:
        raise ValueError(f"Expected {expected_sources} distinct source images")
    matrix = {}
    mean_rows = []
    snrs = sorted({k[0] for k in grouped})
    methods = sorted({k[1] for k in grouped})
    for snr in snrs:
        for method in methods:
            means = []
            for sid in source_order:
                entries = grouped.get((snr, method, sid), [])
                if len(entries) != 3 or {s for s, _, _ in entries} != NOISE_SEEDS:
                    raise ValueError(f"Incomplete or duplicate paired noise grid: {(snr, method, sid)}")
                vals = np.asarray([[v[k] for k in METRICS] for _, v, _ in entries], dtype=np.float64).mean(0)
                means.append(vals)
                actions = {r.get("policy_action", "") for _, _, r in entries}
                lambdas = {r.get("lambda", r.get("lambda_", "")) for _, _, r in entries}
                if len(actions) != 1 or len(lambdas) != 1:
                    raise ValueError("Source-independent frozen SNR policy changed across noise")
                mean_rows.append(dict(source_id=sid, source_index=identity[sid], snr_db=snr,
                    method=method, noise_repeats=3, policy_action=next(iter(actions)),
                    lambda_=next(iter(lambdas)), **dict(zip(METRICS, map(float, vals)))))
            matrix[(snr, method)] = np.asarray(means)
    if any(m not in methods for m in POLICIES):
        raise ValueError("A required primary policy is missing")
    return matrix, mean_rows, source_order, snrs, methods


def interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values[indices].mean(1), [2.5, 97.5])
    return dict(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi))


def analyze(matrix, snrs, methods, indices):
    summary = []
    for snr in snrs:
        for method in methods:
            values = matrix[(snr, method)]
            row = dict(snr_db=snr, method=method, sources=len(values), noise_repeats=3,
                       in_training_range=snr in (1, 4, 7, 13), inference_unit="source_image")
            for i, metric in enumerate(METRICS):
                result = interval(values[:, i], indices)
                row[metric] = result["mean"]
                row[metric + "_ci_low"] = result["ci_low"]
                row[metric + "_ci_high"] = result["ci_high"]
            summary.append(row)
    comparisons = [("V_policy", c, "primary") for c in ("B2", "A1_policy", "A2_policy")]
    comparisons += [(m, "B2", "control_vs_base") for m in ("A1_policy", "A2_policy")]
    for kind in ("common", "tok"):
        comparisons += [("V_" + kind, p + "_" + kind, "content_diagnostic" if kind == "common" else "token_diagnostic")
                        for p in ("A1", "A2") if "V_" + kind in methods and p + "_" + kind in methods]
    for method in methods:
        if method not in POLICIES:
            comparisons.append((method, "B2", "diagnostic_vs_base"))
    paired = []
    lookup = {}
    for snr in snrs:
        for method, control, role in comparisons:
            diff = matrix[(snr, method)] - matrix[(snr, control)]
            for j, metric in enumerate(METRICS):
                raw_diff = diff[:, j]
                stable_diff = np.where(np.abs(raw_diff) <= STATISTICAL_ZERO_TOLERANCE, 0., raw_diff)
                result = interval(stable_diff, indices)
                row = dict(snr_db=snr, method=method, control=control, comparison_role=role,
                           metric="delta_specific" if metric == "dino_match_specificity" else metric,
                           delta=result["mean"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                           raw_delta_mean=float(raw_diff.mean()), statistical_zero_tolerance=STATISTICAL_ZERO_TOLERANCE,
                           sources=len(diff), noise_repeats=3, bootstrap_resamples=RESAMPLES,
                           bootstrap_seed=SEED, multiple_comparison_adjustment="none")
                paired.append(row)
                lookup[(snr, method, control, row["metric"])] = row
    return summary, paired, lookup


def level_for(policy, snr):
    levels = policy.get("levels", policy)
    return levels.get(str(int(snr)), levels.get(str(float(snr)), {}))


def decision_table(snrs, lookup, policy):
    records = []
    for snr in snrs:
        level = level_for(policy, snr)
        action = level.get("methods", {}).get("V", {}).get("policy_action", "UNAVAILABLE")
        psnr = lookup[(snr, "V_policy", "B2", "psnr_db")]
        psnr_ok = psnr["delta"] >= -0.3
        per_control = {}
        for control in ("B2", "A1_policy", "A2_policy"):
            lp = lookup[(snr, "V_policy", control, "lpips_alex")]
            di = lookup[(snr, "V_policy", control, "dino_cosine")]
            specific = lookup[(snr, "V_policy", control, "delta_specific")]
            lp_supported = lp["delta"] < 0 and lp["ci_high"] < 0
            dino_supported = di["delta"] > 0 and di["ci_low"] > 0 and specific["ci_low"] > 0
            branches = []
            if lp_supported and lp["delta"] <= -0.02:
                branches.append("LPIPS")
            if dino_supported and di["delta"] >= 0.02:
                branches.append("DINO_WITH_SPECIFICITY")
            per_control[control] = dict(lpips_supported=lp_supported,
                dino_supported_with_specificity=dino_supported, clear_gain_branches=branches,
                quality_supported=lp_supported or dino_supported,
                delta_lpips=lp, delta_dino=di, delta_specific=specific)
        supported = sum(x["quality_supported"] for x in per_control.values())
        if action.upper() == "BYPASS":
            label = "BYPASS_SELECTED"
        elif psnr_ok and all(x["clear_gain_branches"] for x in per_control.values()):
            label = "G1_CLEAR_GAIN"
        elif psnr_ok and supported == 3:
            label = "SMALL_SUPPORTED_GAIN"
        elif psnr_ok and supported:
            label = "PARTIAL_GAIN"
        else:
            label = "NO_GAIN_ESTABLISHED"
        record = dict(snr_db=snr, in_training_range=snr in (1, 4, 7, 13),
            role="high_SNR_protection" if snr == 13 else "within_training_range" if snr in MAIN_SNRS else "outside_training_range_diagnostic",
            policy_action=action, label=label, psnr_vs_B2_constraint_pass=psnr_ok,
            psnr_vs_B2=psnr, controls=per_control)
        if snr == 13:
            lp = lookup[(snr, "V_policy", "B2", "lpips_alex")]
            record["high_snr_protection"] = dict(psnr_observed_constraint_pass=psnr["delta"] >= -0.2,
                lpips_significantly_worse=lp["ci_low"] > 0,
                observational_check_pass=psnr["delta"] >= -0.2 and lp["ci_low"] <= 0,
                interpretation="bypass protection only" if action.upper() == "BYPASS" else "observational check; statistical equivalence was not tested")
        records.append(record)
    return records


def number(value, digits=5):
    return f"{value:.{digits}f}"


def delta_text(row, digits=5):
    return f"{number(row['delta'], digits)} [{number(row['ci_low'], digits)}, {number(row['ci_high'], digits)}]"


def timing_summary(results_dir):
    path = results_dir / "timing.csv"
    if not path.exists():
        return []
    groups = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("phase", "").lower() == "warmup" or row.get("warmup", "").lower() in ("true", "1"):
            continue
        method = row.get("method", row.get("receiver", ""))
        if not method:
            raise ValueError("Timing method identity missing")
        groups[(float(row["snr_db"]), method)].append(row)
    result = []
    for (snr, method), rows in sorted(groups.items()):
        summary = dict(snr_db=snr, method=method, calls=len(rows), sources=len({r["source_id"] for r in rows}),
                       endpoint="CPU observed waveform -> CPU image; P receive network included")
        flags={r.get("ran_VAR","") for r in rows}
        lambdas={r.get("raw_selected_lambda",r.get("lambda",r.get("lambda_",""))) for r in rows}
        if len(flags)!=1 or len(lambdas)!=1:
            raise ValueError("Timing lambda or VAR execution differs within a frozen SNR method")
        summary["ran_VAR"]=next(iter(flags))
        summary["raw_selected_lambda"]=next(iter(lambdas))
        for key in ("rx_ms", "P_receive_ms", "decode_ms", "correction_ms", "var_ms", "additional_ms", "post_ms", "extra_ms", "correction_only_ms", "var_correction_ms"):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            if vals:
                if not np.isfinite(vals).all() or min(vals) < 0:
                    raise ValueError("Invalid timing measurement")
                summary[key + "_mean"] = float(np.mean(vals))
                summary[key + "_median"] = float(np.median(vals))
                summary[key + "_p95"] = float(np.percentile(vals, 95))
        result.append(summary)
    if result:
        write_csv(results_dir / "timing_summary.csv", result)
    return result


def token_summary(results_dir, source_order, indices):
    path=results_dir/"token_accuracy.csv"
    if not path.exists():
        return []
    grouped=defaultdict(list)
    scopes=set()
    metrics=("acc","path_accuracy","entropy","logp_true")
    with path.open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            scope=(float(row['snr_db']),row['method'],row['mode'])
            scopes.add(scope)
            grouped[(*scope,row['source_id'],int(row['scale']))].append(row)
    result=[]
    scale_means=[]
    for snr,method,mode in sorted(scopes):
        source_values={k:[] for k in metrics}
        weighted_values={k:[] for k in metrics}
        by_scale={k:[] for k in metrics}
        for sid in source_order:
            means={k:[] for k in metrics}
            counts=[]
            for scale in range(1,11):
                rows=grouped.get((snr,method,mode,sid,scale),[])
                if len(rows)!=3 or {int(r['noise_seed']) for r in rows}!=NOISE_SEEDS:
                    raise ValueError('Incomplete token metric source/noise/scale grid')
                c={int(r['token_count']) for r in rows}
                if c!={ (1,2,3,4,5,6,8,10,13,16)[scale-1]**2 }:
                    raise ValueError('Token scale count differs from official ten-scale schedule')
                counts.append(next(iter(c)))
                for metric in metrics:
                    vv=[float(r[metric]) for r in rows if r.get(metric) not in (None,'')]
                    if vv and (len(vv)!=3 or not np.isfinite(vv).all()):
                        raise ValueError('Nonfinite or missing token diagnostic')
                    means[metric].append(float(np.mean(vv)) if vv else None)
            for metric in metrics:
                values=means[metric]
                if all(v is None for v in values):
                    continue
                if any(v is None for v in values):
                    raise ValueError('Token diagnostic missing on some scales')
                source_values[metric].append(float(np.mean(values)))
                weighted_values[metric].append(float(np.average(values,weights=counts)))
                by_scale[metric].append(values)
        for weighting,values in (('scale_equal',source_values),('token_count_weighted',weighted_values)):
            for metric,vector in values.items():
                if not vector:
                    continue
                if len(vector)!=len(source_order):
                    raise ValueError('Token diagnostic source coverage differs')
                ci=interval(vector,indices)
                result.append(dict(snr_db=snr,method=method,mode=mode,weighting=weighting,metric=metric,
                    mean=ci['mean'],ci_low=ci['ci_low'],ci_high=ci['ci_high'],sources=len(vector),
                    noise_repeats=3,scales=10,total_tokens=680,bootstrap_seed=SEED,
                    bootstrap_resamples=RESAMPLES,target='receiver_path_conditioned' if metric=='path_accuracy' else 'original_encoding_tokens' if metric=='acc' else 'score_diagnostic'))
        for metric,values in by_scale.items():
            if values:
                means=np.mean(values,axis=0)
                for scale,value in enumerate(means,1):
                    scale_means.append(dict(snr_db=snr,method=method,mode=mode,scale=scale,
                        metric=metric,mean=float(value),sources=len(values),noise_repeats=3,
                        token_count=(1,2,3,4,5,6,8,10,13,16)[scale-1]**2))
    write_csv(results_dir/'token_summary.csv',result)
    write_csv(results_dir/'token_scale_summary.csv',scale_means)
    return result


def make_figures(results_dir, summary, paired, policy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.bbox": "tight"})
    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    snrs = sorted({r["snr_db"] for r in summary})
    rows = {(r["snr_db"], r["method"]): r for r in summary}
    methods = sorted({r["method"] for r in summary})
    colors = {"B2": "#555555", "A1_policy": "#d8901d", "A2_policy": "#49832c", "V_policy": "#346ac9"}
    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(figures / (name + "." + suffix))
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    selected_metrics = ("psnr_db", "lpips_alex", "dino_cosine", "latent_sq_err_final", "dino_lt_0_6", "lpips_gt_0_35")
    for ax, metric in zip(axes.flat, selected_metrics):
        for method in POLICIES:
            ax.plot(snrs, [rows[(s, method)][metric] for s in snrs], "o-", label=method, color=colors[method])
        ax.axvspan(0.5, 7.5, color="#eeeeee", alpha=.55, zorder=0)
        ax.set(xlabel="Nominal SNR (dB)", ylabel=metric)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "quality_snr")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, ("psnr_db", "lpips_alex", "dino_cosine")):
        for offset, control in zip((-.1, 0, .1), ("B2", "A1_policy", "A2_policy")):
            rr = sorted((r for r in paired if r["method"] == "V_policy" and r["control"] == control and r["metric"] == metric), key=lambda r:r["snr_db"])
            yy = np.array([r["delta"] for r in rr])
            ax.errorbar(np.array([r["snr_db"] for r in rr])+offset, yy,
                yerr=[yy-np.array([r["ci_low"] for r in rr]),np.array([r["ci_high"] for r in rr])-yy],
                marker="o", linestyle="-", capsize=2, label="V - " + control)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(xlabel="Nominal SNR (dB)", ylabel="Paired " + metric + " difference")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, "paired_policy_deltas")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for i, kind in enumerate(("common", "tok")):
        for ax, metric in zip(axes[i], ("psnr_db", "lpips_alex", "dino_cosine")):
            for prefix in ("A1", "A2", "V"):
                method = prefix + "_" + kind
                if method in methods:
                    ax.plot(snrs, [rows[(s,method)][metric] for s in snrs], "o-", label=method)
            ax.set(xlabel="Nominal SNR (dB)", ylabel=metric, title="Common A1 weight" if kind == "common" else "Before fusion")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    save(fig, "common_and_token_diagnostics")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for prior, offset in zip(("A1", "A2", "V"), (-.1,0,.1)):
        levels = [level_for(policy,s) for s in snrs]
        mm = [l.get("methods",{}).get(prior,{}) for l in levels]
        alpha = [np.asarray(m.get("alpha",[]),dtype=float) for m in mm]
        if all(a.size == 32 for a in alpha):
            yy = np.asarray([a.mean() for a in alpha])
            low=np.asarray([a.min() for a in alpha]);high=np.asarray([a.max() for a in alpha])
            axes[0].errorbar(np.asarray(snrs)+offset,yy,yerr=[yy-low,high-yy],fmt="o-",capsize=2,label=prior)
        lambdas=[m.get("raw_selected_lambda",np.nan) for m in mm]
        lambdas=[np.nan if x is None else float(x) for x in lambdas]
        axes[1].plot(np.asarray(snrs)+offset,lambdas,"o-",label=prior)
        axes[2].plot(np.asarray(snrs)+offset,[float(m.get("policy_action","").upper()=="BYPASS") for m in mm],"o-",label=prior)
    axes[0].set(ylabel="Alpha mean; whiskers = channel range",ylim=(-.04,1.04))
    axes[1].set(ylabel="Raw selected lambda",yticks=[0,.25,.5,.75,1,1.5])
    axes[2].set(ylabel="SNR-level BYPASS action",yticks=[0,1],ylim=(-.1,1.1))
    for ax in axes:
        ax.set_xlabel("Nominal SNR (dB)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,"alpha_lambda_policy")
    return [str(p.relative_to(results_dir)) for p in sorted(figures.glob("*"))]


def make_report(results_dir, summary, lookup, decisions, policy, timing, config, figure_files):
    rows = {(r["snr_db"],r["method"]):r for r in summary}
    supported = [r for r in decisions if r["snr_db"] in MAIN_SNRS and r["label"] in ("G1_CLEAR_GAIN","SMALL_SUPPORTED_GAIN")]
    against_base=[s for s in MAIN_SNRS if next(r for r in decisions if r["snr_db"]==s)["controls"]["B2"]["quality_supported"]]
    cost={}
    for row in timing:
        cost[(row["snr_db"],row["method"])]=row.get("rx_ms_mean")
    lines=["# 冻结 P4084 接收端增量验证：A 部分", "", "本轮使用 C priority 第一训练 seed 2026092304 的 P4084 selected27500，完成冻结模型评测；新增训练更新为 0。", "",
        "1. **V 是否胜过原 P4084？** " + ("相对 B2 有统计支持的质量收益点："+"、".join(f"{s} dB" for s in against_base)+"。" if against_base else "1/4/7 dB 范围内未建立相对 B2 的质量收益。") + "逐点指标、PSNR 约束和接收时间列在下表。",
        "2. **V 是否胜过 A1 和 A2？** " + ("三个直接控制均支持质量增量的点："+"、".join(f"{r['snr_db']:g} dB ({r['label']})" for r in supported)+"。" if supported else "1/4/7 dB 范围内未建立超出三个直接控制的质量增量。") + "只胜 B2 时不归因于 VAR 条件先验。",
        "3. **共同融合权重下是否仍有增量？** 下面逐点报告 V_common 相对 A1_common/A2_common 的配对差；共同权重固定为 calibration 的 A1 alpha，未重新选 lambda。旁路时这些是诊断输出。",
        "4. **下一步支持验证什么？** " + ("优先在已有第二训练 seed 的 P4084 上按同规则重复 A，重新校准该 base，以检查当前增量能否复现。" if supported else "若仅范围外出现增量，可在用户独立确认后验证低 SNR 基线适配；否则当前数据支持结束这套直接后处理实现，并保留码本/融合和条件先验的具体归因。") + "B 部分、P_low 和旧队列均不自动启动。", "",
        "## 范围内主结果", "", "以下均先对每张源图的三个噪声平均，再对 100 张源图平均。PSNR 是逐图 PSNR 的均值。", "",
        "| SNR | 方法 | PSNR dB | LPIPS alex | DINO | 原坐标 latent 平方和 | DINO<0.6 | LPIPS>0.35 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for snr in MAIN_SNRS:
        for method in POLICIES:
            r=rows[(snr,method)]
            lines.append(f"| {snr} | {method} | {r['psnr_db']:.4f} | {r['lpips_alex']:.5f} | {r['dino_cosine']:.5f} | {r['latent_sq_err_final']:.3f} | {r['dino_lt_0_6']:.4f} | {r['lpips_gt_0_35']:.4f} |")
    lines += ["", "上述两个阈值比例仅是所设指标阈值的失效比例。latent 误差是 32×16×16 原坐标平方和；均方值可除以 8192，不能把两者混用。", "",
        "| SNR | 控制 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δspecific [95% CI] |", "|---:|---|---|---|---|---|"]
    for snr in MAIN_SNRS:
        for control in ("B2","A1_policy","A2_policy"):
            values=[delta_text(lookup[(snr,"V_policy",control,k)]) for k in ("psnr_db","lpips_alex","dino_cosine","delta_specific")]
            lines.append(f"| {snr} | {control} | " + " | ".join(values) + " |")
    lines += ["", "所有差值方向为 V−控制；LPIPS 和 latent 误差越低越好，PSNR/DINO 越高越好。DINO 增量的判定同时要求对应 Δspecific 区间下界>0；LPIPS 判定独立进行。", "",
        "| SNR | 标签 | 相对 B2 PSNR≥−0.3 | 各控制明显收益分支 |", "|---:|---|---|---|"]
    for record in decisions:
        if record["snr_db"] in MAIN_SNRS:
            branches="; ".join(c+":"+("/".join(v["clear_gain_branches"]) or "未达到0.02幅度") for c,v in record["controls"].items())
            lines.append(f"| {record['snr_db']:g} | {record['label']} | {record['psnr_vs_B2_constraint_pass']} | {branches} |")
    lines += ["", "## 共同权重与融合前诊断", "",
        "| SNR | 输出 | 控制 | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δlatent平方和 [95% CI] |", "|---:|---|---|---|---|---|"]
    for snr in sorted({r["snr_db"] for r in summary}):
        for kind in ("common","tok"):
            for prior in ("A1","A2"):
                key=(snr,"V_"+kind,prior+"_"+kind,"lpips_alex")
                if key in lookup:
                    values=[delta_text(lookup[(snr,key[1],key[2],m)],3 if m.startswith("latent") else 5) for m in ("lpips_alex","dino_cosine","latent_sq_err_final")]
                    lines.append(f"| {snr:g} | {key[1]} | {key[2]} | "+" | ".join(values)+" |")
    lines += ["", "各自校准融合与共同权重结果均保留。共同权重保留的质量差支持候选内容的作用；若只有各自融合成立，需结合 alpha、bias/variance/MSE 与交叉二阶矩说明对融合校准的依赖。token 准确率仅解释机制，不参与策略排名。", "",
        "## 范围外与 13 dB 保护", "",
        "| SNR | 标签 | ΔPSNR V_policy−B2 | ΔLPIPS V_policy−B2 | ΔDINO V_policy−B2 |", "|---:|---|---|---|---|"]
    for record in decisions:
        snr=record["snr_db"]
        if snr not in MAIN_SNRS:
            vv=[delta_text(lookup[(snr,"V_policy","B2",m)]) for m in ("psnr_db","lpips_alex","dino_cosine")]
            lines.append(f"| {snr:g} | {record['label']} | "+" | ".join(vv)+" |")
    h=next((r for r in decisions if r["snr_db"]==13),None)
    if h:
        hh=h["high_snr_protection"]
        lines += ["", f"13 dB 最终策略：{h['policy_action']}；实测 PSNR 差≥−0.2：{hh['psnr_observed_constraint_pass']}；LPIPS 显著恶化：{hh['lpips_significantly_worse']}。"]
        if h["policy_action"].upper()=="BYPASS":
            lines.append("该点说明旁路保护有效，不证明 VAR 修正本身在高 SNR 下无损。")
        lines.append("未发现显著恶化不等于证明统计等效。−5/−2 dB 是训练范围外诊断，不能合并成范围内优势。")
        for name in ("V_raw_fused","V_candidate_fused","V_raw","V_lambda1","V_common","V_tok"):
            if (13,name,"B2","psnr_db") in lookup:
                r=level_for(policy,13).get("methods",{}).get("V",{})
                lines.append(f"13 dB 未旁路 {name}（raw λ={r.get('raw_selected_lambda')}；V_lambda1 固定 λ=1）相对 B2：PSNR {delta_text(lookup[(13,name,'B2','psnr_db')])}，LPIPS {delta_text(lookup[(13,name,'B2','lpips_alex')])}。")
    lines += ["", "## 冻结策略与计算代价", "",
        "| SNR | 接收器 | raw λ | raw可行 | 策略 | alpha均值 [通道范围] |", "|---:|---|---:|---|---|---|"]
    for snr in sorted({r["snr_db"] for r in summary}):
        for prior in ("A1","A2","V"):
            m=level_for(policy,snr).get("methods",{}).get(prior,{})
            aa=np.asarray(m.get("alpha",[]),dtype=float)
            atext=f"{aa.mean():.5f} [{aa.min():.5f}, {aa.max():.5f}]" if aa.size else "未提供"
            lines.append(f"| {snr:g} | {prior} | {m.get('raw_selected_lambda')} | {m.get('raw_feasible')} | {m.get('policy_action')} | {atext} |")
    if timing:
        lines += ["", "| SNR | 方法 | raw λ | 实际运行VAR | 接收时间均值 ms | 中位 ms | p95 ms | 测量次数 |", "|---:|---|---:|---|---:|---:|---:|---:|"]
        for r in timing:
            if "rx_ms_mean" in r:
                label='V_raw 未旁路修正' if r['method']=='V_raw' else r['method']
                lines.append(f"| {r['snr_db']:g} | {label} | {r.get('raw_selected_lambda')} | {r.get('ran_VAR')} | {r['rx_ms_mean']:.3f} | {r['rx_ms_median']:.3f} | {r['rx_ms_p95']:.3f} | {r['calls']} |")
        lines.append("计时边界为 CPU 接收波形→CPU 图像，包括 P receive 网络；新增修正分项见 timing.csv/timing_summary.csv。旁路耗时与实际运行 VAR 的未旁路耗时分别报告。")
        lines.append("V_raw 表示未旁路修正；若选中 lambda=0，修正只运行无先验量化，ran_VAR=False，其速度不能称为运行 VAR 的速度。")
    else:
        lines += ["", "接收计时尚无可读数据，不能据此报告执行代价。"]
    lines += ["", "## 证据与适用范围", "",
        "- 原始 per_frame.csv 保留同一 P 观测下的全量配对图像指标、原坐标 latent 误差、动作、lambda 与实测 N/E。",
        "- 以 100 个源图为推断单位；每张源图先平均三噪声。所有方法共用 10000 组源图 bootstrap 索引，seed=20260930，逐点 95% 百分位区间，未作多重比较校正。",
        "- 该 development 已多轮使用，本轮是探索性验证；区间不包含训练 seed 变异。",
        "- 经验重建误差方差包含学习映射与压缩误差；融合是误差可能相关时的启发式收缩，不是精确物理后验。",
        "- BYPASS 原样返回 B2；lambda=0 仍是量化修正候选。旁路结果不记为 VAR 修正成功。",
        "- TF 使用真实前缀，只作局部判别诊断；CL 的路径条件准确率与对原编码 token 的一致率分别保留。",
        "- calibration 冻结参数后一次运行 development，未按 development 修改 lambda、融合、SNR 子集或 checkpoint。", "",
        "文件身份：config.json `"+sha(results_dir/"config.json")+"`；selected_policy.json `"+sha(results_dir/"selected_policy.json")+"`；per_frame.csv `"+sha(results_dir/"per_frame.csv")+"`。", "",
        "详细表：summary.csv、paired.csv、source_means.csv、decisions.json、error_stats.csv、timing.csv；逐尺度 token 诊断另表保留。", "",
        "配对统计的预先数值保护：每源图差值绝对值≤1e−12 置为数值0后 bootstrap；paired.csv 同时保留 raw_delta_mean。该规则避免理论相同的匹配/错配增益被浮点舍入误记为特异性增益。", ""]
    for name in figure_files:
        if name.endswith(".png"):
            lines += [f"![{Path(name).stem}]({name})", ""]
    (results_dir/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main(out_dir: Path, results_dir: Path):
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    config=read_json(results_dir/"config.json")
    policy=read_json(results_dir/"selected_policy.json")
    with (results_dir/"per_frame.csv").open(encoding="utf-8",newline="") as handle:
        raw=list(csv.DictReader(handle))
    matrix, means, ids, snrs, methods=source_means(raw)
    if snrs != [-5.,-2.,1.,4.,7.,13.]:
        raise ValueError("Registered SNR scope differs from A execution plan")
    indices=np.random.default_rng(SEED).integers(len(ids),size=(RESAMPLES,len(ids)),dtype=np.int64)
    summary,paired,lookup=analyze(matrix,snrs,methods,indices)
    decisions=decision_table(snrs,lookup,policy)
    write_csv(results_dir/"source_means.csv",means)
    write_csv(results_dir/"summary.csv",summary)
    write_csv(results_dir/"paired.csv",paired)
    write_json(results_dir/"decisions.json",decisions)
    tokens=token_summary(results_dir,ids,indices)
    timing=timing_summary(results_dir)
    figure_files=make_figures(results_dir,summary,paired,policy)
    make_report(results_dir,summary,lookup,decisions,policy,timing,config,figure_files)
    completion=dict(status="FROZEN_A_SOURCE_PAIRED_ANALYSIS_COMPLETE",sources=len(ids),noise_repeats=3,
        methods=methods,snrs=snrs,bootstrap_seed=SEED,bootstrap_resamples=RESAMPLES,
        bootstrap_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),source_order=ids,
        files={name:sha(results_dir/name) for name in ("source_means.csv","summary.csv","paired.csv","decisions.json","report.md")},
        figures=figure_files,token_summary_rows=len(tokens),training_updates=0,automatic_B_start=False,automatic_old_queue_resume=False)
    write_json(results_dir/"analysis_completion.json",completion)
    return completion


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    args=parser.parse_args()
    result=main(args.out,args.results)
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_order","files","figures")},ensure_ascii=False))
