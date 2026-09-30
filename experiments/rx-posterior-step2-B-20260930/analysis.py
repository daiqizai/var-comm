"""Source-paired CPU analysis of frozen selected P_low receiver corrections.

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
MAIN_SNRS = (-5, -2, 1)
TRAINING_SNRS = (-5, -2, 1, 4)
EVALUATION_SNRS = (-5, -2, 1, 4, 13)
NOISE_SEEDS = {2001, 2002, 2003}
POLICIES = ("P_low", "A1_policy", "A2_policy", "V_policy")
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
                       in_training_range=snr in (-5, -2, 1, 4), inference_unit="source_image")
            for i, metric in enumerate(METRICS):
                result = interval(values[:, i], indices)
                row[metric] = result["mean"]
                row[metric + "_ci_low"] = result["ci_low"]
                row[metric + "_ci_high"] = result["ci_high"]
            summary.append(row)
    comparisons = [("V_policy", c, "primary") for c in ("P_low", "A1_policy", "A2_policy")]
    comparisons += [(m, "P_low", "control_vs_base") for m in ("A1_policy", "A2_policy")]
    for kind in ("common", "tok"):
        comparisons += [("V_" + kind, p + "_" + kind, "content_diagnostic" if kind == "common" else "token_diagnostic")
                        for p in ("A1", "A2") if "V_" + kind in methods and p + "_" + kind in methods]
    for method in methods:
        if method not in POLICIES:
            comparisons.append((method, "P_low", "diagnostic_vs_base"))
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
        psnr = lookup[(snr, "V_policy", "P_low", "psnr_db")]
        psnr_ok = psnr["delta"] >= -0.3
        per_control = {}
        for control in ("P_low", "A1_policy", "A2_policy"):
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
            label = "G2_CLEAR_GAIN"
        elif psnr_ok and supported == 3:
            label = "SMALL_SUPPORTED_GAIN"
        elif psnr_ok and supported:
            label = "PARTIAL_GAIN"
        else:
            label = "NO_GAIN_ESTABLISHED"
        scope_label = label if snr in MAIN_SNRS else ('SUPPORTING_' if snr == 4 else 'SIDE_EFFECT_') + label
        record = dict(snr_db=snr, in_training_range=snr in TRAINING_SNRS,
            role="outside_training_range_side_effect" if snr == 13 else "G2_primary" if snr in MAIN_SNRS else "within_training_range_supporting",
            G2_applicable=snr in MAIN_SNRS,policy_action=action,label=scope_label,
            quality_label=label,psnr_vs_P_low_constraint_pass=psnr_ok,
            psnr_vs_P_low=psnr, controls=per_control)
        if snr == 13:
            lp = lookup[(snr, "V_policy", "P_low", "lpips_alex")]
            record["high_snr_protection"] = dict(psnr_observed_constraint_pass=psnr["delta"] >= -0.2,
                lpips_significantly_worse=lp["ci_low"] > 0,
                observational_check_pass=psnr["delta"] >= -0.2 and lp["ci_low"] <= 0,
                interpretation="bypass protection only" if action.upper() == "BYPASS" else "observational check; statistical equivalence was not tested")
        records.append(record)
    return records


def compare_original_base(results_dir, config, raw, matrix, source_order, indices):
    """Cross-base pairing uses source/noise/budget, with separately generated y."""
    reference = config['original_base_reference']
    original_path = Path(reference['per_frame'])
    original_config = read_json(reference['config'])
    if original_config['mismatch_permutation'] != config['mismatch_permutation']:
        raise ValueError('Cross-base DINO mismatch targets differ')
    with original_path.open(encoding='utf-8', newline='') as handle:
        original = [r for r in csv.DictReader(handle)
                    if r['method'] == 'B2' and float(r['snr_db']) in EVALUATION_SNRS]
    current = [r for r in raw if r['method'] == 'P_low']
    key = lambda r: (r['source_id'], float(r['snr_db']), int(r['noise_seed']))
    before, after = {key(r): r for r in original}, {key(r): r for r in current}
    expected = {(sid, float(s), n) for sid in source_order
                for s in EVALUATION_SNRS for n in NOISE_SEEDS}
    if len(before) != len(original) or len(after) != len(current) or set(before) != expected or set(after) != expected:
        raise ValueError('Cross-base source/noise grid is incomplete or duplicated')
    same_wave, same_observation, energy_difference = 0, 0, 0.
    frame_rows = []
    for k in sorted(expected):
        a, b = before[k], after[k]
        if a['preprocessing_id'] != b['preprocessing_id'] or int(a['source_index']) != int(b['source_index']):
            raise ValueError('Cross-base source preprocessing or ordering differs')
        if int(a['N']) != 4084 or int(b['N']) != 4084:
            raise ValueError('Cross-base channel-use budgets differ')
        for row in (a, b):
            if not np.isclose(float(row['E']), 8168., atol=.02, rtol=1e-5):
                raise ValueError('Cross-base registered energy differs')
        energy_difference = max(energy_difference, abs(float(a['E']) - float(b['E'])))
        sw = a['waveform_sha256'] == b['waveform_sha256']
        sy = a['observation_sha256'] == b['observation_sha256']
        same_wave += int(sw); same_observation += int(sy)
        frame_rows.append(dict(source_id=k[0], snr_db=k[1], noise_seed=k[2], N=4084,
            E_original=float(a['E']), E_P_low=float(b['E']),
            waveform_sha_original=a['waveform_sha256'],waveform_sha_P_low=b['waveform_sha256'],
            observation_sha_original=a['observation_sha256'],observation_sha_P_low=b['observation_sha256'],
            same_waveform=sw,same_observation=sy,pairing_unit='source/noise_seed/N/E'))
    paired = []
    for snr in EVALUATION_SNRS:
        vectors = []
        for sid in source_order:
            vals = []
            for seed in sorted(NOISE_SEEDS):
                row = before[(sid, float(snr), seed)]
                v = {m: float(row[m]) for m in METRICS[:7]}
                v['dino_lt_0_6'] = float(v['dino_cosine'] < .6)
                v['lpips_gt_0_35'] = float(v['lpips_alex'] > .35)
                v['dino_match_specificity'] = v['dino_cosine'] - v['dino_mismatched']
                vals.append([v[m] for m in METRICS])
            vectors.append(np.asarray(vals).mean(0))
        diff = matrix[(float(snr), 'P_low')] - np.asarray(vectors)
        for j, metric in enumerate(METRICS):
            d = diff[:, j]
            stable = np.where(np.abs(d) <= STATISTICAL_ZERO_TOLERANCE, 0., d)
            ci = interval(stable, indices)
            paired.append(dict(snr_db=snr, method='P_low', control='P4084_original',
                comparison_role='cross_base_source_noise_budget_paired',
                metric='delta_specific' if metric == 'dino_match_specificity' else metric,
                delta=ci['mean'],ci_low=ci['ci_low'],ci_high=ci['ci_high'],raw_delta_mean=float(d.mean()),
                statistical_zero_tolerance=STATISTICAL_ZERO_TOLERANCE,sources=100,noise_repeats=3,
                bootstrap_resamples=RESAMPLES,bootstrap_seed=SEED,multiple_comparison_adjustment='none',
                observation_pairing='each base uses its separately generated waveform/observation'))
    identity = dict(original_per_frame=str(original_path),original_per_frame_sha256=sha(original_path),
        original_config_sha256=sha(Path(reference['config'])),frames=len(frame_rows),
        same_waveform_frames=same_wave,same_observation_frames=same_observation,
        energy_difference_max=energy_difference,pairing='source/preprocessing/SNR/noise_seed/N/E',
        interpretation='cross-base adaptation comparison; excluded from G2 prior-increment gates')
    write_csv(results_dir/'original_base_pairing.csv',frame_rows)
    write_csv(results_dir/'original_base_paired.csv',paired)
    write_json(results_dir/'original_base_identity.json',identity)
    return paired, identity


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
    colors = {"P_low": "#555555", "A1_policy": "#d8901d", "A2_policy": "#49832c", "V_policy": "#346ac9"}
    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(figures / (name + "." + suffix))
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    selected_metrics = ("psnr_db", "lpips_alex", "dino_cosine", "latent_sq_err_final", "dino_lt_0_6", "lpips_gt_0_35")
    for ax, metric in zip(axes.flat, selected_metrics):
        for method in POLICIES:
            ax.plot(snrs, [rows[(s, method)][metric] for s in snrs], "o-", label=method, color=colors[method])
        ax.axvspan(-5.5, 1.5, color="#eeeeee", alpha=.55, zorder=0)
        ax.set(xlabel="Nominal SNR (dB)", ylabel=metric)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "quality_snr")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, ("psnr_db", "lpips_alex", "dino_cosine")):
        for offset, control in zip((-.1, 0, .1), ("P_low", "A1_policy", "A2_policy")):
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
    supported = [r for r in decisions if r["snr_db"] in MAIN_SNRS and r["label"] in ("G2_CLEAR_GAIN","SMALL_SUPPORTED_GAIN")]
    against_base=[s for s in MAIN_SNRS if next(r for r in decisions if r["snr_db"]==s)["controls"]["P_low"]["quality_supported"]]
    cost={}
    for row in timing:
        cost[(row["snr_db"],row["method"])]=row.get("rx_ms_mean")
    selected=config['identity']['selected']
    lines=["# P_low 专门低 SNR 微调后的接收增量：B 部分", "",
        f"本轮 P_low 训练共完成新增更新 {config['completed_added_updates']}；全 1000 张 calibration 选中的 checkpoint 使用新增更新 {selected['step']}，总更新 {selected['total_updates']}；SHA-256 `{selected['checkpoint_sha256']}`。接收后处理评测更新参数次数为 0。", "",
        "1. **V 是否胜过 P_low？** " + ("相对 P_low 有统计支持的质量收益点："+"、".join(f"{s} dB" for s in against_base)+"。" if against_base else "−5/−2/1 dB 的 G2 主点未建立相对 P_low 的质量收益。") + "逐点指标、PSNR 约束和接收时间列在下表。",
        "2. **V 是否胜过 A1 和 A2？** " + ("三个直接控制均支持质量增量的点："+"、".join(f"{r['snr_db']:g} dB ({r['label']})" for r in supported)+"。" if supported else "−5/−2/1 dB 的 G2 主点未建立超出三个直接控制的质量增量。") + "只胜 P_low 时不归因于 VAR 条件先验。",
        "3. **共同融合权重下是否仍有增量？** 下面逐点报告 V_common 相对 A1_common/A2_common 的配对差；共同权重固定为 calibration 的 A1 alpha，未重新选 lambda。旁路时这些是诊断输出。",
        "4. **专门适配低 SNR 后是否保留额外增量？** " + ("上述 G2 主点支持在当前专门训练基线上保留接收增量；下一步可独立复现相同基线与校准流程。" if supported else "当前设置未建立或复现超出三个直接控制的低 SNR 接收增量。") + "这不自动证明新颖性或论文贡献；未通过也不能单凭此结果把 A 的全部收益归因于训练失配。P_low_perc 与旧队列均未启动。", "",
        "## G2 主结果：−5、−2、1 dB", "", "训练 SNR 为 −5/−2/1/4 dB；4 dB 是支持性评测，13 dB 是训练范围外副作用诊断。以下均先对每张源图的三个噪声平均，再对 100 张源图平均。PSNR 是逐图 PSNR 的均值。", "",
        "| SNR | 方法 | PSNR dB | LPIPS alex | DINO | 原坐标 latent 平方和 | DINO<0.6 | LPIPS>0.35 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for snr in MAIN_SNRS:
        for method in POLICIES:
            r=rows[(snr,method)]
            lines.append(f"| {snr} | {method} | {r['psnr_db']:.4f} | {r['lpips_alex']:.5f} | {r['dino_cosine']:.5f} | {r['latent_sq_err_final']:.3f} | {r['dino_lt_0_6']:.4f} | {r['lpips_gt_0_35']:.4f} |")
    lines += ["", "上述两个阈值比例仅是所设指标阈值的失效比例。latent 误差是 32×16×16 原坐标平方和；均方值可除以 8192，不能把两者混用。", "",
        "| SNR | 控制 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO [95% CI] | Δspecific [95% CI] |", "|---:|---|---|---|---|---|"]
    for snr in MAIN_SNRS:
        for control in ("P_low","A1_policy","A2_policy"):
            values=[delta_text(lookup[(snr,"V_policy",control,k)]) for k in ("psnr_db","lpips_alex","dino_cosine","delta_specific")]
            lines.append(f"| {snr} | {control} | " + " | ".join(values) + " |")
    lines += ["", "所有差值方向为 V−控制；LPIPS 和 latent 误差越低越好，PSNR/DINO 越高越好。DINO 增量的判定同时要求对应 Δspecific 区间下界>0；LPIPS 判定独立进行。", "",
        "| SNR | 标签 | 相对 P_low PSNR≥−0.3 | 各控制明显收益分支 |", "|---:|---|---|---|"]
    for record in decisions:
        if record["snr_db"] in MAIN_SNRS:
            branches="; ".join(c+":"+("/".join(v["clear_gain_branches"]) or "未达到0.02幅度") for c,v in record["controls"].items())
            lines.append(f"| {record['snr_db']:g} | {record['label']} | {record['psnr_vs_P_low_constraint_pass']} | {branches} |")
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
        "## 4 dB 支持性结果与 13 dB 副作用", "",
        "| SNR | 标签 | ΔPSNR V_policy−P_low | ΔLPIPS V_policy−P_low | ΔDINO V_policy−P_low |", "|---:|---|---|---|---|"]
    for record in decisions:
        snr=record["snr_db"]
        if snr not in MAIN_SNRS:
            vv=[delta_text(lookup[(snr,"V_policy","P_low",m)]) for m in ("psnr_db","lpips_alex","dino_cosine")]
            lines.append(f"| {snr:g} | {record['label']} | "+" | ".join(vv)+" |")
    lines += ["", "4 dB 是训练范围内支持性结果，不并入 G2 的极低 SNR 主判据。", "",
        "| 4 dB 方法 | PSNR dB | LPIPS alex | DINO |", "|---|---:|---:|---:|"]
    for method in POLICIES:
        r=rows[(4.,method)]
        lines.append(f"| {method} | {r['psnr_db']:.4f} | {r['lpips_alex']:.5f} | {r['dino_cosine']:.5f} |")
    h=next((r for r in decisions if r["snr_db"]==13),None)
    if h:
        hh=h["high_snr_protection"]
        lines += ["", f"13 dB 最终策略：{h['policy_action']}；实测 PSNR 差≥−0.2：{hh['psnr_observed_constraint_pass']}；LPIPS 显著恶化：{hh['lpips_significantly_worse']}。"]
        if h["policy_action"].upper()=="BYPASS":
            lines.append("该点说明旁路保护有效，不证明 VAR 修正本身在高 SNR 下无损。")
        lines.append("未发现显著恶化不等于证明统计等效。13 dB 未进入 P_low 专门微调的 SNR 集合，不参与 G2 极低 SNR 主判据。")
        for name in ("V_raw_fused","V_candidate_fused","V_raw","V_lambda1","V_common","V_tok"):
            if (13,name,"P_low","psnr_db") in lookup:
                r=level_for(policy,13).get("methods",{}).get("V",{})
                lines.append(f"13 dB 未旁路 {name}（raw λ={r.get('raw_selected_lambda')}；V_lambda1 固定 λ=1）相对 P_low：PSNR {delta_text(lookup[(13,name,'P_low','psnr_db')])}，LPIPS {delta_text(lookup[(13,name,'P_low','lpips_alex')])}。")
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
    lines += ["", "## P_low 与原 P4084 的跨基线比较", "",
        "两者按相同源图、预处理、SNR、噪声种子和 N4084/E8168 配对，各自生成收发波形与观测。此比较评价低 SNR 基线适配，未参与 G2 的 VAR 独立增量判据。", "",
        "| SNR | ΔPSNR P_low−原 P | ΔLPIPS P_low−原 P | ΔDINO P_low−原 P |", "|---:|---|---|---|"]
    for snr in EVALUATION_SNRS:
        vv=[delta_text(lookup[(snr,'P_low','P4084_original',m)]) for m in ('psnr_db','lpips_alex','dino_cosine')]
        lines.append(f"| {snr} | "+" | ".join(vv)+" |")
    lines += ["", "## 证据与适用范围", "",
        "- 原始 per_frame.csv 保留同一 P_low 观测下的全量配对图像指标、原坐标 latent 误差、动作、lambda 与实测 N/E。",
        "- 以 100 个源图为推断单位；每张源图先平均三噪声。所有方法共用 10000 组源图 bootstrap 索引，seed=20260930，逐点 95% 百分位区间，未作多重比较校正。",
        "- 该 development 已多轮使用，本轮是探索性验证；区间不包含训练 seed 变异。",
        "- 经验重建误差方差包含学习映射与压缩误差；融合是误差可能相关时的启发式收缩，不是精确物理后验。",
        "- BYPASS 原样返回 P_low；lambda=0 仍是量化修正候选。旁路结果不记为 VAR 修正成功。",
        "- TF 使用真实前缀，只作局部判别诊断；CL 的路径条件准确率与对原编码 token 的一致率分别保留。",
        "- checkpoint 由完整 1000 张 calibration 选择；之后在固定 200 张 calibration 上重新估计误差并冻结 lambda、融合与旁路，再运行原 100 张 development。未按 development 修改参数或 checkpoint。", "",
        "文件身份：config.json `"+sha(results_dir/"config.json")+"`；selected_policy.json `"+sha(results_dir/"selected_policy.json")+"`；per_frame.csv `"+sha(results_dir/"per_frame.csv")+"`。", "",
        "详细表：summary.csv、paired.csv、source_means.csv、decisions.json、error_stats.csv、timing.csv、original_base_paired.csv、original_base_pairing.csv；逐尺度 token 诊断另表保留。", "",
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
    if snrs != [-5.,-2.,1.,4.,13.]:
        raise ValueError("Registered SNR scope differs from B evaluation plan")
    indices=np.random.default_rng(SEED).integers(len(ids),size=(RESAMPLES,len(ids)),dtype=np.int64)
    summary,paired,lookup=analyze(matrix,snrs,methods,indices)
    decisions=decision_table(snrs,lookup,policy)
    original_paired,original_identity=compare_original_base(results_dir,config,raw,matrix,ids,indices)
    paired.extend(original_paired)
    for row in original_paired:
        lookup[(row['snr_db'],row['method'],row['control'],row['metric'])]=row
    write_csv(results_dir/"source_means.csv",means)
    write_csv(results_dir/"summary.csv",summary)
    write_csv(results_dir/"paired.csv",paired)
    write_json(results_dir/"decisions.json",decisions)
    tokens=token_summary(results_dir,ids,indices)
    timing=timing_summary(results_dir)
    figure_files=make_figures(results_dir,summary,paired,policy)
    make_report(results_dir,summary,lookup,decisions,policy,timing,config,figure_files)
    completion=dict(status="FROZEN_B_P_LOW_SOURCE_PAIRED_ANALYSIS_COMPLETE",sources=len(ids),noise_repeats=3,
        methods=methods,snrs=snrs,bootstrap_seed=SEED,bootstrap_resamples=RESAMPLES,
        bootstrap_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),source_order=ids,
        files={name:sha(results_dir/name) for name in ("source_means.csv","summary.csv","paired.csv","decisions.json","report.md")},
        figures=figure_files,token_summary_rows=len(tokens),evaluation_training_updates=0,
        selected_finetuning_updates=config['selected_finetuning_updates'],completed_added_updates=config['completed_added_updates'],
        original_base_comparison=original_identity,
        automatic_additional_training=False,automatic_old_queue_resume=False)
    write_json(results_dir/"analysis_completion.json",completion)
    return completion


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    args=parser.parse_args()
    result=main(args.out,args.results)
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_order","files","figures")},ensure_ascii=False))
