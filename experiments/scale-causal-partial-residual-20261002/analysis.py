"""CPU-only source-paired analysis for the two registered October 2 methods.

No experiment modules, model libraries, or observations are regenerated here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = "SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002"
DEFAULT_OUT = ROOT / "outputs" / RUN
DEFAULT_RESULTS = ROOT / "results/scale_causal_partial_residual_20261002"
SEED = 20261002
REPLICATES = 10000
ZERO_TOLERANCE = 1e-12
DEV_SEEDS = (2001, 2002, 2003)
SNRS = (1, 4, 7, 13, 19)
M1_REQUIRED = ("whole_policy", "raster_policy", "random_policy", "entropy_policy",
               "oracle_policy", "raster_at_entropy", "random_at_entropy", "legacy_policy")
METRICS = ("psnr_db", "lpips_alex", "dino_cosine", "dino_specific",
           "dino_fail_rate", "lpips_fail_rate", "latent_valid_rate", "F_sq_error_valid",
           "E", "header_fail_rate", "prefix_crc_fail_given_header",
           "partial_crc_fail_given_prefix", "partial_used_rate", "prefix_false_accept_rate",
           "gain_crc_fail_given_header", "gain_fields_invalid_given_header", "analog_used_rate",
           "E_header", "E_gain", "E_digital", "E_analog")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, required=False):
    if value is None or str(value).strip().lower() in ("", "none", "null", "nan", "n.a."):
        if required:
            raise ValueError("Required numerical value missing")
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Nonfinite numerical observation: " + str(value))
    return result


def boolean(value):
    if value is None or str(value).strip().lower() in ("", "none", "null", "nan"):
        return None
    if str(value).strip().lower() in ("true", "1", "1.0"):
        return True
    if str(value).strip().lower() in ("false", "0", "0.0"):
        return False
    raise ValueError("Invalid boolean: " + str(value))


def partial_count(row):
    value = row.get("K", row.get("q"))
    return None if number(value) is None else int(float(value))


def metric(row, name):
    """Return None only when a metric is genuinely undefined for this observation."""
    if name in ("psnr_db", "lpips_alex", "dino_cosine", "E", "E_header", "E_gain", "E_digital", "E_analog"):
        return number(row.get(name), required=name in ("psnr_db", "lpips_alex", "dino_cosine"))
    if name == "dino_specific":
        mismatch = number(row.get("dino_mismatched"))
        return None if mismatch is None else number(row["dino_cosine"], True) - mismatch
    if name == "dino_fail_rate":
        return float(number(row["dino_cosine"], True) < .6)
    if name == "lpips_fail_rate":
        return float(number(row["lpips_alex"], True) > .35)
    if name == "latent_valid_rate":
        valid = boolean(row.get("latent_valid"))
        return None if valid is None else float(valid)
    if name == "F_sq_error_valid":
        return number(row.get("latent_sq_err_final"), True) if boolean(row.get("latent_valid")) is True else None
    if name == "header_fail_rate":
        flag = boolean(row.get("header_ok"))
        return None if flag is None else float(not flag)
    if name == "prefix_crc_fail_given_header":
        if boolean(row.get("header_ok")) is not True:
            return None
        flag = boolean(row.get("prefix_crc_ok"))
        return None if flag is None else float(not flag)
    if name == "partial_crc_fail_given_prefix":
        if not partial_count(row) or boolean(row.get("header_ok")) is not True or boolean(row.get("prefix_crc_ok")) is not True:
            return None
        flag = boolean(row.get("partial_crc_ok"))
        return None if flag is None else float(not flag)
    if name in ("gain_crc_fail_given_header", "gain_fields_invalid_given_header"):
        if boolean(row.get("header_ok")) is not True:
            return None
        flag = boolean(row.get("gain_crc_ok" if name.startswith("gain_crc") else "gain_fields_valid"))
        return None if flag is None else float(not flag)
    if name == "prefix_false_accept_rate" and "prefix_actually_correct_offline" in row:
        return float(boolean(row.get("prefix_crc_ok")) is True and boolean(row["prefix_actually_correct_offline"]) is False)
    if name in ("partial_used_rate", "prefix_false_accept_rate", "analog_used_rate"):
        flag = boolean(row.get(name.removesuffix("_rate")))
        return None if flag is None else float(flag)
    raise KeyError(name)


def ordered_sources(rows):
    mapping = {}
    for row in rows:
        sid = str(row["source_id"])
        index = int(row["source_index"])
        if sid in mapping and mapping[sid] != index:
            raise ValueError("Source index changed for " + sid)
        mapping[sid] = index
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Different source identities share an index")
    return sorted(mapping, key=lambda sid: (mapping[sid], sid))


def verify_mismatch_identity(rows, sources, required=False):
    mapping = {}
    for row in rows:
        value = row.get("mismatch_source_id", "")
        if not value:
            if required:
                raise ValueError("Wrong-image reference identity missing")
            continue
        if value not in sources or value == row["source_id"]:
            raise ValueError("Wrong-image reference is outside population or equals true source")
        if row["source_id"] in mapping and mapping[row["source_id"]] != value:
            raise ValueError("Wrong-image reference changed across controls/stages")
        mapping[row["source_id"]] = value
    if required and (len(mapping) != len(sources) or len(set(mapping.values())) != len(sources)):
        raise ValueError("Wrong-image references are not a complete shared derangement")
    return mapping


def verify_registered_inputs(reg, rows, sources):
    bindings = reg.get("data_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(sources):
        raise ValueError("Actual pixel population bindings missing")
    identities = {r["source_id"]: str(r["preprocessing_id"]) for r in rows}
    for sid, binding in zip(sources, bindings):
        if binding.get("rgb_sha256") != identities[sid] or not re.fullmatch(r"[0-9a-fA-F]{64}", str(binding.get("source_npz_sha256", ""))):
            raise ValueError("Registered pixel/preprocessing identity differs")
    for path, digest in reg.get("input_artifacts", {}).items():
        if sha(path) != digest:
            raise ValueError("Frozen input artifact changed: " + path)


def copy_report(report, reports, filename):
    reports.mkdir(parents=True, exist_ok=True)
    text = report.read_text(encoding="utf-8")
    def relocate(match):
        target = match.group(1)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        moved = Path(os.path.relpath(report.parent / target, reports)).as_posix()
        return "](" + moved + ")"
    text = re.sub(r"\]\(([^)]+)\)", relocate, text)
    (reports / filename).write_text(text, encoding="utf-8")


def m1_cell(row):
    return (int(row["N"]), str(row["phy_family"]), int(float(row["snr_db"])), str(row["method"]))


def validate_groups(rows, cell_fn, expected_sources=100, expected_seeds=DEV_SEEDS):
    """Exact identifiers/counts, finite quality, and no duplicate source/noise cells."""
    groups = defaultdict(list)
    identity = {}
    for row in rows:
        sid = str(row["source_id"])
        ident = (int(row["source_index"]), str(row["preprocessing_id"]))
        if not ident[1]:
            raise ValueError("Missing preprocessing identity")
        if sid in identity and identity[sid] != ident:
            raise ValueError("Source/preprocessing identity differs across methods")
        identity[sid] = ident
        for name in ("psnr_db", "lpips_alex", "dino_cosine"):
            metric(row, name)
        metric(row, "F_sq_error_valid")
        groups[cell_fn(row)].append(row)
    if len(identity) != expected_sources:
        raise ValueError(f"Expected {expected_sources} source images, got {len(identity)}")
    sources = ordered_sources(rows)
    for cell, rr in groups.items():
        pairs = [(r["source_id"], int(r["noise_seed"])) for r in rr]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate source/noise observation: " + str(cell))
        expected = {(sid, seed) for sid in sources for seed in expected_seeds}
        if set(pairs) != expected:
            raise ValueError("Incomplete source/noise pairing: " + str(cell))
    return dict(groups), sources


def validate_m1(rows, expected_sources=100, budgets=(512, 1024), phys=("QPSK", "16QAM"), snrs=SNRS):
    groups, sources = validate_groups(rows, m1_cell, expected_sources)
    for N in budgets:
        for phy in phys:
            for snr in snrs:
                required = (*M1_REQUIRED, f"P{N}")
                for method in required:
                    if (N, phy, snr, method) not in groups:
                        raise ValueError("Missing registered development group: " + str((N, phy, snr, method)))
                ent = groups[N, phy, snr, "entropy_policy"]
                reference = {(r["source_id"], int(r["noise_seed"])): r for r in ent}
                for method in ("raster_at_entropy", "random_at_entropy", "oracle_at_entropy"):
                    rr = groups.get((N, phy, snr, method))
                    if rr is None:
                        if method != "oracle_at_entropy":
                            raise ValueError("Missing same-K control")
                        continue
                    for row in rr:
                        target = reference[row["source_id"], int(row["noise_seed"])]
                        if (int(row["m"]), partial_count(row)) != (int(target["m"]), partial_count(target)):
                            raise ValueError("Same-K control changed m or K")
                for method in required:
                    rr = groups[N, phy, snr, method]
                    if method not in ("legacy_policy", f"P{N}"):
                        acts = {(int(r["m"]), partial_count(r), str(r["order"])) for r in rr}
                        if len(acts) != 1:
                            raise ValueError("Frozen policy changed by source/noise")
                    if phy == "QPSK" or method == f"P{N}":
                        for row in rr:
                            energy = number(row.get("E"), True)
                            if abs(energy - 2 * N) > max(.001, 2 * N * 1e-6):
                                raise ValueError("QPSK/P fixed energy budget differs from 2N")
    return groups, sources


class Bootstrap:
    """One shared source resample for each ordered population size."""
    def __init__(self, replicates=REPLICATES, seed=SEED):
        self.replicates = replicates
        self.seed = seed
        self.indices = {}

    def interval(self, values, zero_deltas=False):
        a = np.asarray(values, dtype=np.float64)
        if len(a) == 0:
            return (None, None)
        if not np.isfinite(a).all():
            raise ValueError("Bootstrap input is nonfinite")
        if zero_deltas:
            a = a.copy()
            a[np.abs(a) <= ZERO_TOLERANCE] = 0.
        if len(a) not in self.indices:
            self.indices[len(a)] = np.random.default_rng(self.seed).integers(0, len(a), size=(self.replicates, len(a)))
        estimates = a[self.indices[len(a)]].mean(axis=1)
        lo, hi = np.quantile(estimates, [.025, .975])
        return float(lo), float(hi)


def source_values(rows, name, sources):
    groups = defaultdict(list)
    count = 0
    for row in rows:
        value = metric(row, name)
        if value is not None:
            groups[row["source_id"]].append(value)
            count += 1
    return {sid: float(np.mean(groups[sid])) for sid in sources if groups[sid]}, count


def summarize(groups, sources, bootstrap, cell_fields):
    output, per_source = [], []
    for cell, rows in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        context = dict(zip(cell_fields, cell))
        for name in METRICS:
            values, count = source_values(rows, name, sources)
            a = list(values.values())
            lo, hi = bootstrap.interval(a)
            output.append(dict(**context, metric=name, mean=float(np.mean(a)) if a else None,
                               ci_low=lo, ci_high=hi, n_sources=len(a), n_eligible_frames=count,
                               n_total_sources=len(sources), n_total_frames=len(rows),
                               aggregation="eligible_noise_mean_within_source_then_equal_source_mean"))
            for sid, value in values.items():
                per_source.append(dict(**context, source_id=sid, metric=name, value=value))
    return output, per_source


def paired(rows_a, rows_b, sources, bootstrap, context):
    """Pair first at source/noise; valid-F deltas use the common-valid noise subset."""
    b = {(r["source_id"], int(r["noise_seed"])): r for r in rows_b}
    if len(b) != len(rows_b):
        raise ValueError("Duplicate comparison observation")
    output = []
    for name in METRICS:
        deltas = defaultdict(list)
        n_frames = 0
        for a in rows_a:
            key = (a["source_id"], int(a["noise_seed"]))
            if key not in b:
                raise ValueError("Unpaired comparison observation")
            av, bv = metric(a, name), metric(b[key], name)
            if av is not None and bv is not None:
                deltas[a["source_id"]].append(av - bv)
                n_frames += 1
        values = [float(np.mean(deltas[sid])) for sid in sources if deltas[sid]]
        lo, hi = bootstrap.interval(values, zero_deltas=True)
        output.append(dict(**context, metric=name, delta_mean=float(np.mean(values)) if values else None,
                           ci_low=lo, ci_high=hi, n_sources_paired=len(values), n_frames_paired=n_frames,
                           n_total_sources=len(sources), direction="A_minus_B", conditional=name in (
                               "F_sq_error_valid", "prefix_crc_fail_given_header", "partial_crc_fail_given_prefix",
                               "gain_crc_fail_given_header", "gain_fields_invalid_given_header"),
                           statistical_zero_tolerance=ZERO_TOLERANCE))
    return output


def m1_contrasts(groups, sources, bootstrap):
    output = []
    cells = sorted({key[:3] for key in groups})
    for N, phy, snr in cells:
        comparisons = [("entropy_policy", name) for name in (
            "raster_at_entropy", "random_at_entropy", "whole_policy", "legacy_policy", f"P{N}")]
        if (N, phy, snr, "oracle_at_entropy") in groups:
            comparisons.append(("entropy_policy", "oracle_at_entropy"))
        comparisons += [(name, f"P{N}") for name in (
            "whole_policy", "raster_policy", "random_policy", "oracle_policy")]
        for A, B in comparisons:
            a, b = groups[N, phy, snr, A], groups[N, phy, snr, B]
            historical = B == "legacy_policy" or B.startswith("P") or A == "legacy_policy" or A.startswith("P")
            context = dict(N=N, phy_family=phy, snr_db=snr, method_A=A, method_B=B,
                           pairing="same_source_same_nominal_seed_historical_distinct_noise" if historical else "same_source_shared_physical_noise",
                           same_m_K=B.endswith("_at_entropy"),
                           energy_comparison="strict_E_2N" if phy == "QPSK" else "same_N_modulation_actual_E_may_differ")
            output.extend(paired(a, b, sources, bootstrap, context))
    return output


def high_snr_evidence(contrasts):
    lookup = {(r["N"], r["phy_family"], r["snr_db"], r["method_B"], r["metric"]): r
              for r in contrasts if r["method_A"] == "entropy_policy"}
    output = []
    for N, phy, snr, baseline, _ in sorted(lookup):
        if snr not in (7, 13, 19):
            continue
        key = (N, phy, snr, baseline)
        if any(all(r[k] == v for k, v in zip(("N", "phy_family", "snr_db", "baseline"), key)) for r in output):
            continue
        lp = lookup[key + ("lpips_alex",)]
        ps = lookup[key + ("psnr_db",)]
        di = lookup[key + ("dino_cosine",)]
        gain = lp["ci_high"] is not None and lp["ci_high"] < 0
        protected = ps["ci_low"] is not None and ps["ci_low"] >= -.25
        output.append(dict(N=N, phy_family=phy, snr_db=snr, baseline=baseline,
                           lpips_delta=lp["delta_mean"], lpips_ci_low=lp["ci_low"], lpips_ci_high=lp["ci_high"],
                           psnr_delta=ps["delta_mean"], psnr_ci_low=ps["ci_low"], psnr_ci_high=ps["ci_high"],
                           dino_delta=di["delta_mean"], dino_ci_low=di["ci_low"], dino_ci_high=di["ci_high"],
                           descriptive_status="LPIPS_GAIN_WITH_PSNR_PROTECTION" if gain and protected else "NOT_ESTABLISHED_BY_THIS_CONTRAST",
                           criterion="LPIPS_delta_upper<0_and_PSNR_delta_lower>=-0.25dB",
                           energy_comparison=lp["energy_comparison"], pairing=lp["pairing"]))
    return output


def timing_summary(path, dest, cell_fields):
    if not Path(path).exists():
        return []
    rows = read_csv(path)
    groups = defaultdict(list)
    for row in rows:
        if boolean(row.get("cache_used")) is True:
            raise ValueError("Timing row uses a receiver cache")
        groups[tuple(row.get(k, "") for k in cell_fields)].append(row)
    result = []
    for cell, rr in sorted(groups.items()):
        for field in ("tx_ms", "rx_ms"):
            vals = [number(r.get(field), True) for r in rr]
            result.append(dict(**dict(zip(cell_fields, cell)), metric=field, mean=float(np.mean(vals)),
                               median=float(np.median(vals)), min=min(vals), max=max(vals), n_timed=len(vals),
                               cache_used=False, includes_actual_erasure_paths=True))
    write_csv(dest, result)
    return result


def plot_m1(groups, summary, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = results / "figures"
    folder.mkdir(exist_ok=True)
    colors = {"entropy_policy": "#1769aa", "raster_at_entropy": "#b35e16", "random_at_entropy": "#707070",
              "whole_policy": "#2e8b57", "legacy_policy": "#a45aa0", "P512": "#161616", "P1024": "#161616"}
    created = []
    for N in (512, 1024):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
        for i, phy in enumerate(("QPSK", "16QAM")):
            for j, name in enumerate(("psnr_db", "lpips_alex", "dino_specific")):
                for method, color in colors.items():
                    rr = [r for r in summary if (r["N"], r["phy_family"], r["metric"], r["method"]) == (N, phy, name, method) and r["mean"] is not None]
                    rr.sort(key=lambda r: r["snr_db"])
                    if rr:
                        axes[i, j].plot([r["snr_db"] for r in rr], [r["mean"] for r in rr], "o-", color=color, label=method)
                axes[i, j].set_title(phy + " · " + name)
                axes[i, j].set_xlabel("SNR (dB)")
                axes[i, j].grid(alpha=.2)
            axes[i, 0].legend(fontsize=7)
        fig.suptitle(f"M1 N={N}: 100 sources, noise averaged within source; 16QAM actual E varies")
        p = folder / f"m1_quality_N{N}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, N in zip(axes, (512, 1024)):
        for phy in ("QPSK", "16QAM"):
            for method in ("whole_policy", "raster_policy", "random_policy", "entropy_policy", "oracle_policy"):
                rr = [(key[2], partial_count(rows[0])) for key, rows in groups.items() if key[0] == N and key[1] == phy and key[3] == method]
                rr.sort()
                if rr:
                    ax.plot(*zip(*rr), "o-", label=phy + "/" + method)
        ax.set_title(f"N={N}; K=0 allowed")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Selected K")
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    p = folder / "m1_K_vs_SNR.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(p)
    rate_path = results / "m1_rate_curve.csv"
    if rate_path.exists():
        rows = read_csv(rate_path)
        gg = defaultdict(list)
        for row in rows:
            if boolean(row.get("wireless_claim")) is not False:
                raise ValueError("Source rate curve is not explicitly labelled non-wireless")
            bits = int(float(row["source_bits"])) + int(float(row.get("mask_bits", 0)))
            gg[int(row["m"]), row["order"], bits].append(row)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, met in zip(axes, ("lpips_alex", "dino_specific")):
            for m, order in sorted({key[:2] for key in gg}):
                points = []
                for (mm, oo, bits), rr in sorted(gg.items()):
                    if (mm, oo) == (m, order):
                        vals = [metric(r, met) for r in rr]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            points.append((bits, float(np.mean(vals))))
                if points:
                    ax.plot(*zip(*points), "o-", label=f"m{m}/{order}", markersize=3)
            ax.set_xlabel("Source content + paid oracle-mask bits (no channel coding)")
            ax.set_ylabel(met)
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle("Zero-noise source diagnostic; measured points only; not paid-link performance")
        p = folder / "m1_clean_source_rate_curve.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(p)
    return created


def formatted(value):
    return "不可用" if value is None else f"{value:.6g}"


def report_m1(results, groups, summary, contrasts, high, timing, sources):
    policy = read_json(results / "m1_policy.json")
    if policy.get("development_read") is not False:
        raise ValueError("Calibration policy does not certify development_read=false")
    lines = ["# 方法一：尺度因果的部分尺度传输", "", "## 主要结果", "",
             "统计单位为100个源图像。每源先平均三次登记噪声，再对源做10,000次共享配对bootstrap。区间为重复使用development上的描述性95%区间，未包含训练种子不确定性。", "",
             "熵排序的主要机制比较使用相同m、K的raster/random对照。与历史legacy/P仅按源及名义seed配对，历史物理噪声namespace不同。QPSK满足E=2N；16QAM沿用原符号映射，实际E可随bit变化，其对照仅同N/调制，能量差另列。", "",
             "| N | PHY | SNR | 对照 | ΔLPIPS [95% CI] | ΔPSNR [95% CI] | 判定 |",
             "|---|---|---:|---|---|---|---|"]
    for row in high:
        if row["baseline"] not in ("raster_at_entropy", "random_at_entropy", "whole_policy", "legacy_policy", f"P{row['N']}"):
            continue
        label = "LPIPS下降且PSNR保护成立" if row["descriptive_status"] == "LPIPS_GAIN_WITH_PSNR_PROTECTION" else "该对照未建立此结论"
        lines.append(f"| {row['N']} | {row['phy_family']} | {row['snr_db']} | {row['baseline']} | {formatted(row['lpips_delta'])} [{formatted(row['lpips_ci_low'])}, {formatted(row['lpips_ci_high'])}] | {formatted(row['psnr_delta'])} [{formatted(row['psnr_ci_low'])}, {formatted(row['psnr_ci_high'])}] | {label} |")
    lines += ["", "上述判定逐N/PHY/SNR/对照报告，条件为LPIPS差区间上界<0且PSNR差区间下界≥−0.25dB。它不单独证明总体最优或高SNR已全面解除饱和。熵机制须同时查看两个同K对照；16QAM还须查看ΔE。", "", "## 冻结的校准选择", "",
              f"校准policy SHA256：`{sha(results / 'm1_policy.json')}`。policy在development读取前封存，analysis不重选K。", "",
              "| N | PHY | SNR | 方法 | m | K | 顺序 |", "|---|---|---:|---|---:|---:|---|"]
    for key, rr in sorted(groups.items()):
        if key[3] in ("whole_policy", "raster_policy", "random_policy", "entropy_policy", "oracle_policy"):
            row = rr[0]
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {row['m']} | {partial_count(row)} | {row['order']} |")
    lines += ["", "## 指标、有效分母和失败", "",
              "`m1_summary.csv`保留PSNR、LPIPS、DINO、DINO specificity、DINO<0.6与LPIPS>0.35事件、E、latent有效率和各CRC指标。图像质量包含全部300帧/组，包括灰色erasure。", "",
              "F平方误差仅在实际latent有效的帧计算，不以零填补失败。汇总先对每源可用噪声平均再平均源；`n_eligible_frames/n_sources`注明分母。F配对差只使用双方同一source/noise均有效的交集，并注明其源与帧数。它不能代替全失败率。", "",
              "prefix CRC失败率以header有效为条件；partial CRC失败率以K>0且header/prefix均有效为条件。缺少字段的历史项保持不可用，不当作零失败。oracle按显式mask收费，是错误位置优先的诊断参考，未称全局质量上界。", "",
              "## 证据与图", "",
              "- [完整汇总](m1_summary.csv)、[逐源均值](m1_source_means.csv)、[全部配对区间](m1_paired_intervals.csv)、[高SNR对照](m1_high_snr_evidence.csv)。",
              "- [N512质量](figures/m1_quality_N512.png)、[N1024质量](figures/m1_quality_N1024.png)、[校准K](figures/m1_K_vs_SNR.png)。",
              "- `m1_rate_curve.csv`与对应图是zero-noise source诊断，只连接实测点，不作为实际付费链路性能。",
              "- 样例固定source_index=0/25/50/75、SNR=4/13、seed=2001，位于`examples/m1/`，不按结果挑图。", "", "## 接收与发送时间", ""]
    if timing:
        lines += ["计时来自独立uncached路径；实际header失败/erasure路径保留，不由质量CSV中的缓存运行时间代替。", "", "| N | PHY | SNR | 方法 | 阶段 | 均值ms | 样本数 |", "|---|---|---:|---|---|---:|---:|"]
        for row in timing:
            lines.append(f"| {row['N']} | {row['phy_family']} | {row['snr_db']} | {row['method']} | {row['metric']} | {formatted(row['mean'])} | {row['n_timed']} |")
    else:
        lines.append("独立计时文件尚未提供，时间结论不可用。")
    lines += ["", "## 范围", "", "本报告完成方法一统计。方法二及其条件实际链路分支的状态由各自证据决定；本报告不写整个研究已完成。新颖性与已有先例见`literature_notes.md`。", ""]
    path = results / "m1_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def verify_m1_registration(out, results, rows, sources, allow_synthetic=False):
    paths = (out / "m1_development_registration.json", out / "m1_development_complete.json",
             out / "m1_calibration_complete.json", results / "m1_policy.json")
    if allow_synthetic:
        return {str(p): sha(p) for p in paths if p.exists()}
    if not all(p.exists() for p in paths):
        raise ValueError("Actual registered development/calibration completion evidence missing")
    reg, dev, cal, policy = map(read_json, paths)
    if any(obj.get("synthetic", False) for obj in (reg, dev, cal, policy)):
        raise ValueError("Synthetic evidence requires explicit --allow-synthetic")
    if reg.get("source_ids") != sources:
        raise ValueError("Development registration source order differs")
    identities = {r["source_id"]: str(r["preprocessing_id"]) for r in rows}
    if reg.get("preprocessing_ids") != [identities[sid] for sid in sources]:
        raise ValueError("Development registration preprocessing differs")
    if reg.get("calibration_or_development") != "m1_development" or reg.get("training_updates") != 0:
        raise ValueError("Development role or no-training boundary differs")
    bindings = reg.get("source_bindings")
    if not bindings:
        raise ValueError("Actual registration has no frozen source bindings")
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError("Registered source binding changed: " + path)
    verify_registered_inputs(reg, rows, sources)
    digest = sha(results / "m1_policy.json")
    for obj in (cal, dev):
        if obj.get("status") != "COMPLETE" or obj.get("policy_sha256") != digest or obj.get("training_updates") != 0:
            raise ValueError("Actual completion/policy binding differs")
    if dev.get("sources") != 100 or dev.get("rows") != len(rows):
        raise ValueError("Actual development completion population/count differs")
    if cal.get("development_read") is not False or cal.get("sources") != 1000:
        raise ValueError("Calibration boundary differs")
    if policy.get("development_read") is not False or policy.get("sources") != 1000 or policy.get("noise_seeds") != [4101, 4102, 4103]:
        raise ValueError("Frozen full calibration identity differs")
    actions = {(int(cell["N"]), cell["phy_family"], int(cell["snr_db"]), cell["method"] + "_policy"): cell["action"] for cell in policy["cells"]}
    for row in rows:
        key = m1_cell(row)
        if key in actions:
            action = actions[key]
            if (int(row["m"]), partial_count(row), row["order"]) != (int(action["m"]), int(action["q"]), action["order"]):
                raise ValueError("Development action differs from frozen full-calibration policy")
    timing_path, receipt_path = results / "m1_timing.csv", out / "m1_timing_complete.json"
    timed = read_csv(timing_path)
    receipt = read_json(receipt_path)
    if receipt.get("status") != "COMPLETE" or receipt.get("rows") != len(timed) or receipt.get("cache_used") is not False:
        raise ValueError("M1 independent timing completion differs")
    timed_groups = defaultdict(list)
    for row in timed:
        if boolean(row.get("cache_used")) is not False or boolean(row.get("online_cached_tokens_equal")) is not True or int(row.get("warmup_calls_per_policy", 0)) != 3:
            raise ValueError("M1 independent online/uncached/warmup timing differs")
        timed_groups[m1_cell(row)].append(row)
    if set(timed_groups) != set(actions):
        raise ValueError("M1 timing does not cover every frozen policy")
    for rr in timed_groups.values():
        if len(rr) != 10 or {int(r["source_index"]) for r in rr} != {0, 11, 22, 33, 44, 55, 66, 77, 88, 99}:
            raise ValueError("M1 timing fixed ten-source population differs")
    return {str(p): sha(p) for p in (*paths, timing_path, receipt_path)}


def analysis_m1(out, results, reports=None, allow_synthetic=False):
    input_path = results / "m1_per_frame.csv"
    rows = read_csv(input_path)
    groups, sources = validate_m1(rows)
    verify_mismatch_identity(rows, sources, required=not allow_synthetic)
    verified_inputs = verify_m1_registration(out, results, rows, sources, allow_synthetic)
    bootstrap = Bootstrap()
    summary, per_source = summarize(groups, sources, bootstrap, ("N", "phy_family", "snr_db", "method"))
    contrasts = m1_contrasts(groups, sources, bootstrap)
    high = high_snr_evidence(contrasts)
    outputs = []
    for name, table in (("summary", summary), ("source_means", per_source), ("paired_intervals", contrasts), ("high_snr_evidence", high)):
        path = results / f"m1_{name}.csv"
        write_csv(path, table)
        outputs.append(path)
    timing = timing_summary(results / "m1_timing.csv", results / "m1_timing_summary.csv", ("N", "phy_family", "snr_db", "method"))
    if timing:
        outputs.append(results / "m1_timing_summary.csv")
    outputs.extend(plot_m1(groups, summary, results))
    report = report_m1(results, groups, summary, contrasts, high, timing, sources)
    outputs.append(report)
    if reports is not None:
        copy_report(report, reports, "scale_causal_partial_residual_m1_20261002.md")
        outputs.append(reports / "scale_causal_partial_residual_m1_20261002.md")
    inputs = {str(p): sha(p) for p in (input_path, results / "m1_policy.json")}
    inputs.update(verified_inputs)
    for name in ("m1_calibration_summary.csv", "m1_screen_summary.csv", "m1_timing.csv", "m1_rate_curve.csv", "m1_reference_identity.json"):
        path = results / name
        if path.exists():
            inputs[str(path)] = sha(path)
    receipt = dict(status="COMPLETE", stage="m1_analysis", synthetic=bool(allow_synthetic), whole_study_complete=False, training_updates=0,
                   sources=len(sources), rows=len(rows), groups=len(groups), source_ids=sources,
                   exact_three_noise_pairs_verified=True, bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
                   statistical_zero_tolerance=ZERO_TOLERANCE,
                   historical_noise_pairing="source_and_nominal_seed_only", QPSK_energy="2N_verified",
                   QAM16_energy="actual_E_reported_not_strict_same_E", inputs=inputs,
                   outputs={str(p): sha(p) for p in outputs}, report=str(report))
    write_json(results / "m1_analysis_completion.json", receipt)
    write_json(out / "m1_analysis_completion.json", receipt)
    return receipt


PROJECTIONS = ("g8_c32", "g8_c8", "g6_c8", "g4_c32", "g4_c16", "g4_c8")
CONTROLS = ("UNGUIDED", "LIKELIHOOD", "STATIC", "VAR_GUIDED", "DIRECT")
M2_FIELDS = ("stage", "projection", "snr_db", "control", "N", "phy_family")


def m2_cell(row):
    snr = str(row["snr_db"])
    if snr != "clean":
        snr = str(int(float(snr)))
    return (str(row["stage"]), str(row["projection"]), snr, str(row["control"]),
            str(row.get("N", "")), str(row.get("phy_family", "ORACLE")) or "ORACLE")


def validate_m2_oracle(rows, expected_sources=100, snrs=SNRS, projections=PROJECTIONS):
    identities = {}
    for row in rows:
        ident = (int(row["source_index"]), str(row["preprocessing_id"]))
        if row["source_id"] in identities and identities[row["source_id"]] != ident:
            raise ValueError("M2 clean/noisy source/preprocessing identity differs")
        identities[row["source_id"]] = ident
    clean = [r for r in rows if str(r["snr_db"]) == "clean"]
    noisy = [r for r in rows if str(r["snr_db"]) != "clean"]
    clean_groups, sources = validate_groups(clean, m2_cell, expected_sources, (0,))
    noisy_groups, noisy_sources = validate_groups(noisy, m2_cell, expected_sources, DEV_SEEDS)
    if sources != noisy_sources:
        raise ValueError("M2 clean/noisy source population differs")
    groups = {**clean_groups, **noisy_groups}
    expected_cells = set()
    for projection in projections:
        for snr in ("clean", *(str(s) for s in snrs)):
            stage = ("full_noiseless" if projection == "g8_c32" else "reduced_noiseless") if snr == "clean" else (
                "full_noisy_infeasible_oracle" if projection == "g8_c32" else "reduced_noisy")
            for control in CONTROLS:
                expected_cells.add((stage, projection, snr, control, "", "ORACLE"))
    if set(groups) != expected_cells:
        raise ValueError("Incomplete or unexpected six-projection oracle ladder")
    observations = defaultdict(set)
    for row in rows:
        if boolean(row.get("oracle_side_information")) is not True:
            raise ValueError("Oracle side information is not explicitly labelled")
        if row.get("N", "") not in (None, "") or row.get("E", "") not in (None, ""):
            raise ValueError("Oracle row falsely carries a paid full-link budget")
        if boolean(row.get("exact_posterior_claim")) is not False:
            raise ValueError("M2 working likelihood falsely claims an exact posterior")
        if boolean(row.get("assumed_prefix_correct")) is not True or boolean(row.get("assumed_gain_correct")) is not True:
            raise ValueError("Oracle prefix/gain assumptions missing")
        g, channels = (int(part[1:]) for part in row["projection"].split("_"))
        if int(row["g"]) != g or int(row["channels"]) != channels:
            raise ValueError("Projection name/dimension metadata differs")
        if str(row["snr_db"]) != "clean":
            key = (row["source_id"], row["projection"], str(row["snr_db"]), int(row["noise_seed"]))
            value = row.get("measurement_y_sha256", "")
            if not value:
                raise ValueError("Noisy oracle row lacks actual measurement identity")
            observations[key].add(value)
    if any(len(values) != 1 for values in observations.values()):
        raise ValueError("M2 controls use different noisy measurements")
    return groups, sources


def m2_control_contrasts(groups, sources, bootstrap):
    output = []
    contexts = sorted({(key[0], key[1], key[2], key[4], key[5]) for key in groups})
    for stage, projection, snr, N, phy in contexts:
        A = (stage, projection, snr, "VAR_GUIDED", N, phy)
        if A not in groups:
            raise ValueError("M2 cell lacks VAR_GUIDED")
        for control in ("UNGUIDED", "LIKELIHOOD", "STATIC", "DIRECT"):
            B = (stage, projection, snr, control, N, phy)
            if B not in groups:
                raise ValueError("M2 cell lacks a registered control")
            context = dict(stage=stage, projection=projection, snr_db=snr, N=N, phy_family=phy,
                           method_A="VAR_GUIDED", method_B=control,
                           pairing="same_source_same_projected_measurement",
                           actual_paid_link=stage == "actual_link",
                           oracle_side_information=stage != "actual_link")
            output.extend(paired(groups[A], groups[B], sources, bootstrap, context))
    return output


def m2_ladder_contrasts(groups, sources, bootstrap):
    output = []
    for projection in PROJECTIONS[1:]:
        for control in CONTROLS:
            A = ("reduced_noiseless", projection, "clean", control, "", "ORACLE")
            B = ("full_noiseless", "g8_c32", "clean", control, "", "ORACLE")
            if A in groups and B in groups:
                output.extend(paired(groups[A], groups[B], sources, bootstrap,
                    dict(stage="compression_damage_noiseless", projection=projection, reference_projection="g8_c32",
                         snr_db="clean", method_A=control, method_B=control,
                         pairing="same_source_different_projection_noiseless", actual_paid_link=False)))
    return output


def verify_m2_evidence(out, results, rows, sources, allow_synthetic=False):
    names = (out / "m2_evaluation_registration.json", out / "m2_evaluation_complete.json",
             out / "m2_calibration_complete.json", results / "m2_oracle_policy.json", results / "m2_gate.json",
             results / "m2_resource_ledger.csv")
    if allow_synthetic:
        return {str(p): sha(p) for p in names if p.exists()}
    if not all(p.exists() for p in names):
        raise ValueError("Actual registered M2 evidence missing")
    reg, dev, cal, policy, gate = map(read_json, names[:5])
    if any(obj.get("synthetic", False) for obj in (reg, dev, cal, policy, gate)):
        raise ValueError("Synthetic M2 evidence requires explicit --allow-synthetic")
    identities = {r["source_id"]: str(r["preprocessing_id"]) for r in rows}
    if reg.get("source_ids") != sources or reg.get("preprocessing_ids") != [identities[sid] for sid in sources]:
        raise ValueError("M2 registered development source/preprocessing differs")
    if reg.get("calibration_or_development") != "m2_evaluation" or reg.get("training_updates") != 0:
        raise ValueError("M2 development role/no-training boundary differs")
    for obj in (reg, policy):
        if not obj.get("source_bindings"):
            raise ValueError("M2 frozen source bindings missing")
        for path, digest in obj["source_bindings"].items():
            if sha(path) != digest:
                raise ValueError("M2 source binding changed: " + path)
    verify_registered_inputs(reg, rows, sources)
    for obj, status in ((dev, "M2_EVALUATION_COMPLETE"), (cal, "M2_CALIBRATION_COMPLETE")):
        if obj.get("status") != status or obj.get("training_updates") != 0 or obj.get("synthetic") is not False:
            raise ValueError("M2 actual completion differs")
        for path, digest in obj.get("files", {}).items():
            if sha(path) != digest:
                raise ValueError("M2 evidence hash changed: " + path)
    if dev.get("metric_rows") != len(rows) or dev.get("clean_rows") != 3000 or dev.get("noisy_rows") != 45000:
        raise ValueError("M2 oracle completion counts differ")
    if gate.get("policy_sha256") != sha(out / "m2/calibration/policy.json") or gate.get("per_frame_sha256") != sha(out / "m2/calibration/gate_per_frame.csv"):
        raise ValueError("M2 gate policy/full-calibration binding differs")
    if gate.get("calibration_sources") != 1000 or gate.get("noise_seeds") != [4101, 4102, 4103] or gate.get("snr_db") != 7:
        raise ValueError("M2 gate calibration population differs")
    if policy.get("selection_sources") != 200 or policy.get("selection_noise_seeds") != [4101, 4102, 4103] or policy.get("selected_snr_db") != 7:
        raise ValueError("M2 frozen lambda selection identity differs")
    for row in rows:
        if row["control"] in ("VAR_GUIDED", "STATIC"):
            branch = "var" if row["control"] == "VAR_GUIDED" else "static"
            if number(row["lambda"], True) != float(policy[branch][row["projection"]]["selected_lambda"]):
                raise ValueError("Development guidance lambda differs from frozen calibration")
    m1 = read_json(out / "m1_complete.json")
    if m1.get("status") != "M1_COMPLETE" or m1.get("synthetic") is not False:
        raise ValueError("M2 started without actual completed M1")
    if dev.get("m1_completion_sha256") != sha(out / "m1_complete.json") or cal.get("m1_completion_sha256") != sha(out / "m1_complete.json"):
        raise ValueError("M1-before-M2 completion binding differs")
    return {str(p): sha(p) for p in names}


def audit_m2_gate(out, results, allow_synthetic=False):
    gate = read_json(results / "m2_gate.json")
    path = out / "m2/calibration/gate_per_frame.csv"
    if allow_synthetic and not path.exists():
        return dict(independently_recomputed=False, synthetic=True, status=gate["status"])
    rows = read_csv(path)
    groups, sources = validate_groups(rows, lambda r: (r["projection"], r["control"]), 1000, (4101, 4102, 4103))
    eligible = gate.get("eligible_projections")
    ledger = read_csv(results / "m2_resource_ledger.csv")
    feasible = {r["projection"] if "projection" in r else f"g{r['g']}_c{r['channels']}" for r in ledger if boolean(r.get("feasible")) is True}
    if set(eligible) != feasible or set(groups) != {(p, ctrl) for p in eligible for ctrl in ("VAR_GUIDED", "UNGUIDED")}:
        raise ValueError("Gate does not cover exactly budget-feasible projections")
    boot = Bootstrap()
    decisions = []
    evidence = []
    for projection in sorted(eligible):
        contrasts = paired(groups[projection, "VAR_GUIDED"], groups[projection, "UNGUIDED"], sources, boot,
                           dict(projection=projection, method_A="VAR_GUIDED", method_B="UNGUIDED", snr_db=7))
        evidence.extend(contrasts)
        by_metric = {row["metric"]: row for row in contrasts}
        ps, lp, di = (by_metric[name] for name in ("psnr_db", "lpips_alex", "dino_cosine"))
        passed = ps["delta_mean"] >= .5 and ps["ci_low"] > 0 and lp["delta_mean"] <= -.02 and lp["ci_high"] < 0 and di["delta_mean"] >= -.01
        old = next(r for r in gate["decisions"] if r["projection"] == projection)
        if bool(old["passed"]) != bool(passed):
            raise ValueError("Independent full-calibration gate decision differs")
        for met, recomputed, field in (("psnr_db", ps, "delta_psnr"), ("lpips_alex", lp, "delta_lpips"), ("dino_cosine", di, "delta_dino")):
            reported = old[field]
            for a_name, b_name in (("delta_mean", "mean"), ("ci_low", "lower95"), ("ci_high", "upper95")):
                if abs(recomputed[a_name] - float(reported[b_name])) > 1e-9:
                    raise ValueError("Independent gate interval differs: " + projection + "/" + met)
        decisions.append(dict(projection=projection, passed=bool(passed), delta_psnr=ps, delta_lpips=lp, delta_dino=di))
    passed = any(row["passed"] for row in decisions)
    if bool(gate["passed"]) != passed or gate["status"] != ("PASSED" if passed else "SKIPPED_GATE_NOT_MET"):
        raise ValueError("Overall gate status differs")
    write_csv(results / "m2_gate_independent_paired.csv", evidence)
    audit = dict(independently_recomputed=True, synthetic=bool(allow_synthetic), status=gate["status"],
                 sources=1000, noise_seeds=[4101, 4102, 4103], bootstrap_replicates=REPLICATES,
                 bootstrap_seed=SEED, statistical_zero_tolerance=ZERO_TOLERANCE,
                 selection_and_gate_calibration_overlap=True, decisions=decisions)
    write_json(results / "m2_gate_independent_audit.json", audit)
    return audit


def m2_P_contrasts(groups, source_ids, results, bootstrap):
    path = results / "m1_per_frame.csv"
    if not path.exists():
        return []
    references = defaultdict(list)
    for row in read_csv(path):
        if row["method"] in ("P512", "P1024"):
            references[int(row["N"]), row["phy_family"], str(int(float(row["snr_db"])))].append(row)
    if ordered_sources([r for rr in references.values() for r in rr]) != source_ids:
        raise ValueError("M2/P source identities differ")
    wrong_mapping = verify_mismatch_identity([r for rr in references.values() for r in rr], source_ids, required=True)
    ledger = read_csv(results / "m2_resource_ledger.csv")
    feasible = {(str(r.get("projection", f"g{r['g']}_c{r['channels']}")), int(r["N"]), r.get("phy_family", r.get("phy", ""))): boolean(r["feasible"]) for r in ledger}
    output = []
    for key, rr in sorted(groups.items()):
        stage, projection, snr, control, N, phy = key
        if snr == "clean" or control != "VAR_GUIDED":
            continue
        targets = ((int(N), phy),) if stage == "actual_link" else ((512, "QPSK"), (512, "16QAM"), (1024, "QPSK"), (1024, "16QAM"))
        for budget, family in targets:
            ref = references.get((budget, family, snr))
            if ref is None:
                raise ValueError("Missing frozen P reference")
            context = dict(stage=stage, projection=projection, snr_db=snr, N=budget, phy_family=family,
                           method_A="VAR_GUIDED", method_B=f"P{budget}",
                           actual_paid_link=stage == "actual_link", budget_feasible=feasible.get((projection, budget, family), False),
                           fair_paid_link_comparison=stage == "actual_link",
                           pairing="same_source_same_nominal_seed_historical_distinct_noise",
                           oracle_side_information=stage != "actual_link")
            # Source/preprocessing exact identity is required, even when physical noise differs.
            identity = {r["source_id"]: r["preprocessing_id"] for r in ref}
            if any(identity.get(r["source_id"]) != r["preprocessing_id"] for r in rr):
                raise ValueError("M2/P preprocessing identity differs")
            if any(wrong_mapping.get(r["source_id"]) != r.get("mismatch_source_id") for r in rr):
                raise ValueError("M2/P wrong-image reference identity differs")
            output.extend(paired(rr, ref, source_ids, bootstrap, context))
    return output


def verify_completion(path, status):
    obj = read_json(path)
    if obj.get("status") != status or obj.get("synthetic") is not False or obj.get("training_updates") != 0:
        raise ValueError("Actual completion status/no-training/synthetic boundary differs: " + str(path))
    if not obj.get("files"):
        raise ValueError("Completion lacks hashed evidence: " + str(path))
    for name, digest in obj["files"].items():
        if sha(name) != digest:
            raise ValueError("Completion evidence hash changed: " + name)
    return obj


def validate_m2_actual(rows, policy, ledger, sources, expected_sources=100):
    selected = [r for r in policy.get("choices", []) if r["status"] == "SELECTED"]
    contexts = {(str(r["projection"]), str(int(r["N"])), r["phy_family"], str(int(r["snr_db"]))) for r in selected}
    if len(contexts) != len(selected):
        raise ValueError("Duplicate actual-policy selected context")
    if not rows and not contexts:
        return {}
    groups, actual_sources = validate_groups(rows, m2_cell, expected_sources)
    if actual_sources != sources:
        raise ValueError("Actual-link source population differs")
    expected = {("actual_link", p, s, ctrl, n, phy) for p, n, phy, s in contexts for ctrl in CONTROLS}
    if set(groups) != expected:
        raise ValueError("Actual-link rows do not equal frozen selected contexts and five controls")
    feasible = {(str(r.get("projection", f"g{r['g']}_c{r['channels']}")), str(int(r["N"])), r["phy_family"]): boolean(r["feasible"]) for r in ledger}
    frames = defaultdict(list)
    for row in rows:
        N = int(row["N"])
        projection = row["projection"]
        if boolean(row.get("oracle_side_information")) is not False or boolean(row.get("offline_truth_never_controls_rx")) is not True:
            raise ValueError("Actual-link oracle/truth boundary differs")
        if not feasible.get((projection, str(N), row["phy_family"])):
            raise ValueError("Actual-link projection is not budget-feasible")
        g, channels = (int(part[1:]) for part in projection.split("_"))
        if (int(row["g"]), int(row["channels"])) != (g, channels):
            raise ValueError("Actual-link projection dimension differs")
        parts = [int(row[name]) for name in ("N_header", "N_gain", "N_digital", "N_measurement", "N_padding")]
        if sum(parts) != N or parts[0] != 68 or parts[1] != 32 or parts[3] != g * g * channels // 2 or parts[4] != 1:
            raise ValueError("Paid header/gain/digital/measurement/padding resource ledger differs")
        width = 2 if row["phy_family"] == "QPSK" else 4
        rate = 382 / (parts[2] * width)
        if rate > .9 or abs(number(row.get("effective_prefix_rate"), True) - rate) > 1e-12:
            raise ValueError("Actual-link effective prefix rate differs")
        energy = number(row["E"], True)
        if abs(energy - sum(number(row[k], True) for k in ("E_header", "E_gain", "E_digital", "E_analog"))) > 1e-6:
            raise ValueError("Actual segment energies do not sum to frame energy")
        if row["phy_family"] == "QPSK" and abs(energy - 2 * N) > max(.001, N * 2e-6):
            raise ValueError("Actual QPSK energy differs from 2N")
        if boolean(row.get("padding_ignored")) is not True:
            raise ValueError("Actual padding is not explicitly ignored by RX")
        header = boolean(row.get("header_ok"))
        if header is None:
            raise ValueError("Actual header event missing")
        if not header and boolean(row.get("latent_valid")) is not False:
            raise ValueError("Header erasure falsely has a valid latent")
        if boolean(row.get("analog_used")) is True and not all(boolean(row.get(k)) is True for k in ("header_ok", "prefix_crc_ok", "gain_crc_ok", "gain_fields_valid")):
            raise ValueError("Analog used despite failed paid packet/gain")
        frames[row["source_id"], int(row["noise_seed"]), N, row["phy_family"], int(float(row["snr_db"]))].append(row)
    for rr in frames.values():
        for key in ("waveform_sha256", "observation_sha256", "E", "header_ok", "prefix_crc_ok", "gain_crc_ok", "gain_fields_valid", "analog_used"):
            if not rr[0].get(key) and key in ("waveform_sha256", "observation_sha256"):
                raise ValueError("Actual transmitted/received waveform identity missing")
            if len({str(r.get(key)) for r in rr}) != 1:
                raise ValueError("Actual controls do not share waveform/observation/event: " + key)
        if boolean(rr[0]["analog_used"]) is not True:
            for name in ("psnr_db", "lpips_alex", "dino_cosine", "dino_mismatched", "latent_valid", "latent_sq_err_final"):
                if len({str(r.get(name)) for r in rr}) != 1:
                    raise ValueError("Failed actual packet did not reduce all controls to the same RX output")
    return groups


def verify_m2_actual_evidence(out, results, gate, sources, oracle_rows, allow_synthetic=False):
    path = results / "m2_actual_per_frame.csv"
    if gate["passed"] and not path.exists():
        raise ValueError("Passed gate requires actual-link results before M2 analysis completion")
    rows = read_csv(path) if path.exists() else []
    if not gate["passed"] and rows:
        raise ValueError("Actual-link rows exist after a failed calibration gate")
    policy_path = results / "m2_actual_policy.json"
    receipt_path = out / "m2_actual_complete.json"
    if allow_synthetic and not policy_path.exists():
        if rows:
            raise ValueError("Synthetic actual results still require a frozen selected policy")
        return {}, "SKIPPED_GATE_NOT_MET", {}
    policy = read_json(policy_path)
    if policy.get("development_read") is not False:
        raise ValueError("Actual policy was not frozen before development")
    if gate["passed"]:
        passed_projections = {r["projection"] for r in gate["decisions"] if r["passed"]}
        for choice in policy.get("choices", []):
            if choice["status"] == "SELECTED" and choice["projection"] not in passed_projections:
                raise ValueError("Actual policy selected a projection that failed the calibration gate")
    ledger = read_csv(results / "m2_resource_ledger.csv")
    groups = validate_m2_actual(rows, policy, ledger, sources, len(sources))
    bindings = {str(p): sha(p) for p in (path, policy_path) if p.exists()}
    if allow_synthetic:
        return groups, "ACTUAL_LINK_COMPLETE" if gate["passed"] else "SKIPPED_GATE_NOT_MET", bindings
    receipt = verify_completion(receipt_path, "M2_ACTUAL_COMPLETE")
    bindings[str(receipt_path)] = sha(receipt_path)
    if receipt.get("metric_rows") != len(rows):
        raise ValueError("Actual completion row count differs")
    if sha(policy_path) != sha(out / "m2/calibration/actual_policy.json"):
        raise ValueError("Actual policy mirror differs")
    bindings[str(out / "m2/calibration/actual_policy.json")] = sha(out / "m2/calibration/actual_policy.json")
    if gate["passed"]:
        if receipt.get("branch") != "ACTUAL_LINK_EVALUATED" or receipt.get("selected_contexts") * 100 * 3 * 5 != len(rows):
            raise ValueError("Actual completion selected-context count differs")
        if any(receipt.get(k) is not False for k in ("ideal_prefix_assumption", "ideal_gain_assumption")) or receipt.get("all_failures_in_quality") is not True:
            raise ValueError("Actual paid-link failure/side-information completion differs")
        if policy.get("calibration_sources") != 200 or policy.get("calibration_noise_seeds") != [4101, 4102, 4103]:
            raise ValueError("Actual calibration selection population differs")
        reg_path = out / "m2_actual_registration.json"
        reg = read_json(reg_path)
        if reg.get("calibration_or_development") != "m2_actual" or reg.get("source_ids") != sources or reg.get("training_updates") != 0:
            raise ValueError("Actual development registration differs")
        if rows:
            verify_registered_inputs(reg, rows, sources)
            true_ids = {r["source_id"]: r["preprocessing_id"] for r in oracle_rows}
            wrong_ids = verify_mismatch_identity(oracle_rows, sources, required=True)
            if any(true_ids[r["source_id"]] != r["preprocessing_id"] or wrong_ids[r["source_id"]] != r.get("mismatch_source_id") for r in rows):
                raise ValueError("Actual/oracle pixel or wrong-image identity differs")
            oracle_policy = read_json(results / "m2_oracle_policy.json")
            for row in rows:
                if boolean(row.get("true_class_sent")) is not False:
                    raise ValueError("Actual header silently sends source-class side information")
                expected_lambda = (oracle_policy["var" if row["control"] == "VAR_GUIDED" else "static"][row["projection"]]["selected_lambda"]
                                   if row["control"] in ("VAR_GUIDED", "STATIC") else 0.)
                if number(row["lambda"], True) != float(expected_lambda):
                    raise ValueError("Actual guidance lambda differs from frozen oracle calibration")
        for p, digest in reg.get("source_bindings", {}).items():
            if sha(p) != digest:
                raise ValueError("Actual source binding changed")
        bindings[str(reg_path)] = sha(reg_path)
        status = "ACTUAL_LINK_COMPLETE"
    else:
        if receipt.get("branch") != "SKIPPED_GATE_NOT_MET" or receipt.get("actual_development_read") is not False or policy.get("choices") != []:
            raise ValueError("Failed gate actual skip evidence differs")
        status = "SKIPPED_GATE_NOT_MET"
    return groups, status, bindings


def m2_timing(out, results, allow_synthetic=False):
    path = results / "m2_timing.csv"
    if allow_synthetic and not path.exists():
        return [], {}
    rows = read_csv(path)
    completion = out / "m2_timing_complete.json"
    bindings = {str(path): sha(path)}
    if not allow_synthetic:
        obj = verify_completion(completion, "M2_TIMING_COMPLETE")
        if obj.get("timing_rows") != len(rows) or obj.get("sources") != 10 or obj.get("repeats") != 2 or obj.get("warmups_per_source_control_context") != 1:
            raise ValueError("M2 timing completion design/count differs")
        bindings[str(completion)] = sha(completion)
    fields = ("scope", "N", "phy_family", "projection", "snr_db", "control")
    groups = defaultdict(list)
    for row in rows:
        if boolean(row.get("no_image_output_cache")) is not True:
            raise ValueError("M2 timing image-output cache boundary differs")
        if boolean(row.get("deployment_endpoint")) != (row["scope"] == "actual_paid_frame"):
            raise ValueError("Oracle timing mislabeled as deployment endpoint")
        groups[tuple(row.get(k, "") for k in fields)].append(row)
    output = []
    for key, rr in sorted(groups.items()):
        pairs = {(r["source_id"], int(r["repeat"])) for r in rr}
        ids = {r["source_id"] for r in rr}
        if len(ids) != 10 or len(rr) != 20 or pairs != {(s, rep) for s in ids for rep in (0, 1)}:
            raise ValueError("M2 timing missing ten-source paired repeats")
        for name in ("tx_ms", "rx_ms", "total_ms"):
            values = [np.mean([number(r[name], True) for r in rr if r["source_id"] == sid]) for sid in sorted(ids)]
            output.append(dict(**dict(zip(fields, key)), metric=name, mean=float(np.mean(values)),
                               median=float(np.median(values)), n_sources=10, repeats_per_source=2,
                               n_timed=20, no_image_output_cache=True, deployment_endpoint=key[0] == "actual_paid_frame"))
    write_csv(results / "m2_timing_summary.csv", output)
    return output, bindings


def plot_m2(summary, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = results / "figures"
    folder.mkdir(exist_ok=True)
    colors = {"UNGUIDED": "#707070", "LIKELIHOOD": "#b35e16", "STATIC": "#a45aa0", "VAR_GUIDED": "#1769aa", "DIRECT": "#2e8b57"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, snr in enumerate(("clean", "7")):
        for j, metric_name in enumerate(("psnr_db", "lpips_alex", "dino_specific")):
            for control, color in colors.items():
                points = []
                for row in summary:
                    if row["N"] == "" and (row["snr_db"], row["control"], row["metric"]) == (snr, control, metric_name) and row["mean"] is not None:
                        g, channels = (int(part[1:]) for part in row["projection"].split("_"))
                        points.append((g * g * channels, row["mean"], row["projection"]))
                if points:
                    axes[i, j].scatter([p[0] for p in points], [p[1] for p in points], color=color, label=control, s=25)
                    if control == "VAR_GUIDED":
                        for x, y, name in points:
                            axes[i, j].annotate(name, (x, y), fontsize=6, xytext=(3, 4), textcoords="offset points")
            axes[i, j].set_title(("Noiseless oracle" if snr == "clean" else "7dB oracle with ideal correct prefix/gain") + " · " + metric_name)
            axes[i, j].set_xlabel("Projected real measurement components (not total paid uses)")
            axes[i, j].grid(alpha=.2)
        axes[i, 0].legend(fontsize=7)
    fig.suptitle("M2 oracle ladder: full g8_c32 is budget-infeasible; six measured projections, no interpolation")
    p = folder / "m2_oracle_damage_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created = [p]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, projection in enumerate(("g8_c8", "g4_c8")):
        for j, name in enumerate(("psnr_db", "lpips_alex", "dino_specific")):
            for control, color in colors.items():
                rr = [r for r in summary if r["N"] == "" and (r["projection"], r["control"], r["metric"]) == (projection, control, name) and r["snr_db"] != "clean" and r["mean"] is not None]
                rr.sort(key=lambda r: int(r["snr_db"]))
                if rr:
                    axes[i, j].plot([int(r["snr_db"]) for r in rr], [r["mean"] for r in rr], "o-", color=color, label=control)
            axes[i, j].set_title(projection + " · " + name)
            axes[i, j].set_xlabel("SNR (dB)")
            axes[i, j].grid(alpha=.2)
        axes[i, 0].legend(fontsize=7)
    fig.suptitle("M2 noisy reduced-measurement oracle; ideal correct prefix/gain, no actual-link claim")
    p = folder / "m2_oracle_quality_vs_SNR.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    created.append(p)
    if any(r["stage"] == "actual_link" for r in summary):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for N in ("512", "1024"):
            for i, phy in enumerate(("QPSK", "16QAM")):
                for j, metric_name in enumerate(("psnr_db", "lpips_alex", "dino_specific")):
                    for control, color in colors.items():
                        rr = [r for r in summary if (r["stage"], r["N"], r["phy_family"], r["control"], r["metric"]) == ("actual_link", N, phy, control, metric_name) and r["mean"] is not None]
                        rr.sort(key=lambda r: int(r["snr_db"]))
                        if rr:
                            axes[i, j].plot([int(r["snr_db"]) for r in rr], [r["mean"] for r in rr], "o-" if N == "512" else "s--", color=color, label="N" + N + "/" + control)
                    axes[i, j].set_title(phy + " paid actual frame · " + metric_name)
                    axes[i, j].set_xlabel("SNR (dB)")
                    axes[i, j].grid(alpha=.2)
                axes[i, 0].legend(fontsize=6)
        fig.suptitle("M2 paid header/prefix/gain/measurement/padding; all failed frames included; 16QAM actual E varies")
        p = folder / "m2_actual_quality_vs_SNR.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(p)
    return created


def report_m2(results, summary, contrasts, P_contrasts, gate, actual_status, timing=()):
    lines = ["# 方法二：模拟残差观测与冻结先验引导", "", "## 四个问题的实际结果", "",
             "### 1. 较多无噪声残差信息是否有用", "",
             "g8_c32仍是原16×16 latent的area空间降维观测，并非完整F。以下是其免费正确prefix、无噪声观测的VAR_GUIDED相对各接收控制的真实差值。", "",
             "| 控制 | 指标 | 差值 [95% CI] | 源数 |", "|---|---|---|---:|"]
    for row in contrasts:
        if row["stage"] == "full_noiseless" and row["metric"] in ("psnr_db", "lpips_alex", "dino_cosine", "dino_specific", "F_sq_error_valid"):
            lines.append(f"| {row['method_B']} | {row['metric']} | {formatted(row['delta_mean'])} [{formatted(row['ci_low'])}, {formatted(row['ci_high'])}] | {row['n_sources_paired']} |")
    lines += ["", "### 2. 降维和加噪后是否仍有来源一致的收益", "",
              "下面固定7dB，VAR_GUIDED减去同一投影/同一measurement的UNGUIDED。压缩损害另外见`m2_ladder_paired.csv`的无噪声同源比较。", "",
              "| 投影 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | ΔDINO specificity [95% CI] |", "|---|---|---|---|"]
    for projection in PROJECTIONS:
        rr = {r["metric"]: r for r in contrasts if r["projection"] == projection and r["snr_db"] == "7" and r["method_B"] == "UNGUIDED" and r["N"] == ""}
        if rr:
            vals = [rr[name] for name in ("psnr_db", "lpips_alex", "dino_specific")]
            chunks = [f"{formatted(r['delta_mean'])} [{formatted(r['ci_low'])}, {formatted(r['ci_high'])}]" for r in vals]
            lines.append(f"| {projection} | " + " | ".join(chunks) + " |")
    lines += ["", "### 3. 实际链路是否获准并完成", "",
              f"全1000 calibration、7dB、三noise的登记gate状态：**{gate['status']}**。实际链路分支状态：**{actual_status}**。", "",
              "gate在独立analysis中按原数据重算。固定lambda来自前200 calibration，gate包含这200源，因此gate区间是校准内描述性检查，不是独立泛化验证。development未用于调lambda、选投影或改变gate。若gate失败，完整oracle ladder仍发布，实际链路跳过；oracle质量不升级为付费链路结论。", "",
              "实际投影在前200 calibration按各N/PHY/SNR选择。`selection_fallback=True`表示没有同时满足相对UNGUIDED的PSNR−0.25dB及DINO−0.01约束的候选；报告保留该分支，不把它称为满足质量保护的策略。", "",
              "### 4. 收益是否特属于VAR先验", "",
              "下面列出7dB同一观测下VAR_GUIDED对LIKELIHOOD、STATIC、DIRECT的LPIPS差区间。VAR仅胜UNGUIDED不足以证明先验机制；需同时检查这些控制以及PSNR、DINO specificity、F误差。", "",
              "| 投影 | 对照 | ΔLPIPS [95% CI] | 区间支持LPIPS下降 |", "|---|---|---|---|"]
    for row in contrasts:
        if row["snr_db"] == "7" and row["N"] == "" and row["metric"] == "lpips_alex" and row["method_B"] in ("LIKELIHOOD", "STATIC", "DIRECT"):
            yes = "是" if row["ci_high"] is not None and row["ci_high"] < 0 else "否"
            lines.append(f"| {row['projection']} | {row['method_B']} | {formatted(row['delta_mean'])} [{formatted(row['ci_low'])}, {formatted(row['ci_high'])}] | {yes} |")
    actual_pairs = [r for r in contrasts + P_contrasts if r.get("stage") == "actual_link" and r["metric"] in ("psnr_db", "lpips_alex", "dino_specific")]
    if actual_pairs:
        lines += ["", "## 实际付费链路的配对结果", "", "各接收控制共享同一实际波形、观测和事件；历史P只共享源与名义seed。以下每个区间均来自100源内平均三noise后的配对差。", "", "| N | PHY | SNR | 投影 | 对照 | 指标 | 差值 [95% CI] |", "|---:|---|---:|---|---|---|---|"]
        for r in actual_pairs:
            lines.append(f"| {r['N']} | {r['phy_family']} | {r['snr_db']} | {r['projection']} | {r['method_B']} | {r['metric']} | {formatted(r['delta_mean'])} [{formatted(r['ci_low'])}, {formatted(r['ci_high'])}] |")
        lines += ["", "[实际链路质量曲线](figures/m2_actual_quality_vs_SNR.png)。16QAM质量比较按相同N和调制，实际E在汇总表单独列出，不能称严格同E。"]
    if timing:
        lines += ["", "## 独立计时", "", "每context/control先warmup，再对10个固定源各计时2次；先平均同源两次再平均10源。oracle endpoint假设正确prefix/gain，其速度不能当部署链路速度。实际endpoint包含付费帧收发和真实失败路径；固定算子缓存允许，图像输出缓存禁用。", "", "[分范围TX/RX/总时间](m2_timing_summary.csv)。"]
    lines += ["", "## 预算和oracle边界", "",
              "六投影的noiseless/noisy oracle均假设正确数字m4前缀与正确gain；noisy观测在相应归一化波形上真实加AWGN。g8_c32的完整观测无法放入N512/N1024付费链路。其他投影的可行尺寸由`m2_resource_ledger.csv`列出，尺寸可行仍不等于真实header/CRC/gain可靠。", "",
              "与冻结P512/P1024的源配对差值见`m2_P_paired.csv`。oracle行明确`fair_paid_link_comparison=False`，仅作观测诊断；实际链路才按相应N对比。P的历史namespace不同，名义seed相同不代表同一物理noise。共同错图DINO reference采用历史derangement。", "",
              "似然使用登记的Gaussian working/composite approximation、未观测残差统计及固定前向算子。没有声明精确独立后验、后验采样、全球最优或无推理成本。模型参数训练更新为0；calibration统计/PCA和输入变量迭代属于额外数据/算力使用。", "",
              "## 统计、分母与证据", "",
              "清洁观测每源1次seed0；noisy每源三noise平均，再对100源做10,000次共享bootstrap，seed20261002。所有失败图像留在质量分母；实际有效latent的F误差和双方共同有效F配对另注明分母。", "",
              "- [汇总](m2_summary.csv)、[逐源均值](m2_source_means.csv)、[控制配对](m2_paired_intervals.csv)、[压缩损害](m2_ladder_paired.csv)、[P比较](m2_P_paired.csv)。",
              "- [全量校准gate独立复算](m2_gate_independent_audit.json)、[gate区间](m2_gate_independent_paired.csv)。",
              "- [维度与质量](figures/m2_oracle_damage_ladder.png)、[有噪观测曲线](figures/m2_oracle_quality_vs_SNR.png)。",
              "- 固定样例source_index=0/25/50/75、clean/4/13dB、seed0/2001，目录`m2_examples/`。", "",
              "本报告只确认方法二分析阶段。整个研究和推送状态由supervisor/publication证据确认。", ""]
    p = results / "m2_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def analysis_m2(out, results, reports=None, allow_synthetic=False):
    path = results / "m2_oracle_per_frame.csv"
    rows = read_csv(path)
    groups, sources = validate_m2_oracle(rows)
    verify_mismatch_identity(rows, sources, required=not allow_synthetic)
    bindings = verify_m2_evidence(out, results, rows, sources, allow_synthetic)
    gate_audit = audit_m2_gate(out, results, allow_synthetic)
    gate = read_json(results / "m2_gate.json")
    actual_groups, actual_status, actual_bindings = verify_m2_actual_evidence(out, results, gate, sources, rows, allow_synthetic)
    groups.update(actual_groups)
    bindings.update(actual_bindings)
    timing, timing_bindings = m2_timing(out, results, allow_synthetic)
    bindings.update(timing_bindings)
    boot = Bootstrap()
    summary, means = summarize(groups, sources, boot, M2_FIELDS)
    contrasts = m2_control_contrasts(groups, sources, boot)
    ladder = m2_ladder_contrasts(groups, sources, boot)
    P_contrasts = m2_P_contrasts(groups, sources, results, boot)
    outputs = []
    for name, data in (("summary", summary), ("source_means", means), ("paired_intervals", contrasts), ("ladder_paired", ladder), ("P_paired", P_contrasts)):
        p = results / f"m2_{name}.csv"
        write_csv(p, data)
        outputs.append(p)
    outputs.extend(plot_m2(summary, results))
    report = report_m2(results, summary, contrasts, P_contrasts, gate, actual_status, timing)
    outputs.append(report)
    if timing:
        outputs.append(results / "m2_timing_summary.csv")
    for name in ("m2_gate_independent_audit.json", "m2_gate_independent_paired.csv"):
        p = results / name
        if p.exists():
            outputs.append(p)
    if reports is not None:
        copy_report(report, reports, "scale_causal_partial_residual_m2_20261002.md")
        outputs.append(reports / "scale_causal_partial_residual_m2_20261002.md")
    bindings[str(path)] = sha(path)
    for p in (results / "m1_per_frame.csv", out / "m2/calibration/gate_per_frame.csv", out / "m2/calibration/policy.json"):
        if p.exists():
            bindings[str(p)] = sha(p)
    receipt = dict(status="COMPLETE", stage="m2_analysis", synthetic=bool(allow_synthetic), whole_study_complete=False,
                   training_updates=0, sources=len(sources), oracle_rows=len(rows), clean_rows=sum(r["snr_db"] == "clean" for r in rows),
                   noisy_rows=sum(r["snr_db"] != "clean" for r in rows), oracle_measurements_verified=True,
                   gate_status=gate["status"], gate_independently_recomputed=gate_audit["independently_recomputed"],
                   actual_link_status=actual_status, bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
                   statistical_zero_tolerance=ZERO_TOLERANCE, inputs=bindings,
                   outputs={str(p): sha(p) for p in outputs}, report=str(report))
    write_json(results / "m2_analysis_completion.json", receipt)
    write_json(out / "m2_analysis_completion.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("m1", "m2", "all"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true", help="Mark all receipts synthetic; publication must reject these")
    args = parser.parse_args()
    if args.method in ("m1", "all"):
        analysis_m1(args.out_dir, args.results_dir, args.reports_dir, args.allow_synthetic)
    if args.method in ("m2", "all"):
        analysis_m2(args.out_dir, args.results_dir, args.reports_dir, args.allow_synthetic)


if __name__ == "__main__":
    main()
