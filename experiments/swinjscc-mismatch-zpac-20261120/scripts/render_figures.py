#!/usr/bin/env python3
"""Render publication-quality figures (PDF + PNG) with shared non-overlapping legends for Direction 2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.hygiene import configure_runtime_env, verify_no_links


plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.3,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 240,
    }
)


STYLE_MAP = {
    "Frozen": ("#7f8c8d", "--", "o", "Original SwinJSCC"),
    "AvgSNR": ("#95a5a6", ":", "s", "Avg-SNR Fallback (cT=7dB)"),
    "B0": ("#e67e22", "-.", "^", "B0 (Single-Cond cR)"),
    "B1": ("#2980b9", "-.", "D", "B1 (Decoder Dual [cT,cR])"),
    "B2": ("#8e44ad", ":", "v", "B2 (Front+Dec Dual)"),
    "P_Raw": ("#16a085", "--", "P", "P-Raw ([cR,cT,0] Concat)"),
    "P_ZPAC": ("#e74c3c", "-", "*", "P-ZPAC (Ours, Front)"),
    "P_ZPAC_Hier": ("#900c3f", "-", "X", "P-ZPAC-Hier (Ours, Full)"),
}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_fig1_latent_diagnosis(stage0_json: Path, out_dir: Path) -> None:
    data = json.loads(stage0_json.read_text(encoding="utf-8"))
    ch_vars = data["latent_diagnosis"]["sorted_channel_variances"]
    shifts = data["latent_diagnosis"]["latent_shift_summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.7))

    colors = {
        "cT_1dB": "#1f77b4",
        "cT_4dB": "#2ca02c",
        "cT_7dB": "#ff7f0e",
        "cT_10dB": "#9467bd",
        "cT_13dB": "#d62728",
    }
    for key, label in [
        ("cT_1dB", "cT = 1 dB (Conservative)"),
        ("cT_4dB", "cT = 4 dB"),
        ("cT_7dB", "cT = 7 dB"),
        ("cT_10dB", "cT = 10 dB"),
        ("cT_13dB", "cT = 13 dB (Aggressive)"),
    ]:
        if key in ch_vars:
            vals = np.asarray(ch_vars[key], dtype=np.float64)
            ax1.plot(np.arange(1, len(vals) + 1), vals, label=label, color=colors[key], linewidth=1.9)

    ax1.set_xlabel("Sorted Latent Channel Rank (1..96)")
    ax1.set_ylabel("Normalized Token Variance")
    ax1.set_title("(a) TX Power Allocation Across Latent Channels")
    ax1.legend(loc="upper right", frameon=True)

    bar_specs = [
        ("matched_4_4", "Matched\n(4, 4)"),
        ("over_10_4", "Over +6dB\n(10, 4)"),
        ("over_13_4", "Over +9dB\n(13, 4)"),
        ("matched_10_10", "Matched\n(10, 10)"),
        ("under_4_10", "Under -6dB\n(4, 10)"),
        ("under_1_10", "Under -9dB\n(1, 10)"),
    ]
    labels = [b[1] for b in bar_specs if b[0] in shifts]
    rel_mse = [shifts[b[0]]["received_u_rel_mse_to_oracle"] for b in bar_specs if b[0] in shifts]
    cos_rx = [shifts[b[0]]["received_u_cosine_to_oracle"] for b in bar_specs if b[0] in shifts]

    x = np.arange(len(labels))
    w = 0.36
    ax2.bar(x - w / 2, cos_rx, width=w, label="Received u Cosine to z*(Gamma) [Higher]", color="#2980b9", alpha=0.88)
    ax2.bar(x + w / 2, rel_mse, width=w, label="Relative Latent MSE to z*(Gamma) [Lower]", color="#c0392b", alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.2)
    ax2.set_ylim(0.0, 1.08)
    ax2.set_ylabel("Latent Alignment / Distortion Ratio")
    ax2.set_title("(b) Latent Manifold Drift Under Mismatch")
    ax2.legend(loc="upper right", frameon=True, fontsize=7.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_latent_diagnosis.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_fig2_awgn_mismatch(awgn_csv: Path, out_dir: Path, split_name: str) -> None:
    rows = read_csv_rows(awgn_csv)
    sweep_rows = [r for r in rows if r["category"] == "sweep_g7" and r["model"] != "AvgSNR"]
    extrap_conds = [
        ("extrap_over_13_1_1", "Over +12dB\n(13->1dB)"),
        ("over_10_4_4", "Over +6dB\n(10->4dB)"),
        ("extrap_over_13_4_4", "Over +9dB\n(13->4dB)"),
        ("under_1_7_7", "Under -6dB\n(1->7dB)"),
        ("extrap_under_1_10_10", "Under -9dB\n(1->10dB)"),
        ("extrap_under_1_13_13", "Under -12dB\n(1->13dB)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax1.axvspan(6.0, 8.4, color="#fdebd0", alpha=0.5, label="Unseen Extrapolation")
    ax1.axvline(0.0, color="#2c3e50", linestyle=":", linewidth=1.1)

    handles, leg_labels = [], []
    plot_order = ["Frozen", "B0", "B1", "B2", "P_Raw", "P_ZPAC", "P_ZPAC_Hier"]
    for m in plot_order:
        mrows = sorted([r for r in sweep_rows if r["model"] == m], key=lambda x: float(x["delta_c_db"]))
        if not mrows or m not in STYLE_MAP:
            continue
        col, ls, mk, leg = STYLE_MAP[m]
        xs = [float(r["delta_c_db"]) for r in mrows]
        psnrs = [float(r["psnr_db_mean"]) for r in mrows]
        lw = 2.3 if m.startswith("P_ZPAC") else 1.45
        (ln,) = ax1.plot(xs, psnrs, color=col, linestyle=ls, marker=mk, markersize=5.2, linewidth=lw, label=leg)
        handles.append(ln)
        leg_labels.append(leg)

    ax1.set_xlabel("TX Condition Offset cT - Gamma (dB) at Gamma = 7 dB")
    ax1.set_ylabel("PSNR (dB) [Higher is Better]")
    ax1.set_title(f"(a) Continuous Offset Curve at Gamma=7dB ({split_name})")

    # Panel (b): Gain over Frozen Backbone (dB) across Seen & Unseen Extrapolation Conditions
    bar_models = ["B0", "P_Raw", "B2", "P_ZPAC", "P_ZPAC_Hier"]
    bar_models = [m for m in bar_models if any(r["model"] == m for r in rows)]
    frozen_lookup = {r["condition"]: float(r["psnr_db_mean"]) for r in rows if r["model"] == "Frozen"}
    valid_specs = [sp for sp in extrap_conds if sp[0] in frozen_lookup]

    x = np.arange(len(valid_specs))
    total_w = 0.78
    bw = total_w / max(1, len(bar_models))
    for idx, m in enumerate(bar_models):
        col, _, _, leg = STYLE_MAP[m]
        mlookup = {r["condition"]: float(r["psnr_db_mean"]) for r in rows if r["model"] == m}
        gains = [mlookup[c] - frozen_lookup[c] for c, _ in valid_specs]
        offset = (idx - (len(bar_models) - 1) / 2.0) * bw
        ax2.bar(x + offset, gains, width=bw * 0.92, color=col, alpha=0.90, label=leg)

    ax2.axhline(0.0, color="#2c3e50", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in valid_specs], fontsize=8.0)
    ax2.set_ylabel("PSNR Gain Over Original SwinJSCC (dB)")
    ax2.set_title(f"(b) Seen & Unseen Extrapolation Gains ({split_name})")

    fig.legend(handles, leg_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_awgn_sweep_{split_name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_fig3_fading_delay(fading_csv: Path, out_dir: Path, split_name: str) -> None:
    if not fading_csv.exists():
        return
    rows = [
        r for r in read_csv_rows(fading_csv)
        if abs(float(r["avg_snr_db"]) - 10.0) < 1e-3 and abs(float(r["rho_c"]) - 0.90) < 1e-3
    ]
    if not rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    handles, leg_labels = [], []
    plot_order = ["Frozen", "AvgSNR", "B0", "B1", "B2", "P_Raw", "P_ZPAC", "P_ZPAC_Hier"]
    for m in plot_order:
        mrows = sorted([r for r in rows if r["model"] == m], key=lambda x: int(x["delay_blocks"]))
        if not mrows or m not in STYLE_MAP:
            continue
        col, ls, mk, leg = STYLE_MAP[m]
        ds = [int(r["delay_blocks"]) for r in mrows]
        mean_psnr = [float(r["psnr_db_mean"]) for r in mrows]
        dists_vals = [float(r["dists_mean"]) for r in mrows]
        lw = 2.3 if m.startswith("P_ZPAC") else 1.45
        (ln,) = ax1.plot(ds, mean_psnr, color=col, linestyle=ls, marker=mk, markersize=5.2, linewidth=lw, label=leg)
        ax2.plot(ds, dists_vals, color=col, linestyle=ls, marker=mk, markersize=5.2, linewidth=lw, label=leg)
        handles.append(ln)
        leg_labels.append(leg)

    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xlabel("Feedback Delay d (Blocks, rho_c = 0.90, Avg SNR = 10 dB)")
    ax1.set_ylabel("Mean PSNR (dB) [Higher is Better]")
    ax1.set_title(f"(a) Correlated Rayleigh Fading PSNR ({split_name})")

    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xlabel("Feedback Delay d (Blocks, rho_c = 0.90, Avg SNR = 10 dB)")
    ax2.set_ylabel("Mean Perceptual DISTS [Lower is Better]")
    ax2.set_title(f"(b) Correlated Rayleigh Fading DISTS ({split_name})")

    fig.legend(handles, leg_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_fading_delay_{split_name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def render_fig4_ablations(ablation_csv: Path, out_dir: Path, split_name: str) -> None:
    if not ablation_csv.exists():
        return
    rows = read_csv_rows(ablation_csv)
    ab_order = [
        ("P_Raw", "P-Raw"),
        ("P_NoGate", "w/o Gate"),
        ("P_Sym", "w/o Asym"),
        ("P_NoXi", "w/o Wiener"),
        ("P_Pointwise", "w/o 3x3"),
        ("P_ZPAC", "P-ZPAC"),
        ("P_ZPAC_Hier", "P-ZPAC-Hier"),
    ]
    present = [(k, lbl) for k, lbl in ab_order if any(r["model"] == k for r in rows)]
    if not present:
        return

    over_cats = {"seen_over", "unseen_extrap"}
    under_cats = {"seen_under", "unseen_extrap"}

    over_psnr, under_psnr, under_dists = [], [], []
    for k, _ in present:
        mrows = [r for r in rows if r["model"] == k]
        ov = [float(r["psnr_db_mean"]) for r in mrows if float(r["delta_c_db"]) > 0 and r["category"] in over_cats]
        un = [float(r["psnr_db_mean"]) for r in mrows if float(r["delta_c_db"]) < 0 and r["category"] in under_cats]
        und = [float(r["dists_mean"]) for r in mrows if float(r["delta_c_db"]) < 0 and r["category"] in under_cats]
        over_psnr.append(float(np.mean(ov)))
        under_psnr.append(float(np.mean(un)))
        under_dists.append(float(np.mean(und)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6))
    xs = np.arange(len(present))
    colors = ["#95a5a6", "#e67e22", "#f39c12", "#3498db", "#9b59b6", "#e74c3c", "#900c3f"][: len(present)]

    mean_mis_psnr = [0.5 * (o + u) for o, u in zip(over_psnr, under_psnr)]
    ax1.bar(xs, mean_mis_psnr, color=colors, width=0.58, alpha=0.92)
    ax1.set_ylim(min(mean_mis_psnr) - 0.03, max(mean_mis_psnr) + 0.03)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([lbl for _, lbl in present], fontsize=7.8, rotation=18, ha="right")
    ax1.set_ylabel("Mean Mismatch PSNR (dB) [Higher]")
    ax1.set_title(f"(a) Ablation Progression on Mismatch PSNR ({split_name})")

    ax2.bar(xs, under_dists, color=colors, width=0.58, alpha=0.92)
    ax2.set_ylim(min(under_dists) - 0.002, max(under_dists) + 0.002)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([lbl for _, lbl in present], fontsize=7.8, rotation=18, ha="right")
    ax2.set_ylabel("Underestimation Perceptual DISTS [Lower]")
    ax2.set_title(f"(b) Ablation Progression on Detail DISTS ({split_name})")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig4_ablation_{split_name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-json", type=Path, default=None)
    parser.add_argument("--awgn-csv", type=Path, default=None)
    parser.add_argument("--fading-csv", type=Path, default=None)
    parser.add_argument("--ablation-csv", type=Path, default=None)
    parser.add_argument("--split-name", type=str, default="common500")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_runtime_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage0_json is not None and args.stage0_json.exists():
        render_fig1_latent_diagnosis(args.stage0_json, args.output_dir)
    if args.awgn_csv is not None and args.awgn_csv.exists():
        render_fig2_awgn_mismatch(args.awgn_csv, args.output_dir, args.split_name)
    if args.fading_csv is not None and args.fading_csv.exists():
        render_fig3_fading_delay(args.fading_csv, args.output_dir, args.split_name)
    if args.ablation_csv is not None and args.ablation_csv.exists():
        render_fig4_ablations(args.ablation_csv, args.output_dir, args.split_name)
    verify_no_links([args.output_dir])


if __name__ == "__main__":
    main()
