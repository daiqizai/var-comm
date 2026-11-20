#!/usr/bin/env python3
"""Render IEEE-Transactions camera-ready publication figures for V2 multi-suite experiments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLE_MAP = {
    "Frozen": ("#7f8c8d", "--", "o", "Frozen SwinJSCC"),
    "Frozen_Clamped": ("#95a5a6", ":", "s", "Frozen (CQI-Clamped)"),
    "AvgSNR": ("#e67e22", "-.", "^", "Avg-SNR Conservative"),
    "B1": ("#8e44ad", ":", "d", "B1 (Dual ModNet 4.94M)"),
    "P_Raw": ("#d35400", "--", "v", "P-Raw ([c_R, c_T, 0])"),
    "MS_ZPAC_ZeroMis": ("#16a085", "-.", "P", "MS-ZPAC-ZeroMis (RX)"),
    "MS_ZPAC": ("#2980b9", "-", "s", "MS-ZPAC (v2 RX-Only)"),
    "TR_ZPAC": ("#c0392b", "-", "*", "TR-ZPAC (v2 Joint TX-RX)"),
}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_fading_and_static_figure(
    static_csv: Path,
    fading_csv: Path,
    out_png: Path,
    suptitle_text: str,
) -> None:
    s_rows = load_csv(static_csv)
    f_rows = load_csv(fading_csv)
    if not s_rows or not f_rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.8, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), dpi=220)
    fig.suptitle(suptitle_text, fontsize=12.5, fontweight="bold", y=0.99)

    # Panel (a): OFDM Fading Delay Curve (SNR=10, rho=0.90)
    ax = axes[0]
    models_plot = [m for m in ["Frozen", "AvgSNR", "P_Raw", "MS_ZPAC_ZeroMis", "MS_ZPAC", "TR_ZPAC"] if any(r["model"] == m for r in f_rows)]
    for m in models_plot:
        col, ls, mk, lbl = STYLE_MAP.get(m, ("#333333", "-", "o", m))
        pts = sorted(
            [r for r in f_rows if r["model"] == m and float(r["avg_snr_db"]) == 10.0 and abs(float(r["rho_c"]) - 0.90) < 1e-3],
            key=lambda r: int(r["delay_blocks"]),
        )
        if pts:
            xs = [int(r["delay_blocks"]) for r in pts]
            ys = [float(r["psnr_db_mean"]) for r in pts]
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=2.2, markersize=6.5, label=lbl)
    ax.set_title("(a) 32-RB OFDM Fading vs. Delay (10 dB, ρ=0.90)")
    ax.set_xlabel("Feedback Delay d (blocks)")
    ax.set_ylabel("Mean PSNR (dB) ↑")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.0, loc="best")

    # Panel (b): Fast Fading & Low-SNR Outage p10 PSNR Bar Comparison
    ax2 = axes[1]
    target_scenarios = [
        ("ofdm_snr10_rho070_d2", "Fast Fade\n(10dB, ρ=0.7, d=2)"),
        ("ofdm_snr7_rho085_d2", "Mid SNR\n(7dB, ρ=0.85, d=2)"),
        ("ofdm_snr4_rho085_d2", "Low SNR\n(4dB, ρ=0.85, d=2)"),
    ]
    bar_models = [m for m in ["Frozen", "AvgSNR", "P_Raw", "MS_ZPAC", "TR_ZPAC"] if any(r["model"] == m for r in f_rows)]
    x_pos = np.arange(len(target_scenarios))
    width = 0.155
    for idx, m in enumerate(bar_models):
        col, _, _, lbl = STYLE_MAP.get(m, ("#333333", "-", "o", m))
        vals = []
        for sc_key, _ in target_scenarios:
            match_r = [r for r in f_rows if r["model"] == m and r["scenario"] == sc_key]
            vals.append(float(match_r[0]["frame_p10_psnr_db"]) if match_r else 0.0)
        offset = (idx - (len(bar_models) - 1) / 2.0) * width
        ax2.bar(x_pos + offset, vals, width=width * 0.90, color=col, alpha=0.92, label=lbl)
    ax2.set_title("(b) Worst-10% Outage Tail (p10 Frame PSNR)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([t[1] for t in target_scenarios], fontsize=8.8)
    ax2.set_ylabel("10th-Percentile PSNR (dB) ↑")
    ymin = min(float(r["frame_p10_psnr_db"]) for r in f_rows if r["scenario"] in {t[0] for t in target_scenarios})
    ax2.set_ylim(max(0.0, ymin - 1.2), None)
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel (c): Static Mismatch & Unseen Extrapolation Category Gains over P-Raw & Frozen
    ax3 = axes[2]
    cats = [
        ("matched", "Matched\n(Δc=0)"),
        ("seen_over", "Seen\nOver"),
        ("imperfect_rx", "Imperfect\nRX CSI"),
        ("unseen_extrap", "Unseen\nExtrap"),
        ("uncertain_csi", "Uncertain\nCSI"),
    ]
    comp_models = [m for m in ["AvgSNR", "P_Raw", "MS_ZPAC", "TR_ZPAC"] if any(r["model"] == m for r in s_rows)]
    x_c = np.arange(len(cats))
    w_c = 0.18
    for idx, m in enumerate(comp_models):
        col, _, _, lbl = STYLE_MAP.get(m, ("#333333", "-", "o", m))
        gains = []
        for c_key, _ in cats:
            m_vals = [float(r["psnr_db_mean"]) for r in s_rows if r["model"] == m and r["category"] == c_key]
            f_vals = [float(r["psnr_db_mean"]) for r in s_rows if r["model"] == "Frozen" and r["category"] == c_key]
            gains.append(float(np.mean(m_vals) - np.mean(f_vals)) if (m_vals and f_vals) else 0.0)
        offset = (idx - (len(comp_models) - 1) / 2.0) * w_c
        ax3.bar(x_c + offset, gains, width=w_c * 0.90, color=col, alpha=0.92, label=lbl)
    ax3.axhline(0.0, color="black", linewidth=0.9)
    ax3.set_title("(c) Static Mismatch ΔPSNR vs. Frozen")
    ax3.set_xticks(x_c)
    ax3.set_xticklabels([c[1] for c in cats], fontsize=8.5)
    ax3.set_ylabel("ΔPSNR over Frozen (dB) ↑")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_ablation_and_sara_figure(
    awgn_fading_csv: Path,
    sara_csvs: list[tuple[str, Path, Path]],
    out_png: Path,
) -> None:
    f_rows = load_csv(awgn_fading_csv)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.8, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.7), dpi=220)

    # Panel (a): Component Ablation Ladder on 32-RB OFDM Fast Fading (SNR=10, rho=0.70, d=2)
    ax = axes[0]
    abl_order = [
        ("Frozen", "Frozen"),
        ("B1", "B1\n(4.94M)"),
        ("P_Raw", "P-Raw\n(Concat)"),
        ("MS_ZPAC_NoMS", "w/o 4-Stage\nSwin AdaLN"),
        ("MS_ZPAC_NoIQ", "w/o I/Q\nPair"),
        ("MS_ZPAC", "MS-ZPAC\n(RX Full)"),
        ("TR_ZPAC", "TR-ZPAC\n(Joint)"),
    ]
    present = [(k, lbl) for k, lbl in abl_order if any(r["model"] == k for r in f_rows)]
    if present:
        xs = np.arange(len(present))
        means = []
        for k, _ in present:
            rr = [r for r in f_rows if r["model"] == k and r["scenario"] == "ofdm_snr10_rho070_d2"]
            means.append(float(rr[0]["psnr_db_mean"]) if rr else 0.0)
        colors = [
            "#7f8c8d" if k == "Frozen" else ("#c0392b" if k == "TR_ZPAC" else ("#2980b9" if k == "MS_ZPAC" else "#5dade2"))
            for k, _ in present
        ]
        bars = ax.bar(xs, means, width=0.56, color=colors, alpha=0.92)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in present], fontsize=8.3)
        ax.set_ylim(min(means) - 0.55, max(means) + 0.55)
        ax.set_ylabel("Mean PSNR (dB) ↑")
        ax.set_title("(a) V2 Component Ablation on 32-RB Fast OFDM Fading (ρ=0.70, d=2)")
        ax.grid(True, axis="y", alpha=0.3)

    # Panel (b): Multi-Budget SA&RA Gain over Frozen (dPSNR in dB across N=640, N=1024, N=2048!)
    ax2 = axes[1]
    labels_b = []
    avg_g, ms_fading_g, tr_fading_g, tr_extrap_g = [], [], [], []
    for b_lbl, f_p, s_p in sara_csvs:
        fr = load_csv(f_p)
        sr = load_csv(s_p)
        if not fr or not sr:
            continue
        labels_b.append(b_lbl)

        def get_f(mname: str) -> float:
            rr = [r for r in fr if r["model"] == mname and r["scenario"] == "ofdm_snr4_rho085_d2"]
            return float(rr[0]["psnr_db_mean"]) if rr else 0.0

        f0 = get_f("Frozen")
        avg_g.append(get_f("AvgSNR") - f0)
        ms_fading_g.append(get_f("MS_ZPAC") - f0)
        tr_fading_g.append(get_f("TR_ZPAC") - f0)

        tr_ex = [float(r["psnr_db_mean"]) for r in sr if r["model"] == "TR_ZPAC" and r["category"] == "uncertain_csi"]
        fr_ex = [float(r["psnr_db_mean"]) for r in sr if r["model"] == "Frozen" and r["category"] == "uncertain_csi"]
        tr_extrap_g.append(float(np.mean(tr_ex) - np.mean(fr_ex)) if (tr_ex and fr_ex) else 0.0)

    if labels_b:
        xb = np.arange(len(labels_b))
        wb = 0.19
        ax2.bar(xb - 1.5 * wb, avg_g, width=wb, color="#e67e22", label="Avg-SNR (OFDM 4dB)")
        ax2.bar(xb - 0.5 * wb, ms_fading_g, width=wb, color="#2980b9", label="MS-ZPAC RX (OFDM 4dB)")
        ax2.bar(xb + 0.5 * wb, tr_fading_g, width=wb, color="#c0392b", label="TR-ZPAC Joint (OFDM 4dB)")
        ax2.bar(xb + 1.5 * wb, tr_extrap_g, width=wb, color="#8e44ad", label="TR-ZPAC Joint (Uncertain AWGN)")
        ax2.axhline(0.0, color="black", linewidth=0.9)
        ax2.set_xticks(xb)
        ax2.set_xticklabels(labels_b, fontsize=9.2)
        ax2.set_ylabel("ΔPSNR Gain over Frozen SA&RA (dB) ↑")
        ax2.set_title("(b) Cross-Chapter Multi-Budget Bridge (SwinJSCC w/ SA&RA)")
        ax2.legend(fontsize=8.1, loc="best")
        ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    render_fading_and_static_figure(
        args.v2_eval_root / "rayleigh_common500/static_results_rayleigh_common500.csv",
        args.v2_eval_root / "rayleigh_common500/ofdm_fading_results_rayleigh_common500.csv",
        args.out_dir / "fig_v2_1_rayleigh_common500.png",
        "Suite A — Official SwinJSCC Rayleigh Backbone on ImageNet-500 (3 Seeds)",
    )
    render_fading_and_static_figure(
        args.v2_eval_root / "awgn_common500/static_results_awgn_common500.csv",
        args.v2_eval_root / "awgn_common500/ofdm_fading_results_awgn_common500.csv",
        args.out_dir / "fig_v2_2_awgn_ofdm_common500.png",
        "Suite B — Official SwinJSCC AWGN Backbone on ImageNet-500 (3 Seeds)",
    )
    render_ablation_and_sara_figure(
        args.v2_eval_root / "awgn_common500/ofdm_fading_results_awgn_common500.csv",
        [
            (
                "N=640 (k=3, Zero-Shot)",
                args.v2_eval_root / "sara_k3_common500/ofdm_fading_results_sara_common500.csv",
                args.v2_eval_root / "sara_k3_common500/static_results_sara_common500.csv",
            ),
            (
                "N=1024 (k=6, NeST-Aligned)",
                args.v2_eval_root / "sara_k6_common500/ofdm_fading_results_sara_common500.csv",
                args.v2_eval_root / "sara_k6_common500/static_results_sara_common500.csv",
            ),
            (
                "N=2048 (k=14, High-Rate)",
                args.v2_eval_root / "sara_k14_common500/ofdm_fading_results_sara_common500.csv",
                args.v2_eval_root / "sara_k14_common500/static_results_sara_common500.csv",
            ),
        ],
        args.out_dir / "fig_v2_3_ablation_and_sara.png",
    )
    print("RENDERED ALL V2 FIGURES SUCCESSFULLY!")


if __name__ == "__main__":
    main()
