#!/usr/bin/env python3
"""Render IEEE camera-ready publication figures for V4 Universal Blind, 4-User Broadcast & Digital 16/64-QAM JSCC."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_fig_v4_1(awgn_csv: Path, ofdm_csv: Path, out_png: Path) -> None:
    a_rows = load_csv(awgn_csv)
    o_rows = load_csv(ofdm_csv)
    if not a_rows or not o_rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.6, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), dpi=220)
    fig.suptitle(
        "V4 Suite 1 — Residual-Free Universal Self-Probing Across Official AWGN & Rayleigh Backbones (ImageNet-500)",
        fontsize=12.3,
        fontweight="bold",
        y=0.99,
    )

    # Panel (a): Blind Unknown-SNR AWGN Gap relative to AvgSNR_GeniePwr
    ax1 = axes[0]
    avg_ref = {float(r["true_snr_db"]): float(r["psnr_db_mean"]) for r in a_rows if r["model"] == "AvgSNR_GeniePwr"}
    styles_a = [
        ("AvgSNR_GeniePwr", "#e67e22", "--", "^", "Open-Loop AvgSNR (Genie Pwr)"),
        ("V3_SPH_ZPAC", "#5dade2", "--", "s", "V3 SPH-ZPAC (0-Pilot, 0-CSI)"),
        ("V4_U_SPH_AWGN", "#c0392b", "-", "*", "V4 U-SPH-ZPAC (Ours: Exact Null + dB-Loss)"),
        ("Oracle_MatchedCSI", "#2c3e50", ":", "o", "Oracle Matched-CSI (Genie Upper)"),
    ]
    for m, col, ls, mk, lbl in styles_a:
        pts = sorted([r for r in a_rows if r["model"] == m], key=lambda r: float(r["true_snr_db"]))
        if pts and avg_ref:
            xs = [float(r["true_snr_db"]) for r in pts]
            ys = [float(r["psnr_db_mean"]) - avg_ref[float(r["true_snr_db"])] for r in pts]
            ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=2.3 if m == "V4_U_SPH_AWGN" else 1.8, markersize=6.8, label=lbl)
    ax1.axhline(0.0, color="#e67e22", linewidth=1.0, linestyle="--")
    ax1.set_title("(a) Closing the High-SNR Blind AWGN Gap")
    ax1.set_xlabel("True AWGN SNR (dB) [-2 & 15 dB Unseen OOD]")
    ax1.set_ylabel("ΔPSNR over Genie-Pwr AvgSNR (dB) ↑")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7.8, loc="best")

    # Panel (b): Official SwinJSCC_w_SA_Rayleigh Backbone on Pilot-Free 32-RB Complex OFDM
    ax2 = axes[1]
    styles_b = [
        ("Frozen_Blind_Rayleigh", "#7f8c8d", ":", "x", "Frozen Official Rayleigh (0-Pilot)"),
        ("Pilot4_Frozen_Rayleigh", "#8e44ad", "-.", "^", "4-Pilot Official Rayleigh (128 Pilots)"),
        ("Pilot4_TR_ZPAC_Rayleigh", "#2c3e50", "--", "o", "4-Pilot V2 TR-ZPAC (Rayleigh BB)"),
        ("V4_U_SPH_Rayleigh", "#c0392b", "-", "*", "V4 U-SPH-ZPAC (Rayleigh BB, 0 Pilots!)"),
    ]
    for m, col, ls, mk, lbl in styles_b:
        pts = sorted(
            [r for r in o_rows if r["model"] == m and abs(float(r["rho_f"]) - 0.85) < 1e-3],
            key=lambda r: float(r["true_snr_db"]),
        )
        if pts:
            ax2.plot(
                [float(r["true_snr_db"]) for r in pts],
                [float(r["psnr_db_mean"]) for r in pts],
                color=col,
                linestyle=ls,
                marker=mk,
                linewidth=2.3 if m == "V4_U_SPH_Rayleigh" else 1.8,
                markersize=6.8,
                label=lbl,
            )
    ax2.set_title("(b) Official Rayleigh Backbone: 0-Pilot 32-RB OFDM")
    ax2.set_xlabel("True Channel SNR (dB) [Hidden from TX & RX]")
    ax2.set_ylabel("Mean PSNR (dB) ↑")
    ax2.set_xticks([1, 4, 7, 10, 13])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.8, loc="best")

    # Panel (c): Cross-Backbone OOD Severe Multipath (rho_f = 0.55) at 7 & 10 dB
    ax3 = axes[2]
    conds_c = [
        ("ofdm_ood_snr7_rhof055", "OOD Multipath\n(7dB, $\\rho_f=0.55$)"),
        ("ofdm_ood_snr10_rhof055", "OOD Multipath\n(10dB, $\\rho_f=0.55$)"),
        ("ofdm_snr13_rhof085", "High-SNR OFDM\n(13dB, $\\rho_f=0.85$)"),
    ]
    bars_c = [
        ("Pilot4_Frozen_Rayleigh", "#8e44ad", "4-Pilot Rayleigh (128 Pilots)"),
        ("Pilot4_TR_ZPAC_AWGN", "#34495e", "4-Pilot TR-ZPAC (AWGN BB)"),
        ("V4_U_SPH_Rayleigh", "#e74c3c", "V4 U-SPH (Rayleigh BB, 0-Pilot)"),
        ("V4_U_SPH_AWGN", "#922b21", "V4 U-SPH (AWGN BB, 0-Pilot)"),
    ]
    xp = np.arange(len(conds_c))
    w = 0.19
    for idx, (m, col, lbl) in enumerate(bars_c):
        vals = []
        for ck, _ in conds_c:
            rr = [r for r in o_rows if r["model"] == m and r["condition"] == ck]
            vals.append(float(rr[0]["psnr_db_mean"]) if rr else 0.0)
        off = (idx - (len(bars_c) - 1) / 2.0) * w
        ax3.bar(xp + off, vals, width=w * 0.90, color=col, alpha=0.92, label=lbl)
    ax3.set_title("(c) Cross-Backbone OOD Multipath & High-SNR OFDM")
    ax3.set_xticks(xp)
    ax3.set_xticklabels([c[1] for c in conds_c], fontsize=8.8)
    ax3.set_ylabel("Mean PSNR (dB) ↑")
    ax3.set_ylim(18.0, 34.5)
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(fontsize=7.6, loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_fig_v4_2(bcast_csv: Path, qam_csv: Path, out_png: Path) -> None:
    b_rows = load_csv(bcast_csv)
    q_rows = load_csv(qam_csv)
    if not b_rows or not q_rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.6, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.1), dpi=220)
    fig.suptitle(
        "V4 Suites 2 & 3 — 1-to-4 Simultaneous Wireless Broadcast & Standard 16/64-QAM Zero-Pilot Digital JSCC",
        fontsize=12.3,
        fontweight="bold",
        y=0.99,
    )

    # Panel (a): 1-to-4 Heterogeneous Wireless Broadcast
    ax1 = axes[0]
    b_cols = [
        ("U1_Edge_OOD_1dB_psnr_db", "U1 Edge\n(1dB OOD)"),
        ("U2_Urban_5dB_psnr_db", "U2 Urban\n(5dB OFDM)"),
        ("U3_Suburban_10dB_psnr_db", "U3 Suburb\n(10dB OFDM)"),
        ("U4_NearLoS_14dB_psnr_db", "U4 Near\n(14dB AWGN)"),
        ("mean_4user_psnr_db", "4-User\nMean"),
    ]
    b_models = [
        ("Blind_BlackBox_TCOM25", "#d35400", "TCOM'25 Blind (1x Air)"),
        ("Pilot4_EdgeTuned_cT1", "#8e44ad", "4-Pilot cT=1dB (1x Air)"),
        ("V4_DD_QAM64_SPH_Bcast", "#16a085", "V4 DD-64QAM (1x Air, 0-Pilot)"),
        ("V4_U_SPH_Bcast", "#c0392b", "V4 U-SPH (1x Air, 0-Pilot)"),
        ("Unicast_4x_ClosedLoop_Ceiling", "#2c3e50", "4x Unicast Ceiling (4x Air!)"),
    ]
    xp = np.arange(len(b_cols))
    w = 0.155
    for idx, (sc, col, lbl) in enumerate(b_models):
        rr = [r for r in b_rows if r["scheme"] == sc]
        vals = [float(rr[0][ck]) for ck, _ in b_cols] if rr else [0.0] * len(b_cols)
        off = (idx - (len(b_models) - 1) / 2.0) * w
        ax1.bar(xp + off, vals, width=w * 0.90, color=col, alpha=0.92, label=lbl)
    ax1.set_title("(a) 1-to-4 Heterogeneous Broadcast (100% vs 400% Airtime)")
    ax1.set_xticks(xp)
    ax1.set_xticklabels([c[1] for c in b_cols], fontsize=8.4)
    ax1.set_ylabel("PSNR (dB) ↑")
    ax1.set_ylim(14.0, 37.5)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=7.3, loc="upper left", framealpha=0.92)

    # Panel (b) & (c): 16-QAM and 64-QAM Zero-Pilot 32-RB Complex OFDM
    for ax, q_ord, ptitle in [
        (axes[1], 16, "(b) Standard 16-QAM: 0-Pilot 32-RB Complex OFDM"),
        (axes[2], 64, "(c) Standard 64-QAM: 0-Pilot 32-RB Complex OFDM"),
    ]:
        q_styles = [
            (f"Frozen_QAM{q_ord}_Blind", "#7f8c8d", ":", "x", f"Frozen {q_ord}-QAM (0-Pilot)"),
            (f"TCOM25_QAM{q_ord}_Blind", "#d35400", "--", "v", f"TCOM'25 Blind {q_ord}-QAM (0-Pilot)"),
            (f"Pilot4_QAM{q_ord}_MMSE", "#8e44ad", "-.", "^", f"4-Pilot {q_ord}-QAM (128 Pilots)"),
            (f"DD_QAM{q_ord}_SPH", "#c0392b", "-", "*", f"DD-{q_ord}QAM-SPH (Ours: 0-Pilot!)"),
        ]
        for m, col, ls, mk, lbl in q_styles:
            pts = sorted(
                [
                    r
                    for r in q_rows
                    if int(r["qam_order"]) == q_ord
                    and r["model"] == m
                    and r["channel_type"] == "ofdm_complex"
                    and abs(float(r["rho_f"]) - 0.85) < 1e-3
                ],
                key=lambda r: float(r["true_snr_db"]),
            )
            if pts:
                ax.plot(
                    [float(r["true_snr_db"]) for r in pts],
                    [float(r["psnr_db_mean"]) for r in pts],
                    color=col,
                    linestyle=ls,
                    marker=mk,
                    linewidth=2.3 if m.startswith("DD_QAM") else 1.8,
                    markersize=6.8,
                    label=lbl,
                )
        ax.set_title(ptitle)
        ax.set_xlabel("True Es/N0 (dB) [Exact Discrete QAM Symbols]")
        ax.set_ylabel("Mean PSNR (dB) ↑")
        ax.set_xticks([1, 4, 7, 10, 13])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.6, loc="lower right", bbox_to_anchor=(0.99, 0.13), framealpha=0.93)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_fig_v4_3_qualitative(
    awgn_bb_ckpt: Path,
    v4_ckpt_root: Path,
    out_png: Path,
    physical_gpu: int = 0,
) -> None:
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.models import load_swinjscc_backbone
    from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4
    from dir2_zpac.metrics import compute_psnr_batch
    from evaluate_blind_v3 import make_paired_h_rb, make_paired_iq_noise
    from evaluate_universal_blind_v4 import forward_pilot4_qam_reference

    configure_runtime_env(cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_universal_blind_v4/cache/fig_v4"))
    seed_everything(20261122)
    configure_torch_determinism()
    admit_gpu(physical_gpu)
    device = torch.device("cuda:0")

    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    targets = (kodak.to(dtype=torch.float32) / 255.0).to(device)
    bb = load_swinjscc_backbone(awgn_bb_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    def load_m(mth: str, tag: str | None) -> UniversalBlindTransceiverV4:
        m = UniversalBlindTransceiverV4(bb, method=mth, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)
        if tag is not None:
            m.load_state_dict(torch.load(v4_ckpt_root / tag / "best_v4.pt", map_location="cpu")["state_dict"], strict=False)
        return m.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15.6, 6.6), dpi=220)
    with torch.inference_mode():
        eps_iq = make_paired_iq_noise(128 * 96, list(range(24)), 2001, device=device)
        h_rb_85 = make_paired_h_rb(32, 0.85, list(range(24)), 2001, device=device)
        h_rb_55 = make_paired_h_rb(32, 0.55, list(range(24)), 2002, device=device)

        configs = [
            (16, 10.0, 0.85, h_rb_85, 18, "Standard 16-QAM 32-RB OFDM (10 dB, $\\rho_f=0.85$)"),
            (64, 10.0, 0.55, h_rb_55, 6, "Standard 64-QAM OOD Multipath (10 dB, $\\rho_f=0.55$)"),
        ]
        for row_i, (q_ord, snr_v, rhof_v, h_rb, img_idx, desc) in enumerate(configs):
            m_frz = load_m(f"frozen_qam{q_ord}", None)
            m_tc = load_m(f"tcom25_qam{q_ord}", f"tcom25_qam{q_ord}")
            m_dd = load_m(f"dd_qam{q_ord}_sph", f"dd_qam{q_ord}_sph")

            xb1 = targets[img_idx : img_idx + 1]
            eps1 = eps_iq[img_idx : img_idx + 1]
            h1 = h_rb[img_idx : img_idx + 1]

            axes[row_i, 0].imshow(xb1[0].permute(1, 2, 0).cpu().numpy())
            axes[row_i, 0].set_title(f"Ground Truth (Kodak #{img_idx+1})\n{desc}", fontsize=8.8, fontweight="bold")
            axes[row_i, 0].axis("off")

            r_frz = m_frz.forward_blind(xb1, snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h1, standard_noise_iq=eps1)["recon"].clamp(0, 1)
            r_tc = m_tc.forward_blind(xb1, snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h1, standard_noise_iq=eps1)["recon"].clamp(0, 1)
            r_p4 = forward_pilot4_qam_reference(bb, xb1, q_ord, snr_v, "ofdm_complex", eps1, h1, pilot_seed=777 + row_i).clamp(0, 1)
            r_dd = m_dd.forward_blind(xb1, snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h1, standard_noise_iq=eps1)["recon"].clamp(0, 1)

            panels = [
                (f"Frozen {q_ord}-QAM\n(0-Pilot Blind)", r_frz, False),
                (f"TCOM'25 Blind {q_ord}-QAM\n(0-Pilot E2E)", r_tc, False),
                (f"4-Pilot {q_ord}-QAM\n(128 Extra Pilots)", r_p4, False),
                (f"DD-{q_ord}QAM-SPH (Ours)\n(0 Pilots, Exact {q_ord}-QAM!)", r_dd, True),
            ]
            for c_i, (lbl, rec, is_ours) in enumerate(panels, start=1):
                pv = float(compute_psnr_batch(rec, xb1)[0].item())
                axes[row_i, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
                axes[row_i, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=9.0, fontweight="bold" if is_ours else "normal")
                axes[row_i, c_i].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, hspace=0.25, wspace=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-eval-dir", type=Path, required=True)
    parser.add_argument("--awgn-bb", type=Path, required=True)
    parser.add_argument("--v4-ckpt-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    render_fig_v4_1(
        args.v4_eval_dir / "v4_s1_universal_awgn_common500.csv",
        args.v4_eval_dir / "v4_s1_universal_ofdm_common500.csv",
        args.out_dir / "fig_v4_1_universal_and_rayleigh.png",
    )
    render_fig_v4_2(
        args.v4_eval_dir / "v4_s2_broadcast_4user_common500.csv",
        args.v4_eval_dir / "v4_s3_digital_qam_common500.csv",
        args.out_dir / "fig_v4_2_broadcast_and_digital_qam.png",
    )
    render_fig_v4_3_qualitative(
        args.awgn_bb,
        args.v4_ckpt_root,
        args.out_dir / "fig_v4_3_digital_qam_qualitative_kodak.png",
        physical_gpu=args.physical_gpu,
    )
    print("RENDERED ALL V4 FIGURES SUCCESSFULLY!")


if __name__ == "__main__":
    main()
