#!/usr/bin/env python3
"""Render IEEE camera-ready publication figures for Direction 4:
  - fig_dir4_1_rf_1bit_2bit_impairments.png: Dir-4A 1-Bit & 2-Bit ADC + Rapp HPA + Wiener Phase Noise + OOD Hardware Stress
  - fig_dir4_2_grantfree_noma_semsic.png: Dir-4B Grant-Free 2x2 Near-Far & Overloaded 3x2 Non-Orthogonal Uplink with Sem-SIC
  - fig_dir4_3_qualitative_kodak.png: 3-Row Held-Out Kodak24 Visual Comparison (1-Bit OOD Hardware, Overloaded 3x2 Weak User & 3-Way Collision)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_fig_dir4_1_rf(rf_csv: Path, out_png: Path) -> None:
    rows = load_csv(rf_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2), dpi=220)
    fig.suptitle(
        "Dir-4A — Zero-Pilot, Zero-AGC 1-Bit & 2-Bit ADC + Nonlinear HPA + Wiener Phase Noise Semantic Comm. (ImageNet-500)",
        fontsize=11.9,
        fontweight="bold",
        y=0.99,
    )

    line_styles = [
        ("Frozen_RF_Blind", "#7f8c8d", ":", "x", "Frozen Blind (0-Pilot)"),
        ("BlackBox_RF_Blind", "#2980b9", "--", "s", "Black-Box Blind RF (0-Pilot E2E)"),
        ("V4_USPH_ZeroShot", "#e67e22", "-.", "d", "V4 U-SPH-ZPAC (Static Subspace, High PAPR)"),
        ("Pilot4_Bussgang_LMMSE", "#8e44ad", "-.", "^", "4-Pilot Slot-Header Bussgang (128 Pilots)"),
        ("RF_Bussgang_JSCC", "#c0392b", "-", "*", "RF-Bussgang-JSCC (Ours: 0-Pilot, 0-AGC, CPE+LowPAPR)"),
    ]

    # Panel (a): 1-Bit Zero-AGC Comparator ADC Sweep
    ax1 = axes[0]
    sub1 = [r for r in rows if int(r["adc_bits"]) == 1 and r["is_ood"] == "False"]
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub1 if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
        if not pts:
            continue
        xs = [float(r["true_snr_db"]) for r in pts]
        ys = [float(r["psnr_db_mean"]) for r in pts]
        ylo = [float(r["psnr_db_ci95_lo"]) for r in pts]
        yhi = [float(r["psnr_db_ci95_hi"]) for r in pts]
        lw = 2.45 if mname == "RF_Bussgang_JSCC" else 1.75
        ms = 7.6 if mname == "RF_Bussgang_JSCC" else 5.8
        ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        if mname in ("RF_Bussgang_JSCC", "Pilot4_Bussgang_LMMSE", "V4_USPH_ZeroShot"):
            ax1.fill_between(xs, ylo, yhi, color=col, alpha=0.10)
    ax1.set_title("(a) 1-Bit Zero-AGC Comparator + PA (1 dB IBO) + PN (3.5°)")
    ax1.set_xlabel("Wireless Channel SNR before 1-Bit ADC (dB)")
    ax1.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax1.set_xticks([1, 4, 7, 10, 14])
    ax1.grid(True, alpha=0.30)
    ax1.legend(fontsize=7.3, loc="lower right", framealpha=0.92)

    # Panel (b): 2-Bit Mid-Rise ADC Sweep
    ax2 = axes[1]
    sub2 = [r for r in rows if int(r["adc_bits"]) == 2 and r["is_ood"] == "False"]
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub2 if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
        if not pts:
            continue
        xs = [float(r["true_snr_db"]) for r in pts]
        ys = [float(r["psnr_db_mean"]) for r in pts]
        ylo = [float(r["psnr_db_ci95_lo"]) for r in pts]
        yhi = [float(r["psnr_db_ci95_hi"]) for r in pts]
        lw = 2.45 if mname == "RF_Bussgang_JSCC" else 1.75
        ms = 7.6 if mname == "RF_Bussgang_JSCC" else 5.8
        ax2.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        if mname in ("RF_Bussgang_JSCC", "Pilot4_Bussgang_LMMSE", "V4_USPH_ZeroShot"):
            ax2.fill_between(xs, ylo, yhi, color=col, alpha=0.10)
    ax2.set_title("(b) 2-Bit Mid-Rise ADC + PA (1 dB IBO) + PN (3.5°)")
    ax2.set_xlabel("Wireless Channel SNR before 2-Bit ADC (dB)")
    ax2.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax2.set_xticks([4, 10, 14])
    ax2.grid(True, alpha=0.30)
    ax2.legend(fontsize=7.3, loc="lower right", framealpha=0.92)

    # Panel (c): In-Distribution vs Unseen OOD Extreme RF Stress (0 dB IBO Hard PA Saturation + 5.5 deg/sym PN)
    ax3 = axes[2]
    bar_groups = [
        ("adc1bit_snr10dB", "1-Bit In-Dist\n(1 dB IBO, 3.5° PN)"),
        ("adc1bit_ood_ibo0_pn55_10dB", "1-Bit OOD Stress\n(0 dB IBO, 5.5° PN!)"),
        ("adc2bit_snr10dB", "2-Bit In-Dist\n(1 dB IBO, 3.5° PN)"),
        ("adc2bit_ood_ibo0_pn55_10dB", "2-Bit OOD Stress\n(0 dB IBO, 5.5° PN!)"),
    ]
    bar_models = [
        ("BlackBox_RF_Blind", "#2980b9", "Black-Box Blind (0-P)"),
        ("V4_USPH_ZeroShot", "#e67e22", "V4 U-SPH (Static Subspace)"),
        ("Pilot4_Bussgang_LMMSE", "#8e44ad", "4-Pilot Bussgang (128-P)"),
        ("RF_Bussgang_JSCC", "#c0392b", "RF-Bussgang-JSCC (Ours, 0-P)"),
    ]
    xg = np.arange(len(bar_groups))
    wb = 0.19
    ymax = 0.0
    for b_i, (mname, col, blbl) in enumerate(bar_models):
        vals = []
        for cond, _ in bar_groups:
            hit = [r for r in rows if r["condition"] == cond and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax = max(ymax, v)
        bars = ax3.bar(
            xg + (b_i - 1.5) * wb,
            vals,
            width=wb,
            color=col,
            edgecolor="black" if mname == "RF_Bussgang_JSCC" else "none",
            linewidth=0.9 if mname == "RF_Bussgang_JSCC" else 0.0,
            label=blbl,
        )
        for br, v in zip(bars, vals):
            ax3.text(
                br.get_x() + br.get_width() / 2.0,
                v + 0.30,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.9,
                fontweight="bold" if mname == "RF_Bussgang_JSCC" else "normal",
                color="#900C3F" if mname == "RF_Bussgang_JSCC" else "#2c3e50",
            )
    ax3.set_title("(c) Unseen OOD Extreme Hardware Stress (10 dB)")
    ax3.set_xticks(xg)
    ax3.set_xticklabels([g[1] for g in bar_groups], fontsize=8.3)
    ax3.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax3.set_ylim(10.0, ymax + 5.8)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.4, loc="upper left", ncol=2, framealpha=0.93)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_fig_dir4_2_noma(noma_csv: Path, out_png: Path) -> None:
    rows = load_csv(noma_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=220)
    fig.suptitle(
        "Dir-4B — Grant-Free Zero-Pilot Near-Far & Overloaded 3×2 Non-Orthogonal Uplink with Semantic SIC (ImageNet-500)",
        fontsize=11.9,
        fontweight="bold",
        y=0.99,
    )

    m_order = [
        ("Uncoord_V4_Collision", "#95a5a6", "Uncoord. V4 Collision"),
        ("BlackBox_NOMA_Blind", "#2980b9", "Black-Box Blind NOMA (0-P)"),
        ("Pilot4_Linear_MMSE", "#8e44ad", "4-Pilot Linear MMSE (256-384 P)"),
        ("Genie_CSI_Linear_MMSE", "#34495e", "Genie-CSI Linear MMSE (Oracle CSI)"),
        ("SemSIC_Stage1_Only", "#e67e22", "Ours Stage-1 Only (No Sem-SIC)"),
        ("SemSIC_NOMA_JSCC", "#c0392b", "SemSIC-NOMA-JSCC (Ours: 2-Round SIC)"),
    ]

    # Panel (a): 2-User -> 2-BS-Antenna Grant-Free Uplink (Balanced & Near-Far + OOD 14/2 dB!)
    ax1 = axes[0]
    g2_specs = [
        ("noma2x2_bal_10dB", "ALL_USERS_MEAN", "2×2 Bal.\n(10/10 dB)"),
        ("noma2x2_nearfar_12_3dB", "ALL_USERS_MEAN", "Near-Far\nMean (12/3)"),
        ("noma2x2_nearfar_12_3dB", "User2_3dB", "Weak User\n(3 vs 12 dB)"),
        ("noma2x2_ood_nearfar_14_2dB", "User2_2dB", "OOD Weak\n(2 vs 14 dB!)"),
    ]
    x1 = np.arange(len(g2_specs))
    w1 = 0.135
    ymax1 = 0.0
    for b_i, (mname, col, lbl) in enumerate(m_order):
        vals = []
        for cond, role, _ in g2_specs:
            hit = [r for r in rows if r["condition"] == cond and r["user_role"] == role and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax1 = max(ymax1, v)
        bars = ax1.bar(
            x1 + (b_i - 2.5) * w1,
            vals,
            width=w1,
            color=col,
            edgecolor="black" if mname == "SemSIC_NOMA_JSCC" else "none",
            linewidth=0.85 if mname == "SemSIC_NOMA_JSCC" else 0.0,
            label=lbl,
        )
        if mname in ("SemSIC_NOMA_JSCC", "Genie_CSI_Linear_MMSE"):
            for br, v in zip(bars, vals):
                ax1.text(
                    br.get_x() + br.get_width() / 2.0,
                    v + 0.30,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold" if mname == "SemSIC_NOMA_JSCC" else "normal",
                )
    ax1.set_title("(a) 2-User → 2-Ant. Grant-Free Uplink (200% Spec. Eff.)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([g[2] for g in g2_specs], fontsize=8.4)
    ax1.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax1.set_ylim(7.0, ymax1 + 6.5)
    ax1.grid(True, axis="y", alpha=0.30)
    ax1.legend(fontsize=7.0, loc="upper right", ncol=2, framealpha=0.93)

    # Panel (b): Overloaded 3-User -> 2-BS-Antenna Per-User Breakdown (13 / 8 / 4 dB)
    ax2 = axes[1]
    g3_roles = [
        ("User1_13dB", "User 1\n(13 dB Strong)"),
        ("User2_8dB", "User 2\n(8 dB Med.)"),
        ("User3_4dB", "User 3\n(4 dB Weak!)"),
        ("ALL_USERS_MEAN", "3-User\nMean"),
    ]
    x2 = np.arange(len(g3_roles))
    ymax2 = 0.0
    for b_i, (mname, col, lbl) in enumerate(m_order):
        vals = []
        for role, _ in g3_roles:
            hit = [r for r in rows if r["condition"] == "over3x2_nearfar_13_8_4dB" and r["user_role"] == role and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax2 = max(ymax2, v)
        bars = ax2.bar(
            x2 + (b_i - 2.5) * w1,
            vals,
            width=w1,
            color=col,
            edgecolor="black" if mname == "SemSIC_NOMA_JSCC" else "none",
            linewidth=0.85 if mname == "SemSIC_NOMA_JSCC" else 0.0,
            label=lbl,
        )
        if mname in ("SemSIC_NOMA_JSCC", "SemSIC_Stage1_Only", "Genie_CSI_Linear_MMSE"):
            for br, v in zip(bars, vals):
                ax2.text(
                    br.get_x() + br.get_width() / 2.0,
                    v + 0.30,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.3,
                    fontweight="bold" if mname == "SemSIC_NOMA_JSCC" else "normal",
                )
    ax2.set_title("(b) Overloaded 3-User → 2-Ant. NOMA (13/8/4 dB, 300% Eff.)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([g[1] for g in g3_roles], fontsize=8.4)
    ax2.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax2.set_ylim(7.0, ymax2 + 6.5)
    ax2.grid(True, axis="y", alpha=0.30)
    ax2.legend(fontsize=7.0, loc="upper right", ncol=2, framealpha=0.93)

    # Panel (c): Overloaded 3x2 OOD Generalization (12/9/6, OOD 14/8/2, OOD Equal 10/10/10)
    ax3 = axes[2]
    ood_specs = [
        ("over3x2_mid_12_9_6dB", "ALL_USERS_MEAN", "3×2 Mid.\n(12/9/6 Mean)"),
        ("over3x2_ood_nearfar_14_8_2dB", "ALL_USERS_MEAN", "3×2 OOD\n(14/8/2 Mean)"),
        ("over3x2_ood_nearfar_14_8_2dB", "User3_2dB", "3×2 OOD Weak\n(2 dB User!)"),
        ("over3x2_ood_equal_10_10_10dB", "ALL_USERS_MEAN", "3×2 OOD Eq.\n(10/10/10 dB!)"),
    ]
    x3 = np.arange(len(ood_specs))
    m_sub = m_order[1:]  # Drop Uncoord (~9 dB) for crisp 5-bar spacing
    w3 = 0.16
    ymax3 = 0.0
    for b_i, (mname, col, lbl) in enumerate(m_sub):
        vals = []
        for cond, role, _ in ood_specs:
            hit = [r for r in rows if r["condition"] == cond and r["user_role"] == role and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax3 = max(ymax3, v)
        bars = ax3.bar(
            x3 + (b_i - 2.0) * w3,
            vals,
            width=w3,
            color=col,
            edgecolor="black" if mname == "SemSIC_NOMA_JSCC" else "none",
            linewidth=0.85 if mname == "SemSIC_NOMA_JSCC" else 0.0,
            label=lbl,
        )
        if mname in ("SemSIC_NOMA_JSCC", "SemSIC_Stage1_Only", "Genie_CSI_Linear_MMSE"):
            for br, v in zip(bars, vals):
                ax3.text(
                    br.get_x() + br.get_width() / 2.0,
                    v + 0.30,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    fontweight="bold" if mname == "SemSIC_NOMA_JSCC" else "normal",
                )
    ax3.set_title("(c) Overloaded 3×2 OOD Extreme & Equal-Power Collision")
    ax3.set_xticks(x3)
    ax3.set_xticklabels([g[2] for g in ood_specs], fontsize=8.3)
    ax3.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax3.set_ylim(7.0, ymax3 + 6.5)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.1, loc="upper right", framealpha=0.93)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def render_fig_dir4_3_qualitative(
    backbone_ckpt: Path,
    dir4_ckpt_root: Path,
    v4_usph_ckpt: Path,
    out_png: Path,
    physical_gpu: int = 0,
) -> None:
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.models import load_swinjscc_backbone
    from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4
    from dir2_zpac.phy_v2 import sample_correlated_rbs
    from dir2_zpac.metrics import compute_psnr_batch
    from dir4_rn.rf_1bit_blind import (
        N_OFDM_SC,
        N_OFDM_SYM,
        RFImpairedBlindTransceiver,
        forward_pilot4_rf_bussgang_reference,
        physical_rf_impaired_ofdm_channel,
    )
    from dir4_rn.noma_semsic_blind import (
        GrantFreeNOMATransceiver,
        forward_pilot4_or_genie_linear_noma_reference,
        physical_grantfree_uplink_mac,
        sample_multiuser_uplink_channels,
    )

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir4_rf1bit_noma_v1/cache/fig_dir4")
    )
    seed_everything(20261124)
    configure_torch_determinism()
    admit_gpu(physical_gpu)
    device = torch.device("cuda:0")

    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    targets = (kodak.to(dtype=torch.float32) / 255.0).to(device)
    bb = load_swinjscc_backbone(backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    m_rf_bb1 = RFImpairedBlindTransceiver(bb, method="blackbox_rf").to(device)
    m_rf_bb1.load_state_dict(torch.load(dir4_ckpt_root / "blackbox_rf_1bit/best_dir4.pt", map_location="cpu")["state_dict"], strict=False)
    m_rf_bb1.eval()

    m_rf_ours1 = RFImpairedBlindTransceiver(bb, method="rf_bussgang").to(device)
    m_rf_ours1.load_state_dict(torch.load(dir4_ckpt_root / "rf_bussgang_1bit/best_dir4.pt", map_location="cpu")["state_dict"], strict=False)
    m_rf_ours1.eval()

    m_noma_bb = GrantFreeNOMATransceiver(bb, method="blackbox_noma").to(device)
    m_noma_bb.load_state_dict(torch.load(dir4_ckpt_root / "blackbox_noma/best_dir4.pt", map_location="cpu")["state_dict"], strict=False)
    m_noma_bb.eval()

    m_noma_s1 = GrantFreeNOMATransceiver(bb, method="semsic_stage1_only").to(device)
    m_noma_s1.load_state_dict(torch.load(dir4_ckpt_root / "semsic_stage1_only/best_dir4.pt", map_location="cpu")["state_dict"], strict=False)
    m_noma_s1.eval()

    m_noma_sic = GrantFreeNOMATransceiver(bb, method="semsic_noma").to(device)
    m_noma_sic.load_state_dict(torch.load(dir4_ckpt_root / "semsic_noma/best_dir4.pt", map_location="cpu")["state_dict"], strict=False)
    m_noma_sic.eval()

    m_v4 = UniversalBlindTransceiverV4(bb, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    m_v4.load_state_dict(torch.load(v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    m_v4.eval()

    fig, axes = plt.subplots(3, 5, figsize=(16.0, 9.8), dpi=220)
    fig.suptitle(
        "Held-Out Kodak24 Visual Comparison — Dir-4A 1-Bit OOD Hardware Stress & Dir-4B Overloaded 3×2 Grant-Free NOMA",
        fontsize=12.1,
        fontweight="bold",
        y=0.985,
    )

    # ---------------- ROW 0: Dir-4A 1-Bit Zero-AGC + OOD 0 dB IBO PA + 5.5 deg/sym Wiener PN (10 dB, Kodak #19) ----------------
    idx0 = 18
    xb0 = targets[idx0 : idx0 + 1]
    sb0 = m_v4.extract_encoder_stem(xb0)
    gen0 = torch.Generator(device="cpu").manual_seed(3001 * 1301 + 53 + 55)
    eps0 = torch.randn((1, N_OFDM_SYM, N_OFDM_SC, 2), generator=gen0, dtype=torch.float32).to(device)
    hr0, hi0 = sample_correlated_rbs(1, 32, 0.85, device="cpu", generator=gen0)
    h0 = torch.stack([hr0, hi0], dim=-1).to(device)
    dphi0 = torch.randn((1, N_OFDM_SYM), generator=gen0, dtype=torch.float32) * (5.5 * math.pi / 180.0)
    dphi0[:, 0] = 0.0
    phi0 = torch.cumsum(dphi0, dim=1).to(device)

    r0_bb = m_rf_bb1.forward_rf_blind(sb0, 10.0, adc_bits=1, pa_ibo_db=0.0, pn_step_deg=5.5, h_rb_override=h0, phase_walk_override=phi0, standard_noise_td_iq=eps0, input_is_stem=True)["recon"].clamp(0, 1)
    stx_v4_0 = m_v4.encode_blind_tx(sb0).reshape(1, 32, 384, 2)
    rrx_v4_0, _ = physical_rf_impaired_ofdm_channel(stx_v4_0, 10.0, adc_bits=1, pa_ibo_db=0.0, pn_step_deg=5.5, h_rb_override=h0, phase_walk_override=phi0, standard_noise_td_iq=eps0)
    r0_v4 = m_v4.decode_blind_rx(rrx_v4_0.reshape(1, 12288, 2))[0].clamp(0, 1)
    r0_p4 = forward_pilot4_rf_bussgang_reference(bb, xb0, 10.0, adc_bits=1, pa_ibo_db=0.0, pn_step_deg=5.5, h_rb_override=h0, phase_walk_override=phi0, standard_noise_td_iq=eps0, pilot_seed=911).clamp(0, 1)
    r0_ours = m_rf_ours1.forward_rf_blind(sb0, 10.0, adc_bits=1, pa_ibo_db=0.0, pn_step_deg=5.5, h_rb_override=h0, phase_walk_override=phi0, standard_noise_td_iq=eps0, input_is_stem=True)["recon"].clamp(0, 1)

    axes[0, 0].imshow(xb0[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Ground Truth (Kodak #19)\n1-Bit ADC + OOD PA(0dB) + PN(5.5°)", fontsize=8.5, fontweight="bold")
    axes[0, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Black-Box 1-Bit RF\n(0-Pilot E2E)", r0_bb, False),
        ("V4 U-SPH-ZPAC\n(Static Subspace, High PAPR)", r0_v4, False),
        ("4-Pilot Slot-Header\n(128 Pilots, PN Drift!)", r0_p4, False),
        ("RF-Bussgang-JSCC (Ours)\n(1-Bit, 0-Pilot, 0-AGC!)", r0_ours, True),
    ], start=1):
        pv = float(compute_psnr_batch(rec, xb0)[0].item())
        axes[0, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
        axes[0, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=8.8, fontweight="bold" if is_ours else "normal", color="#900C3F" if is_ours else "black")
        axes[0, c_i].axis("off")

    # ---------------- ROW 1: Dir-4B Overloaded 3x2 NOMA Weakest User 3 (4 dB vs 13 dB & 8 dB Interferers, Kodak #23) ----------------
    u_imgs_r1 = [targets[6:7], targets[14:15], targets[22:23]]  # User 3 (Weak 4 dB) = Kodak #23 Parrots!
    gen1 = torch.Generator(device="cpu").manual_seed(300199)
    hr1, hi1 = sample_multiuser_uplink_channels(1, 3, 2, 32, 0.85, generator=gen1)
    hr1, hi1 = hr1.to(device), hi1.to(device)
    eps1 = torch.randn((1, 2, 32, 384, 2), generator=gen1, dtype=torch.float32).to(device)
    snrs_r1 = torch.tensor([[13.0, 8.0, 4.0]], device=device)

    v4_tx1 = torch.stack([m_v4.encode_blind_tx(m_v4.extract_encoder_stem(u_imgs_r1[i])).reshape(1, 32, 384, 2) for i in range(3)], dim=1)
    y_col1, _ = physical_grantfree_uplink_mac(v4_tx1, snrs_r1, n_rx=2, h_mu_override=(hr1, hi1), standard_noise_iq=eps1)
    r1_col = m_v4.decode_blind_rx(y_col1[:, 0].reshape(1, 12288, 2))[0].clamp(0, 1)
    r1_genie = forward_pilot4_or_genie_linear_noma_reference(bb, u_imgs_r1, snrs_r1, n_rx=2, h_mu_override=(hr1, hi1), standard_noise_iq=eps1, mode="genie_linear_mmse")[2].clamp(0, 1)
    r1_s1 = m_noma_s1.forward_noma_uplink(u_imgs_r1, snrs_r1, h_mu_override=(hr1, hi1), standard_noise_iq=eps1)["recons"][2].clamp(0, 1)
    r1_sic = m_noma_sic.forward_noma_uplink(u_imgs_r1, snrs_r1, h_mu_override=(hr1, hi1), standard_noise_iq=eps1)["recons"][2].clamp(0, 1)

    xb1 = u_imgs_r1[2]
    axes[1, 0].imshow(xb1[0].permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("User 3 Truth (Kodak #23)\nOverloaded 3×2 Weak 4 dB (vs 13/8)", fontsize=8.5, fontweight="bold")
    axes[1, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Uncoord. V4 Collision\n(Multi-Image Ghosting!)", r1_col, False),
        ("Genie-CSI Linear MMSE\n(3×2 Rank-Deficient Floor)", r1_genie, False),
        ("Ours Stage-1 Blind LMMSE\n(0-Pilot, No Sem-SIC)", r1_s1, False),
        ("SemSIC-NOMA-JSCC (Ours)\n(2-Round Sem-SIC, 0 Pilots!)", r1_sic, True),
    ], start=1):
        pv = float(compute_psnr_batch(rec, xb1)[0].item())
        axes[1, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
        axes[1, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=8.8, fontweight="bold" if is_ours else "normal", color="#900C3F" if is_ours else "black")
        axes[1, c_i].axis("off")

    # ---------------- ROW 2: Dir-4B Overloaded 3x2 OOD Equal-Power 10/10/10 dB 3-Way Collision (Kodak #7) ----------------
    u_imgs_r2 = [targets[3:4], targets[6:7], targets[19:20]]  # User 2 = Kodak #7 Hibiscus!
    gen2 = torch.Generator(device="cpu").manual_seed(300288)
    hr2, hi2 = sample_multiuser_uplink_channels(1, 3, 2, 32, 0.85, generator=gen2)
    hr2, hi2 = hr2.to(device), hi2.to(device)
    eps2 = torch.randn((1, 2, 32, 384, 2), generator=gen2, dtype=torch.float32).to(device)
    snrs_r2 = torch.tensor([[10.0, 10.0, 10.0]], device=device)

    r2_bb = m_noma_bb.forward_noma_uplink(u_imgs_r2, snrs_r2, h_mu_override=(hr2, hi2), standard_noise_iq=eps2)["recons"][1].clamp(0, 1)
    r2_genie = forward_pilot4_or_genie_linear_noma_reference(bb, u_imgs_r2, snrs_r2, n_rx=2, h_mu_override=(hr2, hi2), standard_noise_iq=eps2, mode="genie_linear_mmse")[1].clamp(0, 1)
    r2_s1 = m_noma_s1.forward_noma_uplink(u_imgs_r2, snrs_r2, h_mu_override=(hr2, hi2), standard_noise_iq=eps2)["recons"][1].clamp(0, 1)
    r2_sic = m_noma_sic.forward_noma_uplink(u_imgs_r2, snrs_r2, h_mu_override=(hr2, hi2), standard_noise_iq=eps2)["recons"][1].clamp(0, 1)

    xb2 = u_imgs_r2[1]
    axes[2, 0].imshow(xb2[0].permute(1, 2, 0).cpu().numpy())
    axes[2, 0].set_title("User 2 Truth (Kodak #7)\nOOD 3×2 Equal Collision (10/10/10)", fontsize=8.5, fontweight="bold")
    axes[2, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Black-Box Blind NOMA\n(0-Pilot E2E)", r2_bb, False),
        ("Genie-CSI Linear MMSE\n(3×2 Rank-Deficient Floor)", r2_genie, False),
        ("Ours Stage-1 Blind LMMSE\n(0-Pilot, No Sem-SIC)", r2_s1, False),
        ("SemSIC-NOMA-JSCC (Ours)\n(2-Round Sem-SIC, 0 Pilots!)", r2_sic, True),
    ], start=1):
        pv = float(compute_psnr_batch(rec, xb2)[0].item())
        axes[2, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
        axes[2, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=8.8, fontweight="bold" if is_ours else "normal", color="#900C3F" if is_ours else "black")
        axes[2, c_i].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.92, hspace=0.26, wspace=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--dir4-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    render_fig_dir4_1_rf(
        args.eval_dir / "dir4a_rf1bit_results_common500.csv",
        args.out_dir / "fig_dir4_1_rf_1bit_2bit_impairments.png",
    )
    render_fig_dir4_2_noma(
        args.eval_dir / "dir4b_noma_results_common500.csv",
        args.out_dir / "fig_dir4_2_grantfree_noma_semsic.png",
    )
    render_fig_dir4_3_qualitative(
        args.backbone_ckpt,
        args.dir4_ckpt_root,
        args.v4_usph_ckpt,
        args.out_dir / "fig_dir4_3_qualitative_kodak.png",
        physical_gpu=args.physical_gpu,
    )
    print("RENDERED ALL DIRECTION 4 FIGURES SUCCESSFULLY!")


if __name__ == "__main__":
    main()
