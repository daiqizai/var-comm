#!/usr/bin/env python3
"""Publication-Grade Figure Renderer for Direction 6 (`codex/dir6-matched-breakthroughs-v1`):
Generates 3 high-resolution (300 DPI) audit-verified figures:
  1. `fig_dir6_1_burst_otfs_matched.png`: Dir-6A Matched-Control Burst-OTFS vs Byte-Identical OFDM-2D + 2x2 Cross-Switch
  2. `fig_dir6_2_bayesvar_cbr512_256_dinov2.png`: Dir-6B BayesVAR (AR=ON vs Single-Switch AR=OFF) vs Loss-Matched Perceptual Swin + Kodak Visuals
  3. `fig_dir6_3_overloaded_4x2_noma_qualitative.png`: Dir-6C 200% Overloaded (4-User -> 2-Antenna) Disjoint-Tuple NOMA + Kodak 4-User Visuals
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR5_SRC = Path(__file__).resolve().parents[2] / "dir5-otfs-genvar-20261122/src"
DIR4_SRC = Path(__file__).resolve().parents[2] / "dir4-rf1bit-noma-20261122/src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (SCRIPTS_DIR, PKG_SRC, DIR5_SRC, DIR4_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, verify_no_links
from dir2_zpac.metrics import compute_psnr_batch
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels

from dir6_mb import (
    BayesVARTransceiver,
    OverloadedNOMA4x2Transceiver,
    PerceptualSwinUltraLowTransceiver,
    forward_genie_linear_noma_4x2,
    load_frozen_vae_and_var_d16,
)
from dir6_mb.overloaded_noma_4x2 import N_B_4X2

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
})


def render_fig1_dir6a(res_dir: Path, out_dir: Path) -> Path:
    d = json.loads((res_dir / "eval_dir6a_common500.json").read_text())
    conds = [
        ("unblocked_kd010_snr12", "0/48 Unblk\n(12 dB)"),
        ("burst8_kd010_snr12", "8/48 Burst\n(12 dB)"),
        ("burst12_kd010_snr10", "12/48 Burst\n(10 dB)"),
        ("burst12_kd010_snr14", "12/48 Burst\n(14 dB)"),
        ("burst16_kd010_snr14_ood", "16/48 OOD\n(500 km/h)"),
        ("burst16_kd018_snr14_ood", "16/48+900\n(Dbl OOD)"),
    ]

    fig = plt.figure(figsize=(18.2, 5.6), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.42, 1.28, 0.95], wspace=0.28)

    # Panel A: PSNR across Burst Blockage Regimes
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(conds))
    w = 0.26
    specs = [
        ("burst_symp_otfs", "Burst-Symp-OTFS (0-Pilot, Proposed)", "#1b6ca8"),
        ("burst_symp_ofdm_2d", "Burst-Symp-OFDM-2D (Byte-Identical Control)", "#e05a47"),
        ("oracle_genie_csi_ofdm", "Oracle Genie-CSI OFDM (100% DoF)", "#6c757d"),
    ]
    for i, (m_key, lbl, col) in enumerate(specs):
        means = [d["conditions"][c][ "methods"][m_key]["psnr_db"]["mean"] for c, _ in conds]
        cis = [d["conditions"][c]["methods"][m_key]["psnr_db"]["ci95"] for c, _ in conds]
        bars = ax0.bar(x + (i - 1) * w, means, width=w, yerr=cis, capsize=3, color=col, edgecolor="black", linewidth=0.6, label=lbl, alpha=0.93)
        if i == 0:
            for idx_c, (c_key, _) in enumerate(conds):
                delta = d["conditions"][c_key]["paired_delta_otfs_minus_ofdm2d"]["psnr_db"]["mean"]
                if abs(delta) >= 0.4:
                    ax0.text(
                        x[idx_c] - w,
                        means[idx_c] + cis[idx_c] + 0.35,
                        f"{delta:+.2f}dB",
                        ha="center",
                        va="bottom",
                        fontsize=8.2,
                        fontweight="bold",
                        color="#0d47a1",
                    )

    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl in conds], fontsize=8.4)
    ax0.set_ylabel("PSNR (dB) ↑")
    ax0.set_ylim(13.5, 32.0)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.set_title("(a) common500 PSNR under Doubly-Selective Burst Blockage")
    ax0.legend(loc="upper right", framealpha=0.92, fontsize=8.3)

    # Panel B: DINOv2 Cosine & DISTS
    ax1 = fig.add_subplot(gs[1])
    dino_otfs = [d["conditions"][c]["methods"]["burst_symp_otfs"]["dinov2_cos"]["mean"] for c, _ in conds]
    dino_ofdm = [d["conditions"][c]["methods"]["burst_symp_ofdm_2d"]["dinov2_cos"]["mean"] for c, _ in conds]
    dino_orac = [d["conditions"][c]["methods"]["oracle_genie_csi_ofdm"]["dinov2_cos"]["mean"] for c, _ in conds]

    ax1.plot(x, dino_otfs, "-o", color="#1b6ca8", lw=2.3, ms=6.5, label="Burst-Symp-OTFS (DINOv2 ↑)")
    ax1.plot(x, dino_ofdm, "--s", color="#e05a47", lw=2.1, ms=6.0, label="Byte-Identical OFDM-2D (DINOv2 ↑)")
    ax1.plot(x, dino_orac, ":^", color="#6c757d", lw=1.8, ms=5.8, label="Oracle Genie-CSI OFDM (DINOv2 ↑)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in conds], fontsize=8.4)
    ax1.set_ylabel("DINOv2-ViT-L/14 Semantic Cosine ↑")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title("(b) Semantic Representation Preservation (DINOv2)")
    ax1.legend(loc="lower left", framealpha=0.92)

    # Panel C: 2x2 Weight x Waveform Cross-Switch Matrix on OOD 16/48 Blockage
    ax2 = fig.add_subplot(gs[2])
    cs = d["cross_switch_2x2"]["burst16_kd010_snr14_ood"]
    mat = np.array([
        [cs["w_otfs__wave_otfs"]["psnr_db"]["mean"], cs["w_otfs__wave_ofdm"]["psnr_db"]["mean"]],
        [cs["w_ofdm__wave_otfs"]["psnr_db"]["mean"], cs["w_ofdm__wave_ofdm"]["psnr_db"]["mean"]],
    ])
    im = ax2.imshow(mat, cmap="YlGnBu", vmin=mat.min() - 0.6, vmax=mat.max() + 0.4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Inference:\nOTFS (ISFFT/SFFT)", "Inference:\nOFDM-2D (Identity)"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Weights:\nTrained w/ OTFS", "Weights:\nTrained w/ OFDM"])
    for r in range(2):
        for c in range(2):
            val = mat[r, c]
            ax2.text(
                c,
                r,
                f"{val:.2f} dB",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if val > (mat.mean()) else "#111111",
            )
    ax2.set_title("(c) 2×2 Weight×Waveform Audit\n(16/48 Burst OOD, SNR=14 dB)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="PSNR (dB)")

    out_path = out_dir / "fig_dir6_1_burst_otfs_matched.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


@torch.inference_mode()
def render_fig2_dir6b(res_dir: Path, ckpt_root: Path, out_dir: Path, device: torch.device) -> Path:
    d = json.loads((res_dir / "eval_dir6b_common500.json").read_text())
    conds = [
        ("rayleigh_snr4", "4 dB\nRayleigh"),
        ("rayleigh_snr8", "8 dB\nRayleigh"),
        ("rayleigh_snr13", "13 dB\nRayleigh"),
        ("erasure50_snr10_ood", "50% Erasure\n(10 dB OOD)"),
    ]

    fig = plt.figure(figsize=(17.5, 9.6), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.12], hspace=0.34, wspace=0.22)

    m_specs = [
        ("bayes_var_ar_on", "BayesVAR (AR=ON, Proposed 0-Pilot)", "#1565c0", "-o"),
        ("bayes_var_ar_off", "BayesVAR (AR=OFF, Exact Single-Switch)", "#f57c00", "--s"),
        ("swin_holo_perc", "Swin-Holo-Perc (0-Pilot, Loss-Matched)", "#d32f2f", "-.^"),
        ("swin_genie_perc", "Oracle Swin-Genie-Perc (Exact CSI, Loss-Matched)", "#546e7a", ":D"),
    ]

    # Top-Left: CBR = 1/512 DISTS & DINOv2
    ax_tl = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conds))
    c512 = d["by_cbr"]["cbr_512"]["conditions"]
    for m_k, lbl, col, sty in m_specs:
        vals = [c512[c]["methods"][m_k]["dists"]["mean"] for c, _ in conds]
        ax_tl.plot(x, vals, sty, color=col, lw=2.2, ms=6.2, label=lbl)
    ax_tl.set_xticks(x)
    ax_tl.set_xticklabels([l for _, l in conds])
    ax_tl.set_ylabel("DISTS Perceptual Distance ↓ (Lower is Better)")
    ax_tl.grid(True, linestyle="--", alpha=0.35)
    ax_tl.set_title("(a) Extreme CBR = 1/512 (K = 384 Symbols): DISTS Perceptual Error ↓")
    ax_tl.legend(loc="upper right", fontsize=8.3, framealpha=0.93)

    # Top-Right: Held-Out DINOv2-ViT-L/14 Cosine across CBR=1/512 & CBR=1/256
    ax_tr = fig.add_subplot(gs[0, 1])
    c256 = d["by_cbr"]["cbr_256"]["conditions"]
    bar_groups = [
        ("CBR=1/512\n4dB", c512["rayleigh_snr4"]),
        ("CBR=1/512\n8dB", c512["rayleigh_snr8"]),
        ("CBR=1/512\n50% Erasure", c512["erasure50_snr10_ood"]),
        ("CBR=1/256\n8dB", c256["rayleigh_snr8"]),
        ("CBR=1/256\n13dB", c256["rayleigh_snr13"]),
    ]
    xg = np.arange(len(bar_groups))
    bw = 0.19
    for i, (m_k, lbl, col, _) in enumerate(m_specs):
        means = [bg[1]["methods"][m_k]["dinov2_cos"]["mean"] for bg in bar_groups]
        ax_tr.bar(xg + (i - 1.5) * bw, means, width=bw, color=col, edgecolor="black", linewidth=0.55, label=lbl, alpha=0.92)
    all_dino_vals = [
        bg[1]["methods"][m_k]["dinov2_cos"]["mean"]
        for bg in bar_groups
        for m_k, _, _, _ in m_specs
    ]
    ax_tr.set_xticks(xg)
    ax_tr.set_xticklabels([bg[0] for bg in bar_groups])
    ax_tr.set_ylabel("DINOv2-ViT-L/14 Cosine Similarity ↑")
    ax_tr.set_ylim(0.0, min(0.98, max(all_dino_vals) + 0.08))
    ax_tr.grid(axis="y", linestyle="--", alpha=0.35)
    ax_tr.set_title("(b) Held-Out DINOv2 Semantic Fidelity ↑ (Never Seen in Training)")
    ax_tr.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    # Bottom Row: Qualitative Kodak Visual Strip (Classic kodim23 Macaws at CBR = 1/256, K = 768 symbols, SNR = 13 dB)
    kodak_u8 = load_split_u8("kodak24", allow_holdout=True).images_u8
    img_idx = 22  # Classic kodim23 benchmark scene
    gt_01 = kodak_u8[img_idx : img_idx + 1].to(device=device, dtype=torch.float32) / 255.0

    vae, var_d16 = load_frozen_vae_and_var_d16(device=device, load_var=True)
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    bv = BayesVARTransceiver(vae, var_d16, cbr_denom=256).to(device)
    bv.load_state_dict(torch.load(ckpt_root / "bayes_var_cbr256/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"], strict=False)
    bv.eval()

    sh = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=256, method="swin_holo_perc").to(device)
    sh.load_state_dict(torch.load(ckpt_root / "swin_holo_perc_cbr256/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"], strict=False)
    sh.eval()

    sg = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=256, method="swin_genie_perc").to(device)
    sg.load_state_dict(torch.load(ckpt_root / "swin_genie_perc_cbr256/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"], strict=False)
    sg.eval()

    gen = torch.Generator(device="cpu").manual_seed(202606)
    hr, hi = sample_correlated_rbs(1, 8, 0.85, device="cpu", dtype=torch.float32, generator=gen)
    h_ovr = torch.stack([hr, hi], dim=-1)
    eps = torch.randn((1, 8, 96, 2), generator=gen)

    out_bv = bv.forward_ultralow(gt_01, 13.0, h_rb_override=h_ovr, standard_noise_iq=eps, use_ar=True, also_return_ar_off=True, seed_offset=20261123)
    out_sh = sh.forward_ultralow(gt_01, 13.0, h_rb_override=h_ovr, standard_noise_iq=eps)
    out_sg = sg.forward_ultralow(gt_01, 13.0, h_rb_override=h_ovr, standard_noise_iq=eps)

    from evaluate_dir6_matched import Full5MetricEvaluator

    ev5 = Full5MetricEvaluator(device)
    panels = [
        ("Ground Truth (kodim23)\nCBR = 1/256 (Only 768 Symbols!)", gt_01, None),
        ("Swin-Holo-Perc (0-Pilot)\nLoss-Matched L1+MSE+LPIPS", out_sh["recon"], ev5.evaluate_batch(gt_01, out_sh["recon"])),
        ("Oracle Swin-Genie-Perc\nExact CSI + Loss-Matched", out_sg["recon"], ev5.evaluate_batch(gt_01, out_sg["recon"])),
        ("BayesVAR (AR=OFF)\nIdentical Wireless Latent", out_bv["recon_ar_off"], ev5.evaluate_batch(gt_01, out_bv["recon_ar_off"])),
        ("BayesVAR (AR=ON, Proposed)\nVAR-d16 Next-Scale Posterior", out_bv["recon"], ev5.evaluate_batch(gt_01, out_bv["recon"])),
    ]

    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, :], wspace=0.08)
    for idx_p, (title_p, img_t, m_dict) in enumerate(panels):
        ax_p = fig.add_subplot(gs_bot[0, idx_p])
        np_img = img_t[0].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        ax_p.imshow(np_img)
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        if m_dict is not None:
            sub = f"DISTS={m_dict['dists'][0]:.3f} | DINO={m_dict['dinov2_cos'][0]:.3f}\nPSNR={m_dict['psnr_db'][0]:.2f} dB"
            ax_p.set_xlabel(sub, fontsize=8.8, fontweight="bold")
        ax_p.set_title(title_p, fontsize=9.5)

    out_path = out_dir / "fig_dir6_2_bayesvar_cbr512_256_dinov2.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


@torch.inference_mode()
def render_fig3_dir6c(res_dir: Path, ckpt_root: Path, out_dir: Path, device: torch.device) -> Path:
    d = json.loads((res_dir / "eval_dir6c_common500.json").read_text())
    conds = [
        ("sym_4u_snr8", "Sym 4×2\n(8 dB)"),
        ("sym_4u_snr12", "Sym 4×2\n(12 dB)"),
        ("nearfar_14db", "Near-Far 14dB\n[18,13,8,4]"),
        ("nearfar_18db_ood", "Near-Far 18dB\n[20,14,8,2] OOD"),
        ("sporadic_random_k1234", "Sporadic K∈{1..4}\n(Blind CFAR)"),
    ]

    fig = plt.figure(figsize=(17.5, 9.8), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.24)

    # Top-Left: All Active Users Mean PSNR across 125 Disjoint 4-Tuples on common500
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(conds))
    bw = 0.20
    m_specs = [
        ("semsic_3round", "SemSIC-4x2 (3-Round IC, Proposed)", "#1b6ca8"),
        ("owndenoise_3round", "OwnDenoise-4x2 (Byte-Identical Control)", "#e05a47"),
        ("tuned_pic_3round", "Conventional 4-LMMSE PIC (Val-Tuned)", "#f39c12"),
        ("linear_1stage", "1-Stage Blind LMMSE", "#7f8c8d"),
    ]
    for i, (m_k, lbl, col) in enumerate(m_specs):
        means = [d["conditions"][c]["all_active_users"][m_k]["psnr_db"]["mean"] for c, _ in conds]
        cis = [d["conditions"][c]["all_active_users"][m_k]["psnr_db"]["ci95"] for c, _ in conds]
        ax0.bar(x + (i - 1.5) * bw, means, width=bw, yerr=cis, capsize=2.5, color=col, edgecolor="black", linewidth=0.55, label=lbl)
        if i == 0:
            for ic, (c_k, _) in enumerate(conds):
                d_own = d["conditions"][c_k]["paired_delta_semsic_minus_owndenoise"]["psnr_db"]["mean"]
                ax0.text(x[ic] - 1.5 * bw, means[ic] + cis[ic] + 0.3, f"{d_own:+.2f}dB", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0d47a1")

    ax0.set_xticks(x)
    ax0.set_xticklabels([l for _, l in conds])
    ax0.set_ylabel("All-User Mean PSNR (dB) ↑")
    ax0.set_ylim(18.0, 32.5)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.set_title("(a) 200% Overloaded (4 Users → 2 Antennas) Disjoint common500 Tuples")
    ax0.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    # Top-Right: Weakest Edge User (Slot 3: 4 dB & 2 dB under Near-Far) + Oracle Genie Linear Comparison
    ax1 = fig.add_subplot(gs[0, 1])
    nf_conds = [
        ("sym_4u_snr12", "Sym 12 dB\n(User-3: 12dB)"),
        ("nearfar_14db", "14dB Near-Far\n(Weakest: 4dB)"),
        ("nearfar_18db_ood", "18dB Near-Far OOD\n(Weakest: 2dB)"),
    ]
    x_nf = np.arange(len(nf_conds))
    bw2 = 0.17
    nf_specs = [
        ("semsic_3round", "SemSIC-4x2 (0-Pilot)", "#1b6ca8"),
        ("owndenoise_3round", "OwnDenoise-4x2 (Byte-Identical)", "#e05a47"),
        ("tuned_pic_3round", "Tuned 4-LMMSE PIC", "#f39c12"),
        ("oracle_genie_linear_mmse", "Oracle Exact-CSI Linear MMSE", "#2ecc71"),
    ]
    for i, (m_k, lbl, col) in enumerate(nf_specs):
        means = [d["conditions"][c]["weakest_user_slot3"][m_k]["psnr_db"]["mean"] for c, _ in nf_conds]
        ax1.bar(x_nf + (i - 1.5) * bw2, means, width=bw2, color=col, edgecolor="black", linewidth=0.55, label=lbl)
    cfar_acc = d["conditions"]["sporadic_random_k1234"]["blind_cfar_detection"]["exact_subset_acc"]["mean"] * 100.0
    ax1.set_xticks(x_nf)
    ax1.set_xticklabels([l for _, l in nf_conds])
    ax1.set_ylabel("Weakest Edge User PSNR (dB) ↑")
    ax1.set_ylim(10.0, 25.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title(f"(b) Weakest Edge User Recovery (Blind CFAR Subset Acc = {cfar_acc:.1f}%)")
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    # Bottom Row: Visual Comparison of Weakest User-3 (4 dB) & Co-Channel User-2 (8 dB) under 4x2 Near-Far [18,13,8,4] dB!
    kodak_u8 = load_split_u8("kodak24", allow_holdout=True).images_u8
    u_imgs = [kodak_u8[u : u + 1].to(device=device, dtype=torch.float32) / 255.0 for u in (0, 4, 7, 14)]
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    m_sic = OverloadedNOMA4x2Transceiver(backbone, method="semsic_3round").to(device)
    ck_sic = torch.load(ckpt_root / "semsic_3round/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)
    m_sic.load_state_dict(ck_sic["model_state"], strict=False)
    m_sic.eval()

    m_own = OverloadedNOMA4x2Transceiver(backbone, method="owndenoise_3round").to(device)
    ck_own = torch.load(ckpt_root / "owndenoise_3round/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)
    m_own.load_state_dict(ck_own["model_state"], strict=False)
    m_own.eval()

    gen = torch.Generator(device="cpu").manual_seed(202609)
    hr, hi = sample_multiuser_uplink_channels(1, 4, 2, 32, 0.85, generator=gen)
    eps = torch.randn((1, 2, 32, N_B_4X2, 2), generator=gen)
    snr_nf = torch.tensor([[18.0, 13.0, 8.0, 4.0]], device=device)

    out_sic = m_sic.forward_noma_4x2(u_imgs, snr_nf, h_mu_override=(hr, hi), standard_noise_iq=eps)
    out_own = m_own.forward_noma_4x2(u_imgs, snr_nf, h_mu_override=(hr, hi), standard_noise_iq=eps)
    out_pic = m_sic.forward_noma_4x2(
        u_imgs, snr_nf, h_mu_override=(hr, hi), standard_noise_iq=eps, override_mode="tuned_pic_3round"
    )
    out_orc = forward_genie_linear_noma_4x2(backbone, u_imgs, snr_nf, n_rx=2, h_mu_override=(hr, hi), standard_noise_iq=eps)

    u_focus = 3  # Weakest 4 dB user buried -14 dB below User 0 on 200% overloaded 4->2 MAC!
    gt_f = u_imgs[u_focus]
    vis_list = [
        ("Weakest User-4 GT\n(4 dB under +14dB Near-Far)", gt_f),
        ("Oracle Exact-CSI Linear MMSE\n(Rank-Deficient 4→2 Ceiling)", out_orc[u_focus]),
        ("Tuned 4-LMMSE PIC\n(Classical Symbol IC)", out_pic["recons"][u_focus]),
        ("Byte-Identical OwnDenoise\n(No Cross-User Waveform IC)", out_own["recons"][u_focus]),
        ("SemSIC-4x2 (Proposed)\n3-Round Semantic-Physical IC", out_sic["recons"][u_focus]),
    ]
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1, :], wspace=0.08)
    for idx_v, (v_title, v_img) in enumerate(vis_list):
        ax_v = fig.add_subplot(gs_bot[0, idx_v])
        arr = v_img[0].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        ax_v.imshow(arr)
        ax_v.set_xticks([])
        ax_v.set_yticks([])
        if idx_v > 0:
            p_val = float(compute_psnr_batch(gt_f, v_img.clamp(0.0, 1.0))[0].item())
            ax_v.set_xlabel(f"PSNR = {p_val:.2f} dB", fontsize=9.2, fontweight="bold")
        ax_v.set_title(v_title, fontsize=9.4)

    out_path = out_dir / "fig_dir6_3_overloaded_4x2_noma_qualitative.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res-dir",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1/results/dir6_matched_breakthroughs_20261123"),
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir6_matched_breakthroughs_v1"),
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir6_matched_breakthroughs_v1"),
    )
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    admit_gpu(min_free_mib=20000)
    device = torch.device("cuda:0")

    fig_dir = args.res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = render_fig1_dir6a(args.res_dir, fig_dir)
    print(f"[Rendered] {f1}", flush=True)
    f2 = render_fig2_dir6b(args.res_dir, args.ckpt_root, fig_dir, device)
    print(f"[Rendered] {f2}", flush=True)
    f3 = render_fig3_dir6c(args.res_dir, args.ckpt_root, fig_dir, device)
    print(f"[Rendered] {f3}", flush=True)

    verify_no_links([args.res_dir, args.ckpt_root, args.tmp_root])


if __name__ == "__main__":
    main()
