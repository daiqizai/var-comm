#!/usr/bin/env python3
"""Render 3 Publication-Grade 300-DPI Figures for Direction 7 (`codex/dir7-fas-persist-impulse-v1`):
  1. `fig_dir7_1_blind_fas_kriging.png`:        Zero-Pilot 32-Port Fluid / Pixel Antenna JSCC (`Dir-7A`)
  2. `fig_dir7_2_persistguard_digital_outage.png`: Non-Stationary Delayed-Feedback Outage-Constrained VAR AMC (`Dir-7B`, solving `582038a`)
  3. `fig_dir7_3_impulse_shield_qualitative.png`:  Fast-Time Impulsive Interference Cancellation (`Dir-7C`)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
DIR6_SRC = Path(__file__).resolve().parents[2] / "dir6-matched-breakthroughs-20261123/src"
DIR5_SRC = Path(__file__).resolve().parents[2] / "dir5-otfs-genvar-20261122/src"
DIR4_SRC = Path(__file__).resolve().parents[2] / "dir4-rf1bit-noma-20261122/src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (SCRIPTS_DIR, PKG_SRC, DIR6_SRC, DIR5_SRC, DIR4_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, verify_no_links
from dir2_zpac.models import load_swinjscc_backbone
from dir5_og.otfs_blind import N_SC, N_SYM
from dir6_mb.bayes_var_ultralow import load_frozen_vae_and_var_d16
from dir7_fpi import (
    ANCHOR_PORTS_6,
    N_FAS_PORTS,
    BlindFASTransceiver,
    ImpulseShieldTransceiver,
    forward_genie_csi_impulsive_ofdm,
    forward_pilot32_exhaustive_fas,
    sample_fas_spatial_channels,
)
from dir7_fpi.blind_fas_jscc import FAS_APERTURE_LAMBDA, build_jakes_spatial_cov
from dir7_fpi.impulse_shield_blind import N_IMP_RBS, physical_impulsive_ofdm_channel
from dir7_fpi.persist_outage_amc import reconstruct_var_from_prefix
from evaluate_dir7_fpi import Full5MetricEvaluator, SWIN_AWGN_CKPT, _make_det_impulses, _make_det_rbs


def _to_hwc(img_01: torch.Tensor) -> np.ndarray:
    return np.clip(img_01.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


@torch.inference_mode()
def render_fig1_blind_fas(
    eval_dir: Path,
    ckpt_root: Path,
    fig_dir: Path,
    backbone: torch.nn.Module,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> Path:
    d7a = json.loads((eval_dir / "eval_dir7a_common500.json").read_text(encoding="utf-8"))
    conds = d7a["conditions"]

    snrs = [0, 3, 6, 10]
    ckeys = [f"nb1_rayleigh_snr{s}" for s in snrs]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.2, "axes.labelsize": 9.8})
    fig = plt.figure(figsize=(19.2, 9.8), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.30)

    # Panel (a): Mean PSNR & 10th-Percentile Outage PSNR vs Average Per-Port SNR
    ax_a = fig.add_subplot(gs[0, 0])
    styles = [
        ("oracle_genie_port_fas", "Oracle Genie-Port+CSI (Upper Bound)", "#555555", "--", "s"),
        ("fas_kriging", "Blind-FAS (6-Anchor Kriging, Proposed)", "#d62728", "-", "o"),
        ("random_hop", "Random 6-Anchor Hop (Byte-Id. Control)", "#ff7f0e", "-.", "^"),
        ("fixed_port", "Fixed Port 0 (Byte-Id. Control)", "#1f77b4", "-", "v"),
        ("pilot32_exhaustive_fas", "32-Pilot Exhaustive FAS (66.7% Pilot)", "#9467bd", ":", "d"),
    ]
    for m_key, lbl, col, ls, mk in styles:
        ys = [conds[ck]["models"][m_key]["psnr_db"]["mean"] for ck in ckeys]
        ax_a.plot(snrs, ys, label=lbl, color=col, linestyle=ls, marker=mk, linewidth=2.2, markersize=6)
    # Also plot p10 outage PSNR for `fas_kriging` vs `fixed_port` as shaded/thin lines
    p10_fas = [conds[ck]["models"]["fas_kriging"]["p10_psnr_db"]["mean"] for ck in ckeys]
    p10_fix = [conds[ck]["models"]["fixed_port"]["p10_psnr_db"]["mean"] for ck in ckeys]
    ax_a.plot(snrs, p10_fas, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.8, label="Blind-FAS p10 Outage PSNR")
    ax_a.plot(snrs, p10_fix, color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.8, label="Fixed Port p10 Outage PSNR")
    ax_a.set_title("(a) Mean & p10 Outage PSNR (N_rf=1, CBR=1/16)", fontweight="bold")
    ax_a.set_xlabel("Average Per-Port Es/N0 (dB)")
    ax_a.set_ylabel("PSNR (dB)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=7.6, loc="lower right")

    # Panel (b): Held-out OOD Regimes & 3x3 Cross-Switch Matrix
    ax_b = fig.add_subplot(gs[0, 1])
    bar_groups = [
        ("In-Dist\nRayleigh 3dB", "nb1_rayleigh_snr3"),
        ("OOD Wideband\n4-SB Pixel-FAS 4dB", "wb4_pixel_fas_snr4_ood"),
        ("OOD Rician\nK=2 (3dB) 4dB", "nb1_rician_k2_snr4_ood"),
    ]
    x = np.arange(len(bar_groups))
    w = 0.20
    m_bars = [
        ("fixed_port", "Fixed Port 0", "#1f77b4"),
        ("random_hop", "Random 6-Hop", "#ff7f0e"),
        ("pilot32_exhaustive_fas", "32-Pilot FAS", "#9467bd"),
        ("fas_kriging", "Blind-FAS (Ours)", "#d62728"),
    ]
    for i_m, (mk, mlbl, col) in enumerate(m_bars):
        vals = [conds[ck]["models"][mk]["psnr_db"]["mean"] for _, ck in bar_groups]
        rects = ax_b.bar(x + (i_m - 1.5) * w, vals, width=w, label=mlbl, color=col, alpha=0.9)
        for r in rects:
            h = r.get_height()
            ax_b.text(r.get_x() + r.get_width() / 2.0, h + 0.18, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([g[0] for g in bar_groups], fontsize=8.8)
    ax_b.set_ylabel("Mean PSNR (dB)")
    ax_b.set_ylim(16.0, max(31.5, ax_b.get_ylim()[1] + 1.8))
    ax_b.set_title("(b) In-Domain & Unseen OOD Generalization (common500)", fontweight="bold")
    ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.legend(fontsize=8.0, ncol=2, loc="upper left")

    # Panel (c): Live Bessel-J0 Spatial Kriging Standing-Wave Reconstruction across 32 Ports
    ax_c = fig.add_subplot(gs[0, 2])
    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    # Find a representative spatial realization where Port 0 is in a deep fade (< -6 dB) while aperture has a strong peak!
    gen_demo = torch.Generator(device="cpu").manual_seed(2026771)
    h_cand = sample_fas_spatial_channels(32, n_subbands=1, generator=gen_demo, device="cpu")
    p0_pwr = (h_cand[:, 0, 0].real.square() + h_cand[:, 0, 0].imag.square())
    max_pwr = (h_cand[:, :, 0].real.square() + h_cand[:, :, 0].imag.square()).max(dim=1).values
    best_i = int((max_pwr / p0_pwr.clamp_min(1e-4)).argmax().item())
    h_1 = h_cand[best_i : best_i + 1].to(device)  # [1, 32, 1]
    noise_1 = torch.randn((1, N_SYM, N_SC, 2), generator=gen_demo).to(device)

    pos_lam = np.linspace(0.0, FAS_APERTURE_LAMBDA, N_FAS_PORTS)
    true_gain_db = 10.0 * np.log10(
        np.maximum((h_1[0, :, 0].real.square() + h_1[0, :, 0].imag.square()).cpu().numpy(), 1e-4)
    )
    anc = list(ANCHOR_PORTS_6)
    r_sp = build_jakes_spatial_cov(N_FAS_PORTS, FAS_APERTURE_LAMBDA, device="cpu", dtype=torch.float64).numpy()
    h_anc_np = h_1[0, anc, 0].cpu().numpy()
    w_k = np.linalg.solve(r_sp[np.ix_(anc, anc)] + 0.015 * np.eye(len(anc)), r_sp[anc, :])
    h_krig_np = w_k.T @ h_anc_np
    krig_gain_db = 10.0 * np.log10(np.maximum(np.abs(h_krig_np) ** 2, 1e-4))
    p_sel = int(np.argmax(krig_gain_db))

    ax_c.plot(pos_lam, true_gain_db, color="#1f77b4", linewidth=2.2, label="True 32-Port |h(x)|^2 (dB)")
    ax_c.plot(pos_lam, krig_gain_db, color="#d62728", linestyle="--", linewidth=2.0, label="Blind 6-Anchor Kriging Est.")
    ax_c.scatter(pos_lam[anc], true_gain_db[anc], color="#ff7f0e", s=55, zorder=5, label="6 Swept Payload Anchors")
    ax_c.scatter([pos_lam[0]], [true_gain_db[0]], color="#111111", marker="x", s=80, zorder=6, label=f"Port 0 ({true_gain_db[0]:.1f} dB)")
    ax_c.scatter(
        [pos_lam[p_sel]],
        [true_gain_db[p_sel]],
        color="#2ca02c",
        marker="*",
        s=150,
        zorder=7,
        label=f"Locked Port p*={p_sel} (+{true_gain_db[p_sel]:.1f} dB)",
    )
    ax_c.set_title("(c) Single-RF Bessel-J0 Spatial Kriging (W = 2.5 lambda)", fontweight="bold")
    ax_c.set_xlabel("Aperture Position x / lambda (32 Sub-Wavelength Ports)")
    ax_c.set_ylabel("Spatial Channel Gain |h_p|^2 (dB)")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=7.6, loc="lower right")

    # Bottom Row: 4-Column Kodak Visual Strip at SNR = 3 dB under `h_1`
    img_01 = kodak[3:4].to(device=device, dtype=torch.float32) / 255.0
    m_fas = BlindFASTransceiver(backbone, port_mode="fas_kriging").to(device).eval()
    m_fix = BlindFASTransceiver(backbone, port_mode="fixed_port").to(device).eval()
    m_fas.load_state_dict(
        torch.load(ckpt_root / "dir7a_fas_kriging/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"],
        strict=False,
    )
    m_fix.load_state_dict(
        torch.load(ckpt_root / "dir7a_fixed_port/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"],
        strict=False,
    )
    rec_fix = m_fix.forward_fas(img_01, 3.0, h_port_override=h_1, standard_noise_iq=noise_1)["recon"]
    rec_p32 = forward_pilot32_exhaustive_fas(backbone, img_01, 3.0, h_1, noise_1)["recon"]
    rec_fas = m_fas.forward_fas(img_01, 3.0, h_port_override=h_1, standard_noise_iq=noise_1)["recon"]

    ev_fix = evaluator.evaluate_batch(img_01, rec_fix)
    ev_p32 = evaluator.evaluate_batch(img_01, rec_p32)
    ev_fas = evaluator.evaluate_batch(img_01, rec_fas)

    gs_bot = gs[1, :].subgridspec(1, 4, wspace=0.10)
    vis_items = [
        (img_01[0], "Kodak Ground Truth\n(256x256, CBR=1/16, SNR=3 dB)"),
        (
            rec_fix[0],
            f"Fixed Port 0 (Deep Spatial Fade)\nPSNR {ev_fix['psnr_db'][0]:.2f} dB | DINOv2 {ev_fix['dinov2_cos'][0]:.3f}",
        ),
        (
            rec_p32[0],
            f"32-Pilot Exhaustive FAS (66.7% Pilot)\nPSNR {ev_p32['psnr_db'][0]:.2f} dB | DINOv2 {ev_p32['dinov2_cos'][0]:.3f}",
        ),
        (
            rec_fas[0],
            f"Blind-FAS 6-Anchor Kriging (Ours)\nPSNR {ev_fas['psnr_db'][0]:.2f} dB | DINOv2 {ev_fas['dinov2_cos'][0]:.3f}",
        ),
    ]
    for idx_v, (ten, title_v) in enumerate(vis_items):
        ax_v = fig.add_subplot(gs_bot[0, idx_v])
        ax_v.imshow(_to_hwc(ten))
        ax_v.set_title(title_v, fontsize=9.2, fontweight="bold" if idx_v == 3 else "normal")
        ax_v.axis("off")

    fig.suptitle(
        "Dir-7A: Zero-Pilot 32-Port Fluid / Pixel Antenna JSCC via Bessel-J0 Spatial Wiener-Kriging (Single RF Chain)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    out_p = fig_dir / "fig_dir7_1_blind_fas_kriging.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p


@torch.inference_mode()
def render_fig2_persistguard(
    eval_dir: Path,
    fig_dir: Path,
    device: torch.device,
) -> Path:
    d7b = json.loads((eval_dir / "eval_dir7b_common500.json").read_text(encoding="utf-8"))
    traj = d7b["sample_trajectory_regime_switch"]
    reg_sw = d7b["regimes"]["nonstationary_regime_switch"]["policies"]
    reg_mm = d7b["regimes"]["mismatch_slow_to_fast"]["policies"]
    pref_sum = d7b["prefix_curve_summary"]

    fig = plt.figure(figsize=(18.5, 9.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.12, 0.95], hspace=0.34, wspace=0.22)

    # Panel (a): Live 120-Slot Non-Stationary Regime Switch Trajectory (Persistence Tracking & Outage Cliff Suppression)
    gs_left = gs[0, 0].subgridspec(2, 1, hspace=0.25)
    ax_p = fig.add_subplot(gs_left[0, 0])
    slots = np.asarray(traj["slot"])
    ax_p.plot(slots, traj["true_p"], color="#111111", linestyle="--", linewidth=2.2, label="True Channel Persistence p_t")
    ax_p.plot(slots, traj["est_p_bayes"], color="#d62728", linewidth=2.2, label="Online Joint CQI+NACK Bayes Est.")
    ax_p.set_ylabel("Persistence p_t")
    ax_p.set_ylim(0.38, 1.02)
    ax_p.set_title("(a) Live Non-Stationary Delayed-Feedback Persistence Tracking & Outage Control", fontweight="bold")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(fontsize=8.2, loc="lower left")

    ax_r = fig.add_subplot(gs_left[1, 0])
    # Smooth slot-wise episode-mean rejection slightly (5-slot window) for visual clarity while showing exact means in legend!
    ker = np.ones(5) / 5.0
    rej_slow = np.convolve(np.asarray(traj["slow_lp_rej"]) * 100.0, ker, mode="same")
    rej_pgs = np.convolve(np.asarray(traj["pg_single_rej"]) * 100.0, ker, mode="same")
    rej_pgd = np.convolve(np.asarray(traj["pg_dual_rej"]) * 100.0, ker, mode="same")
    m_slow = reg_sw["slow_lp_582038a"]["rejected_null"]["mean"] * 100.0
    m_pgs = reg_sw["persistguard_single"]["rejected_null"]["mean"] * 100.0
    m_pgd = reg_sw["persistguard_duallayer"]["rejected_null"]["mean"] * 100.0

    ax_r.plot(slots, rej_slow, color="#ff7f0e", linewidth=2.1, label=f"Slow_LP_582038a (Mean Outage={m_slow:.2f}%)")
    ax_r.plot(slots, rej_pgs, color="#1f77b4", linewidth=2.2, label=f"PersistGuard_Single (Mean={m_pgs:.2f}%)")
    ax_r.plot(slots, rej_pgd, color="#2ca02c", linewidth=2.2, label=f"PersistGuard_DualLayer (Mean={m_pgd:.2f}%)")
    ax_r.axhline(2.0, color="#d62728", linestyle=":", linewidth=1.8, label="Target Outage Cap (2.0%)")
    ax_r.set_xlabel("Transmission Frame Index t (2-Bit CQI Delay = 2..3 Slots)")
    ax_r.set_ylabel("Frame Rejection (%)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=7.8, loc="upper right")

    # Panel (b): Risk-Utility Frontier on `common500` across `Non-Stationary Switch` & `Slow->Fast Mismatch`
    ax_b = fig.add_subplot(gs[0, 1])
    pol_meta = [
        ("static_safe", "Static Safe", "#7f7f7f", "s"),
        ("fixed_margin_amc", "Fixed-Margin AMC (-3.5dB)", "#9467bd", "d"),
        ("stale_greedy", "Stale Greedy", "#8c564b", "X"),
        ("slow_lp_582038a", "Slow_LP (582038a, p_des=0.95)", "#ff7f0e", "^"),
        ("persistguard_single", "PersistGuard_Single (Ours)", "#1f77b4", "o"),
        ("persistguard_duallayer", "PersistGuard_DualLayer (Ours)", "#2ca02c", "*"),
    ]
    for p_key, p_lbl, col, mk in pol_meta:
        rx_sw = reg_sw[p_key]["rejected_null"]["mean"] * 100.0
        dy_sw = reg_sw[p_key]["dinov2_cos"]["mean"]
        rx_mm = reg_mm[p_key]["rejected_null"]["mean"] * 100.0
        dy_mm = reg_mm[p_key]["dinov2_cos"]["mean"]
        sz = 160 if mk == "*" else 95
        ax_b.scatter([rx_sw], [dy_sw], color=col, marker=mk, s=sz, zorder=5, label=f"{p_lbl} [Switch: {rx_sw:.1f}%, {dy_sw:.3f}]")
        ax_b.scatter([rx_mm], [dy_mm], color=col, marker=mk, s=sz * 0.65, alpha=0.45, edgecolors="black", zorder=4)
    ax_b.axvline(2.0, color="#d62728", linestyle="--", linewidth=2.0, label="2.0% Outage Constraint")
    ax_b.axvspan(0.0, 2.2, color="#2ca02c", alpha=0.07)
    ax_b.set_title("(b) common500 Outage vs Semantic DINOv2 Frontier (N=1,024, CBR=1/192)", fontweight="bold")
    ax_b.set_xlabel("Realized Grey-Screen Frame Rejection Rate (%) [Solid=Switch, Translucent=Fast Mismatch]")
    ax_b.set_ylabel("Delivered Mean DINOv2-ViT-L/14 Cosine (up)")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=7.6, loc="lower right")

    # Bottom Row: Multi-Scale Digital VAR-d16 Prefix Reconstruction Strip on Kodak
    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    img_k = kodak[8:9].to(device=device, dtype=torch.float32) / 255.0
    vae, var_d16 = load_frozen_vae_and_var_d16(device=device, load_var=True)
    gt_idx = vae.img_to_idxBl(img_k * 2.0 - 1.0)
    r_m0 = reconstruct_var_from_prefix(vae, var_d16, gt_idx, 0)[0]
    r_m4 = reconstruct_var_from_prefix(vae, var_d16, gt_idx, 4)[0]
    r_m7 = reconstruct_var_from_prefix(vae, var_d16, gt_idx, 7)[0]
    r_m10 = reconstruct_var_from_prefix(vae, var_d16, gt_idx, 10)[0]

    gs_bot = gs[1, :].subgridspec(1, 5, wspace=0.08)
    strips = [
        (img_k[0], "Kodak Ground Truth\n(256x256, N=1,024 Coded Syms)"),
        (
            r_m0,
            f"m=0: CRC Rejected NULL\ncommon500 DINOv2={pref_sum['m_0']['dinov2_cos']:.3f}\n(Slow_LP hits {m_slow:.1f}% NULLs!)",
        ),
        (
            r_m4,
            f"m=4: Base Anchor (30 tok, QPSK r=1/4)\nPSNR={pref_sum['m_4']['psnr_db']:.1f} dB | DINOv2={pref_sum['m_4']['dinov2_cos']:.3f}\n(DualLayer Fallback: {m_pgd:.2f}% NULLs!)",
        ),
        (
            r_m7,
            f"m=7: Mid Prefix (155 tok, QPSK r=2/3)\nPSNR={pref_sum['m_7']['psnr_db']:.1f} dB | DINOv2={pref_sum['m_7']['dinov2_cos']:.3f}",
        ),
        (
            r_m10,
            f"m=10: Full Prefix (680 tok, 64-QAM)\nPSNR={pref_sum['m_10']['psnr_db']:.1f} dB | DINOv2={pref_sum['m_10']['dinov2_cos']:.3f}",
        ),
    ]
    for i_s, (ten_s, lbl_s) in enumerate(strips):
        ax_s = fig.add_subplot(gs_bot[0, i_s])
        ax_s.imshow(_to_hwc(ten_s))
        ax_s.set_title(lbl_s, fontsize=8.5, fontweight="bold" if i_s in (2, 4) else "normal")
        ax_s.axis("off")

    fig.suptitle(
        "Dir-7B: Solving Non-Stationary Delayed-Feedback Outage Cliffs in Digital Multi-Scale VAR Transmission",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    out_p = fig_dir / "fig_dir7_2_persistguard_digital_outage.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p


@torch.inference_mode()
def render_fig3_impulse_shield(
    eval_dir: Path,
    ckpt_root: Path,
    fig_dir: Path,
    backbone: torch.nn.Module,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> Path:
    d7c = json.loads((eval_dir / "eval_dir7c_common500.json").read_text(encoding="utf-8"))
    conds = d7c["conditions"]
    cross2 = d7c["cross_switch_2x2"]["imp08_sir9_snr14"]

    fig = plt.figure(figsize=(19.2, 10.0), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.31)

    # Panel (a): PSNR across Clean AWGN -> 4% -> 6% -> 8% -> 12% OOD Impulse Density
    ax_a = fig.add_subplot(gs[0, 0])
    c_order = [
        ("Clean\n0%", "clean_awgn_snr14"),
        ("4% Imp\n-6dB", "imp04_sir6_snr12"),
        ("6% Imp\n-7dB", "imp06_sir7_snr12"),
        ("8% Imp\n-9dB", "imp08_sir9_snr14"),
        ("12% OOD\n-10dB", "imp12_sir10_snr12_ood"),
    ]
    xs = np.arange(len(c_order))
    lines = [
        ("sem_impulse_2round", "SemImpulse-2Round (Proposed)", "#d62728", "-", "o"),
        ("huber_owndenoise", "Huber-OwnDenoise (Byte-Id. Control)", "#1f77b4", "--", "s"),
        ("classical_blanking", "Trained Classical Blanking", "#2ca02c", "-.", "^"),
        ("standard_blind", "Trained Standard Blind OFDM", "#ff7f0e", ":", "v"),
        ("oracle_genie_csi_ofdm", "Oracle Genie-CSI Standard OFDM", "#555555", "--", "d"),
    ]
    for mk, mlbl, col, ls, mrk in lines:
        ys = [conds[ck]["models"][mk]["psnr_db"]["mean"] for _, ck in c_order]
        ax_a.plot(xs, ys, label=mlbl, color=col, linestyle=ls, marker=mrk, linewidth=2.2, markersize=6)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([c[0] for c in c_order], fontsize=8.8)
    ax_a.set_ylabel("common500 Mean PSNR (dB)")
    ax_a.set_title("(a) PSNR vs Impulse Density (CBR=1/16)", fontweight="bold")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=7.6, loc="lower left")

    # Panel (b): 2x2 Weight x Fast-Time Peeling Loop Cross-Switch & DINOv2 / DISTS Comparison
    ax_b = fig.add_subplot(gs[0, 1])
    x_keys = [
        ("w_huber_owndenoise__loop_huber_owndenoise", "w_Own +\nLatent-Only"),
        ("w_sem_impulse_2round__loop_huber_owndenoise", "w_Sem +\nLatent-Only"),
        ("w_huber_owndenoise__loop_sem_impulse_2round", "w_Own +\nFastTime-Peel"),
        ("w_sem_impulse_2round__loop_sem_impulse_2round", "w_Sem +\nFastTime-Peel\n(Proposed)"),
    ]
    bx = np.arange(len(x_keys))
    bvals = [cross2[k]["psnr_db"]["mean"] for k, _ in x_keys]
    bcols = ["#1f77b4", "#6baed6", "#ff7f0e", "#d62728"]
    bars = ax_b.bar(bx, bvals, width=0.55, color=bcols, alpha=0.92)
    for r, (k, _) in zip(bars, x_keys):
        h = r.get_height()
        dn = cross2[k]["dinov2_cos"]["mean"]
        ax_b.text(r.get_x() + r.get_width() / 2.0, h + 0.12, f"{h:.2f} dB\n(DINO {dn:.3f})", ha="center", va="bottom", fontsize=7.6)
    ax_b.set_xticks(bx)
    ax_b.set_xticklabels([lbl for _, lbl in x_keys], fontsize=8.4)
    ax_b.set_ylim(min(bvals) - 1.5, max(bvals) + 1.4)
    ax_b.set_ylabel("common500 PSNR (dB) @ 8% Imp, SIR=-9dB")
    ax_b.set_title("(b) 2x2 Weight x Peeling Loop Audit", fontweight="bold")
    ax_b.grid(True, axis="y", alpha=0.3)

    # Panel (c): Physical Fast-Time Envelope vs Signal-Cancelled Impulse Residual on Kodak
    ax_c = fig.add_subplot(gs[0, 2])
    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    img_01 = kodak[7:8].to(device=device, dtype=torch.float32) / 255.0

    m_sem = ImpulseShieldTransceiver(backbone, rx_mode="sem_impulse_2round").to(device).eval()
    m_own = ImpulseShieldTransceiver(backbone, rx_mode="huber_owndenoise").to(device).eval()
    m_blk = ImpulseShieldTransceiver(backbone, rx_mode="classical_blanking").to(device).eval()
    m_sem.load_state_dict(
        torch.load(ckpt_root / "dir7c_sem_impulse_2round/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"],
        strict=False,
    )
    m_own.load_state_dict(
        torch.load(ckpt_root / "dir7c_huber_owndenoise/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"],
        strict=False,
    )
    m_blk.load_state_dict(
        torch.load(ckpt_root / "dir7c_classical_blanking/seed_1001/best_indist.pt", map_location="cpu", weights_only=True)["model_state"],
        strict=False,
    )

    gen_c = torch.Generator(device="cpu").manual_seed(2026799)
    hr_1, hi_1 = _make_det_rbs(1, 0.88, gen_c)
    aw_1 = torch.randn((1, N_SYM, N_SC, 2), generator=gen_c)
    im_1 = _make_det_impulses(1, 0.06, -7.0, gen_c)

    # Trace physical fast-time symbol on an impulsed symbol
    stem_1 = m_sem.extract_encoder_stem(img_01)
    snr_c0 = torch.full((1, 1), m_sem.base_c0_db, device=device)
    enc = m_sem.backbone.encoder
    temp = None
    for i in range(enc.layer_num):
        temp = enc.sm_list[i](stem_1) if i == 0 else enc.sm_list[i](temp)
        temp = temp * enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, 256, -1)
    z_tx = m_sem.hier_tx(enc.head_list(stem_1 * enc.sigmoid(enc.sm_list[-1](temp))), stem_1)
    x_tf_c, _ = m_sem.modem.embed_tx(z_tx)
    r_rx_td, phy_d = physical_impulsive_ofdm_channel(
        x_tf_c, 12.0, p_imp=0.06, imp_sir_db=-7.0, h_rb_override=(hr_1.to(device), hi_1.to(device)), awgn_td_override=aw_1.to(device), impulse_td_override=im_1.to(device)
    )
    imp_mag_all = phy_d["imp_td"][0].abs().cpu().numpy()
    sym_pick = int(np.argmax(imp_mag_all.max(axis=-1)))
    y_clean_sym = (r_rx_td[0, sym_pick] - phy_d["imp_td"][0, sym_pick]).abs().cpu().numpy()
    r_raw_sym = r_rx_td[0, sym_pick].abs().cpu().numpy()
    imp_true_sym = imp_mag_all[sym_pick]

    n_idx = np.arange(128)
    ax_c.plot(n_idx, r_raw_sym[:128], color="#1f77b4", alpha=0.75, linewidth=1.3, label="Raw RX |r[n]| (Signal + Impulses)")
    ax_c.plot(n_idx, y_clean_sym[:128], color="#2ca02c", alpha=0.80, linewidth=1.3, label="OFDM Signal Envelope |y_clean[n]|")
    ax_c.plot(n_idx, imp_true_sym[:128], color="#d62728", linestyle="--", linewidth=1.8, label="Overlapped Impulses |e_imp[n]|")
    ax_c.set_title(f"(c) Symbol #{sym_pick} Envelope vs Overlapped Impulses", fontweight="bold")
    ax_c.set_xlabel("Fast-Time Sample Index n (Before RX FFT)")
    ax_c.set_ylabel("Complex Sample Magnitude")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=7.6, loc="upper right")

    # Bottom Row: 5-Column Kodak Visual Comparison at `p_imp=6%, SIR=-7 dB, SNR=12 dB` (Overlapped PAPR Regime!)
    rec_gen = forward_genie_csi_impulsive_ofdm(
        backbone, img_01, 12.0, p_imp=0.06, imp_sir_db=-7.0, h_rb_override=(hr_1.to(device), hi_1.to(device)), awgn_td_override=aw_1.to(device), impulse_td_override=im_1.to(device)
    )["recon"]
    rec_blk = m_blk.forward_impulsive(
        stem_1, 12.0, p_imp=0.06, imp_sir_db=-7.0, h_rb_override=(hr_1.to(device), hi_1.to(device)), awgn_td_override=aw_1.to(device), impulse_td_override=im_1.to(device), input_is_stem=True
    )["recon"]
    rec_own = m_own.forward_impulsive(
        stem_1, 12.0, p_imp=0.06, imp_sir_db=-7.0, h_rb_override=(hr_1.to(device), hi_1.to(device)), awgn_td_override=aw_1.to(device), impulse_td_override=im_1.to(device), input_is_stem=True
    )["recon"]
    rec_sem = m_sem.forward_impulsive(
        stem_1, 12.0, p_imp=0.06, imp_sir_db=-7.0, h_rb_override=(hr_1.to(device), hi_1.to(device)), awgn_td_override=aw_1.to(device), impulse_td_override=im_1.to(device), input_is_stem=True
    )["recon"]

    ev_gen = evaluator.evaluate_batch(img_01, rec_gen)
    ev_blk = evaluator.evaluate_batch(img_01, rec_blk)
    ev_own = evaluator.evaluate_batch(img_01, rec_own)
    ev_sem = evaluator.evaluate_batch(img_01, rec_sem)

    gs_bot = gs[1, :].subgridspec(1, 5, wspace=0.08)
    cols_v = [
        (img_01[0], "Kodak Ground Truth\n(6% Overlapped Impulse, SIR=-7dB)"),
        (
            rec_gen[0],
            f"Oracle Genie-CSI Standard OFDM\nPSNR {ev_gen['psnr_db'][0]:.2f} dB | DISTS {ev_gen['dists'][0]:.3f}",
        ),
        (
            rec_blk[0],
            f"Trained Classical Blanking\nPSNR {ev_blk['psnr_db'][0]:.2f} dB | DISTS {ev_blk['dists'][0]:.3f}",
        ),
        (
            rec_own[0],
            f"Byte-Identical Huber-OwnDenoise\nPSNR {ev_own['psnr_db'][0]:.2f} dB | DISTS {ev_own['dists'][0]:.3f}",
        ),
        (
            rec_sem[0],
            f"SemImpulse-2Round (Proposed)\nPSNR {ev_sem['psnr_db'][0]:.2f} dB | DISTS {ev_sem['dists'][0]:.3f}",
        ),
    ]
    for i_c, (ten_c, lbl_c) in enumerate(cols_v):
        ax_v = fig.add_subplot(gs_bot[0, i_c])
        ax_v.imshow(_to_hwc(ten_c))
        ax_v.set_title(lbl_c, fontsize=8.6, fontweight="bold" if i_c == 4 else "normal")
        ax_v.axis("off")

    fig.suptitle(
        "Dir-7C: Null-Gated Decision-Directed Fast-Time Impulsive Interference Cancellation vs Byte-Identical Control",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    out_p = fig_dir / "fig_dir7_3_impulse_shield_qualitative.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--fig_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt_root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir7_fas_persist_impulse_v1"),
    )
    parser.add_argument(
        "--tmp_root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir7_fas_persist_impulse_v1"),
    )
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    admit_gpu(min_free_mib=22000)
    device = torch.device("cuda:0")
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    evaluator = Full5MetricEvaluator(device)

    p1 = render_fig1_blind_fas(args.eval_dir, args.ckpt_root, args.fig_dir, backbone, evaluator, device)
    print(f"[Fig1] Rendered {p1}", flush=True)
    p2 = render_fig2_persistguard(args.eval_dir, args.fig_dir, device)
    print(f"[Fig2] Rendered {p2}", flush=True)
    p3 = render_fig3_impulse_shield(args.eval_dir, args.ckpt_root, args.fig_dir, backbone, evaluator, device)
    print(f"[Fig3] Rendered {p3}", flush=True)

    verify_no_links([args.ckpt_root, args.tmp_root, args.fig_dir])


if __name__ == "__main__":
    main()
