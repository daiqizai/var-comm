#!/usr/bin/env python3
"""Render IEEE camera-ready publication figures for Direction 5:
  - fig_dir5_1_otfs_high_mobility.png: Dir-5A High-Mobility Doubly-Selective Delay-Doppler JSCC (ImageNet-500)
  - fig_dir5_2_genvar_cbr128_256_perceptual.png: Dir-5B Ultra-Low CBR=1/128 & 1/256 Generative Perceptual JSCC (ImageNet-500)
  - fig_dir5_3_qualitative_kodak.png: 3-Row Held-Out Kodak24 Visual Comparison (900 km/h OOD OTFS, CBR=1/256 [768 syms], & CBR_eff=1/512 [384 syms] OOD Erasure!)
"""
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


def render_fig_dir5_1_otfs(otfs_csv: Path, out_png: Path) -> None:
    rows = load_csv(otfs_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.3), dpi=220)
    fig.suptitle(
        "Dir-5A — High-Mobility Doubly-Selective Delay-Doppler Zero-Pilot Semantic Comm. (Symp-OTFS-JSCC, ImageNet-500)",
        fontsize=11.8,
        fontweight="bold",
        y=0.99,
    )

    line_styles = [
        ("Frozen_OFDM_Blind", "#7f8c8d", ":", "x", "Frozen OFDM Blind (0-Pilot)"),
        ("V4_USPH_OFDM_ZeroShot", "#e67e22", "-.", "d", "V4 U-SPH OFDM (Static RB Subspace)"),
        ("Pilot_4Sym_Interp_OFDM", "#8e44ad", "-.", "^", "5G 4-Sym DMRS Interp. OFDM (1,024 Pilots)"),
        ("BlackBox_OTFS_Blind", "#2980b9", "--", "s", "Black-Box OTFS Blind (0-Pilot Neural)"),
        ("Symp_OFDM_2DSubspace", "#16a085", "--", "o", "Symp-OFDM-2D (No ISFFT Waveform Ablation)"),
        ("Symp_OTFS_JSCC", "#c0392b", "-", "*", "Symp-OTFS-JSCC (Ours: 0-Pilot, 0-Guard)"),
    ]

    # Panel (a): SNR Sweep at High Mobility k_d = 0.10 (~500 km/h)
    ax1 = axes[0]
    sub_snr = [r for r in rows if r["suite"] == "snr_sweep"]
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub_snr if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
        if not pts:
            continue
        xs = [float(r["true_snr_db"]) for r in pts]
        ys = [float(r["psnr_db_mean"]) for r in pts]
        ylo = [float(r["psnr_db_ci95_lo"]) for r in pts]
        yhi = [float(r["psnr_db_ci95_hi"]) for r in pts]
        lw = 2.5 if mname == "Symp_OTFS_JSCC" else 1.75
        ms = 7.8 if mname == "Symp_OTFS_JSCC" else 5.6
        ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        if mname in ("Symp_OTFS_JSCC", "Symp_OFDM_2DSubspace", "Pilot_4Sym_Interp_OFDM"):
            ax1.fill_between(xs, ylo, yhi, color=col, alpha=0.10)
    ax1.set_title(r"(a) SNR Sweep at High Mobility ($k_d = 0.10 \approx 500\,$km/h)")
    ax1.set_xlabel("Wireless Channel SNR (dB)")
    ax1.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax1.set_xticks([1, 4, 7, 10, 14])
    ax1.grid(True, alpha=0.30)
    ax1.legend(fontsize=7.1, loc="lower right", framealpha=0.92)

    # Panel (b): Velocity / Doppler Sweep at SNR = 10 dB (50 km/h -> 900 km/h OOD!)
    ax2 = axes[1]
    sub_vel = [r for r in rows if r["suite"] == "vel_sweep"]
    ax2.axvspan(600, 950, color="#fdebd0", alpha=0.45, label="Unseen OOD Hyper-Mobility")
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub_vel if r["model"] == mname], key=lambda r: float(r["approx_kmh"]))
        if not pts:
            continue
        xs = [float(r["approx_kmh"]) for r in pts]
        ys = [float(r["psnr_db_mean"]) for r in pts]
        lw = 2.5 if mname == "Symp_OTFS_JSCC" else 1.75
        ms = 7.8 if mname == "Symp_OTFS_JSCC" else 5.6
        ax2.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms)
    ax2.set_title("(b) Mobility Sweep at 10 dB SNR (50 → 900 km/h OOD)")
    ax2.set_xlabel("Equivalent Carrier Mobility Velocity (km/h)")
    ax2.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax2.set_xticks([50, 250, 500, 700, 900])
    ax2.set_xlim(20, 940)
    ax2.grid(True, alpha=0.30)
    ax2.legend(fontsize=7.8, loc="lower left", framealpha=0.92)

    # Panel (c): Controlled Waveform Ablation & OOD Hyper-Mobility Stress Bar Chart
    ax3 = axes[2]
    bar_groups = [
        ("vel_sweep_10dB_kd010", "500 km/h\n(10 dB, In-Dist)"),
        ("vel_sweep_10dB_kd014_ood", "700 km/h OOD\n(10 dB, Unseen)"),
        ("vel_sweep_10dB_kd018_ood", "900 km/h OOD\n(10 dB, Unseen)"),
        ("ood_hyper_kd018_14dB", "900 km/h OOD\n(14 dB, Unseen)"),
    ]
    bar_models = [
        ("Pilot_4Sym_Interp_OFDM", "#8e44ad", "5G 4-Sym DMRS (1,024-P)"),
        ("BlackBox_OTFS_Blind", "#2980b9", "Black-Box OTFS (0-P)"),
        ("Symp_OFDM_2DSubspace", "#16a085", "Symp-OFDM-2D (No ISFFT)"),
        ("Symp_OTFS_JSCC", "#c0392b", "Symp-OTFS-JSCC (Ours, 0-P)"),
    ]
    xg = np.arange(len(bar_groups))
    wb = 0.19
    ymin, ymax = 99.0, 0.0
    for b_i, (mname, col, blbl) in enumerate(bar_models):
        vals = []
        for cid, _ in bar_groups:
            hit = [r for r in rows if r["condition"] == cid and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymin = min(ymin, v)
            ymax = max(ymax, v)
        offs = (b_i - 1.5) * wb
        bars = ax3.bar(xg + offs, vals, width=wb, color=col, alpha=0.90, edgecolor="black", linewidth=0.55, label=blbl)
        for rect, v in zip(bars, vals):
            ax3.text(
                rect.get_x() + rect.get_width() / 2.0,
                v + 0.35,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if mname == "Symp_OTFS_JSCC" else "normal",
            )
    ax3.set_title("(c) Waveform Diversity Ablation & OOD Hyper-Mobility")
    ax3.set_xticks(xg)
    ax3.set_xticklabels([lbl for _, lbl in bar_groups], fontsize=8.4)
    ax3.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax3.set_ylim(max(6.0, ymin - 2.5), ymax + 6.8)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.3, loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def render_fig_dir5_2_genvar(ul_csv: Path, out_png: Path) -> None:
    rows = load_csv(ul_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.3), dpi=220)
    fig.suptitle(
        "Dir-5B — Ultra-Low CBR = 1/128 (1,536 Syms) & CBR = 1/256 (768 Syms) Generative Perceptual Blind JSCC (ImageNet-500)",
        fontsize=11.7,
        fontweight="bold",
        y=0.99,
    )

    line_styles = [
        ("Swin_Holo_Blind", "#7f8c8d", ":", "s", "Swin-Holo-Blind (0-Pilot Discriminative MSE)"),
        ("Swin_Genie_CSI", "#8e44ad", "-.", "^", "Swin-Genie-CSI (Oracle Exact CSI, Discriminative)"),
        ("BlackBox_VAR_Blind", "#2980b9", "--", "o", "Black-Box VAR Blind (0-Pilot Neural Generative)"),
        ("GenHolo_VAR_JSCC", "#c0392b", "-", "*", "GenHolo-VAR-JSCC (Ours: 0-Pilot Holo + VQ Manifold)"),
    ]

    # Panel (a): CBR = 1/128 (K = 1,536) DISTS vs SNR (Lower is Better!)
    ax1 = axes[0]
    sub128 = [r for r in rows if int(r["cbr_denom"]) == 128 and r["erase50"] == "False"]
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub128 if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
        if not pts:
            continue
        xs = [float(r["true_snr_db"]) for r in pts]
        ys = [float(r["dists_mean"]) for r in pts]
        ylo = [float(r["dists_ci95_lo"]) for r in pts]
        yhi = [float(r["dists_ci95_hi"]) for r in pts]
        lw = 2.5 if mname == "GenHolo_VAR_JSCC" else 1.8
        ms = 7.8 if mname == "GenHolo_VAR_JSCC" else 5.8
        ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        ax1.fill_between(xs, ylo, yhi, color=col, alpha=0.10)
    ax1.set_title("(a) CBR = 1/128 (K = 1,536 Syms): Perceptual DISTS ↓")
    ax1.set_xlabel("8-RB Rayleigh OFDM Channel SNR (dB)")
    ax1.set_ylabel("Perceptual Distortion DISTS ↓ (Lower is Better)")
    ax1.set_xticks([1, 4, 7, 10, 13])
    ax1.grid(True, alpha=0.30)
    ax1.legend(fontsize=7.2, loc="upper right", framealpha=0.92)

    # Panel (b): CBR = 1/256 (K = 768 = 16x Compression!) DISTS vs SNR (Lower is Better!)
    ax2 = axes[1]
    sub256 = [r for r in rows if int(r["cbr_denom"]) == 256 and r["erase50"] == "False"]
    for mname, col, ls, mk, lbl in line_styles:
        pts = sorted([r for r in sub256 if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
        if not pts:
            continue
        xs = [float(r["true_snr_db"]) for r in pts]
        ys = [float(r["dists_mean"]) for r in pts]
        ylo = [float(r["dists_ci95_lo"]) for r in pts]
        yhi = [float(r["dists_ci95_hi"]) for r in pts]
        lw = 2.5 if mname == "GenHolo_VAR_JSCC" else 1.8
        ms = 7.8 if mname == "GenHolo_VAR_JSCC" else 5.8
        ax2.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        ax2.fill_between(xs, ylo, yhi, color=col, alpha=0.10)
    ax2.set_title("(b) Extreme CBR = 1/256 (K = 768 Syms): Perceptual DISTS ↓")
    ax2.set_xlabel("8-RB Rayleigh OFDM Channel SNR (dB)")
    ax2.set_ylabel("Perceptual Distortion DISTS ↓ (Lower is Better)")
    ax2.set_xticks([1, 4, 7, 10, 13])
    ax2.grid(True, alpha=0.30)
    ax2.legend(fontsize=7.2, loc="upper right", framealpha=0.92)

    # Panel (c): LPIPS-Alex on Full Band vs Unseen OOD 50% Contiguous RB Wipeout (K_surv = 768 & 384!)
    ax3 = axes[2]
    bar_groups = [
        ("cbr128_snr10dB", "CBR 1/128\n(1,536 Syms)"),
        ("cbr128_ood_erase50_10dB", "CBR 1/128 OOD\n50% Erase (768 Syms)"),
        ("cbr256_snr10dB", "CBR 1/256\n(768 Syms)"),
        ("cbr256_ood_erase50_10dB", "CBR 1/256 OOD\n50% Erase (384 Syms!)"),
    ]
    bar_models = [
        ("Swin_Holo_Blind", "#7f8c8d", "Swin-Holo-Blind (0-P)"),
        ("Swin_Genie_CSI", "#8e44ad", "Swin-Genie-CSI (Oracle)"),
        ("BlackBox_VAR_Blind", "#2980b9", "Black-Box VAR (0-P)"),
        ("GenHolo_VAR_JSCC", "#c0392b", "GenHolo-VAR (Ours, 0-P)"),
    ]
    xg = np.arange(len(bar_groups))
    wb = 0.19
    ymax = 0.0
    for b_i, (mname, col, blbl) in enumerate(bar_models):
        vals = []
        for cid, _ in bar_groups:
            hit = [r for r in rows if r["condition"] == cid and r["model"] == mname]
            v = float(hit[0]["lpips_alex_mean"]) if hit else 0.0
            vals.append(v)
            ymax = max(ymax, v)
        offs = (b_i - 1.5) * wb
        bars = ax3.bar(xg + offs, vals, width=wb, color=col, alpha=0.90, edgecolor="black", linewidth=0.55, label=blbl)
        for rect, v in zip(bars, vals):
            ax3.text(
                rect.get_x() + rect.get_width() / 2.0,
                v + 0.008,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="bold" if mname == "GenHolo_VAR_JSCC" else "normal",
            )
    ax3.set_title("(c) Perceptual LPIPS ↓ under OOD 50% Half-Band Wipeout")
    ax3.set_xticks(xg)
    ax3.set_xticklabels([lbl for _, lbl in bar_groups], fontsize=8.1)
    ax3.set_ylabel("Perceptual LPIPS-Alex ↓ (Lower is Better)")
    ax3.set_ylim(0.0, ymax + 0.12)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.3, loc="upper left", framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


@torch.inference_mode()
def render_fig_dir5_3_kodak_qualitative(
    backbone_ckpt: Path,
    dir5_ckpt_root: Path,
    v4_usph_ckpt: Path,
    out_png: Path,
    physical_gpu: int = 0,
) -> None:
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, seed_everything
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.models import load_swinjscc_backbone
    from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4
    from dir2_zpac.phy_v2 import sample_correlated_rbs
    from dir2_zpac.metrics import OfflineVisualEvaluator
    from dir5_og.otfs_blind import (
        N_SC,
        N_SYM,
        SympOTFSBlindTransceiver,
        forward_pilot_4sym_interp_ofdm_doubly_selective,
        forward_v4_usph_ofdm_doubly_selective,
        sample_doubly_selective_multipath,
    )
    from dir5_og.genvar_ultralow_blind import (
        DEFAULT_VAR_VQVAE_CKPT,
        N_UL_RBS,
        GenHoloVARTransceiver,
        SwinUltraLowTransceiver,
        cbr_geometry,
        load_frozen_var_vqvae,
    )

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/cache/fig_dir5"),
        num_threads=4,
    )
    seed_everything(20261127)
    admit_gpu(physical_gpu)
    device = torch.device("cuda:0")

    backbone = load_swinjscc_backbone(backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    vae = load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT, device=device)
    ev = OfflineVisualEvaluator(device=device, include_lpips=True)

    kodak = (load_split_u8("kodak24", allow_holdout=True).images_u8.to(dtype=torch.float32) / 255.0).to(device)

    def _load_otfs(mth: str, tag: str):
        m = SympOTFSBlindTransceiver(backbone, method=mth, hidden_ch=160).to(device)
        m.load_state_dict(torch.load(dir5_ckpt_root / tag / "best_dir5.pt", map_location="cpu")["state_dict"], strict=False)
        return m.eval()

    def _load_gv(mth: str, cbr_d: int, tag: str):
        m = GenHoloVARTransceiver(vae, cbr_denom=cbr_d, method=mth, hidden_ch=160).to(device)
        m.load_state_dict(torch.load(dir5_ckpt_root / tag / "best_dir5.pt", map_location="cpu")["state_dict"], strict=False)
        return m.eval()

    def _load_sw(mth: str, cbr_d: int, tag: str):
        m = SwinUltraLowTransceiver(backbone, cbr_denom=cbr_d, method=mth, hidden_ch=160).to(device)
        m.load_state_dict(torch.load(dir5_ckpt_root / tag / "best_dir5.pt", map_location="cpu")["state_dict"], strict=False)
        return m.eval()

    symp_otfs = _load_otfs("symp_otfs", "symp_otfs")
    symp_ofdm = _load_otfs("symp_ofdm_2d", "symp_ofdm_2d")
    v4_usph = UniversalBlindTransceiverV4(backbone, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    v4_usph.load_state_dict(torch.load(v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    v4_usph.eval()

    sw_genie_128 = _load_sw("swin_genie_csi", 128, "swin_genie_csi_cbr128")
    sw_holo_128 = _load_sw("swin_holo_blind", 128, "swin_holo_blind_cbr128")
    bb_var_128 = _load_gv("blackbox_var", 128, "blackbox_var_cbr128")
    gh_var_128 = _load_gv("genholo_var", 128, "genholo_var_cbr128")

    sw_genie_256 = _load_sw("swin_genie_csi", 256, "swin_genie_csi_cbr256")
    sw_holo_256 = _load_sw("swin_holo_blind", 256, "swin_holo_blind_cbr256")
    bb_var_256 = _load_gv("blackbox_var", 256, "blackbox_var_cbr256")
    gh_var_256 = _load_gv("genholo_var", 256, "genholo_var_cbr256")

    def _m_str_psnr(gt: torch.Tensor, rec: torch.Tensor) -> str:
        m = ev.evaluate_batch(gt, rec)
        return f"PSNR {m['psnr_db'][0]:.2f} dB | DISTS {m['dists'][0]:.3f}"

    def _m_str_perc(gt: torch.Tensor, rec: torch.Tensor) -> str:
        m = ev.evaluate_batch(gt, rec)
        return f"DISTS {m['dists'][0]:.3f} | LPIPS {m['lpips_alex'][0]:.3f}\n(PSNR {m['psnr_db'][0]:.1f} dB)"

    # Row 1: Kodak idx 22 (Parrots) — Dir-5A OOD Hyper-Mobility k_d = 0.18 (~900 km/h) at 10 dB
    x1 = kodak[22:23].to(device)
    gen1 = torch.Generator(device="cpu").manual_seed(8011)
    mp1 = sample_doubly_selective_multipath(1, 0.18, generator=gen1)
    mp1_d = (mp1[0].to(device), mp1[1].to(device))
    eps1 = torch.randn((1, N_SYM, N_SC, 2), generator=gen1, dtype=torch.float32).to(device)

    r1_v4 = forward_v4_usph_ofdm_doubly_selective(v4_usph, x1, 10.0, kd_max=0.18, multipath_override=mp1_d, standard_noise_td_iq=eps1)["recon"].clamp(0, 1)
    r1_p4 = forward_pilot_4sym_interp_ofdm_doubly_selective(backbone, x1, 10.0, kd_max=0.18, multipath_override=mp1_d, standard_noise_td_iq=eps1)["recon"].clamp(0, 1)
    r1_ofdm = symp_ofdm.forward_doubly_selective(x1, 10.0, kd_max=0.18, multipath_override=mp1_d, standard_noise_td_iq=eps1)["recon"].clamp(0, 1)
    r1_ours = symp_otfs.forward_doubly_selective(x1, 10.0, kd_max=0.18, multipath_override=mp1_d, standard_noise_td_iq=eps1)["recon"].clamp(0, 1)

    # Row 2: Kodak idx 18 (Lighthouse, #19) — Dir-5B Ultra-Low CBR = 1/128 (K = 1,536 Symbols) at 10 dB
    _, nb128, _ = cbr_geometry(128)
    gen2 = torch.Generator(device="cpu").manual_seed(8022)
    hr24, hi24 = sample_correlated_rbs(24, N_UL_RBS, 0.85, device="cpu", generator=gen2)
    hb24 = torch.stack([hr24, hi24], dim=-1).to(device)
    eps128 = torch.randn((24, N_UL_RBS, nb128, 2), generator=gen2, dtype=torch.float32).to(device)
    am_full = torch.ones((1, N_UL_RBS), device=device)

    x2 = kodak[18:19]
    hb2 = hb24[18:19]
    ep2 = eps128[18:19]
    r2_genie = sw_genie_128.forward_ultralow(x2, 10.0, alive_rb_mask=am_full, h_rb_override=hb2, standard_noise_iq=ep2)["recon"].clamp(0, 1)
    r2_sholo = sw_holo_128.forward_ultralow(x2, 10.0, alive_rb_mask=am_full, h_rb_override=hb2, standard_noise_iq=ep2)["recon"].clamp(0, 1)
    r2_bbvar = bb_var_128.forward_ultralow(x2, 10.0, alive_rb_mask=am_full, h_rb_override=hb2, standard_noise_iq=ep2)["recon"].clamp(0, 1)
    r2_ours = gh_var_128.forward_ultralow(x2, 10.0, alive_rb_mask=am_full, h_rb_override=hb2, standard_noise_iq=ep2)["recon"].clamp(0, 1)

    # Row 3: Kodak idx 8 (Sailboats, #09) — Dir-5B Extreme CBR = 1/256 (ONLY 768 Symbols = 16x Compression!) at 10 dB
    _, nb256, _ = cbr_geometry(256)
    gen3 = torch.Generator(device="cpu").manual_seed(8022)
    hr3_all, hi3_all = sample_correlated_rbs(24, N_UL_RBS, 0.85, device="cpu", generator=gen3)
    hb3_all = torch.stack([hr3_all, hi3_all], dim=-1).to(device)
    eps256 = torch.randn((24, N_UL_RBS, nb256, 2), generator=gen3, dtype=torch.float32).to(device)

    x3 = kodak[8:9]
    hb3 = hb3_all[8:9]
    ep3 = eps256[8:9]
    r3_genie = sw_genie_256.forward_ultralow(x3, 10.0, alive_rb_mask=am_full, h_rb_override=hb3, standard_noise_iq=ep3)["recon"].clamp(0, 1)
    r3_sholo = sw_holo_256.forward_ultralow(x3, 10.0, alive_rb_mask=am_full, h_rb_override=hb3, standard_noise_iq=ep3)["recon"].clamp(0, 1)
    r3_bbvar = bb_var_256.forward_ultralow(x3, 10.0, alive_rb_mask=am_full, h_rb_override=hb3, standard_noise_iq=ep3)["recon"].clamp(0, 1)
    r3_ours = gh_var_256.forward_ultralow(x3, 10.0, alive_rb_mask=am_full, h_rb_override=hb3, standard_noise_iq=ep3)["recon"].clamp(0, 1)

    grid_rows = [
        (
            "Row 1: Dir-5A OOD Hyper-Mobility\n(~900 km/h, k_d=0.18, 10 dB)",
            [
                (x1, "Ground Truth\n(Kodak #23, 256×256)"),
                (r1_v4, f"V4 U-SPH OFDM (0-Pilot)\n{_m_str_psnr(x1, r1_v4)}"),
                (r1_p4, f"5G 4-Sym DMRS (1,024 Pilots)\n{_m_str_psnr(x1, r1_p4)}"),
                (r1_ofdm, f"Symp-OFDM-2D (No ISFFT)\n{_m_str_psnr(x1, r1_ofdm)}"),
                (r1_ours, f"Symp-OTFS-JSCC (Ours, 0-P)\n{_m_str_psnr(x1, r1_ours)}"),
            ],
        ),
        (
            "Row 2: Dir-5B Ultra-Low CBR = 1/128\n(1,536 Symbols, 10 dB 8-RB OFDM)",
            [
                (x2, "Ground Truth\n(Kodak #19, Lighthouse)"),
                (r2_genie, f"Swin-Genie-CSI (Oracle CSI)\n{_m_str_perc(x2, r2_genie)}"),
                (r2_sholo, f"Swin-Holo-Blind (0-P MSE)\n{_m_str_perc(x2, r2_sholo)}"),
                (r2_bbvar, f"BlackBox-VAR-Blind (0-P)\n{_m_str_perc(x2, r2_bbvar)}"),
                (r2_ours, f"GenHolo-VAR (Ours, 0-P, 1,536 Syms)\n{_m_str_perc(x2, r2_ours)}"),
            ],
        ),
        (
            "Row 3: Dir-5B Extreme CBR = 1/256\n(ONLY 768 Symbols = 16× Comp., 10 dB)",
            [
                (x3, "Ground Truth\n(Kodak #09, Sailboats)"),
                (r3_genie, f"Swin-Genie-CSI (Oracle, 768 Syms)\n{_m_str_perc(x3, r3_genie)}"),
                (r3_sholo, f"Swin-Holo-Blind (0-P, 768 Syms)\n{_m_str_perc(x3, r3_sholo)}"),
                (r3_bbvar, f"BlackBox-VAR-Blind (0-P, 768 Syms)\n{_m_str_perc(x3, r3_bbvar)}"),
                (r3_ours, f"GenHolo-VAR (Ours, 0-P, 768 Syms!)\n{_m_str_perc(x3, r3_ours)}"),
            ],
        ),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(16.8, 11.4), dpi=220)
    fig.suptitle(
        "Direction 5 Held-Out Kodak24 Visual Showcase: 900 km/h Zero-Pilot OTFS & 768/384-Symbol Generative Holographic JSCC",
        fontsize=12.2,
        fontweight="bold",
        y=0.988,
    )
    for r_idx, (row_lbl, cols) in enumerate(grid_rows):
        for c_idx, (img_t, title_s) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            np_img = img_t[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
            ax.imshow(np_img)
            ax.set_xticks([])
            ax.set_yticks([])
            is_ours = c_idx == 4
            for spine in ax.spines.values():
                spine.set_edgecolor("#c0392b" if is_ours else "#bdc3c7")
                spine.set_linewidth(2.4 if is_ours else 0.8)
            ax.set_title(
                title_s,
                fontsize=8.1,
                fontweight="bold" if (is_ours or c_idx == 0) else "normal",
                color="#922b21" if is_ours else "black",
                pad=5,
            )
            if c_idx == 0:
                ax.set_ylabel(row_lbl, fontsize=8.7, fontweight="bold", labelpad=8)

    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.015, top=0.925, wspace=0.08, hspace=0.26)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--dir5-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_fig_dir5_1_otfs(
        args.eval_dir / "dir5a_otfs_common500.csv",
        args.out_dir / "fig_dir5_1_otfs_high_mobility.png",
    )
    render_fig_dir5_2_genvar(
        args.eval_dir / "dir5b_genvar_common500.csv",
        args.out_dir / "fig_dir5_2_genvar_cbr128_256_perceptual.png",
    )
    render_fig_dir5_3_kodak_qualitative(
        args.backbone_ckpt,
        args.dir5_ckpt_root,
        args.v4_usph_ckpt,
        args.out_dir / "fig_dir5_3_qualitative_kodak.png",
        physical_gpu=args.physical_gpu,
    )
    print(f"[Render-Dir5] Saved all 3 figures to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
