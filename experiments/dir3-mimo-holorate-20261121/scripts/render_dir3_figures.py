#!/usr/bin/env python3
"""Render IEEE camera-ready publication figures for Direction 3:
  - fig_dir3_1_mimo_2x2_4x4.png: Dir-3A Zero-Pilot Spatial-Multiplexed 2x2 (CBR=1/32) & 4x4 (CBR=1/64) MIMO-OFDM + OOD Correlated MIMO
  - fig_dir3_2_holorate_erasure_jamming.png: Dir-3B Anytime-Stopping Rateless Curve, Contiguous Burst Erasures & Hostile Partial-Band Jamming
  - fig_dir3_3_qualitative_kodak.png: 3-Row Held-Out Kodak24 Visual Comparison (4x4 OOD MIMO, 50% Contiguous Burst Loss, +18 dB OOD Jamming)
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
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_fig_dir3_1_mimo(mimo_csv: Path, out_png: Path) -> None:
    rows = load_csv(mimo_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.6, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), dpi=220)
    fig.suptitle(
        "Dir-3A — Zero-Pilot Spatial-Multiplexed 2×2 (CBR=1/32) & 4×4 (CBR=1/64) MIMO-OFDM Semantic Comm. (ImageNet-500)",
        fontsize=12.1,
        fontweight="bold",
        y=0.99,
    )

    line_styles = [
        ("Frozen_Blind_MIMO", "#7f8c8d", ":", "x", "Frozen Blind MIMO (0-Pilot)"),
        ("BlackBox_Blind_MIMO", "#2980b9", "--", "s", "Black-Box Blind MIMO (0-Pilot E2E)"),
        ("Pilot4_MIMO_LMMSE", "#8e44ad", "-.", "^", "4-Pilot Orth. MIMO LMMSE (128 Pilot Slots)"),
        ("OS_MIMO_JSCC", "#c0392b", "-", "*", "OS-MIMO-JSCC (Ours: 0-Pilot Walsh Subspace)"),
        ("Genie_SVD_MIMO_Oracle", "#2c3e50", ":", "o", "Genie SVD Eigenmode Oracle (Exact TX+RX CSI)"),
    ]

    for ax_idx, (cfg, cbr_lbl, spec_lbl) in enumerate([
        ("2x2", "CBR=1/32", "50% Spectrum"),
        ("4x4", "CBR=1/64", "25% Spectrum"),
    ]):
        ax = axes[ax_idx]
        sub = [
            r for r in rows
            if r["mimo_config"] == cfg and r["condition"].startswith("mimo_iid_")
        ]
        for mname, col, ls, mk, lbl in line_styles:
            pts = sorted([r for r in sub if r["model"] == mname], key=lambda r: float(r["true_snr_db"]))
            if not pts:
                continue
            xs = [float(r["true_snr_db"]) for r in pts]
            ys = [float(r["psnr_db_mean"]) for r in pts]
            y_lo = [float(r["psnr_db_ci95_lo"]) for r in pts]
            y_hi = [float(r["psnr_db_ci95_hi"]) for r in pts]
            lw = 2.4 if mname == "OS_MIMO_JSCC" else 1.75
            ms = 7.5 if mname == "OS_MIMO_JSCC" else 6.0
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
            if mname in ("OS_MIMO_JSCC", "BlackBox_Blind_MIMO", "Pilot4_MIMO_LMMSE"):
                ax.fill_between(xs, y_lo, y_hi, color=col, alpha=0.10)

        panel_tag = "(a)" if ax_idx == 0 else "(b)"
        ax.set_title(f"{panel_tag} {cfg} Spatial-Multiplexed MIMO ({cbr_lbl}, {spec_lbl})")
        ax.set_xlabel("Per-Receive-Antenna MIMO-OFDM SNR (dB)")
        ax.set_ylabel("Reconstruction PSNR (dB) ↑")
        ax.set_xticks([1, 4, 7, 10, 13])
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=7.6, loc="lower right", framealpha=0.92)

    # Panel (c): Unseen OOD Kronecker Spatially-Correlated MIMO (rho_ant = 0.60)
    ax3 = axes[2]
    ood_groups = [
        ("2x2", "mimo_ood_corr060_7dB", "2×2 OOD\n(7 dB)"),
        ("2x2", "mimo_ood_corr060_10dB", "2×2 OOD\n(10 dB)"),
        ("4x4", "mimo_ood_corr060_7dB", "4×4 OOD\n(7 dB)"),
        ("4x4", "mimo_ood_corr060_10dB", "4×4 OOD\n(10 dB)"),
    ]
    bar_models = [
        ("Frozen_Blind_MIMO", "#95a5a6", "Frozen Blind (0-P)"),
        ("BlackBox_Blind_MIMO", "#2980b9", "Black-Box Blind (0-P)"),
        ("Pilot4_MIMO_LMMSE", "#8e44ad", "4-Pilot LMMSE (128-P)"),
        ("OS_MIMO_JSCC", "#c0392b", "OS-MIMO-JSCC (Ours, 0-P)"),
        ("Genie_SVD_MIMO_Oracle", "#34495e", "Genie SVD (TX+RX CSI)"),
    ]
    x_pos = np.arange(len(ood_groups))
    width = 0.155
    ymax = 0.0
    for b_i, (mname, col, blbl) in enumerate(bar_models):
        vals = []
        for cfg, cond, _ in ood_groups:
            hit = [r for r in rows if r["mimo_config"] == cfg and r["condition"] == cond and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax = max(ymax, v)
        offs = (b_i - 2.0) * width
        bars = ax3.bar(
            x_pos + offs,
            vals,
            width=width,
            color=col,
            edgecolor="black" if mname == "OS_MIMO_JSCC" else "none",
            linewidth=0.9 if mname == "OS_MIMO_JSCC" else 0.0,
            label=blbl,
        )
        if mname in ("OS_MIMO_JSCC", "Pilot4_MIMO_LMMSE", "BlackBox_Blind_MIMO"):
            for br, v in zip(bars, vals):
                ax3.text(
                    br.get_x() + br.get_width() / 2.0,
                    v + 0.35,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    fontweight="bold" if mname == "OS_MIMO_JSCC" else "normal",
                    color="#900C3F" if mname == "OS_MIMO_JSCC" else "#2c3e50",
                )

    ax3.set_title("(c) Unseen OOD Spatially Correlated MIMO ($\\rho_{\\mathrm{ant}}=0.60$)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([g[2] for g in ood_groups], fontsize=8.8)
    ax3.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax3.set_ylim(6.0, ymax + 6.2)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.93)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_fig_dir3_2_holorate(holo_csv: Path, out_png: Path) -> None:
    rows = load_csv(holo_csv)
    if not rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.6, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), dpi=220)
    fig.suptitle(
        "Dir-3B — Holographic Rateless Anytime-Stopping, Burst-Erasure & Anti-Jamming Blind JSCC (ImageNet-500, 10 dB OFDM)",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )

    models_meta = [
        ("V4_U_SPH_ZeroShot", "#7f8c8d", ":", "^", "V4 U-SPH-ZPAC (Standard Raster Layout)"),
        ("Raster_Blind_Resilient", "#2980b9", "--", "s", "Raster Resilient Blind JSCC (Trained w/ Drops)"),
        ("HoloRate_JSCC", "#c0392b", "-", "*", "HoloRate-JSCC (Ours: 256-Pt Hadamard Dispersion)"),
    ]

    # Panel (a): Anytime-Stopping Rateless Prefix Curve
    ax1 = axes[0]
    rateless_order = [
        ("rateless_8rb_25pct", 25),
        ("rateless_16rb_50pct", 50),
        ("rateless_24rb_75pct", 75),
        ("rateless_32rb_100pct", 100),
    ]
    for mname, col, ls, mk, lbl in models_meta:
        xs, ys, ylo, yhi = [], [], [], []
        for cond, pct in rateless_order:
            hit = [r for r in rows if r["condition"] == cond and r["model"] == mname]
            if hit:
                xs.append(pct)
                ys.append(float(hit[0]["psnr_db_mean"]))
                ylo.append(float(hit[0]["psnr_db_ci95_lo"]))
                yhi.append(float(hit[0]["psnr_db_ci95_hi"]))
        lw = 2.5 if mname == "HoloRate_JSCC" else 1.85
        ms = 8.0 if mname == "HoloRate_JSCC" else 6.2
        ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=ms, label=lbl)
        ax1.fill_between(xs, ylo, yhi, color=col, alpha=0.11)

    ax1.set_title("(a) Anytime-Stopping Rateless Prefix (0 Feedback)")
    ax1.set_xlabel("Received OFDM Prefix Budget (% of 32 RBs / CBR)")
    ax1.set_xticks([25, 50, 75, 100])
    ax1.set_xticklabels(["25% (8 RB)\nCBR 1/64", "50% (16 RB)\nCBR 1/32", "75% (24 RB)\nCBR 3/64", "100% (32 RB)\nCBR 1/16"], fontsize=8.4)
    ax1.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax1.grid(True, alpha=0.30)
    ax1.legend(fontsize=7.7, loc="lower right", framealpha=0.93)

    # Panel (b): Contiguous & Random Packet/RB Erasures
    ax2 = axes[1]
    erasure_groups = [
        ("burst_erasure_25pct", "25% Contig.\nBurst Loss"),
        ("random_erasure_37pct", "37.5% Random\nRB Erasure"),
        ("burst_erasure_50pct", "50% Contig.\nBurst Loss (16 RB)"),
    ]
    x_e = np.arange(len(erasure_groups))
    w_e = 0.25
    ymax_e = 0.0
    for b_i, (mname, col, _, _, lbl) in enumerate(models_meta):
        vals = []
        for cond, _ in erasure_groups:
            hit = [r for r in rows if r["condition"] == cond and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax_e = max(ymax_e, v)
        bars = ax2.bar(
            x_e + (b_i - 1.0) * w_e,
            vals,
            width=w_e,
            color=col,
            edgecolor="black" if mname == "HoloRate_JSCC" else "none",
            linewidth=0.9 if mname == "HoloRate_JSCC" else 0.0,
            label=lbl,
        )
        for br, v in zip(bars, vals):
            ax2.text(
                br.get_x() + br.get_width() / 2.0,
                v + 0.30,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold" if mname == "HoloRate_JSCC" else "normal",
            )
    ax2.set_title("(b) Contiguous Deep-Fade & Burst Packet Erasures")
    ax2.set_xticks(x_e)
    ax2.set_xticklabels([g[1] for g in erasure_groups], fontsize=8.7)
    ax2.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax2.set_ylim(10.0, ymax_e + 4.8)
    ax2.grid(True, axis="y", alpha=0.30)
    ax2.legend(fontsize=7.6, loc="upper right", framealpha=0.93)

    # Panel (c): Hostile Partial-Band Jamming Resilience
    ax3 = axes[2]
    jam_groups = [
        ("rateless_32rb_100pct", "Clean OFDM\n(0% Jammed)"),
        ("jamming_25pct_15dB", "25% Band Jam\n(+15 dB INR)"),
        ("jamming_37pct_15dB", "37.5% Band Jam\n(+15 dB INR)"),
        ("jamming_ood_31pct_18dB", "31% OOD Jam\n(+18 dB INR!)"),
    ]
    x_j = np.arange(len(jam_groups))
    w_j = 0.25
    ymax_j = 0.0
    for b_i, (mname, col, _, _, lbl) in enumerate(models_meta):
        vals = []
        for cond, _ in jam_groups:
            hit = [r for r in rows if r["condition"] == cond and r["model"] == mname]
            v = float(hit[0]["psnr_db_mean"]) if hit else 0.0
            vals.append(v)
            ymax_j = max(ymax_j, v)
        bars = ax3.bar(
            x_j + (b_i - 1.0) * w_j,
            vals,
            width=w_j,
            color=col,
            edgecolor="black" if mname == "HoloRate_JSCC" else "none",
            linewidth=0.9 if mname == "HoloRate_JSCC" else 0.0,
            label=lbl,
        )
        for br, v in zip(bars, vals):
            ax3.text(
                br.get_x() + br.get_width() / 2.0,
                v + 0.30,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if mname == "HoloRate_JSCC" else "normal",
            )
    ax3.set_title("(c) Blind Partial-Band Anti-Jamming (0 Jammer Knowledge)")
    ax3.set_xticks(x_j)
    ax3.set_xticklabels([g[1] for g in jam_groups], fontsize=8.5)
    ax3.set_ylabel("Reconstruction PSNR (dB) ↑")
    ax3.set_ylim(10.0, ymax_j + 4.8)
    ax3.grid(True, axis="y", alpha=0.30)
    ax3.legend(fontsize=7.5, loc="upper right", framealpha=0.93)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def render_fig_dir3_3_qualitative(
    backbone_ckpt: Path,
    dir3_ckpt_root: Path,
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
    from dir3_mh.mimo_blind import BlindMIMOTransceiver, forward_pilot_or_genie_svd_mimo_reference, sample_kronecker_mimo_rbs
    from dir3_mh.holorate_blind import HoloRateBlindTransceiver, physical_hostile_ofdm_channel

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir3_mimo_holorate_v1/cache/fig_dir3")
    )
    seed_everything(20261123)
    configure_torch_determinism()
    admit_gpu(physical_gpu)
    device = torch.device("cuda:0")

    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    targets = (kodak.to(dtype=torch.float32) / 255.0).to(device)
    bb = load_swinjscc_backbone(backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    # Load Dir-3A 4x4 MIMO models
    m_frz4 = BlindMIMOTransceiver(bb, method="frozen_blind_mimo", n_ant=4, num_rbs=None, hidden_ch=160).to(device).eval()
    m_bb4 = BlindMIMOTransceiver(bb, method="blackbox_mimo", n_ant=4, num_rbs=None, hidden_ch=160).to(device)
    m_bb4.load_state_dict(torch.load(dir3_ckpt_root / "blackbox_mimo_4x4/best_dir3.pt", map_location="cpu")["state_dict"], strict=False)
    m_bb4.eval()
    m_os4 = BlindMIMOTransceiver(bb, method="os_mimo", n_ant=4, num_rbs=None, hidden_ch=160).to(device)
    m_os4.load_state_dict(torch.load(dir3_ckpt_root / "os_mimo_4x4/best_dir3.pt", map_location="cpu")["state_dict"], strict=False)
    m_os4.eval()

    # Load Dir-3B & V4 models
    m_holo = HoloRateBlindTransceiver(bb, method="holorate_blind", num_rbs=32, hidden_ch=160).to(device)
    m_holo.load_state_dict(torch.load(dir3_ckpt_root / "holorate_blind/best_dir3.pt", map_location="cpu")["state_dict"], strict=False)
    m_holo.eval()

    m_rast = HoloRateBlindTransceiver(bb, method="raster_blind_resilient", num_rbs=32, hidden_ch=160).to(device)
    m_rast.load_state_dict(torch.load(dir3_ckpt_root / "raster_blind_resilient/best_dir3.pt", map_location="cpu")["state_dict"], strict=False)
    m_rast.eval()

    m_v4 = UniversalBlindTransceiverV4(bb, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    m_v4.load_state_dict(torch.load(v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    m_v4.eval()

    def forward_frozen_hostile_ofdm(
        x_stem: torch.Tensor,
        snr_db: float,
        *,
        alive_rb_mask: torch.Tensor | None = None,
        jam_rb_mask: torch.Tensor | None = None,
        jam_inr_db: float = 0.0,
        h_rb_override: torch.Tensor,
        standard_noise_iq: torch.Tensor,
        jam_noise_iq: torch.Tensor,
    ) -> torch.Tensor:
        enc, dec = bb.encoder, bb.decoder
        B, L, _ = x_stem.shape
        snr_c0 = torch.full((B, 1), 7.0, device=x_stem.device, dtype=x_stem.dtype)
        temp = None
        for i in range(enc.layer_num):
            temp = enc.sm_list[i](x_stem) if i == 0 else enc.sm_list[i](temp)
            temp = temp * enc.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
        z_base = enc.head_list(x_stem * enc.sigmoid(enc.sm_list[-1](temp)))
        K = (L * z_base.shape[-1]) // 2
        pwr = (2.0 * z_base.square().mean(dim=(1, 2))).clamp_min(1e-8)
        flat = (z_base / pwr.sqrt()[:, None, None]).reshape(B, 2 * K)
        stx = torch.stack([flat[:, :K], flat[:, K:]], dim=-1).reshape(B, 32, 384, 2)
        yrx, _ = physical_hostile_ofdm_channel(
            stx,
            snr_db,
            alive_rb_mask=alive_rb_mask,
            jam_rb_mask=jam_rb_mask,
            jam_inr_db=jam_inr_db,
            h_rb_override=h_rb_override,
            standard_noise_iq=standard_noise_iq,
            jam_noise_iq=jam_noise_iq,
        )
        y_iq = yrx.reshape(B, K, 2)
        tot_pwr = (y_iq[:, :, 0].square() + y_iq[:, :, 1].square()).mean(dim=1).clamp_min(0.5)
        u_in = torch.cat([y_iq[:, :, 0], y_iq[:, :, 1]], dim=1).reshape(B, L, -1) * (2.82 / tot_pwr.sqrt()[:, None, None])
        x = dec.head_list(u_in)
        temp = None
        for i in range(dec.layer_num):
            temp = dec.sm_list[i](x) if i == 0 else dec.sm_list[i](temp)
            temp = temp * dec.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
        x = x * dec.sigmoid(dec.sm_list[-1](temp))
        for layer in dec.layers:
            x = layer(x)
        return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous().clamp(0, 1)

    fig, axes = plt.subplots(3, 5, figsize=(16.0, 9.8), dpi=220)
    fig.suptitle(
        "Held-Out Kodak24 Visual Comparison — Dir-3A 4×4 OOD MIMO (CBR=1/64) & Dir-3B 50% Burst Loss / +18 dB OOD Jamming",
        fontsize=12.2,
        fontweight="bold",
        y=0.985,
    )

    # ---------------- ROW 0: 4x4 OOD Correlated MIMO (rho_ant = 0.60, 10 dB, Kodak #19) ----------------
    idx0 = 18
    xb0 = targets[idx0 : idx0 + 1]
    gen0 = torch.Generator(device="cpu").manual_seed(2001 * 1009 + 4 * 37 + 60)
    eps0 = torch.randn((1, 4, 3072, 2), generator=gen0, dtype=torch.float32).to(device)
    hr0, hi0 = sample_kronecker_mimo_rbs(1, 8, 4, 4, rho_f=0.85, rho_ant=0.60, generator=gen0)
    hr0, hi0 = hr0.to(device), hi0.to(device)

    r0_frz = m_frz4.forward_mimo_blind(xb0, 10.0, rho_f=0.85, rho_ant=0.60, h_mimo_override=(hr0, hi0), standard_noise_iq=eps0)["recon"].clamp(0, 1)
    r0_bb = m_bb4.forward_mimo_blind(xb0, 10.0, rho_f=0.85, rho_ant=0.60, h_mimo_override=(hr0, hi0), standard_noise_iq=eps0)["recon"].clamp(0, 1)
    r0_p4 = forward_pilot_or_genie_svd_mimo_reference(bb, xb0, 4, 10.0, hr0, hi0, eps0, mode="pilot4_lmmse", pilot_seed=999).clamp(0, 1)
    r0_os = m_os4.forward_mimo_blind(xb0, 10.0, rho_f=0.85, rho_ant=0.60, h_mimo_override=(hr0, hi0), standard_noise_iq=eps0)["recon"].clamp(0, 1)

    axes[0, 0].imshow(xb0[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Ground Truth (Kodak #19)\n4×4 OOD MIMO ($\\rho_{\\mathrm{ant}}=0.60$, CBR=1/64)", fontsize=8.6, fontweight="bold")
    axes[0, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Frozen 4×4 MIMO\n(0-Pilot Blind)", r0_frz, False),
        ("Black-Box 4×4 MIMO\n(0-Pilot E2E)", r0_bb, False),
        ("4-Pilot Orth. LMMSE\n(128 Extra Pilot Slots)", r0_p4, False),
        ("OS-MIMO-JSCC (Ours)\n(0 Pilots, 4× Spectrum Saving!)", r0_os, True),
    ], start=1):
        pv = float(compute_psnr_batch(rec, xb0)[0].item())
        axes[0, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
        axes[0, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=8.8, fontweight="bold" if is_ours else "normal", color="#900C3F" if is_ours else "black")
        axes[0, c_i].axis("off")

    # ---------------- ROW 1: 50% Contiguous RB Burst Erasure (Middle 16/32 RBs Dropped, 10 dB, Kodak #23) ----------------
    idx1 = 22
    xb1 = targets[idx1 : idx1 + 1]
    sb1 = m_holo.extract_encoder_stem(xb1)
    gen1 = torch.Generator(device="cpu").manual_seed(202611)
    eps1 = torch.randn((1, 32, 384, 2), generator=gen1, dtype=torch.float32).to(device)
    jeps1 = torch.randn((1, 32, 384, 2), generator=gen1, dtype=torch.float32).to(device)
    hr1, hi1 = sample_correlated_rbs(1, 32, 0.85, device="cpu", generator=gen1)
    h1 = torch.stack([hr1, hi1], dim=-1).to(device)
    am1 = torch.ones((1, 32), device=device)
    am1[:, 8:24] = 0.0  # Middle 50% contiguous RBs completely wiped out!

    r1_frz = forward_frozen_hostile_ofdm(sb1, 10.0, alive_rb_mask=am1, h_rb_override=h1, standard_noise_iq=eps1, jam_noise_iq=jeps1)
    stx_v4_1 = m_v4.encode_blind_tx(sb1).reshape(1, 32, 384, 2)
    yrx_v4_1, _ = physical_hostile_ofdm_channel(stx_v4_1, 10.0, alive_rb_mask=am1, h_rb_override=h1, standard_noise_iq=eps1, jam_noise_iq=jeps1)
    r1_v4 = m_v4.decode_blind_rx(yrx_v4_1.reshape(1, 12288, 2))[0].clamp(0, 1)
    r1_rast = m_rast.forward_hostile_blind(sb1, 10.0, alive_rb_mask=am1, h_rb_override=h1, standard_noise_iq=eps1, jam_noise_iq=jeps1, input_is_stem=True)["recon"].clamp(0, 1)
    r1_holo = m_holo.forward_hostile_blind(sb1, 10.0, alive_rb_mask=am1, h_rb_override=h1, standard_noise_iq=eps1, jam_noise_iq=jeps1, input_is_stem=True)["recon"].clamp(0, 1)

    axes[1, 0].imshow(xb1[0].permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("Ground Truth (Kodak #23)\n50% Contig. Burst Loss (RBs 8..23=0)", fontsize=8.6, fontweight="bold")
    axes[1, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Frozen Blind OFDM\n(50% Burst Loss)", r1_frz, False),
        ("V4 U-SPH-ZPAC\n(Raster Spatial Wipeout!)", r1_v4, False),
        ("Raster Resilient Blind\n(Trained w/ Burst Drops)", r1_rast, False),
        ("HoloRate-JSCC (Ours)\n(Hadamard Global Recovery!)", r1_holo, True),
    ], start=1):
        pv = float(compute_psnr_batch(rec, xb1)[0].item())
        axes[1, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
        axes[1, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=8.8, fontweight="bold" if is_ours else "normal", color="#900C3F" if is_ours else "black")
        axes[1, c_i].axis("off")

    # ---------------- ROW 2: Unseen OOD +18 dB Partial-Band Jamming (10/32 RBs Jammed, 10 dB, Kodak #7) ----------------
    idx2 = 6
    xb2 = targets[idx2 : idx2 + 1]
    sb2 = m_holo.extract_encoder_stem(xb2)
    gen2 = torch.Generator(device="cpu").manual_seed(202622)
    eps2 = torch.randn((1, 32, 384, 2), generator=gen2, dtype=torch.float32).to(device)
    jeps2 = torch.randn((1, 32, 384, 2), generator=gen2, dtype=torch.float32).to(device)
    hr2, hi2 = sample_correlated_rbs(1, 32, 0.85, device="cpu", generator=gen2)
    h2 = torch.stack([hr2, hi2], dim=-1).to(device)
    jm2 = torch.zeros((1, 32), device=device)
    jm2[:, [2, 5, 9, 12, 15, 18, 21, 24, 27, 30]] = 1.0  # 10/32 RBs (31.2%) jammed at +18 dB (63x signal power!)

    r2_frz = forward_frozen_hostile_ofdm(sb2, 10.0, jam_rb_mask=jm2, jam_inr_db=18.0, h_rb_override=h2, standard_noise_iq=eps2, jam_noise_iq=jeps2)
    stx_v4_2 = m_v4.encode_blind_tx(sb2).reshape(1, 32, 384, 2)
    yrx_v4_2, _ = physical_hostile_ofdm_channel(stx_v4_2, 10.0, jam_rb_mask=jm2, jam_inr_db=18.0, h_rb_override=h2, standard_noise_iq=eps2, jam_noise_iq=jeps2)
    r2_v4 = m_v4.decode_blind_rx(yrx_v4_2.reshape(1, 12288, 2))[0].clamp(0, 1)
    r2_rast = m_rast.forward_hostile_blind(sb2, 10.0, jam_rb_mask=jm2, jam_inr_db=18.0, h_rb_override=h2, standard_noise_iq=eps2, jam_noise_iq=jeps2, input_is_stem=True)["recon"].clamp(0, 1)
    r2_holo = m_holo.forward_hostile_blind(sb2, 10.0, jam_rb_mask=jm2, jam_inr_db=18.0, h_rb_override=h2, standard_noise_iq=eps2, jam_noise_iq=jeps2, input_is_stem=True)["recon"].clamp(0, 1)

    axes[2, 0].imshow(xb2[0].permute(1, 2, 0).cpu().numpy())
    axes[2, 0].set_title("Ground Truth (Kodak #7)\nOOD +18 dB Jamming (10/32 RBs)", fontsize=8.6, fontweight="bold")
    axes[2, 0].axis("off")
    for c_i, (lbl, rec, is_ours) in enumerate([
        ("Frozen Blind OFDM\n(+18 dB Jammer)", r2_frz, False),
        ("V4 U-SPH-ZPAC\n(Unprotected Null Poisoned)", r2_v4, False),
        ("Raster Resilient Blind\n(Blind Excision + Raster)", r2_rast, False),
        ("HoloRate-JSCC (Ours)\n(Blind Excision + Hadamard!)", r2_holo, True),
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
    parser.add_argument("--dir3-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    render_fig_dir3_1_mimo(
        args.eval_dir / "dir3a_mimo_results_common500.csv",
        args.out_dir / "fig_dir3_1_mimo_2x2_4x4.png",
    )
    render_fig_dir3_2_holorate(
        args.eval_dir / "dir3b_holorate_results_common500.csv",
        args.out_dir / "fig_dir3_2_holorate_erasure_jamming.png",
    )
    render_fig_dir3_3_qualitative(
        args.backbone_ckpt,
        args.dir3_ckpt_root,
        args.v4_usph_ckpt,
        args.out_dir / "fig_dir3_3_qualitative_kodak.png",
        physical_gpu=args.physical_gpu,
    )
    print("RENDERED ALL DIRECTION 3 FIGURES SUCCESSFULLY!")


if __name__ == "__main__":
    main()
