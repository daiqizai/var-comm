#!/usr/bin/env python3
"""Render IEEE camera-ready publication figures for V3 Completely CSI-Free / Pilot-Free Adaptive JSCC."""
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


STYLE = {
    "Frozen_Blind": ("#7f8c8d", ":", "x", "Frozen SwinJSCC (0-Pilot Blind)"),
    "Blind_BlackBox": ("#d35400", "--", "v", "Blind-BlackBox (TCOM'25 E2E)"),
    "Blind_RX_ZPAC": ("#16a085", "-.", "d", "Blind-RX-ZPAC (Frozen TX)"),
    "SPH_ZPAC_NoHierTX": ("#5dade2", "--", "s", "SPH-ZPAC w/o Hier-TX"),
    "SPH_ZPAC": ("#c0392b", "-", "*", "SPH-ZPAC (Ours: 0-Pilot, 0-CSI)"),
    "Pilot4_Frozen_MMSE": ("#8e44ad", "-.", "^", "4-Pilot SwinJSCC (128 Pilots)"),
    "Pilot4_TR_ZPAC_Ceiling": ("#2c3e50", ":", "o", "4-Pilot TR-ZPAC (Closed-Loop Ref)"),
    "AvgSNR_GeniePwr": ("#e67e22", "--", "^", "Open-Loop AvgSNR (Genie Pwr)"),
    "Oracle_MatchedCSI": ("#2c3e50", ":", "o", "Oracle Matched-CSI (Genie Upper)"),
}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_main_v3_figure(awgn_csv: Path, ofdm_csv: Path, out_png: Path) -> None:
    a_rows = load_csv(awgn_csv)
    o_rows = load_csv(ofdm_csv)
    if not a_rows or not o_rows:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.8, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=220)
    fig.suptitle("V3 — Completely CSI-Free, Feedback-Free & Pilot-Free Adaptive Transmission (ImageNet-500, 3 Seeds)", fontsize=12.5, fontweight="bold", y=0.99)

    # Panel (a): Pilot-Free 32-RB Complex OFDM Fading Curve (rho_f = 0.85, SNR = 1..13 dB)
    ax = axes[0]
    for m in ["Frozen_Blind", "Blind_BlackBox", "Pilot4_Frozen_MMSE", "SPH_ZPAC_NoHierTX", "SPH_ZPAC", "Pilot4_TR_ZPAC_Ceiling"]:
        pts = sorted(
            [r for r in o_rows if r["model"] == m and abs(float(r["rho_f"]) - 0.85) < 1e-3],
            key=lambda r: float(r["true_snr_db"]),
        )
        if pts:
            col, ls, mk, lbl = STYLE[m]
            xs = [float(r["true_snr_db"]) for r in pts]
            ys = [float(r["psnr_db_mean"]) for r in pts]
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=2.2 if m == "SPH_ZPAC" else 1.8, markersize=6.8, label=lbl)
    ax.set_title("(a) Pilot-Free 32-RB Complex OFDM Fading")
    ax.set_xlabel("True Channel SNR (dB) [Hidden from TX & RX]")
    ax.set_ylabel("Mean PSNR (dB) ↑")
    ax.set_xticks([1, 4, 7, 10, 13])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.8, loc="best")

    # Panel (b): OOD Severe Multipath (rho_f = 0.55) & Outage p10 Comparison at 10 dB
    ax2 = axes[1]
    conds_b = [
        ("ofdm_snr10_rhof085", "32-RB OFDM\n(10dB, $\\rho_f=0.85$)"),
        ("ofdm_ood_snr10_rhof055", "OOD Multipath\n(10dB, $\\rho_f=0.55$)"),
        ("ofdm_ood_snr7_rhof055", "OOD Multipath\n(7dB, $\\rho_f=0.55$)"),
    ]
    bar_m = [m for m in ["Frozen_Blind", "Blind_BlackBox", "Pilot4_Frozen_MMSE", "SPH_ZPAC"] if any(r["model"] == m for r in o_rows)]
    xp = np.arange(len(conds_b))
    w = 0.18
    for idx, m in enumerate(bar_m):
        col, _, _, lbl = STYLE[m]
        vals = []
        for ck, _ in conds_b:
            rr = [r for r in o_rows if r["model"] == m and r["condition"] == ck]
            vals.append(float(rr[0]["psnr_db_mean"]) if rr else 0.0)
        off = (idx - (len(bar_m) - 1) / 2.0) * w
        ax2.bar(xp + off, vals, width=w * 0.90, color=col, alpha=0.92, label=lbl)
    ax2.set_title("(b) In-Domain & Unseen OOD Multipath ($\\rho_f=0.55$)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels([c[1] for c in conds_b], fontsize=8.8)
    ax2.set_ylabel("Mean PSNR (dB) ↑")
    ax2.set_ylim(0.0, 36.5)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=7.6, loc="upper left", ncol=2, framealpha=0.92)

    # Panel (c): Unknown-SNR AWGN Gap to Open-Loop AvgSNR (Genie Pwr) across [-2..15] dB
    ax3 = axes[2]
    for m in ["Frozen_Blind", "AvgSNR_GeniePwr", "Blind_BlackBox", "Blind_RX_ZPAC", "SPH_ZPAC_NoHierTX", "SPH_ZPAC"]:
        pts = sorted([r for r in a_rows if r["model"] == m], key=lambda r: float(r["true_snr_db"]))
        ora = {float(r["true_snr_db"]): float(r["psnr_db_mean"]) for r in a_rows if r["model"] == "Oracle_MatchedCSI"}
        if pts and ora:
            col, ls, mk, lbl = STYLE[m]
            xs = [float(r["true_snr_db"]) for r in pts]
            # Plot PSNR difference relative to AvgSNR_GeniePwr (strong open-loop reference!)
            avg_ref = {float(r["true_snr_db"]): float(r["psnr_db_mean"]) for r in a_rows if r["model"] == "AvgSNR_GeniePwr"}
            ys = [float(r["psnr_db_mean"]) - avg_ref[float(r["true_snr_db"])] for r in pts]
            ax3.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=2.2 if m in ("SPH_ZPAC", "Blind_RX_ZPAC") else 1.7, markersize=6.3, label=lbl)
    # Also plot Oracle Matched-CSI upper bound relative to AvgSNR_GeniePwr
    pts_o = sorted([r for r in a_rows if r["model"] == "Oracle_MatchedCSI"], key=lambda r: float(r["true_snr_db"]))
    avg_ref = {float(r["true_snr_db"]): float(r["psnr_db_mean"]) for r in a_rows if r["model"] == "AvgSNR_GeniePwr"}
    if pts_o and avg_ref:
        col, ls, mk, lbl = STYLE["Oracle_MatchedCSI"]
        ax3.plot([float(r["true_snr_db"]) for r in pts_o], [float(r["psnr_db_mean"]) - avg_ref[float(r["true_snr_db"])] for r in pts_o], color=col, linestyle=ls, marker=mk, linewidth=1.8, label=lbl)

    ax3.axhline(0.0, color="#e67e22", linewidth=1.0, linestyle="--")
    ax3.set_title("(c) Blind Unknown-SNR AWGN vs. Open-Loop AvgSNR")
    ax3.set_xlabel("True AWGN SNR (dB) [-2 & 15 dB Unseen OOD]")
    ax3.set_ylabel("ΔPSNR over Open-Loop AvgSNR (dB) ↑")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7.6, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render_qualitative_v3(
    backbone_ckpt: Path,
    v3_ckpt_root: Path,
    out_png: Path,
    physical_gpu: int = 0,
) -> None:
    from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
    from dir2_zpac.datasets import load_split_u8
    from dir2_zpac.models import load_swinjscc_backbone
    from dir2_zpac.models_v2 import TransceiverZPACv2
    from dir2_zpac.blind_v3 import BlindCSIFreeTransceiver
    from dir2_zpac.metrics import compute_psnr_batch
    from evaluate_blind_v3 import build_pilot_aided_steps_from_h_rb, make_paired_h_rb, make_paired_iq_noise

    configure_runtime_env(cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_blind_csifree_v3/cache"))
    seed_everything(20261121)
    configure_torch_determinism()
    admit_gpu(physical_gpu)
    device = torch.device("cuda:0")

    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    targets = (kodak.to(dtype=torch.float32) / 255.0).to(device)
    bb = load_swinjscc_backbone(backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    def load_b(mth: str, cp: Path | None) -> BlindCSIFreeTransceiver:
        m = BlindCSIFreeTransceiver(bb, method=mth, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)
        if cp is not None:
            m.load_state_dict(torch.load(cp, map_location="cpu")["state_dict"], strict=False)
        return m.eval()

    m_frz = load_b("frozen_blind", None)
    m_bbx = load_b("blind_blackbox", v3_ckpt_root / "blind_blackbox/best_blind_v3.pt")
    m_sph = load_b("sph_zpac", v3_ckpt_root / "sph_zpac/best_blind_v3.pt")
    m_pil = TransceiverZPACv2(bb, method="frozen", hidden_ch=160).to(device).eval()

    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6), dpi=220)
    with torch.inference_mode():
        eps_iq = make_paired_iq_noise(128 * 96, list(range(24)), 2001, device=device)
        eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(24, 256, 96)
        h_rb_85 = make_paired_h_rb(32, 0.85, list(range(24)), 2001, device=device)
        h_rb_55 = make_paired_h_rb(32, 0.55, list(range(24)), 2002, device=device)

        for row_i, (snr_v, rhof_v, h_rb, img_idx, desc) in enumerate([
            (10.0, 0.85, h_rb_85, 3, "Pilot-Free 32-RB OFDM (10 dB, $\\rho_f=0.85$)"),
            (10.0, 0.55, h_rb_55, 22, "OOD Severe Multipath (10 dB, $\\rho_f=0.55$)"),
        ]):
            gt = targets[img_idx].permute(1, 2, 0).cpu().numpy()
            axes[row_i, 0].imshow(gt)
            axes[row_i, 0].set_title(f"Ground Truth (Kodak #{img_idx+1})\n{desc}", fontsize=9.0, fontweight="bold")
            axes[row_i, 0].axis("off")

            r_frz = m_frz.forward_blind(targets[img_idx:img_idx+1], snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h_rb[img_idx:img_idx+1], standard_noise_iq=eps_iq[img_idx:img_idx+1])["recon"].clamp(0, 1)
            r_bbx = m_bbx.forward_blind(targets[img_idx:img_idx+1], snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h_rb[img_idx:img_idx+1], standard_noise_iq=eps_iq[img_idx:img_idx+1])["recon"].clamp(0, 1)
            p_steps = build_pilot_aided_steps_from_h_rb(h_rb[img_idx:img_idx+1], snr_v, pilot_seed=555 + row_i, pilot_symbols=4)
            r_pil = m_pil.forward_channel_v2(targets[img_idx:img_idx+1], 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=p_steps, standard_noise=eps_pack[img_idx:img_idx+1])["recon"].clamp(0, 1)
            r_sph = m_sph.forward_blind(targets[img_idx:img_idx+1], snr_v, channel_type="ofdm_complex", rho_f=rhof_v, h_rb_override=h_rb[img_idx:img_idx+1], standard_noise_iq=eps_iq[img_idx:img_idx+1])["recon"].clamp(0, 1)

            panels = [
                ("Frozen SwinJSCC\n(0-Pilot Blind)", r_frz, False),
                ("Blind-BlackBox\n(TCOM'25 E2E)", r_bbx, False),
                ("4-Pilot SwinJSCC\n(128 Extra Pilots)", r_pil, False),
                ("SPH-ZPAC (Ours)\n(0 Pilots, 0 CSI!)", r_sph, True),
            ]
            for c_i, (lbl, rec, is_ours) in enumerate(panels, start=1):
                pv = float(compute_psnr_batch(rec, targets[img_idx:img_idx+1])[0].item())
                axes[row_i, c_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
                axes[row_i, c_i].set_title(f"{lbl}\nPSNR: {pv:.2f} dB", fontsize=9.1, fontweight="bold" if is_ours else "normal")
                axes[row_i, c_i].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, hspace=0.25, wspace=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-eval-dir", type=Path, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--v3-ckpt-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    render_main_v3_figure(
        args.v3_eval_dir / "blind_awgn_results_common500.csv",
        args.v3_eval_dir / "blind_ofdm_results_common500.csv",
        args.out_dir / "fig_v3_1_blind_ofdm_and_awgn.png",
    )
    render_qualitative_v3(
        args.backbone_ckpt,
        args.v3_ckpt_root,
        args.out_dir / "fig_v3_2_blind_qualitative_kodak.png",
        physical_gpu=args.physical_gpu,
    )
    print("RENDERED ALL V3 BLIND FIGURES SUCCESSFULLY!")


if __name__ == "__main__":
    main()
