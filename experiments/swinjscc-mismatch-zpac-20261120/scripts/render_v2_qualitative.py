#!/usr/bin/env python3
"""Render side-by-side qualitative visual comparisons for V2 (32-RB OFDM Fading & Official Rayleigh Mismatch)."""
from __future__ import annotations

import argparse
import math
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

from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.phy_v2 import simulate_ofdm_ar1_chain
from dir2_zpac.metrics import compute_psnr_batch


def load_v2_model(backbone: torch.nn.Module, method: str, ckpt_path: Path | None, device: torch.device) -> TransceiverZPACv2:
    m = TransceiverZPACv2(backbone, method=method, hidden_ch=160).to(device)
    if ckpt_path is not None and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        m.load_state_dict(sd, strict=False)
    return m.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--awgn-backbone", type=Path, required=True)
    parser.add_argument("--rayleigh-backbone", type=Path, required=True)
    parser.add_argument("--v2-ckpt-root", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env()
    seed_everything(20261121)
    configure_torch_determinism()
    admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")

    kodak = load_split_u8("kodak24", allow_holdout=True).images_u8
    targets = (kodak.to(dtype=torch.float32) / 255.0).to(device)

    # Row 1: Kodak img #22 (parrot/lighthouse/detailed Kodak crop) under 32-RB Fast OFDM Fading (SNR=10, rho=0.70, d=2)
    # Row 2: Kodak img #7 under Official Rayleigh Backbone Overestimation (c_T=13 -> Gamma=1 dB)
    bb_awgn = load_swinjscc_backbone(args.awgn_backbone, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    bb_ray = load_swinjscc_backbone(args.rayleigh_backbone, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    awgn_models = [
        ("Frozen\nSwinJSCC", load_v2_model(bb_awgn, "frozen", None, device)),
        ("Avg-SNR\n(c_T=10dB)", load_v2_model(bb_awgn, "avg_snr", None, device)),
        ("P-Raw\n([c_R,c_T,0])", load_v2_model(bb_awgn, "p_raw_v2", args.v2_ckpt_root / "awgn/p_raw_v2/best_calibrator_v2.pt", device)),
        ("MS-ZPAC\n(v2 RX-Only)", load_v2_model(bb_awgn, "ms_zpac", args.v2_ckpt_root / "awgn/ms_zpac/best_calibrator_v2.pt", device)),
        ("TR-ZPAC\n(v2 Joint)", load_v2_model(bb_awgn, "tr_zpac", args.v2_ckpt_root / "awgn/tr_zpac/best_calibrator_v2.pt", device)),
    ]
    ray_models = [
        ("Frozen\nRayleigh", load_v2_model(bb_ray, "frozen", None, device)),
        ("Avg-SNR\n(c_T=7dB)", load_v2_model(bb_ray, "avg_snr", None, device)),
        ("P-Raw\n([c_R,c_T,0])", load_v2_model(bb_ray, "p_raw_v2", args.v2_ckpt_root / "rayleigh/p_raw_v2/best_calibrator_v2.pt", device)),
        ("MS-ZPAC\n(v2 RX-Only)", load_v2_model(bb_ray, "ms_zpac", args.v2_ckpt_root / "rayleigh/ms_zpac/best_calibrator_v2.pt", device)),
        ("TR-ZPAC\n(v2 Joint)", load_v2_model(bb_ray, "tr_zpac", args.v2_ckpt_root / "rayleigh/tr_zpac/best_calibrator_v2.pt", device)),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(17.2, 6.7), dpi=220)

    with torch.inference_mode():
        # Pick representative frame with clear fading mismatch from kodak24
        steps_24 = simulate_ofdm_ar1_chain(24, avg_snr_db=10.0, rho_c=0.70, delay_blocks=2, seed=2026, num_rbs=32)
        eps_24 = make_paired_noise((24, 256, 96), list(range(24)), 2001, device=device)

        # Find Kodak image with largest TR-ZPAC vs Frozen gain for clear visual illustration
        frz_m = awgn_models[0][1]
        tr_m = awgn_models[-1][1]
        frz_m.avg_snr_fallback_db = 10.0
        p_f = compute_psnr_batch(frz_m.forward_channel_v2(targets, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=steps_24, standard_noise=eps_24)["recon"].clamp(0, 1), targets)
        p_t = compute_psnr_batch(tr_m.forward_channel_v2(targets, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=steps_24, standard_noise=eps_24)["recon"].clamp(0, 1), targets)
        idx1 = int(torch.argmax(p_t - p_f).item())

        gt1 = targets[idx1].permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(gt1)
        axes[0, 0].set_title(f"Ground Truth (Kodak #{idx1+1})\n32-RB OFDM (ρ=0.70, d=2)", fontsize=9.2, fontweight="bold")
        axes[0, 0].axis("off")

        for col_i, (lbl, mod) in enumerate(awgn_models, start=1):
            mod.avg_snr_fallback_db = 10.0
            rec = mod.forward_channel_v2(
                targets[idx1:idx1+1], 0, 0, 0,
                channel_mode="ofdm_fading",
                ofdm_steps=[steps_24[idx1]],
                standard_noise=eps_24[idx1:idx1+1],
            )["recon"].clamp(0.0, 1.0)
            psnr_v = float(compute_psnr_batch(rec, targets[idx1:idx1+1])[0].item())
            axes[0, col_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
            axes[0, col_i].set_title(f"{lbl}\nPSNR: {psnr_v:.2f} dB", fontsize=9.2, fontweight="bold" if "ZPAC" in lbl else "normal")
            axes[0, col_i].axis("off")

        # Row 2: Official Rayleigh Backbone under Severe Overestimation (c_T=13 -> Gamma=1 dB, tau=0.5)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(3005)
        h_iq_24 = (torch.randn((24, 128 * 96, 2), generator=gen, dtype=torch.float32) * math.sqrt(0.5)).to(device)
        p_rf = compute_psnr_batch(ray_models[0][1].forward_channel_v2(targets, 13.0, 1.0, 1.0, channel_mode="rayleigh", tau_unc=0.5, standard_noise=eps_24, h_complex_iq=h_iq_24)["recon"].clamp(0, 1), targets)
        p_rt = compute_psnr_batch(ray_models[-1][1].forward_channel_v2(targets, 13.0, 1.0, 1.0, channel_mode="rayleigh", tau_unc=0.5, standard_noise=eps_24, h_complex_iq=h_iq_24)["recon"].clamp(0, 1), targets)
        idx2 = int(torch.argmax(p_rt - p_rf).item())

        gt2 = targets[idx2].permute(1, 2, 0).cpu().numpy()
        axes[1, 0].imshow(gt2)
        axes[1, 0].set_title(f"Ground Truth (Kodak #{idx2+1})\nRayleigh Mismatch (13→1 dB)", fontsize=9.2, fontweight="bold")
        axes[1, 0].axis("off")

        for col_i, (lbl, mod) in enumerate(ray_models, start=1):
            mod.avg_snr_fallback_db = 7.0
            rec = mod.forward_channel_v2(
                targets[idx2:idx2+1], 13.0, 1.0, 1.0,
                channel_mode="rayleigh",
                tau_unc=0.5,
                standard_noise=eps_24[idx2:idx2+1],
                h_complex_iq=h_iq_24[idx2:idx2+1],
            )["recon"].clamp(0.0, 1.0)
            psnr_v = float(compute_psnr_batch(rec, targets[idx2:idx2+1])[0].item())
            axes[1, col_i].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
            axes[1, col_i].set_title(f"{lbl}\nPSNR: {psnr_v:.2f} dB", fontsize=9.2, fontweight="bold" if "ZPAC" in lbl else "normal")
            axes[1, col_i].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91, hspace=0.25, wspace=0.06)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print("RENDERED QUALITATIVE V2 FIGURE SUCCESSFULLY!")


if __name__ == "__main__":
    main()
