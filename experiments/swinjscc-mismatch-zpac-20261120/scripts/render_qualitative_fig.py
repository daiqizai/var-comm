#!/usr/bin/env python3
"""Render Fig. 5 (Qualitative Visual Reconstruction + Latent Calibration Residual Heatmaps on Kodak)."""
from __future__ import annotations

import argparse
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

from dir2_zpac.hygiene import admit_gpu, configure_runtime_env, configure_torch_determinism, seed_everything, verify_no_links
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.models import CalibratedSwinJSCC, load_swinjscc_backbone
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.metrics import compute_psnr_batch


def load_cal_model(backbone: torch.nn.Module, method: str, ckpt: Path, device: torch.device) -> CalibratedSwinJSCC:
    blob = torch.load(ckpt, map_location="cpu")
    m = CalibratedSwinJSCC(backbone, method=method, hidden_ch=int(blob.get("hidden_ch", 160)))
    sd = m.state_dict()
    for k, v in blob["delta_state_dict"].items():
        sd[k] = v
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--zpac-ckpt", type=Path, required=True)
    parser.add_argument("--praw-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=3)
    args = parser.parse_args()

    configure_runtime_env()
    admit_gpu(args.physical_gpu, min_free_mib=12000)
    device = torch.device("cuda:0" if "CUDA_VISIBLE_DEVICES" in __import__("os").environ else f"cuda:{args.physical_gpu}")
    torch.cuda.set_device(device)
    configure_torch_determinism(num_threads=4)
    seed_everything(20261120)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, device=device)
    m_frozen = CalibratedSwinJSCC(backbone, method="frozen").to(device).eval()
    m_praw = load_cal_model(backbone, "P_raw", args.praw_ckpt, device)
    m_zpac = load_cal_model(backbone, "P_zpac_hier", args.zpac_ckpt, device)

    kodak = load_split_u8("kodak24", allow_holdout=True)
    # Use kodim08 (index 7, rich architectural edges + sky) or kodim23 (index 22, parrot feathers + smooth background)
    idx_over = 22   # kodim23: smooth background + vivid edges -> ideal for overestimation speckle vs edge preservation
    idx_under = 7   # kodim08: fine roof/window details -> ideal for underestimation detail recovery

    fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.6))
    plt.rcParams.update({"font.size": 8.8})

    cases = [
        (0, idx_over, 13.0, 1.0, 1.0, "Overestimation (cT=13, Gamma=1 dB)"),
        (1, idx_under, 1.0, 13.0, 13.0, "Underestimation (cT=1, Gamma=13 dB)"),
    ]

    with torch.inference_mode():
        for row_i, s_idx, ct, gm, cr, title_prefix in cases:
            xb = kodak.images_u8[s_idx : s_idx + 1].to(device=device, dtype=torch.float32) / 255.0
            eps = make_paired_noise((1, 256, m_zpac.latent_channels), [s_idx], 2001, device)

            out_f = m_frozen.forward_awgn(xb, ct, gm, cr, standard_noise=eps, return_latents=True)
            out_r = m_praw.forward_awgn(xb, ct, gm, cr, standard_noise=eps, return_latents=True)
            out_z = m_zpac.forward_awgn(xb, ct, gm, cr, standard_noise=eps, return_latents=True)

            rec_f = out_f["recon"].clamp(0.0, 1.0)
            rec_r = out_r["recon"].clamp(0.0, 1.0)
            rec_z = out_z["recon"].clamp(0.0, 1.0)

            pf = float(compute_psnr_batch(xb, rec_f).item())
            pr = float(compute_psnr_batch(xb, rec_r).item())
            pz = float(compute_psnr_batch(xb, rec_z).item())

            # Compute 16x16 spatial RMS of latent calibration delta ||u_cal - u_restored||_2 / sqrt(C)
            delta_u = (out_z["u_cal"] - out_z["u_restored"]).reshape(16, 16, -1)
            delta_rms = delta_u.square().mean(dim=-1).sqrt().cpu().numpy()

            # Zoom into informative 128x128 ROI
            y0, x0, sz = (48, 64, 128) if row_i == 0 else (64, 64, 128)

            def to_hwc(t: torch.Tensor) -> np.ndarray:
                return t[0, :, y0 : y0 + sz, x0 : x0 + sz].permute(1, 2, 0).cpu().numpy()

            panels = [
                (to_hwc(xb), f"Ground Truth ({kodak.source_ids[s_idx]})\n{title_prefix}"),
                (to_hwc(rec_f), f"Original SwinJSCC\nPSNR = {pf:.2f} dB"),
                (to_hwc(rec_r), f"P-Raw ([cR,cT,0])\nPSNR = {pr:.2f} dB ({pr - pf:+.2f})"),
                (to_hwc(rec_z), f"P-ZPAC-Hier (Ours)\nPSNR = {pz:.2f} dB ({pz - pf:+.2f})"),
            ]
            for col_i, (img_np, ttl) in enumerate(panels):
                ax = axes[row_i, col_i]
                ax.imshow(np.clip(img_np, 0.0, 1.0))
                ax.set_title(ttl, fontsize=8.5)
                ax.axis("off")

            ax_hm = axes[row_i, 4]
            im = ax_hm.imshow(delta_rms, cmap="magma" if row_i == 0 else "viridis", interpolation="nearest")
            ax_hm.set_title(
                "ZPAC ||Delta u|| Heatmap\n(R+ Noise Shrinkage)" if row_i == 0 else "ZPAC ||Delta u|| Heatmap\n(R- Detail Excitation)",
                fontsize=8.5,
            )
            ax_hm.set_xticks([])
            ax_hm.set_yticks([])
            fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.28)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"fig5_qualitative_kodak.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)
    verify_no_links([args.output_dir])


if __name__ == "__main__":
    main()
