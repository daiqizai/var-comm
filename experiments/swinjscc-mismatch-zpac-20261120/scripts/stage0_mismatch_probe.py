#!/usr/bin/env python3
"""Stage 0 (Checklist 01-05): Zero-training (c_T, Gamma, c_R) split verification, 22-condition probe, and latent shift diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from dir2_zpac.hygiene import (
    admit_gpu,
    configure_runtime_env,
    configure_torch_determinism,
    save_json,
    seed_everything,
    sha256_file,
    verify_no_links,
)
from dir2_zpac.datasets import load_split_u8
from dir2_zpac.models import CalibratedSwinJSCC, encode_with_ct, load_swinjscc_backbone
from dir2_zpac.phy_channel import awgn_forward, make_paired_noise
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci


STAGE0_CONDITIONS: list[tuple[str, str, float, float, float]] = [
    # Group A: Matched Oracle (c_T = Gamma = c_R)
    ("A_matched", "matched_1", 1.0, 1.0, 1.0),
    ("A_matched", "matched_4", 4.0, 4.0, 4.0),
    ("A_matched", "matched_7", 7.0, 7.0, 7.0),
    ("A_matched", "matched_10", 10.0, 10.0, 10.0),
    ("A_matched", "matched_13", 13.0, 13.0, 13.0),
    # Group B_over: TX Overestimation (c_T > Gamma, c_R = Gamma)
    ("B_tx_over", "over_13_1_1", 13.0, 1.0, 1.0),
    ("B_tx_over", "over_10_4_4", 10.0, 4.0, 4.0),
    ("B_tx_over", "over_13_4_4", 13.0, 4.0, 4.0),
    ("B_tx_over", "over_10_7_7", 10.0, 7.0, 7.0),
    ("B_tx_over", "over_13_7_7", 13.0, 7.0, 7.0),
    ("B_tx_over", "over_13_10_10", 13.0, 10.0, 10.0),
    # Group B_under: TX Underestimation (c_T < Gamma, c_R = Gamma)
    ("B_tx_under", "under_1_4_4", 1.0, 4.0, 4.0),
    ("B_tx_under", "under_1_7_7", 1.0, 7.0, 7.0),
    ("B_tx_under", "under_4_7_7", 4.0, 7.0, 7.0),
    ("B_tx_under", "under_1_10_10", 1.0, 10.0, 10.0),
    ("B_tx_under", "under_4_10_10", 4.0, 10.0, 10.0),
    ("B_tx_under", "under_1_13_13", 1.0, 13.0, 13.0),
    ("B_tx_under", "under_7_13_13", 7.0, 13.0, 13.0),
    # Group C: RX-only Mismatch (c_T = Gamma, c_R != Gamma)
    ("C_rx_only", "rx_over_4_4_13", 4.0, 4.0, 13.0),
    ("C_rx_only", "rx_over_7_7_13", 7.0, 7.0, 13.0),
    ("C_rx_only", "rx_under_10_10_4", 10.0, 10.0, 4.0),
    ("C_rx_only", "rx_under_7_7_1", 7.0, 7.0, 1.0),
    # Group D: Both Stale (c_T = c_R != Gamma)
    ("D_both_stale", "both_13_4_13", 13.0, 4.0, 13.0),
    ("D_both_stale", "both_10_4_10", 10.0, 4.0, 10.0),
    ("D_both_stale", "both_13_7_13", 13.0, 7.0, 13.0),
    ("D_both_stale", "both_1_10_1", 1.0, 10.0, 1.0),
    # Group E: Fixed Average-SNR Fallback at TX (c_T = 7 dB, c_R = Gamma)
    ("E_avg_tx7", "avg_7_1_1", 7.0, 1.0, 1.0),
    ("E_avg_tx7", "avg_7_4_4", 7.0, 4.0, 4.0),
    ("E_avg_tx7", "avg_7_10_10", 7.0, 10.0, 10.0),
    ("E_avg_tx7", "avg_7_13_13", 7.0, 13.0, 13.0),
]


@torch.inference_mode()
def verify_upstream_equivalence(
    backbone: torch.nn.Module,
    sample_img: torch.Tensor,
    rate_channels: int | None,
) -> dict:
    """Checklist 02: Verify exact numerical equivalence (< 1e-6) against upstream noiseless & AWGN paths at B=1."""
    img1 = sample_img[:1]
    checks = []
    orig_pass = backbone.pass_channel
    try:
        backbone.pass_channel = False
        for snr in (1, 4, 7, 10, 13):
            ref_recon, cbr, _, _, _ = backbone(img1, given_SNR=snr, given_rate=rate_channels)
            z_raw, mask = encode_with_ct(backbone, img1, float(snr), rate_channels=rate_channels)
            if mask is not None:
                z_in = z_raw * mask
            else:
                z_in = z_raw
            wrap_model = CalibratedSwinJSCC(backbone, method="frozen").to(img1.device).eval()
            our_recon = wrap_model.decode_latent(z_in, float(snr), float(snr))
            max_diff = float((ref_recon - our_recon).abs().max().item())
            assert max_diff < 1e-6, f"Equivalence failed at SNR={snr}: max_diff={max_diff}"
            checks.append({"snr_db": snr, "cbr": float(cbr), "noiseless_max_abs_diff": max_diff})

            # Also verify zero-locked identity of P_zpac at matched condition (c_T == c_R)
            zpac_model = CalibratedSwinJSCC(backbone, method="P_zpac").to(img1.device).eval()
            # Even if we randomize branch weights, zero-locked gate guarantees exact identity when c_T == c_R!
            for p in zpac_model.frontend.parameters():
                if p.ndim >= 2:
                    torch.nn.init.normal_(p, mean=0.0, std=0.05)
            ch_out = awgn_forward(
                z_raw, float(snr), float(snr), standard_noise=torch.zeros_like(z_raw), active_mask=mask
            )
            u_cal, _ = zpac_model.frontend(
                ch_out.u_restored,
                ch_out.u_norm,
                torch.tensor([float(snr)], device=img1.device),
                torch.tensor([float(snr)], device=img1.device),
                ch_out.sigma0_sq_cr,
                active_mask=mask,
            )
            gate_diff = float((u_cal - ch_out.u_restored).abs().max().item())
            assert gate_diff == 0.0, f"Zero-locked gate leaked at matched SNR={snr}: {gate_diff}"
            checks[-1]["zpac_matched_gate_abs_diff"] = gate_diff
    finally:
        backbone.pass_channel = orig_pass
    return {"status": "PASS", "checks": checks}


@torch.inference_mode()
def diagnose_latent_shift(
    backbone: torch.nn.Module,
    images: torch.Tensor,
    rate_channels: int | None,
    device: torch.device,
) -> dict:
    """Export sorted per-channel variance curves and cosine/MSE shift under mismatch (for Paper Fig. 1a)."""
    B = images.shape[0]
    batch_sz = 16
    snrs = [1.0, 4.0, 7.0, 10.0, 13.0]
    channel_vars: dict[str, list[float]] = {}

    for ct in snrs:
        var_accum = []
        for s in range(0, B, batch_sz):
            xb = images[s : s + batch_sz].to(device, dtype=torch.float32) / 255.0
            z_raw, mask = encode_with_ct(backbone, xb, ct, rate_channels=rate_channels)
            ch_out = awgn_forward(z_raw, ct, ct, standard_noise=torch.zeros_like(z_raw), active_mask=mask)
            # Per-channel variance across tokens (B, L, C) -> (B, C)
            v = ch_out.z_norm.var(dim=1, unbiased=False)
            var_accum.append(v.cpu())
        v_all = torch.cat(var_accum, dim=0).mean(dim=0)  # (C,)
        v_sorted, _ = torch.sort(v_all, descending=True)
        channel_vars[f"cT_{int(ct)}dB"] = [float(x) for x in v_sorted.tolist()]

    # Compare cosine similarity & relative L2 error of u_restored(c_T, Gamma) vs clean z_raw(Gamma)
    shift_pairs = [
        ("matched_4_4", 4.0, 4.0),
        ("over_13_4", 13.0, 4.0),
        ("over_10_4", 10.0, 4.0),
        ("matched_7_7", 7.0, 7.0),
        ("over_13_7", 13.0, 7.0),
        ("under_1_7", 1.0, 7.0),
        ("matched_10_10", 10.0, 10.0),
        ("under_1_10", 1.0, 10.0),
        ("under_4_10", 4.0, 10.0),
    ]
    shift_stats = {}
    for label, ct, gm in shift_pairs:
        cos_clean_list = []
        cos_noisy_list = []
        rel_mse_list = []
        for s in range(0, B, batch_sz):
            xb = images[s : s + batch_sz].to(device, dtype=torch.float32) / 255.0
            idx_list = list(range(s, min(B, s + batch_sz)))
            z_oracle, m_oracle = encode_with_ct(backbone, xb, gm, rate_channels=rate_channels)
            z_ct, m_ct = encode_with_ct(backbone, xb, ct, rate_channels=rate_channels)
            eps = make_paired_noise(tuple(z_ct.shape), idx_list, 9101, device)
            ch_ct = awgn_forward(z_ct, gm, gm, standard_noise=eps, active_mask=m_ct)

            zo_flat = z_oracle.flatten(1)
            zct_flat = z_ct.flatten(1)
            uct_flat = ch_ct.u_restored.flatten(1)
            cos_c = F.cosine_similarity(zo_flat, zct_flat, dim=1)
            cos_n = F.cosine_similarity(zo_flat, uct_flat, dim=1)
            rel_mse = (uct_flat - zo_flat).square().sum(dim=1) / zo_flat.square().sum(dim=1).clamp_min(1e-8)
            cos_clean_list.extend(cos_c.cpu().tolist())
            cos_noisy_list.extend(cos_n.cpu().tolist())
            rel_mse_list.extend(rel_mse.cpu().tolist())

        shift_stats[label] = {
            "c_t_db": ct,
            "gamma_db": gm,
            "noiseless_tx_cosine_to_oracle": float(np.mean(cos_clean_list)),
            "received_u_cosine_to_oracle": float(np.mean(cos_noisy_list)),
            "received_u_rel_mse_to_oracle": float(np.mean(rel_mse_list)),
        }
    return {"sorted_channel_variances": channel_vars, "latent_shift_summary": shift_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 Mismatch Diagnostic Probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-mode", type=str, default="SwinJSCC_w/_SA")
    parser.add_argument("--c-bottleneck", type=int, default=96)
    parser.add_argument("--rate-channels", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[9101, 9102, 9103])
    args = parser.parse_args()

    configure_runtime_env()
    gpu_info = admit_gpu(args.physical_gpu, min_free_mib=24000)
    # Note: assume CUDA_VISIBLE_DEVICES is set by caller or bind directly
    device = torch.device("cuda:0" if "CUDA_VISIBLE_DEVICES" in __import__("os").environ else f"cuda:{gpu_info['index']}")
    torch.cuda.set_device(device)
    configure_torch_determinism(num_threads=6)
    seed_everything(20261120)

    t0 = time.time()
    backbone = load_swinjscc_backbone(
        args.checkpoint,
        model_mode=args.model_mode,
        c_bottleneck=args.c_bottleneck,
        device=device,
    )
    model = CalibratedSwinJSCC(backbone, method="frozen").to(device).eval()
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)

    dev_split = load_split_u8("dev64")
    n_imgs = len(dev_split.source_ids)

    sample0 = dev_split.images_u8[:2].to(device=device, dtype=torch.float32) / 255.0
    equiv_report = verify_upstream_equivalence(backbone, sample0, args.rate_channels)
    latent_report = diagnose_latent_shift(backbone, dev_split.images_u8, args.rate_channels, device)

    per_cond_source_scores: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict] = []
    batch_sz = 16

    for group, name, ct, gm, cr in STAGE0_CONDITIONS:
        metrics_acc = {k: np.zeros((n_imgs, len(args.noise_seeds)), dtype=np.float64) for k in ("psnr_db", "ms_ssim", "lpips_alex", "dists")}
        for r_idx, nseed in enumerate(args.noise_seeds):
            for s in range(0, n_imgs, batch_sz):
                e = min(n_imgs, s + batch_sz)
                xb = dev_split.images_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
                src_idxs = list(range(s, e))
                # Determine latent shape for paired noise
                lat_shape = (e - s, 256, model.latent_channels)
                eps = make_paired_noise(lat_shape, src_idxs, nseed, device)
                with torch.inference_mode():
                    out = model.forward_awgn(
                        xb, ct, gm, cr, standard_noise=eps, rate_channels=args.rate_channels
                    )
                    recon = out["recon"].clamp(0.0, 1.0)
                scores = evaluator.evaluate_batch(xb, recon)
                for mk, vals in scores.items():
                    metrics_acc[mk][s:e, r_idx] = vals

        source_means = {mk: arr.mean(axis=1) for mk, arr in metrics_acc.items()}
        per_cond_source_scores[name] = source_means

        row = {
            "group": group,
            "condition": name,
            "c_t_db": ct,
            "gamma_db": gm,
            "c_r_db": cr,
            "delta_c_db": ct - cr,
        }
        for mk, s_arr in source_means.items():
            st = paired_bootstrap_ci(s_arr, num_resamples=10000, seed=20261120)
            row[f"{mk}_mean"] = round(st["mean"], 6)
            row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
            row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
            row[f"{mk}_p10"] = round(st["p10"], 6)
            row[f"{mk}_p90"] = round(st["p90"], 6)
        summary_rows.append(row)
        print(
            f"[Stage0] {name:18s} (cT={ct:4.1f}, G={gm:4.1f}, cR={cr:4.1f}) | "
            f"PSNR={row['psnr_db_mean']:.3f} dB | MS-SSIM={row['ms_ssim_mean']:.4f} | "
            f"DISTS={row['dists_mean']:.4f} | LPIPS={row['lpips_alex_mean']:.4f}",
            flush=True,
        )

    # Compute key Stage 0 Go/No-Go paired gaps (Oracle Matched A minus Mismatched B)
    paired_gaps = {}
    gap_specs = [
        ("over_13_1_vs_matched_1", "matched_1", "over_13_1_1"),
        ("over_13_4_vs_matched_4", "matched_4", "over_13_4_4"),
        ("over_10_4_vs_matched_4", "matched_4", "over_10_4_4"),
        ("over_13_7_vs_matched_7", "matched_7", "over_13_7_7"),
        ("over_10_7_vs_matched_7", "matched_7", "over_10_7_7"),
        ("under_1_7_vs_matched_7", "matched_7", "under_1_7_7"),
        ("under_1_10_vs_matched_10", "matched_10", "under_1_10_10"),
        ("under_1_13_vs_matched_13", "matched_13", "under_1_13_13"),
        ("both_13_4_13_vs_matched_4", "matched_4", "both_13_4_13"),
        ("rx_only_4_4_13_vs_matched_4", "matched_4", "rx_over_4_4_13"),
    ]
    for gname, ref_c, mis_c in gap_specs:
        paired_gaps[gname] = {
            mk: paired_bootstrap_ci(
                per_cond_source_scores[ref_c][mk],
                per_cond_source_scores[mis_c][mk],
                num_resamples=10000,
                seed=20261120,
            )
            for mk in ("psnr_db", "ms_ssim", "dists", "lpips_alex")
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "stage0_grid_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    full_summary = {
        "schema": "DIR2_STAGE0_MISMATCH_PROBE_V1",
        "elapsed_seconds": time.time() - t0,
        "gpu": gpu_info,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "model_mode": args.model_mode,
            "c_bottleneck": args.c_bottleneck,
            "rate_channels": args.rate_channels,
        },
        "population": "dev64 (calibration[:64], holdout untouched)",
        "noise_seeds": args.noise_seeds,
        "upstream_equivalence": equiv_report,
        "paired_oracle_minus_mismatch_gaps": paired_gaps,
        "latent_diagnosis": latent_report,
    }
    save_json(args.output_dir / "stage0_summary.json", full_summary)
    hygiene = verify_no_links([args.output_dir])
    save_json(args.output_dir / "hygiene_receipt.json", hygiene)
    print(f"[Stage0] Completed in {full_summary['elapsed_seconds']:.1f}s -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
