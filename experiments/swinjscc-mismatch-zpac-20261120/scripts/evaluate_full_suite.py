#!/usr/bin/env python3
"""Stage 2 & Stage 3 Full Held-Out Evaluation, Unseen Offset Generalization, AR(1) Fading Delay, Complexity & Figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

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
from dir2_zpac.models import CalibratedSwinJSCC, load_swinjscc_backbone
from dir2_zpac.phy_channel import make_paired_noise, simulate_ar1_fading_chain
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci


AWGN_TEST_CONDITIONS: list[tuple[str, str, float, float, float]] = [
    # 1. Matched conditions (Delta c = 0)
    ("matched", "matched_1", 1.0, 1.0, 1.0),
    ("matched", "matched_4", 4.0, 4.0, 4.0),
    ("matched", "matched_7", 7.0, 7.0, 7.0),
    ("matched", "matched_10", 10.0, 10.0, 10.0),
    ("matched", "matched_13", 13.0, 13.0, 13.0),
    # 2. Seen in-distribution TX Overestimation (+3, +6 dB)
    ("seen_over", "over_7_4_4", 7.0, 4.0, 4.0),
    ("seen_over", "over_10_4_4", 10.0, 4.0, 4.0),
    ("seen_over", "over_10_7_7", 10.0, 7.0, 7.0),
    ("seen_over", "over_13_7_7", 13.0, 7.0, 7.0),
    # 3. Seen in-distribution TX Underestimation (-3, -6 dB)
    ("seen_under", "under_1_4_4", 1.0, 4.0, 4.0),
    ("seen_under", "under_1_7_7", 1.0, 7.0, 7.0),
    ("seen_under", "under_4_7_7", 4.0, 7.0, 7.0),
    ("seen_under", "under_4_10_10", 4.0, 10.0, 10.0),
    ("seen_under", "under_7_13_13", 7.0, 13.0, 13.0),
    # 4. UNSEEN Interpolation Offsets (+-2, +-4 dB)
    ("unseen_interp", "interp_over_6_4_4", 6.0, 4.0, 4.0),
    ("unseen_interp", "interp_over_8_4_4", 8.0, 4.0, 4.0),
    ("unseen_interp", "interp_over_11_7_7", 11.0, 7.0, 7.0),
    ("unseen_interp", "interp_under_5_7_7", 5.0, 7.0, 7.0),
    ("unseen_interp", "interp_under_3_7_7", 3.0, 7.0, 7.0),
    ("unseen_interp", "interp_under_6_10_10", 6.0, 10.0, 10.0),
    # 5. UNSEEN Extrapolation Extreme Offsets (+-8, +-9, +-12 dB)
    ("unseen_extrap", "extrap_over_12_4_4", 12.0, 4.0, 4.0),
    ("unseen_extrap", "extrap_over_13_4_4", 13.0, 4.0, 4.0),
    ("unseen_extrap", "extrap_over_13_1_1", 13.0, 1.0, 1.0),
    ("unseen_extrap", "extrap_under_2_10_10", 2.0, 10.0, 10.0),
    ("unseen_extrap", "extrap_under_1_10_10", 1.0, 10.0, 10.0),
    ("unseen_extrap", "extrap_under_1_13_13", 1.0, 13.0, 13.0),
    # 6. Imperfect Receiver CSI (c_R != Gamma)
    ("imperfect_rx", "imprx_over_13_4_5.5", 13.0, 4.0, 5.5),
    ("imperfect_rx", "imprx_over_10_4_2.5", 10.0, 4.0, 2.5),
    ("imperfect_rx", "imprx_under_1_10_8.5", 1.0, 10.0, 8.5),
    # 7. Sweep at Gamma = 7 dB for Fig. 2 curve: c_T - Gamma in {-6, -4, -2, 0, +2, +4, +6, +8}
    ("sweep_g7", "sweep_g7_m6", 1.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_m4", 3.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_m2", 5.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_0", 7.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_p2", 9.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_p4", 11.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_p6", 13.0, 7.0, 7.0),
    ("sweep_g7", "sweep_g7_p8", 15.0, 7.0, 7.0),
]


def load_method_model(
    backbone: torch.nn.Module,
    spec: str,
    device: torch.device,
) -> tuple[str, CalibratedSwinJSCC]:
    """Parse spec 'label:method[:ckpt_path]' and construct frozen evaluation model."""
    parts = spec.split(":", 2)
    label = parts[0]
    method = parts[1]
    ckpt_path = Path(parts[2]) if len(parts) == 3 and parts[2] else None

    if ckpt_path is not None:
        blob = torch.load(ckpt_path, map_location="cpu")
        hidden_ch = int(blob.get("hidden_ch", 128))
        model = CalibratedSwinJSCC(backbone, method=method, hidden_ch=hidden_ch)
        sd = model.state_dict()
        for k, v in blob["delta_state_dict"].items():
            assert k in sd, f"Unexpected key {k} in {ckpt_path}"
            sd[k] = v
        model.load_state_dict(sd, strict=True)
    else:
        model = CalibratedSwinJSCC(backbone, method=method)
    return label, model.to(device).eval()


@torch.inference_mode()
def benchmark_rx_complexity(
    models_dict: dict[str, CalibratedSwinJSCC],
    device: torch.device,
) -> list[dict]:
    """Checklist 15: Measure parameter count and median RX latency (ms/img) on A800."""
    u_dummy = torch.randn(1, 256, next(iter(models_dict.values())).latent_channels, device=device)
    un_dummy = u_dummy * 0.5
    ct_t = torch.tensor([13.0], device=device)
    cr_t = torch.tensor([4.0], device=device)
    s0_t = torch.tensor([0.5 * (10.0 ** (-4.0 / 10.0))], device=device)

    rows = []
    for label, m in models_dict.items():
        total_p = sum(p.numel() for p in m.parameters())
        enc_p = sum(p.numel() for p in m.backbone.encoder.parameters())
        dec_base_p = sum(p.numel() for p in m.backbone.decoder.parameters())
        front_p = sum(p.numel() for p in m.frontend.parameters()) if m.frontend is not None else 0
        hier_p = sum(p.numel() for p in m.hier_delta.parameters()) if m.hier_delta is not None else 0
        rx_added_p = front_p + hier_p + (
            sum(bm.fc[0].linear.weight[:, 0].numel() for bm in m.backbone.decoder.bm_list) if m.dual_decoder_bm else 0
        )

        # Warmup
        for _ in range(25):
            if m.frontend is not None:
                uc, _ = m.frontend(u_dummy, un_dummy, ct_t, cr_t, s0_t)
            else:
                uc = u_dummy
            _ = m.decode_latent(uc, ct_t, cr_t)
        torch.cuda.synchronize(device)

        timings_ms = []
        for _ in range(120):
            torch.cuda.synchronize(device)
            t_s = time.perf_counter()
            if m.frontend is not None:
                uc, _ = m.frontend(u_dummy, un_dummy, ct_t, cr_t, s0_t)
            else:
                uc = u_dummy
            _ = m.decode_latent(uc, ct_t, cr_t)
            torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - t_s) * 1000.0)

        rows.append(
            {
                "label": label,
                "method": m.method,
                "total_params": total_p,
                "rx_decoder_plus_cal_params": dec_base_p + front_p + hier_p,
                "added_calibrator_params": rx_added_p,
                "added_rx_pct": round(100.0 * rx_added_p / max(1, dec_base_p), 4),
                "rx_median_ms": round(float(np.median(timings_ms)), 4),
                "rx_p90_ms": round(float(np.quantile(timings_ms, 0.90)), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Frozen Calibrators on Independent Test Sets")
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--model-mode", type=str, default="SwinJSCC_w/_SA")
    parser.add_argument("--c-bottleneck", type=int, default=96)
    parser.add_argument("--rate-channels", type=int, default=None)
    parser.add_argument("--models", type=str, nargs="+", required=True, help="List of 'label:method[:ckpt_path]'")
    parser.add_argument("--primary-label", type=str, default="P_ZPAC")
    parser.add_argument("--split", type=str, required=True, choices=["dev64", "val256", "common500", "kodak24"])
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[6201, 6202, 6203])
    parser.add_argument("--skip-awgn", action="store_true")
    parser.add_argument("--run-fading", action="store_true")
    parser.add_argument("--run-complexity", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=None)
    args = parser.parse_args()

    configure_runtime_env()
    gpu_info = admit_gpu(args.physical_gpu, min_free_mib=24000)
    device = torch.device("cuda:0" if "CUDA_VISIBLE_DEVICES" in __import__("os").environ else f"cuda:{gpu_info['index']}")
    torch.cuda.set_device(device)
    configure_torch_determinism(num_threads=6)
    seed_everything(20261120)

    t0 = time.time()
    backbone = load_swinjscc_backbone(
        args.backbone_ckpt,
        model_mode=args.model_mode,
        c_bottleneck=args.c_bottleneck,
        device=device,
    )
    models_dict: dict[str, CalibratedSwinJSCC] = {}
    for spec in args.models:
        lbl, m = load_method_model(backbone, spec, device)
        models_dict[lbl] = m

    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)
    split_data = load_split_u8(args.split, allow_holdout=args.allow_holdout)
    n_imgs = len(split_data.source_ids)
    batch_sz = 25 if n_imgs >= 100 else 16

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. AWGN Evaluation Suite
    awgn_scores: dict[str, dict[str, dict[str, np.ndarray]]] = {lbl: {} for lbl in models_dict}
    awgn_rows: list[dict] = []
    paired_diffs_rows: list[dict] = []

    if not args.skip_awgn:
        for cat, cname, ct, gm, cr in AWGN_TEST_CONDITIONS:
            for lbl, m in models_dict.items():
                acc = {mk: np.zeros((n_imgs, len(args.noise_seeds)), dtype=np.float64) for mk in ("psnr_db", "ms_ssim", "lpips_alex", "dists")}
                for r_idx, nseed in enumerate(args.noise_seeds):
                    for s in range(0, n_imgs, batch_sz):
                        e = min(n_imgs, s + batch_sz)
                        xb = split_data.images_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
                        eps = make_paired_noise((e - s, 256, m.latent_channels), list(range(s, e)), nseed, device)
                        with torch.inference_mode():
                            out = m.forward_awgn(xb, ct, gm, cr, standard_noise=eps, rate_channels=args.rate_channels)
                            recon = out["recon"].clamp(0.0, 1.0)
                        sc = evaluator.evaluate_batch(xb, recon)
                        for mk, vals in sc.items():
                            acc[mk][s:e, r_idx] = vals
                s_means = {mk: arr.mean(axis=1) for mk, arr in acc.items()}
                awgn_scores[lbl][cname] = s_means

                row = {
                    "split": args.split,
                    "category": cat,
                    "condition": cname,
                    "model": lbl,
                    "c_t_db": ct,
                    "gamma_db": gm,
                    "c_r_db": cr,
                    "delta_c_db": ct - cr,
                }
                for mk, s_arr in s_means.items():
                    st = paired_bootstrap_ci(s_arr, num_resamples=10000, seed=20261120)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                    row[f"{mk}_p90"] = round(st["p90"], 6)
                awgn_rows.append(row)

            if args.primary_label in models_dict:
                p_row = next(r for r in reversed(awgn_rows) if r["condition"] == cname and r["model"] == args.primary_label)
                f_row = next((r for r in reversed(awgn_rows) if r["condition"] == cname and r["model"] == "Frozen"), p_row)
                print(
                    f"[Eval:{args.split}] {cname:22s} | Frozen PSNR={f_row['psnr_db_mean']:.3f}, DISTS={f_row['dists_mean']:.4f} "
                    f"-> {args.primary_label} PSNR={p_row['psnr_db_mean']:.3f}, DISTS={p_row['dists_mean']:.4f}",
                    flush=True,
                )

        categories_to_agg = ["matched", "seen_over", "seen_under", "unseen_interp", "unseen_extrap", "imperfect_rx"]
        if args.primary_label in models_dict:
            for cat in categories_to_agg:
                cat_conds = [cname for c, cname, *_ in AWGN_TEST_CONDITIONS if c == cat]
                for other_lbl in models_dict:
                    if other_lbl == args.primary_label:
                        continue
                    row = {"split": args.split, "category": cat, "comparison": f"{args.primary_label}_minus_{other_lbl}"}
                    for mk in ("psnr_db", "ms_ssim", "dists", "lpips_alex"):
                        prim_src = np.mean([awgn_scores[args.primary_label][cn][mk] for cn in cat_conds], axis=0)
                        oth_src = np.mean([awgn_scores[other_lbl][cn][mk] for cn in cat_conds], axis=0)
                        st = paired_bootstrap_ci(prim_src, oth_src, num_resamples=10000, seed=20261120)
                        row[f"{mk}_diff_mean"] = round(st["mean"], 6)
                        row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    paired_diffs_rows.append(row)

        with open(args.output_dir / f"awgn_results_{args.split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(awgn_rows[0].keys()))
            w.writeheader()
            w.writerows(awgn_rows)

        if paired_diffs_rows:
            with open(args.output_dir / f"awgn_paired_diffs_{args.split}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(paired_diffs_rows[0].keys()))
                w.writeheader()
                w.writerows(paired_diffs_rows)

    # 2. Time-Correlated AR(1) Rayleigh Block Fading + Feedback Delay Evaluation (Checklist 16-17)
    fading_rows: list[dict] = []
    if args.run_fading:
        fading_configs = [
            (10.0, 0.90, 0),
            (10.0, 0.90, 1),
            (10.0, 0.90, 2),
            (10.0, 0.90, 3),
            (10.0, 0.70, 1),
            (10.0, 0.70, 2),
            (10.0, 0.70, 3),
            (7.0, 0.85, 2),
        ]
        for avg_snr, rho_c, delay_d in fading_configs:
            # Generate paired fading steps per (realization_seed, source_idx)
            trajs_by_seed = {
                nseed: simulate_ar1_fading_chain(
                    n_imgs, avg_snr_db=avg_snr, rho_c=rho_c, delay_blocks=delay_d, seed=nseed * 17 + int(rho_c * 100)
                )
                for nseed in args.noise_seeds
            }
            for lbl, m in models_dict.items():
                acc = {mk: np.zeros((n_imgs, len(args.noise_seeds)), dtype=np.float64) for mk in ("psnr_db", "ms_ssim", "lpips_alex", "dists")}
                abs_mismatch = []
                for r_idx, nseed in enumerate(args.noise_seeds):
                    steps_all = trajs_by_seed[nseed]
                    abs_mismatch.extend(abs(st.c_t_db - st.c_r_db) for st in steps_all)
                    for s in range(0, n_imgs, batch_sz):
                        e = min(n_imgs, s + batch_sz)
                        xb = split_data.images_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
                        eps = make_paired_noise((e - s, 256, m.latent_channels), list(range(s, e)), nseed + 500, device)
                        with torch.inference_mode():
                            out = m.forward_fading(
                                xb, steps_all[s:e], standard_noise=eps, rate_channels=args.rate_channels
                            )
                            recon = out["recon"].clamp(0.0, 1.0)
                        sc = evaluator.evaluate_batch(xb, recon)
                        for mk, vals in sc.items():
                            acc[mk][s:e, r_idx] = vals
                s_means = {mk: arr.mean(axis=1) for mk, arr in acc.items()}
                frow = {
                    "split": args.split,
                    "avg_snr_db": avg_snr,
                    "rho_c": rho_c,
                    "delay_blocks": delay_d,
                    "mean_abs_mismatch_db": round(float(np.mean(abs_mismatch)), 4),
                    "model": lbl,
                }
                for mk, s_arr in s_means.items():
                    st = paired_bootstrap_ci(s_arr, num_resamples=10000, seed=20261120)
                    frow[f"{mk}_mean"] = round(st["mean"], 6)
                    frow[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    frow[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    frow[f"{mk}_p10"] = round(st["p10"], 6)
                    frow[f"{mk}_p90"] = round(st["p90"], 6)
                fading_rows.append(frow)
            print(f"[Fading:{args.split}] SNR={avg_snr}dB, rho={rho_c}, d={delay_d} done", flush=True)

        with open(args.output_dir / f"fading_results_{args.split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fading_rows[0].keys()))
            w.writeheader()
            w.writerows(fading_rows)

    complexity_rows = benchmark_rx_complexity(models_dict, device) if args.run_complexity else []
    if complexity_rows:
        with open(args.output_dir / "complexity_table.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(complexity_rows[0].keys()))
            w.writeheader()
            w.writerows(complexity_rows)

    save_json(
        args.output_dir / f"eval_summary_{args.split}.json",
        {
            "schema": "DIR2_FULL_EVALUATION_V1",
            "split": args.split,
            "is_holdout": split_data.is_holdout,
            "source_count": n_imgs,
            "noise_seeds": args.noise_seeds,
            "elapsed_seconds": time.time() - t0,
            "gpu": gpu_info,
            "models": args.models,
            "paired_category_differences": paired_diffs_rows,
            "complexity": complexity_rows,
        },
    )
    verify_no_links([args.output_dir])


if __name__ == "__main__":
    main()
