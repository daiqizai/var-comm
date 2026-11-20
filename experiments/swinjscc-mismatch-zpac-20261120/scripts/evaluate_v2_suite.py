#!/usr/bin/env python3
"""V2 Full Held-Out Evaluation Suite: Controlled Mismatch, Extrapolation, and 32-RB 5G/OFDM Delayed Fading."""
from __future__ import annotations

import argparse
import csv
import math
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
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.phy_v2 import OFDMFadingStep, simulate_ofdm_ar1_chain
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci


# Controlled Mismatch & Generalization Grid: (category, name, c_t_db, gamma_db, c_r_db, tau_unc)
STATIC_CONDITIONS: list[tuple[str, str, float, float, float, float]] = [
    # 1. Matched Conditions (Delta c = 0, tau = 0)
    ("matched", "matched_1", 1.0, 1.0, 1.0, 0.0),
    ("matched", "matched_4", 4.0, 4.0, 4.0, 0.0),
    ("matched", "matched_7", 7.0, 7.0, 7.0, 0.0),
    ("matched", "matched_10", 10.0, 10.0, 10.0, 0.0),
    ("matched", "matched_13", 13.0, 13.0, 13.0, 0.0),
    # 2. Seen In-Distribution Overestimation (+3, +6 dB, unaware legacy TX tau=0)
    ("seen_over", "over_4_1_1", 4.0, 1.0, 1.0, 0.0),
    ("seen_over", "over_7_1_1", 7.0, 1.0, 1.0, 0.0),
    ("seen_over", "over_7_4_4", 7.0, 4.0, 4.0, 0.0),
    ("seen_over", "over_10_4_4", 10.0, 4.0, 4.0, 0.0),
    ("seen_over", "over_13_7_7", 13.0, 7.0, 7.0, 0.0),
    # 3. Seen In-Distribution Underestimation (-3, -6 dB, tau=0)
    ("seen_under", "under_1_4_4", 1.0, 4.0, 4.0, 0.0),
    ("seen_under", "under_1_7_7", 1.0, 7.0, 7.0, 0.0),
    ("seen_under", "under_4_7_7", 4.0, 7.0, 7.0, 0.0),
    ("seen_under", "under_4_10_10", 4.0, 10.0, 10.0, 0.0),
    ("seen_under", "under_7_13_13", 7.0, 13.0, 13.0, 0.0),
    # 4. Unseen Interpolation Offsets (+-2, +-4 dB)
    ("unseen_interp", "interp_5_1_1", 5.0, 1.0, 1.0, 0.0),
    ("unseen_interp", "interp_6_4_4", 6.0, 4.0, 4.0, 0.0),
    ("unseen_interp", "interp_8_4_4", 8.0, 4.0, 4.0, 0.0),
    ("unseen_interp", "interp_3_7_7", 3.0, 7.0, 7.0, 0.0),
    ("unseen_interp", "interp_6_10_10", 6.0, 10.0, 10.0, 0.0),
    # 5. Unseen Extreme Extrapolation (+-8, +-9, +-12 dB & Deep -1 dB)
    ("unseen_extrap", "extrap_10_1_1", 10.0, 1.0, 1.0, 0.0),
    ("unseen_extrap", "extrap_13_1_1", 13.0, 1.0, 1.0, 0.0),
    ("unseen_extrap", "extrap_13_4_4", 13.0, 4.0, 4.0, 0.0),
    ("unseen_extrap", "extrap_9_m1_m1", 9.0, -1.0, -1.0, 0.0),
    ("unseen_extrap", "extrap_1_10_10", 1.0, 10.0, 10.0, 0.0),
    ("unseen_extrap", "extrap_1_13_13", 1.0, 13.0, 13.0, 0.0),
    # 6. Uncertain-CSI Protocol Mode (tau_unc = 0.50: TX knows CQI has +-6..12 dB uncertainty)
    ("uncertain_csi", "unc_13_1_1", 13.0, 1.0, 1.0, 0.50),
    ("uncertain_csi", "unc_10_1_1", 10.0, 1.0, 1.0, 0.50),
    ("uncertain_csi", "unc_13_4_4", 13.0, 4.0, 4.0, 0.50),
    ("uncertain_csi", "unc_1_10_10", 1.0, 10.0, 10.0, 0.50),
    ("uncertain_csi", "unc_1_13_13", 1.0, 13.0, 13.0, 0.50),
    # 7. Imperfect RX CSI (c_R != Gamma)
    ("imperfect_rx", "imprx_13_1_2.5", 13.0, 1.0, 2.5, 0.0),
    ("imperfect_rx", "imprx_10_4_2.5", 10.0, 4.0, 2.5, 0.0),
    ("imperfect_rx", "imprx_13_4_5.5", 13.0, 4.0, 5.5, 0.0),
    ("imperfect_rx", "imprx_1_10_8.5", 1.0, 10.0, 8.5, 0.0),
]


OFDM_SCENARIOS: list[tuple[str, float, float, int]] = [
    # (label, avg_snr_db, rho_c, delay_blocks)
    ("ofdm_snr10_rho090_d0", 10.0, 0.90, 0),
    ("ofdm_snr10_rho090_d1", 10.0, 0.90, 1),
    ("ofdm_snr10_rho090_d2", 10.0, 0.90, 2),
    ("ofdm_snr10_rho090_d3", 10.0, 0.90, 3),
    ("ofdm_snr10_rho070_d1", 10.0, 0.70, 1),
    ("ofdm_snr10_rho070_d2", 10.0, 0.70, 2),
    ("ofdm_snr10_rho070_d3", 10.0, 0.70, 3),
    ("ofdm_snr7_rho085_d2", 7.0, 0.85, 2),
    ("ofdm_snr4_rho085_d2", 4.0, 0.85, 2),
]


def make_paired_rayleigh_iq(
    num_complex_symbols: int,
    source_indices: list[int],
    realization_seed: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    inv_sqrt2 = math.sqrt(0.5)
    for s_idx in source_indices:
        gen = torch.Generator(device="cpu")
        comb = ((int(realization_seed) * 2000003) ^ (int(s_idx) * 19937 + 0x9E3779B9)) & ((1 << 62) - 1)
        gen.manual_seed(comb)
        rows.append(torch.randn((num_complex_symbols, 2), generator=gen, dtype=torch.float32) * inv_sqrt2)
    return torch.stack(rows, dim=0).to(device=device)


def parse_model_specs(specs: list[str]) -> list[tuple[str, str, Path | None]]:
    out = []
    for sp in specs:
        parts = sp.split(":", 2)
        if len(parts) == 2:
            out.append((parts[0], parts[1], None))
        elif len(parts) == 3:
            out.append((parts[0], parts[1], Path(parts[2]) if parts[2] else None))
        else:
            raise ValueError(f"Bad model spec: {sp}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, choices=["rayleigh", "awgn", "sara"])
    parser.add_argument("--split", type=str, required=True, choices=["common500", "kodak24"])
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--models", type=str, nargs="+", required=True, help="DisplayName:method[:ckpt_path]")
    parser.add_argument("--rate-channels", type=int, default=6, help="Used when suite == sara")
    parser.add_argument("--hidden-ch", type=int, default=160)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[2001, 2002, 2003])
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env()
    seed_everything(20261121)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_mode = "SwinJSCC_w/_SAandRA" if args.suite == "sara" else "SwinJSCC_w/_SA"
    c_bot = 320 if args.suite == "sara" else 96
    rate_ch = args.rate_channels if args.suite == "sara" else None

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode=model_mode, c_bottleneck=c_bot, device=device)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)

    split_data = load_split_u8(args.split, allow_holdout=True)
    assert split_data.is_holdout, f"Split {args.split} must be holdout!"
    images_u8 = split_data.images_u8
    n_src = int(images_u8.shape[0])
    targets_all = images_u8.to(dtype=torch.float32) / 255.0

    # Instantiate all requested models
    model_specs = parse_model_specs(args.models)
    loaded_models: dict[str, TransceiverZPACv2] = {}
    ckpt_records: dict[str, dict] = {}

    for disp_name, method, ckpt_path in model_specs:
        m = TransceiverZPACv2(backbone, method=method, hidden_ch=args.hidden_ch).to(device)
        if ckpt_path is not None:
            payload = torch.load(ckpt_path, map_location="cpu")
            sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected, f"Unexpected keys in {ckpt_path}: {unexpected}"
            ckpt_records[disp_name] = {"path": str(ckpt_path), "sha256": sha256_file(ckpt_path)}
        m.eval()
        loaded_models[disp_name] = m

    # Pre-extract encoder stems once for the entire test set (exact FP32!)
    ref_model = next(iter(loaded_models.values()))
    stems_list = []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            stems_list.append(ref_model.extract_encoder_stem(targets_all[s:e].to(device)).cpu())
    stems_all = torch.cat(stems_list, dim=0)

    t0 = time.time()
    ch_mode = "rayleigh" if args.suite == "rayleigh" else "awgn"
    eq_mode = "mrc_derot" if args.suite == "rayleigh" else "unbiased_mmse"
    latent_ch = ref_model.latent_channels
    K_iq = 128 * latent_ch
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # Storage: per_source_static[model][cond_name][metric] -> np.ndarray of shape (n_src,) averaged over noise_seeds
    per_source_static: dict[str, dict[str, dict[str, np.ndarray]]] = {
        mname: {} for mname in loaded_models
    }
    static_rows: list[dict] = []

    print(f"=== Evaluating Static Mismatch Grid ({len(STATIC_CONDITIONS)} conditions x {len(args.noise_seeds)} seeds) ===", flush=True)
    with torch.inference_mode():
        for cat, cname, ct, gm, cr, tau in STATIC_CONDITIONS:
            acc = {mname: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for mname in loaded_models}
            for r_seed in args.noise_seeds:
                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps = make_paired_noise((e - s, 256, latent_ch), src_ids, r_seed, device=device)
                    h_iq = make_paired_rayleigh_iq(K_iq, src_ids, r_seed, device=device) if ch_mode == "rayleigh" else None
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)

                    for mname, model in loaded_models.items():
                        out = model.forward_channel_v2(
                            sb,
                            ct,
                            gm,
                            cr,
                            channel_mode=ch_mode,
                            tau_unc=tau,
                            c_prior_db=7.0,
                            standard_noise=eps,
                            h_complex_iq=h_iq,
                            rate_channels=rate_ch,
                            input_is_stem=True,
                        )
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in loaded_models:
                per_source_static[mname][cname] = acc[mname]
                row = {
                    "split": args.split,
                    "suite": args.suite,
                    "category": cat,
                    "condition": cname,
                    "model": mname,
                    "c_t_db": ct,
                    "gamma_db": gm,
                    "c_r_db": cr,
                    "tau_unc": tau,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=10000, seed=20261121)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                    row[f"{mk}_p90"] = round(st["p90"], 6)
                static_rows.append(row)

    # Category-level paired bootstrap comparisons
    categories = sorted({c[0] for c in STATIC_CONDITIONS})
    paired_diff_rows: list[dict] = []
    ref_candidates = [k for k in ("TR_ZPAC", "MS_ZPAC") if k in loaded_models]
    for ref_name in ref_candidates:
        for cat in categories:
            cnames = [c[1] for c in STATIC_CONDITIONS if c[0] == cat]
            ref_src = {
                mk: np.mean([per_source_static[ref_name][cn][mk] for cn in cnames], axis=0)
                for mk in metric_names
            }
            for other_name in loaded_models:
                if other_name == ref_name:
                    continue
                oth_src = {
                    mk: np.mean([per_source_static[other_name][cn][mk] for cn in cnames], axis=0)
                    for mk in metric_names
                }
                drow = {
                    "split": args.split,
                    "suite": args.suite,
                    "category": cat,
                    "comparison": f"{ref_name}_minus_{other_name}",
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(ref_src[mk], oth_src[mk], num_resamples=10000, seed=20261121)
                    drow[f"{mk}_diff_mean"] = round(st["mean"], 6)
                    drow[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    drow[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                paired_diff_rows.append(drow)

    # Evaluate 32-RB 5G/OFDM Delayed Block Fading Scenarios
    print(f"=== Evaluating 32-RB OFDM Fading Scenarios ({len(OFDM_SCENARIOS)} scenarios x {len(args.noise_seeds)} seeds) ===", flush=True)
    fading_rows: list[dict] = []
    fading_paired_rows: list[dict] = []

    with torch.inference_mode():
        for sc_name, avg_snr, rho_c, delay_d in OFDM_SCENARIOS:
            f_acc = {mname: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for mname in loaded_models}
            # Also pool raw frame PSNR across all (source, seed) for true frame-level Outage p10!
            frame_psnr_pool = {mname: [] for mname in loaded_models}
            abs_mis_list = []

            for r_seed in args.noise_seeds:
                chain_seed = int((r_seed * 104729 + round(avg_snr * 10) * 97 + round(rho_c * 100) * 31 + delay_d * 7) & 0x7FFFFFFF)
                steps_all = simulate_ofdm_ar1_chain(
                    n_src,
                    avg_snr_db=avg_snr,
                    rho_c=rho_c,
                    delay_blocks=delay_d,
                    seed=chain_seed,
                    num_rbs=32,
                    rho_f=0.85,
                    pilot_symbols=4,
                    warmup=32,
                )
                abs_mis_list.extend([abs(st.c_t_raw_db - st.c_r_db) for st in steps_all])

                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps = make_paired_noise((e - s, 256, latent_ch), src_ids, r_seed, device=device)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)
                    sub_steps = steps_all[s:e]

                    for mname, model in loaded_models.items():
                        # Adjust AvgSNR fallback dynamically to scenario's true long-term average SNR!
                        orig_fb = model.avg_snr_fallback_db
                        model.avg_snr_fallback_db = avg_snr
                        out = model.forward_channel_v2(
                            sb,
                            0.0,
                            0.0,
                            0.0,
                            channel_mode="ofdm_fading",
                            ofdm_steps=sub_steps,
                            standard_noise=eps,
                            equalizer_mode=eq_mode,
                            rate_channels=rate_ch,
                            input_is_stem=True,
                        )
                        model.avg_snr_fallback_db = orig_fb
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        frame_psnr_pool[mname].extend(mb["psnr_db"])
                        for mk in metric_names:
                            f_acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

            mean_abs_mis = float(np.mean(abs_mis_list))
            for mname in loaded_models:
                frow = {
                    "split": args.split,
                    "suite": args.suite,
                    "scenario": sc_name,
                    "avg_snr_db": avg_snr,
                    "rho_c": rho_c,
                    "delay_blocks": delay_d,
                    "mean_abs_mismatch_db": round(mean_abs_mis, 4),
                    "model": mname,
                    "frame_p10_psnr_db": round(float(np.quantile(frame_psnr_pool[mname], 0.10)), 6),
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(f_acc[mname][mk], num_resamples=10000, seed=20261121)
                    frow[f"{mk}_mean"] = round(st["mean"], 6)
                    frow[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    frow[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    frow[f"{mk}_p10"] = round(st["p10"], 6)
                fading_rows.append(frow)

            for ref_name in ref_candidates:
                for other_name in loaded_models:
                    if other_name == ref_name:
                        continue
                    pd = {
                        "split": args.split,
                        "suite": args.suite,
                        "scenario": sc_name,
                        "avg_snr_db": avg_snr,
                        "rho_c": rho_c,
                        "delay_blocks": delay_d,
                        "comparison": f"{ref_name}_minus_{other_name}",
                        "frame_p10_diff_db": round(
                            float(np.quantile(frame_psnr_pool[ref_name], 0.10) - np.quantile(frame_psnr_pool[other_name], 0.10)),
                            6,
                        ),
                    }
                    for mk in metric_names:
                        st = paired_bootstrap_ci(f_acc[ref_name][mk], f_acc[other_name][mk], num_resamples=10000, seed=20261121)
                        pd[f"{mk}_diff_mean"] = round(st["mean"], 6)
                        pd[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        pd[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    fading_paired_rows.append(pd)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(args.output_dir / f"static_results_{args.suite}_{args.split}.csv", static_rows)
    write_csv(args.output_dir / f"static_paired_diffs_{args.suite}_{args.split}.csv", paired_diff_rows)
    write_csv(args.output_dir / f"ofdm_fading_results_{args.suite}_{args.split}.csv", fading_rows)
    write_csv(args.output_dir / f"ofdm_fading_paired_diffs_{args.suite}_{args.split}.csv", fading_paired_rows)

    save_json(
        args.output_dir / f"eval_v2_summary_{args.suite}_{args.split}.json",
        {
            "schema": "DIR2_V2_EVALUATION",
            "suite": args.suite,
            "split": args.split,
            "is_holdout": True,
            "source_count": n_src,
            "noise_seeds": args.noise_seeds,
            "elapsed_seconds": time.time() - t0,
            "gpu": gpu_info,
            "backbone_ckpt": {"path": str(args.backbone_ckpt), "sha256": sha256_file(args.backbone_ckpt)},
            "loaded_checkpoints": ckpt_records,
        },
    )
    verify_no_links([args.output_dir])
    print(f"COMPLETED V2 EVALUATION: {args.suite} on {args.split} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
