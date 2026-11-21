#!/usr/bin/env python3
"""V3 Full Held-Out Evaluation: Completely CSI-Free / Blind AWGN & Pilot-Free 32-RB Complex OFDM vs Pilot/Oracle Ceilings."""
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
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.blind_v3 import BlindCSIFreeTransceiver
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.phy_v2 import OFDMFadingStep, sample_correlated_rbs
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci


# Track 1: Unknown-SNR AWGN (including unseen OOD -2 dB and 15 dB!)
BLIND_AWGN_SNRS: list[float] = [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0]

# Track 2: Completely Pilot-Free & Feedback-Free 32-RB Complex OFDM Fading (including OOD rho_f=0.55!)
BLIND_OFDM_CONDITIONS: list[tuple[str, float, float]] = [
    # (name, true_snr_db, rho_f)
    ("ofdm_snr1_rhof085", 1.0, 0.85),
    ("ofdm_snr4_rhof085", 4.0, 0.85),
    ("ofdm_snr7_rhof085", 7.0, 0.85),
    ("ofdm_snr10_rhof085", 10.0, 0.85),
    ("ofdm_snr13_rhof085", 13.0, 0.85),
    # Unseen severe frequency-selective multipath (OOD rho_f = 0.55, coherence BW ~ 2 RBs!)
    ("ofdm_ood_snr7_rhof055", 7.0, 0.55),
    ("ofdm_ood_snr10_rhof055", 10.0, 0.55),
]


def make_paired_iq_noise(num_complex_symbols: int, source_indices: list[int], realization_seed: int, device: torch.device) -> torch.Tensor:
    rows = []
    for s_idx in source_indices:
        gen = torch.Generator(device="cpu")
        comb = ((int(realization_seed) * 1000003) ^ (int(s_idx) * 9176 + 0x5DEECE66D)) & ((1 << 62) - 1)
        gen.manual_seed(comb)
        rows.append(torch.randn((num_complex_symbols, 2), generator=gen, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device)


def make_paired_h_rb(num_rbs: int, rho_f: float, source_indices: list[int], realization_seed: int, device: torch.device) -> torch.Tensor:
    rows = []
    keep = math.sqrt(max(0.0, 1.0 - rho_f * rho_f))
    inv_sqrt2 = math.sqrt(0.5)
    for s_idx in source_indices:
        gen = torch.Generator(device="cpu")
        comb = ((int(realization_seed) * 3000017) ^ (int(s_idx) * 29989 + round(rho_f * 1000) * 131)) & ((1 << 62) - 1)
        gen.manual_seed(comb)
        w = torch.randn((num_rbs, 2), generator=gen, dtype=torch.float32) * inv_sqrt2
        hr = [w[0, 0]]
        hi = [w[0, 1]]
        for m in range(1, num_rbs):
            hr.append(rho_f * hr[-1] + keep * w[m, 0])
            hi.append(rho_f * hi[-1] + keep * w[m, 1])
        rows.append(torch.stack([torch.stack(hr), torch.stack(hi)], dim=-1))
    return torch.stack(rows, dim=0).to(device=device)


def build_pilot_aided_steps_from_h_rb(
    h_rb_batch: torch.Tensor,  # (B, M, 2)
    true_snr_db: float,
    pilot_seed: int,
    pilot_symbols: int = 4,
) -> list[OFDMFadingStep]:
    """Construct exact 4-pilot-per-RB LMMSE steps on the EXACT SAME physical h_rb realization for head-to-head comparison!"""
    B, M, _ = h_rb_batch.shape
    sigma_n_sq = float(10.0 ** (-true_snr_db / 10.0))
    np_f = float(pilot_symbols)
    sigma_e_sq = float((sigma_n_sq / np_f) / (1.0 + sigma_n_sq / np_f))
    lmmse_scale = float(1.0 / (1.0 + sigma_n_sq / np_f))
    eff_var = sigma_n_sq + sigma_e_sq

    rng = np.random.default_rng(pilot_seed)
    h_np = h_rb_batch.detach().cpu().numpy()
    steps = []
    for b in range(B):
        hr = h_np[b, :, 0].astype(np.float32)
        hi = h_np[b, :, 1].astype(np.float32)
        vp = rng.standard_normal((M, 2)).astype(np.float32) * math.sqrt(sigma_n_sq / (2.0 * np_f))
        hhat_r = (lmmse_scale * (hr + vp[:, 0])).astype(np.float32)
        hhat_i = (lmmse_scale * (hi + vp[:, 1])).astype(np.float32)
        hhat_sq = hhat_r * hhat_r + hhat_i * hhat_i
        alpha_bar = float(np.mean(hhat_sq / np.maximum(hhat_sq + eff_var, 1e-8)))
        sinr_wb = max(1e-4, alpha_bar) / max(1e-4, 1.0 - alpha_bar)
        cr_db = float(np.clip(10.0 * math.log10(sinr_wb), -5.0, 24.0))
        steps.append(
            OFDMFadingStep(
                c_t_raw_db=7.0,   # Open-loop / feedback-free TX fixed at neutral 7 dB!
                c_r_db=cr_db,
                gamma_avg_db=true_snr_db,
                tau_unc=0.5,
                hr_rb=hr,
                hi_rb=hi,
                hhat_r_rb=hhat_r,
                hhat_i_rb=hhat_i,
                sigma_n_sq=sigma_n_sq,
                sigma_e_sq=sigma_e_sq,
            )
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["common500", "kodak24"])
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--v3-ckpt-root", type=Path, required=True)
    parser.add_argument("--v2-tr-ckpt", type=Path, default=None)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[2001, 2002, 2003])
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_blind_csifree_v3/cache"))
    seed_everything(20261121)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)

    split_data = load_split_u8(args.split, allow_holdout=True)
    images_u8 = split_data.images_u8
    n_src = int(images_u8.shape[0])
    targets_all = images_u8.to(dtype=torch.float32) / 255.0

    # Load all V3 Blind/CSI-Free models
    blind_specs = [
        ("Frozen_Blind", "frozen_blind", None),
        ("Blind_BlackBox", "blind_blackbox", args.v3_ckpt_root / "blind_blackbox/best_blind_v3.pt"),
        ("Blind_RX_ZPAC", "blind_rx_zpac", args.v3_ckpt_root / "blind_rx_zpac/best_blind_v3.pt"),
        ("SPH_ZPAC_NoHierTX", "sph_zpac_nohiertx", args.v3_ckpt_root / "sph_zpac_nohiertx/best_blind_v3.pt"),
        ("SPH_ZPAC", "sph_zpac", args.v3_ckpt_root / "sph_zpac/best_blind_v3.pt"),
    ]
    blind_models: dict[str, BlindCSIFreeTransceiver] = {}
    ckpt_records: dict[str, dict] = {}
    for dname, mth, cp in blind_specs:
        m = BlindCSIFreeTransceiver(backbone, method=mth, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)
        if cp is not None:
            payload = torch.load(cp, map_location="cpu")
            sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected, f"Unexpected keys in {cp}: {unexpected}"
            ckpt_records[dname] = {"path": str(cp), "sha256": sha256_file(cp)}
        m.eval()
        blind_models[dname] = m

    # Also load Reference Pilot/CSI-Aided Models for head-to-head comparison!
    pilot_frozen = TransceiverZPACv2(backbone, method="frozen", hidden_ch=160).to(device).eval()
    oracle_awgn = TransceiverZPACv2(backbone, method="frozen", hidden_ch=160).to(device).eval()
    avgsnr_genie_pwr = TransceiverZPACv2(backbone, method="avg_snr", hidden_ch=160, avg_snr_fallback_db=7.0).to(device).eval()

    pilot_trzpac = None
    if args.v2_tr_ckpt is not None and args.v2_tr_ckpt.exists():
        pilot_trzpac = TransceiverZPACv2(backbone, method="tr_zpac", hidden_ch=160).to(device)
        pl = torch.load(args.v2_tr_ckpt, map_location="cpu")
        pilot_trzpac.load_state_dict(pl["state_dict"], strict=False)
        pilot_trzpac.eval()

    # Pre-extract encoder stems once
    stems_list = []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            stems_list.append(pilot_frozen.extract_encoder_stem(targets_all[s:e].to(device)).cpu())
    stems_all = torch.cat(stems_list, dim=0)

    t0 = time.time()
    K = 128 * 96
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # === TRACK 1: UNKNOWN-SNR AWGN ===
    awgn_models_order = [*blind_models.keys(), "AvgSNR_GeniePwr", "Oracle_MatchedCSI"]
    awgn_rows: list[dict] = []
    awgn_paired_rows: list[dict] = []

    print(f"=== Evaluating V3 Blind Unknown-SNR AWGN ({len(BLIND_AWGN_SNRS)} SNRs x {len(args.noise_seeds)} seeds) ===", flush=True)
    with torch.inference_mode():
        for snr_db in BLIND_AWGN_SNRS:
            acc = {mname: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for mname in awgn_models_order}
            snr_mae_acc = {mname: [] for mname in blind_models}

            for r_seed in args.noise_seeds:
                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps_iq = make_paired_iq_noise(K, src_ids, r_seed, device=device)
                    eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(e - s, 256, 96)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)

                    for mname, mod in blind_models.items():
                        out = mod.forward_blind(sb, snr_db, channel_type="awgn", standard_noise_iq=eps_iq, input_is_stem=True)
                        snr_mae_acc[mname].extend((out["cr_hat_db"] - snr_db).abs().cpu().tolist())
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # AvgSNR with genie pwr side-channel (c_T=7, c_R=7)
                    out_avg = avgsnr_genie_pwr.forward_channel_v2(sb, 7.0, snr_db, 7.0, channel_mode="awgn", standard_noise=eps_pack, input_is_stem=True)
                    mb_avg = evaluator.evaluate_batch(xb, out_avg["recon"])
                    for mk in metric_names:
                        acc["AvgSNR_GeniePwr"][mk][s:e] += np.asarray(mb_avg[mk], dtype=np.float64) / len(args.noise_seeds)

                    # Genie Matched-CSI Oracle (clamped to [1, 13] where pretrained encoder lives, or exact snr_db)
                    ct_ora = max(1.0, min(13.0, snr_db))
                    out_ora = oracle_awgn.forward_channel_v2(sb, ct_ora, snr_db, ct_ora, channel_mode="awgn", standard_noise=eps_pack, input_is_stem=True)
                    mb_ora = evaluator.evaluate_batch(xb, out_ora["recon"])
                    for mk in metric_names:
                        acc["Oracle_MatchedCSI"][mk][s:e] += np.asarray(mb_ora[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in awgn_models_order:
                row = {
                    "split": args.split,
                    "true_snr_db": snr_db,
                    "is_ood_snr": snr_db in (-2.0, 15.0),
                    "model": mname,
                    "blind_snr_mae_db": round(float(np.mean(snr_mae_acc[mname])), 4) if mname in snr_mae_acc else 0.0,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=10000, seed=20261121)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                awgn_rows.append(row)

            for ref_name in ("SPH_ZPAC", "Blind_RX_ZPAC"):
                for oth in awgn_models_order:
                    if oth == ref_name:
                        continue
                    drow = {"split": args.split, "true_snr_db": snr_db, "comparison": f"{ref_name}_minus_{oth}"}
                    for mk in metric_names:
                        st = paired_bootstrap_ci(acc[ref_name][mk], acc[oth][mk], num_resamples=10000, seed=20261121)
                        drow[f"{mk}_diff_mean"] = round(st["mean"], 6)
                        drow[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        drow[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    awgn_paired_rows.append(drow)

    # === TRACK 2: COMPLETELY PILOT-FREE & FEEDBACK-FREE 32-RB COMPLEX OFDM FADING ===
    ofdm_models_order = [*blind_models.keys(), "Pilot4_Frozen_MMSE"]
    if pilot_trzpac is not None:
        ofdm_models_order.append("Pilot4_TR_ZPAC_Ceiling")

    ofdm_rows: list[dict] = []
    ofdm_paired_rows: list[dict] = []

    print(f"=== Evaluating V3 Pilot-Free 32-RB Complex OFDM ({len(BLIND_OFDM_CONDITIONS)} conditions x {len(args.noise_seeds)} seeds) ===", flush=True)
    with torch.inference_mode():
        for cname, snr_db, rho_f in BLIND_OFDM_CONDITIONS:
            acc = {mname: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for mname in ofdm_models_order}
            frame_pool = {mname: [] for mname in ofdm_models_order}

            for r_seed in args.noise_seeds:
                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps_iq = make_paired_iq_noise(K, src_ids, r_seed, device=device)
                    eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(e - s, 256, 96)
                    h_rb = make_paired_h_rb(32, rho_f, src_ids, r_seed, device=device)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)

                    # 1. All Pilot-Free / CSI-Free Blind Models (0 pilots, 0 feedback, 0 side-info!)
                    for mname, mod in blind_models.items():
                        out = mod.forward_blind(
                            sb,
                            snr_db,
                            channel_type="ofdm_complex",
                            rho_f=rho_f,
                            h_rb_override=h_rb,
                            standard_noise_iq=eps_iq,
                            input_is_stem=True,
                        )
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        frame_pool[mname].extend(mb["psnr_db"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # 2. Pilot-Aided Reference Systems (using 4 dedicated pilots/RB = 128 extra pilot symbols + pwr side-info!)
                    p_steps = build_pilot_aided_steps_from_h_rb(h_rb, snr_db, pilot_seed=r_seed * 997 + s, pilot_symbols=4)
                    out_pf = pilot_frozen.forward_channel_v2(sb, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=p_steps, standard_noise=eps_pack, input_is_stem=True)
                    mb_pf = evaluator.evaluate_batch(xb, out_pf["recon"])
                    frame_pool["Pilot4_Frozen_MMSE"].extend(mb_pf["psnr_db"])
                    for mk in metric_names:
                        acc["Pilot4_Frozen_MMSE"][mk][s:e] += np.asarray(mb_pf[mk], dtype=np.float64) / len(args.noise_seeds)

                    if pilot_trzpac is not None:
                        out_pt = pilot_trzpac.forward_channel_v2(sb, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=p_steps, standard_noise=eps_pack, input_is_stem=True)
                        mb_pt = evaluator.evaluate_batch(xb, out_pt["recon"])
                        frame_pool["Pilot4_TR_ZPAC_Ceiling"].extend(mb_pt["psnr_db"])
                        for mk in metric_names:
                            acc["Pilot4_TR_ZPAC_Ceiling"][mk][s:e] += np.asarray(mb_pt[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in ofdm_models_order:
                frow = {
                    "split": args.split,
                    "condition": cname,
                    "true_snr_db": snr_db,
                    "rho_f": rho_f,
                    "is_ood_rhof": abs(rho_f - 0.55) < 1e-3,
                    "model": mname,
                    "dedicated_pilots": 128 if mname.startswith("Pilot4_") else 0,
                    "frame_p10_psnr_db": round(float(np.quantile(frame_pool[mname], 0.10)), 6),
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=10000, seed=20261121)
                    frow[f"{mk}_mean"] = round(st["mean"], 6)
                    frow[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    frow[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                ofdm_rows.append(frow)

            for oth in ofdm_models_order:
                if oth == "SPH_ZPAC":
                    continue
                pd = {
                    "split": args.split,
                    "condition": cname,
                    "true_snr_db": snr_db,
                    "rho_f": rho_f,
                    "comparison": f"SPH_ZPAC_minus_{oth}",
                    "frame_p10_diff_db": round(float(np.quantile(frame_pool["SPH_ZPAC"], 0.10) - np.quantile(frame_pool[oth], 0.10)), 6),
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc["SPH_ZPAC"][mk], acc[oth][mk], num_resamples=10000, seed=20261121)
                    pd[f"{mk}_diff_mean"] = round(st["mean"], 6)
                    pd[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    pd[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                ofdm_paired_rows.append(pd)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(args.output_dir / f"blind_awgn_results_{args.split}.csv", awgn_rows)
    write_csv(args.output_dir / f"blind_awgn_paired_diffs_{args.split}.csv", awgn_paired_rows)
    write_csv(args.output_dir / f"blind_ofdm_results_{args.split}.csv", ofdm_rows)
    write_csv(args.output_dir / f"blind_ofdm_paired_diffs_{args.split}.csv", ofdm_paired_rows)

    save_json(
        args.output_dir / f"eval_v3_blind_summary_{args.split}.json",
        {
            "schema": "DIR2_V3_BLIND_EVALUATION",
            "split": args.split,
            "is_holdout": True,
            "source_count": n_src,
            "noise_seeds": args.noise_seeds,
            "elapsed_seconds": time.time() - t0,
            "gpu": gpu_info,
            "loaded_checkpoints": ckpt_records,
        },
    )
    verify_no_links([args.output_dir])
    print(f"COMPLETED V3 BLIND EVALUATION ON {args.split} IN {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
