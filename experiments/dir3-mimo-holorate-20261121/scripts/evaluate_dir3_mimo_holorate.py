#!/usr/bin/env python3
"""Direction 3 Full Held-Out Evaluation Suite (ImageNet common500 & Kodak24):
  - Track A (Dir-3A): Zero-Pilot Spatial-Multiplexed 2x2 (CBR=1/32) & 4x4 (CBR=1/64) MIMO-OFDM (i.i.d. & OOD Kronecker Correlated)
  - Track B (Dir-3B): Holographic Rateless Anytime-Stopping, Contiguous Burst Erasures & Hostile Partial-Band Jamming
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
import numpy as np
import torch

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci

from dir3_mh.mimo_blind import (
    BlindMIMOTransceiver,
    forward_pilot_or_genie_svd_mimo_reference,
    sample_kronecker_mimo_rbs,
)
from dir3_mh.holorate_blind import HoloRateBlindTransceiver, physical_hostile_ofdm_channel


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["common500", "kodak24"])
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--dir3-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[2001, 2002, 2003])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir3_mimo_holorate_v1/cache") / f"eval_{args.split}",
        num_threads=6,
    )
    seed_everything(20261123)
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

    ckpt_records: dict[str, dict] = {}

    def load_mimo(mth: str, n_ant: int, tag: str | None) -> BlindMIMOTransceiver:
        m = BlindMIMOTransceiver(backbone, method=mth, n_ant=n_ant, num_rbs=None, hidden_ch=160).to(device)
        if tag is not None:
            cp = args.dir3_ckpt_root / tag / "best_dir3.pt"
            pl = torch.load(cp, map_location="cpu")
            missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
            assert not unexpected
            ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    def load_holo(mth: str, tag: str) -> HoloRateBlindTransceiver:
        m = HoloRateBlindTransceiver(backbone, method=mth, num_rbs=32, hidden_ch=160).to(device)
        cp = args.dir3_ckpt_root / tag / "best_dir3.pt"
        pl = torch.load(cp, map_location="cpu")
        missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
        assert not unexpected
        ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    mimo_models = {
        2: {
            "Frozen_Blind_MIMO": load_mimo("frozen_blind_mimo", 2, None),
            "BlackBox_Blind_MIMO": load_mimo("blackbox_mimo", 2, "blackbox_mimo_2x2"),
            "OS_MIMO_JSCC": load_mimo("os_mimo", 2, "os_mimo_2x2"),
        },
        4: {
            "Frozen_Blind_MIMO": load_mimo("frozen_blind_mimo", 4, None),
            "BlackBox_Blind_MIMO": load_mimo("blackbox_mimo", 4, "blackbox_mimo_4x4"),
            "OS_MIMO_JSCC": load_mimo("os_mimo", 4, "os_mimo_4x4"),
        },
    }

    holo_model = load_holo("holorate_blind", "holorate_blind")
    raster_model = load_holo("raster_blind_resilient", "raster_blind_resilient")

    v4_usph = UniversalBlindTransceiverV4(backbone, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    v4_usph.load_state_dict(torch.load(args.v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    v4_usph.eval()
    ckpt_records["v4_u_sph"] = {"path": str(args.v4_usph_ckpt), "sha256": sha256_file(args.v4_usph_ckpt)}

    stems_l = []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            stems_l.append(holo_model.extract_encoder_stem(targets_all[s:e].to(device)).cpu())
    stems_all = torch.cat(stems_l, dim=0)

    t0 = time.time()
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # =========================================================================
    # TRACK A (DIR-3A): 2x2 & 4x4 SPATIAL-MULTIPLEXED MIMO-OFDM EVALUATION
    # =========================================================================
    mimo_conds = [
        ("mimo_iid_1dB", 1.0, 0.85, 0.0),
        ("mimo_iid_4dB", 4.0, 0.85, 0.0),
        ("mimo_iid_7dB", 7.0, 0.85, 0.0),
        ("mimo_iid_10dB", 10.0, 0.85, 0.0),
        ("mimo_iid_13dB", 13.0, 0.85, 0.0),
        ("mimo_ood_corr060_7dB", 7.0, 0.85, 0.60),
        ("mimo_ood_corr060_10dB", 10.0, 0.85, 0.60),
    ]
    mimo_rows: list[dict] = []
    mimo_paired_rows: list[dict] = []

    for n_ant in (2, 4):
        print(f"=== [Dir-3A] Evaluating {n_ant}x{n_ant} Spatial-Multiplexed MIMO-OFDM ({args.split}) ===", flush=True)
        m_rbs = 32 // n_ant
        t_mimo = (128 * 96) // n_ant
        all_mnames = ["Frozen_Blind_MIMO", "BlackBox_Blind_MIMO", "Pilot4_MIMO_LMMSE", "OS_MIMO_JSCC", "Genie_SVD_MIMO_Oracle"]

        with torch.inference_mode():
            for cname, snr_db, rho_f, rho_ant in mimo_conds:
                acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in all_mnames}
                for r_seed in args.noise_seeds:
                    gen = torch.Generator(device="cpu").manual_seed(r_seed * 1009 + n_ant * 37 + round(rho_ant * 100))
                    eps_full = torch.randn((n_src, n_ant, t_mimo, 2), generator=gen, dtype=torch.float32)
                    hr_full, hi_full = sample_kronecker_mimo_rbs(n_src, m_rbs, n_ant, n_ant, rho_f=rho_f, rho_ant=rho_ant, generator=gen)

                    for s in range(0, n_src, args.batch_size):
                        e = min(n_src, s + args.batch_size)
                        xb = targets_all[s:e].to(device)
                        sb = stems_all[s:e].to(device)
                        eps_b = eps_full[s:e].to(device)
                        hr_b = hr_full[s:e].to(device)
                        hi_b = hi_full[s:e].to(device)

                        for mname, mod in mimo_models[n_ant].items():
                            out = mod.forward_mimo_blind(sb, snr_db, rho_f=rho_f, rho_ant=rho_ant, h_mimo_override=(hr_b, hi_b), standard_noise_iq=eps_b, input_is_stem=True)
                            mb = evaluator.evaluate_batch(xb, out["recon"])
                            for mk in metric_names:
                                acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                        rec_p4 = forward_pilot_or_genie_svd_mimo_reference(backbone, xb, n_ant, snr_db, hr_b, hi_b, eps_b, mode="pilot4_lmmse", pilot_seed=r_seed * 997 + s)
                        mb_p4 = evaluator.evaluate_batch(xb, rec_p4)
                        rec_svd = forward_pilot_or_genie_svd_mimo_reference(backbone, xb, n_ant, snr_db, hr_b, hi_b, eps_b, mode="genie_svd")
                        mb_svd = evaluator.evaluate_batch(xb, rec_svd)
                        for mk in metric_names:
                            acc["Pilot4_MIMO_LMMSE"][mk][s:e] += np.asarray(mb_p4[mk], dtype=np.float64) / len(args.noise_seeds)
                            acc["Genie_SVD_MIMO_Oracle"][mk][s:e] += np.asarray(mb_svd[mk], dtype=np.float64) / len(args.noise_seeds)

                for mname in all_mnames:
                    row = {
                        "split": args.split,
                        "mimo_config": f"{n_ant}x{n_ant}",
                        "channel_uses": t_mimo,
                        "cbr": f"1/{16 * n_ant}",
                        "condition": cname,
                        "true_snr_db": snr_db,
                        "rho_ant": rho_ant,
                        "is_ood_corr": rho_ant > 0.5,
                        "model": mname,
                        "dedicated_pilot_slots": 4 * n_ant * m_rbs if mname == "Pilot4_MIMO_LMMSE" else 0,
                    }
                    for mk in metric_names:
                        st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261123)
                        row[f"{mk}_mean"] = round(st["mean"], 6)
                        row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                        row[f"{mk}_p10"] = round(st["p10"], 6)
                    mimo_rows.append(row)

                for oth in all_mnames:
                    if oth == "OS_MIMO_JSCC":
                        continue
                    st_d = paired_bootstrap_ci(acc["OS_MIMO_JSCC"]["psnr_db"], acc[oth]["psnr_db"], num_resamples=5000, seed=20261123)
                    st_di = paired_bootstrap_ci(acc["OS_MIMO_JSCC"]["dists"], acc[oth]["dists"], num_resamples=5000, seed=20261123)
                    mimo_paired_rows.append(
                        {
                            "split": args.split,
                            "mimo_config": f"{n_ant}x{n_ant}",
                            "condition": cname,
                            "comparison": f"OS_MIMO_JSCC_minus_{oth}",
                            "psnr_diff_mean": round(st_d["mean"], 6),
                            "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                            "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                            "dists_diff_mean": round(st_di["mean"], 6),
                        }
                    )

    # =========================================================================
    # TRACK B (DIR-3B): RATELESS, BURST-ERASURE & ANTI-JAMMING EVALUATION
    # =========================================================================
    print(f"=== [Dir-3B] Evaluating Rateless, Burst-Erasure & Partial-Band Jamming ({args.split}) ===", flush=True)
    # Construct deterministic masks for all 10 hostile/rateless conditions (SNR = 10 dB, 0-Pilot 32-RB Complex OFDM)
    def prefix_mask(k_keep: int) -> np.ndarray:
        m = np.zeros((1, 32), dtype=np.float32)
        m[:, :k_keep] = 1.0
        return m

    def burst_drop_mask(st: int, length: int) -> np.ndarray:
        m = np.ones((1, 32), dtype=np.float32)
        m[:, st : st + length] = 0.0
        return m

    hostile_specs = [
        ("rateless_8rb_25pct", "rateless", 8, prefix_mask(8), None, 0.0),
        ("rateless_16rb_50pct", "rateless", 16, prefix_mask(16), None, 0.0),
        ("rateless_24rb_75pct", "rateless", 24, prefix_mask(24), None, 0.0),
        ("rateless_32rb_100pct", "rateless", 32, prefix_mask(32), None, 0.0),
        ("burst_erasure_25pct", "erasure", 24, burst_drop_mask(12, 8), None, 0.0),
        ("burst_erasure_50pct", "erasure", 16, burst_drop_mask(8, 16), None, 0.0),
        ("random_erasure_37pct", "erasure", 20, "random_drop_12", None, 0.0),
        ("jamming_25pct_15dB", "jamming", 32, None, "jam_8", 15.0),
        ("jamming_37pct_15dB", "jamming", 32, None, "jam_12", 15.0),
        ("jamming_ood_31pct_18dB", "jamming", 32, None, "jam_10", 18.0),
    ]

    holo_models_order = ["V4_U_SPH_ZeroShot", "Raster_Blind_Resilient", "HoloRate_JSCC"]
    holo_rows: list[dict] = []
    holo_paired_rows: list[dict] = []

    with torch.inference_mode():
        for cname, cat, eff_rbs, am_spec, jm_spec, jam_inr in hostile_specs:
            acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in holo_models_order}
            for r_seed in args.noise_seeds:
                gen = torch.Generator(device="cpu").manual_seed(r_seed * 2029)
                eps_full = torch.randn((n_src, 32, 384, 2), generator=gen, dtype=torch.float32)
                jeps_full = torch.randn((n_src, 32, 384, 2), generator=gen, dtype=torch.float32)
                hr_f, hi_f = sample_correlated_rbs(n_src, 32, 0.85, device="cpu", generator=gen)
                h_rb_full = torch.stack([hr_f, hi_f], dim=-1)

                rng_m = np.random.default_rng(r_seed * 7919)
                if isinstance(am_spec, str) and am_spec == "random_drop_12":
                    am_full = np.ones((n_src, 32), dtype=np.float32)
                    for i in range(n_src):
                        am_full[i, rng_m.choice(32, size=12, replace=False)] = 0.0
                elif isinstance(am_spec, np.ndarray):
                    am_full = np.repeat(am_spec, n_src, axis=0)
                else:
                    am_full = None

                if isinstance(jm_spec, str) and jm_spec.startswith("jam_"):
                    n_j = int(jm_spec.split("_")[1])
                    jm_full = np.zeros((n_src, 32), dtype=np.float32)
                    for i in range(n_src):
                        jm_full[i, rng_m.choice(32, size=n_j, replace=False)] = 1.0
                else:
                    jm_full = None

                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)
                    am_b = torch.from_numpy(am_full[s:e]).to(device) if am_full is not None else None
                    jm_b = torch.from_numpy(jm_full[s:e]).to(device) if jm_full is not None else None
                    eps_b = eps_full[s:e].to(device)
                    jeps_b = jeps_full[s:e].to(device)
                    h_b = h_rb_full[s:e].to(device)

                    for mname, mod in [("Raster_Blind_Resilient", raster_model), ("HoloRate_JSCC", holo_model)]:
                        out = mod.forward_hostile_blind(
                            sb,
                            10.0,
                            rho_f=0.85,
                            alive_rb_mask=am_b,
                            jam_rb_mask=jm_b,
                            jam_inr_db=jam_inr,
                            h_rb_override=h_b,
                            standard_noise_iq=eps_b,
                            jam_noise_iq=jeps_b,
                            input_is_stem=True,
                        )
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # Pass V4 U-SPH through the EXACT SAME physical hostile channel!
                    stx_v4 = v4_usph.encode_blind_tx(sb).reshape(e - s, 32, 384, 2)
                    yrx_v4, _ = physical_hostile_ofdm_channel(
                        stx_v4,
                        10.0,
                        rho_f=0.85,
                        alive_rb_mask=am_b,
                        jam_rb_mask=jm_b,
                        jam_inr_db=jam_inr,
                        h_rb_override=h_b,
                        standard_noise_iq=eps_b,
                        jam_noise_iq=jeps_b,
                    )
                    rec_v4, _ = v4_usph.decode_blind_rx(yrx_v4.reshape(e - s, 12288, 2))
                    mb_v4 = evaluator.evaluate_batch(xb, rec_v4)
                    for mk in metric_names:
                        acc["V4_U_SPH_ZeroShot"][mk][s:e] += np.asarray(mb_v4[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in holo_models_order:
                row = {
                    "split": args.split,
                    "condition": cname,
                    "category": cat,
                    "effective_rbs": eff_rbs,
                    "jam_inr_db": jam_inr,
                    "model": mname,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261123)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                holo_rows.append(row)

            for oth in holo_models_order:
                if oth == "HoloRate_JSCC":
                    continue
                st_d = paired_bootstrap_ci(acc["HoloRate_JSCC"]["psnr_db"], acc[oth]["psnr_db"], num_resamples=5000, seed=20261123)
                st_di = paired_bootstrap_ci(acc["HoloRate_JSCC"]["dists"], acc[oth]["dists"], num_resamples=5000, seed=20261123)
                holo_paired_rows.append(
                    {
                        "split": args.split,
                        "condition": cname,
                        "comparison": f"HoloRate_JSCC_minus_{oth}",
                        "psnr_diff_mean": round(st_d["mean"], 6),
                        "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                        "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                        "dists_diff_mean": round(st_di["mean"], 6),
                    }
                )

    write_csv(args.output_dir / f"dir3a_mimo_results_{args.split}.csv", mimo_rows)
    write_csv(args.output_dir / f"dir3a_mimo_paired_{args.split}.csv", mimo_paired_rows)
    write_csv(args.output_dir / f"dir3b_holorate_results_{args.split}.csv", holo_rows)
    write_csv(args.output_dir / f"dir3b_holorate_paired_{args.split}.csv", holo_paired_rows)

    save_json(
        args.output_dir / f"eval_dir3_summary_{args.split}.json",
        {
            "schema": "DIR3_MIMO_HOLORATE_EVALUATION",
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
    print(f"COMPLETED DIR-3 EVALUATION ON {args.split} IN {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
