#!/usr/bin/env python3
"""Direction 4 Full Sealed Held-Out Evaluation Suite (ImageNet common500 & Kodak24):
  - Track A (Dir-4A): 1-Bit & 2-Bit Low-Resolution Time-Domain ADCs + Rapp Nonlinear HPA + Wiener Phase Noise
  - Track B (Dir-4B): Grant-Free 2x2 Near-Far & Overloaded 3x2 Non-Orthogonal Uplink with 2-Stage Semantic SIC
"""
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
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR3_SRC, DIR2_SRC):
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

from dir4_rn.rf_1bit_blind import (
    N_OFDM_SC,
    N_OFDM_SYM,
    RFImpairedBlindTransceiver,
    forward_pilot4_rf_bussgang_reference,
    physical_rf_impaired_ofdm_channel,
)
from dir4_rn.noma_semsic_blind import (
    GrantFreeNOMATransceiver,
    forward_pilot4_or_genie_linear_noma_reference,
    physical_grantfree_uplink_mac,
    sample_multiuser_uplink_channels,
)


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
    parser.add_argument("--dir4-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[3001, 3002, 3003])
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir4_rf1bit_noma_v1/cache") / f"eval_{args.split}",
        num_threads=6,
    )
    seed_everything(20261124)
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

    def load_rf(mth: str, tag: str | None) -> RFImpairedBlindTransceiver:
        m = RFImpairedBlindTransceiver(backbone, method=mth, num_rbs=32, hidden_ch=160).to(device)
        if tag is not None:
            cp = args.dir4_ckpt_root / tag / "best_dir4.pt"
            pl = torch.load(cp, map_location="cpu")
            missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
            assert not unexpected
            ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    def load_noma(mth: str, tag: str) -> GrantFreeNOMATransceiver:
        m = GrantFreeNOMATransceiver(backbone, method=mth, n_rx=2, num_rbs=32, hidden_ch=160).to(device)
        cp = args.dir4_ckpt_root / tag / "best_dir4.pt"
        pl = torch.load(cp, map_location="cpu")
        missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
        assert not unexpected
        ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    rf_models = {
        1: {
            "Frozen_RF_Blind": load_rf("frozen_rf", None),
            "BlackBox_RF_Blind": load_rf("blackbox_rf", "blackbox_rf_1bit"),
            "RF_Bussgang_JSCC": load_rf("rf_bussgang", "rf_bussgang_1bit"),
        },
        2: {
            "Frozen_RF_Blind": load_rf("frozen_rf", None),
            "BlackBox_RF_Blind": load_rf("blackbox_rf", "blackbox_rf_2bit"),
            "RF_Bussgang_JSCC": load_rf("rf_bussgang", "rf_bussgang_2bit"),
        },
    }

    noma_models = {
        "BlackBox_NOMA_Blind": load_noma("blackbox_noma", "blackbox_noma"),
        "SemSIC_Stage1_Only": load_noma("semsic_stage1_only", "semsic_stage1_only"),
        "SemSIC_NOMA_JSCC": load_noma("semsic_noma", "semsic_noma"),
    }

    v4_usph = UniversalBlindTransceiverV4(backbone, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    v4_usph.load_state_dict(torch.load(args.v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    v4_usph.eval()
    ckpt_records["v4_u_sph"] = {"path": str(args.v4_usph_ckpt), "sha256": sha256_file(args.v4_usph_ckpt)}

    stems_l = []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            stems_l.append(v4_usph.extract_encoder_stem(targets_all[s:e].to(device)).cpu())
    stems_all = torch.cat(stems_l, dim=0)

    t0 = time.time()
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # =========================================================================
    # TRACK A (DIR-4A): 1-BIT & 2-BIT ADC + RAPP HPA + WIENER PHASE NOISE
    # =========================================================================
    rf_specs = [
        # (cond_name, adc_bits, snr_db, pa_ibo_db, pn_step_deg, is_ood)
        ("adc1bit_snr1dB", 1, 1.0, 1.0, 3.5, False),
        ("adc1bit_snr4dB", 1, 4.0, 1.0, 3.5, False),
        ("adc1bit_snr7dB", 1, 7.0, 1.0, 3.5, False),
        ("adc1bit_snr10dB", 1, 10.0, 1.0, 3.5, False),
        ("adc1bit_snr14dB", 1, 14.0, 1.0, 3.5, False),
        ("adc1bit_ood_ibo0_pn55_10dB", 1, 10.0, 0.0, 5.5, True),
        ("adc2bit_snr4dB", 2, 4.0, 1.0, 3.5, False),
        ("adc2bit_snr10dB", 2, 10.0, 1.0, 3.5, False),
        ("adc2bit_snr14dB", 2, 14.0, 1.0, 3.5, False),
        ("adc2bit_ood_ibo0_pn55_10dB", 2, 10.0, 0.0, 5.5, True),
    ]
    rf_mnames = ["Frozen_RF_Blind", "BlackBox_RF_Blind", "V4_USPH_ZeroShot", "Pilot4_Bussgang_LMMSE", "RF_Bussgang_JSCC"]
    rf_rows: list[dict] = []
    rf_paired_rows: list[dict] = []
    papr_summary: dict[str, float] = {}

    print(f"=== [Dir-4A] Evaluating 1-Bit & 2-Bit Hardware-Impaired OFDM ({args.split}) ===", flush=True)
    with torch.inference_mode():
        for cname, b_adc, snr_db, ibo_db, pn_deg, is_ood in rf_specs:
            acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in rf_mnames}
            papr_v4_l, papr_ours_l = [], []

            for r_seed in args.noise_seeds:
                gen = torch.Generator(device="cpu").manual_seed(r_seed * 1301 + b_adc * 53 + round(pn_deg * 10))
                eps_td_full = torch.randn((n_src, N_OFDM_SYM, N_OFDM_SC, 2), generator=gen, dtype=torch.float32)
                hr_f, hi_f = sample_correlated_rbs(n_src, 32, 0.85, device="cpu", generator=gen)
                h_rb_full = torch.stack([hr_f, hi_f], dim=-1)
                dphi = torch.randn((n_src, N_OFDM_SYM), generator=gen, dtype=torch.float32) * (pn_deg * math.pi / 180.0)
                dphi[:, 0] = 0.0
                phi_full = torch.cumsum(dphi, dim=1)

                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)
                    eps_b = eps_td_full[s:e].to(device)
                    h_b = h_rb_full[s:e].to(device)
                    phi_b = phi_full[s:e].to(device)

                    for mname, mod in rf_models[b_adc].items():
                        out = mod.forward_rf_blind(
                            sb,
                            snr_db,
                            adc_bits=b_adc,
                            pa_ibo_db=ibo_db,
                            pn_step_deg=pn_deg,
                            h_rb_override=h_b,
                            phase_walk_override=phi_b,
                            standard_noise_td_iq=eps_b,
                            input_is_stem=True,
                        )
                        if mname == "RF_Bussgang_JSCC":
                            papr_ours_l.extend(out["papr_db"].cpu().tolist())
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # V4 U-SPH Zero-Shot on exact same RF hardware-impaired channel
                    stx_v4 = v4_usph.encode_blind_tx(sb).reshape(e - s, 32, 384, 2)
                    rrx_v4, dv4 = physical_rf_impaired_ofdm_channel(
                        stx_v4,
                        snr_db,
                        adc_bits=b_adc,
                        pa_ibo_db=ibo_db,
                        pn_step_deg=pn_deg,
                        h_rb_override=h_b,
                        phase_walk_override=phi_b,
                        standard_noise_td_iq=eps_b,
                    )
                    papr_v4_l.extend(dv4["papr_db"].cpu().tolist())
                    rec_v4, _ = v4_usph.decode_blind_rx(rrx_v4.reshape(e - s, 12288, 2))
                    mb_v4 = evaluator.evaluate_batch(xb, rec_v4)

                    # 128-Pilot Closed-Loop Bussgang-LMMSE Reference
                    rec_p4 = forward_pilot4_rf_bussgang_reference(
                        backbone,
                        xb,
                        snr_db,
                        adc_bits=b_adc,
                        pa_ibo_db=ibo_db,
                        pn_step_deg=pn_deg,
                        h_rb_override=h_b,
                        phase_walk_override=phi_b,
                        standard_noise_td_iq=eps_b,
                        pilot_seed=r_seed * 991 + s,
                    )
                    mb_p4 = evaluator.evaluate_batch(xb, rec_p4)

                    for mk in metric_names:
                        acc["V4_USPH_ZeroShot"][mk][s:e] += np.asarray(mb_v4[mk], dtype=np.float64) / len(args.noise_seeds)
                        acc["Pilot4_Bussgang_LMMSE"][mk][s:e] += np.asarray(mb_p4[mk], dtype=np.float64) / len(args.noise_seeds)

            papr_summary[f"{cname}_v4_papr_db"] = round(float(np.mean(papr_v4_l)), 4)
            papr_summary[f"{cname}_ours_papr_db"] = round(float(np.mean(papr_ours_l)), 4)

            for mname in rf_mnames:
                row = {
                    "split": args.split,
                    "condition": cname,
                    "adc_bits": b_adc,
                    "true_snr_db": snr_db,
                    "pa_ibo_db": ibo_db,
                    "pn_step_deg": pn_deg,
                    "is_ood": is_ood,
                    "model": mname,
                    "dedicated_pilots": 128 if mname == "Pilot4_Bussgang_LMMSE" else 0,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261124)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                rf_rows.append(row)

            for oth in rf_mnames:
                if oth == "RF_Bussgang_JSCC":
                    continue
                st_d = paired_bootstrap_ci(acc["RF_Bussgang_JSCC"]["psnr_db"], acc[oth]["psnr_db"], num_resamples=5000, seed=20261124)
                st_di = paired_bootstrap_ci(acc["RF_Bussgang_JSCC"]["dists"], acc[oth]["dists"], num_resamples=5000, seed=20261124)
                rf_paired_rows.append({
                    "split": args.split,
                    "condition": cname,
                    "adc_bits": b_adc,
                    "comparison": f"RF_Bussgang_JSCC_minus_{oth}",
                    "psnr_diff_mean": round(st_d["mean"], 6),
                    "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                    "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                    "dists_diff_mean": round(st_di["mean"], 6),
                })

    # =========================================================================
    # TRACK B (DIR-4B): GRANT-FREE 2x2 & OVERLOADED 3x2 NOMA UPLINK
    # =========================================================================
    print(f"=== [Dir-4B] Evaluating Grant-Free 2x2 & Overloaded 3x2 NOMA Uplink ({args.split}) ===", flush=True)
    noma_specs = [
        # (cond_name, user_snrs_list, is_overloaded, is_ood)
        ("noma2x2_bal_4dB", [4.0, 4.0], False, False),
        ("noma2x2_bal_10dB", [10.0, 10.0], False, False),
        ("noma2x2_nearfar_12_3dB", [12.0, 3.0], False, False),
        ("noma2x2_ood_nearfar_14_2dB", [14.0, 2.0], False, True),
        ("over3x2_nearfar_13_8_4dB", [13.0, 8.0, 4.0], True, False),
        ("over3x2_mid_12_9_6dB", [12.0, 9.0, 6.0], True, False),
        ("over3x2_ood_nearfar_14_8_2dB", [14.0, 8.0, 2.0], True, True),
        ("over3x2_ood_equal_10_10_10dB", [10.0, 10.0, 10.0], True, True),
    ]
    noma_mnames = [
        "Uncoord_V4_Collision",
        "BlackBox_NOMA_Blind",
        "Pilot4_Linear_MMSE",
        "Genie_CSI_Linear_MMSE",
        "SemSIC_Stage1_Only",
        "SemSIC_NOMA_JSCC",
    ]
    noma_rows: list[dict] = []
    noma_paired_rows: list[dict] = []

    with torch.inference_mode():
        for cname, snr_list, is_over, is_ood in noma_specs:
            U = len(snr_list)
            # Store per-user metric arrays [n_src] AND mean-across-users array [n_src]
            acc_u = {
                m: {u_i: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for u_i in range(U)}
                for m in noma_mnames
            }
            shifts = [0, n_src // 3, (2 * n_src) // 3][:U] if n_src >= 24 else [0, 8, 16][:U]

            for r_seed in args.noise_seeds:
                gen = torch.Generator(device="cpu").manual_seed(r_seed * 2089 + U * 97 + round(snr_list[0] * 10))
                hr_full, hi_full = sample_multiuser_uplink_channels(n_src, U, 2, 32, 0.85, generator=gen)
                eps_full = torch.randn((n_src, 2, 32, 384, 2), generator=gen, dtype=torch.float32)

                bs_noma = min(args.batch_size, 25)
                for s in range(0, n_src, bs_noma):
                    e = min(n_src, s + bs_noma)
                    cb = e - s
                    snr_t = torch.tensor([snr_list], dtype=torch.float32, device=device).expand(cb, U)
                    hr_b = hr_full[s:e].to(device)
                    hi_b = hi_full[s:e].to(device)
                    eps_b = eps_full[s:e].to(device)

                    x_users, s_users = [], []
                    for u_i in range(U):
                        idx_u = (np.arange(s, e) + shifts[u_i]) % n_src
                        x_users.append(targets_all[idx_u].to(device))
                        s_users.append(stems_all[idx_u].to(device))

                    # 1. Trained Dir-4B Models (BlackBox, Stage1-Only Ablation, Full SemSIC)
                    for mname, mod in noma_models.items():
                        out_m = mod.forward_noma_uplink(
                            s_users,
                            snr_t,
                            h_mu_override=(hr_b, hi_b),
                            standard_noise_iq=eps_b,
                            input_is_stem=True,
                        )
                        for u_i in range(U):
                            mb = evaluator.evaluate_batch(x_users[u_i], out_m["recons"][u_i])
                            for mk in metric_names:
                                acc_u[mname][u_i][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # 2. Uncoordinated V4 U-SPH Collision (all users transmit V4 waveforms without orthogonal signatures)
                    v4_tx_u = torch.stack([v4_usph.encode_blind_tx(s_users[u_i]).reshape(cb, 32, 384, 2) for u_i in range(U)], dim=1)
                    y_col, _ = physical_grantfree_uplink_mac(v4_tx_u, snr_t, n_rx=2, h_mu_override=(hr_b, hi_b), standard_noise_iq=eps_b)
                    for u_i in range(U):
                        ant_pick = u_i % 2
                        rec_col, _ = v4_usph.decode_blind_rx(y_col[:, ant_pick].reshape(cb, 12288, 2))
                        mb_c = evaluator.evaluate_batch(x_users[u_i], rec_col)
                        for mk in metric_names:
                            acc_u["Uncoord_V4_Collision"][u_i][mk][s:e] += np.asarray(mb_c[mk], dtype=np.float64) / len(args.noise_seeds)

                    # 3. Pilot4 Linear MMSE & Genie-CSI Oracle Linear MMSE
                    recons_p4 = forward_pilot4_or_genie_linear_noma_reference(
                        backbone, x_users, snr_t, n_rx=2, h_mu_override=(hr_b, hi_b), standard_noise_iq=eps_b, mode="pilot4_linear_mmse", pilot_seed=r_seed * 773 + s
                    )
                    recons_ge = forward_pilot4_or_genie_linear_noma_reference(
                        backbone, x_users, snr_t, n_rx=2, h_mu_override=(hr_b, hi_b), standard_noise_iq=eps_b, mode="genie_linear_mmse"
                    )
                    for u_i in range(U):
                        mb_p4 = evaluator.evaluate_batch(x_users[u_i], recons_p4[u_i])
                        mb_ge = evaluator.evaluate_batch(x_users[u_i], recons_ge[u_i])
                        for mk in metric_names:
                            acc_u["Pilot4_Linear_MMSE"][u_i][mk][s:e] += np.asarray(mb_p4[mk], dtype=np.float64) / len(args.noise_seeds)
                            acc_u["Genie_CSI_Linear_MMSE"][u_i][mk][s:e] += np.asarray(mb_ge[mk], dtype=np.float64) / len(args.noise_seeds)

            # Record both ALL_USERS_MEAN and individual user roles (e.g. Weakest Far User!)
            for mname in noma_mnames:
                mean_arrs = {mk: sum(acc_u[mname][u_i][mk] for u_i in range(U)) / float(U) for mk in metric_names}
                for role_lbl, src_dict in [("ALL_USERS_MEAN", mean_arrs)] + [(f"User{u_i+1}_{int(snr_list[u_i])}dB", acc_u[mname][u_i]) for u_i in range(U)]:
                    row = {
                        "split": args.split,
                        "condition": cname,
                        "n_users": U,
                        "n_rx": 2,
                        "is_overloaded": is_over,
                        "is_ood": is_ood,
                        "user_role": role_lbl,
                        "model": mname,
                        "dedicated_pilots": 4 * U * 32 if mname == "Pilot4_Linear_MMSE" else 0,
                    }
                    for mk in metric_names:
                        st = paired_bootstrap_ci(src_dict[mk], num_resamples=5000, seed=20261124)
                        row[f"{mk}_mean"] = round(st["mean"], 6)
                        row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                        row[f"{mk}_p10"] = round(st["p10"], 6)
                    noma_rows.append(row)

            # Paired Bootstrap CI on ALL_USERS_MEAN and Weakest User (u_i = U - 1)
            for role_tag, u_pick in [("ALL_USERS_MEAN", None), (f"Weakest_User{U}_{int(snr_list[-1])}dB", U - 1)]:
                ours_psnr = (
                    sum(acc_u["SemSIC_NOMA_JSCC"][u_i]["psnr_db"] for u_i in range(U)) / float(U)
                    if u_pick is None
                    else acc_u["SemSIC_NOMA_JSCC"][u_pick]["psnr_db"]
                )
                ours_dists = (
                    sum(acc_u["SemSIC_NOMA_JSCC"][u_i]["dists"] for u_i in range(U)) / float(U)
                    if u_pick is None
                    else acc_u["SemSIC_NOMA_JSCC"][u_pick]["dists"]
                )
                for oth in noma_mnames:
                    if oth == "SemSIC_NOMA_JSCC":
                        continue
                    oth_psnr = (
                        sum(acc_u[oth][u_i]["psnr_db"] for u_i in range(U)) / float(U)
                        if u_pick is None
                        else acc_u[oth][u_pick]["psnr_db"]
                    )
                    oth_dists = (
                        sum(acc_u[oth][u_i]["dists"] for u_i in range(U)) / float(U)
                        if u_pick is None
                        else acc_u[oth][u_pick]["dists"]
                    )
                    st_d = paired_bootstrap_ci(ours_psnr, oth_psnr, num_resamples=5000, seed=20261124)
                    st_di = paired_bootstrap_ci(ours_dists, oth_dists, num_resamples=5000, seed=20261124)
                    noma_paired_rows.append({
                        "split": args.split,
                        "condition": cname,
                        "user_role": role_tag,
                        "comparison": f"SemSIC_NOMA_JSCC_minus_{oth}",
                        "psnr_diff_mean": round(st_d["mean"], 6),
                        "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                        "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                        "dists_diff_mean": round(st_di["mean"], 6),
                    })

    write_csv(args.output_dir / f"dir4a_rf1bit_results_{args.split}.csv", rf_rows)
    write_csv(args.output_dir / f"dir4a_rf1bit_paired_{args.split}.csv", rf_paired_rows)
    write_csv(args.output_dir / f"dir4b_noma_results_{args.split}.csv", noma_rows)
    write_csv(args.output_dir / f"dir4b_noma_paired_{args.split}.csv", noma_paired_rows)

    save_json(
        args.output_dir / f"eval_dir4_summary_{args.split}.json",
        {
            "schema": "DIR4_RF1BIT_NOMA_EVALUATION",
            "split": args.split,
            "is_holdout": True,
            "source_count": n_src,
            "noise_seeds": args.noise_seeds,
            "papr_diagnostics_db": papr_summary,
            "elapsed_seconds": time.time() - t0,
            "gpu": gpu_info,
            "loaded_checkpoints": ckpt_records,
        },
    )
    verify_no_links([args.output_dir])
    print(f"COMPLETED DIR-4 EVALUATION ON {args.split} IN {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
