#!/usr/bin/env python3
"""V4 Comprehensive Held-Out Evaluation Suite:
  Suite 1: Upgraded Universal Analog Blind Self-Probing (AWGN & Official Rayleigh Backbones)
  Suite 2: Simultaneous 1-to-4 Heterogeneous Wireless Semantic Broadcast (100% vs 400% Airtime)
  Suite 3: Hardware-Standard 16-QAM & 64-QAM Zero-Pilot Digital JSCC vs. IEEE TCOM Nov. 2025
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
from dir2_zpac.models import encode_with_ct, load_swinjscc_backbone
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.blind_v3 import BlindCSIFreeTransceiver
from dir2_zpac.universal_blind_v4 import (
    UniversalBlindTransceiverV4,
    get_pam_levels,
    physical_blind_channel_v4,
    ste_pam_quantize,
)
from dir2_zpac.phy_v2 import expand_rb_to_symbols
from dir2_zpac.metrics import OfflineVisualEvaluator
from dir2_zpac.stats import paired_bootstrap_ci

from evaluate_blind_v3 import (
    BLIND_AWGN_SNRS,
    BLIND_OFDM_CONDITIONS,
    build_pilot_aided_steps_from_h_rb,
    make_paired_h_rb,
    make_paired_iq_noise,
)


# 4-User Heterogeneous Wireless Broadcast Profile: (user_id, channel_type, snr_db, rho_f)
BROADCAST_USERS: list[tuple[str, str, float, float]] = [
    ("U1_Edge_OOD_1dB", "ofdm_complex", 1.0, 0.55),
    ("U2_Urban_5dB", "ofdm_complex", 5.0, 0.85),
    ("U3_Suburban_10dB", "ofdm_complex", 10.0, 0.85),
    ("U4_NearLoS_14dB", "awgn", 14.0, 1.00),
]


@torch.inference_mode()
def forward_pilot4_qam_reference(
    backbone: torch.nn.Module,
    xb: torch.Tensor,
    qam_order: int,
    snr_db: float,
    channel_type: str,
    eps_iq: torch.Tensor,
    h_rb: torch.Tensor | None,
    pilot_seed: int,
    pilot_symbols: int = 4,
) -> torch.Tensor:
    """Conventional 4-Pilot-per-RB (128 extra QAM pilots + genie pwr side-channel!) Digital QAM LMMSE Reference."""
    B = xb.shape[0]
    device, dtype = xb.device, xb.dtype
    z_raw, _ = encode_with_ct(backbone, xb, 7.0)
    L, C = z_raw.shape[1], z_raw.shape[2]
    K = (L * C) // 2
    M = 32
    N_b = K // M

    pwr = (2.0 * z_raw.square().mean(dim=(1, 2))).clamp_min(1e-8)
    p_img = pwr.sqrt()
    g0 = 0.78 if qam_order == 16 else 0.70
    flat = (z_raw / p_img[:, None, None] * g0).reshape(B, 2 * K)
    levels = get_pam_levels(qam_order, device=device, dtype=dtype)
    q_r = ste_pam_quantize(flat[:, :K], levels)
    q_i = ste_pam_quantize(flat[:, K:], levels)
    s_tx = torch.stack([q_r, q_i], dim=-1)

    y_rx, diag = physical_blind_channel_v4(
        s_tx, snr_db, channel_type=channel_type, num_rbs=M, h_rb_override=h_rb, standard_noise_iq=eps_iq
    )
    sigma_n_sq = diag["true_sigma_n_sq"]  # (B,)

    if channel_type == "awgn":
        hhat_r = torch.ones((B, M), device=device, dtype=dtype)
        hhat_i = torch.zeros((B, M), device=device, dtype=dtype)
        sigma_e_sq = torch.zeros((B,), device=device, dtype=dtype)
    else:
        assert h_rb is not None
        np_f = float(pilot_symbols)
        gen = torch.Generator(device="cpu").manual_seed(pilot_seed)
        vp = torch.randn((B, M, 2), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        # Pilots use unit-energy outer/inner average QAM symbols
        sig_p = (sigma_n_sq / (2.0 * np_f)).sqrt()[:, None, None]
        lmmse_w = (1.0 / (1.0 + (sigma_n_sq / np_f)))[:, None]
        hhat_r = lmmse_w * (h_rb[:, :, 0] + vp[:, :, 0] * sig_p.squeeze(-1))
        hhat_i = lmmse_w * (h_rb[:, :, 1] + vp[:, :, 1] * sig_p.squeeze(-1))
        sigma_e_sq = (sigma_n_sq / np_f) / (1.0 + sigma_n_sq / np_f)

    y_r = y_rx[:, :, 0].reshape(B, M, N_b)
    y_i = y_rx[:, :, 1].reshape(B, M, N_b)
    h_sq = (hhat_r.square() + hhat_i.square()).clamp_min(1e-8)
    h_abs = h_sq.sqrt()
    denom = (h_sq + (sigma_n_sq + sigma_e_sq)[:, None]).clamp_min(1e-6)
    alpha_rb = h_sq / denom
    alpha_bar = alpha_rb.mean(dim=1, keepdim=True).clamp_min(0.05)

    cos_rb = (hhat_r / h_abs)[:, :, None]
    sin_rb = (hhat_i / h_abs)[:, :, None]
    yd_r = cos_rb * y_r + sin_rb * y_i
    yd_i = -sin_rb * y_r + cos_rb * y_i
    w_ub = ((h_abs / denom) / alpha_bar)[:, :, None] / g0

    u_in = torch.cat([(yd_r * w_ub).reshape(B, K), (yd_i * w_ub).reshape(B, K)], dim=1).reshape(B, L, C) * p_img[:, None, None]
    dec = backbone.decoder
    x = dec.head_list(u_in)
    snr_c0 = torch.full((B, 1), 7.0, device=device, dtype=dtype)
    temp = None
    for i in range(dec.layer_num):
        temp = dec.sm_list[i](x.detach()) if i == 0 else dec.sm_list[i](temp)
        temp = temp * dec.bm_list[i](snr_c0).unsqueeze(1).expand(-1, L, -1)
    x = x * dec.sigmoid(dec.sm_list[-1](temp))
    for layer in dec.layers:
        x = layer(x)
    return x.reshape(B, dec.H, dec.W, -1).permute(0, 3, 1, 2).contiguous()


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
    parser.add_argument("--awgn-bb", type=Path, required=True)
    parser.add_argument("--rayleigh-bb", type=Path, required=True)
    parser.add_argument("--v4-ckpt-root", type=Path, required=True)
    parser.add_argument("--v3-ckpt-root", type=Path, required=True)
    parser.add_argument("--v2-ckpt-root", type=Path, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[2001, 2002, 2003])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_universal_blind_v4/cache") / f"eval_{args.split}",
        num_threads=6,
    )
    seed_everything(20261122)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bb_awgn = load_swinjscc_backbone(args.awgn_bb, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    bb_ray = load_swinjscc_backbone(args.rayleigh_bb, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)

    split_data = load_split_u8(args.split, allow_holdout=True)
    images_u8 = split_data.images_u8
    n_src = int(images_u8.shape[0])
    targets_all = images_u8.to(dtype=torch.float32) / 255.0

    ckpt_records: dict[str, dict] = {}

    def load_v4(bb: torch.nn.Module, mth: str, tag: str | None) -> UniversalBlindTransceiverV4:
        m = UniversalBlindTransceiverV4(bb, method=mth, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)
        if tag is not None:
            cp = args.v4_ckpt_root / tag / "best_v4.pt"
            pl = torch.load(cp, map_location="cpu")
            missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
            assert not unexpected, f"Unexpected keys in {cp}: {unexpected}"
            ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    def load_v3(bb: torch.nn.Module, mth: str, sub: str | None) -> BlindCSIFreeTransceiver:
        m = BlindCSIFreeTransceiver(bb, method=mth, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)
        if sub is not None:
            cp = args.v3_ckpt_root / sub / "best_blind_v3.pt"
            pl = torch.load(cp, map_location="cpu")
            m.load_state_dict(pl["state_dict"], strict=False)
            ckpt_records[f"v3_{sub}"] = {"path": str(cp), "sha256": sha256_file(cp)}
        return m.eval()

    # Instantiate V4, V3, and V2 reference models
    v4_usph_awgn = load_v4(bb_awgn, "u_sph", "u_sph_awgn")
    v4_usph_ray = load_v4(bb_ray, "u_sph", "u_sph_rayleigh")
    v3_sph_awgn = load_v3(bb_awgn, "sph_zpac", "sph_zpac")
    v3_bbx_awgn = load_v3(bb_awgn, "blind_blackbox", "blind_blackbox")
    frz_blind_ray = load_v3(bb_ray, "frozen_blind", None)

    v2_pilot_awgn = TransceiverZPACv2(bb_awgn, method="frozen", hidden_ch=160).to(device).eval()
    v2_avg_awgn = TransceiverZPACv2(bb_awgn, method="avg_snr", hidden_ch=160, avg_snr_fallback_db=7.0).to(device).eval()
    v2_tr_awgn = TransceiverZPACv2(bb_awgn, method="tr_zpac", hidden_ch=160).to(device)
    v2_tr_awgn.load_state_dict(torch.load(args.v2_ckpt_root / "awgn/tr_zpac/best_calibrator_v2.pt", map_location="cpu")["state_dict"], strict=False)
    v2_tr_awgn.eval()

    v2_pilot_ray = TransceiverZPACv2(bb_ray, method="frozen", hidden_ch=160).to(device).eval()
    v2_tr_ray = TransceiverZPACv2(bb_ray, method="tr_zpac", hidden_ch=160).to(device)
    v2_tr_ray.load_state_dict(torch.load(args.v2_ckpt_root / "rayleigh/tr_zpac/best_calibrator_v2.pt", map_location="cpu")["state_dict"], strict=False)
    v2_tr_ray.eval()

    # Digital QAM models (16-QAM & 64-QAM)
    v4_dd_qam16 = load_v4(bb_awgn, "dd_qam16_sph", "dd_qam16_sph")
    v4_tc_qam16 = load_v4(bb_awgn, "tcom25_qam16", "tcom25_qam16")
    v4_frz_qam16 = load_v4(bb_awgn, "frozen_qam16", None)

    v4_dd_qam64 = load_v4(bb_awgn, "dd_qam64_sph", "dd_qam64_sph")
    v4_tc_qam64 = load_v4(bb_awgn, "tcom25_qam64", "tcom25_qam64")
    v4_frz_qam64 = load_v4(bb_awgn, "frozen_qam64", None)

    # Pre-extract encoder stems for AWGN & Rayleigh backbones
    stems_awgn_l, stems_ray_l = [], []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            xb = targets_all[s:e].to(device)
            stems_awgn_l.append(v4_usph_awgn.extract_encoder_stem(xb).cpu())
            stems_ray_l.append(v4_usph_ray.extract_encoder_stem(xb).cpu())
    stems_awgn = torch.cat(stems_awgn_l, dim=0)
    stems_ray = torch.cat(stems_ray_l, dim=0)

    t0 = time.time()
    K = 128 * 96
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # =========================================================================
    # SUITE 1A: V4 U-SPH-ZPAC vs V3 SPH-ZPAC & Oracles on Unknown-SNR AWGN
    # =========================================================================
    print(f"=== [V4 Suite 1A] Universal Blind AWGN ({args.split}) ===", flush=True)
    s1_awgn_models = ["V3_SPH_ZPAC", "V4_U_SPH_AWGN", "AvgSNR_GeniePwr", "Oracle_MatchedCSI", "Frozen_Blind_Rayleigh", "V4_U_SPH_Rayleigh"]
    s1_awgn_rows: list[dict] = []
    with torch.inference_mode():
        for snr_db in BLIND_AWGN_SNRS:
            acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in s1_awgn_models}
            mae_acc = {m: [] for m in ["V3_SPH_ZPAC", "V4_U_SPH_AWGN", "V4_U_SPH_Rayleigh"]}
            for r_seed in args.noise_seeds:
                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps_iq = make_paired_iq_noise(K, src_ids, r_seed, device=device)
                    eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(e - s, 256, 96)
                    xb = targets_all[s:e].to(device)
                    sa = stems_awgn[s:e].to(device)
                    sr = stems_ray[s:e].to(device)

                    for mname, mod, st_in in [
                        ("V3_SPH_ZPAC", v3_sph_awgn, sa),
                        ("V4_U_SPH_AWGN", v4_usph_awgn, sa),
                        ("Frozen_Blind_Rayleigh", frz_blind_ray, sr),
                        ("V4_U_SPH_Rayleigh", v4_usph_ray, sr),
                    ]:
                        out = mod.forward_blind(st_in, snr_db, channel_type="awgn", standard_noise_iq=eps_iq, input_is_stem=True)
                        if mname in mae_acc:
                            mae_acc[mname].extend((out["cr_hat_db"] - snr_db).abs().cpu().tolist())
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    out_avg = v2_avg_awgn.forward_channel_v2(sa, 7.0, snr_db, 7.0, channel_mode="awgn", standard_noise=eps_pack, input_is_stem=True)
                    mb_avg = evaluator.evaluate_batch(xb, out_avg["recon"])
                    ct_ora = max(1.0, min(13.0, snr_db))
                    out_ora = v2_pilot_awgn.forward_channel_v2(sa, ct_ora, snr_db, ct_ora, channel_mode="awgn", standard_noise=eps_pack, input_is_stem=True)
                    mb_ora = evaluator.evaluate_batch(xb, out_ora["recon"])
                    for mk in metric_names:
                        acc["AvgSNR_GeniePwr"][mk][s:e] += np.asarray(mb_avg[mk], dtype=np.float64) / len(args.noise_seeds)
                        acc["Oracle_MatchedCSI"][mk][s:e] += np.asarray(mb_ora[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in s1_awgn_models:
                row = {
                    "split": args.split,
                    "true_snr_db": snr_db,
                    "is_ood_snr": snr_db in (-2.0, 15.0),
                    "model": mname,
                    "blind_snr_mae_db": round(float(np.mean(mae_acc[mname])), 4) if mname in mae_acc else 0.0,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261122)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                s1_awgn_rows.append(row)

    # =========================================================================
    # SUITE 1B: V4 U-SPH-ZPAC on Both AWGN & Official Rayleigh Backbones (OFDM)
    # =========================================================================
    print(f"=== [V4 Suite 1B] Universal Pilot-Free 32-RB OFDM ({args.split}) ===", flush=True)
    s1_ofdm_models = [
        "V3_SPH_ZPAC",
        "V4_U_SPH_AWGN",
        "Pilot4_TR_ZPAC_AWGN",
        "Frozen_Blind_Rayleigh",
        "Pilot4_Frozen_Rayleigh",
        "Pilot4_TR_ZPAC_Rayleigh",
        "V4_U_SPH_Rayleigh",
    ]
    s1_ofdm_rows: list[dict] = []
    with torch.inference_mode():
        for cname, snr_db, rho_f in BLIND_OFDM_CONDITIONS:
            acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in s1_ofdm_models}
            fpool = {m: [] for m in s1_ofdm_models}
            for r_seed in args.noise_seeds:
                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    src_ids = list(range(s, e))
                    eps_iq = make_paired_iq_noise(K, src_ids, r_seed, device=device)
                    eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(e - s, 256, 96)
                    h_rb = make_paired_h_rb(32, rho_f, src_ids, r_seed, device=device)
                    xb = targets_all[s:e].to(device)
                    sa = stems_awgn[s:e].to(device)
                    sr = stems_ray[s:e].to(device)

                    for mname, mod, st_in in [
                        ("V3_SPH_ZPAC", v3_sph_awgn, sa),
                        ("V4_U_SPH_AWGN", v4_usph_awgn, sa),
                        ("Frozen_Blind_Rayleigh", frz_blind_ray, sr),
                        ("V4_U_SPH_Rayleigh", v4_usph_ray, sr),
                    ]:
                        out = mod.forward_blind(st_in, snr_db, channel_type="ofdm_complex", rho_f=rho_f, h_rb_override=h_rb, standard_noise_iq=eps_iq, input_is_stem=True)
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        fpool[mname].extend(mb["psnr_db"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    p_steps = build_pilot_aided_steps_from_h_rb(h_rb, snr_db, pilot_seed=r_seed * 997 + s, pilot_symbols=4)
                    for mname, mod, st_in in [
                        ("Pilot4_TR_ZPAC_AWGN", v2_tr_awgn, sa),
                        ("Pilot4_Frozen_Rayleigh", v2_pilot_ray, sr),
                        ("Pilot4_TR_ZPAC_Rayleigh", v2_tr_ray, sr),
                    ]:
                        out = mod.forward_channel_v2(st_in, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=p_steps, standard_noise=eps_pack, input_is_stem=True)
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        fpool[mname].extend(mb["psnr_db"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

            for mname in s1_ofdm_models:
                row = {
                    "split": args.split,
                    "condition": cname,
                    "true_snr_db": snr_db,
                    "rho_f": rho_f,
                    "model": mname,
                    "dedicated_pilots": 128 if mname.startswith("Pilot4_") else 0,
                    "frame_p10_psnr_db": round(float(np.quantile(fpool[mname], 0.10)), 6),
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261122)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                s1_ofdm_rows.append(row)

    # =========================================================================
    # SUITE 2: Simultaneous 1-to-4 Heterogeneous Wireless Broadcast
    # =========================================================================
    print(f"=== [V4 Suite 2] 1-to-4 Simultaneous Heterogeneous Wireless Broadcast ({args.split}) ===", flush=True)
    bcast_schemes = [
        "Frozen_Blind_Bcast",
        "Blind_BlackBox_TCOM25",
        "Pilot4_EdgeTuned_cT1",
        "Pilot4_NearTuned_cT13",
        "V3_SPH_ZPAC_Bcast",
        "V4_DD_QAM64_SPH_Bcast",
        "V4_U_SPH_Bcast",
        "Unicast_4x_ClosedLoop_Ceiling",
    ]
    bcast_acc = {sc: {u[0]: np.zeros(n_src, dtype=np.float64) for u in BROADCAST_USERS} for sc in bcast_schemes}
    bcast_dists = {sc: {u[0]: np.zeros(n_src, dtype=np.float64) for u in BROADCAST_USERS} for sc in bcast_schemes}

    frz_blind_awgn = load_v3(bb_awgn, "frozen_blind", None)
    with torch.inference_mode():
        for r_seed in args.noise_seeds:
            for s in range(0, n_src, args.batch_size):
                e = min(n_src, s + args.batch_size)
                src_ids = list(range(s, e))
                xb = targets_all[s:e].to(device)
                sa = stems_awgn[s:e].to(device)

                # Single broadcast waveform transmitted ONCE over the air (100% airtime!)
                pkt_frz = frz_blind_awgn.encode_blind_tx(sa)
                pkt_bbx = v3_bbx_awgn.encode_blind_tx(sa)
                pkt_v3 = v3_sph_awgn.encode_blind_tx(sa)
                pkt_q64 = v4_dd_qam64.encode_blind_tx(sa)
                pkt_v4 = v4_usph_awgn.encode_blind_tx(sa)

                for u_idx, (uid, ch_t, u_snr, u_rhof) in enumerate(BROADCAST_USERS):
                    u_seed = r_seed * 17 + u_idx * 101
                    eps_iq = make_paired_iq_noise(K, src_ids, u_seed, device=device)
                    eps_pack = torch.cat([eps_iq[:, :, 0], eps_iq[:, :, 1]], dim=1).reshape(e - s, 256, 96)
                    h_rb = make_paired_h_rb(32, u_rhof, src_ids, u_seed, device=device) if ch_t == "ofdm_complex" else None

                    # Receive single broadcast waveform at User u_idx
                    for sc_name, rx_mod, pkt in [
                        ("Frozen_Blind_Bcast", frz_blind_awgn, pkt_frz),
                        ("Blind_BlackBox_TCOM25", v3_bbx_awgn, pkt_bbx),
                        ("V3_SPH_ZPAC_Bcast", v3_sph_awgn, pkt_v3),
                        ("V4_DD_QAM64_SPH_Bcast", v4_dd_qam64, pkt_q64),
                        ("V4_U_SPH_Bcast", v4_usph_awgn, pkt_v4),
                    ]:
                        y_u, _ = physical_blind_channel_v4(pkt, u_snr, channel_type=ch_t, num_rbs=32, rho_f=u_rhof, h_rb_override=h_rb, standard_noise_iq=eps_iq)
                        rec_u, _ = rx_mod.decode_blind_rx(y_u)
                        mb = evaluator.evaluate_batch(xb, rec_u)
                        bcast_acc[sc_name][uid][s:e] += np.asarray(mb["psnr_db"], dtype=np.float64) / len(args.noise_seeds)
                        bcast_dists[sc_name][uid][s:e] += np.asarray(mb["dists"], dtype=np.float64) / len(args.noise_seeds)

                    # Pilot-aided single-cT broadcast baselines & 4x unicast ceiling
                    if ch_t == "ofdm_complex":
                        assert h_rb is not None
                        p_steps_base = build_pilot_aided_steps_from_h_rb(h_rb, u_snr, pilot_seed=u_seed * 997 + s, pilot_symbols=4)
                        for sc_name, ct_fix, mod_p in [
                            ("Pilot4_EdgeTuned_cT1", 1.0, v2_pilot_awgn),
                            ("Pilot4_NearTuned_cT13", 13.0, v2_pilot_awgn),
                            ("Unicast_4x_ClosedLoop_Ceiling", max(1.0, min(13.0, u_snr)), v2_tr_awgn),
                        ]:
                            steps_c = [
                                type(st)(
                                    c_t_raw_db=ct_fix,
                                    c_r_db=st.c_r_db,
                                    gamma_avg_db=st.gamma_avg_db,
                                    tau_unc=0.0 if sc_name.startswith("Unicast_") else 0.5,
                                    hr_rb=st.hr_rb,
                                    hi_rb=st.hi_rb,
                                    hhat_r_rb=st.hhat_r_rb,
                                    hhat_i_rb=st.hhat_i_rb,
                                    sigma_n_sq=st.sigma_n_sq,
                                    sigma_e_sq=st.sigma_e_sq,
                                )
                                for st in p_steps_base
                            ]
                            out_p = mod_p.forward_channel_v2(sa, 0, 0, 0, channel_mode="ofdm_fading", ofdm_steps=steps_c, standard_noise=eps_pack, input_is_stem=True)
                            mb = evaluator.evaluate_batch(xb, out_p["recon"])
                            bcast_acc[sc_name][uid][s:e] += np.asarray(mb["psnr_db"], dtype=np.float64) / len(args.noise_seeds)
                            bcast_dists[sc_name][uid][s:e] += np.asarray(mb["dists"], dtype=np.float64) / len(args.noise_seeds)
                    else:
                        for sc_name, ct_fix, mod_p in [
                            ("Pilot4_EdgeTuned_cT1", 1.0, v2_pilot_awgn),
                            ("Pilot4_NearTuned_cT13", 13.0, v2_pilot_awgn),
                            ("Unicast_4x_ClosedLoop_Ceiling", 13.0, v2_tr_awgn),
                        ]:
                            out_p = mod_p.forward_channel_v2(sa, ct_fix, u_snr, min(13.0, u_snr), channel_mode="awgn", standard_noise=eps_pack, input_is_stem=True)
                            mb = evaluator.evaluate_batch(xb, out_p["recon"])
                            bcast_acc[sc_name][uid][s:e] += np.asarray(mb["psnr_db"], dtype=np.float64) / len(args.noise_seeds)
                            bcast_dists[sc_name][uid][s:e] += np.asarray(mb["dists"], dtype=np.float64) / len(args.noise_seeds)

    bcast_rows: list[dict] = []
    for sc_name in bcast_schemes:
        u_mats = np.stack([bcast_acc[sc_name][u[0]] for u in BROADCAST_USERS], axis=1)  # (N_src, 4)
        d_mats = np.stack([bcast_dists[sc_name][u[0]] for u in BROADCAST_USERS], axis=1)
        row = {
            "split": args.split,
            "scheme": sc_name,
            "airtime_pct": 400 if sc_name.startswith("Unicast_") else 100,
            "dedicated_pilots": 512 if sc_name.startswith("Unicast_") else (128 if sc_name.startswith("Pilot4_") else 0),
        }
        for ui, (uid, _, _, _) in enumerate(BROADCAST_USERS):
            row[f"{uid}_psnr_db"] = round(float(np.mean(u_mats[:, ui])), 4)
            row[f"{uid}_dists"] = round(float(np.mean(d_mats[:, ui])), 4)
        mean_4u = u_mats.mean(axis=1)
        min_4u = u_mats.min(axis=1)
        st_mean = paired_bootstrap_ci(mean_4u, num_resamples=5000, seed=20261122)
        st_min = paired_bootstrap_ci(min_4u, num_resamples=5000, seed=20261122)
        row["mean_4user_psnr_db"] = round(st_mean["mean"], 4)
        row["mean_4user_ci95_lo"] = round(st_mean["ci95_lo"], 4)
        row["mean_4user_ci95_hi"] = round(st_mean["ci95_hi"], 4)
        row["worst_user_psnr_db"] = round(st_min["mean"], 4)
        row["mean_4user_dists"] = round(float(d_mats.mean()), 4)
        bcast_rows.append(row)

    # =========================================================================
    # SUITE 3: Hardware-Standard 16-QAM & 64-QAM Zero-Pilot Digital JSCC
    # =========================================================================
    print(f"=== [V4 Suite 3] Hardware-Standard 16-QAM & 64-QAM Digital Blind JSCC ({args.split}) ===", flush=True)
    qam_conds = [
        ("awgn", 1.0, 1.00, "awgn_1dB"),
        ("awgn", 7.0, 1.00, "awgn_7dB"),
        ("awgn", 13.0, 1.00, "awgn_13dB"),
        ("awgn", 15.0, 1.00, "awgn_ood_15dB"),
        ("ofdm_complex", 1.0, 0.85, "ofdm_1dB_rhof085"),
        ("ofdm_complex", 4.0, 0.85, "ofdm_4dB_rhof085"),
        ("ofdm_complex", 7.0, 0.85, "ofdm_7dB_rhof085"),
        ("ofdm_complex", 10.0, 0.85, "ofdm_10dB_rhof085"),
        ("ofdm_complex", 13.0, 0.85, "ofdm_13dB_rhof085"),
        ("ofdm_complex", 10.0, 0.55, "ofdm_ood_10dB_rhof055"),
    ]
    qam_rows: list[dict] = []
    qam_paired_rows: list[dict] = []

    with torch.inference_mode():
        for ch_t, snr_db, rho_f, clabel in qam_conds:
            for q_ord, m_frz, m_tc, m_dd in [
                (16, v4_frz_qam16, v4_tc_qam16, v4_dd_qam16),
                (64, v4_frz_qam64, v4_tc_qam64, v4_dd_qam64),
            ]:
                mnames = [f"Frozen_QAM{q_ord}_Blind", f"TCOM25_QAM{q_ord}_Blind", f"Pilot4_QAM{q_ord}_MMSE", f"DD_QAM{q_ord}_SPH"]
                acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in mnames}
                for r_seed in args.noise_seeds:
                    for s in range(0, n_src, args.batch_size):
                        e = min(n_src, s + args.batch_size)
                        src_ids = list(range(s, e))
                        eps_iq = make_paired_iq_noise(K, src_ids, r_seed, device=device)
                        h_rb = make_paired_h_rb(32, rho_f, src_ids, r_seed, device=device) if ch_t == "ofdm_complex" else None
                        xb = targets_all[s:e].to(device)
                        sa = stems_awgn[s:e].to(device)

                        for mname, mod in [
                            (f"Frozen_QAM{q_ord}_Blind", m_frz),
                            (f"TCOM25_QAM{q_ord}_Blind", m_tc),
                            (f"DD_QAM{q_ord}_SPH", m_dd),
                        ]:
                            out = mod.forward_blind(sa, snr_db, channel_type=ch_t, rho_f=rho_f, h_rb_override=h_rb, standard_noise_iq=eps_iq, input_is_stem=True)
                            # Verify strict constellation compliance on first batch!
                            if s == 0 and r_seed == args.noise_seeds[0]:
                                lv = get_pam_levels(q_ord, device=device, dtype=xb.dtype)
                                err_q = (out["s_tx_iq"].unsqueeze(-1) - lv).abs().min(dim=-1).values.max().item()
                                assert err_q < 1e-5, f"{mname} violated {q_ord}-QAM lattice: {err_q}"
                            mb = evaluator.evaluate_batch(xb, out["recon"])
                            for mk in metric_names:
                                acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                        rec_p4 = forward_pilot4_qam_reference(bb_awgn, xb, q_ord, snr_db, ch_t, eps_iq, h_rb, pilot_seed=r_seed * 997 + s)
                        mb_p4 = evaluator.evaluate_batch(xb, rec_p4)
                        for mk in metric_names:
                            acc[f"Pilot4_QAM{q_ord}_MMSE"][mk][s:e] += np.asarray(mb_p4[mk], dtype=np.float64) / len(args.noise_seeds)

                for mname in mnames:
                    row = {
                        "split": args.split,
                        "qam_order": q_ord,
                        "condition": clabel,
                        "channel_type": ch_t,
                        "true_snr_db": snr_db,
                        "rho_f": rho_f,
                        "model": mname,
                        "dedicated_pilots": 128 if "Pilot4_" in mname else 0,
                    }
                    for mk in metric_names:
                        st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261122)
                        row[f"{mk}_mean"] = round(st["mean"], 6)
                        row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                        row[f"{mk}_p10"] = round(st["p10"], 6)
                    qam_rows.append(row)

                dd_key = f"DD_QAM{q_ord}_SPH"
                for oth in mnames:
                    if oth == dd_key:
                        continue
                    st_d = paired_bootstrap_ci(acc[dd_key]["psnr_db"], acc[oth]["psnr_db"], num_resamples=5000, seed=20261122)
                    st_di = paired_bootstrap_ci(acc[dd_key]["dists"], acc[oth]["dists"], num_resamples=5000, seed=20261122)
                    qam_paired_rows.append(
                        {
                            "split": args.split,
                            "qam_order": q_ord,
                            "condition": clabel,
                            "comparison": f"{dd_key}_minus_{oth}",
                            "psnr_diff_mean": round(st_d["mean"], 6),
                            "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                            "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                            "dists_diff_mean": round(st_di["mean"], 6),
                        }
                    )

    write_csv(args.output_dir / f"v4_s1_universal_awgn_{args.split}.csv", s1_awgn_rows)
    write_csv(args.output_dir / f"v4_s1_universal_ofdm_{args.split}.csv", s1_ofdm_rows)
    write_csv(args.output_dir / f"v4_s2_broadcast_4user_{args.split}.csv", bcast_rows)
    write_csv(args.output_dir / f"v4_s3_digital_qam_{args.split}.csv", qam_rows)
    write_csv(args.output_dir / f"v4_s3_digital_qam_paired_{args.split}.csv", qam_paired_rows)

    save_json(
        args.output_dir / f"eval_v4_summary_{args.split}.json",
        {
            "schema": "DIR2_V4_UNIVERSAL_EVALUATION",
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
    print(f"COMPLETED V4 EVALUATION ON {args.split} IN {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
