#!/usr/bin/env python3
"""Direction 5 Sealed Held-Out Evaluation Suite (ImageNet common500 & Kodak24):
  - Track A (Dir-5A): High-Mobility Doubly-Selective Delay-Doppler Channels (SNR sweep @ k_d=0.10 + Mobility sweep k_d=0.01..0.18 OOD)
  - Track B (Dir-5B): Ultra-Low-Bandwidth CBR = 1/128 (K=1,536) & CBR = 1/256 (K=768) + OOD 50% Contiguous RB Erasure
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

from dir5_og.otfs_blind import (
    N_SC,
    N_SYM,
    SympOTFSBlindTransceiver,
    forward_frozen_ofdm_doubly_selective,
    forward_pilot_4sym_interp_ofdm_doubly_selective,
    forward_v4_usph_ofdm_doubly_selective,
    sample_doubly_selective_multipath,
)
from dir5_og.genvar_ultralow_blind import (
    DEFAULT_VAR_VQVAE_CKPT,
    N_UL_RBS,
    GenHoloVARTransceiver,
    SwinUltraLowTransceiver,
    cbr_geometry,
    load_frozen_var_vqvae,
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
    parser.add_argument("--dir5-ckpt-root", type=Path, required=True)
    parser.add_argument("--v4-usph-ckpt", type=Path, required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[5001, 5002, 5003])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--skip-dir5a", action="store_true", help="Keep existing Dir-5A CSVs and only evaluate Dir-5B")
    args = parser.parse_args()

    eval_cache = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/cache") / f"eval_{args.split}"
    configure_runtime_env(cache_root=eval_cache, num_threads=6)
    seed_everything(20261126)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval-Dir5:{args.split}] Admitted GPU {args.physical_gpu}: {gpu_info}", flush=True)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    vae = load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT, device=device)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)

    split_data = load_split_u8(args.split, allow_holdout=True)
    images_u8 = split_data.images_u8
    n_src = int(images_u8.shape[0])
    targets_all = images_u8.to(dtype=torch.float32) / 255.0

    ckpt_records: dict[str, dict] = {
        "swin_backbone": {"path": str(args.backbone_ckpt), "sha256": sha256_file(args.backbone_ckpt)},
        "var_vqvae": {"path": str(DEFAULT_VAR_VQVAE_CKPT), "sha256": sha256_file(DEFAULT_VAR_VQVAE_CKPT)},
    }

    def load_otfs(mth: str, tag: str) -> SympOTFSBlindTransceiver:
        m = SympOTFSBlindTransceiver(backbone, method=mth, hidden_ch=160).to(device)
        cp = args.dir5_ckpt_root / tag / "best_dir5.pt"
        pl = torch.load(cp, map_location="cpu")
        missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
        assert not unexpected, f"Unexpected keys in {cp}: {unexpected}"
        ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp), "best_step": pl.get("step", -1)}
        return m.eval()

    def load_genvar(mth: str, cbr_d: int, tag: str) -> GenHoloVARTransceiver:
        m = GenHoloVARTransceiver(vae, cbr_denom=cbr_d, method=mth, hidden_ch=160).to(device)
        cp = args.dir5_ckpt_root / tag / "best_dir5.pt"
        pl = torch.load(cp, map_location="cpu")
        missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
        assert not unexpected, f"Unexpected keys in {cp}: {unexpected}"
        ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp), "best_step": pl.get("step", -1)}
        return m.eval()

    def load_swin_ul(mth: str, cbr_d: int, tag: str) -> SwinUltraLowTransceiver:
        m = SwinUltraLowTransceiver(backbone, cbr_denom=cbr_d, method=mth, hidden_ch=160).to(device)
        cp = args.dir5_ckpt_root / tag / "best_dir5.pt"
        pl = torch.load(cp, map_location="cpu")
        missing, unexpected = m.load_state_dict(pl["state_dict"], strict=False)
        assert not unexpected, f"Unexpected keys in {cp}: {unexpected}"
        ckpt_records[tag] = {"path": str(cp), "sha256": sha256_file(cp), "best_step": pl.get("step", -1)}
        return m.eval()

    otfs_trained = {
        "BlackBox_OTFS_Blind": load_otfs("blackbox_otfs", "blackbox_otfs"),
        "Symp_OFDM_2DSubspace": load_otfs("symp_ofdm_2d", "symp_ofdm_2d"),
        "Symp_OTFS_JSCC": load_otfs("symp_otfs", "symp_otfs"),
    }

    v4_usph = UniversalBlindTransceiverV4(backbone, method="u_sph", num_rbs=32, hidden_ch=160).to(device)
    v4_usph.load_state_dict(torch.load(args.v4_usph_ckpt, map_location="cpu")["state_dict"], strict=False)
    v4_usph.eval()
    ckpt_records["v4_u_sph"] = {"path": str(args.v4_usph_ckpt), "sha256": sha256_file(args.v4_usph_ckpt)}

    ul_models = {
        cbr_d: {
            "Swin_Genie_CSI": load_swin_ul("swin_genie_csi", cbr_d, f"swin_genie_csi_cbr{cbr_d}"),
            "Swin_Holo_Blind": load_swin_ul("swin_holo_blind", cbr_d, f"swin_holo_blind_cbr{cbr_d}"),
            "BlackBox_VAR_Blind": load_genvar("blackbox_var", cbr_d, f"blackbox_var_cbr{cbr_d}"),
            "GenHolo_VAR_JSCC": load_genvar("genholo_var", cbr_d, f"genholo_var_cbr{cbr_d}"),
        }
        for cbr_d in (128, 256)
    }

    # Pre-extract deterministic held-out stems & VAR f_enc once in float32 (mathematically identical to full forward!)
    stems_l, fenc_l = [], []
    with torch.inference_mode():
        for s in range(0, n_src, 32):
            e = min(n_src, s + 32)
            xb = targets_all[s:e].to(device)
            stems_l.append(v4_usph.extract_encoder_stem(xb).cpu())
            fe, _ = ul_models[128]["GenHolo_VAR_JSCC"].extract_var_latents(xb)
            fenc_l.append(fe.cpu())
    stems_all = torch.cat(stems_l, dim=0)
    fenc_all = torch.cat(fenc_l, dim=0)

    t0 = time.time()
    metric_names = ["psnr_db", "ms_ssim", "lpips_alex", "dists"]

    # =========================================================================
    # TRACK A (DIR-5A): HIGH-MOBILITY DOUBLY-SELECTIVE DELAY-DOPPLER CHANNELS
    # =========================================================================
    print(f"=== [Dir-5A] Evaluating Doubly-Selective Mobility & SNR Sweeps ({args.split}) ===", flush=True)
    otfs_specs = [
        # (cond_name, suite, snr_db, kd_max, approx_kmh, is_ood)
        ("snr_sweep_kd010_1dB", "snr_sweep", 1.0, 0.10, 500, False),
        ("snr_sweep_kd010_4dB", "snr_sweep", 4.0, 0.10, 500, False),
        ("snr_sweep_kd010_7dB", "snr_sweep", 7.0, 0.10, 500, False),
        ("snr_sweep_kd010_10dB", "snr_sweep", 10.0, 0.10, 500, False),
        ("snr_sweep_kd010_14dB", "snr_sweep", 14.0, 0.10, 500, False),
        ("vel_sweep_10dB_kd001", "vel_sweep", 10.0, 0.01, 50, False),
        ("vel_sweep_10dB_kd005", "vel_sweep", 10.0, 0.05, 250, False),
        ("vel_sweep_10dB_kd010", "vel_sweep", 10.0, 0.10, 500, False),
        ("vel_sweep_10dB_kd014_ood", "vel_sweep", 10.0, 0.14, 700, True),
        ("vel_sweep_10dB_kd018_ood", "vel_sweep", 10.0, 0.18, 900, True),
        ("ood_hyper_kd018_14dB", "ood_stress", 14.0, 0.18, 900, True),
    ]
    otfs_mnames = [
        "Frozen_OFDM_Blind",
        "V4_USPH_OFDM_ZeroShot",
        "Pilot_4Sym_Interp_OFDM",
        "BlackBox_OTFS_Blind",
        "Symp_OFDM_2DSubspace",
        "Symp_OTFS_JSCC",
    ]
    otfs_rows: list[dict] = []
    otfs_paired_rows: list[dict] = []
    csi_nmse_summary: dict[str, dict[str, float]] = {}
    prev_summary_p = args.output_dir / f"summary_dir5_{args.split}.json"
    if args.skip_dir5a and prev_summary_p.exists():
        import json
        csi_nmse_summary = json.loads(prev_summary_p.read_text(encoding="utf-8")).get("csi_nmse_summary_db", {})

    with torch.inference_mode():
        for cname, suite, snr_db, kd_max, kmh, is_ood in ([] if args.skip_dir5a else otfs_specs):
            acc = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in otfs_mnames}
            nmse_ours_l, nmse_pilot_l = [], []

            for r_seed in args.noise_seeds:
                gen = torch.Generator(device="cpu").manual_seed(r_seed * 1709 + round(kd_max * 1000) * 13 + round(snr_db))
                eps_td_full = torch.randn((n_src, N_SYM, N_SC, 2), generator=gen, dtype=torch.float32)
                mp_full = sample_doubly_selective_multipath(n_src, kd_max, generator=gen)

                for s in range(0, n_src, args.batch_size):
                    e = min(n_src, s + args.batch_size)
                    xb = targets_all[s:e].to(device)
                    sb = stems_all[s:e].to(device)
                    eps_b = eps_td_full[s:e].to(device)
                    mp_b = (mp_full[0][s:e].to(device), mp_full[1][s:e].to(device))

                    # 1. Trained Dir-5A models
                    for mname, mod in otfs_trained.items():
                        out = mod.forward_doubly_selective(
                            sb,
                            snr_db,
                            kd_max=kd_max,
                            multipath_override=mp_b,
                            standard_noise_td_iq=eps_b,
                            input_is_stem=True,
                        )
                        if mname == "Symp_OTFS_JSCC":
                            ht, hh = out["h_true_tf"], out["h_hat_tf"]
                            nmse_val = ((hh - ht).abs().square().mean(dim=(1, 2)) / ht.abs().square().mean(dim=(1, 2)).clamp_min(1e-8))
                            nmse_ours_l.extend((10.0 * torch.log10(nmse_val.clamp_min(1e-8))).cpu().tolist())
                        mb = evaluator.evaluate_batch(xb, out["recon"])
                        for mk in metric_names:
                            acc[mname][mk][s:e] += np.asarray(mb[mk], dtype=np.float64) / len(args.noise_seeds)

                    # 2. Frozen OFDM Blind
                    out_frz = forward_frozen_ofdm_doubly_selective(
                        backbone, xb, snr_db, kd_max=kd_max, multipath_override=mp_b, standard_noise_td_iq=eps_b
                    )
                    mb_frz = evaluator.evaluate_batch(xb, out_frz["recon"])

                    # 3. V4 U-SPH Zero-Shot over Doubly-Selective OFDM
                    out_v4 = forward_v4_usph_ofdm_doubly_selective(
                        v4_usph, sb, snr_db, kd_max=kd_max, multipath_override=mp_b, standard_noise_td_iq=eps_b, input_is_stem=True
                    )
                    mb_v4 = evaluator.evaluate_batch(xb, out_v4["recon"])

                    # 4. Generous 1,024-Pilot 4-Symbol DMRS Interpolated OFDM
                    out_pil = forward_pilot_4sym_interp_ofdm_doubly_selective(
                        backbone, xb, snr_db, kd_max=kd_max, multipath_override=mp_b, standard_noise_td_iq=eps_b
                    )
                    ht_p, hh_p = out_pil["h_true_tf"], out_pil["h_hat_tf"]
                    nmse_p = ((hh_p - ht_p).abs().square().mean(dim=(1, 2)) / ht_p.abs().square().mean(dim=(1, 2)).clamp_min(1e-8))
                    nmse_pilot_l.extend((10.0 * torch.log10(nmse_p.clamp_min(1e-8))).cpu().tolist())
                    mb_pil = evaluator.evaluate_batch(xb, out_pil["recon"])

                    for mk in metric_names:
                        acc["Frozen_OFDM_Blind"][mk][s:e] += np.asarray(mb_frz[mk], dtype=np.float64) / len(args.noise_seeds)
                        acc["V4_USPH_OFDM_ZeroShot"][mk][s:e] += np.asarray(mb_v4[mk], dtype=np.float64) / len(args.noise_seeds)
                        acc["Pilot_4Sym_Interp_OFDM"][mk][s:e] += np.asarray(mb_pil[mk], dtype=np.float64) / len(args.noise_seeds)

            csi_nmse_summary[cname] = {
                "symp_otfs_blind_nmse_db": round(float(np.mean(nmse_ours_l)), 3),
                "pilot_4sym_interp_nmse_db": round(float(np.mean(nmse_pilot_l)), 3),
            }

            for mname in otfs_mnames:
                row = {
                    "split": args.split,
                    "condition": cname,
                    "suite": suite,
                    "true_snr_db": snr_db,
                    "kd_max": kd_max,
                    "approx_kmh": kmh,
                    "is_ood": is_ood,
                    "model": mname,
                    "dedicated_pilots": 1024 if mname == "Pilot_4Sym_Interp_OFDM" else 0,
                }
                for mk in metric_names:
                    st = paired_bootstrap_ci(acc[mname][mk], num_resamples=5000, seed=20261126)
                    row[f"{mk}_mean"] = round(st["mean"], 6)
                    row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                    row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                    row[f"{mk}_p10"] = round(st["p10"], 6)
                otfs_rows.append(row)

            for oth in otfs_mnames:
                if oth == "Symp_OTFS_JSCC":
                    continue
                st_d = paired_bootstrap_ci(
                    acc["Symp_OTFS_JSCC"]["psnr_db"], acc[oth]["psnr_db"], num_resamples=5000, seed=20261126
                )
                st_di = paired_bootstrap_ci(
                    acc["Symp_OTFS_JSCC"]["dists"], acc[oth]["dists"], num_resamples=5000, seed=20261126
                )
                otfs_paired_rows.append({
                    "split": args.split,
                    "condition": cname,
                    "kd_max": kd_max,
                    "true_snr_db": snr_db,
                    "comparison": f"Symp_OTFS_JSCC_minus_{oth}",
                    "psnr_diff_mean": round(st_d["mean"], 6),
                    "psnr_ci95_lo": round(st_d["ci95_lo"], 6),
                    "psnr_ci95_hi": round(st_d["ci95_hi"], 6),
                    "dists_diff_mean": round(st_di["mean"], 6),
                })
            print(
                f"  [Dir-5A] {cname}: Symp_OTFS={acc['Symp_OTFS_JSCC']['psnr_db'].mean():.2f} dB | "
                f"Symp_OFDM={acc['Symp_OFDM_2DSubspace']['psnr_db'].mean():.2f} dB | "
                f"Pilot4Sym={acc['Pilot_4Sym_Interp_OFDM']['psnr_db'].mean():.2f} dB | "
                f"V4_USPH={acc['V4_USPH_OFDM_ZeroShot']['psnr_db'].mean():.2f} dB",
                flush=True,
            )

    # =========================================================================
    # TRACK B (DIR-5B): ULTRA-LOW CBR = 1/128 (K=1536) & CBR = 1/256 (K=768)
    # =========================================================================
    print(f"=== [Dir-5B] Evaluating Ultra-Low CBR = 1/128 & 1/256 Generative Perceptual JSCC ({args.split}) ===", flush=True)
    ul_specs = [
        # (cond_suffix, snr_db, erase50, is_ood)
        ("snr1dB", 1.0, False, False),
        ("snr4dB", 4.0, False, False),
        ("snr7dB", 7.0, False, False),
        ("snr10dB", 10.0, False, False),
        ("snr13dB", 13.0, False, False),
        ("ood_erase50_10dB", 10.0, True, True),
    ]
    ul_mnames = ["Swin_Genie_CSI", "Swin_Holo_Blind", "BlackBox_VAR_Blind", "GenHolo_VAR_JSCC"]
    ul_rows: list[dict] = []
    ul_paired_rows: list[dict] = []

    with torch.inference_mode():
        for cbr_d in (128, 256):
            k_sym, n_b, _ = cbr_geometry(cbr_d)
            for csuf, snr_db, erase50, is_ood in ul_specs:
                cname = f"cbr{cbr_d}_{csuf}"
                acc_u = {m: {mk: np.zeros(n_src, dtype=np.float64) for mk in metric_names} for m in ul_mnames}
                amask_full = (
                    torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32).expand(n_src, N_UL_RBS)
                    if erase50
                    else torch.ones((n_src, N_UL_RBS), dtype=torch.float32)
                )

                for r_seed in args.noise_seeds:
                    # Use shared seed across cbr_d for h_rb so 8-RB channel realizations match!
                    gen_h = torch.Generator(device="cpu").manual_seed(r_seed * 2311 + round(snr_db) * 19 + (77 if erase50 else 0))
                    hr_f, hi_f = sample_correlated_rbs(n_src, N_UL_RBS, 0.85, device="cpu", generator=gen_h)
                    h_rb_full = torch.stack([hr_f, hi_f], dim=-1)
                    gen_n = torch.Generator(device="cpu").manual_seed(r_seed * 3517 + cbr_d + round(snr_db) * 29)
                    eps_full = torch.randn((n_src, N_UL_RBS, n_b, 2), generator=gen_n, dtype=torch.float32)

                    for s in range(0, n_src, args.batch_size):
                        e = min(n_src, s + args.batch_size)
                        xb = targets_all[s:e].to(device)
                        sb = stems_all[s:e].to(device)
                        feb = fenc_all[s:e].to(device)
                        hb = h_rb_full[s:e].to(device)
                        eb = eps_full[s:e].to(device)
                        amb = amask_full[s:e].to(device)

                        for mname in ("Swin_Genie_CSI", "Swin_Holo_Blind"):
                            mod_s = ul_models[cbr_d][mname]
                            out_s = mod_s.forward_ultralow(
                                xb if mname == "Swin_Genie_CSI" else sb,
                                snr_db,
                                alive_rb_mask=amb,
                                h_rb_override=hb,
                                standard_noise_iq=eb,
                                input_is_stem=(mname != "Swin_Genie_CSI"),
                            )
                            mb_s = evaluator.evaluate_batch(xb, out_s["recon"])
                            for mk in metric_names:
                                acc_u[mname][mk][s:e] += np.asarray(mb_s[mk], dtype=np.float64) / len(args.noise_seeds)

                        for mname in ("BlackBox_VAR_Blind", "GenHolo_VAR_JSCC"):
                            mod_v = ul_models[cbr_d][mname]
                            out_v = mod_v.forward_from_fenc(
                                feb,
                                snr_db,
                                alive_rb_mask=amb,
                                h_rb_override=hb,
                                standard_noise_iq=eb,
                                decode_pixels=True,
                            )
                            mb_v = evaluator.evaluate_batch(xb, out_v["recon"])
                            for mk in metric_names:
                                acc_u[mname][mk][s:e] += np.asarray(mb_v[mk], dtype=np.float64) / len(args.noise_seeds)

                k_surviving = (k_sym // 2) if erase50 else k_sym
                for mname in ul_mnames:
                    row = {
                        "split": args.split,
                        "condition": cname,
                        "cbr_denom": cbr_d,
                        "complex_symbols_tx": k_sym,
                        "complex_symbols_surviving": k_surviving,
                        "true_snr_db": snr_db,
                        "erase50": erase50,
                        "is_ood": is_ood,
                        "model": mname,
                        "csi_requirement": "Oracle_Exact_CSI" if mname == "Swin_Genie_CSI" else "Zero_Pilot_Zero_CSI",
                    }
                    for mk in metric_names:
                        st = paired_bootstrap_ci(acc_u[mname][mk], num_resamples=5000, seed=20261126)
                        row[f"{mk}_mean"] = round(st["mean"], 6)
                        row[f"{mk}_ci95_lo"] = round(st["ci95_lo"], 6)
                        row[f"{mk}_ci95_hi"] = round(st["ci95_hi"], 6)
                        row[f"{mk}_p10"] = round(st["p10"], 6)
                    ul_rows.append(row)

                for oth in ul_mnames:
                    if oth == "GenHolo_VAR_JSCC":
                        continue
                    st_di = paired_bootstrap_ci(
                        acc_u["GenHolo_VAR_JSCC"]["dists"], acc_u[oth]["dists"], num_resamples=5000, seed=20261126
                    )
                    st_lp = paired_bootstrap_ci(
                        acc_u["GenHolo_VAR_JSCC"]["lpips_alex"], acc_u[oth]["lpips_alex"], num_resamples=5000, seed=20261126
                    )
                    st_ps = paired_bootstrap_ci(
                        acc_u["GenHolo_VAR_JSCC"]["psnr_db"], acc_u[oth]["psnr_db"], num_resamples=5000, seed=20261126
                    )
                    ul_paired_rows.append({
                        "split": args.split,
                        "condition": cname,
                        "cbr_denom": cbr_d,
                        "comparison": f"GenHolo_VAR_JSCC_minus_{oth}",
                        "dists_diff_mean": round(st_di["mean"], 6),
                        "dists_ci95_lo": round(st_di["ci95_lo"], 6),
                        "dists_ci95_hi": round(st_di["ci95_hi"], 6),
                        "lpips_diff_mean": round(st_lp["mean"], 6),
                        "lpips_ci95_lo": round(st_lp["ci95_lo"], 6),
                        "lpips_ci95_hi": round(st_lp["ci95_hi"], 6),
                        "psnr_diff_mean": round(st_ps["mean"], 6),
                    })
                print(
                    f"  [Dir-5B] {cname} (K_surv={k_surviving}): "
                    f"GenHolo_VAR DISTS={acc_u['GenHolo_VAR_JSCC']['dists'].mean():.4f} (LPIPS={acc_u['GenHolo_VAR_JSCC']['lpips_alex'].mean():.4f}, PSNR={acc_u['GenHolo_VAR_JSCC']['psnr_db'].mean():.2f}) | "
                    f"Swin_Genie DISTS={acc_u['Swin_Genie_CSI']['dists'].mean():.4f} (PSNR={acc_u['Swin_Genie_CSI']['psnr_db'].mean():.2f}) | "
                    f"BlackBox_VAR DISTS={acc_u['BlackBox_VAR_Blind']['dists'].mean():.4f} (PSNR={acc_u['BlackBox_VAR_Blind']['psnr_db'].mean():.2f})",
                    flush=True,
                )

    if not args.skip_dir5a:
        write_csv(args.output_dir / f"dir5a_otfs_{args.split}.csv", otfs_rows)
        write_csv(args.output_dir / f"dir5a_otfs_paired_{args.split}.csv", otfs_paired_rows)
    write_csv(args.output_dir / f"dir5b_genvar_{args.split}.csv", ul_rows)
    write_csv(args.output_dir / f"dir5b_genvar_paired_{args.split}.csv", ul_paired_rows)

    save_json(
        args.output_dir / f"summary_dir5_{args.split}.json",
        {
            "split": args.split,
            "num_images": n_src,
            "noise_seeds": args.noise_seeds,
            "gpu_info": gpu_info,
            "checkpoints": ckpt_records,
            "csi_nmse_summary_db": csi_nmse_summary,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    )
    verify_no_links([args.output_dir, eval_cache])
    print(f"[Eval-Dir5:{args.split}] ALL DONE in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
