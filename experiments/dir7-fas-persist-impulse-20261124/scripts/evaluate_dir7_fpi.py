#!/usr/bin/env python3
"""Sealed Held-Out Evaluation Suite for Direction 7 (`codex/dir7-fas-persist-impulse-v1`):
Evaluates all matched-control models across 3 paired random seeds (`1001, 2002, 3003`) on `common500` & `kodak24`
with full 5-metric visual & semantic suite (`PSNR`, `MS-SSIM`, `LPIPS-Alex`, `DISTS`, `DINOv2-ViT-L/14` cosine):
  - Suite `dir7a`: Blind-FAS (`fas_kriging`) vs Byte-Identical `fixed_port` & `random_hop` + `pilot32_exhaustive_fas`
                   + `oracle_genie_port_fas` + 10th-percentile outage PSNR (`p10_psnr_db`) + 3x3 Cross-Switch + `last.pt`
  - Suite `dir7b`: PersistGuard (`persistguard_single` & `persistguard_duallayer`) vs `slow_lp_582038a`, `fixed_margin_amc`,
                   `stale_greedy`, and `static_safe` over frozen `VQVAE + VAR-d16` (`N=1,024, CBR=1/192`), calibrated strictly
                   on `val256` ONLY (`2%` & `1%` outage caps across stationary & abrupt regime-switching trajectories)
  - Suite `dir7c`: ImpulseShield (`sem_impulse_2round`) vs Byte-Identical `huber_owndenoise` + end-to-end trained
                   `classical_blanking` & `standard_blind` + `oracle_genie_csi_ofdm` + 2x2 Cross-Switch + `last.pt`
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR6_SRC = Path(__file__).resolve().parents[2] / "dir6-matched-breakthroughs-20261123/src"
DIR5_SRC = Path(__file__).resolve().parents[2] / "dir5-otfs-genvar-20261122/src"
DIR4_SRC = Path(__file__).resolve().parents[2] / "dir4-rf1bit-noma-20261122/src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR6_SRC, DIR5_SRC, DIR4_SRC, DIR3_SRC, DIR2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dir2_zpac.datasets import load_split_u8
from dir2_zpac.hygiene import (
    admit_gpu,
    configure_runtime_env,
    configure_torch_determinism,
    save_json,
    verify_no_links,
)
from dir2_zpac.metrics import METRIC_MODELS_PY, OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir5_og.otfs_blind import N_SC, N_SYM
from dir6_mb.bayes_var_ultralow import load_frozen_vae_and_var_d16

from dir7_fpi import (
    BlindFASTransceiver,
    ImpulseShieldTransceiver,
    build_var_prefix_quality_bank,
    calibrate_policies_on_val256,
    forward_genie_csi_impulsive_ofdm,
    forward_oracle_fas_ofdm,
    forward_pilot32_exhaustive_fas,
    sample_fas_spatial_channels,
    simulate_persist_amc_trajectories,
)
from dir7_fpi.impulse_shield_blind import N_IMP_RBS

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)
ARPC_METRIC_IDENTITY = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1/experiments/arpc-n2048-feasibility-20261019/inputs/arpc_metric_identity.json"
)
DINOV2_WEIGHTS_PTH = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028/"
    "var_comm_runtime_20261010/assets/pilot_metric_weights_seed_v1/VAR_COMM/outputs/UNIFIED-METRICS-20261002/assets/weights/dinov2_vitl14_pretrain.pth"
)
METRIC_KEYS = ("psnr_db", "ms_ssim", "lpips_alex", "dists", "dinov2_cos")


class Full5MetricEvaluator:
    """Offline SHA256-verified 5-Metric Evaluator: PSNR, MS-SSIM, LPIPS-Alex, DISTS, and DINOv2-ViT-L/14 Cosine."""

    def __init__(self, device: torch.device):
        self.device = device
        self.vis_eval = OfflineVisualEvaluator(device=device, include_lpips=True)

        spec = importlib.util.spec_from_file_location("_dir7_eval_mm", str(METRIC_MODELS_PY))
        assert spec is not None and spec.loader is not None
        self.mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mm)
        dino_id = json.loads(ARPC_METRIC_IDENTITY.read_text(encoding="utf-8"))
        dino_spec = dict(dino_id["assets"]["dinov2_manifest"]["dinov2_vitl14"])
        dino_spec["weights"] = dict(dino_spec["weights"], path=str(DINOV2_WEIGHTS_PTH))
        with self.mm.no_network():
            self.dinov2 = self.mm.load_dinov2_vitl14(dino_spec, device)

    @torch.inference_mode()
    def extract_ref_dino(self, images_u8: torch.Tensor, bs: int = 32) -> torch.Tensor:
        feats = []
        for s in range(0, images_u8.shape[0], bs):
            x = images_u8[s : s + bs].to(device=self.device, dtype=torch.float32) / 255.0
            feats.append(self.mm.dinov2_vitl14_features(self.dinov2, x).cpu())
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def evaluate_batch(
        self,
        target_01: torch.Tensor,
        recon_01: torch.Tensor,
        ref_dino_batch: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        x = target_01.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        y = recon_01.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        out = self.vis_eval.evaluate_batch(x, y)
        fx = (
            ref_dino_batch.to(device=self.device, dtype=torch.float32)
            if ref_dino_batch is not None
            else self.mm.dinov2_vitl14_features(self.dinov2, x)
        )
        fy = self.mm.dinov2_vitl14_features(self.dinov2, y)
        d_cos = F.cosine_similarity(fx, fy, dim=-1)
        out["dinov2_cos"] = [float(v) for v in d_cos.cpu().tolist()]
        return out


def summarize_seeds(seed_means: list[dict[str, float]]) -> dict[str, dict[str, float | list[float]]]:
    summary: dict[str, dict[str, float | list[float]]] = {}
    for k in seed_means[0].keys():
        vals = [float(sm[k]) for sm in seed_means]
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
        ci95 = float(1.96 * sd / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        summary[k] = {"mean": mean, "std": sd, "ci95": ci95, "per_seed": vals}
    return summary


def summarize_paired_delta(
    prop_seeds: list[dict[str, float]],
    base_seeds: list[dict[str, float]],
) -> dict[str, dict[str, float | list[float]]]:
    diffs = [{k: float(p[k] - b[k]) for k in p.keys()} for p, b in zip(prop_seeds, base_seeds)]
    return summarize_seeds(diffs)


def _extract_swin_stems(backbone: torch.nn.Module, imgs_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    enc = backbone.encoder
    outs = []
    for s in range(0, imgs_u8.shape[0], 32):
        xb = imgs_u8[s : s + 32].to(device=device, dtype=torch.float32) / 255.0
        x = enc.patch_embed(xb)
        for layer in enc.layers:
            x = layer(x)
        outs.append(enc.norm(x).cpu())
    return torch.cat(outs, dim=0)


@torch.inference_mode()
def evaluate_suite_dir7a(
    split_name: str,
    seeds: list[int],
    ckpt_root: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict:
    imgs_u8 = load_split_u8(split_name, allow_holdout=True).images_u8
    N = imgs_u8.shape[0]
    ref_dino = evaluator.extract_ref_dino(imgs_u8)
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    stems_all = _extract_swin_stems(backbone, imgs_u8, device)

    # (cond_name, snr_db, n_subbands, rician_k_linear, is_ood)
    conditions = [
        ("nb1_rayleigh_snr0", 0.0, 1, 0.0, False),
        ("nb1_rayleigh_snr3", 3.0, 1, 0.0, False),
        ("nb1_rayleigh_snr6", 6.0, 1, 0.0, False),
        ("nb1_rayleigh_snr10", 10.0, 1, 0.0, False),
        ("wb4_pixel_fas_snr4_ood", 4.0, 4, 0.0, True),
        ("nb1_rician_k2_snr4_ood", 4.0, 1, 2.0, True),
    ]

    port_modes = ("fas_kriging", "fixed_port", "random_hop")
    models_best: dict[int, dict[str, BlindFASTransceiver]] = {sd: {} for sd in seeds}
    models_last: dict[int, dict[str, BlindFASTransceiver]] = {sd: {} for sd in seeds}
    for sd in seeds:
        for pm in port_modes:
            mb = BlindFASTransceiver(backbone, port_mode=pm).to(device).eval()
            ck_b = torch.load(ckpt_root / f"dir7a_{pm}" / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            mb.load_state_dict(ck_b["model_state"], strict=False)
            models_best[sd][pm] = mb

            ml = BlindFASTransceiver(backbone, port_mode=pm).to(device).eval()
            ck_l = torch.load(ckpt_root / f"dir7a_{pm}" / f"seed_{sd}" / "last.pt", map_location="cpu", weights_only=True)
            ml.load_state_dict(ck_l["model_state"], strict=False)
            models_last[sd][pm] = ml

    bs = 25
    suite_out: dict = {"conditions": {}, "cross_switch_3x3": {}, "last_step_check": {}}

    for idx_c, (cname, snr_db, n_sb, ric_k, is_ood) in enumerate(conditions):
        methods = ("fas_kriging", "fixed_port", "random_hop", "pilot32_exhaustive_fas", "oracle_genie_port_fas")
        seed_res: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        last_res: dict[str, list[dict[str, float]]] = {pm: [] for pm in port_modes}
        cross_res: dict[str, list[dict[str, float]]] = {
            f"w_{w_pm}__sched_{s_pm}": [] for w_pm in port_modes for s_pm in port_modes
        }

        for sd in seeds:
            gen = torch.Generator(device="cpu").manual_seed(sd * 100 + idx_c * 17 + 71)
            h_all = sample_fas_spatial_channels(N, n_subbands=n_sb, rician_k_linear=ric_k, generator=gen, device="cpu")
            n_all = torch.randn((N, N_SYM, N_SC, 2), generator=gen)

            acc_m = {m: {k: [] for k in METRIC_KEYS} for m in methods}
            acc_l = {pm: {k: [] for k in METRIC_KEYS} for pm in port_modes}
            acc_x = {xk: {k: [] for k in METRIC_KEYS} for xk in cross_res}

            for s in range(0, N, bs):
                e = min(N, s + bs)
                x01 = imgs_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
                st_b = stems_all[s:e].to(device=device, dtype=torch.float32)
                rd = ref_dino[s:e]
                hb = h_all[s:e].to(device)
                nb = n_all[s:e].to(device)

                for pm in port_modes:
                    ob = models_best[sd][pm].forward_fas(
                        st_b, snr_db, n_subbands=n_sb, rician_k_linear=ric_k, h_port_override=hb, standard_noise_iq=nb, input_is_stem=True
                    )
                    ev_b = evaluator.evaluate_batch(x01, ob["recon"], rd)
                    for k in METRIC_KEYS:
                        acc_m[pm][k].extend(ev_b[k])

                op32 = forward_pilot32_exhaustive_fas(backbone, x01, snr_db, hb, nb)
                ev_p32 = evaluator.evaluate_batch(x01, op32["recon"], rd)
                oorc = forward_oracle_fas_ofdm(backbone, x01, snr_db, hb, nb)
                ev_orc = evaluator.evaluate_batch(x01, oorc["recon"], rd)
                for k in METRIC_KEYS:
                    acc_m["pilot32_exhaustive_fas"][k].extend(ev_p32[k])
                    acc_m["oracle_genie_port_fas"][k].extend(ev_orc[k])

                # 3x3 Weight x Port-Policy Cross-Switch on representative `nb1_rayleigh_snr3`
                if cname == "nb1_rayleigh_snr3":
                    for w_pm in port_modes:
                        for s_pm in port_modes:
                            ox = models_best[sd][w_pm].forward_fas(
                                st_b,
                                snr_db,
                                n_subbands=n_sb,
                                rician_k_linear=ric_k,
                                h_port_override=hb,
                                standard_noise_iq=nb,
                                input_is_stem=True,
                                override_port_mode=s_pm,
                            )
                            ev_x = evaluator.evaluate_batch(x01, ox["recon"], rd)
                            for k in METRIC_KEYS:
                                acc_x[f"w_{w_pm}__sched_{s_pm}"][k].extend(ev_x[k])

                # Fixed final step `last.pt` audit on representative `nb1_rayleigh_snr3` & `wb4_pixel_fas_snr4_ood`
                if cname in ("nb1_rayleigh_snr3", "wb4_pixel_fas_snr4_ood"):
                    for pm in port_modes:
                        ol = models_last[sd][pm].forward_fas(
                            st_b, snr_db, n_subbands=n_sb, rician_k_linear=ric_k, h_port_override=hb, standard_noise_iq=nb, input_is_stem=True
                        )
                        ev_l = evaluator.evaluate_batch(x01, ol["recon"], rd)
                        for k in METRIC_KEYS:
                            acc_l[pm][k].extend(ev_l[k])

            for m in methods:
                row_m = {k: float(np.mean(acc_m[m][k])) for k in METRIC_KEYS}
                row_m["p10_psnr_db"] = float(np.percentile(acc_m[m]["psnr_db"], 10))
                seed_res[m].append(row_m)

            if cname == "nb1_rayleigh_snr3":
                for xk in cross_res:
                    row_x = {k: float(np.mean(acc_x[xk][k])) for k in METRIC_KEYS}
                    row_x["p10_psnr_db"] = float(np.percentile(acc_x[xk]["psnr_db"], 10))
                    cross_res[xk].append(row_x)

            if cname in ("nb1_rayleigh_snr3", "wb4_pixel_fas_snr4_ood"):
                for pm in port_modes:
                    row_l = {k: float(np.mean(acc_l[pm][k])) for k in METRIC_KEYS}
                    row_l["p10_psnr_db"] = float(np.percentile(acc_l[pm]["psnr_db"], 10))
                    last_res[pm].append(row_l)

        suite_out["conditions"][cname] = {
            "snr_db": snr_db,
            "n_subbands": n_sb,
            "rician_k_linear": ric_k,
            "is_ood": is_ood,
            "models": {m: summarize_seeds(seed_res[m]) for m in methods},
            "paired_deltas": {
                "fas_kriging_vs_fixed_port": summarize_paired_delta(seed_res["fas_kriging"], seed_res["fixed_port"]),
                "fas_kriging_vs_random_hop": summarize_paired_delta(seed_res["fas_kriging"], seed_res["random_hop"]),
                "fas_kriging_vs_pilot32": summarize_paired_delta(seed_res["fas_kriging"], seed_res["pilot32_exhaustive_fas"]),
            },
        }
        if cname == "nb1_rayleigh_snr3":
            suite_out["cross_switch_3x3"][cname] = {xk: summarize_seeds(cross_res[xk]) for xk in cross_res}
        if cname in ("nb1_rayleigh_snr3", "wb4_pixel_fas_snr4_ood"):
            suite_out["last_step_check"][cname] = {
                pm: summarize_seeds(last_res[pm]) for pm in port_modes
            }
            suite_out["last_step_check"][cname]["fas_vs_fixed_delta"] = summarize_paired_delta(
                last_res["fas_kriging"], last_res["fixed_port"]
            )

    return suite_out


def _load_or_build_var_bank(
    split_name: str,
    cache_dir: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict[str, np.ndarray]:
    npz_path = cache_dir / f"var_prefix_bank_{split_name}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        return {k: np.asarray(data[k], dtype=np.float64) for k in data.files}

    imgs_u8 = load_split_u8(split_name, allow_holdout=(split_name != "val256")).images_u8
    vae, var_d16 = load_frozen_vae_and_var_d16(device=device, load_var=True)
    bank = build_var_prefix_quality_bank(imgs_u8, vae, var_d16, evaluator, device, batch_size=24)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **bank)
    return bank


@torch.inference_mode()
def evaluate_suite_dir7b(
    split_name: str,
    seeds: list[int],
    cache_dir: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict:
    # Calibration strictly on `val256` ONLY (zero leakage from `common500` or `kodak24`!)
    val_bank = _load_or_build_var_bank("val256", cache_dir, evaluator, device)
    eval_bank = _load_or_build_var_bank(split_name, cache_dir, evaluator, device)

    cal_2pct = calibrate_policies_on_val256(val_bank, outage_cap=0.02, cal_seed=20261124)
    cal_1pct = calibrate_policies_on_val256(val_bank, outage_cap=0.01, cal_seed=20261125)

    regimes = [
        ("stationary_slow_p095", "stationary_slow_p095", cal_2pct, 0.02),
        ("mismatch_slow_to_fast", "mismatch_slow_to_fast", cal_2pct, 0.02),
        ("nonstationary_regime_switch", "nonstationary_regime_switch", cal_2pct, 0.02),
        ("strict_cap1pct_regime_switch", "nonstationary_regime_switch", cal_1pct, 0.01),
    ]
    policies = (
        "static_safe",
        "stale_greedy",
        "fixed_margin_amc",
        "slow_lp_582038a",
        "persistguard_single",
        "persistguard_duallayer",
    )

    out_b: dict = {
        "cbr": "1/192 (N=1024 complex symbols = 64 header + 960 payload)",
        "prefix_curve_summary": {
            f"m_{m}": {k: float(eval_bank[k][:, m].mean()) for k in METRIC_KEYS}
            for m in (0, 4, 5, 6, 7, 8, 9, 10)
        },
        "regimes": {},
        "sample_trajectory_regime_switch": None,
    }

    for r_key, r_mode, cal_tbl, cap_val in regimes:
        seed_pol: dict[str, list[dict[str, float]]] = {p: [] for p in policies}
        for i_s, sd in enumerate(seeds):
            want_traj = (r_key == "nonstationary_regime_switch" and i_s == 0)
            sim = simulate_persist_amc_trajectories(
                eval_bank,
                cal_tbl,
                regime=r_mode,
                seed=sd,
                n_episodes=50,
                episode_len=120,
                return_trajectory=want_traj,
            )
            if want_traj:
                out_b["sample_trajectory_regime_switch"] = sim["trajectory"]
                summary = sim["summary"]
            else:
                summary = sim
            for p in policies:
                seed_pol[p].append(summary[p])

        out_b["regimes"][r_key] = {
            "outage_cap": cap_val,
            "policies": {p: summarize_seeds(seed_pol[p]) for p in policies},
            "paired_deltas": {
                "persistguard_single_vs_slow_lp_582038a": summarize_paired_delta(
                    seed_pol["persistguard_single"], seed_pol["slow_lp_582038a"]
                ),
                "persistguard_single_vs_static_safe": summarize_paired_delta(
                    seed_pol["persistguard_single"], seed_pol["static_safe"]
                ),
                "persistguard_duallayer_vs_slow_lp_582038a": summarize_paired_delta(
                    seed_pol["persistguard_duallayer"], seed_pol["slow_lp_582038a"]
                ),
            },
        }

    return out_b


def _make_det_impulses(N: int, p_imp: float, imp_sir_db: float, gen: torch.Generator) -> torch.Tensor:
    if p_imp <= 0.0:
        return torch.zeros((N, N_SYM, N_SC, 2), dtype=torch.float32)
    u1 = torch.rand((N, N_SYM, N_SC), generator=gen)
    u2 = torch.rand((N, N_SYM, N_SC), generator=gen)
    mask_seed = (u1 < (0.65 * float(p_imp))).to(torch.float32)
    mask_ring = torch.roll(mask_seed, shifts=1, dims=-1) * (u2 < 0.55).to(torch.float32)
    mask = (mask_seed + mask_ring).clamp(0.0, 1.0)[..., None]
    amp = math.pow(10.0, -float(imp_sir_db) / 20.0) / math.sqrt(2.0)
    g_iq = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
    return mask * amp * g_iq


def _make_det_rbs(N: int, rho_f: float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    z_hr = torch.randn((N, N_IMP_RBS), generator=gen)
    z_hi = torch.randn((N, N_IMP_RBS), generator=gen)
    hr = torch.zeros_like(z_hr)
    hi = torch.zeros_like(z_hi)
    hr[:, 0] = z_hr[:, 0] * math.sqrt(0.5)
    hi[:, 0] = z_hi[:, 0] * math.sqrt(0.5)
    for m in range(1, N_IMP_RBS):
        hr[:, m] = rho_f * hr[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hr[:, m]
        hi[:, m] = rho_f * hi[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hi[:, m]
    return hr, hi


@torch.inference_mode()
def evaluate_suite_dir7c(
    split_name: str,
    seeds: list[int],
    ckpt_root: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict:
    imgs_u8 = load_split_u8(split_name, allow_holdout=True).images_u8
    N = imgs_u8.shape[0]
    ref_dino = evaluator.extract_ref_dino(imgs_u8)
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    stems_all = _extract_swin_stems(backbone, imgs_u8, device)

    # (cond_name, snr_db, p_imp, imp_sir_db, is_ood)
    conditions = [
        ("clean_awgn_snr14", 14.0, 0.00, -7.0, False),
        ("imp04_sir6_snr12", 12.0, 0.04, -6.0, False),
        ("imp06_sir7_snr12", 12.0, 0.06, -7.0, False),
        ("imp08_sir8_snr10", 10.0, 0.08, -8.0, False),
        ("imp08_sir9_snr14", 14.0, 0.08, -9.0, False),
        ("imp12_sir10_snr12_ood", 12.0, 0.12, -10.0, True),
    ]

    rx_modes = ("sem_impulse_2round", "huber_owndenoise", "classical_blanking", "standard_blind")
    models_best: dict[int, dict[str, ImpulseShieldTransceiver]] = {sd: {} for sd in seeds}
    models_last: dict[int, dict[str, ImpulseShieldTransceiver]] = {sd: {} for sd in seeds}
    for sd in seeds:
        for rm in rx_modes:
            mb = ImpulseShieldTransceiver(backbone, rx_mode=rm).to(device).eval()
            ck_b = torch.load(ckpt_root / f"dir7c_{rm}" / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            mb.load_state_dict(ck_b["model_state"], strict=False)
            models_best[sd][rm] = mb

            if rm in ("sem_impulse_2round", "huber_owndenoise"):
                ml = ImpulseShieldTransceiver(backbone, rx_mode=rm).to(device).eval()
                ck_l = torch.load(ckpt_root / f"dir7c_{rm}" / f"seed_{sd}" / "last.pt", map_location="cpu", weights_only=True)
                ml.load_state_dict(ck_l["model_state"], strict=False)
                models_last[sd][rm] = ml

    bs = 25
    suite_c: dict = {"conditions": {}, "cross_switch_2x2": {}, "last_step_check": {}}

    for idx_c, (cname, snr_db, p_imp, sir_db, is_ood) in enumerate(conditions):
        methods = (*rx_modes, "oracle_genie_csi_ofdm")
        seed_res: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        last_res: dict[str, list[dict[str, float]]] = {"sem_impulse_2round": [], "huber_owndenoise": []}
        cross_res: dict[str, list[dict[str, float]]] = {
            f"w_{w_rm}__loop_{l_rm}": []
            for w_rm in ("sem_impulse_2round", "huber_owndenoise")
            for l_rm in ("sem_impulse_2round", "huber_owndenoise")
        }

        for sd in seeds:
            gen = torch.Generator(device="cpu").manual_seed(sd * 130 + idx_c * 29 + 73)
            hr_all, hi_all = _make_det_rbs(N, 0.88, gen)
            awgn_all = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
            imp_all = _make_det_impulses(N, p_imp, sir_db, gen)

            acc_m = {m: {k: [] for k in METRIC_KEYS} for m in methods}
            acc_l = {rm: {k: [] for k in METRIC_KEYS} for rm in ("sem_impulse_2round", "huber_owndenoise")}
            acc_x = {xk: {k: [] for k in METRIC_KEYS} for xk in cross_res}

            for s in range(0, N, bs):
                e = min(N, s + bs)
                x01 = imgs_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
                st_b = stems_all[s:e].to(device=device, dtype=torch.float32)
                rd = ref_dino[s:e]
                hr_b, hi_b = hr_all[s:e].to(device), hi_all[s:e].to(device)
                aw_b = awgn_all[s:e].to(device)
                im_b = imp_all[s:e].to(device)

                for rm in rx_modes:
                    ob = models_best[sd][rm].forward_impulsive(
                        st_b,
                        snr_db,
                        p_imp=p_imp,
                        imp_sir_db=sir_db,
                        h_rb_override=(hr_b, hi_b),
                        awgn_td_override=aw_b,
                        impulse_td_override=im_b,
                        input_is_stem=True,
                    )
                    ev_b = evaluator.evaluate_batch(x01, ob["recon"], rd)
                    for k in METRIC_KEYS:
                        acc_m[rm][k].extend(ev_b[k])

                ogen = forward_genie_csi_impulsive_ofdm(
                    backbone,
                    x01,
                    snr_db,
                    p_imp=p_imp,
                    imp_sir_db=sir_db,
                    h_rb_override=(hr_b, hi_b),
                    awgn_td_override=aw_b,
                    impulse_td_override=im_b,
                )
                ev_gen = evaluator.evaluate_batch(x01, ogen["recon"], rd)
                for k in METRIC_KEYS:
                    acc_m["oracle_genie_csi_ofdm"][k].extend(ev_gen[k])

                if cname == "imp08_sir9_snr14":
                    for w_rm in ("sem_impulse_2round", "huber_owndenoise"):
                        for l_rm in ("sem_impulse_2round", "huber_owndenoise"):
                            ox = models_best[sd][w_rm].forward_impulsive(
                                st_b,
                                snr_db,
                                p_imp=p_imp,
                                imp_sir_db=sir_db,
                                h_rb_override=(hr_b, hi_b),
                                awgn_td_override=aw_b,
                                impulse_td_override=im_b,
                                input_is_stem=True,
                                override_rx_mode=l_rm,
                            )
                            ev_x = evaluator.evaluate_batch(x01, ox["recon"], rd)
                            for k in METRIC_KEYS:
                                acc_x[f"w_{w_rm}__loop_{l_rm}"][k].extend(ev_x[k])

                if cname in ("imp08_sir9_snr14", "imp12_sir10_snr12_ood"):
                    for rm in ("sem_impulse_2round", "huber_owndenoise"):
                        ol = models_last[sd][rm].forward_impulsive(
                            st_b,
                            snr_db,
                            p_imp=p_imp,
                            imp_sir_db=sir_db,
                            h_rb_override=(hr_b, hi_b),
                            awgn_td_override=aw_b,
                            impulse_td_override=im_b,
                            input_is_stem=True,
                        )
                        ev_l = evaluator.evaluate_batch(x01, ol["recon"], rd)
                        for k in METRIC_KEYS:
                            acc_l[rm][k].extend(ev_l[k])

            for m in methods:
                seed_res[m].append({k: float(np.mean(acc_m[m][k])) for k in METRIC_KEYS})

            if cname == "imp08_sir9_snr14":
                for xk in cross_res:
                    cross_res[xk].append({k: float(np.mean(acc_x[xk][k])) for k in METRIC_KEYS})

            if cname in ("imp08_sir9_snr14", "imp12_sir10_snr12_ood"):
                for rm in ("sem_impulse_2round", "huber_owndenoise"):
                    last_res[rm].append({k: float(np.mean(acc_l[rm][k])) for k in METRIC_KEYS})

        suite_c["conditions"][cname] = {
            "snr_db": snr_db,
            "p_imp": p_imp,
            "imp_sir_db": sir_db,
            "is_ood": is_ood,
            "models": {m: summarize_seeds(seed_res[m]) for m in methods},
            "paired_deltas": {
                "sem_impulse_vs_huber_owndenoise": summarize_paired_delta(
                    seed_res["sem_impulse_2round"], seed_res["huber_owndenoise"]
                ),
                "sem_impulse_vs_classical_blanking": summarize_paired_delta(
                    seed_res["sem_impulse_2round"], seed_res["classical_blanking"]
                ),
                "sem_impulse_vs_standard_blind": summarize_paired_delta(
                    seed_res["sem_impulse_2round"], seed_res["standard_blind"]
                ),
                "sem_impulse_vs_oracle_genie_ofdm": summarize_paired_delta(
                    seed_res["sem_impulse_2round"], seed_res["oracle_genie_csi_ofdm"]
                ),
            },
        }
        if cname == "imp08_sir9_snr14":
            suite_c["cross_switch_2x2"][cname] = {xk: summarize_seeds(cross_res[xk]) for xk in cross_res}
        if cname in ("imp08_sir9_snr14", "imp12_sir10_snr12_ood"):
            suite_c["last_step_check"][cname] = {
                rm: summarize_seeds(last_res[rm]) for rm in ("sem_impulse_2round", "huber_owndenoise")
            }
            suite_c["last_step_check"][cname]["sem_vs_own_delta"] = summarize_paired_delta(
                last_res["sem_impulse_2round"], last_res["huber_owndenoise"]
            )

    return suite_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["dir7a", "dir7b", "dir7c", "all"], default="all")
    parser.add_argument("--splits", type=str, default="common500,kodak24")
    parser.add_argument("--seeds", type=str, default="1001,2002,3003")
    parser.add_argument(
        "--ckpt_root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir7_fas_persist_impulse_v1"),
    )
    parser.add_argument(
        "--tmp_root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir7_fas_persist_impulse_v1"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    admit_gpu(min_free_mib=22000)
    device = torch.device("cuda:0")

    evaluator = Full5MetricEvaluator(device)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.tmp_root / "cache"

    suites = ["dir7a", "dir7b", "dir7c"] if args.suite == "all" else [args.suite]
    for st_name in suites:
        for sp in splits:
            print(f"[Eval] Running {st_name} on {sp} (seeds={seeds})...", flush=True)
            if st_name == "dir7a":
                res = evaluate_suite_dir7a(sp, seeds, args.ckpt_root, evaluator, device)
            elif st_name == "dir7b":
                res = evaluate_suite_dir7b(sp, seeds, cache_dir, evaluator, device)
            else:
                res = evaluate_suite_dir7c(sp, seeds, args.ckpt_root, evaluator, device)
            out_file = args.out_dir / f"eval_{st_name}_{sp}.json"
            save_json(out_file, res)
            print(f"[Eval] Saved {out_file}", flush=True)

    verify_no_links([args.ckpt_root, args.tmp_root, args.out_dir])


if __name__ == "__main__":
    main()
