#!/usr/bin/env python3
"""Sealed Held-Out Evaluation Suite for Direction 6 (`codex/dir6-matched-breakthroughs-v1`):
Evaluates all matched-control models across 3 paired random seeds (`1001, 2002, 3003`) on `common500` & `kodak24`
with full 5-metric visual & semantic suite (`PSNR`, `MS-SSIM`, `LPIPS-Alex`, `DISTS`, and `DINOv2-ViT-L/14` cosine):
  - Suite `dir6a`: Burst_Symp_OTFS vs Byte-Identical Burst_Symp_OFDM_2D + Oracle_Genie_CSI_OFDM + 2x2 Cross-Switch + `last.pt`
  - Suite `dir6b`: BayesVAR (AR=ON) vs Exact Single-Switch BayesVAR (AR=OFF) vs Loss-Matched Swin_Holo_Perc & Swin_Genie_Perc
                   at CBR=1/512 (K=384) & CBR=1/256 (K=768) + `last.pt`
  - Suite `dir6c`: SemSIC_NOMA_4x2 vs Byte-Identical OwnDenoise_NOMA_4x2 vs Val-Tuned Conventional_PIC_3Round vs
                   Linear_LMMSE_1Stage vs Oracle_Genie_Linear_MMSE on STRICTLY DISJOINT 4-Image Tuples + Blind CFAR Subset Audit!
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

PKG_SRC = Path(__file__).resolve().parents[1] / "src"
DIR5_SRC = Path(__file__).resolve().parents[2] / "dir5-otfs-genvar-20261122/src"
DIR4_SRC = Path(__file__).resolve().parents[2] / "dir4-rf1bit-noma-20261122/src"
DIR3_SRC = Path(__file__).resolve().parents[2] / "dir3-mimo-holorate-20261121/src"
DIR2_SRC = Path(__file__).resolve().parents[2] / "swinjscc-mismatch-zpac-20261120/src"
for p in (PKG_SRC, DIR5_SRC, DIR4_SRC, DIR3_SRC, DIR2_SRC):
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
from dir2_zpac.metrics import METRIC_MODELS_PY, SUPPLEMENT_MANIFEST, OfflineVisualEvaluator
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
from dir5_og.otfs_blind import N_SC, N_SYM, sample_doubly_selective_multipath

from dir6_mb import (
    BayesVARTransceiver,
    BurstMatchedTransceiver,
    OverloadedNOMA4x2Transceiver,
    PerceptualSwinUltraLowTransceiver,
    forward_genie_csi_ofdm_burst,
    forward_genie_linear_noma_4x2,
    load_frozen_vae_and_var_d16,
    sample_blockage_envelope,
    ultralow_dir6_geometry,
)
from dir6_mb.overloaded_noma_4x2 import N_B_4X2

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
    """Offline SHA256-verified 5-Metric Evaluator: PSNR, MS-SSIM, LPIPS-Alex, DISTS, and DINOv2-ViT-L/14 Cosine!"""

    def __init__(self, device: torch.device):
        self.device = device
        self.vis_eval = OfflineVisualEvaluator(device=device, include_lpips=True)

        spec = importlib.util.spec_from_file_location("_dir6_eval_mm", str(METRIC_MODELS_PY))
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
        return torch.cat(feats, dim=0)  # [N, 1024]

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


@torch.inference_mode()
def evaluate_suite_dir6a(
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

    # Precompute encoder stems once for fast evaluation
    stems_list = []
    enc = backbone.encoder
    for s in range(0, N, 32):
        xb = imgs_u8[s : s + 32].to(device=device, dtype=torch.float32) / 255.0
        x = enc.patch_embed(xb)
        for layer in enc.layers:
            x = layer(x)
        stems_list.append(enc.norm(x).cpu())
    stems_all = torch.cat(stems_list, dim=0)

    conditions = [
        ("unblocked_kd010_snr12", 12.0, 0.10, 0, False),
        ("burst8_kd010_snr12", 12.0, 0.10, 8, False),
        ("burst12_kd010_snr10", 10.0, 0.10, 12, False),
        ("burst12_kd010_snr14", 14.0, 0.10, 12, False),
        ("burst16_kd010_snr14_ood", 14.0, 0.10, 16, True),
        ("burst16_kd018_snr14_ood", 14.0, 0.18, 16, True),
    ]

    models_by_seed: dict[int, dict[str, BurstMatchedTransceiver]] = {}
    last_models_by_seed: dict[int, dict[str, BurstMatchedTransceiver]] = {}
    for sd in seeds:
        models_by_seed[sd] = {}
        last_models_by_seed[sd] = {}
        for m_name, u_otfs in [("burst_symp_otfs", True), ("burst_symp_ofdm_2d", False)]:
            mdl = BurstMatchedTransceiver(backbone, use_otfs=u_otfs).to(device)
            ck = torch.load(ckpt_root / m_name / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            mdl.load_state_dict(ck["model_state"], strict=False)
            mdl.eval()
            models_by_seed[sd][m_name] = mdl

            mdl_last = BurstMatchedTransceiver(backbone, use_otfs=u_otfs).to(device)
            ck_l = torch.load(ckpt_root / m_name / f"seed_{sd}" / "last.pt", map_location="cpu", weights_only=True)
            mdl_last.load_state_dict(ck_l["model_state"], strict=False)
            mdl_last.eval()
            last_models_by_seed[sd][m_name] = mdl_last

    results: dict[str, dict] = {"split": split_name, "num_images": N, "cbr_per_rgb_scalar": "1/16 (K=12,288 REs)", "conditions": {}, "cross_switch_2x2": {}, "last_step_audit": {}}

    for cond_name, snr_v, kd_v, n_blk, is_ood in conditions:
        per_method_seeds: dict[str, list[dict[str, float]]] = {
            "burst_symp_otfs": [],
            "burst_symp_ofdm_2d": [],
            "oracle_genie_csi_ofdm": [],
        }
        cross_seeds: dict[str, list[dict[str, float]]] = {
            "w_otfs__wave_otfs": [],
            "w_otfs__wave_ofdm": [],
            "w_ofdm__wave_otfs": [],
            "w_ofdm__wave_ofdm": [],
        }
        last_seeds: dict[str, list[dict[str, float]]] = {"burst_symp_otfs_last": [], "burst_symp_ofdm_2d_last": []}

        for s_idx, sd in enumerate(seeds):
            gen = torch.Generator(device="cpu").manual_seed(710000 + sd * 31 + n_blk * 7 + int(round(kd_v * 100)))
            mp = sample_doubly_selective_multipath(N, kd_v, generator=gen)
            eps = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
            blk = sample_blockage_envelope(N, n_blk, generator=gen, device="cpu")

            acc = {m: {k: [] for k in METRIC_KEYS} for m in per_method_seeds}
            acc_x = {m: {k: [] for k in METRIC_KEYS} for m in cross_seeds}
            acc_l = {m: {k: [] for k in METRIC_KEYS} for m in last_seeds}

            bs = 24
            for st in range(0, N, bs):
                en = min(N, st + bs)
                tgt = imgs_u8[st:en].to(device=device, dtype=torch.float32) / 255.0
                sb = stems_all[st:en].to(device)
                rd = ref_dino[st:en]
                mp_b = (mp[0][st:en], mp[1][st:en])
                eps_b = eps[st:en]
                blk_b = blk[st:en].to(device)

                for m_name in ("burst_symp_otfs", "burst_symp_ofdm_2d"):
                    out = models_by_seed[sd][m_name].forward_burst(
                        sb, snr_v, kd_max=kd_v, blockage_env=blk_b, multipath_override=mp_b, standard_noise_td_iq=eps_b, input_is_stem=True
                    )
                    ev = evaluator.evaluate_batch(tgt, out["recon"], rd)
                    for k in METRIC_KEYS:
                        acc[m_name][k].extend(ev[k])

                out_g = forward_genie_csi_ofdm_burst(
                    backbone, tgt, snr_v, kd_max=kd_v, blockage_env=blk_b, multipath_override=mp_b, standard_noise_td_iq=eps_b
                )
                ev_g = evaluator.evaluate_batch(tgt, out_g["recon"], rd)
                for k in METRIC_KEYS:
                    acc["oracle_genie_csi_ofdm"][k].extend(ev_g[k])

                if cond_name in ("burst12_kd010_snr14", "burst16_kd010_snr14_ood"):
                    # 2x2 Weight x Waveform Cross-Switch Audit + Fixed Final Step Audit!
                    for x_key, w_key, wave_flag in [
                        ("w_otfs__wave_otfs", "burst_symp_otfs", True),
                        ("w_otfs__wave_ofdm", "burst_symp_otfs", False),
                        ("w_ofdm__wave_otfs", "burst_symp_ofdm_2d", True),
                        ("w_ofdm__wave_ofdm", "burst_symp_ofdm_2d", False),
                    ]:
                        ox = models_by_seed[sd][w_key].forward_burst(
                            sb, snr_v, kd_max=kd_v, blockage_env=blk_b, multipath_override=mp_b, standard_noise_td_iq=eps_b, input_is_stem=True, override_use_otfs=wave_flag
                        )
                        evx = evaluator.evaluate_batch(tgt, ox["recon"], rd)
                        for k in METRIC_KEYS:
                            acc_x[x_key][k].extend(evx[k])

                    for l_key, m_base in [("burst_symp_otfs_last", "burst_symp_otfs"), ("burst_symp_ofdm_2d_last", "burst_symp_ofdm_2d")]:
                        ol = last_models_by_seed[sd][m_base].forward_burst(
                            sb, snr_v, kd_max=kd_v, blockage_env=blk_b, multipath_override=mp_b, standard_noise_td_iq=eps_b, input_is_stem=True
                        )
                        evl = evaluator.evaluate_batch(tgt, ol["recon"], rd)
                        for k in METRIC_KEYS:
                            acc_l[l_key][k].extend(evl[k])

            for m in per_method_seeds:
                per_method_seeds[m].append({k: float(np.mean(acc[m][k])) for k in METRIC_KEYS})
            if cond_name in ("burst12_kd010_snr14", "burst16_kd010_snr14_ood"):
                for m in cross_seeds:
                    cross_seeds[m].append({k: float(np.mean(acc_x[m][k])) for k in METRIC_KEYS})
                for m in last_seeds:
                    last_seeds[m].append({k: float(np.mean(acc_l[m][k])) for k in METRIC_KEYS})

        cond_summary = {
            "is_ood": is_ood,
            "snr_db": snr_v,
            "kd_max": kd_v,
            "num_blocked_syms": n_blk,
            "methods": {m: summarize_seeds(per_method_seeds[m]) for m in per_method_seeds},
            "paired_delta_otfs_minus_ofdm2d": summarize_paired_delta(
                per_method_seeds["burst_symp_otfs"], per_method_seeds["burst_symp_ofdm_2d"]
            ),
            "paired_delta_otfs_minus_oracle_ofdm": summarize_paired_delta(
                per_method_seeds["burst_symp_otfs"], per_method_seeds["oracle_genie_csi_ofdm"]
            ),
        }
        results["conditions"][cond_name] = cond_summary
        if cond_name in ("burst12_kd010_snr14", "burst16_kd010_snr14_ood"):
            results["cross_switch_2x2"][cond_name] = {m: summarize_seeds(cross_seeds[m]) for m in cross_seeds}
            results["last_step_audit"][cond_name] = {
                "methods": {m: summarize_seeds(last_seeds[m]) for m in last_seeds},
                "paired_delta_last_otfs_minus_ofdm2d": summarize_paired_delta(
                    last_seeds["burst_symp_otfs_last"], last_seeds["burst_symp_ofdm_2d_last"]
                ),
            }
        print(
            f"[Eval-Dir6A:{split_name}:{cond_name}] OTFS={cond_summary['methods']['burst_symp_otfs']['psnr_db']['mean']:.3f} dB vs OFDM2D={cond_summary['methods']['burst_symp_ofdm_2d']['psnr_db']['mean']:.3f} dB (Delta={cond_summary['paired_delta_otfs_minus_ofdm2d']['psnr_db']['mean']:+.3f} dB)",
            flush=True,
        )

    return results


@torch.inference_mode()
def evaluate_suite_dir6b(
    split_name: str,
    seeds: list[int],
    ckpt_root: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict:
    imgs_u8 = load_split_u8(split_name, allow_holdout=True).images_u8
    N = imgs_u8.shape[0]
    ref_dino = evaluator.extract_ref_dino(imgs_u8)

    vae, var_d16 = load_frozen_vae_and_var_d16(device=device, load_var=True)
    assert var_d16 is not None
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    # Precompute frozen f_enc and Swin stems for split
    fenc_list, stem_list = [], []
    enc = backbone.encoder
    for s in range(0, N, 32):
        x01 = imgs_u8[s : s + 32].to(device=device, dtype=torch.float32) / 255.0
        fenc_list.append(vae.quant_conv(vae.encoder(x01 * 2.0 - 1.0)).cpu())
        xs = enc.patch_embed(x01)
        for layer in enc.layers:
            xs = layer(xs)
        stem_list.append(enc.norm(xs).cpu())
    fenc_all = torch.cat(fenc_list, dim=0)
    stems_all = torch.cat(stem_list, dim=0)

    conditions = [
        ("rayleigh_snr4", 4.0, False),
        ("rayleigh_snr8", 8.0, False),
        ("rayleigh_snr13", 13.0, False),
        ("erasure50_snr10_ood", 10.0, True),
    ]

    out_all: dict = {"split": split_name, "num_images": N, "by_cbr": {}}

    for cbr_d in (512, 256):
        k_sym, nb, _, _ = ultralow_dir6_geometry(cbr_d)
        bv_models, bv_last, sw_holo, sw_genie = {}, {}, {}, {}
        for sd in seeds:
            m_bv = BayesVARTransceiver(vae, var_d16, cbr_denom=cbr_d).to(device)
            ck = torch.load(ckpt_root / f"bayes_var_cbr{cbr_d}" / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            m_bv.load_state_dict(ck["model_state"], strict=False)
            m_bv.eval()
            bv_models[sd] = m_bv

            m_bvl = BayesVARTransceiver(vae, var_d16, cbr_denom=cbr_d).to(device)
            ck_l = torch.load(ckpt_root / f"bayes_var_cbr{cbr_d}" / f"seed_{sd}" / "last.pt", map_location="cpu", weights_only=True)
            m_bvl.load_state_dict(ck_l["model_state"], strict=False)
            m_bvl.eval()
            bv_last[sd] = m_bvl

            m_sh = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=cbr_d, method="swin_holo_perc").to(device)
            ck_sh = torch.load(ckpt_root / f"swin_holo_perc_cbr{cbr_d}" / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            m_sh.load_state_dict(ck_sh["model_state"], strict=False)
            m_sh.eval()
            sw_holo[sd] = m_sh

            m_sg = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=cbr_d, method="swin_genie_perc").to(device)
            ck_sg = torch.load(ckpt_root / f"swin_genie_perc_cbr{cbr_d}" / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            m_sg.load_state_dict(ck_sg["model_state"], strict=False)
            m_sg.eval()
            sw_genie[sd] = m_sg

        cbr_res = {"k_complex_symbols": k_sym, "cbr_label": f"1/{cbr_d}", "conditions": {}, "last_step_audit": {}}

        for cond_name, snr_v, is_erasure in conditions:
            m_names = ("bayes_var_ar_on", "bayes_var_ar_off", "swin_holo_perc", "swin_genie_perc")
            seed_collect = {m: [] for m in m_names}
            last_collect = {"bayes_var_ar_on_last": [], "bayes_var_ar_off_last": []}

            for sd in seeds:
                gen = torch.Generator(device="cpu").manual_seed(720000 + sd * 37 + cbr_d + int(snr_v * 10) + (99 if is_erasure else 0))
                hr, hi = sample_correlated_rbs(N, 8, 0.85, device="cpu", dtype=torch.float32, generator=gen)
                h_ovr = torch.stack([hr, hi], dim=-1)
                eps = torch.randn((N, 8, nb, 2), generator=gen)
                if is_erasure:
                    amask = torch.ones((N, 8), dtype=torch.float32)
                    amask[:, 4:] = 0.0  # Contiguous 50% half-band deep fade / jammer erasure!
                else:
                    amask = torch.ones((N, 8), dtype=torch.float32)

                acc = {m: {k: [] for k in METRIC_KEYS} for m in m_names}
                acc_l = {m: {k: [] for k in METRIC_KEYS} for m in last_collect}

                bs = 20
                for st in range(0, N, bs):
                    en = min(N, st + bs)
                    tgt = imgs_u8[st:en].to(device=device, dtype=torch.float32) / 255.0
                    rd = ref_dino[st:en]
                    fe_b = fenc_all[st:en].to(device)
                    sb_b = stems_all[st:en].to(device)
                    h_b = h_ovr[st:en]
                    eps_b = eps[st:en]
                    am_b = amask[st:en]

                    # Single-switch evaluation on the EXACT SAME wireless forward pass!
                    out_bv = bv_models[sd].forward_from_fenc(
                        fe_b,
                        snr_v,
                        alive_rb_mask=am_b,
                        h_rb_override=h_b,
                        standard_noise_iq=eps_b,
                        use_ar=True,
                        also_return_ar_off=True,
                        seed_offset=20261123 + sd + st,
                    )
                    ev_on = evaluator.evaluate_batch(tgt, out_bv["recon"], rd)
                    ev_off = evaluator.evaluate_batch(tgt, out_bv["recon_ar_off"], rd)
                    for k in METRIC_KEYS:
                        acc["bayes_var_ar_on"][k].extend(ev_on[k])
                        acc["bayes_var_ar_off"][k].extend(ev_off[k])

                    out_sh = sw_holo[sd].forward_ultralow(
                        sb_b, snr_v, alive_rb_mask=am_b, h_rb_override=h_b, standard_noise_iq=eps_b, input_is_stem=True
                    )
                    ev_sh = evaluator.evaluate_batch(tgt, out_sh["recon"], rd)
                    out_sg = sw_genie[sd].forward_ultralow(
                        sb_b, snr_v, alive_rb_mask=am_b, h_rb_override=h_b, standard_noise_iq=eps_b, input_is_stem=True
                    )
                    ev_sg = evaluator.evaluate_batch(tgt, out_sg["recon"], rd)
                    for k in METRIC_KEYS:
                        acc["swin_holo_perc"][k].extend(ev_sh[k])
                        acc["swin_genie_perc"][k].extend(ev_sg[k])

                    if cond_name == "rayleigh_snr8":
                        out_bvl = bv_last[sd].forward_from_fenc(
                            fe_b, snr_v, alive_rb_mask=am_b, h_rb_override=h_b, standard_noise_iq=eps_b, use_ar=True, also_return_ar_off=True, seed_offset=20261123 + sd + st
                        )
                        ev_on_l = evaluator.evaluate_batch(tgt, out_bvl["recon"], rd)
                        ev_off_l = evaluator.evaluate_batch(tgt, out_bvl["recon_ar_off"], rd)
                        for k in METRIC_KEYS:
                            acc_l["bayes_var_ar_on_last"][k].extend(ev_on_l[k])
                            acc_l["bayes_var_ar_off_last"][k].extend(ev_off_l[k])

                for m in m_names:
                    seed_collect[m].append({k: float(np.mean(acc[m][k])) for k in METRIC_KEYS})
                if cond_name == "rayleigh_snr8":
                    for m in last_collect:
                        last_collect[m].append({k: float(np.mean(acc_l[m][k])) for k in METRIC_KEYS})

            csum = {
                "is_ood": is_erasure,
                "snr_db": snr_v,
                "methods": {m: summarize_seeds(seed_collect[m]) for m in m_names},
                "single_switch_ar_on_minus_ar_off": summarize_paired_delta(
                    seed_collect["bayes_var_ar_on"], seed_collect["bayes_var_ar_off"]
                ),
                "paired_delta_ar_on_minus_swin_holo_perc": summarize_paired_delta(
                    seed_collect["bayes_var_ar_on"], seed_collect["swin_holo_perc"]
                ),
                "paired_delta_ar_on_minus_swin_genie_perc": summarize_paired_delta(
                    seed_collect["bayes_var_ar_on"], seed_collect["swin_genie_perc"]
                ),
            }
            cbr_res["conditions"][cond_name] = csum
            if cond_name == "rayleigh_snr8":
                cbr_res["last_step_audit"][cond_name] = {
                    m: summarize_seeds(last_collect[m]) for m in last_collect
                }
            print(
                f"[Eval-Dir6B:{split_name}:CBR=1/{cbr_d}:{cond_name}] DISTS: AR_ON={csum['methods']['bayes_var_ar_on']['dists']['mean']:.4f} vs AR_OFF={csum['methods']['bayes_var_ar_off']['dists']['mean']:.4f} vs SwinHoloPerc={csum['methods']['swin_holo_perc']['dists']['mean']:.4f} | DINOv2: AR_ON={csum['methods']['bayes_var_ar_on']['dinov2_cos']['mean']:.4f} vs SwinHoloPerc={csum['methods']['swin_holo_perc']['dinov2_cos']['mean']:.4f}",
                flush=True,
            )

        out_all["by_cbr"][f"cbr_{cbr_d}"] = cbr_res
    return out_all


@torch.inference_mode()
def evaluate_suite_dir6c(
    split_name: str,
    seeds: list[int],
    ckpt_root: Path,
    evaluator: Full5MetricEvaluator,
    device: torch.device,
) -> dict:
    imgs_u8 = load_split_u8(split_name, allow_holdout=True).images_u8
    N = imgs_u8.shape[0]
    n_tuples = N // 4  # 125 strictly disjoint 4-tuples on common500 (500 unique images); 6 on kodak24 (24 unique images)!
    assert n_tuples * 4 == N, f"Expected N divisible by 4, got {N}"
    ref_dino = evaluator.extract_ref_dino(imgs_u8)

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    enc = backbone.encoder
    stems_list = []
    for s in range(0, N, 32):
        xb = imgs_u8[s : s + 32].to(device=device, dtype=torch.float32) / 255.0
        x = enc.patch_embed(xb)
        for layer in enc.layers:
            x = layer(x)
        stems_list.append(enc.norm(x).cpu())
    stems_all = torch.cat(stems_list, dim=0)

    models_by_seed: dict[int, dict[str, OverloadedNOMA4x2Transceiver]] = {}
    last_by_seed: dict[int, dict[str, OverloadedNOMA4x2Transceiver]] = {}
    pic_betas_by_seed: dict[int, tuple[float, float, float]] = {}

    for sd in seeds:
        models_by_seed[sd] = {}
        last_by_seed[sd] = {}
        for m_name in ("semsic_3round", "owndenoise_3round", "linear_1stage"):
            mdl = OverloadedNOMA4x2Transceiver(backbone, method=m_name).to(device)
            ck = torch.load(ckpt_root / m_name / f"seed_{sd}" / "best_indist.pt", map_location="cpu", weights_only=True)
            mdl.load_state_dict(ck["model_state"], strict=False)
            if m_name == "semsic_3round" and "pic_betas_val256" in ck:
                betas_t = tuple(float(v) for v in ck["pic_betas_val256"])
                mdl.modem.pic_betas = betas_t
                pic_betas_by_seed[sd] = betas_t
            mdl.eval()
            models_by_seed[sd][m_name] = mdl

            if m_name in ("semsic_3round", "owndenoise_3round"):
                ml = OverloadedNOMA4x2Transceiver(backbone, method=m_name).to(device)
                ck_l = torch.load(ckpt_root / m_name / f"seed_{sd}" / "last.pt", map_location="cpu", weights_only=True)
                ml.load_state_dict(ck_l["model_state"], strict=False)
                ml.eval()
                last_by_seed[sd][m_name] = ml

    conditions = [
        ("sym_4u_snr8", [8.0, 8.0, 8.0, 8.0], False, False),
        ("sym_4u_snr12", [12.0, 12.0, 12.0, 12.0], False, False),
        ("nearfar_14db", [18.0, 13.0, 8.0, 4.0], False, False),
        ("nearfar_18db_ood", [20.0, 14.0, 8.0, 2.0], True, False),
        ("sporadic_random_k1234", [17.0, 12.0, 8.0, 4.0], True, True),
    ]

    res_out = {
        "split": split_name,
        "num_unique_images": N,
        "num_disjoint_4tuples": n_tuples,
        "image_reuse_count": 0,
        "pic_betas_tuned_val256": {str(sd): list(pic_betas_by_seed.get(sd, (0.55, 0.72, 0.85))) for sd in seeds},
        "conditions": {},
        "last_step_audit": {},
    }

    all_methods = (
        "semsic_3round",
        "owndenoise_3round",
        "tuned_pic_3round",
        "linear_1stage",
        "oracle_genie_linear_mmse",
    )

    for cond_name, snr_vec, is_ood, is_sporadic in conditions:
        m_list = ("semsic_3round", "owndenoise_3round", "tuned_pic_3round", "linear_1stage") if is_sporadic else all_methods
        seed_all_u = {m: [] for m in m_list}
        seed_weak_u = {m: [] for m in m_list}
        seed_cfar = []
        last_collect = {"semsic_3round_last": [], "owndenoise_3round_last": []}

        for sd in seeds:
            gen = torch.Generator(device="cpu").manual_seed(730000 + sd * 41 + int(snr_vec[-1] * 10) + (77 if is_sporadic else 0))
            hr, hi = sample_multiuser_uplink_channels(n_tuples, 4, 2, 32, 0.85, generator=gen)
            eps = torch.randn((n_tuples, 2, 32, N_B_4X2, 2), generator=gen)
            snr_mat = torch.tensor(snr_vec, dtype=torch.float32)[None, :].expand(n_tuples, 4).clone()

            act_mask = torch.ones((n_tuples, 4), dtype=torch.float32)
            if is_sporadic:
                for t_i in range(n_tuples):
                    k_act = (t_i % 4) + 1  # Balanced exact 1, 2, 3, 4 active users across disjoint tuples!
                    keep = torch.randperm(4, generator=gen)[:k_act]
                    row = torch.zeros(4)
                    row[keep] = 1.0
                    act_mask[t_i] = row

            acc_all = {m: {k: [] for k in METRIC_KEYS} for m in m_list}
            acc_weak = {m: {k: [] for k in METRIC_KEYS} for m in m_list}
            acc_last = {m: {k: [] for k in METRIC_KEYS} for m in last_collect}
            exact_subset_hits = []
            slot_hits = []

            bs_t = 12
            for st in range(0, n_tuples, bs_t):
                en = min(n_tuples, st + bs_t)
                u_stems = [stems_all[4 * st + u : 4 * en : 4].to(device) for u in range(4)]
                u_tgts = [imgs_u8[4 * st + u : 4 * en : 4].to(device=device, dtype=torch.float32) / 255.0 for u in range(4)]
                u_rd = [ref_dino[4 * st + u : 4 * en : 4] for u in range(4)]
                hm_b = (hr[st:en], hi[st:en])
                eps_b = eps[st:en]
                snr_b = snr_mat[st:en].to(device)
                act_b = act_mask[st:en].to(device)

                for m_name in m_list:
                    if m_name == "oracle_genie_linear_mmse":
                        recons_m = forward_genie_linear_noma_4x2(
                            backbone, u_tgts, snr_b, n_rx=2, h_mu_override=hm_b, standard_noise_iq=eps_b
                        )
                    elif m_name == "tuned_pic_3round":
                        out_m = models_by_seed[sd]["semsic_3round"].forward_noma_4x2(
                            u_stems,
                            snr_b,
                            active_user_mask=act_b,
                            h_mu_override=hm_b,
                            standard_noise_iq=eps_b,
                            input_is_stem=True,
                            override_mode="tuned_pic_3round",
                            pic_betas_override=pic_betas_by_seed.get(sd, (0.55, 0.72, 0.85)),
                        )
                        recons_m = out_m["recons"]
                    else:
                        out_m = models_by_seed[sd][m_name].forward_noma_4x2(
                            u_stems,
                            snr_b,
                            active_user_mask=act_b,
                            h_mu_override=hm_b,
                            standard_noise_iq=eps_b,
                            input_is_stem=True,
                        )
                        recons_m = out_m["recons"]
                        if m_name == "semsic_3round":
                            ahat = out_m["active_hat"]
                            exact_subset_hits.extend((ahat == act_b).all(dim=1).to(torch.float32).cpu().tolist())
                            slot_hits.extend((ahat == act_b).to(torch.float32).reshape(-1).cpu().tolist())

                    for u in range(4):
                        ev_u = evaluator.evaluate_batch(u_tgts[u], recons_m[u], u_rd[u])
                        active_flags = act_b[:, u].cpu().tolist()
                        for b_i, flag in enumerate(active_flags):
                            if flag > 0.5:
                                for k in METRIC_KEYS:
                                    acc_all[m_name][k].append(ev_u[k][b_i])
                                if u == 3:
                                    for k in METRIC_KEYS:
                                        acc_weak[m_name][k].append(ev_u[k][b_i])

                if cond_name == "nearfar_14db":
                    for l_key, m_base in [("semsic_3round_last", "semsic_3round"), ("owndenoise_3round_last", "owndenoise_3round")]:
                        ol = last_by_seed[sd][m_base].forward_noma_4x2(
                            u_stems, snr_b, active_user_mask=act_b, h_mu_override=hm_b, standard_noise_iq=eps_b, input_is_stem=True
                        )
                        for u in range(4):
                            ev_l = evaluator.evaluate_batch(u_tgts[u], ol["recons"][u], u_rd[u])
                            for k in METRIC_KEYS:
                                acc_last[l_key][k].extend(ev_l[k])

            for m in m_list:
                seed_all_u[m].append({k: float(np.mean(acc_all[m][k])) for k in METRIC_KEYS})
                if acc_weak[m]["psnr_db"]:
                    seed_weak_u[m].append({k: float(np.mean(acc_weak[m][k])) for k in METRIC_KEYS})
            seed_cfar.append({
                "exact_subset_acc": float(np.mean(exact_subset_hits)),
                "slot_acc": float(np.mean(slot_hits)),
            })
            if cond_name == "nearfar_14db":
                for m in last_collect:
                    last_collect[m].append({k: float(np.mean(acc_last[m][k])) for k in METRIC_KEYS})

        csum: dict = {
            "is_ood": is_ood,
            "is_sporadic_activity": is_sporadic,
            "user_snrs_db": snr_vec,
            "blind_cfar_detection": summarize_seeds(seed_cfar),
            "all_active_users": {m: summarize_seeds(seed_all_u[m]) for m in m_list},
            "weakest_user_slot3": {m: summarize_seeds(seed_weak_u[m]) for m in m_list if seed_weak_u[m]},
            "paired_delta_semsic_minus_owndenoise": summarize_paired_delta(
                seed_all_u["semsic_3round"], seed_all_u["owndenoise_3round"]
            ),
            "paired_delta_semsic_minus_tuned_pic": summarize_paired_delta(
                seed_all_u["semsic_3round"], seed_all_u["tuned_pic_3round"]
            ),
            "paired_delta_semsic_minus_linear1": summarize_paired_delta(
                seed_all_u["semsic_3round"], seed_all_u["linear_1stage"]
            ),
        }
        if not is_sporadic:
            csum["paired_delta_semsic_minus_oracle_linear"] = summarize_paired_delta(
                seed_all_u["semsic_3round"], seed_all_u["oracle_genie_linear_mmse"]
            )
        res_out["conditions"][cond_name] = csum
        if cond_name == "nearfar_14db":
            res_out["last_step_audit"][cond_name] = {
                "methods": {m: summarize_seeds(last_collect[m]) for m in last_collect},
                "paired_delta_last_semsic_minus_owndenoise": summarize_paired_delta(
                    last_collect["semsic_3round_last"], last_collect["owndenoise_3round_last"]
                ),
            }

        print(
            f"[Eval-Dir6C:{split_name}:{cond_name}] SemSIC={csum['all_active_users']['semsic_3round']['psnr_db']['mean']:.3f} dB vs OwnDenoise={csum['all_active_users']['owndenoise_3round']['psnr_db']['mean']:.3f} dB (Delta={csum['paired_delta_semsic_minus_owndenoise']['psnr_db']['mean']:+.3f} dB) | vs TunedPIC={csum['paired_delta_semsic_minus_tuned_pic']['psnr_db']['mean']:+.3f} dB | CFAR={csum['blind_cfar_detection']['exact_subset_acc']['mean']*100:.2f}%",
            flush=True,
        )

    return res_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=["dir6a", "dir6b", "dir6c"])
    parser.add_argument("--splits", nargs="+", default=["common500", "kodak24"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 2002, 3003])
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir6_matched_breakthroughs_v1"),
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir6_matched_breakthroughs_v1"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM.dir2-swinjscc-mismatch-v1/results/dir6_matched_breakthroughs_20261123"),
    )
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    gpu_info = admit_gpu(min_free_mib=25000)
    device = torch.device("cuda:0")
    evaluator = Full5MetricEvaluator(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for sp in args.splits:
        if args.suite == "dir6a":
            res = evaluate_suite_dir6a(sp, args.seeds, args.ckpt_root, evaluator, device)
        elif args.suite == "dir6b":
            res = evaluate_suite_dir6b(sp, args.seeds, args.ckpt_root, evaluator, device)
        else:
            res = evaluate_suite_dir6c(sp, args.seeds, args.ckpt_root, evaluator, device)
        res["gpu_info"] = gpu_info
        res["elapsed_s"] = round(time.time() - t0, 2)
        out_file = args.out_dir / f"eval_{args.suite}_{sp}.json"
        save_json(out_file, res)
        print(f"[Eval Saved] {out_file}", flush=True)

    verify_no_links([args.ckpt_root, args.tmp_root, args.out_dir])


if __name__ == "__main__":
    main()
