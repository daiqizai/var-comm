#!/usr/bin/env python3
"""Unified 4-GPU Training Driver for Direction 6 (`codex/dir6-matched-breakthroughs-v1`):
GPT-Audit-Grade Matched-Control Breakthroughs across 3 Wireless Regimes:
  - Track `dir6a`:      Trains `burst_symp_otfs` (`use_otfs=True`) & `burst_symp_ofdm_2d` (`use_otfs=False`)
                        with BYTE-IDENTICAL initialization, IDENTICAL mini-batch & channel sequence, and
                        PURE IN-DOMAIN `val256` selection + fixed final step `last.pt`.
  - Track `dir6b_var`:  Trains `bayes_var` (`CBR = 1/512` [K=384] & `CBR = 1/256` [K=768]) with Stage-1 latent+ResNet50
                        distillation and Stage-2 perceptual (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`) loss.
  - Track `dir6b_swin`: Trains `swin_holo_perc` & `swin_genie_perc` (`CBR = 1/512` & `1/256`) with the EXACT SAME
                        `8x8` pyramid geometry and EXACT SAME perceptual (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`) loss!
  - Track `dir6c`:      Trains `semsic_3round` & `owndenoise_3round` (and `linear_1stage`) on 200% Overloaded 4x2
                        Grant-Free Uplink with BYTE-IDENTICAL initialization, IDENTICAL 4-user tuple sequence, and
                        EXACT LOSS PARITY + validation-only tuning of `tuned_pic_3round` damping!
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
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

from dir2_zpac.datasets import BASELINE_DATA_ROOT, load_split_u8
from dir2_zpac.hygiene import (
    admit_gpu,
    configure_runtime_env,
    configure_torch_determinism,
    save_json,
    seed_everything,
    sha256_file,
    verify_no_links,
)
from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir4_rn.noma_semsic_blind import sample_multiuser_uplink_channels
from dir5_og.otfs_blind import N_SC, N_SYM, sample_doubly_selective_multipath

from dir6_mb import (
    BayesVARTransceiver,
    BurstMatchedTransceiver,
    OverloadedNOMA4x2Transceiver,
    PerceptualSwinUltraLowTransceiver,
    load_frozen_vae_and_var_d16,
    sample_blockage_envelope,
    tune_pic_damping_val256,
    ultralow_dir6_geometry,
)
from dir6_mb.overloaded_noma_4x2 import N_B_4X2

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)
DIR5_CACHE_DIR = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/cache")


@torch.inference_mode()
def ensure_caches(cache_dir: Path, device: torch.device) -> tuple[Path, Path, Path]:
    """Reuse pre-extracted Swin stems & VAR latents under liulu PFS (or build if missing) + precache ResNet50 top-8 teacher."""
    swin_cache = DIR5_CACHE_DIR / "swin_stems_fp16.pt"
    var_cache = DIR5_CACHE_DIR / "var_latents_fp16.pt"
    if not swin_cache.exists():
        swin_cache = cache_dir / "swin_stems_fp16.pt"
    if not var_cache.exists():
        var_cache = cache_dir / "var_latents_fp16.pt"

    if not swin_cache.exists():
        backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
        enc = backbone.encoder

        def _ext_swin(imgs_u8: torch.Tensor, bs: int = 64) -> torch.Tensor:
            outs = []
            for s in range(0, imgs_u8.shape[0], bs):
                xb = imgs_u8[s : s + bs].to(device=device, dtype=torch.float32) / 255.0
                x = enc.patch_embed(xb)
                for layer in enc.layers:
                    x = layer(x)
                outs.append(enc.norm(x).to(torch.float16).cpu())
            return torch.cat(outs, dim=0)

        tr_stems = _ext_swin(load_split_u8("train20k").images_u8)
        va_stems = _ext_swin(load_split_u8("val256").images_u8).to(torch.float32)
        swin_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train_stems_fp16": tr_stems, "val_stems_fp32": va_stems}, swin_cache)

    if not var_cache.exists():
        vae, _ = load_frozen_vae_and_var_d16(device=device, load_var=False)

        def _ext_var(imgs_u8: torch.Tensor, bs: int = 64, dt: torch.dtype = torch.float16):
            fe_l, fvq_l = [], []
            for s in range(0, imgs_u8.shape[0], bs):
                x01 = imgs_u8[s : s + bs].to(device=device, dtype=torch.float32) / 255.0
                fe = vae.quant_conv(vae.encoder(x01 * 2.0 - 1.0))
                fvq = vae.quantize.f_to_idxBl_or_fhat(fe, to_fhat=True)[-1]
                fe_l.append(fe.to(dt).cpu())
                fvq_l.append(fvq.to(dt).cpu())
            return torch.cat(fe_l, dim=0), torch.cat(fvq_l, dim=0)

        tr_fe, tr_fvq = _ext_var(load_split_u8("train20k").images_u8, dt=torch.float16)
        va_fe, va_fvq = _ext_var(load_split_u8("val256").images_u8, dt=torch.float32)
        var_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"train_f_enc_fp16": tr_fe, "train_f_vq_fp16": tr_fvq, "val_f_enc_fp32": va_fe, "val_f_vq_fp32": va_fvq},
            var_cache,
        )

    cls_cache = cache_dir / "train20k_labels_i64.pt"
    if not cls_cache.exists():
        t0 = time.time()
        shards = sorted((BASELINE_DATA_ROOT / "inputs/rgb/train").glob("shard_*.pt"))
        lbls = [torch.load(sp, map_location="cpu", weights_only=True)["labels"].to(torch.int64) for sp in shards]
        cls_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"labels_i64": torch.cat(lbls, dim=0)}, cls_cache)
        print(f"[Precache-Labels] Saved {cls_cache} in {time.time() - t0:.1f}s", flush=True)

    return swin_cache, var_cache, cls_cache


def perceptual_pixel_loss(recon: torch.Tensor, target: torch.Tensor, lpips_net: torch.nn.Module) -> tuple[torch.Tensor, float]:
    """Exact shared perceptual objective (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`) used identically across Dir-6B models!"""
    rec_c = recon.clamp(0.0, 1.0)
    l1 = F.l1_loss(rec_c, target)
    mse = F.mse_loss(rec_c, target)
    lp = lpips_net(rec_c * 2.0 - 1.0, target * 2.0 - 1.0).mean()
    tot = 1.0 * l1 + 0.4 * mse + 0.25 * lp
    return tot, float(lp.item())


@torch.inference_mode()
def eval_dir6a_indomain_val256(
    model: BurstMatchedTransceiver,
    val_stems: torch.Tensor,
    val_u8: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Pure in-domain `val256` validation (ONLY inside training support: kd=0.10, blk in {8, 12}, SNR in {10, 14} dB!)."""
    model.eval()
    N = val_stems.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(6101)
    mp = sample_doubly_selective_multipath(N, 0.10, generator=gen)
    eps = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
    blk8 = sample_blockage_envelope(N, 8, fixed_start=18, device="cpu")
    blk12 = sample_blockage_envelope(N, 12, fixed_start=16, device="cpu")

    scores = {}
    for tag, snr_v, blk in [("snr10_blk8", 10.0, blk8), ("snr14_blk12", 14.0, blk12)]:
        psnrs = []
        for s in range(0, N, 32):
            e = min(N, s + 32)
            out = model.forward_burst(
                val_stems[s:e].to(device),
                snr_v,
                kd_max=0.10,
                blockage_env=blk[s:e].to(device),
                multipath_override=(mp[0][s:e], mp[1][s:e]),
                standard_noise_td_iq=eps[s:e],
                input_is_stem=True,
            )
            tgt = val_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            psnrs.extend(compute_psnr_batch(tgt, out["recon"].clamp(0.0, 1.0)).cpu().tolist())
        scores[tag] = float(sum(psnrs) / len(psnrs))
    scores["val_indomain_psnr"] = 0.5 * (scores["snr10_blk8"] + scores["snr14_blk12"])
    return scores


def train_track_dir6a(
    method: str,
    seed: int,
    steps: int,
    batch_size: int,
    ckpt_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> None:
    use_otfs = method == "burst_symp_otfs"
    assert method in ("burst_symp_otfs", "burst_symp_ofdm_2d")
    swin_cache, _, _ = ensure_caches(cache_dir, device)
    c_data = torch.load(swin_cache, map_location="cpu", weights_only=True)
    tr_stems = c_data["train_stems_fp16"]
    va_stems = c_data["val_stems_fp32"]
    tr_u8 = load_split_u8("train20k").images_u8
    va_u8 = load_split_u8("val256").images_u8

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    # STRICT BYTE-IDENTICAL INIT & IDENTICAL DATA/CHANNEL RNG SEQUENCE ACROSS BOTH ARMS!
    seed_everything(seed)
    model = BurstMatchedTransceiver(backbone, use_otfs=use_otfs).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1.5e-5)

    data_gen = torch.Generator(device="cpu").manual_seed(seed + 70001)
    out_dir = ckpt_dir / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "val_curve.csv"

    best_score = -1e9
    N_tr = tr_stems.shape[0]
    t0 = time.time()

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "loss", "snr10_blk8", "snr14_blk12", "val_indomain_psnr", "elapsed_s"])

        for step in range(1, steps + 1):
            model.train()
            idx = torch.randint(0, N_tr, (batch_size,), generator=data_gen)
            snr_b = torch.empty((batch_size,), dtype=torch.float32).uniform_(4.0, 18.0, generator=data_gen)
            kd_b = torch.empty((batch_size,), dtype=torch.float32).uniform_(0.02, 0.12, generator=data_gen)
            blk_choices = [0, 4, 8, 12]
            nb_pick = blk_choices[int(torch.randint(0, 4, (1,), generator=data_gen).item())]
            blk_env = sample_blockage_envelope(batch_size, nb_pick, generator=data_gen, device=device)
            mp_tr = sample_doubly_selective_multipath(batch_size, kd_b, generator=data_gen)
            eps_tr = torch.randn((batch_size, N_SYM, N_SC, 2), generator=data_gen)

            stems_b = tr_stems[idx].to(device=device, dtype=torch.float32)
            tgt_b = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0

            opt.zero_grad(set_to_none=True)
            out = model.forward_burst(
                stems_b,
                snr_b.to(device),
                kd_max=kd_b.to(device),
                blockage_env=blk_env,
                multipath_override=mp_tr,
                standard_noise_td_iq=eps_tr,
                input_is_stem=True,
            )
            loss = F.l1_loss(out["recon"], tgt_b) + 0.25 * F.mse_loss(out["recon"], tgt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step == 1 or step % 250 == 0 or step == steps:
                val_m = eval_dir6a_indomain_val256(model, va_stems, va_u8, device)
                el = time.time() - t0
                writer.writerow([step, f"{loss.item():.5f}", f"{val_m['snr10_blk8']:.3f}", f"{val_m['snr14_blk12']:.3f}", f"{val_m['val_indomain_psnr']:.3f}", f"{el:.1f}"])
                fcsv.flush()
                print(
                    f"[Dir6A:{method} s={seed}] step {step}/{steps} | loss={loss.item():.4f} | val_psnr={val_m['val_indomain_psnr']:.3f} dB ({el:.1f}s)",
                    flush=True,
                )
                ckpt_payload = {
                    "method": method,
                    "use_otfs": use_otfs,
                    "seed": seed,
                    "step": step,
                    "val_metrics": val_m,
                    "trainable_params": sum(p.numel() for p in params),
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                }
                if val_m["val_indomain_psnr"] >= best_score:
                    best_score = val_m["val_indomain_psnr"]
                    torch.save(ckpt_payload, out_dir / "best_indist.pt")
                if step == steps:
                    torch.save(ckpt_payload, out_dir / "last.pt")


@torch.inference_mode()
def eval_dir6b_indomain_val256(
    model: torch.nn.Module,
    is_var: bool,
    val_feats: torch.Tensor,
    val_u8: torch.Tensor,
    evaluator: OfflineVisualEvaluator,
    device: torch.device,
) -> dict[str, float]:
    """Pure in-domain `val256` validation on 8-RB Rayleigh (`SNR in {4, 10} dB`, NO OOD erasure!)."""
    model.eval()
    N = val_feats.shape[0]
    nb = model.n_b
    gen = torch.Generator(device="cpu").manual_seed(6202)
    hr, hi = sample_correlated_rbs(N, 8, 0.85, device="cpu", dtype=torch.float32, generator=gen)
    h_ovr = torch.stack([hr, hi], dim=-1)
    eps = torch.randn((N, 8, nb, 2), generator=gen)

    lp_all, psnr_all = [], []
    for snr_v in (4.0, 10.0):
        for s in range(0, N, 32):
            e = min(N, s + 32)
            fb = val_feats[s:e].to(device)
            if is_var:
                out = model.forward_from_fenc(
                    fb, snr_v, h_rb_override=h_ovr[s:e], standard_noise_iq=eps[s:e], use_ar=False
                )
            else:
                out = model.forward_ultralow(
                    fb, snr_v, h_rb_override=h_ovr[s:e], standard_noise_iq=eps[s:e], input_is_stem=True
                )
            tgt = val_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            m = evaluator.evaluate_batch(tgt, out["recon"].clamp(0.0, 1.0))
            lp_all.extend(m["lpips_alex"])
            psnr_all.extend(m["psnr_db"])

    mean_lp = float(sum(lp_all) / len(lp_all))
    mean_psnr = float(sum(psnr_all) / len(psnr_all))
    # Higher score is better: minimizes LPIPS while preserving structural PSNR
    score = -mean_lp + 0.012 * mean_psnr
    return {"val_lpips": mean_lp, "val_psnr": mean_psnr, "val_score": score}


def train_track_dir6b_var(
    cbr_denom: int,
    seed: int,
    s1_steps: int,
    s2_steps: int,
    ckpt_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> None:
    _, var_cache, cls_cache = ensure_caches(cache_dir, device)
    v_data = torch.load(var_cache, map_location="cpu", weights_only=True)
    tr_fe = v_data["train_f_enc_fp16"]
    tr_fvq = v_data["train_f_vq_fp16"]
    va_fe = v_data["val_f_enc_fp32"]

    c_cls = torch.load(cls_cache, map_location="cpu", weights_only=True)
    tr_labels = c_cls["labels_i64"]

    tr_u8 = load_split_u8("train20k").images_u8
    va_u8 = load_split_u8("val256").images_u8

    vae, _ = load_frozen_vae_and_var_d16(device=device, load_var=False)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)
    assert evaluator.lpips_model is not None

    seed_everything(seed + cbr_denom)
    model = BayesVARTransceiver(vae, None, cbr_denom=cbr_denom).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    data_gen = torch.Generator(device="cpu").manual_seed(seed + 81000 + cbr_denom)
    N_tr = tr_fe.shape[0]
    out_dir = ckpt_dir / f"bayes_var_cbr{cbr_denom}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _cls_distill_loss(cls_logits: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        tgt_y = tr_labels[batch_idx].to(device=device, dtype=torch.int64)
        return F.cross_entropy(cls_logits.float(), tgt_y, label_smoothing=0.10)

    # Stage 1: Fast Latent + RX Semantic Class Distillation (`decode_pixels=False`)
    opt1 = torch.optim.AdamW(params, lr=4e-4, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, s1_steps), eta_min=8e-5)
    bs1 = 64
    for step in range(1, s1_steps + 1):
        model.train()
        idx = torch.randint(0, N_tr, (bs1,), generator=data_gen)
        snr_b = torch.empty((bs1,), dtype=torch.float32).uniform_(1.0, 16.0, generator=data_gen).to(device)
        rho_v = float(torch.empty((1,)).uniform_(0.65, 0.92, generator=data_gen).item())
        fe_b = tr_fe[idx].to(device=device, dtype=torch.float32)
        fvq_b = tr_fvq[idx].to(device=device, dtype=torch.float32)

        opt1.zero_grad(set_to_none=True)
        out = model.forward_from_fenc(fe_b, snr_b, rho_f=rho_v, decode_pixels=False, use_ar=False)
        loss = (
            F.l1_loss(out["f_coarse"], fvq_b)
            + 0.5 * F.mse_loss(out["f_coarse"], fe_b)
            + 0.12 * _cls_distill_loss(out["cls_logits"], idx)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt1.step()
        sch1.step()

    # Stage 2: Exact Loss-Matched Perceptual Pixel Fine-Tuning (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`)
    opt2 = torch.optim.AdamW(params, lr=1.8e-4, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, s2_steps), eta_min=1.5e-5)
    bs2 = 16
    best_score = -1e9
    csv_path = out_dir / "val_curve.csv"

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "loss", "lpips_tr", "val_lpips", "val_psnr", "val_score", "elapsed_s"])
        for step in range(1, s2_steps + 1):
            model.train()
            idx = torch.randint(0, N_tr, (bs2,), generator=data_gen)
            snr_b = torch.empty((bs2,), dtype=torch.float32).uniform_(1.0, 16.0, generator=data_gen).to(device)
            rho_v = float(torch.empty((1,)).uniform_(0.65, 0.92, generator=data_gen).item())
            fe_b = tr_fe[idx].to(device=device, dtype=torch.float32)
            fvq_b = tr_fvq[idx].to(device=device, dtype=torch.float32)
            tgt_b = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0

            opt2.zero_grad(set_to_none=True)
            out = model.forward_from_fenc(fe_b, snr_b, rho_f=rho_v, decode_pixels=True, use_ar=False)
            p_loss, lp_v = perceptual_pixel_loss(out["recon"], tgt_b, evaluator.lpips_model)
            loss = p_loss + 0.15 * F.l1_loss(out["f_coarse"], fvq_b) + 0.08 * _cls_distill_loss(out["cls_logits"], idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt2.step()
            sch2.step()

            if step == 1 or step % 225 == 0 or step == s2_steps:
                vm = eval_dir6b_indomain_val256(model, True, va_fe, va_u8, evaluator, device)
                el = time.time() - t0
                writer.writerow([step, f"{loss.item():.5f}", f"{lp_v:.4f}", f"{vm['val_lpips']:.4f}", f"{vm['val_psnr']:.3f}", f"{vm['val_score']:.4f}", f"{el:.1f}"])
                fcsv.flush()
                print(
                    f"[Dir6B:bayes_var_cbr{cbr_denom} s={seed}] s2 {step}/{s2_steps} | val_lpips={vm['val_lpips']:.4f}, val_psnr={vm['val_psnr']:.2f} ({el:.1f}s)",
                    flush=True,
                )
                payload = {
                    "method": "bayes_var",
                    "cbr_denom": cbr_denom,
                    "seed": seed,
                    "step": step,
                    "val_metrics": vm,
                    "trainable_params": sum(p.numel() for p in params),
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("vae.")},
                }
                if vm["val_score"] >= best_score:
                    best_score = vm["val_score"]
                    torch.save(payload, out_dir / "best_indist.pt")
                if step == s2_steps:
                    torch.save(payload, out_dir / "last.pt")


def train_track_dir6b_swin(
    method: str,
    cbr_denom: int,
    seed: int,
    s1_steps: int,
    s2_steps: int,
    ckpt_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> None:
    assert method in ("swin_holo_perc", "swin_genie_perc")
    swin_cache, _, _ = ensure_caches(cache_dir, device)
    s_data = torch.load(swin_cache, map_location="cpu", weights_only=True)
    tr_stems = s_data["train_stems_fp16"]
    va_stems = s_data["val_stems_fp32"]
    tr_u8 = load_split_u8("train20k").images_u8
    va_u8 = load_split_u8("val256").images_u8

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)
    assert evaluator.lpips_model is not None

    seed_everything(seed + cbr_denom)
    model = PerceptualSwinUltraLowTransceiver(backbone, cbr_denom=cbr_denom, method=method).to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    data_gen = torch.Generator(device="cpu").manual_seed(seed + 82000 + cbr_denom)
    N_tr = tr_stems.shape[0]
    out_dir = ckpt_dir / f"{method}_cbr{cbr_denom}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Stage 1: Pixel Warmup (`1.0*L1 + 0.4*MSE`)
    opt1 = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, s1_steps), eta_min=8e-5)
    bs1 = 24
    for step in range(1, s1_steps + 1):
        model.train()
        idx = torch.randint(0, N_tr, (bs1,), generator=data_gen)
        snr_b = torch.empty((bs1,), dtype=torch.float32).uniform_(1.0, 16.0, generator=data_gen).to(device)
        rho_v = float(torch.empty((1,)).uniform_(0.65, 0.92, generator=data_gen).item())
        sb = tr_stems[idx].to(device=device, dtype=torch.float32)
        tb = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0

        opt1.zero_grad(set_to_none=True)
        out = model.forward_ultralow(sb, snr_b, rho_f=rho_v, input_is_stem=True)
        loss = F.l1_loss(out["recon"], tb) + 0.4 * F.mse_loss(out["recon"], tb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt1.step()
        sch1.step()

    # Stage 2: EXACT SAME Perceptual Objective (`1.0*L1 + 0.4*MSE + 0.25*LPIPS_Alex`) & step budget as BayesVAR!
    opt2 = torch.optim.AdamW(params, lr=1.8e-4, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, s2_steps), eta_min=1.5e-5)
    bs2 = 16
    best_score = -1e9
    csv_path = out_dir / "val_curve.csv"

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "loss", "lpips_tr", "val_lpips", "val_psnr", "val_score", "elapsed_s"])
        for step in range(1, s2_steps + 1):
            model.train()
            idx = torch.randint(0, N_tr, (bs2,), generator=data_gen)
            snr_b = torch.empty((bs2,), dtype=torch.float32).uniform_(1.0, 16.0, generator=data_gen).to(device)
            rho_v = float(torch.empty((1,)).uniform_(0.65, 0.92, generator=data_gen).item())
            sb = tr_stems[idx].to(device=device, dtype=torch.float32)
            tb = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0

            opt2.zero_grad(set_to_none=True)
            out = model.forward_ultralow(sb, snr_b, rho_f=rho_v, input_is_stem=True)
            loss, lp_v = perceptual_pixel_loss(out["recon"], tb, evaluator.lpips_model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt2.step()
            sch2.step()

            if step == 1 or step % 225 == 0 or step == s2_steps:
                vm = eval_dir6b_indomain_val256(model, False, va_stems, va_u8, evaluator, device)
                el = time.time() - t0
                writer.writerow([step, f"{loss.item():.5f}", f"{lp_v:.4f}", f"{vm['val_lpips']:.4f}", f"{vm['val_psnr']:.3f}", f"{vm['val_score']:.4f}", f"{el:.1f}"])
                fcsv.flush()
                print(
                    f"[Dir6B:{method}_cbr{cbr_denom} s={seed}] s2 {step}/{s2_steps} | val_lpips={vm['val_lpips']:.4f}, val_psnr={vm['val_psnr']:.2f} ({el:.1f}s)",
                    flush=True,
                )
                payload = {
                    "method": method,
                    "cbr_denom": cbr_denom,
                    "seed": seed,
                    "step": step,
                    "val_metrics": vm,
                    "trainable_params": sum(p.numel() for p in params),
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                }
                if vm["val_score"] >= best_score:
                    best_score = vm["val_score"]
                    torch.save(payload, out_dir / "best_indist.pt")
                if step == s2_steps:
                    torch.save(payload, out_dir / "last.pt")


@torch.inference_mode()
def eval_dir6c_indomain_val256(
    model: OverloadedNOMA4x2Transceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Pure in-domain `val256` validation on 64 strictly disjoint 4-image tuples (`256 / 4 = 64`, 0 image reuse!)."""
    model.eval()
    n_tup = val_stems.shape[0] // 4
    gen = torch.Generator(device="cpu").manual_seed(6303)
    hr, hi = sample_multiuser_uplink_channels(n_tup, 4, model.n_rx, model.num_rbs, 0.85, generator=gen)
    eps = torch.randn((n_tup, model.n_rx, model.num_rbs, N_B_4X2, 2), generator=gen)

    snr_sym = torch.full((n_tup, 4), 10.0, dtype=torch.float32)
    snr_nf = torch.tensor([18.0, 13.0, 8.0, 4.0], dtype=torch.float32)[None, :].expand(n_tup, 4)

    res = {}
    for tag, snr_mat in [("sym10", snr_sym), ("nf_18_13_8_4", snr_nf)]:
        psnrs = []
        for s in range(0, n_tup, 16):
            e = min(n_tup, s + 16)
            u_stems = [val_stems[4 * s + u : 4 * e : 4].to(device) for u in range(4)]
            out = model.forward_noma_4x2(
                u_stems,
                snr_mat[s:e].to(device),
                h_mu_override=(hr[s:e], hi[s:e]),
                standard_noise_iq=eps[s:e],
                input_is_stem=True,
            )
            for u in range(4):
                tgt_u = val_targets[4 * s + u : 4 * e : 4].to(device)
                psnrs.extend(compute_psnr_batch(tgt_u, out["recons"][u].clamp(0.0, 1.0)).cpu().tolist())
        res[tag] = float(sum(psnrs) / len(psnrs))
    res["val_indomain_psnr"] = 0.5 * (res["sym10"] + res["nf_18_13_8_4"])
    return res


def train_track_dir6c(
    method: str,
    seed: int,
    steps: int,
    batch_size: int,
    ckpt_dir: Path,
    cache_dir: Path,
    device: torch.device,
) -> None:
    assert method in ("semsic_3round", "owndenoise_3round", "linear_1stage")
    swin_cache, _, _ = ensure_caches(cache_dir, device)
    s_data = torch.load(swin_cache, map_location="cpu", weights_only=True)
    tr_stems = s_data["train_stems_fp16"]
    va_stems = s_data["val_stems_fp32"]
    tr_u8 = load_split_u8("train20k").images_u8
    va_targets = load_split_u8("val256").images_u8.to(torch.float32) / 255.0

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)

    # STRICT BYTE-IDENTICAL INIT & IDENTICAL 4-TUPLE SEQUENCE BETWEEN `semsic_3round` AND `owndenoise_3round`!
    seed_everything(seed)
    model = OverloadedNOMA4x2Transceiver(backbone, method=method).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1.5e-5)

    data_gen = torch.Generator(device="cpu").manual_seed(seed + 90001)
    N_tr = tr_stems.shape[0]
    out_dir = ckpt_dir / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "val_curve.csv"

    best_score = -1e9
    t0 = time.time()

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["step", "loss", "sym10", "nf_18_13_8_4", "val_indomain_psnr", "elapsed_s"])

        for step in range(1, steps + 1):
            model.train()
            # Sample `batch_size * 4` distinct images per step via random permutation slice
            perm_4b = torch.randperm(N_tr, generator=data_gen)[: batch_size * 4].reshape(4, batch_size)
            u_stems = [tr_stems[perm_4b[u]].to(device=device, dtype=torch.float32) for u in range(4)]
            u_tgts = [tr_u8[perm_4b[u]].to(device=device, dtype=torch.float32) / 255.0 for u in range(4)]

            # 50% Near-Far spread, 50% general uniform spread; 80% full 4-user overload, 20% random 2..3 active subset
            is_nf = bool(torch.rand((), generator=data_gen).item() < 0.55)
            if is_nf:
                base_nf = torch.stack([
                    torch.empty((batch_size,)).uniform_(16.0, 20.0, generator=data_gen),
                    torch.empty((batch_size,)).uniform_(11.0, 15.0, generator=data_gen),
                    torch.empty((batch_size,)).uniform_(6.0, 10.0, generator=data_gen),
                    torch.empty((batch_size,)).uniform_(2.5, 6.0, generator=data_gen),
                ], dim=1)
                # Randomly permute user slots so slot order carries zero power-ranking bias!
                slot_perm = torch.argsort(torch.rand((batch_size, 4), generator=data_gen), dim=1)
                snr_mat = torch.gather(base_nf, 1, slot_perm)
            else:
                snr_mat = torch.empty((batch_size, 4)).uniform_(4.0, 16.0, generator=data_gen)

            act_mask = torch.ones((batch_size, 4), dtype=torch.float32)
            if torch.rand((), generator=data_gen).item() < 0.20:
                for b_i in range(batch_size):
                    k_act = int(torch.randint(2, 5, (1,), generator=data_gen).item())
                    keep = torch.randperm(4, generator=data_gen)[:k_act]
                    m_row = torch.zeros(4)
                    m_row[keep] = 1.0
                    act_mask[b_i] = m_row

            hr_tr, hi_tr = sample_multiuser_uplink_channels(batch_size, 4, model.n_rx, model.num_rbs, 0.85, generator=data_gen)
            eps_tr = torch.randn((batch_size, model.n_rx, model.num_rbs, N_B_4X2, 2), generator=data_gen)

            opt.zero_grad(set_to_none=True)
            out = model.forward_noma_4x2(
                u_stems,
                snr_mat.to(device),
                active_user_mask=act_mask.to(device),
                h_mu_override=(hr_tr, hi_tr),
                standard_noise_iq=eps_tr,
                input_is_stem=True,
            )

            act_dev = act_mask.to(device)
            tot_act = act_dev.sum().clamp_min(1.0)
            pix_terms = []
            for u in range(4):
                w_u = act_dev[:, u]
                per_sample_l1 = (out["recons"][u] - u_tgts[u]).abs().mean(dim=(1, 2, 3))
                per_sample_mse = (out["recons"][u] - u_tgts[u]).square().mean(dim=(1, 2, 3))
                pix_terms.append(((per_sample_l1 + 0.25 * per_sample_mse) * w_u).sum())
            loss = sum(pix_terms) / tot_act

            if len(out["z_denoised_users"]) > 0:
                n_rounds_z = len(out["z_denoised_users"]) // 4
                lat_terms = []
                for idx_z, z_den in enumerate(out["z_denoised_users"]):
                    u = idx_z % 4
                    w_u = act_dev[:, u]
                    lat_mse = (z_den - out["z_tx_users"][u].detach()).square().mean(dim=(1, 2))
                    lat_terms.append((lat_mse * w_u).sum())
                loss = loss + 0.10 * (sum(lat_terms) / (tot_act * float(max(1, n_rounds_z))))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step == 1 or step % 250 == 0 or step == steps:
                vm = eval_dir6c_indomain_val256(model, va_stems, va_targets, device)
                el = time.time() - t0
                writer.writerow([step, f"{loss.item():.5f}", f"{vm['sym10']:.3f}", f"{vm['nf_18_13_8_4']:.3f}", f"{vm['val_indomain_psnr']:.3f}", f"{el:.1f}"])
                fcsv.flush()
                print(
                    f"[Dir6C:{method} s={seed}] step {step}/{steps} | loss={loss.item():.4f} | val_psnr={vm['val_indomain_psnr']:.3f} dB ({el:.1f}s)",
                    flush=True,
                )
                pic_betas, pic_val_psnr = (model.modem.pic_betas, 0.0)
                if method == "semsic_3round" and (vm["val_indomain_psnr"] >= best_score or step == steps):
                    pic_betas, pic_val_psnr = tune_pic_damping_val256(model, va_stems, va_targets, device)

                payload = {
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "val_metrics": vm,
                    "pic_betas_val256": list(pic_betas),
                    "pic_val256_psnr": pic_val_psnr,
                    "trainable_params": sum(p.numel() for p in params),
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                }
                if vm["val_indomain_psnr"] >= best_score:
                    best_score = vm["val_indomain_psnr"]
                    torch.save(payload, out_dir / "best_indist.pt")
                if step == steps:
                    torch.save(payload, out_dir / "last.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["precache", "dir6a", "dir6b_var", "dir6b_swin", "dir6c"])
    parser.add_argument("--method", default="burst_symp_otfs")
    parser.add_argument("--cbr-denom", type=int, default=512, choices=[512, 256])
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--s1-steps", type=int, default=1200)
    parser.add_argument("--s2-steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=24)
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
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    gpu_info = admit_gpu(min_free_mib=25000)
    device = torch.device("cuda:0")
    cache_dir = args.tmp_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"[Dir-6 Train] track={args.track} method={args.method} seed={args.seed} gpu={gpu_info}", flush=True)

    if args.track == "precache":
        sw, va, cl = ensure_caches(cache_dir, device)
        save_json(
            cache_dir / "precache_manifest.json",
            {"swin_cache": str(sw), "var_cache": str(va), "cls_cache": str(cl), "cls_sha256": sha256_file(cl)},
        )
    elif args.track == "dir6a":
        train_track_dir6a(args.method, args.seed, args.steps, args.batch_size, args.ckpt_root, cache_dir, device)
    elif args.track == "dir6b_var":
        train_track_dir6b_var(args.cbr_denom, args.seed, args.s1_steps, args.s2_steps, args.ckpt_root, cache_dir, device)
    elif args.track == "dir6b_swin":
        train_track_dir6b_swin(args.method, args.cbr_denom, args.seed, args.s1_steps, args.s2_steps, args.ckpt_root, cache_dir, device)
    elif args.track == "dir6c":
        train_track_dir6c(args.method, args.seed, args.steps, args.batch_size, args.ckpt_root, cache_dir, device)

    verify_no_links([args.ckpt_root, args.tmp_root])


if __name__ == "__main__":
    main()
