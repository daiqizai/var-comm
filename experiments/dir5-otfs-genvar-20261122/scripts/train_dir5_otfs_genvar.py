#!/usr/bin/env python3
"""Direction 5 Unified Pre-Caching & Training Engine:
  - Mode `precache_swin`: Extracts frozen SwinJSCC encoder stems for train20k + val256 into liulu cache (.pt)
  - Mode `precache_var`:  Extracts frozen VAR VQ-VAE (f_enc, f_vq) latents for train20k + val256 into liulu cache (.pt)
  - Track `otfs`:         Trains Symp_OTFS_JSCC, Symp_OFDM_2DSubspace, or BlackBox_OTFS_Blind on Doubly-Selective Channels
  - Track `genvar`:       2-Stage Latent Manifold + End-to-End Frozen VQVAE/LPIPS Training for GenHolo_VAR_JSCC & BlackBox_VAR_Blind
  - Track `swin_ul`:      Trains Swin_Holo_Blind & Swin_Genie_CSI at Ultra-Low CBR = 1/128 (K=1536) & CBR = 1/256 (K=768)
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
import torch.nn.functional as F

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
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.metrics import OfflineVisualEvaluator, compute_psnr_batch

from dir5_og.otfs_blind import (
    N_SC,
    N_SYM,
    SympOTFSBlindTransceiver,
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

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)


@torch.inference_mode()
def run_precache_swin(cache_file: Path, device: torch.device) -> None:
    if cache_file.exists():
        print(f"[Precache-Swin] Already exists: {cache_file}", flush=True)
        return
    t0 = time.time()
    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    enc = backbone.encoder

    def _extract(imgs_u8: torch.Tensor, bs: int = 64) -> torch.Tensor:
        outs = []
        N = imgs_u8.shape[0]
        for s in range(0, N, bs):
            e = min(N, s + bs)
            xb = imgs_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            x = enc.patch_embed(xb)
            for layer in enc.layers:
                x = layer(x)
            outs.append(enc.norm(x).to(torch.float16).cpu())
        return torch.cat(outs, dim=0)

    tr_split = load_split_u8("train20k")
    va_split = load_split_u8("val256")
    tr_stems = _extract(tr_split.images_u8, bs=64)
    va_stems = _extract(va_split.images_u8, bs=64).to(torch.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_stems_fp16": tr_stems, "val_stems_fp32": va_stems}, cache_file)
    print(f"[Precache-Swin] Saved {cache_file} in {time.time() - t0:.1f}s | shape={tuple(tr_stems.shape)}", flush=True)


@torch.inference_mode()
def run_precache_var(cache_file: Path, device: torch.device) -> None:
    if cache_file.exists():
        print(f"[Precache-VAR] Already exists: {cache_file}", flush=True)
        return
    t0 = time.time()
    vae = load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT, device=device)

    def _extract(imgs_u8: torch.Tensor, bs: int = 64, out_dtype: torch.dtype = torch.float16):
        fe_list, fvq_list = [], []
        N = imgs_u8.shape[0]
        for s in range(0, N, bs):
            e = min(N, s + bs)
            x01 = imgs_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            f_enc = vae.quant_conv(vae.encoder(x01 * 2.0 - 1.0))
            f_vq = vae.quantize.f_to_idxBl_or_fhat(f_enc, to_fhat=True)[-1]
            fe_list.append(f_enc.to(out_dtype).cpu())
            fvq_list.append(f_vq.to(out_dtype).cpu())
        return torch.cat(fe_list, dim=0), torch.cat(fvq_list, dim=0)

    tr_split = load_split_u8("train20k")
    va_split = load_split_u8("val256")
    tr_fe, tr_fvq = _extract(tr_split.images_u8, bs=64, out_dtype=torch.float16)
    va_fe, va_fvq = _extract(va_split.images_u8, bs=64, out_dtype=torch.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_f_enc_fp16": tr_fe,
            "train_f_vq_fp16": tr_fvq,
            "val_f_enc_fp32": va_fe,
            "val_f_vq_fp32": va_fvq,
        },
        cache_file,
    )
    print(f"[Precache-VAR] Saved {cache_file} in {time.time() - t0:.1f}s | shape={tuple(tr_fe.shape)}", flush=True)


@torch.inference_mode()
def evaluate_otfs_val256(
    model: SympOTFSBlindTransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    gen = torch.Generator(device="cpu").manual_seed(9701)
    eps_td = torch.randn((N, N_SYM, N_SC, 2), generator=gen, dtype=torch.float32)
    mp_005 = sample_doubly_selective_multipath(N, 0.05, generator=gen)
    mp_010 = sample_doubly_selective_multipath(N, 0.10, generator=gen)
    mp_018 = sample_doubly_selective_multipath(N, 0.18, generator=gen)  # OOD ~900 km/h!

    conds = [
        ("kd005_10dB", 10.0, 0.05, mp_005),
        ("kd010_6dB", 6.0, 0.10, mp_010),
        ("kd010_12dB", 12.0, 0.10, mp_010),
        ("kd018_ood_10dB", 10.0, 0.18, mp_018),
    ]
    res: dict[str, float] = {}
    for label, snr, kd, mp in conds:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_doubly_selective(
                val_stems[s:e].to(device),
                snr,
                kd_max=kd,
                multipath_override=(mp[0][s:e].to(device), mp[1][s:e].to(device)),
                standard_noise_td_iq=eps_td[s:e].to(device),
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
        res[label] = float(np.mean(vals))
    res["composite"] = 0.25 * sum(res[k] for k, _, _, _ in conds)
    return res


@torch.inference_mode()
def evaluate_genvar_val256(
    model: GenHoloVARTransceiver,
    val_f_enc: torch.Tensor,
    val_targets: torch.Tensor,
    evaluator: OfflineVisualEvaluator,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_f_enc.shape[0]
    bs = 32
    _, n_b, _ = cbr_geometry(model.cbr_denom)
    gen = torch.Generator(device="cpu").manual_seed(9811 + model.cbr_denom)
    hr, hi = sample_correlated_rbs(N, N_UL_RBS, 0.85, device="cpu", generator=gen)
    h_rb = torch.stack([hr, hi], dim=-1)
    eps = torch.randn((N, N_UL_RBS, n_b, 2), generator=gen, dtype=torch.float32)

    mask_full = torch.ones((N, N_UL_RBS), dtype=torch.float32)
    mask_ood50 = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32).expand(N, N_UL_RBS)

    conds = [
        ("ul_4dB", 4.0, mask_full),
        ("ul_10dB", 10.0, mask_full),
        ("ul_ood_erase50_10dB", 10.0, mask_ood50),
    ]
    res: dict[str, float] = {}
    comp_terms = []
    for label, snr, amask in conds:
        psnrs, dists_l, lpips_l = [], [], []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_from_fenc(
                val_f_enc[s:e].to(device),
                snr,
                alive_rb_mask=amask[s:e].to(device),
                h_rb_override=h_rb[s:e].to(device),
                standard_noise_iq=eps[s:e].to(device),
                decode_pixels=True,
            )
            m = evaluator.evaluate_batch(val_targets[s:e].to(device), out["recon"])
            psnrs.extend(m["psnr_db"])
            dists_l.extend(m["dists"])
            lpips_l.extend(m["lpips_alex"])
        p_m = float(np.mean(psnrs))
        d_m = float(np.mean(dists_l))
        l_m = float(np.mean(lpips_l))
        res[f"{label}_psnr"] = p_m
        res[f"{label}_dists"] = d_m
        res[f"{label}_lpips"] = l_m
        # Higher composite is better: balances pixel fidelity + DISTS + LPIPS!
        comp_terms.append(p_m - 25.0 * d_m - 15.0 * l_m)
    res["composite"] = float(np.mean(comp_terms))
    return res


@torch.inference_mode()
def evaluate_swin_ul_val256(
    model: SwinUltraLowTransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    _, n_b, _ = cbr_geometry(model.cbr_denom)
    gen = torch.Generator(device="cpu").manual_seed(9811 + model.cbr_denom)
    hr, hi = sample_correlated_rbs(N, N_UL_RBS, 0.85, device="cpu", generator=gen)
    h_rb = torch.stack([hr, hi], dim=-1)
    eps = torch.randn((N, N_UL_RBS, n_b, 2), generator=gen, dtype=torch.float32)

    mask_full = torch.ones((N, N_UL_RBS), dtype=torch.float32)
    mask_ood50 = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32).expand(N, N_UL_RBS)

    conds = [
        ("ul_4dB", 4.0, mask_full),
        ("ul_10dB", 10.0, mask_full),
        ("ul_ood_erase50_10dB", 10.0, mask_ood50),
    ]
    res: dict[str, float] = {}
    for label, snr, amask in conds:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_ultralow(
                val_stems[s:e].to(device),
                snr,
                alive_rb_mask=amask[s:e].to(device),
                h_rb_override=h_rb[s:e].to(device),
                standard_noise_iq=eps[s:e].to(device),
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
        res[label] = float(np.mean(vals))
    res["composite"] = float(np.mean([res[c[0]] for c in conds]))
    return res


def sample_train_alive_mask(batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """70% full 8 RBs alive; 30% random 1 or 2 RBs deep-faded/erased (strictly holding out 50% contiguous erasure for OOD!)."""
    mask = np.ones((batch_size, N_UL_RBS), dtype=np.float32)
    for b in range(batch_size):
        if rng.random() < 0.30:
            n_drop = 1 if rng.random() < 0.55 else 2
            drop_idx = rng.choice(N_UL_RBS, size=n_drop, replace=False)
            mask[b, drop_idx] = 0.0
    return torch.from_numpy(mask).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "precache_swin", "precache_var"], default="train")
    parser.add_argument("--track", choices=["otfs", "genvar", "swin_ul"], default="otfs")
    parser.add_argument(
        "--method",
        choices=[
            "symp_otfs",
            "symp_ofdm_2d",
            "blackbox_otfs",
            "genholo_var",
            "blackbox_var",
            "swin_holo_blind",
            "swin_genie_csi",
        ],
        default="symp_otfs",
    )
    parser.add_argument("--cbr-denom", type=int, choices=[128, 256], default=128)
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--total-steps", type=int, default=7000)
    parser.add_argument("--stage1-steps", type=int, default=6000)  # For genvar 2-stage training
    parser.add_argument("--stage2-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--eval-every", type=int, default=700)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/cache"),
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_cache = args.shared_cache_dir / args.run_tag
    configure_runtime_env(cache_root=run_cache, num_threads=4)
    seed_everything(args.seed)
    configure_torch_determinism()

    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")
    print(f"[Dir5:{args.run_tag}] Admitted GPU {args.physical_gpu}: {gpu_info}", flush=True)

    swin_cache = args.shared_cache_dir / "swin_stems_fp16.pt"
    var_cache = args.shared_cache_dir / "var_latents_fp16.pt"

    if args.mode == "precache_swin":
        run_precache_swin(swin_cache, device)
        verify_no_links([args.shared_cache_dir])
        return
    if args.mode == "precache_var":
        run_precache_var(var_cache, device)
        verify_no_links([args.shared_cache_dir])
        return

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr_split = load_split_u8("train20k")
    va_split = load_split_u8("val256")
    train_u8 = tr_split.images_u8  # [20000, 3, 256, 256] uint8 CPU
    val_targets = va_split.images_u8.to(torch.float32) / 255.0  # [256, 3, 256, 256] CPU
    n_train = train_u8.shape[0]
    db_factor = 10.0 / math.log(10.0)

    def save_light_ckpt(model: torch.nn.Module, path: Path, step: int, val_dict: dict[str, float]) -> None:
        keep_keys = {name for name, p in model.named_parameters() if p.requires_grad}
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep_keys}
        torch.save(
            {
                "step": step,
                "track": args.track,
                "method": args.method,
                "cbr_denom": args.cbr_denom,
                "run_tag": args.run_tag,
                "state_dict": sd,
                "val": val_dict,
            },
            path,
        )

    history: list[dict[str, float]] = []
    best_path = args.ckpt_dir / "best_dir5.pt"
    t0 = time.time()

    if args.track == "genvar":
        assert var_cache.exists(), f"Missing {var_cache} — run --mode precache_var first!"
        vc = torch.load(var_cache, map_location="cpu", weights_only=True)
        tr_fe = vc["train_f_enc_fp16"]
        tr_fvq = vc["train_f_vq_fp16"]
        va_fe = vc["val_f_enc_fp32"]

        vae = load_frozen_var_vqvae(DEFAULT_VAR_VQVAE_CKPT, device=device)
        evaluator = OfflineVisualEvaluator(device=device, include_lpips=True)
        lpips_net = evaluator.lpips_model
        assert lpips_net is not None

        model = GenHoloVARTransceiver(vae, cbr_denom=args.cbr_denom, method=args.method).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"[Dir5:{args.run_tag}] GenHoloVAR CBR=1/{args.cbr_denom} trainable params: {n_params:,}", flush=True)

        val0 = evaluate_genvar_val256(model, va_fe, val_targets, evaluator, device)
        print(f"[Dir5:{args.run_tag}] Step    0 | comp={val0['composite']:.3f} | {val0}", flush=True)
        best_val = val0
        best_step = 0
        save_light_ckpt(model, best_path, 0, val0)
        history.append({"step": 0, "stage": 0, "train_loss": 0.0, **val0})

        _, n_b, _ = cbr_geometry(args.cbr_denom)

        # Stage 1: Rapid Latent Manifold Alignment (batch = 64, decode_pixels = False)
        opt1 = torch.optim.AdamW(trainable, lr=1.0e-3, weight_decay=1e-4)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, args.stage1_steps), eta_min=1.0e-4)
        bs1 = 64
        run_loss = 0.0
        run_cnt = 0

        for step in range(1, args.stage1_steps + 1):
            model.train()
            idx_rng = np.random.default_rng(args.seed * 73331 + step)
            ch_rng = np.random.default_rng(args.seed * 1000003 + step * 89)
            ngen = torch.Generator(device="cpu").manual_seed(args.seed * 2000003 + step * 41)

            bidx = idx_rng.choice(n_train, size=bs1, replace=False)
            fe_b = tr_fe[bidx].to(device=device, dtype=torch.float32)
            fvq_b = tr_fvq[bidx].to(device=device, dtype=torch.float32)

            snr_v = torch.from_numpy(ch_rng.uniform(0.5, 14.5, size=bs1).astype(np.float32)).to(device)
            rho_f = float(ch_rng.uniform(0.75, 0.92))
            hr_b, hi_b = sample_correlated_rbs(bs1, N_UL_RBS, rho_f, device="cpu", generator=ngen)
            h_rb = torch.stack([hr_b, hi_b], dim=-1).to(device)
            eps_b = torch.randn((bs1, N_UL_RBS, n_b, 2), generator=ngen, dtype=torch.float32).to(device)
            amask_b = sample_train_alive_mask(bs1, ch_rng, device)

            out = model.forward_from_fenc(
                fe_b,
                snr_v,
                rho_f=rho_f,
                alive_rb_mask=amask_b,
                h_rb_override=h_rb,
                standard_noise_iq=eps_b,
                decode_pixels=False,
            )
            loss = (
                F.mse_loss(out["f_out"], fvq_b)
                + 0.35 * F.l1_loss(out["f_out"], fvq_b)
                + 0.30 * F.mse_loss(out["f_pre_vq"], fe_b)
            )
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.5)
            opt1.step()
            sched1.step()
            run_loss += float(loss.item())
            run_cnt += 1

            if step == args.stage1_steps or (args.stage1_steps >= 2000 and step % 2000 == 0):
                val_s1 = evaluate_genvar_val256(model, va_fe, val_targets, evaluator, device)
                avg_l = run_loss / max(1, run_cnt)
                run_loss, run_cnt = 0.0, 0
                history.append({"step": step, "stage": 1, "train_loss": avg_l, **val_s1})
                if val_s1["composite"] > best_val["composite"]:
                    best_val = val_s1
                    best_step = step
                    save_light_ckpt(model, best_path, step, val_s1)
                print(
                    f"[Dir5:{args.run_tag}] S1 Step {step:4d}/{args.stage1_steps} ({time.time()-t0:.1f}s) | "
                    f"loss={avg_l:.4f} | comp={val_s1['composite']:.3f} (best={best_val['composite']:.3f}) | "
                    f"10dB: PSNR={val_s1['ul_10dB_psnr']:.2f}, DISTS={val_s1['ul_10dB_dists']:.4f}, LPIPS={val_s1['ul_10dB_lpips']:.4f}",
                    flush=True,
                )

        # Stage 2: End-to-End Frozen VQVAE Decoder + Offline LPIPS-Alex + Pixel L1/MSE Fine-Tuning
        opt2 = torch.optim.AdamW(trainable, lr=2.5e-4, weight_decay=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.stage2_steps), eta_min=2.0e-5)
        bs2 = 16
        s2_eval_every = max(1, min(300, args.stage2_steps))

        for s2 in range(1, args.stage2_steps + 1):
            gstep = args.stage1_steps + s2
            model.train()
            idx_rng = np.random.default_rng(args.seed * 91193 + gstep)
            ch_rng = np.random.default_rng(args.seed * 1300007 + gstep * 97)
            ngen = torch.Generator(device="cpu").manual_seed(args.seed * 2500009 + gstep * 53)

            bidx = idx_rng.choice(n_train, size=bs2, replace=False)
            fe_b = tr_fe[bidx].to(device=device, dtype=torch.float32)
            fvq_b = tr_fvq[bidx].to(device=device, dtype=torch.float32)
            xb = train_u8[bidx].to(device=device, dtype=torch.float32) / 255.0

            snr_v = torch.from_numpy(ch_rng.uniform(0.5, 14.5, size=bs2).astype(np.float32)).to(device)
            rho_f = float(ch_rng.uniform(0.75, 0.92))
            hr_b, hi_b = sample_correlated_rbs(bs2, N_UL_RBS, rho_f, device="cpu", generator=ngen)
            h_rb = torch.stack([hr_b, hi_b], dim=-1).to(device)
            eps_b = torch.randn((bs2, N_UL_RBS, n_b, 2), generator=ngen, dtype=torch.float32).to(device)
            amask_b = sample_train_alive_mask(bs2, ch_rng, device)

            out = model.forward_from_fenc(
                fe_b,
                snr_v,
                rho_f=rho_f,
                alive_rb_mask=amask_b,
                h_rb_override=h_rb,
                standard_noise_iq=eps_b,
                decode_pixels=True,
            )
            rec = out["recon"]
            lp_val = lpips_net(rec * 2.0 - 1.0, xb * 2.0 - 1.0).mean()
            loss = (
                0.80 * F.mse_loss(out["f_out"], fvq_b)
                + 1.00 * F.l1_loss(rec, xb)
                + 0.40 * F.mse_loss(rec, xb)
                + 0.25 * lp_val
            )
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.2)
            opt2.step()
            sched2.step()
            run_loss += float(loss.item())
            run_cnt += 1

            if s2 % s2_eval_every == 0 or s2 == args.stage2_steps:
                val_s2 = evaluate_genvar_val256(model, va_fe, val_targets, evaluator, device)
                avg_l = run_loss / max(1, run_cnt)
                run_loss, run_cnt = 0.0, 0
                history.append({"step": gstep, "stage": 2, "train_loss": avg_l, **val_s2})
                if val_s2["composite"] > best_val["composite"]:
                    best_val = val_s2
                    best_step = gstep
                    save_light_ckpt(model, best_path, gstep, val_s2)
                print(
                    f"[Dir5:{args.run_tag}] S2 Step {s2:4d}/{args.stage2_steps} ({time.time()-t0:.1f}s) | "
                    f"loss={avg_l:.4f} | comp={val_s2['composite']:.3f} (best={best_val['composite']:.3f} @{best_step}) | "
                    f"10dB: PSNR={val_s2['ul_10dB_psnr']:.2f}, DISTS={val_s2['ul_10dB_dists']:.4f}, LPIPS={val_s2['ul_10dB_lpips']:.4f} | "
                    f"OOD50%: PSNR={val_s2['ul_ood_erase50_10dB_psnr']:.2f}, DISTS={val_s2['ul_ood_erase50_10dB_dists']:.4f}",
                    flush=True,
                )
        final_step = args.stage1_steps + args.stage2_steps
        save_light_ckpt(model, args.ckpt_dir / "last_dir5.pt", final_step, history[-1])

    else:
        assert swin_cache.exists(), f"Missing {swin_cache} — run --mode precache_swin first!"
        sc = torch.load(swin_cache, map_location="cpu", weights_only=True)
        tr_stems = sc["train_stems_fp16"]
        va_stems = sc["val_stems_fp32"]

        backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
        if args.track == "otfs":
            model = SympOTFSBlindTransceiver(backbone, method=args.method).to(device)
            val0 = evaluate_otfs_val256(model, va_stems, val_targets, device)
        else:
            model = SwinUltraLowTransceiver(backbone, cbr_denom=args.cbr_denom, method=args.method).to(device)
            val0 = evaluate_swin_ul_val256(model, va_stems, val_targets, device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"[Dir5:{args.run_tag}] {args.track}:{args.method} trainable params: {n_params:,}", flush=True)
        print(f"[Dir5:{args.run_tag}] Step    0 | comp={val0['composite']:.3f} | {val0}", flush=True)

        best_val = val0
        best_step = 0
        save_light_ckpt(model, best_path, 0, val0)
        history.append({"step": 0, "train_loss": 0.0, **val0})

        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.total_steps), eta_min=args.lr * 0.05)
        run_loss, run_cnt = 0.0, 0
        _, n_b, _ = cbr_geometry(args.cbr_denom)

        for step in range(1, args.total_steps + 1):
            model.train()
            idx_rng = np.random.default_rng(args.seed * 73331 + step)
            ch_rng = np.random.default_rng(args.seed * 1000003 + step * 89)
            ngen = torch.Generator(device="cpu").manual_seed(args.seed * 2000003 + step * 41)

            bidx = idx_rng.choice(n_train, size=args.batch_size, replace=False)
            stems_b = tr_stems[bidx].to(device=device, dtype=torch.float32)
            xb = train_u8[bidx].to(device=device, dtype=torch.float32) / 255.0

            if args.track == "otfs":
                snr_v = torch.from_numpy(ch_rng.uniform(1.0, 15.0, size=args.batch_size).astype(np.float32)).to(device)
                # Strictly hold out extreme OOD kd_max = 0.18 (~900 km/h) for zero-shot evaluation!
                kd_v = torch.from_numpy(ch_rng.uniform(0.01, 0.12, size=args.batch_size).astype(np.float32))
                mp_b = sample_doubly_selective_multipath(args.batch_size, kd_v, generator=ngen)
                eps_td = torch.randn((args.batch_size, N_SYM, N_SC, 2), generator=ngen, dtype=torch.float32).to(device)
                out = model.forward_doubly_selective(
                    stems_b,
                    snr_v,
                    kd_max=kd_v.to(device),
                    multipath_override=(mp_b[0].to(device), mp_b[1].to(device)),
                    standard_noise_td_iq=eps_td,
                    input_is_stem=True,
                )
            else:
                snr_v = torch.from_numpy(ch_rng.uniform(0.5, 14.5, size=args.batch_size).astype(np.float32)).to(device)
                rho_f = float(ch_rng.uniform(0.75, 0.92))
                hr_b, hi_b = sample_correlated_rbs(args.batch_size, N_UL_RBS, rho_f, device="cpu", generator=ngen)
                h_rb = torch.stack([hr_b, hi_b], dim=-1).to(device)
                eps_b = torch.randn((args.batch_size, N_UL_RBS, n_b, 2), generator=ngen, dtype=torch.float32).to(device)
                amask_b = sample_train_alive_mask(args.batch_size, ch_rng, device)
                out = model.forward_ultralow(
                    stems_b,
                    snr_v,
                    rho_f=rho_f,
                    alive_rb_mask=amask_b,
                    h_rb_override=h_rb,
                    standard_noise_iq=eps_b,
                    input_is_stem=True,
                )

            per_mse = (out["recon"] - xb).square().mean(dim=(1, 2, 3))
            loss = db_factor * torch.log(per_mse + 1e-6).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.2)
            opt.step()
            sched.step()
            run_loss += float(loss.item())
            run_cnt += 1

            if step % args.eval_every == 0 or step == args.total_steps:
                val_cur = (
                    evaluate_otfs_val256(model, va_stems, val_targets, device)
                    if args.track == "otfs"
                    else evaluate_swin_ul_val256(model, va_stems, val_targets, device)
                )
                avg_l = run_loss / max(1, run_cnt)
                run_loss, run_cnt = 0.0, 0
                history.append({"step": step, "train_loss": avg_l, **val_cur})
                if val_cur["composite"] > best_val["composite"]:
                    best_val = val_cur
                    best_step = step
                    save_light_ckpt(model, best_path, step, val_cur)
                print(
                    f"[Dir5:{args.run_tag}] Step {step:4d}/{args.total_steps} ({time.time()-t0:.1f}s) | "
                    f"loss={avg_l:.3f} | comp={val_cur['composite']:.3f} (best={best_val['composite']:.3f} @{best_step}) | {val_cur}",
                    flush=True,
                )
        final_step = args.total_steps
        save_light_ckpt(model, args.ckpt_dir / "last_dir5.pt", final_step, history[-1])

    csv_path = args.out_dir / f"history_{args.run_tag}.csv"
    all_fields: list[str] = []
    for row in history:
        for k in row.keys():
            if k not in all_fields:
                all_fields.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "run_tag": args.run_tag,
        "track": args.track,
        "method": args.method,
        "cbr_denom": args.cbr_denom,
        "best_step": best_step,
        "best_val": best_val,
        "best_ckpt_path": str(best_path),
        "best_ckpt_sha256": sha256_file(best_path),
        "elapsed_sec": time.time() - t0,
    }
    save_json(args.out_dir / f"summary_{args.run_tag}.json", summary)
    verify_no_links([args.ckpt_dir, args.out_dir, run_cache])
    print(f"[Dir5:{args.run_tag}] Training complete! Best step={best_step} | {best_val}", flush=True)


if __name__ == "__main__":
    main()
