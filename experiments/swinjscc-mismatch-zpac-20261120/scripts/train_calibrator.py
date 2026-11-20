#!/usr/bin/env python3
"""Stage 1 & Stage 2 Training Engine for SwinJSCC Mismatch Calibration (Checklist 06-11).

Supports strictly paired mixed-condition training across all methods:
  B0, B1, B2, P_raw, P_nogate, P_sym, P_noxi, P_pointwise, P_zpac, P_zpac_hier
with identical (image, c_T, Gamma, c_R, noise) batch sequences across methods for fair comparison.
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
from dir2_zpac.models import CalibratedSwinJSCC, encode_with_ct, load_swinjscc_backbone
from dir2_zpac.phy_channel import make_paired_noise
from dir2_zpac.metrics import compute_psnr_batch


VAL_CONDITIONS: list[tuple[str, float, float, float]] = [
    # Matched across full [1, 13] dB range
    ("matched_1", 1.0, 1.0, 1.0),
    ("matched_4", 4.0, 4.0, 4.0),
    ("matched_7", 7.0, 7.0, 7.0),
    ("matched_10", 10.0, 10.0, 10.0),
    ("matched_13", 13.0, 13.0, 13.0),
    # Seen TX Overestimation (+3, +6 dB)
    ("over_4_1_1", 4.0, 1.0, 1.0),
    ("over_7_1_1", 7.0, 1.0, 1.0),
    ("over_10_4_4", 10.0, 4.0, 4.0),
    ("over_13_7_7", 13.0, 7.0, 7.0),
    # Seen TX Underestimation (-3, -6 dB)
    ("under_1_4_4", 1.0, 4.0, 4.0),
    ("under_1_7_7", 1.0, 7.0, 7.0),
    ("under_4_10_10", 4.0, 10.0, 10.0),
    ("under_7_13_13", 7.0, 13.0, 13.0),
]


def sample_mixed_conditions(
    batch_size: int,
    step: int,
    base_seed: int,
    device: torch.device,
    *,
    cr_jitter_prob: float = 0.20,
    cr_jitter_std: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically sample 50% matched (Delta c = 0) and 50% mismatched (Delta c in {+-3, +-6} dB).

    Unseen offsets (+-2, +-4, +-8, +-9, +-12 dB) are strictly held out from training!
    """
    rng = np.random.default_rng(base_seed * 1000003 + step)
    snrs_grid = np.array([1.0, 4.0, 7.0, 10.0, 13.0], dtype=np.float32)

    # Pre-enumerate all valid mismatched pairs from grid with |c_T - Gamma| in {3, 6} dB
    valid_mis_pairs: list[tuple[float, float]] = []
    for gm in snrs_grid:
        for delta in (-6.0, -3.0, 3.0, 6.0):
            ct = gm + delta
            if 1.0 <= ct <= 13.0:
                valid_mis_pairs.append((float(ct), float(gm)))

    ct_arr = np.zeros(batch_size, dtype=np.float32)
    gm_arr = np.zeros(batch_size, dtype=np.float32)
    cr_arr = np.zeros(batch_size, dtype=np.float32)

    half = batch_size // 2
    for b in range(batch_size):
        if b < half:
            # Matched sample (70% discrete grid, 30% continuous uniform [1, 13])
            if rng.random() < 0.70:
                gm = float(snrs_grid[rng.integers(0, len(snrs_grid))])
            else:
                gm = float(rng.uniform(1.0, 13.0))
            ct = gm
            cr = gm
        else:
            # Mismatched sample with |Delta c| strictly in {3, 6} dB (never +-2, +-4, +-8, +-9, +-12!)
            if rng.random() < 0.75:
                ct, gm = valid_mis_pairs[int(rng.integers(0, len(valid_mis_pairs)))]
            else:
                delta = float(rng.choice([-6.0, -3.0, 3.0, 6.0]))
                gm_lo = max(1.0, 1.0 - delta)
                gm_hi = min(13.0, 13.0 - delta)
                gm = float(rng.uniform(gm_lo, gm_hi))
                ct = gm + delta
            cr = gm
            if rng.random() < cr_jitter_prob:
                cr = float(np.clip(gm + rng.normal(0.0, cr_jitter_std), 0.5, 14.0))

        ct_arr[b] = ct
        gm_arr[b] = gm
        cr_arr[b] = cr

    perm = rng.permutation(batch_size)
    return (
        torch.from_numpy(ct_arr[perm]).to(device),
        torch.from_numpy(gm_arr[perm]).to(device),
        torch.from_numpy(cr_arr[perm]).to(device),
    )


@torch.inference_mode()
def evaluate_validation(
    model: CalibratedSwinJSCC,
    val_images_u8: torch.Tensor,
    device: torch.device,
    rate_channels: int | None = None,
    noise_seed: int = 8101,
) -> dict:
    was_training = model.training
    model.eval()
    n_imgs = int(val_images_u8.shape[0])
    batch_sz = 32

    cond_psnrs: dict[str, float] = {}
    matched_vals = []
    over_vals = []
    under_vals = []

    for cname, ct, gm, cr in VAL_CONDITIONS:
        psnr_list: list[float] = []
        for s in range(0, n_imgs, batch_sz):
            e = min(n_imgs, s + batch_sz)
            xb = val_images_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            eps = make_paired_noise((e - s, 256, model.latent_channels), list(range(s, e)), noise_seed, device)
            out = model.forward_awgn(xb, ct, gm, cr, standard_noise=eps, rate_channels=rate_channels)
            recon = out["recon"].clamp(0.0, 1.0)
            psnr_list.extend(compute_psnr_batch(xb, recon).cpu().tolist())
        mean_p = float(np.mean(psnr_list))
        cond_psnrs[cname] = round(mean_p, 5)
        if cname.startswith("matched_"):
            matched_vals.append(mean_p)
        elif cname.startswith("over_"):
            over_vals.append(mean_p)
        elif cname.startswith("under_"):
            under_vals.append(mean_p)

    if was_training:
        model.train()
    matched_mean = float(np.mean(matched_vals))
    mis_mean = float(np.mean(over_vals + under_vals))
    # Balanced selection objective: 0.4 * matched + 0.6 * mismatched
    composite = 0.4 * matched_mean + 0.6 * mis_mean
    return {
        "composite_psnr_db": round(composite, 5),
        "matched_psnr_db": round(matched_mean, 5),
        "mismatch_psnr_db": round(mis_mean, 5),
        "over_psnr_db": round(float(np.mean(over_vals)), 5),
        "under_psnr_db": round(float(np.mean(under_vals)), 5),
        "cells": cond_psnrs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SwinJSCC Mismatch Calibrator")
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--model-mode", type=str, default="SwinJSCC_w/_SA")
    parser.add_argument("--c-bottleneck", type=int, default=96)
    parser.add_argument("--rate-channels", type=int, default=None)
    parser.add_argument("--method", type=str, required=True, choices=list(CalibratedSwinJSCC.METHODS))
    parser.add_argument("--hidden-ch", type=int, default=128)
    parser.add_argument("--open-decoder-bm", action="store_true", help="Open decoder.bm_list condition layers (Checklist 08)")
    parser.add_argument("--open-decoder-sm", action="store_true", help="Also open decoder.sm_list in ChannelModNet (full ModNet B1/B2)")
    parser.add_argument("--phase1-frozen-steps", type=int, default=0, help="Initial steps with decoder.bm_list frozen before opening")
    parser.add_argument("--latent-align-weight", type=float, default=0.0, help="Optional weight lambda_lat for oracle clean matched-latent MSE")
    parser.add_argument("--train-split", type=str, default="train20k", choices=["train20k", "dev64"])
    parser.add_argument("--max-train-shards", type=int, default=None)
    parser.add_argument("--val-split", type=str, default="val256", choices=["val256", "dev64"])
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bm-lr-scale", type=float, default=0.25, help="Lower LR multiplier on pretrained decoder.bm_list")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=None)
    args = parser.parse_args()

    configure_runtime_env()
    gpu_info = admit_gpu(args.physical_gpu, min_free_mib=24000)
    device = torch.device("cuda:0" if "CUDA_VISIBLE_DEVICES" in __import__("os").environ else f"cuda:{gpu_info['index']}")
    torch.cuda.set_device(device)
    configure_torch_determinism(num_threads=6)
    seed_everything(args.seed)

    t0 = time.time()
    backbone = load_swinjscc_backbone(
        args.backbone_ckpt,
        model_mode=args.model_mode,
        c_bottleneck=args.c_bottleneck,
        device=device,
    )
    model = CalibratedSwinJSCC(backbone, method=args.method, hidden_ch=args.hidden_ch).to(device)

    train_data = load_split_u8(args.train_split, max_train_shards=args.max_train_shards)
    val_data = load_split_u8(args.val_split)
    n_train = int(train_data.images_u8.shape[0])

    # Baseline step-0 validation before any update
    val0 = evaluate_validation(model, val_data.images_u8, device, rate_channels=args.rate_channels)
    print(f"[{args.method}] Step 0 | comp={val0['composite_psnr_db']:.3f} | match={val0['matched_psnr_db']:.3f} | mis={val0['mismatch_psnr_db']:.3f}", flush=True)

    def build_optimizer(open_bm_now: bool) -> tuple[torch.optim.Optimizer, list[str]]:
        tnames = model.configure_trainable_parameters(
            open_decoder_bm=open_bm_now, open_decoder_sm=(args.open_decoder_sm and (open_bm_now or model.dual_decoder_bm))
        )
        front_params = []
        bm_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone.decoder.bm_list" in n or "backbone.decoder.sm_list" in n:
                bm_params.append(p)
            else:
                front_params.append(p)
        groups = []
        if front_params:
            groups.append({"params": front_params, "lr": args.lr, "weight_decay": args.weight_decay})
        if bm_params:
            groups.append({"params": bm_params, "lr": args.lr * args.bm_lr_scale, "weight_decay": 0.0})
        if not groups:
            raise RuntimeError(f"Method {args.method} has 0 trainable parameters!")
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        return opt, tnames

    initial_open_bm = args.open_decoder_bm and (args.phase1_frozen_steps <= 0)
    optimizer, trainable_names = build_optimizer(initial_open_bm)
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    best_val = val0
    best_step = 0
    best_ckpt_path = args.ckpt_dir / "best_calibrator.pt"

    def save_light_ckpt(path: Path, step_num: int, val_metrics: dict) -> None:
        # Save ONLY modified/trainable state dict keys (tiny ~1-5 MB instead of duplicating 120 MB frozen backbone!)
        save_keys = set(trainable_names)
        if model.frontend is not None:
            save_keys.update(f"frontend.{k}" for k in model.frontend.state_dict().keys())
        if model.hier_delta is not None:
            save_keys.update(f"hier_delta.{k}" for k in model.hier_delta.state_dict().keys())
        if model.dual_decoder_bm or args.open_decoder_bm:
            save_keys.update(f"backbone.decoder.bm_list.{k}" for k in model.backbone.decoder.bm_list.state_dict().keys())
        if args.open_decoder_sm:
            save_keys.update(f"backbone.decoder.sm_list.{k}" for k in model.backbone.decoder.sm_list.state_dict().keys())
        full_sd = model.state_dict()
        delta_sd = {k: full_sd[k].cpu() for k in sorted(save_keys) if k in full_sd}
        payload = {
            "method": args.method,
            "hidden_ch": args.hidden_ch,
            "open_decoder_bm": args.open_decoder_bm,
            "latent_align_weight": args.latent_align_weight,
            "model_mode": args.model_mode,
            "c_bottleneck": args.c_bottleneck,
            "rate_channels": args.rate_channels,
            "step": step_num,
            "val_metrics": val_metrics,
            "delta_state_dict": delta_sd,
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        __import__("os").replace(tmp, path)

    save_light_ckpt(best_ckpt_path, 0, val0)

    history_rows: list[dict] = [
        {
            "step": 0,
            "train_loss": float("nan"),
            "lr": args.lr,
            **{k: v for k, v in val0.items() if k != "cells"},
        }
    ]

    rng_idx = np.random.default_rng(args.seed + 77)
    perm = rng_idx.permutation(n_train)
    cursor = 0

    model.train()
    running_loss = 0.0
    running_count = 0

    for step in range(1, args.total_steps + 1):
        if args.open_decoder_bm and args.phase1_frozen_steps > 0 and step == args.phase1_frozen_steps + 1:
            optimizer, trainable_names = build_optimizer(True)
            trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[{args.method}] Step {step}: Opened decoder.bm_list ({trainable_param_count} trainable params)", flush=True)

        # Cosine LR schedule with 5% linear warmup
        warmup_steps = max(50, int(0.05 * args.total_steps))
        if step <= warmup_steps:
            lr_mult = step / float(warmup_steps)
        else:
            prog = (step - warmup_steps) / float(max(1, args.total_steps - warmup_steps))
            lr_mult = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg_idx, pg in enumerate(optimizer.param_groups):
            base_group_lr = args.lr if (pg_idx == 0 and model.frontend is not None) else (
                args.lr * args.bm_lr_scale if "bm_list" in trainable_names[-1] and pg_idx > 0 else args.lr
            )
            pg["lr"] = base_group_lr * lr_mult

        if cursor + args.batch_size > n_train:
            perm = rng_idx.permutation(n_train)
            cursor = 0
        batch_indices = perm[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        xb = train_data.images_u8[batch_indices].to(device=device, dtype=torch.float32) / 255.0
        ct_vec, gm_vec, cr_vec = sample_mixed_conditions(args.batch_size, step, args.seed, device)

        noise_gen = torch.Generator(device="cpu")
        noise_gen.manual_seed(args.seed * 1000003 + step * 31)
        eps = torch.randn((args.batch_size, 256, model.latent_channels), generator=noise_gen, dtype=torch.float32).to(device)

        need_lat = args.latent_align_weight > 0.0 and model.frontend is not None
        out = model.forward_awgn(
            xb,
            ct_vec,
            gm_vec,
            cr_vec,
            standard_noise=eps,
            rate_channels=args.rate_channels,
            return_latents=need_lat,
        )
        # Note: do NOT clamp before MSE loss during training so out-of-range gradients aren't zeroed
        loss_rec = F.mse_loss(out["recon"], xb)
        if need_lat:
            with torch.no_grad():
                z_oracle, _ = encode_with_ct(model.backbone, xb, gm_vec, rate_channels=args.rate_channels)
            loss_lat = F.mse_loss(out["u_cal"], z_oracle.detach()) / (z_oracle.detach().square().mean().clamp_min(1e-6))
            loss = loss_rec + args.latent_align_weight * loss_lat
        else:
            loss = loss_rec

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        running_loss += float(loss_rec.item())
        running_count += 1

        if step % args.eval_interval == 0 or step == args.total_steps:
            avg_train_mse = running_loss / max(1, running_count)
            running_loss = 0.0
            running_count = 0

            val_res = evaluate_validation(model, val_data.images_u8, device, rate_channels=args.rate_channels)
            cur_lr = optimizer.param_groups[0]["lr"]
            is_best = val_res["composite_psnr_db"] > best_val["composite_psnr_db"]
            # Guardrail: reject checkpoints that regress matched PSNR by more than 0.08 dB vs pretrained
            if (val0["matched_psnr_db"] - val_res["matched_psnr_db"]) > 0.08 and best_step > 0:
                is_best = False
            if is_best:
                best_val = val_res
                best_step = step
                save_light_ckpt(best_ckpt_path, step, val_res)

            print(
                f"[{args.method}] Step {step:4d}/{args.total_steps} | train_mse={avg_train_mse:.6f} | "
                f"comp={val_res['composite_psnr_db']:.3f} | match={val_res['matched_psnr_db']:.3f} | "
                f"over={val_res['over_psnr_db']:.3f} | under={val_res['under_psnr_db']:.3f}"
                + (" *BEST*" if is_best else ""),
                flush=True,
            )
            history_rows.append(
                {
                    "step": step,
                    "train_loss": round(avg_train_mse, 7),
                    "lr": round(cur_lr, 8),
                    **{k: v for k, v in val_res.items() if k != "cells"},
                }
            )

    save_light_ckpt(args.ckpt_dir / "last_calibrator.pt", args.total_steps, val_res)

    hist_csv = args.summary_dir / f"train_history_{args.method}.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "schema": "DIR2_CALIBRATOR_TRAINING_V1",
        "method": args.method,
        "elapsed_seconds": time.time() - t0,
        "gpu": gpu_info,
        "backbone_ckpt": {"path": str(args.backbone_ckpt), "sha256": sha256_file(args.backbone_ckpt)},
        "hyperparameters": {
            "hidden_ch": args.hidden_ch,
            "open_decoder_bm": args.open_decoder_bm,
            "phase1_frozen_steps": args.phase1_frozen_steps,
            "latent_align_weight": args.latent_align_weight,
            "total_steps": args.total_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "bm_lr_scale": args.bm_lr_scale,
            "seed": args.seed,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "holdout_touched": False,
        },
        "parameter_counts": {
            "total": total_param_count,
            "trainable": trainable_param_count,
            "overhead_pct": round(100.0 * trainable_param_count / total_param_count, 4),
        },
        "step0_validation": val0,
        "best_step": best_step,
        "best_validation": best_val,
        "best_checkpoint": {"path": str(best_ckpt_path), "sha256": sha256_file(best_ckpt_path)},
    }
    save_json(args.summary_dir / f"train_summary_{args.method}.json", summary)
    verify_no_links([args.ckpt_dir, args.summary_dir])


if __name__ == "__main__":
    main()
