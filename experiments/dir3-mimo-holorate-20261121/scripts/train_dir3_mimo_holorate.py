#!/usr/bin/env python3
"""Direction 3 Unified Training Engine:
  - Track A (mimo): 2x2 & 4x4 Zero-Pilot Spatial-Multiplexed MIMO-OFDM (os_mimo vs blackbox_mimo)
  - Track B (holorate): Rateless Anytime-Stopping, Burst-Erasure & Partial-Band Anti-Jamming (holorate_blind vs raster_blind_resilient)
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
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.metrics import compute_psnr_batch

from dir3_mh.mimo_blind import BlindMIMOTransceiver, sample_kronecker_mimo_rbs
from dir3_mh.holorate_blind import HoloRateBlindTransceiver


@torch.inference_mode()
def evaluate_mimo_val256(
    model: BlindMIMOTransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    nt = model.n_ant
    t_mimo = (128 * model.latent_channels) // nt

    gen = torch.Generator(device="cpu").manual_seed(9301)
    eps_all = torch.randn((N, nt, t_mimo, 2), generator=gen, dtype=torch.float32)
    m_rbs = model.num_rbs
    hr_iid, hi_iid = sample_kronecker_mimo_rbs(N, m_rbs, nt, nt, rho_f=0.85, rho_ant=0.0, generator=gen)
    hr_cor, hi_cor = sample_kronecker_mimo_rbs(N, m_rbs, nt, nt, rho_f=0.85, rho_ant=0.60, generator=gen)

    res: dict[str, float] = {}
    for label, snr, hr_b, hi_b in [
        ("mimo_4", 4.0, hr_iid, hi_iid),
        ("mimo_10", 10.0, hr_iid, hi_iid),
        ("mimo_13", 13.0, hr_iid, hi_iid),
        ("mimo_corr10", 10.0, hr_cor, hi_cor),
    ]:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_mimo_blind(
                val_stems[s:e].to(device),
                snr,
                h_mimo_override=(hr_b[s:e].to(device), hi_b[s:e].to(device)),
                standard_noise_iq=eps_all[s:e].to(device),
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
        res[label] = float(np.mean(vals))

    res["composite"] = (res["mimo_4"] + res["mimo_10"] + res["mimo_13"] + res["mimo_corr10"]) / 4.0
    return res


@torch.inference_mode()
def evaluate_holorate_val256(
    model: HoloRateBlindTransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    gen = torch.Generator(device="cpu").manual_seed(9401)
    eps_all = torch.randn((N, 32, 384, 2), generator=gen, dtype=torch.float32)
    jeps_all = torch.randn((N, 32, 384, 2), generator=gen, dtype=torch.float32)
    hr, hi = sample_correlated_rbs(N, 32, 0.85, device="cpu", generator=gen)
    h_rb_all = torch.stack([hr, hi], dim=-1)

    m_r25 = torch.zeros((1, 32)); m_r25[:, :8] = 1.0
    m_r50 = torch.zeros((1, 32)); m_r50[:, :16] = 1.0
    m_b50 = torch.ones((1, 32)); m_b50[:, 8:24] = 0.0
    j_31 = torch.zeros((1, 32)); j_31[:, 6:16] = 1.0

    conds = [
        ("full_10", 10.0, None, None),
        ("rate25_10", 10.0, m_r25, None),
        ("rate50_10", 10.0, m_r50, None),
        ("burst50_10", 10.0, m_b50, None),
        ("jam31_10", 10.0, None, j_31),
    ]
    res: dict[str, float] = {}
    for label, snr, amask, jmask in conds:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            cur_b = e - s
            out = model.forward_hostile_blind(
                val_stems[s:e].to(device),
                snr,
                alive_rb_mask=amask.expand(cur_b, 32).to(device) if amask is not None else None,
                jam_rb_mask=jmask.expand(cur_b, 32).to(device) if jmask is not None else None,
                jam_inr_db=15.0,
                h_rb_override=h_rb_all[s:e].to(device),
                standard_noise_iq=eps_all[s:e].to(device),
                jam_noise_iq=jeps_all[s:e].to(device),
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
        res[label] = float(np.mean(vals))

    res["composite"] = (
        0.25 * res["full_10"]
        + 0.15 * res["rate25_10"]
        + 0.20 * res["rate50_10"]
        + 0.20 * res["burst50_10"]
        + 0.20 * res["jam31_10"]
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, required=True, choices=["mimo", "holorate"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n-ant", type=int, default=2)
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir3_mimo_holorate_v1/cache") / args.run_tag,
        num_threads=4,
    )
    seed_everything(args.seed)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    if args.track == "mimo":
        model = BlindMIMOTransceiver(backbone, method=args.method, n_ant=args.n_ant, num_rbs=None, hidden_ch=160).to(device)
    else:
        model = HoloRateBlindTransceiver(backbone, method=args.method, num_rbs=32, hidden_ch=160).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Dir3:{args.run_tag}] track={args.track} method={args.method} | trainable={n_trainable:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

    train_u8 = load_split_u8("train20k").images_u8
    val_u8 = load_split_u8("val256").images_u8
    n_train = int(train_u8.shape[0])

    val_targets = val_u8.to(dtype=torch.float32) / 255.0
    val_stems_list = []
    with torch.inference_mode():
        for s in range(0, val_u8.shape[0], 32):
            e = min(val_u8.shape[0], s + 32)
            val_stems_list.append(model.extract_encoder_stem(val_targets[s:e].to(device)).cpu())
    val_stems = torch.cat(val_stems_list, dim=0)

    stem_cache: dict[int, torch.Tensor] = {}

    def get_train_stems_and_targets(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        xb = (train_u8[indices].to(device=device, dtype=torch.float32)) / 255.0
        missing_pos = [i for i, idx in enumerate(indices.tolist()) if idx not in stem_cache]
        if missing_pos:
            miss_t = torch.tensor(missing_pos, device=device, dtype=torch.long)
            with torch.no_grad():
                miss_stems = model.extract_encoder_stem(xb.index_select(0, miss_t)).cpu()
            for k_i, pos in enumerate(missing_pos):
                stem_cache[int(indices[pos])] = miss_stems[k_i]
        stems = torch.stack([stem_cache[int(idx)] for idx in indices.tolist()], dim=0).to(device=device)
        return stems, xb

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr * 0.05)

    def save_light_ckpt(path: Path, step: int, val_dict: dict) -> None:
        keep_keys = {name for name, p in model.named_parameters() if p.requires_grad}
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep_keys}
        torch.save({"step": step, "track": args.track, "method": args.method, "run_tag": args.run_tag, "state_dict": sd, "val": val_dict}, path)

    t0 = time.time()
    eval_fn = evaluate_mimo_val256 if args.track == "mimo" else evaluate_holorate_val256
    val0 = eval_fn(model, val_stems, val_targets, device)
    print(f"[Dir3:{args.run_tag}] Step    0 | comp={val0['composite']:.3f} | {val0}", flush=True)

    best_val = val0
    best_step = 0
    best_path = args.ckpt_dir / "best_dir3.pt"
    save_light_ckpt(best_path, 0, val0)

    history = [{"step": 0, "train_loss_db": 0.0, **val0}]
    run_loss = 0.0
    run_cnt = 0
    db_factor = 10.0 / math.log(10.0)

    for step in range(1, args.total_steps + 1):
        model.train()
        idx_rng = np.random.default_rng(args.seed * 73331 + step)
        batch_idx = idx_rng.choice(n_train, size=args.batch_size, replace=False)
        stems_b, xb = get_train_stems_and_targets(batch_idx)

        ch_rng = np.random.default_rng(args.seed * 1000003 + step * 89)
        ngen = torch.Generator(device="cpu").manual_seed(args.seed * 2000003 + step * 41)

        if args.track == "mimo":
            snr_arr = ch_rng.uniform(1.0, 14.0, size=args.batch_size).astype(np.float32)
            snr_vec = torch.from_numpy(snr_arr).to(device)
            rho_f = float(ch_rng.uniform(0.78, 0.92))
            # Hold out severe rho_ant = 0.60 for OOD testing! Train only on [0.0, 0.35]
            rho_ant = 0.0 if ch_rng.random() < 0.60 else float(ch_rng.uniform(0.10, 0.35))
            nt = args.n_ant
            t_mimo = (128 * 96) // nt
            eps_iq = torch.randn((args.batch_size, nt, t_mimo, 2), generator=ngen, dtype=torch.float32).to(device)
            hr_b, hi_b = sample_kronecker_mimo_rbs(args.batch_size, model.num_rbs, nt, nt, rho_f=rho_f, rho_ant=rho_ant, generator=ngen)
            out = model.forward_mimo_blind(
                stems_b,
                snr_vec,
                rho_f=rho_f,
                rho_ant=rho_ant,
                h_mimo_override=(hr_b.to(device), hi_b.to(device)),
                standard_noise_iq=eps_iq,
                input_is_stem=True,
            )
        else:
            snr_arr = ch_rng.uniform(2.0, 14.0, size=args.batch_size).astype(np.float32)
            snr_vec = torch.from_numpy(snr_arr).to(device)
            rho_f = float(ch_rng.uniform(0.78, 0.92))
            eps_iq = torch.randn((args.batch_size, 32, 384, 2), generator=ngen, dtype=torch.float32).to(device)
            jeps_iq = torch.randn((args.batch_size, 32, 384, 2), generator=ngen, dtype=torch.float32).to(device)
            hr_b, hi_b = sample_correlated_rbs(args.batch_size, 32, rho_f, device="cpu", generator=ngen)
            h_rb = torch.stack([hr_b, hi_b], dim=-1).to(device)

            draw = ch_rng.random()
            alive_np = np.ones((args.batch_size, 32), dtype=np.float32)
            jam_np = np.zeros((args.batch_size, 32), dtype=np.float32)
            jam_inr = 15.0

            if draw < 0.25:
                # Regime 1: Clean full-rate 32 RBs
                pass
            elif draw < 0.60:
                # Regime 2: Rateless Anytime-Stopping Prefix (B_keep in 6..31 RBs)
                for b_i in range(args.batch_size):
                    k_keep = int(ch_rng.integers(6, 32))
                    alive_np[b_i, k_keep:] = 0.0
            elif draw < 0.80:
                # Regime 3: Contiguous or Random Burst Packet Erasures (20%..50% dropped!)
                for b_i in range(args.batch_size):
                    n_drop = int(ch_rng.integers(6, 17))
                    if ch_rng.random() < 0.65:
                        st_rb = int(ch_rng.integers(0, 32 - n_drop + 1))
                        alive_np[b_i, st_rb : st_rb + n_drop] = 0.0
                    else:
                        d_idx = ch_rng.choice(32, size=n_drop, replace=False)
                        alive_np[b_i, d_idx] = 0.0
            else:
                # Regime 4: Hostile Partial-Band Jamming (15%..38% of RBs jammed at +12..+16 dB; +18 dB held out OOD!)
                jam_inr = float(ch_rng.uniform(12.0, 16.0))
                for b_i in range(args.batch_size):
                    n_jam = int(ch_rng.integers(5, 13))
                    j_idx = ch_rng.choice(32, size=n_jam, replace=False)
                    jam_np[b_i, j_idx] = 1.0

            out = model.forward_hostile_blind(
                stems_b,
                snr_vec,
                rho_f=rho_f,
                alive_rb_mask=torch.from_numpy(alive_np).to(device),
                jam_rb_mask=torch.from_numpy(jam_np).to(device),
                jam_inr_db=jam_inr,
                h_rb_override=h_rb,
                standard_noise_iq=eps_iq,
                jam_noise_iq=jeps_iq,
                input_is_stem=True,
            )

        per_img_mse = (out["recon"] - xb).square().mean(dim=(1, 2, 3))
        loss = db_factor * torch.log(per_img_mse + 1e-6).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        run_loss += float(loss.item())
        run_cnt += 1

        if step % args.eval_interval == 0 or step == args.total_steps:
            avg_loss = run_loss / max(1, run_cnt)
            run_loss = 0.0
            run_cnt = 0
            vres = eval_fn(model, val_stems, val_targets, device)
            is_best = vres["composite"] > best_val["composite"]
            if is_best:
                best_val = vres
                best_step = step
                save_light_ckpt(best_path, step, vres)
            summary_kv = " | ".join(f"{k}={v:.2f}" for k, v in vres.items() if k != "composite")
            print(
                f"[Dir3:{args.run_tag}] Step {step:4d}/{args.total_steps} | loss_db={avg_loss:.3f} | "
                f"comp={vres['composite']:.3f} | {summary_kv}" + (" *BEST*" if is_best else ""),
                flush=True,
            )
            history.append({"step": step, "train_loss_db": round(avg_loss, 5), **vres})

    save_light_ckpt(args.ckpt_dir / "last_dir3.pt", args.total_steps, vres)
    hist_csv = args.summary_dir / f"train_dir3_{args.run_tag}.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    save_json(
        args.summary_dir / f"train_dir3_{args.run_tag}.json",
        {
            "schema": "DIR3_MIMO_HOLORATE_TRAINING",
            "run_tag": args.run_tag,
            "track": args.track,
            "method": args.method,
            "n_ant": args.n_ant if args.track == "mimo" else 1,
            "elapsed_seconds": time.time() - t0,
            "gpu": gpu_info,
            "backbone_ckpt": {"path": str(args.backbone_ckpt), "sha256": sha256_file(args.backbone_ckpt)},
            "parameter_counts": {
                "total": n_total,
                "trainable": n_trainable,
                "overhead_pct": round(100.0 * n_trainable / n_total, 4),
            },
            "step0_val": val0,
            "best_step": best_step,
            "best_val": best_val,
            "best_checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
            "holdout_touched": False,
        },
    )
    verify_no_links([args.ckpt_dir, args.summary_dir])


if __name__ == "__main__":
    main()
