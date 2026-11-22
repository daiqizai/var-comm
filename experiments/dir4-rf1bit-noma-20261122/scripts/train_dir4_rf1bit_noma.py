#!/usr/bin/env python3
"""Direction 4 Unified Training Engine:
  - Track A (rf): RF-Bussgang-JSCC vs BlackBox-RF under 1-Bit / 2-Bit Time-Domain ADC, Rapp Nonlinear HPA & Wiener Phase Noise
  - Track B (noma): SemSIC-NOMA-JSCC vs Stage1-Only Ablation vs BlackBox-NOMA on Grant-Free 2x2 & Overloaded 3x2 Uplink
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
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.metrics import compute_psnr_batch

from dir4_rn.rf_1bit_blind import N_OFDM_SC, N_OFDM_SYM, RFImpairedBlindTransceiver
from dir4_rn.noma_semsic_blind import GrantFreeNOMATransceiver, sample_multiuser_uplink_channels


@torch.inference_mode()
def evaluate_rf_val256(
    model: RFImpairedBlindTransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    adc_bits: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    gen = torch.Generator(device="cpu").manual_seed(9501 + adc_bits * 17)
    eps_td = torch.randn((N, N_OFDM_SYM, N_OFDM_SC, 2), generator=gen, dtype=torch.float32)
    hr, hi = sample_correlated_rbs(N, 32, 0.85, device="cpu", generator=gen)
    h_rb = torch.stack([hr, hi], dim=-1)

    dphi_35 = torch.randn((N, N_OFDM_SYM), generator=gen, dtype=torch.float32) * (3.5 * math.pi / 180.0)
    dphi_35[:, 0] = 0.0
    phi_35 = torch.cumsum(dphi_35, dim=1)

    dphi_55 = torch.randn((N, N_OFDM_SYM), generator=gen, dtype=torch.float32) * (5.5 * math.pi / 180.0)
    dphi_55[:, 0] = 0.0
    phi_55 = torch.cumsum(dphi_55, dim=1)

    conds = [
        ("rf_4dB", 4.0, 1.0, phi_35),
        ("rf_10dB", 10.0, 1.0, phi_35),
        ("rf_14dB", 14.0, 1.0, phi_35),
        ("rf_ood_severe10", 10.0, 0.0, phi_55),  # OOD 0 dB IBO saturation + 5.5 deg/sym severe PN!
    ]
    res: dict[str, float] = {}
    for label, snr, ibo, phi_w in conds:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_rf_blind(
                val_stems[s:e].to(device),
                snr,
                adc_bits=adc_bits,
                pa_ibo_db=ibo,
                h_rb_override=h_rb[s:e].to(device),
                phase_walk_override=phi_w[s:e].to(device),
                standard_noise_td_iq=eps_td[s:e].to(device),
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
        res[label] = float(np.mean(vals))

    res["composite"] = 0.25 * (res["rf_4dB"] + res["rf_10dB"] + res["rf_14dB"] + res["rf_ood_severe10"])
    return res


@torch.inference_mode()
def evaluate_noma_val256(
    model: GrantFreeNOMATransceiver,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 16
    gen = torch.Generator(device="cpu").manual_seed(9607)

    # Condition 1: 2-User Near-Far (12 dB + 3 dB -> 2-BS-Antenna)
    hr2, hi2 = sample_multiuser_uplink_channels(N, 2, 2, 32, 0.85, generator=gen)
    eps2 = torch.randn((N, 2, 32, 384, 2), generator=gen, dtype=torch.float32)
    idx_u1_2 = (np.arange(N) + 85) % N

    vals_2u_near, vals_2u_far = [], []
    for s in range(0, N, bs):
        e = min(N, s + bs)
        cb = e - s
        snrs2 = torch.tensor([[12.0, 3.0]], device=device).expand(cb, 2)
        stems_pair = [val_stems[s:e].to(device), val_stems[idx_u1_2[s:e]].to(device)]
        out2 = model.forward_noma_uplink(
            stems_pair,
            snrs2,
            h_mu_override=(hr2[s:e].to(device), hi2[s:e].to(device)),
            standard_noise_iq=eps2[s:e].to(device),
            input_is_stem=True,
        )
        vals_2u_near.extend(compute_psnr_batch(out2["recons"][0].clamp(0, 1), val_targets[s:e].to(device)).cpu().tolist())
        vals_2u_far.extend(compute_psnr_batch(out2["recons"][1].clamp(0, 1), val_targets[idx_u1_2[s:e]].to(device)).cpu().tolist())

    # Condition 2: Overloaded 3-User -> 2-BS-Antenna Near-Far NOMA (13 dB + 8 dB + 4 dB!)
    hr3, hi3 = sample_multiuser_uplink_channels(N, 3, 2, 32, 0.85, generator=gen)
    eps3 = torch.randn((N, 2, 32, 384, 2), generator=gen, dtype=torch.float32)
    idx_u1_3 = (np.arange(N) + 85) % N
    idx_u2_3 = (np.arange(N) + 170) % N

    vals_3u = [[], [], []]
    for s in range(0, N, bs):
        e = min(N, s + bs)
        cb = e - s
        snrs3 = torch.tensor([[13.0, 8.0, 4.0]], device=device).expand(cb, 3)
        stems_tri = [
            val_stems[s:e].to(device),
            val_stems[idx_u1_3[s:e]].to(device),
            val_stems[idx_u2_3[s:e]].to(device),
        ]
        targs_tri = [
            val_targets[s:e].to(device),
            val_targets[idx_u1_3[s:e]].to(device),
            val_targets[idx_u2_3[s:e]].to(device),
        ]
        out3 = model.forward_noma_uplink(
            stems_tri,
            snrs3,
            h_mu_override=(hr3[s:e].to(device), hi3[s:e].to(device)),
            standard_noise_iq=eps3[s:e].to(device),
            input_is_stem=True,
        )
        for u_i in range(3):
            vals_3u[u_i].extend(compute_psnr_batch(out3["recons"][u_i].clamp(0, 1), targs_tri[u_i]).cpu().tolist())

    res = {
        "noma2_near12": float(np.mean(vals_2u_near)),
        "noma2_far3": float(np.mean(vals_2u_far)),
        "over3x2_u1_13": float(np.mean(vals_3u[0])),
        "over3x2_u2_8": float(np.mean(vals_3u[1])),
        "over3x2_u3_4": float(np.mean(vals_3u[2])),
    }
    res["over3x2_mean"] = (res["over3x2_u1_13"] + res["over3x2_u2_8"] + res["over3x2_u3_4"]) / 3.0
    res["composite"] = 0.35 * (0.5 * (res["noma2_near12"] + res["noma2_far3"])) + 0.65 * res["over3x2_mean"]
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, required=True, choices=["rf", "noma"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--adc-bits", type=int, default=1)
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir4_rf1bit_noma_v1/cache") / args.run_tag,
        num_threads=4,
    )
    seed_everything(args.seed)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    if args.track == "rf":
        model: torch.nn.Module = RFImpairedBlindTransceiver(backbone, method=args.method, num_rbs=32, hidden_ch=160).to(device)
    else:
        model = GrantFreeNOMATransceiver(backbone, method=args.method, n_rx=2, num_rbs=32, hidden_ch=160).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Dir4:{args.run_tag}] track={args.track} method={args.method} | trainable={n_trainable:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

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
        torch.save(
            {"step": step, "track": args.track, "method": args.method, "adc_bits": args.adc_bits, "run_tag": args.run_tag, "state_dict": sd, "val": val_dict},
            path,
        )

    t0 = time.time()
    val0 = (
        evaluate_rf_val256(model, val_stems, val_targets, args.adc_bits, device)
        if args.track == "rf"
        else evaluate_noma_val256(model, val_stems, val_targets, device)
    )
    print(f"[Dir4:{args.run_tag}] Step    0 | comp={val0['composite']:.3f} | {val0}", flush=True)

    best_val = val0
    best_step = 0
    best_path = args.ckpt_dir / "best_dir4.pt"
    save_light_ckpt(best_path, 0, val0)

    history = [{"step": 0, "train_loss_db": 0.0, **val0}]
    run_loss = 0.0
    run_cnt = 0
    db_factor = 10.0 / math.log(10.0)

    for step in range(1, args.total_steps + 1):
        model.train()
        idx_rng = np.random.default_rng(args.seed * 73331 + step)
        ch_rng = np.random.default_rng(args.seed * 1000003 + step * 89)
        ngen = torch.Generator(device="cpu").manual_seed(args.seed * 2000003 + step * 41)

        if args.track == "rf":
            batch_idx = idx_rng.choice(n_train, size=args.batch_size, replace=False)
            stems_b, xb = get_train_stems_and_targets(batch_idx)

            snr_arr = ch_rng.uniform(1.0, 15.0, size=args.batch_size).astype(np.float32)
            snr_vec = torch.from_numpy(snr_arr).to(device)
            rho_f = float(ch_rng.uniform(0.78, 0.92))
            # Hold out severe IBO = 0.0 dB & pn = 5.5 deg/sym strictly for OOD evaluation!
            ibo_db = float(ch_rng.uniform(0.6, 3.0))
            pn_deg = float(ch_rng.uniform(1.5, 4.0))

            eps_td = torch.randn((args.batch_size, N_OFDM_SYM, N_OFDM_SC, 2), generator=ngen, dtype=torch.float32).to(device)
            hr_b, hi_b = sample_correlated_rbs(args.batch_size, 32, rho_f, device="cpu", generator=ngen)
            h_rb = torch.stack([hr_b, hi_b], dim=-1).to(device)
            dphi = torch.randn((args.batch_size, N_OFDM_SYM), generator=ngen, dtype=torch.float32) * (pn_deg * math.pi / 180.0)
            dphi[:, 0] = 0.0
            phi_w = torch.cumsum(dphi, dim=1).to(device)

            out = model.forward_rf_blind(
                stems_b,
                snr_vec,
                adc_bits=args.adc_bits,
                pa_ibo_db=ibo_db,
                pn_step_deg=pn_deg,
                rho_f=rho_f,
                h_rb_override=h_rb,
                phase_walk_override=phi_w,
                standard_noise_td_iq=eps_td,
                input_is_stem=True,
            )
            per_img_mse = (out["recon"] - xb).square().mean(dim=(1, 2, 3))
            loss = db_factor * torch.log(per_img_mse + 1e-6).mean()
        else:
            # 35% 2-User Grant-Free Uplink, 65% Overloaded 3-User -> 2-BS-Antenna Grant-Free NOMA!
            n_u = 2 if ch_rng.random() < 0.35 else 3
            all_idx = idx_rng.choice(n_train, size=args.batch_size * n_u, replace=False)
            stems_u, targets_u = [], []
            for u_i in range(n_u):
                su, xu = get_train_stems_and_targets(all_idx[u_i * args.batch_size : (u_i + 1) * args.batch_size])
                stems_u.append(su)
                targets_u.append(xu)

            # Sample realistic uncoordinated Near-Far & Balanced per-user SNRs (hold out extreme 14/2 dB OOD!)
            snr_mat = np.zeros((args.batch_size, n_u), dtype=np.float32)
            for b_i in range(args.batch_size):
                if n_u == 2:
                    if ch_rng.random() < 0.55:
                        # Near-Far 2-user
                        s_hi = float(ch_rng.uniform(9.0, 13.5))
                        s_lo = float(ch_rng.uniform(2.0, 6.5))
                        snr_mat[b_i] = [s_hi, s_lo] if ch_rng.random() < 0.5 else [s_lo, s_hi]
                    else:
                        s0 = float(ch_rng.uniform(1.5, 13.5))
                        snr_mat[b_i] = [s0, s0 + float(ch_rng.uniform(-1.5, 1.5))]
                else:
                    # Overloaded 3-User -> 2-BS-Antenna NOMA!
                    u_snrs = np.array([
                        ch_rng.uniform(10.0, 14.0),
                        ch_rng.uniform(6.0, 10.5),
                        ch_rng.uniform(2.5, 6.5),
                    ], dtype=np.float32)
                    ch_rng.shuffle(u_snrs)
                    snr_mat[b_i] = u_snrs

            rho_f = float(ch_rng.uniform(0.78, 0.92))
            hr_m, hi_m = sample_multiuser_uplink_channels(args.batch_size, n_u, 2, 32, rho_f, generator=ngen)
            eps_m = torch.randn((args.batch_size, 2, 32, 384, 2), generator=ngen, dtype=torch.float32).to(device)

            out_n = model.forward_noma_uplink(
                stems_u,
                torch.from_numpy(snr_mat).to(device),
                rho_f=rho_f,
                h_mu_override=(hr_m.to(device), hi_m.to(device)),
                standard_noise_iq=eps_m,
                input_is_stem=True,
            )
            recon_losses = []
            for u_i in range(n_u):
                mse_u = (out_n["recons"][u_i] - targets_u[u_i]).square().mean(dim=(1, 2, 3))
                recon_losses.append(db_factor * torch.log(mse_u + 1e-6).mean())
            loss = sum(recon_losses) / float(n_u)

            # Direct Latent Denoising Supervision for Sem-SIC Prior Network!
            z_den = out_n["z_denoised_users"]
            if z_den:
                z_tgt = out_n["z_tx_users"]
                lat_loss = sum((z_den[u_i] - z_tgt[u_i].detach()).square().mean() for u_i in range(n_u)) / float(n_u)
                loss = loss + 0.35 * lat_loss

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
            vres = (
                evaluate_rf_val256(model, val_stems, val_targets, args.adc_bits, device)
                if args.track == "rf"
                else evaluate_noma_val256(model, val_stems, val_targets, device)
            )
            is_best = vres["composite"] > best_val["composite"]
            if is_best:
                best_val = vres
                best_step = step
                save_light_ckpt(best_path, step, vres)
            summary_kv = " | ".join(f"{k}={v:.2f}" for k, v in vres.items() if k != "composite")
            print(
                f"[Dir4:{args.run_tag}] Step {step:4d}/{args.total_steps} | loss={avg_loss:.3f} | "
                f"comp={vres['composite']:.3f} | {summary_kv}" + (" *BEST*" if is_best else ""),
                flush=True,
            )
            history.append({"step": step, "train_loss_db": round(avg_loss, 5), **vres})

    save_light_ckpt(args.ckpt_dir / "last_dir4.pt", args.total_steps, vres)
    hist_csv = args.summary_dir / f"train_dir4_{args.run_tag}.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    save_json(
        args.summary_dir / f"train_dir4_{args.run_tag}.json",
        {
            "schema": "DIR4_RF1BIT_NOMA_TRAINING",
            "run_tag": args.run_tag,
            "track": args.track,
            "method": args.method,
            "adc_bits": args.adc_bits if args.track == "rf" else 32,
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
