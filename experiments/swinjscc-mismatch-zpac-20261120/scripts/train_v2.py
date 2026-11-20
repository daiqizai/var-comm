#!/usr/bin/env python3
"""V2 Training Engine for Suite A (Official SA_Rayleigh), Suite B (Official SA_AWGN + OFDM), and Suite C (SAandRA Multi-Rate)."""
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
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.models_v2 import TransceiverZPACv2
from dir2_zpac.phy_v2 import OFDMFadingStep, simulate_ofdm_ar1_chain
from dir2_zpac.metrics import compute_psnr_batch


def build_training_step_batch(
    suite: str,
    batch_size: int,
    step: int,
    base_seed: int,
    latent_ch: int,
    device: torch.device,
    fixed_rate_channels: int | None = None,
) -> dict:
    """Deterministic paired condition & channel sampler strictly holding out +-2, +-4, +-8, +-9, +-12 dB."""
    rng = np.random.default_rng(base_seed * 1000003 + step * 97)
    snrs_grid = np.array([1.0, 4.0, 7.0, 10.0, 13.0], dtype=np.float32)

    valid_mis_pairs: list[tuple[float, float]] = []
    for gm in snrs_grid:
        for delta in (-6.0, -3.0, 3.0, 6.0):
            ct = gm + delta
            if 1.0 <= ct <= 13.0:
                valid_mis_pairs.append((float(ct), float(gm)))

    u_draw = rng.random()

    # Branch 1 (35% of steps): Correlated 32-RB OFDM / Rayleigh Fading with Delayed Feedback (tau_unc >= 0)
    if u_draw < 0.35:
        avg_snr = float(rng.choice([4.0, 7.0, 10.0, 13.0]))
        rho_c = float(rng.choice([0.95, 0.90, 0.85, 0.70, 0.55]))
        delay_d = int(rng.choice([0, 1, 2, 3], p=[0.20, 0.30, 0.30, 0.20]))
        steps = simulate_ofdm_ar1_chain(
            batch_size,
            avg_snr_db=avg_snr,
            rho_c=rho_c,
            delay_blocks=delay_d,
            seed=int((base_seed * 7919 + step * 104729) & 0x7FFFFFFF),
            num_rbs=32,
            rho_f=0.85,
            pilot_symbols=4,
            warmup=16,
        )
        eq_mode = "mrc_derot" if suite == "rayleigh" else "unbiased_mmse"
        rate_ch = (int(fixed_rate_channels) if fixed_rate_channels is not None else int(rng.choice([3, 6, 14]))) if suite == "sara" else None
        return {
            "channel_mode": "ofdm_fading",
            "ofdm_steps": steps,
            "equalizer_mode": eq_mode,
            "ct_vec": None,
            "gm_vec": None,
            "cr_vec": None,
            "tau_vec": None,
            "prior_vec": avg_snr,
            "rate_channels": rate_ch,
        }

    # Branch 2 (65% of steps): 40% Matched + 60% Seen Mismatch (+-3, +-6 dB ONLY!)
    ct_arr = np.zeros(batch_size, dtype=np.float32)
    gm_arr = np.zeros(batch_size, dtype=np.float32)
    cr_arr = np.zeros(batch_size, dtype=np.float32)
    tau_arr = np.zeros(batch_size, dtype=np.float32)
    prior_arr = np.full(batch_size, 7.0, dtype=np.float32)

    n_match = max(4, int(round(batch_size * 0.38)))
    for b in range(batch_size):
        if b < n_match:
            gm = float(snrs_grid[rng.integers(0, len(snrs_grid))]) if rng.random() < 0.7 else float(rng.uniform(1.0, 13.0))
            ct = gm
            cr = gm
            tau = 0.0
        else:
            if rng.random() < 0.75:
                ct, gm = valid_mis_pairs[int(rng.integers(0, len(valid_mis_pairs)))]
            else:
                delta = float(rng.choice([-6.0, -3.0, 3.0, 6.0]))
                gm_lo = max(0.5, 1.0 - delta)
                gm_hi = min(13.0, 13.0 - delta)
                gm = float(rng.uniform(gm_lo, gm_hi))
                ct = float(np.clip(gm + delta, 1.0, 13.0))
            cr = gm
            if rng.random() < 0.20:
                cr = float(np.clip(gm + rng.normal(0.0, 0.6), 0.0, 14.5))
            # Half of mismatched samples have tau=0 (unaware legacy TX), half have known feedback uncertainty
            if rng.random() < 0.40:
                tau = float(rng.uniform(0.15, 0.80))
                # Under uncertain feedback, simulate raw CQI before TX shrinkage
                ct = float(gm + rng.choice([-6.0, -3.0, 3.0, 6.0]) + rng.normal(0.0, 1.0))
                prior_arr[b] = float(rng.choice([4.0, 7.0, 10.0]))
            else:
                tau = 0.0

        ct_arr[b] = ct
        gm_arr[b] = gm
        cr_arr[b] = cr
        tau_arr[b] = tau

    perm = rng.permutation(batch_size)
    ch_mode = "rayleigh" if suite == "rayleigh" else "awgn"
    rate_ch = (int(fixed_rate_channels) if fixed_rate_channels is not None else int(rng.choice([3, 6, 14]))) if suite == "sara" else None

    return {
        "channel_mode": ch_mode,
        "ofdm_steps": None,
        "equalizer_mode": "mrc_derot" if suite == "rayleigh" else "unbiased_mmse",
        "ct_vec": torch.from_numpy(ct_arr[perm]).to(device),
        "gm_vec": torch.from_numpy(gm_arr[perm]).to(device),
        "cr_vec": torch.from_numpy(cr_arr[perm]).to(device),
        "tau_vec": torch.from_numpy(tau_arr[perm]).to(device),
        "prior_vec": torch.from_numpy(prior_arr[perm]).to(device),
        "rate_channels": rate_ch,
    }


@torch.inference_mode()
def evaluate_val256(
    model: TransceiverZPACv2,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    suite: str,
    device: torch.device,
    fixed_rate_channels: int | None = None,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    rate_ch = (int(fixed_rate_channels) if fixed_rate_channels is not None else 6) if suite == "sara" else None
    ch_mode = "rayleigh" if suite == "rayleigh" else "awgn"
    eq_mode = "mrc_derot" if suite == "rayleigh" else "unbiased_mmse"

    gen = torch.Generator(device="cpu")
    gen.manual_seed(8801)
    eps_all = torch.randn((N, 256, model.latent_channels), generator=gen, dtype=torch.float32)
    K = 128 * model.latent_channels
    h_iq_all = torch.randn((N, K, 2), generator=gen, dtype=torch.float32) * math.sqrt(0.5)

    conds = [
        ("match_4", 4.0, 4.0, 4.0),
        ("match_10", 10.0, 10.0, 10.0),
        ("over_7_1", 7.0, 1.0, 1.0),
        ("over_10_4", 10.0, 4.0, 4.0),
        ("under_1_7", 1.0, 7.0, 7.0),
        ("under_4_10", 4.0, 10.0, 10.0),
    ]
    res: dict[str, float] = {}
    for name, ct, gm, cr in conds:
        vals = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            out = model.forward_channel_v2(
                val_stems[s:e].to(device),
                ct,
                gm,
                cr,
                channel_mode=ch_mode,
                tau_unc=0.0,
                standard_noise=eps_all[s:e].to(device),
                h_complex_iq=h_iq_all[s:e].to(device) if ch_mode == "rayleigh" else None,
                rate_channels=rate_ch,
                input_is_stem=True,
            )
            vals.extend(compute_psnr_batch(out["recon"], val_targets[s:e].to(device)).cpu().tolist())
        res[name] = float(np.mean(vals))

    # Also evaluate 32-RB OFDM delayed fading on val256 (rho=0.80, d=2, avg=10 dB)
    ofdm_steps = simulate_ofdm_ar1_chain(N, avg_snr_db=10.0, rho_c=0.80, delay_blocks=2, seed=8802, num_rbs=32)
    f_vals = []
    for s in range(0, N, bs):
        e = min(N, s + bs)
        out = model.forward_channel_v2(
            val_stems[s:e].to(device),
            0.0,
            0.0,
            0.0,
            channel_mode="ofdm_fading",
            ofdm_steps=ofdm_steps[s:e],
            standard_noise=eps_all[s:e].to(device),
            equalizer_mode=eq_mode,
            rate_channels=rate_ch,
            input_is_stem=True,
        )
        f_vals.extend(compute_psnr_batch(out["recon"], val_targets[s:e].to(device)).cpu().tolist())
    res["ofdm_d2"] = float(np.mean(f_vals))

    res["matched_mean"] = 0.5 * (res["match_4"] + res["match_10"])
    res["mismatch_mean"] = 0.25 * (res["over_7_1"] + res["over_10_4"] + res["under_1_7"] + res["under_4_10"])
    res["composite"] = 0.20 * res["matched_mean"] + 0.45 * res["mismatch_mean"] + 0.35 * res["ofdm_d2"]
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, choices=["rayleigh", "awgn", "sara"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-ch", type=int, default=160)
    parser.add_argument("--rate-channels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env()
    seed_everything(args.seed)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    model_mode = "SwinJSCC_w/_SAandRA" if args.suite == "sara" else "SwinJSCC_w/_SA"
    c_bot = 320 if args.suite == "sara" else 96

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode=model_mode, c_bottleneck=c_bot, device=device)
    model = TransceiverZPACv2(backbone, method=args.method, hidden_ch=args.hidden_ch).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[V2-{args.suite}:{args.method}] trainable={n_trainable:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

    train_u8 = load_split_u8("train20k").images_u8
    val_u8 = load_split_u8("val256").images_u8
    n_train = int(train_u8.shape[0])

    # Pre-extract val256 encoder stems once (exact FP32, 32 MB)
    val_targets = val_u8.to(dtype=torch.float32) / 255.0
    val_stems_list = []
    with torch.inference_mode():
        for s in range(0, val_u8.shape[0], 32):
            e = min(val_u8.shape[0], s + 32)
            val_stems_list.append(model.extract_encoder_stem(val_targets[s:e].to(device)).cpu())
    val_stems = torch.cat(val_stems_list, dim=0)

    # Lazy CPU stem cache for train20k (~6.1 GiB max in host RAM, 100% exact FP32!)
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
        torch.save({"step": step, "suite": args.suite, "method": args.method, "state_dict": sd, "val": val_dict}, path)

    t0 = time.time()
    val0 = evaluate_val256(model, val_stems, val_targets, args.suite, device, fixed_rate_channels=args.rate_channels)
    tag = f"{args.suite}_k{args.rate_channels}" if args.rate_channels is not None else args.suite
    print(
        f"[{tag}:{args.method}] Step    0 | comp={val0['composite']:.3f} | "
        f"match={val0['matched_mean']:.3f} | mis={val0['mismatch_mean']:.3f} | ofdm_d2={val0['ofdm_d2']:.3f}",
        flush=True,
    )
    best_val = val0
    best_step = 0
    best_path = args.ckpt_dir / "best_calibrator_v2.pt"
    save_light_ckpt(best_path, 0, val0)

    history = [{"step": 0, "train_mse": 0.0, **val0}]
    run_loss = 0.0
    run_cnt = 0

    K_iq = 128 * model.latent_channels
    for step in range(1, args.total_steps + 1):
        model.train()
        idx_rng = np.random.default_rng(args.seed * 31337 + step)
        batch_idx = idx_rng.choice(n_train, size=args.batch_size, replace=False)
        stems_b, xb = get_train_stems_and_targets(batch_idx)

        cfg = build_training_step_batch(
            args.suite,
            args.batch_size,
            step,
            args.seed,
            model.latent_channels,
            device,
            fixed_rate_channels=args.rate_channels,
        )

        ngen = torch.Generator(device="cpu")
        ngen.manual_seed(args.seed * 1000003 + step * 53)
        eps = torch.randn((args.batch_size, 256, model.latent_channels), generator=ngen, dtype=torch.float32).to(device)
        h_iq = (
            (torch.randn((args.batch_size, K_iq, 2), generator=ngen, dtype=torch.float32) * math.sqrt(0.5)).to(device)
            if cfg["channel_mode"] == "rayleigh"
            else None
        )

        out = model.forward_channel_v2(
            stems_b,
            cfg["ct_vec"] if cfg["ct_vec"] is not None else 0.0,
            cfg["gm_vec"] if cfg["gm_vec"] is not None else 0.0,
            cfg["cr_vec"] if cfg["cr_vec"] is not None else 0.0,
            channel_mode=cfg["channel_mode"],
            tau_unc=cfg["tau_vec"] if cfg["tau_vec"] is not None else 0.0,
            c_prior_db=cfg["prior_vec"],
            ofdm_steps=cfg["ofdm_steps"],
            standard_noise=eps,
            h_complex_iq=h_iq,
            equalizer_mode=cfg["equalizer_mode"],
            rate_channels=cfg["rate_channels"],
            input_is_stem=True,
        )
        loss = F.mse_loss(out["recon"], xb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        run_loss += float(loss.item())
        run_cnt += 1

        if step % args.eval_interval == 0 or step == args.total_steps:
            avg_mse = run_loss / max(1, run_cnt)
            run_loss = 0.0
            run_cnt = 0
            vres = evaluate_val256(model, val_stems, val_targets, args.suite, device, fixed_rate_channels=args.rate_channels)
            is_best = vres["composite"] > best_val["composite"]
            if is_best:
                best_val = vres
                best_step = step
                save_light_ckpt(best_path, step, vres)
            print(
                f"[{tag}:{args.method}] Step {step:4d}/{args.total_steps} | mse={avg_mse:.6f} | "
                f"comp={vres['composite']:.3f} | match={vres['matched_mean']:.3f} | "
                f"mis={vres['mismatch_mean']:.3f} | ofdm_d2={vres['ofdm_d2']:.3f}"
                + (" *BEST*" if is_best else ""),
                flush=True,
            )
            history.append({"step": step, "train_mse": round(avg_mse, 7), **vres})

    save_light_ckpt(args.ckpt_dir / "last_calibrator_v2.pt", args.total_steps, vres)
    hist_csv = args.summary_dir / f"train_v2_{tag}_{args.method}.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_json(
        args.summary_dir / f"train_v2_{tag}_{args.method}.json",
        {
            "schema": "DIR2_V2_TRAINING",
            "suite": tag,
            "method": args.method,
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
