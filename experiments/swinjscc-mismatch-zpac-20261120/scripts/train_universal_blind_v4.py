#!/usr/bin/env python3
"""V4 Training Engine: dB-Balanced Universal Self-Probing (AWGN & Rayleigh Backbones) & Digital 16/64-QAM Blind JSCC."""
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
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.universal_blind_v4 import UniversalBlindTransceiverV4
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir2_zpac.metrics import compute_psnr_batch


@torch.inference_mode()
def evaluate_v4_val256(
    model: UniversalBlindTransceiverV4,
    val_stems: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    N = val_stems.shape[0]
    bs = 32
    K = 128 * model.latent_channels

    gen = torch.Generator(device="cpu")
    gen.manual_seed(9101)
    eps_all = torch.randn((N, K, 2), generator=gen, dtype=torch.float32)
    hr_rb, hi_rb = sample_correlated_rbs(N, model.num_rbs, 0.85, device="cpu", generator=gen)
    h_rb_all = torch.stack([hr_rb, hi_rb], dim=-1)

    res: dict[str, float] = {}
    for ch_type, snrs in [("awgn", [1.0, 7.0, 13.0]), ("ofdm_complex", [4.0, 10.0, 13.0])]:
        for snr in snrs:
            vals = []
            snr_errs = []
            for s in range(0, N, bs):
                e = min(N, s + bs)
                out = model.forward_blind(
                    val_stems[s:e].to(device),
                    snr,
                    channel_type=ch_type,
                    h_rb_override=h_rb_all[s:e].to(device) if ch_type == "ofdm_complex" else None,
                    standard_noise_iq=eps_all[s:e].to(device),
                    input_is_stem=True,
                )
                vals.extend(compute_psnr_batch(out["recon"].clamp(0.0, 1.0), val_targets[s:e].to(device)).cpu().tolist())
                if ch_type == "awgn":
                    snr_errs.extend((out["cr_hat_db"] - snr).abs().cpu().tolist())
            key = f"{'awgn' if ch_type == 'awgn' else 'ofdm'}_{int(snr)}"
            res[key] = float(np.mean(vals))
            if ch_type == "awgn":
                res[f"snr_mae_awgn_{int(snr)}"] = float(np.mean(snr_errs))

    res["awgn_mean"] = (res["awgn_1"] + res["awgn_7"] + res["awgn_13"]) / 3.0
    res["ofdm_mean"] = (res["ofdm_4"] + res["ofdm_10"] + res["ofdm_13"]) / 3.0
    res["composite"] = 0.45 * res["awgn_mean"] + 0.55 * res["ofdm_mean"]
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["u_sph", "dd_qam16_sph", "dd_qam64_sph", "tcom25_qam16", "tcom25_qam64"],
    )
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--backbone-ckpt", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()

    configure_runtime_env(
        cache_root=Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir2_universal_blind_v4/cache") / args.run_tag,
        num_threads=4,
    )
    seed_everything(args.seed)
    configure_torch_determinism()
    gpu_info = admit_gpu(args.physical_gpu)
    device = torch.device("cuda:0")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_swinjscc_backbone(args.backbone_ckpt, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    model = UniversalBlindTransceiverV4(backbone, method=args.method, num_rbs=32, base_c0_db=7.0, hidden_ch=160).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[V4:{args.run_tag}] method={args.method} | trainable={n_trainable:,} ({100.0*n_trainable/n_total:.2f}%)", flush=True)

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
        torch.save({"step": step, "method": args.method, "run_tag": args.run_tag, "state_dict": sd, "val": val_dict}, path)

    t0 = time.time()
    val0 = evaluate_v4_val256(model, val_stems, val_targets, device)
    print(
        f"[V4:{args.run_tag}] Step    0 | comp={val0['composite']:.3f} | "
        f"awgn={val0['awgn_mean']:.3f} ({val0['awgn_1']:.2f}/{val0['awgn_7']:.2f}/{val0['awgn_13']:.2f}) | "
        f"ofdm={val0['ofdm_mean']:.3f} ({val0['ofdm_4']:.2f}/{val0['ofdm_10']:.2f}/{val0['ofdm_13']:.2f})",
        flush=True,
    )
    best_val = val0
    best_step = 0
    best_path = args.ckpt_dir / "best_v4.pt"
    save_light_ckpt(best_path, 0, val0)

    history = [{"step": 0, "train_loss_db": 0.0, **val0}]
    run_loss = 0.0
    run_cnt = 0
    K = 128 * model.latent_channels
    db_factor = 10.0 / math.log(10.0)

    for step in range(1, args.total_steps + 1):
        model.train()
        idx_rng = np.random.default_rng(args.seed * 73331 + step)
        batch_idx = idx_rng.choice(n_train, size=args.batch_size, replace=False)
        stems_b, xb = get_train_stems_and_targets(batch_idx)

        # Strictly hold out OOD -2 dB, 15 dB, and fast multipath rho_f=0.55!
        ch_rng = np.random.default_rng(args.seed * 1000003 + step * 89)
        ch_type = "awgn" if ch_rng.random() < 0.45 else "ofdm_complex"
        snr_arr = (
            ch_rng.uniform(0.0, 13.5, size=args.batch_size).astype(np.float32)
            if ch_type == "awgn"
            else ch_rng.uniform(1.0, 14.0, size=args.batch_size).astype(np.float32)
        )
        snr_vec = torch.from_numpy(snr_arr).to(device)

        ngen = torch.Generator(device="cpu")
        ngen.manual_seed(args.seed * 2000003 + step * 41)
        eps_iq = torch.randn((args.batch_size, K, 2), generator=ngen, dtype=torch.float32).to(device)
        if ch_type == "ofdm_complex":
            rho_train = float(ch_rng.uniform(0.78, 0.92))
            hr_rb, hi_rb = sample_correlated_rbs(args.batch_size, 32, rho_train, device="cpu", generator=ngen)
            h_rb = torch.stack([hr_rb, hi_rb], dim=-1).to(device)
        else:
            rho_train = 0.85
            h_rb = None

        out = model.forward_blind(
            stems_b,
            snr_vec,
            channel_type=ch_type,
            rho_f=rho_train,
            h_rb_override=h_rb,
            standard_noise_iq=eps_iq,
            input_is_stem=True,
        )
        # dB-Balanced Log-MSE Universal Objective: equal gradient weight per dB across 0..14 dB!
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
            vres = evaluate_v4_val256(model, val_stems, val_targets, device)
            is_best = vres["composite"] > best_val["composite"]
            if is_best:
                best_val = vres
                best_step = step
                save_light_ckpt(best_path, step, vres)
            print(
                f"[V4:{args.run_tag}] Step {step:4d}/{args.total_steps} | loss_db={avg_loss:.3f} | "
                f"comp={vres['composite']:.3f} | awgn={vres['awgn_mean']:.3f} ({vres['awgn_1']:.2f}/{vres['awgn_13']:.2f}) | "
                f"ofdm={vres['ofdm_mean']:.3f} ({vres['ofdm_4']:.2f}/{vres['ofdm_10']:.2f})"
                + (" *BEST*" if is_best else ""),
                flush=True,
            )
            history.append({"step": step, "train_loss_db": round(avg_loss, 5), **vres})

    save_light_ckpt(args.ckpt_dir / "last_v4.pt", args.total_steps, vres)
    hist_csv = args.summary_dir / f"train_v4_{args.run_tag}.csv"
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    save_json(
        args.summary_dir / f"train_v4_{args.run_tag}.json",
        {
            "schema": "DIR2_V4_UNIVERSAL_BLIND_TRAINING",
            "run_tag": args.run_tag,
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
