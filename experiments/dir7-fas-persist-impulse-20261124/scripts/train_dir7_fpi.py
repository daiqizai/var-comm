#!/usr/bin/env python3
"""Unified Multi-GPU Training Driver for Direction 7 (`codex/dir7-fas-persist-impulse-v1`):
GPT-Audit-Grade Matched-Control Training across:
  - Track `dir7a`: Zero-Pilot 32-Port Fluid / Pixel Antenna JSCC (`fas_kriging` vs byte-identical `fixed_port`
                   vs byte-identical `random_hop`) with paired mini-batch/channel sequences, pure in-domain `val256`
                   selection (`best_indist.pt`), and fixed final step (`last.pt`).
                   (`n_subbands=4` wideband pixel-FAS and `rician_k=2.0` LoS fields are strictly held-out OOD!)
  - Track `dir7c`: Fast-Time Impulsive Interference Cancellation (`sem_impulse_2round` vs byte-identical
                   `huber_owndenoise` + end-to-end trained `classical_blanking` & `standard_blind`) with identical
                   loss parity, paired mini-batch/channel draws, pure in-domain `val256` selection (`best_indist.pt`),
                   and fixed final step (`last.pt`).
                   (`p_imp=0.12, imp_sir_db=-10.0` heavy burst is strictly held-out OOD!)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time
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
    seed_everything,
    verify_no_links,
)
from dir2_zpac.metrics import compute_psnr_batch
from dir2_zpac.models import load_swinjscc_backbone
from dir2_zpac.phy_v2 import sample_correlated_rbs
from dir5_og.otfs_blind import N_SC, N_SYM

from dir7_fpi import (
    BlindFASTransceiver,
    ImpulseShieldTransceiver,
    sample_fas_spatial_channels,
)
from dir7_fpi.impulse_shield_blind import N_IMP_RBS

SWIN_AWGN_CKPT = Path(
    "/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/dir2_swinjscc_mismatch_v1/pretrained/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model"
)
DIR5_SWIN_CACHE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/dir5_otfs_genvar_v1/cache/swin_stems_fp16.pt")


@torch.inference_mode()
def ensure_swin_cache(cache_dir: Path, device: torch.device) -> Path:
    if DIR5_SWIN_CACHE.exists():
        return DIR5_SWIN_CACHE
    local_cache = cache_dir / "swin_stems_fp16.pt"
    if local_cache.exists():
        return local_cache

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    enc = backbone.encoder

    def _ext(imgs_u8: torch.Tensor, bs: int = 64) -> torch.Tensor:
        outs = []
        for s in range(0, imgs_u8.shape[0], bs):
            xb = imgs_u8[s : s + bs].to(device=device, dtype=torch.float32) / 255.0
            x = enc.patch_embed(xb)
            for layer in enc.layers:
                x = layer(x)
            outs.append(enc.norm(x).to(torch.float16).cpu())
        return torch.cat(outs, dim=0)

    tr_s = _ext(load_split_u8("train20k").images_u8)
    va_s = _ext(load_split_u8("val256").images_u8).to(torch.float32)
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_stems_fp16": tr_s, "val_stems_fp32": va_s}, local_cache)
    return local_cache


@torch.inference_mode()
def eval_dir7a_indomain_val256(
    model: BlindFASTransceiver,
    val_stems: torch.Tensor,
    val_u8: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Pure in-domain `val256` validation (`n_subbands=1, rician_k=0.0`, SNR in {2, 6, 10} dB)."""
    model.eval()
    N = val_u8.shape[0]
    bs = 32
    scores = {}
    for idx_c, snr_db in enumerate((2.0, 6.0, 10.0)):
        gen = torch.Generator(device="cpu").manual_seed(2026710 + idx_c * 31)
        h_all = sample_fas_spatial_channels(N, n_subbands=1, rician_k_linear=0.0, generator=gen, device="cpu")
        n_all = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
        psnrs = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            tgt = val_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            out = model.forward_fas(
                val_stems[s:e].to(device=device, dtype=torch.float32),
                snr_db,
                n_subbands=1,
                rician_k_linear=0.0,
                h_port_override=h_all[s:e].to(device),
                standard_noise_iq=n_all[s:e].to(device),
                input_is_stem=True,
            )
            psnrs.extend(compute_psnr_batch(tgt, out["recon"]).cpu().tolist())
        scores[f"psnr_snr{int(snr_db)}"] = float(sum(psnrs) / len(psnrs))
    scores["mean_psnr_db"] = float(sum(scores.values()) / len(scores))
    return scores


def train_dir7a_single(
    port_mode: str,
    seed: int,
    steps: int,
    batch_size: int,
    val_every: int,
    backbone: torch.nn.Module,
    tr_u8: torch.Tensor,
    tr_stems: torch.Tensor,
    val_u8: torch.Tensor,
    val_stems: torch.Tensor,
    ckpt_root: Path,
    device: torch.device,
) -> None:
    out_dir = ckpt_root / f"dir7a_{port_mode}" / f"seed_{seed}"
    last_pt = out_dir / "last.pt"
    if last_pt.exists():
        print(f"[Skip] {last_pt} already exists.", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Byte-identical model initialization across `fas_kriging`, `fixed_port`, and `random_hop`!
    seed_everything(seed)
    model = BlindFASTransceiver(backbone, port_mode=port_mode).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)

    opt = torch.optim.AdamW(trainable_params, lr=3.5e-4, weight_decay=1e-4)
    gen_data = torch.Generator(device="cpu").manual_seed(seed + 1011)
    gen_chan = torch.Generator(device="cpu").manual_seed(seed + 2022)
    N_tr = tr_u8.shape[0]

    best_psnr = -1e9
    hist_rows = []
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        lr_now = 1.5e-5 + 0.5 * (3.5e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * float(step) / float(max(1, steps))))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        idx = torch.randint(0, N_tr, (batch_size,), generator=gen_data)
        tgt = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0
        stems = tr_stems[idx].to(device=device, dtype=torch.float32)

        snr_b = torch.empty((batch_size,), dtype=torch.float32).uniform_(0.0, 14.0, generator=gen_chan).to(device)
        h_b = sample_fas_spatial_channels(batch_size, n_subbands=1, rician_k_linear=0.0, generator=gen_chan, device=device)
        n_b = torch.randn((batch_size, N_SYM, N_SC, 2), generator=gen_chan).to(device)

        opt.zero_grad(set_to_none=True)
        out = model.forward_fas(
            stems, snr_b, n_subbands=1, rician_k_linear=0.0, h_port_override=h_b, standard_noise_iq=n_b, input_is_stem=True
        )
        rec = out["recon"]
        loss = F.l1_loss(rec, tgt) + 0.30 * F.mse_loss(rec, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        if step % val_every == 0 or step == steps:
            val_m = eval_dir7a_indomain_val256(model, val_stems, val_u8, device)
            cur_p = val_m["mean_psnr_db"]
            row = {
                "step": step,
                "train_loss": round(float(loss.item()), 6),
                "lr": round(lr_now, 7),
                **{k: round(v, 4) for k, v in val_m.items()},
                "elapsed_s": round(time.time() - t0, 1),
            }
            hist_rows.append(row)
            print(f"[Dir7A:{port_mode} s={seed}] step {step}/{steps} | val_psnr={cur_p:.3f} dB ({val_m})", flush=True)

            payload = {
                "track": "dir7a",
                "port_mode": port_mode,
                "seed": seed,
                "step": step,
                "n_trainable": n_trainable,
                "val_metrics": val_m,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
            }
            if cur_p > best_psnr:
                best_psnr = cur_p
                torch.save(payload, out_dir / "best_indist.pt")
            if step == steps:
                torch.save(payload, out_dir / "last.pt")

    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)
    save_json(
        out_dir / "run_manifest.json",
        {
            "track": "dir7a",
            "port_mode": port_mode,
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "n_trainable": n_trainable,
            "best_val256_psnr_db": best_psnr,
            "wall_time_s": round(time.time() - t0, 2),
        },
    )


def _make_deterministic_impulse_batch(
    batch_size: int,
    p_imp: float,
    imp_sir_db: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """CPU-generator deterministic fast-time clustered impulse tensor `[B, 48, 256, 2]` for exact paired training/val!"""
    if p_imp <= 0.0:
        return torch.zeros((batch_size, N_SYM, N_SC, 2), dtype=torch.float32)
    u1 = torch.rand((batch_size, N_SYM, N_SC), generator=gen)
    u2 = torch.rand((batch_size, N_SYM, N_SC), generator=gen)
    mask_seed = (u1 < (0.65 * float(p_imp))).to(torch.float32)
    mask_ring = torch.roll(mask_seed, shifts=1, dims=-1) * (u2 < 0.55).to(torch.float32)
    mask = (mask_seed + mask_ring).clamp(0.0, 1.0)[..., None]
    amp = math.pow(10.0, -float(imp_sir_db) / 20.0) / math.sqrt(2.0)
    g_iq = torch.randn((batch_size, N_SYM, N_SC, 2), generator=gen)
    return mask * amp * g_iq


@torch.inference_mode()
def eval_dir7c_indomain_val256(
    model: ImpulseShieldTransceiver,
    val_stems: torch.Tensor,
    val_u8: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Pure in-domain `val256` validation (`clean AWGN` + `6%` & `8%` impulsive; `12%` heavy burst is unseen OOD!)."""
    model.eval()
    N = val_u8.shape[0]
    bs = 32
    conds = [
        ("clean_snr14", 14.0, 0.00, -7.0),
        ("imp06_sir7_snr12", 12.0, 0.06, -7.0),
        ("imp08_sir8_snr10", 10.0, 0.08, -8.0),
    ]
    scores = {}
    for idx_c, (cname, snr_db, p_imp, sir_db) in enumerate(conds):
        gen = torch.Generator(device="cpu").manual_seed(2026730 + idx_c * 47)
        # Deterministic RB channels & noise on CPU
        z_hr = torch.randn((N, N_IMP_RBS), generator=gen)
        z_hi = torch.randn((N, N_IMP_RBS), generator=gen)
        rho_f = 0.88
        hr_all = torch.zeros_like(z_hr)
        hi_all = torch.zeros_like(z_hi)
        hr_all[:, 0] = z_hr[:, 0] * math.sqrt(0.5)
        hi_all[:, 0] = z_hi[:, 0] * math.sqrt(0.5)
        for m in range(1, N_IMP_RBS):
            hr_all[:, m] = rho_f * hr_all[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hr[:, m]
            hi_all[:, m] = rho_f * hi_all[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hi[:, m]
        awgn_all = torch.randn((N, N_SYM, N_SC, 2), generator=gen)
        imp_all = _make_deterministic_impulse_batch(N, p_imp, sir_db, gen)

        psnrs = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            tgt = val_u8[s:e].to(device=device, dtype=torch.float32) / 255.0
            out = model.forward_impulsive(
                val_stems[s:e].to(device=device, dtype=torch.float32),
                snr_db,
                p_imp=p_imp,
                imp_sir_db=sir_db,
                h_rb_override=(hr_all[s:e].to(device), hi_all[s:e].to(device)),
                awgn_td_override=awgn_all[s:e].to(device),
                impulse_td_override=imp_all[s:e].to(device),
                input_is_stem=True,
            )
            psnrs.extend(compute_psnr_batch(tgt, out["recon"]).cpu().tolist())
        scores[cname] = float(sum(psnrs) / len(psnrs))
    scores["mean_psnr_db"] = float(sum(scores.values()) / len(scores))
    return scores


def train_dir7c_single(
    rx_mode: str,
    seed: int,
    steps: int,
    batch_size: int,
    val_every: int,
    backbone: torch.nn.Module,
    tr_u8: torch.Tensor,
    tr_stems: torch.Tensor,
    val_u8: torch.Tensor,
    val_stems: torch.Tensor,
    ckpt_root: Path,
    device: torch.device,
) -> None:
    out_dir = ckpt_root / f"dir7c_{rx_mode}" / f"seed_{seed}"
    last_pt = out_dir / "last.pt"
    if last_pt.exists():
        print(f"[Skip] {last_pt} already exists.", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Byte-identical initialization between `sem_impulse_2round` and `huber_owndenoise`!
    seed_everything(seed)
    model = ImpulseShieldTransceiver(backbone, rx_mode=rx_mode).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)

    opt = torch.optim.AdamW(trainable_params, lr=3.5e-4, weight_decay=1e-4)
    gen_data = torch.Generator(device="cpu").manual_seed(seed + 3033)
    gen_chan = torch.Generator(device="cpu").manual_seed(seed + 4044)
    N_tr = tr_u8.shape[0]

    best_psnr = -1e9
    hist_rows = []
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        lr_now = 1.5e-5 + 0.5 * (3.5e-4 - 1.5e-5) * (1.0 + math.cos(math.pi * float(step) / float(max(1, steps))))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        idx = torch.randint(0, N_tr, (batch_size,), generator=gen_data)
        tgt = tr_u8[idx].to(device=device, dtype=torch.float32) / 255.0
        stems = tr_stems[idx].to(device=device, dtype=torch.float32)

        snr_b = torch.empty((batch_size,), dtype=torch.float32).uniform_(6.0, 18.0, generator=gen_chan).to(device)
        # Paired curriculum: 25% clean AWGN (`p_imp=0`), 25% mild (`p_imp=0.03`), 50% moderate (`p_imp in [0.05, 0.08]`)
        coin = float(torch.rand((1,), generator=gen_chan).item())
        if coin < 0.25:
            p_imp_s, sir_s = 0.0, -7.0
        elif coin < 0.50:
            p_imp_s, sir_s = 0.03, -5.5
        else:
            p_imp_s = float(torch.empty((1,)).uniform_(0.05, 0.085, generator=gen_chan).item())
            sir_s = float(torch.empty((1,)).uniform_(-9.5, -4.5, generator=gen_chan).item())

        z_hr = torch.randn((batch_size, N_IMP_RBS), generator=gen_chan)
        z_hi = torch.randn((batch_size, N_IMP_RBS), generator=gen_chan)
        rho_f = 0.88
        hr_b = torch.zeros_like(z_hr)
        hi_b = torch.zeros_like(z_hi)
        hr_b[:, 0] = z_hr[:, 0] * math.sqrt(0.5)
        hi_b[:, 0] = z_hi[:, 0] * math.sqrt(0.5)
        for m in range(1, N_IMP_RBS):
            hr_b[:, m] = rho_f * hr_b[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hr[:, m]
            hi_b[:, m] = rho_f * hi_b[:, m - 1] + math.sqrt((1.0 - rho_f**2) * 0.5) * z_hi[:, m]

        awgn_b = torch.randn((batch_size, N_SYM, N_SC, 2), generator=gen_chan)
        imp_b = _make_deterministic_impulse_batch(batch_size, p_imp_s, sir_s, gen_chan)

        opt.zero_grad(set_to_none=True)
        out = model.forward_impulsive(
            stems,
            snr_b,
            p_imp=p_imp_s,
            imp_sir_db=sir_s,
            h_rb_override=(hr_b.to(device), hi_b.to(device)),
            awgn_td_override=awgn_b.to(device),
            impulse_td_override=imp_b.to(device),
            input_is_stem=True,
        )
        rec = out["recon"]
        pix_loss = F.l1_loss(rec, tgt) + 0.30 * F.mse_loss(rec, tgt)
        z_rounds = out["z_denoised_rounds"]
        if len(z_rounds) > 0:
            z_tgt = out["z_tx"].detach()
            lat_loss = sum(F.mse_loss(zr, z_tgt) for zr in z_rounds) / float(len(z_rounds))
            loss = pix_loss + 0.08 * lat_loss
        else:
            loss = pix_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

        if step % val_every == 0 or step == steps:
            val_m = eval_dir7c_indomain_val256(model, val_stems, val_u8, device)
            cur_p = val_m["mean_psnr_db"]
            row = {
                "step": step,
                "train_loss": round(float(loss.item()), 6),
                "lr": round(lr_now, 7),
                **{k: round(v, 4) for k, v in val_m.items()},
                "elapsed_s": round(time.time() - t0, 1),
            }
            hist_rows.append(row)
            print(f"[Dir7C:{rx_mode} s={seed}] step {step}/{steps} | val_psnr={cur_p:.3f} dB ({val_m})", flush=True)

            payload = {
                "track": "dir7c",
                "rx_mode": rx_mode,
                "seed": seed,
                "step": step,
                "n_trainable": n_trainable,
                "val_metrics": val_m,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
            }
            if cur_p > best_psnr:
                best_psnr = cur_p
                torch.save(payload, out_dir / "best_indist.pt")
            if step == steps:
                torch.save(payload, out_dir / "last.pt")

    with (out_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)
    save_json(
        out_dir / "run_manifest.json",
        {
            "track": "dir7c",
            "rx_mode": rx_mode,
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "n_trainable": n_trainable,
            "best_val256_psnr_db": best_psnr,
            "wall_time_s": round(time.time() - t0, 2),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["dir7a", "dir7c"], required=True)
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="1001,2002,3003")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--val_every", type=int, default=250)
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
    args = parser.parse_args()

    configure_runtime_env(args.tmp_root)
    configure_torch_determinism()
    admit_gpu(min_free_mib=22000)
    device = torch.device("cuda:0")

    swin_cache = ensure_swin_cache(args.tmp_root / "cache", device)
    cache_blob = torch.load(swin_cache, map_location="cpu", weights_only=True)
    tr_stems = cache_blob["train_stems_fp16"]
    val_stems = cache_blob["val_stems_fp32"]
    tr_u8 = load_split_u8("train20k").images_u8
    val_u8 = load_split_u8("val256").images_u8

    backbone = load_swinjscc_backbone(SWIN_AWGN_CKPT, model_mode="SwinJSCC_w/_SA", c_bottleneck=96, device=device)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    if args.track == "dir7a":
        modes = (
            ["fas_kriging", "fixed_port", "random_hop"]
            if args.models == "all"
            else [m.strip() for m in args.models.split(",") if m.strip()]
        )
        for sd in seeds:
            for pm in modes:
                train_dir7a_single(
                    pm, sd, args.steps, args.batch_size, args.val_every, backbone, tr_u8, tr_stems, val_u8, val_stems, args.ckpt_root, device
                )
    else:
        modes = (
            ["sem_impulse_2round", "huber_owndenoise", "classical_blanking", "standard_blind"]
            if args.models == "all"
            else [m.strip() for m in args.models.split(",") if m.strip()]
        )
        for sd in seeds:
            for rm in modes:
                train_dir7c_single(
                    rm, sd, args.steps, args.batch_size, args.val_every, backbone, tr_u8, tr_stems, val_u8, val_stems, args.ckpt_root, device
                )

    verify_no_links([args.ckpt_root, args.tmp_root])


if __name__ == "__main__":
    main()
