#!/usr/bin/env python3
"""Evaluate official and decoder-only-tuned VAR decoders on frozen ImageNet-100."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.var_decoder_tuning import (  # noqa: E402
    CONDITIONS,
    bootstrap_mean_ci,
    build_vae_only,
    build_vae_var,
    dino_cls,
    generate_condition_fhats,
    json_dump,
    load_config,
    load_dino,
    paired_win_rates,
    preprocess_path,
    psnr_from_mse,
    require_sha,
    sha256_file,
    state_sha256,
)


SCRIPT = Path(__file__).resolve()
DEFAULT_CONFIG = ROOT / "configs/var_decoder_only_m89_imagenet.yaml"
METRICS = ("psnr_db", "ssim", "lpips_alex", "dino_cosine")
HIGHER_IS_BETTER = {
    "psnr_db": True,
    "ssim": True,
    "lpips_alex": False,
    "dino_cosine": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_lpips(device: torch.device) -> torch.nn.Module:
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)


def save_image(tensor_01: torch.Tensor, path: Path) -> None:
    array = (
        tensor_01.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def decode_conditions(vae: torch.nn.Module, fhats: torch.Tensor) -> torch.Tensor:
    batch_size = fhats.shape[0]
    flat = fhats.reshape(batch_size * len(CONDITIONS), 32, 16, 16).float()
    with torch.inference_mode():
        output = vae.decoder(vae.post_quant_conv(flat)).clamp(-1, 1)
    return output.add(1).mul(0.5).reshape(batch_size, len(CONDITIONS), 3, 256, 256)


def calculate_metrics(
    source_01: torch.Tensor,
    reconstructions_01: torch.Tensor,
    *,
    lpips_model: torch.nn.Module,
    dino_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    from pytorch_msssim import ssim

    batch_size, arms = reconstructions_01.shape[:2]
    flat = reconstructions_01.reshape(batch_size * arms, 3, 256, 256)
    target = source_01[:, None].expand(-1, arms, -1, -1, -1).reshape_as(flat)
    with torch.inference_mode():
        mse = (flat - target).square().flatten(1).mean(1)
        psnr = psnr_from_mse(mse)
        ssim_value = ssim(flat, target, data_range=1.0, size_average=False)
        lpips_value = lpips_model(flat.mul(2).sub(1), target.mul(2).sub(1)).flatten()
        features = dino_cls(dino_model, torch.cat((source_01, flat), dim=0))
        reference = features[:batch_size]
        estimate = features[batch_size:].reshape(batch_size, arms, -1)
        dino = F.cosine_similarity(
            estimate, reference[:, None].expand_as(estimate), dim=-1
        ).reshape(-1)
    return {
        "psnr_db": psnr.reshape(batch_size, arms).cpu(),
        "ssim": ssim_value.reshape(batch_size, arms).cpu(),
        "lpips_alex": lpips_value.reshape(batch_size, arms).cpu(),
        "dino_cosine": dino.reshape(batch_size, arms).cpu(),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    paired = []
    for method_index, method in enumerate(("official", "tuned")):
        for condition_index, condition in enumerate(CONDITIONS):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["condition"] == condition
            ]
            item: dict[str, Any] = {
                "method": method,
                "condition": condition,
                "n_images": len(selected),
            }
            for metric_index, metric in enumerate(METRICS):
                values = [float(row[metric]) for row in selected]
                mean, low, high = bootstrap_mean_ci(
                    values,
                    resamples=resamples,
                    seed=seed + 1000 * method_index + 100 * condition_index + metric_index,
                )
                item[f"{metric}_mean"] = mean
                item[f"{metric}_ci95_low"] = low
                item[f"{metric}_ci95_high"] = high
            summary.append(item)

    for condition_index, condition in enumerate(CONDITIONS):
        official = {
            str(row["image_id"]): row
            for row in rows
            if row["method"] == "official" and row["condition"] == condition
        }
        tuned = {
            str(row["image_id"]): row
            for row in rows
            if row["method"] == "tuned" and row["condition"] == condition
        }
        if official.keys() != tuned.keys():
            raise RuntimeError(f"paired key mismatch for {condition}")
        ids = sorted(official)
        for metric_index, metric in enumerate(METRICS):
            tuned_values = np.asarray([float(tuned[key][metric]) for key in ids])
            official_values = np.asarray([float(official[key][metric]) for key in ids])
            differences = tuned_values - official_values
            mean, low, high = bootstrap_mean_ci(
                differences,
                resamples=resamples,
                seed=seed + 10000 + 100 * condition_index + metric_index,
            )
            rates = paired_win_rates(
                tuned_values,
                official_values,
                higher_is_better=HIGHER_IS_BETTER[metric],
            )
            paired.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "delta_definition": "tuned_minus_official",
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "mean_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    **rates,
                }
            )
    return summary, paired


def plot_paired(paired: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for axis, metric in zip(axes.flat, METRICS):
        selected = [row for row in paired if row["metric"] == metric]
        values = np.asarray([float(row["mean_delta"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        axis.errorbar(
            np.arange(len(CONDITIONS)),
            values,
            yerr=np.vstack((values - low, high - values)),
            marker="o",
            capsize=4,
        )
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(np.arange(len(CONDITIONS)), CONDITIONS)
        axis.set_ylabel(f"tuned − official {metric}")
        axis.grid(True, alpha=0.25)
    figure.suptitle("Decoder-only fine-tuning: paired ImageNet-100 differences")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_grid(root: Path, *, count: int, path: Path) -> None:
    labels = ["source"] + [f"official {name}" for name in CONDITIONS] + [
        f"tuned {name}" for name in CONDITIONS
    ]
    columns = len(labels)
    label_height = 24
    canvas = Image.new("RGB", (columns * 256, count * (256 + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    for row_index in range(count):
        sample = root / f"sample_{row_index:03d}"
        files = [sample / "source.png"] + [
            sample / f"official_{name}.png" for name in CONDITIONS
        ] + [sample / f"tuned_{name}.png" for name in CONDITIONS]
        for column, (label, file_path) in enumerate(zip(labels, files)):
            with Image.open(file_path) as image:
                canvas.paste(
                    image.convert("RGB"),
                    (column * 256, row_index * (256 + label_height) + label_height),
                )
            draw.text(
                (column * 256 + 4, row_index * (256 + label_height) + 4),
                label,
                fill="black",
                font=font,
            )
    canvas.save(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    train_root = (ROOT / str(config["outputs"]["train"])).resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else train_root / "checkpoints/best.pt"
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (ROOT / str(config["outputs"]["test"])).resolve()
    )
    samples_path = require_sha(
        Path(config["paths"]["test_samples"]),
        str(config["paths"]["test_samples_sha256"]),
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if len(samples) != int(config["population"]["test"]["count"]):
        raise RuntimeError("frozen ImageNet-100 count changed")
    if args.limit is not None:
        samples = samples[: args.limit]
    plan = {
        "checkpoint": str(checkpoint_path),
        "output": str(output),
        "images": len(samples),
        "conditions": list(CONDITIONS),
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != config["experiment_ids"]["train"]:
        raise RuntimeError("decoder checkpoint experiment ID changed")
    if checkpoint.get("official_vae_checkpoint_sha256") != config["paths"]["vae_checkpoint_sha256"]:
        raise RuntimeError("decoder checkpoint official VAE parent changed")
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(output if output.exists() else building)
    building.mkdir(parents=True, exist_ok=False)
    representatives = building / "representatives"
    representatives.mkdir()
    shutil.copy2(config_path, building / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, building / SCRIPT.name)
    shutil.copy2(ROOT / "src/cadsd_jscc/var_decoder_tuning.py", building / "var_decoder_tuning_snapshot.py")
    json_dump(building / "selected_samples.json", samples)

    device = torch.device(args.device)
    paths = config["paths"]
    vae_checkpoint = require_sha(Path(paths["vae_checkpoint"]), str(paths["vae_checkpoint_sha256"]))
    var_checkpoint = require_sha(Path(paths["var_checkpoint"]), str(paths["var_checkpoint_sha256"]))
    dino_checkpoint = require_sha(Path(paths["dino_checkpoint"]), str(paths["dino_checkpoint_sha256"]))
    print("loading metrics and frozen models", flush=True)
    dino_model = load_dino(Path(paths["dino_source"]), dino_checkpoint, device)
    lpips_model = make_lpips(device)
    official_vae, var = build_vae_var(
        Path(paths["var_source"]), vae_checkpoint, var_checkpoint, device
    )
    tuned_vae = build_vae_only(Path(paths["var_source"]), vae_checkpoint, device)
    incompatible = tuned_vae.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"tuned decoder checkpoint mismatch: {incompatible}")
    if state_sha256(official_vae, exclude_prefix="decoder.") != state_sha256(
        tuned_vae, exclude_prefix="decoder."
    ):
        raise RuntimeError("official and tuned non-decoder VQ-VAE state differs")
    official_vae.eval().requires_grad_(False)
    tuned_vae.eval().requires_grad_(False)

    rows: list[dict[str, Any]] = []
    representative_count = min(int(config["evaluation"]["representative_images"]), len(samples))
    started = time.perf_counter()
    for batch_start in range(0, len(samples), args.batch_size):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        source_m11 = torch.stack(
            [preprocess_path(Path(str(sample["path"]))) for sample in batch_samples]
        ).to(device)
        source_01 = source_m11.add(1).mul(0.5)
        labels = torch.as_tensor(
            [int(sample["class_index"]) for sample in batch_samples],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            fhats = generate_condition_fhats(official_vae, var, source_m11, labels)
            official = decode_conditions(official_vae, fhats)
            tuned = decode_conditions(tuned_vae, fhats)
        combined = torch.cat((official, tuned), dim=1)
        metrics = calculate_metrics(
            source_01,
            combined,
            lpips_model=lpips_model,
            dino_model=dino_model,
        )
        for local_index, sample in enumerate(batch_samples):
            global_index = batch_start + local_index
            if global_index < representative_count:
                sample_root = representatives / f"sample_{global_index:03d}"
                sample_root.mkdir()
                save_image(source_01[local_index], sample_root / "source.png")
                for condition_index, condition in enumerate(CONDITIONS):
                    save_image(
                        official[local_index, condition_index],
                        sample_root / f"official_{condition}.png",
                    )
                    save_image(
                        tuned[local_index, condition_index],
                        sample_root / f"tuned_{condition}.png",
                    )
            for method_index, method in enumerate(("official", "tuned")):
                for condition_index, condition in enumerate(CONDITIONS):
                    arm_index = method_index * len(CONDITIONS) + condition_index
                    rows.append(
                        {
                            "image_index": global_index,
                            "image_id": str(sample["image_id"]),
                            "path": str(sample["path"]),
                            "class_index": int(sample["class_index"]),
                            "synset": str(sample["synset"]),
                            "method": method,
                            "condition": condition,
                            **{
                                metric: float(metrics[metric][local_index, arm_index])
                                for metric in METRICS
                            },
                        }
                    )
        write_csv(building / "per_image_metrics.partial.csv", rows)
        print(
            f"completed {min(batch_start + len(batch_samples), len(samples))}/{len(samples)}; "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    summary, paired = summarize_rows(
        rows,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    write_csv(building / "per_image_metrics.csv", rows)
    write_csv(building / "mean_metrics.csv", summary)
    write_csv(building / "paired_differences_and_win_rates.csv", paired)
    plot_paired(paired, building / "paired_decoder_differences.png")
    build_grid(representatives, count=representative_count, path=building / "representative_grid.png")
    completion = {
        "status": "complete",
        "analysis_id": config["experiment_ids"]["test"],
        "images": len(samples),
        "rows": len(rows),
        "conditions": list(CONDITIONS),
        "metrics": list(METRICS),
        "elapsed_seconds": time.perf_counter() - started,
        "decoder_checkpoint": str(checkpoint_path),
        "decoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": int(checkpoint["epoch"]),
        "official_vae_checkpoint_sha256": sha256_file(vae_checkpoint),
        "var_checkpoint_sha256": sha256_file(var_checkpoint),
        "test_samples_sha256": sha256_file(samples_path),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(SCRIPT),
        "primary_claim_scope": "paired_official_vs_tuned_same_latent_decoder_ablation",
    }
    json_dump(building / "summary.json", {"completion": completion, "means": summary, "paired": paired})
    json_dump(building / "completion.json", completion)
    building.rename(output)
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
