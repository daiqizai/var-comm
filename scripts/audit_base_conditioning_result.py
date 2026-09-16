#!/usr/bin/env python3
"""Read-only post-completion PHY/metric verification and fixed preview figures."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from evaluate_communication_modes import Population
from var_comm.hybrid_correction import encode_digital, receive_digital
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import seeded_noise, sha256, write_json


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def run(arguments):
    root = arguments.run.resolve()
    if not (root / "pipeline_001/completion.json").exists():
        raise RuntimeError("this audit cannot run before the registered pair finishes")
    output = root / "report_assets_001"
    output.mkdir(exist_ok=False)
    assert_gpu_available()
    config = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
    binding = json.loads((root / "pipeline_001/bindings.json").read_text())
    if any(sha256(path) != expected for path, expected in binding.items()):
        raise RuntimeError("a registered input implementation changed")
    quality = root / "development_001"
    rows = read_rows(quality / "per_frame.csv")
    arms = config["arms"] + ["conditioned_spatial_shuffle"]
    lookup = {(row["arm"], *key(row)): row for row in rows}
    if len(rows) != 4500 or len(lookup) != 4500:
        raise RuntimeError("complete pair/ablation coverage is missing or duplicated")
    expected = {(index, snr, seed) for index in range(100) for snr in config["snrs_db"] for seed in config["development_seeds"]}
    for arm in arms:
        if {key(row) for row in rows if row["arm"] == arm} != expected:
            raise RuntimeError("a failed frame or registered SNR/source was removed")
    original = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    for name in ("alexnet_checkpoint", "dino_checkpoint"):
        if sha256(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError("frozen metric weights changed")
    perceptual, dino, _weights = load_quality_models(paths, device)
    maximum = {"psnr_db": 0., "lpips": 0., "dino": 0., "CPU_mse": 0.}
    limits = {"psnr_db": 1e-4, "lpips": 1e-5, "dino": 1e-5, "CPU_mse": 1e-7}
    for index in range(100):
        pixels, scales, target = original.source(index, None, None)
        np.testing.assert_array_equal(pixels, np.load(quality / "images" / f"source_{index:04d}.npy", allow_pickle=False))
        signal = encode_digital(np.concatenate(scales[:7]), int(target["class_index"]))
        for snr in config["snrs_db"]:
            for seed in config["development_seeds"]:
                received = signal + seeded_noise(target["image_id"], seed, (3060, 2))[:1950] / np.sqrt(10 ** (snr / 10))
                decoded = receive_digital(received, snr)
                correct = bool(decoded["header_usable"] and decoded["label"] == int(target["class_index"]) and np.array_equal(decoded["tokens"], np.concatenate(scales[:7])))
                events = {"header_usable": decoded["header_usable"], "body_crc_accepted": decoded["crc_accepted"],
                          "source_correct": correct, "accepted_correct": correct and decoded["crc_accepted"],
                          "false_acceptance": decoded["crc_accepted"] and not correct}
                for arm in arms:
                    row = lookup[arm, index, snr, seed]
                    if any((row[field] == "True") != bool(value) for field, value in events.items()):
                        raise RuntimeError("actual PHY events differ from archived observations")
                    if row["received_digital_sha256"] != hashlib.sha256(received.tobytes()).hexdigest():
                        raise RuntimeError("receiver did not use the original registered digital observation")
                    if int(row["complex_uses"]) != 3060 or abs(float(row["total_energy"]) - 6120) > .01:
                        raise RuntimeError("physical budget mismatch")
        selected = [row for row in rows if int(row["image_index"]) == index]
        images = []
        for row in selected:
            path = quality / row["image_path"]
            if sha256(path) != row["image_sha256"]:
                raise RuntimeError("stored image hash changed")
            image = np.load(path, allow_pickle=False)
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise RuntimeError("invalid reconstructed image")
            if row["header_usable"] == "False" and not np.all(image == .5):
                raise RuntimeError("header failure received free information")
            images.append(image)
        reference = pixels.astype(np.float32) / 255
        recalculated, _original_feature, _features = quality_metrics(reference, images, perceptual, dino, device)
        for row, scores, image in zip(selected, recalculated, images):
            for stored, measured in (("psnr_db", "psnr_db"), ("lpips", "lpips_alex"), ("dino", "dino_cosine")):
                if not np.isfinite(scores[measured]):
                    raise RuntimeError("nonfinite recomputed quality")
                maximum[stored] = max(maximum[stored], abs(float(row[stored]) - scores[measured]))
            mse = float(np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2))
            maximum["CPU_mse"] = max(maximum["CPU_mse"], abs(mse - float(row["mse"])))
        if any(maximum[name] > limits[name] for name in maximum):
            raise RuntimeError(f"independent metric recomputation differs: {maximum}")
        if (index + 1) % 10 == 0:
            print(f"verified {index+1}/100 sources and {(index+1)*45} images", flush=True)
    write_json(output / "audit.json", {"status": "PASS", "PHY_replays": 1500, "images": 4500,
               "normal_receiver_images": 3000, "context_ablation_images": 1500, "source_images": 100,
               "metric_max_errors": maximum, "tolerances": limits, "script_sha256": sha256(Path(__file__)),
               "frozen_implementation_unchanged": True, "new_training_or_holdout": False})
    del perceptual, dino
    history = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"
    digital = read_rows(history / "DEVELOPMENT_001/per_frame.csv")
    policy = json.loads((history / "POLICIES_001/policies.json").read_text())["actions"]
    deep_root = ROOT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
    deep = [row for row in read_rows(deep_root / "per_frame.csv") if row["arm"] == "perceptual_deepjscc"]
    methods = ["source", "unconditioned", "conditioned", "conditioned_spatial_shuffle", "raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc"]
    for snr in (1., 7., 19.):
        figure, axes = plt.subplots(4, 7, figsize=(17.5, 10))
        for row_index, index in enumerate(config["visual_preview_indices"]):
            for column, method in enumerate(methods):
                score = None
                if method == "source":
                    image = np.load(quality / "images" / f"source_{index:04d}.npy").astype(np.float32) / 255
                elif method in arms:
                    record = lookup[method,index,snr,2001]
                    image = np.load(quality / record["image_path"], allow_pickle=False)
                    score = float(record["psnr_db"]), float(record["lpips"])
                elif method in ("raw_adaptive", "arithmetic_adaptive"):
                    family = method.split("_")[0]
                    mode = policy[family]["quality"][str(snr)]
                    record = next(row for row in digital if key(row)==(index,snr,2001) and row["family"]==family and int(row["mode"])==mode)
                    with np.load(history / "DEVELOPMENT_001/images" / f"{index:04d}" / "reconstructions.npz", allow_pickle=False) as archive:
                        image = archive["images"][int(record["image_index_in_archive"])]
                    score = float(record["psnr_db"]), float(record["lpips"])
                else:
                    record = next(row for row in deep if key(row)==(index,snr,2001))
                    if not record["image_ref"].startswith("new:"):
                        raise RuntimeError("unexpected frozen Deep image reference")
                    with np.load(deep_root / "images" / f"{index:03d}" / "reconstructions.npz", allow_pickle=False) as archive:
                        image = archive["images"][int(record["image_ref"].split(":")[1])]
                    score = float(record["psnr_db"]), float(record["lpips_alex"])
                axis = axes[row_index,column]
                axis.imshow(np.asarray(image).transpose(1,2,0).clip(0,1), interpolation="nearest")
                axis.set_axis_off()
                title = f"source {index}" if score is None else f"{method}\nPSNR {score[0]:.2f} | LPIPS {score[1]:.3f}"
                axis.set_title(title, fontsize=7)
        figure.suptitle(f"Fixed sources [0,25,50,75] · {snr:g} dB · N3060/E6120 · shuffled context is diagnostic only")
        figure.tight_layout()
        figure.savefig(output / f"fixed_examples_{snr:g}dB.png", dpi=180)
        plt.close(figure)
    write_json(output / "completion.json", {"status": "CONDITIONING_RESULT_AUDIT_AND_FIXED_FIGURES_COMPLETE",
               "finished_at": datetime.now().astimezone().isoformat(), "audit_sha256": sha256(output / "audit.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args())
