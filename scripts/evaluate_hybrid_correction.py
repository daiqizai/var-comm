#!/usr/bin/env python3
"""Original development only; frozen calibration selection and strong references."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from evaluate_communication_modes import Population
from var_comm.hybrid_correction import encode_digital, ResidualLink
from var_comm.hybrid_training import DigitalRenderer, evaluate
from var_comm.next_scale_prior import load_models
from var_comm.progressive import complete_image
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.study import seeded_noise, sha256, snapshot, verify_snapshot, write_csv, write_json

HISTORY = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"


class Development:
    def __init__(self, vae, var, device, limit):
        original = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
        self.ids, self.labels, self.images, self.bases, self.tokens, self.digital = [], [], [], [], [], []
        self.original_bindings = original.bindings
        for index in range(limit):
            pixels, scales, target = original.source(index, vae, device)
            tokens = np.concatenate(scales[:7])
            label = int(target["class_index"])
            self.ids.append(target["image_id"])
            self.labels.append(label)
            self.images.append(pixels)
            self.bases.append(complete_image(vae, var, scales[:7], label, device))
            self.tokens.append(tokens)
            self.digital.append(encode_digital(tokens, label))
        self.labels = np.asarray(self.labels)
        self.images = torch.from_numpy(np.stack(self.images))
        self.bases, self.tokens, self.digital = np.stack(self.bases), np.stack(self.tokens), np.stack(self.digital)


def frozen_references(population, config):
    policies = json.loads((HISTORY / "POLICIES_001/policies.json").read_text())["actions"]
    lookup = {identifier: index for index, identifier in enumerate(population.ids)}
    hashes = {identifier: hashlib.sha256(population.images[index].numpy().tobytes()).hexdigest()
              for identifier, index in lookup.items()}
    rows = []
    matrix = HISTORY / "DEVELOPMENT_001/per_frame.csv"
    receipt = json.loads((HISTORY / "DEVELOPMENT_001/completion.json").read_text())
    expected = receipt.get("output_hashes", {}).get("per_frame.csv")
    if expected is not None and sha256(matrix) != expected:
        raise RuntimeError("frozen development rows changed")
    with matrix.open(newline="") as handle:
        for source in csv.DictReader(handle):
            identifier, snr = source["image_id"], float(source["snr_db"])
            if identifier not in lookup or snr not in config["snrs_db"]:
                continue
            if source["source_pixels_sha256"] != hashes[identifier] or int(source["complex_uses"]) != 3060 or float(source["total_energy"]) != 6120:
                raise RuntimeError("digital source pixels or physical ledger do not match")
            family = source["family"]
            names = [f"raw_m{source['mode']}"] if family == "raw" else []
            if int(source["mode"]) == policies[family]["quality"][source["snr_db"]]:
                names.append(f"{family}_adaptive")
            for arm in names:
                rows.append({"image_index": lookup[identifier], "image_id": identifier, "snr_db": snr,
                             "seed": int(source["seed"]), "arm": arm, "psnr_db": float(source["psnr_db"]),
                             "mse": 10 ** (-float(source["psnr_db"]) / 10), "lpips": float(source["lpips"]),
                             "dino": float(source["dino"]), "complex_uses": 3060, "total_energy": 6120})
    deep_root = ROOT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
    deep_receipt = json.loads((deep_root / "completion.json").read_text())
    if sha256(deep_root / "per_frame.csv") != deep_receipt["output_hashes"]["per_frame.csv"]:
        raise RuntimeError("frozen Deep reference rows changed")
    with (deep_root / "per_frame.csv").open(newline="") as handle:
        for source in csv.DictReader(handle):
            identifier, snr = source["image_id"], float(source["snr_db"])
            if identifier not in lookup or snr not in config["snrs_db"] or source["arm"] != "perceptual_deepjscc":
                continue
            noise = seeded_noise(identifier, int(source["seed"]), (3060, 2))
            if hashlib.sha256(noise.tobytes()).hexdigest() != source["noise_sha256"]:
                raise RuntimeError("strong Deep did not receive the same standard noise")
            if int(source["total_complex_uses"]) != 3060 or abs(float(source["data_power"]) - 2) > 1e-5:
                raise RuntimeError("Deep reference physical budget differs")
            rows.append({"image_index": lookup[identifier], "image_id": identifier, "snr_db": snr,
                         "seed": int(source["seed"]), "arm": "perceptual_deepjscc", "psnr_db": float(source["psnr_db"]),
                         "mse": 10 ** (-float(source["psnr_db"]) / 10), "lpips": float(source["lpips_alex"]),
                         "dino": float(source["dino_cosine"]), "complex_uses": 3060, "total_energy": 6120})
    required = {"raw_adaptive", "arithmetic_adaptive", "raw_m7", "raw_m8", "raw_m9", "perceptual_deepjscc"}
    if {row["arm"] for row in rows} != required:
        raise RuntimeError("missing a required strong resource-allocation control")
    expected_keys = {(identifier, snr, seed) for identifier in lookup for snr in config["snrs_db"] for seed in config["development_seeds"]}
    for arm in required:
        selected = [row for row in rows if row["arm"] == arm]
        keys = {(row["image_id"], row["snr_db"], row["seed"]) for row in selected}
        if len(selected) != len(expected_keys) or keys != expected_keys:
            raise RuntimeError("incomplete or duplicate paired strong-reference frames")
    return rows, {str(matrix): sha256(matrix), str(deep_root / "per_frame.csv"): sha256(deep_root / "per_frame.csv"),
                  str(HISTORY / "POLICIES_001/policies.json"): sha256(HISTORY / "POLICIES_001/policies.json")}


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
    metadata = json.loads((arguments.training / "metadata.json").read_text())
    verify_snapshot(metadata["bindings"])
    if arguments.mode == "full":
        selected = json.loads((arguments.training / "selection.json").read_text())["selected"]
        if not (arguments.training / "completion.json").exists():
            raise RuntimeError("formal development must follow completed training/calibration")
    else:
        checkpoint = torch.load(arguments.training / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        selected = {arm: {"step": checkpoint["step"], "feasible_on_calibration": False} for arm in config["arms"]}
        del checkpoint
    sources = snapshot(output, [Path(__file__), ROOT / "src/var_comm/hybrid_training.py", ROOT / "src/var_comm/hybrid_correction.py"])
    write_json(output / "selection_frozen_before_development.json", {"selected": selected,
                "mode": arguments.mode, "training_metadata_sha256": sha256(arguments.training / "metadata.json")})
    device = torch.device("cuda:0")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    vae, var = load_models(paths, device)
    perceptual, dino, _weights = load_quality_models(quality_paths, device)
    models, weights = {}, {}
    for arm in config["arms"]:
        path = arguments.training / "checkpoints" / f"step_{selected[arm]['step']:07d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ResidualLink(config["model"], arm).to(device).eval()
        model.load_state_dict(checkpoint["models"][arm], strict=True)
        models[arm] = model
        weights[str(path.resolve())] = sha256(path)
        del checkpoint
    limit = 100 if arguments.mode == "full" else 2
    population = Development(vae, var, device, limit)
    references, reference_bindings = frozen_references(population, config)
    write_csv(output / "frozen_reference_rows.csv", references)
    write_json(output / "bindings.json", {"models": weights, "references": reference_bindings,
                                         "data": population.original_bindings})
    image_directory = output / "images"
    image_directory.mkdir()
    for index in range(limit):
        np.save(image_directory / f"source_{index:04d}.npy", population.images[index].numpy())
    renderer = DigitalRenderer(vae, var, device, (population,))
    base_rows, seen = [], set()

    def save_image(row, image, base):
        frame = f"{row['image_index']:04d}_{row['snr_db']:g}_{row['seed']}"
        path = image_directory / f"{row['arm']}_{frame}.npy"
        np.save(path, image)
        row["image_path"] = str(path.relative_to(output))
        row["image_sha256"] = sha256(path)
        row["calibration_feasible"] = selected[row["arm"]]["feasible_on_calibration"]
        if frame not in seen:
            seen.add(frame)
            base_path = image_directory / f"shortened_base_{frame}.npy"
            np.save(base_path, base)
            original = population.images[row["image_index"]].numpy().astype(np.float32) / 255
            metrics, _source, _features = quality_metrics(original, [base], perceptual, dino, device)
            base_rows.append({"image_id": row["image_id"], "image_index": row["image_index"], "snr_db": row["snr_db"],
                              "seed": row["seed"], "arm": "shortened_base_diagnostic", "psnr_db": metrics[0]["psnr_db"],
                              "lpips": metrics[0]["lpips_alex"], "dino": metrics[0]["dino_cosine"],
                              "mse": float(np.mean((base - original) ** 2)), "complex_uses": 3060, "total_energy": 6120,
                              "diagnostic_ignores_already_paid_analog_branch": True})

    rows = evaluate(models, population, np.arange(limit), config, renderer, perceptual, device, dino, save_image)
    write_csv(output / "per_frame.csv", rows)
    write_csv(output / "shortened_base_diagnostic.csv", base_rows)
    verify_snapshot(metadata["bindings"])
    verify_snapshot(sources)
    for path, expected in weights.items():
        if sha256(path) != expected:
            raise RuntimeError("selected checkpoint changed during development")
    write_json(output / "completion.json", {"status": "DEVELOPMENT_COMPLETE" if arguments.mode == "full" else "ENGINEERING_REPLAY_COMPLETE",
               "sources": limit, "new_rows": len(rows), "frozen_reference_rows": len(references),
               "new_holdout_accessed": False, "per_frame_sha256": sha256(output / "per_frame.csv"),
               "core_hypothesis_proven": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
