#!/usr/bin/env python3
"""Reuse the four frozen systems; add only common SSIM and image-input manifests."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "scripts")]

import numpy as np
import torch
from pytorch_msssim import ssim

from evaluate_communication_modes import Population
from var_comm.prefix_training_data import read_image_population
from var_comm.study import seeded_noise, sha256, write_csv, write_json

SNRS = (1., 4., 7., 13., 19.)
SEEDS = (2001, 2002, 2003)
CALIBRATION_INDICES = (0, 125, 250, 375, 500, 625, 750, 999)


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    history = PROJECT / "outputs/COMMUNICATION-CONVERGENCE-20260915"
    digital_path = history / "DEVELOPMENT_001/per_frame.csv"
    deep_root = PROJECT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
    r3_root = PROJECT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000"
    paths = (digital_path, deep_root / "per_frame.csv", r3_root / "per_frame.csv", history / "POLICIES_001/policies.json")
    bindings = {str(path): sha256(path) for path in paths}
    policies = json.loads(paths[-1].read_text())["actions"]
    selected = []
    for row in read_rows(digital_path):
        snr = float(row["snr_db"])
        if snr not in SNRS or int(row["mode"]) != policies[row["family"]]["quality"][row["snr_db"]]:
            continue
        if int(row["complex_uses"]) != 3060 or float(row["total_energy"]) != 6120:
            raise RuntimeError("frozen digital budget differs")
        selected.append({"method": row["family"] + "_adaptive", "row": row,
                         "archive": history / "DEVELOPMENT_001/images" / f"{int(row['image_index']):04d}/reconstructions.npz",
                         "image_slot": int(row["image_index_in_archive"]), "psnr_db": float(row["psnr_db"]),
                         "lpips": float(row["lpips"]), "dino": float(row["dino"]), "previous_ssim": None})
    for row in read_rows(deep_root / "per_frame.csv"):
        if row["arm"] != "perceptual_deepjscc" or float(row["snr_db"]) not in SNRS:
            continue
        if not row["image_ref"].startswith("new:") or int(row["total_complex_uses"]) != 3060 or abs(float(row["data_power"]) - 2) > 1e-5:
            raise RuntimeError("frozen Deep image reference or resource ledger differs")
        selected.append({"method": "perceptual_deepjscc", "row": row,
                         "archive": deep_root / "images" / f"{int(row['image_index']):03d}/reconstructions.npz",
                         "image_slot": int(row["image_ref"].split(":")[1]), "psnr_db": float(row["psnr_db"]),
                         "lpips": float(row["lpips_alex"]), "dino": float(row["dino_cosine"]), "previous_ssim": None})
    for row in read_rows(r3_root / "per_frame.csv"):
        if row["arm"] != "r3__full_grid_prediction_features" or float(row["snr_db"]) not in SNRS:
            continue
        if (row["checkpoint_sha256"] != "667abc43639c09b0f0c465bc3bf669117ff727095ffdb7e563f6dad71c11ab3e" or
                int(row["total_complex_uses"]) != 3060 or abs(float(row["total_energy"]) - 6120) > .01):
            raise RuntimeError("R3 is not the originally frozen selected point")
        selected.append({"method": "wetok_r3", "row": row, "archive": Path(row["image_archive"]),
                         "image_slot": int(row["image_ref"]), "psnr_db": float(row["psnr_db"]),
                         "lpips": float(row["lpips"]), "dino": float(row["dino"]), "previous_ssim": float(row["ssim"])})
    expected = {(method, index, snr, seed) for method in ("raw_adaptive", "arithmetic_adaptive", "perceptual_deepjscc", "wetok_r3")
                for index in range(100) for snr in SNRS for seed in SEEDS}
    keys = {(item["method"], int(item["row"]["image_index"]), float(item["row"]["snr_db"]), int(item["row"]["seed"])) for item in selected}
    if len(selected) != 6000 or keys != expected:
        raise RuntimeError("source/noise/SNR reference coverage is incomplete")
    population = Population("development", json.loads((PROJECT / "configs/communication_decision_study.json").read_text()))
    rows, inputs = [], []
    maximum_psnr, maximum_ssim = 0., 0.
    source_directory = output / "development"
    source_directory.mkdir()
    for index in range(100):
        pixels, _scales, target = population.source(index, None, None)
        image_path = source_directory / f"{index:04d}.npy"
        np.save(image_path, pixels, allow_pickle=False)
        source_hash = digest(pixels)
        inputs.append({"image_index": index, "image_id": target["image_id"], "class_index": int(target["class_index"]),
                       "source_pixels_sha256": source_hash, "path": str(image_path), "file_sha256": sha256(image_path)})
        reference = torch.from_numpy(pixels).float().div(255)[None]
        items = [item for item in selected if int(item["row"]["image_index"]) == index]
        archives = {}
        for path in {item["archive"] for item in items}:
            with np.load(path, allow_pickle=False) as archive:
                archives[path] = archive["images"].copy()
        for item in items:
            archived = item["row"]
            if archived["image_id"] != target["image_id"]:
                raise RuntimeError("reference source identity differs")
            if archived.get("source_pixels_sha256") and archived["source_pixels_sha256"] != source_hash:
                raise RuntimeError("digital reference used different source pixels")
            noise = seeded_noise(target["image_id"], int(archived["seed"]), (3060, 2))
            if archived.get("noise_sha256") and archived["noise_sha256"] != digest(noise):
                raise RuntimeError("reference standard channel noise differs")
            image = archives[item["archive"]][item["image_slot"]]
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise RuntimeError("invalid frozen RGB output")
            if archived.get("image_sha256") and archived["image_sha256"] != digest(image):
                raise RuntimeError("R3 output no longer matches its frozen pixel hash")
            reconstructed = torch.from_numpy(image.copy())[None].float()
            with torch.no_grad():
                psnr = float(-10 * (reconstructed - reference).square().mean().log10())
                structural = float(ssim(reconstructed, reference, data_range=1., size_average=False)[0])
            maximum_psnr = max(maximum_psnr, abs(psnr - item["psnr_db"]))
            if item["previous_ssim"] is not None:
                maximum_ssim = max(maximum_ssim, abs(structural - item["previous_ssim"]))
            rows.append({"method": item["method"], "image_index": index, "image_id": target["image_id"],
                         "snr_db": float(archived["snr_db"]), "seed": int(archived["seed"]),
                         "complex_uses": 3060, "total_energy": 6120, "psnr_db": item["psnr_db"],
                         "ssim": structural, "lpips": item["lpips"], "dino": item["dino"],
                         "metric_origin": "frozen_PSNR_LPIPS_DINO_plus_common_SSIM_only",
                         "image_archive": str(item["archive"]), "image_slot": item["image_slot"],
                         "image_sha256": digest(image), "source_pixels_sha256": source_hash,
                         "standard_noise_sha256": digest(noise), "all_failures_retained": True})
        del archives
        if maximum_psnr > 1e-4 or maximum_ssim > 2e-4:
            raise RuntimeError(f"frozen image metrics differ: {maximum_psnr}, {maximum_ssim}")
        if (index + 1) % 10 == 0:
            print(f"Prepared {index + 1}/100 sources and {(index + 1) * 60}/6000 common reference metrics", flush=True)
    write_csv(output / "reference_per_frame.csv", rows)
    write_json(output / "development_inputs.json", inputs)
    images, labels, identifiers, _bindings = read_image_population("calibration", verify=True)
    calibration = []
    directory = output / "calibration"
    directory.mkdir()
    development_ids = {item["image_id"] for item in inputs}
    for index in CALIBRATION_INDICES:
        pixels = images[index].numpy()
        path = directory / f"{index:04d}.npy"
        if identifiers[index] in development_ids:
            raise RuntimeError("calibration source leaked from development")
        np.save(path, pixels, allow_pickle=False)
        calibration.append({"image_index": index, "image_id": identifiers[index], "class_index": int(labels[index]),
                            "source_pixels_sha256": digest(pixels), "path": str(path), "file_sha256": sha256(path)})
    write_json(output / "calibration_inputs.json", calibration)
    for path, value in bindings.items():
        if sha256(path) != value:
            raise RuntimeError("historical inputs changed during read-only preparation")
    write_json(output / "completion.json", {"status": "INPUTS_AND_FROZEN_REFERENCES_READY", "reference_rows": len(rows),
               "development_sources": 100, "calibration_sources": len(calibration), "bindings": bindings,
               "max_PSNR_pixel_check_error": maximum_psnr, "max_R3_SSIM_CPU_GPU_difference": maximum_ssim,
               "SSIM": "pytorch_msssim.ssim 1.0.0, float32 RGB, data_range1, win11, sigma1.5",
               "existing_model_forward_reruns": 0, "new_training_or_holdout": False,
               "source_manifest_sha256": sha256(output / "development_inputs.json"),
               "reference_csv_sha256": sha256(output / "reference_per_frame.csv"),
               "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
