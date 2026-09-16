#!/usr/bin/env python3
"""Post-completion artifact/metric audit and fixed report figures; no training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

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
from var_comm.study import paired_interval, seeded_noise, sha256, write_csv, write_json


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return int(row["image_index"]), float(row["snr_db"]), int(row["seed"])


def run(arguments):
    source = arguments.run.resolve()
    output = source / "report_assets_001"
    output.mkdir(exist_ok=False)
    if not (source / "pipeline_001/completion.json").exists():
        raise RuntimeError("do not audit an incomplete pipeline as a finished experiment")
    assert_gpu_available()
    config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
    plan = json.loads((source / "final_report_view_plan.json").read_text())
    quality = source / "development_001"
    rows = read_rows(quality / "per_frame.csv")
    references = read_rows(quality / "frozen_reference_rows.csv")
    bases = read_rows(quality / "shortened_base_diagnostic.csv")
    if len(rows) != 4500 or len(references) != 9000 or len(bases) != 1500:
        raise RuntimeError("unexpected complete source/frame coverage")
    lookup = {(row["arm"], *key(row)): row for row in rows}
    if len(lookup) != len(rows):
        raise RuntimeError("duplicate new-system frames")
    expected_keys = {(index, snr, seed) for index in range(100) for snr in config["snrs_db"] for seed in config["development_seeds"]}
    for arm in config["arms"]:
        if {key(row) for row in rows if row["arm"] == arm} != expected_keys:
            raise RuntimeError("a failure/source/SNR disappeared from evaluation")
    original = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    perceptual, dino, _weights = load_quality_models(paths, device)
    errors = {"psnr_db": 0.0, "lpips": 0.0, "dino": 0.0, "mse_CPU": 0.0}
    tolerances = {"psnr_db": 1e-4, "lpips": 1e-5, "dino": 1e-5, "mse_CPU": 1e-7}
    source_rows = {}
    for index in range(100):
        pixels, scales, target = original.source(index, None, None)
        stored = np.load(quality / "images" / f"source_{index:04d}.npy", allow_pickle=False)
        np.testing.assert_array_equal(pixels, stored)
        source_rows[index] = target
        transmitted = encode_digital(np.concatenate(scales[:7]), int(target["class_index"]))
        for snr in config["snrs_db"]:
            for seed in config["development_seeds"]:
                observed = transmitted + seeded_noise(target["image_id"], seed, (3060, 2))[:1950] / np.sqrt(10 ** (snr / 10))
                decoded = receive_digital(observed, snr)
                correct = bool(decoded["header_usable"] and decoded["label"] == int(target["class_index"]) and
                               np.array_equal(decoded["tokens"], np.concatenate(scales[:7])))
                flags = {"header_usable": decoded["header_usable"], "body_crc_accepted": decoded["crc_accepted"],
                         "source_correct": correct, "accepted_correct": correct and decoded["crc_accepted"],
                         "false_acceptance": decoded["crc_accepted"] and not correct}
                for arm in config["arms"]:
                    row = lookup[arm, index, snr, seed]
                    if any((row[name] == "True") != bool(value) for name, value in flags.items()):
                        raise RuntimeError("stored failure/reliability flags differ from actual PHY replay")
                    if row["received_digital_sha256"] != hashlib.sha256(observed.tobytes()).hexdigest():
                        raise RuntimeError("stored digital observation differs from the original noise")
                    if int(row["complex_uses"]) != 3060 or abs(float(row["total_energy"]) - 6120) > .01:
                        raise RuntimeError("incorrect physical ledger")
        selected = [row for row in rows if int(row["image_index"]) == index]
        selected_base = [row for row in bases if int(row["image_index"]) == index]
        images = []
        for row in selected:
            path = quality / row["image_path"]
            if sha256(path) != row["image_sha256"]:
                raise RuntimeError("saved reconstruction bytes changed")
            image = np.load(path, allow_pickle=False)
            if row["header_usable"] == "False" and not np.all(image == .5):
                raise RuntimeError("header failure did not retain the fixed gray rule")
            images.append(image)
        for row in selected_base:
            images.append(np.load(quality / "images" / f"shortened_base_{index:04d}_{float(row['snr_db']):g}_{row['seed']}.npy", allow_pickle=False))
        normalized = pixels.astype(np.float32) / 255
        recomputed, _source_features, _image_features = quality_metrics(normalized, images, perceptual, dino, device)
        for expected, actual, image in zip(selected + selected_base, recomputed, images):
            for stored_name, actual_name in (("psnr_db", "psnr_db"), ("lpips", "lpips_alex"), ("dino", "dino_cosine")):
                error = abs(float(expected[stored_name]) - actual[actual_name])
                errors[stored_name] = max(errors[stored_name], error)
            mse = float(np.mean((image.astype(np.float64) - normalized.astype(np.float64)) ** 2))
            errors["mse_CPU"] = max(errors["mse_CPU"], abs(mse - float(expected["mse"])))
        if any(errors[name] > tolerances[name] for name in errors):
            raise RuntimeError(f"independent metric recomputation exceeds predeclared numerical tolerances: {errors}")
        if (index + 1) % 10 == 0:
            print(f"audited {index+1}/100 sources, {60*(index+1)} saved images", flush=True)
    write_json(output / "audit.json", {"status": "PASS", "sources": 100, "actual_PHY_replays": 1500,
               "new_images_checked": 4500, "shortened_base_images_checked": 1500, "metric_max_errors": errors,
               "tolerances": tolerances, "new_holdout": False, "new_training": False,
               "script_sha256": sha256(Path(__file__)), "models": {name: paths[name] for name in ("alexnet_checkpoint", "dino_checkpoint")}})
    del perceptual, dino
    matched_subset = set(np.linspace(0, 999, 100, dtype=int).tolist())
    calibration = []
    for directory in sorted((source / "training_001/calibration").iterdir()):
        step = int(directory.name.split("_")[1])
        records = read_rows(directory / "per_frame.csv")
        for arm in config["arms"]:
            for region, snrs in (("primary", config["primary_snrs_db"]), ("high", [13., 19.])):
                selected = [row for row in records if row["arm"] == arm and int(row["image_index"]) in matched_subset and float(row["snr_db"]) in snrs]
                if len(selected) != 100 * len(snrs) * 3:
                    raise RuntimeError("calibration curve changes its source/noise population")
                calibration.append({"step": step, "arm": arm, "region": region,
                                    **{metric: float(np.mean([float(row[metric]) for row in selected])) for metric in ("psnr_db", "lpips", "mse")}})
    write_csv(output / "matched_calibration_curves.csv", calibration)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row_index, region in enumerate(("primary", "high")):
        for column, metric in enumerate(("psnr_db", "lpips")):
            axis = axes[row_index, column]
            for arm in config["arms"]:
                selected = sorted([row for row in calibration if row["region"] == region and row["arm"] == arm], key=lambda row: row["step"])
                axis.plot([row["step"] for row in selected], [row[metric] for row in selected], marker="o", label=arm)
            axis.set(xlabel="Updates per arm", ylabel=metric, title=region)
            axis.grid(alpha=.25)
    axes[0, 1].legend()
    figure.suptitle("Identical 100 calibration sources and noise at every checkpoint")
    figure.tight_layout()
    figure.savefig(output / "matched_calibration_curves.png", dpi=180)
    figure.savefig(output / "matched_calibration_curves.pdf")
    plt.close(figure)
    shortened = {key(row): row for row in bases}
    contributions = []
    for arm in config["arms"]:
        for region, snrs in [("primary", config["primary_snrs_db"]), ("high", [13., 19.])] + [(f"snr_{snr:g}", [snr]) for snr in config["snrs_db"]]:
            selected = [row for row in rows if row["arm"] == arm and float(row["snr_db"]) in snrs]
            for metric in ("psnr_db", "lpips", "dino"):
                differences = [np.mean([float(row[metric])-float(shortened[key(row)][metric]) for row in selected if int(row["image_index"])==index]) for index in range(100)]
                interval = paired_interval(differences, 20260915, 10000)
                contributions.append({"arm": arm, "region": region, "metric": metric,
                                      "difference_vs_shortened_base": interval["gain"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
    write_csv(output / "correction_vs_shortened_base.csv", contributions)
    digital_directory = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/DEVELOPMENT_001"
    digital = read_rows(digital_directory / "per_frame.csv")
    policy = json.loads((ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/POLICIES_001/policies.json").read_text())["actions"]
    deep_directory = ROOT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
    deep = [row for row in read_rows(deep_directory / "per_frame.csv") if row["arm"] == "perceptual_deepjscc"]
    for snr in plan["visual_snrs_db"]:
        figure, axes = plt.subplots(4, 7, figsize=(17.5, 10))
        for row_index, index in enumerate(plan["source_indices_for_visuals"]):
            for column, arm in enumerate(plan["visual_methods"]):
                metrics = None
                if arm == "source":
                    image = np.load(quality / "images" / f"source_{index:04d}.npy").astype(np.float32) / 255
                elif arm in ("raw_adaptive", "arithmetic_adaptive"):
                    family = arm.split("_")[0]
                    mode = policy[family]["quality"][str(snr)]
                    record = next(row for row in digital if key(row)==(index,snr,2001) and row["family"]==family and int(row["mode"])==mode)
                    with np.load(digital_directory / "images" / f"{index:04d}" / "reconstructions.npz", allow_pickle=False) as archive:
                        image = archive["images"][int(record["image_index_in_archive"])]
                    metrics = float(record["psnr_db"]), float(record["lpips"])
                elif arm == "perceptual_deepjscc":
                    record = next(row for row in deep if key(row)==(index,snr,2001))
                    if not record["image_ref"].startswith("new:"):
                        raise RuntimeError("unexpected frozen Deep image reference")
                    with np.load(deep_directory / "images" / f"{index:03d}" / "reconstructions.npz", allow_pickle=False) as archive:
                        image = archive["images"][int(record["image_ref"].split(":")[1])]
                    metrics = float(record["psnr_db"]), float(record["lpips_alex"])
                else:
                    record = lookup[arm,index,snr,2001]
                    image = np.load(quality / record["image_path"], allow_pickle=False)
                    metrics = float(record["psnr_db"]), float(record["lpips"])
                axis = axes[row_index,column]
                axis.imshow(np.asarray(image).transpose(1,2,0).clip(0,1), interpolation="nearest")
                axis.set_axis_off()
                title = f"source {index}" if metrics is None else f"{arm}\nPSNR {metrics[0]:.2f} | LPIPS {metrics[1]:.3f}"
                axis.set_title(title, fontsize=8)
        figure.suptitle(f"Fixed previews [0,25,50,75] · SNR {snr:g} dB · seed 2001 · N=3060, E=6120 · development only")
        figure.tight_layout()
        figure.savefig(output / f"fixed_examples_{snr:g}dB.png", dpi=180)
        plt.close(figure)
    write_json(output / "completion.json", {"status": "REPORT_ASSETS_AND_RECOMPUTATION_COMPLETE", "audit_sha256": sha256(output / "audit.json"),
               "finished_unix": time.time(), "new_training": False, "new_holdout": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args())
