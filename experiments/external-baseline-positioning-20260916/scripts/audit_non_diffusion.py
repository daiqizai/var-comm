#!/usr/bin/env python3
"""Recheck new PHY/selection, summary means and fixed preview pixels on CPU."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src"), str(EXPERIMENT / "scripts")]

import numpy as np
import torch
from pytorch_msssim import ssim

from evaluate_digital_budgets import digest, legacy_source, physical_state, read_csv
from external_positioning.analysis import METRICS, SCOPES, validate_groups
from external_positioning.digital_budget import receive, transmit
from var_comm.study import seeded_noise, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    root, analysis = arguments.root, arguments.analysis
    sources = {int(item["image_index"]): item for item in json.loads((root / "inputs_001/development_inputs.json").read_text())}
    policies = json.loads((root / "digital_policies_001.json").read_text())
    calibration = root / "digital_calibration_001"
    development = root / "digital_development_001"
    if sha256(calibration / "per_frame.csv") != policies["calibration_per_frame_sha256"]:
        raise RuntimeError("the frozen mode selection no longer matches calibration")
    metadata = json.loads((development / "metadata.json").read_text())
    if metadata["policy_sha256"] != sha256(root / "digital_policies_001.json"):
        raise RuntimeError("development did not use the frozen calibration policies")
    digital_rows = read_csv(development / "per_frame.csv")
    validate_groups(digital_rows, sources)
    checked = 0
    for index in range(100):
        _directory, _receipt, original, label, packed, _old_rows = legacy_source("development", index)
        directory = development / "images" / f"{index:04d}"
        with np.load(directory / "payloads.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        records = json.loads((directory / "phy.json").read_text())["records"]
        signals = {}
        for record in records:
            family, budget, mode = record["family"], record["complex_uses"], record["mode"]
            expected_mode = policies["budgets"][str(budget)]["actions"][family]["quality"][str(float(record["snr_db"]))]
            if mode != expected_mode:
                raise RuntimeError("a development mode differs from the calibrated SNR-only action")
            key = family, budget, mode
            if key not in signals:
                signals[key] = transmit(original, packed[f"payload_m{mode}"], label, mode, family, budget)[0]
            signal = signals[key]
            noise = seeded_noise(record["image_id"], record["seed"], (budget, 2))
            observed = signal + noise / np.sqrt(10 ** (record["snr_db"] / 10))
            decoded = receive(observed, record["snr_db"], family)
            archived = physical_state(record, arrays)
            if digest(signal) != record["transmitted_sha256"] or digest(observed) != record["received_sha256"]:
                raise RuntimeError("actual new-budget signal/observation differs")
            if decoded["header"]["accepted"] != archived["header"]["accepted"] or decoded["body_crc_accepted"] != archived["body_crc_accepted"]:
                raise RuntimeError("replayed CRC/header decisions differ")
            if decoded["label"] != archived["label"] or decoded["mode"] != archived["mode"] or not np.array_equal(decoded["payload"], archived["payload"]):
                raise RuntimeError("replayed actual receiver candidate differs")
            checked += 1
    external = read_csv(root / "fast_author_metrics_001/per_frame.csv")
    refs = read_csv(root / "inputs_001/reference_per_frame.csv")
    for row in external:
        row["total_energy"] = row["actual_total_energy"]
    for row in refs:
        row["protocol"] = "common_paid_information"
    rows = refs + external + digital_rows
    groups = validate_groups(rows, sources)
    means_error = 0.
    supports = dict(SCOPES)
    for reported in read_csv(analysis / "summary.csv"):
        selected = groups[reported["method"], reported["protocol"], int(reported["complex_uses"])]
        source_metric = defaultdict(list)
        for row in selected:
            if float(row["snr_db"]) in supports[reported["scope"]]:
                source_metric[row["image_id"]].append([float(row[name]) for name in METRICS])
        measured = np.mean([np.mean(values, axis=0) for values in source_metric.values()], axis=0)
        means_error = max(means_error, max(abs(measured[position] - float(reported[name])) for position, name in enumerate(METRICS)))
    pixel_checks, psnr_error, ssim_error = 0, 0., 0.
    seen = set()
    for row in rows:
        key = row["image_archive"], row["image_slot"]
        if int(row["image_index"]) not in (0, 25, 50, 75) or int(row["seed"]) != 2001 or key in seen:
            continue
        seen.add(key)
        pixels = np.load(sources[int(row["image_index"])]["path"], allow_pickle=False).astype(np.float32) / 255
        with np.load(row["image_archive"], allow_pickle=False) as archive:
            image = archive["images"][int(row["image_slot"])].copy()
        if row.get("image_sha256") and digest(image) != row["image_sha256"]:
            raise RuntimeError("reconstruction pixel digest differs")
        measured_psnr = -10 * np.log10(np.mean((image.astype(np.float64) - pixels) ** 2))
        with torch.no_grad():
            measured_ssim = float(ssim(torch.from_numpy(image)[None], torch.from_numpy(pixels)[None], data_range=1., size_average=False)[0])
        psnr_error = max(psnr_error, abs(measured_psnr - float(row["psnr_db"])))
        ssim_error = max(ssim_error, abs(measured_ssim - float(row["ssim"])))
        pixel_checks += 1
    if means_error > 1e-10 or psnr_error > 1e-4 or ssim_error > 2e-4:
        raise RuntimeError("independent metric/aggregation check failed")
    write_json(analysis / "audit.json", {"status": "NON_DIFFUSION_CPU_REPLAY_AND_AGGREGATION_PASS", "actual_new_budget_PHY_replays": checked,
               "complete_input_rows": len(rows), "fixed_source_pixel_checks": pixel_checks,
               "max_mean_error": means_error, "max_PSNR_error": psnr_error, "max_SSIM_error": ssim_error,
               "LPIPS_DINO_independently_rerun": False, "bootstrap_independently_recomputed": False,
               "no_training_or_new_holdout": True, "at": datetime.now().astimezone().isoformat()})
    print(f"PASS: {checked} actual PHY, {len(rows)} row coverage, {pixel_checks} pixel/PSNR/SSIM checks")


if __name__ == "__main__":
    main()
