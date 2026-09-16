#!/usr/bin/env python3
"""Audit frozen holdout reference files, observations, power and saved image metrics on CPU."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from benchmark_frozen_systems import digest
from var_comm.frozen_timing_adapters import install_frozen_paths
from var_comm.next_scale_prior import preprocess
from var_comm.study import create_output, seeded_noise, sha256, write_json


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-method", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run_dir.resolve()
    receipt = json.loads((run / "completion.json").read_text())
    manifest = json.loads(arguments.holdout_manifest.read_text())
    config = json.loads((ROOT / "configs/communication_decision_study.json").read_text())
    if (receipt["status"] != "FROZEN_HOLDOUT_REFERENCES_COMPLETE" or receipt["rows"] != 84000 or
            receipt["manifest_sha256"] != sha256(arguments.holdout_manifest) or receipt["method_sha256"] != sha256(arguments.frozen_method) or
            receipt["frozen_before"] != receipt["frozen_after"]):
        raise RuntimeError("holdout reference run is incomplete, changed or not bound to the frozen method")
    output = create_output(arguments.output_dir)
    started = time.perf_counter()
    try:
        for relative, expected in receipt["output_hashes"].items():
            if sha256(run / relative) != expected:
                raise RuntimeError("held-out reference artifact changed")
        for path, expected in receipt["source_hashes"].items():
            if sha256(path) != expected:
                raise RuntimeError("held-out reference source code changed")
        install_frozen_paths()
        from wetok_comm.digital import WeTokDigital
        modem = WeTokDigital()
        names = ("r2__full_grid_innovation", "r3__full_grid_prediction_features", "perceptual_deepjscc", "wetok_8PSK_FEC")
        maximum_noise_error, maximum_psnr_error, maximum_dino_error, total, crc_failures = 0., 0., 0., 0, 0
        for index, target in enumerate(manifest["images"]):
            directory = run / "images" / f"{index:04d}"
            rows = read_rows(directory / "per_frame.csv")
            expected_keys = {(name, snr, seed) for name in names for snr in config["snrs_db"] for seed in config["holdout_seeds"]}
            keys = {(row["arm"], float(row["snr_db"]), int(row["seed"])) for row in rows}
            if len(rows) != 84 or keys != expected_keys or any(row["image_id"] != target["image_id"] for row in rows):
                raise RuntimeError("holdout references omitted or duplicated a source/noise/SNR/method")
            normalized, pixel_hash = preprocess(Path(target["path"]))
            if pixel_hash != target["preprocessed_rgb_sha256"]:
                raise RuntimeError("holdout source preprocessing changed")
            source = normalized.add(1).mul(.5).numpy()
            with np.load(directory / "reconstructions.npz", allow_pickle=False) as archive, np.load(directory / "waveforms.npz", allow_pickle=False) as waves:
                images = archive["images"]
                if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                    raise RuntimeError("invalid reference RGB image")
                psnr = -10 * np.log10(np.square(images - source[None]).mean(axis=(1, 2, 3), dtype=np.float64))
                features = archive["reconstruction_dino"].astype(np.float64)
                original_features = archive["source_dino"].astype(np.float64)
                dino = (features @ original_features) / (np.linalg.norm(features, axis=1) * np.linalg.norm(original_features))
                for row in rows:
                    snr, seed, name = float(row["snr_db"]), int(row["seed"]), row["arm"]
                    signal = waves[f"{name}_snr{snr}_tx"]
                    received = waves[f"{name}_snr{snr}_seed{seed}_rx"]
                    if signal.shape != (3060, 2) or received.shape != (3060, 2) or abs(float(np.square(signal.astype(np.float64)).sum()) - 6120) > .02:
                        raise RuntimeError("reference violated the same N/E physical budget")
                    if digest(signal) != row["transmitted_sha256"] or digest(received) != row["received_sha256"]:
                        raise RuntimeError("reference waveform digests changed")
                    noise = seeded_noise(target["image_id"], seed, (3060, 2))
                    if name == "wetok_8PSK_FEC":
                        expected_received = signal + noise / np.sqrt(10 ** (snr / 10))
                        decoded = modem.receive(received, snr)
                        if int(decoded["crc_accepted"]) != int(row["crc_accepted"]):
                            raise RuntimeError("reference CRC decision changed")
                        crc_failures += int(not decoded["crc_accepted"])
                    else:
                        expected_received = signal + noise.astype(np.float32) * np.float32(10 ** (-snr / 20))
                    maximum_noise_error = max(maximum_noise_error, float(np.max(np.abs(received - expected_received))))
                    image_index = int(row["image_index_in_archive"])
                    if digest(images[image_index]) != row["image_sha256"]:
                        raise RuntimeError("reference image digest changed")
                    maximum_psnr_error = max(maximum_psnr_error, abs(float(psnr[image_index]) - float(row["psnr_db"])))
                    maximum_dino_error = max(maximum_dino_error, abs(float(dino[image_index]) - float(row["dino"])))
                    if not np.isfinite(float(row["lpips"])):
                        raise RuntimeError("invalid saved LPIPS")
            total += len(rows)
            if (index + 1) % 50 == 0:
                print(f"holdout reference CPU audit {index+1}/1000", flush=True)
        if maximum_noise_error > 1e-6 or maximum_psnr_error > 1e-4 or maximum_dino_error > 1e-5 or total != 84000:
            raise RuntimeError("reference source/noise/pixel audit differs from the primary evaluation")
        write_json(output / "completion.json", {"status": "FROZEN_HOLDOUT_REFERENCE_CPU_AUDIT_PASS", "completed_local": datetime.now().astimezone().isoformat(),
            "input_receipt_sha256": sha256(run / "completion.json"), "rows": total, "source_images": 1000,
            "max_noise_error": maximum_noise_error, "max_PSNR_error": maximum_psnr_error, "max_DINO_error": maximum_dino_error,
            "retained_WeTok_CRC_failure_frames": crc_failures, "new_model_inference": False, "LPIPS_independently_rerun": False,
            "source_script_sha256": sha256(Path(__file__)), "elapsed_seconds": time.perf_counter() - started, "research_goal_complete": False})
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
