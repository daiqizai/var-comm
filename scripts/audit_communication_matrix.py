#!/usr/bin/env python3
"""CPU replay of every PHY observation and independent saved-image/statistic invariants."""

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
from var_comm.next_scale_prior import preprocess
from var_comm.prefix_training_data import read_image_population
from var_comm.progressive import prefix_key, receive_whole, split_prefix, transmit_whole
from var_comm.scale_channel import indices_to_bits
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, verify_snapshot, write_json
from var_comm.whole_entropy import decode_phy, transmit


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run_dir.resolve()
    receipt = json.loads((run / "completion.json").read_text())
    config = json.loads((ROOT / "configs/communication_decision_study.json").read_text())
    if receipt["status"] != "FIXED_MODE_MATRIX_COMPLETE" or receipt["config_sha256"] != sha256(ROOT / "configs/communication_decision_study.json"):
        raise RuntimeError("communication matrix is incomplete or its configuration changed")
    verify_snapshot(receipt["source_hashes"])
    if receipt["frozen_before"] != receipt["frozen_after"]:
        raise RuntimeError("a frozen model changed")
    output = create_output(arguments.output_dir)
    started = time.perf_counter()
    try:
        for relative, expected in receipt["output_hashes"].items():
            if sha256(run / relative) != expected:
                raise RuntimeError(f"completed communication artifact changed: {relative}")
        population = json.loads((run / "population.json").read_text())
        metadata = json.loads((run / "metadata.json").read_text())
        role = receipt["population"]
        calibration_images = read_image_population("calibration")[0] if role == "calibration" else None
        source_indices = metadata["source_indices"]
        total, max_psnr, max_dino, complete_payload_checks = 0, 0., 0., 0
        for completed, index in enumerate(source_indices, 1):
            directory = run / "images" / f"{index:04d}"
            target = population[index]
            if role == "calibration":
                pixels = calibration_images[index].numpy()
            elif role == "development":
                with np.load(ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001" / f"images/{index:03d}/reconstructions.npz", allow_pickle=False) as archive:
                    pixels = np.rint(archive["source"] * 255).astype(np.uint8)
            else:
                normalized, pixel_hash = preprocess(Path(target["path"]))
                pixels = normalized.add(1).mul(127.5).round().to(torch.uint8).numpy()
                if pixel_hash != target["preprocessed_rgb_sha256"]:
                    raise RuntimeError("frozen holdout source preprocessing changed")
            rows = read_rows(directory / "per_frame.csv")
            expected_keys = {(family, mode, snr, seed) for family in config["coding_families"] for mode in config["prefix_modes"]
                             for snr in config["snrs_db"] for seed in config[role + "_seeds"]}
            keys = {(row["family"], int(row["mode"]), float(row["snr_db"]), int(row["seed"])) for row in rows}
            if len(rows) != len(expected_keys) or keys != expected_keys or any(row["image_id"] != target["image_id"] for row in rows):
                raise RuntimeError("per-source candidate matrix is incomplete or duplicated")
            with np.load(directory / "source_tokens.npz", allow_pickle=False) as source_archive:
                source = split_prefix(source_archive["tokens"], 10)
                label = int(source_archive["label"])
            with np.load(directory / "transmissions.npz", allow_pickle=False) as transmissions, np.load(directory / "candidates.npz", allow_pickle=False) as prefixes, np.load(directory / "reconstructions.npz", allow_pickle=False) as archive:
                images = archive["images"]
                if images.shape[1:] != (3, 256, 256) or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                    raise RuntimeError("invalid archived RGB output")
                ground = torch.from_numpy(pixels).float().div(127.5).sub(1).add(1).mul(.5).numpy()
                errors = np.square(images - ground[None]).mean(axis=(1, 2, 3), dtype=np.float64)
                psnr = -10 * np.log10(errors)
                features = archive["reconstruction_dino"].astype(np.float64)
                source_features = archive["source_dino"].astype(np.float64)
                cosine = (features @ source_features) / (np.linalg.norm(features, axis=1) * np.linalg.norm(source_features))
                signals = {}
                for family in config["coding_families"]:
                    for mode in config["prefix_modes"]:
                        signal = transmissions[f"signal_{family}_m{mode}"].astype(np.float64)
                        row = next(row for row in rows if row["family"] == family and int(row["mode"]) == mode)
                        if family == "raw":
                            expected_signal = transmit_whole(source, label, mode)
                        else:
                            expected_signal, ledger = transmit({"payload": transmissions[f"payload_m{mode}"],
                                "length_field": int(row["arithmetic_length_field"]), "attempted_arithmetic_bits": int(row["attempted_arithmetic_bits"])}, label, mode)
                        if not np.array_equal(signal, expected_signal) or signal.shape != (3060, 2) or np.square(signal).sum() != 6120:
                            raise RuntimeError("actual coded waveform, header or energy ledger changed")
                        signals[family, mode] = signal
                for row in rows:
                    family, mode, snr, seed = row["family"], int(row["mode"]), float(row["snr_db"]), int(row["seed"])
                    signal = signals[family, mode]
                    observed = signal + seeded_noise(target["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                    image_index = int(row["image_index_in_archive"])
                    if digest(observed) != row["received_sha256"] or digest(signal) != row["transmitted_sha256"] or digest(pixels) != row["source_pixels_sha256"]:
                        raise RuntimeError("source, transmitted or received bytes changed")
                    prefix = split_prefix(prefixes[f"prefix_{image_index}"], int(row["candidate_prefix_scales"])) if int(row["candidate_prefix_scales"]) else []
                    if family == "raw":
                        decoded = receive_whole(observed, snr)
                        header, decoded_prefix = decoded["header"], decoded["prefix"]
                        body_ok = bool(decoded["events"][0]["accepted"]) if decoded["events"] else False
                        fields_correct = header["accepted"] and header["label"] == label and header["mode"] == mode
                        payload_correct = len(decoded_prefix) == mode and all(np.array_equal(first, second) for first, second in zip(decoded_prefix, source[:mode]))
                        if len(prefix) != len(decoded_prefix) or any(not np.array_equal(first, second) for first, second in zip(prefix, decoded_prefix)):
                            raise RuntimeError("raw failed-candidate prefix differs from independent PHY replay")
                    else:
                        decoded = decode_phy(observed, snr)
                        header, body_ok = decoded["header"], decoded["body_crc_accepted"]
                        fields_correct = header["accepted"] and header["label"] == label and header["mode"] == mode and header["length_field"] == int(row["arithmetic_length_field"])
                        payload_correct = decoded["payload"] is not None and np.array_equal(decoded["payload"], transmissions[f"payload_m{mode}"])
                    if (int(header["accepted"]) != int(row["header_accepted"]) or int(body_ok) != int(row["body_crc_accepted"]) or
                            int(fields_correct) != int(row["header_fields_correct"]) or int(payload_correct) != int(row["payload_bits_correct"])):
                        raise RuntimeError("stored PHY/CRC/header integrity differs from independent replay")
                    receiver_label = int(row["receiver_label"]) if row["receiver_label"] != "" else None
                    if not header["accepted"]:
                        if prefix or receiver_label is not None or not np.all(images[image_index] == .5):
                            raise RuntimeError("header failure acquired free information or changed gray output")
                    elif receiver_label != header["label"] or len(prefix) > header["mode"]:
                        raise RuntimeError("receiver acquired a class or source scale not supplied by its actual header/payload")
                    key = "header_erasure" if receiver_label is None else prefix_key(prefix, receiver_label)
                    correct = bool(int(row["source_complete"]) and len(prefix) == mode and receiver_label == label and
                        header["mode"] == mode and all(np.array_equal(first, second) for first, second in zip(prefix, source[:mode])))
                    accepted = bool(correct and fields_correct and payload_correct and body_ok)
                    if key != row["prefix_sha256"] or int(correct) != int(row["source_correct"]) or int(accepted) != int(row["accepted_correct"]):
                        raise RuntimeError("source correctness or reliable-success accounting changed")
                    if fields_correct and payload_correct:
                        complete_payload_checks += 1
                        if not correct:
                            raise RuntimeError("an exactly recovered source payload did not invert the supposedly lossless source codec")
                    max_psnr = max(max_psnr, abs(float(psnr[image_index]) - float(row["psnr_db"])))
                    max_dino = max(max_dino, abs(float(cosine[image_index]) - float(row["dino"])))
                    if not np.isfinite(float(row["lpips"])):
                        raise RuntimeError("invalid stored LPIPS")
            total += len(rows)
            if completed % 25 == 0 or completed == len(source_indices):
                print(f"CPU matrix audit {completed}/{len(source_indices)} rows={total}", flush=True)
        if total != receipt["rows"] or max_psnr > 1e-4 or max_dino > 1e-5:
            raise RuntimeError("independent pixel/feature audit did not reproduce the primary records")
        write_json(output / "completion.json", {"status": "COMMUNICATION_MATRIX_CPU_AUDIT_PASS", "completed_local": datetime.now().astimezone().isoformat(),
            "input_receipt_sha256": sha256(run / "completion.json"), "population": role, "sources": len(source_indices), "rows": total,
            "all_PHY_observations_replayed": True, "exact_payload_source_inverse_checks": complete_payload_checks,
            "max_PSNR_error": max_psnr, "max_DINO_error": max_dino, "LPIPS_independently_rerun": False,
            "VAR_independently_rerun": False, "new_model_inference": False, "source_script_sha256": sha256(Path(__file__)),
            "elapsed_seconds": time.perf_counter() - started, "research_goal_complete": False})
    except BaseException as error:
        write_json(output / "failure.json", {"error": repr(error), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
