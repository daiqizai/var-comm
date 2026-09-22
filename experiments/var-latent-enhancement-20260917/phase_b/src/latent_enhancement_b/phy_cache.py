from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time

import numpy as np
import torch

from latent_enhancement.latent import observed_status
from latent_enhancement.runtime import ROOT, digest, save_torch, verify_snapshot, write_json
from var_comm.progressive import receive_whole, split_prefix, transmit_whole
from var_comm.scale_channel import load_native
from var_comm.study import seeded_noise

from .common import (CONFIG_B, OUT_B, bind_files, cache_shard, config_b,
                     sample_key_digest, validate_sample_grid)


def decode_source(item):
    tokens, label, identifier, snrs, seeds = item
    waveform = transmit_whole(split_prefix(tokens, 8), int(label), 8)
    if waveform.shape != (3060, 2) or float(np.square(waveform).sum()) != 6120:
        raise RuntimeError("original m8 PHY ledger changed")
    rows = []
    for snr in snrs:
        for seed in seeds:
            observation = waveform + seeded_noise(identifier, seed, waveform.shape) / np.sqrt(10 ** (snr / 10))
            received = receive_whole(observation, snr)
            candidate = np.concatenate(received["prefix"]) if received["prefix"] else np.empty(0, dtype=np.int64)
            source_equal = (received["label"] == label and len(candidate) == len(tokens) and np.array_equal(candidate, tokens))
            rows.append({"candidate": candidate, "label": received["label"], "status": observed_status(received),
                         "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                         "score": received["events"][0]["score"] if received["events"] else 0.0,
                         "source_equal_diagnostic_only": source_equal})
    return hashlib.sha256(waveform.tobytes()).hexdigest(), rows


def main():
    torch.set_num_threads(2)
    config = config_b()
    directory = OUT_B / "phy_cache"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [__file__, CONFIG_B, ROOT / "src/var_comm/progressive.py", ROOT / "src/var_comm/scale_channel.py",
             ROOT / "src/var_comm/token_trellis.cpp", ROOT / "src/var_comm/study.py"]
    source_bindings = bind_files(paths)
    scope = {"cache_kind": "phy", "population_roles": ["calibration", "train"],
             "protocol_id": "original_m8_v1", "noise_model": "seeded_noise_v1:variance=1/SNR",
             "config_sha256": digest(CONFIG_B), "sample_key_schema": "image_id/source/SNR/noise_seed"}
    registration = directory / "registration.json"
    if registration.exists():
        registered = json.loads(registration.read_text())
        verify_snapshot(registered["source_bindings"])
        if registered.get("scope") != scope:
            raise RuntimeError("PHY cache scope/protocol configuration changed")
    else:
        write_json(registration, {"source_bindings": source_bindings, "torch_cuda_initialized": False,
                                  "digital_pool": config["training_base_noise_pool"], "scope": scope,
                                  "created_at": time.time()})
    if (directory / "completion.json").exists():
        return
    load_native()
    started = time.perf_counter()
    newly_decoded = 0
    all_receipts = {}
    with ThreadPoolExecutor(max_workers=config["cpu_phy_workers"]) as executor:
        for population, shard_count in (("calibration", 10), ("train", 200)):
            destination = directory / population
            destination.mkdir(exist_ok=True)
            seeds = config["training_base_noise_seeds"] if population == "train" else config["calibration_noise_seeds"]
            for shard_index in range(shard_count):
                path = destination / f"shard_{shard_index:04d}.pt"
                receipt_path = path.with_suffix(".json")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if digest(path) != receipt["sha256"]:
                        raise RuntimeError("committed PHY replay cache changed")
                    if receipt.get("scope") != scope or receipt.get("snrs_db") != config["snrs_db"] or receipt.get("noise_seeds") != seeds:
                        raise RuntimeError("committed PHY replay cache grid/scope changed")
                    all_receipts[str(path.relative_to(directory))] = receipt["sha256"]
                    continue
                source, source_path, source_receipt = cache_shard(population, shard_index)
                count = len(source["image_ids"])
                grid = (count, len(config["snrs_db"]), len(seeds))
                candidates = np.full((*grid, 424), -1, dtype=np.int16)
                lengths = np.zeros(grid, dtype=np.int16)
                labels = np.full(grid, -1, dtype=np.int16)
                status = np.zeros((*grid, 3), dtype=np.float32)
                scores = np.zeros(grid, dtype=np.float64)
                equality = np.zeros(grid, dtype=np.bool_)
                waveform_hashes, observation_hashes = [], []
                inputs = [(source["base_tokens"][index].numpy(), int(source["labels"][index]), identifier,
                           config["snrs_db"], seeds) for index, identifier in enumerate(source["image_ids"])]
                for source_index, (waveform_sha, rows) in enumerate(executor.map(decode_source, inputs)):
                    waveform_hashes.append(waveform_sha)
                    observation_hashes.append([row["observation_sha256"] for row in rows])
                    for row_index, row in enumerate(rows):
                        snr_index, seed_index = divmod(row_index, len(seeds))
                        position = (source_index, snr_index, seed_index)
                        candidate = row["candidate"]
                        if len(candidate) > candidates.shape[-1]:
                            raise RuntimeError("actual received mode exceeds registered original protocol")
                        candidates[position][:len(candidate)] = candidate
                        lengths[position] = len(candidate)
                        labels[position] = -1 if row["label"] is None else row["label"]
                        status[position] = row["status"]
                        scores[position] = row["score"]
                        equality[position] = row["source_equal_diagnostic_only"]
                record = {"candidate_tokens": torch.from_numpy(candidates), "lengths": torch.from_numpy(lengths),
                          "received_labels": torch.from_numpy(labels), "rx_status": torch.from_numpy(status),
                          "scores": torch.from_numpy(scores), "source_equal_diagnostic_only": torch.from_numpy(equality),
                          "image_ids": source["image_ids"], "snrs_db": config["snrs_db"], "noise_seeds": seeds,
                          "waveform_sha256": waveform_hashes, "observation_sha256": observation_hashes,
                          "source_cache_sha256": source_receipt["sha256"], "scope": scope,
                          "sample_key_sha256": sample_key_digest(source["image_ids"], config["snrs_db"], seeds)}
                validate_sample_grid(record, image_ids=source["image_ids"], snrs_db=config["snrs_db"],
                                     noise_seeds=seeds, arrays=("candidate_tokens", "lengths", "received_labels", "rx_status", "scores"))
                save_torch(path, record)
                newly_decoded += int(np.prod(grid))
                verify_snapshot(source_bindings)
                receipt = {"sha256": digest(path), "frames": int(np.prod(grid)), "source_count": count,
                           "source_cache_sha256": source_receipt["sha256"], "base_cache_path": str(source_path),
                           "header_failures": int((status[..., 0] == 0).sum()),
                           "body_crc_failures_after_header_accept": int(((status[..., 0] == 1) & (status[..., 1] == 0)).sum()),
                           "all_failures_retained": True, "TX_truth_not_used_to_repair_decoded_candidate": True,
                           "scope": scope, "snrs_db": config["snrs_db"], "noise_seeds": seeds,
                           "sample_key_sha256": record["sample_key_sha256"]}
                write_json(receipt_path, receipt)
                all_receipts[str(path.relative_to(directory))] = receipt["sha256"]
                elapsed = time.perf_counter() - started
                write_json(directory / "status.json", {"status": "ACTUAL_ORIGINAL_M8_PHY_REPLAY_CPU_ONLY", "pid": os.getpid(),
                    "population": population, "source_count_completed": (shard_index + 1) * count,
                    "session_frames": newly_decoded, "seconds_per_frame": elapsed / newly_decoded, "timestamp": time.time()})
                print(f"PHY {population} {(shard_index + 1) * count} sources {elapsed / newly_decoded:.5f}s/frame", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    write_json(directory / "completion.json", {"status": "ACTUAL_M8_PHY_REPLAY_READY", "train_frames": 200000,
        "calibration_frames": 15000, "output_hashes": all_receipts, "registration_sha256": digest(registration),
        "elapsed_seconds": time.perf_counter() - started, "torch_cuda_initialized": False,
        "scope": scope, "sample_key_schema": "image_id/source/SNR/noise_seed"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT_B / "phy_cache" / f"failure_{time.time_ns()}.json", {"error": str(error), "type": type(error).__name__})
        raise
