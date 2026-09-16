#!/usr/bin/env python3
"""Only the new pair: actual CPU endpoints including spatial feature extraction."""

import argparse
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

from benchmark_frozen_systems import assert_gpu_available, gpu_state
from benchmark_hybrid_correction import transmit
from evaluate_base_conditioning import load_selected
from evaluate_communication_modes import Population
from var_comm.hybrid_correction import receive_digital, receiver_features
from var_comm.next_scale_prior import load_models
from var_comm.progressive import complete_image
from var_comm.study import seeded_noise, sha256, verify_snapshot, write_csv, write_json


@torch.no_grad()
def receive(waveform, snr, model, vae, var, device):
    result = receive_digital(waveform[:1950], snr)
    if not result["header_usable"]:
        return np.full((3, 256, 256), .5, dtype=np.float32)
    base = complete_image(vae, var, result["prefix"], result["label"], device)
    received = torch.from_numpy(waveform[1950:].reshape(1, 2220)).to(device).float()
    base_tensor = torch.from_numpy(base[None]).to(device)
    correction = model.decode(received, torch.tensor([snr], device=device), base_tensor)
    image, _fixed_gain = model.fuse(base_tensor, correction,
                                   torch.from_numpy(receiver_features(result, snr)[None]).to(device),
                                   torch.tensor([True], device=device))
    return image[0].cpu().numpy()


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
    device = torch.device("cuda:0")
    models, selection, hashes, metadata = load_selected(arguments.training, config, device, arguments.mode)
    for model in models.values():
        model.requires_grad_(False)
    registered_selection = json.loads((arguments.quality / "selection_frozen_before_development.json").read_text())
    if registered_selection["selected"] != selection["selected"]:
        raise RuntimeError("timing must use the same fixed quality checkpoint")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    vae, var = load_models(paths, device)
    original = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
    indices = np.rint(np.linspace(0, 99, 32)).astype(int) if arguments.mode == "full" else np.array([0, 1])
    repeats = 3 if arguments.mode == "full" else 1
    write_json(output / "protocol.json", {"mode": arguments.mode, "source_indices": indices.tolist(),
               "repeats": repeats, "warmups_per_SNR_arm": 2, "batch": 1, "complex_uses": 3060, "energy": 6120,
               "context_feature_extraction_included": True, "both_shared_feature_calls_included": True,
               "cached_TX_or_RX_base_allowed": False, "image_replay_tolerance": .0002,
               "TX_endpoint": "CPU_uint8_RGB_to_CPU_waveform", "RX_endpoint": "CPU_waveform_to_CPU_RGB",
               "excludes": ["disk_IO", "model_loading", "warmup", "AWGN_generation", "metrics", "airtime", "queueing", "class_acquisition", "CSI_estimation"],
               "selected_checkpoints": hashes, "start_gpu": gpu_state()})
    rows = []
    for snr in config["snrs_db"]:
        pixels, _scales, target = original.source(int(indices[0]), vae, device)
        for model in models.values():
            for _warmup in range(2):
                signal = transmit(pixels, int(target["class_index"]), snr, model, vae, var, device)
                observed = signal + seeded_noise(target["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                receive(observed, snr, model, vae, var, device)
        for index in indices:
            pixels, _scales, target = original.source(int(index), vae, device)
            noise = seeded_noise(target["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
            for arm, model in models.items():
                expected = np.load(arguments.quality / "images" / f"{arm}_{index:04d}_{snr:g}_2001.npy", allow_pickle=False)
                for repetition in range(repeats):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    signal = transmit(pixels, int(target["class_index"]), snr, model, vae, var, device)
                    torch.cuda.synchronize(device)
                    tx_seconds = time.perf_counter() - start
                    observed = signal + noise
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    image = receive(observed, snr, model, vae, var, device)
                    torch.cuda.synchronize(device)
                    rx_seconds = time.perf_counter() - start
                    error = float(np.max(np.abs(image - expected)))
                    energy = float(np.sum(signal ** 2))
                    if len(signal) != 3060 or abs(energy - 6120) > .02 or error > .0002:
                        raise RuntimeError(f"timing ledger/replay mismatch: energy={energy}, image_error={error}")
                    rows.append({"arm": arm, "image_index": int(index), "image_id": target["image_id"],
                                 "snr_db": snr, "seed": 2001, "repetition": repetition,
                                 "tx_seconds": tx_seconds, "rx_seconds": rx_seconds, "processing_seconds": tx_seconds + rx_seconds,
                                 "complex_uses": 3060, "total_energy": energy, "image_max_error": error})
            write_csv(output / "per_call.csv", rows)
    verify_snapshot(metadata["bindings"])
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise RuntimeError("selected checkpoint changed during timing")
    write_json(output / "completion.json", {"status": "CONDITIONING_PAIR_TIMING_COMPLETE", "sources": len(indices), "rows": len(rows),
               "image_max_error": max(row["image_max_error"] for row in rows), "per_call_sha256": sha256(output / "per_call.csv"),
               "end_gpu": gpu_state(), "old_benchmark_rerun": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
