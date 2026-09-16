#!/usr/bin/env python3
"""New hybrid only, batch-one CPU endpoints, with no clean-base cache."""

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
from evaluate_communication_modes import Population
from var_comm.hybrid_correction import encode_digital, receive_digital, receiver_features, ResidualLink
from var_comm.next_scale_prior import load_models
from var_comm.progressive import complete_image
from var_comm.study import seeded_noise, sha256, write_csv, write_json, verify_snapshot


@torch.no_grad()
def transmit(pixels, label, snr, model, vae, var, device):
    image = torch.from_numpy(pixels[None]).to(device).float().div(255)
    scales = [tokens[0].cpu().numpy() for tokens in vae.img_to_idxBl(image * 2 - 1)]
    base = complete_image(vae, var, scales[:7], label, device)
    residual = image - torch.from_numpy(base[None]).to(device)
    signal = model.encode(residual, torch.tensor([snr], device=device)).cpu().numpy().reshape(1110, 2)
    digital = encode_digital(np.concatenate(scales[:7]), label)
    return np.concatenate((digital, signal))


@torch.no_grad()
def receive(waveform, snr, model, vae, var, device):
    decoded = receive_digital(waveform[:1950], snr)
    if not decoded["header_usable"]:
        return np.full((3, 256, 256), 0.5, dtype=np.float32)
    base = complete_image(vae, var, decoded["prefix"], decoded["label"], device)
    condition = torch.tensor([snr], device=device)
    analog = torch.from_numpy(waveform[1950:].reshape(1, 2220)).to(device).float()
    correction = model.decode(analog, condition)
    image, _gain = model.fuse(torch.from_numpy(base[None]).to(device), correction,
                             torch.from_numpy(receiver_features(decoded, snr)[None]).to(device),
                             torch.tensor([True], device=device))
    return image[0].cpu().numpy()


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs/hybrid_source_correction.json").read_text())
    training = json.loads((arguments.training / "metadata.json").read_text())
    verify_snapshot(training["bindings"])
    selected = json.loads((arguments.quality / "selection_frozen_before_development.json").read_text())["selected"]
    specification = {"batch": 1, "TX": "CPU_uint8_RGB_to_CPU_waveform_including_VQ_and_TX_VAR",
                     "RX": "CPU_received_waveform_to_CPU_RGB_including_FEC_RX_VAR_and_residual_decoder",
                     "clean_base_cache": False, "image_replay_tolerance": 0.0002, "energy_tolerance": 0.02,
                     "excludes": ["disk_IO", "model_loading", "warmup", "AWGN_generation", "metrics", "airtime", "queueing", "class_acquisition", "CSI_estimation"],
                     "mode": arguments.mode, "quality_receipt_sha256": sha256(arguments.quality / "completion.json")}
    write_json(output / "protocol.json", specification)
    device = torch.device("cuda:0")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    vae, var = load_models(paths, device)
    models = {}
    for arm, choice in selected.items():
        checkpoint = torch.load(arguments.training / "checkpoints" / f"step_{choice['step']:07d}.pt", map_location="cpu", weights_only=False)
        model = ResidualLink(config["model"], arm).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["models"][arm], strict=True)
        models[arm] = model
        del checkpoint
    original = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
    indices = np.rint(np.linspace(0, 99, 32)).astype(int) if arguments.mode == "full" else np.array([0, 1])
    repeats = 3 if arguments.mode == "full" else 1
    rows = []
    write_json(output / "start_gpu.json", {"state": gpu_state()})
    for snr in config["snrs_db"]:
        pixels, _source, target = original.source(int(indices[0]), vae, device)
        for model in models.values():
            for _warmup in range(2):
                signal = transmit(pixels, int(target["class_index"]), snr, model, vae, var, device)
                noise = seeded_noise(target["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                receive(signal + noise, snr, model, vae, var, device)
        for index in indices:
            pixels, _source, target = original.source(int(index), vae, device)
            noise = seeded_noise(target["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
            for arm, model in models.items():
                expected = np.load(arguments.quality / "images" / f"{arm}_{index:04d}_{snr:g}_2001.npy")
                for repetition in range(repeats):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    signal = transmit(pixels, int(target["class_index"]), snr, model, vae, var, device)
                    torch.cuda.synchronize(device)
                    tx_seconds = time.perf_counter() - start
                    waveform = signal + noise
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    reconstructed = receive(waveform, snr, model, vae, var, device)
                    torch.cuda.synchronize(device)
                    rx_seconds = time.perf_counter() - start
                    error = float(np.max(np.abs(reconstructed - expected)))
                    energy = float(np.sum(signal ** 2))
                    if error > specification["image_replay_tolerance"] or abs(energy - 6120) > specification["energy_tolerance"]:
                        raise RuntimeError(f"timing replay/energy differs: image_error={error}, energy={energy}")
                    rows.append({"arm": arm, "image_index": int(index), "image_id": target["image_id"],
                                 "snr_db": snr, "seed": 2001, "repetition": repetition,
                                 "tx_seconds": tx_seconds, "rx_seconds": rx_seconds,
                                 "processing_seconds": tx_seconds + rx_seconds,
                                 "image_max_error": error, "total_complex_uses": len(signal), "total_energy": energy})
            write_csv(output / "per_frame.csv", rows)
    verify_snapshot(training["bindings"])
    write_json(output / "completion.json", {"status": "HYBRID_ONLY_TIMING_COMPLETE", "rows": len(rows),
               "sources": len(indices), "image_max_error": max(row["image_max_error"] for row in rows),
               "per_frame_sha256": sha256(output / "per_frame.csv"), "end_gpu": gpu_state()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
