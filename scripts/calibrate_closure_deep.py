#!/usr/bin/env python3
"""Only the missing frozen DeepJSCC calibration for content-independent mixing."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/wetok-comm-v2-20260912/src")]

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available
from evaluate_communication_modes import Population
from var_comm.prefix_training_data import read_image_population, MANIFEST_SHA
from var_comm.quality import dino_features, load_quality_models
from var_comm.study import seeded_noise, sha256, snapshot, verify_snapshot, write_csv, write_json
from wetok_comm.deep_support import FrozenDeepSupport, load_deep_support


@torch.no_grad()
def receive_batch(system, pixels, snrs, standard_noise, device):
    images = torch.from_numpy(np.stack(pixels)).to(device).float().div(255)
    condition = torch.tensor(snrs, dtype=torch.float32, device=device)
    encoded = system.model.encode(images, condition)
    normalized, _power = system.model.normalize_channel_input(encoded)
    active = normalized.flatten(1).index_select(1, system.model.active_real_indices)
    transmitted = active.cpu().numpy().reshape(-1, 3060, 2).astype(np.float64)
    observed = transmitted + standard_noise / np.sqrt(10 ** (np.asarray(snrs)[:, None, None] / 10))
    received = torch.from_numpy(observed.reshape(-1, 6120)).to(device).float()
    flat = received.new_zeros(len(images), system.model.native_real_symbols)
    latent = flat.index_copy(1, system.model.active_real_indices, received).reshape(len(images), *system.layout)
    decoded = system.model.decode(latent, condition).clamp(0, 1)
    energies = np.sum(transmitted ** 2, axis=(1, 2))
    if np.max(np.abs(energies - 6120)) > .02:
        raise RuntimeError("frozen Deep is not operating at the actual registered energy")
    return images, decoded, transmitted, observed, energies


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    base = json.loads((ROOT / "configs/hybrid_base_conditioning.json").read_text())
    sources = snapshot(output, [Path(__file__), ROOT / "configs/hybrid_weight_closure.json"])
    support = load_deep_support()
    system = FrozenDeepSupport(support, device)
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    perceptual, dino, _weights = load_quality_models(quality_paths, device)
    old = ROOT / "outputs/VAR-PREFIX-JSCC-EVAL-001"
    receipt = json.loads((old / "completion.json").read_text())
    if sha256(old / "per_frame.csv") != receipt["output_hashes"]["per_frame.csv"]:
        raise RuntimeError("frozen original Deep reference changed")
    with (old / "per_frame.csv").open() as handle:
        references = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"])): row for row in csv.DictReader(handle) if row["arm"] == "perceptual_deepjscc"}
    development = Population("development", json.loads((ROOT / "configs/communication_decision_study.json").read_text()))
    replay_max = 0.
    for index in (0, 99):
        pixels, _scales, target = development.source(index, None, None)
        for snr in base["snrs_db"]:
            entries = references[index, snr, 2001]
            noise = seeded_noise(target["image_id"], 2001, (3060, 2))[None]
            _source, reconstructed, _signal, _received, _energies = receive_batch(system, [pixels], [snr], noise, device)
            if not entries["image_ref"].startswith("new:"):
                raise RuntimeError("unknown frozen Deep archive reference")
            with np.load(old / "images" / f"{index:03d}" / "reconstructions.npz", allow_pickle=False) as archive:
                expected = archive["images"][int(entries["image_ref"].split(":")[1])]
            difference = float(np.max(np.abs(reconstructed[0].cpu().numpy() - expected)))
            replay_max = max(replay_max, difference)
            if difference > 1e-4:
                raise RuntimeError("frozen Deep actual-physics replay changed")
    write_json(output / "replay_check.json", {"status": "PASS", "source_indices": [0, 99], "SNRs": base["snrs_db"], "image_max_error": replay_max,
               "old_reference_sha256": sha256(old / "per_frame.csv"), "weights_sha256": support["checkpoint_sha256"], "new_training": False})
    if arguments.mode == "smoke":
        write_json(output / "completion.json", {"status": "FROZEN_DEEP_REPLAY_PASS", "new_training": False})
        return
    images, labels, identifiers, bindings = read_image_population("calibration")
    if len(images) != 1000:
        raise RuntimeError("expected the original 1000-source calibration population")
    population = [{"image_id": identifier, "class_index": int(label), "binding": binding}
                  for identifier, label, binding in zip(identifiers, labels, bindings)]
    write_json(output / "population.json", population)
    write_json(output / "metadata.json", {"role": "missing_calibration_only", "population_manifest_sha256": MANIFEST_SHA,
               "weights_sha256": support["checkpoint_sha256"], "seeds": base["calibration_seeds"], "SNRs": base["snrs_db"],
               "N": 3060, "E": 6120, "header": 0, "noise_real_variance": "1/gamma", "new_training": False})
    items = [(index, snr, seed) for index in range(1000) for snr in base["snrs_db"] for seed in base["calibration_seeds"]]
    records = []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(items), 16):
            entries = items[offset:offset + 16]
            pixels = [images[index].numpy() for index, _snr, _seed in entries]
            snrs = [snr for _index, snr, _seed in entries]
            noise = np.stack([seeded_noise(identifiers[index], seed, (3060, 2)) for index, _snr, seed in entries])
            source, decoded, transmitted, received, energies = receive_batch(system, pixels, snrs, noise, device)
            mse = (source - decoded).square().flatten(1).mean(1)
            lpips = perceptual(decoded * 2 - 1, source * 2 - 1).reshape(-1)
            similarity = torch.nn.functional.cosine_similarity(dino_features(dino, decoded), dino_features(dino, source), dim=1)
            for local, (index, snr, seed) in enumerate(entries):
                metrics = {"mse": float(mse[local]), "psnr_db": float(-10 * mse[local].log10()), "lpips": float(lpips[local]), "dino": float(similarity[local])}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("a frozen Deep calibration frame cannot be silently dropped")
                records.append({"image_index": index, "image_id": identifiers[index], "snr_db": snr, "seed": seed,
                                "arm": "perceptual_deepjscc", **metrics, "complex_uses": 3060, "total_energy": float(energies[local]),
                                "source_pixels_sha256": hashlib.sha256(pixels[local].tobytes()).hexdigest(),
                                "standard_noise_sha256": hashlib.sha256(noise[local].tobytes()).hexdigest(),
                                "transmitted_sha256": hashlib.sha256(transmitted[local].tobytes()).hexdigest(),
                                "received_sha256": hashlib.sha256(received[local].tobytes()).hexdigest(),
                                "image_sha256": hashlib.sha256(decoded[local].cpu().numpy().tobytes()).hexdigest()})
            if len(records) % 800 == 0 or offset + 16 >= len(items):
                write_json(output / "status.json", {"status": "FROZEN_DEEP_CALIBRATION", "frames": len(records), "expected": 15000,
                           "seconds": time.perf_counter() - start, "updated_at": datetime.now().astimezone().isoformat()})
                print(f"Deep calibration {len(records)}/15000", flush=True)
    write_csv(output / "per_frame.csv", records)
    verify_snapshot(sources)
    write_json(output / "completion.json", {"status": "FROZEN_DEEP_CALIBRATION_COMPLETE", "sources": 1000, "frames": len(records),
               "new_training": False, "new_holdout": False, "per_frame_sha256": sha256(output / "per_frame.csv"),
               "seconds_this_stage": time.perf_counter() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / f"failure_{time.time_ns()}.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
