#!/usr/bin/env python3
"""Build deterministic clean TX bases; never simulate oracle RX information."""

import argparse
from datetime import datetime
import json
import os
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

from var_comm.hybrid_correction import HybridBudget, encode_digital, receive_digital
from var_comm.next_scale_prior import load_models
from var_comm.prefix_training_data import read_image_population, MANIFEST_SHA
from var_comm.progressive import complete_image, split_prefix
from var_comm.study import sha256, write_json


def prepare(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/hybrid_source_correction.json"
    config = json.loads(config_path.read_text())
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    bindings = {str(config_path): sha256(config_path), "source_manifest": MANIFEST_SHA,
                str(Path(__file__)): sha256(Path(__file__)),
                str(ROOT / "src/var_comm/hybrid_correction.py"): sha256(ROOT / "src/var_comm/hybrid_correction.py"),
                str(ROOT / "src/var_comm/progressive.py"): sha256(ROOT / "src/var_comm/progressive.py")}
    for name in ("vae_checkpoint", "var_checkpoint"):
        if sha256(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError("frozen visual checkpoint changed")
        bindings[paths[name]] = paths[name + "_sha256"]
    metadata = {"bindings": bindings, "precision": "float32_batch1_no_tf32",
                "model": "official_frozen_VAR_d16", "budget": HybridBudget().ledger(),
                "purpose": "TX_clean_source_bases_not_RX_oracle", "limit": arguments.limit}
    if (output / "metadata.json").exists():
        if json.loads((output / "metadata.json").read_text()) != metadata:
            raise RuntimeError("resume cannot change cache source, code, budget, or population")
    else:
        write_json(output / "metadata.json", metadata)
    if (output / "completion.json").exists():
        raise RuntimeError("cache is already complete; do not restart")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    vae, var = load_models(paths, device)
    start = time.perf_counter()
    for population in ("train", "calibration"):
        images, labels, identifiers, source_bindings = read_image_population(population)
        count = min(arguments.limit, len(images)) if arguments.limit else len(images)
        directory = output / population
        directory.mkdir(exist_ok=True)
        populations = [{"image_id": identifiers[index], "class_index": int(labels[index]),
                        "source": source_bindings[index]} for index in range(count)]
        population_path = directory / "population.json"
        if population_path.exists() and json.loads(population_path.read_text()) != populations:
            raise RuntimeError("population changed on resume")
        write_json(population_path, populations)
        progress_path = directory / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": 0}
        mode = "r+" if (directory / "bases.npy").exists() else "w+"
        arrays = {"bases": np.lib.format.open_memmap(directory / "bases.npy", mode=mode, dtype=np.float32,
                                                    shape=(count, 3, 256, 256)),
                  "tokens": np.lib.format.open_memmap(directory / "tokens.npy", mode=mode, dtype=np.uint16,
                                                     shape=(count, 155)),
                  "digital": np.lib.format.open_memmap(directory / "digital.npy", mode=mode, dtype=np.int8,
                                                      shape=(count, 1950, 2))}
        for index in range(progress["completed"], count):
            with torch.no_grad():
                source = images[index:index + 1].to(device).float().div(127.5).sub(1)
                scales = [value[0].cpu().numpy() for value in vae.img_to_idxBl(source)]
                prefix = np.concatenate(scales[:7])
                base = complete_image(vae, var, scales[:7], int(labels[index]), device)
            waveform = encode_digital(prefix, int(labels[index]))
            if index < 2:
                received = receive_digital(waveform, 19.0)
                if not received["crc_accepted"] or received["label"] != int(labels[index]):
                    raise RuntimeError("noiseless digital acceptance failed")
                np.testing.assert_array_equal(received["tokens"], prefix)
                replay = complete_image(vae, var, split_prefix(received["tokens"], 7), received["label"], device)
                np.testing.assert_array_equal(base, replay)
            arrays["bases"][index] = base
            arrays["tokens"][index] = prefix
            arrays["digital"][index] = waveform.astype(np.int8)
            if (index + 1) % 50 == 0 or index + 1 == count:
                for array in arrays.values():
                    array.flush()
                state = {"status": "BUILDING_TX_CLEAN_BASES", "population": population,
                         "completed": index + 1, "count": count, "pid": os.getpid(),
                         "seconds": time.perf_counter() - start, "updated_at": datetime.now().astimezone().isoformat()}
                write_json(progress_path, state)
                write_json(output / "status.json", state)
                print(json.dumps(state), flush=True)
        del arrays, images
        files = {name: sha256(directory / name) for name in ("bases.npy", "tokens.npy", "digital.npy", "population.json")}
        write_json(directory / "completion.json", {"sources": count, "hashes": files})
    write_json(output / "completion.json", {"status": "COMPLETE", "seconds_this_session": time.perf_counter() - start,
                                            "metadata_sha256": sha256(output / "metadata.json"),
                                            "train_receipt_sha256": sha256(output / "train/completion.json"),
                                            "calibration_receipt_sha256": sha256(output / "calibration/completion.json")})
    write_json(output / "status.json", {"status": "COMPLETE", "pid": os.getpid(),
                                       "updated_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED", "pid": os.getpid(), "traceback": traceback.format_exc()}
        write_json(args.output / f"failure_{time.time_ns()}.json", failure)
        write_json(args.output / "status.json", failure)
        raise
