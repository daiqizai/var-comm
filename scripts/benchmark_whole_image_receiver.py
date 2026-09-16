#!/usr/bin/env python3
"""Time waveform-to-image completion for the three core receivers on the fixed timing subset."""

import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from benchmark_whole_receiver import receive_api
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import complete_image, split_prefix
from var_comm.scale_channel import bits_to_indices, load_native
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, write_csv, write_json
from var_comm.whole_frame import source_tables
from var_comm.whole_list import load_list_native


def main():
    run = ROOT / "outputs/VAR-WHOLE-FRAME-PRIOR-001"
    receipt = verify_artifacts(run, "completion.json")
    config = yaml.safe_load((run / "snapshots/configs/whole_frame_prior.yaml").read_text())
    parent = ROOT / config["input_run"]
    model_config = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())
    output = create_output(ROOT / "outputs/VAR-WHOLE-IMAGE-LATENCY-001")
    sources = snapshot(output, [Path(__file__), ROOT / "scripts/benchmark_whole_receiver.py"])
    write_json(output / "protocol.json", {"role": "post-primary timing only; cannot replace quality outcomes", "image_indices": list(range(0, 100, 5)),
                                         "snr_db": [6.0, 7.0], "seed": 2001, "arms": ["whole_m9_ml", "whole_m9_list65", "whole_m9_var"],
                                         "boundary": "received waveform to RGB image; includes source and image decoding; excludes loading and IO"})
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    vae, var = load_models(model_config["paths"], device)
    load_native()
    load_list_native()
    with np.load(ROOT / "outputs/VAR-WHOLE-PRIOR-SELFCHECK-001/hypotheses.npz", allow_pickle=False) as cache:
        source_tables(vae, var, cache["bits"], int(cache["label"]), device)
        prefix = split_prefix(bits_to_indices(cache["bits"][0]), 8) + [np.zeros(169, dtype=np.int64)]
        complete_image(vae, var, prefix, int(cache["label"]), device)
    targets = json.loads((parent / "populations.json").read_text())["target"]
    rows = []
    for index in range(0, 100, 5):
        directory = run / "images" / f"{index:03d}"
        original = parent / "images" / f"{index:03d}"
        with np.load(original / "waveforms.npz", allow_pickle=False) as cache:
            transmitted = cache["whole_m9_0"].copy()
        with np.load(original / "reconstructions.npz", allow_pickle=False) as cache:
            old_images = cache["images"].copy()
        with np.load(directory / "reconstructions.npz", allow_pickle=False) as cache:
            new_images = cache["images"].copy()
        records = json.loads((directory / "receivers.json").read_text())
        for snr in (6.0, 7.0):
            received = transmitted + seeded_noise(targets[index]["image_id"], 2001, (3060, 2)) / np.sqrt(10 ** (snr / 10))
            record = next(frame for frame in records if frame["snr_db"] == snr and frame["seed"] == 2001)
            for name in ("whole_m9_ml", "whole_m9_list65", "whole_m9_var"):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                tick = time.perf_counter()
                result = receive_api(received, snr, name, vae, var, device, config["search"])
                torch.cuda.synchronize(device)
                token_seconds = time.perf_counter() - tick
                image = complete_image(vae, var, result["prefix"], result["label"], device)
                torch.cuda.synchronize(device)
                total_seconds = time.perf_counter() - tick
                kind, position = record["receivers"][name]["image_ref"].split(":")
                reference = (old_images if kind == "input" else new_images)[int(position)]
                error = float(np.max(np.abs(image - reference)))
                if error != 0:
                    raise RuntimeError("full image timing changed output")
                rows.append({"image_index": index, "snr_db": snr, "seed": 2001, "arm": name, "token_API_seconds": token_seconds,
                             "waveform_to_image_seconds": total_seconds, "image_stage_seconds": total_seconds - token_seconds,
                             "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(device), "max_pixel_error": error})
    summary = []
    for snr in (6.0, 7.0):
        for name in ("whole_m9_ml", "whole_m9_list65", "whole_m9_var"):
            selected = [row for row in rows if row["snr_db"] == snr and row["arm"] == name]
            values = [row["waveform_to_image_seconds"] for row in selected]
            summary.append({"snr_db": snr, "arm": name, "samples": len(values), "mean_seconds": float(np.mean(values)),
                            "p95_seconds": float(np.percentile(values, 95)), "mean_image_stage_seconds": float(np.mean([row["image_stage_seconds"] for row in selected])),
                            "max_GPU_allocated_bytes": max(row["peak_GPU_allocated_bytes"] for row in selected)})
    models = {"vae": state_sha256(vae), "var": state_sha256(var)}
    if any(value != receipt["frozen_before"][name] for name, value in models.items()):
        raise RuntimeError("image timing modified models")
    write_csv(output / "per_call.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "completion.json", {"status": "WAVEFORM_TO_IMAGE_TIMING_COMPLETE", "calls": len(rows), "source_images": 20,
                                            "source_hashes": sources, "input_completion_sha256": sha256(run / "completion.json"),
                                            "max_pixel_error": 0.0, "frozen_models": models, "output_hashes": artifact_hashes(output)})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
