#!/usr/bin/env python3
"""Measure complete receiver API latency on a fixed subset without revising quality results."""

import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from evaluate_whole_frame_prior import completed_receiver, serial_choice
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import receive_whole
from var_comm.scale_channel import channel_evidence, load_native, rate_match_indices
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json
from var_comm.whole_frame import cache_empty, conditional_branches, source_tables
from var_comm.whole_list import load_list_native, ordered_paths


def receive_api(received, snr, name, vae, var, device, search):
    original = receive_whole(received, snr)
    trigger = original["header"]["accepted"] and original["header"]["mode"] == 9 and not original["events"][0]["accepted"]
    if name in ("whole_m8", "whole_m9_ml") or not trigger:
        return original
    evidence = channel_evidence(received[68:], rate_match_indices(10220, 5984), 5110, snr)
    if name in ("whole_m9_list65", "whole_m9_list_time"):
        timed = name == "whole_m9_list_time"
        listed = ordered_paths(evidence, search["time_control_max_candidates"] if timed else search["count_control_candidates"],
                               payload_bits=5088, stop_on_crc=True, seconds=search["time_control_seconds"] if timed else -1,
                               queue_limit=search["maximum_heap_nodes"])
        return completed_receiver(original, serial_choice(listed))
    prefixes = ordered_paths(evidence, search["distinct_prefixes"], stop_bits=3060, queue_limit=search["maximum_heap_nodes"])
    if name == "whole_m9_var":
        tables, prefix_scores = source_tables(vae, var, prefixes["bits"], original["label"], device)
        result = conditional_branches(evidence, prefixes, search, tables, prefix_scores)
    else:
        result = conditional_branches(evidence, prefixes, search)
    return completed_receiver(original, result["information"])


def main():
    run = ROOT / "outputs/VAR-WHOLE-FRAME-PRIOR-001"
    receipt = verify_artifacts(run, "completion.json")
    verify_snapshot(receipt["source_hashes"])
    config = yaml.safe_load((run / "snapshots/configs/whole_frame_prior.yaml").read_text())
    parent = ROOT / config["input_run"]
    upstream = yaml.safe_load((parent / "snapshots/configs/progressive_channel.yaml").read_text())
    model_config = yaml.safe_load((ROOT / upstream["model_config"]).read_text())
    output = create_output(ROOT / "outputs/VAR-WHOLE-RECEIVER-LATENCY-001")
    sources = snapshot(output, [Path(__file__), ROOT / "scripts/evaluate_whole_frame_prior.py", ROOT / "reports/whole_frame_latency_protocol_2026-09-07.md"])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    vae, var = load_models(model_config["paths"], device)
    before = {"vae": state_sha256(vae), "var": state_sha256(var)}
    load_native()
    load_list_native()
    with np.load(ROOT / "outputs/VAR-WHOLE-PRIOR-SELFCHECK-001/hypotheses.npz", allow_pickle=False) as cache:
        source_tables(vae, var, cache["bits"], int(cache["label"]), device)
    targets = json.loads((parent / "populations.json").read_text())["target"]
    names = config["arms"][:6]
    rows = []
    for index in range(0, 100, 5):
        target = targets[index]
        with np.load(parent / "images" / f"{index:03d}" / "waveforms.npz", allow_pickle=False) as cache:
            waves = {mode: cache[f"whole_m{mode}_0"].copy() for mode in (8, 9)}
        frames = json.loads((run / "images" / f"{index:03d}" / "receivers.json").read_text())
        seed = config["noise_seeds"][0]
        for snr in config["snr_db"]:
            frame = next(record for record in frames if record["seed"] == seed and record["snr_db"] == snr)
            noise = seeded_noise(target["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
            for name in names:
                observed = waves[8 if name == "whole_m8" else 9] + noise
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                tick = time.perf_counter()
                result = receive_api(observed, snr, name, vae, var, device, config["search"])
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - tick
                original = frame["receivers"][name]
                match = (result["header"] == original["header"] and result["label"] == original["label"]
                         and [scale.tolist() for scale in result["prefix"]] == original["output_prefix"]
                         and result["last_accepted_scale"] == original["last_accepted_scale"])
                if (not match and name != "whole_m9_list_time") or not cache_empty(var):
                    raise RuntimeError("fixed-budget receiver output changed in API replay")
                rows.append({"image_index": index, "image_id": target["image_id"], "snr_db": snr, "seed": seed, "arm": name,
                             "receiver_API_seconds": elapsed, "matches_primary": int(match),
                             "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        print(f"receiver API timing {index // 5 + 1}/20", flush=True)
    summary = []
    for snr in config["snr_db"]:
        for name in names:
            selected = [row for row in rows if row["snr_db"] == snr and row["arm"] == name]
            values = [row["receiver_API_seconds"] for row in selected]
            summary.append({"snr_db": snr, "arm": name, "samples": len(values), "mean_seconds": float(np.mean(values)),
                            "p95_seconds": float(np.percentile(values, 95)), "max_seconds": max(values),
                            "output_mismatches": sum(1 - row["matches_primary"] for row in selected),
                            "max_GPU_allocated_bytes": max(row["peak_GPU_allocated_bytes"] for row in selected),
                            "max_GPU_reserved_bytes": max(row["peak_GPU_reserved_bytes"] for row in selected)})
    if before != {"vae": state_sha256(vae), "var": state_sha256(var)}:
        raise RuntimeError("timing run modified models")
    write_csv(output / "per_call.csv", rows)
    write_csv(output / "summary.csv", summary)
    verify_snapshot(sources)
    result = {"status": "RECEIVER_API_TIMING_COMPLETE", "calls": len(rows), "source_images": 20,
              "main_completion_sha256": sha256(run / "completion.json"), "source_hashes": sources, "frozen_models": before,
              "output_mismatches": sum(1 - row["matches_primary"] for row in rows),
              "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "excludes": ["model loading", "file IO", "final image generation"], "output_hashes": artifact_hashes(output)}
    write_json(output / "completion.json", result)
    print({name: value for name, value in result.items() if name not in ("source_hashes", "output_hashes")}, flush=True)


if __name__ == "__main__":
    main()
