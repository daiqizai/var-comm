#!/usr/bin/env python3
"""Supplement only the new budgets' complete CPU-to-CPU digital processing costs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src"), str(PROJECT / "scripts"), str(EXPERIMENT / "scripts")]

import numpy as np
import torch
import yaml

from benchmark_frozen_systems import assert_gpu_available, gpu_state
from evaluate_digital_budgets import digest, read_csv
from external_positioning.analysis import SCOPES, SNRS
from external_positioning.digital_budget import receive, transmit
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.online_timing import SOURCE_INDICES, compare_arrays, timed_call
from var_comm.study import seeded_noise, sha256, write_csv, write_json
from var_comm.whole_entropy import decode_source, encode_prefixes


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = json.loads((arguments.development / "completion.json").read_text())
    if receipt["status"] != "COMMON_BUDGET_DIGITAL_COMPLETE" or receipt["population"] != "development":
        raise RuntimeError("new budget timing requires complete frozen development results")
    arguments.output.mkdir(parents=True, exist_ok=False)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((PROJECT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    vae, var = load_models(paths, device)
    before = {"vae": state_sha256(vae), "var": state_sha256(var)}
    if any(before[name] != receipt["frozen_models"][name] for name in before):
        raise RuntimeError("timing visual weights differ from quality evaluation")
    counts = {"vae": sum(parameter.numel() for parameter in vae.parameters()),
              "var": sum(parameter.numel() for parameter in var.parameters())}
    source_inputs = {int(item["image_index"]): item for item in json.loads((arguments.root / "inputs_001/development_inputs.json").read_text())}
    all_quality = read_csv(arguments.development / "per_frame.csv")
    references = {(int(row["image_index"]), float(row["snr_db"]), row["family"], int(row["complex_uses"])): row
                  for row in all_quality if int(row["seed"]) == 2001}
    inputs = [Path(__file__), EXPERIMENT / "configs/digital_common_budget.json", arguments.development / "per_frame.csv"]
    bindings = {str(path): sha256(path) for path in inputs}
    write_json(arguments.output / "metadata.json", {"GPU": gpu_state(), "FP32_TF32_disabled": True, "batch": 1,
               "endpoints": "CPU_source_pixels_to_complete_CPU_waveform;CPU_received_waveform_to_CPU_RGB",
               "excludes": ["loading", "file_IO", "metrics", "airtime", "queueing", "class_acquisition", "CSI_estimation"],
               "resident_models": ["frozen_VAE", "frozen_VAR"], "parameters": counts,
               "no_RX_or_TX_encoding_cache": True, "bindings": bindings})

    def transmitter(pixels, label, mode, family, budget):
        image = torch.from_numpy(pixels[None]).float().div(127.5).sub(1).to(device)
        tokens = vae.img_to_idxBl(image)
        prefix = [item[0].cpu().numpy() for item in tokens[:mode]]
        payload = encode_prefixes(vae, var, prefix, label, device, modes=(mode,))[mode]["payload"] if family == "arithmetic" else None
        return transmit(prefix, payload, label, mode, family, budget)[0]

    def receiver(observed, snr, family):
        physical = receive(observed, snr, family)
        if family == "raw":
            physical["header"]["length_field"] = 0
        return decode_source(physical, vae, var, device)["image"]

    started, records, warmed = time.perf_counter(), [], set()
    for position, index in enumerate(SOURCE_INDICES):
        assert_gpu_available()
        item = source_inputs[index]
        pixels = np.load(item["path"], allow_pickle=False)
        for snr in SNRS:
            for budget in (4204, 4498):
                for family in ("raw", "arithmetic"):
                    reference = references[index, snr, family, budget]
                    mode = int(reference["mode"])
                    with np.load(reference["image_archive"], allow_pickle=False) as archive:
                        expected_image = archive["images"][int(reference["image_slot"])].copy()
                    signal = transmitter(pixels, item["class_index"], mode, family, budget)
                    observed = signal + seeded_noise(item["image_id"], 2001, (budget, 2)) / np.sqrt(10 ** (snr / 10))
                    if digest(signal) != reference["transmitted_sha256"] or digest(observed) != reference["received_sha256"]:
                        raise RuntimeError("timing did not reproduce the quality run's actual waveform")
                    if (family, budget, mode) not in warmed:
                        receiver(observed, snr, family)
                        warmed.add((family, budget, mode))
                    for repeat in range(3):
                        torch.cuda.reset_peak_memory_stats(device)
                        actual_signal, tx_seconds = timed_call(lambda: transmitter(pixels, item["class_index"], mode, family, budget), torch.cuda.synchronize)
                        actual_image, rx_seconds = timed_call(lambda: receiver(observed, snr, family), torch.cuda.synchronize)
                        signal_error, signal_exact = compare_arrays(actual_signal, signal, 0., "digital waveform")
                        image_error, image_exact = compare_arrays(actual_image, expected_image, 1e-5, "digital image")
                        records.append({"image_index": index, "image_id": item["image_id"], "snr_db": snr, "seed": 2001,
                                        "repeat": repeat, "method": reference["method"], "mode": mode, "family": family,
                                        "complex_uses": budget, "total_energy": 2 * budget,
                                        "TX_seconds": tx_seconds, "RX_seconds": rx_seconds,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                        "parameters_vae": counts["vae"], "parameters_var": counts["var"],
                                        "signal_exact": signal_exact, "image_exact": image_exact,
                                        "signal_max_error": signal_error, "image_max_error": image_error,
                                        "header_accepted": reference["header_accepted"], "body_crc_accepted": reference["body_crc_accepted"],
                                        "actual_image_sha256": digest(actual_image), "actual_signal_sha256": digest(actual_signal)})
        write_json(arguments.output / "status.json", {"status": "NEW_BUDGET_ONLINE_TIMING", "sources_completed": position + 1,
                   "timing_rows": len(records), "seconds": time.perf_counter() - started, "updated_at": datetime.now().astimezone().isoformat()})
        if (position + 1) % 8 == 0:
            print(f"Online timing: {position + 1}/32 sources, {len(records)} calls", flush=True)
    write_csv(arguments.output / "per_call.csv", records)
    summary = []
    for method in sorted({row["method"] for row in records}):
        for scope, support in SCOPES:
            selected = [row for row in records if row["method"] == method and row["snr_db"] in support]
            tx = np.array([row["TX_seconds"] * 1000 for row in selected])
            rx = np.array([row["RX_seconds"] * 1000 for row in selected])
            summary.append({"method": method, "complex_uses": selected[0]["complex_uses"], "scope": scope,
                            "source_images": len(SOURCE_INDICES), "timing_rows": len(selected), "TX_mean_ms": float(tx.mean()),
                            "RX_mean_ms": float(rx.mean()), "TX_p95_ms": float(np.quantile(tx, .95)), "RX_p95_ms": float(np.quantile(rx, .95)),
                            "single_method_peak_allocated_MiB": max(row["peak_allocated_bytes"] for row in selected) / 2**20,
                            "communication_parameters": 0, "NFE_mean": 0,
                            "scope_note": f"VAE+VAR total parameters {sum(counts.values())}; single digital system resident; uncached CPU_to_CPU"})
    write_csv(arguments.output / "summary.csv", summary)
    if before != {"vae": state_sha256(vae), "var": state_sha256(var)}:
        raise RuntimeError("timing changed frozen models")
    write_json(arguments.output / "completion.json", {"status": "NEW_BUDGET_DIGITAL_TIMING_COMPLETE", "rows": len(records),
               "seconds": time.perf_counter() - started, "parameters": counts, "bindings": bindings,
               "per_call_sha256": sha256(arguments.output / "per_call.csv"), "summary_sha256": sha256(arguments.output / "summary.csv"),
               "no_old_full_benchmark_rerun": True, "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    main()
