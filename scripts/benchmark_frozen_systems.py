#!/usr/bin/env python3
"""Measure existing frozen systems at common CPU endpoints, never train or select models."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from var_comm.online_timing import ARMS, SNRS, SOURCE_INDICES, compare_arrays, measurement_order, summarize, timed_call, validate_config, validate_rows
from var_comm.frozen_timing_adapters import ArchiveInputs, FrozenSystems, install_frozen_paths, loaded_local_sources, rows_from
from var_comm.study import artifact_hashes, sha256, write_csv, write_json


def now():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    write_json(temporary, value)
    temporary.replace(path)


def digest(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def assert_gpu_available():
    output = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("cannot verify GPU process ownership")
    foreign = {int(line) for line in lines} - {os.getpid()}
    if foreign:
        raise RuntimeError(f"GPU occupied by other processes {sorted(foreign)}; leaving them untouched")


def gpu_state():
    return subprocess.check_output(["nvidia-smi", "-i", "0",
        "--query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader"], text=True).strip()


def save_sources(output, paths):
    records = {}
    for path in paths:
        relative = path.relative_to(ROOT.parent)
        target = output / "snapshots" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        records[str(path)] = sha256(path)
    return records


def draw(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names = {"whole_m7": "Digital VAR m7", "whole_m8": "Digital VAR m8", "whole_m9": "Digital VAR m9",
             "whole_adaptive": "Digital VAR adaptive", "r2__full_grid_innovation": "Frozen R2 residual",
             "perceptual_deepjscc": "Perceptual DeepJSCC", "wetok_8PSK_FEC": "WeTok 8PSK + FEC"}
    for arm in ARMS:
        selected = [next(row for row in summary if row["scope"] == str(int(snr)) and row["arm"] == arm) for snr in SNRS]
        for axis, metric, title in zip(axes, ("TX_mean_ms", "RX_mean_ms"), ("CPU RGB to CPU waveform", "CPU waveform to CPU RGB")):
            axis.plot(SNRS, [row[metric] for row in selected], marker="o", label=names[arm])
            axis.set(xlabel="SNR (dB)", ylabel="Mean processing time (ms)", title=title)
            axis.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Frozen systems: N=3060, E=6120; 32 development sources; three timing repeats\nSame CPU endpoints; airtime, queueing, IO and model loading excluded", fontsize=10)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"same_endpoint_latency.{extension}", dpi=180)
    plt.close(figure)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frozen_system_online_timing.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    validate_config(config)
    if not arguments.execute:
        print("PLAN ONLY: existing frozen systems, no GPU execution without --execute")
        return
    indices = list(SOURCE_INDICES if arguments.mode == "full" else config["smoke_source_indices"])
    repeats = config["repeats"] if arguments.mode == "full" else config["smoke_repeats"]
    output = arguments.output_dir.resolve()
    if not output.is_relative_to((ROOT / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915").resolve()):
        raise ValueError("new timing output must remain in its declared project directory")
    if (output / "completion.json").exists():
        raise RuntimeError("this timing run is already complete")
    output.mkdir(parents=True, exist_ok=arguments.resume)
    (output / "sessions").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    session = output / "sessions" / f"{time.time_ns()}_{os.getpid()}.json"
    started = time.perf_counter()
    completed = 0
    local = []

    def status(value, **extra):
        record = {"status": value, "pid": os.getpid(), "updated_local": now(), "completed_sources": completed,
                  "source_count": len(indices), "elapsed_process_seconds": time.perf_counter() - started,
                  "research_goal_complete": False, **extra}
        atomic_json(session, record)
        atomic_json(output / "status.json", record)

    try:
        status("VERIFYING_AND_LOADING_FROZEN_SYSTEMS")
        if arguments.mode == "full":
            if arguments.smoke_receipt is None:
                raise RuntimeError("full timing requires the successful same-source smoke receipt")
            smoke = json.loads(arguments.smoke_receipt.read_text())
            if smoke["status"] != "FROZEN_SYSTEM_TIMING_SMOKE_COMPLETE" or smoke["config_sha256"] != sha256(arguments.config):
                raise RuntimeError("smoke protocol does not match full timing")
            for path, expected in smoke["source_hashes"].items():
                if sha256(path) != expected:
                    raise RuntimeError(f"source changed after GPU smoke: {path}")
        assert_gpu_available()
        install_frozen_paths()
        inputs = ArchiveInputs()
        systems = FrozenSystems(inputs, "cuda:0")
        files = loaded_local_sources() + [Path(__file__).resolve(), arguments.config.resolve(),
                ROOT / "reports/frozen_system_online_timing_protocol_20260915.md"]
        source_hashes = {str(path): sha256(path) for path in set(files)}
        if arguments.resume:
            metadata = json.loads((output / "metadata.json").read_text())
            if (metadata["config_sha256"] != sha256(arguments.config) or metadata["mode"] != arguments.mode or
                    metadata["model_hashes"] != systems.before or metadata["source_hashes"] != source_hashes):
                raise RuntimeError("resume changed frozen source, model or measurement boundaries")
        else:
            source_hashes = save_sources(output, sorted(set(files)))
            metadata = {"started_local": now(), "command": sys.argv, "mode": arguments.mode,
                        "config_sha256": sha256(arguments.config), "source_hashes": source_hashes,
                        "model_hashes": systems.before, "selected_R2_checkpoint": systems.choice,
                        "source_indices": indices, "repeats": repeats, "python": sys.version,
                        "torch": torch.__version__, "numpy": np.__version__, "GPU": gpu_state(),
                        "GPU_isolation_observation": "polled before/after groups, not proof of continuous exclusivity",
                        "memory_scope": "all frozen models co-resident, not single-model deployment memory",
                        "new_training": False, "new_architecture": False, "new_holdout": False}
            atomic_json(output / "metadata.json", metadata)
        rows, warmed = [], set()
        synchronize = lambda: torch.cuda.synchronize(0)
        for position, index in enumerate(indices):
            directory = output / "images" / f"{index:03d}"
            committed = directory / "receipt.json"
            if committed.exists():
                saved = json.loads(committed.read_text())
                for relative, expected in saved["output_hashes"].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError("committed timing source was changed")
                rows.extend(rows_from(directory / "per_call.csv"))
                completed += 1
                continue
            assert_gpu_available()
            pixels, target, references = inputs.source(index, systems.modem)
            systems.check_native_source(pixels, np.array(inputs.codes[index, 0], copy=True))
            directory.mkdir(exist_ok=True)
            local, hardware = [], [{"phase": "before_source", "value": gpu_state()}]
            for snr_index, snr in enumerate(SNRS):
                for arm in ARMS:
                    if (arm, snr) not in warmed:
                        for warmup in range(config["warmups_per_arm_and_snr"]):
                            systems.transmit(pixels, snr, arm, target["class_index"])
                            systems.receive(references[snr, arm]["received"], snr, arm)
                        warmed.add((arm, snr))
                for repeat in range(repeats):
                    assert_gpu_available()
                    order = measurement_order(position * len(SNRS) + snr_index, repeat)
                    for order_index, arm in enumerate(order):
                        reference = references[snr, arm]
                        received = reference["received"]
                        observed_hash = digest(received)
                        torch.cuda.reset_peak_memory_stats(0)
                        signal, tx_seconds = timed_call(lambda: systems.transmit(pixels, snr, arm, target["class_index"]), synchronize)
                        (image, flags), rx_seconds = timed_call(lambda: systems.receive(received, snr, arm), synchronize)
                        tolerance = config["digital_signal_max_error"] if arm.startswith("whole_") or arm == "wetok_8PSK_FEC" else config["signal_max_error"]
                        signal_error, same_signal = compare_arrays(signal, reference["signal"], tolerance, "transmitted waveform")
                        image_error, same_image = compare_arrays(image, reference["image"], config["image_max_error"], "received RGB")
                        energy = float(np.square(signal.astype(np.float64)).sum())
                        if (signal.shape != (3060, 2) or abs(energy - config["total_energy"]) > config["energy_absolute_tolerance"] or
                                digest(received) != observed_hash or flags != {key: reference[key] for key in flags}):
                            raise RuntimeError("physical resources, observed waveform or failure decision changed")
                        failure = "header" if flags["header_accepted"] is False else ("body_crc" if flags["body_crc_accepted"] is False else
                                  ("accepted" if flags["body_crc_accepted"] is True else "not_applicable"))
                        local.append({"image_index": index, "image_id": target["image_id"], "snr_db": snr,
                            "noise_seed": config["noise_seed"], "arm": arm, "repeat": repeat, "order_index": order_index,
                            "total_complex_uses": 3060, "total_energy": energy,
                            "header_uses": 68 if arm.startswith("whole_") else 0,
                            "data_uses": 2992 if arm.startswith("whole_") else 3060,
                            "TX_seconds": tx_seconds, "RX_seconds": rx_seconds, "processing_sum_seconds": tx_seconds + rx_seconds,
                            "timing_scope": "CPU_to_CPU_contiguous_TX_and_RX", "failure_stratum": failure,
                            "received_sha256": observed_hash, "actual_transmitted_sha256": digest(signal), "reference_transmitted_sha256": digest(reference["signal"]),
                            "actual_image_sha256": digest(image), "reference_image_sha256": digest(reference["image"]),
                            "signal_max_error": signal_error, "image_max_error": image_error,
                            "signal_exact": same_signal, "image_exact": same_image,
                            "GPU_all_models_peak_allocated_bytes": torch.cuda.max_memory_allocated(0)})
                    assert_gpu_available()
            hardware.append({"phase": "after_source", "value": gpu_state()})
            write_csv(directory / "per_call.csv", local)
            atomic_json(directory / "hardware.json", hardware)
            atomic_json(committed, {"image_id": target["image_id"], "rows": len(local), "source_sha256": digest(pixels),
                                   "output_hashes": artifact_hashes(directory)})
            rows.extend(local)
            completed += 1
            status("MEASURING_CPU_ENDPOINT_LATENCY", last_image_index=index, completed_rows=len(rows))
            print(f"{now()} {arguments.mode} sources={completed}/{len(indices)} rows={len(rows)}", flush=True)
        status("VERIFYING_MODELS_AND_SUMMARIZING", completed_rows=len(rows))
        assert_gpu_available()
        after = systems.model_hashes()
        if after != systems.before:
            raise RuntimeError("frozen model state changed during timing")
        for path, expected in source_hashes.items():
            if sha256(path) != expected:
                raise RuntimeError(f"source changed during timing: {path}")
        for path, expected in inputs.hashes.items():
            if sha256(path) != expected:
                raise RuntimeError(f"frozen timing input changed: {path}")
        validate_rows(rows, indices, repeats)
        summary, paired = summarize(rows, config, indices, repeats)
        write_csv(output / "per_call.csv", rows)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired.csv", paired)
        draw(summary, output)
        finished = "FROZEN_SYSTEM_CPU_ENDPOINT_TIMING_COMPLETE" if arguments.mode == "full" else "FROZEN_SYSTEM_TIMING_SMOKE_COMPLETE"
        status(finished, completed_rows=len(rows))
        result = {"status": finished, "completed_local": now(), "mode": arguments.mode, "rows": len(rows),
                  "source_images": len(indices), "repeats": repeats, "timing_conditions_not_independent_sources": True,
                  "config_sha256": sha256(arguments.config), "source_hashes": source_hashes, "input_sha256": inputs.hashes,
                  "frozen_models_before": systems.before, "frozen_models_after": after,
                  "max_signal_error": max(float(row["signal_max_error"]) for row in rows),
                  "max_image_error": max(float(row["image_max_error"]) for row in rows),
                  "all_failures_included": True, "new_quality_results": False, "new_training": False,
                  "new_architecture": False, "new_holdout": False, "research_goal_complete": False,
                  "process_wall_seconds_this_session": time.perf_counter() - started,
                  "output_hashes": artifact_hashes(output)}
        atomic_json(output / "completion.json", result)
        print(finished, flush=True)
    except BaseException as error:
        failed = output / "failures"
        failed.mkdir(exist_ok=True)
        tag = f"{time.time_ns()}_{os.getpid()}"
        if local:
            write_csv(failed / f"{tag}_partial_calls.csv", local)
        record = {"failed_local": now(), "error": repr(error), "traceback": traceback.format_exc(),
                  "completed_sources": completed, "partial_calls_retained": len(local)}
        atomic_json(failed / f"{tag}.json", record)
        status("STOPPED_WITH_RETAINED_FAILURE", error=repr(error))
        raise


if __name__ == "__main__":
    main()
