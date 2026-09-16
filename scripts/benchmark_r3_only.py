#!/usr/bin/env python3
"""Add only the frozen R3 CPU-endpoint timing; reuse all previously measured controls."""

import argparse
import csv
from datetime import datetime
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

from benchmark_frozen_systems import assert_gpu_available, digest, gpu_state
from var_comm.frozen_timing_adapters import install_frozen_paths, loaded_local_sources
from var_comm.online_timing import SOURCE_INDICES, SNRS, compare_arrays, single_waveform, timed_call
from var_comm.study import artifact_hashes, create_output, paired_interval, sha256, write_csv, write_json

NAME = "r3__full_grid_prediction_features"
QUALITY = ROOT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000"
REFERENCE = ROOT / "outputs/FROZEN-SYSTEM-ONLINE-TIMING-20260915/full_001"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows, controls):
    scopes = [(str(int(snr)), (snr,)) for snr in SNRS] + [("primary_1_4_7", SNRS[:3]), ("high_13_19", SNRS[3:]), ("all_5_snrs", SNRS)]
    summary, paired = [], []
    names = sorted({row["arm"] for row in controls})
    for scope, snrs in scopes:
        for metric in ("TX_seconds", "RX_seconds", "processing_sum_seconds"):
            values = np.array([np.mean([float(row[metric]) for row in rows if int(row["image_index"]) == index and float(row["snr_db"]) in snrs]) * 1000 for index in SOURCE_INDICES])
            selected = [float(row[metric]) * 1000 for row in rows if float(row["snr_db"]) in snrs]
            summary.append({"scope": scope, "arm": NAME, "metric": metric, "source_images": 32,
                            "mean_ms": float(values.mean()), "raw_call_p95_ms": float(np.percentile(selected, 95))})
            for control in names:
                reference = np.array([np.mean([float(row[metric]) for row in controls if row["arm"] == control and
                    int(row["image_index"]) == index and float(row["snr_db"]) in snrs]) * 1000 for index in SOURCE_INDICES])
                paired.append({"scope": scope, "method": NAME, "control": control, "metric": metric,
                    "source_images": 32, "units": "ms_R3_minus_control", "session_relation": "paired_sources_separate_sessions",
                    **paired_interval(values - reference, 9142026, 10000)})
    return summary, paired


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/R3_CPU_TIMING_001")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: 480 R3-only calls; no existing control rerun or training")
        return
    output = create_output(arguments.output_dir)
    started = time.perf_counter()
    rows = []
    try:
        install_frozen_paths()
        sys.path.insert(0, str(ROOT / "experiments/wetok-reencoding-vector-control-r3/src"))
        from vector_control.evaluation import load_evaluation, model_registry
        from wetok_comm.common import configure_torch
        from wetok_comm.native import FrozenWeTok, indices_to_features
        from wetok_comm.training import module_sha256
        configure_torch()
        assert_gpu_available()
        quality_receipt = json.loads((QUALITY / "completion.json").read_text())
        reference_receipt = json.loads((REFERENCE / "completion.json").read_text())
        if quality_receipt["status"] != "R3_QUALITY_COMPLETE" or reference_receipt["status"] != "FROZEN_SYSTEM_CPU_ENDPOINT_TIMING_COMPLETE":
            raise RuntimeError("R3 quality or frozen CPU-endpoint control timing is incomplete")
        for path, receipt in ((QUALITY / "per_frame.csv", quality_receipt), (REFERENCE / "per_call.csv", reference_receipt)):
            if sha256(path) != receipt["output_hashes"][path.name]:
                raise RuntimeError("frozen quality/timing reference changed")
        quality = {(int(row["image_index"]), float(row["snr_db"])): row for row in read_rows(QUALITY / "per_frame.csv")
                   if row["arm"] == NAME and int(row["seed"]) == 2001}
        controls = read_rows(REFERENCE / "per_call.csv")
        evaluation, config, r2, original, grid, reference, base, parent_record = load_evaluation()
        milestone_path = ROOT / "outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/milestones/step_0010000.json"
        if sha256(milestone_path) != quality_receipt["r3_milestone_sha256"]:
            raise RuntimeError("R3 selected milestone changed")
        milestone = json.loads(milestone_path.read_text())
        if milestone["selected"]["checkpoint_sha256"] != "667abc43639c09b0f0c465bc3bf669117ff727095ffdb7e563f6dad71c11ab3e":
            raise RuntimeError("R3 checkpoint was reselected")
        parent, models, choices = model_registry(config, reference, base, milestone, torch.device("cuda:0"))
        model = models[NAME].eval().requires_grad_(False)
        del parent
        native = FrozenWeTok("cuda:0", "both")
        before = {NAME: module_sha256(model), "native": module_sha256(native.codec)}
        if any(value != quality_receipt["frozen_models_before"][name] for name, value in before.items()):
            raise RuntimeError("R3 timing uses different frozen models")
        source_hashes = {str(path): sha256(path) for path in loaded_local_sources()}
        write_json(output / "metadata.json", {"started_local": datetime.now().astimezone().isoformat(),
            "checkpoint": choices[NAME], "frozen_models": before, "source_hashes": source_hashes,
            "quality_receipt_sha256": sha256(QUALITY / "completion.json"), "control_timing_receipt_sha256": sha256(REFERENCE / "completion.json"),
            "GPU": gpu_state(), "co_resident_models": [NAME, "native"], "control_times_reused_not_remeasured": True,
            "session_caveat": "same_sources_and_endpoints_but_not_same_interleaved_session", "new_training": False})
        original_root = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"
        population = json.loads((original_root / "populations.json").read_text())["target"]
        original_receipt = json.loads((original_root / "completion.json").read_text())
        native_codes = np.load(ROOT / "outputs/WETOK-NATIVE-CACHE-20260912/development_codes.npy", mmap_mode="r")

        def transmit(pixels, snr):
            source = torch.from_numpy(pixels[None]).to("cuda:0").float().div(255)
            encoded = native.codec.encoder(source.mul(2).sub(1))
            quantized, auxiliary, indices = native.quantizer(encoded)
            indices = indices.reshape(1, 16, 16, 4).to(torch.uint8)
            signal = model.transmit(indices_to_features(indices), torch.full((1,), float(snr), device="cuda:0"))
            return signal[0].cpu().numpy()

        def receive(observation, snr):
            result = model.receive(torch.from_numpy(observation[None]).to("cuda:0"), torch.full((1,), float(snr), device="cuda:0"))
            return native.decode(result["receiver_features"])[0].cpu().numpy()

        warmed = set()
        for completed, index in enumerate(SOURCE_INDICES, 1):
            assert_gpu_available()
            relative = f"images/{index:03d}"
            original_image = original_root / relative / "reconstructions.npz"
            if sha256(original_image) != original_receipt["output_hashes"][relative + "/reconstructions.npz"]:
                raise RuntimeError("source pixels changed")
            with np.load(original_image, allow_pickle=False) as archive:
                pixels = np.rint(archive["source"] * 255).astype(np.uint8)
            identifier = population[index]["image_id"]
            checked = native.encode(torch.from_numpy(pixels[None]).to("cuda:0").float().div(255))
            if not np.array_equal(checked[0].cpu().numpy(), native_codes[index, 0]):
                raise RuntimeError("native source differs from the frozen cache")
            waveform_path = QUALITY / relative / "waveforms.npz"
            if sha256(waveform_path) != quality_receipt["output_hashes"][relative + "/waveforms.npz"]:
                raise RuntimeError("R3 waveform archive changed")
            with np.load(waveform_path, allow_pickle=False) as waves:
                for snr in SNRS:
                    record = quality[index, snr]
                    signal = single_waveform(waves[f"{NAME}__snr{snr}"])
                    observation = single_waveform(waves[f"{NAME}__snr{snr}__seed2001"])
                    with np.load(record["image_archive"], allow_pickle=False) as images:
                        image = images["images"][int(record["image_ref"])].copy()
                    if (record["image_id"] != identifier or digest(signal) != record["transmitted_sha256"] or
                            digest(observation) != record["received_sha256"] or digest(image) != record["image_sha256"]):
                        raise RuntimeError("R3 quality observation, source or pixels changed")
                    if snr not in warmed:
                        for warmup in range(2):
                            transmit(pixels, snr)
                            receive(observation, snr)
                        warmed.add(snr)
                    for repeat in range(3):
                        assert_gpu_available()
                        actual_signal, tx_seconds = timed_call(lambda: transmit(pixels, snr), torch.cuda.synchronize)
                        actual_image, rx_seconds = timed_call(lambda: receive(observation, snr), torch.cuda.synchronize)
                        signal_error, same_signal = compare_arrays(actual_signal, signal, 1e-6, "R3 TX")
                        image_error, same_image = compare_arrays(actual_image, image, 1e-6, "R3 RGB")
                        energy = float(np.square(actual_signal.astype(np.float64)).sum())
                        if abs(energy - 6120) > .02 or digest(observation) != record["received_sha256"]:
                            raise RuntimeError("R3 power or received bytes changed")
                        rows.append({"image_index": index, "image_id": identifier, "snr_db": snr, "noise_seed": 2001,
                            "arm": NAME, "repeat": repeat, "TX_seconds": tx_seconds, "RX_seconds": rx_seconds,
                            "processing_sum_seconds": tx_seconds + rx_seconds, "total_complex_uses": 3060, "total_energy": energy,
                            "timing_scope": "CPU_to_CPU_contiguous_TX_and_RX", "signal_max_error": signal_error, "image_max_error": image_error,
                            "signal_exact": same_signal, "image_exact": same_image, "received_sha256": digest(observation),
                            "actual_image_sha256": digest(actual_image), "reference_image_sha256": digest(image)})
                    assert_gpu_available()
            write_csv(output / "per_call.csv", rows)
            write_json(output / "status.json", {"status": "R3_ONLY_TIMING", "pid": __import__("os").getpid(), "completed_sources": completed,
                                               "rows": len(rows), "updated_local": datetime.now().astimezone().isoformat()})
            print(f"R3 CPU timing {completed}/32 rows={len(rows)}", flush=True)
        after = {NAME: module_sha256(model), "native": module_sha256(native.codec)}
        if after != before or len(rows) != 480:
            raise RuntimeError("R3 timing incomplete or model changed")
        for path, expected in source_hashes.items():
            if sha256(path) != expected:
                raise RuntimeError("source changed during R3 timing")
        summary, paired = summarize(rows, controls)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired_historical_controls.csv", paired)
        write_json(output / "completion.json", {"status": "R3_ONLY_CPU_ENDPOINT_TIMING_COMPLETE", "completed_local": datetime.now().astimezone().isoformat(),
            "rows": 480, "frozen_before": before, "frozen_after": after, "source_hashes": source_hashes,
            "max_signal_error": max(row["signal_max_error"] for row in rows), "max_image_error": max(row["image_max_error"] for row in rows),
            "control_times_reused_not_remeasured": True, "new_training": False, "research_goal_complete": False,
            "process_seconds": time.perf_counter() - started, "output_hashes": artifact_hashes(output)})
    except BaseException as error:
        if rows:
            write_csv(output / "partial_calls.csv", rows)
        write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
