#!/usr/bin/env python3
"""Fixed author checkpoints on the frozen development population; no training."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src")]

import numpy as np
import torch

from external_positioning.author_models import DiffComAuthor, SwinAuthor
from external_positioning.radio import decode_metadata, encode_metadata
from var_comm.study import seeded_noise, sha256, write_csv, write_json


def tensor_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def assert_gpu_available(allow_self=False):
    response = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    active = {int(line.strip()) for line in response.splitlines() if line.strip()}
    if allow_self:
        active.discard(os.getpid())
    if active:
        raise RuntimeError(f"OTHER_GPU_PROCESS_PRESENT_PAUSE_NOT_PERFORMANCE_FAILURE:{sorted(active)}")


def same_context(first, second, family):
    if np.float32(first["power"]).tobytes() != np.float32(second["power"]).tobytes():
        return False
    return family != "swin" or first["indices"] == second["indices"]


def save_frame(directory, images, rows, frame_key):
    directory.mkdir(parents=True, exist_ok=False)
    stack = []
    hashes = {}
    for image in images:
        identity = tensor_digest(image)
        if identity not in hashes:
            hashes[identity] = len(stack)
            stack.append(image)
    archive = directory / "reconstructions.npz"
    np.savez(archive, images=np.stack(stack))
    for row, image in zip(rows, images):
        identity = tensor_digest(image)
        row.update(image_archive=str(archive), image_slot=hashes[identity], image_sha256=identity)
    write_json(directory / "frame.json", {"frame_key": frame_key, "rows": rows, "archive_sha256": sha256(archive)})


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "completion.json").exists():
        raise RuntimeError("completed author study must not be rerun")
    protocol_path = EXPERIMENT / "configs/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = arguments.inputs / ("calibration_inputs.json" if arguments.calibration else "development_inputs.json")
    population = json.loads(source.read_text())
    if arguments.limit:
        population = population[:arguments.limit]
    rates = [32, 64, 96] if arguments.family == "swin" else [2, 4, 6] if arguments.family == "adjscc" else [2]
    snrs = [1., 4., 7., 13., 19.]
    seeds = [4101] if arguments.calibration else [2001, 2002, 2003]
    bindings = {str(path): sha256(path) for path in (Path(__file__), protocol_path,
                EXPERIMENT / "src/external_positioning/author_models.py", EXPERIMENT / "src/external_positioning/radio.py", source)}
    paired = {}
    if arguments.paired_adjscc:
        import csv
        with arguments.paired_adjscc.open(newline="") as handle:
            paired = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"])): row for row in csv.DictReader(handle)
                      if row["method"] == "adjscc_c2" and row["protocol"] == "common_paid_information"}
        bindings[str(arguments.paired_adjscc)] = sha256(arguments.paired_adjscc)
    if arguments.family == "hifi" and not arguments.calibration and not paired:
        raise RuntimeError("formal HiFi evaluation must use a frozen same-ADJSCC observation reference")
    metadata = {"family": arguments.family, "rates": rates, "SNRs": snrs, "seeds": seeds,
                "sources": len(population), "calibration_only": arguments.calibration,
                "bindings": bindings, "quality_batch_size": 1, "timing_batch_size": 1,
                "metadata_after_data": True, "DINO_or_quality_selection": False}
    if (output / "metadata.json").exists() and json.loads((output / "metadata.json").read_text()) != metadata:
        raise RuntimeError("resume changed the registered execution inputs")
    write_json(output / "metadata.json", metadata)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    receipts = list(output.glob("frames/*/frame.json"))
    completed = {json.loads(path.read_text())["frame_key"] for path in receipts}
    if len(completed) != len(receipts):
        raise RuntimeError("duplicate completed frame rather than a clean resume")
    all_rows = []
    for path in receipts:
        saved = json.loads(path.read_text())
        if sha256(path.parent / "reconstructions.npz") != saved["archive_sha256"]:
            raise RuntimeError("committed author reconstruction changed")
        all_rows.extend(saved["rows"])
    started = time.perf_counter()
    total_frames = len(population) * len(rates) * len(snrs) * len(seeds)
    checkpoints = EXPERIMENT / "checkpoints"
    for rate in rates:
        checkpoint_name = "SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model" if arguments.family == "swin" else f"ADJSCC_C={rate}.pth.tar"
        if sha256(checkpoints / checkpoint_name) != protocol["checkpoints"][checkpoint_name]:
            raise RuntimeError("author communication checkpoint changed")
        if arguments.family == "swin":
            model = SwinAuthor(checkpoints / checkpoint_name)
        else:
            diffusion_path = checkpoints / "256x256_diffusion_uncond.pt" if arguments.family == "hifi" else None
            if diffusion_path and sha256(diffusion_path) != protocol["checkpoints"][diffusion_path.name]:
                raise RuntimeError("author diffusion checkpoint changed")
            model = DiffComAuthor(checkpoints / checkpoint_name, rate, diffusion_path)
        warmed = False
        for seed in seeds:
            for snr_index, snr in enumerate(snrs):
                for item in population:
                    key = f"rate{rate}_seed{seed}_snr{snr:g}_source{item['image_index']:04d}"
                    if key in completed:
                        continue
                    assert_gpu_available(allow_self=True)
                    pixels = np.load(item["path"], allow_pickle=False)
                    if sha256(item["path"]) != item["file_sha256"]:
                        raise RuntimeError("frozen source pixels changed")
                    frame_seed = 2026091600 + int(item["image_index"]) * 32 + snr_index * 3 + (seed - seeds[0])
                    if not warmed:
                        warm_signal, warm_context = model.transmit(pixels, snr, rate if arguments.family == "swin" else frame_seed)
                        if arguments.family == "swin":
                            model.receive(warm_signal, snr, warm_context)
                        elif arguments.family == "adjscc":
                            model.base_receive(warm_signal, snr, frame_seed)
                        else:
                            model.hifi_receive(warm_signal, snr, warm_context, sampling_seed=23, step_limit=2)
                        warmed = True
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    tx_begin = time.perf_counter()
                    signal, context = model.transmit(pixels, snr, rate if arguments.family == "swin" else frame_seed)
                    codec_tx_seconds = time.perf_counter() - tx_begin
                    if arguments.family == "swin":
                        packet = encode_metadata(context["power"], context["indices"], 320)
                    elif arguments.family == "hifi":
                        packet = encode_metadata(context["power"])
                    else:
                        packet = None
                    waveform = np.concatenate((signal, packet["symbols"])) if packet else signal
                    torch.cuda.synchronize()
                    tx_seconds = time.perf_counter() - tx_begin
                    energy = float(np.square(waveform).sum())
                    if abs(energy - 2 * len(waveform)) > .1:
                        raise RuntimeError("author waveform left the fixed-energy contract")
                    noise = seeded_noise(item["image_id"], seed, waveform.shape)
                    observed = waveform + noise / np.sqrt(10 ** (snr / 10))
                    data = observed[:len(signal)]
                    if paired:
                        original = paired[item["image_index"], snr, seed]
                        if (tensor_digest(signal) != original["data_transmitted_sha256"] or
                                tensor_digest(data) != original["data_observed_sha256"]):
                            raise RuntimeError("HiFi and its actual ADJSCC reference no longer share the same waveform and observation")
                    base = None
                    base_seconds = None
                    if arguments.family == "hifi":
                        torch.cuda.synchronize()
                        baseline_begin = time.perf_counter()
                        base = model.base_receive(data, snr, frame_seed)
                        torch.cuda.synchronize()
                        base_seconds = time.perf_counter() - baseline_begin
                    torch.cuda.synchronize()
                    rx_begin = time.perf_counter()
                    if packet:
                        decoded = decode_metadata(observed[len(signal):], snr, 320 if arguments.family == "swin" else None,
                                                  rate if arguments.family == "swin" else None)
                        decoded["frame_seed"] = frame_seed
                    else:
                        decoded = {"usable": True, "crc_accepted": None, "frame_seed": frame_seed}
                    fallback = ""
                    sampling = {"NFE": 0}
                    if not decoded["usable"]:
                        fallback = "invalid_metadata_gray" if arguments.family == "swin" else "invalid_metadata_same_observation_ADJSCC"
                        image = np.full((3, 256, 256), .5, dtype=np.float32) if arguments.family == "swin" else model.base_receive(data, snr, frame_seed)
                    elif arguments.family == "swin":
                        image = model.receive(data, snr, decoded)
                    elif arguments.family == "adjscc":
                        image = model.base_receive(data, snr, frame_seed)
                    else:
                        image, sampling = model.hifi_receive(data, snr, decoded, sampling_seed=23)
                    torch.cuda.synchronize()
                    rx_seconds = time.perf_counter() - rx_begin
                    peak_memory = torch.cuda.max_memory_allocated()
                    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                        raise RuntimeError("unexpected nonfinite or illegal RGB; preserve and investigate, not a weak-baseline score")
                    common = {"image_index": item["image_index"], "image_id": item["image_id"], "snr_db": snr,
                              "seed": seed, "rate": rate, "source_pixels_sha256": item["source_pixels_sha256"],
                              "checkpoint_sha256": protocol["checkpoints"][checkpoint_name], "data_complex_uses": len(signal),
                              "data_transmitted_sha256": tensor_digest(signal), "data_observed_sha256": tensor_digest(data),
                              "standard_data_noise_sha256": tensor_digest(noise[:len(signal)]),
                              "source_pwr_at_TX": context["power"], "frame_counter_seed": frame_seed,
                              "parameters_communication": model.parameters,
                              "parameters_diffusion": model.diffusion_parameters if arguments.family == "hifi" else 0,
                              "precision": "FP32_TF32_disabled", "batch_size": 1,
                              "TX_seconds": tx_seconds, "RX_seconds": rx_seconds,
                              "GPU_peak_allocated_bytes": peak_memory,
                              "performance_population": "calibration" if arguments.calibration else "development"}
                    method = f"swin_ra{rate}" if arguments.family == "swin" else f"adjscc_c{rate}" if arguments.family == "adjscc" else "hifi_diffcom_c2"
                    row = {**common, "method": method, "protocol": "common_paid_information",
                           "complex_uses": len(waveform), "actual_total_energy": energy,
                           "metadata_complex_uses": packet["complex_uses"] if packet else 0,
                           "metadata_crc_accepted": decoded["crc_accepted"], "metadata_usable": decoded["usable"],
                           "offline_metadata_exact": bool(np.array_equal(decoded["payload"], packet["payload"])) if packet else None,
                           "fallback": fallback, **sampling}
                    images, rows = [image], [row]
                    if arguments.family == "hifi":
                        rows.append({**row, "method": "adjscc_c2", "complex_uses": len(signal),
                                     "actual_total_energy": float(np.square(signal).sum()), "metadata_complex_uses": 0,
                                     "metadata_crc_accepted": None, "metadata_usable": True, "offline_metadata_exact": None,
                                     "fallback": "", "NFE": 0, "parameters_diffusion": 0,
                                     "TX_seconds": codec_tx_seconds, "RX_seconds": base_seconds,
                                     "GPU_peak_allocated_bytes": None,
                                     "timing_caveat": "same-observation attribution; prior resident; standalone timing separate"})
                        images.append(base)
                    if packet:
                        identical_inputs = decoded["usable"] and same_context(context, decoded, arguments.family)
                        if identical_inputs:
                            author_image = image
                        elif arguments.family == "swin":
                            author_image = model.receive(data, snr, context)
                        else:
                            author_image, _author_sampling = model.hifi_receive(data, snr, context, sampling_seed=23)
                        rows.append({**row, "protocol": "author_assumed_information", "complex_uses": len(signal),
                                     "actual_total_energy": float(np.square(signal).sum()), "metadata_complex_uses": 0,
                                     "metadata_crc_accepted": None, "metadata_usable": True, "offline_metadata_exact": None,
                                     "fallback": "", "TX_seconds": codec_tx_seconds, "RX_seconds": None,
                                     "GPU_peak_allocated_bytes": None,
                                     "unmetered_sender_metadata": "power_and_channel_mask" if arguments.family == "swin" else "normalization_power",
                                     "same_receive_inputs_reused": bool(identical_inputs),
                                     "NFE": sampling["NFE"] if identical_inputs else _author_sampling["NFE"] if arguments.family == "hifi" else 0})
                        images.append(author_image)
                    directory = output / "frames" / key
                    if directory.exists():
                        directory = output / "frames" / f"{key}_attempt{time.time_ns()}"
                    save_frame(directory, images, rows, key)
                    all_rows.extend(rows)
                    completed.add(key)
                    elapsed = time.perf_counter() - started
                    write_json(output / "status.json", {"status": "EVALUATING_FROZEN_AUTHORS", "family": arguments.family,
                               "frames_completed": len(completed), "target_frames": total_frames, "rows": len(all_rows),
                               "seconds_this_session": elapsed, "pid": os.getpid(),
                               "last": {"image_index": item["image_index"], "snr_db": snr, "seed": seed,
                                        "NFE": sampling["NFE"], "RX_seconds": rx_seconds},
                               "updated_at": datetime.now().astimezone().isoformat()})
                    if len(completed) % 10 == 0:
                        print(f"{arguments.family}: {len(completed)}/{total_frames} frames; last RX {rx_seconds:.3f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    if len(completed) != total_frames:
        raise RuntimeError("registered author population incomplete")
    for path, value in bindings.items():
        if sha256(path) != value:
            raise RuntimeError("author protocol or implementation changed during execution")
    fields = sorted({name for row in all_rows for name in row})
    write_csv(output / "per_frame.csv", [{name: row.get(name, "") for name in fields} for row in all_rows])
    receipt = {"status": "AUTHOR_TRANSMISSIONS_COMPLETE", "family": arguments.family, "frames": total_frames,
               "rows": len(all_rows), "sources": len(population), "new_training_or_holdout": False,
               "GPU_hours_this_session": (time.perf_counter() - started) / 3600,
               "per_frame_sha256": sha256(output / "per_frame.csv"), "finished_at": datetime.now().astimezone().isoformat()}
    write_json(output / "completion.json", receipt)
    write_json(output / "status.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["swin", "adjscc", "hifi"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--paired-adjscc", type=Path)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        arguments.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "PAUSED_OR_FAILED_REQUIRES_REVIEW_NO_SILENT_FRAME_REMOVAL", "traceback": traceback.format_exc(),
                   "at": datetime.now().astimezone().isoformat()}
        write_json(arguments.output / f"failure_{time.time_ns()}.json", failure)
        write_json(arguments.output / "status.json", failure)
        raise
