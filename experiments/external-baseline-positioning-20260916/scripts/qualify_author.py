#!/usr/bin/env python3
"""Small calibration-only strict-load, actual waveform and upstream equivalence checks."""

import argparse
from datetime import datetime
import hashlib
import json
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
from var_comm.study import seeded_noise, sha256, write_json


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU already occupied; do not overlap author qualification")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    records = json.loads((arguments.inputs / "calibration_inputs.json").read_text())[:arguments.sources]
    checkpoint_root = EXPERIMENT / "checkpoints"
    protocol = json.loads((EXPERIMENT / "configs/protocol.json").read_text())
    required = ["SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model"] if arguments.family == "swin" else ["ADJSCC_C=2.pth.tar"]
    if arguments.family == "hifi":
        required.append("256x256_diffusion_uncond.pt")
    for name in required:
        if sha256(checkpoint_root / name) != protocol["checkpoints"][name]:
            raise RuntimeError("uploaded checkpoint changed since integrity verification")
    write_json(output / "inputs.json", {"checkpoint_SHA256": {name: protocol["checkpoints"][name] for name in required},
               "protocol_sha256": sha256(EXPERIMENT / "configs/protocol.json"),
               "adapter_sha256": sha256(EXPERIMENT / "src/external_positioning/author_models.py"),
               "script_sha256": sha256(Path(__file__))})
    if arguments.family == "swin":
        model = SwinAuthor(checkpoint_root / "SwinJSCC_w_SAandRA_AWGN_HRimage_cbr_psnr_snr.model")
        rates = [32, 64, 96]
    else:
        model = DiffComAuthor(checkpoint_root / "ADJSCC_C=2.pth.tar", 2,
                              checkpoint_root / "256x256_diffusion_uncond.pt" if arguments.family == "hifi" else None)
        rates = [2]
    events = []
    for record in records:
        pixels = np.load(record["path"], allow_pickle=False)
        if sha256(record["path"]) != record["file_sha256"]:
            raise RuntimeError("registered calibration pixels changed")
        for rate in rates:
            snr = 10.
            seed = 4101
            frame_seed = 2026091600 + int(record["image_index"]) * 32 + 1
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            signal, metadata = model.transmit(pixels, snr, rate if arguments.family == "swin" else frame_seed)
            metadata_packet = encode_metadata(metadata["power"], metadata["indices"], 320) if arguments.family == "swin" else encode_metadata(metadata["power"])
            waveform = np.concatenate((signal, metadata_packet["symbols"]))
            torch.cuda.synchronize()
            tx_seconds = time.perf_counter() - started
            energy = float(np.square(waveform).sum())
            if abs(energy - 2 * len(waveform)) > .05:
                raise RuntimeError("physical normalization is not equivalent to Ecomplex=2")
            noise = seeded_noise(record["image_id"], seed, waveform.shape)
            observed = waveform + noise / np.sqrt(10 ** (snr / 10))
            data = observed[:len(signal)]
            decoded = decode_metadata(observed[len(signal):], snr, 320 if arguments.family == "swin" else None,
                                      rate if arguments.family == "swin" else None)
            if not decoded["usable"]:
                raise RuntimeError("qualification metadata is not usable")
            exact = np.array_equal(decoded["payload"], metadata_packet["payload"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            if arguments.family == "swin":
                image = model.receive(data, snr, decoded)
            else:
                image = model.base_receive(data, snr, frame_seed)
            torch.cuda.synchronize()
            rx_seconds = time.perf_counter() - started
            reference = model.native_reference(pixels, snr, rate if arguments.family == "swin" else frame_seed, noise[:len(signal)])
            maximum_error = float(np.max(np.abs(image - reference)))
            if exact and maximum_error > 2e-4:
                raise RuntimeError(f"adapted receive path does not match author path: {maximum_error}")
            event = {"family": arguments.family, "image_id": record["image_id"], "rate": rate,
                     "snr_db": snr, "seed": seed, "data_complex_uses": len(signal),
                     "metadata_complex_uses": metadata_packet["complex_uses"], "total_complex_uses": len(waveform),
                     "actual_total_energy": energy, "metadata_exact": bool(exact), "metadata_CRC": decoded["crc_accepted"],
                     "author_path_max_RGB_difference": maximum_error, "TX_seconds": tx_seconds,
                     "base_RX_seconds": rx_seconds, "communication_parameters": model.parameters}
            np.save(output / f"{arguments.family}_{record['image_index']}_{rate}_base.npy", image, allow_pickle=False)
            if arguments.family == "hifi":
                decoded["frame_seed"] = frame_seed
                torch.cuda.synchronize()
                started = time.perf_counter()
                image, sampling = model.hifi_receive(data, snr, decoded, sampling_seed=23, step_limit=arguments.profile_steps)
                torch.cuda.synchronize()
                event.update(**sampling, full_RX_seconds=time.perf_counter() - started,
                             diffusion_parameters=model.diffusion_parameters)
                np.save(output / f"hifi_{record['image_index']}_{rate}_output.npy", image, allow_pickle=False)
            event["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            events.append(event)
            write_json(output / "events.json", events)
            print(json.dumps(event), flush=True)
    write_json(output / "completion.json", {"status": "AUTHOR_CALIBRATION_FUNCTIONAL_CHECK_COMPLETE",
               "family": arguments.family, "sources": len(records), "events": len(events),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "engineering_only": arguments.profile_steps is not None,
               "new_training_or_development_or_holdout": False,
               "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["swin", "adjscc", "hifi"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=1)
    parser.add_argument("--profile-steps", type=int)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"status": "ENGINEERING_CHECK_FAILED_NOT_A_PERFORMANCE_RESULT",
                   "traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()})
        raise
