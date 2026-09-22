#!/usr/bin/env python3
"""Two original full repeats on the already fixed first calibration observation."""

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
BRANCH = Path(__file__).resolve().parent
EXPERIMENT = BRANCH.parent
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(BRANCH), str(PROJECT / "src")]

import numpy as np
import torch

from profile_attention import DiffComAuthor, assert_gpu_available, digest, invariants, matched_step, receive, warm_step
from var_comm.study import sha256, write_csv, write_json


def run(arguments):
    assert_gpu_available()
    profile = arguments.stage / "attention_profile_001"
    if not (profile / "completion.json").exists():
        raise RuntimeError("finish the original paired profile first")
    output = arguments.stage / "repeatability_001"
    output.mkdir(exist_ok=False)
    case = profile / "case_00"
    recorded = json.loads((case / "input.json").read_text())
    with np.load(case / "actual_observation.npz", allow_pickle=False) as archive:
        observed = archive["received"].copy()
    if digest(observed) != recorded["received_sha256"]:
        raise RuntimeError("repeatability observation changed")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = EXPERIMENT / "checkpoints"
    model = DiffComAuthor(checkpoint / "ADJSCC_C=2.pth.tar", 2, checkpoint / "256x256_diffusion_uncond.pt")
    before = invariants(model)
    images, timings = [], []
    for repeat in range(2):
        gc.collect()
        torch.cuda.empty_cache()
        warm_step(model, observed, recorded["snr_db"], recorded["frame_seed"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        image, sampling = receive(model, observed, recorded["snr_db"], recorded["frame_seed"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        images.append(image)
        np.save(output / f"original_repeat_{repeat}.npy", image, allow_pickle=False)
        timings.append({"repeat": repeat, "RX_seconds": elapsed, "NFE": sampling["NFE"],
                        "received_sha256": digest(observed), "RGB_sha256": digest(image),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "sampler_or_precision_changed": False})
        write_csv(output / "timings.csv", timings)
        print(json.dumps(timings[-1]), flush=True)
    previous = np.load(case / "original_timed_RGB.npy", allow_pickle=False)
    comparisons = []
    for label, first, second in (("fresh_original_repeat0_vs_repeat1", images[0], images[1]),
                                  ("previous_original_vs_fresh0", previous, images[0]),
                                  ("previous_original_vs_fresh1", previous, images[1])):
        error = first.astype(np.float64) - second
        comparisons.append({"comparison": label, "max_abs_RGB_error": float(np.max(np.abs(error))),
                            "mean_abs_RGB_error": float(np.mean(np.abs(error))), "exact": bool(np.array_equal(first, second)),
                            "original_strict_2e4_tolerance_passed": bool(np.max(np.abs(error)) <= 2e-4)})
    write_csv(output / "original_repeat_comparisons.csv", comparisons)
    with np.load(case / "reference_state_000.npz", allow_pickle=False) as archive:
        snapshot = {"state": archive["state"].copy(), "cpu_rng": torch.from_numpy(archive["cpu_rng"].copy()),
                    "cuda_rng": torch.from_numpy(archive["cuda_rng"].copy())}
    for variant in ("original", "attention_direct"):
        with torch.autograd.profiler.profile(use_cuda=False, record_shapes=False) as trace:
            matched_step(model, observed, recorded["snr_db"], recorded["frame_seed"], 0, snapshot, variant)
        rows = [{"operation": item.key, "calls": item.count, "CPU_total_us_inclusive": item.cpu_time_total}
                for item in trace.key_averages() if any(word in item.key.lower() for word in ("backward", "convolution", "checkpoint"))]
        write_csv(output / f"{variant}_one_step_CPU_operations.csv", rows)
    if invariants(model) != before or any(parameter.grad is not None for network in (model.unet, model.operator.model) for parameter in network.parameters()):
        raise RuntimeError("repeatability check changed model parameters or flags")
    write_json(output / "completion.json", {"status": "BOUNDED_ORIGINAL_REPEATABILITY_DIAGNOSTIC_COMPLETE",
               "same_source_observation_seed_steps_precision": True, "new_calibration_sources": 0,
               "full_original_repeats": 2, "diagnostic_one_step_calls": 2, "no_tolerance_relaxed": True,
               "original_queue_not_replaced": True, "protocol_sha256": sha256(arguments.stage / "protocol.json"),
               "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    run(parser.parse_args())
