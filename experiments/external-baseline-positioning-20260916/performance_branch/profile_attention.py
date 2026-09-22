#!/usr/bin/env python3
"""Paired calibration-only performance/gradient validation, never a queue replacement."""

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
BRANCH = Path(__file__).resolve().parent
EXPERIMENT = BRANCH.parent
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(BRANCH), str(EXPERIMENT / "src"), str(PROJECT / "src"), str(PROJECT / "scripts")]

import numpy as np
import torch

from benchmark_frozen_systems import assert_gpu_available, gpu_state
from external_positioning.author_models import DiffComAuthor
from external_positioning.radio import decode_metadata, encode_metadata
from instrumentation import AttentionExecution, ExecutionRecorder
from var_comm.study import seeded_noise, sha256, write_csv, write_json


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def invariants(model):
    models = (model.unet, model.operator.model)
    return [(name, str(parameter.dtype), tuple(parameter.shape), parameter.requires_grad, parameter._version)
            for index, network in enumerate(models) for name, parameter in network.named_parameters()]


def receive(model, observation, snr, frame_seed):
    decoded = decode_metadata(observation[4096:], snr)
    decoded["frame_seed"] = frame_seed
    if not decoded["usable"]:
        return model.base_receive(observation[:4096], snr, frame_seed), {"NFE": 0, "fallback": True}
    image, sampling = model.hifi_receive(observation[:4096], snr, decoded, sampling_seed=23)
    return image, {**sampling, "fallback": False}


def warm_step(model, observation, snr, frame_seed):
    decoded = decode_metadata(observation[4096:], snr)
    decoded["frame_seed"] = frame_seed
    if not decoded["usable"]:
        return
    measurement = model.setup_receive(observation[:4096], snr, decoded)
    schedule = model.schedule_type(model.config, model.logger, model.device)
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    state = schedule.sqrt_alphas_cumprod[schedule.t_start] * (2 * measurement["x_mse"] - 1) + schedule.sqrt_1m_alphas_cumprod[schedule.t_start] * torch.randn_like(measurement["x_mse"])
    model.conditioner(model.config, 0, schedule, state, None, None, measurement, model.unet, model.diffusion,
                      model.operator, model.loss, last_timestep=False)
    torch.cuda.synchronize()


def matched_step(model, observation, snr, frame_seed, index, snapshot, variant):
    decoded = decode_metadata(observation[4096:], snr)
    decoded["frame_seed"] = frame_seed
    measurement = model.setup_receive(observation[:4096], snr, decoded)
    schedule = model.schedule_type(model.config, model.logger, model.device)
    state = torch.from_numpy(snapshot["state"].copy()).to(model.device)
    torch.set_rng_state(snapshot["cpu_rng"])
    torch.cuda.set_rng_state(snapshot["cuda_rng"], model.device)
    with AttentionExecution(model, variant), ExecutionRecorder(model, capture=False) as recorder:
        result = model.conditioner(model.config, index, schedule, state, None, None, measurement, model.unet, model.diffusion,
                                   model.operator, model.loss, last_timestep=False)
        torch.cuda.synchronize()
        prediction = result[0].detach().cpu().numpy().copy()
        next_state = result[2].detach().cpu().numpy().copy()
        gradient = recorder.last_outer_gradient
        details = recorder.summary()
    return {"gradient": gradient, "prediction": prediction, "next_state": next_state}, details


def run(arguments):
    assert_gpu_available()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((arguments.stage / "protocol.json").read_text())
    original = json.loads((EXPERIMENT / "configs/protocol.json").read_text())
    calibration = {int(row["image_index"]): row for row in json.loads((arguments.root / "inputs_001/calibration_inputs.json").read_text())}
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoints = EXPERIMENT / "checkpoints"
    for name in ("ADJSCC_C=2.pth.tar", "256x256_diffusion_uncond.pt"):
        if sha256(checkpoints / name) != original["checkpoints"][name]:
            raise RuntimeError("registered checkpoint changed")
    paths = [Path(__file__), BRANCH / "instrumentation.py", arguments.stage / "protocol.json",
             EXPERIMENT / "src/external_positioning/author_models.py", EXPERIMENT / "src/external_positioning/radio.py",
             EXPERIMENT / "vendor/diffcom_code/guided_diffusion/unet.py", EXPERIMENT / "vendor/diffcom_code/guided_diffusion/nn.py"]
    bindings = {str(path): sha256(path) for path in paths}
    started = time.perf_counter()
    model = DiffComAuthor(checkpoints / "ADJSCC_C=2.pth.tar", 2, checkpoints / "256x256_diffusion_uncond.pt")
    signature = invariants(model)
    if any(parameter.dtype != torch.float32 for network in (model.unet, model.operator.model) for parameter in network.parameters()):
        raise RuntimeError("profiling must preserve FP32")
    attention_count = len(AttentionExecution(model, "original").modules)
    write_json(output / "metadata.json", {"bindings": bindings, "torch": torch.__version__, "GPU": gpu_state(),
               "attention_blocks": attention_count, "attention_use_checkpoint_attributes": sorted({module.use_checkpoint for module in AttentionExecution(model, "original").modules}),
               "ResBlocks_checkpointed": sum(bool(module.use_checkpoint) for module in model.unet.modules() if module.__class__.__name__ == "ResBlock"),
               "FP32": True, "new_training": False, "profile_scope": "calibration_only_no_GT_quality_selection"})
    timings, comparisons, gradients, traces = [], [], [], []
    direct_available = True
    for ordinal, case in enumerate(protocol["calibration_profile_cases"]):
        assert_gpu_available()
        index, snr = int(case["image_index"]), float(case["snr_db"])
        item = calibration[index]
        if sha256(item["path"]) != item["file_sha256"]:
            raise RuntimeError("calibration pixels changed")
        pixels = np.load(item["path"], allow_pickle=False)
        frame_seed = 2026091600 + index * 32 + protocol["snrs_db"].index(snr) * 3
        signal, context = model.transmit(pixels, snr, frame_seed)
        packet = encode_metadata(context["power"])
        waveform = np.concatenate((signal, packet["symbols"]))
        noise = seeded_noise(item["image_id"], 4101, waveform.shape)
        observation = waveform + noise / np.sqrt(10 ** (snr / 10))
        directory = output / f"case_{ordinal:02d}"
        directory.mkdir()
        np.savez(directory / "actual_observation.npz", transmitted=waveform, received=observation, standard_noise=noise)
        write_json(directory / "input.json", {"image_index": index, "snr_db": snr, "noise_seed": 4101, "sampling_seed": 23,
                   "frame_seed": frame_seed, "source_pixels_sha256": item["source_pixels_sha256"],
                   "received_sha256": digest(observation), "total_uses": len(waveform), "energy": float(np.square(waveform).sum())})
        results, captured = {}, {}
        order = ("original", "attention_direct") if ordinal % 2 == 0 else ("attention_direct", "original")
        for instrumented in (False, True):
            for variant in order:
                if variant == "attention_direct" and not direct_available:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    with AttentionExecution(model, variant):
                        warm_step(model, observation, snr, frame_seed)
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        if instrumented:
                            with ExecutionRecorder(model) as recorder:
                                begin = time.perf_counter()
                                image, sampling = receive(model, observation, snr, frame_seed)
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - begin
                                detail = recorder.summary()
                                if variant == "original":
                                    captured = recorder.snapshots
                                traces.append({"case": ordinal, "variant": variant, "snr_db": snr, **detail})
                                write_json(directory / f"{variant}_instrumentation.json", detail)
                        else:
                            begin = time.perf_counter()
                            image, sampling = receive(model, observation, snr, frame_seed)
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - begin
                    if invariants(model) != signature or any(parameter.grad is not None for network in (model.unet, model.operator.model) for parameter in network.parameters()):
                        raise RuntimeError("profiling changed fixed model parameters/flags or accumulated parameter grads")
                    key = variant, instrumented
                    results[key] = image.copy()
                    np.save(directory / f"{variant}_{'instrumented' if instrumented else 'timed'}_RGB.npy", image, allow_pickle=False)
                    row = {"case": ordinal, "image_index": index, "snr_db": snr, "variant": variant,
                           "instrumented": instrumented, "full_RX_seconds": elapsed, "NFE_reverse_steps": int(sampling["NFE"]),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "input_observation_sha256": digest(observation), "output_sha256": digest(image), "fallback": sampling["fallback"],
                           "weights_and_autograd_flags_unchanged": True}
                    timings.append(row)
                    write_csv(output / "timings.csv", timings)
                    print(json.dumps(row), flush=True)
                except RuntimeError as failure:
                    if "out of memory" not in str(failure).lower() or variant != "attention_direct":
                        raise
                    direct_available = False
                    write_json(directory / "attention_direct_OOM.json", {"failure": str(failure), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "same_precision_and_steps": True, "no_automatic_fallback_or_memory_reconfiguration": True})
                    gc.collect()
                    torch.cuda.empty_cache()
        for variant in ("original", "attention_direct"):
            if (variant, False) in results and (variant, True) in results:
                error = float(np.max(np.abs(results[variant, False].astype(np.float64) - results[variant, True])))
                comparisons.append({"case": ordinal, "comparison": variant + "_instrumentation_vs_timed", "max_abs_RGB_error": error,
                                    "mean_abs_RGB_error": float(np.mean(np.abs(results[variant, False] - results[variant, True]))),
                                    "RGB_tolerance_passed": error <= protocol["full_RGB_max_abs_tolerance"]})
        if ("original", False) in results and ("attention_direct", False) in results:
            delta = results["attention_direct", False].astype(np.float64) - results["original", False]
            comparisons.append({"case": ordinal, "comparison": "attention_direct_vs_original_full_output", "max_abs_RGB_error": float(np.max(np.abs(delta))),
                                "mean_abs_RGB_error": float(np.mean(np.abs(delta))),
                                "RGB_tolerance_passed": bool(np.max(np.abs(delta)) <= protocol["full_RGB_max_abs_tolerance"])})
        if direct_available:
            for step, snapshot in sorted(captured.items()):
                if "gradient" not in snapshot:
                    raise RuntimeError("registered guided state has no input gradient")
                np.savez(directory / f"reference_state_{step:03d}.npz", state=snapshot["state"], gradient=snapshot["gradient"],
                         cpu_rng=snapshot["cpu_rng"].numpy(), cuda_rng=snapshot["cuda_rng"].numpy())
                reference, _reference_details = matched_step(model, observation, snr, frame_seed, step, snapshot, "original")
                direct, _direct_details = matched_step(model, observation, snr, frame_seed, step, snapshot, "attention_direct")
                original_gradient, direct_gradient = reference["gradient"].astype(np.float64), direct["gradient"].astype(np.float64)
                norm = float(np.linalg.norm(original_gradient))
                relative = float(np.linalg.norm(direct_gradient - original_gradient) / max(norm, 1e-30))
                passed = bool(np.isfinite(direct_gradient).all() and norm > 0 and np.linalg.norm(direct_gradient) > 0 and
                              np.allclose(direct_gradient, original_gradient, rtol=protocol["input_gradient_rtol"], atol=protocol["input_gradient_atol"]) and
                              relative <= protocol["input_gradient_relative_L2_tolerance"])
                gradients.append({"case": ordinal, "step": step, "snr_db": snr, "original_input_gradient_norm": norm,
                                  "direct_input_gradient_norm": float(np.linalg.norm(direct_gradient)), "relative_L2_error": relative,
                                  "max_abs_gradient_error": float(np.max(np.abs(direct_gradient - original_gradient))),
                                  "next_state_max_abs_error": float(np.max(np.abs(reference["next_state"] - direct["next_state"]))),
                                  "input_gradient_passed": passed, "same_state_RNG_observation": True})
        write_csv(output / "full_output_comparisons.csv", comparisons)
        if gradients:
            write_csv(output / "matched_input_gradients.csv", gradients)
        write_json(output / "status.json", {"status": "CALIBRATION_PROFILE", "completed_cases": ordinal + 1,
                   "new_training": False, "updated_at": datetime.now().astimezone().isoformat()})
    if invariants(model) != signature or any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("immutable profile inputs changed")
    speed = []
    for ordinal in range(len(protocol["calibration_profile_cases"])):
        chosen = {row["variant"]: row for row in timings if row["case"] == ordinal and not row["instrumented"]}
        if set(chosen) == {"original", "attention_direct"}:
            speed.append({"case": ordinal, "image_index": chosen["original"]["image_index"], "snr_db": chosen["original"]["snr_db"],
                          "original_RX_seconds": chosen["original"]["full_RX_seconds"], "direct_RX_seconds": chosen["attention_direct"]["full_RX_seconds"],
                          "speed_ratio_original_over_direct": chosen["original"]["full_RX_seconds"] / chosen["attention_direct"]["full_RX_seconds"],
                          "original_peak_allocated_bytes": chosen["original"]["peak_allocated_bytes"],
                          "direct_peak_allocated_bytes": chosen["attention_direct"]["peak_allocated_bytes"]})
    if speed:
        write_csv(output / "uninstrumented_speed_comparison.csv", speed)
    write_json(output / "completion.json", {"status": "ISOLATED_CALIBRATION_PROFILE_COMPLETE", "bindings": bindings,
               "direct_attention_completed": direct_available, "full_output_checks_passed": all(row["RGB_tolerance_passed"] for row in comparisons),
               "input_gradient_checks_passed": len(gradients) == 9 and all(row["input_gradient_passed"] for row in gradients),
               "weights_unchanged": True, "no_original_queue_or_vendor_overwrite": True, "no_automatic_deployment": True,
               "profile_seconds": time.perf_counter() - started, "finished_at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        arguments.output.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output / "failure.json", {"traceback": traceback.format_exc(), "no_original_inputs_modified": True})
        raise
