#!/usr/bin/env python3
"""One finite paired training run, including real base-link errors."""

import argparse
from datetime import datetime
import fcntl
import hashlib
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

from benchmark_frozen_systems import assert_gpu_available
from var_comm.hybrid_correction import ResidualLink, HybridBudget
from var_comm.hybrid_training import (
    CachedPopulation, DigitalRenderer, aggregate, choose_checkpoint, evaluate,
    make_batch, model_forward, reference_calibration, training_draw,
)
from var_comm.next_scale_prior import load_models
from var_comm.quality import load_quality_models
from var_comm.study import sha256, snapshot, verify_snapshot, write_csv, write_json


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def run(arguments):
    output = arguments.output.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("output must remain in VAR_COMM")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = ROOT / "configs/hybrid_source_correction.json"
    config = json.loads(config_path.read_text())
    settings = config["training"]
    if (output / "completion.json").exists():
        raise RuntimeError("completed bounded run must not be restarted")
    if not (arguments.cache / "completion.json").exists():
        raise RuntimeError("full deterministic source cache has not completed")
    sources = [config_path, Path(__file__), ROOT / "src/var_comm/hybrid_correction.py",
               ROOT / "src/var_comm/hybrid_training.py", ROOT / "src/var_comm/progressive.py",
               ROOT / "src/var_comm/scale_channel.py", ROOT / "src/var_comm/token_trellis.cpp",
               ROOT / "src/var_comm/quality.py", ROOT / "src/var_comm/next_scale_prior.py",
               ROOT / "reports/hybrid_source_correction_protocol_20260915.md"]
    bindings = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    metadata = {"bindings": bindings, "cache": str(arguments.cache.resolve()),
                "cache_receipt_sha256": sha256(arguments.cache / "completion.json"),
                "config_sha256": sha256(config_path), "mode": arguments.mode,
                "profile_steps": arguments.profile_steps if arguments.mode == "profile" else None,
                "budget": HybridBudget().ledger()}
    if (output / "metadata.json").exists():
        if json.loads((output / "metadata.json").read_text()) != metadata:
            raise RuntimeError("training resume changed a frozen binding")
        verify_snapshot(bindings)
    else:
        snapshot(output, sources)
        write_json(output / "metadata.json", metadata)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    train = CachedPopulation(arguments.cache, "train")
    calibration = CachedPopulation(arguments.cache, "calibration")
    if arguments.mode != "profile" and (len(train), len(calibration)) != (20000, 1000):
        raise RuntimeError("formal run must use all original training/calibration sources")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    for name in ("vae_checkpoint", "var_checkpoint"):
        if sha256(paths[name]) != paths[name + "_sha256"]:
            raise RuntimeError("frozen visual weight changed")
    for name in ("alexnet_checkpoint", "dino_checkpoint"):
        if sha256(quality_paths[name]) != quality_paths[name + "_sha256"]:
            raise RuntimeError("frozen quality weight changed")
    vae, var = load_models(paths, device)
    perceptual, unused_dino, _linear_weights = load_quality_models(quality_paths, device)
    del unused_dino
    renderer = DigitalRenderer(vae, var, device, (train, calibration))
    models, optimizers = {}, {}
    for arm in config["arms"]:
        torch.manual_seed(settings["initialization_seed"])
        models[arm] = ResidualLink(config["model"], arm).to(device).train()
        optimizers[arm] = torch.optim.AdamW(models[arm].parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    checkpoint_directory = output / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    first_step = 0
    latest = checkpoint_directory / "latest.pt"
    total_wall = 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["bindings"] != bindings:
            raise RuntimeError("checkpoint code bindings differ")
        first_step = int(checkpoint["step"])
        total_wall = checkpoint["training_wall_seconds"]
        for arm in config["arms"]:
            models[arm].load_state_dict(checkpoint["models"][arm], strict=True)
            optimizers[arm].load_state_dict(checkpoint["optimizers"][arm])
        del checkpoint
    else:
        common = models["add"].state_dict()
        for arm in config["arms"][1:]:
            for name, value in common.items():
                torch.testing.assert_close(value, models[arm].state_dict()[name], rtol=0, atol=0)
        atomic_checkpoint(checkpoint_directory / "initial.pt", {"models": {arm: model.state_dict() for arm, model in models.items()},
                          "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}, "bindings": bindings})
    parameter_counts = {arm: sum(value.numel() for value in model.parameters()) for arm, model in models.items()}
    write_json(output / "initialization.json", {"parameter_counts": parameter_counts, "common_ED_identical": True,
                                               "gain_initial_value": 1, "planned_formal_presentations_per_arm": 160000})
    target = arguments.profile_steps if arguments.mode == "profile" else settings["updates_per_arm"]
    start = time.perf_counter()
    last_time = start
    trace = (output / f"updates_from_{first_step:06d}_{time.time_ns()}.jsonl").open("a", buffering=1)
    subset = np.linspace(0, len(calibration) - 1, min(settings["calibration_subset_sources"], len(calibration)), dtype=int)
    full_summaries = {}
    gradient_checks = {}
    for step in range(first_step + 1, target + 1):
        indices, snrs, noise = training_draw(step, len(train), config)
        batch = make_batch(train, indices, snrs, noise, renderer, device)
        results = {}
        for arm, model in models.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            prediction, transmitted, gain = model_forward(model, batch)
            mse = (prediction - batch["images"]).square().mean()
            lpips = perceptual(prediction * 2 - 1, batch["images"] * 2 - 1).mean()
            loss = mse + settings["lpips_weight"] * lpips
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss in {arm}; all paired arms stopped")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            if step == first_step + 1:
                encoder_norm = sum(float(value.grad.square().sum()) for value in model.encoder.parameters() if value.grad is not None) ** 0.5
                decoder_norm = sum(float(value.grad.square().sum()) for value in model.decoder.parameters() if value.grad is not None) ** 0.5
                if encoder_norm <= 0 or decoder_norm <= 0:
                    raise RuntimeError("image loss does not reach both continuous communication ends")
                gradient_checks[arm] = {"encoder_gradient_norm_after_clip": encoder_norm,
                                        "decoder_gradient_norm_after_clip": decoder_norm}
            optimizer.step()
            energy = transmitted.detach().square().sum(1)
            if torch.max(torch.abs(energy - 2220)) > 0.01:
                raise RuntimeError("incorrect actual transmitted energy")
            results[arm] = {"loss": float(loss.detach()), "mse": float(mse.detach()), "lpips": float(lpips.detach()),
                            "gradient_norm_before_clip": float(norm), "gain_mean": float(gain.detach().mean()),
                            "analog_energy_max_error": float(torch.abs(energy - 2220).max())}
            del prediction, transmitted, gain, loss, mse, lpips
        now = time.perf_counter()
        event = {"step": step, "indices": indices.tolist(), "snrs_db": snrs.tolist(),
                 "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(), "arms": results,
                 "header_failures": sum(not row["header_usable"] for row in batch["events"]),
                 "body_crc_failures": sum(row["header_usable"] and not row["body_crc_accepted"] for row in batch["events"]),
                 "false_acceptances": sum(row["false_acceptance"] for row in batch["events"]),
                 "digital_event_sha256": hashlib.sha256(json.dumps(batch["events"], sort_keys=True).encode()).hexdigest(),
                 "iteration_seconds_including_actual_digital_RX": now - last_time,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        trace.write(json.dumps(event) + "\n")
        last_time = now
        del batch
        state = {"status": "PROFILE_TRAINING" if arguments.mode == "profile" else "PAIRED_TRAINING",
                 "step_per_arm": step, "target_per_arm": target, "pid": os.getpid(),
                 "session_elapsed_seconds": now - start, "updated_at": datetime.now().astimezone().isoformat(),
                 "last": event, "renderer_forward_calls": renderer.calls, "renderer_cache_hits": renderer.cache_hits}
        write_json(output / "status.json", state)
        if step % 25 == 0 or step == first_step + 1:
            print(json.dumps({key: value for key, value in state.items() if key != "last"}), flush=True)
        if step == first_step + 1:
            if any(value.grad is not None for value in var.parameters()) or any(value.grad is not None for value in vae.parameters()):
                raise RuntimeError("frozen visual backbone received parameter gradients")
            write_json(output / "gradient_check.json", {"arms": gradient_checks, "visual_parameters_frozen": True,
                                                       "real_digital_failures_in_first_batch": event["body_crc_failures"]})
        checkpoint_due = step % settings["checkpoint_interval"] == 0 or step == target
        if checkpoint_due:
            verify_snapshot(bindings)
            checkpoint = {"step": step, "bindings": bindings,
                          "training_wall_seconds": total_wall + time.perf_counter() - start,
                          "models": {arm: model.state_dict() for arm, model in models.items()},
                          "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}}
            path = checkpoint_directory / f"step_{step:07d}.pt"
            atomic_checkpoint(path, checkpoint)
            temporary = latest.with_suffix(".link")
            if temporary.exists():
                temporary.unlink()
            temporary.symlink_to(path.name)
            temporary.replace(latest)
            del checkpoint
        calibration_due = step % settings["calibration_interval"] == 0 or (arguments.mode == "profile" and step == target)
        if calibration_due:
            full = step in settings["full_calibration_steps"] and arguments.mode != "profile"
            selected = np.arange(len(calibration)) if full else subset
            calibration_directory = output / "calibration" / f"step_{step:07d}_{'full' if full else 'subset'}"
            calibration_directory.mkdir(parents=True, exist_ok=True)
            write_json(output / "status.json", {**state, "status": "FULL_CALIBRATION" if full else "SUBSET_CALIBRATION"})
            rows = evaluate(models, calibration, selected, config, renderer, perceptual, device)
            summary = aggregate(rows, config["primary_snrs_db"])
            write_csv(calibration_directory / "per_frame.csv", rows)
            write_json(calibration_directory / "summary.json", summary)
            write_json(calibration_directory / "completion.json", {"frames": len(rows), "sources": len(selected),
                       "per_frame_sha256": sha256(calibration_directory / "per_frame.csv"), "DINO_used": False})
            if full:
                full_summaries[str(step)] = summary
            last_time = time.perf_counter()
    trace.close()
    if arguments.mode == "train":
        for step in settings["full_calibration_steps"]:
            directory = output / "calibration" / f"step_{step:07d}_full"
            if not (directory / "completion.json").exists():
                checkpoint = torch.load(checkpoint_directory / f"step_{step:07d}.pt", map_location=device, weights_only=False)
                for arm, model in models.items():
                    model.load_state_dict(checkpoint["models"][arm])
                del checkpoint
                directory.mkdir(parents=True, exist_ok=True)
                rows = evaluate(models, calibration, np.arange(len(calibration)), config, renderer, perceptual, device)
                write_csv(directory / "per_frame.csv", rows)
                write_json(directory / "summary.json", aggregate(rows, config["primary_snrs_db"]))
                write_json(directory / "completion.json", {"frames": len(rows), "sources": len(calibration),
                           "per_frame_sha256": sha256(directory / "per_frame.csv"), "DINO_used": False})
            full_summaries[str(step)] = json.loads((directory / "summary.json").read_text())
        references = reference_calibration(calibration.ids, config)
        selected = choose_checkpoint(full_summaries, references, config)
        write_json(output / "selection.json", {"selected": selected, "references": references,
                   "config_sha256": sha256(config_path), "development_used": False, "holdout_used": False})
    verify_snapshot(bindings)
    finished = {"status": "PROFILE_COMPLETE" if arguments.mode == "profile" else "TRAINING_AND_CALIBRATION_COMPLETE",
                "steps_per_arm": target, "arms": config["arms"], "parameter_counts": parameter_counts,
                "wall_seconds_including_calibration_this_session": time.perf_counter() - start,
                "GPU_hours_this_session": (time.perf_counter() - start) / 3600,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "quality_gain_proven": False, "independent_holdout_used": False}
    write_json(output / "completion.json", finished)
    write_json(output / "status.json", finished)
    print(json.dumps(finished), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "train"), default="train")
    parser.add_argument("--profile-steps", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED_ALL_ARMS_STOPPED", "pid": os.getpid(), "traceback": traceback.format_exc(),
                   "at": datetime.now().astimezone().isoformat()}
        write_json(args.output / f"failure_{time.time_ns()}.json", failure)
        write_json(args.output / "status.json", failure)
        raise
