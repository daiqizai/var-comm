#!/usr/bin/env python3
"""One matched pair; calibration-driven stopping with an immutable upper budget."""

import argparse
import csv
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
from var_comm.base_conditioning import BaseConditionedLink, restore_parent, select_checkpoints, stopping_decision, summarize_regions
from var_comm.hybrid_training import CachedPopulation, DigitalRenderer, evaluate, make_batch, model_forward, training_draw
from var_comm.hybrid_correction import ResidualLink
from var_comm.next_scale_prior import load_models
from var_comm.quality import load_quality_models
from var_comm.study import sha256, snapshot, verify_snapshot, write_csv, write_json


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def initial_reference_audit(rows):
    path = ROOT / "outputs/HYBRID-SOURCE-CORRECTION-20260915/training_001/calibration/step_0010000_full/per_frame.csv"
    with path.open() as handle:
        references = {(row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in csv.DictReader(handle) if row["arm"] == "add"}
    maximum = {"psnr_db": 0., "lpips": 0.}
    for row in rows:
        reference = references[row["image_id"], float(row["snr_db"]), int(row["seed"])]
        for metric in maximum:
            maximum[metric] = max(maximum[metric], abs(float(row[metric]) - float(reference[metric])))
    if maximum["psnr_db"] > 1e-4 or maximum["lpips"] > 1e-5:
        raise RuntimeError(f"zero-injection initial function differs from the audited parent: {maximum}")
    return {"status": "PASS", "reference_sha256": sha256(path), "maximum_metric_errors": maximum, "rows": len(rows)}


def run(arguments):
    output = arguments.output.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("new artifacts must stay in VAR_COMM")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "completion.json").exists():
        raise RuntimeError("completed pair must not be restarted")
    config_path = ROOT / "configs/hybrid_base_conditioning.json"
    config = json.loads(config_path.read_text())
    settings = config["training"]
    for path, expected in ((ROOT / config["initializer"], config["initializer_sha256"]),
                           (ROOT / config["parent_config"], config["parent_config_sha256"])):
        if sha256(path) != expected:
            raise RuntimeError("frozen initializer or historical recipe changed")
    cache = ROOT / config["cache"]
    sources = [Path(__file__), config_path, ROOT / "src/var_comm/base_conditioning.py",
               ROOT / "src/var_comm/hybrid_correction.py", ROOT / "src/var_comm/hybrid_training.py",
               ROOT / "src/var_comm/progressive.py", ROOT / "src/var_comm/scale_channel.py",
               ROOT / "src/var_comm/token_trellis.cpp", ROOT / "src/var_comm/quality.py",
               ROOT / "src/var_comm/next_scale_prior.py", ROOT / "reports/hybrid_base_conditioning_protocol_20260915.md"]
    bindings = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    metadata = {"bindings": bindings, "config_sha256": sha256(config_path), "mode": arguments.mode,
                "profile_updates": arguments.profile_updates if arguments.mode == "profile" else None,
                "initializer": config["initializer"], "initializer_sha256": config["initializer_sha256"],
                "initializer_role": config["initializer_role"], "cache_receipt_sha256": sha256(cache / "completion.json")}
    if (output / "metadata.json").exists():
        if json.loads((output / "metadata.json").read_text()) != metadata:
            raise RuntimeError("resume changed the registered pair")
        verify_snapshot(bindings)
    else:
        snapshot(output, sources)
        write_json(output / "metadata.json", metadata)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    train, calibration = CachedPopulation(cache, "train"), CachedPopulation(cache, "calibration")
    if (len(train), len(calibration)) != (20000, 1000):
        raise RuntimeError("original data populations must not be silently reduced")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    for specification, names in ((paths, ("vae_checkpoint", "var_checkpoint")), (quality_paths, ("alexnet_checkpoint", "dino_checkpoint"))):
        for name in names:
            if sha256(specification[name]) != specification[name + "_sha256"]:
                raise RuntimeError("frozen visual/metric weight changed")
    vae, var = load_models(paths, device)
    perceptual, unused_dino, _weights = load_quality_models(quality_paths, device)
    del unused_dino
    renderer = DigitalRenderer(vae, var, device, (train, calibration))
    parent = torch.load(ROOT / config["initializer"], map_location="cpu", weights_only=False)
    models, optimizers, parameter_names = {}, {}, {}
    for arm in config["arms"]:
        torch.manual_seed(settings["initialization_seed"])
        model = BaseConditionedLink(config["model"], arm, config["mechanism"]["context_ablation_seed"]).to(device).train()
        optimizer, names = restore_parent(model, parent, config)
        models[arm], optimizers[arm], parameter_names[arm] = model, optimizer, names
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    if len(set(counts.values())) != 1 or counts["conditioned"] != 1648125:
        raise RuntimeError("pair parameter counts no longer match the registered small adapter")
    for name, value in models["unconditioned"].state_dict().items():
        torch.testing.assert_close(value, models["conditioned"].state_dict()[name], rtol=0, atol=0)
    checkpoint_directory = output / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    latest = checkpoint_directory / "latest.pt"
    first_step = 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["bindings"] != bindings:
            raise RuntimeError("checkpoint bindings changed")
        first_step = int(saved["new_step"])
        for arm in config["arms"]:
            models[arm].load_state_dict(saved["models"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        del saved
    start = time.perf_counter()
    full_summaries = {}
    calibration_root = output / "calibration"
    calibration_root.mkdir(exist_ok=True)
    for path in sorted(calibration_root.glob("step_*_full/completion.json")):
        step = path.parent.name.split("_")[1]
        if sha256(path.parent / "per_frame.csv") != json.loads(path.read_text())["per_frame_sha256"]:
            raise RuntimeError("completed calibration artifact changed")
        full_summaries[str(int(step))] = json.loads((path.parent / "summary.json").read_text())
    subset = np.linspace(0, 999, 100, dtype=int)
    profile_subset = np.asarray([0, 999])

    def status(stage, step, **extra):
        value = {"status": stage, "new_updates_per_arm": step,
                 "shared_parameter_updates": 10000 + step, "maximum_new_updates": settings["maximum_new_updates"],
                 "pid": os.getpid(), "session_seconds": time.perf_counter() - start,
                 "updated_at": datetime.now().astimezone().isoformat(), **extra}
        write_json(output / "status.json", value)
        return value

    def checkpoint(step):
        verify_snapshot(bindings)
        path = checkpoint_directory / f"step_{step:07d}.pt"
        if path.exists():
            return path
        value = {"new_step": step, "shared_parent_updates": 10000, "bindings": bindings,
                 "models": {arm: model.state_dict() for arm, model in models.items()},
                 "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
                 "parameter_names": parameter_names}
        save_checkpoint(path, value)
        temporary = latest.with_suffix(".link")
        if temporary.exists():
            temporary.unlink()
        temporary.symlink_to(path.name)
        temporary.replace(latest)
        return path

    def calibrate(step, full):
        directory = calibration_root / f"step_{step:07d}_{'full' if full else 'subset'}"
        if (directory / "completion.json").exists():
            return json.loads((directory / "summary.json").read_text())
        directory.mkdir(exist_ok=True)
        selected = np.arange(1000) if full else profile_subset if arguments.mode == "profile" else subset
        status("FULL_CALIBRATION" if full else "SUBSET_CALIBRATION", step, calibration_sources=len(selected))
        records = evaluate(models, calibration, selected, config, renderer, perceptual, device)
        summary = summarize_regions(records, config)
        write_csv(directory / "per_frame.csv", records)
        write_json(directory / "summary.json", summary)
        if step == 0:
            write_json(directory / "parent_replay_audit.json", initial_reference_audit(records))
        receipt = {"sources": len(selected), "rows": len(records), "DINO_used": False,
                   "per_frame_sha256": sha256(directory / "per_frame.csv"),
                   "checkpoint_sha256": sha256(checkpoint_directory / f"step_{step:07d}.pt")}
        write_json(directory / "completion.json", receipt)
        if full:
            full_summaries[str(step)] = summary
        return summary

    if first_step == 0:
        checkpoint(0)
        indices, snrs, noise = training_draw(1, 20000, config)
        batch = make_batch(train, indices, snrs, noise, renderer, device)
        reference = ResidualLink(config["model"], "add").to(device).eval()
        reference.load_state_dict(parent["models"]["add"])
        with torch.no_grad():
            expected = model_forward(reference, batch)
            errors = {}
            for arm, model in models.items():
                output_image, waveform, _fixed_gain = model_forward(model, batch)
                errors[arm] = {"image": float((output_image - expected[0]).abs().max()),
                               "waveform": float((waveform - expected[1]).abs().max())}
                if errors[arm]["image"] > 1e-6 or errors[arm]["waveform"] > 1e-6:
                    raise RuntimeError("actual initial function differs from the common parent")
        write_json(output / "initialization.json", {"parameter_counts": counts, "shared_state_identical": True,
                   "forward_max_errors": errors, "parent_Adam_restored_by_name": True,
                   "parent_is_only_diagnostic": True, "fresh_parameters_per_arm": 5488})
        del expected, batch, reference
        calibrate(0, arguments.mode != "profile")
    elif arguments.mode != "profile" and first_step in settings["full_calibration_steps"] and str(first_step) not in full_summaries:
        calibrate(first_step, True)
    del parent
    target = arguments.profile_updates if arguments.mode == "profile" else settings["maximum_new_updates"]
    stop = stopping_decision(full_summaries, first_step, config) if arguments.mode != "profile" and first_step >= 10000 else {"stop": False}
    final_step = first_step
    trace = (output / f"updates_from_{first_step:07d}_{time.time_ns()}.jsonl").open("a", buffering=1)
    gradient_checks = {}
    previous_time = time.perf_counter()
    if not stop["stop"]:
        for step in range(first_step + 1, target + 1):
            indices, snrs, noise = training_draw(step, 20000, config)
            batch = make_batch(train, indices, snrs, noise, renderer, device)
            values = {}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                image, waveform, _fixed_gain = model_forward(model, batch)
                mse = (image - batch["images"]).square().mean()
                lpips = perceptual(image * 2 - 1, batch["images"] * 2 - 1).mean()
                loss = mse + settings["lpips_weight"] * lpips
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss stops both paired arms")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
                if step <= first_step + 2:
                    norms = {name: sum(float(parameter.grad.square().sum()) for parameter in module.parameters() if parameter.grad is not None) ** .5
                             for name, module in (("encoder", model.encoder), ("decoder", model.decoder),
                                                  ("context_extractor", model.context_extractor), ("context_projections", model.context_projections))}
                    if min(norms["encoder"], norms["decoder"], norms["context_projections"]) <= 0:
                        raise RuntimeError("image gradient does not reach the new receiver and both communication ends")
                    if step == first_step + 2 and norms["context_extractor"] <= 0:
                        raise RuntimeError("shared context extractor remained dead after injection update")
                    gradient_checks[f"{arm}_step{step}"] = norms
                optimizer.step()
                energy_error = float((waveform.detach().square().sum(1) - 2220).abs().max())
                if energy_error > .01:
                    raise RuntimeError("analog power no longer matches the paid physical budget")
                values[arm] = {"loss": float(loss.detach()), "mse": float(mse.detach()), "lpips": float(lpips.detach()),
                               "gradient_norm": float(norm), "analog_energy_error": energy_error}
                del image, waveform, loss, mse, lpips
            now = time.perf_counter()
            event = {"new_step": step, "indices": indices.tolist(), "snrs_db": snrs.tolist(),
                     "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(), "arms": values,
                     "body_CRC_failures": sum(row["header_usable"] and not row["body_crc_accepted"] for row in batch["events"]),
                     "header_failures": sum(not row["header_usable"] for row in batch["events"]),
                     "iteration_seconds": now - previous_time, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
            trace.write(json.dumps(event) + "\n")
            previous_time = now
            del batch
            final_step = step
            state = status("PROFILE_UPDATES" if arguments.mode == "profile" else "PAIRED_TRAINING", step, last=event)
            if step % 25 == 0 or step == first_step + 1:
                print(json.dumps({name: value for name, value in state.items() if name != "last"}), flush=True)
            if step == first_step + 2:
                if any(parameter.grad is not None for parameter in var.parameters()) or any(parameter.grad is not None for parameter in vae.parameters()):
                    raise RuntimeError("frozen visual backbone unexpectedly received parameter gradients")
                write_json(output / "gradient_check.json", {"status": "PASS", "norms": gradient_checks,
                           "frozen_visual_parameters_have_no_gradients": True, "RGB_gain_trained": False})
            if step % settings["checkpoint_interval"] == 0 or step == target:
                checkpoint(step)
            if step % settings["calibration_interval"] == 0 or step == target:
                full = arguments.mode != "profile" and step in settings["full_calibration_steps"]
                calibrate(step, full)
                previous_time = time.perf_counter()
                if full:
                    stop = stopping_decision(full_summaries, step, config)
                    write_json(output / f"stop_review_{step:07d}.json", stop)
                    if stop["stop"]:
                        break
    trace.close()
    if arguments.mode != "profile":
        selected = select_checkpoints(full_summaries, config)
        write_json(output / "selection.json", {"selected": selected, "stopping": stop, "full_calibration_steps": sorted(map(int, full_summaries)),
                   "new_updates_opportunity_per_arm": final_step, "old_shared_updates": 10000,
                   "development_used": False, "holdout_used": False, "DINO_used": False})
    verify_snapshot(bindings)
    done = {"status": "PROFILE_COMPLETE" if arguments.mode == "profile" else "PAIR_TRAINING_CALIBRATION_COMPLETE",
            "new_updates_per_arm": final_step, "shared_parameter_updates": 10000 + final_step,
            "parameter_counts": counts, "stopping": stop, "session_seconds_including_calibration": time.perf_counter() - start,
            "GPU_hours_this_session": (time.perf_counter() - start) / 3600, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "mechanism_or_system_success_proven": False, "new_holdout_accessed": False,
            "finished_at": datetime.now().astimezone().isoformat()}
    write_json(output / "completion.json", done)
    write_json(output / "status.json", done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "train"), default="train")
    parser.add_argument("--profile-updates", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED_BOTH_ARMS_STOPPED", "pid": os.getpid(), "traceback": traceback.format_exc(),
                   "at": datetime.now().astimezone().isoformat()}
        write_json(args.output / f"failure_{time.time_ns()}.json", failure)
        write_json(args.output / "status.json", failure)
        raise
