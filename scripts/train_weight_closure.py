#!/usr/bin/env python3
"""Only four new weight arms; the two audited 0.01 histories remain read-only."""

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
from var_comm.base_conditioning import BaseConditionedLink, restore_parent, summarize_regions
from var_comm.hybrid_training import CachedPopulation, DigitalRenderer, evaluate, make_batch, model_forward, training_draw
from var_comm.next_scale_prior import load_models
from var_comm.quality import load_quality_models
from var_comm.study import sha256, snapshot, verify_snapshot, write_csv, write_json
from var_comm.weight_closure import load_configs, select_actual_points, specifications, weight_name


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "completion.json").exists():
        raise RuntimeError("completed fixed-budget weight closure must not be restarted")
    closure, base, runtime = load_configs()
    all_specs = specifications(closure)
    trained_specs = {name: item for name, item in all_specs.items() if not item["reused_training"]}
    qualification = arguments.qualification.resolve()
    if json.loads((qualification / "reuse_audit.json").read_text())["status"] != "PASS":
        raise RuntimeError("0.01 reuse must be qualified before any new training")
    old = ROOT / closure["base_run"] / "training_001"
    sources = [Path(__file__), ROOT / "src/var_comm/weight_closure.py", ROOT / "configs/hybrid_weight_closure.json",
               ROOT / "reports/hybrid_weight_closure_protocol_20260916.md", ROOT / "src/var_comm/base_conditioning.py",
               ROOT / "src/var_comm/hybrid_training.py", ROOT / "src/var_comm/hybrid_correction.py",
               ROOT / "src/var_comm/quality.py", ROOT / "src/var_comm/scale_channel.py", ROOT / "src/var_comm/progressive.py"]
    bindings = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    metadata = {"bindings": bindings, "mode": arguments.mode, "profile_updates": arguments.profile_updates if arguments.mode == "profile" else None,
                "reuse_audit_sha256": sha256(qualification / "reuse_audit.json"), "base_config_sha256": sha256(ROOT / closure["base_config"]),
                "trained_arms": trained_specs, "reused_arms": [name for name, item in all_specs.items() if item["reused_training"]],
                "new_budget_each": 20000, "loss_only_changed": True}
    if (output / "metadata.json").exists():
        if json.loads((output / "metadata.json").read_text()) != metadata:
            raise RuntimeError("resume changed frozen weight-study bindings")
    else:
        snapshot(output, sources)
        write_json(output / "metadata.json", metadata)
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    train = CachedPopulation(ROOT / base["cache"], "train")
    calibration = CachedPopulation(ROOT / base["cache"], "calibration")
    paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    quality_paths = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
    for specification, names in ((paths, ("vae_checkpoint", "var_checkpoint")), (quality_paths, ("alexnet_checkpoint", "dino_checkpoint"))):
        for name in names:
            if sha256(specification[name]) != specification[name + "_sha256"]:
                raise RuntimeError("frozen visual or metric weight changed")
    vae, var = load_models(paths, device)
    perceptual, dino, _linear = load_quality_models(quality_paths, device)
    renderer = DigitalRenderer(vae, var, device, (train, calibration))
    if sha256(ROOT / base["initializer"]) != base["initializer_sha256"]:
        raise RuntimeError("common warm-start checkpoint changed")
    parent = torch.load(ROOT / base["initializer"], map_location="cpu", weights_only=False)
    initial_old = torch.load(old / "checkpoints/step_0000000.pt", map_location="cpu", weights_only=False)
    models, optimizers = {}, {}
    for arm, item in trained_specs.items():
        torch.manual_seed(base["training"]["initialization_seed"])
        model = BaseConditionedLink(base["model"], item["variant"], base["mechanism"]["context_ablation_seed"]).to(device).train()
        optimizer, _names = restore_parent(model, parent, runtime)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor.cpu(), initial_old["models"][item["variant"]][name], rtol=0, atol=0)
        models[arm], optimizers[arm] = model, optimizer
    del parent, initial_old
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    first_step = 0
    if (checkpoints / "latest.pt").exists():
        saved = torch.load(checkpoints / "latest.pt", map_location=device, weights_only=False)
        if saved["bindings"] != bindings:
            raise RuntimeError("checkpoint implementation binding mismatch")
        first_step = saved["new_step"]
        for arm in models:
            models[arm].load_state_dict(saved["models"][arm])
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        del saved
    parameter_counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    if any(count != 1648125 for count in parameter_counts.values()):
        raise RuntimeError("weight sweep changed architecture size")
    write_json(output / "initialization.json", {"parameter_counts": parameter_counts, "same_as_original_0p01_initial_state": True,
               "parent_shared_Adam_copied": True, "0p01_is_not_retrained": True})
    started = time.perf_counter()
    full_summaries = {}
    cal_root = output / "calibration"
    cal_root.mkdir(exist_ok=True)
    for path in cal_root.glob("step_*_full/completion.json"):
        if sha256(path.parent / "per_frame.csv") != json.loads(path.read_text())["per_frame_sha256"]:
            raise RuntimeError("completed calibration was modified")
        full_summaries[str(int(path.parent.name.split("_")[1]))] = json.loads((path.parent / "summary.json").read_text())

    def status(stage, step, **extra):
        value = {"status": stage, "new_updates_per_trained_arm": step, "target_per_arm": 20000,
                 "trained_arms": list(models), "reused_weight": .01, "pid": os.getpid(),
                 "seconds_this_session": time.perf_counter() - started, "updated_at": datetime.now().astimezone().isoformat(), **extra}
        write_json(output / "status.json", value)
        return value

    def checkpoint(step):
        verify_snapshot(bindings)
        path = checkpoints / f"step_{step:07d}.pt"
        if not path.exists():
            save(path, {"new_step": step, "shared_parent_updates": 10000, "bindings": bindings,
                        "models": {name: model.state_dict() for name, model in models.items()},
                        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()}})
            link = checkpoints / "latest.link"
            if link.exists():
                link.unlink()
            link.symlink_to(path.name)
            link.replace(checkpoints / "latest.pt")
        return path

    def calibrate(step, full):
        directory = cal_root / f"step_{step:07d}_{'full' if full else 'subset'}"
        if (directory / "completion.json").exists():
            return
        directory.mkdir(exist_ok=True)
        selected = np.arange(1000) if full else np.asarray([0, 999]) if arguments.mode == "profile" else np.linspace(0, 999, 100, dtype=int)
        status("FULL_CALIBRATION_ALL_SIX" if full else "SUBSET_CALIBRATION", step, sources=len(selected), DINO_forward_only=bool(full))
        evaluated = dict(models)
        historical = {}
        if full:
            frozen = torch.load(old / "checkpoints" / f"step_{step:07d}.pt", map_location="cpu", weights_only=False)
            for variant in closure["variants"]:
                name = f"{weight_name(.01)}__{variant}"
                model = BaseConditionedLink(base["model"], variant, base["mechanism"]["context_ablation_seed"]).to(device).eval().requires_grad_(False)
                model.load_state_dict(frozen["models"][variant], strict=True)
                historical[name] = model
            evaluated.update(historical)
            del frozen
        records = evaluate(evaluated, calibration, selected, base, renderer, perceptual, device, dino if full else None)
        if arguments.mode != "profile":
            kind = "full" if full else "subset"
            archived = read_rows(old / "calibration" / f"step_{step:07d}_{kind}" / "per_frame.csv")
            previous = {(row["arm"], row["image_id"], float(row["snr_db"]), int(row["seed"])): row for row in archived}
            if full:
                maximum = {"psnr_db": 0., "lpips": 0.}
                for row in records:
                    if all_specs[row["arm"]]["reused_training"]:
                        item = previous[all_specs[row["arm"]]["variant"], row["image_id"], float(row["snr_db"]), int(row["seed"])]
                        for metric in maximum:
                            maximum[metric] = max(maximum[metric], abs(float(row[metric]) - float(item[metric])))
                if maximum["psnr_db"] > 1e-4 or maximum["lpips"] > 1e-5:
                    raise RuntimeError(f"0.01 replay no longer matches history: {maximum}")
                write_json(directory / "old_0p01_replay.json", {"status": "PASS", "max_errors": maximum, "DINO_newly_measured_not_used_for_selection": True})
            else:
                for row in archived:
                    row = dict(row)
                    row["arm"] = f"{weight_name(.01)}__{row['arm']}"
                    records.append(row)
        summary = summarize_regions(records, base)
        write_csv(directory / "per_frame.csv", records)
        write_json(directory / "summary.json", summary)
        write_json(directory / "completion.json", {"sources": len(selected), "rows": len(records),
                   "DINO_forward_only": bool(full), "DINO_used_for_training_selection_or_ratio": False,
                   "per_frame_sha256": sha256(directory / "per_frame.csv")})
        if full:
            full_summaries[str(step)] = summary
        del evaluated, historical

    if first_step == 0:
        checkpoint(0)
        calibrate(0, arguments.mode != "profile")
    elif first_step in closure["calibration_full_steps"] and str(first_step) not in full_summaries:
        calibrate(first_step, True)
    target = arguments.profile_updates if arguments.mode == "profile" else 20000
    trace = (output / f"updates_from_{first_step:07d}_{time.time_ns()}.jsonl").open("a", buffering=1)
    previous_time = time.perf_counter()
    for step in range(first_step + 1, target + 1):
        indices, snrs, noise = training_draw(step, 20000, base)
        batch = make_batch(train, indices, snrs, noise, renderer, device)
        scores = {}
        for arm, model in models.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, transmitted, _fixed_gain = model_forward(model, batch)
            mse = (reconstructed - batch["images"]).square().mean()
            lpips = perceptual(reconstructed * 2 - 1, batch["images"] * 2 - 1).mean()
            loss = mse + trained_specs[arm]["weight"] * lpips
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; stop all new weight arms without changing the recipe")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base["training"]["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            energy_error = float((transmitted.detach().square().sum(1) - 2220).abs().max())
            if energy_error > .01:
                raise RuntimeError("weight change altered the physical energy contract")
            scores[arm] = {"lambda": trained_specs[arm]["weight"], "loss": float(loss.detach()), "mse": float(mse.detach()),
                           "lpips": float(lpips.detach()), "gradient_norm": float(norm), "energy_error": energy_error}
            del reconstructed, transmitted, loss, mse, lpips
        now = time.perf_counter()
        event = {"new_step": step, "indices": indices.tolist(), "snrs_db": snrs.tolist(), "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                 "arms": scores, "body_crc_failures": sum(row["header_usable"] and not row["body_crc_accepted"] for row in batch["events"]),
                 "iteration_seconds": now - previous_time, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        trace.write(json.dumps(event) + "\n")
        del batch
        state = status("PROFILE_UPDATES" if arguments.mode == "profile" else "FOUR_NEW_WEIGHT_ARMS_TRAINING", step, last=event)
        if step % 25 == 0 or step == first_step + 1:
            print(json.dumps({name: value for name, value in state.items() if name != "last"}), flush=True)
        previous_time = now
        if step % 1000 == 0 or step == target:
            checkpoint(step)
            calibrate(step, arguments.mode != "profile" and step in closure["calibration_full_steps"])
            previous_time = time.perf_counter()
    trace.close()
    if arguments.mode != "profile":
        for step in closure["calibration_full_steps"]:
            if str(step) not in full_summaries:
                raise RuntimeError("formal closure cannot select from incomplete matched calibration history")
        selected = select_actual_points(full_summaries, closure)
        write_json(output / "selection.json", {"selected": selected, "selection_source": "primary_full_calibration_only",
                   "development_used": False, "DINO_used": False, "historical_report_rewritten": False,
                   "all_six_arms_received_20000_new_update_opportunities": True})
    verify_snapshot(bindings)
    receipt = {"status": "PROFILE_COMPLETE" if arguments.mode == "profile" else "WEIGHT_CLOSURE_TRAINING_AND_SELECTION_COMPLETE",
               "new_updates_each_new_arm": target, "new_arms": trained_specs, "reused_0p01_new_updates": 20000,
               "parameter_counts": parameter_counts, "GPU_hours_this_session": (time.perf_counter() - started) / 3600,
               "wall_seconds_this_session": time.perf_counter() - started, "new_holdout": False,
               "finished_at": datetime.now().astimezone().isoformat()}
    write_json(output / "completion.json", receipt)
    write_json(output / "status.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "profile"), default="train")
    parser.add_argument("--profile-updates", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        error = {"status": "FAILED_NEW_WEIGHT_ARMS_STOPPED", "traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()}
        write_json(args.output / f"failure_{time.time_ns()}.json", error)
        write_json(args.output / "status.json", error)
        raise
