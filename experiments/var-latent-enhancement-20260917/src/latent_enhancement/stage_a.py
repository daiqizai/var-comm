from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from var_comm.next_scale_prior import load_models, state_sha256

from .latent import ContinuousDecoder, original_rgb
from .runtime import (
    CONFIG, EXPERIMENT, OUTPUT, ResourceBusy, configure, digest, image_losses, model_paths,
    perceptual_model, require_available, save_torch, settings, snapshot, verify_snapshot,
    wait_for_available, write_json,
)
from .training import (
    LatentPopulation, PairedOrder, calibration_summary, draw_mixture, mixed_input, monitor, paired_interval,
)


def calibrate(decoder, vae, perceptual, population, alpha, indices, directory, step, full, device):
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{'full' if full else 'subset'}_{step:07d}"
    result_path = directory / (stem + ".json")
    arrays_path = directory / (stem + ".npz")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if digest(arrays_path) != result["arrays_sha256"]:
            raise RuntimeError("calibration arrays changed")
        return result
    decoder.eval()
    branches = ("F", "Fq", "interpolation", "Fb_TX")
    records = {name: [] for name in branches}
    if step == 0 and full:
        records.update({"D0_F": [], "D0_Fq": []})
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            require_available()
            selected = indices[start:start + 4]
            batch = population.batch(selected, device)
            coefficient = alpha[selected].to(device).reshape(-1, 1, 1, 1)
            inputs = {"F": batch["F"], "Fq": batch["Fq"], "Fb_TX": batch["Fb_TX"],
                      "interpolation": batch["Fb_TX"] + coefficient * (batch["F"] - batch["Fb_TX"])}
            for branch in branches:
                mse, lpips = image_losses(decoder(inputs[branch]), batch["target"], perceptual)
                records[branch].append(torch.stack((mse, lpips), dim=1).cpu().numpy())
            if step == 0 and full:
                for name, branch in (("D0_F", "F"), ("D0_Fq", "Fq")):
                    mse, lpips = image_losses(original_rgb(vae, batch[branch]), batch["target"], perceptual)
                    records[name].append(torch.stack((mse, lpips), dim=1).cpu().numpy())
    torch.cuda.synchronize()
    arrays = {name: np.concatenate(rows) for name, rows in records.items()}
    temporary = arrays_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, indices=indices.numpy(), image_ids=np.asarray([population.identifiers[index] for index in indices]),
                 alpha=alpha[indices].numpy(), **arrays)
    temporary.replace(arrays_path)
    result = {"step": step, "role": "full_calibration" if full else "fixed_calibration_subset", "sources": len(indices),
              "summary": calibration_summary(arrays), "arrays_sha256": digest(arrays_path),
              "elapsed_seconds": time.perf_counter() - started, "DINO_used": False,
              "perfect_F_and_interpolation_are_not_communication_results": True}
    write_json(result_path, result)
    print(f"calibration {stem}: {json.dumps(result['summary'])}", flush=True)
    decoder.train()
    return result


def selected_usefulness(output, selection):
    selected_step = selection["step"]
    with np.load(output / "calibration/full_0000000.npz", allow_pickle=False) as archive:
        original = archive["D0_Fq"].astype(np.float64)
        original_continuous = archive["D0_F"].astype(np.float64)
        identifiers = archive["image_ids"].copy()
    with np.load(output / f"calibration/full_{selected_step:07d}.npz", allow_pickle=False) as archive:
        adapted = archive["F"].astype(np.float64)
        if not np.array_equal(identifiers, archive["image_ids"]):
            raise RuntimeError("calibration source pairing changed")
    psnr = paired_interval(-10 * np.log10(np.maximum(adapted[:, 0], 1e-12)) + 10 * np.log10(np.maximum(original[:, 0], 1e-12)))
    lpips = paired_interval(adapted[:, 1] - original[:, 1])
    utility = paired_interval(adapted[:, 0] + 0.1 * adapted[:, 1] - original[:, 0] - 0.1 * original[:, 1])
    qualifies = selected_step > 0 and utility["mean"] < 0 and (psnr["low95"] > 0 or lpips["high95"] < 0)
    return {"stage_B_eligible": bool(qualifies), "selection": selection, "calibration_only_not_holdout": True,
            "Dc_F_minus_D0_Fq": {"psnr_db": psnr, "lpips": lpips, "image_utility": utility},
            "reference_means": {name: {"psnr_db": float((-10 * np.log10(np.maximum(values[:, 0], 1e-12))).mean()),
                                       "lpips": float(values[:, 1].mean()), "mse": float(values[:, 0].mean())}
                                for name, values in (("D0_Fq", original), ("D0_F", original_continuous), ("Dc_F", adapted))},
            "interpretation": "continuous-information reference only; not a finite-bandwidth system result",
            "next_action": "prepare_and_qualify_stage_B_with_same_selected_Dc" if qualifies else "report_representation_issue_no_automatic_enhancement_training"}


def run(args):
    output, cache = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    if not (cache / "completion.json").exists():
        raise RuntimeError("exact train/calibration cache is not complete")
    if not (OUTPUT / "qualification_001/completion.json").exists():
        raise RuntimeError("real-model qualification is missing")
    if (output / "completion.json").exists():
        print("stage A is already complete; no extra updates are authorized by this launcher", flush=True)
        return 0
    wait_for_available(output / "status.json")
    config = settings()
    recipe = config["stage_A"]
    registration_path = output / "registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
        verify_snapshot(registration["source_snapshot"])
        if registration["cache_completion_sha256"] != digest(cache / "completion.json"):
            raise RuntimeError("training cache receipt changed")
    else:
        registration = {"config_sha256": digest(CONFIG), "source_snapshot": snapshot([EXPERIMENT / "docs/stage_A_execution.md"]),
                        "cache_completion_sha256": digest(cache / "completion.json"), "created_at": time.time(),
                        "subset_rule": "floor(arange(100)*1000/100)", "warm_start": "independent_official_D0_copy_fresh_AdamW",
                        "no_retained_qualification_update": True}
        write_json(registration_path, registration)
    train = LatentPopulation(cache, "train")
    calibration = LatentPopulation(cache, "calibration")
    if len(train) != 20000 or len(calibration) != 1000 or train.shape != calibration.shape:
        raise RuntimeError("registered training/calibration shape or count mismatch")
    alpha = torch.rand(len(calibration), generator=torch.Generator().manual_seed(recipe["calibration_alpha_seed"]))
    full_indices = torch.arange(len(calibration))
    subset_indices = torch.arange(100) * len(calibration) // 100
    device = torch.device("cuda:0")
    require_available()
    vae, var = load_models(model_paths(), device)
    del var
    torch.cuda.empty_cache()
    frozen_vae_sha = state_sha256(vae)
    decoder = ContinuousDecoder(vae).to(device)
    perceptual = perceptual_model(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    order = PairedOrder(len(train), recipe["data_seed"])
    mixture = torch.Generator().manual_seed(recipe["mixture_seed"])
    state = {"step": 0, "last_full_step": -1, "best_monitor": {}, "plateau_checks": 0,
             "selection": {"step": 0, "utility": None}, "update_seconds": 0.0, "calibration_seconds": 0.0,
             "data_trace_sha256": hashlib.sha256(b"latent-enhancement-stageA-v1").hexdigest()}
    checkpoint_files = sorted((output / "checkpoints").glob("update_*.pt"))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[-1]
        receipt = json.loads(checkpoint_path.with_suffix(".json").read_text())
        if digest(checkpoint_path) != receipt["sha256"]:
            raise RuntimeError("checkpoint changed")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["registration_sha256"] != digest(registration_path):
            raise RuntimeError("checkpoint registration mismatch")
        decoder.load_state_dict(saved["decoder"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        order.load_state_dict(saved["order"])
        mixture.set_state(saved["mixture_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state = saved["training_state"]
        del saved
        print(f"resume stageA from exact model/Adam/order at {state['step']}", flush=True)
    stop_requested = [False]

    def request_stop(_signal, _frame):
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    attempt = f"{time.time_ns()}_{os.getpid()}"
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(reason):
        verify_snapshot(registration["source_snapshot"])
        directory = output / "checkpoints"
        directory.mkdir(exist_ok=True)
        checkpoint_path = directory / f"update_{state['step']:07d}_{time.time_ns()}.pt"
        save_torch(checkpoint_path, {"decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
            "order": order.state_dict(), "mixture_rng": mixture.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "training_state": state, "latent_shape": list(train.shape),
            "registration_sha256": digest(registration_path), "frozen_vae_state_sha256": frozen_vae_sha})
        receipt = {"step": state["step"], "reason": reason, "sha256": digest(checkpoint_path),
                   "path": str(checkpoint_path), "attempt": attempt, "state": state}
        write_json(checkpoint_path.with_suffix(".json"), receipt)
        write_json(output / "latest.json", receipt)
        if state["step"] == state["selection"]["step"]:
            write_json(output / "selected.json", {**state["selection"], "checkpoint": str(checkpoint_path),
                                                     "checkpoint_sha256": receipt["sha256"]})
        return checkpoint_path

    def full_calibration():
        result = calibrate(decoder, vae, perceptual, calibration, alpha, full_indices,
                           output / "calibration", state["step"], True, device)
        if state["last_full_step"] < state["step"]:
            state["calibration_seconds"] += result["elapsed_seconds"]
            state["best_monitor"], improved = monitor(result["summary"], state["best_monitor"], config["stopping"])
            state["plateau_checks"] = 0 if improved else state["plateau_checks"] + 1
            state["last_full_step"] = state["step"]
            utility = result["summary"]["mixture_utility"]
            if state["selection"]["utility"] is None or utility < state["selection"]["utility"]:
                state["selection"] = {"step": state["step"], "utility": utility}
        save_checkpoint("full_calibration_complete")

    try:
        if state["step"] % recipe["full_calibration_interval"] == 0 and state["last_full_step"] < state["step"]:
            full_calibration()
        with (output / "training.jsonl").open("a", buffering=1) as log:
            while state["step"] < recipe["safety_maximum_updates"]:
                if stop_requested[0]:
                    save_checkpoint("explicit_interrupt_safe_boundary")
                    write_json(output / "status.json", {"status": "PAUSED_EXPLICIT_INTERRUPT", "step": state["step"], "pid": os.getpid()})
                    return 130
                require_available()
                if (state["step"] >= recipe["minimum_updates"] and
                        state["plateau_checks"] >= config["stopping"]["full_calibration_plateau_checks"]):
                    break
                decoder.train()
                indices = order.next(recipe["logical_batch_size"])
                branches, coefficient = draw_mixture(len(indices), mixture)
                trace = hashlib.sha256(bytes.fromhex(state["data_trace_sha256"]) + indices.numpy().tobytes()
                                       + branches.numpy().tobytes() + coefficient.numpy().tobytes()).hexdigest()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                losses = np.zeros(2, dtype=np.float64)
                branch_sums = np.zeros((3, 3), dtype=np.float64)
                for start in range(0, len(indices), recipe["microbatch_size"]):
                    stop = min(start + recipe["microbatch_size"], len(indices))
                    batch = train.batch(indices[start:stop], device)
                    branch_ids = branches[start:stop].to(device)
                    latent = mixed_input(batch, branch_ids, coefficient[start:stop].to(device))
                    mse, lpips = image_losses(decoder(latent), batch["target"], perceptual)
                    ((mse + recipe["LPIPS_weight"] * lpips).sum() / len(indices)).backward()
                    values = torch.stack((mse.detach(), lpips.detach()), dim=1).cpu().numpy()
                    losses += values.sum(0)
                    for branch in range(3):
                        mask = branches[start:stop].numpy() == branch
                        branch_sums[branch, :2] += values[mask].sum(0)
                        branch_sums[branch, 2] += mask.sum()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                state["step"] += 1
                state["update_seconds"] += elapsed
                state["data_trace_sha256"] = trace
                row = {"step": state["step"], "attempt": attempt, "mse": float(losses[0] / len(indices)),
                    "lpips": float(losses[1] / len(indices)), "image_loss": float((losses[0] + 0.1 * losses[1]) / len(indices)),
                    "branches": {name: {"mse": float(branch_sums[branch, 0] / branch_sums[branch, 2]),
                                         "lpips": float(branch_sums[branch, 1] / branch_sums[branch, 2]),
                                         "count": int(branch_sums[branch, 2])}
                                 for branch, name in enumerate(("F", "Fq", "interpolation"))},
                    "gradient_norm_before_clip": gradient_norm, "update_seconds": elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "data_trace_sha256": trace}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if state["step"] % 10 == 0 or state["step"] == 1:
                    status = {"status": "STAGE_A_TRAINING", "step": state["step"], "pid": os.getpid(), "timestamp": time.time(),
                        "source_presentations": state["step"] * recipe["logical_batch_size"], "last_loss": row["image_loss"],
                        "mean_update_seconds": state["update_seconds"] / state["step"], "last_full_step": state["last_full_step"],
                        "selected": state["selection"], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "minimum_updates": recipe["minimum_updates"], "safety_cap_not_convergence": recipe["safety_maximum_updates"]}
                    write_json(output / "status.json", status)
                    print(f"stageA {state['step']} loss={row['image_loss']:.6f} {elapsed:.3f}s/update", flush=True)
                if state["step"] % recipe["full_calibration_interval"] == 0:
                    full_calibration()
                else:
                    if state["step"] % recipe["subset_calibration_interval"] == 0:
                        result = calibrate(decoder, vae, perceptual, calibration, alpha, subset_indices,
                                           output / "calibration", state["step"], False, device)
                        state["calibration_seconds"] += result["elapsed_seconds"]
                    if state["step"] % recipe["checkpoint_interval"] == 0:
                        save_checkpoint("regular_checkpoint")
        if state["last_full_step"] != state["step"]:
            full_calibration()
        if state_sha256(vae) != frozen_vae_sha:
            raise RuntimeError("original VAE was modified")
        verify_snapshot(registration["source_snapshot"])
        selection = json.loads((output / "selected.json").read_text())
        usefulness = selected_usefulness(output, selection)
        write_json(output / "continuous_reference_result.json", usefulness)
        completion = {"status": "STAGE_A_COMPLETE_NOT_FULL_EXPERIMENT", "updates": state["step"],
            "stop_reason": "safety_cap_not_convergence_claim" if state["step"] >= recipe["safety_maximum_updates"] else "registered_three_full_check_plateau",
            "selection": selection, "stage_B_eligible": usefulness["stage_B_eligible"], "stage_B_started": False,
            "training_gpu_hours": state["update_seconds"] / 3600, "calibration_gpu_hours": state["calibration_seconds"] / 3600,
            "frozen_vae_state_unchanged": True, "data_trace_sha256": state["data_trace_sha256"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "timestamp": time.time()}
        write_json(output / "completion.json", completion)
        write_json(output / "status.json", completion)
        print(json.dumps(completion), flush=True)
        return 0
    except ResourceBusy as error:
        save_checkpoint("resource_yield_safe_boundary")
        write_json(output / "status.json", {"status": "YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK", "step": state["step"], "reason": str(error), "pid": os.getpid()})
        return 75


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(OUTPUT / "cache_v1"))
    parser.add_argument("--output", default=str(OUTPUT / "stage_A_v1"))
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except ResourceBusy as error:
        write_json(Path(args.output) / "status.json", {"status": "YIELDED_BEFORE_TRAINING", "reason": str(error), "pid": os.getpid()})
        raise SystemExit(75)
    except Exception as error:
        write_json(Path(args.output) / f"failure_{time.time_ns()}.json", {"type": type(error).__name__, "message": str(error), "timestamp": time.time()})
        raise


if __name__ == "__main__":
    main()
