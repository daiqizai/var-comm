from __future__ import annotations

import hashlib
import json
import os
import signal
import time

import numpy as np
import torch

from latent_enhancement.latent import original_rgb
from latent_enhancement.runtime import ResourceBusy, configure, digest, image_losses, model_paths, perceptual_model, require_available, save_torch, settings, verify_snapshot, write_json
from latent_enhancement.training import PairedOrder
from var_comm.next_scale_prior import load_models, state_sha256

from .common import NAMES, OUT_B, load_decoder, load_gate, scale_statistics, stage_b_sources, validate_gpu_qualification
from .data import MatchedPopulation, paired_stop, update_monitor
from .evaluate import calibrate
from .model import build_arms, latent_errors, render_received


def main():
    configure()
    load_gate()
    output = OUT_B / "training"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        return 0
    for relative in ("qualification/completion.json", "rx_cache/completion.json"):
        if not (OUT_B / relative).exists():
            raise RuntimeError(f"required B prerequisite missing: {relative}")
    validate_gpu_qualification()
    registration_path = output / "registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
        verify_snapshot(registration["source_bindings"])
        if registration["rx_cache_completion_sha256"] != digest(OUT_B / "rx_cache/completion.json"):
            raise RuntimeError("actual-RX cache receipt changed")
        if registration["decoder_gate_sha256"] != digest(OUT_B / "decoder_gate.json"):
            raise RuntimeError("B training selected-Dc gate changed")
    else:
        registration = {"source_bindings": stage_b_sources(), "decoder_gate_sha256": digest(OUT_B / "decoder_gate.json"),
                        "rx_cache_completion_sha256": digest(OUT_B / "rx_cache/completion.json"),
                        "qualification_sha256": digest(OUT_B / "qualification/completion.json"), "created_at": time.time(),
                        "all_three_arms_share_every_update": True}
        write_json(registration_path, registration)
    require_available()
    config = settings()
    recipe = config["stage_B"]
    train = MatchedPopulation("train")
    calibration = MatchedPopulation("calibration")
    if len(train) != 20000 or len(calibration) != 1000:
        raise RuntimeError("source population differs from registration")
    device = torch.device("cuda:0")
    require_available()
    vae, var = load_models(model_paths(), device)
    del var
    torch.cuda.empty_cache()
    decoder = load_decoder(vae, device)
    scale = scale_statistics(device)
    arms = build_arms(train.shape, scale, recipe).to(device)
    perceptual = perceptual_model(device)
    original_sha, decoder_sha = state_sha256(vae), state_sha256(decoder)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
                  for name, model in arms.items()}
    order = PairedOrder(len(train), recipe["data_seed"])
    channel_rng = torch.Generator().manual_seed(recipe["channel_seed"])
    state = {"step": 0, "last_full_step": -1, "best_monitor": {name: {} for name in NAMES},
             "plateau_checks": {name: 0 for name in NAMES}, "selection": {name: {"step": 0, "utility": None} for name in NAMES},
             "update_seconds": 0.0, "calibration_seconds": 0.0,
             "data_trace_sha256": hashlib.sha256(b"paired-latent-enhancement-B-v1").hexdigest()}
    checkpoints = sorted(path for path in (output / "checkpoints").glob("update_*.pt") if path.with_suffix(".json").exists())
    if checkpoints:
        path = checkpoints[-1]
        receipt = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != receipt["sha256"]:
            raise RuntimeError("paired B checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["registration_sha256"] != digest(registration_path):
            raise RuntimeError("paired B checkpoint registration mismatch")
        arms.load_state_dict(checkpoint["arms"], strict=True)
        for name in NAMES:
            optimizers[name].load_state_dict(checkpoint["optimizers"][name])
        order.load_state_dict(checkpoint["order"])
        channel_rng.set_state(checkpoint["channel_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        state = checkpoint["training_state"]
        del checkpoint
        print(f"B resumed all three arms/Adam/order at update {state['step']}", flush=True)
    initial_path = output / "initialization.json"
    if not initial_path.exists():
        write_json(initial_path, {"arm_parameters": {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in arms.items()},
            "initial_arm_sha256": {name: state_sha256(model) for name, model in arms.items()},
            "frozen_Dc_state_sha256": decoder_sha, "frozen_original_vae_state_sha256": original_sha,
            "received_latent_shape": list(train.shape), "no_qualification_optimizer_reused": True,
            "budget": {name: {"N": 3060 + model.uses, "E": 2 * (3060 + model.uses)} for name, model in arms.items()}})
    stop_requested = [False]

    def request_stop(_signal, _frame):
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    attempt = f"{time.time_ns()}_{os.getpid()}"
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(reason):
        verify_snapshot(registration["source_bindings"])
        path = output / "checkpoints" / f"update_{state['step']:07d}_{time.time_ns()}.pt"
        save_torch(path, {"arms": arms.state_dict(), "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "order": order.state_dict(), "channel_rng": channel_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "training_state": state, "latent_shape": list(train.shape),
            "registration_sha256": digest(registration_path), "frozen_Dc_state_sha256": decoder_sha})
        receipt = {"step": state["step"], "reason": reason, "path": str(path), "sha256": digest(path), "attempt": attempt, "state": state}
        write_json(path.with_suffix(".json"), receipt)
        write_json(output / "latest.json", receipt)
        for name in NAMES:
            if state["selection"][name]["step"] == state["step"]:
                write_json(output / f"selected_{name}.json", {**state["selection"][name], "arm": name,
                    "checkpoint": str(path), "checkpoint_sha256": receipt["sha256"], "all_arms_trained_updates_at_checkpoint": state["step"]})

    def full_calibration():
        write_json(output / "status.json", {"status": "STAGE_B_FULL_CALIBRATION", "step": state["step"], "pid": os.getpid(), "timestamp": time.time()})
        result = calibrate(arms, decoder, lambda latent: original_rgb(vae, latent), perceptual, calibration, scale,
                           output, state["step"], True, device, recipe["microbatch_size"])
        if state["last_full_step"] < state["step"]:
            state["calibration_seconds"] += result["elapsed_seconds"]
            for name in NAMES:
                summary = result["summary"][name]
                state["best_monitor"][name], improved = update_monitor(summary, state["best_monitor"][name], config["stopping"])
                state["plateau_checks"][name] = 0 if improved else state["plateau_checks"][name] + 1
                utility = summary["all"]["utility"]
                if state["selection"][name]["utility"] is None or utility < state["selection"][name]["utility"]:
                    state["selection"][name] = {"step": state["step"], "utility": utility}
            state["last_full_step"] = state["step"]
        save_checkpoint("paired_full_calibration_complete")

    try:
        if state["step"] % recipe["full_calibration_interval"] == 0 and state["last_full_step"] < state["step"]:
            full_calibration()
        with (output / "training.jsonl").open("a", buffering=1) as log:
            while state["step"] < recipe["safety_maximum_updates"]:
                if stop_requested[0]:
                    save_checkpoint("explicit_interrupt_at_paired_update_boundary")
                    write_json(output / "status.json", {"status": "PAUSED_EXPLICIT_INTERRUPT", "step": state["step"], "pid": os.getpid()})
                    return 130
                require_available()
                if paired_stop(state, recipe, config["stopping"]["full_calibration_plateau_checks"]):
                    break
                arms.train()
                indices = order.next(recipe["logical_batch_size"])
                snr_indices = torch.randint(len(train.snrs), (len(indices),), generator=channel_rng)
                noise_indices = torch.randint(len(train.seeds), (len(indices),), generator=channel_rng)
                enhanced_seeds = torch.full_like(indices, recipe["channel_seed"] + state["step"])
                batch = train.batch(indices, snr_indices, noise_indices, enhanced_seeds, device)
                trace = hashlib.sha256(bytes.fromhex(state["data_trace_sha256"]) + indices.numpy().tobytes() + snr_indices.numpy().tobytes()
                                       + noise_indices.numpy().tobytes() + enhanced_seeds.numpy().tobytes()).hexdigest()
                torch.cuda.synchronize()
                started = time.perf_counter()
                rows = {}
                for name, model in arms.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    totals = np.zeros(3, dtype=np.float64)
                    for start in range(0, len(indices), recipe["microbatch_size"]):
                        selection = slice(start, min(start + recipe["microbatch_size"], len(indices)))
                        part = {key: value[selection] for key, value in batch.items()}
                        latent, _ = model.receive_training_sample(part["F"], part["Fb_TX"], part["Fb_RX"], part["snr_db"], part["rx_status"], part["standard_noise"])
                        predicted = render_received(decoder, latent, part["rx_status"])
                        mse, lpips = image_losses(predicted, part["target"], perceptual)
                        auxiliary = latent_errors(latent, part["F"], scale, part["rx_status"])
                        ((mse + recipe["LPIPS_weight"] * lpips + recipe["normalized_latent_weight"] * auxiliary).sum() / len(indices)).backward()
                        totals += torch.stack((mse.detach(), lpips.detach(), auxiliary.detach()), dim=1).sum(0).cpu().numpy()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    rows[name] = {"mse": float(totals[0] / len(indices)), "lpips": float(totals[1] / len(indices)),
                        "normalized_latent": float(totals[2] / len(indices)), "total_loss": float((totals[0] + 0.1 * totals[1] + 0.01 * totals[2]) / len(indices)),
                        "gradient_norm_before_clip": gradient_norm}
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                state["step"] += 1
                state["update_seconds"] += elapsed
                state["data_trace_sha256"] = trace
                row = {"step": state["step"], "attempt": attempt, "arms": rows, "paired_update_seconds": elapsed,
                       "header_failures": int((batch["rx_status"][:, 0] == 0).sum()), "data_trace_sha256": trace,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if state["step"] % 10 == 0 or state["step"] == 1:
                    write_json(output / "status.json", {"status": "STAGE_B_PAIRED_TRAINING", "step_per_arm": state["step"],
                        "pid": os.getpid(), "timestamp": time.time(), "source_presentations_per_arm": state["step"] * len(indices),
                        "mean_paired_update_seconds": state["update_seconds"] / state["step"], "latest_losses": rows,
                        "last_full_step": state["last_full_step"], "selection": state["selection"], "plateau_checks": state["plateau_checks"],
                        "minimum_updates_per_arm": recipe["minimum_updates"], "safety_cap_per_arm": recipe["safety_maximum_updates"],
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                    print(f"B paired update {state['step']} {elapsed:.3f}s " + json.dumps({name: values['total_loss'] for name, values in rows.items()}), flush=True)
                if state["step"] % recipe["full_calibration_interval"] == 0:
                    full_calibration()
                else:
                    if state["step"] % recipe["subset_calibration_interval"] == 0:
                        write_json(output / "status.json", {"status": "STAGE_B_SUBSET_CALIBRATION", "step": state["step"], "pid": os.getpid(), "timestamp": time.time()})
                        result = calibrate(arms, decoder, lambda latent: original_rgb(vae, latent), perceptual, calibration, scale,
                                           output, state["step"], False, device, recipe["microbatch_size"])
                        state["calibration_seconds"] += result["elapsed_seconds"]
                    if state["step"] % recipe["checkpoint_interval"] == 0:
                        save_checkpoint("paired_regular_checkpoint")
        if state["last_full_step"] != state["step"]:
            full_calibration()
        if state_sha256(vae) != original_sha or state_sha256(decoder) != decoder_sha:
            raise RuntimeError("frozen original VAE or selected Dc modified during training")
        verify_snapshot(registration["source_bindings"])
        completion = {"status": "STAGE_B_TRAINING_COMPLETE_DEVELOPMENT_AND_DIGITAL_COMPARISONS_PENDING", "updates_per_arm": state["step"],
            "selection": {name: json.loads((output / f"selected_{name}.json").read_text()) for name in NAMES},
            "stop_reason": "safety_cap_not_convergence_claim" if state["step"] >= recipe["safety_maximum_updates"] else "all_arms_three_full_plateau_checks",
            "training_gpu_hours": state["update_seconds"] / 3600, "calibration_gpu_hours": state["calibration_seconds"] / 3600,
            "original_D0_and_selected_Dc_unchanged": True, "data_trace_sha256": state["data_trace_sha256"],
            "full_experiment_complete": False, "timestamp": time.time()}
        write_json(output / "completion.json", completion)
        write_json(output / "status.json", completion)
        return 0
    except ResourceBusy as error:
        save_checkpoint("yield_to_other_authorized_GPU_task")
        write_json(output / "status.json", {"status": "YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK", "step": state["step"], "reason": str(error), "pid": os.getpid()})
        return 75


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceBusy as error:
        write_json(OUT_B / "training/status.json", {"status": "YIELDED_BEFORE_TRAINING", "reason": str(error), "pid": os.getpid()})
        raise SystemExit(75)
    except Exception as error:
        write_json(OUT_B / "training" / f"failure_{time.time_ns()}.json", {"type": type(error).__name__, "error": str(error)})
        raise
