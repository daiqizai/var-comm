from __future__ import annotations

import json
import time

import numpy as np
import torch

from latent_enhancement.latent import append_observations, base_transmission, complete_latent, enhancement_noise, observed_status, original_rgb
from latent_enhancement.runtime import ResourceBusy, configure, digest, image_losses, model_paths, perceptual_model, require_available, settings, verify_snapshot, write_json
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_training_data import IMAGE_CACHE
from var_comm.progressive import complete_image, split_prefix

from .common import OUT_B, cache_shard, load_decoder, load_gate, scale_statistics, stage_b_sources
from .model import build_arms, latent_errors, render_received


def main():
    configure()
    gate = load_gate()
    directory = OUT_B / "qualification"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completion.json").exists():
        receipt = json.loads((directory / "completion.json").read_text())
        verify_snapshot(receipt["source_bindings"])
        if receipt["decoder_gate_sha256"] != digest(OUT_B / "decoder_gate.json"):
            raise RuntimeError("qualified decoder gate changed")
        return 0
    require_available()
    attempt = directory / f"attempt_{time.time_ns()}"
    attempt.mkdir()
    bindings = stage_b_sources()
    started = time.perf_counter()
    device = torch.device("cuda:0")
    vae, var = load_models(model_paths(), device)
    source, _, _ = cache_shard("train", 0)
    descriptor = source["source_descriptor"]
    image_path = IMAGE_CACHE / descriptor["path"]
    if digest(image_path) != descriptor["sha256"]:
        raise RuntimeError("qualification training image shard changed")
    images = torch.load(image_path, map_location="cpu", weights_only=True)
    target = images["targets_u8"][:4].to(device).float().div(255)
    base_tx, continuous = source["Fb_TX"][:4].to(device), source["F"][:4].to(device)
    snrs = torch.tensor([1., 4., 7., 19.], device=device)
    base_rx, statuses, waveforms, observations, originals, records = [], [], [], [], [], []
    for source_index, snr in enumerate(snrs.cpu().tolist()):
        require_available()
        prefix = split_prefix(source["base_tokens"][source_index].numpy(), 8)
        identifier = source["image_ids"][source_index]
        waveform, observation, reception = base_transmission(prefix, int(source["labels"][source_index]), identifier, 2026091721, snr)
        latent = complete_latent(vae, var, reception["prefix"], reception["label"], device)
        original = complete_image(vae, var, reception["prefix"], reception["label"], device)
        reproduced = np.full_like(original, 0.5) if latent is None else original_rgb(vae, latent)[0].cpu().numpy()
        np.testing.assert_array_equal(original, reproduced)
        base_rx.append(torch.zeros_like(base_tx[:1]) if latent is None else latent)
        statuses.append(observed_status(reception))
        waveforms.append(waveform)
        observations.append(observation)
        originals.append(original)
        records.append({"snr_db": snr, "header_accepted": reception["header"]["accepted"],
                        "body_accepted": bool(reception["events"] and reception["events"][0]["accepted"]),
                        "original_RGB_max_difference": float(np.abs(reproduced - original).max())})
    del var
    torch.cuda.empty_cache()
    base_rx = torch.cat(base_rx)
    statuses = torch.tensor(np.stack(statuses), device=device)
    scale = scale_statistics(device)
    decoder = load_decoder(vae, device)
    vae_sha, decoder_sha = state_sha256(vae), state_sha256(decoder)
    perceptual = perceptual_model(device)
    recipe = settings()["stage_B"]
    arms = build_arms(tuple(continuous.shape[1:]), scale, recipe).to(device)
    noise = torch.tensor(np.stack([enhancement_noise(identifier, 2026091721, 1024) for identifier in source["image_ids"][:4]]),
                         device=device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    results = {}
    saved_outputs = {"source": target.cpu().numpy(), "original_base": np.stack(originals)}
    for name, model in arms.items():
        require_available()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        encoder_image_gradient = 0.0
        update_times = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update_started = time.perf_counter()
            latent, waveform = model.receive_training_sample(continuous, base_tx, base_rx, snrs, statuses, noise)
            if step == 0:
                torch.testing.assert_close(latent, base_rx, rtol=0, atol=0)
            if waveform is not None:
                torch.testing.assert_close(waveform.square().sum((1, 2)), torch.full((len(waveform),), 2.0 * model.uses, device=device), rtol=1e-5, atol=1e-4)
                if step == 0:
                    for source_index in range(4):
                        received, transmitted = append_observations(waveforms[source_index], waveform[source_index].detach().cpu().numpy(),
                            source["image_ids"][source_index], 2026091721, float(snrs[source_index]))
                        np.testing.assert_array_equal(received[:3060], observations[source_index])
                        np.testing.assert_array_equal(transmitted[:3060], waveforms[source_index])
            reconstructed = render_received(decoder, latent, statuses)
            mse, lpips = image_losses(reconstructed, target, perceptual)
            image_loss = (mse + 0.1 * lpips).mean()
            if model.encoder is not None and step > 0:
                gradients = torch.autograd.grad(image_loss, tuple(model.encoder.parameters()), retain_graph=True, allow_unused=True)
                encoder_image_gradient = sum(float(gradient.detach().abs().sum()) for gradient in gradients if gradient is not None)
                if not encoder_image_gradient > 0:
                    raise RuntimeError("image loss does not reach the communication encoder through frozen Dc")
            auxiliary = latent_errors(latent, continuous, scale, statuses)
            (image_loss + 0.01 * auxiliary.mean()).backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update_times.append(time.perf_counter() - update_started)
            if not gradient_norm > 0:
                raise RuntimeError("communication model has no gradient")
        with torch.no_grad():
            hypothetical_rejection = statuses.clone()
            hypothetical_rejection[0, 0] = 0
            rejected = render_received(decoder, latent, hypothetical_rejection)
            torch.testing.assert_close(rejected[0], torch.full_like(rejected[0], 0.5), rtol=0, atol=0)
            assert float(latent_errors(latent, continuous, scale, hypothetical_rejection)[0]) == 0
        saved_outputs[name] = reconstructed.detach().cpu().numpy()
        results[name] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                         "four_source_update_seconds": update_times, "image_only_encoder_gradient_l1_after_zero_head_warmup": encoder_image_gradient,
                         "total_complex_uses": 3060 + model.uses, "total_energy": 2 * (3060 + model.uses)}
    if state_sha256(vae) != vae_sha or state_sha256(decoder) != decoder_sha or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError("frozen visual model or selected decoder changed")
    verify_snapshot(bindings)
    np.savez(attempt / "qualification_outputs.npz", **saved_outputs)
    receipt = {"status": "STAGE_B_REAL_PHY_AND_GRADIENT_QUALIFICATION_PASS", "source_bindings": bindings,
        "decoder_gate_sha256": digest(OUT_B / "decoder_gate.json"), "selected_stage_A_step": gate["selection"]["step"],
        "source_role": "training_only", "qualification_updates_discarded": True, "actual_base_checks": records,
        "arms": results, "frozen_D0_and_Dc_unchanged": True, "synthetic_header_rejection_rule_also_checked": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.perf_counter() - started, "output_sha256": digest(attempt / "qualification_outputs.npz")}
    write_json(attempt / "completion.json", receipt)
    write_json(directory / "completion.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceBusy as error:
        write_json(OUT_B / "qualification" / f"yield_{time.time_ns()}.json", {"reason": str(error)})
        raise SystemExit(75)
    except Exception as error:
        write_json(OUT_B / "qualification" / f"failure_{time.time_ns()}.json", {"type": type(error).__name__, "error": str(error)})
        raise
