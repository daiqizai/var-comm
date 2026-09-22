from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_training_data import IMAGE_CACHE, MANIFEST_SHA
from var_comm.progressive import complete_image, split_prefix

from .latent import (
    ContinuousDecoder, append_observations, base_transmission, complete_latent,
    normalize_enhancement, original_rgb, quantized_latent,
)
from .runtime import (
    OUTPUT, configure, digest, image_losses, model_paths, perceptual_model,
    require_available, snapshot, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(OUTPUT / "qualification_001"))
    args = parser.parse_args()
    from pathlib import Path

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    configure()
    require_available()
    started = time.perf_counter()
    sources = snapshot()
    try:
        if digest(IMAGE_CACHE / "manifest.json") != MANIFEST_SHA:
            raise RuntimeError("source manifest changed")
        manifest = json.loads((IMAGE_CACHE / "manifest.json").read_text())
        population = next(entry for entry in manifest["populations"] if entry["name"] == "train")
        descriptor = population["shards"][0]
        path = IMAGE_CACHE / descriptor["path"]
        if digest(path) != descriptor["sha256"]:
            raise RuntimeError("source shard changed")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        device = torch.device("cuda:0")
        vae, var = load_models(model_paths(), device)
        vae_before = state_sha256(vae)
        pixels = shard["targets_u8"][:4].to(device).float().div(255)
        with torch.no_grad():
            continuous = vae.quant_conv(vae.encoder(pixels * 2 - 1))
            scales = vae.quantize.f_to_idxBl_or_fhat(continuous, to_fhat=False)
            quantized = quantized_latent(vae, scales)
            mapped = vae.quantize.f_to_idxBl_or_fhat(continuous, to_fhat=True)[-1]
        torch.testing.assert_close(quantized, mapped, rtol=0, atol=2e-6)
        source = [scale[0].cpu().numpy() for scale in scales]
        label = int(shard["labels"][0])
        image_id = shard["image_ids"][0]
        checks = []
        for snr in (1, 19):
            require_available()
            waveform, observation, received = base_transmission(source, label, image_id, 4101, snr)
            old = complete_image(vae, var, received["prefix"], received["label"], device)
            latent = complete_latent(vae, var, received["prefix"], received["label"], device)
            new = (np.full_like(old, 0.5) if latent is None else original_rgb(vae, latent)[0].cpu().numpy())
            np.testing.assert_array_equal(new, old)
            for uses in (512, 1024):
                added = normalize_enhancement(torch.randn(1, uses, 2))[0].numpy()
                appended, transmitted = append_observations(waveform, added, image_id, 4101, snr)
                np.testing.assert_array_equal(appended[:3060], observation)
                np.testing.assert_array_equal(transmitted[:3060], waveform)
            checks.append({"snr_db": snr, "header_accepted": received["header"]["accepted"],
                           "body_accepted": bool(received["events"] and received["events"][0]["accepted"]),
                           "base_output_max_difference": float(np.abs(new - old).max())})
        assert complete_latent(vae, var, [], None, device) is None
        del var
        torch.cuda.empty_cache()
        decoder = ContinuousDecoder(vae).to(device)
        original_pointers = {parameter.data_ptr() for parameter in vae.parameters()}
        assert all(parameter.data_ptr() not in original_pointers for parameter in decoder.parameters())
        with torch.no_grad():
            torch.testing.assert_close(decoder(quantized), original_rgb(vae, quantized), rtol=0, atol=0)
        perceptual = perceptual_model(device)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        mse, lpips = image_losses(decoder(continuous), pixels, perceptual)
        (mse + 0.1 * lpips).mean().backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1))
        assert gradient > 0 and np.isfinite(gradient)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        microbatch_seconds = time.perf_counter() - update_started
        assert state_sha256(vae) == vae_before
        decoder.requires_grad_(False)
        probe = continuous[:1].detach().clone().requires_grad_(True)
        decoder(probe).square().mean().backward()
        assert probe.grad is not None and float(probe.grad.abs().sum()) > 0
        record = {"status": "ENGINEERING_QUALIFICATION_PASS_NOT_SCIENTIFIC_RESULT", "shape": list(continuous.shape[1:]),
                  "decoder_trainable_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
                  "copy_storage_disjoint": True, "original_vae_state_unchanged": True,
                  "frozen_decoder_input_gradient_l1": float(probe.grad.abs().sum()),
                  "full_quantizer_latent_max_difference": float((mapped - quantized).abs().max()),
                  "base_checks": checks, "header_failure_original_gray": True,
                  "gradient_norm_before_clip": gradient, "microbatch4_update_seconds": microbatch_seconds,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "source_snapshot": sources,
                  "elapsed_seconds": time.perf_counter() - started,
                  "qualification_update_discarded": True, "source_role": "training_only"}
        write_json(output / "completion.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                              "source_snapshot": sources, "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
