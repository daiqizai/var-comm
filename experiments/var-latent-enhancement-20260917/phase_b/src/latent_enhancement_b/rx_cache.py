from __future__ import annotations

import json
import os
import time

import torch

from latent_enhancement.latent import complete_latent
from latent_enhancement.runtime import ResourceBusy, configure, digest, model_paths, require_available, save_torch, verify_snapshot, write_json
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import prefix_key, split_prefix

from .common import (CONFIG_B, OUT_B, cache_shard, config_b, load_gate, sample_key_digest,
                     stage_b_sources, validate_gpu_qualification, validate_sample_grid)


def main():
    configure()
    load_gate()
    if not (OUT_B / "qualification/completion.json").exists():
        raise RuntimeError("selected-Dc real-model qualification has not passed")
    validate_gpu_qualification()
    phy = OUT_B / "phy_cache"
    if not (phy / "completion.json").exists():
        raise RuntimeError("actual original-PHY candidate cache is incomplete")
    output = OUT_B / "rx_cache"
    output.mkdir(parents=True, exist_ok=True)
    config = config_b()
    require_available()
    device = torch.device("cuda:0")
    vae, var = load_models(model_paths(), device)
    model_scope = {"vae_sha256": state_sha256(vae), "var_sha256": state_sha256(var),
                   "decoder_gate_sha256": digest(OUT_B / "decoder_gate.json")}
    scope = {"cache_kind": "rx", "protocol_id": "original_m8_v1",
             "noise_model": "seeded_noise_v1:variance=1/SNR", "config_sha256": digest(CONFIG_B),
             **model_scope, "sample_key_schema": "image_id/source/SNR/noise_seed"}
    if (output / "completion.json").exists():
        completion = json.loads((output / "completion.json").read_text())
        if completion.get("scope") != scope:
            raise RuntimeError("actual-RX cache scope/model/decoder identity changed")
        return 0
    registration_path = output / "registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
        verify_snapshot(registration["source_bindings"])
        if digest(phy / "completion.json") != registration["phy_completion_sha256"]:
            raise RuntimeError("PHY replay receipt changed")
        if registration.get("scope") != scope:
            raise RuntimeError("actual-RX registration scope/model/decoder identity changed")
    else:
        registration = {"source_bindings": stage_b_sources(), "phy_completion_sha256": digest(phy / "completion.json"),
                        "decoder_gate_sha256": digest(OUT_B / "decoder_gate.json"), "scope": scope,
                        "created_at": time.time()}
        write_json(registration_path, registration)
    started = time.perf_counter()
    new_completions = 0
    frame_count = 0
    output_hashes = {}
    try:
        for population, shard_count in (("calibration", 10), ("train", 200)):
            destination = output / population
            destination.mkdir(exist_ok=True)
            for shard_index in range(shard_count):
                path = destination / f"shard_{shard_index:04d}.pt"
                receipt_path = path.with_suffix(".json")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if digest(path) != receipt["sha256"]:
                        raise RuntimeError("committed actual-RX latent cache changed")
                    if receipt.get("scope") != scope:
                        raise RuntimeError("committed actual-RX shard scope/model changed")
                    output_hashes[str(path.relative_to(output))] = receipt["sha256"]
                    continue
                source, _, source_receipt = cache_shard(population, shard_index)
                phy_path = phy / population / path.name
                phy_receipt = json.loads(phy_path.with_suffix(".json").read_text())
                if digest(phy_path) != phy_receipt["sha256"] or source_receipt["sha256"] != phy_receipt["source_cache_sha256"]:
                    raise RuntimeError("candidate cache or source binding changed")
                candidates = torch.load(phy_path, map_location="cpu", weights_only=True)
                expected_seeds = config["training_base_noise_seeds"] if population == "train" else config["calibration_noise_seeds"]
                if candidates.get("scope") != {"cache_kind": "phy", "population_roles": ["calibration", "train"],
                                                "protocol_id": "original_m8_v1", "noise_model": "seeded_noise_v1:variance=1/SNR",
                                                "config_sha256": digest(CONFIG_B), "sample_key_schema": "image_id/source/SNR/noise_seed"}:
                    raise RuntimeError("candidate cache PHY scope changed")
                validate_sample_grid(candidates, image_ids=source["image_ids"], snrs_db=config["snrs_db"],
                                     noise_seeds=expected_seeds,
                                     arrays=("candidate_tokens", "lengths", "received_labels", "rx_status", "scores"))
                if candidates.get("sample_key_sha256") != sample_key_digest(source["image_ids"], config["snrs_db"], expected_seeds):
                    raise RuntimeError("candidate cache sample-key digest changed")
                if source["image_ids"] != candidates["image_ids"]:
                    raise RuntimeError("candidate/source IDs not paired")
                if candidates["snrs_db"] != config["snrs_db"]:
                    raise RuntimeError("SNR grid changed")
                count = len(source["image_ids"])
                grid = candidates["rx_status"].shape[:-1]
                latents = torch.zeros((*grid, *source["F"].shape[1:]), dtype=torch.float32)
                hits = 0
                for source_index in range(count):
                    require_available()
                    original_prefix = split_prefix(source["base_tokens"][source_index].numpy(), 8)
                    memo = {prefix_key(original_prefix, int(source["labels"][source_index])): source["Fb_TX"][source_index]}
                    for snr_index in range(grid[1]):
                        for seed_index in range(grid[2]):
                            position = (source_index, snr_index, seed_index)
                            status = candidates["rx_status"][position]
                            if not bool(status[0]):
                                continue
                            received_mode = int(status[2])
                            length = int(candidates["lengths"][position])
                            received_label = int(candidates["received_labels"][position])
                            if received_label < 0 or not length:
                                raise RuntimeError("accepted header has no actual usable candidate")
                            prefix = split_prefix(candidates["candidate_tokens"][position][:length].numpy(), received_mode)
                            key = prefix_key(prefix, received_label)
                            if key in memo:
                                hits += 1
                            else:
                                memo[key] = complete_latent(vae, var, prefix, received_label, device)[0].cpu()
                                new_completions += 1
                            latents[position].copy_(memo[key])
                if not torch.isfinite(latents).all():
                    raise RuntimeError("nonfinite actual RX latent")
                save_torch(path, {"Fb_RX": latents, "rx_status": candidates["rx_status"], "image_ids": source["image_ids"],
                                 "snrs_db": candidates["snrs_db"], "noise_seeds": candidates["noise_seeds"],
                                 "phy_cache_sha256": phy_receipt["sha256"], "source_cache_sha256": source_receipt["sha256"],
                                 "scope": scope, "sample_key_sha256": candidates["sample_key_sha256"]})
                frames = int(candidates["rx_status"][..., 0].numel())
                frame_count += frames
                receipt = {"sha256": digest(path), "frames": frames, "sources": count,
                           "phy_cache_sha256": phy_receipt["sha256"], "source_cache_sha256": source_receipt["sha256"],
                           "exact_received_prefix_memo_hits": hits, "all_failures_retained": True,
                           "base_truth_never_used_to_repair_candidate": True, "scope": scope,
                           "sample_key_sha256": candidates["sample_key_sha256"]}
                write_json(receipt_path, receipt)
                output_hashes[str(path.relative_to(output))] = receipt["sha256"]
                elapsed = time.perf_counter() - started
                write_json(output / "status.json", {"status": "RENDERING_ACTUAL_DECODED_PREFIXES", "pid": os.getpid(),
                    "population": population, "sources_completed": (shard_index + 1) * count,
                    "session_frames": frame_count, "new_VAR_completions": new_completions, "elapsed_seconds": elapsed,
                    "timestamp": time.time()})
                print(f"RX cache {population} {(shard_index + 1) * count} sources, {new_completions} new VAR completions", flush=True)
        verify_snapshot(registration["source_bindings"])
        write_json(output / "completion.json", {"status": "ACTUAL_RX_LATENT_CACHE_READY", "output_hashes": output_hashes,
            "registration_sha256": digest(registration_path), "session_elapsed_seconds": time.perf_counter() - started,
            "all_failures_retained": True, "no_free_RX_information": True, "scope": scope,
            "sample_key_schema": "image_id/source/SNR/noise_seed"})
        return 0
    except ResourceBusy as error:
        write_json(output / "status.json", {"status": "YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK", "reason": str(error), "pid": os.getpid()})
        return 75


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceBusy:
        raise SystemExit(75)
    except Exception as error:
        write_json(OUT_B / "rx_cache" / f"failure_{time.time_ns()}.json", {"type": type(error).__name__, "error": str(error)})
        raise
