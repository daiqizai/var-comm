from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.prefix_training_data import IMAGE_CACHE, MANIFEST_SHA
from var_comm.progressive import split_prefix

from .latent import complete_latent, quantized_latent
from .runtime import (
    CONFIG, OUTPUT, ROOT, ResourceBusy, configure, digest, model_paths, require_available,
    save_torch, settings, snapshot, verify_snapshot, wait_for_available, write_json,
)


def descriptors(population):
    if digest(IMAGE_CACHE / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("source image manifest changed")
    manifest = json.loads((IMAGE_CACHE / "manifest.json").read_text())
    return next(entry for entry in manifest["populations"] if entry["name"] == population)["shards"]


def canonical_prefixes(population):
    path = ROOT / "outputs/VAR-PREFIX-TRAINING-DATA-001" / f"{population}.npz"
    with np.load(path, allow_pickle=False) as archive:
        tokens = archive["tokens"][:, 0].copy()
        identifiers = archive["image_ids"].tolist()
        return dict(zip(identifiers, tokens)), digest(path)


def sample_key_sha256(image_ids):
    payload = json.dumps([str(value) for value in image_ids], ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    wait_for_available(output / "status.json")
    if not (OUTPUT / "qualification_001/completion.json").is_file():
        raise RuntimeError("real-model interface qualification is required")
    config = settings()
    device = torch.device("cuda:0")
    vae, var = load_models(model_paths(), device)
    model_scope = {"vae_sha256": state_sha256(vae), "var_sha256": state_sha256(var)}
    scope = {"cache_kind": "latent", "protocol_id": "official_latent_cache_v1", "noise_model": "none",
             "config_sha256": digest(CONFIG), "image_manifest_sha256": MANIFEST_SHA,
             "sample_key_schema": "image_id", **model_scope}
    registration = output / "registration.json"
    if registration.exists():
        metadata = json.loads(registration.read_text())
        verify_snapshot(metadata["source_snapshot"])
        if metadata.get("scope") != scope:
            raise RuntimeError("latent cache scope/model identity changed")
    else:
        metadata = {"config_sha256": digest(CONFIG), "source_snapshot": snapshot(),
                    "image_manifest_sha256": MANIFEST_SHA, "precision": "float32",
                    "old_fp16_fhat_used": False, "scope": scope, "created_at": time.time()}
        write_json(registration, metadata)
    if (output / "completion.json").exists():
        completion = json.loads((output / "completion.json").read_text())
        if completion.get("scope") != scope:
            raise RuntimeError("completed latent cache scope/model identity changed")
        print("exact latent cache already complete; preserving immutable completion receipt", flush=True)
        return 0
    started = time.perf_counter()
    processed_session = 0
    try:
        for population in ("calibration", "train"):
            directory = output / population
            directory.mkdir(exist_ok=True)
            prefixes, prefix_sha = canonical_prefixes(population)
            offset = 0
            for shard_index, descriptor in enumerate(descriptors(population)):
                path = directory / f"shard_{shard_index:04d}.pt"
                receipt_path = path.with_suffix(".json")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if digest(path) != receipt["sha256"]:
                        raise RuntimeError(f"committed latent shard changed: {path}")
                    if receipt.get("scope") != scope:
                        raise RuntimeError("committed latent shard scope/model changed")
                    offset += receipt["count"]
                    continue
                require_available()
                source_path = IMAGE_CACHE / descriptor["path"]
                if digest(source_path) != descriptor["sha256"]:
                    raise RuntimeError("source image shard changed")
                source = torch.load(source_path, map_location="cpu", weights_only=True)
                count = len(source["targets_u8"])
                continuous_values, quantized_values, base_values, token_values = [], [], [], []
                differences = 0
                batch = config["data"]["cache_encoder_batch"] if population == "train" else 1
                with torch.no_grad():
                    for start in range(0, count, batch):
                        require_available()
                        images = source["targets_u8"][start:start + batch].to(device).float().div(127.5).sub(1)
                        continuous = vae.quant_conv(vae.encoder(images))
                        scales = vae.quantize.f_to_idxBl_or_fhat(continuous, to_fhat=False)
                        quantized = quantized_latent(vae, scales)
                        continuous_values.append(continuous.cpu())
                        quantized_values.append(quantized.cpu())
                        joined = torch.cat(scales, dim=1).cpu().numpy().astype(np.uint16)
                        for within in range(len(images)):
                            source_index = start + within
                            identifier = source["image_ids"][source_index]
                            if identifier not in prefixes:
                                raise RuntimeError("source not in original training/calibration token population")
                            canonical = prefixes[identifier]
                            differences += int(np.count_nonzero(joined[within, :len(canonical)] != canonical))
                            base_prefix = (canonical if population == "train" else joined[within, :len(canonical)])
                            base = complete_latent(vae, var, split_prefix(base_prefix, 8),
                                                   int(source["labels"][source_index]), device)
                            base_values.append(base.cpu())
                        token_values.append(torch.from_numpy(joined.copy()).to(torch.int32))
                record = {"F": torch.cat(continuous_values), "Fq": torch.cat(quantized_values),
                          "Fb_TX": torch.cat(base_values), "full_tokens": torch.cat(token_values),
                          "base_tokens": torch.stack([torch.as_tensor(prefixes[identifier].astype(np.int64))
                                                       for identifier in source["image_ids"]]) if population == "train"
                          else torch.cat(token_values)[:, :255].long(),
                          "labels": source["labels"], "image_ids": source["image_ids"],
                          "source_indices": torch.arange(offset, offset + count),
                          "source_descriptor": descriptor, "scope": scope,
                          "sample_key_sha256": sample_key_sha256(source["image_ids"])}
                if not all(torch.isfinite(record[name]).all() for name in ("F", "Fq", "Fb_TX")):
                    raise RuntimeError("nonfinite latent cache")
                save_torch(path, record)
                elapsed = time.perf_counter() - started
                processed_session += count
                offset += count
                write_json(receipt_path, {"sha256": digest(path), "count": count, "shape": list(record["F"].shape[1:]),
                                          "source_shard_sha256": descriptor["sha256"], "canonical_prefix_cache_sha256": prefix_sha,
                                          "retokenized_prefix_difference_count": differences,
                                          "base_token_source": "original_prefix_cache" if population == "train" else "original_single_image_official_tokenizer",
                                          "elapsed_seconds_this_session": elapsed, "scope": scope,
                                          "sample_key_sha256": record["sample_key_sha256"]})
                write_json(output / "status.json", {"status": "CACHING_EXACT_LATENTS", "population": population,
                    "population_completed": offset, "pid": os.getpid(), "timestamp": time.time(),
                    "session_sources": processed_session, "seconds_per_source": elapsed / processed_session,
                    "latent_shape": list(record["F"].shape[1:])})
                print(f"cache {population} {offset}/{config['data'][population + '_sources']} {elapsed / processed_session:.3f}s/source", flush=True)
            if offset != config["data"][population + "_sources"]:
                raise RuntimeError("latent cache population count mismatch")
        sums = None
        squares = None
        positions = 0
        hashes = {}
        for path in sorted((output / "train").glob("shard_*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            latent = record["F"].double()
            current_sum = latent.sum((0, 2, 3))
            current_squares = latent.square().sum((0, 2, 3))
            sums = current_sum if sums is None else sums + current_sum
            squares = current_squares if squares is None else squares + current_squares
            positions += latent.shape[0] * latent.shape[2] * latent.shape[3]
            hashes[str(path.relative_to(output))] = digest(path)
        mean = sums / positions
        standard_deviation = (squares / positions - mean.square()).clamp_min(0).sqrt().clamp_min(config["s_F_floor"])
        write_json(output / "training_statistics.json", {"source_role": "training_only", "positions_per_channel": positions,
            "mean": mean.tolist(), "s_F": standard_deviation.tolist(), "floor": config["s_F_floor"],
            "accumulation_dtype": "float64", "training_shard_sha256": hashes})
        verify_snapshot(metadata["source_snapshot"])
        write_json(output / "completion.json", {"status": "EXACT_FP32_LATENT_CACHE_COMPLETE", "source_counts": {"train": 20000, "calibration": 1000},
            "registration_sha256": digest(registration), "statistics_sha256": digest(output / "training_statistics.json"),
            "session_elapsed_seconds": time.perf_counter() - started, "base_tokens_pinned_separately_from_full_retokenization": True,
            "scope": scope, "sample_key_schema": "image_id"})
    except ResourceBusy as error:
        write_json(output / "status.json", {"status": "YIELDED_TO_OTHER_AUTHORIZED_GPU_TASK", "reason": str(error), "pid": os.getpid()})
        return 75
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT / "cache_v1"))
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
