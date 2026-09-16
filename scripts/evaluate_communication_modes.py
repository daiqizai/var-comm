#!/usr/bin/env python3
"""Fixed raw/arithmetic prefix-mode matrix, with calibration/development/holdout boundaries."""

import argparse
import csv
from datetime import datetime
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

from benchmark_frozen_systems import assert_gpu_available, digest, gpu_state
from var_comm.next_scale_prior import load_models, preprocess, state_sha256
from var_comm.prefix_training_data import IMAGE_CACHE, read_image_population
from var_comm.progressive import complete_image, prefix_key, receive_whole, split_prefix, transmit_whole
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import indices_to_bits, load_native
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_snapshot, write_csv, write_json
from var_comm.whole_entropy import decode_phy, decode_source, encode_prefixes, payload_key, transmit


CONFIG = ROOT / "configs/communication_decision_study.json"
DIGITAL = ROOT / "outputs/VAR-PROGRESSIVE-CHANNEL-001"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Population:
    def __init__(self, role, config, frozen_method=None, manifest_path=None):
        self.role, self.images, self.token_lookup = role, None, None
        self.bindings = {}
        if role == "calibration":
            self.images, labels, identifiers, bindings = read_image_population("calibration")
            expected_ids = json.loads((ROOT / "outputs/WETOK-NATIVE-CACHE-20260912/calibration_ids.json").read_text())
            if identifiers != expected_ids:
                raise RuntimeError("calibration is not the original independent 1000-source population")
            self.targets = [{**binding, "class_index": int(label), "image_id": identifier}
                            for binding, label, identifier in zip(bindings, labels, identifiers)]
            self.bindings[str(IMAGE_CACHE / "manifest.json")] = sha256(IMAGE_CACHE / "manifest.json")
            self.bindings["native_calibration_ids_sha256"] = sha256(ROOT / "outputs/WETOK-NATIVE-CACHE-20260912/calibration_ids.json")
        elif role == "development":
            path = DIGITAL / "populations.json"
            self.targets = json.loads(path.read_text())["target"]
            self.bindings[str(path)] = sha256(path)
            tokens_path = ROOT / "outputs/VAR-NEXT-SCALE-PRIOR-DIAG-001/source_tokens.npz"
            self.bindings[str(tokens_path)] = sha256(tokens_path)
            with np.load(tokens_path, allow_pickle=False) as archive:
                self.token_lookup = dict(zip(archive["image_ids"].tolist(), archive["tokens"].copy()))
        else:
            if frozen_method is None or manifest_path is None:
                raise RuntimeError("holdout requires a frozen method and concrete frozen manifest before pixels are accessed")
            method = json.loads(frozen_method.read_text())
            manifest = json.loads(manifest_path.read_text())
            if (method["status"] != "FINAL_METHOD_AND_HOLDOUT_PROTOCOL_FROZEN" or
                    method["config_sha256"] != sha256(CONFIG) or method["holdout_manifest_sha256"] != sha256(manifest_path) or
                    manifest["role"] != "independent_holdout_after_method_freeze"):
                raise RuntimeError("holdout protocol is not frozen or its data list changed")
            self.targets = manifest["images"]
            self.bindings[str(frozen_method)] = sha256(frozen_method)
            self.bindings[str(manifest_path)] = sha256(manifest_path)
        expected = config[role + "_sources"]
        identifiers = [target["image_id"] for target in self.targets]
        if len(self.targets) != expected or len(set(identifiers)) != expected:
            raise RuntimeError("source population is incomplete or duplicated")
        development_ids = set(json.loads((ROOT / "outputs/WETOK-NATIVE-CACHE-20260912/development_ids.json").read_text()))
        if role != "development" and development_ids.intersection(identifiers):
            raise RuntimeError("calibration/holdout overlaps original development")
        self.development_pixel_hashes = {target["preprocessed_rgb_sha256"] for target in json.loads((DIGITAL / "populations.json").read_text())["target"]}

    def source(self, index, vae, device):
        target = self.targets[index]
        if self.role == "calibration":
            pixels = self.images[index].numpy().copy()
        elif self.role == "development":
            path = DIGITAL / f"images/{index:03d}/reconstructions.npz"
            with np.load(path, allow_pickle=False) as archive:
                pixels = np.rint(archive["source"] * 255).astype(np.uint8)
        else:
            path = Path(target["path"])
            if sha256(path) != target["file_sha256"]:
                raise RuntimeError("registered holdout image bytes changed")
            normalized, rgb_sha = preprocess(path)
            pixels = normalized.add(1).mul(127.5).round().to(torch.uint8).numpy()
            if rgb_sha != target["preprocessed_rgb_sha256"]:
                raise RuntimeError("registered holdout preprocessing changed")
        if self.token_lookup is not None:
            source = split_prefix(self.token_lookup[target["image_id"]], 10)
        else:
            normalized = torch.from_numpy(pixels[None]).float().div(127.5).sub(1).to(device)
            source = [tokens[0].cpu().numpy() for tokens in vae.img_to_idxBl(normalized)]
        if self.role != "development" and digest(pixels) in self.development_pixel_hashes:
            raise RuntimeError("calibration/holdout duplicates a development image by pixel content")
        return pixels, source, target


def compare_source(prefix, label, recovered_mode, source, target, sent_mode, complete, header_ok, body_ok):
    correct = (complete and recovered_mode == sent_mode and label == int(target["class_index"]) and len(prefix) == sent_mode and
               all(np.array_equal(received, truth) for received, truth in zip(prefix, source[:sent_mode])))
    bit_error = ""
    if len(prefix) == sent_mode:
        recovered = indices_to_bits(np.concatenate(prefix))
        truth = indices_to_bits(np.concatenate(source[:sent_mode]))
        if recovered.shape == truth.shape:
            bit_error = float(np.mean(recovered != truth))
    return bool(correct), bool(correct and header_ok and body_ok), bit_error


def header_fields_correct(family, header, label, mode, transmitted_length=0):
    if not header["accepted"] or header["label"] != label or header["mode"] != mode:
        return False
    return family == "raw" or header["length_field"] == transmitted_length


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("calibration", "development", "holdout"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--frozen-method", type=Path)
    parser.add_argument("--holdout-manifest", type=Path)
    arguments = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if config["new_training"] or config["new_backbone"] or config["automatic_candidate_expansion"]:
        raise RuntimeError("this study cannot train, expand candidates or change the backbone")
    if not arguments.execute:
        print("PLAN ONLY: six fixed candidate modes, no training or policy search")
        return
    if arguments.population == "holdout" and arguments.mode != "full":
        raise RuntimeError("do not inspect holdout through an engineering smoke")
    policy_sha = None
    if arguments.population != "calibration":
        if arguments.policies is None:
            raise RuntimeError("freeze calibration policies before development or holdout evaluation")
        policies = json.loads(arguments.policies.read_text())
        if policies["status"] != "CALIBRATION_POLICIES_FROZEN" or policies["config_sha256"] != sha256(CONFIG):
            raise RuntimeError("policies are not frozen to the current preregistered protocol")
        policy_sha = sha256(arguments.policies)
    output = arguments.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"):
        raise ValueError("communication study output escaped its declared project directory")
    if (output / "completion.json").exists():
        raise RuntimeError("this source population is already complete")
    if arguments.resume:
        metadata = json.loads((output / "metadata.json").read_text())
        verify_snapshot(metadata["source_hashes"])
        if (metadata["config_sha256"] != sha256(CONFIG) or metadata["population"] != arguments.population or
                metadata["mode"] != arguments.mode or metadata["policy_sha256"] != policy_sha):
            raise RuntimeError("resume changed protocol or policy selection")
    else:
        create_output(output)
    (output / "images").mkdir(exist_ok=True)
    (output / "sessions").mkdir(exist_ok=True)
    started = time.perf_counter()
    session = output / "sessions" / f"{time.time_ns()}_{os.getpid()}.json"
    completed, rows = 0, []

    def status(value, **extra):
        record = {"status": value, "pid": os.getpid(), "population": arguments.population, "mode": arguments.mode,
            "updated_local": datetime.now().astimezone().isoformat(), "completed_sources": completed,
            "process_seconds": time.perf_counter() - started, "new_training": False, "research_goal_complete": False, **extra}
        write_json(output / "status.json", record)
        write_json(session, record)

    try:
        status("LOADING_FIXED_COMMUNICATION_STUDY")
        assert_gpu_available()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        check_path = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915/ENTROPY_GPU_CHECK_001/completion.json"
        check = json.loads(check_path.read_text())
        if check["status"] != "WHOLE_ENTROPY_CODEC_GPU_CHECK_PASS":
            raise RuntimeError("whole entropy implementation has not passed its GPU roundtrip check")
        verify_snapshot(check["source_hashes"])
        paths = yaml.safe_load((ROOT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
        for name in ("vae_checkpoint", "var_checkpoint"):
            if sha256(paths[name]) != paths[name + "_sha256"]:
                raise RuntimeError("frozen source model checkpoint changed")
        vae, var = load_models(paths, device)
        quality_config = yaml.safe_load((ROOT / "configs/progressive_channel.yaml").read_text())["quality"]
        perceptual, dino, linear_weights = load_quality_models(quality_config, device)
        models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
        before = {name: state_sha256(model) for name, model in models.items()}
        load_native()
        population = Population(arguments.population, config, arguments.frozen_method, arguments.holdout_manifest)
        source_indices = list(range(len(population.targets))) if arguments.mode == "full" else [0, len(population.targets) - 1]
        source_files = [Path(__file__), CONFIG, ROOT / "reports/whole_entropy_and_mode_policy_protocol_20260915.md",
            *[ROOT / "src/var_comm" / name for name in ("whole_entropy.py", "entropy.py", "next_scale_prior.py", "progressive.py", "scale_channel.py", "quality.py", "study.py", "prefix_training_data.py")]]
        if not arguments.resume:
            metadata = {"started_local": datetime.now().astimezone().isoformat(), "config_sha256": sha256(CONFIG),
                "population": arguments.population, "mode": arguments.mode, "policy_sha256": policy_sha,
                "frozen_method_sha256": sha256(arguments.frozen_method) if arguments.frozen_method else None,
                "source_indices": source_indices, "source_hashes": snapshot(output, source_files),
                "population_bindings": population.bindings, "frozen_models": before, "GPU": gpu_state(),
                "quality_cache_scope": "within_source_same_received_prefix_or_decoded_payload_only_not_GT_repair",
                "no_online_latency_claim_from_cached_quality": True}
            write_json(output / "metadata.json", metadata)
            write_json(output / "population.json", population.targets)
        elif metadata["frozen_models"] != before or metadata["population_bindings"] != population.bindings:
            raise RuntimeError("resumed population or frozen models changed")
        seeds = config[arguments.population + "_seeds"]
        total_expected = len(source_indices) * len(config["prefix_modes"]) * len(config["coding_families"]) * len(config["snrs_db"]) * len(seeds)
        for image_index in source_indices:
            directory = output / "images" / f"{image_index:04d}"
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved["image_id"] != population.targets[image_index]["image_id"]:
                    raise RuntimeError("committed source identity changed")
                for relative, expected in saved["output_hashes"].items():
                    if sha256(directory / relative) != expected:
                        raise RuntimeError("committed source archive changed")
                rows.extend(read_rows(directory / "per_frame.csv"))
                completed += 1
                continue
            assert_gpu_available()
            directory.mkdir(exist_ok=True)
            tick = time.perf_counter()
            pixels, source, target = population.source(image_index, vae, device)
            label = int(target["class_index"])
            payloads = encode_prefixes(vae, var, source, label, device)
            signals, ledgers, packed = {}, {}, {}
            for mode in config["prefix_modes"]:
                signals["raw", mode] = transmit_whole(source, label, mode)
                raw_bits = 12 * sum(len(tokens) for tokens in source[:mode])
                ledgers["raw", mode] = {"raw_payload_bits": raw_bits, "actual_payload_bits": raw_bits,
                    "attempted_arithmetic_bits": "", "arithmetic_length_field": "", "raw_fallback": False,
                    "header_uses": 68, "data_uses": 2992, "complex_uses": 3060, "total_energy": 6120.,
                    "header_payload_bits": 12, "header_crc_bits": 16, "header_tail_bits": 6, "header_coded_bits": 136,
                    "data_crc_bits": 16, "data_tail_bits": 6, "data_mother_coded_bits": 2 * (raw_bits + 22),
                    "data_coded_bits": 5984, "data_information_rate": (raw_bits + 22) / 5984}
                signals["arithmetic", mode], ledgers["arithmetic", mode] = transmit(payloads[mode], label, mode)
                packed[f"payload_m{mode}"] = payloads[mode]["payload"]
            for key, signal in signals.items():
                if signal.shape != (3060, 2) or not np.all(np.abs(signal) == 1) or np.square(signal).sum() != 6120:
                    raise RuntimeError("candidate signal changed the fixed N/E budget")
                packed[f"signal_{key[0]}_m{key[1]}"] = signal.astype(np.int8)
            images, image_lookup, decoded_cache, prefixes, local_rows = [], {}, {}, {}, []
            for snr in config["snrs_db"]:
                for seed in seeds:
                    noise = seeded_noise(target["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                    for family in config["coding_families"]:
                        for mode in config["prefix_modes"]:
                            signal = signals[family, mode]
                            observed = signal + noise
                            if family == "raw":
                                result = receive_whole(observed, snr)
                                prefix, receiver_label = result["prefix"], result["label"]
                                header_ok = bool(result["header"]["accepted"])
                                receiver_mode = result["header"]["mode"] if header_ok else None
                                body_ok = bool(result["events"][0]["accepted"]) if result["events"] else False
                                complete = bool(header_ok and len(prefix) == receiver_mode)
                                image, padding, source_error = None, 0, ""
                                fields_correct = header_fields_correct(family, result["header"], label, mode)
                                payload_correct = len(prefix) == mode and all(np.array_equal(received, truth) for received, truth in zip(prefix, source[:mode]))
                            else:
                                phy = decode_phy(observed, snr)
                                key = payload_key(phy)
                                if key not in decoded_cache:
                                    decoded_cache[key] = decode_source(phy, vae, var, device)
                                result = decoded_cache[key]
                                prefix, receiver_label, receiver_mode = result["prefix"], phy["label"], phy["mode"]
                                header_ok, body_ok = bool(phy["header"]["accepted"]), bool(phy["body_crc_accepted"])
                                complete = result["source_complete"]
                                image, padding, source_error = result["image"], result["arithmetic_padding_reads"], result["source_error"]
                                fields_correct = header_fields_correct(family, phy["header"], label, mode, payloads[mode]["length_field"])
                                payload_correct = phy["payload"] is not None and np.array_equal(phy["payload"], payloads[mode]["payload"])
                            key = "header_erasure" if receiver_label is None else prefix_key(prefix, receiver_label)
                            if key not in image_lookup:
                                if image is None:
                                    image = complete_image(vae, var, prefix, receiver_label, device)
                                image_lookup[key] = len(images)
                                images.append(image)
                                prefixes[f"prefix_{len(images) - 1}"] = np.concatenate(prefix).astype(np.uint16) if prefix else np.empty(0, dtype=np.uint16)
                            elif image is not None and not np.array_equal(image, images[image_lookup[key]]):
                                raise RuntimeError("identical actual received prefixes produced different renderers")
                            correct, reliable, bit_error = compare_source(prefix, receiver_label, receiver_mode, source, target, mode, complete,
                                bool(header_ok and fields_correct and payload_correct), body_ok)
                            local_rows.append({"population": arguments.population, "image_index": image_index, "image_id": target["image_id"],
                                "class_index": label, "snr_db": snr, "seed": seed, "family": family, "mode": mode,
                                "arm": f"{family}_m{mode}", "header_accepted": int(header_ok), "body_crc_accepted": int(body_ok),
                                "header_false_acceptance": int(header_ok and not fields_correct), "header_fields_correct": int(fields_correct),
                                "payload_bits_correct": int(payload_correct),
                                "source_complete": int(complete), "source_correct": int(correct), "accepted_correct": int(reliable),
                                "source_bit_error_rate": bit_error, "candidate_prefix_scales": len(prefix),
                                "receiver_label": receiver_label if receiver_label is not None else "", "arithmetic_padding_reads": padding,
                                "source_error": source_error, "image_index_in_archive": image_lookup[key], "prefix_sha256": key,
                                "source_pixels_sha256": digest(pixels), "received_sha256": digest(observed), "transmitted_sha256": digest(signal),
                                **ledgers[family, mode]})
            metric_source = torch.from_numpy(pixels).float().div(127.5).sub(1).add(1).mul(.5).numpy()
            quality, source_features, image_features = quality_metrics(metric_source, images, perceptual, dino, device)
            for row in local_rows:
                metric = quality[row["image_index_in_archive"]]
                row.update(psnr_db=metric["psnr_db"], lpips=metric["lpips_alex"], dino=metric["dino_cosine"])
            assert_gpu_available()
            np.savez(directory / "transmissions.npz", **packed)
            np.savez(directory / "source_tokens.npz", tokens=np.concatenate(source).astype(np.uint16), label=np.asarray(label))
            np.savez(directory / "candidates.npz", **prefixes)
            np.savez(directory / "reconstructions.npz", images=np.stack(images), source_dino=source_features, reconstruction_dino=image_features)
            write_csv(directory / "per_frame.csv", local_rows)
            write_json(receipt_path, {"image_id": target["image_id"], "source_pixels_sha256": digest(pixels), "rows": len(local_rows),
                "unique_images": len(images), "unique_entropy_decodes": len(decoded_cache), "source_seconds": time.perf_counter() - tick,
                "output_hashes": artifact_hashes(directory)})
            rows.extend(local_rows)
            completed += 1
            status("EVALUATING_FIXED_MODE_MATRIX", source_count=len(source_indices), rows=len(rows), last_source_seconds=time.perf_counter() - tick)
            print(f"{arguments.population} {completed}/{len(source_indices)} rows={len(rows)} unique_images={len(images)} seconds={time.perf_counter()-tick:.2f}", flush=True)
        if len(rows) != total_expected:
            raise RuntimeError("fixed candidate/SNR/noise/source matrix is incomplete")
        after = {name: state_sha256(model) for name, model in models.items()}
        if after != before:
            raise RuntimeError("frozen models changed during communication evaluation")
        verify_snapshot(metadata["source_hashes"])
        write_csv(output / "per_frame.csv", rows)
        status("FIXED_MODE_MATRIX_COMPLETE", source_count=len(source_indices), rows=len(rows))
        write_json(output / "completion.json", {"status": "FIXED_MODE_MATRIX_COMPLETE", "completed_local": datetime.now().astimezone().isoformat(),
            "population": arguments.population, "mode": arguments.mode, "sources": len(source_indices), "rows": len(rows),
            "config_sha256": sha256(CONFIG), "policy_sha256": policy_sha, "source_hashes": metadata["source_hashes"],
            "frozen_before": before, "frozen_after": after, "all_failures_included": True, "new_training": False,
            "research_goal_complete": False, "output_hashes": artifact_hashes(output)})
    except BaseException as error:
        path = output / f"failure_{time.time_ns()}.json"
        write_json(path, {"error": repr(error), "traceback": traceback.format_exc(), "completed_sources": completed})
        status("STOPPED_WITH_FAILURE_RETAINED", error=repr(error))
        raise


if __name__ == "__main__":
    main()
