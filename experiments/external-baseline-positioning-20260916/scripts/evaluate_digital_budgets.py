#!/usr/bin/env python3
"""Existing whole-frame digital codecs at two paid author-compatible budgets."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "src"), str(PROJECT / "src"), str(PROJECT / "scripts")]

import numpy as np
import torch
import yaml
from pytorch_msssim import ssim

from benchmark_frozen_systems import assert_gpu_available
from evaluate_communication_modes import Population, compare_source, header_fields_correct
from external_positioning.digital_budget import raw_prefix, receive, transmit
from var_comm.mode_policies import fit_actions, summarize_candidates
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import complete_image, prefix_key, split_prefix
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import indices_to_bits
from var_comm.study import seeded_noise, sha256, write_csv, write_json
from var_comm.whole_entropy import decode_source, payload_key


HISTORY = PROJECT / "outputs/COMMUNICATION-CONVERGENCE-20260915"
CONFIG = EXPERIMENT / "configs/digital_common_budget.json"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def indices_for(role):
    return np.rint(np.linspace(0, 999, 100)).astype(int).tolist() if role == "calibration" else list(range(100))


def implementation_bindings():
    files = [Path(__file__), CONFIG, EXPERIMENT / "src/external_positioning/digital_budget.py"]
    files.extend(PROJECT / "src/var_comm" / name for name in
                 ("mode_policies.py", "whole_entropy.py", "entropy.py", "progressive.py", "scale_channel.py", "token_trellis.cpp", "quality.py", "next_scale_prior.py"))
    return {str(path): sha256(path) for path in files}


def legacy_source(role, index):
    directory = HISTORY / ("CALIBRATION_001" if role == "calibration" else "DEVELOPMENT_001") / "images" / f"{index:04d}"
    receipt = json.loads((directory / "receipt.json").read_text())
    with np.load(directory / "source_tokens.npz", allow_pickle=False) as archive:
        source, label = split_prefix(archive["tokens"], 10), int(archive["label"])
    with np.load(directory / "transmissions.npz", allow_pickle=False) as archive:
        transmissions = {key: archive[key].copy() for key in archive.files}
    rows = read_csv(directory / "per_frame.csv")
    return directory, receipt, source, label, transmissions, rows


def physical_state(record, arrays):
    header = {"accepted": bool(record["header_accepted"]), "label": record["receiver_label"],
              "mode": record["receiver_mode"], "length_field": record["receiver_length_field"]}
    return {"header": header, "label": record["receiver_label"], "mode": record["receiver_mode"],
            "payload": arrays[record["payload_key"]] if record["header_accepted"] else None,
            "body_crc_accepted": bool(record["body_crc_accepted"])}


def prepare_phy(arguments):
    config = json.loads(CONFIG.read_text())
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "phy_completion.json").exists():
        raise RuntimeError("completed PHY must not be rerun")
    bindings = implementation_bindings()
    policies = None
    if arguments.population == "development":
        if arguments.policies is None:
            raise ValueError("development requires a policy frozen on calibration")
        policies = json.loads(arguments.policies.read_text())
        if policies["status"] != "COMMON_BUDGET_CALIBRATION_POLICIES_FROZEN":
            raise ValueError("calibration must finish before new-budget development")
        bindings[str(arguments.policies)] = sha256(arguments.policies)
    metadata = {"population": arguments.population, "indices": indices_for(arguments.population), "bindings": bindings,
                "no_training_or_holdout": True, "policy_sha256": sha256(arguments.policies) if arguments.policies else None}
    if (output / "metadata.json").exists() and json.loads((output / "metadata.json").read_text()) != metadata:
        raise RuntimeError("resume changed the registered digital protocol")
    write_json(output / "metadata.json", metadata)
    (output / "images").mkdir(exist_ok=True)
    total, started = 0, time.perf_counter()
    for position, index in enumerate(metadata["indices"]):
        destination = output / "images" / f"{index:04d}"
        destination.mkdir(exist_ok=True)
        if (destination / "phy.json").exists():
            saved = json.loads((destination / "phy.json").read_text())
            if sha256(destination / "payloads.npz") != saved["payloads_sha256"]:
                raise RuntimeError("committed receiver candidates changed")
            total += len(saved["records"])
            continue
        _old, receipt, source, label, packed, _rows = legacy_source(arguments.population, index)
        records, arrays, signals = [], {}, {}
        for budget in config["budgets"]:
            for snr in config["snrs_db"]:
                for seed in config[arguments.population + "_seeds"]:
                    noise = seeded_noise(receipt["image_id"], seed, (budget, 2))
                    for family in config["families"]:
                        modes = config["modes"] if policies is None else [policies["budgets"][str(budget)]["actions"][family]["quality"][str(float(snr))]]
                        for mode in modes:
                            key = budget, family, mode
                            if key not in signals:
                                signals[key] = transmit(source, packed[f"payload_m{mode}"], label, mode, family, budget)
                            signal, ledger = signals[key]
                            observed = signal + noise / np.sqrt(10 ** (snr / 10))
                            physical = receive(observed, snr, family)
                            payload_name = f"payload_{len(records):04d}"
                            if physical["payload"] is not None:
                                arrays[payload_name] = physical["payload"]
                            correct_header = header_fields_correct(family, physical["header"], label, mode, ledger["length_field"])
                            sent_bits = indices_to_bits(np.concatenate(source[:mode])) if family == "raw" else packed[f"payload_m{mode}"]
                            correct_payload = physical["payload"] is not None and np.array_equal(physical["payload"], sent_bits)
                            records.append({"population": arguments.population, "image_index": index, "image_id": receipt["image_id"],
                                            "source_pixels_sha256": receipt["source_pixels_sha256"], "class_index": label,
                                            "snr_db": float(snr), "seed": seed, "family": family, "mode": mode,
                                            "header_accepted": int(physical["header"]["accepted"]),
                                            "body_crc_accepted": int(physical["body_crc_accepted"]),
                                            "header_fields_correct": int(correct_header), "payload_bits_correct": int(correct_payload),
                                            "receiver_label": physical["label"], "receiver_mode": physical["mode"],
                                            "receiver_length_field": physical["header"].get("length_field", 0),
                                            "payload_key": payload_name, "standard_noise_sha256": digest(noise),
                                            "received_sha256": digest(observed), "transmitted_sha256": digest(signal),
                                            "header_crc_bits": 16, "data_crc_bits": 16, "tail_bits_per_block": 6,
                                            "data_coded_bits": 2 * ledger["data_uses"],
                                            "data_mother_coded_bits": 2 * (len(sent_bits) + 22), **ledger})
        np.savez(destination / "payloads.npz", **arrays)
        write_json(destination / "phy.json", {"records": records, "payloads_sha256": sha256(destination / "payloads.npz"),
                   "legacy_receipt_sha256": sha256(_old / "receipt.json"), "candidates_are_actual_PHY_outputs": True})
        total += len(records)
        write_json(output / "status.json", {"status": "COMMON_BUDGET_CPU_PHY", "population": arguments.population,
                   "sources_completed": position + 1, "transmissions": total, "pid": os.getpid(),
                   "elapsed_seconds": time.perf_counter() - started, "updated_at": datetime.now().astimezone().isoformat()})
        if (position + 1) % 10 == 0:
            print(f"{arguments.population} PHY: {position + 1}/100 sources; {total} transmissions", flush=True)
    if any(sha256(path) != expected for path, expected in bindings.items()):
        raise RuntimeError("digital protocol changed during CPU evaluation")
    write_json(output / "phy_completion.json", {"status": "COMMON_BUDGET_PHY_COMPLETE", "transmissions": total,
               "seconds": time.perf_counter() - started, "bindings": bindings})


def legacy_receiver_cache(role, index, directory, packed, rows):
    with np.load(directory / "candidates.npz", allow_pickle=False) as archive:
        candidates = {key: archive[key].copy() for key in archive.files}
    images, entropy = {}, {}
    for row in rows:
        key = row["prefix_sha256"]
        if key not in images:
            flat = candidates[f"prefix_{row['image_index_in_archive']}"]
            prefix = split_prefix(flat, int(row["candidate_prefix_scales"])) if len(flat) else []
            images[key] = {"prefix": prefix, "source_complete": bool(int(row["source_complete"])),
                           "arithmetic_padding_reads": int(row["arithmetic_padding_reads"]), "source_error": row["source_error"],
                           "image_archive": str(directory / "reconstructions.npz"), "image_slot": int(row["image_index_in_archive"]),
                           "psnr_db": float(row["psnr_db"]), "lpips": float(row["lpips"]), "dino": float(row["dino"])}
    seed = 4101 if role == "calibration" else 2001
    for mode in (7, 8, 9):
        chosen = next(row for row in rows if row["family"] == "arithmetic" and int(row["mode"]) == mode
                      and float(row["snr_db"]) == 19 and int(row["seed"]) == seed)
        observed = packed[f"signal_arithmetic_m{mode}"].astype(np.float64) + seeded_noise(chosen["image_id"], seed, (3060, 2)) / np.sqrt(10 ** 1.9)
        if digest(observed) != chosen["received_sha256"]:
            raise RuntimeError("old observable receiver inputs cannot be reproduced")
        physical = receive(observed, 19., "arithmetic")
        entropy[payload_key(physical)] = images[chosen["prefix_sha256"]]
    return images, entropy


@torch.no_grad()
def render(arguments):
    output = arguments.output
    if (output / "completion.json").exists():
        raise RuntimeError("completed digital measurements must not be repeated")
    metadata = json.loads((output / "metadata.json").read_text())
    physical_receipt = json.loads((output / "phy_completion.json").read_text())
    if any(sha256(path) != expected for path, expected in metadata["bindings"].items()):
        raise RuntimeError("digital code changed after physical candidates were produced")
    assert_gpu_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    paths = yaml.safe_load((PROJECT / "configs/next_scale_prior_diagnostic.yaml").read_text())["paths"]
    vae, var = load_models(paths, device)
    quality_paths = yaml.safe_load((PROJECT / "configs/progressive_channel.yaml").read_text())["quality"]
    perceptual, dino, _weights = load_quality_models(quality_paths, device)
    models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
    before = {name: state_sha256(model) for name, model in models.items()}
    original_config = json.loads((PROJECT / "configs/communication_decision_study.json").read_text())
    population = Population(arguments.population, original_config)
    started, all_rows, reused_count, fresh_count = time.perf_counter(), [], 0, 0
    for position, index in enumerate(metadata["indices"]):
        assert_gpu_available()
        destination = output / "images" / f"{index:04d}"
        if (destination / "render_completion.json").exists():
            receipt = json.loads((destination / "render_completion.json").read_text())
            if sha256(destination / "per_frame.csv") != receipt["per_frame_sha256"]:
                raise RuntimeError("completed digital quality rows changed")
            all_rows.extend(read_csv(destination / "per_frame.csv"))
            continue
        directory, legacy_receipt, source, label, packed, old_rows = legacy_source(arguments.population, index)
        target = population.targets[index]
        if target["image_id"] != legacy_receipt["image_id"] or int(target["class_index"]) != label:
            raise RuntimeError("legacy source no longer matches the original population")
        if arguments.population == "calibration":
            pixels = population.images[index].numpy().copy()
        else:
            with np.load(PROJECT / "outputs/VAR-PROGRESSIVE-CHANNEL-001/images" / f"{index:03d}/reconstructions.npz", allow_pickle=False) as archive:
                pixels = np.rint(archive["source"] * 255).astype(np.uint8)
        if digest(pixels) != legacy_receipt["source_pixels_sha256"]:
            raise RuntimeError("source preprocessing changed")
        cached_images, entropy_cache = legacy_receiver_cache(arguments.population, index, directory, packed, old_rows)
        with np.load(destination / "payloads.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        records = json.loads((destination / "phy.json").read_text())["records"]
        fresh, fresh_keys, local_rows, pixel_cache = [], [], [], {}
        for record in records:
            physical = physical_state(record, arrays)
            state_key = payload_key(physical)
            result = None
            image = None
            if record["family"] == "raw":
                prefix = raw_prefix(physical)
                result = {"prefix": prefix, "source_complete": physical["label"] is not None,
                          "arithmetic_padding_reads": 0, "source_error": "" if physical["label"] is not None else "header_failure"}
            elif state_key in entropy_cache:
                result = entropy_cache[state_key]
            else:
                result = decode_source(physical, vae, var, device)
                image = result["image"]
                entropy_cache[state_key] = result
            prefix = result["prefix"]
            key = "header_erasure" if physical["label"] is None else prefix_key(prefix, physical["label"])
            if key not in cached_images:
                if image is None:
                    image = complete_image(vae, var, prefix, physical["label"], device)
                cached_images[key] = {"image_archive": str(destination / "reconstructions.npz"), "image_slot": len(fresh)}
                fresh.append(image)
                fresh_keys.append(key)
            correct, accepted_correct, ber = compare_source(prefix, physical["label"], physical["mode"], source, target,
                record["mode"], result["source_complete"], bool(record["header_fields_correct"] and record["payload_bits_correct"]), record["body_crc_accepted"])
            method = f"{record['family']}_m{record['mode']}_N{record['complex_uses']}" if arguments.population == "calibration" else f"{record['family']}_adaptive_N{record['complex_uses']}"
            local_rows.append({**record, "method": method, "protocol": "common_paid_information", "prefix_sha256": key,
                               "source_correct": int(correct), "accepted_correct": int(accepted_correct), "source_bit_error_rate": ber,
                               "candidate_prefix_scales": len(prefix), "source_complete": int(result["source_complete"]),
                               "arithmetic_padding_reads": result["arithmetic_padding_reads"], "source_error": result["source_error"]})
        normalized = pixels.astype(np.float32) / 255
        if fresh:
            scores, _features, _image_features = quality_metrics(normalized, fresh, perceptual, dino, device)
            np.savez(destination / "reconstructions.npz", images=np.stack(fresh))
            for key, scores_row, image in zip(fresh_keys, scores, fresh):
                cached_images[key].update(psnr_db=scores_row["psnr_db"], lpips=scores_row["lpips_alex"], dino=scores_row["dino_cosine"])
                pixel_cache[key] = image
        source_tensor = torch.from_numpy(normalized)[None].to(device)
        for row in local_rows:
            key = row["prefix_sha256"]
            cached = cached_images[key]
            if "ssim" not in cached:
                if key not in pixel_cache:
                    with np.load(cached["image_archive"], allow_pickle=False) as archive:
                        pixel_cache[key] = archive["images"][cached["image_slot"]].copy()
                image = pixel_cache[key]
                cached["ssim"] = float(ssim(torch.from_numpy(image)[None].to(device), source_tensor, data_range=1., size_average=False)[0])
                cached["image_sha256"] = digest(image)
                actual_psnr = float(-10 * np.log10(np.mean((image.astype(np.float64) - normalized) ** 2)))
                if abs(actual_psnr - cached["psnr_db"]) > 1e-4:
                    raise RuntimeError("cached image no longer matches its actual source quality")
            row.update({name: cached[name] for name in ("image_archive", "image_slot", "psnr_db", "lpips", "dino", "ssim", "image_sha256")})
            row["metric_origin"] = "identical_observable_RX_argument_memoization" if key not in fresh_keys else "new_actual_RX_render"
            row["all_failures_retained"] = True
        write_csv(destination / "per_frame.csv", local_rows)
        write_json(destination / "render_completion.json", {"rows": len(local_rows), "new_images": len(fresh),
                   "cached_images": len(set(row["prefix_sha256"] for row in local_rows)) - len(fresh),
                   "per_frame_sha256": sha256(destination / "per_frame.csv"), "source_pixels_sha256": digest(pixels)})
        all_rows.extend(local_rows)
        fresh_count += len(fresh)
        reused_count += len(local_rows) - sum(row["prefix_sha256"] in fresh_keys for row in local_rows)
        write_json(output / "status.json", {"status": "COMMON_BUDGET_RENDER_AND_SCORE", "population": arguments.population,
                   "sources_completed": position + 1, "rows": len(all_rows), "new_images": fresh_count,
                   "pid": os.getpid(), "seconds": time.perf_counter() - started, "updated_at": datetime.now().astimezone().isoformat()})
        if (position + 1) % 10 == 0:
            print(f"{arguments.population} RGB: {position + 1}/100; rows {len(all_rows)}; new images {fresh_count}", flush=True)
    after = {name: state_sha256(model) for name, model in models.items()}
    if before != after or len(all_rows) != physical_receipt["transmissions"]:
        raise RuntimeError("frozen model changed or transmissions disappeared")
    write_csv(output / "per_frame.csv", all_rows)
    write_json(output / "completion.json", {"status": "COMMON_BUDGET_DIGITAL_COMPLETE", "population": arguments.population,
               "rows": len(all_rows), "source_images": 100, "frozen_models": before,
               "per_frame_sha256": sha256(output / "per_frame.csv"), "seconds_this_render_session": time.perf_counter() - started,
               "new_images_this_session": fresh_count, "memoized_frame_rows_this_session": reused_count,
               "no_cached_quality_timing_claim": True, "new_training_or_holdout": False,
               "policy_sha256": metadata["policy_sha256"], "finished_at": datetime.now().astimezone().isoformat()})


def fit(arguments):
    receipt = json.loads((arguments.output / "completion.json").read_text())
    if receipt["population"] != "calibration" or receipt["status"] != "COMMON_BUDGET_DIGITAL_COMPLETE":
        raise RuntimeError("only complete calibration can determine a mode policy")
    if arguments.policies is None or arguments.policies.exists():
        raise ValueError("supply a fresh immutable policy output")
    rows = read_csv(arguments.output / "per_frame.csv")
    original = json.loads((PROJECT / "configs/communication_decision_study.json").read_text())
    original["snrs_db"] = [1., 4., 7., 13., 19.]
    budgets = {}
    candidates = []
    for budget in (4204, 4498):
        selected = [row for row in rows if int(row["complex_uses"]) == budget]
        summary = summarize_candidates(selected)
        actions, family, primary = fit_actions(summary, original)
        budgets[str(budget)] = {"actions": actions, "calibration_global_family": family, "primary_LPIPS": primary}
        candidates.extend({"complex_uses": budget, **row} for row in summary)
    write_csv(arguments.output / "candidate_summary.csv", candidates)
    write_json(arguments.policies, {"status": "COMMON_BUDGET_CALIBRATION_POLICIES_FROZEN", "budgets": budgets,
               "calibration_sources": indices_for("calibration"), "calibration_per_frame_sha256": sha256(arguments.output / "per_frame.csv"),
               "calibration_completion_sha256": sha256(arguments.output / "completion.json"), "DINO_used_for_selection": False,
               "development_accessed_for_selection": False, "frozen_at": datetime.now().astimezone().isoformat()})
    print(json.dumps(budgets, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("phy", "render", "fit"), required=True)
    parser.add_argument("--population", choices=("calibration", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", type=Path)
    args = parser.parse_args()
    {"phy": prepare_phy, "render": render, "fit": fit}[args.stage](args)
