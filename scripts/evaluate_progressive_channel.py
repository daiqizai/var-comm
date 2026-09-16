#!/usr/bin/env python3
"""Run calibration then the complete fixed-budget matched/strong-control study."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import load_models, preprocess, state_sha256
from var_comm.progressive import (PriorCache, allocation, complete_image, prefix_key, receive_group, receive_whole,
                                  split_prefix, transmit_group, transmit_whole)
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import load_native
from var_comm.study import (artifact_hashes, create_output, paired_interval, seeded_noise, sha256, snapshot,
                            verify_artifacts, verify_snapshot, write_csv, write_json)

IMAGE_METRICS = ("psnr_db", "lpips_alex", "dino_cosine")


def receiver_metrics(result, source, true_label, grouped, expected_mode=9):
    prefix = result["prefix"]
    label_correct = result["label"] == true_label
    prefix_correct = all(np.array_equal(scale, source[index]) for index, scale in enumerate(prefix))
    trusted = result["last_accepted_scale"]
    header_false = result["header"]["accepted"] and (result["header"]["label"] != true_label or result["header"]["mode"] != expected_mode)
    false_blocks = 0
    for event in result["events"]:
        if not event["accepted"]:
            continue
        if grouped:
            group = event["group"]
            reference = np.concatenate(source[:6]) if group == 0 else source[group + 5]
            false_blocks += int(not np.array_equal(np.asarray(event["recovered_tokens"]), reference))
        else:
            false_blocks += int(not prefix_correct)
    count = sum(len(scale) for scale in prefix[:trusted]) if prefix_correct and label_correct and not header_false else 0
    return {"header_accepted": int(result["header"]["accepted"]), "header_false_acceptance": int(header_false),
            "source_crc_false_acceptance_count": false_blocks, "last_accepted_scale": trusted,
            "correct_accepted_tokens": count, "correct_m9_accepted": int(trusted == 9 and prefix_correct and label_correct and not header_false),
            "output_prefix_scales": len(prefix), "output_prefix_correct": int(prefix_correct and label_correct)}


def calibrate(samples, token_lookup, config, priors, static, output, smoke):
    rows, choices = [], {}
    seeds = config["calibration"]["noise_seeds"][:1] if smoke else config["calibration"]["noise_seeds"]
    encoded_cache = {}
    for sample_index, sample in enumerate(samples):
        source = split_prefix(token_lookup[sample["image_id"]], 10)
        label = sample["class_index"]
        for snr in config["snr_db"]:
            for family in config["calibration"]["independent_selection_per_snr_and_family"]:
                for base in config["calibration"]["base_uses_candidates"]:
                    waveform, _ledger = transmit_group(source, label, base, priors, family == "group_entropy", encoded_cache)
                    for seed in seeds:
                        received = waveform + seeded_noise(sample["image_id"], seed, waveform.shape) / np.sqrt(10 ** (snr / 10))
                        result = receive_group(received, snr, base, family, priors, static)
                        metrics = receiver_metrics(result, source, label, True)
                        rows.append({"image_id": sample["image_id"], "family": family, "snr_db": snr, "seed": seed,
                                     "base_uses": base, "correct_accepted_tokens": metrics["correct_accepted_tokens"]})
        print(f"resource calibration {sample_index + 1}/{len(samples)}", flush=True)
    for family in config["calibration"]["independent_selection_per_snr_and_family"]:
        choices[family] = {}
        for snr in config["snr_db"]:
            candidates = []
            for base in config["calibration"]["base_uses_candidates"]:
                values = [row["correct_accepted_tokens"] for row in rows if row["family"] == family and row["snr_db"] == snr and row["base_uses"] == base]
                candidates.append((float(np.mean(values)), -base))
            choices[family][str(snr)] = -max(candidates)[1]
    write_csv(output / "calibration.csv", rows)
    write_json(output / "selected_policies.json", {"local_frozen": datetime.now().astimezone().isoformat(),
                                                  "choices": choices, "target_results_observed": False})
    return choices


def aggregate(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["snr_db"], row["arm"])].append(row)
    summary = []
    for (snr, arm), group in sorted(groups.items()):
        metrics = (*IMAGE_METRICS, "correct_accepted_tokens", "correct_m9_accepted", "last_accepted_scale", "header_accepted", "receiver_seconds")
        summary.append({"snr_db": snr, "arm": arm, "frames": len(group),
                        **{metric: float(np.mean([row[metric] for row in group])) for metric in metrics},
                        "false_acceptances": sum(row["header_false_acceptance"] + row["source_crc_false_acceptance_count"] for row in group)})
    comparisons = []
    image_ids = sorted({row["image_id"] for row in rows})
    controls = [config["gate"]["mechanism_control"], *config["gate"]["system_controls"]]
    for selected in [[snr] for snr in config["snr_db"]] + [config["gate"]["primary_snr_db"]]:
        for control in controls:
            for metric in IMAGE_METRICS:
                values = defaultdict(list)
                for row in rows:
                    if row["snr_db"] in selected and row["arm"] in ("group_var", control):
                        values[(row["image_id"], row["arm"])].append(row[metric])
                differences = [np.mean(values[(identifier, "group_var")]) - np.mean(values[(identifier, control)]) for identifier in image_ids]
                interval = paired_interval(differences, config["bootstrap"]["seed"], config["bootstrap"]["resamples"])
                comparisons.append({"snr_db": "+".join(str(snr) for snr in selected), "control": control,
                                    "metric": metric, "VAR_minus_control": interval.pop("gain"), **interval,
                                    "control_mean": float(np.mean([np.mean(values[(identifier, control)]) for identifier in image_ids]))})
    verdicts = {}
    primary_label = "+".join(str(snr) for snr in config["gate"]["primary_snr_db"])
    for control in controls:
        compared = {row["metric"]: row for row in comparisons if row["snr_db"] == primary_label and row["control"] == control}
        lpips = compared["lpips_alex"]
        checks = {"lpips_relative_gain": -lpips["VAR_minus_control"] >= config["gate"]["minimum_relative_lpips_reduction"] * lpips["control_mean"],
                  "lpips_interval": lpips["ci_high"] < 0,
                  "psnr_guard": compared["psnr_db"]["VAR_minus_control"] >= -config["gate"]["maximum_psnr_drop_db"],
                  "dino_guard": compared["dino_cosine"]["VAR_minus_control"] >= -config["gate"]["maximum_dino_drop"],
                  "observed_false_acceptances": sum(row["header_false_acceptance"] + row["source_crc_false_acceptance_count"] for row in rows if row["arm"] == "group_var") <= config["gate"]["maximum_observed_VAR_false_acceptances"]}
        verdicts[control] = {"passed": all(checks.values()), "checks": checks}
    return summary, comparisons, verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/progressive_channel.yaml")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["status"] != "preregistered_before_group_calibration_and_target_results":
        raise ValueError("progressive contract changed")
    prior_run = ROOT / config["prior_run"]
    prior_receipt = verify_artifacts(prior_run, "completion.json", config["prior_completion_sha256"])
    previous = verify_artifacts(ROOT / config["single_scale_run"], "completion.json", config["single_scale_completion_sha256"])
    if previous["status"] != "PASS_SINGLE_SCALE_GATE_ONLY" or sha256(ROOT / config["single_scale_audit"]) != config["single_scale_audit_sha256"]:
        raise RuntimeError("single-scale gate/audit boundary not met")
    selfcheck = json.loads((ROOT / config["protocol_selfcheck"]).read_text())
    if selfcheck["status"] != "PROGRESSIVE_SELFCHECK_PASS":
        raise RuntimeError("protocol selfcheck failed")
    verify_snapshot(selfcheck["source_hashes"])
    output = create_output(args.output_dir or ROOT / config["outputs"][args.mode])
    (output / "images").mkdir()
    paths = [Path(__file__), args.config, ROOT / "reports/progressive_channel_preregistration_2026-09-07.md",
             *sorted((ROOT / "src/var_comm").glob("*.py")), ROOT / "src/var_comm/token_trellis.cpp"]
    records = snapshot(output, paths)
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv, "mode": args.mode,
                "source_hashes": records, "single_scale_completion_sha256": config["single_scale_completion_sha256"],
                "single_scale_audit_sha256": config["single_scale_audit_sha256"], "python": sys.version,
                "torch": torch.__version__, "numpy": np.__version__}
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        model_config = yaml.safe_load((ROOT / config["model_config"]).read_text())
        for name in ("vae_checkpoint", "var_checkpoint"):
            if sha256(model_config["paths"][name]) != model_config["paths"][name + "_sha256"]:
                raise RuntimeError("model checkpoint changed")
        for name in ("dino_checkpoint", "alexnet_checkpoint"):
            if sha256(config["quality"][name]) != config["quality"][name + "_sha256"]:
                raise RuntimeError("quality checkpoint changed")
        for original, record in prior_receipt["source_files"].items():
            if sha256(original) != record["sha256"]:
                raise RuntimeError("upstream prior source changed")
        vae, var = load_models(model_config["paths"], device)
        perceptual, dino, linear_weights = load_quality_models(config["quality"], device)
        models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
        before = {name: state_sha256(model) for name, model in models.items()}
        write_json(output / "model_integrity_before.json", {"state_sha256": before, "lpips_linear_sha256": sha256(linear_weights),
            "dino_sources": {str(path): sha256(path) for path in sorted(Path(config["quality"]["dino_source"]).rglob("*.py"))}})
        priors = PriorCache(vae, var, device, output / "probabilities")
        manifest = json.loads((prior_run / "population_manifest.json").read_text())
        targets = [row for row in manifest if row["role"] == "target_development"]
        fitting = [row for row in manifest if row["role"] == "frequency_fit"]
        selected_indices = ((np.arange(config["population"]["calibration_count"]) + 0.5) * len(fitting) / config["population"]["calibration_count"]).astype(int)
        calibration = [fitting[index] for index in selected_indices]
        with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
            token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
        static = {}
        for scale, start, stop in ((7, 91, 155), (8, 155, 255), (9, 255, 424)):
            counts = np.bincount(np.concatenate([token_lookup[row["image_id"]][start:stop] for row in fitting]), minlength=4096).astype(np.float64) + 0.5
            static[scale] = np.log(counts / counts.sum())
        seeds = config["noise_seeds"]
        if args.mode == "smoke":
            targets, calibration, seeds = targets[:2], calibration[:2], seeds[:1]
        if {row["image_id"] for row in targets} & {row["image_id"] for row in calibration}:
            raise RuntimeError("resource calibration overlaps target population")
        write_json(output / "populations.json", {"target": targets, "resource_calibration": calibration})
        load_native()
        choices = calibrate(calibration, token_lookup, config, priors, static, output, args.mode == "smoke")
        policy_sha = sha256(output / "selected_policies.json")
        rows, ledgers, consistency_count = [], [], 0
        for image_index, target in enumerate(targets):
            source = split_prefix(token_lookup[target["image_id"]], 10)
            label = target["class_index"]
            original, rgb_sha = preprocess(Path(target["path"]))
            if rgb_sha != target["preprocessed_rgb_sha256"] or sha256(target["path"]) != target["file_sha256"]:
                raise RuntimeError("source image identity changed")
            source_image = original.add(1).mul(0.5).numpy()
            image_directory = output / "images" / f"{image_index:03d}"
            image_directory.mkdir()
            reconstruction_keys, reconstructed, frame_records, waveforms = {}, [], [], {}
            local_rows, encoded_cache = [], {}
            for snr_index, snr in enumerate(config["snr_db"]):
                raw_base = choices["group_var"][str(snr)]
                entropy_base = choices["group_entropy"][str(snr)]
                raw_signal, raw_ledger = transmit_group(source, label, raw_base, priors)
                entropy_signal, entropy_ledger = transmit_group(source, label, entropy_base, priors, True, encoded_cache)
                for family, signal, ledger in (("group_raw", raw_signal, raw_ledger), ("group_entropy", entropy_signal, entropy_ledger)):
                    waveforms[f"{family}_{snr_index}"] = signal
                    ledgers.append({"image_index": image_index, "snr_db": snr, "family": family, **ledger})
                for mode in (7, 8, 9):
                    waveforms[f"whole_m{mode}_{snr_index}"] = transmit_whole(source, label, mode)
                for seed in seeds:
                    noise = seeded_noise(target["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                    results, receiver_times = {}, {}
                    for arm in config["arms"]:
                        if arm == "whole_adaptive":
                            chosen = f"whole_m{config['adaptive_modes'][str(snr)]}"
                            results[arm], receiver_times[arm] = results[chosen], receiver_times[chosen]
                            continue
                        tick = time.perf_counter()
                        if arm.startswith("group_"):
                            family = "group_entropy" if arm == "group_entropy" else "group_raw"
                            received = waveforms[f"{family}_{snr_index}"] + noise
                            result = receive_group(received, snr, entropy_base if arm == "group_entropy" else raw_base, arm, priors, static)
                        else:
                            received = waveforms[f"{arm}_{snr_index}"] + noise
                            result = receive_whole(received, snr)
                        receiver_times[arm] = time.perf_counter() - tick
                        result["received_sha256"] = hashlib.sha256(received.tobytes()).hexdigest()
                        results[arm] = result
                    raw_hashes = {results[arm]["received_sha256"] for arm in ("group_ml", "group_static", "group_var")}
                    if len(raw_hashes) != 1:
                        raise RuntimeError("matched grouped receivers saw different observations")
                    correct_output_keys = []
                    for arm in config["arms"]:
                        result = results[arm]
                        key = "header_erasure" if result["label"] is None else prefix_key(result["prefix"], result["label"])
                        if key not in reconstruction_keys:
                            reconstruction_keys[key] = len(reconstructed)
                            reconstructed.append(complete_image(vae, var, result["prefix"], result["label"], device))
                        expected_mode = 9 if arm.startswith("group_") else (config["adaptive_modes"][str(snr)] if arm == "whole_adaptive" else int(arm[-1]))
                        metrics = receiver_metrics(result, source, label, arm.startswith("group_"), expected_mode)
                        if len(result["prefix"]) == 9 and metrics["output_prefix_correct"]:
                            correct_output_keys.append(key)
                        row = {"image_index": image_index, "image_id": target["image_id"], "snr_db": snr, "seed": seed,
                               "arm": arm, "complex_uses": 3060, "received_sha256": result["received_sha256"],
                               "reconstruction_index": reconstruction_keys[key], "reconstruction_key": key,
                               "receiver_seconds": receiver_times[arm], **metrics}
                        local_rows.append(row)
                        frame_records.append({"snr_db": snr, "seed": seed, "arm": arm, "reconstruction_index": reconstruction_keys[key],
                            **{name: value for name, value in result.items() if name != "prefix"},
                            "output_prefix": [scale.tolist() for scale in result["prefix"]]})
                    if correct_output_keys:
                        if len(set(correct_output_keys)) != 1:
                            raise RuntimeError("correct equal prefixes led to different output conditions")
                        consistency_count += len(correct_output_keys)
            quality, source_embedding, query_embeddings = quality_metrics(source_image, reconstructed, perceptual, dino, device)
            for row in local_rows:
                row.update(quality[row["reconstruction_index"]])
            rows.extend(local_rows)
            np.savez(image_directory / "reconstructions.npz", images=np.stack(reconstructed), source=source_image,
                     source_dino=source_embedding, reconstruction_dino=query_embeddings)
            np.savez(image_directory / "waveforms.npz", **waveforms)
            write_json(image_directory / "receivers.json", frame_records)
            write_csv(output / "per_frame.csv", rows)
            print(f"progressive {image_index + 1}/{len(targets)} rows={len(rows)} unique_images={len(reconstructed)}", flush=True)
        expected = len(targets) * len(config["snr_db"]) * len(seeds) * len(config["arms"])
        if len(rows) != expected or sha256(output / "selected_policies.json") != policy_sha:
            raise RuntimeError("incomplete grid or target-driven policy modification")
        summary, comparisons, gates = aggregate(rows, config)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired_quality.csv", comparisons)
        write_json(output / "wire_ledgers.json", ledgers)
        after = {name: state_sha256(model) for name, model in models.items()}
        if after != before or any(parameter.requires_grad or parameter.grad is not None for model in models.values() for parameter in model.parameters()):
            raise RuntimeError("model freeze boundary violated")
        verify_snapshot(records)
        mechanism = gates[config["gate"]["mechanism_control"]]["passed"]
        system = mechanism and all(gates[name]["passed"] for name in config["gate"]["system_controls"])
        status = "PASS_DEVELOPMENT_SYSTEM_GATE" if system else ("MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN" if mechanism else "STOP_THIS_GROUPED_CONFIGURATION")
        if args.mode == "smoke":
            status = "SMOKE_COMPLETE_NOT_SCIENTIFIC_RESULT"
        result = {**metadata, "status": status, "rows": len(rows), "target_count": len(targets), "gate": gates,
                  "selected_policies_sha256": policy_sha, "choices": choices,
                  "frozen_before": before, "frozen_after": after, "correct_m9_identical_output_cases": consistency_count,
                  "prior_requests": priors.requests, "unique_prior_tables": len(priors.tables), "prior_compute_seconds": priors.compute_seconds,
                  "elapsed_seconds": time.perf_counter() - started, "output_hashes": artifact_hashes(output)}
        write_json(output / "completion.json", result)
        print(json.dumps({name: value for name, value in result.items() if name not in ("source_hashes", "output_hashes")}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_CLOSED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
