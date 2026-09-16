#!/usr/bin/env python3
"""Run the registered receiver-only whole-frame finite-prefix prototype."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import complete_image, prefix_key, receive_whole, split_prefix
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import bits_to_indices, channel_evidence, crc_accepts, indices_to_bits, load_native, rate_match_indices
from var_comm.study import artifact_hashes, create_output, seeded_noise, sha256, snapshot, verify_artifacts, verify_snapshot, write_csv, write_json
from var_comm.whole_frame import assisted_receivers, cache_empty, public_assisted_record, public_native_record, source_tables
from var_comm.whole_list import load_list_native, ordered_paths
from var_comm.whole_study import METRICS, PROFILE_FIELDS, aggregate


def completed_receiver(original, information):
    result = deepcopy(original)
    if information is not None:
        if len(information) != 5110 or not crc_accepts(information[:-6]) or information[-6:].any():
            raise RuntimeError("new accepted whole-frame candidate violates received CRC/tail")
        result["prefix"] = split_prefix(bits_to_indices(information[:5088]), 9)
        result["last_accepted_scale"] = 9
        result["events"] = [{"accepted": True, "decoded_bits": information.tolist(), "new_candidate": True}]
    return result


def serial_choice(listed):
    selected = np.flatnonzero(listed["accepted"])
    return None if not len(selected) else listed["bits"][int(selected[0])]


def profile_records(records, base_seconds, extra_seconds, gpu_bytes, prior_seconds=0.0, prior_count=0):
    profile = {field: 0 for field in PROFILE_FIELDS}
    profile.update(receiver_seconds=base_seconds + extra_seconds, auxiliary_seconds=extra_seconds, prior_seconds=prior_seconds,
                   prior_prefix_count=prior_count, prior_batch_calls=int(prior_count > 0), gpu_peak_allocated_bytes=gpu_bytes)
    for record in records:
        for name in ("channel_products", "backward_edges", "deviation_edges", "heap_comparisons", "traceback_layers"):
            profile[name] += record[name]
        profile["native_owned_peak_bytes"] = max(profile["native_owned_peak_bytes"], record["native_owned_peak_bytes"])
    return profile


def run_study(config, output, metadata, receipt):
    parent = ROOT / config["input_run"]
    upstream = yaml.safe_load((parent / "snapshots/configs/progressive_channel.yaml").read_text())
    model_config = yaml.safe_load((ROOT / upstream["model_config"]).read_text())
    for paths, names in ((model_config["paths"], ("vae_checkpoint", "var_checkpoint")), (upstream["quality"], ("alexnet_checkpoint", "dino_checkpoint"))):
        for name in names:
            if sha256(paths[name]) != paths[name + "_sha256"]:
                raise RuntimeError("frozen checkpoint changed")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    vae, var = load_models(model_config["paths"], device)
    with mock.patch("torch.hub.download_url_to_file", side_effect=RuntimeError("weight downloads forbidden")):
        perceptual, dino, _weights = load_quality_models(upstream["quality"], device)
    models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
    before = {name: state_sha256(model) for name, model in models.items()}
    if before != receipt["frozen_before"]:
        raise RuntimeError("models differ from the frozen baseline")
    load_native()
    load_list_native()
    with np.load(ROOT / "outputs/VAR-WHOLE-PRIOR-SELFCHECK-001/hypotheses.npz", allow_pickle=False) as cache:
        source_tables(vae, var, cache["bits"], int(cache["label"]), device)
    ordered_paths(np.zeros(10220), 1)
    prior_run = ROOT / upstream["prior_run"]
    prior_record = json.loads((prior_run / "completion.json").read_text())
    if sha256(prior_run / "completion.json") != upstream["prior_completion_sha256"] or sha256(prior_run / "source_tokens.npz") != prior_record["output_hashes"]["source_tokens.npz"]:
        raise RuntimeError("diagnostic source tokens changed")
    with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
        token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
    targets = json.loads((parent / "populations.json").read_text())["target"]
    if len(targets) != 100 or any(row["role"] != "target_development" for row in targets):
        raise RuntimeError("development population changed")
    with (parent / "per_frame.csv").open() as handle:
        historical = list(csv.DictReader(handle))
    old_lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in historical}
    rows, new_renders, legacy_replays, triggered, total_prior_queries = [], 0, 0, 0, 0
    mapping = rate_match_indices(10220, 5984)
    for image_index, target in enumerate(targets):
        original_directory = parent / "images" / f"{image_index:03d}"
        image_directory = output / "images" / f"{image_index:03d}"
        image_directory.mkdir(parents=True)
        original_records = json.loads((original_directory / "receivers.json").read_text())
        original_record_lookup = {(record["snr_db"], record["seed"], record["arm"]): record for record in original_records}
        with np.load(original_directory / "waveforms.npz", allow_pickle=False) as cache:
            transmissions = {mode: cache[f"whole_m{mode}_0"].copy() for mode in (8, 9)}
            for mode in (8, 9):
                if any(not np.array_equal(cache[f"whole_m{mode}_{position}"], transmissions[mode]) for position in range(5)):
                    raise RuntimeError("whole-mode transmitter unexpectedly changes with SNR")
        with np.load(original_directory / "reconstructions.npz", allow_pickle=False) as cache:
            source_image, old_images = cache["source"].copy(), cache["images"].copy()
        cache_map = {row["reconstruction_key"]: {"image_ref": "input:" + row["reconstruction_index"], **{metric: float(row[metric]) for metric in METRICS}}
                     for row in historical if int(row["image_index"]) == image_index}
        arrays, frame_records, local_rows, images, image_states = {}, [], [], [], []
        source_tokens = token_lookup[target["image_id"]]
        true_prefix_bits = indices_to_bits(source_tokens[:255])
        for snr_index, snr in enumerate(config["snr_db"]):
            for seed_index, seed in enumerate(config["noise_seeds"]):
                key = f"frame_{snr_index}_{seed_index}"
                noise = seeded_noise(target["image_id"], seed, (3060, 2)) / np.sqrt(10 ** (snr / 10))
                originals, base_times, hashes = {}, {}, {}
                for mode in (8, 9):
                    received = transmissions[mode] + noise
                    tick = time.perf_counter()
                    originals[mode] = receive_whole(received, snr)
                    base_times[mode] = time.perf_counter() - tick
                    hashes[mode] = hashlib.sha256(received.tobytes()).hexdigest()
                    if snr in (4.0, 7.0):
                        previous = original_record_lookup[snr, seed, f"whole_m{mode}"]
                        current = originals[mode]
                        if hashes[mode] != previous["received_sha256"] or current["header"] != previous["header"] or current["events"] != previous["events"] or [scale.tolist() for scale in current["prefix"]] != previous["output_prefix"]:
                            raise RuntimeError("old whole-frame waveform or ML result changed")
                        legacy_replays += 1
                original = originals[9]
                baseline_gpu = torch.cuda.memory_allocated(device)
                names = ("whole_m9_list65", "whole_m9_list_time", "whole_m9_hyp_uniform", "whole_m9_var")
                receivers = {"whole_m8": originals[8], "whole_m9_ml": original, **{name: deepcopy(original) for name in names}}
                profiles = {name: profile_records([], base_times[8] if name == "whole_m8" else base_times[9], 0.0, baseline_gpu) for name in receivers}
                new_accepted = {name: 0 for name in receivers}
                trigger = bool(original["header"]["accepted"] and original["header"]["mode"] == 9 and not original["events"][0]["accepted"])
                details, prefix_covered, prefix_rank, var_over_time, capacity_limited = {}, "", "", 0, 0
                if trigger:
                    triggered += 1
                    evidence = channel_evidence((transmissions[9] + noise)[68:], mapping, 5110, snr)
                    assisted = assisted_receivers(evidence, original["label"], vae, var, device, config["search"])
                    total_prior_queries += assisted["prior_prefix_count"]
                    details["assisted"] = public_assisted_record(assisted, arrays, key)
                    for name, family in (("whole_m9_var", "VAR"), ("whole_m9_hyp_uniform", "uniform")):
                        result = assisted[family]
                        receivers[name] = completed_receiver(original, result["information"])
                        new_accepted[name] = int(result["information"] is not None)
                        native_stats = [assisted["prefixes"]["stats"], *[branch["list"]["stats"] for branch in result["branches"]]]
                        profiles[name] = profile_records(native_stats, base_times[9], assisted[family + "_seconds"],
                                                        assisted["peak_gpu_allocated_bytes"] if family == "VAR" else baseline_gpu,
                                                        assisted["prior_seconds"] if family == "VAR" else 0.0,
                                                        assisted["prior_prefix_count"] if family == "VAR" else 0)
                        profiles[name]["candidates_checked"] = sum(len(branch["list"]["bits"]) for branch in result["branches"])
                    for name, maximum, seconds in (("whole_m9_list65", config["search"]["count_control_candidates"], -1),
                                                    ("whole_m9_list_time", config["search"]["time_control_max_candidates"], config["search"]["time_control_seconds"])):
                        tick = time.perf_counter()
                        listed = ordered_paths(evidence, maximum, payload_bits=5088, stop_on_crc=True, seconds=seconds, queue_limit=config["search"]["maximum_heap_nodes"])
                        elapsed = time.perf_counter() - tick
                        if not len(listed["bits"]) or not np.array_equal(listed["bits"][0], original["events"][0]["decoded_bits"]):
                            raise RuntimeError("list rank zero differs from original ML")
                        information = serial_choice(listed)
                        receivers[name] = completed_receiver(original, information)
                        new_accepted[name] = int(information is not None)
                        profiles[name] = profile_records([listed["stats"]], base_times[9], elapsed, baseline_gpu)
                        profiles[name]["candidates_checked"] = len(listed["bits"])
                        details[name] = public_native_record(listed, arrays, key + "_" + name)
                        if name == "whole_m9_list_time":
                            capacity_limited = int(information is None and listed["stats"]["stop_reason"] in (0, 4))
                    var_over_time = int(assisted["VAR_seconds"] > config["search"]["time_control_seconds"])
                    matching = [index for index, bits in enumerate(assisted["prefixes"]["bits"]) if np.array_equal(bits, true_prefix_bits)]
                    prefix_covered = int(bool(matching))
                    prefix_rank = matching[0] if matching else -1
                mode = config["adaptive_modes"][str(snr)]
                for name, source_name in (("whole_adaptive", "whole_m8" if mode == 8 else "whole_m9_ml"),
                                           ("whole_adaptive_var", "whole_m8" if mode == 8 else "whole_m9_var")):
                    receivers[name], profiles[name], new_accepted[name] = receivers[source_name], dict(profiles[source_name]), new_accepted[source_name]
                baseline_key = "header_erasure" if original["label"] is None else prefix_key(original["prefix"], original["label"])
                public_receivers = {}
                for name in config["arms"]:
                    result = receivers[name]
                    expected_mode = mode if name.startswith("whole_adaptive") else (8 if name == "whole_m8" else 9)
                    image_key = "header_erasure" if result["label"] is None else prefix_key(result["prefix"], result["label"])
                    if image_key not in cache_map:
                        tick = time.perf_counter()
                        pixels = complete_image(vae, var, result["prefix"], result["label"], device)
                        cache_map[image_key] = {"image_ref": "new:" + str(len(images))}
                        images.append(pixels)
                        image_states.append({"key": image_key, "index": len(images) - 1, "prefix": [scale.tolist() for scale in result["prefix"]],
                                             "label": result["label"], "completion_seconds": time.perf_counter() - tick})
                    received_tokens = np.concatenate(result["prefix"]) if result["prefix"] else np.empty(0, dtype=int)
                    reference = source_tokens[:255 if expected_mode == 8 else 424]
                    correct = result["label"] == target["class_index"] and result["header"]["mode"] == expected_mode and np.array_equal(received_tokens, reference)
                    accepted = bool(result["events"] and result["events"][0]["accepted"])
                    header_false = result["header"]["accepted"] and (result["header"]["label"] != target["class_index"] or result["header"]["mode"] != expected_mode)
                    uses_aux = expected_mode == 9 and name not in ("whole_m9_ml", "whole_adaptive")
                    row = {"image_index": image_index, "image_id": target["image_id"], "snr_db": snr, "seed": seed, "arm": name,
                           "mode": expected_mode, "complex_uses": 3060, "received_sha256": hashes[expected_mode],
                           "source_bler": int(not correct), "correct_accepted": int(accepted and correct),
                           "correct_m9_accepted": int(expected_mode == 9 and accepted and correct), "undetected_error": int(accepted and not correct),
                           "header_false_acceptance": int(header_false), "aux_triggered": int(trigger and uses_aux),
                           "new_candidate_accepted": new_accepted[name], "new_false_acceptance": int(new_accepted[name] and not correct),
                           "image_condition_changed": int(uses_aux and image_key != baseline_key),
                           "true_prefix_covered": prefix_covered if trigger and name in ("whole_m9_var", "whole_m9_hyp_uniform", "whole_adaptive_var") and expected_mode == 9 else "",
                           "true_prefix_rank": prefix_rank if trigger and name == "whole_m9_var" else "",
                           "VAR_above_time_allowance": var_over_time if name == "whole_m9_var" else 0,
                           "time_control_capacity_limited": capacity_limited if name in ("whole_m9_var", "whole_m9_list_time") else 0,
                           "reconstruction_key": image_key, **cache_map[image_key], **profiles[name]}
                    local_rows.append(row)
                    public_receivers[name] = {"header": result["header"], "label": result["label"], "last_accepted_scale": result["last_accepted_scale"],
                                              "output_prefix": [scale.tolist() for scale in result["prefix"]], "accepted": accepted,
                                              "image_key": image_key, "image_ref": cache_map[image_key]["image_ref"]}
                frame_records.append({"snr_db": snr, "seed": seed, "trigger": trigger, "receivers": public_receivers, "details": details})
        if images:
            quality, _source_embedding, embeddings = quality_metrics(source_image, images, perceptual, dino, device)
            for state, scores in zip(image_states, quality):
                cache_map[state["key"]].update(scores)
        else:
            embeddings = np.empty((0, 384), dtype=np.float32)
        for row in local_rows:
            row.update({metric: cache_map[row["reconstruction_key"]][metric] for metric in METRICS})
        clean = old_lookup[image_index, 19.0, config["noise_seeds"][0], "whole_m9"]
        correct_record = original_record_lookup[19.0, config["noise_seeds"][0], "whole_m9"]
        check = complete_image(vae, var, [np.asarray(scale) for scale in correct_record["output_prefix"]], correct_record["label"], device)
        if not np.array_equal(check, old_images[int(clean["reconstruction_index"])]) or not cache_empty(var):
            raise RuntimeError("hypothesis branch changed clean m9 image output or cached state")
        write_json(image_directory / "receivers.json", frame_records)
        write_json(image_directory / "image_states.json", image_states)
        np.savez(image_directory / "candidates.npz", **arrays)
        np.savez(image_directory / "reconstructions.npz", images=np.stack(images) if images else np.empty((0, 3, 256, 256), dtype=np.float32), reconstruction_dino=embeddings)
        rows.extend(local_rows)
        new_renders += len(images)
        write_csv(output / "per_frame.csv", rows)
        print(f"whole-frame {image_index + 1}/100 triggers={triggered} rows={len(rows)} new_renders={new_renders}", flush=True)
    summary, comparisons, changes, decision = aggregate(rows, config)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_quality.csv", comparisons)
    write_csv(output / "changed_frames.csv", changes)
    write_json(output / "decision.json", decision)
    after = {name: state_sha256(model) for name, model in models.items()}
    if before != after or any(parameter.requires_grad or parameter.grad is not None for model in models.values() for parameter in model.parameters()):
        raise RuntimeError("model freeze boundary violated")
    verify_snapshot(metadata["source_hashes"])
    verify_artifacts(parent, "completion.json", config["input_completion_sha256"])
    return {"status": decision["status"], "decision": decision, "rows": len(rows), "triggered_transmissions": triggered,
            "prior_prefix_queries": total_prior_queries, "unique_new_reconstructions": new_renders,
            "old_4_and_7dB_ML_exact_replays": legacy_replays, "clean_m9_pixel_exact_replays": 100,
            "frozen_before": before, "frozen_after": after, "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/whole_frame_prior.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    requested = (args.output_dir or ROOT / config["outputs"]["full"]).resolve()
    if not requested.is_relative_to(ROOT / "outputs"):
        raise ValueError("new output must stay under VAR_COMM/outputs")
    if requested.exists():
        raise FileExistsError(requested)
    if config["status"] != "preregistered_before_new_target_channel_results":
        raise RuntimeError("unregistered configuration")
    parent = ROOT / config["input_run"]
    receipt = verify_artifacts(parent, "completion.json", config["input_completion_sha256"])
    verify_snapshot(receipt["source_hashes"])
    if sha256(ROOT / config["input_audit"]) != config["input_audit_sha256"]:
        raise RuntimeError("baseline audit changed")
    for name, status in (("list_selfcheck", "WHOLE_LIST_SELFCHECK_PASS"), ("neural_selfcheck", "WHOLE_PRIOR_SELFCHECK_PASS")):
        path = ROOT / config[name]
        checked = verify_artifacts(path.parent, path.name)
        if checked["status"] != status:
            raise RuntimeError("selfcheck did not pass")
        verify_snapshot(checked["source_hashes"])
    output = create_output(requested)
    paths = [Path(__file__), args.config, ROOT / "reports/whole_frame_prior_preregistration_2026-09-07.md",
             *sorted((ROOT / "src/var_comm").glob("*.py")), *sorted((ROOT / "src/var_comm").glob("*.cpp"))]
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv,
                "input_completion_sha256": config["input_completion_sha256"], "source_hashes": snapshot(output, paths),
                "selfcheck_sha256": {name: sha256(ROOT / config[name]) for name in ("list_selfcheck", "neural_selfcheck")},
                "torch": torch.__version__, "numpy": np.__version__}
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        result = run_study(config, output, metadata, receipt)
        write_json(output / "completion.json", {**metadata, **result, "elapsed_seconds": time.perf_counter() - started,
                                                "output_hashes": artifact_hashes(output)})
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
