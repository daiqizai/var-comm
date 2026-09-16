#!/usr/bin/env python3
"""Audit frozen whole-frame candidates with independent encoding, CRC and neural scores."""

from __future__ import annotations

import argparse
import binascii
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lpips
import numpy as np
import torch
import yaml

from audit_progressive_channel import context_key, divide_prefix, number, score_images, token_bits, token_indices
from audit_single_scale_channel import require
from check_whole_prior import teacher_scores
from var_comm.next_scale_prior import load_models, next_scale_log_probs, state_sha256
from var_comm.progressive import receive_whole
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, verify_snapshot, write_json

METRICS = ("psnr_db", "lpips_alex", "dino_cosine")


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def independent_crc(bits):
    length = len(bits) // 8 * 8
    value = binascii.crc_hqx(np.packbits(bits[:length]).tobytes(), 65535)
    for bit in bits[length:]:
        high = (value >> 15) ^ int(bit)
        value = (value << 1) & 65535
        if high:
            value ^= 0x1021
    return value


def mother_code(candidates):
    values = np.asarray(candidates, dtype=np.uint8)
    outputs = np.zeros((*values.shape, 2), dtype=np.uint8)
    for output, generator in enumerate((0o171, 0o133)):
        for delay in range(7):
            if (generator >> delay) & 1:
                outputs[:, delay:, output] ^= values[:, :values.shape[1] - delay]
    return outputs


def channel_costs(candidates, evidence):
    costs = []
    for start in range(0, len(candidates), 64):
        coded = mother_code(candidates[start:start + 64]).reshape(-1, len(evidence))
        costs.extend(-((1.0 - 2.0 * coded.astype(np.float64)) @ evidence))
    return np.asarray(costs)


def unpack(record, arrays):
    packed = arrays[record["array_key"]]
    bits = np.unpackbits(packed, axis=1)[:, :record["information_bits"]]
    require(len(bits) == record["candidate_count"] > 0, "missing candidate paths")
    require(len({row.tobytes() for row in packed}) == len(bits), "duplicate list candidates")
    require(np.isfinite(record["rank_scores"]).all() and np.all(np.diff(record["rank_scores"]) >= -1e-7), "unordered path list")
    require(record["stats"]["candidates"] == len(bits) and record["stats"]["native_seconds"] >= record["stats"]["setup_seconds"], "invalid native accounting")
    return bits


def check_crc_list(record, full_candidates):
    flags = [independent_crc(bits[:5088]) == number(bits[5088:5104]) for bits in full_candidates]
    require(flags == record["accepted"] and not np.asarray(full_candidates)[:, -6:].any(), "candidate CRC/tail changed")
    if any(flags):
        require(flags[-1] and not any(flags[:-1]) and record["stats"]["stop_reason"] == 1, "search did not stop on first accepted candidate")
    return None if not any(flags) else full_candidates[-1]


def independent_prefix_scores(evidence):
    states = np.arange(64)
    registers = (states[:, None] << 1) | np.arange(2)[None]
    signs = np.array([[[1 - 2 * ((int(value) & generator).bit_count() % 2) for generator in (0o171, 0o133)]
                       for value in row] for row in registers])
    next_states = registers & 63
    costs = np.full(64, np.inf)
    costs[0] = 0
    for step in range(5109, 3059, -1):
        branches = -(signs * evidence[2 * step:2 * step + 2]).sum(axis=2)
        costs = np.min(branches + costs[next_states], axis=1)
    previous_zero, previous_one = states >> 1, (states >> 1) | 32
    last_bits = states & 1
    paths = np.full((64, 4), np.inf)
    paths[0, 0] = 0
    for step in range(3060):
        branches = -(signs * evidence[2 * step:2 * step + 2]).sum(axis=2)
        options = np.concatenate((paths[previous_zero] + branches[previous_zero, last_bits, None],
                                  paths[previous_one] + branches[previous_one, last_bits, None]), axis=1)
        paths = np.sort(options, axis=1)[:, :4]
    return np.sort((paths + costs[:, None]).ravel())[:4]


def audit(run, output, config, receipt):
    parent = ROOT / config["input_run"]
    previous = verify_artifacts(parent, "completion.json", config["input_completion_sha256"])
    verify_snapshot(previous["source_hashes"])
    upstream = yaml.safe_load((parent / "snapshots/configs/progressive_channel.yaml").read_text())
    model_config = yaml.safe_load((ROOT / upstream["model_config"]).read_text())
    targets = json.loads((parent / "populations.json").read_text())["target"]
    rows = read_csv(run / "per_frame.csv")
    lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in rows}
    expected = {(index, snr, seed, arm) for index in range(100) for snr in config["snr_db"] for seed in config["noise_seeds"] for arm in config["arms"]}
    require(len(rows) == len(lookup) == 9600 and set(lookup) == expected, "incomplete receiver grid")
    with np.load(ROOT / upstream["prior_run"] / "source_tokens.npz", allow_pickle=False) as cache:
        token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.hub.set_dir(str(Path(upstream["quality"]["alexnet_checkpoint"]).parent.parent))
    with mock.patch("torch.hub.download_url_to_file", side_effect=RuntimeError("downloads forbidden")):
        perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
    dino = torch.hub.load(upstream["quality"]["dino_source"], "dinov2_vits14", source="local", pretrained=False)
    dino.load_state_dict(torch.load(upstream["quality"]["dino_checkpoint"], map_location="cpu", weights_only=True), strict=True)
    dino = dino.to(device).eval().requires_grad_(False)
    vae, var = load_models(model_config["paths"], device)
    models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
    require({name: state_sha256(model) for name, model in models.items()} == receipt["frozen_before"], "independent model fingerprint mismatch")
    sys.path.insert(0, "/workspace/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/src")
    from cadsd_jscc.var_prefix_consistency import complete_received_prefix

    maximum_metrics = np.zeros(3)
    max_source_error = max_channel_error = max_pixel_error = 0.0
    candidate_count = prefix_count = reconstructed_count = image_count = triggers = exact_prefix_checks = 0
    scored = {}
    mapping = np.floor((np.arange(5984) + 0.5) * 10220 / 5984).astype(int)
    for image_index, target in enumerate(targets):
        original_directory = parent / "images" / f"{image_index:03d}"
        directory = run / "images" / f"{image_index:03d}"
        records = json.loads((directory / "receivers.json").read_text())
        require(len(records) == 12, "missing channel records")
        states = json.loads((directory / "image_states.json").read_text())
        with np.load(directory / "candidates.npz", allow_pickle=False) as cache:
            arrays = {name: cache[name] for name in cache.files}
        with np.load(directory / "reconstructions.npz", allow_pickle=False) as cache:
            new_images = cache["images"].copy()
        with np.load(original_directory / "reconstructions.npz", allow_pickle=False) as cache:
            original, old_images = cache["source"].copy(), cache["images"].copy()
        with np.load(original_directory / "waveforms.npz", allow_pickle=False) as cache:
            waves = {mode: cache[f"whole_m{mode}_0"].copy() for mode in (8, 9)}
        old_records = json.loads((original_directory / "receivers.json").read_text())
        ref_keys = {"input:" + str(record["reconstruction_index"]): ("header_erasure" if record["label"] is None else context_key(record["output_prefix"], record["label"])) for record in old_records}
        ref_keys.update({"new:" + str(state["index"]): state["key"] for state in states})
        local_rows = [row for row in rows if int(row["image_index"]) == image_index]
        references = sorted({row["image_ref"] for row in local_rows})
        images = np.stack([(old_images if reference.startswith("input:") else new_images)[int(reference.split(":")[1])] for reference in references])
        metric_values = score_images(original, images, perceptual, dino, device)
        image_count += len(images)
        for position, reference in enumerate(references):
            scored[image_index, reference] = {metric: float(values[position]) for metric, values in zip(METRICS, metric_values)}
        for row in local_rows:
            require(ref_keys[row["image_ref"]] == row["reconstruction_key"], "image reference does not belong to decoded prefix")
            maximum_metrics = np.maximum(maximum_metrics, [abs(scored[image_index, row["image_ref"]][metric] - float(row[metric])) for metric in METRICS])
        for state in states:
            require(state["key"] == context_key(state["prefix"], state["label"]), "new image has an invalid prefix key")
            with torch.no_grad():
                prefix = [torch.tensor(scale, dtype=torch.long, device=device)[None] for scale in state["prefix"]]
                latent = complete_received_prefix(var, vae, prefix, torch.tensor([state["label"]], device=device))
                pixels = vae.fhat_to_img(latent).clamp(-1, 1).add(1).mul(0.5)[0].cpu().numpy()
            max_pixel_error = max(max_pixel_error, float(np.max(np.abs(pixels - new_images[state["index"]]))))
            reconstructed_count += 1
        truth = token_lookup[target["image_id"]]
        for record in records:
            snr, seed = record["snr_db"], record["seed"]
            decoded = record["receivers"]
            digest = hashlib.sha256(f"{target['image_id']}|{seed}".encode()).digest()
            noise = np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal((3060, 2)) / np.sqrt(10 ** (snr / 10))
            observations = {mode: waves[mode] + noise for mode in (8, 9)}
            for mode, name in ((8, "whole_m8"), (9, "whole_m9_ml")):
                replayed = receive_whole(observations[mode], snr)
                require(replayed["header"] == decoded[name]["header"] and replayed["label"] == decoded[name]["label"], "original header/ML receiver changed")
                require([scale.tolist() for scale in replayed["prefix"]] == decoded[name]["output_prefix"], "original ML payload changed")
                accepted = bool(replayed["events"] and replayed["events"][0]["accepted"])
                require(accepted == decoded[name]["accepted"], "original ML CRC decision changed")
                if replayed["events"]:
                    bits = np.asarray(replayed["events"][0]["decoded_bits"], dtype=np.uint8)
                    require((independent_crc(bits[:-22]) == number(bits[-22:-6])) == accepted, "independent baseline source CRC differs")
            for arm, receiver in decoded.items():
                row = lookup[image_index, snr, seed, arm]
                mode = int(row["mode"])
                require(int(row["complex_uses"]) == 3060 and hashlib.sha256(observations[mode].tobytes()).hexdigest() == row["received_sha256"], "mode, budget or actual waveform changed")
                header = receiver["header"]
                bits = np.asarray(header["decoded_bits"], dtype=np.uint8)
                valid = independent_crc(bits[:12]) == number(bits[12:28]) and number(bits[:10]) < 1000 and number(bits[10:12]) + 6 in (7, 8, 9)
                require(valid == header["accepted"] and not bits[-6:].any(), "header verification mismatch")
                require(number(bits[:10]) == header["label"] and number(bits[10:12]) + 6 == header["mode"], "decoded header fields were replaced")
                require(receiver["label"] == (header["label"] if valid else None), "oracle label supplied")
                flat = np.concatenate(receiver["output_prefix"]) if receiver["output_prefix"] else []
                correct = receiver["label"] == target["class_index"] and header["mode"] == mode and np.array_equal(flat, truth[:255 if mode == 8 else 424])
                require(int(row["source_bler"]) == int(not correct) and int(row["correct_accepted"]) == int(correct and receiver["accepted"]), "source correctness changed")
                require(int(row["undetected_error"]) == int(receiver["accepted"] and not correct), "false acceptance omitted")
                require(int(row["new_false_acceptance"]) == int(int(row["new_candidate_accepted"]) and not correct), "new CRC false acceptance omitted")
                image_key = "header_erasure" if receiver["label"] is None else context_key(receiver["output_prefix"], receiver["label"])
                require(image_key == receiver["image_key"] == row["reconstruction_key"], "image selection used another state")
            old = decoded["whole_m9_ml"]
            trigger = old["header"]["accepted"] and old["header"]["mode"] == 9 and not old["accepted"]
            require(trigger == record["trigger"], "assistance ran outside failed m9 CRC boundary")
            selected = {name: None for name in ("whole_m9_list65", "whole_m9_list_time", "whole_m9_var", "whole_m9_hyp_uniform")}
            if trigger:
                triggers += 1
                evidence = np.bincount(mapping, weights=observations[9][68:].ravel(), minlength=10220) * (10 ** (snr / 10))
                details = record["details"]
                full_lists = {}
                for name in ("whole_m9_list65", "whole_m9_list_time"):
                    listed = details[name]
                    candidates = unpack(listed, arrays)
                    require(len(candidates) <= (65 if name.endswith("65") else 4096), "ordinary list exceeded registered cap")
                    costs = channel_costs(candidates, evidence)
                    max_channel_error = max(max_channel_error, float(np.max(np.abs(costs - listed["local_scores"]))))
                    selected[name] = check_crc_list(listed, candidates)
                    full_lists[name] = candidates
                    candidate_count += len(candidates)
                    require(np.array_equal(token_indices(candidates[0][:5088]), np.concatenate(old["output_prefix"])), "list did not start at ML")
                overlap = min(len(value) for value in full_lists.values())
                require(np.array_equal(full_lists["whole_m9_list65"][:overlap], full_lists["whole_m9_list_time"][:overlap]), "ordinary budget controls do not share an ordered list")
                assisted = details["assisted"]
                prefixes = unpack(assisted["prefixes"], arrays)
                require(len(prefixes) == 4 and prefixes.shape[1] == 3060, "wrong distinct prefix budget")
                prefix_costs = channel_costs(prefixes, evidence[:6120])
                max_channel_error = max(max_channel_error, float(np.max(np.abs(prefix_costs - assisted["prefixes"]["local_scores"]))))
                if image_index < 4 and seed == config["noise_seeds"][0]:
                    expected_prefix_scores = independent_prefix_scores(evidence)
                    require(np.max(np.abs(expected_prefix_scores - assisted["prefixes"]["rank_scores"])) < 1e-7, "independent top-four prefix dynamic program differs")
                    exact_prefix_checks += 1
                prefix_values = [divide_prefix(token_indices(bits), 8) for bits in prefixes]
                tensors = [torch.tensor(np.stack([prefix[index] for prefix in prefix_values]), dtype=torch.long, device=device) for index in range(8)]
                label = old["label"]
                tables = next_scale_log_probs(var, vae, tensors, torch.full((4,), label, dtype=torch.long, device=device)).cpu().numpy()
                require([hashlib.sha256(table.tobytes()).hexdigest() for table in tables] == assisted["q9_sha256"], "independent q9 does not reproduce receiver tables")
                _teacher_table, teacher_prefix = teacher_scores(vae, var, prefixes, label, device)
                source_error = float(np.max(np.abs(teacher_prefix - assisted["prefix_log_probs"])))
                max_source_error = max(max_source_error, source_error)
                require(source_error < 0.005, "full prefix source probability is incomplete")
                require([context_key(prefix, label) for prefix in prefix_values] == assisted["prefix_keys"], "source cache used a different prefix or class")
                prefix_count += len(prefixes)
                for family, name in (("VAR", "whole_m9_var"), ("uniform", "whole_m9_hyp_uniform")):
                    choices, completed = [], []
                    for branch in assisted[family]["branches"]:
                        index = branch["prefix_rank"]
                        listed = branch["list"]
                        suffixes = unpack(listed, arrays)
                        require(len(suffixes) <= 16 and suffixes.shape[1] == 2050, "suffix budget or CRC/tail length changed")
                        require(branch["initial_state"] == number(prefixes[index][-6:]) and branch["prefix_crc_state"] == independent_crc(prefixes[index]), "suffix encoder/CRC state was reset")
                        full = np.concatenate((np.broadcast_to(prefixes[index], (len(suffixes), 3060)), suffixes), axis=1)
                        candidate_count += len(full)
                        costs = channel_costs(full, evidence)
                        selected_bits = check_crc_list(listed, full)
                        suffix_nll = np.zeros(len(full))
                        prefix_nll = 0.0
                        if family == "VAR":
                            prefix_nll = -float(np.sum(assisted["prefix_log_probs"][index]))
                            for position, bits in enumerate(suffixes):
                                values = token_indices(bits[:2028])
                                suffix_nll[position] = -float(tables[index, np.arange(169), values].astype(np.float64).sum())
                        joint = costs + suffix_nll + prefix_nll
                        max_channel_error = max(max_channel_error, float(np.max(np.abs(costs - prefix_costs[index] + suffix_nll - listed["local_scores"]))),
                                                float(np.max(np.abs(joint - branch["joint_scores"]))))
                        require(abs(prefix_nll - branch["prefix_source_nll"]) < 1e-8, "prefix prior omitted or repeated")
                        if selected_bits is not None:
                            choices.append((float(branch["joint_scores"][-1]), index, len(full) - 1))
                        completed.append(full)
                    expected_choice = min(choices) if choices else None
                    require((None if expected_choice is None else list(expected_choice)) == assisted[family]["selected"], "candidate chosen by a nonregistered criterion")
                    selected[name] = None if expected_choice is None else completed[expected_choice[1]][expected_choice[2]]
                matching = [index for index, bits in enumerate(prefixes) if np.array_equal(bits, token_bits(truth[:255]))]
                row = lookup[image_index, snr, seed, "whole_m9_var"]
                require(int(row["true_prefix_covered"]) == int(bool(matching)) and int(row["true_prefix_rank"]) == (matching[0] if matching else -1), "diagnostic true-prefix coverage changed")
                require(int(row["VAR_above_time_allowance"]) == int(assisted["VAR_seconds"] > 0.5), "time allowance caveat hidden")
            else:
                require(not record["details"], "unused assistance leaked source information")
            for name, information in selected.items():
                receiver = decoded[name]
                row = lookup[image_index, snr, seed, name]
                if information is None:
                    require(receiver == old, "failed assistance did not preserve the ML output")
                else:
                    require(receiver["output_prefix"] == divide_prefix(token_indices(information[:5088]), 9) and receiver["accepted"], "output does not use the selected whole CRC candidate")
                require(int(row["new_candidate_accepted"]) == int(information is not None), "new acceptance count mismatch")
            mode = config["adaptive_modes"][str(snr)]
            require(decoded["whole_adaptive"] == decoded["whole_m8" if mode == 8 else "whole_m9_ml"], "old adaptive threshold changed")
            require(decoded["whole_adaptive_var"] == decoded["whole_m8" if mode == 8 else "whole_m9_var"], "new receiver changed the mode strategy")
        if (image_index + 1) % 10 == 0:
            print(f"whole audit {image_index + 1}/100 paths={candidate_count} prefixes={prefix_count} images={image_count}", flush=True)
    require(max_channel_error < 2e-7 and maximum_metrics.max() < 2e-5 and max_pixel_error == 0.0, "waveform, metric or legacy image mismatch")
    for record in read_csv(run / "summary.csv"):
        selected = [row for row in rows if row["arm"] == record["arm"] and float(row["snr_db"]) == float(record["snr_db"])]
        require(len(selected) == 300, "summary excluded failures")
        for metric in METRICS:
            mean = np.mean([scored[int(row["image_index"]), row["image_ref"]][metric] for row in selected])
            require(abs(mean - float(record[metric])) < 2e-5, "summary differs from independently rescored images")
    draws = np.random.default_rng(config["bootstrap"]["seed"]).integers(100, size=(10000, 100))
    interval_error = 0.0
    for record in read_csv(run / "paired_quality.csv"):
        interval = [float(value) for value in record["snr_db"].split("+")]
        differences = []
        for image_index in range(100):
            values = []
            for snr in interval:
                for seed in config["noise_seeds"]:
                    method = lookup[image_index, snr, seed, record["method"]]
                    control = lookup[image_index, snr, seed, record["control"]]
                    values.append(scored[image_index, method["image_ref"]][record["metric"]] - scored[image_index, control["image_ref"]][record["metric"]])
            differences.append(np.mean(values))
        differences = np.asarray(differences)
        low, high = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
        interval_error = max(interval_error, abs(differences.mean() - float(record["delta"])), abs(low - float(record["ci_low"])), abs(high - float(record["ci_high"])))
    require(interval_error < 2e-5, "source-image paired intervals mismatch")
    require({name: state_sha256(model) for name, model in models.items()} == receipt["frozen_before"] == receipt["frozen_after"], "model freeze failed")
    return {"status": "AUDIT_PASS", "receiver_rows_checked": len(rows), "triggered_transmissions": triggers,
            "full_candidate_paths_checked": candidate_count, "hypothesis_tables_recomputed": prefix_count,
            "independent_top_four_prefix_checks": exact_prefix_checks, "max_channel_and_joint_score_error": max_channel_error,
            "max_teacher_prefix_logprob_error": max_source_error, "unique_images_neurally_rescored": image_count,
            "new_images_replayed_with_legacy_renderer": reconstructed_count, "legacy_max_pixel_error": max_pixel_error,
            "max_metric_errors_PSNR_LPIPS_DINO": maximum_metrics.tolist(), "max_paired_interval_error": interval_error,
            "scope": "all candidate feasibility/CRC/source scores checked; exhaustive ordering tests plus independent real prefix DP; timed search lengths not replayed as deterministic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/VAR-WHOLE-FRAME-PRIOR-001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-WHOLE-FRAME-PRIOR-AUDIT-001")
    args = parser.parse_args()
    receipt = verify_artifacts(args.run_dir, "completion.json")
    verify_snapshot(receipt["source_hashes"])
    output = create_output(args.output_dir)
    sources = snapshot(output, [Path(__file__), ROOT / "scripts/check_whole_prior.py", ROOT / "scripts/audit_progressive_channel.py", ROOT / "scripts/audit_single_scale_channel.py"])
    config = yaml.safe_load((args.run_dir / "snapshots/configs/whole_frame_prior.yaml").read_text())
    started = time.perf_counter()
    try:
        result = audit(args.run_dir, output, config, receipt)
        verify_snapshot(sources)
        write_json(output / "audit.json", {**result, "local_completed": datetime.now().astimezone().isoformat(),
                                           "run_completion_sha256": sha256(args.run_dir / "completion.json"), "source_hashes": sources,
                                           "elapsed_seconds": time.perf_counter() - started, "output_hashes": artifact_hashes(output)})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
