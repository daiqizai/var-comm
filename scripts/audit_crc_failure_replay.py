#!/usr/bin/env python3
"""Independently audit frozen candidate provenance, renders, and paired statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import csv
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

from audit_progressive_channel import audit_receiver, context_key, divide_prefix, score_images, token_bits
from audit_single_scale_channel import require
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, verify_snapshot, write_json

METRICS = ("psnr_db", "lpips_alex", "dino_cosine")


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def independent_branch(record):
    prefix = [list(scale) for scale in record["output_prefix"]]
    if record["label"] is None:
        return prefix, "header_failure", False
    failed = [event for event in record["events"] if not event["accepted"]]
    if not failed:
        return prefix, "m9_accepted", False
    require(len(failed) == 1 and failed[0] is record["events"][-1], "not the first and terminal failure")
    event = failed[0]
    group = event["group"]
    stage = "B0_failure" if group == 0 else f"B{group + 6}_failure"
    candidate = event.get("recovered_tokens", [])
    expected_length = (91, 64, 100, 169)[group]
    valid = len(candidate) == expected_length and all(type(token) is int and 0 <= token < 4096 for token in candidate)
    if not valid:
        return prefix, stage, False
    retained = divide_prefix(candidate, 6) if group == 0 else prefix + [list(candidate)]
    return retained, stage, True


def audit(run, output, config, receipt):
    input_run = ROOT / config["input_run"]
    parent = verify_artifacts(input_run, "completion.json", config["input_completion_sha256"])
    verify_snapshot(parent["source_hashes"])
    upstream = yaml.safe_load((input_run / "snapshots/configs/progressive_channel.yaml").read_text())
    model_config = yaml.safe_load((ROOT / upstream["model_config"]).read_text())
    populations = json.loads((input_run / "populations.json").read_text())
    targets = populations["target"]
    require(len(targets) == config["target_count"] == 100, "changed development population")
    old_rows = read_csv(input_run / "per_frame.csv")
    old_lookup = {(int(row["image_index"]), int(row["seed"]), row["arm"]): row for row in old_rows if float(row["snr_db"]) == 7}
    rows = read_csv(run / "per_frame.csv")
    lookup = {(int(row["image_index"]), int(row["seed"]), row["arm"]): row for row in rows}
    arms = [family + "_" + policy for family in config["grouped_arms"] for policy in config["output_policies"]] + ["whole_m9"]
    expected = {(index, seed, arm) for index in range(100) for seed in config["noise_seeds"] for arm in arms}
    require(len(rows) == len(lookup) == 3000 and set(lookup) == expected, "incomplete paired population")
    require(all(float(row["snr_db"]) == 7 and int(row["complex_uses"]) == 3060 for row in rows), "changed SNR or budget")
    prior_run = ROOT / upstream["prior_run"]
    require(sha256(prior_run / "completion.json") == upstream["prior_completion_sha256"], "prior provenance changed")
    prior_receipt = json.loads((prior_run / "completion.json").read_text())
    require(sha256(prior_run / "source_tokens.npz") == prior_receipt["output_hashes"]["source_tokens.npz"], "source tokens changed")
    with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
        tokens = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
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
    require({name: state_sha256(model) for name, model in models.items()} == parent["frozen_before"], "independently loaded models differ")
    legacy_source = Path("/workspace/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc/src")
    sys.path.insert(0, str(legacy_source))
    from cadsd_jscc.var_prefix_consistency import complete_received_prefix

    score_lookup, table_cache, maximum_errors = {}, {}, np.zeros(3)
    trace_count, neural_count, pixel_count, pixel_error, unchanged_count = 0, 0, 0, 0.0, 0
    ledger_rows = json.loads((input_run / "wire_ledgers.json").read_text())
    ledger_lookup = {(row["image_index"], row["snr_db"], row["family"]): row for row in ledger_rows}
    for image_index, target in enumerate(targets):
        original_directory = input_run / "images" / f"{image_index:03d}"
        replay_directory = run / "images" / f"{image_index:03d}"
        records = [record for record in json.loads((original_directory / "receivers.json").read_text())
                   if record["snr_db"] == 7 and record["arm"] in config["grouped_arms"] + ["whole_m9"]]
        saved_states = json.loads((replay_directory / "replay_states.json").read_text())
        states = {(state["seed"], state["family"]): state for state in saved_states}
        require(len(saved_states) == len(states) == 9, "missing or duplicated render branch state")
        source = divide_prefix(tokens[target["image_id"]], 10)
        with np.load(original_directory / "reconstructions.npz", allow_pickle=False) as cache:
            old_images, original = cache["images"].copy(), cache["source"].copy()
        with np.load(replay_directory / "retained.npz", allow_pickle=False) as cache:
            new_images = cache["images"].copy()
        with np.load(original_directory / "waveforms.npz", allow_pickle=False) as cache:
            waves = {name: cache[name].copy() for name in ("group_raw_2", "group_entropy_2", "whole_m9_2")}
        local_rows = [row for row in rows if int(row["image_index"]) == image_index]
        references = sorted({row["image_ref"] for row in local_rows})
        selected_images = []
        for reference in references:
            kind, position = reference.split(":")
            require(kind in ("input", "new"), "invalid reconstruction source")
            selected_images.append((old_images if kind == "input" else new_images)[int(position)])
        selected_images = np.stack(selected_images)
        require(np.isfinite(selected_images).all() and selected_images.min() >= 0 and selected_images.max() <= 1, "invalid image pixels")
        scores = score_images(original, selected_images, perceptual, dino, device)
        neural_count += len(selected_images)
        for position, reference in enumerate(references):
            score_lookup[image_index, reference] = {metric: float(score[position]) for metric, score in zip(METRICS, scores)}
        for row in local_rows:
            independently_scored = score_lookup[image_index, row["image_ref"]]
            maximum_errors = np.maximum(maximum_errors, [abs(float(row[metric]) - independently_scored[metric]) for metric in METRICS])
        new_states = json.loads((replay_directory / "image_states.json").read_text())
        require(len(new_states) == len(new_images), "unaccounted new reconstruction")
        for state in new_states:
            require(state["key"] == context_key(state["render_prefix"], state["label"]), "new render key depends on hidden state")
            require(any(branch["render_prefix"] == state["render_prefix"] and branch["label"] == state["label"] for branch in saved_states), "new render has no allowed receive candidate")
            with torch.no_grad():
                prefix = [torch.tensor(scale, device=device, dtype=torch.long)[None] for scale in state["render_prefix"]]
                latent = complete_received_prefix(var, vae, prefix, torch.tensor([state["label"]], device=device))
                pixels = vae.fhat_to_img(latent).clamp(-1, 1).add(1).mul(0.5)[0].cpu().numpy()
            pixel_error = max(pixel_error, float(np.max(np.abs(pixels - new_images[state["image_index"]]))))
            pixel_count += 1
            require(all(not block.attn.caching and block.attn.cached_k is None and block.attn.cached_v is None for block in var.blocks), "legacy reconstruction left a live cache")
        for record in records:
            family, seed = record["arm"], record["seed"]
            old = old_lookup[image_index, seed, family]
            wave_family = "group_entropy" if family == "group_entropy" else ("group_raw" if family.startswith("group_") else family)
            seed_digest = hashlib.sha256(f"{target['image_id']}|{seed}".encode()).digest()
            seed_integer = int.from_bytes(seed_digest[:8], "big")
            received = waves[wave_family + "_2"] + np.random.default_rng(seed_integer).standard_normal((3060, 2)) / np.sqrt(10 ** 0.7)
            require(hashlib.sha256(received.tobytes()).hexdigest() == record["received_sha256"] == old["received_sha256"], "replay changed the received waveform")
            uses = None
            if family.startswith("group_"):
                ledger = ledger_lookup[image_index, 7.0, wave_family]
                uses = [792, 413, 640, 1075] if family == "group_entropy" else [720, 441, 683, 1148]
                require(ledger["block_uses"] == uses and sum(uses) + ledger["header_uses"] == 3060, "resource allocation changed")
                for event in record["events"]:
                    if "position" in event:
                        fragment = received[event["position"]:event["position"] + event["complex_uses"]]
                        require(hashlib.sha256(fragment.tobytes()).hexdigest() == event["received_sha256"], "wrong frozen scale observation")
            audit_receiver(record, old, source, target["class_index"], 9, input_run, table_cache, uses)
            trace_count += 1
            if family == "whole_m9":
                row = lookup[image_index, seed, "whole_m9"]
                require(row["image_ref"] == "input:" + str(record["reconstruction_index"]), "whole-frame output was weakened")
                require(all(float(row[metric]) == float(old[metric]) for metric in METRICS), "whole-frame scores changed")
                continue
            retained, stage, available = independent_branch(record)
            state = states[seed, family]
            expected_digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            require(state["frozen_record_sha256"] == expected_digest, "trusted receive trace was mutated")
            require(state["render_prefix"] == retained and state["trusted_prefix"] == record["output_prefix"], "render and trusted prefixes were not isolated")
            require(state["label"] == record["label"] and state["candidate_available"] == available and state["failure_stage"] == stage, "candidate availability used nonreceiver information")
            discard_row = lookup[image_index, seed, family + "_discard"]
            retain_row = lookup[image_index, seed, family + "_retain"]
            oracle_row = lookup[image_index, seed, family + "_oracle_lpips"]
            require(all(float(discard_row[metric]) == float(old[metric]) for metric in METRICS), "discard baseline changed")
            require(discard_row["image_ref"] == "input:" + str(record["reconstruction_index"]), "discard baseline image changed")
            expected_key = "header_erasure" if record["label"] is None else context_key(retained, record["label"])
            require(retain_row["reconstruction_key"] == expected_key and int(retain_row["render_scales"]) == len(retained), "retain image came from another candidate")
            for row in (discard_row, retain_row, oracle_row):
                require(row["received_sha256"] == old["received_sha256"] and int(row["trusted_scales"]) == record["last_accepted_scale"], "output policy changed receive information or trust")
                require(int(row["correct_m9_accepted"]) == int(old["correct_m9_accepted"]), "retained candidate was promoted to reliable")
                require(row["failure_stage"] == stage and int(row["candidate_available"]) == available, "failure strata changed")
                if available:
                    group = record["events"][-1]["group"]
                    truth = np.concatenate(source[:6]) if group == 0 else source[group + 5]
                    ber = float(np.mean(token_bits(truth) != token_bits(record["events"][-1]["recovered_tokens"])))
                    require(float(row["failed_candidate_true_ber"]) == ber, "post-hoc BER mismatch")
                else:
                    require(row["failed_candidate_true_ber"] == "", "BER exists without a candidate")
            chosen = retain_row if float(retain_row["lpips_alex"]) < float(discard_row["lpips_alex"]) else discard_row
            require(oracle_row["image_ref"] == chosen["image_ref"] and oracle_row["oracle_selected_policy"] == chosen["policy"], "oracle did not choose the lower-LPIPS allowed image")
            require(all(float(oracle_row[metric]) == float(chosen[metric]) for metric in METRICS) and int(oracle_row["deployable"]) == 0, "oracle used separate metric choices or was called deployable")
            if not available:
                require(discard_row["image_ref"] == retain_row["image_ref"], "a no-candidate output changed")
                unchanged_count += 1
        if (image_index + 1) % 20 == 0:
            print(f"replay audit {image_index + 1}/100 traces={trace_count} rerenders={pixel_count}", flush=True)
    require(maximum_errors.max() < 2e-5 and pixel_error == 0.0, "independent metric or renderer mismatch")
    means = {}
    for arm in arms:
        for metric in METRICS:
            means[arm, metric] = np.array([np.mean([score_lookup[index, lookup[index, seed, arm]["image_ref"]][metric]
                                                   for seed in config["noise_seeds"]]) for index in range(100)])
    for row in read_csv(run / "summary.csv"):
        require(int(row["transmissions"]) == 300 and int(row["source_images"]) == 100, "summary conditions on success")
        for metric in METRICS:
            require(abs(float(row[metric]) - means[row["arm"], metric].mean()) < 2e-5, "overall summary mismatch")
    draws = np.random.default_rng(config["bootstrap"]["seed"]).integers(100, size=(10000, 100))
    max_interval_error = 0.0
    for row in read_csv(run / "paired_quality.csv"):
        metric = row["metric"]
        if row["method"] == "VAR_retention_change":
            differences = (means["group_var_retain", metric] - means["group_var_discard", metric]
                           - means["group_ml_retain", metric] + means["group_ml_discard", metric])
        else:
            differences = means[row["method"], metric] - means[row["control"], metric]
        low, high = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
        max_interval_error = max(max_interval_error, abs(differences.mean() - float(row["delta"])),
                                 abs(low - float(row["ci_low"])), abs(high - float(row["ci_high"])))
    require(max_interval_error < 2e-5, "source-image paired intervals mismatch")
    for stratum in read_csv(run / "failure_strata.csv"):
        family, stage = stratum["family"], stratum["failure_stage"]
        selected = [row for row in rows if row["family"] == family and row["policy"] == "discard" and row["failure_stage"] == stage]
        require(len(selected) == int(stratum["transmissions"]), "failure samples were omitted")
        require(sum(int(row["candidate_available"]) for row in selected) == int(stratum["available_candidates"]), "failure candidate count mismatch")
        for metric in METRICS:
            differences, contributions = [], np.zeros(100)
            for row in selected:
                index, seed = int(row["image_index"]), int(row["seed"])
                other = lookup[index, seed, family + "_retain"]
                difference = score_lookup[index, other["image_ref"]][metric] - score_lookup[index, row["image_ref"]][metric]
                differences.append(difference)
                contributions[index] += difference / 3
            if selected:
                require(abs(float(stratum["delta_" + metric]) - np.mean(differences)) < 2e-5, "stratified mean mismatch")
            low, high = np.quantile(contributions[draws].mean(axis=1), [0.025, 0.975])
            require(max(abs(float(stratum["overall_contribution_" + metric]) - contributions.mean()),
                        abs(float(stratum["contribution_ci_low_" + metric]) - low),
                        abs(float(stratum["contribution_ci_high_" + metric]) - high)) < 2e-5, "stratum contribution bootstrap mismatch")
    decision = json.loads((run / "decision.json").read_text())
    retained_gap = means["group_var_retain", "lpips_alex"].mean() - means["whole_m9", "lpips_alex"].mean()
    oracle_gap = means["group_var_oracle_lpips", "lpips_alex"].mean() - means["whole_m9", "lpips_alex"].mean()
    status = ("STOP_TWO_CANDIDATE_SELECTOR_FIXED_M9" if oracle_gap > 0 else "ORACLE_HEADROOM_ONLY_NO_DEPLOYABLE_GAIN"
              if retained_gap > 0 else "RETAIN_MEAN_GAP_CLOSED_CHECK_FAIR_CONTROLS")
    require(status == decision["status"] == receipt["status"], "pre-registered stop decision mismatch")
    require({name: state_sha256(model) for name, model in models.items()} == receipt["frozen_before"] == receipt["frozen_after"], "model freeze failed")
    return {"status": "AUDIT_PASS", "decision": status, "receiver_traces_checked": trace_count,
            "new_images_replayed_with_legacy_renderer": pixel_count, "legacy_max_pixel_error": pixel_error,
            "unique_images_neurally_rescored": neural_count, "max_metric_errors_PSNR_LPIPS_DINO": maximum_errors.tolist(),
            "max_paired_interval_error": max_interval_error, "unchanged_no_candidate_group_transmissions": unchanged_count,
            "accepted_prefix_tables_verified": len(table_cache), "channel_decoding_or_new_candidate_searches": 0,
            "limitations": ["reuses the registered arithmetic codec and previous independent CRC/receiver auditor",
                            "canonical metrics and legacy renderer independently reloaded; no new test data"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/VAR-CRC-FAILURE-REPLAY-001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-CRC-FAILURE-REPLAY-AUDIT-001")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    receipt = verify_artifacts(run, "completion.json")
    verify_snapshot(receipt["source_hashes"])
    config = yaml.safe_load((run / "snapshots/configs/crc_failure_replay.yaml").read_text())
    output = create_output(args.output_dir)
    sources = snapshot(output, [Path(__file__), ROOT / "scripts/audit_progressive_channel.py", ROOT / "scripts/audit_single_scale_channel.py"])
    started = time.perf_counter()
    try:
        result = audit(run, output, config, receipt)
        verify_snapshot(sources)
        verify_artifacts(run, "completion.json")
        write_json(output / "audit.json", {**result, "local_completed": datetime.now().astimezone().isoformat(),
                                           "run_completion_sha256": sha256(run / "completion.json"), "source_hashes": sources,
                                           "elapsed_seconds": time.perf_counter() - started, "output_hashes": artifact_hashes(output)})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
