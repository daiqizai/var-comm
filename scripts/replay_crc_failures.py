#!/usr/bin/env python3
"""Replay only the image output of the first frozen CRC-failed candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
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

import numpy as np
import torch
import yaml

from var_comm.entropy import arithmetic_decode, probability_cdf
from var_comm.failure_replay import METRICS, choose_oracle, image_only_state, render_from_scratch, summarize
from var_comm.next_scale_prior import load_models, state_sha256
from var_comm.progressive import prefix_key, split_prefix
from var_comm.quality import load_quality_models, quality_metrics
from var_comm.scale_channel import bits_to_indices, crc16, indices_to_bits
from var_comm.study import (artifact_hashes, create_output, sha256, snapshot, verify_artifacts,
                            verify_snapshot, write_csv, write_json)


def fail_channel_call(*args, **kwargs):
    raise RuntimeError("image-only replay must not run channel decoding or source-prior searches")


def check_saved_candidates(record, input_run):
    prefix, entropy_checks = [], 0
    for event in record["events"]:
        if "decoded_bits" not in event:
            continue
        bits = np.asarray(event["decoded_bits"], dtype=np.uint8)
        length = event["source_wire_bits"]
        group = event["group"]
        if len(bits) != length + 22 or np.any(bits > 1) or bits[-6:].any():
            raise RuntimeError("invalid frozen decoder output")
        compressed = record["arm"] == "group_entropy" and group > 0 and record["header"]["lengths"][group - 1] > 0
        if compressed:
            key = prefix_key(prefix, record["label"])
            if key != event["prior_key"]:
                raise RuntimeError("entropy candidate was not conditioned on the accepted prefix")
            with np.load(input_run / "probabilities" / (key + ".npz"), allow_pickle=False) as cache:
                if int(cache["label"]) != record["label"] or not np.array_equal(cache["prefix"], np.concatenate(prefix)):
                    raise RuntimeError("entropy probability cache context changed")
                cdf = probability_cdf(cache["log_probs"])
            recovered = arithmetic_decode(bits[:length], cdf)
            entropy_checks += 1
        else:
            recovered = bits_to_indices(bits[:length])
        if recovered.tolist() != event["recovered_tokens"]:
            raise RuntimeError("saved source candidate differs from actual decoded bits")
        accepted = crc16(indices_to_bits(recovered)) == int(bits_to_indices(bits[length:length + 16], 16)[0])
        if accepted != event["accepted"]:
            raise RuntimeError("source CRC result changed in frozen trace")
        if not accepted:
            break
        prefix = split_prefix(recovered, 6) if group == 0 else prefix + [recovered]
    return entropy_checks


def replay(config, output, metadata, input_receipt):
    input_run = ROOT / config["input_run"]
    upstream = yaml.safe_load((input_run / "snapshots/configs/progressive_channel.yaml").read_text())
    model_config = yaml.safe_load((ROOT / upstream["model_config"]).read_text())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    for paths, names in ((model_config["paths"], ("vae_checkpoint", "var_checkpoint")),
                         (upstream["quality"], ("alexnet_checkpoint", "dino_checkpoint"))):
        for name in names:
            if sha256(paths[name]) != paths[name + "_sha256"]:
                raise RuntimeError("frozen model checkpoint changed")
    vae, var = load_models(model_config["paths"], device)
    with mock.patch("torch.hub.download_url_to_file", side_effect=RuntimeError("downloads forbidden")):
        perceptual, dino, _weights = load_quality_models(upstream["quality"], device)
    models = {"vae": vae, "var": var, "lpips": perceptual, "dino": dino}
    before = {name: state_sha256(model) for name, model in models.items()}
    if before != input_receipt["frozen_before"]:
        raise RuntimeError("replay models differ from the original experiment")
    populations = json.loads((input_run / "populations.json").read_text())
    targets = populations["target"]
    if len(targets) != config["target_count"] or any(target["role"] != "target_development" for target in targets):
        raise RuntimeError("development population changed")
    prior_run = ROOT / upstream["prior_run"]
    if sha256(prior_run / "completion.json") != upstream["prior_completion_sha256"]:
        raise RuntimeError("diagnostic source-token receipt changed")
    prior_receipt = json.loads((prior_run / "completion.json").read_text())
    if sha256(prior_run / "source_tokens.npz") != prior_receipt["output_hashes"]["source_tokens.npz"]:
        raise RuntimeError("diagnostic source tokens changed")
    with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
        token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
    with (input_run / "per_frame.csv").open() as handle:
        frozen_rows = [row for row in csv.DictReader(handle) if float(row["snr_db"]) == config["snr_db"]
                       and row["arm"] in config["grouped_arms"] + [config["unchanged_control"]]]
    old_lookup = {(int(row["image_index"]), int(row["seed"]), row["arm"]): row for row in frozen_rows}
    if len(old_lookup) != config["target_count"] * len(config["noise_seeds"]) * 4:
        raise RuntimeError("incomplete frozen 7 dB source grid")
    rows, render_count, replay_count, entropy_checks, isolation_error = [], 0, 0, 0, 0.0
    for image_index, target in enumerate(targets):
        directory = input_run / "images" / f"{image_index:03d}"
        records = json.loads((directory / "receivers.json").read_text())
        selected = [record for record in records if record["snr_db"] == config["snr_db"]
                    and record["arm"] in config["grouped_arms"] + [config["unchanged_control"]]]
        frozen_copy = deepcopy(selected)
        plans = {}
        for record in selected:
            if record["arm"] in config["grouped_arms"]:
                state = image_only_state(record)
                entropy_checks += check_saved_candidates(record, input_run)
                plans[record["seed"], record["arm"]] = state
        with np.load(directory / "reconstructions.npz", allow_pickle=False) as cache:
            old_images, reference = cache["images"].copy(), cache["source"].copy()
        cached, images, image_states = {}, [], []
        for record in selected:
            old = old_lookup[image_index, record["seed"], record["arm"]]
            key = "header_erasure" if record["label"] is None else prefix_key(record["output_prefix"], record["label"])
            if old["reconstruction_key"] != key or old["received_sha256"] != record["received_sha256"]:
                raise RuntimeError("original receiver/image association changed")
            cached[key] = {"image_ref": "input:" + str(record["reconstruction_index"]),
                           **{metric: float(old[metric]) for metric in METRICS}}
        for state in plans.values():
            key = "header_erasure" if state["label"] is None else prefix_key(state["render_prefix"], state["label"])
            if key not in cached:
                cached[key] = {"image_ref": "new:" + str(len(images))}
                images.append(render_from_scratch(vae, var, state["render_prefix"], state["label"], device))
                image_states.append({"key": key, "image_index": len(images) - 1,
                                     "label": state["label"], "render_prefix": deepcopy(state["render_prefix"])})
        baseline = next(record for record in selected if record["arm"] == "whole_m9")
        check = render_from_scratch(vae, var, baseline["output_prefix"], baseline["label"], device)
        error = float(np.max(np.abs(check - old_images[baseline["reconstruction_index"]])))
        isolation_error = max(isolation_error, error)
        if error != 0.0 or selected != frozen_copy:
            raise RuntimeError("retained branch changed a later baseline render or trusted receiver trace")
        if images:
            quality, _source_features, features = quality_metrics(reference, images, perceptual, dino, device)
            for state, metrics in zip(image_states, quality):
                cached[state["key"]].update(metrics)
        else:
            features = np.empty((0, 384), dtype=np.float32)
        local_rows, traces = [], []
        source = split_prefix(token_lookup[target["image_id"]], 10)
        for record in selected:
            family, seed = record["arm"], record["seed"]
            old = old_lookup[image_index, seed, family]
            if int(old["complex_uses"]) != config["complex_uses"]:
                raise RuntimeError("budget changed")
            base = {"image_index": image_index, "image_id": target["image_id"], "snr_db": config["snr_db"],
                    "seed": seed, "family": family, "arm": family, "policy": "original", "deployable": 1,
                    "complex_uses": int(old["complex_uses"]), "received_sha256": record["received_sha256"],
                    "trusted_scales": record["last_accepted_scale"], "render_scales": len(record["output_prefix"]),
                    "correct_m9_accepted": int(old["correct_m9_accepted"]), "failure_stage": "unchanged_whole",
                    "candidate_available": 0, "retention_applied": 0, "failed_candidate_true_ber": "",
                    "image_ref": "input:" + str(record["reconstruction_index"]), "reconstruction_key": old["reconstruction_key"],
                    "oracle_selected_policy": "", **{metric: float(old[metric]) for metric in METRICS}}
            if family == "whole_m9":
                local_rows.append(base)
                continue
            state = plans[seed, family]
            base.update(failure_stage=state["failure_stage"], candidate_available=int(state["candidate_available"]))
            if state["candidate_available"]:
                group = state["failure_group"]
                truth = np.concatenate(source[:6]) if group == 0 else source[group + 5]
                candidate = np.asarray(record["events"][-1]["recovered_tokens"])
                base["failed_candidate_true_ber"] = float(np.mean(indices_to_bits(truth) != indices_to_bits(candidate)))
            discarded = dict(base, arm=family + "_discard", policy="discard")
            retained = dict(base, arm=family + "_retain", policy="retain", render_scales=state["render_scales"],
                            retention_applied=int(state["retention_applied"]))
            key = "header_erasure" if state["label"] is None else prefix_key(state["render_prefix"], state["label"])
            retained.update(reconstruction_key=key, **cached[key])
            if (not state["retention_applied"] or int(old["correct_m9_accepted"])) and any(retained[metric] != discarded[metric] for metric in METRICS):
                raise RuntimeError("a no-candidate or correct accepted output changed")
            local_rows.extend((discarded, retained, choose_oracle(discarded, retained)))
            traces.append({"seed": seed, "family": family,
                           "frozen_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), **state})
            replay_count += 1
        image_output = output / "images" / f"{image_index:03d}"
        image_output.mkdir(parents=True)
        np.savez(image_output / "retained.npz", images=np.stack(images) if images else np.empty((0, 3, 256, 256), dtype=np.float32),
                 reconstruction_dino=features)
        write_json(image_output / "image_states.json", image_states)
        write_json(image_output / "replay_states.json", traces)
        rows.extend(local_rows)
        render_count += len(images)
        write_csv(output / "per_frame.csv", rows)
        if (image_index + 1) % 10 == 0:
            print(f"image-only replay {image_index + 1}/{len(targets)} new_renders={render_count} rows={len(rows)}", flush=True)
    summary, comparisons, strata, decision = summarize(rows, config)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_quality.csv", comparisons)
    write_csv(output / "failure_strata.csv", strata)
    write_json(output / "decision.json", decision)
    after = {name: state_sha256(model) for name, model in models.items()}
    if after != before or any(parameter.requires_grad or parameter.grad is not None for model in models.values() for parameter in model.parameters()):
        raise RuntimeError("model freeze violated")
    verify_snapshot(metadata["source_hashes"])
    verify_artifacts(input_run, "completion.json", config["input_completion_sha256"])
    return {"status": decision["status"], "rows": len(rows), "replayed_group_receivers": replay_count,
            "unique_new_renders": render_count, "actual_entropy_candidates_replayed": entropy_checks,
            "channel_decoder_calls": 0, "new_candidate_searches": 0,
            "baseline_renders_after_retained_branches": len(targets), "max_baseline_pixel_difference": isolation_error,
            "frozen_before": before, "frozen_after": after, "decision": decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/crc_failure_replay.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_path = (args.output_dir or ROOT / config["output"]).resolve()
    if not output_path.is_relative_to(ROOT / "outputs"):
        raise ValueError("new outputs must physically remain under VAR_COMM/outputs")
    if output_path.exists():
        raise FileExistsError(output_path)
    if config["status"] != "preregistered_before_retained_image_results":
        raise ValueError("replay contract changed")
    input_run = ROOT / config["input_run"]
    input_receipt = verify_artifacts(input_run, "completion.json", config["input_completion_sha256"])
    verify_snapshot(input_receipt["source_hashes"])
    if sha256(ROOT / config["input_audit"]) != config["input_audit_sha256"]:
        raise RuntimeError("upstream audit changed")
    audit = json.loads((ROOT / config["input_audit"]).read_text())
    if audit["status"] != "AUDIT_PASS" or audit["run_completion_sha256"] != config["input_completion_sha256"]:
        raise RuntimeError("input audit does not cover the frozen run")
    selfcheck_path = ROOT / config["selfcheck"]
    selfcheck = verify_artifacts(selfcheck_path.parent, selfcheck_path.name)
    if selfcheck["status"] != "FAILURE_REPLAY_SELFCHECK_PASS":
        raise RuntimeError("state-isolation selfcheck did not pass")
    verify_snapshot(selfcheck["source_hashes"])
    output = create_output(output_path)
    paths = [Path(__file__), args.config, ROOT / "reports/crc_failure_replay_preregistration_2026-09-07.md",
             *sorted((ROOT / "src/var_comm").glob("*.py"))]
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv,
                "input_completion_sha256": config["input_completion_sha256"], "source_hashes": snapshot(output, paths),
                "selfcheck_sha256": sha256(selfcheck_path), "input_audit_sha256": config["input_audit_sha256"]}
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        with mock.patch.multiple("var_comm.progressive", receive_group=fail_channel_call, receive_whole=fail_channel_call,
                                 decode_map=fail_channel_call, next_scale_log_probs=fail_channel_call), \
             mock.patch("var_comm.scale_channel.decode_map", side_effect=fail_channel_call):
            result = replay(config, output, metadata, input_receipt)
        write_json(output / "completion.json", {**metadata, **result, "elapsed_seconds": time.perf_counter() - started,
                                                "output_hashes": artifact_hashes(output)})
        print(json.dumps(result["decision"], ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
