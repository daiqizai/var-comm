#!/usr/bin/env python3
"""Matched-observation scale-9 decoding; the source prefix is explicitly oracle."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
import yaml

from var_comm.scale_channel import (
    bit_marginal_table, channel_evidence, crc_accepts, decode_map, encode_packet,
    indices_to_bits, load_native, path_score,
)
from var_comm.study import (
    artifact_hashes, create_output, paired_interval, seeded_noise, sha256,
    snapshot, verify_artifacts, verify_snapshot, write_csv, write_json,
)

METRICS = ("source_ber", "source_bler", "protected_bler", "crc_acceptance", "accepted_source_recovery",
           "crc_false_acceptance", "correct_source_crc_rejection", "decode_seconds")


def decode_job(evidence, table):
    started = time.perf_counter()
    decoded, score = decode_map(evidence, table)
    return decoded, score, time.perf_counter() - started


def summarize(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["snr_db"], row["arm"])].append(row)
    summary = []
    for (snr, arm), group in sorted(groups.items()):
        summary.append({"snr_db": snr, "arm": arm, "frames": len(group),
                        **{metric: float(np.mean([row[metric] for row in group])) for metric in METRICS},
                        "crc_false_acceptance_count": sum(row["crc_false_acceptance"] for row in group)})
    comparisons = []
    identifiers = sorted({row["image_id"] for row in rows})
    for selected_snr in ([value] for value in config["snr_db"]):
        for control in config["primary_controls"]:
            comparisons.append(compare(rows, identifiers, selected_snr, control, config))
    primary = [compare(rows, identifiers, config["gate"]["primary_snr_db"], control, config)
               for control in config["primary_controls"]]
    comparisons.extend(primary)
    checks = {row["control"]: row["gain"] >= config["gate"]["minimum_absolute_accepted_source_recovery_gain_over_each_control"] and row["ci_low"] > 0
              for row in primary}
    high = compare(rows, identifiers, [config["gate"]["high_snr_guard_db"]], "uniform", config)
    checks["high_snr_guard"] = high["gain"] >= -config["gate"]["maximum_high_snr_recovery_drop"]
    checks["observed_crc_false_acceptances"] = sum(row["crc_false_acceptance"] for row in rows if row["arm"] == "true_source_prefix") <= config["gate"]["maximum_observed_VAR_crc_false_acceptances"]
    return summary, comparisons, {"passed": all(checks.values()), "checks": checks}


def compare(rows, identifiers, selected_snr, control, config):
    measurements = defaultdict(list)
    for row in rows:
        if row["snr_db"] in selected_snr and row["arm"] in (control, "true_source_prefix"):
            measurements[(row["image_id"], row["arm"])].append(row["accepted_source_recovery"])
    differences = [np.mean(measurements[(identifier, "true_source_prefix")]) - np.mean(measurements[(identifier, control)]) for identifier in identifiers]
    interval = paired_interval(differences, config["bootstrap"]["seed"], config["bootstrap"]["resamples"])
    return {"snr_db": "+".join(str(value) for value in selected_snr), "control": control, "metric": "accepted_source_recovery", **interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/single_scale_channel.yaml")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["status"] != "preregistered_before_target_channel_results" or config["target_scale"] != 9:
        raise ValueError("single-scale contract changed")
    prior_run = ROOT / config["prior_run"]
    prior_receipt = verify_artifacts(prior_run, "completion.json", config["prior_completion_sha256"])
    if prior_receipt["status"] != "PASS_PRIOR_GATE_ONLY":
        raise RuntimeError("prior gate has not passed")
    selfcheck = json.loads((ROOT / config["selfcheck"]).read_text())
    if selfcheck["status"] != "CHANNEL_SELFCHECK_PASS":
        raise RuntimeError("decoder selfcheck has not passed")
    verify_snapshot(selfcheck["source_hashes"])
    output = create_output(args.output_dir or ROOT / config["outputs"][args.mode])
    (output / "frames").mkdir()
    records = snapshot(output, [Path(__file__), args.config, ROOT / "src/var_comm/scale_channel.py",
        ROOT / "src/var_comm/token_trellis.cpp", ROOT / "src/var_comm/study.py",
        ROOT / "scripts/check_scale_channel.py", ROOT / "reports/single_scale_channel_preregistration_2026-09-07.md"])
    metadata = {"local_started": datetime.now().astimezone().isoformat(), "command": sys.argv,
                "mode": args.mode, "source_hashes": records, "prior_completion_sha256": config["prior_completion_sha256"],
                "selfcheck_sha256": sha256(ROOT / config["selfcheck"]), "python": sys.version, "numpy": np.__version__}
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    try:
        manifest = json.loads((prior_run / "population_manifest.json").read_text())
        targets = [record for record in manifest if record["role"] == "target_development"]
        if len(targets) != config["target_count"]:
            raise RuntimeError("target population changed")
        with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
            tokens = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
        with np.load(prior_run / "frequency_counts.npz", allow_pickle=False) as cache:
            weights = cache["scale_9"].astype(np.float64) + 0.5
        static = np.broadcast_to(np.log(weights / weights.sum()), (169, 4096)).copy()
        uniform = np.full((169, 4096), -np.log(4096))
        seeds = config["noise_seeds"]
        if args.mode == "smoke":
            targets, seeds = targets[:2], seeds[:1]
        write_json(output / "target_manifest.json", targets)
        rows, native_rows = [], []
        load_native()
        max_score_error = 0.0
        with ThreadPoolExecutor(max_workers=1 if args.mode == "smoke" else config["workers"]) as executor:
            for image_index, target in enumerate(targets):
                source_tokens = tokens[target["image_id"]][255:424]
                if len(source_tokens) != 169:
                    raise RuntimeError("wrong scale-9 token slice")
                payload = indices_to_bits(source_tokens)
                packet = encode_packet(payload, config["phy"]["complex_uses"])
                if len(packet["information"]) != 2050 or len(packet["mapping"]) != 2448:
                    raise RuntimeError("packet budget differs from registration")
                with np.load(prior_run / "probabilities" / f"{image_index:03d}.npz", allow_pickle=False) as cache:
                    true = cache["true_source_prefix_scale_9"].astype(np.float64)
                    donor = cache["same_class_other_prefix_scale_9"].astype(np.float64)
                tables = {"uniform": uniform, "static_scale_frequency": static, "true_source_prefix": true,
                          "bit_marginal_true": bit_marginal_table(true), "same_class_other_prefix": donor}
                shape = (len(config["snr_db"]), len(seeds), len(config["arms"]))
                decoded_cache = np.empty((*shape, len(packet["information"])), dtype=np.uint8)
                score_cache = np.empty(shape)
                received_cache = np.empty((*shape[:2], config["phy"]["complex_uses"], 2))
                native_cache = np.empty((*shape[:2], len(packet["information"])), dtype=np.uint8)
                for snr_index, snr in enumerate(config["snr_db"]):
                    for seed_index, seed in enumerate(seeds):
                        noise = seeded_noise(target["image_id"], seed, packet["symbols"].shape)
                        received = packet["symbols"] + noise / np.sqrt(10 ** (snr / 10))
                        received_cache[snr_index, seed_index] = received
                        evidence = channel_evidence(received, packet["mapping"], len(packet["information"]), snr)
                        jobs = {arm: executor.submit(decode_job, evidence, tables[arm]) for arm in config["arms"]}
                        native, native_score, native_seconds = decode_job(evidence, None)
                        native_cache[snr_index, seed_index] = native
                        native_rows.append({"image_id": target["image_id"], "snr_db": snr, "seed": seed,
                                            "score": native_score, "decode_seconds": native_seconds})
                        received_hash = hashlib.sha256(received.tobytes()).hexdigest()
                        for arm_index, arm in enumerate(config["arms"]):
                            decoded, score, seconds = jobs[arm].result()
                            recomputed = path_score(decoded, evidence, tables[arm])
                            truth_score = path_score(packet["information"], evidence, tables[arm])
                            max_score_error = max(max_score_error, abs(score - recomputed))
                            if abs(score - recomputed) > 1e-7 or score > truth_score + 1e-7 or np.any(decoded[-6:]):
                                raise RuntimeError("MAP score or known termination audit failed")
                            if arm == "uniform" and abs(score - native_score - 169 * np.log(4096)) > 1e-7:
                                raise RuntimeError("ordinary Viterbi has a better channel candidate")
                            source_errors = int(np.count_nonzero(decoded[:2028] != payload))
                            protected_errors = int(np.count_nonzero(decoded[:-6] != packet["information"][:-6]))
                            accepts = crc_accepts(decoded[:-6])
                            row = {"image_index": image_index, "image_id": target["image_id"], "snr_db": snr, "seed": seed,
                                   "arm": arm, "source_bit_errors": source_errors, "source_bits": 2028,
                                   "source_ber": source_errors / 2028, "source_bler": int(source_errors != 0),
                                   "protected_bler": int(protected_errors != 0), "crc_acceptance": int(accepts),
                                   "accepted_source_recovery": int(accepts and source_errors == 0),
                                   "crc_false_acceptance": int(accepts and source_errors != 0),
                                   "correct_source_crc_rejection": int(not accepts and source_errors == 0),
                                   "decode_seconds": seconds, "map_score": score, "true_path_score": truth_score,
                                   "received_sha256": received_hash, "complex_uses": packet["complex_uses"]}
                            rows.append(row)
                            decoded_cache[snr_index, seed_index, arm_index] = decoded
                            score_cache[snr_index, seed_index, arm_index] = score
                np.savez(output / "frames" / f"{image_index:03d}.npz", payload=payload, information=packet["information"],
                         mapping=packet["mapping"], transmitted=packet["symbols"], received=received_cache,
                         decoded=decoded_cache, map_scores=score_cache, native_decoded=native_cache,
                         seeds=np.asarray(seeds), snr_db=np.asarray(config["snr_db"]), arms=np.asarray(config["arms"]))
                write_csv(output / "per_frame.csv", rows)
                if (image_index + 1) % 10 == 0 or args.mode == "smoke":
                    print(f"scale9 {image_index + 1}/{len(targets)} frames={len(rows)}", flush=True)
        expected = len(targets) * len(config["snr_db"]) * len(seeds) * len(config["arms"])
        if len(rows) != expected:
            raise RuntimeError("incomplete single-scale trial grid")
        summary, comparisons, gate = summarize(rows, config)
        write_csv(output / "summary.csv", summary)
        write_csv(output / "paired_gains.csv", comparisons)
        write_csv(output / "native_viterbi.csv", native_rows)
        verify_snapshot(records)
        status = "PASS_SINGLE_SCALE_GATE_ONLY" if gate["passed"] else "STOP_THIS_SINGLE_SCALE_CONFIGURATION"
        if args.mode == "smoke":
            status = "SMOKE_COMPLETE_NOT_SCIENTIFIC_RESULT"
        result = {**metadata, "status": status, "rows": len(rows), "target_count": len(targets), "gate": gate,
                  "max_recomputed_MAP_score_error": max_score_error, "elapsed_seconds": time.perf_counter() - started,
                  "ledger": config["phy"], "output_hashes": artifact_hashes(output)}
        write_json(output / "completion.json", result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("output_hashes", "source_hashes")}, indent=2), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_CLOSED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
