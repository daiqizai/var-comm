#!/usr/bin/env python3
"""Recompute waveform, CRC, metrics and selected exact MAP optima independently."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml
from var_comm.study import artifact_hashes, create_output, sha256, snapshot, verify_artifacts, write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encode_independent(bits):
    impulse = [[(generator >> delay) & 1 for delay in range(7)] for generator in (0o171, 0o133)]
    return np.stack([np.convolve(bits, taps)[:len(bits)] % 2 for taps in impulse], axis=1).ravel()


def crc_independent(bits):
    length = len(bits)
    packed = int.from_bytes(np.packbits(bits).tobytes(), "big") >> ((-length) % 8)
    remainder = (packed << 16) ^ (0xFFFF << length)
    while remainder.bit_length() > 16:
        remainder ^= 0x11021 << (remainder.bit_length() - 17)
    return remainder


def indices(bits):
    return np.asarray(bits[:2028]).reshape(169, 12) @ (1 << np.arange(11, -1, -1))


def numpy_map_score(evidence, priors, half_symbols, bit_symbols):
    metrics = np.full(64, np.inf)
    metrics[0] = 0
    for token, table in enumerate(priors):
        first = -(half_symbols @ evidence[token * 24:token * 24 + 12]).reshape(64, 64)
        second = -(half_symbols @ evidence[token * 24 + 12:token * 24 + 24]).reshape(64, 64)
        middle = (metrics[:, None] + first).min(axis=0)
        metrics = (middle[:, None] + second - table.reshape(64, 64)).min(axis=0)
    predecessors = np.stack((np.arange(64) >> 1, (np.arange(64) >> 1) | 32), axis=1)
    for offset in range(2028, 2050):
        symbols = bit_symbols[predecessors, (np.arange(64) & 1)[:, None]]
        costs = -(symbols * evidence[offset * 2:offset * 2 + 2]).sum(axis=2)
        metrics = (metrics[predecessors] + costs).min(axis=1)
    return float(metrics[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/VAR-SINGLE-SCALE-CHANNEL-001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-SINGLE-SCALE-CHANNEL-AUDIT-001")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    receipt = verify_artifacts(run, "completion.json")
    require(receipt["mode"] == "full", "this audit expects the registered full trial")
    config = yaml.safe_load((run / "snapshots/configs/single_scale_channel.yaml").read_text())
    prior_run = ROOT / config["prior_run"]
    verify_artifacts(prior_run, "completion.json", config["prior_completion_sha256"])
    output = create_output(args.output_dir or ROOT / "outputs/VAR-SINGLE-SCALE-CHANNEL-AUDIT-001")
    snapshot(output, [Path(__file__)])
    with (run / "per_frame.csv").open() as handle:
        recorded = list(csv.DictReader(handle))
    lookup = {(int(row["image_index"]), float(row["snr_db"]), int(row["seed"]), row["arm"]): row for row in recorded}
    require(len(lookup) == len(recorded) == 7500, "incomplete or duplicate frame rows")
    with np.load(prior_run / "frequency_counts.npz", allow_pickle=False) as cache:
        mass = cache["scale_9"].astype(np.float64) + 0.5
    static = np.log(mass / mass.sum())
    with np.load(prior_run / "source_tokens.npz", allow_pickle=False) as cache:
        token_lookup = dict(zip(cache["image_ids"].tolist(), cache["tokens"].copy()))
    half_symbols = []
    for value in range(4096):
        bits = ((value >> np.arange(11, -1, -1)) & 1).astype(np.uint8)
        half_symbols.append(1.0 - 2.0 * encode_independent(bits)[12:])
    half_symbols = np.asarray(half_symbols)
    bit_symbols = np.empty((64, 2, 2))
    for state in range(64):
        for bit in range(2):
            history = np.concatenate((((state >> np.arange(5, -1, -1)) & 1).astype(np.uint8), [bit]))
            bit_symbols[state, bit] = 1.0 - 2.0 * encode_independent(history)[-2:]
    require(crc_independent(np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))) == 0x29B1, "independent CRC implementation failed")
    recomputed, maximum_score_error, maximum_optimum_error, optimum_cases = [], 0.0, 0.0, 0
    for image_index in range(100):
        with np.load(prior_run / "probabilities" / f"{image_index:03d}.npz", allow_pickle=False) as priors:
            true = priors["true_source_prefix_scale_9"].astype(np.float64)
            donor = priors["same_class_other_prefix_scale_9"].astype(np.float64)
        true_mass = np.exp(true)
        with np.load(run / "frames" / f"{image_index:03d}.npz", allow_pickle=False) as cache:
            information = cache["information"]
            identifier = lookup[(image_index, 4.0, 1001, "uniform")]["image_id"]
            require(np.array_equal(indices(information), token_lookup[identifier][255:424]), "wrong transmitted scale or source")
            crc_value = int(information[2028:2044] @ (1 << np.arange(15, -1, -1)))
            require(crc_independent(information[:2028]) == crc_value and not information[-6:].any(), "transmitted CRC/tail invalid")
            mapping = np.floor((np.arange(2448) + 0.5) * 4100 / 2448).astype(np.int64)
            require(np.array_equal(mapping, cache["mapping"]), "rate matching changed")
            signal = (1.0 - 2.0 * encode_independent(information)[mapping]).reshape(1224, 2)
            require(np.array_equal(signal, cache["transmitted"]), "transmitted waveform mismatch")
            for snr_index, snr in enumerate(config["snr_db"]):
                gamma = 10 ** (snr / 10)
                for seed_index, seed in enumerate(config["noise_seeds"]):
                    seed_value = int.from_bytes(hashlib.sha256(f"{identifier}|{seed}".encode()).digest()[:8], "big")
                    expected_y = signal + np.random.default_rng(seed_value).standard_normal(signal.shape) / np.sqrt(gamma)
                    received = cache["received"][snr_index, seed_index]
                    require(np.array_equal(expected_y, received), "AWGN realization or SNR scale changed")
                    for arm_index, arm in enumerate(config["arms"]):
                        row = lookup[(image_index, snr, seed, arm)]
                        decoded = cache["decoded"][snr_index, seed_index, arm_index]
                        require(hashlib.sha256(received.tobytes()).hexdigest() == row["received_sha256"], "arms did not use the same observations")
                        require(not decoded[-6:].any(), "decoded tail is not terminated")
                        token_values = indices(decoded)
                        if arm == "uniform":
                            log_mass = np.full(169, -np.log(4096))
                        elif arm == "static_scale_frequency":
                            log_mass = static[token_values]
                        elif arm in ("true_source_prefix", "same_class_other_prefix"):
                            log_mass = (true if arm == "true_source_prefix" else donor)[np.arange(169), token_values]
                        else:
                            log_mass = np.zeros(169)
                            for shift in range(12):
                                mask = ((np.arange(4096) >> shift) & 1).astype(bool)
                                matching = np.where((token_values >> shift) & 1, true_mass[:, mask].sum(axis=1), true_mass[:, ~mask].sum(axis=1))
                                log_mass += np.log(matching)
                        candidate_signal = 1.0 - 2.0 * encode_independent(decoded)[mapping]
                        score = -gamma * float(received.ravel() @ candidate_signal) - float(log_mass.sum())
                        error = abs(score - float(row["map_score"]))
                        maximum_score_error = max(maximum_score_error, error)
                        require(error < 1e-7, "direct waveform MAP score mismatch")
                        errors = int(np.count_nonzero(decoded[:2028] != information[:2028]))
                        accepts = crc_independent(decoded[:2028]) == int(decoded[2028:2044] @ (1 << np.arange(15, -1, -1)))
                        metrics = {"source_bit_errors": errors, "source_ber": errors / 2028, "source_bler": int(errors > 0),
                                   "protected_bler": int(np.any(decoded[:-6] != information[:-6])), "crc_acceptance": int(accepts),
                                   "accepted_source_recovery": int(accepts and errors == 0), "crc_false_acceptance": int(accepts and errors > 0),
                                   "correct_source_crc_rejection": int(not accepts and errors == 0)}
                        require(all(abs(value - float(row[name])) < 1e-12 for name, value in metrics.items()), "CRC or frame metric mismatch")
                        recomputed.append({"image_index": image_index, "snr_db": snr, "arm": arm, **metrics})
                        if image_index == 0 and seed_index == 0 and arm in config["primary_controls"] + ["true_source_prefix"]:
                            evidence = np.bincount(mapping, weights=gamma * received.ravel(), minlength=4100)
                            table = true if arm == "true_source_prefix" else np.broadcast_to(static if arm == "static_scale_frequency" else np.full(4096, -np.log(4096)), (169, 4096))
                            optimum = numpy_map_score(evidence, table, half_symbols, bit_symbols)
                            maximum_optimum_error = max(maximum_optimum_error, abs(optimum - score))
                            require(abs(optimum - score) < 1e-7, "independent NumPy MAP optimum disagrees")
                            optimum_cases += 1
        if (image_index + 1) % 20 == 0:
            print(f"audit scale9 {image_index + 1}/100", flush=True)
    with (run / "summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            group = [entry for entry in recomputed if entry["arm"] == row["arm"] and entry["snr_db"] == float(row["snr_db"])]
            for metric in ("source_ber", "source_bler", "protected_bler", "crc_acceptance", "accepted_source_recovery", "crc_false_acceptance"):
                require(abs(np.mean([entry[metric] for entry in group]) - float(row[metric])) < 1e-12, "aggregate mismatch")
    max_ci_error = 0.0
    with (run / "paired_gains.csv").open() as handle:
        for row in csv.DictReader(handle):
            selected = [float(value) for value in row["snr_db"].split("+")]
            grouped = defaultdict(list)
            for entry in recomputed:
                if entry["snr_db"] in selected:
                    grouped[(entry["image_index"], entry["arm"])].append(entry["accepted_source_recovery"])
            differences = np.array([np.mean(grouped[(index, "true_source_prefix")]) - np.mean(grouped[(index, row["control"])]) for index in range(100)])
            selections = np.random.default_rng(config["bootstrap"]["seed"]).integers(100, size=(10000, 100))
            lower, upper = np.quantile(differences[selections].mean(axis=1), [0.025, 0.975])
            for field, value in (("gain", differences.mean()), ("ci_low", lower), ("ci_high", upper)):
                max_ci_error = max(max_ci_error, abs(value - float(row[field])))
    require(max_ci_error < 1e-12, "paired image interval mismatch")
    result = {"status": "AUDIT_PASS", "run": str(run), "run_completion_sha256": sha256(run / "completion.json"),
              "rows_recomputed": len(recomputed), "independent_numpy_MAP_cases": optimum_cases,
              "max_direct_waveform_score_error": maximum_score_error, "max_independent_MAP_optimum_error": maximum_optimum_error,
              "max_paired_interval_error": max_ci_error, "crc_method": "independent_GF2_polynomial_long_division",
              "output_hashes": artifact_hashes(output)}
    write_json(output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
