#!/usr/bin/env python3
"""Exhaustive small-case verification of the new token-prior channel decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from var_comm.scale_channel import (
    append_crc, bits_to_indices, channel_evidence, convolutional_encode, crc16,
    crc_accepts, decode_map, encode_packet, indices_to_bits, load_native,
    path_score, rate_match_indices,
)


def verify():
    rng = np.random.default_rng(2026090701)
    checks = {}
    check_bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    if crc16(check_bits) != 0x29B1:
        raise RuntimeError("CRC-16/CCITT-FALSE known-answer mismatch")
    for length in (1, 12, 31, 2028):
        source = rng.integers(0, 2, length, dtype=np.uint8)
        packet = append_crc(source)
        if not crc_accepts(packet):
            raise RuntimeError("CRC append/check mismatch")
        packet[length // 2] ^= 1
        if crc_accepts(packet):
            raise RuntimeError("CRC did not detect a single-bit error")
    checks["crc_known_answer_and_unaligned_payload"] = "PASS"
    if not np.array_equal(bits_to_indices(indices_to_bits(np.arange(4096))), np.arange(4096)):
        raise RuntimeError("12-bit MSB-first index conversion mismatch")
    checks["token_bit_order"] = "PASS"
    load_native()
    max_exhaustive_error = 0.0
    for memory, token_count in ((2, 1), (2, 2), (2, 3), (6, 1)):
        width = 2 * memory
        length = width * token_count
        candidates = np.asarray([np.concatenate((indices_to_bits([value], length), np.zeros(memory, dtype=np.uint8)))
                                 for value in range(1 << length)])
        signals = np.asarray([1.0 - 2.0 * convolutional_encode(candidate, memory) for candidate in candidates])
        token_indices = np.asarray([bits_to_indices(candidate[:length], width) for candidate in candidates])
        for trial in range(6):
            evidence = rng.normal(size=2 * (length + memory)) * (0.2 + trial)
            if trial % 2:
                evidence[::3] = 0
            mass = rng.lognormal(0, 2, (token_count, 1 << width))
            priors = np.log(mass / mass.sum(axis=1, keepdims=True))
            decoded, score = decode_map(evidence, priors, memory=memory)
            all_scores = -signals @ evidence - priors[np.arange(token_count)[None], token_indices].sum(axis=1)
            expected = float(np.min(all_scores))
            recomputed = path_score(decoded, evidence, priors, memory)
            error = max(abs(score - expected), abs(recomputed - expected))
            if error > 1e-8 or np.any(decoded[-memory:]):
                raise RuntimeError(f"exhaustive MAP mismatch: memory={memory}, tokens={token_count}, error={error}")
            max_exhaustive_error = max(max_exhaustive_error, error)
    checks["24_exhaustive_joint_token_MAP_cases"] = "PASS"
    for slots in (12, 20, 48):
        information = np.concatenate((rng.integers(0, 2, 8, dtype=np.uint8), np.zeros(2, dtype=np.uint8)))
        mother = convolutional_encode(information, 2)
        mapping = rate_match_indices(len(mother), slots)
        symbols = 1.0 - 2.0 * mother[mapping]
        received = symbols + rng.normal(size=slots)
        evidence = channel_evidence(received, mapping, len(information), 3.0)
        for value in range(256):
            candidate = np.concatenate((indices_to_bits([value], 8), np.zeros(2, dtype=np.uint8)))
            trial_symbols = 1.0 - 2.0 * convolutional_encode(candidate, 2)[mapping]
            direct = 0.5 * (10 ** 0.3) * float(np.sum((received - trial_symbols) ** 2))
            omitted = 0.5 * (10 ** 0.3) * float(np.sum(received ** 2 + 1))
            if abs(direct - omitted - path_score(candidate, evidence, memory=2)) > 1e-9:
                raise RuntimeError("puncturing/repetition AWGN likelihood scale mismatch")
    checks["768_direct_waveform_likelihood_cases"] = "PASS"
    uniform = np.full((169, 4096), -np.log(4096))
    timings = []
    for uses in (1100, 1224, 2050, 2500):
        tokens = rng.integers(0, 4096, 169)
        packet = encode_packet(indices_to_bits(tokens), uses)
        evidence = channel_evidence(packet["symbols"], packet["mapping"], len(packet["information"]), 20)
        tick = time.perf_counter()
        decoded, score = decode_map(evidence, uniform)
        timings.append(time.perf_counter() - tick)
        vanilla, vanilla_score = decode_map(evidence)
        if not np.array_equal(decoded, packet["information"]) or not crc_accepts(decoded[:-6]):
            raise RuntimeError("noiseless terminated packet failed")
        if not np.array_equal(vanilla, decoded) or abs(score - vanilla_score - 169 * np.log(4096)) > 1e-7:
            raise RuntimeError("uniform MAP differs from ordinary exact Viterbi")
        noisy_evidence = evidence / 100 + rng.normal(size=len(evidence))
        noisy_token, token_score = decode_map(noisy_evidence, uniform)
        noisy_bit, bit_score = decode_map(noisy_evidence)
        if abs(token_score - bit_score - 169 * np.log(4096)) > 1e-7:
            raise RuntimeError("noisy uniform token MAP score differs from ordinary Viterbi")
        if abs(path_score(noisy_token, noisy_evidence) - path_score(noisy_bit, noisy_evidence)) > 1e-7:
            raise RuntimeError("noisy uniform token MAP path is not ML")
    checks["noiseless_recovery_and_uniform_native_ML_equivalence"] = "PASS"
    return {"status": "CHANNEL_SELFCHECK_PASS", "checks": checks,
            "max_exhaustive_score_error": max_exhaustive_error,
            "mean_token_map_seconds_169_tokens": float(np.mean(timings)),
            "crc_role": "post_decode_detection_not_a_constraint_in_the_MAP_search",
            "search": "exact terminated convolutional trellis under the factorized token prior"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-SCALE-CHANNEL-SELFCHECK-001")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("selfcheck outputs must stay under VAR_COMM/outputs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = verify()
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"status": "CHANNEL_SELFCHECK_FAILED", "error": repr(error)}, indent=2) + "\n")
        raise
    result["source_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), ROOT / "src/var_comm/scale_channel.py", ROOT / "src/var_comm/token_trellis.cpp")}
    (output / "selfcheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
