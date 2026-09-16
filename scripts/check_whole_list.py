#!/usr/bin/env python3
"""Exhaustively check ordered lists, nonzero states, and unreset whole-frame boundaries."""

import argparse
import itertools
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from var_comm.scale_channel import channel_evidence, crc16, decode_map, encode_packet, rate_match_indices
from var_comm.study import artifact_hashes, create_output, snapshot, write_json
from var_comm.whole_list import ordered_paths


def direct_score(bits, evidence, initial, memory, table=None):
    state, score = initial, 0.0
    generators = (0o171, 0o133) if memory == 6 else (7, 5)
    for position, bit in enumerate(bits):
        register = (state << 1) | int(bit)
        for output, generator in enumerate(generators):
            score -= (1 - 2 * ((register & generator).bit_count() % 2)) * evidence[2 * position + output]
        state = register & ((1 << memory) - 1)
    if table is not None:
        for token, probabilities in enumerate(table):
            value = int("".join(str(int(bit)) for bit in bits[token * 2 * memory:(token + 1) * 2 * memory]), 2)
            score -= probabilities[value]
    return score, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/VAR-WHOLE-LIST-SELFCHECK-001")
    args = parser.parse_args()
    output = create_output(args.output_dir)
    random = np.random.default_rng(2026090707)
    exhaustive = boundaries = real = 0
    try:
        for initial in range(4):
            for use_prior in (False, True):
                for terminal in (-1, 0, 2):
                    evidence = random.normal(size=20)
                    table = random.normal(size=(2, 16)) if use_prior else None
                    expected = []
                    for candidate in itertools.product((0, 1), repeat=10):
                        score, end = direct_score(candidate, evidence, initial, 2, table)
                        if terminal == -1 or end == terminal:
                            expected.append((score, candidate))
                    expected.sort()
                    actual = ordered_paths(evidence, len(expected), memory=2, initial_state=initial, final_state=terminal, prior=table)
                    assert np.allclose(actual["rank_scores"], [entry[0] for entry in expected], atol=1e-10, rtol=0)
                    assert [tuple(value) for value in actual["bits"]] == [entry[1] for entry in expected]
                    exhaustive += 1
            evidence = random.normal(size=20)
            expected = {}
            for candidate in itertools.product((0, 1), repeat=10):
                score, end = direct_score(candidate, evidence, initial, 2)
                if end == 0:
                    key = candidate[:5]
                    expected[key] = min(score, expected.get(key, float("inf")))
            ordered = sorted((score, candidate) for candidate, score in expected.items())
            actual = ordered_paths(evidence, len(ordered), memory=2, initial_state=initial, stop_bits=5)
            assert [tuple(value) for value in actual["bits"]] == [entry[1] for entry in ordered]
            assert np.allclose(actual["rank_scores"], [entry[0] for entry in ordered], atol=1e-10, rtol=0)
            boundaries += 1
        for initial in (1, 7, 31, 63):
            evidence = random.normal(size=36)
            table = random.normal(size=(1, 4096))
            expected = []
            for payload in itertools.product((0, 1), repeat=12):
                bits = (*payload, 0, 0, 0, 0, 0, 0)
                score, end = direct_score(bits, evidence, initial, 6, table)
                assert end == 0
                expected.append((score, bits))
            expected.sort()
            actual = ordered_paths(evidence, 32, prior=table, initial_state=initial)
            assert [tuple(value) for value in actual["bits"]] == [entry[1] for entry in expected[:32]]
            assert np.allclose(actual["rank_scores"], [entry[0] for entry in expected[:32]], atol=1e-10, rtol=0)
            exhaustive += 1
        for snr in (4.0, 6.0, 7.0):
            payload = random.integers(0, 2, 5088, dtype=np.uint8)
            packet = encode_packet(payload, 2992)
            received = packet["symbols"] + random.normal(size=(2992, 2)) / np.sqrt(10 ** (snr / 10))
            evidence = channel_evidence(received, packet["mapping"], 5110, snr)
            ml, _score = decode_map(evidence)
            listed = ordered_paths(evidence, 8, payload_bits=5088)
            assert np.array_equal(listed["bits"][0], ml)
            prefix = ordered_paths(evidence, 4, stop_bits=3060)
            assert np.array_equal(prefix["bits"][0], ml[:3060])
            for candidate, accepts in zip(listed["bits"], listed["accepted"]):
                crc_value = int("".join(str(int(bit)) for bit in candidate[5088:5104]), 2)
                assert bool(accepts) == (crc16(candidate[:5088]) == crc_value)
            for index, bits in enumerate(prefix["bits"]):
                state = int("".join(str(int(bit)) for bit in bits[-6:]), 2)
                suffix = ordered_paths(evidence[6120:], 1, initial_state=state, payload_bits=2028, crc_initial=crc16(bits))
                complete = np.concatenate((bits, suffix["bits"][0]))
                score, end = direct_score(complete, evidence, 0, 6)
                assert end == 0 and abs(score - prefix["rank_scores"][index]) < 1e-7
                assert abs(prefix["local_scores"][index] + suffix["local_scores"][0] - score) < 1e-7
                uniform = ordered_paths(evidence[6120:], 1, prior=np.zeros((169, 4096)), initial_state=state)
                assert np.array_equal(uniform["bits"][0], suffix["bits"][0])
                boundaries += 1
            mapping = rate_match_indices(10220, 5984)
            full = listed["bits"][0]
            encoded = encode_packet(full[:5088], 2992)
            noiseless = ordered_paths(channel_evidence(encoded["symbols"], mapping, 5110, snr), 65,
                                      payload_bits=5088, stop_on_crc=True)
            assert len(noiseless["bits"]) == 1 and noiseless["accepted"][0]
            assert np.array_equal(noiseless["bits"][0][:5088], full[:5088])
            real += 1
        sources = snapshot(output, [Path(__file__), ROOT / "src/var_comm/whole_list.py", ROOT / "src/var_comm/whole_list.cpp"])
        result = {"status": "WHOLE_LIST_SELFCHECK_PASS", "exhaustive_ordered_lists": exhaustive,
                  "prefix_and_nonzero_state_boundaries": boundaries, "real_length_punctured_frames": real,
                  "source_hashes": sources, "output_hashes": artifact_hashes(output)}
        write_json(output / "selfcheck.json", result)
        print(result, flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
