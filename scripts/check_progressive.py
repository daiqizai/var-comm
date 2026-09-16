#!/usr/bin/env python3
"""Noiseless packet, header, arithmetic and trusted-prefix state-machine checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from var_comm.entropy import arithmetic_decode, arithmetic_encode, probability_cdf
from var_comm.progressive import RAW_BITS, allocation, receive_group, receive_whole, transmit_group, transmit_whole
from var_comm.study import artifact_hashes, create_output, snapshot, write_json


class ToyPrior:
    def __init__(self):
        self.calls = []

    def table(self, prefix, label):
        self.calls.append(len(prefix))
        length = (8, 10, 13)[len(prefix) - 6] ** 2
        return np.full((length, 4096), -np.log(4096)), f"toy_{len(prefix)}_{label}"

    def cdf(self, prefix, label):
        table, key = self.table(prefix, label)
        return probability_cdf(table), key


def main():
    rng = np.random.default_rng(2026090703)
    output = create_output(ROOT / "outputs/VAR-PROGRESSIVE-SELFCHECK-001")
    paths = [Path(__file__), ROOT / "src/var_comm/entropy.py", ROOT / "src/var_comm/progressive.py"]
    records = snapshot(output, paths)
    try:
        maximum_overhead = -np.inf
        for length in (1, 2, 64, 100, 169):
            for spread in (0.0, 1.0, 8.0):
                logits = rng.normal(size=(length, 4096)) * spread
                logits -= np.max(logits, axis=1, keepdims=True)
                logarithms = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
                cdf = probability_cdf(logarithms)
                tokens = rng.integers(0, 4096, length)
                stream = arithmetic_encode(tokens, cdf)
                recovered = arithmetic_decode(stream, cdf)
                if not np.array_equal(tokens, recovered) or np.any(np.diff(cdf) <= 0):
                    raise RuntimeError("arithmetic coding or full support check failed")
                quantized_nll = -np.log2(np.diff(cdf)[np.arange(length), tokens] / (1 << 24)).sum()
                maximum_overhead = max(maximum_overhead, len(stream) - quantized_nll)
        prior = ToyPrior()
        source = [rng.integers(0, 4096, size ** 2) for size in (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)]
        static = {scale: np.full(4096, -np.log(4096)) for scale in (7, 8, 9)}
        cases = 0
        for base in (720, 792, 864):
            for arm in ("group_ml", "group_static", "group_var", "group_entropy"):
                waveform, ledger = transmit_group(source, 123, base, prior, arm == "group_entropy")
                result = receive_group(waveform, 30.0, base, arm, prior, static)
                if result["last_accepted_scale"] != 9 or not all(np.array_equal(received, sent) for received, sent in zip(result["prefix"], source[:9])):
                    raise RuntimeError(f"noiseless grouped packet mismatch: {base}, {arm}")
                if ledger["total_complex_uses"] != 3060 or waveform.shape != (3060, 2):
                    raise RuntimeError("group budget changed")
                cases += 1
        for mode in (7, 8, 9):
            waveform = transmit_whole(source, 123, mode)
            result = receive_whole(waveform, 30.0)
            if result["last_accepted_scale"] != mode or not all(np.array_equal(received, sent) for received, sent in zip(result["prefix"], source[:mode])):
                raise RuntimeError("noiseless whole-frame mismatch")
        waveform, _ledger = transmit_group(source, 123, 792, prior)
        header_uses, uses = allocation(792)
        waveform[header_uses:header_uses + uses[0]] = rng.normal(size=(uses[0], 2))
        prior.calls.clear()
        result = receive_group(waveform, 7.0, 792, "group_var", prior, static)
        if result["prefix"] or prior.calls or len(result["events"]) != 1 or result["events"][0]["accepted"]:
            raise RuntimeError("failed base block entered a trusted VAR context")
        result = {"status": "PROGRESSIVE_SELFCHECK_PASS", "arithmetic_roundtrips": 15,
                  "maximum_coder_bits_above_quantized_NLL": float(maximum_overhead), "noiseless_group_cases": cases,
                  "noiseless_whole_cases": 3, "failed_base_blocks_prior_calls": "PASS",
                  "source_hashes": records, "output_hashes": artifact_hashes(output)}
        write_json(output / "selfcheck.json", result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED_CLOSED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
