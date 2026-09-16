"""The existing raw/arithmetic whole-frame PHY at explicitly charged new budgets."""

from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from var_comm.next_scale_prior import PATCH_NUMS
from var_comm.progressive import decode_header as raw_header, header_bits, split_prefix
from var_comm.scale_channel import bits_to_indices, channel_evidence, crc_accepts, decode_map, encode_packet, indices_to_bits, rate_match_indices
from var_comm.whole_entropy import decode_header as entropy_header


def transmit(source, payload, label, mode, family, budget):
    if mode not in (7, 8, 9) or family not in ("raw", "arithmetic"):
        raise ValueError("only the originally retained digital candidate set is allowed")
    raw = indices_to_bits(np.concatenate(source[:mode]))
    if family == "raw":
        data, header_uses, length_field = raw, 68, 0
        fields = header_bits(label, mode)
    else:
        data, header_uses = np.asarray(payload, dtype=np.uint8), 94
        length_field = len(data) if len(data) < len(raw) else 0
        if not length_field and not np.array_equal(data, raw):
            raise RuntimeError("archived raw-fallback payload differs from the actual source")
        fields = np.concatenate((indices_to_bits([label], 10), indices_to_bits([mode - 6], 2), indices_to_bits([length_field], 13)))
    header = encode_packet(fields, header_uses)
    body = encode_packet(data, budget - header_uses)
    signal = np.concatenate((header["symbols"], body["symbols"]))
    return signal, {"family": family, "mode": mode, "header_uses": header_uses,
                    "data_uses": budget - header_uses, "complex_uses": budget, "total_energy": float(np.square(signal).sum()),
                    "raw_payload_bits": len(raw), "actual_payload_bits": len(data), "length_field": length_field}


def receive(observed, snr, family):
    budget = len(observed)
    header_uses = 68 if family == "raw" else 94
    header = raw_header(observed[:68], snr) if family == "raw" else entropy_header(observed[:94], snr)
    if not header["accepted"]:
        return {"header": header, "label": None, "mode": None, "payload": None, "body_crc_accepted": False}
    mode = header["mode"]
    raw_bits = 12 * sum(size ** 2 for size in PATCH_NUMS[:mode])
    payload_length = raw_bits if family == "raw" else header["length_field"] or raw_bits
    information_bits = payload_length + 22
    mapping = rate_match_indices(2 * information_bits, 2 * (budget - header_uses))
    evidence = channel_evidence(observed[header_uses:], mapping, information_bits, snr)
    decoded, _score = decode_map(evidence)
    return {"header": header, "label": header["label"], "mode": mode,
            "payload": decoded[:payload_length], "body_crc_accepted": bool(crc_accepts(decoded[:-6]))}


def raw_prefix(physical):
    if physical["payload"] is None:
        return []
    return split_prefix(bits_to_indices(physical["payload"]), physical["mode"])
