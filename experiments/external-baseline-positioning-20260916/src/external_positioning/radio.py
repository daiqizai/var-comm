"""Finite metadata packets using the already audited convolutional PHY."""

import math
from pathlib import Path
import struct
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from var_comm.scale_channel import channel_evidence, crc_accepts, decode_map, encode_packet, rate_match_indices


def integer_bits(value, width):
    if not 0 <= value < 1 << width:
        raise ValueError("integer does not fit the registered field")
    return np.asarray([(value >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def bits_integer(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def rank_subset(indices, width):
    indices = sorted(int(index) for index in indices)
    if len(set(indices)) != len(indices) or any(not 0 <= index < width for index in indices):
        raise ValueError("invalid active-channel set")
    return sum(math.comb(index, ordinal + 1) for ordinal, index in enumerate(indices))


def unrank_subset(rank, width, count):
    if not 0 <= rank < math.comb(width, count):
        raise ValueError("received active-channel rank is not legal")
    selected = []
    upper = width - 1
    for ordinal in range(count, 0, -1):
        lower, high = ordinal - 1, upper
        while lower < high:
            middle = (lower + high + 1) // 2
            if math.comb(middle, ordinal) <= rank:
                lower = middle
            else:
                high = middle - 1
        selected.append(lower)
        rank -= math.comb(lower, ordinal)
        upper = lower - 1
    return sorted(selected)


def metadata_layout(width=None, count=None):
    subset_bits = 0 if width is None else (math.comb(width, count) - 1).bit_length()
    payload_bits = 32 + subset_bits
    return {"power_bits": 32, "subset_bits": subset_bits, "payload_bits": payload_bits,
            "crc_bits": 16, "tail_bits": 6, "complex_uses": 2 * (payload_bits + 22)}


def encode_metadata(power, indices=None, width=None):
    power = float(np.float32(power))
    if not math.isfinite(power) or power <= 0:
        raise ValueError("transmitted normalization power must be positive and finite")
    power_integer = int.from_bytes(struct.pack(">f", power), "big")
    count = len(indices) if indices is not None else None
    layout = metadata_layout(width, count)
    fields = [integer_bits(power_integer, 32)]
    if indices is not None:
        fields.append(integer_bits(rank_subset(indices, width), layout["subset_bits"]))
    payload = np.concatenate(fields)
    packet = encode_packet(payload, layout["complex_uses"])
    return {**packet, **layout, "payload": payload}


def decode_metadata(received, snr_db, width=None, count=None):
    layout = metadata_layout(width, count)
    information_bits = layout["payload_bits"] + 22
    if np.asarray(received).shape != (layout["complex_uses"], 2):
        raise ValueError("metadata observation length mismatch")
    mapping = rate_match_indices(2 * information_bits, 2 * layout["complex_uses"])
    evidence = channel_evidence(received, mapping, information_bits, snr_db)
    decoded, score = decode_map(evidence)
    payload = decoded[:layout["payload_bits"]]
    power = struct.unpack(">f", bits_integer(payload[:32]).to_bytes(4, "big"))[0]
    indices = None
    legal = math.isfinite(power) and power > 0
    if width is not None:
        try:
            indices = unrank_subset(bits_integer(payload[32:]), width, count)
        except ValueError:
            legal = False
    return {"power": power, "indices": indices, "usable": bool(legal),
            "crc_accepted": bool(crc_accepts(decoded[:-6])), "path_score": score,
            "payload": payload, "complex_uses": layout["complex_uses"]}
