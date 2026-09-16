"""Positive-frequency, finite-precision arithmetic coding of complete token priors."""

from __future__ import annotations

import numpy as np

TOTAL = 1 << 24
FULL = 1 << 32
HALF = FULL >> 1
QUARTER = FULL >> 2


def probability_cdf(log_probs):
    logarithms = np.asarray(log_probs, dtype=np.float64)
    if logarithms.ndim != 2 or logarithms.shape[1] != 4096 or not np.isfinite(logarithms).all():
        raise ValueError("expected a complete finite 4096-symbol table")
    mass = np.exp(logarithms)
    mass /= mass.sum(axis=1, keepdims=True)
    counts = np.floor(mass * (TOTAL - 4096)).astype(np.int64) + 1
    counts[np.arange(len(counts)), np.argmax(mass, axis=1)] += TOTAL - counts.sum(axis=1)
    return np.concatenate((np.zeros((len(counts), 1), dtype=np.int64), counts.cumsum(axis=1)), axis=1)


def arithmetic_encode(tokens, cdf):
    values = np.asarray(tokens, dtype=np.int64)
    if cdf.shape != (len(values), 4097) or np.any(values < 0) or np.any(values >= 4096):
        raise ValueError("token/CDF shape or alphabet mismatch")
    low, high, pending = 0, FULL - 1, 0
    bits = []

    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0

    for token, cumulative in zip(values, cdf):
        interval = high - low + 1
        high = low + interval * int(cumulative[token + 1]) // TOTAL - 1
        low = low + interval * int(cumulative[token]) // TOTAL
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
    pending += 1
    emit(0 if low < QUARTER else 1)
    return np.asarray(bits, dtype=np.uint8)


def arithmetic_decode(bits, cdf):
    stream = np.asarray(bits, dtype=np.uint8)
    if stream.ndim != 1 or np.any(stream > 1):
        raise ValueError("expected a binary arithmetic bitstream")
    position = 0

    def read_bit():
        nonlocal position
        bit = int(stream[position]) if position < len(stream) else 0
        position += 1
        return bit

    low, high, value = 0, FULL - 1, 0
    for _bit in range(32):
        value = value * 2 + read_bit()
    result = []
    for cumulative in cdf:
        interval = high - low + 1
        scaled = ((value - low + 1) * TOTAL - 1) // interval
        token = int(np.searchsorted(cumulative, scaled, side="right") - 1)
        if token < 0 or token >= 4096:
            raise ValueError("invalid arithmetic decoder state")
        result.append(token)
        high = low + interval * int(cumulative[token + 1]) // TOTAL - 1
        low = low + interval * int(cumulative[token]) // TOTAL
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                value -= QUARTER
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low *= 2
            high = high * 2 + 1
            value = value * 2 + read_bit()
    return np.asarray(result, dtype=np.int64)
