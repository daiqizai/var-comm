"""Positive-frequency, finite-precision arithmetic coding of complete token priors."""

from __future__ import annotations

import numpy as np

TOTAL = 1 << 24
FULL = 1 << 32
HALF = FULL >> 1
QUARTER = FULL >> 2


def validate_bits(bits, *, name="bits"):
    values = np.asarray(bits)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(values.dtype, np.floating):
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")
    elif np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0/1 values")
    else:
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain only finite 0/1 values") from error
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")
        values = numeric
    return values.astype(np.uint8, copy=False)


def validate_cdf(cdf, *, rows=None):
    values = np.asarray(cdf)
    if values.ndim != 2 or values.shape[1] != 4097:
        raise ValueError("CDF must have shape [tokens, 4097]")
    if rows is not None and values.shape[0] != int(rows):
        raise ValueError("CDF/token row count mismatch")
    if not np.issubdtype(values.dtype, np.integer):
        try:
            finite = np.isfinite(values).all()
            integral = np.equal(values, np.floor(values)).all()
        except (TypeError, ValueError):
            raise ValueError("CDF values must be finite integers")
        if not finite or not integral:
            raise ValueError("CDF values must be finite integers")
        try:
            values = values.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("CDF values must be finite integers") from error
    else:
        values = values.astype(np.int64, copy=False)
    if values.shape[0] == 0:
        return values
    if np.any(values[:, 0] != 0) or np.any(values[:, -1] != TOTAL):
        raise ValueError("CDF must start at 0 and end at TOTAL")
    if np.any(values < 0) or np.any(values > TOTAL) or np.any(np.diff(values, axis=1) <= 0):
        raise ValueError("CDF must be strictly increasing within [0, TOTAL]")
    return values


def probability_cdf(log_probs):
    logarithms = np.asarray(log_probs, dtype=np.float64)
    if logarithms.ndim != 2 or logarithms.shape[1] != 4096 or not np.isfinite(logarithms).all():
        raise ValueError("expected a complete finite 4096-symbol table")
    shifted = logarithms - np.max(logarithms, axis=1, keepdims=True)
    mass = np.exp(shifted)
    mass /= mass.sum(axis=1, keepdims=True)
    counts = np.floor(mass * (TOTAL - 4096)).astype(np.int64) + 1
    counts[np.arange(len(counts)), np.argmax(mass, axis=1)] += TOTAL - counts.sum(axis=1)
    cdf = np.concatenate((np.zeros((len(counts), 1), dtype=np.int64), counts.cumsum(axis=1)), axis=1)
    return validate_cdf(cdf)


def arithmetic_encode(tokens, cdf):
    raw_values = np.asarray(tokens)
    if raw_values.ndim != 1:
        raise ValueError("tokens must be finite integers")
    if not np.issubdtype(raw_values.dtype, np.integer):
        try:
            numeric = raw_values.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("tokens must be finite integers") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("tokens must be finite integers")
        raw_values = numeric
    values = raw_values.astype(np.int64, copy=False)
    cdf = validate_cdf(cdf, rows=len(values))
    if np.any(values < 0) or np.any(values >= 4096):
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
    stream = validate_bits(bits, name="arithmetic bitstream")
    cdf = validate_cdf(cdf)
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
