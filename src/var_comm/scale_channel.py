"""Terminated convolutional PHY and exact token-factor MAP; CRC is post-detection."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment host is Linux
    fcntl = None

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = None


def binary(values):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("expected a one-dimensional binary array")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        try:
            array = array.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("expected a one-dimensional binary array") from error
        if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
            raise ValueError("expected a one-dimensional binary array")
    elif not np.isin(array, (0, 1)).all():
        raise ValueError("expected a one-dimensional binary array")
    return array.astype(np.uint8, copy=False)


def indices_to_bits(tokens, width=12):
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= 1 << width):
        raise ValueError("token index outside fixed-width alphabet")
    return ((values[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def bits_to_indices(bits, width=12):
    values = binary(bits)
    if len(values) % width:
        raise ValueError("token bit length is not a multiple of index width")
    return values.reshape(-1, width).astype(np.int64) @ (1 << np.arange(width - 1, -1, -1))


def crc16(bits):
    value = 0xFFFF
    for bit in binary(bits):
        feedback = ((value >> 15) & 1) ^ int(bit)
        value = (value << 1) & 0xFFFF
        if feedback:
            value ^= 0x1021
    return value


def append_crc(bits):
    payload = binary(bits)
    return np.concatenate((payload, indices_to_bits([crc16(payload)], 16)))


def crc_accepts(bits):
    frame = binary(bits)
    return len(frame) >= 16 and crc16(frame[:-16]) == int(bits_to_indices(frame[-16:], 16)[0])


def convolutional_encode(bits, memory=6):
    payload = binary(bits)
    if memory not in (2, 6):
        raise ValueError("only memory-2 verification and memory-6 experimental codes are defined")
    generators = (0o171, 0o133) if memory == 6 else (0o7, 0o5)
    coded = np.empty(2 * len(payload), dtype=np.uint8)
    state = 0
    for offset, bit in enumerate(payload):
        register = (state << 1) | int(bit)
        coded[2 * offset] = (register & generators[0]).bit_count() & 1
        coded[2 * offset + 1] = (register & generators[1]).bit_count() & 1
        state = register & ((1 << memory) - 1)
    return coded


def rate_match_indices(mother_length, slots):
    if mother_length <= 0 or slots <= 0 or slots % 2:
        raise ValueError("positive lengths and even transmitted coded-bit slots required")
    mapping = ((2 * np.arange(slots, dtype=np.int64) + 1) * int(mother_length)) // (2 * int(slots))
    if np.min(mapping) < 0 or np.max(mapping) >= mother_length:
        raise ValueError("rate matching index escaped the mother codeword")
    return mapping


def encode_packet(payload, complex_uses, *, with_crc=True):
    source = binary(payload)
    protected = append_crc(source) if with_crc else source
    information = np.concatenate((protected, np.zeros(6, dtype=np.uint8)))
    mother = convolutional_encode(information)
    mapping = rate_match_indices(len(mother), 2 * int(complex_uses))
    symbols = (1.0 - 2.0 * mother[mapping].astype(np.float64)).reshape(-1, 2)
    return {"information": information, "mapping": mapping, "symbols": symbols,
            "raw_payload_bits": len(source), "crc_bits": 16 if with_crc else 0,
            "tail_bits": 6, "coded_bits": len(mapping), "complex_uses": int(complex_uses)}


def channel_evidence(received, mapping, information_bits, snr_db):
    observations = np.asarray(received, dtype=np.float64).ravel()
    mapping = np.asarray(mapping, dtype=np.int64)
    if observations.shape != mapping.shape or not np.isfinite(observations).all():
        raise ValueError("observation/map mismatch or nonfinite received signal")
    if mapping.min() < 0 or mapping.max() >= 2 * information_bits:
        raise ValueError("rate matching map escaped the trellis")
    return np.bincount(mapping, weights=observations, minlength=2 * information_bits) * (10.0 ** (float(snr_db) / 10.0))


def load_native():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name("token_trellis.cpp")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = ROOT / "outputs" / "build" / source_hash[:16]
    if not destination.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("native build escaped VAR_COMM outputs")
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "token_trellis.so"
    descriptor_path = destination / "build.json"
    lock_path = destination / "build.lock"

    def valid_build():
        if not library.is_file() or not descriptor_path.is_file():
            return False
        try:
            descriptor = json.loads(descriptor_path.read_text())
            return (descriptor.get("source_sha256") == source_hash and
                    descriptor.get("library_sha256") == hashlib.sha256(library.read_bytes()).hexdigest())
        except (OSError, ValueError, KeyError):
            return False

    with lock_path.open("a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # A process can die after compiling but before the atomic replace.
            # Remove those abandoned private build directories while holding
            # the same lock used by builders; never touch the live artefact.
            for stale in destination.glob(".build-*"):
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
            if not valid_build():
                temp = Path(tempfile.mkdtemp(prefix=f".build-{os.getpid()}-", dir=str(destination)))
                temp_library = temp / "token_trellis.so"
                command = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared", str(source), "-o", str(temp_library)]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    descriptor = {"source_sha256": source_hash, "command": command,
                                  "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                                  "library_sha256": hashlib.sha256(temp_library.read_bytes()).hexdigest()}
                    temp_descriptor = temp / "build.json"
                    temp_descriptor.write_text(json.dumps(descriptor, indent=2) + "\n")
                    os.replace(temp_library, library)
                    os.replace(temp_descriptor, descriptor_path)
                finally:
                    for child in temp.glob("*"):
                        child.unlink(missing_ok=True)
                    temp.rmdir()
            if not valid_build():
                raise RuntimeError("native build integrity mismatch")
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    _LIBRARY = ctypes.CDLL(str(library))
    floating = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integral = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    _LIBRARY.decode_token_map.argtypes = [ctypes.c_int] * 4 + [floating, floating, ctypes.c_int, integral, floating]
    _LIBRARY.decode_token_map.restype = ctypes.c_int
    return _LIBRARY


def decode_map(evidence, log_prior=None, *, memory=6, final_state=0):
    observations = np.ascontiguousarray(evidence, dtype=np.float64)
    if observations.ndim != 1 or len(observations) % 2 or not np.isfinite(observations).all():
        raise ValueError("expected finite, flat mother-code evidence")
    if memory not in (2, 6):
        raise ValueError("unsupported convolutional memory")
    length = len(observations) // 2
    if log_prior is None:
        token_count, stride = 0, 0
        prior = np.zeros(1 << (2 * memory), dtype=np.float64)
    else:
        prior = np.ascontiguousarray(log_prior, dtype=np.float64)
        if prior.ndim != 2 or prior.shape[1] != 1 << (2 * memory) or not np.isfinite(prior).all():
            raise ValueError("expected a finite complete log-prior table for each token")
        token_count, stride = prior.shape[0], prior.shape[1]
        if token_count * 2 * memory > length:
            raise ValueError("prior extends beyond the information bitstream")
    decoded = np.empty(length, dtype=np.uint8)
    score = np.zeros(1, dtype=np.float64)
    status = load_native().decode_token_map(memory, token_count, length, final_state, observations, prior, stride, decoded, score)
    if status != 0:
        raise RuntimeError(f"native trellis decoder failed: {status}")
    return decoded, float(score[0])


def path_score(bits, evidence, log_prior=None, memory=6):
    candidate = binary(bits)
    value = -float(np.dot(1.0 - 2.0 * convolutional_encode(candidate, memory), evidence))
    if log_prior is not None:
        table = np.asarray(log_prior)
        tokens = bits_to_indices(candidate[:len(table) * 2 * memory], 2 * memory)
        value -= float(table[np.arange(len(tokens)), tokens].sum())
    return value


def bit_marginal_table(log_prior):
    logarithms = np.asarray(log_prior, dtype=np.float64)
    if logarithms.ndim != 2 or logarithms.shape[0] == 0 or logarithms.shape[1] != 4096 or not np.isfinite(logarithms).all():
        raise ValueError("expected full 12-bit token distributions")
    normalizer = np.max(logarithms, axis=1) + np.log(np.exp(logarithms - np.max(logarithms, axis=1, keepdims=True)).sum(axis=1))
    if not np.isfinite(normalizer).all() or np.max(np.abs(normalizer)) > 1e-5:
        raise ValueError("log-prior rows must be finite and normalized")
    table = np.zeros_like(logarithms)
    values = np.arange(4096)
    def logsumexp(values):
        maximum = np.max(values, axis=1)
        return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    for shift in range(11, -1, -1):
        selected = ((values >> shift) & 1).astype(bool)
        table[:, selected] += logsumexp(logarithms[:, selected])[:, None]
        table[:, ~selected] += logsumexp(logarithms[:, ~selected])[:, None]
    return table
