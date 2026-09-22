"""Exact ordered paths on a conditional trellis; finite source hypotheses stay separate."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np

from .study import ROOT, sha256, write_json

_LIBRARY = None
COUNTERS = ("channel_products", "backward_edges", "deviation_edges", "heap_pushes", "heap_pops",
            "heap_comparisons", "traceback_layers", "peak_heap_nodes", "native_owned_peak_bytes",
            "setup_seconds", "native_seconds", "stop_reason", "candidates", "deadline_overrun_seconds")


def load_list_native():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_suffix(".cpp")
    digest = sha256(source)
    directory = ROOT / "outputs/build" / ("whole_list_" + digest[:16])
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "whole_list.so"
    if not library.exists():
        command = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared", str(source), "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        write_json(directory / "build.json", {"source_sha256": digest, "library_sha256": sha256(library), "command": command,
                                              "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]})
    record = json.loads((directory / "build.json").read_text())
    if record["source_sha256"] != digest or record["library_sha256"] != sha256(library):
        raise RuntimeError("native list build changed")
    _LIBRARY = ctypes.CDLL(str(library))
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    floating = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    binary = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    _LIBRARY.enumerate_whole_paths.argtypes = [integer, floating, floating, ctypes.c_double, binary, floating, binary, floating]
    _LIBRARY.enumerate_whole_paths.restype = ctypes.c_int
    return _LIBRARY


def ordered_paths(evidence, maximum, *, prior=None, initial_state=0, final_state=0, stop_bits=None,
                  memory=6, payload_bits=None, crc_initial=65535, stop_on_crc=False,
                  seconds=-1.0, queue_limit=8000000):
    evidence = np.ascontiguousarray(evidence, dtype=np.float64)
    if evidence.ndim != 1 or len(evidence) % 2 or not np.isfinite(evidence).all() or memory not in (2, 6):
        raise ValueError("invalid mother-code evidence or memory")
    length = len(evidence) // 2
    output_bits = length if stop_bits is None else int(stop_bits)
    if maximum < 1 or not 0 < output_bits <= length or not 0 <= initial_state < 1 << memory or not -1 <= final_state < 1 << memory:
        raise ValueError("invalid list size, boundary or encoder state")
    if prior is None:
        tokens, table = 0, np.zeros(1, dtype=np.float64)
    else:
        table = np.ascontiguousarray(prior, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != 1 << (2 * memory) or not np.isfinite(table).all():
            raise ValueError("expected full finite token priors")
        tokens = len(table)
        if tokens * 2 * memory > length:
            raise ValueError("token prior extends beyond payload")
    if payload_bits is not None and (payload_bits + 16 + memory != output_bits or output_bits != length):
        raise ValueError("CRC/tail must belong to the complete decoded suffix")
    settings = np.array([memory, tokens, length, initial_state, final_state, output_bits, maximum,
                         -1 if payload_bits is None else payload_bits, crc_initial, int(stop_on_crc), queue_limit], dtype=np.int32)
    bits = np.empty((maximum, output_bits), dtype=np.uint8)
    scores = np.empty((maximum, 2), dtype=np.float64)
    accepted = np.empty(maximum, dtype=np.uint8)
    counters = np.empty(len(COUNTERS), dtype=np.float64)
    returned = load_list_native().enumerate_whole_paths(settings, evidence, table, seconds, bits, scores, accepted, counters)
    if returned < 0:
        raise RuntimeError(f"native conditional list decoder failed: {returned}")
    stats = {name: (float(value) if "seconds" in name else int(value)) for name, value in zip(COUNTERS, counters)}
    return {"bits": bits[:returned].copy(), "rank_scores": scores[:returned, 0].copy(),
            "local_scores": scores[:returned, 1].copy(), "accepted": accepted[:returned].astype(bool), "stats": stats}
