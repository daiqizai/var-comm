"""Validation and paired aggregation for evaluation result tables.

The validator deliberately works on row identity before doing any means.  This
prevents a missing seed, a fractional SNR, or a mixed decoder/protocol scope
from silently changing the estimand.  It is usable for the original
development table and for follow-up/mechanism tables with the same row
contract.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


METRICS = ("psnr_db", "lpips_alex", "dino_cosine")
SAMPLE_FIELDS = ("source_index", "snr_db", "noise_seed")
SCOPE_FIELDS = (
    "population_role", "family", "N", "renderer", "model_sha256", "decoder_sha256",
    "protocol_id", "noise_model", "budget_scope",
)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, object], field: str, *, integer: bool = False) -> float | int:
    if field not in row or row[field] in (None, ""):
        raise ValueError(f"missing {field}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {row[field]!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    if integer and not value.is_integer():
        raise ValueError(f"noninteger {field}: {value}")
    return int(value) if integer else value


def _seed_field(row: Mapping[str, object]) -> str:
    if "noise_seed" in row:
        return "noise_seed"
    if "seed" in row:
        return "seed"
    raise ValueError("missing noise_seed/seed")


def sample_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Return the complete source/SNR/noise identity for one result row."""
    seed_field = _seed_field(row)
    source = _number(row, "source_index", integer=True)
    snr = _number(row, "snr_db")
    seed = _number(row, seed_field, integer=True)
    image_id = str(row.get("image_id", ""))
    if not image_id:
        raise ValueError("missing image_id; source_index alone is not an identity")
    return image_id, source, snr, seed


def scope_key(row: Mapping[str, object], fields: Sequence[str] = SCOPE_FIELDS) -> tuple[tuple[str, str], ...]:
    """Return the explicit execution scope; absent fields are an error.

    Every table must record these fields (or pass the same values in
    ``scope`` to :func:`validate_rows`).  Treating absent values as ``None``
    would allow rows from different Decoder/protocol runs to be merged.
    """
    result = []
    for field in fields:
        if field not in row or row[field] in (None, ""):
            raise ValueError(f"missing execution scope field {field}")
        result.append((field, str(row[field])))
    return tuple(result)


def validate_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_methods: Sequence[str] | None = None,
    expected_sources: Sequence[int] | None = None,
    expected_snrs: Sequence[float] | None = None,
    expected_seeds: Sequence[int] | None = None,
    scope: Mapping[str, object] | None = None,
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, object]]:
    """Validate complete sample coverage and return normalized rows.

    Validation is exact: duplicates, missing cells, unsupported/fractional SNRs,
    nonfinite metrics, and mixed run scope all fail before aggregation.
    """
    normalized: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    methods = set(expected_methods) if expected_methods is not None else None
    sources = set(int(value) for value in expected_sources) if expected_sources is not None else None
    snrs = set(float(value) for value in expected_snrs) if expected_snrs is not None else None
    seeds = set(int(value) for value in expected_seeds) if expected_seeds is not None else None
    expected_scope = None
    method_scopes: dict[str, tuple[tuple[str, str], ...]] = {}
    if scope is not None:
        expected_scope = tuple((field, str(scope[field])) for field in SCOPE_FIELDS if field in scope)
        if len(expected_scope) != len(scope):
            raise ValueError("scope contains unsupported or missing fields")
    for original in rows:
        row = dict(original)
        method = str(row.get("method", ""))
        if not method:
            raise ValueError("missing method")
        if methods is not None and method not in methods:
            raise ValueError(f"unexpected method {method}")
        key = sample_key(row)
        _image_id, source, snr, seed = key
        if sources is not None and source not in sources:
            raise ValueError(f"unexpected source_index {source}")
        if snrs is not None and snr not in snrs:
            raise ValueError(f"unexpected SNR {snr}; fractional values are not truncated")
        if seeds is not None and seed not in seeds:
            raise ValueError(f"unexpected noise seed {seed}")
        if expected_scope is not None:
            actual_scope = scope_key(row, [field for field, _ in expected_scope])
            if actual_scope != expected_scope:
                raise ValueError(f"mixed execution scope for {method}: {actual_scope!r}")
        else:
            if any(field not in row or row[field] in (None, "") for field in SCOPE_FIELDS):
                raise ValueError("missing execution scope; decoder/protocol/noise identity is required")
            actual_scope = scope_key(row)
            previous_scope = method_scopes.setdefault(method, actual_scope)
            if actual_scope != previous_scope:
                raise ValueError(f"mixed execution scope for method {method}")
        cell = (method,) + key
        if cell in seen:
            raise ValueError(f"duplicate sample cell {cell!r}")
        seen.add(cell)
        for metric in metrics:
            row[metric] = _number(row, metric)
        row["source_index"], row["snr_db"], row["noise_seed"] = int(source), float(snr), int(seed)
        normalized.append(row)
    if expected_methods is not None and sources is not None and snrs is not None and seeds is not None:
        expected_count = len(methods) * len(sources) * len(snrs) * len(seeds)
        if len(seen) != expected_count:
            raise ValueError(f"incomplete sample grid: got {len(seen)}, expected {expected_count}")
    return normalized


def aggregate_source_snr(rows: Iterable[Mapping[str, object]], metrics: Sequence[str] = METRICS) -> list[dict[str, object]]:
    """Average noise seeds only, retaining method/source/SNR and scope."""
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), str(row["image_id"]), int(row["source_index"]), float(row["snr_db"])) + scope_key(row)
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: item[0][:4]):
        method, image_id, source, snr = key[:4]
        row = {"method": method, "image_id": image_id, "source_index": source, "snr_db": snr,
               **dict(key[4:])}
        for metric in metrics:
            row[metric] = float(np.mean([float(value[metric]) for value in values]))
        row["noise_replicates"] = len(values)
        output.append(row)
    return output


def aggregate_source(rows: Iterable[Mapping[str, object]], metrics: Sequence[str] = METRICS) -> list[dict[str, object]]:
    """Average SNR cells after seed aggregation, retaining source identity."""
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), str(row["image_id"]), int(row["source_index"])) + scope_key(row)
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: item[0][:3]):
        method, image_id, source = key[:3]
        row = {"method": method, "image_id": image_id, "source_index": source, **dict(key[3:])}
        for metric in metrics:
            row[metric] = float(np.mean([float(value[metric]) for value in values]))
        row["snr_replicates"] = len(values)
        output.append(row)
    return output


def paired_source_differences(rows: Iterable[Mapping[str, object]], reference: str,
                              metrics: Sequence[str] = METRICS) -> dict[str, np.ndarray]:
    """Compute paired method-reference differences on the same source keys."""
    source_rows = aggregate_source(rows, metrics)
    grouped: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in source_rows:
        # Execution scope is retained on each aggregate, but pairing is across
        # methods at the same source identity.  Different methods necessarily
        # have different family/budget/Decoder scopes and must still be paired.
        key = (str(row["image_id"]), int(row["source_index"]))
        if str(row["method"]) in grouped[key]:
            raise ValueError(f"duplicate method scope for source {key!r}")
        grouped[key][str(row["method"])] = row
    methods = sorted({str(row["method"]) for row in source_rows} - {reference})
    result = {}
    for method in methods:
        differences = []
        for key, values in grouped.items():
            if reference not in values or method not in values:
                raise ValueError(f"unpaired source {key!r} for {method} vs {reference}")
            differences.append([float(values[method][metric]) - float(values[reference][metric]) for metric in metrics])
        result[method] = np.asarray(differences, dtype=np.float64)
    return result
