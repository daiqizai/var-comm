"""Small, local-only helpers for frozen experiment artifacts and image pairing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to report an empty experiment")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_output(path):
    output = Path(path).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("new outputs must physically remain under VAR_COMM/outputs")
    output.mkdir(parents=True, exist_ok=False)
    return output


def snapshot(output, paths):
    records = {}
    for source in paths:
        source = Path(source).resolve()
        relative = str(source.relative_to(ROOT))
        target = output / "snapshots" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[relative] = sha256(source)
    return records


def verify_snapshot(records):
    for relative, expected in records.items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed during experiment: {relative}")


def artifact_hashes(output):
    return {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}


def verify_artifacts(directory, receipt, expected_sha=None):
    directory = Path(directory)
    if expected_sha is not None and sha256(directory / receipt) != expected_sha:
        raise RuntimeError("input receipt SHA mismatch")
    record = json.loads((directory / receipt).read_text())
    for relative, expected in record["output_hashes"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise RuntimeError(f"input artifact mismatch: {relative}")
    return record


def seeded_noise(image_id, seed, shape):
    key = hashlib.sha256(f"{image_id}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big")).standard_normal(shape)


def paired_interval(values, seed, resamples):
    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("expected finite paired image differences")
    indices = np.random.default_rng(seed).integers(len(differences), size=(resamples, len(differences)))
    lower, upper = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    return {"gain": float(differences.mean()), "ci_low": float(lower), "ci_high": float(upper)}
