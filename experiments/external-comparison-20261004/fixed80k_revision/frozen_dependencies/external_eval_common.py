"""CPU-only contracts and atomic caches for the registered external comparison."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

SNRS = (1, 7, 13)
SEEDS = (2001, 2002, 2003)
BUDGETS = (1024, 2048)
METHODS = ("SwinJSCC_new_shared", "HiFiDiffCom_SwinJSCC")
SOURCES = 100
PHYSICAL_FRAMES = 1800
ROWS = 3600
RGB_DOMAIN = b"float32:3,256,256:RGB\0"


class PauseRequested(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def seal(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise RuntimeError("Immutable external evaluation receipt differs: " + str(path))
    if not path.exists():
        write(path, value)


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise RuntimeError("Registered external evaluation input changed: " + str(path))


def pixels(value):
    a = np.asarray(value)
    if a.dtype != np.float32 or a.shape != (3, 256, 256):
        raise ValueError("Exact CHW float32 RGB256 required")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Nonfinite/out-of-domain reconstructed RGB")
    return np.ascontiguousarray(a)


def rgb_sha(value):
    return hashlib.sha256(RGB_DOMAIN + pixels(value).tobytes()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    domain = (str(a.dtype) + ":" + str(tuple(a.shape)) + "\0").encode()
    return hashlib.sha256(domain + a.tobytes()).hexdigest()


def atomic_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return sha(path)


def frame_specs(source_index):
    if type(source_index) is not int or not 0 <= source_index < SOURCES:
        raise ValueError("Unregistered source index")
    return [dict(source_index=source_index, N=n, snr_db=snr, noise_seed=seed)
            for n in BUDGETS for snr in SNRS for seed in SEEDS]


def frame_key(spec):
    return "source{source_index:04d}_N{N}_snr{snr_db}_seed{noise_seed}".format(**spec)


def row_id(spec, method):
    if method not in METHODS:
        raise ValueError("Unregistered external method")
    return identity(dict(study="EXTERNAL_SWIN_HIFI_20261004", **spec, method=method))


def expected_ids(source_index):
    return [row_id(spec, method) for spec in frame_specs(source_index) for method in METHODS]


def validate_full_sampler(receipt, accepted):
    if receipt.get("header_accepted") is not accepted:
        raise RuntimeError("Swin/HiFi header decision differs")
    if not accepted:
        if receipt.get("NFE") != 0 or receipt.get("fallback") != "fixed_gray_0.5":
            raise RuntimeError("Rejected header must bypass generation and use fixed gray")
        return
    if (receipt.get("diagnostic_probe") is not False
            or receipt.get("complete_author_schedule") is not True
            or receipt.get("model_parameters_unchanged") is not True
            or receipt.get("parameter_gradients_accumulated") is not False
            or receipt.get("schedule_mode") != "actual_data_cbr"
            or type(receipt.get("NFE")) is not int
            or receipt["NFE"] != receipt.get("t_start") or receipt["NFE"] < 2):
        raise RuntimeError("Main evaluation requires every registered posterior reverse step")


def audit_received_context(tx_power, tx_indices, rx):
    """Offline audit only. Its output must never control a receiver or fallback."""
    exact = bool(rx.accepted and np.float32(tx_power).tobytes() == np.float32(rx.power).tobytes()
                 and tuple(tx_indices) == tuple(rx.indices))
    return {"offline_metadata_exact": exact,
            "offline_false_accept": bool(rx.accepted and not exact),
            "audit_used_to_control_receiver": False}


def validate_frame_receipt(value, binding, spec, source):
    if value.get("binding") != binding or value.get("frame") != spec:
        raise RuntimeError("Frame resume identity differs")
    if value.get("payload_sha256") != identity({k: v for k, v in value.items() if k != "payload_sha256"}):
        raise RuntimeError("Frame receipt checksum differs")
    if value.get("source_id") != source["image_id"] or value.get("reference_sha256") != rgb_sha(source["rgb"]):
        raise RuntimeError("Frame source pixels/identity differ")
    rows = value["rows"]
    if [r["replay_row_id"] for r in rows] != [row_id(spec, m) for m in METHODS]:
        raise RuntimeError("Frame does not contain its exact paired methods")
    if sha(value["archive"]) != value["archive_sha256"]:
        raise RuntimeError("Committed physical frame archive changed")
    with np.load(value["archive"], allow_pickle=False) as archive:
        if set(archive.files) != {"images", "observed", "transmitted"}:
            raise RuntimeError("Physical frame cache schema differs")
        images, observed, signal = archive["images"], archive["observed"], archive["transmitted"]
        if images.shape != (2, 3, 256, 256):
            raise RuntimeError("Physical frame cached shape differs")
        if observed.shape != (spec["N"], 2) or signal.shape != (spec["N"], 2):
            raise RuntimeError("Physical waveform count differs")
        if array_sha(observed) != value["observed_sha256"] or array_sha(signal) != value["transmitted_sha256"]:
            raise RuntimeError("Paired physical waveform changed")
        if not np.isfinite(observed).all() or abs(float(np.square(signal).sum()) - 2 * spec["N"]) > .02:
            raise RuntimeError("Physical waveform energy/finite check differs")
        for i, row in enumerate(rows):
            if row.get("method") != METHODS[i] or any(row.get(k) != v for k, v in spec.items()):
                raise RuntimeError("Scientific frame fields do not match the actual pair")
            if rgb_sha(images[i]) != row["image_sha256"] or row["observed_sha256"] != value["observed_sha256"]:
                raise RuntimeError("Scored pixels or shared observation identity differ")
            if (type(row.get("header_accepted")) is not bool
                    or row["header_accepted"] != rows[0]["header_accepted"]
                    or row.get("audit_used_to_control_receiver") is not False):
                raise RuntimeError("Receiver decisions or offline-only audit boundary differ")
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                   for k in ("TX_seconds", "RX_seconds")):
                raise RuntimeError("Actual TX/RX timing must be finite and nonnegative")
            if not row["header_accepted"] and not np.all(images[i] == np.float32(.5)):
                raise RuntimeError("Failed header reconstruction differs from registered gray")
        validate_full_sampler(value["hifi_receipt"], rows[0]["header_accepted"])
    return value


def validate_source(value, binding, source_index):
    if (value.get("binding") != binding or value.get("source_index") != source_index
            or value.get("payload_sha256") != identity({k: v for k, v in value.items() if k != "payload_sha256"})):
        raise RuntimeError("Source checkpoint identity/checksum differs")
    rows = value["rows"]
    if [r["replay_row_id"] for r in rows] != expected_ids(source_index):
        raise RuntimeError("Source checkpoint lacks exact 36-row coverage")
    verify(value["frame_bindings"])
    proof = value["float_reconstructions"]
    if sha(proof["path"]) != proof["sha256"]:
        raise RuntimeError("Source floating reconstruction archive changed")
    with np.load(proof["path"], allow_pickle=False) as data:
        if set(data.files) != {"images", "source_rgb", "row_ids", "image_slots"}:
            raise RuntimeError("Source floating cache schema differs")
        if data["row_ids"].tolist() != expected_ids(source_index):
            raise RuntimeError("Source cache row IDs differ")
        images, target, slots = data["images"], pixels(data["source_rgb"]), data["image_slots"].tolist()
        if slots != proof["image_slots"] or len(slots) != 36 or any(type(s) is not int or not 0 <= s < len(images) for s in slots):
            raise RuntimeError("Source reconstruction slots differ")
        for row, slot in zip(rows, slots):
            if row["image_sha256"] != rgb_sha(images[slot]) or row["reference_sha256"] != rgb_sha(target):
                raise RuntimeError("Source cache differs from scientific pixels")
    return value


def gpu_available():
    output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    entries = [s.strip() for s in output.splitlines() if s.strip()]
    if any(not s.isdecimal() for s in entries):
        raise RuntimeError("Cannot determine GPU ownership")
    if set(map(int, entries)) - {os.getpid()}:
        raise PauseRequested("GPU has another owner; checkpointed work is preserved")


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise RuntimeError("Cannot emit an empty scientific table")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
