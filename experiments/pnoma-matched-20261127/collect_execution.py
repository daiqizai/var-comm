"""Verify task receipts and index the regular files used by this comparison."""
from pathlib import Path
import hashlib
import json
import stat
import shutil

from launch import ROOT, HERE, TEMP


def digest(path):
    assert stat.S_ISREG(path.lstat().st_mode) and path.stat().st_nlink == 1, path
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    out = HERE / "execution.json"
    assert not out.exists()
    ledger = {}
    for role in ("train", "evaluate", "solo"):
        path = TEMP / ("job_" + role)
        registration = json.loads((path/"registration.json").read_text())
        pid = json.loads((path/"pid.json").read_text())
        wait = json.loads((path/"wait.json").read_text())
        assert wait["actual_wait"] == 0 and 0 < wait["elapsed_s"] < registration["seconds_limit"]
        proc = Path(f"/proc/{pid['child']}/stat")
        assert not proc.exists() or proc.read_text().split()[21] != str(pid["start_ticks"])
        assert all(digest(HERE/f) == sig for f, sig in registration["source"].items())
        ledger[role] = dict(registration=registration, process=pid, wait=wait,
                            done=json.loads((HERE/role/"done.json").read_text()))
    assert (TEMP/"job_train/wait.json").stat().st_mtime <= ledger["evaluate"]["registration"]["started"]
    assert (TEMP/"job_evaluate/wait.json").stat().st_mtime <= ledger["solo"]["registration"]["started"]
    assert (HERE/"solo_protocol.json").stat().st_mtime < ledger["evaluate"]["registration"]["started"]
    train = json.loads((HERE/"train/sources.json").read_text())
    test = json.loads((HERE/"evaluate/sources.json").read_text())
    a, b, c = (set(train["train_pixels"]), set(train["val_pixels"]), set(test["pixels"]))
    assert not a & b and not a & c and not b & c
    assert len(train["train_ids"]) == 20000 and len(train["val_ids"]) == 256 and len(c) == 500
    modeldir = ROOT/"liberai-checkpoint/liulu/var_comm/pnoma_matched_20261127_v1"
    checkpoints = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(modeldir.glob("*.pt"))}
    vendor = TEMP/"upstream/vendor_identity.json"
    destination = HERE/"vendor_identity.json"
    if not destination.exists():
        shutil.copyfile(vendor, destination)
    assert digest(destination) == digest(vendor)
    metadata = dict(parent=json.loads((HERE/"protocol.json").read_text())["parent"],roles=ledger,
        gpu_device_hours=sum(r["wait"]["elapsed_s"] for r in ledger.values()) / 3600,
        resource_definition="One assigned A800 times each GPU worker wall time, including startup and I/O. No claim of exclusive node time.",
        counts=dict(train_sources=20000, validation_sources=256, test_sources=500,
                    unique_train_pixels=len(a), unique_validation_pixels=len(b),
                    registered_logical_external_frames=750, new_primary_external_source_reconstructions=3000,
                    local_collision_score_rows_reused=2250, primary_timed_receiver_calls=256,
                    timing_warmups=12, supplemental_single_active_frames=500,
                    supplemental_all_active_rows_reused=500),
        training_source_manifest_sha256=digest(HERE/"train/sources.json"), test_source_manifest_sha256=digest(HERE/"evaluate/sources.json"),
        vendor_manifest_sha256=digest(vendor), training_data_shared_with_test=0, val_data_shared_with_test=0,
        weight_location=str(modeldir), weights=checkpoints,
        diagnostic_protocol_mtime=(HERE/"solo_protocol.json").stat().st_mtime,
        formal_registration_unix=ledger["evaluate"]["registration"]["started"],
        stats=json.loads((HERE/"analysis/verification.json").read_text()))
    out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("verified", len(ledger), "exited GPU jobs,", round(metadata["gpu_device_hours"], 4), "device hours")

if __name__ == "__main__":
    main()
