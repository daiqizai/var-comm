"""Export lightweight, identity-checked Pilot50 evidence from the scientific work area."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

from verify_pfs_cache import filehash, require


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    with path.open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    p, out = args.pilot, args.out
    out.mkdir()
    plan = read(p / "sp1_plan/completion.json")
    qual = read(p / "quality/completion.json")
    require(filehash(p / "quality/scores.csv") == qual["scores_sha256"], "four-metric table")
    require(read(p / "decision/decision.json")["plan_sha256"] == filehash(p / "sp1_plan/completion.json"), "registration")
    for original, name in (
        ("source/identity.json", "source_identity.json"), ("source/completion.json", "source_completion.json"),
        ("source/independent_rx_pass.json", "independent_rx_pass.json"), ("source_control/actual_wait.json", "source_actual_wait.json"),
        ("quality/identity.json", "quality_identity.json"), ("quality/completion.json", "quality_completion.json"),
        ("quality/scores.csv", "clean_scores.csv"), ("quality_control/actual_wait.json", "quality_actual_wait.json"),
        ("sp1_plan/completion.json", "sp1_plan_identity.json"), ("sp1_plan/protocol.json", "sp1_protocol.json"),
        ("rx_catalog/completion.json", "rx_catalog_identity.json")):
        shutil.copyfile(p / original, out / name)
    for path in sorted((p / "decision").iterdir()):
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as src, (out / path.stem).open("xb") as dst:
                shutil.copyfileobj(src, dst)
        elif path.suffix in (".json", ".csv"):
            shutil.copyfile(path, out / path.name)
    save(out / "source_manifest.json", [
        dict(slot=i, source_id=v, source_pixel_sha256=read(p / "source" / f"slot_{i:02d}" / "done.json")["source_pixels_sha256"])
        for i, v in enumerate(plan["roster"])])
    rows = []
    for i, v in enumerate(plan["roster"]):
        meta = read(p / "source" / f"slot_{i:02d}" / "done.json")
        require(meta["source_id"] == v and not meta["source_truth_used_at_rx"], ("independent stream identity", i))
        rows.extend(dict(slot=i, source_id=v, **r) for r in meta["lengths"])
    require(len(rows) == 3600 and sum(row["m"] == 10 for row in rows) == 200, "complete two-layer table")
    with (out / "double_stream_lengths.csv").open("x", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    jobs = {}
    for name in ("source", "quality", "rx_catalog", "sp1_sp", "sp1_single", "sp1_oma", "sp1_controls", "decision"):
        control = p / (name + "_control")
        wait, child = read(control / "actual_wait.json"), read(control / "child.json")
        require(wait["returncode"] == 0 and not wait["timed_out"] and wait["pid"] == child["worker"]["pid"], name)
        jobs[name] = dict(launch=child, actual_wait=wait)
        reg = p / name / "registration.json"
        if reg.exists():
            jobs[name]["registration"] = read(reg)
    save(out / "execution.json", jobs)
    physical = {name: read(p / ("sp1_" + name) / "completion.json") for name in ("single", "sp", "oma")}
    save(out / "qualified_ldpc_layouts.json", {name: x["qualified_layouts"] for name, x in physical.items()})
    source = read(out / "source_completion.json")
    seconds = {"independent_source": source["wall_seconds"], "source_quality": qual["gpu_wall_seconds"],
               **{name: x["wall_seconds"] for name, x in physical.items()}}
    science = read(out / "decision.json")
    shared = p.parent.parent
    save(out / "resource_ledger.json", dict(
        device=read(out / "source_identity.json")["physical_name"], gpu_uuid=read(out / "source_identity.json")["gpu_uuid"],
        same_device_scientific_stage_wall_seconds=seconds, active_stage_device_hours=sum(seconds.values()) / 3600.,
        accounting="Elapsed times within the five instrumented scientific stages on one GPU, including their local CPU and I/O. Launch, queue, earlier qualification and historical CPU diagnostics are separate.",
        source_counts=source["counts"], quality_counts=qual["counts"], independent_receiver_lookup=read(out / "rx_catalog_identity.json")["counts"],
        unique_waveforms={k:x["waves"] for k,x in physical.items()},
        received_observations={k:x["new_receiver_observations"] for k,x in physical.items()},
        unresolved_received_strings={k:x["unresolved_source_inputs"] for k,x in physical.items()},
        training_steps=0, formal_common500_frames=0, pilot_gate=science["status"],
        prior_four_arm_pause_marker=(shared/"rdvq_prefix_20261104_v1/STOP_P2").is_file(),
        main_training_stop_marker=(shared/"multirate_reconstruction_20261105_v1/STOP_MAIN").is_file(),
        raw_PFS_root=str(p)))
    save(out / "evidence_hashes.json", {f.name:filehash(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(len(rows), "two-layer streams;", len(jobs), "actual successful process waits", flush=True)


if __name__ == "__main__":
    main()
