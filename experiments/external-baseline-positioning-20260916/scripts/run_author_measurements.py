#!/usr/bin/env python3
"""Finite pretrained-model measurement queue; never starts any training."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(arguments):
    root = arguments.root.resolve()
    directory = root / "author_queue_001"
    directory.mkdir(exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / "completion.json").exists():
        raise RuntimeError("completed finite measurements must not be repeated")
    for required in ("inputs_001", "calibration_metrics_001", "qualify_swin_001", "qualify_adjscc_001", "qualify_hifi_full_001"):
        if not (root / required / "completion.json").exists():
            raise RuntimeError(f"required qualification incomplete: {required}")
    sources = [Path(__file__), EXPERIMENT / "configs/protocol.json", EXPERIMENT / "scripts/evaluate_authors.py",
               EXPERIMENT / "scripts/score_author_outputs.py", EXPERIMENT / "src/external_positioning/author_models.py",
               EXPERIMENT / "src/external_positioning/radio.py"]
    bindings = {str(path): checksum(path) for path in sources}
    if (directory / "bindings.json").exists() and json.loads((directory / "bindings.json").read_text()) != bindings:
        raise RuntimeError("measurement queue implementation changed")
    write(directory / "bindings.json", bindings)
    python = str(EXPERIMENT / ".venv/bin/python")
    metric_python = str(PROJECT / "experiments/backbone-eval-20260912/.venv/bin/python")
    inputs = root / "inputs_001"
    outputs = {name: root / f"development_{name}_001" for name in ("swin", "adjscc", "hifi")}
    stages = []
    for name in ("swin", "adjscc", "hifi"):
        command = [python, str(EXPERIMENT / "scripts/evaluate_authors.py"), "--family", name,
                   "--inputs", str(inputs), "--output", str(outputs[name])]
        if name == "hifi":
            command += ["--paired-adjscc", str(outputs["adjscc"] / "per_frame.csv")]
        stages.append((name, command, outputs[name]))
        if name == "adjscc":
            stages.append(("fast_metrics", [metric_python, str(EXPERIMENT / "scripts/score_author_outputs.py"),
                           "--runs", str(outputs["swin"]), str(outputs["adjscc"]), "--inputs", str(inputs),
                           "--output", str(root / "fast_author_metrics_001")], root / "fast_author_metrics_001"))
    stages.append(("hifi_metrics", [metric_python, str(EXPERIMENT / "scripts/score_author_outputs.py"),
                   "--runs", str(outputs["hifi"]), "--inputs", str(inputs), "--output", str(root / "hifi_metrics_001")],
                   root / "hifi_metrics_001"))
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for stage, command, output in stages:
        if any(checksum(path) != expected for path, expected in bindings.items()):
            raise RuntimeError("registered measurement code changed during execution")
        if (output / "completion.json").exists():
            continue
        log_path = directory / f"{stage}_{time.time_ns()}.log"
        started = time.perf_counter()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write(directory / "status.json", {"status": "RUNNING_FROZEN_AUTHOR_MEASUREMENTS", "stage": stage,
                  "pid": os.getpid(), "child_pid": process.pid, "log": str(log_path),
                  "new_training": False, "updated_at": datetime.now().astimezone().isoformat()})
            code = process.wait()
        if code or not (output / "completion.json").exists():
            raise RuntimeError(f"{stage} did not complete; preserve outputs and review {log_path}")
        write(directory / f"{stage}_completed.json", {"seconds": time.perf_counter() - started,
              "completion_sha256": checksum(output / "completion.json")})
    receipt = {"status": "AUTHOR_MEASUREMENTS_AND_METRICS_COMPLETE", "new_training_or_holdout": False,
               "research_positioning_report_complete": False,
               "remaining": "common-budget digital calibration/evaluation and final evidence review",
               "finished_at": datetime.now().astimezone().isoformat()}
    write(directory / "completion.json", receipt)
    write(directory / "status.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        directory = arguments.root / "author_queue_001"
        directory.mkdir(parents=True, exist_ok=True)
        failure = {"status": "STOPPED_FOR_REVIEW_NO_NEW_SEARCH", "traceback": traceback.format_exc(),
                   "at": datetime.now().astimezone().isoformat()}
        write(directory / f"failure_{time.time_ns()}.json", failure)
        write(directory / "status.json", failure)
        raise
