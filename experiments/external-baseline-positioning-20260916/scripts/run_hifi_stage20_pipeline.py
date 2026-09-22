#!/usr/bin/env python3
"""Finite original-sampler exploratory queue; never resume1500 or deploy profile variants."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]


def write(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(arguments):
    root, stage = arguments.root, arguments.stage
    queue = stage / "pipeline_001"
    queue.mkdir(exist_ok=True)
    lock = (queue / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (queue / "completion.json").exists():
        raise RuntimeError("completed stage cannot be repeated")
    frozen = json.loads((root / "author_queue_001/bindings.json").read_text())
    for path in (Path(__file__), EXPERIMENT / "scripts/evaluate_hifi_stage20.py", stage / "protocol.json", stage / "development_inputs.json"):
        frozen[str(path)] = sha(path)
    write(queue / "bindings.json", frozen)
    while not (stage / "repeatability_001/completion.json").exists():
        if (stage / "attention_profile_001/failure.json").exists():
            raise RuntimeError("isolated profile requires engineering review")
        write(queue / "status.json", {"status": "WAITING_FOR_BOUNDED_PROFILE_REPEATABILITY", "pid": os.getpid(),
              "new_training": False, "at": datetime.now().astimezone().isoformat()})
        time.sleep(10)
    while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        write(queue / "status.json", {"status": "WAITING_FOR_GPU_RELEASE_DO_NOT_DISTURB_OTHERS", "pid": os.getpid()})
        time.sleep(5)
    author_python = str(EXPERIMENT / ".venv/bin/python")
    metric_python = str(PROJECT / "experiments/backbone-eval-20260912/.venv/bin/python")
    actions = [("original_author_stage20", [author_python, "-u", str(EXPERIMENT / "scripts/evaluate_hifi_stage20.py"),
               "--root", str(root), "--stage", str(stage)], stage / "inference_completed.json"),
               ("stage20_metrics", [metric_python, "-u", str(EXPERIMENT / "scripts/score_author_outputs.py"),
                "--runs", str(stage / "inference_001"), "--inputs", str(stage), "--output", str(stage / "metrics_001")], stage / "metrics_001/completion.json"),
               ("stage20_analysis", [metric_python, "-u", str(EXPERIMENT / "scripts/analyze_hifi_stage20.py"),
                "--root", str(root), "--stage", str(stage), "--output", str(stage / "analysis_001")], stage / "analysis_001/completion.json")]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for name, command, completion in actions:
        if completion.exists():
            continue
        if any(sha(path) != expected for path, expected in frozen.items()):
            raise RuntimeError("registered original sampler or exploratory scope changed")
        if not Path(command[2]).is_file():
            raise RuntimeError("analysis implementation must be ready; never fabricate its completion")
        if name == "stage20_analysis":
            write(queue / "analysis_source.json", {"script": command[2], "sha256": sha(command[2])})
        log_path = queue / f"{name}_{time.time_ns()}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write(queue / "status.json", {"status": "RUNNING_EXPLORATORY_STAGE20_ONLY", "stage": name, "pid": os.getpid(),
                  "child_pid": process.pid, "log": str(log_path), "original1500_complete": False,
                  "original_sampler_retained": True, "at": datetime.now().astimezone().isoformat()})
            code = process.wait()
        if code or not completion.exists():
            raise RuntimeError(f"stage {name} failed; retain log and stop, not a method-performance failure")
    receipt = {"status": "EXPLORATORY20_SOURCE100_TRANSMISSIONS_COMPLETE", "original1500_complete": False,
               "new_training_or_architecture": False, "old_full_queue_not_automatically_resumed": True,
               "finished_at": datetime.now().astimezone().isoformat()}
    write(queue / "completion.json", receipt)
    write(queue / "status.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception:
        directory = arguments.stage / "pipeline_001"
        directory.mkdir(exist_ok=True)
        record = {"status": "STOPPED_FOR_REVIEW_NO_NEW_RESEARCH", "traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()}
        write(directory / f"failure_{time.time_ns()}.json", record)
        write(directory / "status.json", record)
        raise
