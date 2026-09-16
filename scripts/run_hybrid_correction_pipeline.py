#!/usr/bin/env python3
"""Finite, failure-stopping queue for one registered hybrid experiment."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from var_comm.study import sha256, write_json


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(arguments):
    root = arguments.root.resolve()
    directory = root / "pipeline_001"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / "completion.json").exists():
        raise RuntimeError("bounded pipeline already completed")
    files = [ROOT / "scripts" / name for name in ("prepare_hybrid_correction.py", "train_hybrid_correction.py",
             "evaluate_hybrid_correction.py", "benchmark_hybrid_correction.py", "summarize_hybrid_correction.py",
             "check_hybrid_correction.py", "run_hybrid_correction_pipeline.py")]
    files += [ROOT / "src/var_comm/hybrid_correction.py", ROOT / "src/var_comm/hybrid_training.py",
              ROOT / "configs/hybrid_source_correction.json", ROOT / "reports/hybrid_source_correction_protocol_20260915.md"]
    bindings = {str(path): sha256(path) for path in files}
    if (directory / "bindings.json").exists():
        if json.loads((directory / "bindings.json").read_text()) != bindings:
            raise RuntimeError("pipeline resume cannot silently change its registered implementation")
    else:
        write_json(directory / "bindings.json", bindings)
    cache = root / "cache_001"
    while alive(arguments.cache_pid):
        if (cache / "status.json").exists() and json.loads((cache / "status.json").read_text())["status"] == "FAILED":
            raise RuntimeError("cache preparation failed")
        write_json(directory / "status.json", {"status": "WAITING_FOR_OWN_CACHE", "pid": os.getpid(),
                   "cache_pid": arguments.cache_pid, "updated_at": datetime.now().astimezone().isoformat()})
        time.sleep(10)
    if not (cache / "completion.json").exists():
        raise RuntimeError("cache process exited without a successful completion receipt")
    python = sys.executable
    script = lambda name: str(ROOT / "scripts" / name)
    profile, training = root / "profile_001", root / "training_001"
    smoke_quality, smoke_timing = root / "quality_smoke_001", root / "timing_smoke_001"
    quality, timing = root / "development_001", root / "timing_001"
    stages = [
        ("cpu_regression", [python, script("check_hybrid_correction.py")], None),
        ("profile", [python, script("train_hybrid_correction.py"), "--cache", str(cache), "--output", str(profile), "--mode", "profile"], profile),
        ("quality_smoke", [python, script("evaluate_hybrid_correction.py"), "--training", str(profile), "--output", str(smoke_quality), "--mode", "smoke"], smoke_quality),
        ("timing_smoke", [python, script("benchmark_hybrid_correction.py"), "--training", str(profile), "--quality", str(smoke_quality), "--output", str(smoke_timing), "--mode", "smoke"], smoke_timing),
        ("analysis_smoke", [python, script("summarize_hybrid_correction.py"), "--training", str(profile), "--quality", str(smoke_quality), "--timing", str(smoke_timing), "--output", str(root / "analysis_smoke_001")], root / "analysis_smoke_001"),
        ("training", [python, script("train_hybrid_correction.py"), "--cache", str(cache), "--output", str(training)], training),
        ("development", [python, script("evaluate_hybrid_correction.py"), "--training", str(training), "--output", str(quality)], quality),
        ("timing", [python, script("benchmark_hybrid_correction.py"), "--training", str(training), "--quality", str(quality), "--output", str(timing)], timing),
        ("analysis", [python, script("summarize_hybrid_correction.py"), "--training", str(training), "--quality", str(quality), "--timing", str(timing), "--output", str(root / "analysis_001")], root / "analysis_001"),
    ]
    for stage, command, artifacts in stages:
        for path, expected in bindings.items():
            if sha256(path) != expected:
                raise RuntimeError(f"implementation changed during the queue: {path}")
        receipt_path = directory / f"{stage}_done.json"
        if receipt_path.exists():
            continue
        if artifacts is not None and (artifacts / "completion.json").exists():
            write_json(receipt_path, {"reused_completed_stage": True, "completion_sha256": sha256(artifacts / "completion.json")})
            continue
        log_path = directory / f"{stage}_{time.time_ns()}.log"
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        start = time.perf_counter()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(directory / "status.json", {"status": "RUNNING", "stage": stage, "pid": os.getpid(),
                       "child_pid": process.pid, "command": command, "log": str(log_path),
                       "updated_at": datetime.now().astimezone().isoformat()})
            code = process.wait()
        if code != 0 or (artifacts is not None and not (artifacts / "completion.json").exists()):
            raise RuntimeError(f"stage {stage} failed with return code {code}; no architecture/budget fallback; see {log_path}")
        write_json(receipt_path, {"stage": stage, "seconds": time.perf_counter() - start, "command": command,
                                 "completion_sha256": sha256(artifacts / "completion.json") if artifacts is not None else None,
                                 "finished_at": datetime.now().astimezone().isoformat()})
    completed = {"status": "REGISTERED_HYBRID_DEVELOPMENT_STUDY_COMPLETE", "new_holdout_accessed": False,
                 "automatic_extensions": False, "report": str(root / "analysis_001/report.md"),
                 "finished_at": datetime.now().astimezone().isoformat()}
    write_json(directory / "completion.json", completed)
    write_json(directory / "status.json", completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-pid", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        directory = args.root / "pipeline_001"
        directory.mkdir(parents=True, exist_ok=True)
        failure = {"status": "STOPPED_ON_FAILURE_NO_NEW_BRANCH", "traceback": traceback.format_exc(),
                   "at": datetime.now().astimezone().isoformat()}
        write_json(directory / f"failure_{time.time_ns()}.json", failure)
        write_json(directory / "status.json", failure)
        raise
