#!/usr/bin/env python3
"""Bounded closing study only; any future CSI topic is a report, never a queued trainer."""

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


def run(arguments):
    root = arguments.root.resolve()
    directory = root / "pipeline_001"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / "completion.json").exists():
        raise RuntimeError("the finite weight grid already completed")
    for phase in ("profile_001", "quality_smoke_001", "analysis_smoke_001", "DEEP_CALIBRATION_001"):
        if not (root / phase / "completion.json").exists():
            raise RuntimeError(f"required qualification missing: {phase}")
    sources = [ROOT / "scripts" / name for name in ("train_weight_closure.py", "fit_weight_closure.py", "evaluate_weight_closure.py",
               "summarize_weight_closure.py", "run_weight_closure_pipeline.py", "check_weight_closure.py")]
    sources += [ROOT / "src/var_comm/weight_closure.py", ROOT / "configs/hybrid_weight_closure.json",
                ROOT / "reports/hybrid_weight_closure_protocol_20260916.md"]
    binding = {str(path): sha256(path) for path in sources}
    if (directory / "bindings.json").exists() and json.loads((directory / "bindings.json").read_text()) != binding:
        raise RuntimeError("pipeline code changed on resume")
    write_json(directory / "bindings.json", binding)
    python = sys.executable
    training, freeze, quality = root / "training_001", root / "CALIBRATION_FREEZE_001", root / "development_001"
    stages = [
        ("training", [python, str(ROOT / "scripts/train_weight_closure.py"), "--output", str(training), "--qualification", str(root / "reuse_qualification")], training),
        ("freeze", [python, str(ROOT / "scripts/fit_weight_closure.py"), "--training", str(training), "--deep-calibration", str(root / "DEEP_CALIBRATION_001"), "--output", str(freeze)], freeze),
        ("development", [python, str(ROOT / "scripts/evaluate_weight_closure.py"), "--training", str(training), "--freeze", str(freeze), "--output", str(quality)], quality),
        ("analysis", [python, str(ROOT / "scripts/summarize_weight_closure.py"), "--training", str(training), "--quality", str(quality), "--output", str(root / "analysis_001")], root / "analysis_001"),
    ]
    for stage, command, output in stages:
        if any(sha256(path) != expected for path, expected in binding.items()):
            raise RuntimeError("registered finite grid changed during execution")
        if (directory / f"{stage}_done.json").exists():
            continue
        if (output / "completion.json").exists():
            write_json(directory / f"{stage}_done.json", {"reused_completed_stage": True, "receipt_sha256": sha256(output / "completion.json")})
            continue
        log_path = directory / f"{stage}_{time.time_ns()}.log"
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        start = time.perf_counter()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(directory / "status.json", {"status": "RUNNING", "stage": stage, "pid": os.getpid(), "child_pid": process.pid,
                       "command": command, "log": str(log_path), "updated_at": datetime.now().astimezone().isoformat()})
            result = process.wait()
        if result or not (output / "completion.json").exists():
            raise RuntimeError(f"{stage} failed; preserve evidence and stop, not expand the search: {log_path}")
        write_json(directory / f"{stage}_done.json", {"seconds": time.perf_counter() - start, "receipt_sha256": sha256(output / "completion.json"),
                   "finished_at": datetime.now().astimezone().isoformat()})
    completed = {"status": "FIXED_WEIGHT_GRID_NUMERICS_COMPLETE_NO_FURTHER_TRAINING", "new_holdout": False,
                 "new_CSI_topic_training_started": False, "report_requires_evidence_and_literature_review": True,
                 "finished_at": datetime.now().astimezone().isoformat()}
    write_json(directory / "completion.json", completed)
    write_json(directory / "status.json", completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        directory = args.root / "pipeline_001"
        directory.mkdir(parents=True, exist_ok=True)
        failure = {"status": "STOPPED_ON_FAILURE_NO_EXPANSION", "traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()}
        write_json(directory / f"failure_{time.time_ns()}.json", failure)
        write_json(directory / "status.json", failure)
        raise
