#!/usr/bin/env python3
"""Single authorized receiver-pair queue; no model/resource search or new holdout."""

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
    output = root / "pipeline_001"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "completion.json").exists():
        raise RuntimeError("the bounded receiver pair already completed")
    files = [ROOT / "scripts" / name for name in ("train_base_conditioning.py", "evaluate_base_conditioning.py", "benchmark_base_conditioning.py",
             "summarize_base_conditioning.py", "run_base_conditioning_pipeline.py", "check_base_conditioning.py")]
    files += [ROOT / "src/var_comm/base_conditioning.py", ROOT / "configs/hybrid_base_conditioning.json",
              ROOT / "reports/hybrid_base_conditioning_protocol_20260915.md"]
    bindings = {str(path): sha256(path) for path in files}
    if (output / "bindings.json").exists() and json.loads((output / "bindings.json").read_text()) != bindings:
        raise RuntimeError("frozen pipeline implementation changed")
    write_json(output / "bindings.json", bindings)
    for phase in ("profile_001", "quality_smoke_001", "timing_smoke_001", "analysis_smoke_001"):
        if not (root / phase / "completion.json").exists():
            raise RuntimeError(f"required completed engineering phase missing: {phase}")
    python = sys.executable
    training, quality, timing = root / "training_001", root / "development_001", root / "timing_001"
    stages = [
        ("training", [python, str(ROOT / "scripts/train_base_conditioning.py"), "--output", str(training)], training),
        ("development", [python, str(ROOT / "scripts/evaluate_base_conditioning.py"), "--training", str(training), "--output", str(quality)], quality),
        ("timing", [python, str(ROOT / "scripts/benchmark_base_conditioning.py"), "--training", str(training), "--quality", str(quality), "--output", str(timing)], timing),
        ("analysis", [python, str(ROOT / "scripts/summarize_base_conditioning.py"), "--training", str(training), "--quality", str(quality), "--timing", str(timing), "--output", str(root / "analysis_001")], root / "analysis_001"),
    ]
    for phase, command, directory in stages:
        if any(sha256(path) != expected for path, expected in bindings.items()):
            raise RuntimeError("registered pair code changed during execution")
        if (output / f"{phase}_done.json").exists():
            continue
        if (directory / "completion.json").exists():
            write_json(output / f"{phase}_done.json", {"reused_completed_stage": True, "receipt_sha256": sha256(directory / "completion.json")})
            continue
        log_path = output / f"{phase}_{time.time_ns()}.log"
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        start = time.perf_counter()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / "status.json", {"status": "RUNNING", "stage": phase, "pid": os.getpid(), "child_pid": process.pid,
                       "command": command, "log": str(log_path), "updated_at": datetime.now().astimezone().isoformat()})
            code = process.wait()
        if code != 0 or not (directory / "completion.json").exists():
            raise RuntimeError(f"stage {phase} failed, no new branch or automatic recipe change; see {log_path}")
        write_json(output / f"{phase}_done.json", {"seconds": time.perf_counter() - start, "receipt_sha256": sha256(directory / "completion.json"),
                   "finished_at": datetime.now().astimezone().isoformat()})
    completion = {"status": "BASE_CONDITIONING_REGISTERED_STUDY_COMPLETE", "report": str(root / "analysis_001/report.md"),
                  "automatic_extensions": False, "new_holdout_accessed": False, "finished_at": datetime.now().astimezone().isoformat()}
    write_json(output / "completion.json", completion)
    write_json(output / "status.json", completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
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
