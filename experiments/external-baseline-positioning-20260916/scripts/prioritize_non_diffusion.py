#!/usr/bin/env python3
"""Temporarily yield this study's diffusion worker, finish short work, then resume it."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]


def write(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def pause_our_hifi(root, directory):
    queue = json.loads((root / "author_queue_001/status.json").read_text())
    if queue.get("stage") != "hifi":
        raise RuntimeError("only the current study's HiFi worker may be paused")
    child, parent = int(queue["child_pid"]), int(queue["pid"])
    for pid, expected in ((child, "evaluate_authors.py"), (parent, "run_author_measurements.py")):
        process = Path(f"/proc/{pid}")
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if process.stat().st_uid != os.getuid() or str(EXPERIMENT) not in command or expected not in command:
            raise RuntimeError("refusing to signal an unrelated process")
    status_path = root / "development_hifi_001/status.json"
    before = json.loads(status_path.read_text())
    write(directory / "pause_requested.json", {"reason": "user_prioritizes_non_diffusion_completion", "queue_before": queue,
          "hifi_before": before, "at": datetime.now().astimezone().isoformat(), "sampling_or_protocol_changed": False})
    deadline = time.monotonic() + 180
    while alive(child):
        status = json.loads(status_path.read_text())
        if status.get("frames_completed", 0) > before["frames_completed"]:
            os.kill(child, signal.SIGINT)
            break
        if time.monotonic() > deadline:
            raise RuntimeError("no committed frame in180 seconds; do not interrupt an unreviewed worker")
        time.sleep(.25)
    deadline = time.monotonic() + 120
    while alive(child) or alive(parent):
        if time.monotonic() > deadline:
            raise RuntimeError("our paused queue did not exit; no forced kill or concurrent GPU work")
        time.sleep(.5)
    write(directory / "paused.json", {"committed_frames": json.loads(status_path.read_text()).get("frames_completed"),
          "all_committed_artifacts_retained": True, "uncommitted_frame_will_resume": True,
          "at": datetime.now().astimezone().isoformat()})


def run(arguments):
    root = arguments.root.resolve()
    directory = root / "non_diffusion_priority_queue_001"
    directory.mkdir(exist_ok=False)
    if not (root / "digital_calibration_001/phy_completion.json").exists():
        raise RuntimeError("CPU calibration preparation must finish first")
    python = str(PROJECT / "experiments/backbone-eval-20260912/.venv/bin/python")
    digital = str(EXPERIMENT / "scripts/evaluate_digital_budgets.py")
    policies = root / "digital_policies_001.json"
    calibration = root / "digital_calibration_001"
    development = root / "digital_development_001"
    timing = root / "digital_timing_001"
    stages = [
        ("calibration_RGB", [python, digital, "--stage", "render", "--population", "calibration", "--output", str(calibration)]),
        ("freeze_policies", [python, digital, "--stage", "fit", "--population", "calibration", "--output", str(calibration), "--policies", str(policies)]),
        ("development_PHY", [python, digital, "--stage", "phy", "--population", "development", "--output", str(development), "--policies", str(policies)]),
        ("development_RGB", [python, digital, "--stage", "render", "--population", "development", "--output", str(development), "--policies", str(policies)]),
        ("new_budget_timing", [python, str(EXPERIMENT / "scripts/time_digital_budgets.py"), "--root", str(root), "--development", str(development), "--output", str(timing)]),
        ("non_diffusion_tables", [python, str(EXPERIMENT / "scripts/summarize_available.py"), "--root", str(root), "--output", str(root / "non_diffusion_analysis_001"), "--digital-runs", str(development), "--digital-timing", str(timing)]),
    ]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    paused = False
    try:
        pause_our_hifi(root, directory)
        paused = True
        for stage, command in stages:
            log_path = directory / f"{stage}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT)
                write(directory / "status.json", {"stage": stage, "pid": os.getpid(), "child_pid": process.pid,
                      "log": str(log_path), "at": datetime.now().astimezone().isoformat()})
                code = process.wait()
            if code:
                raise RuntimeError(f"priority stage {stage} failed; preserve and inspect {log_path}")
        write(directory / "completion.json", {"status": "NON_DIFFUSION_QUALITY_AND_COST_COMPLETE", "HiFi_complete": False,
              "final_positioning_complete": False, "at": datetime.now().astimezone().isoformat()})
    except Exception:
        write(directory / "failure.json", {"traceback": traceback.format_exc(), "at": datetime.now().astimezone().isoformat()})
        raise
    finally:
        if paused:
            command = [str(EXPERIMENT / ".venv/bin/python"), "-u", str(EXPERIMENT / "scripts/run_author_measurements.py"), "--root", str(root)]
            with (directory / "resumed_author_queue.log").open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(directory / "author_resumed.json", {"pid": process.pid, "command": command, "same_frozen_protocol": True,
                  "at": datetime.now().astimezone().isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args())
