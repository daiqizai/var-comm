#!/usr/bin/env python3
"""Resume only unchanged stage20 work, yielding our GPU worker on external contention."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]


def write(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_pids():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def run(arguments):
    stage = arguments.stage
    directory = stage / "resource_resume_001"
    directory.mkdir(exist_ok=False)
    write(directory / "reason.json", {"reason": "original_guard_detected_external_CAP_VPR_process367282;not_sampling_failure",
          "initial_committed_frames": 91, "all_original_frames_retained": True, "no_other_job_signalled": True,
          "model_protocol_and_scope_unchanged": True, "at": datetime.now().astimezone().isoformat()})
    attempt = 0
    while not (stage / "pipeline_001/completion.json").exists():
        idle_since = None
        while idle_since is None or time.monotonic() - idle_since < 20:
            active = gpu_pids()
            if active:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            write(directory / "status.json", {"status": "WAITING_FOR_GPU_IDLE_NO_SHARED_EXECUTION", "other_GPU_pids": sorted(active),
                  "pid": os.getpid(), "at": datetime.now().astimezone().isoformat()})
            time.sleep(2)
        attempt += 1
        command = [str(PROJECT / "experiments/backbone-eval-20260912/.venv/bin/python"), "-u",
                   str(EXPERIMENT / "scripts/run_hifi_stage20_pipeline.py"), "--root", str(arguments.root), "--stage", str(stage)]
        with (directory / f"resume_{attempt:03d}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=PROJECT.parent, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                                       stdout=log, stderr=subprocess.STDOUT)
            yielded = False
            while process.poll() is None:
                status = json.loads((stage / "pipeline_001/status.json").read_text())
                active = gpu_pids()
                child = status.get("child_pid")
                if status.get("stage") in ("original_author_stage20", "stage20_metrics") and child in active and active - {child}:
                    process_path = Path(f"/proc/{child}")
                    command_line = (process_path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                    if process_path.stat().st_uid != os.getuid() or str(EXPERIMENT) not in command_line:
                        raise RuntimeError("refusing to signal an unrelated process")
                    write(directory / f"contention_{time.time_ns()}.json", {"own_child": child, "other_pids": sorted(active - {child}),
                          "action": "interrupt_our_worker_only", "sampling_failure": False, "at": datetime.now().astimezone().isoformat()})
                    os.kill(child, signal.SIGINT)
                    yielded = True
                    break
                write(directory / "status.json", {"status": "MONITORING_UNCHANGED_STAGE20", "attempt": attempt,
                      "stage": status.get("stage"), "child_pid": child, "GPU_pids": sorted(active), "pid": os.getpid(),
                      "at": datetime.now().astimezone().isoformat()})
                time.sleep(1)
            code = process.wait()
        if code and not yielded:
            status = json.loads((stage / "pipeline_001/status.json").read_text())
            log_path = status.get("log")
            logs = sorted((stage / "pipeline_001").glob("*.log"), key=lambda path: path.stat().st_mtime)
            content = logs[-1].read_text() if logs else ""
            if "OTHER_GPU_PROCESS_PRESENT" not in content:
                write(directory / "status.json", {"status": "STOPPED_NON_RESOURCE_ERROR_REQUIRES_REVIEW", "code": code,
                      "preserve_outputs": True, "at": datetime.now().astimezone().isoformat()})
                raise RuntimeError("not a resource interruption; do not retry or change the model")
    write(directory / "completion.json", {"status": "STAGE20_RESUMPTION_COMPLETE", "attempts": attempt,
          "other_jobs_modified": False, "sampler_changed": False, "at": datetime.now().astimezone().isoformat()})
    write(directory / "status.json", json.loads((directory / "completion.json").read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    run(parser.parse_args())
