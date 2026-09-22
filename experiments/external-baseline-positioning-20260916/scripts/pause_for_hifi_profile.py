#!/usr/bin/env python3
"""Yield only our documented full queue to an isolated profile and stage20 schedule."""

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


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def verify_owner(pid, expected):
    process = Path(f"/proc/{pid}")
    command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if process.stat().st_uid != os.getuid() or str(EXPERIMENT) not in command or expected not in command:
        raise RuntimeError("refusing to stop an unrelated process")
    return command


def run(arguments):
    root, stage = arguments.root, arguments.stage
    directory = stage / "original_queue_pause_001"
    directory.mkdir(exist_ok=False)
    author_path, closure_path = root / "author_queue_001/status.json", root / "closure_queue_001/status.json"
    author, closure = json.loads(author_path.read_text()), json.loads(closure_path.read_text())
    child, parent = int(author["child_pid"]), int(author["pid"])
    if author.get("stage") != "hifi":
        raise RuntimeError("only the registered HiFi stage can yield")
    commands = {str(child): verify_owner(child, "evaluate_authors.py"), str(parent): verify_owner(parent, "run_author_measurements.py")}
    if closure.get("status") == "WAITING_FOR_EXISTING_HIFI_AND_METRICS_NO_GPU":
        closure_pid = int(closure["pid"])
        commands[str(closure_pid)] = verify_owner(closure_pid, "finish_hifi_positioning.py")
        write(directory / "closure_before.json", closure)
        os.kill(closure_pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while alive(closure_pid):
            if time.monotonic() > deadline:
                raise RuntimeError("our CPU closure did not exit")
            time.sleep(.25)
        write(closure_path, {"status": "PAUSED_BY_USER_STAGE20_PRIORITY", "previous_pid": closure_pid,
              "original1500_plan_retained": True, "resume_requires_original_complete_receipts": True,
              "at": datetime.now().astimezone().isoformat()})
    before = json.loads((root / "development_hifi_001/status.json").read_text())
    write(directory / "author_before.json", author)
    write(directory / "inference_before.json", before)
    write(directory / "reason.json", {"reason": "authorized_isolated_performance_profile_and_quality_independent_stage20",
          "commands_checked": commands, "protocol_or_sampler_overwritten": False, "at": datetime.now().astimezone().isoformat()})
    deadline = time.monotonic() + 180
    while alive(child):
        current = json.loads((root / "development_hifi_001/status.json").read_text())
        if current.get("frames_completed", 0) > before["frames_completed"]:
            os.kill(child, signal.SIGINT)
            break
        if time.monotonic() > deadline:
            raise RuntimeError("no completed-frame boundary; no forced kill")
        time.sleep(.25)
    deadline = time.monotonic() + 120
    while alive(child) or alive(parent):
        if time.monotonic() > deadline:
            raise RuntimeError("our queue did not release the GPU")
        time.sleep(.5)
    final = json.loads((root / "development_hifi_001/status.json").read_text())
    write(directory / "last_committed_status.json", final)
    write(directory / "parent_exit_status.json", json.loads(author_path.read_text()))
    write(author_path, {"status": "PAUSED_BY_USER_STAGE20_PRIORITY", "original_target_frames": 1500,
          "committed_frames": final["frames_completed"], "sampler_and_checkpoints_unchanged": True,
          "original_plan_retained_for_resume": True, "pause_receipts": str(directory), "at": datetime.now().astimezone().isoformat()})
    command = [str(EXPERIMENT / ".venv/bin/python"), "-u", str(EXPERIMENT / "performance_branch/profile_attention.py"),
               "--root", str(root), "--stage", str(stage), "--output", str(stage / "attention_profile_001")]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    with (stage / "attention_profile_001.log").open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT.parent, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(stage / "profile_launched.json", {"pid": process.pid, "command": command, "new_training": False,
          "original_queue_paused_not_replaced": True, "at": datetime.now().astimezone().isoformat()})
    print(json.dumps({"profile_pid": process.pid, "original_committed_frames": final["frames_completed"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    run(parser.parse_args())
