"""Read-only CPU observer; never signals workers or changes the frozen experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "outputs/VAR-LATENT-ENHANCEMENT-20260917"


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_state(pid, module):
    if not isinstance(pid, int) or pid <= 0:
        return {"matches": False, "pid": pid}
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        matches = any(command[index:index + 2] == ["-m", module] for index in range(len(command) - 1))
        return {"pid": pid, "matches": matches and fields[0] != "Z", "state": fields[0],
                "start_ticks": fields[19], "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return {"matches": False, "pid": pid}


def gpu_state():
    command = ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, timeout=10).strip().split(",")
    return dict(zip(("memory_used_MiB", "memory_total_MiB", "utilization_percent", "temperature_C"),
                    (float(value.strip()) for value in output)))


def latest_training_row(path):
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            continue
    return {}


def source_integrity(registration):
    changed = []
    for filename, expected in registration.get("source_snapshot", {}).items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(filename)
    return changed


def classify(snapshot, stale_seconds=600):
    alerts = []
    if snapshot.get("completion"):
        return "STAGE_A_COMPLETE_B_NOT_STARTED", alerts
    pipeline = snapshot.get("pipeline", {})
    pipeline_status = pipeline.get("status", "")
    stage_status = snapshot.get("training", {}).get("status", "")
    if "FAILURE" in pipeline_status or "INTERRUPT" in pipeline_status or stage_status == "PAUSED_EXPLICIT_INTERRUPT":
        return "STOPPED_REQUIRES_REVIEW", ["worker_failure_or_explicit_interrupt"]
    if snapshot.get("changed_sources"):
        alerts.append("frozen_source_or_config_changed")
    if not snapshot.get("pipeline_process", {}).get("matches"):
        alerts.append("pipeline_process_missing_or_pid_reused")
    waiting = ("WAITING" in pipeline_status or "YIELDED" in stage_status)
    if waiting:
        state = "WAITING_FOR_AUTHORIZED_RESOURCES"
    elif pipeline_status == "RUNNING_STAGE_A":
        state = "STAGE_A_RUNNING"
        if not snapshot.get("worker_process", {}).get("matches"):
            alerts.append("stage_A_worker_missing_or_pid_reused")
        elif snapshot.get("seconds_since_activity", 0) > stale_seconds:
            alerts.append("no_training_calibration_or_checkpoint_activity_for_10_minutes")
    else:
        state = pipeline_status or "AWAITING_PIPELINE"
    for key in ("image_loss", "mse", "lpips", "gradient_norm_before_clip", "update_seconds"):
        value = snapshot.get("last_training_row", {}).get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            alerts.append("nonfinite_" + key)
    if snapshot.get("disk_free_GiB", 100) < 40:
        alerts.append("disk_free_below_40_GiB_no_automatic_deletion")
    if snapshot.get("gpu", {}).get("temperature_C", 0) >= 85:
        alerts.append("gpu_temperature_at_least_85C_no_automatic_clock_change")
    return state, alerts


def collect(run):
    stage = run / "stage_A_v1"
    pipeline = read_json(run / "pipeline_001/status.json")
    training = read_json(stage / "status.json")
    latest = read_json(stage / "latest.json")
    calibration = [read_json(path) for path in sorted((stage / "calibration").glob("full_*.json"))]
    paths = [stage / "status.json", stage / "training.jsonl", stage / "latest.json", run / "stage_A_v1.log"]
    paths.extend((stage / "calibration").glob("*.json"))
    activity = max((path.stat().st_mtime for path in paths if path.exists()), default=time.time())
    now = time.time()
    snapshot = {"timestamp": now, "local_time": datetime.now().astimezone().isoformat(),
        "utc_time": datetime.now(timezone.utc).isoformat(), "monitor_pid": os.getpid(),
        "pipeline": pipeline, "training": training,
        "pipeline_process": process_state(pipeline.get("pipeline_pid", pipeline.get("pid")), "latent_enhancement.pipeline"),
        "worker_process": process_state(pipeline.get("worker_pid"), "latent_enhancement.stage_a"),
        "gpu": gpu_state(), "disk_free_GiB": shutil.disk_usage(run).free / 2 ** 30,
        "seconds_since_activity": now - activity, "last_training_row": latest_training_row(stage / "training.jsonl"),
        "last_checkpoint_step": latest.get("step"), "plateau_checks": latest.get("state", {}).get("plateau_checks"),
        "selected": read_json(stage / "selected.json"), "completion": read_json(stage / "completion.json"),
        "continuous_reference_result": read_json(stage / "continuous_reference_result.json"),
        "full_calibration_curve": [{"step": record["step"], "sources": record["sources"], "summary": record["summary"]}
                                   for record in calibration],
        "changed_sources": source_integrity(read_json(stage / "registration.json")),
        "observer_does_not_change_models_optimizer_config_or_resources": True}
    snapshot["state"], snapshot["alerts"] = classify(snapshot)
    return snapshot


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {name: safe_json(item) for name, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("observer interval must be at least ten seconds")
    directory = args.run / "monitor_001"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "observer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_event = None
        while True:
            try:
                snapshot = collect(args.run)
                curve = snapshot["full_calibration_curve"]
                key = (snapshot["state"], tuple(snapshot["alerts"]), curve[-1]["step"] if curve else -1,
                       snapshot.get("selected", {}).get("step"))
                serializable = safe_json(snapshot)
                atomic_json(directory / "status.json", serializable)
                with (directory / "history.jsonl").open("a", buffering=1) as handle:
                    handle.write(json.dumps({name: value for name, value in serializable.items()
                                            if name not in ("full_calibration_curve", "continuous_reference_result")}, ensure_ascii=False) + "\n")
                if key != previous_event:
                    event = {"local_time": snapshot["local_time"], "state": snapshot["state"], "alerts": snapshot["alerts"],
                        "step": snapshot["training"].get("step"), "selected_step": snapshot.get("selected", {}).get("step"),
                        "latest_full_calibration": curve[-1] if curve else None,
                        "continuous_reference_result": snapshot["continuous_reference_result"],
                        "chat_notification_sent": False, "no_external_notification_channel_configured": True}
                    with (directory / "events.jsonl").open("a", buffering=1) as handle:
                        handle.write(json.dumps(safe_json(event), ensure_ascii=False) + "\n")
                    print(json.dumps(safe_json(event), ensure_ascii=False), flush=True)
                    previous_event = key
                if snapshot["completion"]:
                    atomic_json(directory / "stage_A_completion_notice.json", {"completion": snapshot["completion"],
                        "continuous_reference_result": snapshot["continuous_reference_result"],
                        "stage_B_started": False, "experiment_complete": False})
            except Exception as error:
                failure = {"timestamp": time.time(), "monitor_pid": os.getpid(), "state": "OBSERVER_READ_ERROR_RETRY_NEXT_INTERVAL",
                           "error_type": type(error).__name__, "error": str(error), "worker_untouched": True}
                atomic_json(directory / "status.json", failure)
                with (directory / "observer_errors.jsonl").open("a", buffering=1) as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(json.dumps(failure, ensure_ascii=False), flush=True)
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
