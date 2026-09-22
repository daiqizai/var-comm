from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from latent_enhancement.runtime import OUTPUT, foreign_gpu_processes, write_json

from .common import OUT_B, freeze_selected_decoder


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def process_matches(pid, module):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z" and any(command[index:index + 2] == ["-m", module] for index in range(len(command) - 1))
    except FileNotFoundError:
        return False


def handoff_decision(a_completion, a_alive, a_status):
    if a_completion:
        return "CHECK_FINAL_A_QUALIFICATION"
    if "FAILURE" in a_status or "INTERRUPT" in a_status:
        return "STOP_A_FAILED_OR_PAUSED"
    return "WAIT_FOR_A" if a_alive else "STOP_A_MISSING_WITHOUT_COMPLETION"


def gpu_summary():
    try:
        result = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                                          "--format=csv,noheader,nounits"], text=True, timeout=10)
        return dict(zip(("memory_used_MiB", "utilization_percent", "temperature_C"), map(float, result.strip().split(","))))
    except Exception as error:
        return {"read_error": str(error)}


def write_status(directory, status, **extra):
    record = {"status": status, "controller_pid": os.getpid(), "timestamp": time.time(), **extra}
    write_json(directory / "status.json", record)
    return record


def wait_gpu(directory, module):
    while True:
        processes = foreign_gpu_processes()
        if not processes:
            return
        write_status(directory, "WAITING_FOR_AUTHORIZED_GPU0", next_module=module, other_GPU_processes=processes)
        time.sleep(30)


def run_child(directory, module, receipt):
    while not receipt.exists():
        if module != "phy_cache":
            wait_gpu(directory, module)
        log_path = OUTPUT / f"phase_B_{module}.log"
        environment = os.environ.copy()
        if module == "phy_cache":
            environment.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        with log_path.open("a", buffering=1) as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "latent_enhancement_b." + module],
                                     stdout=log, stderr=subprocess.STDOUT, env=environment)
            while child.poll() is None:
                stage_directory = OUT_B / ("training" if module == "train" else module)
                child_status = read_json(stage_directory / "status.json")
                calibration_status = read_json(stage_directory / "calibration_status.json")
                activity_paths = [log_path, stage_directory / "status.json", stage_directory / "training.jsonl",
                                  stage_directory / "calibration_status.json", stage_directory / "latest.json"]
                activity = max((path.stat().st_mtime for path in activity_paths if path.exists()), default=time.time())
                warning = []
                if time.time() - activity > 600:
                    warning.append("no_worker_activity_for_10_minutes_review_required")
                gpu = gpu_summary()
                if gpu.get("temperature_C", 0) >= 85:
                    warning.append("GPU_temperature_at_least_85C_no_clock_change")
                record = write_status(directory, "RUNNING_" + module.upper(), worker_pid=child.pid, worker_status=child_status,
                    calibration_status=calibration_status, gpu=gpu, alerts=warning, log=str(log_path),
                    no_new_training_recipe_or_other_route=True)
                with (directory / "monitor_history.jsonl").open("a", buffering=1) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                time.sleep(30)
            code = child.returncode
        if code == 75:
            write_json(directory / f"resource_yield_{time.time_ns()}.json", {"module": module, "worker_pid": child.pid, "exit_code": code})
            time.sleep(30)
            continue
        if code or not receipt.exists():
            write_status(directory, "STOPPED_CHILD_FAILURE_OR_INTERRUPT", module=module, exit_code=code,
                         receipt_missing=not receipt.exists(), log=str(log_path), automatic_failure_retry=False)
            raise RuntimeError(f"{module} stopped with exit {code}; no automatic general-failure retry")


def main():
    directory = OUT_B / "auto_001"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            completion = read_json(OUTPUT / "stage_A_v1/completion.json")
            training = read_json(OUTPUT / "stage_A_v1/status.json")
            pipeline = read_json(OUTPUT / "pipeline_001/status.json")
            alive = (process_matches(training.get("pid"), "latent_enhancement.stage_a") or
                     process_matches(pipeline.get("pipeline_pid", pipeline.get("pid")), "latent_enhancement.pipeline"))
            state = handoff_decision(completion, alive, pipeline.get("status", "") + "/" + training.get("status", ""))
            if state == "CHECK_FINAL_A_QUALIFICATION":
                break
            write_status(directory, state, stage_A_step=training.get("step"), stage_A_status=training.get("status"),
                         stage_B_training_started=False)
            if state != "WAIT_FOR_A":
                raise RuntimeError(state)
            time.sleep(30)
        gate = freeze_selected_decoder()
        if gate is None:
            write_status(directory, "STAGE_A_NOT_QUALIFIED_NO_B_TRAINING", stage_B_training_started=False,
                         no_automatic_model_or_loss_search=True)
            return
        write_status(directory, "STAGE_A_PASSED_SELECTED_DC_FROZEN", selected_stage_A_step=gate["selection"]["step"])
        run_child(directory, "qualify", OUT_B / "qualification/completion.json")
        while not (OUT_B / "phy_cache/completion.json").exists():
            phy_status = read_json(OUT_B / "phy_cache/status.json")
            if process_matches(phy_status.get("pid"), "latent_enhancement_b.phy_cache"):
                write_status(directory, "WAITING_FOR_EXISTING_CPU_PHY_REPLAY", worker_status=phy_status)
                time.sleep(30)
            elif list((OUT_B / "phy_cache").glob("failure_*.json")):
                raise RuntimeError("existing CPU PHY preparation failed; preserve evidence and review")
            else:
                run_child(directory, "phy_cache", OUT_B / "phy_cache/completion.json")
        run_child(directory, "rx_cache", OUT_B / "rx_cache/completion.json")
        run_child(directory, "train", OUT_B / "training/completion.json")
        completion = read_json(OUT_B / "training/completion.json")
        write_status(directory, "STAGE_B_TRAINING_FINISHED_SCIENTIFIC_COMPARISONS_PENDING", completion=completion,
                     full_experiment_complete=False, new_holdout_used=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(OUT_B / "auto_001" / f"failure_{time.time_ns()}.json", {"error": str(error), "type": type(error).__name__})
        raise
