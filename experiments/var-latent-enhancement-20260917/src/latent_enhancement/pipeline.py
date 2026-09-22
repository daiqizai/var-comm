from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

from .runtime import OUTPUT, wait_for_available, write_json


def main():
    directory = OUTPUT / "pipeline_001"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for module, receipt in (("cache", OUTPUT / "cache_v1/completion.json"),
                                ("stage_a", OUTPUT / "stage_A_v1/completion.json")):
            if receipt.exists():
                continue
            while True:
                wait_for_available(directory / "status.json")
                log_path = OUTPUT / ("cache_v1.log" if module == "cache" else "stage_A_v1.log")
                with log_path.open("a", buffering=1) as log:
                    child = subprocess.Popen([sys.executable, "-u", "-m", "latent_enhancement." + module], stdout=log, stderr=subprocess.STDOUT)
                    write_json(directory / "status.json", {"status": "RUNNING_" + module.upper(),
                        "pipeline_pid": os.getpid(), "worker_pid": child.pid, "timestamp": time.time(),
                        "log": str(log_path), "HiFi_resumption": False})
                    code = child.wait()
                if code == 75:
                    write_json(directory / f"resource_yield_{time.time_ns()}.json", {"module": module, "exit_code": code,
                                                                                     "worker_pid": child.pid})
                    time.sleep(30)
                    continue
                if code or not receipt.exists():
                    write_json(directory / "status.json", {"status": "STOPPED_CHILD_FAILURE_OR_INTERRUPT", "module": module,
                        "exit_code": code, "worker_pid": child.pid, "timestamp": time.time(), "no_automatic_failure_retry": True})
                    raise SystemExit(code or 1)
                break
        result = json.loads((OUTPUT / "stage_A_v1/continuous_reference_result.json").read_text())
        write_json(directory / "status.json", {"status": "STAGE_A_FINISHED_REFERENCE_REVIEW_REQUIRED",
            "stage_B_eligible": result["stage_B_eligible"], "stage_B_started": False,
            "full_experiment_complete": False, "timestamp": time.time(),
            "next_action": result["next_action"], "no_HiFi_or_old_hybrid_restart": True})


if __name__ == "__main__":
    main()
