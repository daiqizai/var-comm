#!/usr/bin/env python3
"""Bounded authorized finisher: existing holdout matrix -> fixed references -> audits/statistics."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/COMMUNICATION-CONVERGENCE-20260915"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-pid", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("PLAN ONLY: no training, architecture, timing rerun or parameter search")
        return
    output = RUN / "VALIDATION_FINISHER_001"
    output.mkdir(exist_ok=False)
    frozen = RUN / "FINAL_FREEZE_001"
    method, manifest = frozen / "frozen_method.json", frozen / "holdout_manifest.json"
    bindings = {str(path): digest(path) for path in (method, manifest, RUN / "POLICIES_001/policies.json")}

    def status(value, **extra):
        record = {"status": value, "pid": os.getpid(), "updated_local": datetime.now().astimezone().isoformat(),
                  "frozen_inputs": bindings, "new_training_or_search": False, "research_goal_complete": False, **extra}
        temporary = output / "status.pending.json"
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def unchanged():
        if any(digest(path) != value for path, value in bindings.items()):
            raise RuntimeError("frozen method, data list or policy changed during validation")

    def start(name, script, options):
        unchanged()
        command = [sys.executable, "-u", str(ROOT / "scripts" / script), *map(str, options)]
        log = (RUN / f"{name}.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        log.close()
        (output / f"{name}.json").write_text(json.dumps({"pid": process.pid, "command": command,
            "started_local": datetime.now().astimezone().isoformat()}, indent=2) + "\n")
        return process

    try:
        status("WAITING_FOR_EXISTING_LIVE_HOLDOUT_MATRIX", matrix_pid=arguments.matrix_pid)
        while True:
            process_path = Path(f"/proc/{arguments.matrix_pid}/cmdline")
            if not process_path.exists():
                break
            command = process_path.read_bytes().replace(b"\0", b" ")
            if b"evaluate_communication_modes.py" not in command or b"--population holdout" not in command:
                break
            unchanged()
            time.sleep(10)
        matrix_receipt = RUN / "HOLDOUT_001/completion.json"
        matrix = json.loads(matrix_receipt.read_text())
        if matrix["status"] != "FIXED_MODE_MATRIX_COMPLETE" or matrix["sources"] != 1000 or matrix["rows"] != 126000:
            raise RuntimeError("the existing holdout matrix did not complete; do not silently restart it")
        audit = start("holdout_audit_001", "audit_communication_matrix.py", ["--run-dir", RUN / "HOLDOUT_001",
            "--output-dir", RUN / "HOLDOUT_AUDIT_001"])
        references = start("holdout_references_001", "evaluate_frozen_holdout_references.py", ["--execute", "--mode", "holdout",
            "--frozen-method", method, "--holdout-manifest", manifest, "--output-dir", RUN / "HOLDOUT_REFERENCES_001"])
        status("FIXED_REFERENCES_AND_MATRIX_AUDIT_RUNNING", reference_pid=references.pid, matrix_audit_pid=audit.pid)
        reference_return = references.wait()
        audit_return = audit.wait()
        if reference_return or audit_return:
            raise RuntimeError(f"reference/audit stopped ({reference_return}, {audit_return}); logs and partial results retained")
        reference_audit = start("holdout_reference_audit_001", "audit_frozen_holdout_references.py", ["--run-dir", RUN / "HOLDOUT_REFERENCES_001",
            "--frozen-method", method, "--holdout-manifest", manifest, "--output-dir", RUN / "HOLDOUT_REFERENCE_AUDIT_001"])
        status("REFERENCE_CPU_AUDIT_RUNNING", reference_audit_pid=reference_audit.pid)
        if reference_audit.wait():
            raise RuntimeError("holdout reference CPU audit failed")
        analysis = start("holdout_analysis_001", "analyze_communication_policies.py", ["--matrix-run", RUN / "HOLDOUT_001",
            "--matrix-audit", RUN / "HOLDOUT_AUDIT_001/completion.json", "--policies", RUN / "POLICIES_001/policies.json",
            "--reference-run", RUN / "HOLDOUT_REFERENCES_001", "--output-dir", RUN / "HOLDOUT_ANALYSIS_001"])
        status("FIXED_HOLDOUT_STATISTICS_RUNNING", analysis_pid=analysis.pid)
        if analysis.wait():
            raise RuntimeError("frozen holdout policy analysis failed")
        unchanged()
        receipts = [RUN / name / "completion.json" for name in ("HOLDOUT_001", "HOLDOUT_AUDIT_001", "HOLDOUT_REFERENCES_001",
                                                               "HOLDOUT_REFERENCE_AUDIT_001", "HOLDOUT_ANALYSIS_001")]
        status("FROZEN_VALIDATION_NUMERICS_COMPLETE_REPORT_PENDING", receipt_sha256={str(path): digest(path) for path in receipts})
        (output / "completion.json").write_text(json.dumps({"status": "FROZEN_VALIDATION_NUMERICS_COMPLETE_REPORT_PENDING",
            "completed_local": datetime.now().astimezone().isoformat(), "receipts": {str(path): digest(path) for path in receipts},
            "frozen_inputs": bindings, "research_goal_complete": False}, indent=2) + "\n")
    except BaseException as error:
        status("STOPPED_WITH_FAILURE_RETAINED", error=repr(error))
        (output / "failure.json").write_text(json.dumps({"error": repr(error), "time": datetime.now().astimezone().isoformat()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
