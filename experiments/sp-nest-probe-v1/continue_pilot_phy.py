"""Run the two Pilot50 controls on the same GPU after the superposition worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_bytes())


def record(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2)


def ticks(pid):
    try:
        return int(Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def confirmed(pilot, role, deadline):
    name = "sp1_" + role
    control = pilot / (name + "_control")
    child = read(control / "child.json")["worker"]
    while time.time() < deadline:
        receipt = control / "actual_wait.json"
        if receipt.exists() and ticks(child["pid"]) != child["start_ticks"]:
            done = read(receipt)
            if done["returncode"] or done["timed_out"] or done["pid"] != child["pid"]:
                raise RuntimeError((name, done))
            return read(pilot / name / "completion.json")
        time.sleep(3)
    raise TimeoutError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    pilot = args.pilot.resolve()
    first = pilot / "sp1_sp_control"
    source = read(pilot / "source_control" / "child.json")
    deadline = source["started"] + 86400
    owner = read(first / "owner.json")
    command = read(first / "child.json")["argv"]
    worker = pilot / "code" / "a800_broadcast_phy.py"
    if hashlib.sha256(worker.read_bytes()).hexdigest() != owner["binding_sha256"]:
        raise RuntimeError("PHY implementation changed")
    if Path(command[2]) != worker or command[command.index("--family") + 1] != "sp":
        raise ValueError("unexpected original launch")
    if os.environ["CUDA_VISIBLE_DEVICES"] != owner["gpu"]:
        raise RuntimeError("GPU differs from the original worker")
    plan = read(pilot / "sp1_plan" / "completion.json")
    receipts = {}
    for role in ("sp", "single", "oma"):
        name = "sp1_" + role
        if role != "sp":
            out, control = pilot / name, pilot / (name + "_control")
            if out.exists():
                raise FileExistsError(out)
            stride = int(min(12 * 3600, deadline - time.time()))
            if stride < 1:
                raise TimeoutError(name)
            control.mkdir()
            argv = list(command)
            argv[argv.index("--family") + 1] = role
            argv[argv.index("--out") + 1] = str(out)
            with (control / "worker.log").open("xb") as log:
                proc = subprocess.Popen([sys.executable, str(pilot / "wait_stage.py"),
                    str(control), str(stride), *argv], cwd=worker.parent,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                record(control / "owner.json", dict(owner=proc.pid, start_ticks=ticks(proc.pid),
                    binding_sha256=owner["binding_sha256"], gpu=owner["gpu"], deadline=deadline))
                code = proc.wait()
            if code:
                raise RuntimeError((name, "recorded failure", code))
        receipts[role] = confirmed(pilot, role, deadline)
        measured = receipts[role]
        if measured["waves"] != 3 * plan["unique_cases"][role] or measured["new_receiver_observations"] != 3 * measured["waves"]:
            raise ValueError(name + " bounds")
        print(name, measured["waves"], "complete", flush=True)
    record(pilot / "sp1_controls_control" / "families.json", receipts)


if __name__ == "__main__":
    main()
