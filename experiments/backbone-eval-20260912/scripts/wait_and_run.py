#!/usr/bin/env python3
"""Bounded, standard-library-only, one-shot local GPU evaluation queue.

No CUDA context is created by this supervisor. Only read-only nvidia-smi queries
are used. It does not train, download assets, reset GPUs, or signal foreign jobs.
A per-run flock is NOT a cluster reservation: unrelated users can still race the
last idle observation. The evaluation relinquishes the GPU on detected overlap.

Usage: python3 -u -B wait_and_run.py --config /absolute/path/queue.json
The optional --command remainder replaces config.command (no shell expansion).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT / "outputs"
DEFAULT_BLOCKER_STATUS = (
    "/home/liulu/projects/var-next-scale-comm/outputs/"
    "VAR-TOKEN-BACKBONE-20260911-PIPELINE/status.json"
)
TERMINAL_BLOCKER_STATES = {"ALL_STAGES_COMPLETE", "STOPPED_NOT_COMPLETE"}
KNOWN_BLOCKER_STATES = TERMINAL_BLOCKER_STATES | {"RUNNING", "STAGE_COMPLETE", "STARTING"}


class SafetyError(RuntimeError):
    """Insufficient evidence to start/continue safely; fail closed."""


@dataclass(frozen=True)
class ProcIdentity:
    pid: int
    start_ticks: int
    boot_id: str
    uid: int
    pgrp: int
    session: int
    state: str

    def fingerprint(self) -> tuple[int, int, str, int]:
        # state can change while the process identity stays the same.
        return self.pid, self.start_ticks, self.boot_id, self.uid


def parse_proc_stat(text: str, uid: int, boot_id: str) -> ProcIdentity:
    left, right = text.find("("), text.rfind(")")
    if left < 1 or right <= left:
        raise SafetyError("Malformed /proc PID stat")
    parts = text[right + 1:].split()
    if len(parts) < 20:
        raise SafetyError("Truncated /proc PID stat")
    try:
        return ProcIdentity(int(text[:left].strip()), int(parts[19]), boot_id,
                            uid, int(parts[2]), int(parts[3]), parts[0])
    except (ValueError, IndexError) as error:
        raise SafetyError("Invalid /proc PID stat fields") from error


def read_proc(pid: int, boot_id: str) -> ProcIdentity | None:
    path = Path("/proc") / str(pid)
    try:
        first = parse_proc_stat((path / "stat").read_text(), path.stat().st_uid, boot_id)
        second = parse_proc_stat((path / "stat").read_text(), path.stat().st_uid, boot_id)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise SafetyError(f"Cannot verify process {pid}: {error}") from error
    if first.fingerprint() != second.fingerprint():
        raise SafetyError(f"Process {pid} changed identity during inspection")
    return second


@dataclass(frozen=True)
class GPUSnapshot:
    uuid: str
    utilization_percent: int
    memory_used_mib: int
    compute_pids: tuple[int, ...]

    def idle(self, utilization_limit: int, memory_limit: int) -> bool:
        return (not self.compute_pids and self.utilization_percent <= utilization_limit
                and self.memory_used_mib <= memory_limit)


class GPUProbe:
    def __init__(self, uuid: str, timeout: float = 10):
        self.uuid, self.timeout = uuid, timeout

    def _query(self, field: str) -> list[list[str]]:
        command = ["nvidia-smi", "-i", self.uuid, field, "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False,
                                    timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SafetyError(f"Read-only nvidia-smi query failed: {error}") from error
        if result.returncode != 0 or result.stderr.strip():
            raise SafetyError(f"nvidia-smi not clean: rc={result.returncode}, "
                              f"stderr={result.stderr.strip()[:1000]}")
        return [[part.strip() for part in row] for row in csv.reader(io.StringIO(result.stdout))
                if row and any(part.strip() for part in row)]

    def sample(self) -> GPUSnapshot:
        rows = self._query("--query-gpu=uuid,utilization.gpu,memory.used")
        if len(rows) != 1 or len(rows[0]) != 3 or rows[0][0] != self.uuid:
            raise SafetyError(f"Unexpected GPU UUID/count/fields: {rows!r}")
        try:
            utilization, memory = int(rows[0][1]), int(rows[0][2])
        except ValueError as error:
            raise SafetyError(f"GPU utilization or memory is unavailable: {rows!r}") from error
        if not 0 <= utilization <= 100 or memory < 0:
            raise SafetyError("Invalid GPU utilization or memory")
        pids = []
        for row in self._query("--query-compute-apps=gpu_uuid,pid"):
            if len(row) != 2 or row[0] != self.uuid:
                raise SafetyError(f"Unexpected compute-process record: {row!r}")
            try:
                pid = int(row[1])
            except ValueError as error:
                raise SafetyError(f"Unavailable GPU compute PID: {row!r}") from error
            if pid <= 1:
                raise SafetyError("Unsafe/unavailable GPU compute PID")
            pids.append(pid)
        return GPUSnapshot(self.uuid, utilization, memory, tuple(sorted(set(pids))))


@dataclass(frozen=True)
class Config:
    task_name: str
    output_dir: Path
    allowed_gpu_uuid: str
    command: tuple[str, ...]
    cwd: Path
    ready_paths: tuple[Path, ...] = ()
    blocker_pid: int = 1496953
    blocker_status_path: Path = Path(DEFAULT_BLOCKER_STATUS)
    poll_seconds: float = 60
    idle_polls: int = 3
    utilization_limit_percent: int = 5
    memory_limit_mib: int = 512
    max_wait_seconds: float = 86400
    max_run_seconds: float = 14400
    running_poll_seconds: float = 5
    terminate_grace_seconds: float = 15

    @classmethod
    def load(cls, path: Path, command: list[str] | None = None) -> "Config":
        raw = json.loads(path.read_text())
        if command:
            raw["command"] = command[1:] if command[0] == "--" else command
        required = {"task_name", "output_dir", "allowed_gpu_uuid", "command", "cwd"}
        if not isinstance(raw, dict) or required - raw.keys():
            raise ValueError(f"Required config keys: {sorted(required)}")
        extra = raw.keys() - cls.__dataclass_fields__.keys()
        if extra:
            raise ValueError(f"Unknown config keys: {sorted(extra)}")
        if (not isinstance(raw["command"], list) or not raw["command"]
                or any(not isinstance(value, str) or not value for value in raw["command"])):
            raise ValueError("command must be a nonempty string array (never a shell string)")
        for key in ("output_dir", "cwd", "blocker_status_path"):
            if key in raw:
                path_value = Path(raw[key])
                if not path_value.is_absolute():
                    raise ValueError(f"{key} must be absolute")
                raw[key] = path_value.resolve()
        raw["command"] = tuple(raw["command"])
        if "ready_paths" in raw:
            if not isinstance(raw["ready_paths"], list) or any(not isinstance(item, str) for item in raw["ready_paths"]):
                raise ValueError("ready_paths must be an array of absolute file paths")
            raw["ready_paths"] = tuple(Path(item) for item in raw["ready_paths"])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if (not isinstance(self.task_name, str) or len(self.task_name) > 63
                or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", self.task_name)
                or "liulu" not in self.task_name.split("-")):
            raise ValueError("task_name must use lowercase/hyphens, include liulu, and be <=63 chars")
        if (not self.output_dir.is_relative_to(OUTPUT_ROOT.resolve())
                or self.output_dir.name != self.task_name):
            raise ValueError(f"output_dir must be a new {self.task_name} run under {OUTPUT_ROOT}")
        if any(not path.is_absolute() for path in self.ready_paths):
            raise ValueError("ready_paths must contain only absolute file paths")
        if not self.cwd.is_dir() or not Path(self.command[0]).is_absolute():
            raise ValueError("cwd must exist and command executable must be an absolute path")
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{8,}", self.allowed_gpu_uuid):
            raise ValueError("allowed_gpu_uuid must be one explicit physical GPU UUID")
        if not isinstance(self.blocker_pid, int) or isinstance(self.blocker_pid, bool) or self.blocker_pid <= 1:
            raise ValueError("blocker_pid must be a positive non-system PID")
        bounds = {
            "poll_seconds": (60, 3600), "idle_polls": (3, 60),
            "utilization_limit_percent": (0, 5), "memory_limit_mib": (0, 512),
            "max_wait_seconds": (120, 86400), "max_run_seconds": (1, 86400),
            "running_poll_seconds": (1, 60), "terminate_grace_seconds": (1, 60),
        }
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
        if not isinstance(self.idle_polls, int):
            raise ValueError("idle_polls must be an integer")
        if self.max_wait_seconds < (self.idle_polls - 1) * self.poll_seconds:
            raise ValueError("max_wait_seconds is shorter than the required idle confirmation")

    def serializable(self) -> dict[str, Any]:
        result = {key: str(value) if isinstance(value, Path) else value
                  for key, value in asdict(self).items()}
        result["ready_paths"] = [str(path) for path in self.ready_paths]
        return result

    def missing_ready_paths(self) -> list[str]:
        return [str(path) for path in self.ready_paths if not path.is_file()]


class Blocker:
    """Gate on a whole existing train/evaluate pipeline, never a single CUDA PID."""
    def __init__(self, config: Config, boot_id: str):
        self.config, self.boot_id = config, boot_id
        self.original = read_proc(config.blocker_pid, boot_id)
        initial = self.sample()
        if self.original is None and not initial["pipeline_terminal"]:
            raise SafetyError("Original supervisor already absent but pipeline has no terminal status; "
                              "manual verification required, no automatic launch")

    def sample(self) -> dict[str, Any]:
        current = read_proc(self.config.blocker_pid, self.boot_id)
        same = (self.original is not None and current is not None
                and current.fingerprint() == self.original.fingerprint())
        # A zombie no longer executes; the terminal status must still be present.
        alive = same and current.state != "Z"
        try:
            status = json.loads(self.config.blocker_status_path.read_text())
        except (OSError, ValueError) as error:
            raise SafetyError(f"Cannot read existing pipeline status: {error}") from error
        if (not isinstance(status, dict) or status.get("supervisor_pid") != self.config.blocker_pid
                or status.get("status") not in KNOWN_BLOCKER_STATES):
            raise SafetyError("Existing pipeline status identity/state changed or is unknown")
        terminal = status["status"] in TERMINAL_BLOCKER_STATES
        return {"original_identity": asdict(self.original) if self.original else None,
                "original_supervisor_alive": alive,
                "pid_reused": current is not None and not same,
                "pipeline_status": status["status"], "pipeline_stage": status.get("stage"),
                "pipeline_terminal": terminal, "ready": not alive and terminal}


class RunFiles:
    def __init__(self, config: Config):
        self.config, self.lock = config, None
        self.started = datetime.now().astimezone().isoformat()
        self.latest: dict[str, Any] = {}

    def __enter__(self) -> "RunFiles":
        directory = self.config.output_dir
        directory.mkdir(parents=True, exist_ok=True)
        self.lock = (directory / "queue.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Persistent marker prevents a sequential second launch, not just overlap.
            with (directory / "queue-started.json").open("x") as file:
                json.dump({"started_local": self.started, "supervisor_pid": os.getpid(),
                           "task_name": self.config.task_name}, file, indent=2)
            with (directory / "queue-config.json").open("x") as file:
                json.dump(self.config.serializable(), file, indent=2)
            if (directory / "evaluation.log").exists() or (directory / "queue-status.json").exists():
                raise SafetyError("Refusing to overwrite existing evaluation/queue output")
        except BaseException:
            self.lock.close()
            raise
        return self

    def record(self, state: str, **details: Any) -> None:
        record = {"task_name": self.config.task_name, "state": state,
                  "started_local": self.started,
                  "updated_local": datetime.now().astimezone().isoformat(),
                  "supervisor_pid": os.getpid(), **details}
        directory = self.config.output_dir
        temporary = directory / f".queue-status-{os.getpid()}.pending"
        with temporary.open("w") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(directory / "queue-status.json")
        with (directory / "queue-events.jsonl").open("a") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.latest = record
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def __exit__(self, *args: Any) -> None:
        if self.lock is not None:
            self.lock.close()


def child_environment(uuid: str, task_name: str) -> dict[str, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TOKENIZERS_PARALLELISM": "false", "VAR_COMM_TASK_NAME": task_name}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "2"
    return env


def lower_own_priority() -> None:
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))


class OwnChild:
    """A retained, unreaped direct-child PID anchors safe process-group signals.

    Do NOT call Popen.poll() while supervising: it reaps an exited leader, which
    can permit PID reuse before a group signal. WNOWAIT retains the leader until
    its own process group has been cleaned up.
    """
    def __init__(self, config: Config, boot_id: str, logfile: Any):
        self.boot_id, self.reaped = boot_id, False
        self.process = subprocess.Popen(
            list(config.command), cwd=config.cwd, env=child_environment(config.allowed_gpu_uuid, config.task_name),
            stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=lower_own_priority,
        )
        try:
            self.identity = read_proc(self.process.pid, boot_id)
            if (self.identity is None or self.identity.pgrp != self.process.pid
                    or self.identity.session != self.process.pid or self.identity.uid != os.getuid()):
                raise SafetyError("Cannot establish evaluation process-group ownership")
        except BaseException:
            # Direct unreaped child is ours, but an unverified group is not signalled.
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.reaped = True
            raise

    def exited(self) -> bool:
        return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def _verified_leader(self) -> ProcIdentity:
        if self.reaped:
            raise SafetyError("Refusing to signal a reaped child group")
        current = read_proc(self.process.pid, self.boot_id)
        if (current is None or current.fingerprint() != self.identity.fingerprint()
                or current.pgrp != self.process.pid or current.session != self.process.pid):
            raise SafetyError("Evaluation process-group ownership no longer verifiable")
        return current

    def owns_gpu_pid(self, pid: int) -> bool:
        self._verified_leader()
        current = read_proc(pid, self.boot_id)
        return (current is not None and current.pgrp == self.process.pid
                and current.session == self.process.pid and current.uid == self.identity.uid
                and current.start_ticks >= self.identity.start_ticks)

    def _members(self) -> list[ProcIdentity]:
        self._verified_leader()
        members = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            item = read_proc(int(path.name), self.boot_id)
            if item is not None and item.pgrp == self.process.pid:
                if (item.session != self.process.pid or item.uid != self.identity.uid
                        or item.start_ticks < self.identity.start_ticks):
                    raise SafetyError("Unexpected identity in evaluation process group")
                members.append(item)
        return members

    def signal_group(self, signum: int) -> None:
        self._verified_leader()
        # The direct child has not been reaped. Its PID cannot be recycled between
        # the fingerprint verification above and this signal.
        os.killpg(self.process.pid, signum)

    def stop(self, grace: float) -> int:
        self.signal_group(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(item.state != "Z" for item in self._members()):
                break
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        if any(item.state != "Z" for item in self._members()):
            self.signal_group(signal.SIGKILL)
        result = self.process.wait(timeout=5)
        self.reaped = True
        return result

    def finish(self) -> tuple[int, bool]:
        descendants = any(item.pid != self.process.pid and item.state != "Z" for item in self._members())
        if descendants:
            return self.stop(5), True
        result = self.process.wait(timeout=5)
        self.reaped = True
        return result, False


class Supervisor:
    def __init__(self, config: Config, files: RunFiles):
        self.config, self.files = config, files
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not self.boot_id:
            raise SafetyError("Missing Linux boot identity")
        self.probe = GPUProbe(config.allowed_gpu_uuid)
        self.blocker = Blocker(config, self.boot_id)
        self.signal_received: int | None = None
        self.child: OwnChild | None = None
        self.previous_handlers: dict[int, Any] = {}

    def signal_handler(self, signum: int, _frame: Any) -> None:
        # No group signal from an asynchronous handler; safe synchronous cleanup
        # below validates only our direct child's group, never blocker/foreign PIDs.
        self.signal_received = signum

    def sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0, seconds)
        while not self.signal_received and time.monotonic() < deadline:
            time.sleep(max(0.0, min(1.0, deadline - time.monotonic())))

    def wait_for_idle(self) -> bool:
        deadline = time.monotonic() + self.config.max_wait_seconds
        streak = 0
        last_idle = None
        while time.monotonic() < deadline:
            if self.signal_received:
                self.files.record("CANCELLED_SIGNAL", signal=self.signal_received, evaluation_launched=False)
                return False
            blocker = self.blocker.sample()
            gpu = self.probe.sample()
            now = time.monotonic()
            if now > deadline:
                break
            missing_ready = self.config.missing_ready_paths()
            ready = (not missing_ready and blocker["ready"]
                     and gpu.idle(self.config.utilization_limit_percent, self.config.memory_limit_mib))
            if ready:
                if last_idle is None or now - last_idle >= self.config.poll_seconds:
                    streak += 1
                    last_idle = now
            else:
                streak, last_idle = 0, None
            state = ("WAITING_ASSETS" if missing_ready else
                     "WAITING_PIPELINE" if not blocker["ready"] else
                     "IDLE_CONFIRMATION" if ready else "WAITING_GPU")
            self.files.record(state, blocker=blocker, gpu=asdict(gpu), idle_streak=streak,
                              missing_ready_paths=missing_ready,
                              required_idle_polls=self.config.idle_polls,
                              wait_remaining_seconds=max(0, round(deadline - now, 2)),
                              evaluation_launched=False)
            if streak >= self.config.idle_polls:
                # One final fresh pair of queries closes long I/O gaps before launch.
                final_blocker, final_gpu = self.blocker.sample(), self.probe.sample()
                if (not self.signal_received and time.monotonic() <= deadline and final_blocker["ready"]
                        and not self.config.missing_ready_paths()
                        and final_gpu.idle(self.config.utilization_limit_percent, self.config.memory_limit_mib)):
                    return True
                streak, last_idle = 0, None
            self.sleep(min(self.config.poll_seconds, max(0, deadline - time.monotonic())))
        self.files.record("WAIT_TIMEOUT", max_wait_seconds=self.config.max_wait_seconds,
                          evaluation_launched=False)
        return False

    def run_child(self) -> int:
        with (self.config.output_dir / "evaluation.log").open("x") as logfile:
            self.child = OwnChild(self.config, self.boot_id, logfile)
            deadline = time.monotonic() + self.config.max_run_seconds
            try:
                self.files.record("RUNNING", child_identity=asdict(self.child.identity),
                                  command=list(self.config.command), evaluation_launched=True,
                                  max_run_seconds=self.config.max_run_seconds)
                while True:
                    reason, details = None, {}
                    if self.signal_received:
                        reason, details = "CANCELLED_SIGNAL", {"signal": self.signal_received}
                    elif time.monotonic() >= deadline:
                        reason = "RUN_TIMEOUT"
                    else:
                        gpu = self.probe.sample()
                        foreign = [pid for pid in gpu.compute_pids if not self.child.owns_gpu_pid(pid)]
                        if foreign:
                            reason, details = "ABORTED_FOREIGN_GPU_PROCESS", {"foreign_compute_pids": foreign}
                        elif self.child.exited():
                            code, cleaned_descendants = self.child.finish()
                            state = ("ABORTED_BACKGROUND_DESCENDANTS" if cleaned_descendants else
                                     "COMPLETED" if code == 0 else "EVALUATION_FAILED")
                            self.files.record(state, child_returncode=code, evaluation_launched=True,
                                              cleaned_background_descendants=cleaned_descendants)
                            return 0 if code == 0 and not cleaned_descendants else 1
                        else:
                            self.files.record("RUNNING", child_identity=asdict(self.child.identity),
                                              gpu=asdict(gpu), evaluation_launched=True,
                                              run_remaining_seconds=max(0, round(deadline - time.monotonic(), 2)))
                    if reason:
                        code = self.child.stop(self.config.terminate_grace_seconds)
                        self.files.record(reason, child_returncode=code, evaluation_launched=True, **details)
                        return 1
                    self.sleep(min(self.config.running_poll_seconds, max(0, deadline - time.monotonic())))
            except BaseException as error:
                cleanup_error = None
                if self.child is not None and not self.child.reaped:
                    try:
                        self.child.stop(self.config.terminate_grace_seconds)
                    except BaseException as stopping:
                        cleanup_error = repr(stopping)
                self.files.record("ABORTED_MONITOR_ERROR", error=repr(error), cleanup_error=cleanup_error,
                                  evaluation_launched=True)
                return 1

    def run(self) -> int:
        try:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                self.previous_handlers[signum] = signal.signal(signum, self.signal_handler)
            self.files.record("QUEUED", allowed_gpu_uuid=self.config.allowed_gpu_uuid,
                              blocker=self.blocker.sample(), evaluation_launched=False,
                              command_sha256=hashlib.sha256(json.dumps(self.config.command).encode()).hexdigest())
            if not self.wait_for_idle():
                return 1
            if self.signal_received:
                self.files.record("CANCELLED_SIGNAL", signal=self.signal_received, evaluation_launched=False)
                return 1
            return self.run_child()
        except (SafetyError, OSError, ValueError) as error:
            self.files.record("ABORTED_MONITOR_ERROR", error=repr(error), evaluation_launched=self.child is not None)
            return 1
        finally:
            for signum, previous in self.previous_handlers.items():
                signal.signal(signum, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate", action="store_true",
                        help="CPU-only config validation; no monitor, output writes, or evaluation")
    parser.add_argument("--task-name", help="Optional visible ownership tag; must match config.task_name")
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="Optional argv override; no shell parsing/expansion")
    args = parser.parse_args()
    try:
        config = Config.load(args.config, args.command)
        if args.task_name is not None and args.task_name != config.task_name:
            raise ValueError("--task-name must match config.task_name exactly")
        if args.validate:
            print(json.dumps({"valid": True, "no_gpu_query": True, "no_evaluation_launch": True,
                              "no_output_files_written": True,
                              "missing_ready_paths": config.missing_ready_paths(),
                              "config": config.serializable()}, indent=2))
            return 0
        lower_own_priority()
        with RunFiles(config) as files:
            try:
                return Supervisor(config, files).run()
            except (SafetyError, OSError, ValueError) as error:
                files.record("ABORTED_MONITOR_ERROR", error=repr(error), evaluation_launched=False)
                return 1
    except (OSError, ValueError, SafetyError) as error:
        print(f"Refusing queue start: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
