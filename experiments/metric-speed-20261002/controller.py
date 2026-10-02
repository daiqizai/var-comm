"""Replace a paused supervisor after its current worker seals an M1 checkpoint.

This controller never changes the original stage list or experiment code. Only
the retired parent is signalled; the running worker is allowed to finish. Linux
pidfds and the original advisory lock establish ownership before launch/status
records are replaced. Model/runtime work is delegated to unchanged stages or
the separately qualified M2 wrapper.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_NAME = "SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002"
EXTENSION_NAME = "METRIC-SPEED-20261002"
ORIGINAL_DIRECTORY = "experiments/scale-causal-partial-residual-20261002"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seal(path, record):
    path = Path(path)
    if path.exists():
        if read(path) != record:
            raise RuntimeError("An immutable controller record differs: " + str(path))
    else:
        write(path, record)


def resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def verify_bindings(root, items):
    if not isinstance(items, dict) or not items:
        raise RuntimeError("Required frozen bindings are missing")
    for path, expected in items.items():
        if not isinstance(expected, str) or len(expected) != 64 or sha(resolve(root, path)) != expected:
            raise RuntimeError("Frozen input changed: " + str(path))


def sources(directory):
    directory = Path(directory).resolve()
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".py", ".md"):
            continue
        if path.is_symlink() or not path.is_file() or path.parent != directory:
            raise RuntimeError("Runtime extension sources must be regular files in one directory: " + str(path))
        inventory[str(path)] = sha(path)
    if not inventory:
        raise RuntimeError("Empty runtime extension source inventory")
    return inventory


def checked_identity(record):
    if not isinstance(record, dict) or isinstance(record.get("pid"), bool):
        raise RuntimeError("Missing process identity")
    pid = int(record["pid"])
    ticks = str(record["start_ticks"])
    if pid <= 1 or not ticks.isdigit() or int(ticks) <= 0:
        raise RuntimeError("Unsafe process identity")
    return {"pid": pid, "start_ticks": ticks}


def process_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root) / str(int(pid))
    try:
        raw = (directory / "stat").read_text()
    except FileNotFoundError:
        return None
    # The process name in parentheses may itself contain spaces or ')'.
    boundary = raw.rfind(") ")
    if boundary < 0:
        raise RuntimeError("Unreadable process stat: " + str(pid))
    fields = raw[boundary + 2:].split()
    if len(fields) < 20:
        raise RuntimeError("Incomplete process stat: " + str(pid))
    try:
        command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except FileNotFoundError:
        command = ""
    return {"pid": int(pid), "state": fields[0], "start_ticks": fields[19], "command": command}


def exited(expected, actual):
    return actual is None or str(actual["start_ticks"]) != str(expected["start_ticks"]) or actual["state"] == "Z"


def retire_with_pidfd(expected, reader=process_identity, *, allow_running=False):
    """Signal an identity-bound parent; never fall back to an unbound kill(pid)."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required for safe parent retirement")
    expected = checked_identity(expected)
    descriptor = os.pidfd_open(expected["pid"], 0)
    try:
        actual = reader(expected["pid"])
        if actual is None or str(actual["start_ticks"]) != expected["start_ticks"] or actual["state"] == "Z":
            raise RuntimeError("Old parent identity changed before retirement")
        if actual["state"] not in ("T", "t") and not allow_running:
            raise RuntimeError("Old supervisor must already be paused by the operator")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
        # A pending SIGTERM cannot take effect until a stopped task is resumed.
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGCONT, None, 0)
        except ProcessLookupError:
            pass
    finally:
        os.close(descriptor)


def acquire_lock(path):
    import fcntl  # Native execution is Linux; CPU fixtures can import on Windows.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def published(record):
    commit = record.get("commit", "")
    if (record.get("status") != "PUSHED" or record.get("checks") != "PASS"
            or record.get("remote_commit") != commit or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise RuntimeError("Verified pushed publication is required")


class Controller:
    def __init__(self, root=ROOT, here=HERE, *, reader=process_identity,
                 retire=retire_with_pidfd, lock=acquire_lock, execute=None,
                 sleep=time.sleep, now=time.time, pid=None):
        self.root, self.here = Path(root).resolve(), Path(here).resolve()
        self.parent = self.root / "outputs" / PARENT_NAME
        self.out = self.root / "outputs" / EXTENSION_NAME
        self.original = self.root / ORIGINAL_DIRECTORY
        self.reader, self.retire, self.lock = reader, retire, lock
        self.execute = execute or self.execute_native
        self.sleep, self.now = sleep, now
        self.pid = os.getpid() if pid is None else int(pid)
        self.owns_parent = False
        self.parent_lock = None
        self.registration = None

    def status(self, state, **extra):
        record = dict(pid=self.pid, time=self.now(), status=state, **extra)
        write(self.out / "controller_status.json", record)
        if self.owns_parent:
            write(self.parent / "supervisor_status.json", record)

    def qualification_inputs(self, publication, dependencies):
        receipt_path = self.parent / "qualification.json"
        receipt = read(receipt_path)
        report_path = resolve(self.root, receipt["qualification_path"])
        report = read(report_path)
        for record in (receipt, report):
            if (record.get("status") != "REAL_WEIGHT_QUALIFICATION_PASS"
                    or record.get("training_updates") != 0 or record.get("development_read") is not False):
                raise RuntimeError("Original calibration-only model qualification is missing")
        if sha(report_path) != receipt.get("qualification_sha256") or report.get("calibration_sources") != 4:
            raise RuntimeError("Original model qualification scope or report differs")
        qualified = report.get("qualification_source_bindings")
        if not isinstance(qualified, dict) or not qualified or qualified != receipt.get("source_bindings"):
            raise RuntimeError("Original qualification source records differ")
        final = dict(dependencies)
        for path, digest in publication["source_bindings"].items():
            if path in final and final[path] != digest:
                raise RuntimeError("Original source/dependency records disagree")
            final[path] = digest
            if qualified.get(path) != digest:
                raise RuntimeError("Original execution source was not qualified: " + path)
        for path, digest in qualified.items():
            if final.get(path) != digest:
                raise RuntimeError("Qualified dependency differs from original execution input")
        verify_bindings(self.root, qualified)
        engineering_path = self.parent / "engineering_probe.json"
        engineering = read(engineering_path)
        if (engineering.get("status") != "REAL_RUNNER_ENGINEERING_PASS"
                or engineering.get("frames") != 925 or engineering.get("online_cached_tokens_equal") is not True
                or engineering.get("scientific_result") is not False or engineering.get("development_read") is not False
                or engineering.get("source_bindings") != qualified):
            raise RuntimeError("Original final runner qualification differs")
        return {str(path): sha(path) for path in (receipt_path, report_path, engineering_path)}

    def archive(self, files, old_parent):
        directory = self.out / "retired_supervisor_evidence" / (str(old_parent["pid"]) + "_" + old_parent["start_ticks"])
        directory.mkdir(parents=True, exist_ok=True)
        bindings = {}
        for name, source in files.items():
            target = directory / name
            data = Path(source).read_bytes()
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError("Retired evidence archive differs: " + str(target))
            else:
                target.write_bytes(data)
            bindings[str(target)] = sha(target)
        return bindings

    def initialize(self):
        handoff_path = self.out / "handoff.json"
        self.handoff = read(handoff_path)
        self.old_parent = checked_identity(self.handoff["oldsupervisor"])
        self.old_worker = checked_identity(self.handoff["oldworker"])
        if len({self.old_parent["pid"], self.old_worker["pid"], self.pid}) != 3:
            raise RuntimeError("Parent, worker and replacement must have distinct identities")
        self.bound = sources(self.here)
        if self.handoff.get("new_source_bindings") != self.bound or str(self.here / "controller.py") not in self.bound:
            raise RuntimeError("Handoff must bind the complete runtime extension, including controller")
        publication_path = self.parent / "source_publication.json"
        dependency_path = self.parent / "dependency_bindings.json"
        registration_path = self.parent / "supervisor_registration.json"
        stages_path = self.original / "stages.json"
        publication, dependencies, stages = read(publication_path), read(dependency_path), read(stages_path)
        published(publication)
        verify_bindings(self.root, publication["source_bindings"])
        verify_bindings(self.root, dependencies)
        if publication["source_bindings"].get(str(stages_path)) != sha(stages_path):
            raise RuntimeError("Original pushed sources must bind the original stage list")
        registration = read(registration_path)
        if registration.get("bindings") != publication["source_bindings"] or registration.get("dependencies") != dependencies or registration.get("stages") != stages:
            raise RuntimeError("Original supervisor registration differs from pushed execution inputs")
        self.stages = stages
        names = [stage["name"] for stage in stages]
        expected_names = ["m1_screen", "m1_calibration", "m1_development", "m1_rate_curve", "m1_timing", "m1_analysis", "m1_publication",
                          "m2_calibration", "m2_evaluation", "m2_actual", "m2_timing", "m2_analysis", "m2_publication"]
        if names != expected_names:
            raise RuntimeError("Unexpected original authorized stage sequence")
        self.expected_index = names.index(self.handoff["expected_stage"])
        if self.handoff["expected_stage"] != "m1_calibration":
            raise RuntimeError("This handoff is authorized at the M1 calibration boundary")
        stage = stages[self.expected_index]
        self.expected_receipt = resolve(self.root, stage["receipt"])
        if resolve(self.root, self.handoff["expected_receipt"]) != self.expected_receipt:
            raise RuntimeError("Expected handoff receipt differs from the original stage")
        qualification = self.qualification_inputs(publication, dependencies)
        original_inputs = {str(path): sha(path) for path in (publication_path, dependency_path, registration_path, stages_path)}
        original_inputs.update(qualification)
        for field, path in (("original_source_publication_sha256", publication_path),
                            ("original_dependency_bindings_sha256", dependency_path),
                            ("original_registration_sha256", registration_path)):
            if field in self.handoff and self.handoff[field] != sha(path):
                raise RuntimeError("Pinned original input differs: " + field)
        path = self.out / "controller_registration.json"
        if path.exists():
            previous = read(path)
            if (previous.get("source_bindings") != self.bound or previous.get("handoff_sha256") != sha(handoff_path)
                    or previous.get("original_inputs") != original_inputs
                    or previous.get("original_source_bindings") != publication["source_bindings"]
                    or previous.get("dependency_bindings") != dependencies):
                raise RuntimeError("Replacement controller registration changed on resume")
            self.registration = previous
        else:
            launch_path, status_path = self.parent / "supervisor_launch.json", self.parent / "supervisor_status.json"
            launch, status = read(launch_path), read(status_path)
            if sha(launch_path) != self.handoff["oldlaunch_sha256"] or checked_identity(launch) != self.old_parent:
                raise RuntimeError("Old supervisor launch identity differs from handoff")
            if (int(status.get("pid", -1)) != self.old_parent["pid"] or status.get("stage") != stage["name"]
                    or int(status.get("worker_pid", -1)) != self.old_worker["pid"] or status.get("status") != "RUNNING"):
                raise RuntimeError("Old supervisor status is not the nominated running M1 worker")
            actual = self.reader(self.old_parent["pid"])
            if exited(self.old_parent, actual) or actual["state"] not in ("T", "t"):
                raise RuntimeError("Operator must pause the identity-checked old parent before controller launch")
            archive_files = {"supervisor_launch.json": launch_path, "supervisor_status.json": status_path,
                             "supervisor_registration.json": registration_path, "source_publication.json": publication_path,
                             "dependency_bindings.json": dependency_path, "stages.json": stages_path,
                             "handoff.json": handoff_path}
            for index, file in enumerate(qualification):
                archive_files["qualification_" + str(index) + ".json"] = Path(file)
            archive_bindings = self.archive(archive_files, self.old_parent)
            self.registration = dict(status="REGISTERED", source_bindings=self.bound,
                handoff_sha256=sha(handoff_path), original_inputs=original_inputs,
                original_source_bindings=publication["source_bindings"], dependency_bindings=dependencies,
                retired_evidence_bindings=archive_bindings, original_source_commit=publication["commit"],
                oldsupervisor=self.old_parent, oldworker=self.old_worker, stages=stages,
                training_updates=0, policy_selection_updates=0, created=self.now())
            seal(path, self.registration)
        self.verify()

    def verify(self):
        if sources(self.here) != self.bound:
            raise RuntimeError("Runtime extension source inventory changed after launch")
        if sha(self.out / "handoff.json") != self.registration["handoff_sha256"]:
            raise RuntimeError("Handoff changed after launch")
        if read(self.out / "controller_registration.json") != self.registration:
            raise RuntimeError("Controller registration changed")
        for field in ("source_bindings", "original_inputs", "original_source_bindings", "dependency_bindings", "retired_evidence_bindings"):
            verify_bindings(self.root, self.registration[field])

    def receipt(self, stage):
        path = resolve(self.root, stage["receipt"])
        return path.exists() and read(path).get("status") in stage["statuses"]

    def wait_worker(self):
        while True:
            self.verify()
            parent = self.reader(self.old_parent["pid"])
            if exited(self.old_parent, parent) or parent["state"] not in ("T", "t"):
                raise RuntimeError("Paused original parent changed while its worker was running")
            worker = self.reader(self.old_worker["pid"])
            if exited(self.old_worker, worker):
                if not self.receipt(self.stages[self.expected_index]):
                    raise RuntimeError("Old worker exited without the expected complete M1 checkpoint")
                for stage in self.stages[:self.expected_index]:
                    if not self.receipt(stage):
                        raise RuntimeError("An earlier original M1 stage is incomplete")
                return
            if worker["state"] in ("T", "t"):
                raise RuntimeError("M1 worker is unexpectedly stopped; it must be allowed to finish")
            self.status("WAITING_FOR_M1_WORKER", stage=self.handoff["expected_stage"], worker_pid=self.old_worker["pid"])
            self.sleep(30)

    def takeover(self):
        retired_path = self.out / "retirement_complete.json"
        intent_path = self.out / "retirement_intent.json"
        intent = dict(oldsupervisor=self.old_parent, oldworker=self.old_worker,
                      handoff_sha256=self.registration["handoff_sha256"], controller_registration_sha256=sha(self.out / "controller_registration.json"))
        if retired_path.exists():
            retired = read(retired_path)
            if retired.get("status") != "OLD_SUPERVISOR_RETIRED" or retired.get("intent") != intent:
                raise RuntimeError("Prior parent retirement receipt differs")
            if not exited(self.old_parent, self.reader(self.old_parent["pid"])):
                raise RuntimeError("Retired original parent is still alive")
        else:
            if not intent_path.exists():
                self.wait_worker()
                seal(intent_path, intent)
                allow_running = False
            else:
                if read(intent_path) != intent:
                    raise RuntimeError("Prior retirement intent differs")
                if not exited(self.old_worker, self.reader(self.old_worker["pid"])) or not self.receipt(self.stages[self.expected_index]):
                    raise RuntimeError("Retirement resume lacks the sealed M1 checkpoint and worker exit")
                allow_running = True  # Resume only an identity-bound, already recorded retirement.
            actual = self.reader(self.old_parent["pid"])
            if not exited(self.old_parent, actual):
                self.retire(self.old_parent, self.reader, allow_running=allow_running)
            deadline = self.now() + 30
            while not exited(self.old_parent, self.reader(self.old_parent["pid"])):
                self.verify()
                if self.now() >= deadline:
                    raise RuntimeError("Old supervisor did not exit after identity-bound retirement")
                self.sleep(1)
            seal(retired_path, dict(status="OLD_SUPERVISOR_RETIRED", intent=intent))
        self.verify()
        self.parent_lock = self.lock(self.parent / "supervisor.lock")
        try:
            self_identity = self.reader(self.pid)
            if self_identity is None or self_identity["state"] == "Z":
                raise RuntimeError("Cannot bind replacement controller identity")
            self.start_ticks = str(self_identity["start_ticks"])
            previous_launch = read(self.parent / "supervisor_launch.json")
            previous_identity = checked_identity(previous_launch)
            if previous_identity != self.old_parent and not exited(previous_identity, self.reader(previous_identity["pid"])):
                raise RuntimeError("A previous replacement owner is still alive")
            if previous_identity != self.old_parent:
                if (previous_launch.get("controller_registration_sha256") != sha(self.out / "controller_registration.json")
                        or previous_launch.get("source_bindings") != self.bound):
                    raise RuntimeError("Previous replacement launch lineage differs")
                previous_status = read(self.parent / "supervisor_status.json")
                if int(previous_status.get("pid", -1)) != previous_identity["pid"]:
                    raise RuntimeError("Previous replacement status owner differs")
                self.require_previous_worker_exit(previous_status)
                self.archive({"supervisor_launch.json": self.parent / "supervisor_launch.json",
                              "supervisor_status.json": self.parent / "supervisor_status.json"}, previous_identity)
            self.owns_parent = True
            launch = dict(pid=self.pid, start_ticks=self.start_ticks, command=[sys.executable, "-u", str(self.here / "controller.py")],
                time=self.now(), source_bindings=self.bound, original_source_commit=self.registration["original_source_commit"],
                retired_original_supervisor=self.old_parent, retired_original_worker=self.old_worker,
                controller_registration_sha256=sha(self.out / "controller_registration.json"), retirement_receipt_sha256=sha(retired_path),
                lineage="M1_WORKER_COMPLETED_THEN_ORIGINAL_PARENT_RETIRED", automatic_further_experiments=False)
            write(self.parent / "supervisor_launch.json", launch)
            self.status("RUNNING", stage="M1_TAKEOVER_COMPLETE", worker_pid=None)
        except BaseException:
            self.parent_lock.close()
            self.parent_lock = None
            self.owns_parent = False
            raise

    def require_previous_worker_exit(self, previous_status):
        pid = previous_status.get("worker_pid")
        if pid is None:
            return
        actual = self.reader(int(pid))
        if actual is None or actual["state"] == "Z":
            return
        ticks = previous_status.get("worker_start_ticks")
        if ticks is not None and str(actual["start_ticks"]) != str(ticks):
            return  # The registered worker exited and its PID has been reused.
        # A legacy status without ticks cannot prove that a live PID is safe.
        raise RuntimeError("A previous controller worker is still active; manual review is required")

    def execute_native(self, argv, log, started):
        with Path(log).open("a") as stream:
            worker = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT)
            started(worker.pid)
            return worker.wait()

    def run_job(self, name, argv, validator):
        if validator():
            return
        attempt = 0
        while True:
            self.verify()
            attempt += 1
            log = self.out / (name + "_" + str(time.time_ns()) + ".log")
            self.status("RUNNING", stage=name, worker_pid=None, attempt=attempt, log=str(log))
            def started(pid):
                actual = self.reader(pid)
                ticks = None if actual is None else str(actual["start_ticks"])
                self.status("RUNNING", stage=name, worker_pid=pid, worker_start_ticks=ticks, attempt=attempt, log=str(log))
            code = self.execute(argv, log, started)
            self.verify()
            if code == 75:
                self.status("WAITING_FOR_SAFE_RESOURCE", stage=name, worker_pid=None, attempt=attempt, log=str(log))
                self.sleep(30)
                continue
            if code != 0:
                raise RuntimeError(f"{name} failed; exit={code}; log={log}")
            if not validator():
                raise RuntimeError(f"{name} exited without a valid completed receipt; log={log}")
            return

    def verify_m1_publication(self):
        record = read(self.parent / "m1_complete.json")
        if record.get("status") != "M1_COMPLETE" or record.get("synthetic") is not False or record.get("training_updates") != 0:
            raise RuntimeError("Original M1 scientific completion is required before the speed extension")
        publication = record.get("publication", {})
        published(publication)
        if read(self.parent / "m1_publication.json") != publication:
            raise RuntimeError("M1 publication receipt and completion record differ")
        if publication.get("source_bindings") != self.registration["original_source_bindings"]:
            raise RuntimeError("M1 publication must preserve all original pushed execution sources")
        verify_bindings(self.root, publication["source_bindings"])
        return record

    def qualification_complete(self):
        path = self.out / "m2_speed_qualification.json"
        if not path.exists():
            return False
        receipt = read(path)
        if (receipt.get("source_bindings") != self.bound or receipt.get("development_read") is not False
                or receipt.get("training_updates") != 0):
            raise RuntimeError("Speed qualification scope or source bindings differ")
        if receipt.get("status") == "RUNNING":
            return False  # An interrupted attempt is incomplete, never qualified.
        state = (receipt.get("status"), receipt.get("selected_implementation"))
        if state not in (("QUALIFIED", "accelerated"), ("USE_ORIGINAL", "original")):
            raise RuntimeError("Speed qualification failed or has an inconsistent implementation decision")
        verify_bindings(self.root, receipt["base_source_bindings"])
        if state[0] == "QUALIFIED":
            speedup = float(receipt.get("speedup", 0))
            if receipt.get("strict_equality_passed") is not True or not math.isfinite(speedup) or speedup < 1.10:
                raise RuntimeError("Accelerated receiver lacks strict equality and sufficient measured speedup")
        return True

    def source_published(self):
        path = self.out / "source_publication.json"
        if not path.exists():
            return False
        record = read(path)
        if record.get("status") == "COMMITTED":
            return False
        published(record)
        if record.get("source_bindings") != self.bound or record.get("qualification_sha256") != sha(self.out / "m2_speed_qualification.json"):
            raise RuntimeError("Runtime extension publication differs from the qualified frozen source")
        verify_bindings(self.root, record["published_files"])
        return True

    def prepare_m2(self):
        self.verify_m1_publication()
        self.run_job("m2_speed_qualification", [sys.executable, "-u", str(self.here / "benchmark.py")], self.qualification_complete)
        self.run_job("speed_source_publication", [sys.executable, "-u", str(self.here / "publish_source.py")], self.source_published)
        self.qualification_complete()
        self.source_published()

    def stage_argv(self, stage):
        script = stage["script"]
        if Path(script).name != script or script not in ("m1_runner.py", "m2_runner.py", "analysis.py", "publish.py"):
            raise RuntimeError("Unexpected original stage command")
        if script == "m2_runner.py":
            args = stage.get("args", [])
            if len(args) != 2 or args[0] != "--stage" or args[1] not in ("calibration", "evaluation", "actual", "timing"):
                raise RuntimeError("Unexpected original M2 stage arguments")
            return [sys.executable, "-u", str(self.here / "wrapper.py"), *args]
        return [sys.executable, "-u", str(self.original / script), *stage.get("args", [])]

    def finish(self):
        path = self.parent / "completion.json"
        record = read(path)
        if (record.get("status") != "AUTHORIZED_TWO_METHODS_COMPLETE" or record.get("synthetic") is not False
                or record.get("training_updates") != 0 or record.get("stop") is not True):
            raise RuntimeError("Original authorized final completion is invalid")
        published(record.get("publication", {}))
        if read(self.parent / "m2_publication.json") != record["publication"]:
            raise RuntimeError("Final completion and original publication receipt differ")
        verify_bindings(self.root, record["publication"]["source_bindings"])
        self.verify()
        write(self.out / "controller_completion.json", dict(status="COMPLETE", pid=self.pid,
            start_ticks=self.start_ticks, original_completion_path=str(path), original_completion_sha256=sha(path),
            selected_implementation=read(self.out / "m2_speed_qualification.json")["selected_implementation"],
            source_bindings=self.bound, training_updates=0, policy_selection_updates=0,
            stopped_after_authorized_scope=True, automatic_further_experiments=False,
            retired_original_supervisor=self.old_parent, time=self.now()))
        self.status("COMPLETE", worker_pid=None, stopped_after_authorized_scope=True)

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        controller_lock = self.lock(self.out / "controller.lock")
        try:
            self.initialize()
            self.takeover()
            prepared = False
            for stage in self.stages:
                self.verify()
                if stage["name"].startswith("m2_"):
                    if not prepared:
                        self.prepare_m2()
                        prepared = True
                    else:
                        self.verify_m1_publication()
                        self.qualification_complete()
                        self.source_published()
                self.run_job(stage["name"], self.stage_argv(stage), lambda stage=stage: self.receipt(stage))
            self.finish()
        except BaseException as error:
            # Preserve a started child's identity if failure interrupts wait or
            # status handling. A later launch must not overlook an orphan.
            previous = read(self.parent / "supervisor_status.json") if self.owns_parent else {}
            self.status("FAILED", worker_pid=previous.get("worker_pid"),
                        worker_start_ticks=previous.get("worker_start_ticks"),
                        stage=previous.get("stage"), error=str(error))
            raise
        finally:
            if self.parent_lock is not None:
                self.parent_lock.close()
            controller_lock.close()


def main():
    Controller().run()


if __name__ == "__main__":
    main()
