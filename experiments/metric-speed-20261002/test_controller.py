"""CPU engineering fixtures only: no real process signal, GPU, or Git operation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("speed_controller_tested", Path(__file__).with_name("controller.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def stages():
    specifications = [
        ("m1_screen", "m1_runner.py", "--stage", "screen", "COMPLETE"),
        ("m1_calibration", "m1_runner.py", "--stage", "calibration", "COMPLETE"),
        ("m1_development", "m1_runner.py", "--stage", "development", "COMPLETE"),
        ("m1_rate_curve", "m1_runner.py", "--stage", "rate_curves", "COMPLETE"),
        ("m1_timing", "m1_runner.py", "--stage", "timing", "COMPLETE"),
        ("m1_analysis", "analysis.py", "--method", "m1", "COMPLETE"),
        ("m1_publication", "publish.py", "--scope", "m1", "M1_COMPLETE"),
        ("m2_calibration", "m2_runner.py", "--stage", "calibration", "M2_CALIBRATION_COMPLETE"),
        ("m2_evaluation", "m2_runner.py", "--stage", "evaluation", "M2_EVALUATION_COMPLETE"),
        ("m2_actual", "m2_runner.py", "--stage", "actual", "M2_ACTUAL_COMPLETE"),
        ("m2_timing", "m2_runner.py", "--stage", "timing", "M2_TIMING_COMPLETE"),
        ("m2_analysis", "analysis.py", "--method", "m2", "COMPLETE"),
        ("m2_publication", "publish.py", "--scope", "m2", "AUTHORIZED_TWO_METHODS_COMPLETE"),
    ]
    receipt_names = {"m1_publication": "m1_complete.json", "m2_publication": "completion.json",
                     "m1_analysis": "m1_analysis_completion.json", "m2_analysis": "m2_analysis_completion.json"}
    return [dict(name=name, script=script, args=[flag, value], statuses=[status],
                 receipt="outputs/" + c.PARENT_NAME + "/" + receipt_names.get(name, name + "_complete.json"))
            for name, script, flag, value, status in specifications]


class FakeLock:
    def __init__(self, path):
        self.path, self.closed = path, False

    def close(self):
        self.closed = True


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.here = self.root / "experiments/metric-speed-20261002"
        self.original = self.root / c.ORIGINAL_DIRECTORY
        self.parent = self.root / "outputs" / c.PARENT_NAME
        self.out = self.root / "outputs" / c.EXTENSION_NAME
        for directory in (self.here, self.original, self.parent, self.out):
            directory.mkdir(parents=True)
        for name in ("controller.py", "test_controller.py", "benchmark.py", "publish_source.py", "wrapper.py", "protocol.md"):
            (self.here / name).write_text("# ENGINEERING_SYNTHETIC_FIXTURE " + name + "\n")
        for name in ("supervisor.py", "m1_runner.py", "m2_runner.py", "analysis.py", "publish.py", "common.py"):
            (self.original / name).write_text("# ORIGINAL_SYNTHETIC_FIXTURE " + name + "\n")
        self.stages = stages()
        c.write(self.original / "stages.json", self.stages)
        self.original_bindings = {str(path): c.sha(path) for path in self.original.iterdir()}
        dependency = self.root / "frozen_dependency.dat"
        dependency.write_bytes(b"frozen model/third-party dependency fixture")
        self.dependencies = {str(dependency): c.sha(dependency)}
        self.publication = dict(status="PUSHED", checks="PASS", commit="a" * 40,
                                remote_commit="a" * 40, source_bindings=self.original_bindings)
        c.write(self.parent / "source_publication.json", self.publication)
        c.write(self.parent / "dependency_bindings.json", self.dependencies)
        qualified = dict(self.original_bindings, **self.dependencies)
        report = self.parent / "real_weight_report.json"
        c.write(report, dict(status="REAL_WEIGHT_QUALIFICATION_PASS", training_updates=0, development_read=False,
                             calibration_sources=4, qualification_source_bindings=qualified))
        c.write(self.parent / "qualification.json", dict(status="REAL_WEIGHT_QUALIFICATION_PASS", training_updates=0,
            development_read=False, qualification_path=str(report), qualification_sha256=c.sha(report), source_bindings=qualified))
        c.write(self.parent / "engineering_probe.json", dict(status="REAL_RUNNER_ENGINEERING_PASS", frames=925,
            online_cached_tokens_equal=True, scientific_result=False, development_read=False, source_bindings=qualified))
        c.write(self.parent / "supervisor_registration.json", dict(bindings=self.original_bindings,
            dependencies=self.dependencies, stages=self.stages))
        c.write(self.parent / "supervisor_launch.json", dict(pid=100, start_ticks="10", command=["python", "supervisor.py"]))
        c.write(self.parent / "supervisor_status.json", dict(pid=100, status="RUNNING", stage="m1_calibration", worker_pid=200))
        self.old_launch = (self.parent / "supervisor_launch.json").read_bytes()
        c.write(self.out / "handoff.json", dict(oldsupervisor=dict(pid=100, start_ticks="10"), oldworker=dict(pid=200, start_ticks="20"),
            expected_stage="m1_calibration", expected_receipt=self.stages[1]["receipt"],
            oldlaunch_sha256=c.sha(self.parent / "supervisor_launch.json"), new_source_bindings=c.sources(self.here)))
        for stage in self.stages[:2]:
            c.write(self.root / stage["receipt"], dict(status="COMPLETE", training_updates=0, development_read=False))
        self.processes = {100: dict(pid=100, start_ticks="10", state="T"),
                          200: dict(pid=200, start_ticks="20", state="Z"),
                          300: dict(pid=300, start_ticks="30", state="R")}
        self.clock = 1000
        self.events, self.locks, self.calls = [], [], []
        self.codes = {}
        self.qualification_status = "USE_ORIGINAL"
        self.remove_worker_on_sleep = False
        self.controller = self.make_controller()

    def make_controller(self, pid=300):
        return c.Controller(self.root, self.here, reader=lambda pid: self.processes.get(pid), retire=self.retire,
                            lock=self.lock, execute=self.execute, sleep=self.sleep, now=lambda: self.clock, pid=pid)

    def lock(self, path):
        result = FakeLock(path)
        self.locks.append(result)
        self.events.append("lock:" + Path(path).name)
        return result

    def retire(self, expected, reader, *, allow_running=False):
        if expected != dict(pid=100, start_ticks="10"):
            raise AssertionError("Only the nominated old parent may be retired")
        if not allow_running and reader(100)["state"] != "T":
            raise AssertionError("Old parent must be paused")
        self.events.append("retire:100")
        self.processes.pop(100)

    def sleep(self, seconds):
        self.events.append("sleep:" + str(seconds))
        self.clock += seconds
        if self.remove_worker_on_sleep:
            self.processes.pop(200, None)
            self.remove_worker_on_sleep = False

    def execute(self, argv, log, started):
        script = Path(argv[2]).name
        value = argv[-1]
        if script in ("benchmark.py", "publish_source.py"):
            name = "m2_speed_qualification" if script == "benchmark.py" else "speed_source_publication"
        elif script == "wrapper.py":
            name = "m2_" + value
        else:
            name = next(stage["name"] for stage in self.stages if stage["script"] == script and stage["args"][-1] == value)
        self.calls.append((name, list(argv)))
        self.events.append("execute:" + name)
        started(400 + len(self.calls))
        codes = self.codes.get(name, [])
        code = codes.pop(0) if codes else 0
        if code:
            return code
        bound = c.sources(self.here)
        if name == "m2_speed_qualification":
            state = self.qualification_status
            c.write(self.out / "m2_speed_qualification.json", dict(status=state,
                selected_implementation="accelerated" if state == "QUALIFIED" else "original", source_bindings=bound,
                base_source_bindings=dict(self.original_bindings, **self.dependencies), development_read=False,
                training_updates=0, strict_equality_passed=state == "QUALIFIED", speedup=1.3 if state == "QUALIFIED" else .9))
        elif name == "speed_source_publication":
            q = self.out / "m2_speed_qualification.json"
            c.write(self.out / "source_publication.json", dict(status="PUSHED", checks="PASS", commit="b" * 40,
                remote_commit="b" * 40, source_bindings=bound, qualification_sha256=c.sha(q),
                published_files={str(path): c.sha(path) for path in (q, self.out / "controller_registration.json")}))
        elif name in ("m1_publication", "m2_publication"):
            method = name[:2]
            publication = dict(self.publication, commit=("c" if method == "m1" else "d") * 40,
                               remote_commit=("c" if method == "m1" else "d") * 40)
            c.write(self.parent / (method + "_publication.json"), publication)
            stage = next(stage for stage in self.stages if stage["name"] == name)
            c.write(self.root / stage["receipt"], dict(status=stage["statuses"][0], synthetic=False, training_updates=0,
                publication=publication, **({"stop": True} if method == "m2" else {})))
        else:
            stage = next(stage for stage in self.stages if stage["name"] == name)
            c.write(self.root / stage["receipt"], dict(status=stage["statuses"][0]))
        return 0


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(self.temporary.name)

    def test_complete_takeover_preserves_original_sources_and_published_receipt(self):
        f = self.fixture
        f.controller.run()
        names = [name for name, _ in f.calls]
        self.assertEqual(names[:5], ["m1_development", "m1_rate_curve", "m1_timing", "m1_analysis", "m1_publication"])
        self.assertEqual(names[5:7], ["m2_speed_qualification", "speed_source_publication"])
        self.assertEqual(names[7:], ["m2_calibration", "m2_evaluation", "m2_actual", "m2_timing", "m2_analysis", "m2_publication"])
        self.assertLess(f.events.index("retire:100"), f.events.index("lock:supervisor.lock"))
        archive = f.out / "retired_supervisor_evidence/100_10/supervisor_launch.json"
        self.assertEqual(archive.read_bytes(), f.old_launch)
        c.verify_bindings(f.root, f.original_bindings)
        status, launch = c.read(f.parent / "supervisor_status.json"), c.read(f.parent / "supervisor_launch.json")
        self.assertEqual((status["status"], status["pid"], launch["pid"], launch["start_ticks"]), ("COMPLETE", 300, 300, "30"))
        self.assertTrue(status["stopped_after_authorized_scope"])
        self.assertEqual(c.read(f.parent / "completion.json")["publication"], c.read(f.parent / "m2_publication.json"))
        self.assertEqual(c.read(f.out / "controller_completion.json")["selected_implementation"], "original")
        self.assertTrue(all(lock.closed for lock in f.locks))

    def test_worker_is_allowed_to_finish_without_any_worker_signal(self):
        f = self.fixture
        f.processes[200]["state"] = "R"
        f.remove_worker_on_sleep = True
        f.controller.run()
        self.assertLess(f.events.index("sleep:30"), f.events.index("retire:100"))
        self.assertEqual([event for event in f.events if event.startswith("retire:")], ["retire:100"])

    def test_worker_zombie_counts_as_exited_but_receipt_is_required(self):
        f = self.fixture
        (f.root / f.stages[1]["receipt"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "without the expected complete"):
            f.controller.run()
        self.assertNotIn("retire:100", f.events)
        self.assertFalse(f.calls)
        self.assertEqual(c.read(f.parent / "supervisor_launch.json")["pid"], 100)

    def test_wrong_checkpoint_status_refuses_retirement(self):
        f = self.fixture
        c.write(f.root / f.stages[1]["receipt"], dict(status="RUNNING"))
        with self.assertRaisesRegex(RuntimeError, "without the expected complete"):
            f.controller.run()
        self.assertFalse(f.calls)

    def test_worker_pid_reuse_never_signals_replacement(self):
        f = self.fixture
        f.processes[200] = dict(pid=200, start_ticks="999", state="R")
        f.controller.run()
        self.assertEqual(f.processes[200]["start_ticks"], "999")
        self.assertNotIn("retire:200", f.events)

    def test_parent_pid_reuse_and_unpaused_parent_fail_before_signal(self):
        f = self.fixture
        for state, ticks in (("T", "999"), ("S", "10")):
            f.processes[100] = dict(pid=100, start_ticks=ticks, state=state)
            with self.subTest(state=state, ticks=ticks), self.assertRaisesRegex(RuntimeError, "pause"):
                f.controller.run()
        self.assertNotIn("retire:100", f.events)

    def test_source_and_dependency_mutation_rejected_before_signal(self):
        f = self.fixture
        dependency = next(iter(f.dependencies))
        Path(dependency).write_bytes(b"mutated")
        with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
            f.controller.run()
        self.assertNotIn("retire:100", f.events)

    def test_own_source_inventory_must_match_handoff(self):
        f = self.fixture
        (f.here / "unregistered.py").write_text("# unexpected\n")
        with self.assertRaisesRegex(RuntimeError, "complete runtime extension"):
            f.controller.run()

    def test_new_source_mutation_during_wait_refuses_takeover(self):
        f = self.fixture
        f.processes[200]["state"] = "R"
        def corrupt_sleep(seconds):
            (f.here / "wrapper.py").write_text("# hot edit\n")
            f.processes.pop(200)
        f.controller.sleep = corrupt_sleep
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            f.controller.run()
        self.assertNotIn("retire:100", f.events)

    def test_only_exit75_is_automatically_retried(self):
        f = self.fixture
        f.codes["m1_development"] = [75, 75, 0]
        f.controller.run()
        self.assertEqual(sum(name == "m1_development" for name, _ in f.calls), 3)
        self.assertEqual(f.events.count("sleep:30"), 2)

    def test_unknown_failure_stops_before_later_stages(self):
        f = self.fixture
        f.codes["m1_development"] = [1]
        with self.assertRaisesRegex(RuntimeError, "exit=1"):
            f.controller.run()
        self.assertEqual([name for name, _ in f.calls], ["m1_development"])
        self.assertEqual(c.read(f.parent / "supervisor_status.json")["status"], "FAILED")
        self.assertFalse((f.parent / "completion.json").exists())

    def test_failed_benchmark_cannot_publish_or_start_m2(self):
        f = self.fixture
        f.qualification_status = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "qualification failed"):
            f.controller.run()
        self.assertEqual(f.calls[-1][0], "m2_speed_qualification")
        self.assertFalse((f.out / "source_publication.json").exists())

    def test_running_benchmark_is_incomplete_and_can_rerun_with_same_sources(self):
        f = self.fixture
        c.write(f.out / "m2_speed_qualification.json", dict(status="RUNNING", selected_implementation=None,
            source_bindings=c.sources(f.here), development_read=False, training_updates=0))
        f.controller.initialize()
        self.assertFalse(f.controller.qualification_complete())
        f.controller.run()
        self.assertEqual(sum(name == "m2_speed_qualification" for name, _ in f.calls), 1)
        self.assertEqual(c.read(f.out / "m2_speed_qualification.json")["status"], "USE_ORIGINAL")

    def test_running_benchmark_wrong_sources_or_scope_cannot_resume(self):
        f = self.fixture
        f.controller.initialize()
        record = dict(status="RUNNING", source_bindings=c.sources(f.here), development_read=False, training_updates=0)
        for delta in ({"source_bindings": {}}, {"development_read": True}, {"training_updates": 1}):
            c.write(f.out / "m2_speed_qualification.json", dict(record, **delta))
            with self.subTest(delta=delta), self.assertRaisesRegex(RuntimeError, "scope or source"):
                f.controller.qualification_complete()

    def test_acceleration_requires_equality_and_ten_percent_speedup(self):
        f = self.fixture
        f.controller.initialize()
        qpath = f.out / "m2_speed_qualification.json"
        record = dict(status="QUALIFIED", selected_implementation="accelerated", source_bindings=c.sources(f.here),
            base_source_bindings=f.original_bindings, development_read=False, training_updates=0, strict_equality_passed=True, speedup=1.1)
        c.write(qpath, record)
        self.assertTrue(f.controller.qualification_complete())
        for delta in ({"speedup": 1.099}, {"strict_equality_passed": False}, {"speedup": None},
                      {"selected_implementation": "original"}, {"development_read": True}):
            c.write(qpath, dict(record, **delta))
            with self.subTest(delta=delta), self.assertRaises((RuntimeError, TypeError)):
                f.controller.qualification_complete()

    def test_qualified_m2_uses_wrapper_and_original_analysis_publication(self):
        f = self.fixture
        f.qualification_status = "QUALIFIED"
        f.controller.run()
        for name, argv in f.calls:
            if name in ("m2_calibration", "m2_evaluation", "m2_actual", "m2_timing"):
                self.assertEqual(Path(argv[2]), f.here / "wrapper.py")
                self.assertEqual(argv[-2], "--stage")
            elif name in ("m2_analysis", "m2_publication"):
                self.assertEqual(Path(argv[2]).parent, f.original)
        self.assertEqual(c.read(f.out / "controller_completion.json")["selected_implementation"], "accelerated")

    def test_resume_after_failure_preserves_initial_registration_and_lineage(self):
        f = self.fixture
        f.codes["m1_development"] = [1]
        with self.assertRaises(RuntimeError):
            f.controller.run()
        registration_bytes = (f.out / "controller_registration.json").read_bytes()
        f.processes.pop(300)
        f.processes[301] = dict(pid=301, start_ticks="31", state="R")
        f.controller = f.make_controller(pid=301)
        f.controller.run()
        self.assertEqual((f.out / "controller_registration.json").read_bytes(), registration_bytes)
        self.assertEqual(f.events.count("retire:100"), 1)
        self.assertEqual(c.read(f.parent / "supervisor_launch.json")["pid"], 301)
        self.assertTrue((f.out / "retired_supervisor_evidence/300_30/supervisor_launch.json").exists())

    def test_resume_refuses_a_live_previous_controller(self):
        f = self.fixture
        f.codes["m1_development"] = [1]
        with self.assertRaises(RuntimeError):
            f.controller.run()
        f.processes[301] = dict(pid=301, start_ticks="31", state="R")
        replacement = f.make_controller(pid=301)
        with self.assertRaisesRegex(RuntimeError, "previous replacement owner is still alive"):
            replacement.run()
        self.assertEqual(c.read(f.parent / "supervisor_launch.json")["pid"], 300)

    def test_resume_refuses_orphan_worker_before_new_launch_or_job(self):
        f = self.fixture
        f.codes["m1_development"] = [1]
        with self.assertRaises(RuntimeError):
            f.controller.run()
        f.processes.pop(300)
        f.processes[301] = dict(pid=301, start_ticks="31", state="R")
        f.processes[444] = dict(pid=444, start_ticks="44", state="S")
        c.write(f.parent / "supervisor_status.json", dict(pid=300, status="RUNNING", stage="m2_speed_qualification",
            worker_pid=444, worker_start_ticks="44"))
        calls = len(f.calls)
        with self.assertRaisesRegex(RuntimeError, "previous controller worker is still active"):
            f.make_controller(pid=301).run()
        self.assertEqual(len(f.calls), calls)
        self.assertEqual(c.read(f.parent / "supervisor_launch.json")["pid"], 300)

    def test_orphan_worker_guard_handles_zombie_pid_reuse_and_missing_ticks(self):
        f = self.fixture
        record = dict(worker_pid=444, worker_start_ticks="44")
        for actual in (None, dict(pid=444, start_ticks="44", state="Z"), dict(pid=444, start_ticks="999", state="R")):
            if actual is None:
                f.processes.pop(444, None)
            else:
                f.processes[444] = actual
            f.controller.require_previous_worker_exit(record)
        f.processes[444] = dict(pid=444, start_ticks="44", state="S")
        with self.assertRaisesRegex(RuntimeError, "still active"):
            f.controller.require_previous_worker_exit(dict(worker_pid=444))

    def test_exception_after_worker_start_preserves_orphan_identity(self):
        f = self.fixture
        f.processes[444] = dict(pid=444, start_ticks="44", state="S")
        def interrupted_execute(argv, log, started):
            started(444)
            raise RuntimeError("fixture controller wait interrupted")
        f.controller.execute = interrupted_execute
        with self.assertRaisesRegex(RuntimeError, "wait interrupted"):
            f.controller.run()
        status = c.read(f.parent / "supervisor_status.json")
        self.assertEqual((status["worker_pid"], status["worker_start_ticks"]), (444, "44"))
        f.processes.pop(300)
        f.processes[301] = dict(pid=301, start_ticks="31", state="R")
        with self.assertRaisesRegex(RuntimeError, "worker is still active"):
            f.make_controller(pid=301).run()

    def test_original_member_verification_allows_additional_tracked_sources(self):
        # Git universe changes after speed publication do not rewrite original bindings.
        f = self.fixture
        f.controller.initialize()
        (f.root / "other_new_tracked.py").write_text("# additional source outside old member set\n")
        f.controller.verify()

    def test_nested_sources_are_rejected_instead_of_silently_unbound(self):
        f = self.fixture
        (f.here / "nested").mkdir()
        (f.here / "nested/code.py").write_text("# nested\n")
        with self.assertRaisesRegex(RuntimeError, "one directory"):
            c.sources(f.here)

    def test_parse_proc_stat_with_spaces_and_parentheses(self):
        proc = Path(self.temporary.name) / "proc/123"
        proc.mkdir(parents=True)
        fields = ["S"] + ["0"] * 18 + ["4567"]
        (proc / "stat").write_text("123 (name with ) parens) " + " ".join(fields))
        (proc / "cmdline").write_bytes(b"python\0supervisor.py\0")
        identity = c.process_identity(123, proc.parent)
        self.assertEqual((identity["state"], identity["start_ticks"], identity["command"]), ("S", "4567", "python supervisor.py"))
        self.assertIsNone(c.process_identity(456, proc.parent))

    def test_pidfd_signals_only_bound_parent_and_closes_descriptor(self):
        expected = dict(pid=100, start_ticks="10")
        with mock.patch.object(c.os, "pidfd_open", return_value=9, create=True) as opened, \
             mock.patch.object(c.signal, "pidfd_send_signal", create=True) as sent, \
             mock.patch.object(c.signal, "SIGCONT", 18, create=True), \
             mock.patch.object(c.os, "close") as closed:
            c.retire_with_pidfd(expected, lambda pid: dict(pid=pid, state="T", start_ticks="10"))
            opened.assert_called_once_with(100, 0)
            self.assertEqual(sent.call_args_list, [mock.call(9, signal.SIGTERM, None, 0), mock.call(9, signal.SIGCONT, None, 0)])
            closed.assert_called_once_with(9)

    def test_pidfd_identity_recheck_prevents_signalling_reused_pid(self):
        with mock.patch.object(c.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(c.signal, "pidfd_send_signal", create=True) as sent, \
             mock.patch.object(c.os, "close"):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                c.retire_with_pidfd(dict(pid=100, start_ticks="10"), lambda pid: dict(pid=pid, state="T", start_ticks="999"))
            sent.assert_not_called()

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
                         "Native Linux pidfd/flock fixture requires Linux")
    def test_native_paused_parent_worker_checkpoint_and_real_lock_takeover(self):
        # These are newly created fixture processes. No live experiment process
        # is inspected or signalled by this test, and the fixture imports no ML.
        f = self.fixture
        receipt = f.root / f.stages[1]["receipt"]
        receipt.unlink()
        ready, permit = f.out / "native_fixture_ready.json", f.out / "native_worker_permit"
        worker_code = (
            "import json,pathlib,sys,time\n"
            "receipt,permit=map(pathlib.Path,sys.argv[1:])\n"
            "deadline=time.monotonic()+15\n"
            "while not permit.exists():\n"
            " if time.monotonic()>deadline: raise RuntimeError('fixture worker permit timeout')\n"
            " time.sleep(.01)\n"
            "receipt.write_text(json.dumps(dict(status='COMPLETE',training_updates=0,development_read=False)))\n"
        )
        parent_code = (
            "import fcntl,json,os,pathlib,subprocess,sys,time\n"
            "lock=pathlib.Path(sys.argv[1]).open('a')\n"
            "fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "worker=subprocess.Popen([sys.executable,'-u','-c',sys.argv[4],sys.argv[2],sys.argv[3]])\n"
            "def identity(pid):\n"
            " s=pathlib.Path('/proc',str(pid),'stat').read_text(); f=s[s.rfind(') ')+2:].split()\n"
            " return dict(pid=pid,start_ticks=f[19])\n"
            "pathlib.Path(sys.argv[5]).write_text(json.dumps(dict(parent=identity(os.getpid()),worker=identity(worker.pid))))\n"
            "while True: time.sleep(.1)\n"
        )
        parent = subprocess.Popen([sys.executable, "-u", "-c", parent_code,
            str(f.parent / "supervisor.lock"), str(receipt), str(permit), worker_code, str(ready)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        identities = None
        def cleanup():
            if identities:
                for expected in (identities["parent"], identities["worker"]):
                    actual = c.process_identity(expected["pid"])
                    if not c.exited(expected, actual):
                        c.retire_with_pidfd(expected, allow_running=True)
            else:
                # Parent is owned by this Popen; record its /proc identity before
                # cleanup rather than sending to a bare, potentially reused PID.
                actual = c.process_identity(parent.pid)
                if actual and actual["state"] != "Z":
                    c.retire_with_pidfd(actual, allow_running=True)
            parent.wait(timeout=5)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if parent.poll() is not None or time.monotonic() > deadline:
                self.fail("Native parent fixture did not become ready")
            time.sleep(.01)
        identities = c.read(ready)
        descriptor = os.pidfd_open(identities["parent"]["pid"], 0)
        try:
            actual = c.process_identity(identities["parent"]["pid"])
            self.assertEqual(actual["start_ticks"], identities["parent"]["start_ticks"])
            signal.pidfd_send_signal(descriptor, signal.SIGSTOP, None, 0)
        finally:
            os.close(descriptor)
        deadline = time.monotonic() + 5
        while c.process_identity(parent.pid)["state"] not in ("T", "t"):
            if time.monotonic() > deadline:
                self.fail("Fixture parent did not stop")
            time.sleep(.01)
        c.write(f.parent / "supervisor_launch.json", dict(identities["parent"], command=["NATIVE_ENGINEERING_FIXTURE"]))
        c.write(f.parent / "supervisor_status.json", dict(pid=parent.pid, status="RUNNING", stage="m1_calibration",
            worker_pid=identities["worker"]["pid"]))
        handoff = c.read(f.out / "handoff.json")
        handoff.update(oldsupervisor=identities["parent"], oldworker=identities["worker"],
                       oldlaunch_sha256=c.sha(f.parent / "supervisor_launch.json"))
        c.write(f.out / "handoff.json", handoff)
        retired = []
        def retire(expected, reader, **options):
            retired.append(expected["pid"])
            c.retire_with_pidfd(expected, reader, **options)
        controller = c.Controller(f.root, f.here, retire=retire, execute=f.execute, sleep=lambda seconds: time.sleep(.01))
        controller.initialize()
        permit.write_text("finish the existing worker\n")
        controller.run()
        parent.wait(timeout=5)
        self.assertEqual(retired, [identities["parent"]["pid"]])
        self.assertEqual(parent.returncode, -signal.SIGTERM)
        self.assertEqual(c.read(f.parent / "supervisor_status.json")["pid"], os.getpid())
        with c.acquire_lock(f.parent / "supervisor.lock"):
            pass  # Both the old parent and completed replacement released it.


if __name__ == "__main__":
    unittest.main()
