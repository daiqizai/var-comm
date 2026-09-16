"""CPU-only safety tests: no nvidia-smi call, model import, or GPU launch."""
from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wait_and_run.py"
SPEC = importlib.util.spec_from_file_location("wait_and_run_under_test", MODULE_PATH)
queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)

UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
IDLE = queue.GPUSnapshot(UUID, 0, 400, ())
BUSY = queue.GPUSnapshot(UUID, 0, 400, (777,))
READY = {"ready": True, "pipeline_terminal": True}
WAITING = {"ready": False, "pipeline_terminal": False}
ORIGINAL = queue.ProcIdentity(1496953, 111111, "boot-id", os.getuid(), 1496953, 1496953, "S")
CHILD = queue.ProcIdentity(4242, 222222, "boot-id", os.getuid(), 4242, 4242, "S")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = queue.Config(
            task_name="ei-liulu-backbone-eval-local-20260912-v1",
            output_dir=self.root / "ei-liulu-backbone-eval-local-20260912-v1",
            allowed_gpu_uuid=UUID, command=("/usr/bin/python3", "eval.py"), cwd=self.root,
            blocker_status_path=self.root / "old-status.json", max_wait_seconds=900)
        self.write_blocker_status("RUNNING")

    def tearDown(self):
        self.temp.cleanup()

    def write_blocker_status(self, state, pid=1496953):
        self.config.blocker_status_path.write_text(json.dumps({
            "status": state, "supervisor_pid": pid, "stage": "evaluate"}))

    def supervisor(self, gpu_samples=None, blocker_samples=None):
        supervisor = queue.Supervisor.__new__(queue.Supervisor)
        supervisor.config = self.config
        supervisor.files = Mock()
        supervisor.boot_id = "boot-id"
        supervisor.probe = Mock()
        supervisor.probe.sample = Mock(side_effect=gpu_samples) if gpu_samples is not None else Mock(return_value=IDLE)
        supervisor.blocker = Mock()
        supervisor.blocker.sample = (Mock(side_effect=blocker_samples) if blocker_samples is not None
                                     else Mock(return_value=READY))
        supervisor.signal_received = None
        supervisor.child = None
        supervisor.previous_handlers = {}
        clock = FakeClock()
        supervisor.sleep = clock.sleep
        return supervisor, clock

    def test_only_standard_library_imports(self):
        names = []
        for node in ast.walk(ast.parse(MODULE_PATH.read_text())):
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                names.append(node.module.split(".")[0])
        self.assertTrue(all(name in sys.stdlib_module_names or name == "__future__" for name in names))
        self.assertFalse({"torch", "jax", "tensorflow", "pynvml"}.intersection(names))

    def test_proc_stat_with_nested_parentheses_and_spaces(self):
        values = ["S", "1", "4242", "4242"] + ["0"] * 15 + ["123456"] + ["0"] * 4
        identity = queue.parse_proc_stat("4242 (strange )( process name) " + " ".join(values), os.getuid(), "boot-id")
        self.assertEqual(identity.start_ticks, 123456)
        self.assertEqual(identity.pgrp, 4242)
        self.assertEqual(identity.session, 4242)

    def test_bad_proc_stat_fails_closed(self):
        with self.assertRaises(queue.SafetyError):
            queue.parse_proc_stat("4242 ???", 1000, "boot-id")

    def test_gpu_probe_strict_uuid_and_readonly_queries(self):
        outputs = [subprocess.CompletedProcess([], 0, f"{UUID}, 1, 410\n", ""),
                   subprocess.CompletedProcess([], 0, "", "")]
        with patch.object(queue.subprocess, "run", side_effect=outputs) as run:
            sample = queue.GPUProbe(UUID).sample()
        self.assertEqual(sample, queue.GPUSnapshot(UUID, 1, 410, ()))
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0:3], ["nvidia-smi", "-i", UUID])
            self.assertTrue(call.args[0][3].startswith("--query-"))
            self.assertEqual(call.kwargs["timeout"], 10)

    def test_nvidia_error_is_not_idle(self):
        for result in [subprocess.CompletedProcess([], 1, "", "NVML error"),
                       subprocess.CompletedProcess([], 0, f"{UUID}, N/A, 200\n", ""),
                       subprocess.CompletedProcess([], 0, "GPU-wrong, 0, 0\n", ""),
                       subprocess.CompletedProcess([], 0, f"{UUID}, 0, 1\n", "warning")]:
            with self.subTest(result=result), patch.object(queue.subprocess, "run", return_value=result):
                with self.assertRaises(queue.SafetyError):
                    queue.GPUProbe(UUID).sample()
        with patch.object(queue.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 10)):
            with self.assertRaises(queue.SafetyError):
                queue.GPUProbe(UUID).sample()

    def test_unknown_gpu_process_format_fails_closed(self):
        outputs = [subprocess.CompletedProcess([], 0, f"{UUID}, 0, 100\n", ""),
                   subprocess.CompletedProcess([], 0, "No running processes found\n", "")]
        with patch.object(queue.subprocess, "run", side_effect=outputs):
            with self.assertRaises(queue.SafetyError):
                queue.GPUProbe(UUID).sample()

    def test_busy_means_any_compute_pid_or_threshold(self):
        self.assertFalse(BUSY.idle(5, 512))
        self.assertFalse(replace(IDLE, utilization_percent=6).idle(5, 512))
        self.assertFalse(replace(IDLE, memory_used_mib=513).idle(5, 512))
        self.assertTrue(replace(IDLE, utilization_percent=5, memory_used_mib=512).idle(5, 512))

    def test_pipeline_still_alive_even_when_terminal_status_blocks(self):
        self.write_blocker_status("ALL_STAGES_COMPLETE")
        with patch.object(queue, "read_proc", return_value=ORIGINAL):
            blocker = queue.Blocker(self.config, "boot-id")
            sample = blocker.sample()
        self.assertFalse(sample["ready"])
        self.assertTrue(sample["original_supervisor_alive"])

    def test_pipeline_supervisor_exit_plus_terminal_status_ready(self):
        with patch.object(queue, "read_proc", return_value=ORIGINAL):
            blocker = queue.Blocker(self.config, "boot-id")
        self.write_blocker_status("ALL_STAGES_COMPLETE")
        with patch.object(queue, "read_proc", return_value=None):
            sample = blocker.sample()
        self.assertTrue(sample["ready"])
        self.assertEqual(sample["original_identity"]["start_ticks"], 111111)

    def test_pid_reuse_is_not_original_supervisor(self):
        with patch.object(queue, "read_proc", return_value=ORIGINAL):
            blocker = queue.Blocker(self.config, "boot-id")
        self.write_blocker_status("ALL_STAGES_COMPLETE")
        with patch.object(queue, "read_proc", return_value=replace(ORIGINAL, start_ticks=999999)):
            sample = blocker.sample()
        self.assertTrue(sample["pid_reused"])
        self.assertFalse(sample["original_supervisor_alive"])
        self.assertTrue(sample["ready"])

    def test_absent_original_with_stale_running_status_refuses_start(self):
        with patch.object(queue, "read_proc", return_value=None):
            with self.assertRaises(queue.SafetyError):
                queue.Blocker(self.config, "boot-id")

    def test_changed_pipeline_supervisor_or_unknown_state_fails_closed(self):
        with patch.object(queue, "read_proc", return_value=ORIGINAL):
            blocker = queue.Blocker(self.config, "boot-id")
            self.write_blocker_status("RUNNING", pid=98765)
            with self.assertRaises(queue.SafetyError):
                blocker.sample()
            self.write_blocker_status("UNKNOWN")
            with self.assertRaises(queue.SafetyError):
                blocker.sample()

    def test_no_launch_in_train_eval_gaps(self):
        supervisor, clock = self.supervisor(blocker_samples=[WAITING] * 3 + [READY] * 4)
        with patch.object(queue.time, "monotonic", clock.monotonic), patch.object(queue, "OwnChild") as launch:
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 300)
        self.assertEqual(supervisor.probe.sample.call_count, 7)
        launch.assert_not_called()  # Waiting never creates a subprocess itself.

    def test_three_idle_polls_at_least_sixty_seconds_apart(self):
        supervisor, clock = self.supervisor()
        with patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.sleeps, [60, 60])
        self.assertEqual(supervisor.probe.sample.call_count, 4)  # Last call = immediate prelaunch recheck.

    def test_missing_assets_prevent_idle_counting_until_receipt_arrives(self):
        supervisor, clock = self.supervisor()
        supervisor.config = replace(self.config, ready_paths=(self.root / "ready.json",))
        missing = str(self.root / "ready.json")
        with patch.object(queue.Config, "missing_ready_paths", side_effect=[[missing], [missing], [], [], [], []]), \
                patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 240)
        states = [call.args[0] for call in supervisor.files.record.call_args_list]
        self.assertEqual(states[:2], ["WAITING_ASSETS", "WAITING_ASSETS"])
        streaks = [call.kwargs["idle_streak"] for call in supervisor.files.record.call_args_list]
        self.assertEqual(streaks, [0, 0, 1, 2, 3])

    def test_ready_receipt_disappearance_blocks_final_launch(self):
        supervisor, clock = self.supervisor()
        with patch.object(queue.Config, "missing_ready_paths", side_effect=[[], [], [], ["vanished"], [], [], [], []]), \
                patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 300)

    def test_readiness_requires_file_not_directory(self):
        receipt = self.root / "receipt.json"
        config = replace(self.config, ready_paths=(receipt, self.root))
        self.assertEqual(config.missing_ready_paths(), [str(receipt), str(self.root)])
        receipt.write_text('{}')
        self.assertEqual(config.missing_ready_paths(), [str(self.root)])

    def test_busy_gpu_resets_idle_confirmation(self):
        supervisor, clock = self.supervisor(gpu_samples=[IDLE, BUSY, IDLE, IDLE, IDLE, IDLE])
        with patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 240)
        streaks = [call.kwargs["idle_streak"] for call in supervisor.files.record.call_args_list]
        self.assertEqual(streaks, [1, 0, 1, 2, 3])

    def test_final_race_recheck_prevents_launch(self):
        supervisor, clock = self.supervisor(gpu_samples=[IDLE, IDLE, IDLE, BUSY, IDLE, IDLE, IDLE, IDLE])
        with patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertTrue(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 300)
        self.assertEqual(supervisor.probe.sample.call_count, 8)

    def test_wait_is_bounded(self):
        supervisor, clock = self.supervisor()
        supervisor.config = replace(self.config, max_wait_seconds=180)
        supervisor.probe.sample.return_value = BUSY
        with patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertFalse(supervisor.wait_for_idle())
        self.assertEqual(clock.now, 180)
        self.assertEqual(supervisor.files.record.call_args.args[0], "WAIT_TIMEOUT")

    def test_signal_during_wait_never_launches(self):
        supervisor, clock = self.supervisor()
        supervisor.signal_handler(signal.SIGTERM, None)
        with patch.object(queue.time, "monotonic", clock.monotonic), patch.object(queue, "OwnChild") as child:
            self.assertFalse(supervisor.wait_for_idle())
        child.assert_not_called()
        supervisor.probe.sample.assert_not_called()
        self.assertEqual(supervisor.files.record.call_args.args[0], "CANCELLED_SIGNAL")

    def test_monitor_error_never_launches(self):
        supervisor, clock = self.supervisor(gpu_samples=[queue.SafetyError("NVML unavailable")])
        with patch.object(queue.time, "monotonic", clock.monotonic), patch.object(queue, "OwnChild") as child:
            self.assertEqual(supervisor.run(), 1)
        child.assert_not_called()
        self.assertEqual(supervisor.files.record.call_args.args[0], "ABORTED_MONITOR_ERROR")

    def test_child_environment_thread_limits_and_uuid(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "64", "CUDA_VISIBLE_DEVICES": "0,1"}):
            env = queue.child_environment(UUID, self.config.task_name)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual(env["OMP_NUM_THREADS"], "2")
        self.assertEqual(env["MKL_NUM_THREADS"], "2")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "2")
        self.assertEqual(env["VAR_COMM_TASK_NAME"], self.config.task_name)

    def test_own_child_launch_no_shell_new_session_low_priority(self):
        process = Mock(pid=4242)
        with patch.object(queue.subprocess, "Popen", return_value=process) as popen, \
                patch.object(queue, "read_proc", return_value=CHILD):
            child = queue.OwnChild(self.config, "boot-id", Mock())
        self.assertEqual(child.identity, CHILD)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertIs(popen.call_args.kwargs["preexec_fn"], queue.lower_own_priority)
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(popen.call_count, 1)

    def test_child_status_uses_wnowait_not_popen_poll(self):
        child = queue.OwnChild.__new__(queue.OwnChild)
        child.process = Mock(pid=4242)
        with patch.object(queue.os, "waitid", return_value=None) as waitid:
            self.assertFalse(child.exited())
        self.assertTrue(waitid.call_args.args[2] & os.WNOWAIT)
        child.process.poll.assert_not_called()

    def own_child_stub(self):
        child = queue.OwnChild.__new__(queue.OwnChild)
        child.process = Mock(pid=4242)
        child.identity = CHILD
        child.boot_id = "boot-id"
        child.reaped = False
        return child

    def test_never_signal_reused_child_pid(self):
        child = self.own_child_stub()
        with patch.object(queue, "read_proc", return_value=replace(CHILD, start_ticks=333333)), \
                patch.object(queue.os, "killpg") as killpg:
            with self.assertRaises(queue.SafetyError):
                child.signal_group(signal.SIGTERM)
        killpg.assert_not_called()

    def test_never_signal_reaped_child_or_wrong_group(self):
        child = self.own_child_stub()
        for reaped, identity in [(True, CHILD), (False, replace(CHILD, pgrp=9876))]:
            child.reaped = reaped
            with patch.object(queue, "read_proc", return_value=identity), patch.object(queue.os, "killpg") as killpg:
                with self.assertRaises(queue.SafetyError):
                    child.signal_group(signal.SIGTERM)
                killpg.assert_not_called()

    def test_signal_only_verified_own_child_group(self):
        child = self.own_child_stub()
        with patch.object(queue, "read_proc", return_value=CHILD), patch.object(queue.os, "killpg") as killpg:
            child.signal_group(signal.SIGTERM)
        killpg.assert_called_once_with(4242, signal.SIGTERM)

    def test_foreign_pid_does_not_count_as_own_child(self):
        child = self.own_child_stub()
        foreign = replace(CHILD, pid=8888, pgrp=8888, session=8888)
        with patch.object(queue, "read_proc", side_effect=[CHILD, foreign]):
            self.assertFalse(child.owns_gpu_pid(8888))
        own = replace(CHILD, pid=9999, start_ticks=333333)
        with patch.object(queue, "read_proc", side_effect=[CHILD, own]):
            self.assertTrue(child.owns_gpu_pid(9999))

    def test_foreign_gpu_work_stops_own_evaluation_no_restart(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor(gpu_samples=[queue.GPUSnapshot(UUID, 60, 12000, (4242, 8888))])
        child = Mock(identity=CHILD, reaped=False)
        child.owns_gpu_pid.side_effect = lambda pid: pid == 4242
        child.stop.return_value = -15
        with patch.object(queue, "OwnChild", return_value=child) as launch, \
                patch.object(queue.time, "monotonic", clock.monotonic), patch.object(queue.os, "killpg") as killpg:
            self.assertEqual(supervisor.run_child(), 1)
        launch.assert_called_once()
        child.stop.assert_called_once_with(15)
        killpg.assert_not_called()  # Only OwnChild.stop is entrusted with verified signals.
        self.assertEqual(supervisor.files.record.call_args.args[0], "ABORTED_FOREIGN_GPU_PROCESS")
        self.assertEqual(supervisor.files.record.call_args.kwargs["foreign_compute_pids"], [8888])

    def test_running_nvidia_failure_stops_only_own_child(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor(gpu_samples=[queue.SafetyError("NVML failure")])
        child = Mock(identity=CHILD, reaped=False)
        with patch.object(queue, "OwnChild", return_value=child), patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertEqual(supervisor.run_child(), 1)
        child.stop.assert_called_once_with(15)
        self.assertEqual(supervisor.files.record.call_args.args[0], "ABORTED_MONITOR_ERROR")

    def test_running_timeout_is_bounded_and_stops_own_child(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor()
        supervisor.config = replace(self.config, max_run_seconds=1)
        child = Mock(identity=CHILD, reaped=False)
        child.exited.return_value = False
        child.stop.return_value = -15
        with patch.object(queue, "OwnChild", return_value=child), patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertEqual(supervisor.run_child(), 1)
        self.assertEqual(clock.now, 1)
        child.stop.assert_called_once_with(15)
        self.assertEqual(supervisor.files.record.call_args.args[0], "RUN_TIMEOUT")

    def test_signal_during_run_terminates_only_own_child(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor()
        child = Mock(identity=CHILD, reaped=False)
        child.exited.return_value = False
        child.stop.return_value = -15
        def sleeping(seconds):
            clock.sleep(seconds)
            supervisor.signal_handler(signal.SIGTERM, None)
        supervisor.sleep = sleeping
        with patch.object(queue, "OwnChild", return_value=child), patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertEqual(supervisor.run_child(), 1)
        child.stop.assert_called_once_with(15)
        self.assertEqual(supervisor.files.record.call_args.args[0], "CANCELLED_SIGNAL")
        self.assertEqual(supervisor.files.record.call_args.kwargs["signal"], signal.SIGTERM)

    def test_cleanup_waits_for_own_group_and_does_not_reap_before_signalling(self):
        child = self.own_child_stub()
        child.process.wait.return_value = -15
        child.signal_group = Mock()
        child._members = Mock(return_value=[replace(CHILD, state="Z")])
        with patch.object(queue.time, "monotonic", return_value=0):
            self.assertEqual(child.stop(15), -15)
        child.signal_group.assert_called_once_with(signal.SIGTERM)
        child.process.wait.assert_called_once_with(timeout=5)
        self.assertTrue(child.reaped)

    def test_successful_evaluation_finishes_once(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor()
        child = Mock(identity=CHILD, reaped=False)
        child.exited.return_value = True
        child.finish.return_value = (0, False)
        with patch.object(queue, "OwnChild", return_value=child) as launch, patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertEqual(supervisor.run_child(), 0)
        launch.assert_called_once()
        child.stop.assert_not_called()
        self.assertEqual(supervisor.files.record.call_args.args[0], "COMPLETED")

    def test_no_background_descendants_silently_marked_complete(self):
        self.config.output_dir.mkdir()
        supervisor, clock = self.supervisor()
        child = Mock(identity=CHILD, reaped=False)
        child.exited.return_value = True
        child.finish.return_value = (0, True)
        with patch.object(queue, "OwnChild", return_value=child), patch.object(queue.time, "monotonic", clock.monotonic):
            self.assertEqual(supervisor.run_child(), 1)
        self.assertEqual(supervisor.files.record.call_args.args[0], "ABORTED_BACKGROUND_DESCENDANTS")

    def test_one_run_lock_and_persistent_marker(self):
        with queue.RunFiles(self.config) as files:
            with self.assertRaises(BlockingIOError):
                with queue.RunFiles(self.config):
                    self.fail("Second supervisor acquired lock")
            with patch("builtins.print"):
                files.record("QUEUED", evaluation_launched=False)
        with self.assertRaises(FileExistsError):
            with queue.RunFiles(self.config):
                self.fail("Finished run directory was reused")
        self.assertEqual(json.loads((self.config.output_dir / "queue-status.json").read_text())["state"], "QUEUED")

    def test_configuration_rejects_unsafe_scope_and_unbounded_wait(self):
        invalid = [replace(self.config, task_name="bad_Name"),
                   replace(self.config, task_name="ei-otheruser-backbone-eval"),
                   replace(self.config, max_wait_seconds=86401),
                   replace(self.config, poll_seconds=1), replace(self.config, idle_polls=1),
                   replace(self.config, memory_limit_mib=2048),
                   replace(self.config, utilization_limit_percent=20),
                   replace(self.config, allowed_gpu_uuid="0"),
                   replace(self.config, output_dir=Path("/tmp/not-owned"))]
        with patch.object(queue, "OUTPUT_ROOT", self.root):
            self.config.validate()
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    config.validate()

    def test_config_file_command_is_array_and_optional_override(self):
        path = self.root / "queue-config.json"
        path.write_text(json.dumps(self.config.serializable()))
        with patch.object(queue, "OUTPUT_ROOT", self.root):
            loaded = queue.Config.load(path, ["/usr/bin/python3", "other.py"])
            self.assertEqual(loaded.command, ("/usr/bin/python3", "other.py"))
            raw = self.config.serializable()
            raw["command"] = "python3 eval.py && danger"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                queue.Config.load(path)

    def test_validate_cli_never_queries_gpu_or_creates_run(self):
        config_path = self.root / "queue.json"
        config_path.write_text(json.dumps(self.config.serializable()))
        argv = [str(MODULE_PATH), "--config", str(config_path), "--validate", "--task-name", self.config.task_name]
        with patch.object(queue, "OUTPUT_ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(queue, "Supervisor") as supervisor, patch.object(queue, "RunFiles") as files, \
                patch.object(queue, "lower_own_priority") as nice, patch("builtins.print"):
            self.assertEqual(queue.main(), 0)
        supervisor.assert_not_called()
        files.assert_not_called()
        nice.assert_not_called()
        self.assertFalse(self.config.output_dir.exists())

    def test_config_accepts_readiness_files_not_yet_existing(self):
        config_path = self.root / "queue.json"
        raw = self.config.serializable()
        raw["ready_paths"] = [str(self.root / "not-ready.json")]
        config_path.write_text(json.dumps(raw))
        with patch.object(queue, "OUTPUT_ROOT", self.root):
            config = queue.Config.load(config_path)
        self.assertEqual(config.ready_paths, (self.root / "not-ready.json",))
        self.assertEqual(config.missing_ready_paths(), raw["ready_paths"])


if __name__ == "__main__":
    unittest.main()
