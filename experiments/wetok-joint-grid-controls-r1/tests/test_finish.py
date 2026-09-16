import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/finish.py'
SPEC = importlib.util.spec_from_file_location('grid_finish_fixture', SCRIPT)
FINISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINISHER)


class FinishTests(unittest.TestCase):
    def test_only_matching_current_resource_guard_exits_can_retry(self):
        timing = {'pid': 123, 'status': 'GRID_TIMING_FAILED_OR_INTERRUPTED'}
        busy = 'Traceback\nRuntimeError: other GPU compute workloads are present: [456]; not stopping them\n'
        self.assertTrue(FINISHER.retryable_guard('timing', 1, busy, timing, 123))
        self.assertFalse(FINISHER.retryable_guard('timing', 1, busy, timing, 999))
        self.assertFalse(FINISHER.retryable_guard('timing', -9, busy, timing, 123))
        self.assertFalse(FINISHER.retryable_guard('timing', 1, 'RuntimeError: source hash changed', timing, 123))
        quality = {'pid': 123, 'status': 'GRID_QUALITY_FAILED_OR_INTERRUPTED'}
        self.assertTrue(FINISHER.retryable_guard('quality', 1,
            'RuntimeError: insufficient free GPU memory between quality source images', quality, 123))
        self.assertFalse(FINISHER.retryable_guard('quality', 1, 'torch.OutOfMemoryError: allocator cap exceeded', quality, 123))
        self.assertFalse(FINISHER.retryable_guard('analysis', 1, busy, timing, 123))

    def test_plan_only_never_inspects_owned_processes_or_starts_a_child(self):
        with mock.patch.object(sys, 'argv', ['finish.py', '--training-pid', '123', '--review-watcher-pid', '456']), \
                mock.patch.object(FINISHER, 'owned_handle') as ownership, mock.patch.object(FINISHER.subprocess, 'Popen') as process, \
                mock.patch('builtins.print'):
            FINISHER.main()
        ownership.assert_not_called()
        process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
