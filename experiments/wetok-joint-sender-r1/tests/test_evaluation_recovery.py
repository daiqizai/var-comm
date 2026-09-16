import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/resume_joint_evaluation.py'
SPEC = importlib.util.spec_from_file_location('joint_evaluation_recovery', SCRIPT)
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class EvaluationRecoveryTests(unittest.TestCase):
    def test_only_explicit_gpu_contention_status_is_eligible(self):
        message = 'other GPU compute workloads are present: [123, 456]; not stopping them'
        self.assertTrue(RECOVERY.busy_error(message))
        self.assertTrue(RECOVERY.known_contention_status({'status': 'JOINT_EVALUATION_FAILED_OR_INTERRUPTED',
            'error': repr(RuntimeError(message))}))
        self.assertFalse(RECOVERY.busy_error('GPU process ownership cannot be verified'))
        self.assertFalse(RECOVERY.busy_error('other GPU compute workloads are present: []; not stopping them'))
        self.assertFalse(RECOVERY.known_contention_status({'status': 'EVALUATING', 'error': repr(RuntimeError(message))}))
        self.assertFalse(RECOVERY.known_contention_status({'status': 'JOINT_EVALUATION_FAILED_OR_INTERRUPTED',
            'error': repr(RuntimeError('frozen reference changed'))}))

    def test_actual_child_guard_is_required_for_retry(self):
        message = 'other GPU compute workloads are present: [123]; not stopping them'
        log = 'Traceback\nRuntimeError: ' + message + '\n'
        state = {'pid': 456, 'status': 'JOINT_EVALUATION_FAILED_OR_INTERRUPTED', 'error': repr(RuntimeError(message))}
        self.assertTrue(RECOVERY.retryable_exit(1, log, state, 456))
        self.assertTrue(RECOVERY.retryable_exit(1, log, state, 789))
        self.assertFalse(RECOVERY.retryable_exit(-9, log, state, 456))
        self.assertFalse(RECOVERY.retryable_exit(1, 'ValueError: source changed\n', state, 456))
        self.assertFalse(RECOVERY.retryable_exit(1, log, {'pid': 456, 'status': 'EVALUATING'}, 456))
        self.assertFalse(RECOVERY.retryable_exit(1, '', state, 456))

    def test_committed_sources_cannot_be_removed_or_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'images/048'
            source.mkdir(parents=True)
            receipt = source / 'receipt.json'
            receipt.write_text('{"image_id":"same"}')
            committed = RECOVERY.committed_receipts(root)
            self.assertEqual(len(committed), 1)
            RECOVERY.check_commits(root, committed)
            receipt.write_text('{"image_id":"different"}')
            with self.assertRaisesRegex(RuntimeError, 'committed source'):
                RECOVERY.check_commits(root, committed)
            receipt.unlink()
            with self.assertRaises(FileNotFoundError):
                RECOVERY.check_commits(root, committed)

    def test_dry_run_never_launches_or_inspects_gpu(self):
        with mock.patch.object(sys, 'argv', ['resume_joint_evaluation.py', '--output', '/unused']), \
                mock.patch.object(RECOVERY.subprocess, 'Popen') as launcher, \
                mock.patch.object(RECOVERY, 'require_uncontended_gpu') as gpu, mock.patch('builtins.print'):
            RECOVERY.main()
        launcher.assert_not_called()
        gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
