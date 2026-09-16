import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/observe_gpu_trial.py'
SPEC = importlib.util.spec_from_file_location('passive_gpu_observer', SCRIPT)
OBSERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBSERVER)


class GPUObserverTests(unittest.TestCase):
    def make_process(self, root, pid, parent, started='100', state='S'):
        process = root / str(pid)
        process.mkdir(exist_ok=True)
        fields = [state, str(parent), *['0'] * 17, started]
        (process / 'stat').write_text(f'{pid} (python (worker)) ' + ' '.join(fields))
        (process / 'cmdline').write_bytes(b'/usr/bin/python\0/expected/script.py\0')

    def test_owned_process_descendants_foreign_and_reused_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_process(root, 1, 0)
            self.make_process(root, 123, 1)
            self.make_process(root, 456, 123)
            self.make_process(root, 789, 1)
            owner = OBSERVER.owned_handle(123, Path('/expected/script.py'), root)
            self.assertTrue(OBSERVER.alive(owner, root))
            self.assertEqual(OBSERVER.classify(456, [owner], root), 'owned_trial')
            self.assertEqual(OBSERVER.classify(789, [owner], root), 'foreign')
            self.make_process(root, 123, 1, started='200')
            self.assertFalse(OBSERVER.alive(owner, root))
            self.assertEqual(OBSERVER.classify(456, [owner], root), 'foreign')

    def test_missing_zombie_and_wrong_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_process(root, 123, 1)
            owner = OBSERVER.owned_handle(123, Path('/expected/script.py'), root)
            self.assertTrue(OBSERVER.classify(999, [owner], root).startswith('unknown'))
            with self.assertRaises(RuntimeError):
                OBSERVER.owned_handle(123, Path('/different/script.py'), root)
            self.make_process(root, 123, 1, state='Z')
            self.assertFalse(OBSERVER.alive(owner, root))
            with self.assertRaisesRegex(RuntimeError, 'terminal'):
                OBSERVER.owned_handle(123, Path('/expected/script.py'), root)

    def test_incomplete_gpu_output_not_treated_as_idle(self):
        self.assertEqual(OBSERVER.parse_table('', ('pid', 'memory')), [])
        self.assertEqual(OBSERVER.parse_table('123, 456\n', ('pid', 'memory')), [{'pid': 123, 'memory': '456'}])
        with self.assertRaises(RuntimeError):
            OBSERVER.parse_table('123\n', ('pid', 'memory'))
        with self.assertRaises(ValueError):
            OBSERVER.parse_table('unknown, 100\n', ('pid', 'memory'))

    def test_gpu_queries_are_read_only_and_have_timeout(self):
        with mock.patch.object(OBSERVER.subprocess, 'check_output', return_value='123, python, 100\n') as runner:
            OBSERVER.query('compute-apps', ('pid', 'process_name', 'used_gpu_memory'))
        command = runner.call_args.args[0]
        self.assertEqual(command[:3], ['nvidia-smi', '-i', '0'])
        self.assertTrue(command[3].startswith('--query-compute-apps='))
        self.assertEqual(runner.call_args.kwargs['timeout'], 15)

    def test_dry_run_never_inspects_processes_or_creates_output(self):
        with mock.patch.object(sys, 'argv', ['observe_gpu_trial.py', '--owner', '123=/expected/script.py', '--output', '/unused']), \
                mock.patch.object(OBSERVER, 'owned_handle') as ownership, mock.patch.object(OBSERVER, 'query') as query, \
                mock.patch('builtins.print'):
            OBSERVER.main()
        ownership.assert_not_called()
        query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
