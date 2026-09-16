import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/finish_registered_trial.py'
SPEC = importlib.util.spec_from_file_location('joint_completion_watcher', SCRIPT)
WATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCHER)


class CompletionWatcherTests(unittest.TestCase):
    def make_process(self, root, pid, script, arguments, started='12345', state='S'):
        process = root / str(pid)
        process.mkdir(exist_ok=True)
        command = [sys.executable, str(script), *arguments, '--execute']
        (process / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in command) + b'\0')
        fields = [state, *['0'] * 18, started]
        (process / 'stat').write_text(f'{pid} (python worker) ' + ' '.join(fields))

    def test_process_identity_and_wrong_milestone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = WATCHER.EXPERIMENT / 'scripts/train.py'
            self.make_process(root, 123, script, ['--until-update', '5000'])
            handle = WATCHER.owned_handle(123, script, {'--until-update': 5000}, root)
            self.assertTrue(WATCHER.alive(handle, root))
            with self.assertRaisesRegex(RuntimeError, 'different milestone'):
                WATCHER.owned_handle(123, script, {'--until-update': 2500}, root)
            self.make_process(root, 123, script, ['--until-update', '5000'], started='99999')
            self.assertFalse(WATCHER.alive(handle, root))
            self.make_process(root, 123, script, ['--until-update', '5000'], state='Z')
            self.assertFalse(WATCHER.alive(handle, root))
            with self.assertRaisesRegex(RuntimeError, 'already exited'):
                WATCHER.owned_handle(123, script, {'--until-update': 5000}, root)
            self.assertFalse(WATCHER.alive((999, '12345'), root))

    def test_wrong_script_or_trainer_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = WATCHER.EXPERIMENT / 'scripts/watch.py'
            self.make_process(root, 456, script, ['--step', '5000', '--training-pid', '123'])
            with self.assertRaisesRegex(RuntimeError, 'different milestone or trainer'):
                WATCHER.owned_handle(456, script, {'--step': 5000, '--training-pid': 999}, root)
            with self.assertRaisesRegex(RuntimeError, 'registered owned executable'):
                WATCHER.owned_handle(456, script.with_name('train.py'), {}, root)

    def test_receipt_bindings_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(WATCHER, 'verify_sources'):
            root = Path(directory)
            artifact = root / 'result.csv'
            artifact.write_text('value\n0.2\n')
            receipt = root / 'completion.json'
            record = {'status': 'COMPLETE', 'milestone_sha256': 'frozen', 'source_hashes': {},
                'output_hashes': {'result.csv': WATCHER.sha256(artifact)}}
            receipt.write_text(json.dumps(record))
            WATCHER.validate_receipt(receipt, 'COMPLETE', {'milestone_sha256': 'frozen'})
            with self.assertRaisesRegex(RuntimeError, 'different result or prerequisite'):
                WATCHER.validate_receipt(receipt, 'COMPLETE', {'milestone_sha256': 'other'})
            artifact.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
                WATCHER.validate_receipt(receipt, 'COMPLETE', {'milestone_sha256': 'frozen'})
            record['output_hashes'] = {'../outside': 'bad'}
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, 'escaped'):
                WATCHER.validate_receipt(receipt, 'COMPLETE', {'milestone_sha256': 'frozen'})

    def test_completed_stage_not_rerun(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(WATCHER, 'verify_sources'), \
                mock.patch.object(WATCHER.subprocess, 'run') as runner:
            root = Path(directory)
            receipt = root / 'completion.json'
            record = {'status': 'COMPLETE', 'milestone_sha256': 'frozen', 'source_hashes': {}, 'output_hashes': {}}
            receipt.write_text(json.dumps(record))
            WATCHER.run_stage('evaluate.py', receipt, 'COMPLETE', {'milestone_sha256': 'frozen'}, root / 'log', {})
            runner.assert_not_called()

    def test_partial_evaluation_resumes_and_receipt_is_required(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(WATCHER, 'verify_sources'):
            root = Path(directory)
            receipt = root / 'completion.json'
            (root / 'metadata.json').write_text('{}')

            def execute(command, **keywords):
                self.assertIn('--resume', command)
                self.assertEqual(command[command.index('--step') + 1], '5000')
                self.assertEqual(keywords['stdin'], subprocess.DEVNULL)
                self.assertTrue(keywords['check'])
                receipt.write_text(json.dumps({'status': 'COMPLETE', 'milestone_sha256': 'frozen',
                    'source_hashes': {}, 'output_hashes': {}}))

            with mock.patch.object(WATCHER.subprocess, 'run', side_effect=execute):
                WATCHER.run_stage('evaluate.py', receipt, 'COMPLETE', {'milestone_sha256': 'frozen'}, root / 'log', {})
            receipt.unlink()
            with mock.patch.object(WATCHER.subprocess, 'run'):
                with self.assertRaises(FileNotFoundError):
                    WATCHER.run_stage('evaluate.py', receipt, 'COMPLETE', {'milestone_sha256': 'frozen'}, root / 'log', {})

    def test_dry_run_does_not_inspect_or_launch_processes(self):
        with mock.patch.object(sys, 'argv', ['finish_registered_trial.py', '--training-pid', '123', '--review-watcher-pid', '456']), \
                mock.patch.object(WATCHER, 'owned_handle') as ownership, \
                mock.patch.object(WATCHER.subprocess, 'run') as runner, mock.patch('builtins.print'):
            WATCHER.main()
        ownership.assert_not_called()
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
