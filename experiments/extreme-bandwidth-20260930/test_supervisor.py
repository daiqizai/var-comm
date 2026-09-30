"""Completion receipts must reflect executed stages and verified publication."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('extreme_bw_supervisor_test', Path(__file__).with_name('supervisor.py'))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class CompletionTests(unittest.TestCase):
    def check(self, stage, record):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            if record is not None:
                path.write_text(json.dumps(record))
            return supervisor.receipt_complete(stage, path)

    def test_commit_before_failed_push_is_incomplete(self):
        stage = dict(name='publication', complete_status='PUSHED')
        self.assertFalse(self.check(stage, dict(status='COMMITTED', commit='abc', checks='PASS')))

    def test_remote_mismatch_is_incomplete(self):
        stage = dict(name='publication', complete_status='PUSHED')
        self.assertFalse(self.check(stage, dict(status='PUSHED', commit='abc', remote_commit='def', checks='PASS')))
        self.assertFalse(self.check(stage, dict(status='PUSHED', commit='', remote_commit='', checks='PASS')))
        self.assertTrue(self.check(stage, dict(status='PUSHED', commit='abc', remote_commit='abc', checks='PASS')))

    def test_existence_and_wrong_status_do_not_complete_stage(self):
        stage = dict(name='training', complete_status='DONE', required_fields={'development_read': False})
        self.assertFalse(self.check(stage, None))
        self.assertFalse(self.check(stage, dict(status='FAILED', development_read=False)))
        self.assertFalse(self.check(stage, dict(status='DONE', development_read=True)))
        self.assertTrue(self.check(stage, dict(status='DONE', development_read=False)))


if __name__ == '__main__':
    unittest.main()
