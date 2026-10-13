"""Synthetic CPU tests: no original owner, resource polling or scientific call."""
import importlib.util
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('clean_wait_once.py');s=importlib.util.spec_from_file_location('waiter_under_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ZERO=dict(zero_science_proven=True,existing_attempt_markers=[],active_owner_or_worker=[])
EXIT=dict(actual_wait=True,returncode=0,timed_out=False)

class Tests(unittest.TestCase):
    def test_unambiguous_race_wait_only(self):
        self.assertEqual(m.classify_child(EXIT,'{"status":"WAIT","model_calls":0}',ZERO),'WAIT_RACE_PROVEN_ZERO_SCIENCE')
    def test_nonzero_cannot_retry(self):
        self.assertEqual(m.classify_child(dict(EXIT,returncode=1),'{}',ZERO),'STOP_UNKNOWN_OR_FAILED_NO_RETRY')
    def test_timeout_cannot_retry(self):
        self.assertEqual(m.classify_child(dict(EXIT,timed_out=True),'{}',ZERO),'STOP_UNKNOWN_OR_FAILED_NO_RETRY')
    def test_absent_actual_wait_cannot_retry(self):
        self.assertEqual(m.classify_child(dict(EXIT,actual_wait=False),'{}',ZERO),'STOP_UNKNOWN_OR_FAILED_NO_RETRY')
    def test_wait_with_attempt_marker_cannot_retry(self):
        self.assertEqual(m.classify_child(EXIT,'{"status":"WAIT","model_calls":0}',dict(ZERO,zero_science_proven=False)),'STOP_UNCLOSED_OR_AMBIGUOUS_NO_RETRY')
    def test_wait_requires_explicit_zero_calls(self):
        self.assertEqual(m.classify_child(EXIT,'{"status":"WAIT"}',ZERO),'STOP_UNCLOSED_OR_AMBIGUOUS_NO_RETRY')
    def test_unknown_stdout_cannot_retry(self):
        self.assertEqual(m.classify_child(EXIT,'trace\n{"status":"WAIT","model_calls":0}',ZERO),'STOP_UNRECOGNIZED_EXIT_NO_RETRY')
        self.assertEqual(m.classify_child(EXIT,'[]',ZERO),'STOP_UNRECOGNIZED_EXIT_NO_RETRY')
    def test_receipt_is_not_completion(self):
        self.assertEqual(m.classify_child(EXIT,'{"status":"COMPLETE_CLEAN5400"}',ZERO),'STOP_UNCLOSED_OR_AMBIGUOUS_NO_RETRY')
    def test_complete_with_original_actual_proof(self):
        c=dict(status='COMPLETE_CLEAN5400',all_actual_exit_zero=True)
        self.assertEqual(m.classify_child(EXIT,'{"status":"COMPLETE_CLEAN5400"}',dict(ZERO,zero_science_proven=False),c),'COMPLETE')
    def test_original_stage_blocks_even_without_owner_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clean=types.SimpleNamespace(TMP=p/'tmp',OUT=p/'out');(clean.TMP/'render').mkdir(parents=True)
            result=m.zero_science_scene(clean,live=[]);self.assertFalse(result['zero_science_proven'])
    def test_live_child_blocks_even_without_markers(self):
        with tempfile.TemporaryDirectory() as d:
            clean=types.SimpleNamespace(TMP=Path(d)/'tmp',OUT=Path(d)/'out')
            self.assertFalse(m.zero_science_scene(clean,live=[dict(pid=12)])['zero_science_proven'])
    def test_resource_wait_files_do_not_count_as_science(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clean=types.SimpleNamespace(TMP=p/'tmp',OUT=p/'out');clean.TMP.mkdir();clean.OUT.mkdir()
            (clean.OUT/'resource_wait.json').write_text('{}',encoding='utf8');(clean.TMP/'prelaunch').mkdir()
            self.assertTrue(m.zero_science_scene(clean,live=[])['zero_science_proven'])
    def test_probe_acquires_shared_timing_and_exclusive_owner(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clean=types.SimpleNamespace(TMP=p/'tmp',TIMING_LOCK=p/'timing.lock');events=[]
            f=types.SimpleNamespace(LOCK_SH=1,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=lambda fd,op:events.append((Path(fd.name).name,op)))
            with m.probe_locks(clean,f) as locked:self.assertEqual(locked['status'],'LOCKED')
            self.assertEqual(events,[('timing.lock',5),('owner.lock',6),('owner.lock',8),('timing.lock',8)])
    def test_probe_timing_busy_returns_wait(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clean=types.SimpleNamespace(TMP=p/'tmp',TIMING_LOCK=p/'timing.lock')
            f=types.SimpleNamespace(LOCK_SH=1,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=lambda *a:(_ for _ in ()).throw(BlockingIOError()))
            with m.probe_locks(clean,f) as locked:self.assertEqual(locked['status'],'WAIT')
    def test_probe_exception_releases_both(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clean=types.SimpleNamespace(TMP=p/'tmp',TIMING_LOCK=p/'timing.lock');events=[]
            f=types.SimpleNamespace(LOCK_SH=1,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=lambda fd,op:events.append(op))
            with self.assertRaisesRegex(ValueError,'stop'):
                with m.probe_locks(clean,f):raise ValueError('stop')
            self.assertEqual(events,[5,6,8,8])
    def test_review_gate_precedes_all_calls(self):
        with patch.object(m,'load_clean',side_effect=AssertionError('must not load')):self.assertRaises(RuntimeError,m.run,False)
    def test_wait_budget_is_separate_from_science(self):
        self.assertEqual(m.POLL_SECONDS,300);self.assertEqual(m.WAIT_SECONDS,86400)
    def test_dispatch_rechecks_entry_bytes_after_wait(self):
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}),patch.object(m,'sha',return_value='changed'):
            self.assertRaisesRegex(RuntimeError,'changed while waiting',m.dispatch,None,{},None,0)
    def test_dispatch_rechecks_zero_science_before_marking(self):
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}),patch.object(m,'sha',side_effect=[m.ENTRY_SHA,m.REG_SHA]),patch.object(m,'zero_science_scene',return_value=dict(ZERO,zero_science_proven=False)):
            self.assertRaisesRegex(RuntimeError,'Attempt appeared',m.dispatch,None,{},None,0)
    def test_owned_exit_wait_is_actual(self):
        child=types.SimpleNamespace(pid=17,wait=lambda timeout:0)
        self.assertEqual(m.wait_owned_child(child,100)['returncode'],0)
    def test_timeout_terminates_owned_group_and_waits(self):
        calls=[]
        def wait(timeout=None):
            calls.append('wait')
            if calls.count('wait')==1:raise m.subprocess.TimeoutExpired('owned',1)
            return -15
        child=types.SimpleNamespace(pid=17,wait=wait)
        with patch.object(m.os,'getpgid',return_value=17,create=True),patch.object(m.os,'killpg',side_effect=lambda *a:calls.append('killpg'),create=True):r=m.wait_owned_child(child,1)
        self.assertEqual(calls,['wait','killpg','wait']);self.assertTrue(r['actual_wait']);self.assertTrue(r['timed_out']);self.assertEqual(r['returncode'],-15)

if __name__=='__main__':unittest.main()
