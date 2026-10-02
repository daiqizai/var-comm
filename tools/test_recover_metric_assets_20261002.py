"""Operational recovery tests with fake children; no network or experiment execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

helper=Path(__file__).with_name('recover_assets.py')
if not helper.exists():helper=Path(__file__).with_name('recover_metric_assets_20261002.py')
spec=importlib.util.spec_from_file_location('recover_assets_test_target',helper)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def test_process_scan_binds_exact_script_start_ticks_and_ignores_zombies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';source.mkdir();proc=root/'proc';proc.mkdir()
            for pid,name,state in ((10001,'prepare_assets.py','S'),(10002,'supervisor.py','Z'),(10003,'other.py','S')):
                folder=proc/str(pid);folder.mkdir()
                (folder/'stat').write_text(f'{pid} (name with spaces) '+' '.join([state]+['0']*18+['98765']))
                (folder/'cmdline').write_bytes(b'python\0'+(source/name).as_posix().encode()+b'\0')
            with mock.patch.object(r.os,'getpid',return_value=1):
                self.assertEqual(r.named_processes(source,proc),[dict(pid=10001,start_ticks='98765',state='S',script='prepare_assets.py')])

    def fixture(self,root):
        source=root/'experiments/unified-metrics-20261002';source.mkdir(parents=True)
        out=root/'outputs/UNIFIED-METRICS-20261002';out.mkdir(parents=True)
        for name in ('prepare_assets.py','supervisor.py'):(source/name).write_text('# ENGINEERING fixture\n')
        python=root/'original-python';python.write_text('ENGINEERING interpreter placeholder')
        bound={str(p.resolve()):r.sha(p) for p in source.iterdir()}
        r.write(out/'supervisor_registration.json',dict(source_bindings=bound))
        self.failure(out,'<urlopen error timed out>')
        return source,out,python

    def failure(self,out,error):r.write(out/'assets_status.json',dict(status='FAILED',error=error,time=time.time()))

    def ready(self,out,python):
        (out/'modelmanifest.json').write_text('{"ENGINEERING":true}')
        (out/'environment_freeze.txt').write_text('ENGINEERING fixture')
        r.write(out/'assets_complete.json',dict(status='ASSETS_READY',training_updates=0,
            manifest_sha256=r.sha(out/'modelmanifest.json'),environment=str(python),
            environment_freeze_sha256=r.sha(out/'environment_freeze.txt')))

    def test_transient_whitelist_does_not_accept_integrity_env_or_program_errors(self):
        for text in ('<urlopen error timed out>','Incomplete HTTP asset response',
                     '<urlopen error [Errno -3] Temporary failure in name resolution>',
                     '[Errno 104] Connection reset by peer','HTTP Error 503: Service Unavailable',
                     'IncompleteRead(56 bytes read, 1024 more expected)'):
            self.assertTrue(r.transient(text),text)
        for text in ('Downloaded asset hash differs: file','Bad CRC-32 for file','File is not a zip file',
                     'Metric asset manifest changed','ModuleNotFoundError: pip','HTTP Error 403: Forbidden',
                     'HTTP Error 416: Requested Range Not Satisfiable','Command timed out during pip install',
                     '<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>'):
            self.assertFalse(r.transient(text),text)

    def test_two_attempt_recovery_preserves_partial_and_receipts_then_launches_supervisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            part=out/'dreamsim.zip.part';part.write_bytes(b'ORIGINAL PART DO NOT TOUCH')
            initial=r.sha(part);calls=[]
            def start(script,interpreter,cwd,log):
                calls.append(script.name);number=len(calls);log.write_text('ENGINEERING child log')
                launch=dict(pid=number+100,start_ticks=str(number),state='S',script=str(script),
                            python=str(interpreter),log=str(log),launched_at=time.time())
                def wait():
                    if number==1:self.failure(out,'Incomplete HTTP asset response');return 1
                    self.ready(out,python);return 0
                return SimpleNamespace(pid=number+100,wait=wait,poll=lambda:None),launch
            with mock.patch.dict(sys.modules,{'fcntl':SimpleNamespace(flock=lambda *a:None,LOCK_EX=2,LOCK_NB=4)}), \
                 mock.patch.object(r,'named_processes',return_value=[]),mock.patch.object(r,'start',side_effect=start), \
                 mock.patch.object(r.time,'sleep') as sleep,mock.patch.object(r,'process_identity',return_value=dict(pid=103,start_ticks='3',state='S')):
                r.recover(root,python,max_attempts=3,initial_delay=1,max_delay=4)
            state=r.read(out/'assets_recovery/state.json')
            self.assertEqual(calls,['prepare_assets.py','prepare_assets.py','supervisor.py'])
            self.assertEqual(state['status'],'SUPERVISOR_RESTARTED')
            self.assertEqual([x['returncode'] for x in state['attempts']],[1,0])
            self.assertEqual(r.sha(part),initial)
            self.assertEqual(len(list((out/'assets_recovery').glob('attempt_*/launch.json'))),2)
            self.assertTrue(all(entry['previous_receipts'] for entry in state['attempts']))
            self.assertEqual([x.args for x in sleep.call_args_list],[(1,),(2,)])

    def test_nontransient_failure_never_launches_or_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            self.failure(out,'Downloaded asset hash differs: registered.pt')
            with mock.patch.dict(sys.modules,{'fcntl':SimpleNamespace(flock=lambda *a:None,LOCK_EX=2,LOCK_NB=4)}), \
                 mock.patch.object(r,'named_processes',return_value=[]),mock.patch.object(r,'start') as start:
                with self.assertRaisesRegex(RuntimeError,'explicit transient'):r.recover(root,python)
                start.assert_not_called()

    def test_bounded_attempts_and_relaunch_preserve_failure_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            def start(script,interpreter,cwd,log):
                launch=dict(pid=123,start_ticks='456',state='S',launched_at=time.time())
                def wait():self.failure(out,'<urlopen error timed out>');return 1
                return SimpleNamespace(wait=wait),launch
            with mock.patch.dict(sys.modules,{'fcntl':SimpleNamespace(flock=lambda *a:None,LOCK_EX=2,LOCK_NB=4)}), \
                 mock.patch.object(r,'named_processes',return_value=[]),mock.patch.object(r,'start',side_effect=start) as launch:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'exhausted'):r.recover(root,python,max_attempts=1)
                self.assertEqual(launch.call_count,1)

    def test_live_named_process_refuses_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            active=[dict(pid=456,start_ticks='789',state='S',script='prepare_assets.py')]
            with mock.patch.dict(sys.modules,{'fcntl':SimpleNamespace(flock=lambda *a:None,LOCK_EX=2,LOCK_NB=4)}), \
                 mock.patch.object(r,'named_processes',return_value=active),mock.patch.object(r,'start') as start:
                with self.assertRaisesRegex(RuntimeError,'still alive'):r.recover(root,python)
                start.assert_not_called()

    def test_frozen_source_and_completed_asset_checksums_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            r.frozen_sources(source,out);self.ready(out,python);self.assertIsNotNone(r.ready_assets(out))
            (out/'modelmanifest.json').write_text('modified')
            with self.assertRaisesRegex(RuntimeError,'invalid'):r.ready_assets(out)
            (source/'supervisor.py').write_text('modified')
            with self.assertRaisesRegex(RuntimeError,'inventory'):r.frozen_sources(source,out)

    def test_interrupted_helper_does_not_bypass_previous_hard_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,out,python=self.fixture(root)
            attempts=[dict(launch=dict(launched_at=time.time()-1))]
            self.failure(out,'Bad CRC-32 for file')
            with self.assertRaisesRegex(RuntimeError,'explicit transient'):r.require_retryable_failure(out,attempts)
            self.failure(out,'timed out');r.require_retryable_failure(out,attempts)
            attempts[0]['returncode']=-9
            with self.assertRaisesRegex(RuntimeError,'verified retryable'):r.require_retryable_failure(out,attempts)


if __name__=='__main__':unittest.main(verbosity=2)
