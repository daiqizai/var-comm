"""CPU fixtures for exact cohort admission and unchanged source checkpoint scope."""
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cohort_runtime as c
import source_worker as w


def proc(pid):return dict(pid=pid,start_ticks=str(1000+pid),uid=42,argv=['python','source_worker.py','--config',str(pid)+'.json'])


def fixture(folder):
    owner=proc(10);one=dict(proc(11),worker_id='w0',source_indices=[0,2]);two=dict(proc(12),worker_id='w1',source_indices=[1,3])
    manifest=dict(schema='M1_GPU_COHORT_V1',status='READY',stage='benchmark',owner=owner,workers=[one,two])
    path=folder/'cohort.json';c.write(path,manifest)
    config=dict(stage='benchmark',worker_id='w0',source_indices=[0,2],cohort_manifest=str(path))
    processes={p['pid']:{k:p[k] for k in ('pid','start_ticks','uid','argv')} for p in (owner,one,two)}
    return config,manifest,processes


class CohortTests(unittest.TestCase):
    def test_only_exact_registered_live_GPU_processes_are_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp));guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=processes.__getitem__,current_pid=11)
            guard.check_identities([dict(pid=12)])
            for key,value in [('start_ticks','999'),('uid',43),('argv',['unrelated'])]:
                original=processes[12][key];processes[12][key]=value
                with self.assertRaisesRegex(RuntimeError,'identity changed'):guard.check_identities([dict(pid=12)])
                processes[12][key]=original

    def test_finished_peer_need_not_remain_alive_when_absent_from_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp));guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=processes.__getitem__,current_pid=11)
            del processes[12];guard.check_identities([])
            with self.assertRaises(KeyError):guard.check_identities([dict(pid=12)])

    def test_foreign_GPU_and_changed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp));guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=processes.__getitem__,current_pid=11)
            with self.assertRaisesRegex(RuntimeError,'unregistered process'):guard.check_identities([dict(pid=99)])
            c.write(config['cohort_manifest'],dict(manifest,extra=True))
            with self.assertRaisesRegex(RuntimeError,'manifest changed'):guard.check_identities([])

    def test_overlapping_sources_and_cross_user_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,_=fixture(Path(tmp));bad=copy.deepcopy(manifest);bad['workers'][1]['source_indices']=[2,3]
            with self.assertRaisesRegex(RuntimeError,'overlap'):c.validate_manifest(bad,config)
            bad=copy.deepcopy(manifest);bad['workers'][1]['uid']=99
            with self.assertRaisesRegex(RuntimeError,'user ownership'):c.validate_manifest(bad,config)

    def test_owner_and_current_worker_identity_are_always_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp));guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=processes.__getitem__,current_pid=11)
            processes[10]['argv']=['changed']
            with self.assertRaisesRegex(RuntimeError,'identity changed'):guard.check_identities([])

    def test_install_preserves_original_require_and_thermal_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp));guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=processes.__getitem__,current_pid=11)
            namespace={};exec('''
class ResourceBusy(RuntimeError): pass
def foreign_gpu_processes(): return [{'pid':12}]
def require_available():
    if foreign_gpu_processes(): raise ResourceBusy('foreign')
class Safety:
    def __init__(self): self.hot_count=0
    def check(self):
        require_available()
        self.hot_count+=1
        return self.hot_count>=3
''',namespace)
            class Runtime:
                def __getattr__(self,key):return namespace[key]
                def __setattr__(self,key,value):namespace[key]=value
            runtime=Runtime();probe=SimpleNamespace(Safety=namespace['Safety'],require_available=namespace['require_available'])
            original_require=probe.require_available;original_check=probe.Safety.check
            guard.install(probe,runtime)
            self.assertIs(probe.require_available,original_require);self.assertIs(probe.Safety.check,original_check)
            self.assertIs(runtime.require_available,original_require);original_require()
            safety=probe.Safety();self.assertEqual([safety.check() for _ in range(3)],[False,False,True])
            processes[12]['start_ticks']='reused'
            with self.assertRaises(namespace['ResourceBusy']):original_require()

    def test_proc_stat_with_spaces_preserves_start_tick_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'123';folder.mkdir();tail=['S']+['0']*18+['4567']+['0']*3
            (folder/'stat').write_text('123 (python source worker) '+' '.join(tail))
            (folder/'status').write_text('Name:\tpython\nUid:\t42\t42\t42\t42\n')
            (folder/'cmdline').write_bytes(b'python\0source_worker.py\0')
            self.assertEqual(c.process_identity(123,Path(tmp)),dict(pid=123,start_ticks='4567',uid=42,argv=['python','source_worker.py']))

    def test_just_exited_registered_peer_is_requeried_but_mismatch_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,manifest,processes=fixture(Path(tmp))
            def reader(pid):
                if pid==12:raise FileNotFoundError('exited')
                return processes[pid]
            guard=c.CohortGuard(config,manifest,c.sha(config['cohort_manifest']),reader=reader,current_pid=11)
            guard.runtime=SimpleNamespace(ResourceBusy=RuntimeError)
            calls=[]
            def query():
                calls.append(1);return [dict(pid=12)] if len(calls)==1 else []
            guard.original_foreign_gpu_processes=query
            with patch.object(c.time,'sleep'):guard.require_available()
            self.assertEqual(len(calls),2)
            guard.reader=processes.__getitem__;processes[12]['start_ticks']='wrong';calls.clear()
            with self.assertRaisesRegex(RuntimeError,'identity changed'):guard.require_available()
            self.assertEqual(len(calls),1)


@contextmanager
def fake_lock(*_):yield


def checkpoint(path,binding,index):
    value=c.read(path);digest=value.pop('payload_sha256')
    c.require(c.identity(value)==digest and value['binding']==binding and value['source_index']==index,'bad checkpoint')
    return value


def save(path,value):c.write(path,dict(value,payload_sha256=c.identity(value)))


class SourceTests(unittest.TestCase):
    def setup(self,folder,mode='benchmark'):
        original=folder/'original';scratch=folder/'scratch';reg=original/'calibrate_registration.json';c.write(reg,dict(stage='calibrate'))
        source=folder/'source.py';source.write_text('frozen')
        config=dict(original_out=str(original),scratch_out=str(scratch),stage=mode,source_bindings={str(source):c.sha(source)},
            expected_calibration_registration_sha256=c.sha(reg))
        binding=c.identity(c.read(reg));rows=[dict(frame=i,image_sha256='same') for i in range(2070)]
        saved=dict(binding=binding,source_index=0,rows=rows,synthetic=False,development_read=False,seconds=2.)
        reference=original/'calibrate/source_checkpoints/0000.json';save(reference,saved)
        class Runner:
            def __init__(self):self.out=original;self.calls=[];self.change=False
            def guard(self):pass
            def calibration_source(self,stage,binding,data,index,actions):
                self.calls.append((self.out,stage,binding,index));path=self.out/stage/'source_checkpoints/0000.json'
                value=copy.deepcopy(saved)
                if self.change:value['rows'][0]['image_sha256']='different'
                if path.exists():value=checkpoint(path,binding,index)
                else:save(path,value)
                return value,path
            def validate_calibration_rows(self,rows,data,index,actions):assert len(rows)==2070
        return config,binding,Runner(),SimpleNamespace(identity=c.identity,sha=c.sha,checkpoint=checkpoint),reference

    def test_benchmark_recomputes_to_scratch_with_original_binding_and_exact_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,binding,runner,u,reference=self.setup(Path(tmp));before=c.sha(reference)
            with patch.object(c,'source_lock',fake_lock):saved,path,digest=w.source_once(runner,u,config,binding,None,None,0)
            self.assertEqual(runner.calls,[(Path(config['scratch_out']),'cohort_benchmark',binding,0)])
            self.assertEqual(runner.out,Path(config['original_out']));self.assertEqual(c.sha(reference),before)
            self.assertEqual(digest,c.identity(saved['rows']));self.assertTrue(str(path).startswith(config['scratch_out']))

    def test_benchmark_any_original_row_change_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,binding,runner,u,_=self.setup(Path(tmp));runner.change=True
            with patch.object(c,'source_lock',fake_lock):
                with self.assertRaisesRegex(RuntimeError,'2070-row benchmark differs'):w.source_once(runner,u,config,binding,None,None,0)
            self.assertEqual(runner.out,Path(config['original_out']))

    def test_benchmark_never_times_reused_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,binding,runner,u,_=self.setup(Path(tmp))
            with patch.object(c,'source_lock',fake_lock):
                w.source_once(runner,u,config,binding,None,None,0)
                with self.assertRaisesRegex(RuntimeError,'scratch source already exists'):w.source_once(runner,u,config,binding,None,None,0)
            self.assertEqual(len(runner.calls),1)

    def test_production_uses_original_path_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,binding,runner,u,reference=self.setup(Path(tmp),'production');before=c.sha(reference)
            with patch.object(c,'source_lock',fake_lock):_,path,_=w.source_once(runner,u,config,binding,None,None,0)
            self.assertEqual(path,reference);self.assertEqual(runner.calls,[(Path(config['original_out']),'calibrate',binding,0)])
            self.assertEqual(c.sha(reference),before)

    def test_changed_registration_fails_before_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,binding,runner,u,_=self.setup(Path(tmp));c.write(Path(config['original_out'])/'calibrate_registration.json',dict(changed=True))
            with patch.object(c,'source_lock',fake_lock):
                with self.assertRaisesRegex(RuntimeError,'registration changed'):w.source_once(runner,u,config,binding,None,None,0)
            self.assertEqual(runner.calls,[])

    @unittest.skipIf(os.name=='nt','POSIX source locks exercised on the GPU server')
    def test_two_writers_cannot_own_same_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with c.source_lock(tmp,5):
                with self.assertRaisesRegex(RuntimeError,'Another writer'):
                    with c.source_lock(tmp,5):pass


if __name__=='__main__':unittest.main()
