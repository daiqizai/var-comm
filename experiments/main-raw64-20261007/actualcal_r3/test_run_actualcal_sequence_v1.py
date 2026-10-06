from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch,Mock
import run_actualcal_sequence_v1 as s


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return str(path)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def fixture(self):
        rp=self.root/'rootreg.json';cp=put(self.root/'rootcfg.json',dict(registration=str(rp),python='python',root=str(self.root),deadline_unix=1e20,stop_file=str(self.root/'STOP')))
        u=str(Path(s.d.u.__file__).absolute());ledger=str(Path(u).with_name('main_raw64_ledger.py'));put(rp,dict(config_sha256=s.sha(cp),source_bindings={p:s.sha(p) for p in (u,ledger)}))
        old=put(self.root/'oldreg.json',dict(source_bindings={}))
        m=dict(root_config=cp,legacy_batch={'registration':old},source_paths=[str(Path(s.__file__).absolute())],test_metadata='explicit-test-metadata',pythonpath=[str(Path(s.d.__file__).parent),str(Path(u).parent)],prepared_qualification=str(self.root/'qualification/completion.json'))
        return m

    def test_recovery_verifies_all_originals_copies_and_actual_failure_contract(self):
        import shutil
        old=self.root/'old_sequence';archive=self.root/'recovery';old.mkdir();archive.mkdir()
        owner=dict(pid=100,start_ticks=1);child=dict(pid=101,start_ticks=2)
        log=old/'qualification.log';log.write_text('Ran 31 tests\nFAILED (failures=1)\n')
        failure=put(old/'failure.json',dict(original_owner_success=False,retry_allowed=False))
        put(old/'identity.json',owner)
        end=put(old/'qualification_exit.json',dict(process_waited=True,exit_code=1,capture_or_monitor_error=None,identity=child,log=str(log),log_sha256=s.sha(log)))
        for i in range(16):(old/f'preserved{i}.txt').write_text(str(i))
        items=[]
        for p in sorted(old.iterdir()):
            cp=archive/('copy_'+p.name);shutil.copyfile(p,cp)
            items.append(dict(original=str(p),archive=str(cp),sha256=s.sha(p),bytes=p.stat().st_size))
        prep=put(archive/'archive_preparation_failure.json',{'historical_error':True})
        ap=put(archive/'failure_archive_receipt.json',dict(status='FAILED_ACTUALCAL_QUALIFICATION_ORIGINALS_PRESERVED',items=items,originals_retained=True,original_owner_success=False,processes_exited=[100,101],archive_preparation_failure_sha256=s.sha(prep)))
        dp=put(archive/'diagnosis.json',dict(status='ACTUALCAL_PRESTART_HASH_CACHE_FAILURE_DIAGNOSED',archive_sha256=s.sha(ap),tests_run=31,tests_passed=30,tests_failed=1,failed_test='test_hash_cache_revalidates_changed_file',actual_calibration_registered=False,actual_calibration_calls=0,development_calls=0,holdout_calls=0,original_scientific_protocol_changed=False,qualification_exit_sha256=s.sha(end),failure_sha256=s.sha(failure)))
        m={'recovery':{'archive':{'path':ap,'sha256':s.sha(ap)},'diagnosis':{'path':dp,'sha256':s.sha(dp)}}}
        with patch.object(s.d,'gone') as gone:
            pins=s.recovery_hash(m);self.assertEqual(gone.call_count,2);self.assertEqual(len(pins),43)
            Path(items[-1]['archive']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed original or archive'):s.recovery_hash(m)

    def test_actual_downloaded_recovery_schema_matches_reader(self):
        folder=Path(os.environ.get('ACTUALCAL_R2_TEST_FAILURE_METADATA',str(Path(__file__).resolve().parents[2]/'current/actualcal_failure_v1')))
        archive=s.read(folder/'failure_archive_receipt.json');diag=s.read(folder/'diagnosis.json')
        end=s.read(folder/'qualification_exit.json');identity=s.read(folder/'identity.json');failure=s.read(folder/'failure.json')
        self.assertEqual(archive['status'],'FAILED_ACTUALCAL_QUALIFICATION_ORIGINALS_PRESERVED')
        self.assertEqual(len(archive['items']),20);self.assertEqual(diag['archive_sha256'],s.sha(folder/'failure_archive_receipt.json'))
        self.assertTrue(all(set(('original','archive','sha256','bytes'))<=set(x) for x in archive['items']))
        self.assertEqual(set(archive['processes_exited']),{identity['pid'],end['identity']['pid']})
        self.assertEqual(diag['qualification_exit_sha256'],s.sha(folder/'qualification_exit.json'))
        self.assertEqual(diag['failure_sha256'],s.sha(folder/'failure.json'))
        self.assertTrue(end['process_waited']);self.assertEqual(end['exit_code'],1);self.assertIsNone(end['capture_or_monitor_error'])
        self.assertFalse(failure['original_owner_success']);self.assertFalse(diag['actual_calibration_registered'])
        self.assertEqual((diag['tests_run'],diag['tests_passed'],diag['tests_failed']),(31,30,1))

    def test_source_inventory_matches_qualified_original_runtime(self):
        m=self.fixture();sources,cfg,ledger=s.source_inventory(m)
        for name in ['main_raw64_actual_driver']+s.d.TEST_MODULES:self.assertEqual(sources[str(Path(s.d.__file__).with_name(name+'.py'))],s.sha(Path(s.d.__file__).with_name(name+'.py')))
        reg=s.read(cfg['registration']);reg['source_bindings'].pop(ledger);put(Path(cfg['registration']),reg)
        with self.assertRaisesRegex(ValueError,'ORIGINAL'):s.source_inventory(m)

    def test_receipt_recovery_full21_gate_and_changed_copy_rejected(self):
        import shutil
        old=self.root/'failed_r2';arc=self.root/'arc';old.mkdir();arc.mkdir()
        log=old/'qualification.log';log.write_text('Ran 34 tests in 24.121s\n\nOK\n')
        owner=dict(pid=901,start_ticks=2);child=dict(pid=902,start_ticks=3)
        put(old/'identity.json',owner)
        failure=put(old/'failure.json',dict(original_owner_success=False,retry_allowed=False,traceback="KeyError: 'budget_after'"))
        end=put(old/'qualification_exit.json',dict(identity=child,process_waited=True,exit_code=0,capture_or_monitor_error=None,log=str(log),log_sha256=s.sha(log)))
        put(old/'qualified/completion.json',dict(status=s.d.QSTATUS,tests_run=34,test_modules=s.d.TEST_MODULES,process_waited=True,exit_code=0,GPU_used=False,new_packet_decodes=0,log=str(log),outputs={end:s.sha(end),str(log):s.sha(log)}))
        for i in range(16):(old/f'original{i}.txt').write_text(str(i))
        entries=[]
        for p in sorted(old.rglob('*')):
            if not p.is_file():continue
            cp=arc/p.relative_to(old);cp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,cp)
            entries.append(dict(source=str(p),copy=str(cp),sha256=s.sha(p),bytes=p.stat().st_size))
        self.assertEqual(len(entries),21)
        ap=put(arc/'failure_archive_receipt.json',dict(status='FAILED_R2_PREPARE_EVIDENCE_PRESERVED',originals_unchanged=True,entries=entries,source_failure=failure,source_failure_sha256=s.sha(failure)))
        diag=put(arc/'diagnosis.json',dict(status='ORIGINAL_RECEIPT_SCHEMA_MISMATCH_DIAGNOSED',failure_archive_sha256=s.sha(ap),actual_qualification_tests=34,actual_qualification_passed=True,registration_created=False,calibration_calls=0,original_owner_success=False,processes_exited=True))
        m={'receipt_recovery':{k:dict(path=p,sha256=s.sha(p)) for k,p in [('archive',ap),('diagnosis',diag)]}}
        with patch.object(s.d,'gone') as gone:
            pins=s.recovery_receipts(m);self.assertEqual(len(pins),44);self.assertEqual(gone.call_count,2)
            Path(entries[-1]['copy']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed R2'):s.recovery_receipts(m)

    def test_real_downloaded_receipt_failure_metadata(self):
        base=Path(os.environ.get('ACTUALCAL_R3_TEST_FAILURE_METADATA',str(Path(__file__).resolve().parents[2]/'current')))
        a=s.read(base/'actualcal_receipt_recovery_r3/failure_archive_receipt.json');diag=s.read(base/'actualcal_receipt_recovery_r3/diagnosis.json')
        end=s.read(base/'actualcal_sequence_r2/qualification_exit.json');fail=s.read(base/'actualcal_sequence_r2/failure.json');q=s.read(base/'actualcal_cpu_qualification_r2/completion.json')
        self.assertEqual(a['status'],'FAILED_R2_PREPARE_EVIDENCE_PRESERVED');self.assertTrue(a['originals_unchanged']);self.assertEqual(len(a['entries']),21)
        self.assertEqual(diag['failure_archive_sha256'],s.sha(base/'actualcal_receipt_recovery_r3/failure_archive_receipt.json'))
        self.assertEqual(a['source_failure_sha256'],s.sha(base/'actualcal_sequence_r2/failure.json'))
        self.assertFalse(fail['original_owner_success']);self.assertIn('budget_after',fail['traceback'])
        self.assertTrue(end['process_waited']);self.assertEqual(end['exit_code'],0);self.assertIsNone(end['capture_or_monitor_error'])
        self.assertEqual(q['tests_run'],34);self.assertEqual(q['log'],end['log']);self.assertEqual(q['outputs'][end['log']],end['log_sha256'])
        self.assertFalse(diag['registration_created']);self.assertEqual(diag['calibration_calls'],0)

    def test_formal_metadata_manifest_binds_all_originals(self):
        entries=[]
        for i in range(47):
            p=put(self.root/f'm{i}.json',{'actual_fixture':i});entries.append(dict(remote=p,path=p,sha256=s.sha(p)))
        mp=put(self.root/'materials.json',{'scope':'fixture'})
        fp=put(self.root/'fixture.json',dict(status='ACTUALCAL_R3_REAL_PREPARE_CONTRACT_TEST_ASSETS',materials=dict(path=mp,sha256=s.sha(mp)),files=entries))
        pins=s.test_metadata({'test_metadata':fp});self.assertEqual(len(pins),49)
        Path(entries[-1]['path']).write_text('{}')
        with self.assertRaises(ValueError):s.test_metadata({'test_metadata':fp})

    def test_new_qualification_receipt_consumed_by_unchanged_driver(self):
        m=self.fixture();sources,cfg,ledger=s.source_inventory(m);e=self.root/'sequence';e.mkdir()
        def call(e,name,argv,env,aff,cfg,**kw):
            self.assertEqual(argv[-3:],s.d.TEST_MODULES);self.assertEqual(env['MAIN_RAW64_ROOT_LEDGER_MODULE'],ledger);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            log=e/'qualification.log';log.write_text(f'Ran {s.d.TEST_COUNT} tests in 1s\n\nOK\n');return put(e/'qualification_exit.json',dict(exit_code=0,process_waited=True,log=str(log),log_sha256=s.sha(log),identity={'pid':1}))
        with patch.object(s,'call',call):q=s.qualification(m,e,sources,cfg,ledger,[28,29])
        got=s.d.qualification(dict(prepared_qualification=q,python='python',source_bindings=sources),{q:s.sha(q)})
        self.assertEqual(got['tests_run'],s.d.TEST_COUNT);self.assertEqual(got['source_bindings'],sources)

    def test_partial_predecessor_failure_does_not_start_registered_stage(self):
        m=self.fixture();cfg=s.read(m['root_config']);m['root_completion']=str(self.root/'old_cpu/completion.json');m['root_owner_exit']=str(self.root/'old_cpu/exit.json')
        m['prescreen']={k:str(self.root/'pre'/f'{k}.json') for k in ('completion','science','worker_exit','sequence_completion','owner_exit','owner_identity','sequence_identity')}
        put(self.root/'old_cpu/failure.json',{'failed':True})
        with patch.object(s.d,'build_request') as prepare:
            with self.assertRaisesRegex(ValueError,'predecessor failed'):s.wait_predecessors(m,cfg)
            prepare.assert_not_called()

    def test_actual_owner_capture_error_marker_failure_still_waits_and_closes_log(self):
        cfg=dict(stop_file=str(self.root/'STOP'));child=Mock(pid=55);order=[]
        def wait():order.append('wait');return 1
        child.wait.side_effect=wait
        def stop(*a):order.append('stop');raise OSError('marker write')
        with patch.object(s.subprocess,'Popen',return_value=child),patch.object(s.d.u,'capture',side_effect=RuntimeError('capture')),patch.object(s.d.u,'request_stop',side_effect=stop):
            with self.assertRaisesRegex(ValueError,'normally'):s.call(self.root,'owner',['python'],{},[24,25],cfg,actual_owner=True)
        receipt=s.read(self.root/'owner_exit.json');self.assertEqual(order,['stop','wait']);self.assertTrue(receipt['process_waited']);self.assertIn('STOP_WRITE_FAILED',receipt['capture_or_monitor_error']);self.assertEqual(s.sha(receipt['log']),receipt['log_sha256'])

    def test_sequence_only_qualify_wait_prepare_register_owner_and_normal_consumer(self):
        actual=self.root/'actual';out=self.root/'out';m=self.fixture();cfg=s.read(m['root_config']);cfg['python']=str(Path(__import__('sys').executable).absolute());cfg['registration']=str(self.root/'rootreg.json')
        m.update(execution_dir=str(actual),out=str(out),resources={'affinities':[[24,25],[26,27]]});mp=put(self.root/'materials.json',m);e=self.root/'sequence';order=[];sources={str(Path(s.__file__).absolute()):s.sha(s.__file__)}
        @contextmanager
        def lock(cfg):yield
        def qualify(*a):order.append('qualify');return put(Path(m['prepared_qualification']),{})
        def prepare(mpath,target):order.append('prepare');put(Path(target),dict(source_bindings=sources));return {'sha256':s.sha(target)}
        def call(e,name,argv,*args,**kwargs):
            order.append(name);log=e/(name+'.log');log.write_text('closed')
            if name=='registration':
                put(actual/'config.json',dict(registration=str(actual/'registration.json')));put(actual/'registration.json',{})
            if name=='owner':put(actual/'completion.json',{});put(out/'completion.json',{})
            return put(e/(name+'_exit.json'),dict(log=str(log),log_sha256=s.sha(log),process_waited=True,exit_code=0))
        def closed(spec):
            order.append('closed');self.assertEqual(s.sha(spec['registration']),spec['bindings'][spec['registration']]);return dict(points=[1],records=[1],done=dict(frame_count=3,new_packet_calls=2,reused_frames=2),budget_before={'charged':4})
        with patch.object(s,'test_metadata',return_value={}),patch.object(s,'recovery',return_value={}),patch.object(s,'source_inventory',return_value=(sources,cfg,'ledger')),patch.object(s.sys,'platform','linux'),patch.object(s,'check'),patch.object(s.d.u,'set_resources'),patch.object(s,'sequence_lock',lock),patch.object(s.d.u,'process',return_value={'pid':7}),patch.object(s,'qualification',qualify),patch.object(s,'wait_predecessors',side_effect=lambda *a:order.append('wait')),patch.object(s.d,'build_request',prepare),patch.object(s,'call',call),patch.object(s.d,'load_registered',return_value={'cfg':{'registration':str(actual/'registration.json')}}),patch.object(s.d,'closed_calibration',closed):
            got=s.run(mp,e,[28,29])
        self.assertEqual(order,['qualify','wait','prepare','registration','owner','closed']);self.assertEqual(got['status'],s.STATUS);self.assertFalse(got['automatic_visual']);self.assertFalse(got['automatic_development'])


if __name__=='__main__':unittest.main()
