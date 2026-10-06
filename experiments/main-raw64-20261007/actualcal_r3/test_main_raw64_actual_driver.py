import copy
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
import numpy as np

import main_raw64_actual_driver as d
import main_raw64_calibration_checkpoints as c
import test_main_raw64_calibration_checkpoints as fixture
import test_main_raw64_calibration_budget as bf


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(c.canonical(value),encoding='utf-8'); return str(path)


def ident(pid=7,argv=None):
    return dict(pid=pid,start_ticks=99,uid=1002,argv=argv or ['python'],ppid=1,state='S')


class ActualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)

    def test_prescreen_true_writer_contract_and_failed_exit_rejected(self):
        root=self.root; source=put(root/'science_registration.json',{}); cat=put(root/'catalogue.json',{})
        request=put(root/'request.json',dict(deadline_unix=99999)); wi=ident(argv=['python','-B','prescreen','--worker',request])
        log=root/'worker.log';log.write_text('normal');end=put(root/'worker_exit.json',dict(identity=wi,exit_code=0,process_waited=True,log=str(log),log_sha256=d.sha(log)))
        short=put(root/'shortlist.json',dict(catalogue_sha256=d.sha(cat),science_registration_sha256=d.sha(source),original_construction_source_ids=[str(i) for i in range(200)],final_objective=c.PRIMARY,development_used=False,holdout_used=False))
        science=put(root/'science.json',dict(status='RAW64_SIX_SNR_FINITE_SHORTLIST_COMPLETE',source_count=200,wire_count=433,SNRs=list(c.SNRS),new_PHY_calls=0,GPU_used=False,automatic_successor=False,request_sha256=d.sha(request),worker_identity=wi,outputs={short:d.sha(short)}))
        normal=put(root/'completion.json',dict(status='RAW64_PRESCREEN_NORMAL_COMPLETE',original_worker_success=True,automatic_calibration=False,science_completion_sha256=d.sha(science),exit_receipt_sha256=d.sha(end)))
        reg=put(root/'registration.json',dict(status='FROZEN_READ_ONLY_PRESCREEN_EXECUTION',request_sha256=d.sha(request)))
        oi=ident(8,argv=['python','-B','prescreen','--request',request]);owner=put(root/'owner_identity.json',oi)
        oe=put(root/'owner_exit.json',dict(identity=oi,process_waited=True,exit_code=0,log=str(log),log_sha256=d.sha(log),capture_error=None))
        outer=put(root/'outer_completion.json',dict(status='RAW64_PRESCREEN_SEQUENCE_NORMAL_COMPLETE',original_owner_success=True,owner_completion_sha256=d.sha(normal),owner_exit_sha256=d.sha(oe),automatic_calibration=False,new_PHY=0,GPU=False))
        si=put(root/'sequence_identity.json',ident(9));spec=dict(completion=normal,science=science,registration=reg,request=request,worker_exit=end,owner_identity=owner,shortlist=short,owner_exit=oe,sequence_completion=outer,sequence_identity=si)
        bound={p:d.sha(p) for p in list(spec.values())+[str(log)]}
        with patch.object(d,'gone'):
            self.assertEqual(d.prescreen_closed(spec,bound,cat,source),d.read(short))
            x=d.read(end);x['exit_code']=1;put(Path(end),x);bound[end]=d.sha(end)
            normalvalue=d.read(normal);normalvalue['exit_receipt_sha256']=d.sha(end);put(Path(normal),normalvalue);bound[normal]=d.sha(normal)
            with self.assertRaisesRegex(ValueError,'wait0'):d.prescreen_closed(spec,bound,cat,source)

    def test_original_history_uses_wait_completed_not_invented_process_waited(self):
        r=self.root; op=put(r/'owner_config.json',dict(registration=str(r/'reg.json'))); cp=put(r/'worker_config.json',dict(registration=str(r/'reg.json')))
        sources={};inputs={op:d.sha(op),cp:d.sha(cp)}
        reg=put(r/'reg.json',dict(status='MAIN_ACTUAL_CALIBRATION_EXECUTION_REGISTERED_R1',source_bindings=sources,input_bindings=inputs)); wi=ident(); budget=dict(unresolved=0)
        sp=put(r/'science.json',dict(status='MAIN_ACTUAL_CALIBRATION_CPU_COMPLETE',registration_sha256=d.sha(reg),config_sha256=d.sha(cp),frame_count=30000,source_count=1000,packet_calls=60000,budget=budget,outputs={},source_bindings=sources,input_bindings=inputs))
        outputs={sp:d.sha(sp)}
        for i,name in enumerate(('worker_0','worker_1','merge')):
            folder=r/name;folder.mkdir();log=folder/'worker.log';log.write_text('closed')
            launch=put(folder/'launch.json',dict(identity=ident(i+30),owner_identity=wi)); end=put(folder/'exit.json',dict(wait_completed=True,exit_code=0,identity=ident(i+30),registration_sha256=d.sha(reg),launch_sha256=d.sha(launch),log_sha256=d.sha(log),completion_sha256=d.sha(sp)))
            outputs.update({p:d.sha(p) for p in (str(log),launch,end)})
        np_=put(r/'normal.json',dict(status='MAIN_ACTUAL_CALIBRATION_NORMALLY_COMPLETE_R1',registration_sha256=d.sha(reg),owner_config_sha256=d.sha(op),frame_count=30000,packet_calls=60000,budget_after=budget,owner_identity=wi,outputs=outputs,source_bindings=sources,input_bindings=inputs))
        obs=put(r/'observation.json',dict(status='MAIN_ACTUAL1000_CPU_NORMAL_CLOSED_OUTPUTS_VERIFIED',original_owner_success=True,processes_exited=True,original_receipts_validated=True,paid_inventory_validated=True,registration_sha256=d.sha(reg),science_completion_sha256=d.sha(sp),owner_completion_sha256=d.sha(np_)))
        spec=dict(observation=obs,registration=reg,completion=np_,science=sp,config=cp,owner_config=op);bound={p:d.sha(p) for p in spec.values()}
        with patch.object(d,'gone'):
            value=d.historical(spec,bound);self.assertIn(str(r/'merge/exit.json'),value['bound'])
            p=r/'worker_0/exit.json';x=d.read(p);x['wait_completed']=False;put(p,x)
            with self.assertRaisesRegex(ValueError,'pin'):d.historical(spec,bound)

    def test_real_downloaded_prepare_chain_keeps_original_receipt_bytes(self):
        # File-access mapping only: historical/prescreen/assets/inspect/qualification
        # execute unchanged. Unavailable old Python files are opaque sealed hashes;
        # remote qualification separately hashes their actual bytes before this test.
        from pathlib import PurePosixPath
        manifest_path=os.environ.get('ACTUALCAL_R3_TEST_METADATA',str(Path(__file__).with_name('prepare_contract_metadata.local.json')))
        meta=d.read(manifest_path);real_sha=d.sha;real_read=d.read;real_path=d.Path
        self.assertEqual(meta['status'],'ACTUALCAL_R3_REAL_PREPARE_CONTRACT_TEST_ASSETS')
        self.assertEqual(real_sha(meta['materials']['path']),meta['materials']['sha256'])
        lookup={x['remote']:x['path'] for x in meta['files']}
        for x in meta['files']:self.assertEqual(real_sha(x['path']),x['sha256'])
        materials=real_read(meta['materials']['path'])
        rootcfg=real_read(lookup[materials['root_config']]);rootreg=real_read(lookup[rootcfg['registration']])
        oldreg=real_read(lookup[materials['legacy_batch']['registration']])
        opaque=d.combine(rootreg['source_bindings'],oldreg['source_bindings'])
        materials['source_paths']=[str(Path(__file__).with_name(Path(p).name).absolute()) for p in materials['source_paths']]
        materials['receiver_module']=str(Path(__file__).with_name('main_raw64_keep_receiver.py').absolute())
        materials['prepared_qualification']=str(self.root/'qualified.json')
        mp=put(self.root/'materials.json',materials)
        sources=d.combine(opaque,{p:real_sha(p) for p in materials['source_paths']})
        for mod in (d.u,d.c,d.budget):sources[str(Path(mod.__file__).absolute())]=real_sha(mod.__file__)
        for name in ['main_raw64_actual_driver']+d.TEST_MODULES:
            p=str(Path(d.__file__).with_name(name+'.py').absolute());sources[p]=real_sha(p)
        log=self.root/'formal_fixture.log';log.write_text(f'Ran {d.TEST_COUNT} tests in 1s\n\nOK\n')
        put(Path(materials['prepared_qualification']),dict(status=d.QSTATUS,tests_run=d.TEST_COUNT,test_modules=d.TEST_MODULES,python=rootcfg['python'],process_waited=True,exit_code=0,GPU_used=False,new_packet_decodes=0,source_bindings=sources,outputs={str(log):real_sha(log)},log=str(log)))
        accessed=set()
        def key(p):return str(p).replace('\\','/')
        def mapped_read(p):
            k=key(p)
            if k.startswith('/home/'):
                self.assertIn(k,lookup,'Unexpected metadata read must be downloaded, never invented');accessed.add(k)
                return real_read(lookup[k])
            return real_read(p)
        def mapped_sha(p):
            k=key(p)
            if k in lookup:return real_sha(lookup[k])
            if k in opaque:return opaque[k]
            if real_path(p).is_file():return real_sha(p)
            self.assertFalse(k.startswith('/home/'),'Unbound/unavailable non-source hash '+k)
            return real_sha(p)
        class RemotePath(PurePosixPath):
            def absolute(self):return self
            def exists(self):return str(self) in lookup
            def read_text(self,*a,**kw):return Path(lookup[str(self)]).read_text(*a,**kw)
        def paths(p):return RemotePath(key(p)) if key(p).startswith('/home/') else real_path(p)
        with patch.object(d,'read',mapped_read),patch.object(d,'sha',mapped_sha),patch.object(d,'Path',paths),patch.object(d,'gone'),patch.object(d.u,'verify',side_effect=lambda pins:[self.assertEqual(mapped_sha(p),h) for p,h in pins.items()]):
            target=str(self.root/'prepared_request.json');result=d.build_request(mp,target)
            req=real_read(target);ctx=d.inspect_inputs(req);d.qualification(req,ctx['bound'])
        self.assertEqual((result['source_count'],result['frame_count'],result['maximum_new_packet_calls']),(1000,90000,180000))
        self.assertEqual(ctx['legacy']['normal']['budget_after'],ctx['legacy']['science']['budget'])
        self.assertNotIn('budget_after',ctx['legacy']['science'])
        self.assertEqual(len(ctx['records']),1000);self.assertEqual(len(ctx['plan']['schedule']),30)
        self.assertEqual(req['source_bindings'],sources)
        for tail in ('merge/exit.json','worker_0/exit.json','worker_1/exit.json'):
            self.assertTrue(any(x.endswith(tail) for x in accessed))
        self.assertIn(materials['prescreen']['owner_exit'],accessed)
        self.assertIn(ctx['cfg']['asset_manifest'],accessed)
        self.assertFalse(any(x.endswith(('.npz','.png','.jpg')) for x in accessed))

    def test_tokens_only_never_decompress_pixels(self):
        tokens=np.arange(680,dtype=np.int64);archive=self.root/'source.npz';archive.write_bytes(b'bound NPZ fixture')
        h=__import__('hashlib').sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()
        cp=put(self.root/'source.json',dict(source_index=0,source_id='a',tokens_sha256=h,archive=str(archive),outputs={str(archive):d.sha(archive)}))
        record=dict(source_index=0,source_id='a',tokens_sha256=h,archive=str(archive),checkpoint=cp)
        class Npz:
            files=['tokens','pixels']
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def __getitem__(self,key):
                if key!='tokens':raise AssertionError('Pixels must not be read')
                return tokens
        with patch.object(d.np,'load',return_value=Npz()):
            rec,scales=d.token_source(dict(records=[record],bound={cp:d.sha(cp),str(archive):d.sha(archive)}),0)
        self.assertEqual([len(x) for x in scales],[1,4,9,16,25,36,64,100,169,256])
        self.assertTrue(np.array_equal(np.concatenate(scales),tokens))

    def test_original_sqlite_without_request_is_readonly(self):
        import sqlite3
        p=self.root/'legacy.sqlite'
        with d.closing(sqlite3.connect(p)) as db:
            db.execute('CREATE TABLE events(event_id TEXT PRIMARY KEY,request_sha TEXT,result TEXT,status TEXT)');db.execute('INSERT INTO events VALUES(?,?,?,?)',('a','b','{}','COMPLETE'));db.commit()
        before=p.read_bytes();rows=d.paid_rows(p,['a']);self.assertNotIn('request',rows['a']);self.assertEqual(p.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'Missing'):d.paid_rows(p,['missing'])

    def test_execute_point_reuses_three_exact_old_frames_without_decode(self):
        case=fixture.CheckpointTests('test_exact_original_counter_and_float64_noise');case.setUp();self.addCleanup(case.doCleanups)
        rows={};events={};data=[]
        for seed in c.NOISE:
            row,expected,y,ev=case.frame(seed);rows[(row['candidate_id'],row['snr_db'],seed)]=row;events.update(ev);data.append(row)
        tp=put(self.root/'oldtrace.json',data);normal=put(self.root/'normal.json',{});cp=put(self.root/'source.json',{})
        record=dict(case.record,checkpoint=cp);cfg=dict(out=str(self.root/'out'),ledger='new');ctx=dict(cfg=cfg,records=[record],bound={cp:d.sha(cp)},plan=case.plan,regsha='a'*64,legacy=dict(cfg={'ledger':'old'},receipts=[normal]))
        adapter=types.SimpleNamespace(backend=None,legacy=None,header=None)
        raw=types.SimpleNamespace(serialize_raw=lambda scales,p:np.zeros(12,np.uint8),present_actual=fixture.present)
        tx=types.SimpleNamespace(transmit_frame=lambda cat,*args:(np.ones((1024,2),np.float64),data[list(c.NOISE).index(4101+(args[-1]%3))]['transmission']))
        runtime=dict(cat=case.current,legacy=case.old,old=raw,adapter=adapter,rx=tx,receiver=Mock())
        with patch.object(d,'check_live'),patch.object(d,'paid_rows',return_value=events):
            pin=d.execute_point(ctx,{'worker_index':0},runtime,record,[],case.point,rows,tp,{tp:d.sha(tp),normal:d.sha(normal)})
        got=c.read_point(pin,case.plan,7,case.point,'a'*64)
        self.assertEqual((got['newly_charged_packet_calls'],got['reused_frames']),(0,3));runtime['receiver'].receive.assert_not_called()

    def test_execute_point_new_paid_rows_and_three_noise_checkpoint(self):
        case=fixture.CheckpointTests('test_exact_original_counter_and_float64_noise');case.setUp();self.addCleanup(case.doCleanups)
        rows={}; events={}
        for seed in c.NOISE:
            row,expected,y,ev=case.frame(seed,current=True);rows[seed]=row;events.update(ev)
        cp=put(self.root/'source.json',{});ap=put(self.root/'execution/worker_0/admission.json',{})
        ctx=dict(cfg=dict(out=str(self.root/'out'),execution_dir=str(self.root/'execution'),ledger='new'),bound={cp:d.sha(cp)},plan=case.plan,regsha='e'*64,legacy=dict(cfg={'ledger':'old'},receipts=[]))
        record=dict(case.record,checkpoint=cp)
        class RX:
            def receive(self,y,snr,counter,**kwargs):return copy.deepcopy(rows[4101+counter%3])
        tx=types.SimpleNamespace(transmit_frame=lambda cat,*args:(np.ones((1024,2),np.float64),rows[4101+args[-1]%3]['transmission']))
        runtime=dict(cat=case.current,legacy=case.old,old=types.SimpleNamespace(serialize_raw=lambda *a:np.zeros(12,np.uint8),present_actual=fixture.present),adapter=types.SimpleNamespace(backend=None,legacy=None,header=None),rx=tx,receiver=RX())
        with patch.object(d,'check_live'),patch.object(d,'paid_rows',return_value=events):
            pin=d.execute_point(ctx,dict(worker_index=0),runtime,record,[],case.point,{},'',{})
        got=c.read_point(pin,case.plan,7,case.point,'e'*64);self.assertEqual((got['newly_charged_packet_calls'],got['reused_frames']),(6,0));self.assertEqual(len(list((self.root/'out/raw_traces').rglob('*.json'))),3)

    def test_capture_failure_retains_spawned_child_for_drain(self):
        cfg=dict(execution_dir=str(self.root),python='python',_path='config',resources=dict(affinities=[[1,2],[3,4]]),pythonpath=[]);children=[];child=Mock(pid=44)
        with patch.object(d.subprocess,'Popen',return_value=child),patch.object(d.u,'capture',side_effect=RuntimeError('identity')):
            with self.assertRaisesRegex(RuntimeError,'identity'):d.spawn(dict(cfg=cfg,regsha='r'),None,children)
        self.assertEqual(len(children),1);self.assertIs(children[0][0],child);children[0][1].close()

    def test_owner_marker_failure_still_drains_inside_lock(self):
        e=self.root/'e';e.mkdir();out=self.root/'out';cfg=dict(execution_dir=str(e),out=str(out),python=str(Path(__import__('sys').executable).absolute()),_path='config',resources=dict(affinities=[[1,2],[3,4]]))
        ctx=dict(cfg=cfg,regsha='r',reg=dict(budget_before={'n':0}));state={'locked':False};calls=[]
        @contextmanager
        def lock(cfg):
            state['locked']=True
            try:yield
            finally:state['locked']=False;calls.append('unlock')
        def spawn(ctx,index,children):children.append(['tracked']);raise RuntimeError('spawn after child')
        def drain(children):self.assertTrue(state['locked']);self.assertEqual(children,[['tracked']]);calls.append('drain');return {}
        with patch.object(d,'load_registered',return_value=ctx),patch.object(d.sys,'platform','linux'),patch.object(d.u,'check_stop'),patch.object(d.u,'set_resources'),patch.object(d.u,'owner_lock',lock),patch.object(d.u,'process',return_value=ident()),patch.object(d,'budget_snapshot',return_value={'n':0}),patch.object(d,'spawn',spawn),patch.object(d.u,'request_stop',side_effect=OSError('disk')),patch.object(d.u,'drain_children',drain):
            with self.assertRaisesRegex(RuntimeError,'spawn'):d.run('config')
        self.assertEqual(calls,['drain','unlock']);self.assertIn('STOP_WRITE_FAILED',d.read(e/'failure.json')['traceback'])

    def test_live_admission_identity_argv_and_merge_index(self):
        cfg=dict(execution_dir=str(self.root),python='python',_path='cfg');me=ident(argv=d.command(cfg,None));owner=ident(50)
        path=d.proof_path(cfg,None);put(path,dict(registration_sha256='r',config_sha256='h',worker_index=None,worker_identity=me,owner_identity=owner))
        with patch.object(d,'sha',return_value='h'),patch.object(d,'check_live') as live:
            got=d.admit_worker(dict(cfg=cfg,regsha='r'),None);self.assertEqual(got['worker_identity'],me);live.assert_called_once()
            x=d.read(path);x['worker_identity']['argv']=['wrong'];put(path,x)
            with self.assertRaisesRegex(ValueError,'Foreign'):d.admit_worker(dict(cfg=cfg,regsha='r'),None)

    def test_no_fallback_after_bad_reuse_evidence(self):
        case=fixture.CheckpointTests('test_exact_original_counter_and_float64_noise');case.setUp();self.addCleanup(case.doCleanups)
        row,exp,y,events=case.frame();row['source_id']='corrupt';tp=put(self.root/'old.json',[row]);normal=put(self.root/'normal.json',{});cp=put(self.root/'source.json',{})
        runtime=dict(cat=case.current,legacy=case.old,old=types.SimpleNamespace(serialize_raw=lambda *a:np.zeros(12,np.uint8),present_actual=fixture.present),adapter=types.SimpleNamespace(backend=None,legacy=None,header=None),rx=types.SimpleNamespace(transmit_frame=lambda *a:(np.ones((1024,2),np.float64),row['transmission'])),receiver=Mock())
        ctx=dict(cfg={'out':str(self.root/'out')},bound={cp:d.sha(cp)},plan=case.plan,regsha='a'*64,legacy=dict(cfg={'ledger':'old'},receipts=[normal]))
        with patch.object(d,'check_live'),patch.object(d,'paid_rows',return_value=events):
            with self.assertRaisesRegex(ValueError,'identity'):d.execute_point(ctx,{},runtime,dict(case.record,checkpoint=cp),[],case.point,{(row['candidate_id'],13,4101):row},tp,{tp:d.sha(tp),normal:d.sha(normal)})
        runtime['receiver'].receive.assert_not_called()

    def test_qualification_real_consumer_covers_all_sources_and_no_skips(self):
        paths=[str(Path(d.__file__).absolute().with_name(n+'.py')) for n in d.TEST_MODULES]+[str(Path(d.__file__).absolute())]
        sources={p:d.sha(p) for p in paths};log=self.root/'tests.log';log.write_text(f'Ran {d.TEST_COUNT} tests in 1s\n\nOK\n')
        qp=put(self.root/'qualification.json',dict(status=d.QSTATUS,tests_run=d.TEST_COUNT,test_modules=d.TEST_MODULES,python='python',process_waited=True,exit_code=0,GPU_used=False,new_packet_decodes=0,source_bindings=sources,log=str(log),outputs={str(log):d.sha(log)}))
        req=dict(prepared_qualification=qp,python='python',source_bindings=sources)
        self.assertEqual(d.qualification(req,{qp:d.sha(qp)})['tests_run'],d.TEST_COUNT)
        q=d.read(qp);q['source_bindings'].pop(paths[0]);put(Path(qp),q)
        with self.assertRaisesRegex(ValueError,'omitted'):d.qualification(req,{qp:d.sha(qp)})

    def test_hash_cache_revalidates_changed_file(self):
        p=self.root/'changed';p.write_bytes(b'one');one=c.file_sha(p);self.assertEqual(c.file_sha(p),one);p.write_bytes(b'two');self.assertNotEqual(c.file_sha(p),one)

    def test_owner_waits_workers_before_merge_and_seals_normal(self):
        e=self.root/'e';e.mkdir();out=self.root/'out';cfg=dict(execution_dir=str(e),out=str(out),python=str(Path(__import__('sys').executable).absolute()),_path=str(e/'config.json'),resources=dict(affinities=[[1,2],[3,4]]));put(e/'config.json',cfg)
        ctx=dict(cfg=cfg,regsha='r',reg=dict(budget_before={'charged':2}));order=[]
        @contextmanager
        def lock(cfg):yield
        def spawn(ctx,index,children):
            order.append(('spawn',index));folder=e/('merge' if index is None else 'worker_'+str(index));folder.mkdir();log=folder/'worker.log';handle=log.open('wb');who=ident(3 if index is None else index+1)
            put(folder/'admission.json',{});child=Mock();child.poll.return_value=0
            def wait():order.append(('wait',index));return 0
            child.wait.side_effect=wait;row=[child,handle,log,who,folder/'exit.json'];children.append(row)
            if index is None:
                put(out/'completion.json',dict(status=d.DONE,registration_sha256='r',worker_identity=who,budget_after={'charged':8},frame_count=90000,new_packet_calls=6,reused_frames=0))
        with patch.object(d,'load_registered',return_value=ctx),patch.object(d.sys,'platform','linux'),patch.object(d.u,'check_stop'),patch.object(d.u,'set_resources'),patch.object(d.u,'owner_lock',lock),patch.object(d.u,'process',return_value=ident(50)),patch.object(d,'budget_snapshot',side_effect=[{'charged':2},{'charged':8}]),patch.object(d,'spawn',spawn):
            result=d.run(cfg['_path'])
        self.assertEqual(order,[('spawn',0),('spawn',1),('wait',0),('wait',1),('spawn',None),('wait',None)])
        self.assertEqual(result['status'],d.NORMAL);self.assertFalse(result['automatic_successor_started']);d.u.verify(result['outputs'])

    def test_closed_calibration_real_receipt_consumer_and_current_budget(self):
        e=self.root/'e';out=self.root/'out';e.mkdir();out.mkdir();cfg=dict(execution_dir=str(e),out=str(out),python='python',_path=str(e/'config.json'),registration=str(e/'reg.json'))
        put(e/'config.json',cfg);rp=put(e/'reg.json',{});regsha=d.sha(rp);owner=ident(50,argv=['python','-B',str(Path(d.__file__).absolute()),'--config',cfg['_path'],'--stage','run']);child=ident(51,d.command(cfg,None));before={'charged':2};after={'charged':8}
        attempt=put(e/'attempt.json',dict(identity=owner,registration_sha256=regsha));stage=e/'merge';stage.mkdir();log=stage/'worker.log';log.write_text('normal merge')
        admission=put(stage/'admission.json',dict(worker_identity=child));end=put(stage/'exit.json',dict(process_waited=True,exit_code=0,identity=child,log=str(log),log_sha256=d.sha(log)))
        source={'source':'h'};inputs={'input':'j'};audit=dict(points=[{'point':'fixture'}],outputs={},frame_count=90000,new_packet_calls=6,reused_frames=89997,budget=after)
        science=put(out/'completion.json',dict(status=d.DONE,registration_sha256=regsha,config_sha256=d.sha(cfg['_path']),worker_identity=child,points=audit['points'],outputs={},source_bindings=source,input_bindings=inputs,GPU_used=False,quality_ranked=False,development_used=False,holdout_used=False,automatic_successor_started=False,frame_count=90000,new_packet_calls=6,reused_frames=89997,budget_before=before,budget_after=after))
        normal=put(e/'completion.json',dict(status=d.NORMAL,registration_sha256=regsha,config_sha256=d.sha(cfg['_path']),owner_identity=owner,all_waited=True,worker_exit_codes=[0,0],merge_exit_code=0,automatic_successor_started=False,outputs={p:d.sha(p) for p in (science,attempt,end,admission,str(log))},frame_count=90000,new_packet_calls=6,reused_frames=89997,budget_before=before,budget_after=after))
        ol=self.root/'owner.log';ol.write_text('normal owner');oe=put(self.root/'owner_exit.json',dict(process_waited=True,exit_code=0,identity=owner,log=str(ol),log_sha256=d.sha(ol)))
        spec=dict(config=cfg['_path'],registration=rp,completion=normal,owner_exit=oe,bindings={p:d.sha(p) for p in (cfg['_path'],rp,normal,oe,str(ol))})
        ctx=dict(cfg=cfg,regsha=regsha,reg=dict(source_bindings=source,input_bindings=inputs,budget_before=before),plan={'frozen':True},records=['original'],bound={})
        with patch.object(d,'load_registered',return_value=ctx),patch.object(d,'audit_inventory',return_value=audit),patch.object(d,'gone'):
            got=d.closed_calibration(spec);self.assertEqual(got['budget_before'],after);self.assertEqual(got['done']['budget_before'],before);self.assertEqual(got['bindings'][rp],regsha)
            x=d.read(oe);x['exit_code']=1;put(Path(oe),x);spec['bindings'][oe]=d.sha(oe)
            with self.assertRaisesRegex(ValueError,'wait0'):d.closed_calibration(spec)

    def test_real_sqlite_writer_closed_workers_and_inventory_tamper(self):
        # Full source partition, one synthetic point to bound fixture size.
        # Production load still obtains all30 points from strict make_plan.
        br,bp,ledger=bf.fixture(self.root/'budget',actual_cap=3000);gate=d.budget.CalibrationBudget.install(bp)
        pool=fixture.profiles();plan=c.make_plan(pool,fixture.shortlist(pool),[f's{i}' for i in range(1000)],science_sha256='a'*64,shortlist_sha256='b'*64,source_manifest_sha256='c'*64)
        point=plan['schedule'][0];plan['schedule']=[point];plan['frame_count']=3000
        e=self.root/'execution';out=self.root/'out';e.mkdir();out.mkdir();cfg=dict(execution_dir=str(e),out=str(out),_path=str(e/'config.json'),python='python',original_receiver_module='purefixture');put(e/'config.json',cfg)
        owner=ident(900);put(e/'attempt.json',dict(registration_sha256='r'*64,identity=owner));records=[dict(source_id=f's{i}',tokens_sha256='a'*64) for i in range(1000)];worker_outputs=[{},{}];who=[ident(901,d.command(cfg,0)),ident(902,d.command(cfg,1))]
        admissions=[]
        for index in (0,1):
            folder=e/f'worker_{index}';folder.mkdir();log=folder/'worker.log';log.write_text('normally waited')
            admissions.append(put(folder/'admission.json',dict(registration_sha256='r'*64,config_sha256=d.sha(cfg['_path']),worker_identity=who[index],owner_identity=owner,worker_index=index)))
            put(folder/'exit.json',dict(process_waited=True,exit_code=0,identity=who[index],log=str(log),log_sha256=d.sha(log)))
        for i in range(1000):
            base=[];proof=[]
            for seed in c.NOISE:
                header=dict(header_crc_ok=False,header_fields_legal=False,header_ok=False,profile_id=None,score=0.)
                row=dict(status='MAIN_RAW64_KEEP_RECEPTION_COMPLETE',source_index=i,source_id=f's{i}',source_tokens_sha256='a'*64,population_role='calibration',noise_seed=seed,public_frame_counter=c.frame_counter(i,point['snr_db'],seed),received_sha256='f'*64,codebook_sha256=plan['catalogue_digest'],aliases_sha256='c'*64,header=header,body=None,logical_packet_calls=1,**point,**fixture.present(header,None,None))
                request=dict(schema='MAIN_RAW64_ACTUAL_RX_REQUEST_V1',received_sha256=row['received_sha256'],public_frame_counter=row['public_frame_counter'],snr_db=float(point['snr_db']),N=1024,body_session='actual-body',body_group=0,component='header',codebook_sha256=row['codebook_sha256'],aliases_sha256=row['aliases_sha256'])
                eid=f'MAIN_CALIBRATION_V1/source{i}/noise{seed}.header';ev=dict(phase='actual_calibration',kind='header',phy_key='header',event_id=eid)
                gate.decode_once(ev,request,lambda:header,who[i%2]);item=dict(event_id=eid,kind='header',phy_key='header',request_sha256=c.digest(request),result_sha256=c.digest(header));row.update(packet_events=[item],packet_event_ids=[eid]);base.append(row)
            tp=put(out/'raw'/f'{i:04d}.json',base);ap=admissions[i%2]
            for row in base:
                v=copy.deepcopy(row);v.update(effective_catalogue_digest=plan['catalogue_digest'],effective_aliases_digest='c'*64)
                v['execution_provenance']=dict(status='MAIN_RAW64_EXACT_RX_PROVENANCE_V1',origin='NEW_PAID_RECEIVE',receiver_interpretation_unchanged=False,newly_charged_packet_calls=1,original_row_sha256=c.digest(row),original_trace={'path':tp,'sha256':d.sha(tp)},normal_receipts=[],admission_receipts=[ap],input_bindings={tp:d.sha(tp),ap:d.sha(ap)})
                proof.append(v)
            cp=c.write_point(out,plan,i,point,proof,'r'*64);worker_outputs[i%2][cp['path']]=cp['sha256']
        for index in (0,1):
            put(out/f'worker_{index}/completion.json',dict(status=d.WORKER_DONE,registration_sha256='r'*64,config_sha256=d.sha(cfg['_path']),worker_identity=who[index],worker_index=index,source_indices=list(range(index,1000,2)),frame_count=1500,new_packet_calls=1500,reused_frames=0,outputs=worker_outputs[index]))
        ctx=dict(cfg=cfg,regsha='r'*64,gate=gate,plan=plan,records=records,profiles=pool,bound={},legacy=dict(cfg={'ledger':'unused'},receipts=[]))
        with patch.object(d,'gone'),patch.object(d,'module',return_value=types.SimpleNamespace(present_actual=fixture.present)):
            got=d.audit_inventory(ctx);self.assertEqual((len(got['points']),got['new_packet_calls'],got['reused_frames']),(1000,3000,0));self.assertEqual(got['budget']['phase_charged']['qualification'],1)
            with gate.ledger.connect() as db:db.execute("UPDATE events SET request_sha='changed' WHERE event_id=(SELECT event_id FROM events WHERE phase='actual_calibration' LIMIT 1)")
            with self.assertRaisesRegex(ValueError,'request/result'):d.audit_inventory(ctx)


if __name__=='__main__':unittest.main()
