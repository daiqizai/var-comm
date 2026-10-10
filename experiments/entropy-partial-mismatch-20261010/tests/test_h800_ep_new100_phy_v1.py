"""Finite-owner and portable241 proof tests; synthetic bytes only, no PHY."""
import contextlib
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_new100_phy_v1 as owner
import ep_t1_portable_qualification_v1 as q


def fixture(root):
    mapping=[];old=q.OLD_ROOT;output=old+'/outputs/test';source=old+'/src/source.py'
    def put(oldpath,value):
        path=root/str(len(mapping));path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        pin=q.io.pin(path);mapping.append(dict(original_path=oldpath,**pin));return pin
    sp=put(source,b'# synthetic source only\n');reference=put(old+'/legacy.json',{})
    backend={'synthetic_backend':True};catalogue=dict(catalogue_sha256='c'*64,profiles=[{} for _ in range(144)],layouts=[{} for _ in range(12)])
    request=dict(root=old,source_bindings={source:sp['sha256']},input_bindings={old+'/legacy.json':reference['sha256']},
        catalogue=catalogue,backend_identity=backend,header_ids=list(range(144))+[4095],packet_cap=241,body_decoder_calls=96,header_decoder_calls=145)
    request_pin=put(output+'/request.json',request)
    db_path=root/'ledger.sqlite';db=sqlite3.connect(db_path)
    db.execute('CREATE TABLE config(id INTEGER PRIMARY KEY, hash TEXT, cap INTEGER)');db.execute('INSERT INTO config VALUES(1,?,241)',(request_pin['sha256'],))
    db.execute('CREATE TABLE events(event_id TEXT PRIMARY KEY,event TEXT,request TEXT,result TEXT,status TEXT)')
    outputs={}
    for i in range(241):
        kind='header' if i<145 else 'body';eid=str(i);received=str(i);result={'synthetic_output':i}
        ev={'kind':kind};req=dict(received_sha256=received,catalogue_sha256='c'*64)
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(eid,json.dumps(ev),json.dumps(req),json.dumps(result),'COMPLETE'))
        value=dict(event_id=eid,received_sha256=received,correct=True,**result) if kind=='header' else dict(
            row=dict(event_id=eid,received_sha256=received,correct=True),rx=result)
        name=output+'/'+kind+'_cases/'+str(i)+'.json';outputs[name]=put(name,value)['sha256']
    db.commit();db.close();ledger=q.io.pin(db_path);mapping.append(dict(original_path=output+'/qualification_ledger.sqlite',**ledger))
    completed=dict(status='PASS',schema='WCL_T1_REAL_PHY_QUALIFICATION_20261009_V1',actual_constructor_validation=True,
        profile_count=144,actual_layout_count=12,actual_body_decodes=96,actual_header_decodes=145,packet_decode_count=241,
        ledger=dict(total=241,unresolved=0,cap=241),request_sha256=request_pin['sha256'],request_path=output+'/request.json',
        source_bindings=request['source_bindings'],input_bindings=request['input_bindings'],outputs=outputs,
        ledger_path=output+'/qualification_ledger.sqlite',catalogue_sha256='c'*64,backend_identity=backend)
    completion_pin=put(output+'/completion.json',completed)
    spec=dict(schema=q.SCHEMA,status='ORIGINAL_T1_241_BYTES_RELOCATED_SHA_CLOSED_NO_NEW_QUALIFICATION',original_actual_decodes=241,
        new_actual_decodes=0,new_qualification_calls=0,original_ledger_read_only=True,original_receipt_bytes_modified=False,
        qualified_runtime_admitted=False,original_root=old,actual_project_root=str(root),mapping=mapping,completion=completion_pin,
        request=request_pin,ledger=ledger,original_catalogue_sha256='c'*64,original_backend_identity=backend)
    desc=q.io.save(root/'spec.json',spec)
    return desc,spec,completed,request


class Portable241(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        real=q.io.bytes_checked
        self.patches=[patch.object(q.io,'inside',side_effect=lambda p,*a,**k:Path(p)),
            patch.object(q.io,'bytes_checked',side_effect=lambda d,**kw:real(d,base=self.root,**{k:v for k,v in kw.items() if k!='base'}))]
        for p in self.patches:p.start()
        self.desc,self.spec,self.done,self.request=fixture(self.root)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def repin_spec(self):
        path=Path(self.desc['path']);path.write_text(json.dumps(self.spec));self.desc=q.io.pin(path)
    def test_exact_original241_readonly_and_complete_outputs(self):
        before=q.io.sha(self.spec['ledger']['path']);s,d,r,m=q.inspect_binding(self.desc)
        self.assertEqual(d['packet_decode_count'],241);self.assertEqual(q.io.sha(s['ledger']['path']),before)
    def test_changed_source_blocks_before_constructor(self):
        Path(self.spec['mapping'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Relocated original'):q.inspect_binding(self.desc)
    def test_duplicate_old_mapping_blocks(self):
        self.spec['mapping'].append(self.spec['mapping'][0]);self.repin_spec()
        with self.assertRaisesRegex(ValueError,'Ambiguous'):q.inspect_binding(self.desc)
    def test_original_receipt_must_not_claim_current_admission(self):
        self.spec['qualified_runtime_admitted']=True;self.repin_spec()
        with self.assertRaisesRegex(ValueError,'no newly claimed'):q.inspect_binding(self.desc)
    def fake_runtime(self,bad=False):
        fixture=self
        class Runtime:
            def __init__(self,*a,**kw):
                self.qualified=False;self.backend=SimpleNamespace(identity={'wrong':True} if bad else fixture.done['backend_identity'])
            def use_catalogue(self,cat):self.catalogue=cat;self.profiles={str(i):{} for i in range(144)}
        path=self.root/'phy.py';path.write_bytes(b'# fake metadata runtime\n')
        m={x['original_path']:x for x in self.spec['mapping']}
        phy=SimpleNamespace(__file__=str(path),LEGACY_QUALIFICATION_SHA=self.spec['mapping'][1]['sha256'],Runtime=Runtime,
            collect_source_bindings=lambda root:{m[k]['path']:v for k,v in self.done['source_bindings'].items()})
        env=self.root/'env.json';env.write_text('{}');return phy,q.io.pin(env)
    def test_actual_constructor_fullcatalogue_identity_then_admit(self):
        phy,env=self.fake_runtime()
        with patch.object(q,'T1_SHA',q.io.sha(phy.__file__)):
            rt=q.create_runtime(self.desc,self.root,phy,env)
        self.assertTrue(rt.qualified);self.assertEqual(rt.qualification['new_qualification_calls'],0)
        self.assertEqual(rt.qualification['current_CPU_environment'],env)
    def test_different_actual_backend_does_not_admit(self):
        phy,env=self.fake_runtime(bad=True)
        with patch.object(q,'T1_SHA',q.io.sha(phy.__file__)),self.assertRaisesRegex(ValueError,'Actual current backend'):
            q.create_runtime(self.desc,self.root,phy,env)
    def test_nonclosed_original_ledger_cannot_be_hidden_by_spec(self):
        path=Path(self.spec['ledger']['path']);db=sqlite3.connect(path);db.execute("UPDATE events SET status='RESERVED' WHERE event_id='0'");db.commit();db.close()
        self.spec['ledger']=q.io.pin(path)
        for row in self.spec['mapping']:
            if row['path']==str(path):row.update(self.spec['ledger'])
        self.repin_spec()
        with self.assertRaisesRegex(ValueError,'actual callbacks incomplete'):q.inspect_binding(self.desc)
    def test_ledger_outputs_must_match_not_only_count(self):
        row=next(x for x in self.spec['mapping'] if '/body_cases/' in x['original_path']);path=Path(row['path'])
        value=json.loads(path.read_bytes());value['rx']={'synthetic_output':'wrong'};path.write_text(json.dumps(value));row.update(q.io.pin(path))
        self.done['outputs'][row['original_path']]=row['sha256'];cp=Path(self.spec['completion']['path']);cp.write_text(json.dumps(self.done));self.spec['completion']=q.io.pin(cp)
        for r in self.spec['mapping']:
            if r['path']==str(cp):r.update(self.spec['completion'])
        self.repin_spec()
        with self.assertRaisesRegex(ValueError,'body qualification output'):q.inspect_binding(self.desc)


class BoundedOwner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.request=self.root/'request.json';self.request.write_text('{}')
        self.e=owner.execution(time.time()+1000,900,[2,3]);self.r=dict(execution=self.e,root=str(self.root),CPU_python=sys.executable,frames=[])
        self.shared=SimpleNamespace(owner_lock=lambda p:contextlib.nullcontext(),cpu_snapshot=lambda:dict(effective_cpu_cores=2,
            available_memory_bytes=3*(1<<30)),child_limits=lambda *a:None,wait_owned=lambda child,sec:dict(success=child.returncode==0))
        self.ps=[patch.object(owner,'registration',return_value=(self.r,{})),patch.object(owner.g,'inside',side_effect=Path),
            patch.object(owner.g,'helper',return_value=self.shared),patch.object(owner.sys,'platform','linux'),
            patch.object(owner.os,'sched_getaffinity',return_value={2,3},create=True),
            patch.object(owner.gate,'controlled_cpu_environment',return_value={})]
        for p in self.ps:p.start()
        self.a=SimpleNamespace(request=str(self.request),request_sha256='a'*64)
    def tearDown(self):
        for p in reversed(self.ps):p.stop()
        self.temp.cleanup()
    def child(self,code):return SimpleNamespace(pid=42,returncode=code,poll=lambda:code)
    def test_provider_failure_stops_successors_and_records_actual_wait(self):
        with patch.object(owner.subprocess,'Popen',return_value=self.child(1)) as launch:
            with self.assertRaisesRegex(RuntimeError,'Actual provider child failed'):owner.run(self.a)
        self.assertEqual(launch.call_count,1)
        failure=json.loads((self.root/'run/failure.json').read_bytes());self.assertEqual(failure['actual_waits'],[{'success':False}])
        self.assertFalse((self.root/'run/whole').exists());self.assertFalse((self.root/'run/completion.json').exists())
    def test_second_failure_preserves_first_group_and_never_starts_third(self):
        fake=dict(status=owner.PASS,request_sha256='a'*64,group='raw',CUDA_initialized=False,results={'logical_count':1800})
        with patch.object(owner.subprocess,'Popen',side_effect=[self.child(0),self.child(1)]) as launch, \
            patch.object(owner.core,'pin',return_value={}),patch.object(owner.source,'readpin',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'Actual provider child failed'):owner.run(self.a)
        self.assertEqual(launch.call_count,2);self.assertFalse((self.root/'run/partial').exists())
        self.assertEqual(len(json.loads((self.root/'run/failure.json').read_bytes())['actual_waits']),2)
    def test_existing_attempt_is_not_replayed(self):
        (self.root/'run').mkdir()
        with patch.object(owner.subprocess,'Popen') as launch,self.assertRaisesRegex(RuntimeError,'already attempted'):
            owner.run(self.a)
        launch.assert_not_called()
    def test_insufficient_resources_persists_snapshot_no_child(self):
        self.shared.cpu_snapshot=lambda:dict(effective_cpu_cores=1,available_memory_bytes=1<<30)
        with patch.object(owner.subprocess,'Popen') as launch,self.assertRaisesRegex(RuntimeError,'TwoCPU cores'):
            owner.run(self.a)
        launch.assert_not_called();self.assertTrue((self.root/'run/prelaunch_resources.json').is_file());self.assertTrue((self.root/'run/failure.json').is_file())
    def test_wrong_CPU_slot_count_and_over_budget_rejected(self):
        for slots in ([2],[2,2],[1,2,3]):
            with self.assertRaises(RuntimeError):owner.execution(time.time()+1000,900,slots)
        with self.assertRaises(RuntimeError):owner.execution(time.time()+1000,21601,[2,3])


class WorkerIdentity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.path=self.root/'registered/request.json'
        self.path.parent.mkdir();self.path.write_text('{}');self.digest=owner.g.sha(self.path)
        self.r=dict(CPU_python='/frozen/bin/python',execution=dict(CPU_slots=[2,3]));self.group='whole';self.out=self.path.parent/'run'/self.group;self.out.mkdir(parents=True)
        self.intent=dict(request_sha256=self.digest,owner_pid=111,execution=self.r['execution'],packet_cap=7200)
        self.launch=dict(pid=222,owner_pid=111,request_sha256=self.digest,CPU_slots=[2,3],argv=[self.r['CPU_python'],'-B','-u',
            str(Path(owner.__file__).absolute()),'_worker','--request',str(self.path),'--request-sha256',self.digest,
            '--owner-pid','111','--group',self.group])
        self.ps=[patch.object(owner.g,'inside',side_effect=Path),patch.object(owner.os,'getpid',return_value=222),patch.object(owner.os,'getppid',return_value=111)]
        for p in self.ps:p.start()
    def tearDown(self):
        for p in reversed(self.ps):p.stop()
        self.temp.cleanup()
    def check(self):
        (self.out.parent/'intent.json').write_text(json.dumps(self.intent));(self.out/'child_started.json').write_text(json.dumps(self.launch))
        return owner.worker_identity(self.path,self.digest,self.r,111,self.group)
    def test_exact_three_provider_layout_admitted(self):self.assertEqual(self.check(),self.launch)
    def test_wrong_group_argv_rejected(self):
        self.launch['argv'][-1]='partial'
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()
    def test_wrong_registered_request_argument_rejected(self):
        self.launch['argv'][6]=str(self.out.parent/'request.json')
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()
    def test_wrong_parent_intent_rejected(self):
        self.intent['owner_pid']=333
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()
    def test_wrong_actual_PID_rejected(self):
        self.launch['pid']=444
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()
    def test_changed_request_bytes_rejected(self):
        self.path.write_text('{"changed":true}')
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()
    def test_wrong_CPU_launch_receipt_rejected(self):
        self.launch['CPU_slots']=[0,1]
        with self.assertRaisesRegex(RuntimeError,'group/argv'):self.check()


if __name__=='__main__':unittest.main(verbosity=2)
