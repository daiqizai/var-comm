"""Prepared CPU driver tests: temporary assets, SQLite and scripted PHY only."""
import copy
import contextlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'phy_codec',HERE.parent/'payload'):sys.path.insert(0,str(p))
import h_full_payload_driver as d
import h_full_payload_cpu as core
from test_h_full_payload_cpu import fixture,seal,FullLedger,FakeBackend,FakeHeader
from h64_catalog import SIZES,token_count


def dump(p,v):
    p.write_text(json.dumps(v),encoding='utf-8');return str(p)


def completed_fixture():
    ids=[f's_{i:04d}' for i in range(1000)]
    arows=[dict(source_index=i,source_id=s,checkpoint=f'/ASSET/{i}.json',checkpoint_sha256='a'*64) for i,s in enumerate(ids)]
    crows=[dict(source_index=i,source_id=s,checkpoint=f'/CODEC/{i}.json',sha256='b'*64) for i,s in enumerate(ids)]
    assets=dict(status='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',source_count=1000,
                outputs={r['checkpoint']:r['checkpoint_sha256'] for r in arows})
    manifest=dict(status=assets['status'],source_ids=ids,records=arows)
    source=dict(status='H_FULL1000_SOURCE_CODEC_COMPLETE',source_count=1000,source1000_codec_complete=True,
                pending_source_encoding_count=0,reused_codec_sources=200,newly_encoded_sources=800,
                development_used=False,holdout_used=False,new_packet_decodes=0,source_ids=ids,records=crows,
                outputs={r['checkpoint']:r['sha256'] for r in crows})
    return source,assets,manifest


class AssetTests(unittest.TestCase):
    def test_incomplete_800_cannot_be_used_as_full_codec(self):
        s,a,m=completed_fixture()
        self.assertEqual(len(d.completed_sources(s,a,m)),1000)
        for bad in (dict(s,status=a['status']),dict(s,pending_source_encoding_count=800),
                    dict(s,source1000_codec_complete=False),dict(s,newly_encoded_sources=0)):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):d.completed_sources(bad,a,m)

    def test_source_order_and_unbound_checkpoint_rejected(self):
        s,a,m=completed_fixture();s['source_ids']=list(reversed(s['source_ids']))
        with self.assertRaisesRegex(RuntimeError,'order'):d.completed_sources(s,a,m)
        s,a,m=completed_fixture();s['records'][0]['sha256']='c'*64
        with self.assertRaisesRegex(RuntimeError,'completed scope'):d.completed_sources(s,a,m)

    def fixture_asset(self,p):
        tokens=np.arange(680,dtype=np.int64)%4096;raw=p/'asset.npz'
        np.savez(raw,tokens=tokens,unused_pixels=np.asarray([object()],dtype=object))
        ap=p/'asset.json';acp=dict(source_index=999,source_id='full_0999',tokens_sha256=d.token_sha(tokens),
             archive=str(raw),outputs={str(raw):d.sha(raw)})
        dump(ap,acp)
        bits={f'm{m}_bits':np.zeros(300+m,dtype=np.uint8) for m in (6,7,8,9)}
        bp=p/'codec.npz';np.savez(bp,**bits)
        cp=p/'codec.json';c=dict(source_index=999,source_id='full_0999',registration_sha256='b'*64,
            independent_roundtrip=True,tokens_sha256=d.token_sha(tokens),source_assets_checkpoint={'path':str(ap),'sha256':d.sha(ap)},
            lengths=[dict(m=m,raw_bits=12*token_count(m),arithmetic_bits=300+m,zero_extension_reads=30) for m in (6,7,8,9)],
            outputs={str(bp):d.sha(bp)})
        dump(cp,c)
        ar=dict(source_index=999,source_id='full_0999',checkpoint=str(ap),checkpoint_sha256=d.sha(ap),
                archive=str(raw),tokens_sha256=d.token_sha(tokens))
        cr=dict(source_index=999,source_id='full_0999',checkpoint=str(cp),sha256=d.sha(cp))
        ctx=dict(cfg={},core=core,context={'source_ids':[f'full_{i:04d}' for i in range(1000)]},
             source=dict(registration_sha256='b'*64,records=[None]*999+[cr],outputs={str(bp):d.sha(bp)}),
             manifest={'records':[None]*999+[ar]})
        return ctx,cp,c

    def test_actual_npz_source_schema_hash_and_original1000_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx,cp,c=self.fixture_asset(Path(tmp));result=d.load_source(ctx,999)
            self.assertEqual(sum(len(x) for x in result['scales']),680)
            self.assertEqual(set(result['arithmetic_bits']),{6,7,8,9})
            self.assertEqual(len(result['source_bindings']),4)
            # The old200-only loader would reject this valid original index999.
            c['source_index']=199;dump(cp,c);ctx['source']['records'][999]['sha256']=d.sha(cp)
            with self.assertRaisesRegex(RuntimeError,'linked'):d.load_source(ctx,999)

    def test_source_bits_or_token_link_changed_is_not_silently_raw_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx,cp,c=self.fixture_asset(Path(tmp))
            c['tokens_sha256']='0'*64;dump(cp,c);ctx['source']['records'][999]['sha256']=d.sha(cp)
            with self.assertRaisesRegex(RuntimeError,'linked'):d.load_source(ctx,999)
        with tempfile.TemporaryDirectory() as tmp:
            ctx,cp,c=self.fixture_asset(Path(tmp))
            c['lengths'][0]['arithmetic_bits']+=1;dump(cp,c);ctx['source']['records'][999]['sha256']=d.sha(cp)
            with self.assertRaisesRegex(RuntimeError,'bit count'):d.load_source(ctx,999)


class TraceLedgerTests(unittest.TestCase):
    def make_traces(self,p,phase,mixed=False):
        args=fixture();reg=p/'reg.json';dump(reg,{'fake_test_only':True})
        contract=seal(args);contract['execution_registration_sha256']=d.sha(reg)
        context=core.prepare(**args,contract=contract)
        ctx=dict(cfg={'registration':str(reg),'phase':phase},context=context,core=core)
        ledger=FullLedger();paths=[]
        scales=[(np.arange(s*s,dtype=np.int64)+43*i)%4096 for i,s in enumerate(SIZES)]
        streams={m:np.zeros(12*token_count(m)+2,dtype=np.uint8) for m in (6,7,8,9)}
        for i in (0,1):
            frames=[]
            for entry in context['schedule']:
                if entry['phase']!=phase:continue
                for seed in core.SEEDS:
                    backend,header=FakeBackend(),FakeHeader()
                    first_slot=0 if phase=='whole_calibration' else 8
                    if mixed and i==0 and entry['full_slot']==first_slot:
                        if seed==6102:header.reject=True
                        if seed==6103:
                            rp=next(p for p in context['catalogue']['profiles'] if p['q']==6 and p['m']==6
                                    and p['K']==0 and p['mode']=='arithmetic' and p['nominal_rate']=='1/2')
                            header.force_id=rp['profile_id']
                            backend.decoded_override=core.phy.pack_body(np.zeros(20,dtype=np.uint8),rp)
                    frames.append(core.run_frame(ctx=context,full_slot=entry['full_slot'],source_id=f'full_{i:04d}',
                        source_index=i,noise_seed=seed,scales=scales,arithmetic_bits=streams,
                        backend=backend,header=header,ledger=ledger))
            sp=p/f'{i}.json';dump(sp,dict(status='H_FULL_CPU_SOURCE_TRACES',phase=phase,
                registration_sha256=d.sha(reg),source_id=f'full_{i:04d}',source_index=i,frames=frames,source_bindings={}))
            paths.append(sp)
        db=p/'ledger.sqlite'
        with contextlib.closing(sqlite3.connect(db)) as conn:
            conn.execute('CREATE TABLE events(event_id TEXT,phase TEXT,kind TEXT,status TEXT,result TEXT,result_sha TEXT)')
            conn.execute('CREATE TABLE counters(phase TEXT,charged INTEGER)')
            for ev in ledger.events:
                result=ledger.results[ev['event_id']][1]
                conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(ev['event_id'],phase,ev['kind'],'COMPLETE',d.canonical(result),d.digest(result)))
            conn.execute('INSERT INTO counters VALUES(?,?)',(phase,len(ledger.events)))
            conn.commit()
        return ctx,paths,db

    def test_real_core_fake_backend_trace_to_sqlite_audit_roundtrip_both_phases(self):
        for phase,n in (('whole_calibration',48),('partial_calibration',36)):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
                ctx,paths,db=self.make_traces(Path(tmp),phase)
                events,summary=d.trace_inventory(ctx,paths)
                self.assertEqual(summary['frame_count'],n)
                result=d.audit_ledger(db,events,phase)
                self.assertEqual(result['paid_events'],n*2)

    def test_missing_frame_extra_source_and_result_forgery_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
            ctx,paths,db=self.make_traces(Path(tmp),'partial_calibration')
            source=d.read(paths[0]);source['frames'].pop();dump(paths[0],source)
            with self.assertRaisesRegex(RuntimeError,'Missing source frames'):d.trace_inventory(ctx,paths)
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
            ctx,paths,db=self.make_traces(Path(tmp),'whole_calibration')
            events,_=d.trace_inventory(ctx,paths)
            with contextlib.closing(sqlite3.connect(db)) as conn:
                conn.execute("UPDATE events SET result='{}' WHERE kind='body'");conn.commit()
            with self.assertRaisesRegex(RuntimeError,'Paid result'):d.audit_ledger(db,events,'whole_calibration')

    def test_pending_canonical_cannot_be_claimed_as_gray_or_success(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
            ctx,paths,db=self.make_traces(Path(tmp),'partial_calibration')
            source=d.read(paths[0]);source['frames'][0]['rx_source_status']='ARITHMETIC_SOURCE_DECODED'
            dump(paths[0],source)
            with self.assertRaisesRegex(RuntimeError,'canonical status'):d.trace_inventory(ctx,paths)

    def test_actual_mixed_raw_gray_and_pending_canonical_have_distinct_counts(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
            ctx,paths,db=self.make_traces(Path(tmp),'partial_calibration',mixed=True)
            events,summary=d.trace_inventory(ctx,paths)
            self.assertEqual(summary['wire_gray_frames'],1)
            self.assertEqual(summary['arithmetic_pending_frames'],1)
            self.assertEqual(summary['frame_count'],36)
            self.assertEqual(summary['logical_packet_events'],71)
            self.assertEqual(d.audit_ledger(db,events,'partial_calibration')['paid_events'],71)

    def test_paid_reserved_and_counter_mismatch_cannot_pass(self):
        for change in ("UPDATE events SET status='RESERVED' WHERE kind='body'","UPDATE counters SET charged=charged+1"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp,patch.object(d,'COUNT',2):
                ctx,paths,db=self.make_traces(Path(tmp),'partial_calibration');events,_=d.trace_inventory(ctx,paths)
                with contextlib.closing(sqlite3.connect(db)) as conn:conn.execute(change);conn.commit()
                with self.assertRaises(RuntimeError):d.audit_ledger(db,events,'partial_calibration')

    def test_worker_partition_and_full1000_cardinality(self):
        self.assertEqual(len(d.indices(0)),500);self.assertEqual(len(d.indices(1)),500)
        self.assertEqual(set(d.indices(0))|set(d.indices(1)),set(range(1000)))
        self.assertFalse(set(d.indices(0))&set(d.indices(1)))
        with self.assertRaises(RuntimeError):d.indices(2)


class SchedulingTests(unittest.TestCase):
    def fixture_gate(self,p,phase):
        cfg=dict(phase=phase,owner_module=str(p/'old_owner.py'),wait_module=str(p/'old_wait.py'),predecessor_batches={})
        bound={};science={};ident={'pid':123,'uid':1002,'start_ticks':987}
        for kind in (['selection','source']+(['whole'] if phase=='partial_calibration' else [])):
            cp=p/(kind+'_config.json');rp=p/(kind+'_reg.json');lp=p/(kind+'_launch.json');done=p/(kind+'_science.json')
            owner=dict(registration=str(rp));registration={'TEST_ONLY':kind}
            argv=['/test/python','-B',cfg['owner_module'],'--config',str(cp)]
            dump(cp,owner);dump(rp,registration)
            dump(lp,dict(identity=dict(ident,argv=argv),argv=argv,registration_sha256=d.sha(rp),owner_config_sha256=d.sha(cp)))
            dump(done,{'TEST_ONLY':kind});science[str(done)]=d.sha(done)
            cfg['predecessor_batches'][kind]=dict(config=str(cp),registration=str(rp),launch=str(lp))
            cfg[{'selection':'selection_completion','source':'source_completion','whole':'whole_cpu_completion'}[kind]]=str(done)
            for x in (cp,rp,lp):bound[str(x)]=d.sha(x)
        selected=p/'selected.json';source=Path(cfg['source_completion']);dump(selected,{'TEST_ONLY':'winners'})
        cfg['selected']=str(selected)
        if phase=='partial_calibration':
            whole=Path(cfg['whole_cpu_completion']);dump(whole,dict(status='H_FULL1000_CPU_RECEIVE_COMPLETE',phase='whole_calibration',
                source_count=1000,frame_count=24000,selected_sha256=d.sha(selected),source_completion_sha256=d.sha(source),source_decode_complete=False))
            science[str(whole)]=d.sha(whole)
        a=SimpleNamespace(validate_config=lambda *a:None,raw_process_state=lambda p:None)
        w=SimpleNamespace(exited=lambda *a:True,verify_batch=lambda *a:dict(bindings=science))
        return cfg,bound,a,w

    def test_partial_requires_closed_whole_batch_and_exact_source_winners(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,bound,a,w=self.fixture_gate(Path(tmp),'partial_calibration')
            with patch.object(d,'module',side_effect=[a,w]):d.verify_predecessors(cfg,{},bound)
            bad=copy.deepcopy(cfg);bad['predecessor_batches'].pop('whole')
            with patch.object(d,'module',side_effect=[a,w]),self.assertRaisesRegex(RuntimeError,'prerequisite'):
                d.verify_predecessors(bad,{},bound)
            whole=Path(cfg['whole_cpu_completion']);value=d.read(whole);value['frame_count']=18000;dump(whole,value)
            # The closure itself must also bind this current file; failing either gate is sufficient.
            with patch.object(d,'module',side_effect=[a,w]),self.assertRaises(RuntimeError):d.verify_predecessors(cfg,{},bound)

    def test_live_source_owner_blocks_before_any_physical_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,bound,a,w=self.fixture_gate(Path(tmp),'whole_calibration');w.exited=lambda *a:False
            with patch.object(d,'module',side_effect=[a,w]),self.assertRaisesRegex(RuntimeError,'still live'):
                d.verify_predecessors(cfg,{},bound)


if __name__=='__main__':unittest.main()
