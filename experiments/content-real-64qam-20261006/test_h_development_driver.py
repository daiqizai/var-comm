"""Synthetic closure, source, bounded schedule and ledger tests; no real PHY."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import h_development_driver as d
import register_h_development_cpu as r
from test_h_development_cpu import fixture,seal,FakeBackend,FakeHeader,FakeLedger,core,SIZES,token_count,np


def write(p,value):d.atomic(p,value);return str(p)
def budget(n=0):
    counts=dict.fromkeys(d.PHASES,0);counts.update(qualification=1608,coarse=49152,initial_true200=19200,whole_calibration=48000,partial_calibration=36000,development=n)
    return dict(created=True,charged=sum(counts.values()),phase_charged=counts,unresolved=0,failed=0,development_remaining=13200-n)


class DriverTests(unittest.TestCase):
    def gate_fixture(self,root):
        owner=root/'owner';owner.mkdir();out=root/'source100';out.mkdir();entry=root/'stage_owner.py';entry.write_text('# fixture')
        regp,ocp,lp=root/'reg.json',root/'owner.json',root/'launch.json';science=out/'completion.json'
        data=out/'manifest.json';write(data,{'source_count':100})
        job=dict(id='source100',argv=['python','-B','source.py','--config','source.json'],out=str(out),completion=str(science),
            accepted_statuses=['H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE'],receipt_expect={'source_count':100})
        oc=dict(out=str(root),owner_out=str(owner),registration=str(regp),stages=[dict(id='source',jobs=[job])]);write(ocp,oc)
        reg=dict(allowed_stage_ids=['source'],source_bindings={},input_bindings={});write(regp,reg)
        oid=dict(pid=100,start_ticks=1000,uid=1002,argv=['python','-B',str(entry),'--config',str(ocp)])
        child=dict(pid=101,start_ticks=1001,uid=1002,argv=job['argv'])
        write(lp,dict(identity=oid,argv=oid['argv'],registration_sha256=d.sha(regp),owner_config_sha256=d.sha(ocp)))
        write(owner/'owner_identity.json',dict(oid,registration_sha256=d.sha(regp),config_sha256=d.sha(ocp)))
        write(science,dict(status='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE',source_count=100,development_used=True,holdout_used=False,outputs=d.bind((data,))))
        sd=owner/'stages/source';wd=sd/'workers/source100';wd.mkdir(parents=True);log=wd/'worker.log';log.write_text('closed fixture log\n')
        event=dict(job_id='source100',exit_code=0,registration_sha256=d.sha(regp),identity=child,
            closed_log_sha256=d.sha(log),completion=str(science),completion_sha256=d.sha(science))
        write(wd/'exit_receipt.json',event);write(wd/'launch.json',dict(identity=child,argv=child['argv'],registration_sha256=d.sha(regp)))
        write(sd/'completion.json',dict(status='REGISTERED_STAGE_COMPLETE',stage='source',registration_sha256=d.sha(regp),budget=budget(),jobs=[event]))
        write(owner/'completion.json',dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=d.sha(regp),allowed_stage_ids=['source'],
            qualification_passed=True,future_stages_started=False,H_full_delivery_claimed=False,C_started=False,holdout_started=False,
            budget=budget(),completed=[dict(stage='source',completion=str(sd/'completion.json'),sha256=d.sha(sd/'completion.json'))]))
        def receipt(p,statuses,regsha,expected,directory):
            result=d.read(p);d.require(result['status'] in statuses,'Wrong status');d.verify(result['outputs'])
            for k,v in expected.items():d.require(result[k]==v,'Wrong expected');return result
        api=SimpleNamespace(validate_config=lambda *a:None,same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv')),receipt=receipt)
        states={100:None,101:None};wait=SimpleNamespace(exited=lambda ident,reader:reader(ident['pid']) is None)
        args=dict(api=api,wait=wait,spec=dict(config=str(ocp),registration=str(regp),launch=str(lp),completion=str(science)),
            owner_module=str(entry),science_expected={'status':'H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE','source_count':100},state_reader=states.get)
        return args,states

    def test_exact_normal_source_gate_allows_source_preparation_not_failed_or_live_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            args,states=self.gate_fixture(Path(td));got=d.closed_batch(**args)
            self.assertTrue(got['normal_owner_success']);self.assertTrue(got['done']['development_used'])
            for pid in (100,101):
                for state in ('R','Z'):
                    states[pid]={'state':state}
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped'):d.closed_batch(**args)
                states[pid]=None
            (Path(td)/'owner/failure.json').write_text('preserve')
            with self.assertRaisesRegex(RuntimeError,'Failed/stopped'):d.closed_batch(**args)

    def test_changed_science_or_stop_blocks_without_silent_recovery(self):
        for change in ('stop','output'):
            with tempfile.TemporaryDirectory() as td:
                args,_=self.gate_fixture(Path(td))
                if change=='stop':(Path(td)/'source100/STOP').write_text('stop')
                else:(Path(td)/'source100/manifest.json').write_text('{}')
                with self.assertRaises(RuntimeError):d.closed_batch(**args)

    def test_only_original_dev100_complete_codec_admitted(self):
        a=fixture();pop=a['development_registration'];pop['identity']={'models':'frozen'};records=[];codec=[];ao={};co={}
        for i,sid in enumerate(pop['source_ids']):
            cp=f'/dev/asset/{i}.json';cc=f'/dev/codec/{i}.json'
            records.append(dict(source_index=i,source_id=sid,preprocessing_id=pop['preprocessing_ids'][i],checkpoint=cp,checkpoint_sha256='a'*64))
            codec.append(dict(source_index=i,source_id=sid,checkpoint=cc,sha256='b'*64));ao[cp]='a'*64;co[cc]='b'*64
        source=dict(status='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE',source_count=100,source_codec_complete=True,pending_source_encoding_count=0,
            independent_roundtrip=True,population='development',development_used=True,holdout_used=False,new_packet_decodes=0,
            new_image_renders=0,new_metric_calls=0,training_updates=0,source_ids=pop['source_ids'],records=codec,outputs=co)
        assets=dict(status='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',source_count=100,outputs=ao)
        manifest=dict(status=assets['status'],source_ids=pop['source_ids'],preprocessing_ids=pop['preprocessing_ids'],
            original_data_bindings=pop['data_bindings'],visual_identity=pop['identity'],records=records)
        self.assertEqual(d.validate_sources(pop,source,assets,manifest),pop['source_ids'])
        for k,v in [('status','H_FULL1000_SOURCE_CODEC_COMPLETE'),('pending_source_encoding_count',1),('source_count',200),('independent_roundtrip',False)]:
            with self.assertRaises(RuntimeError):d.validate_sources(pop,dict(source,**{k:v}),assets,manifest)
        bad=copy.deepcopy(manifest);bad['source_ids'][0]='calibration_source'
        with self.assertRaises(RuntimeError):d.validate_sources(pop,source,assets,bad)

    def test_unbound_or_missing_actual_input_refused_before_backend_import(self):
        cfg={k:'/unavailable/'+k+'.json' for k in d.REQUIRED}
        with self.assertRaises((RuntimeError,FileNotFoundError)):d.prepare_context(cfg,{})
        with tempfile.TemporaryDirectory() as td:
            request=Path(td)/'r.json';write(request,{'schema':'H_DEVELOPMENT_CPU_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux'):r.register(request)

    def test_source_qualification_must_cover_the_actual_executed_entry(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'actual_source.py';other=Path(td)/'unrelated.py'
            source.write_text('# actual frozen source entry');other.write_text('# unrelated qualification')
            owner={'stages':[{'jobs':[{'argv':['python','-B',str(source),'--config','cfg']}]}]}
            api=SimpleNamespace(command_entry=lambda argv:Path(argv[2]))
            with self.assertRaisesRegex(RuntimeError,'actual executed entry'):
                d.source_qualification_covers_entries({'source_bindings':d.bind((other,))},owner,api)
            d.source_qualification_covers_entries({'source_bindings':d.bind((source,other))},owner,api)
            with self.assertRaisesRegex(RuntimeError,'no bound implementation'):
                d.source_qualification_covers_entries({'source_bindings':{}},owner,api)

    def test_H_admission_preserves_all_other_phases_and_MAIN2400(self):
        before=budget()
        for n in (0,1,5400,10800):d.budget_admission(before,budget(n))
        with self.assertRaises(RuntimeError):d.budget_admission(before,budget(10801))
        with self.assertRaises(RuntimeError):d.budget_admission(budget(1),budget(1))
        other=budget(3);other['phase_charged']['engineering_reserve']=1;other['charged']+=1
        with self.assertRaises(RuntimeError):d.budget_admission(before,other)
        with self.assertRaises(RuntimeError):d.budget_admission(before,budget(1),['H:development:Hslot18:src0000:seed6201:header'])

    def test_generated_CPU_only_owner_contract_keeps_MAIN_pending(self):
        old=dict(root='/root',out='/H',cpu_affinities=[[24,25],[26,27]],phase_limits=d.PHASES,stages=[])
        draft=dict(root='/root',H_out='/H',stop_file='/H/STOP')
        request=dict(execution_dir='/H/dev_execution',payload_out='/H/devCPU',python='/python',max_worker_seconds=3600,merge_seconds=300)
        cfg,oc=r.make_configs(request,draft,old,[{'status':'FROZEN_POLICY_READY'}]*18)
        self.assertEqual([s['id'] for s in oc['stages']],['development','report'])
        self.assertEqual([len(s['jobs']) for s in oc['stages']],[2,1])
        self.assertEqual(oc['stages'][0]['jobs'][0]['receipt_expect']['frame_count'],2700)
        self.assertEqual(oc['stages'][1]['jobs'][0]['receipt_expect']['frame_count'],5400)
        self.assertFalse(oc['stages'][1]['jobs'][0]['receipt_expect']['MAIN_complete'])
        self.assertEqual(oc['stages'][1]['jobs'][0]['receipt_expect']['MAIN_reserved_packet_calls'],2400)
        self.assertNotIn('execution_registration_sha256',cfg)

    def test_full_source_reader_fields_reused_without_importing_old_full_runner(self):
        # The loading adapter must reject absent original cache evidence even
        # if an otherwise plausible source decoder returns tokens.
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'asset.json';write(cp,{'preprocessing_id':'a'*64,'original_development_data_binding':{'index':999}})
            ctx=dict(source_module=SimpleNamespace(load_source=lambda ctx,i:{'scales':'test'}),manifest={'records':[{'checkpoint':str(cp)}]},
                args={'development_registration':{'preprocessing_ids':['a'*64],'data_bindings':[{'index':0}]}})
            with self.assertRaisesRegex(RuntimeError,'original development'):d.load_source(ctx,0)

    def test_trace_cartesian_missing_frame_and_actual_ledger_tamper_rejected(self):
        a=fixture();context=core.prepare(**a,contract=seal(a));ledger=FakeLedger();header=FakeHeader(reject=True);backend=FakeBackend()
        scales=[np.arange(s*s,dtype=np.int64)%4096 for s in SIZES];bits={m:np.zeros(12*token_count(m)+2,dtype=np.uint8) for m in (6,7,8,9)}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reg=root/'reg.json';write(reg,{'test_only':True});context['contract']['execution_registration_sha256']=d.sha(reg)
            frames=[core.run_frame(ctx=context,development_slot=s,source_id='dev_000',source_index=0,noise_seed=n,
                scales=scales,arithmetic_bits=bits,backend=backend,header=header,ledger=ledger) for s in range(18) for n in core.SEEDS]
            p=root/'source.json';doc=dict(status='H_DEVELOPMENT_CPU_SOURCE_TRACES',source_index=0,source_id='dev_000',registration_sha256=d.sha(reg),frames=frames,source_bindings={});write(p,doc)
            ctx=dict(cfg={'registration':str(reg)},core=core,context=context,rows=context['schedule'],ids=context['source_ids'])
            with patch.object(d,'indices',return_value=[0]):events,summary=d.trace_inventory(ctx,[p],0)
            self.assertEqual(summary['frame_count'],54);self.assertEqual(len(events),54)
            db=root/'ledger.sqlite'
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE events(event_id,phase,kind,status,result,result_sha)');conn.execute('CREATE TABLE counters(phase,charged)')
                for eid,v in events.items():
                    result=ledger.results[eid][1];conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(eid,'development',v['kind'],'COMPLETE',d.canonical(result),d.digest(result)))
                conn.execute('INSERT INTO counters VALUES(?,?)',('development',54))
            conn.close()
            self.assertEqual(d.audit_ledger(db,events)['paid_events'],54)
            with sqlite3.connect(db) as conn:conn.execute("UPDATE events SET result='{}' WHERE event_id=?",(next(iter(events)),))
            conn.close()
            with self.assertRaises(RuntimeError):d.audit_ledger(db,events)
            doc['frames'].pop();write(p,doc)
            with patch.object(d,'indices',return_value=[0]):
                with self.assertRaisesRegex(RuntimeError,'Missing policy'):d.trace_inventory(ctx,[p],0)


if __name__=='__main__':unittest.main()
