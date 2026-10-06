"""Read-only closure/config fixtures; no model, actual decoder or job launch."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import register_full1000_selection as r


def rewrite(p,value):Path(p).write_text(json.dumps(value),encoding='utf-8')
def budget(whole=48000,partial=36000):
    c=dict.fromkeys(r.PHASES,0);c.update(qualification=1608,coarse=49152,initial_true200=19200,whole_calibration=whole,partial_calibration=partial)
    return dict(created=True,charged=sum(c.values()),phase_charged=c,unresolved=0,failed=0,development_remaining=13200)


class FullSelectionRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'owner';base.mkdir();out=root/'render';out.mkdir();p={k:root/(k+'.json') for k in ('render_registration','render_config','render_owner_config','render_launch')}
        entry=root/'stage_owner.py';entry.write_text('# immutable owner')
        workerentry=root/'visual.py';workerentry.write_text('# immutable visual')
        graph=base/'visual_source_closure.json';r.save(graph,dict(status='EXACT_SOURCE_CLOSURE_MATCH',source_bindings=r.bind((entry,workerentry))))
        numerical=root/'numeric.json';cal=root/'cal.json';r.save(numerical,{'numerical_runtime':{'threads':6,'fp32':True}});r.save(cal,{'identity':'frozen'})
        cfg=dict(registration=str(p['render_registration']),visual_owner_config=str(p['render_owner_config']),out=str(out),
            H_out=str(root),stop_file=str(root/'STOP'),owner_module=str(entry),visual_driver_module=str(workerentry),
            numerical_reference=str(numerical),calibration_registration=str(cal))
        r.save(p['render_config'],cfg)
        job=dict(id='full1000_render',argv=['python','-B',str(workerentry),'--config',str(p['render_config'])],out=str(out),
            completion=str(out/'completion.json'),accepted_statuses=['H_FULL1000_RX_COMPLETE'],receipt_expect=copy.deepcopy(r.EXPECTED))
        oc=dict(registration=cfg['registration'],owner_out=str(base),out=str(root),stages=[dict(id='render',resource='gpu',jobs=[job])])
        r.save(p['render_owner_config'],oc)
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',allowed_stage_ids=['render'],source_stage_scope='FULL1000_ACTUAL_RX_IMAGES_ONLY',
            source_bindings=r.bind((entry,workerentry)),input_bindings=r.bind((graph,numerical,cal)),budget_before=budget())
        r.save(p['render_registration'],reg)
        oid=dict(pid=100,uid=1002,start_ticks=900,argv=['python','-B',str(entry),'--config',str(p['render_owner_config'])])
        child=dict(pid=101,uid=1002,start_ticks=910,argv=job['argv'])
        r.save(p['render_launch'],dict(identity=oid,argv=oid['argv'],registration_sha256=r.sha(p['render_registration']),owner_config_sha256=r.sha(p['render_owner_config'])))
        ip=base/'owner_identity.json';r.save(ip,dict(oid,registration_sha256=r.sha(p['render_registration']),config_sha256=r.sha(p['render_owner_config'])))
        cp=out/'completion.json';done=dict(status='H_FULL1000_RX_COMPLETE',**r.EXPECTED,registration_sha256=r.sha(p['render_registration']),
            config_sha256=r.sha(p['render_config']),source_bindings=reg['source_bindings'],input_bindings=reg['input_bindings'],predecessor_closure_bindings={},
            supervision={'bindings':r.bind((ip,)),'owner_identity':oid,'worker_identity':child},visual_source_bindings=reg['source_bindings'],
            numerical_runtime={'threads':6,'fp32':True},frozen_visual_identity='frozen',budget_before=budget(),budget_after=budget())
        r.save(cp,done);r.save(base/'completion.json',dict(budget=budget()))
        closure={'bindings':r.bind((cp,)),'child_identities':[child]};states={100:None,101:None}
        a=SimpleNamespace(validate_config=lambda *_:None,same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','start_ticks','argv')))
        calls=[]
        def checked_batch(*args):calls.append(args);return copy.deepcopy(closure)
        w=SimpleNamespace(exited=lambda ident,reader:reader(ident['pid']) is None,verify_batch=checked_batch)
        return dict(a=a,w=w,paths=p,state_reader=lambda pid:states[pid]),states,closure,cp,calls

    def test_normal_exit_requires_explicit_receipt_contract_and_preserves_historical_closure(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,closure,cp,calls=self.fixture(Path(td));ctx=r.normal_render_closed(**args)
            self.assertEqual(len(calls),1);self.assertEqual(ctx['completion'],cp)
            self.assertFalse(ctx['graph']['current_source_graph_reenumerated'])
            # A new independent CPU module does not rewrite the completed GPU graph.
            (Path(td)/'new_selector.py').write_text('# separate post-GPU CPU stage')
            r.normal_render_closed(**args)

    def test_live_zombie_owner_or_child_and_unbound_completion_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,closure,cp,_=self.fixture(Path(td))
            for pid in (100,101):
                for state in ('R','Z'):
                    states[pid]={'state':state}
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped|unproven'):r.normal_render_closed(**args)
                states[pid]=None
            closure['bindings'][str(cp)]='0'*64
            with self.assertRaisesRegex(RuntimeError,'completion differs'):r.normal_render_closed(**args)

    def test_original_failure_stop_and_missing_expected_frame_count_rejected(self):
        for name in ('owner/failure.json','owner/registration_failure.json','owner/STOP','render/failure.json','render/STOP','STOP'):
            with tempfile.TemporaryDirectory() as td:
                args,*_=self.fixture(Path(td));(Path(td)/name).write_text('preserve')
                with self.assertRaisesRegex(RuntimeError,'failure/STOP'):r.normal_render_closed(**args)
        with tempfile.TemporaryDirectory() as td:
            args,*_=self.fixture(Path(td));p=args['paths']['render_owner_config'];oc=r.read(p);del oc['stages'][0]['jobs'][0]['receipt_expect']['frame_count'];rewrite(p,oc)
            with self.assertRaisesRegex(RuntimeError,'exact expectations'):r.normal_render_closed(**args)

    def test_changed_old_visual_source_never_excused_by_new_cpu_graph(self):
        with tempfile.TemporaryDirectory() as td:
            args,*_=self.fixture(Path(td));(Path(td)/'visual.py').write_text('# changed old source')
            with self.assertRaisesRegex(RuntimeError,'Changed bound'):r.normal_render_closed(**args)
        with self.assertRaisesRegex(RuntimeError,'Conflicting'):r.merge({'/old.py':'old'},{'/old.py':'new'})

    def test_actual_budget_derived_no_free_body_or_hardcoded_total(self):
        for whole,partial in ((48000,36000),(47999,35999),(24000,18000)):
            b=budget(whole,partial);self.assertEqual(r.frozen_budget(b,{'budget_before':b},{'budget_before':b,'budget_after':b},{'budget':b}),b)
            changed=copy.deepcopy(b);changed['charged']+=1
            with self.assertRaisesRegex(RuntimeError,'ledger differ'):r.frozen_budget(changed,{'budget_before':b},{'budget_before':b,'budget_after':b},{'budget':b})
        bad=budget();bad['unresolved']=1
        with self.assertRaisesRegex(RuntimeError,'quiescent'):r.frozen_budget(bad,{'budget_before':bad},{'budget_before':bad,'budget_after':bad},{'budget':bad})

    def test_single_cpu_stage_no_hash_cycle_no_automatic_development(self):
        old=dict(out='/H',cpu_affinities=[[24,25],[26,27]],gpu_affinity=list(range(24,30)),gpu_threads=6,phase_limits=r.PHASES,stages=[])
        cfg,oc=r.make_configs(dict(request_path='/request',selection_out='/H/selection',max_seconds=600,python='/python'),
            dict(old=old,cfg={'root':'/root','H_out':'/H','stop_file':'/H/STOP'}),Path('/H/execution'))
        self.assertEqual([(x['id'],x['resource']) for x in oc['stages']],[('freeze','cpu')]);self.assertEqual(len(oc['stages'][0]['jobs']),1)
        job=oc['stages'][0]['jobs'][0];self.assertEqual(job['receipt_expect']['whole_count'],8);self.assertEqual(job['receipt_expect']['selected_partial_count'],2)
        self.assertFalse(job['receipt_expect']['whole_policy_reselected']);self.assertFalse(job['receipt_expect']['GPU_used'])
        self.assertEqual(oc['phase_limits'],old['phase_limits']);self.assertNotIn('registration_sha256',cfg)

    def test_owner_precreated_empty_output_once_and_nonlinux_register_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'selected';out.mkdir();r.claim_output(out,'a'*64)
            with self.assertRaisesRegex(RuntimeError,'Prior selection'):r.claim_output(out,'a'*64)
            request=Path(td)/'request.json';r.save(request,{'schema':'H_FULL1000_SELECTION_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux'):r.register(request)


class SourceAuditTests(unittest.TestCase):
    def test_all1000_source_cp_match_aggregate_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);[(out/n).mkdir() for n in ('images','sources','source_checkpoints')]
            rows=[];ids=[f's{i}' for i in range(1000)];outputs={}
            for i,sid in enumerate(ids):
                image=out/'images'/f'{i:04d}.npz';image.write_bytes(b'fixture arrays already SHA checked by original receipt')
                values=out/'sources'/f'{i:04d}.json';local=[dict(source_index=i,source_id=sid,image_archive=str(image),slot=k//3,noise_seed=6101+k%3) for k in range(42)]
                r.save(values,local);cp=out/'source_checkpoints'/f'{i:04d}.json';r.save(cp,dict(status='H_FULL1000_RX_SOURCE_COMPLETE',registration_sha256='a'*64,
                    source_index=i,source_id=sid,frame_count=42,phase_frame_counts={'whole_calibration':24,'partial_calibration':18},images_scored=True,
                    source_decode_complete=True,new_packet_decodes=0,policy_selection=False,outputs=r.bind((image,values)),input_bindings={}))
                outputs.update(r.bind((image,values,cp)));rows.extend(local)
            done=dict(source_ids=ids,outputs=outputs,registration_sha256='a'*64)
            audit=r.source_grid_audit({'out':str(out)},done,rows);self.assertEqual(audit['frame_count'],42000)
            rows[0]['noise_seed']=9999
            with self.assertRaisesRegex(RuntimeError,'aggregate metrics'):r.source_grid_audit({'out':str(out)},done,rows)


if __name__=='__main__':unittest.main()
