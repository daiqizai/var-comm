"""Pure metadata/closed-process tests, never actual PHY, GPU or jobs."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import register_partial1000_cpu as r
import register_full1000_assets as t
import test_register_full1000_assets as at
import test_register_whole1000_cpu as wt


def budgets(whole=48000):
    counts=dict(t.COUNTS);counts['whole_calibration']=whole
    owner=dict(created=True,charged=sum(counts.values()),phase_charged=counts,unresolved=0,failed=0,development_remaining=13200)
    rows=[dict(phase=p,kind='body',status='COMPLETE',count=n) for p,n in t.COUNTS.items() if n]
    rows.append(dict(phase='whole_calibration',kind='header',status='COMPLETE',count=24000))
    if whole>24000:rows.append(dict(phase='whole_calibration',kind='body',status='COMPLETE',count=whole-24000))
    ledger=dict(schema='PACKET_PRECHARGE_V1',registration_sha256='b'*64,branch='H',limits=t.PHASES,total_cap=200000,
        charged=sum(counts.values()),phase_charged=counts,phase_remaining={p:t.PHASES[p]-n for p,n in counts.items()},
        unresolved=0,counts=rows,development_reserve_transferable=False)
    science=dict(ledger=ledger,logical_packet_events=whole,ledger_audit=dict(status='FULL_CPU_LEDGER_TRACE_MATCH',all_complete=True,paid_events=whole))
    return owner,science


class PartialRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        original,states,_=wt.WholeRegistrationTests().fixture(root);oldpaths=original['paths']
        oldpaths['driver_module']=Path(__file__).with_name('h_full_payload_driver.py').resolve()
        h=root/'H';base=h/'whole_exec';base.mkdir();out=h/'whole';out.mkdir()
        paths={k:base/v for k,v in dict(whole_config='payload_config.json',whole_registration='execution_registration.json',
            whole_owner_config='owner_config.json',whole_launch='launch.json',whole_owner_completion='completion.json',whole_request='request.json').items()}
        paths['whole_completion']=out/'completion.json';paths['whole_registrar_module']=Path(__file__).with_name('register_whole1000_cpu.py').resolve()
        selected=root/'selected.json';partial=root/'partial.json';at.rewrite(selected,dict(test_only=True));at.rewrite(partial,dict(test_only=True))
        cfg=dict(schema='H_FULL_PAYLOAD_CONFIG_V1',phase='whole_calibration',workers=2,root=str(root),out=str(out),registration=str(paths['whole_registration']),
            owner_config=str(paths['whole_owner_config']),phase_limits=t.PHASES,cpu_affinities=[[0,1],[2,3]],selected=str(selected),partial_reference=str(partial),
            source_completion=str(oldpaths['source_completion']),predecessor_batches={'source':{'config':'source'},'selection':{'config':'selection'}})
        at.rewrite(paths['whole_config'],cfg);at.rewrite(paths['whole_request'],{'test_only':True})
        oc=r.read(oldpaths['source_owner_config']);oc.update(owner_out=str(base),registration=str(paths['whole_registration']),budget_registration_sha256='b'*64)
        stages=[]
        for sid in ('whole_calibration','report'):
            jobs=[]
            for i in range(2 if sid=='whole_calibration' else 1):
                worker=sid=='whole_calibration';folder=out/f'worker_{i}' if worker else out
                suffix=['--stage','worker','--worker-index',str(i)] if worker else ['--stage','merge']
                jobs.append(dict(id=f'whole_worker_{i}' if worker else 'whole_merge',argv=[sys.executable,'-B',str(oldpaths['driver_module']),'--config',str(paths['whole_config']),*suffix],
                    cwd=str(root),out=str(folder),completion=str(folder/'completion.json'),accepted_statuses=['H_FULL1000_CPU_WORKER_COMPLETE' if worker else 'H_FULL1000_CPU_RECEIVE_COMPLETE']))
            stages.append(dict(id=sid,resource='cpu',requires=[] if sid=='whole_calibration' else ['whole_calibration'],max_seconds=100,jobs=jobs))
        oc['stages']=stages;at.rewrite(paths['whole_owner_config'],oc)
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',source_stage_scope='WHOLE1000_ACTUAL_CPU_RECEIVE_ONLY',allowed_stage_ids=['whole_calibration','report'],
            phase_limits=t.PHASES,owner_config_sha256=r.sha(paths['whole_owner_config']),budget_before=at.budget(),
            source_bindings=r.bind((oldpaths['driver_module'],oldpaths['owner_module'],paths['whole_registrar_module'])),
            input_bindings=r.bind((paths['whole_config'],paths['whole_request'])))
        at.rewrite(paths['whole_registration'],reg);regsha=r.sha(paths['whole_registration'])
        oid=dict(pid=200,uid=1002,start_ticks=2000,argv=[sys.executable,'-B',str(oldpaths['owner_module']),'--config',str(paths['whole_owner_config'])])
        at.rewrite(paths['whole_launch'],dict(identity=oid,argv=oid['argv'],registration_sha256=regsha,owner_config_sha256=r.sha(paths['whole_owner_config'])))
        at.rewrite(base/'owner_identity.json',dict(oid,registration_sha256=regsha,config_sha256=r.sha(paths['whole_owner_config'])))
        states[200]=None;ob,ledger=budgets();sealed_outputs={};completed=[];worker_evidence=[]
        common=dict(phase='whole_calibration',workers=2,registration_sha256=regsha,driver_config_sha256=r.sha(paths['whole_config']),
            selected_sha256=r.sha(selected),partial_reference_sha256=r.sha(partial),source_completion_sha256=r.sha(oldpaths['source_completion']),
            budget_registration_sha256='b'*64,maximum_packet_calls=48000,images_scored=False,source_decode_complete=False,arithmetic_source_decode_complete=False,
            GPU_used=False,development_used=False,holdout_used=False,policy_selection=False,full1000_calibration_complete=False,visual_stage_started=False,
            requires_both_cpu_groups_closed_before_visual_stage=True,initial_physical_results_reused=False,source_bindings=reg['source_bindings'],input_bindings=reg['input_bindings'])
        for stage in stages:
            events=[]
            for i,job in enumerate(stage['jobs']):
                worker=stage['id']=='whole_calibration';folder=Path(job['out']);folder.mkdir(exist_ok=True)
                if worker:
                    output=folder/'metadata.json';at.rewrite(output,dict(test_only=True));outputs=r.bind((output,))
                else:outputs=sealed_outputs.copy()
                done=dict(common,status=job['accepted_statuses'][0],worker_index=i if worker else None,source_count=500 if worker else 1000,
                    frame_count=12000 if worker else 24000,source_indices=list(range(i,1000,2)) if worker else list(range(1000)),outputs=outputs)
                if not worker:done.update(ledger)
                cp=Path(job['completion']);at.rewrite(cp,done);sealed_outputs.update(outputs);sealed_outputs[str(cp)]=r.sha(cp)
                wp=base/'stages'/stage['id']/'workers'/job['id'];wp.mkdir(parents=True);worker_evidence.append(wp)
                pid=201+i if worker else 203;ident=dict(pid=pid,uid=1002,start_ticks=pid*10,argv=job['argv']);states[pid]=None
                log=wp/'worker.log';log.write_bytes(b'closed\n');event=dict(job_id=job['id'],identity=ident,exit_code=0,registration_sha256=regsha,
                    completion=job['completion'],completion_sha256=r.sha(cp),closed_log_sha256=r.sha(log))
                at.rewrite(wp/'exit_receipt.json',event);at.rewrite(wp/'launch.json',dict(identity=ident,registration_sha256=regsha));events.append(event)
            sp=base/'stages'/stage['id']/'completion.json';at.rewrite(sp,dict(status='REGISTERED_STAGE_COMPLETE',stage=stage['id'],registration_sha256=regsha,budget=ob,jobs=events))
            completed.append(dict(stage=stage['id'],completion=str(sp),sha256=r.sha(sp)))
        at.rewrite(paths['whole_owner_completion'],dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=regsha,allowed_stage_ids=['whole_calibration','report'],
            qualification_passed=True,future_stages_started=False,H_full_delivery_claimed=False,C_started=False,holdout_started=False,budget=ob,completed=completed))
        return dict(a=original['a'],w=original['w'],t=t,paths=paths,oldpaths=oldpaths,state_reader=lambda pid:states[pid]),states,worker_evidence

    def test_real_frozen_batch_verifier_requires_three_successful_child_exits(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));ctx=r.verify_whole_closed(**args)
            self.assertEqual(len(ctx['closure']['child_identities']),3);self.assertTrue(ctx['closure']['whole_owner_success'])
            self.assertEqual(ctx['budget']['charged'],117960);self.assertFalse(ctx['closure']['original_render_owner_success'])

    def test_owner_either_worker_or_merge_still_live_or_zombie_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,workers=self.fixture(Path(td))
            ids=[r.read(args['paths']['whole_launch'])['identity']]+[r.read(p/'launch.json')['identity'] for p in workers]
            for ident in ids:
                for state in ('R','Z'):
                    states[ident['pid']]=dict(ident,state=state)
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped|reap'):r.verify_whole_closed(**args)
                states[ident['pid']]=None

    def test_stops_failures_and_unclosed_logs_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,workers=self.fixture(Path(td));p=args['paths'];oc=r.read(p['whole_owner_config']);out=Path(r.read(p['whole_config'])['out'])
            for path in (Path(oc['out'])/'STOP',Path(oc['owner_out'])/'failure.json',out/'worker_0/STOP',out/'worker_1/failure.json'):
                path.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'STOP|Failed'):r.verify_whole_closed(**args)
                path.unlink()
            (workers[2]/'worker.log').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'log changed'):r.verify_whole_closed(**args)

    def test_budget_allows_skipped_bodies_but_never_missing_headers_or_other_phase(self):
        for n in (24000,47999,48000):
            ob,s=budgets(n);self.assertEqual(r.check_closed_budget(at.budget(),ob,s,t),ob)
        ob,s=budgets();s['ledger']['counts'][-2]['count']=23999
        with self.assertRaisesRegex(RuntimeError,'payment count'):r.check_closed_budget(at.budget(),ob,s,t)
        ob,s=budgets();s['ledger']['phase_charged']['partial_calibration']=1
        with self.assertRaisesRegex(RuntimeError,'Only the original whole'):r.check_closed_budget(at.budget(),ob,s,t)
        ob,s=budgets();s['ledger']['counts'][-1]['status']='RESERVED'
        with self.assertRaisesRegex(RuntimeError,'unresolved'):r.check_closed_budget(at.budget(),ob,s,t)
        ob,s=budgets();ob['charged']+=1
        with self.assertRaisesRegex(RuntimeError,'snapshots disagree'):r.check_closed_budget(at.budget(),ob,s,t)

    def test_paid_audit_identity_and_duplicate_group_counts_are_checked(self):
        for change in ('audit','duplicates','reserve'):
            ob,s=budgets()
            if change=='audit':s['ledger_audit']['paid_events']-=1
            elif change=='duplicates':s['ledger']['counts'].append(copy.deepcopy(s['ledger']['counts'][0]))
            else:s['ledger']['development_reserve_transferable']=True
            with self.assertRaises(RuntimeError):r.check_closed_budget(at.budget(),ob,s,t)

    def test_complete_receipt_wrong_partition_or_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));out=Path(r.read(args['paths']['whole_config'])['out'])
            cp=out/'worker_0/completion.json';d=r.read(cp);d['source_indices'][0]=1;at.rewrite(cp,d)
            with self.assertRaisesRegex(RuntimeError,'changed|binding'):r.verify_whole_closed(**args)

    def test_partial_config_only_uses_existing_six_and_full_shared_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=self.fixture(root);ctx=r.verify_whole_closed(**args);ctx['paths']=args['paths']
            oldcfg=copy.deepcopy(ctx['cfg']);request=dict(execution_dir=str(root/'H/partial_exec'),payload_out=str(root/'H/partial'),
                python=sys.executable,max_worker_seconds=14400,merge_seconds=1800)
            cfg,oc=r.make_configs(request,ctx,args['oldpaths'])
            self.assertEqual([(s['id'],s['resource'],len(s['jobs'])) for s in oc['stages']],[('partial_calibration','cpu',2),('report','cpu',1)])
            self.assertEqual(set(cfg['predecessor_batches']),{'source','selection','whole'});self.assertEqual(ctx['cfg'],oldcfg)
            for k in ('selected','partial_reference','source_completion','cpu_affinities','phase_limits'):self.assertEqual(cfg[k],oldcfg[k])
            self.assertEqual(cfg['whole_cpu_completion'],str(args['paths']['whole_completion']))
            self.assertEqual(oc['stages'][0]['jobs'][1]['receipt_expect']['frame_count'],9000)
            self.assertEqual(oc['stages'][1]['jobs'][0]['receipt_expect']['frame_count'],18000)
            self.assertEqual(oc['stages'][1]['jobs'][0]['receipt_expect']['maximum_packet_calls'],36000)
            reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256='test',allowed_stage_ids=['partial_calibration','report'],
                phase_limits=t.PHASES,source_bindings=r.bind((args['oldpaths']['driver_module'],)))
            args['a'].validate_config(oc,reg,'test')

    def test_nonlinux_registration_refused_without_files_or_experiment_calls(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';at.rewrite(p,dict(schema='H_PARTIAL1000_CPU_REGISTRATION_REQUEST_V1'))
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([p.name for p in Path(td).iterdir()],['request.json'])
        tree=ast.parse(Path(r.__file__).read_text());forbidden={'run_frame','receive_frame','decode_once','build_runtime','build_native','Popen','system'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,forbidden)


if __name__=='__main__':unittest.main()
