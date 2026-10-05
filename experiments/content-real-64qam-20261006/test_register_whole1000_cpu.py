"""Pure CPU registration tests: fake source receipts, never real PHY or GPU."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import register_whole1000_cpu as r
import register_full1000_assets as t
import test_register_full1000_assets as at
import test_register_full1000_source as st


class WholeRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        old,states,oldwp=st.SourceRegistrationTests().fixture(root)
        paths=old['paths'];cfg=r.read(paths['assets_config']);oc=r.read(paths['assets_owner_config']);reg=r.read(paths['assets_registration'])
        rename={'source_config':'assets_config','source_registration':'assets_registration','source_owner_config':'assets_owner_config',
            'source_launch':'assets_owner_launch','source_owner_completion':'assets_owner_completion','source_completion':'assets_completion',
            'source_module':'asset_module'}
        p={new:paths[k] for new,k in rename.items()}
        p.update({k:paths[k] for k in ('owner_module','wait_module','assets_registrar_module')})
        cfg.update(source_owner_config=str(p['source_owner_config']),H_out=oc['out']);at.rewrite(p['source_config'],cfg)
        stage=oc['stages'][0];stage['resource']='gpu';job=stage['jobs'][0]
        job.update(id='full1000_source',accepted_statuses=['H_FULL1000_SOURCE_CODEC_COMPLETE'])
        at.rewrite(p['source_owner_config'],oc)
        reg.update(source_stage_scope='FULL1000_SOURCE_CODEC_ONLY',owner_config_sha256=r.sha(p['source_owner_config']),budget_before=at.budget())
        reg['input_bindings'].update(r.bind((p['source_config'],)))
        at.rewrite(p['source_registration'],reg);rs=r.sha(p['source_registration'])
        launch=r.read(p['source_launch']);launch.update(registration_sha256=rs,owner_config_sha256=r.sha(p['source_owner_config']))
        at.rewrite(p['source_launch'],launch);base=Path(oc['owner_out'])
        at.rewrite(base/'owner_identity.json',dict(launch['identity'],registration_sha256=rs,config_sha256=r.sha(p['source_owner_config'])))
        done=dict(status='H_FULL1000_SOURCE_CODEC_COMPLETE',registration_sha256=rs,config_sha256=r.sha(p['source_config']),
            source_count=1000,source1000_codec_complete=True,reused_codec_sources=200,newly_encoded_sources=800,
            pending_source_encoding_count=0,independent_roundtrip=True,new_canonical_roundtrips=3200,
            new_packet_decodes=0,new_image_renders=0,new_metric_calls=0,development_used=False,holdout_used=False,
            training_updates=0,full1000_calibration_complete=False,GPU_used=True,budget_before=at.budget(),budget_after=at.budget(),
            source_bindings=reg['source_bindings'],input_bindings=reg['input_bindings'],outputs=r.bind((paths['assets_manifest'],)))
        at.rewrite(p['source_completion'],done)
        wp=base/'stages/source/workers/full1000_source';wp.mkdir()
        log=wp/'worker.log';log.write_bytes(b'source complete\n');ident=dict(pid=101,uid=1002,start_ticks=1001,argv=job['argv'])
        event=dict(job_id='full1000_source',identity=ident,exit_code=0,registration_sha256=rs,
            completion=str(p['source_completion']),completion_sha256=r.sha(p['source_completion']),closed_log_sha256=r.sha(log))
        at.rewrite(wp/'exit_receipt.json',event);at.rewrite(wp/'launch.json',dict(identity=ident,registration_sha256=rs))
        sp=base/'stages/source/completion.json';at.rewrite(sp,dict(status='REGISTERED_STAGE_COMPLETE',stage='source',registration_sha256=rs,budget=at.budget(),jobs=[event]))
        at.rewrite(p['source_owner_completion'],dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=rs,allowed_stage_ids=['source'],
            qualification_passed=True,future_stages_started=False,H_full_delivery_claimed=False,C_started=False,holdout_started=False,
            budget=at.budget(),completed=[dict(stage='source',completion=str(sp),sha256=r.sha(sp))]))
        return dict(a=old['a'],w=old['w'],t=t,paths=p,state_reader=lambda pid:states[pid]),states,wp

    def reseal_done(self,args,wp,done):
        paths=args['paths'];at.rewrite(paths['source_completion'],done)
        ep=wp/'exit_receipt.json';event=r.read(ep);event['completion_sha256']=r.sha(paths['source_completion']);at.rewrite(ep,event)
        base=paths['source_owner_completion'].parent;sp=base/'stages/source/completion.json';sd=r.read(sp);sd['jobs']=[event];at.rewrite(sp,sd)
        od=r.read(paths['source_owner_completion']);od['completed'][0]['sha256']=r.sha(sp);at.rewrite(paths['source_owner_completion'],od)

    def test_real_original_owner_closure_passes_only_complete_source(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));ctx=r.verify_source_closed(**args)
            self.assertTrue(ctx['closure']['owner_exited']);self.assertTrue(ctx['closure']['source_owner_success'])
            self.assertFalse(ctx['closure']['original_render_owner_success']);self.assertEqual(ctx['source_done']['newly_encoded_sources'],800)

    def test_live_zombie_owner_or_source_worker_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,wp=self.fixture(Path(td))
            for pid in (100,101):
                ident=r.read(args['paths']['source_launch'])['identity'] if pid==100 else r.read(wp/'launch.json')['identity']
                for state in ('R','Z'):
                    states[pid]=dict(ident,state=state)
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped|reap'):r.verify_source_closed(**args)
                states[pid]=None

    def test_global_local_stop_failure_and_changed_log_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,wp=self.fixture(Path(td));oc=r.read(args['paths']['source_owner_config'])
            for p in (Path(oc['out'])/'STOP',Path(oc['owner_out'])/'STOP',Path(oc['stages'][0]['jobs'][0]['out'])/'STOP',
                      Path(oc['owner_out'])/'failure.json',Path(oc['owner_out'])/'registration_failure.json'):
                p.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'STOP|failed'):r.verify_source_closed(**args)
                p.unlink()
            (wp/'worker.log').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'log changed'):r.verify_source_closed(**args)

    def test_incomplete_800_codec_or_wrong_budget_not_promoted(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,wp=self.fixture(Path(td));base=r.read(args['paths']['source_completion'])
            for key,value in (('newly_encoded_sources',799),('pending_source_encoding_count',1),('source1000_codec_complete',False),('new_canonical_roundtrips',3196),('new_packet_decodes',1)):
                done=copy.deepcopy(base);done[key]=value;self.reseal_done(args,wp,done)
                with self.assertRaisesRegex(RuntimeError,'scope differs'):r.verify_source_closed(**args)
            done=copy.deepcopy(base);done['budget_after']['charged']+=1;self.reseal_done(args,wp,done)
            with self.assertRaisesRegex(RuntimeError,'packet budget'):r.verify_source_closed(**args)

    def test_launch_identity_swap_refused(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));p=args['paths']['source_launch'];v=r.read(p);v['identity']['start_ticks']+=1;at.rewrite(p,v)
            with self.assertRaisesRegex(RuntimeError,'identity differs'):r.verify_source_closed(**args)

    def test_two_cpu_workers_then_one_merge_preserves_budget_no_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args,_,_=self.fixture(root);ctx=r.verify_source_closed(**args);p=args['paths']
            p.update(driver_module=Path(__file__).with_name('h_full_payload_driver.py').resolve())
            for k in ('selection_owner_config','selection_registration','selection_launch','selected','partial_reference','selection_completion'):
                p[k]=root/(k+'.json')
            ctx['source_cfg'].update(assets_completion='/assets/completion.json',assets_manifest='/assets/manifest.json')
            ctx['old'].update(budget_path='/ledger',budget_registration='/budget',budget_registration_sha256='x'*64)
            initial={k:'/old/'+k for k in r.INHERITED_KEYS};initial.update(root=str(root),runtime_dir='/frozen',budget_registration_sha256='x'*64)
            ctx.update(paths=p,initial=initial)
            request=dict(execution_dir=str(root/'H/whole_exec'),payload_out=str(root/'H/whole'),python=sys.executable,max_worker_seconds=14400,merge_seconds=1800)
            cfg,oc=r.make_configs(request,ctx)
            self.assertEqual([(s['id'],s['resource'],len(s['jobs'])) for s in oc['stages']],[('whole_calibration','cpu',2),('report','cpu',1)])
            self.assertEqual(cfg['phase'],'whole_calibration');self.assertEqual(set(cfg['predecessor_batches']),{'source','selection'})
            self.assertEqual(cfg['cpu_affinities'],ctx['old']['cpu_affinities']);self.assertEqual(oc['phase_limits'],ctx['old']['phase_limits'])
            self.assertEqual(oc['stages'][1]['requires'],['whole_calibration']);self.assertEqual(cfg['max_worker_seconds'],14400)
            self.assertEqual(oc['stages'][0]['jobs'][1]['argv'][-4:],['--stage','worker','--worker-index','1'])
            self.assertEqual(oc['stages'][1]['jobs'][0]['receipt_expect']['frame_count'],24000)
            reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256='test',allowed_stage_ids=['whole_calibration','report'],phase_limits=t.PHASES,
                source_bindings=r.bind((p['driver_module'],)))
            args['a'].validate_config(oc,reg,'test')
            self.assertNotIn('execution_registration_sha256',cfg)

    def test_shared_public_schedule_and_full_catalogue_sealed_without_calls(self):
        # Existing fake-backend fixture only constructs layouts; it never decodes.
        import test_h_full_payload_cpu as ct
        v=ct.fixture();core=ct.core;contract=r.make_contract(core,v['protocol'],v['catalogue'],v['selected'],v['partial_reference'],v['shortlist'],v['source_ids'])
        self.assertNotIn('execution_registration_sha256',contract)
        self.assertEqual(contract['noise_stage'],'full_calibration');self.assertTrue(contract['complete_public_receive_catalogue'])
        self.assertEqual(contract['public_schedule_canonical_sha256'],core.digest(contract['public_schedule']))
        self.assertEqual(len(contract['public_schedule']),14);self.assertEqual(len(contract['public_schedule'][:8])*1000*3,24000)
        self.assertEqual(contract['catalogue_canonical_sha256'],core.digest(v['catalogue']))
        self.assertGreater(len(v['catalogue']['profiles']),14)
        other=copy.deepcopy(v['selected']);other['selected_candidates'][0]['target_m']+=1
        with self.assertRaises(ValueError):r.make_contract(core,v['protocol'],v['catalogue'],other,v['partial_reference'],v['shortlist'],v['source_ids'])

    def test_registration_never_starts_on_nonlinux_and_has_no_launch_or_decoder_call(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';at.rewrite(p,dict(schema='H_WHOLE1000_CPU_REGISTRATION_REQUEST_V1'))
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])
        tree=ast.parse(Path(r.__file__).read_text())
        forbidden={'run_frame','receive_frame','decode_once','build_runtime','build_native','Popen','system','check_call'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,forbidden)


if __name__=='__main__':unittest.main()
