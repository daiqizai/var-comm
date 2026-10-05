"""Synthetic source registration, process closure and source graph tests."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import register_full1000_source as r
import register_full1000_assets as t
import test_register_full1000_assets as at


class SourceRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        old,states,oldwp=at.RegistrationTests().fixture(root)
        pp=old['paths'];oc=t.read(pp['selection_owner_config']);ac=t.read(pp['selection_config']);ar=t.read(pp['selection_registration'])
        base=Path(oc['owner_out']);out=Path(ac['out']);wp=base/'stages/source/workers/full1000_assets';wp.mkdir(parents=True)
        helper=root/'register_full1000_assets.py';helper.write_text('# fixture registrar\n')
        owner=old['owner'];wait=old['prior'];adapter=root/'full1000_assets.py';adapter.write_text('# fixture adapter\n')
        paths=dict(assets_config=pp['selection_config'],assets_registration=pp['selection_registration'],
            assets_owner_config=pp['selection_owner_config'],assets_owner_launch=pp['selection_launch'],
            assets_owner_completion=pp['selection_owner_completion'],assets_completion=pp['selection_completion'],
            assets_manifest=out/'manifest.json',owner_module=Path(owner.__file__),wait_module=Path(wait.__file__),
            asset_module=adapter,assets_registrar_module=helper)
        ac.update(protocol=str(root/'protocol.json'),calibration_registration=str(root/'cal.json'))
        at.rewrite(paths['assets_config'],ac)
        oc['stages'][0]['id']='source';job=oc['stages'][0]['jobs'][0]
        job.update(id='full1000_assets',argv=[sys.executable,'-B',str(adapter),'--config',str(paths['assets_config'])],
            accepted_statuses=['H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'])
        at.rewrite(paths['assets_owner_config'],oc)
        ar.update(allowed_stage_ids=['source'],source_stage_scope='FULL1000_CPU_ASSETS_ONLY',owner_config_sha256=t.sha(paths['assets_owner_config']))
        ar['source_bindings'].update(t.bind((adapter,helper,paths['wait_module'])))
        ar['input_bindings'][str(paths['assets_config'])]=t.sha(paths['assets_config']);at.rewrite(paths['assets_registration'],ar)
        rs=t.sha(paths['assets_registration'])
        launch=t.read(paths['assets_owner_launch']);launch.update(registration_sha256=rs,owner_config_sha256=t.sha(paths['assets_owner_config']))
        at.rewrite(paths['assets_owner_launch'],launch)
        at.rewrite(base/'owner_identity.json',dict(launch['identity'],registration_sha256=rs,config_sha256=t.sha(paths['assets_owner_config'])))
        at.rewrite(paths['assets_manifest'],{'status':'H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE','source_count':1000})
        done=dict(status='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',registration_sha256=rs,
            source_count=1000,reused_codec_sources=200,pending_source_encoding_count=800,source1000_codec_complete=False,
            new_source_encoding=False,GPU_used=False,new_visual_inference=0,new_packet_decodes=0,
            development_used=False,holdout_used=False,full1000_calibration_complete=False,
            input_bindings=ar['input_bindings'],source_bindings=ar['source_bindings'],outputs=t.bind((paths['assets_manifest'],)))
        at.rewrite(paths['assets_completion'],done)
        log=wp/'worker.log';log.write_bytes(b'asset complete\n');ident=dict(pid=101,uid=1002,start_ticks=1001,argv=job['argv'])
        event=dict(job_id='full1000_assets',identity=ident,exit_code=0,registration_sha256=rs,
            completion=str(paths['assets_completion']),completion_sha256=t.sha(paths['assets_completion']),closed_log_sha256=t.sha(log))
        at.rewrite(wp/'exit_receipt.json',event);at.rewrite(wp/'launch.json',dict(identity=ident,registration_sha256=rs))
        sp=base/'stages/source/completion.json';at.rewrite(sp,dict(status='REGISTERED_STAGE_COMPLETE',stage='source',
            registration_sha256=rs,budget=at.budget(),jobs=[event]))
        at.rewrite(paths['assets_owner_completion'],dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=rs,
            allowed_stage_ids=['source'],qualification_passed=True,future_stages_started=False,H_full_delivery_claimed=False,
            C_started=False,holdout_started=False,budget=at.budget(),completed=[dict(stage='source',completion=str(sp),sha256=t.sha(sp))]))
        args=dict(a=owner,w=wait,t=t,paths=paths,state_reader=lambda pid:states[pid])
        return args,states,wp

    def test_actual_frozen_batch_verifier_accepts_only_closed_assets(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));v=r.verify_assets_closed(**args)
            self.assertTrue(v['closure']['owner_exited']);self.assertFalse(v['closure']['original_render_owner_success'])
            self.assertEqual(v['assets_done']['pending_source_encoding_count'],800)

    def test_live_or_zombie_owner_and_child_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,wp=self.fixture(Path(td))
            for pid in (100,101):
                ident=r.read(args['paths']['assets_owner_launch'])['identity'] if pid==100 else r.read(wp/'launch.json')['identity']
                for state in ('R','Z'):
                    states[pid]=dict(ident,state=state)
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped|reap'):r.verify_assets_closed(**args)
                states[pid]=None

    def test_hstop_failed_owner_or_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));oc=r.read(args['paths']['assets_owner_config'])
            for p in (Path(oc['out'])/'STOP',Path(oc['owner_out'])/'failure.json'):
                p.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'STOP|failure'):r.verify_assets_closed(**args)
                p.unlink()
            args['paths']['assets_manifest'].write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'changed|differs'):r.verify_assets_closed(**args)

    def test_only_one_gpu_source_stage_preserves_original_resources(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));ctx=r.verify_assets_closed(**args);paths=args['paths']
            for key in ('source_module','static_closure_module','numerical_reference'):
                p=Path(td)/(key+'.py');p.write_text('# fixture\n');paths[key]=p
            ctx.update(paths=paths,visual=dict(runtime_dir=str(Path(td)),native_runtime='/native',uep_runtime='/uep',source_driver_module='/frozen/h_source_driver.py'),
                static_reference={'inputs':dict(var_source='/VAR',dino_source='/DINO')})
            ctx['old'].update(budget_path='/ledger',budget_registration='/budget')
            request=dict(execution_dir=str(Path(td)/'H/new_execution'),source_out=str(Path(td)/'H/new_source'),python=sys.executable,max_seconds=10800)
            cfg,oc=r.make_configs(request,ctx)
            self.assertEqual([(x['id'],x['resource']) for x in oc['stages']],[('source','gpu')])
            job=oc['stages'][0]['jobs'][0];self.assertEqual(job['receipt_expect']['newly_encoded_sources'],800)
            for key in ('cpu_affinities','gpu_affinity','gpu_threads','phase_limits'):
                self.assertEqual(oc[key],ctx['old'][key])
            self.assertEqual(cfg['numerical_reference_field'],['numerical_runtime'])
            self.assertEqual(cfg['assets_module'],str(paths['asset_module']))
            self.assertNotIn('registration_sha256',cfg);self.assertNotIn('owner_config_sha256',oc)
            self.assertEqual(ctx['old']['stages'][0]['resource'],'cpu')
            reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256='fixture',
                allowed_stage_ids=['source'],phase_limits=t.PHASES,source_bindings=t.bind((paths['source_module'],)))
            args['a'].validate_config(oc,reg,'fixture')

    def graph_fixture(self,root):
        here=Path(__file__).resolve().parent
        helper_path=here/'visual_source_closure.py'
        if not helper_path.is_file():helper_path=here.parent/'recovery/visual_source_closure.py'
        helper=r.module(helper_path,'test_source_reg_static_closure')
        var=root/'VAR';dino=root/'DINO';(var/'models').mkdir(parents=True);(dino/'dinov2/nested').mkdir(parents=True)
        vp=var/'models/var.py';dp=dino/'dinov2/nested/backbone.py';vp.write_text('# originalVAR');dp.write_text('# originalDINO')
        old=root/'old.py';new=root/'new.py';old.write_text('# old');new.write_text('# newly tracked')
        visual=dict(native_runtime='/native',uep_runtime='/quality')
        ref=dict(status='EXACT_SOURCE_CLOSURE_MATCH',inputs=dict(root=str(root),runtime='/native',quality_runtime='/quality',var_source=str(var),dino_source=str(dino)))
        return helper,visual,ref,r.bind((vp,dp,old)),r.bind((vp,dp,old,new)),new,old

    def test_static_closure_new_tracked_sources_added_existing_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            helper,visual,ref,old,actual,new,_=self.graph_fixture(Path(td))
            with patch.object(helper,'collect_bindings',return_value=actual) as collect:
                got,delta=r.static_source_closure(helper,str(Path(td)),visual,ref,old)
            self.assertEqual(got,actual);self.assertEqual(delta['missing'],r.bind((new,)));self.assertFalse(delta['changed'])
            collect.assert_called_once_with(str(Path(td)),visual['native_runtime'],ref['inputs']['var_source'],ref['inputs']['dino_source'],visual['uep_runtime'])

    def test_changed_old_source_or_unbound_external_models_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            helper,visual,ref,old,actual,_,p=self.graph_fixture(Path(td));actual[str(p)]='0'*64
            with patch.object(helper,'collect_bindings',return_value=actual):
                with self.assertRaisesRegex(RuntimeError,'existing source changed'):r.static_source_closure(helper,str(Path(td)),visual,ref,old)
            del old[next(k for k in old if '/VAR/' in k.replace('\\','/'))]
            with self.assertRaisesRegex(RuntimeError,'External model source'):r.static_source_closure(helper,str(Path(td)),visual,ref,old)

    def test_renamed_native_directory_cannot_replace_original(self):
        with tempfile.TemporaryDirectory() as td:
            helper,visual,ref,old,*_=self.graph_fixture(Path(td));visual['native_runtime']='/other'
            with self.assertRaisesRegex(RuntimeError,'location differs'):r.static_source_closure(helper,str(Path(td)),visual,ref,old)

    def test_eight_native_input_categories_are_passed_and_bound(self):
        tree=ast.parse(Path(r.__file__).read_text())
        names={k.value for node in ast.walk(tree) if isinstance(node,ast.Dict) for k in node.keys if isinstance(k,ast.Constant)}
        text=Path(r.__file__).read_text()
        for key in ('assets_registration','assets_completion','assets_manifest','calibration_registration','static_closure_module',
                    'numerical_reference','source_driver_module','visual_reference_config'):
            self.assertIn(key,text)
        self.assertNotIn('torch',[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names])
        self.assertNotIn('.run(',text);self.assertNotIn('build_native(',text)

    def test_registration_never_starts_on_nonlinux(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';at.rewrite(p,{'schema':'H_FULL1000_SOURCE_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])


if __name__=='__main__':unittest.main()
