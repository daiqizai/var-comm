"""CPU synthetic receipt/metadata tests; no arrays, codec, PHY, GPU or jobs."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import register_full1000_assets as r


def rewrite(path,value):Path(path).write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')
def budget():return dict(created=True,charged=69960,phase_charged=dict(r.COUNTS),unresolved=0,failed=0,development_remaining=13200)


class RegistrationTests(unittest.TestCase):
    def fixture(self,root):
        ownerfile=root/'stage_owner.py';ownerfile.write_text('# synthetic owner\n')
        workerfile=root/'selector.py';workerfile.write_text('# synthetic CPU selector\n')
        h=root/'H';h.mkdir();base=h/'execution';base.mkdir();out=h/'selected';out.mkdir()
        paths={k:base/name for k,name in dict(selection_owner_config='owner_config.json',
            selection_registration='execution_registration.json',selection_launch='launch.json',
            selection_owner_completion='completion.json',selection_config='selection_config.json').items()}
        paths['selection_completion']=out/'completion.json'
        cfg=dict(branch='H',root=str(root),out=str(h),owner_out=str(base),registration=str(paths['selection_registration']),
            phase_limits=dict(r.PHASES),overall_deadline_unix=r.DEADLINE,
            cpu_affinities=[[0,1],[2,3]],gpu_affinity=[4,5,6,7,8,9],gpu_threads=6,
            qualification_started_unix=1791219005.9549868,qualification_deadline_unix=r.DEADLINE,
            stages=[dict(id='freeze',resource='cpu',requires=[],max_seconds=100,jobs=[dict(id='initial_selection',
                argv=[sys.executable,'-B',str(workerfile),'--config',str(paths['selection_config'])],cwd=str(root),
                out=str(out),completion=str(paths['selection_completion']),accepted_statuses=['H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE'])])])
        rewrite(paths['selection_owner_config'],cfg)
        rewrite(paths['selection_config'],dict(root=str(root),out=str(out),registration=cfg['registration'],owner_config=str(paths['selection_owner_config'])))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256=r.sha(paths['selection_owner_config']),
            allowed_stage_ids=['freeze'],phase_limits=r.PHASES,source_bindings=r.bind((ownerfile,workerfile)),
            input_bindings=r.bind((paths['selection_config'],)))
        rewrite(paths['selection_registration'],reg);regsha=r.sha(paths['selection_registration'])
        oid=dict(pid=100,uid=1002,start_ticks=1000,argv=[sys.executable,'-B',str(ownerfile),'--config',str(paths['selection_owner_config'])])
        rewrite(paths['selection_launch'],dict(identity=oid,argv=oid['argv'],registration_sha256=regsha,
            owner_config_sha256=r.sha(paths['selection_owner_config'])))
        rewrite(base/'owner_identity.json',dict(oid,registration_sha256=regsha,config_sha256=r.sha(paths['selection_owner_config'])))
        result=out/'selected_whole.json';rewrite(result,{'selected_count':8})
        done=dict(status='H_INITIAL_TRUE200_WHOLE_SELECTION_COMPLETE',registration_sha256=regsha,source_count=200,
            measured_frames=9600,candidate_count=16,selected_count=8,partial_reselected=False,new_packet_decodes=0,
            new_visual_inference=0,GPU_used=False,development_used=False,holdout_used=False,full1000_calibration_complete=False,
            original_render_owner_success=False,outputs=r.bind((result,)),budget=budget())
        rewrite(paths['selection_completion'],done)
        wp=base/'stages/freeze/workers/initial_selection';wp.mkdir(parents=True)
        worker=dict(pid=101,uid=1002,start_ticks=1001,argv=cfg['stages'][0]['jobs'][0]['argv'])
        log=wp/'worker.log';log.write_bytes(b'closed\n')
        event=dict(job_id='initial_selection',identity=worker,exit_code=0,registration_sha256=regsha,
            completion=str(paths['selection_completion']),completion_sha256=r.sha(paths['selection_completion']),closed_log_sha256=r.sha(log))
        rewrite(wp/'launch.json',dict(identity=worker,registration_sha256=regsha));rewrite(wp/'exit_receipt.json',event)
        stage=base/'stages/freeze/completion.json';rewrite(stage,dict(status='REGISTERED_STAGE_COMPLETE',stage='freeze',
            registration_sha256=regsha,budget=budget(),jobs=[event]))
        rewrite(paths['selection_owner_completion'],dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=regsha,
            allowed_stage_ids=['freeze'],qualification_passed=True,future_stages_started=False,H_full_delivery_claimed=False,
            C_started=False,holdout_started=False,budget=budget(),completed=[dict(stage='freeze',completion=str(stage),sha256=r.sha(stage))]))
        here=Path(__file__).resolve().parent
        runner=here if (here/'stage_owner.py').is_file() else here.parent/'runner'
        self.assertTrue((runner/'stage_owner.py').is_file() and (runner/'wait_then_coarse.py').is_file())
        real=r.module(runner/'stage_owner.py','test_full1000_original_owner')
        prior=r.module(runner/'wait_then_coarse.py','test_full1000_original_closure')
        api=SimpleNamespace(__file__=str(ownerfile),validate_config=real.validate_config,read=r.read,sha=r.sha,
                            require=r.require,receipt=real.receipt,verify=r.verify)
        states={100:None,101:None}
        return dict(owner=api,prior=prior,paths=paths,state_reader=lambda pid:states[pid]),states,wp

    def test_real_normal_closure_and_original_failed_render_truth_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));got=r.verify_selection(**args)
            self.assertTrue(got['closure']['owner_exited']);self.assertFalse(got['closure']['original_render_owner_success'])
            self.assertEqual(len(got['closure']['child_identities']),1)

    def test_owner_and_worker_live_or_zombie_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,_=self.fixture(Path(td))
            for pid in (100,101):
                for state in ('R','Z'):
                    states[pid]=dict(pid=pid,uid=1002,start_ticks=1000 if pid==100 else 1001,state=state,
                        argv=r.read(args['paths']['selection_launch'])['argv'] if pid==100 else r.read(args['paths']['selection_owner_config'])['stages'][0]['jobs'][0]['argv'])
                    with self.assertRaisesRegex(RuntimeError,'live|unreaped|reap'):r.verify_selection(**args)
                states[pid]=None

    def test_failure_stop_and_tampered_closed_log_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,wp=self.fixture(Path(td));base=args['paths']['selection_owner_completion'].parent
            for p in (base/'failure.json',base.parent/'STOP'):
                p.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'failure|STOP'):r.verify_selection(**args)
                p.unlink()
            (wp/'worker.log').write_bytes(b'mutated\n')
            with self.assertRaisesRegex(RuntimeError,'log changed'):r.verify_selection(**args)

    def test_unbound_or_incomplete_science_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));path=args['paths']['selection_completion'];done=r.read(path)
            done['measured_frames']=9599;rewrite(path,done)
            with self.assertRaisesRegex(RuntimeError,'completion failed/changed'):r.verify_selection(**args)

    def test_launch_identity_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));path=args['paths']['selection_launch'];data=r.read(path)
            data['identity']['start_ticks']+=1;rewrite(path,data)
            with self.assertRaisesRegex(RuntimeError,'identity differs'):r.verify_selection(**args)

    def test_exact_budget_all_caps_reserve_and_unresolved(self):
        r.check_budget(budget())
        for key,value in (('charged',69961),('unresolved',1),('failed',1),('development_remaining',13199)):
            data=budget();data[key]=value
            with self.assertRaises(RuntimeError):r.check_budget(data)
        data=budget();data['phase_charged']['engineering_reserve']=1
        with self.assertRaises(RuntimeError):r.check_budget(data)

    def test_shards_sidecar_bound_and_no_array_load(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);latent=root/'latent.pt';pixel=root/'pixel.pt';side=latent.with_suffix('.json')
            latent.write_bytes(b'not a tensor; must only hash');pixel.write_bytes(b'pixels not loaded')
            rewrite(side,{'sha256':r.sha(latent)})
            row=dict(latent_shard=str(latent),latent_sha256=r.sha(latent),pixel_shard=str(pixel),pixel_sha256=r.sha(pixel))
            values=r.collect_shard_bindings({'records':[row,row]})
            self.assertEqual(values,r.bind((latent,pixel,side)))
            rewrite(side,{'sha256':'0'*64})
            with self.assertRaisesRegex(RuntimeError,'sidecar'):r.collect_shard_bindings({'records':[row]})

    def test_missing_or_changed_shard_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'calibration.pt';p.write_bytes(b'changed')
            row=dict(latent_shard=str(p),latent_sha256='0'*64,pixel_shard=str(p),pixel_sha256='0'*64)
            with self.assertRaisesRegex(RuntimeError,'Changed immutable'):r.collect_shard_bindings({'records':[row]})

    def test_generated_config_is_one_cpu_source_job_no_hash_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));old=r.read(args['paths']['selection_owner_config'])
            paths={k:Path(td)/(k+'.json') for k in (*r.INPUT_NAMES,'s1_source_ids')}
            paths['asset_module']=Path(__file__).resolve().parent/'full1000_assets.py'
            ctx=dict(old=old,paths=paths,plan={'visual_identity':{'frozen':'identity'}},adapter=SimpleNamespace(DONE='H_FULL1000_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'))
            req=dict(execution_dir=str(Path(td)/'H/new_execution'),assets_out=str(Path(td)/'H/new_assets'),python=sys.executable,max_seconds=3600)
            cfg,owner=r.make_configs(req,ctx)
            self.assertEqual([(s['id'],s['resource']) for s in owner['stages']],[('source','cpu')])
            job=owner['stages'][0]['jobs'][0]
            self.assertEqual(job['id'],'full1000_assets');self.assertEqual(job['receipt_expect']['pending_source_encoding_count'],800)
            self.assertFalse(job['receipt_expect']['source1000_codec_complete'])
            self.assertEqual(owner['phase_limits'],r.PHASES);self.assertEqual(owner['cpu_affinities'],old['cpu_affinities'])
            self.assertNotIn('registration_sha256',cfg);self.assertNotIn('owner_config_sha256',owner)
            self.assertEqual(old['stages'][0]['id'],'freeze')
            fake=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256='fixture',allowed_stage_ids=['source'],
                phase_limits=r.PHASES,source_bindings=r.bind((paths['asset_module'],)))
            args['owner'].validate_config(owner,fake,'fixture')

    def test_nonlinux_does_not_start_or_write(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';rewrite(p,{'schema':'H_FULL1000_ASSETS_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.register(p)
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])

    def test_no_model_phy_or_subprocess_import_or_call(self):
        source=Path(r.__file__).read_text();tree=ast.parse(source)
        imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [x.name for node in ast.walk(tree) if isinstance(node,ast.Import) for x in node.names]
        self.assertFalse(set(imports)&{'torch','tensorflow','subprocess','h64_phy','h64_source'})
        self.assertNotIn('.source(',source);self.assertNotIn('.run(',source)


if __name__=='__main__':unittest.main()
