"""Synthetic registered closeout evidence; no process, GPU, image or PHY work."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import register_initial_selection_r2 as r
import test_register_initial_selection as original_tests


def rewrite(path,value):Path(path).write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')


class CertifiedCloseoutTests(unittest.TestCase):
    def fixture(self,root):
        args,states,unused,done_path=original_tests.SelectionRegistrationTests().fixture(root)
        cfg=r.read(args['owner_path']);base=Path(cfg['owner_out']);job=cfg['stages'][0]['jobs'][0]
        job.update(id='render',accepted_statuses=['H_INITIAL_TRUE200_RX_COMPLETE'])
        rewrite(args['owner_path'],cfg)
        launch=r.read(args['launch_path']);launch['owner_config_sha256']=r.sha(args['owner_path']);rewrite(args['launch_path'],launch)
        oid=r.read(base/'owner_identity.json');oid['config_sha256']=r.sha(args['owner_path']);rewrite(base/'owner_identity.json',oid)
        failure=base/'failure.json';r.save(failure,dict(status='FAILED',error='Process no longer live'))
        child=dict(pid=101,uid=1002,start_ticks=1001,argv=job['argv'])
        worker=base/'stages/render/workers/render';worker.mkdir(parents=True)
        log=worker/'worker.log';log.write_bytes(b'complete\n')
        event=dict(identity=child,exit_code=0,registration_sha256=r.sha(args['reg_path']),completion=str(done_path),
                   completion_sha256=r.sha(done_path),closed_log_sha256=r.sha(log))
        ep=worker/'exit_receipt.json';lp=worker/'launch.json';r.save(ep,event)
        r.save(lp,dict(identity=child,registration_sha256=r.sha(args['reg_path'])))
        close=root/'closeout';close.mkdir();reg=close/'execution_registration.json';certificate=close/'completion.json'
        release=close/'release_receipt.json';terminal=close/'closeout_completion.json';preserved=close/'STOP.original'
        preserved.write_bytes(b'original owned STOP\n')
        local_stop=Path(job['out'])/'STOP';local_preserved=close/'renderer_STOP.original';local_preserved.write_bytes(b'original owned STOP\n')
        supplement_copy=root/'supplement_STOP.original';supplement_copy.write_bytes(b'original owned STOP\n')
        supplement=root/'renderer_stop_evidence.json';r.save(supplement,dict(status='RENDER_LOCAL_STOP_EVIDENCE_PRESERVED',
            original_path=str(local_stop),preserved_path=str(supplement_copy),sha256=r.sha(supplement_copy),owner_failure_sha256=r.sha(failure)))
        request=root/'closeout_request.json';r.save(request,dict(renderer_stop_evidence={'path':str(supplement),'sha256':r.sha(supplement)}))
        budget=dict(created=True,charged=69960,phase_charged=dict(r.EXPECTED_CHARGES),unresolved=0,failed=0)
        dependencies=r.bind((args['owner_path'],args['reg_path'],args['launch_path'],args['render_path'],
            base/'owner_identity.json',failure,ep,lp,log,done_path))
        dependencies.update(r.bind((supplement,supplement_copy,request)))
        r.save(reg,dict(status='H_RENDER_CLOSEOUT_READ_ONLY_REGISTERED',input_bindings=dependencies,source_bindings={},
            request={'path':str(request),'sha256':r.sha(request)},budget_before=budget,
            expected_stop={'path':str(Path(cfg['out'])/'STOP'),'sha256':r.sha(preserved)},
            expected_stops=[{'path':str(Path(cfg['out'])/'STOP'),'sha256':r.sha(preserved)},
                            {'path':str(local_stop),'sha256':r.sha(local_preserved)}]))
        cert=dict(status='H_RENDER_OUTPUTS_AND_EXIT_CERTIFIED_AFTER_OWNER_RACE',original_owner_success=False,
            scientific_outputs_complete=True,original_registration_sha256=r.sha(args['reg_path']),
            closeout_registration_sha256=r.sha(reg),source_count=200,frame_count=9600,new_packet_decodes=0,GPU_used=False,
            original_owner_config={'path':str(args['owner_path']),'sha256':r.sha(args['owner_path'])},
            original_launch={'path':str(args['launch_path']),'sha256':r.sha(args['launch_path'])},
            render_completion={'path':str(done_path),'sha256':r.sha(done_path)},
            original_failure={'path':str(failure),'sha256':r.sha(failure)},
            worker_exit_receipt={'path':str(ep),'sha256':r.sha(ep)},
            owner_identity=launch['identity'],worker_identity=child,bindings=dependencies,outputs={},source_bindings={},budget=budget)
        r.save(certificate,cert)
        released=dict(status='H_RENDER_CLOSEOUT_EXACT_STOP_ARCHIVED',
            certificate_sha256=r.sha(certificate),closeout_registration_sha256=r.sha(reg),
            original_path=str(Path(cfg['out'])/'STOP'),preserved_path=str(preserved),sha256=r.sha(preserved),released=True,
            original_owner_success=False,old_owner_and_worker_exited=True,budget_unchanged=True,
            released_markers=[dict(original_path=str(Path(cfg['out'])/'STOP'),preserved_path=str(preserved),sha256=r.sha(preserved)),
                              dict(original_path=str(local_stop),preserved_path=str(local_preserved),sha256=r.sha(local_preserved))])
        r.save(release,released)
        r.save(terminal,dict(status='H_RENDER_CLOSEOUT_COMPLETE_STOP_RELEASED',closeout_registration_sha256=r.sha(reg),
            original_owner_success=False,scientific_outputs_complete=True,original_owner_completion_written=False,
            new_packet_decodes=0,new_visual_inference=0,GPU_used=False,budget=budget,original_registration_sha256=r.sha(args['reg_path']),
            outputs=r.bind((certificate,release,preserved,local_preserved))))
        args['owner'].same_identity=lambda x,y:all(x[k]==y[k] for k in ('pid','uid','start_ticks','argv'))
        args['owner'].receipt=lambda *args:None
        def prohibited(*args):raise AssertionError('Cannot fabricate successful old verify_batch')
        args['prior'].verify_batch=prohibited
        paths=dict(closeout_registration=reg,closeout_certificate=certificate,stop_release=release,closeout_completion=terminal)
        args['closeout_paths']=paths
        return args,states,paths

    def rebind(self,paths):
        release=r.read(paths['stop_release']);release['certificate_sha256']=r.sha(paths['closeout_certificate'])
        rewrite(paths['stop_release'],release)
        final=r.read(paths['closeout_completion']);final['outputs']=r.bind((paths['closeout_certificate'],paths['stop_release'],
            *(Path(x['preserved_path']) for x in release['released_markers'])))
        rewrite(paths['closeout_completion'],final)

    def test_truthful_failed_owner_closeout_admitted_without_verify_batch(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));value=r.verify_render_closed(**args)
            self.assertFalse(value['original_owner_success']);self.assertEqual(len(value['child_identities']),1)
            self.assertEqual(value['status'],'H_RENDER_SCIENCE_CERTIFIED_AFTER_FAILED_OWNER')

    def test_certificate_alone_without_terminal_release_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));paths['closeout_completion'].unlink()
            with self.assertRaises(FileNotFoundError):r.verify_render_closed(**args)

    def test_lying_owner_success_and_wrong_budget_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));original=r.read(paths['closeout_certificate'])
            for field,value in (('original_owner_success',True),('new_packet_decodes',1),('GPU_used',True)):
                cert=copy.deepcopy(original);cert[field]=value;rewrite(paths['closeout_certificate'],cert);self.rebind(paths)
                with self.assertRaisesRegex(RuntimeError,'changed original science'):r.verify_render_closed(**args)
            cert=copy.deepcopy(original);cert['budget']['charged']=69961;rewrite(paths['closeout_certificate'],cert);self.rebind(paths)
            with self.assertRaisesRegex(RuntimeError,'ledger differs'):r.verify_render_closed(**args)

    def test_fabricated_old_owner_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_=self.fixture(Path(td));owner=r.read(args['owner_path']);r.save(Path(owner['owner_out'])/'completion.json',{'status':'COMPLETE'})
            with self.assertRaisesRegex(RuntimeError,'fabricated success'):r.verify_render_closed(**args)

    def test_live_or_unreaped_original_process_still_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,states,_=self.fixture(Path(td))
            for pid in (100,101):
                states[pid]={'state':'Z'}
                with self.assertRaisesRegex(RuntimeError,'unreaped|still present'):r.verify_render_closed(**args)
                states[pid]=None

    def test_new_global_stop_and_even_bound_unreleased_local_stop_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));cfg=r.read(args['owner_path']);global_stop=Path(cfg['out'])/'STOP'
            global_stop.parent.mkdir();global_stop.write_text('new STOP')
            with self.assertRaisesRegex(RuntimeError,'global or renderer STOP'):r.verify_render_closed(**args)
            global_stop.unlink();local=Path(cfg['stages'][0]['jobs'][0]['out'])/'STOP';local.write_text('historical STOP')
            with self.assertRaisesRegex(RuntimeError,'global or renderer STOP'):r.verify_render_closed(**args)
            cert=r.read(paths['closeout_certificate']);cert['bindings'][str(local)]=r.sha(local)
            reg=r.read(paths['closeout_registration']);reg['input_bindings'][str(local)]=r.sha(local)
            rewrite(paths['closeout_registration'],reg);cert['closeout_registration_sha256']=r.sha(paths['closeout_registration'])
            rewrite(paths['closeout_certificate'],cert)
            release=r.read(paths['stop_release']);release['closeout_registration_sha256']=r.sha(paths['closeout_registration'])
            rewrite(paths['stop_release'],release)
            final=r.read(paths['closeout_completion']);final['closeout_registration_sha256']=r.sha(paths['closeout_registration'])
            rewrite(paths['closeout_completion'],final);self.rebind(paths)
            with self.assertRaisesRegex(RuntimeError,'global or renderer STOP'):r.verify_render_closed(**args)
            self.assertTrue(local.exists())

    def test_single_stop_release_is_insufficient(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));release=r.read(paths['stop_release']);release['released_markers']=release['released_markers'][:1]
            rewrite(paths['stop_release'],release);self.rebind(paths)
            with self.assertRaisesRegex(RuntimeError,'Both exact registered STOP'):r.verify_render_closed(**args)

    def test_supplemental_stop_evidence_changed_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));reg=r.read(paths['closeout_registration']);request=r.read(reg['request']['path'])
            evidence=Path(request['renderer_stop_evidence']['path']);evidence.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Changed immutable'):r.verify_render_closed(**args)

    def test_modified_failure_and_worker_log_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));cfg=r.read(args['owner_path']);failure=Path(cfg['owner_out'])/'failure.json'
            original=failure.read_bytes();failure.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Changed immutable'):r.verify_render_closed(**args)
            failure.write_bytes(original);log=Path(cfg['owner_out'])/'stages/render/workers/render/worker.log';log.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Changed immutable'):r.verify_render_closed(**args)

    def test_stop_release_of_different_certificate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,paths=self.fixture(Path(td));release=r.read(paths['stop_release']);release['certificate_sha256']='0'*64
            rewrite(paths['stop_release'],release)
            terminal=r.read(paths['closeout_completion']);terminal['outputs'][str(paths['stop_release'])]=r.sha(paths['stop_release'])
            rewrite(paths['closeout_completion'],terminal)
            with self.assertRaisesRegex(RuntimeError,'exact certificate'):r.verify_render_closed(**args)


class FrozenSelectionContractTests(unittest.TestCase):
    def test_scientific_selector_invocation_and_output_scope_identical_to_r1(self):
        folder=Path(__file__).parent;trees=[ast.parse((folder/name).read_text()) for name in ('register_initial_selection.py','register_initial_selection_r2.py')]
        calls=[]
        for tree in trees:
            matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='select_initial']
            self.assertEqual(len(matches),1);calls.append(ast.dump(matches[0],include_attributes=False))
        self.assertEqual(*calls)
        old=(folder/'register_initial_selection.py').read_bytes()
        self.assertEqual(__import__('hashlib').sha256(old).hexdigest(),'6b9896baa1e167c33ef64291a54e6984a3efb166273b2e09f7e3adc588d6ace3')

    def test_r2_is_one_cpu_freeze_and_same_whole_scope(self):
        old=dict(out='/H',cpu_affinities=[[0,1],[2,3]],phase_limits=r.PHASES,stages=[])
        request=dict(request_path='/request',selection_out='/H/select',max_seconds=3600,python='/python')
        cfg,owner=r.make_configs(request,old,dict(root='/repo',stop_file='/H/STOP'),Path('/H/exec'))
        self.assertEqual(cfg['schema'],'H_INITIAL_SELECTION_CONFIG_R2')
        stage=owner['stages'][0];self.assertEqual((stage['id'],stage['resource']),('freeze','cpu'))
        self.assertTrue(stage['jobs'][0]['argv'][2].endswith('register_initial_selection_r2.py'))
        self.assertEqual(stage['jobs'][0]['receipt_expect']['selected_count'],8)
        self.assertFalse(stage['jobs'][0]['receipt_expect']['full1000_calibration_complete'])


if __name__=='__main__':unittest.main()
