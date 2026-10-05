"""Receipt/configuration tests only: no GPU, model, decoder or process launch."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import register_render as r


class ClosureTests(unittest.TestCase):
    def fixture(self, root):
        owner_path=root/'owner.json'; reg_path=root/'reg.json'; launch_path=root/'launch.json'; cpu_path=root/'cpu.json'
        owner_source=root/'stage_owner.py';owner_source.write_text('# fixture\n')
        cpu_out=root/'cpu';cpu_out.mkdir(); owner_out=root/'cpu_owner';owner_out.mkdir()
        identities=[dict(pid=i,start_ticks=100+i,uid=1002,argv=['python','cpu.py',str(i)]) for i in (2,3,4)]
        owner=dict(pid=1,start_ticks=101,uid=1002,argv=['python','-B',str(owner_source.resolve()),'--config',str(owner_path)])
        completion=cpu_out/'completion.json'
        jobs=[dict(id='w'+str(i),argv=['python','-B','cpu.py','--config',str(cpu_path)],out=str(root/('w'+str(i))),
                   completion=str(root/('w'+str(i))/'completion.json')) for i in (0,1)]
        merged=dict(id='merge',argv=['python','-B','cpu.py','--config',str(cpu_path)],out=str(cpu_out),completion=str(completion))
        cfg=dict(registration=str(reg_path),owner_out=str(owner_out),out=str(root/'H'),stages=[
            dict(id='initial_true200',resource='cpu',jobs=jobs),dict(id='report',resource='cpu',jobs=[merged])])
        reg=dict(allowed_stage_ids=['initial_true200','report'])
        cpu=dict(out=str(cpu_out),registration=str(reg_path))
        for p,d in ((owner_path,cfg),(reg_path,reg),(cpu_path,cpu)):r.save(p,d)
        done=dict(status='H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE',registration_sha256=r.sha(reg_path),
            driver_config_sha256=r.sha(cpu_path),source_count=200,images_scored=False,source_decode_complete=False,
            GPU_used=False,development_used=False,worker_identities=identities[:2],outputs={},input_bindings={},source_bindings={})
        r.save(completion,done)
        r.save(launch_path,dict(identity=owner,argv=owner['argv'],registration_sha256=r.sha(reg_path),owner_config_sha256=r.sha(owner_path)))
        closure=dict(bindings={str(completion):r.sha(completion)},child_identities=identities)
        state={i:None for i in (1,2,3,4)}
        api=SimpleNamespace(__file__=str(owner_source),validate_config=lambda *args:None,
            same_identity=lambda a,b:all(a[k]==b[k] for k in ('pid','start_ticks','uid','argv')),
            verify=lambda values:[self.assertEqual(r.sha(p),s) for p,s in values.items()])
        prior=SimpleNamespace(exited=lambda expected,reader:reader(expected['pid']) is None,
            verify_batch=lambda *args:copy.deepcopy(closure))
        return dict(a=api,prior=prior,config_path=owner_path,registration_path=reg_path,launch_path=launch_path,
            cpu_config_path=cpu_path,state_reader=lambda pid:state[pid]), state, closure, completion

    def test_closed_owner_workers_and_exact_merge_admitted(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,closure,cp=self.fixture(Path(td)); result=r.verify_cpu_closed(**args)
            self.assertEqual(result['status'],'H_INITIAL_CPU_OWNER_AND_ALL_CHILDREN_CLOSED')
            self.assertEqual(result['cpu_completion'],str(cp)); self.assertEqual(len(result['child_identities']),3)

    def test_live_or_unreaped_owner_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,_,_=self.fixture(Path(td))
            for status in ('R','Z'):
                state[1]={'state':status}
                with self.assertRaisesRegex(RuntimeError,'owner is live or unreaped'):r.verify_cpu_closed(**args)

    def test_worker_present_or_not_in_closure_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,closure,_=self.fixture(Path(td));state[2]={'state':'Z'}
            with self.assertRaisesRegex(RuntimeError,'worker not closed'):r.verify_cpu_closed(**args)
            state[2]=None;closure['child_identities']=closure['child_identities'][1:]
            with self.assertRaisesRegex(RuntimeError,'worker not closed'):r.verify_cpu_closed(**args)

    def test_changed_launch_and_unbound_merge_block(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,closure,cp=self.fixture(Path(td));closure['bindings'][str(cp)]='0'*64
            with self.assertRaisesRegex(RuntimeError,'merge receipt is unbound'):r.verify_cpu_closed(**args)
            closure['bindings'][str(cp)]=r.sha(cp)
            launch=r.read(args['launch_path']);launch['owner_config_sha256']='0'*64
            args['launch_path'].write_text(json.dumps(launch))
            with self.assertRaisesRegex(RuntimeError,'launch changed'):r.verify_cpu_closed(**args)

    def test_cpu_scored_claim_or_missing_source_decode_field_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,closure,cp=self.fixture(Path(td)); done=r.read(cp);done['source_decode_complete']=True
            cp.write_text(json.dumps(done));closure['bindings'][str(cp)]=r.sha(cp)
            with self.assertRaisesRegex(RuntimeError,'expected scope'):r.verify_cpu_closed(**args)
            del done['source_decode_complete'];cp.write_text(json.dumps(done));closure['bindings'][str(cp)]=r.sha(cp)
            with self.assertRaises(KeyError):r.verify_cpu_closed(**args)

    def test_stop_or_preserved_merge_failure_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            args,state,closure,cp=self.fixture(Path(td));failure=cp.parent/'merge_failure.json';failure.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'failure/STOP'):r.verify_cpu_closed(**args)
            failure.unlink(); owner=r.read(args['config_path']); stop=Path(owner['owner_out'])/'STOP';stop.write_text('stop')
            with self.assertRaisesRegex(RuntimeError,'STOP blocks'):r.verify_cpu_closed(**args)


class ConfigurationTests(unittest.TestCase):
    def test_single_gpu_keeps_original_visual_contract_and_independent_paths(self):
        previous=dict(out='/H',gpu_threads=6,gpu_affinity=[4,5,6,7,8,9],gpu_device=0,visual_lock_path='/shared_visual.lock',
            cpu_affinities=[[0,1],[2,3]],phase_limits=r.PHASES,prior_qualification_gate={'path':'old_q','sha256':'q'},
            S1_gate={'path':'old_s1','sha256':'s'},stages=[{'id':'old','resource':'cpu'}])
        cpu={k:k for k in ('root','protocol','catalogue','shortlist','source200','S1_completion','S1_assets_completion',
                          'budget_registration','ledger','stop_file')}
        cpu.update(phase_limits=r.PHASES,runtime_dir='/runtime',out='/H/initial')
        visual=dict(uep_runtime='/oldUEP',native_runtime='/oldM1',calibration_registration='/calibration')
        request=dict(render_out='/H/render',owner_module='/runtime/stage_owner.py',wait_module='/runtime/wait_then_coarse.py',
            cpu_config={'path':'cpu_config'},cpu_registration={'path':'cpu_registration'},
            cpu_owner_config={'path':'cpu_owner_config'},cpu_owner_launch={'path':'cpu_owner_launch'},
            max_seconds=14400,python='/python')
        cfg,owner=r.make_configs(request,previous,cpu,visual,Path('/H/render_execution'))
        self.assertEqual([s['id'] for s in owner['stages']],['render']);self.assertEqual(owner['stages'][0]['resource'],'gpu')
        self.assertEqual(len(owner['stages'][0]['jobs']),1)
        for key in ('gpu_threads','gpu_affinity','gpu_device','visual_lock_path','phase_limits','prior_qualification_gate','S1_gate'):
            self.assertEqual(owner[key],previous[key])
        self.assertEqual(cfg['calibration_registration'],visual['calibration_registration'])
        self.assertEqual(cfg['overall_deadline_unix'],r.DEADLINE)
        self.assertEqual(cfg['cpu_completion'],str(Path('/H/initial/completion.json')))
        self.assertNotIn('registration_sha256',cfg);self.assertNotIn('owner_config_sha256',owner)
        self.assertEqual(previous['stages'],[{'id':'old','resource':'cpu'}])
        self.assertEqual(owner['stages'][0]['jobs'][0]['receipt_expect']['new_packet_decodes'],0)

    def test_conflicting_bindings_and_immutable_overwrite_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Conflicting'):r.merge({'/file':'a'},{'/file':'b'})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sealed.json';r.save(p,{'a':1})
            with self.assertRaises(FileExistsError):r.save(p,{'a':2})
            self.assertEqual(r.read(p),{'a':1})
            with self.assertRaisesRegex(RuntimeError,'Pinned input changed'):r.pinned({'path':str(p),'sha256':'0'*64})

    def test_actual_registration_requires_linux_before_any_gate_or_write(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json';r.save(p,{'schema':'H_RENDER_REGISTRATION_REQUEST_V1'})
            with patch.object(r.sys,'platform','win32'):
                with self.assertRaisesRegex(RuntimeError,'Linux process'):r.prepare(str(p))
            self.assertEqual([x.name for x in Path(td).iterdir()],['request.json'])


if __name__=='__main__':unittest.main()
