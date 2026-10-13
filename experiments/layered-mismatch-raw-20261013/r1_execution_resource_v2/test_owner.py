"""Finite owner integration with fake children only; no runtime/model/PHY."""
import ast
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import owner
import shared as s
import registration_builder as builder


def actual_stage_contract(stage):
    filename, function = {
        'phy': (s.EXPERIMENT/'r1_phy.py', 'run'),
        'projection': (Path(__file__).with_name('receiver_visual.py'), 'project'),
        'visual': (Path(__file__).with_name('receiver_visual.py'), 'visual'),
        'score': (Path(__file__).with_name('score_selection.py'), 'score_worker'),
        'selection': (Path(__file__).with_name('score_selection.py'), 'select'),
    }[stage]
    tree = ast.parse(filename.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    statuses = {kw.value.value for node in ast.walk(fn) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg == 'status' and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str) and kw.value.value.startswith('COMPLETE_R1_')}
    if len(statuses) != 1:
        raise AssertionError('Ambiguous actual worker completion contract: ' + stage)
    cli = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument' and n.args
           and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)}
    return statuses.pop(), cli


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.root = self.base/'runtime'; self.out = self.base/'results'
        self.regpin = dict(path='/synthetic/registration.json', sha256='a'*64)
        self.phy = dict(path='/synthetic/phy.json', sha256='b'*64)
        self.visual = dict(path='/synthetic/visual.json', sha256='c'*64)
        self.runtime = dict(root=str(self.root),python_cpu='original_cpu_python',python_visual='original_visual_python',
            target_gpu=3,cpu_affinity=[0,1],phy_cpu_affinity=[2,3],deadline_unix=owner.time.time()+500,
            max_seconds=500,stop_file=str(self.out/'STOP'))
        self.runtime.update({stage+'_directory':str(self.root/stage) for stage in owner.STAGES})
        self.reg = dict(schema=s.MASTER_SCHEMA,runtime=self.runtime,phy_registration=self.phy,
                        visual_registration=self.visual,expected_physical_frames=1,expected_logical_var_images=2)
        self.calls=[]; self.locks=[]

    def window(self, *_):
        p=self.base/('lock'+str(len(self.locks))); lock=p.open('w+');self.locks.append(lock)
        return lock,dict(status='READY')

    def make_popen(self, failed=None):
        test=self
        class FakeChild:
            def __init__(self, argv, **kw):
                stage=owner.STAGES[len(test.calls)];test.calls.append((stage,argv,kw));self.pid=900+len(test.calls)
                self.code=1 if stage==failed else 0
                status,flags=actual_stage_contract(stage)
                for token in argv:
                    if token.startswith('--'):
                        test.assertIn(token,flags,'owner flag is absent from actual child parser')
                test.assertEqual(len(kw['pass_fds']),2)
                test.assertEqual(len(set(kw['pass_fds'])),2)
                test.assertTrue(all(isinstance(fd,int) and fd>=0 for fd in kw['pass_fds']))
                d=Path(test.runtime[stage+'_directory']);d.mkdir()
                if self.code:
                    s.write(d/'failure.json',dict(status='SYNTHETIC_SOFTWARE_FAILURE'))
                    return
                expected=test.phy if stage=='phy' else (test.visual if stage=='visual' else test.regpin)
                done=dict(status=status,registration=expected)
                # The fixture uses actual child status literals and documented
                # receipt field names, rather than owner's status constants.
                empty=s.write(d/'synthetic_rows.json',[])
                if stage=='phy':
                    done.update(physical_frames=empty,logical_frames=empty,ledger=dict(physical_complete=1,
                        physical_reserved=1,packet_decodes_complete=2,packet_decodes_reserved=2,unresolved_packets=0),
                        reconstruction='NOT_RUN',metrics='NOT_RUN',policy_selection='NOT_RUN',R2='NOT_RUN')
                elif stage=='projection':
                    done.update(model_inputs=empty,logical_join=empty,audit=empty,visual_input_completion=empty,
                                logical_count=2,scientific_calls=0)
                elif stage=='visual':
                    done.update(image_rows=empty,logical_rows=2,model_inputs=empty,model_bundle=empty,
                                ledger_file=empty,input_access_audit=empty,source_truth_supplied_to_RX=False)
                elif stage=='score':
                    done.update(scored_rows=empty,source_preparations=empty,logical_rows=2,unique_actual_pairs=1,
                                source_count=200,noise_count=3,bootstrap_calls=0,ConvNeXt_calls=0)
                else:
                    done.update(selection=empty,source_means=empty,score_completion=empty,score_actual_wait=empty,
                                scientific_calls=0,bootstrap_calls=0,R2='NOT_REGISTERED_NOT_RUN')
                s.write(d/'completion.json',done)
            def wait(self,timeout=None):return self.code
            def poll(self):return self.code
            def terminate(self):test.fail('successful fake child should not be terminated')
            def kill(self):test.fail('successful fake child should not be killed')
        return FakeChild

    def patches(self, failed=None, window=None):
        stack=contextlib.ExitStack()
        fake_fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_SH=2,LOCK_NB=4,flock=lambda *a:None)
        stack.enter_context(mock.patch.dict(sys.modules,{'fcntl':fake_fcntl}))
        stack.enter_context(mock.patch.object(s,'OUT',self.out))
        stack.enter_context(mock.patch.object(s,'validate_registration',return_value=copy.deepcopy(self.reg)))
        stack.enter_context(mock.patch.object(s,'process_identity',side_effect=lambda pid:dict(pid=pid,start_ticks=77,state='S')))
        stack.enter_context(mock.patch.object(os,'sched_getaffinity',return_value={0,1,2,3},create=True))
        stack.enter_context(mock.patch.object(owner,'acquire_window',side_effect=window or self.window))
        stack.enter_context(mock.patch.object(owner.subprocess,'Popen',side_effect=self.make_popen(failed)))
        return stack

    def test_five_stage_actual_cli_schema_integration(self):
        with self.patches():result=owner.run(self.regpin)
        self.assertEqual([x[0] for x in self.calls],list(owner.STAGES))
        closed=s.readpin(result);self.assertEqual(closed['status'],'COMPLETE_R1_CHAIN')
        self.assertEqual(closed['R2'],'NOT_REGISTERED_NOT_RUN')
        self.assertEqual(len(closed['stages']),5)
        self.assertEqual(self.calls[0][1][0],'original_cpu_python')
        self.assertEqual(self.calls[2][1][0],'original_visual_python')
        self.assertEqual(self.calls[3][1][0],'original_visual_python')
        self.assertIn(self.phy['path'],self.calls[0][1]);self.assertIn(self.visual['path'],self.calls[2][1])
        for stage,argv,kw in self.calls:
            waited=s.read(self.root/stage/'actual_wait.json')
            owner.actual_success(waited)
            self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'3' if stage in ('visual','score') else '')
            self.assertEqual(kw['env']['OMP_NUM_THREADS'],'6' if stage in ('visual','score') else '2')
        self.assertEqual(s.read(self.root/'owner_launch.json')['deadline_unix'],self.runtime['deadline_unix'])

    def test_failed_stage_preserved_and_never_reentered(self):
        with self.patches(failed='visual'):
            with self.assertRaises(ValueError):owner.run(self.regpin)
            before=list(self.calls)
            with self.assertRaises(ValueError):owner.run(self.regpin)
        self.assertEqual(self.calls,before)
        self.assertEqual([x[0] for x in self.calls],['phy','projection','visual'])
        self.assertEqual(s.read(self.root/'visual/actual_wait.json')['returncode'],1)
        self.assertEqual(len(s.read(self.root/'owner_failure.json')['completed_stages']),2)
        self.assertFalse((self.root/'score').exists())

    def test_prelaunch_wait_has_no_scientific_launch_and_no_deadline_reset(self):
        with self.patches(window=lambda *a:(None,dict(status='WAIT',reason='Synthetic timing priority'))):
            result=owner.run(self.regpin)
        self.assertFalse(self.calls);self.assertFalse((self.root/'owner_launch.json').exists())
        self.assertEqual(result['scientific_calls'],0);self.assertFalse(result['dispatched'])
        self.assertEqual(s.read(self.out/'resource_wait.json')['deadline_unix'],self.runtime['deadline_unix'])

    def test_timing_priority_only_registered_gpu0(self):
        reg=dict(GPU_index=0,GPU_memory_fraction_cap_GiB=16,free_safety_GiB=2,initial_GPU_utilization_max_percent=25)
        def query(*a,**k):return types.SimpleNamespace(stdout='0,10000,0\n1,8000,0\n2,5000,1\n3,60000,0\n')
        ready,_=owner.timing_gpu_eligible(query,reg);self.assertFalse(ready)
        def query0(*a,**k):return types.SimpleNamespace(stdout='0,18432,25\n1,0,0\n2,0,0\n3,60000,0\n')
        self.assertTrue(owner.timing_gpu_eligible(query0,reg)[0])

    def test_expired_deadline_no_child(self):
        self.reg['runtime']['deadline_unix']=owner.time.time()-1
        with self.patches(),self.assertRaises(ValueError):owner.run(self.regpin)
        self.assertFalse(self.calls)

    def test_prelaunch_stop_no_child(self):
        self.out.mkdir();(self.out/'STOP').touch()
        with self.patches(),self.assertRaises(ValueError):owner.run(self.regpin)
        self.assertFalse(self.calls);self.assertFalse((self.root/'owner_launch.json').exists())

    def test_receipt_write_failure_still_waits_already_exited_child(self):
        actual_write=s.write
        def fail_launch(path,value):
            if Path(path).name=='phy_launch.json':raise OSError('Synthetic receipt I/O failure')
            return actual_write(path,value)
        with self.patches(),mock.patch.object(s,'write',side_effect=fail_launch),self.assertRaises(OSError):
            owner.run(self.regpin)
        self.assertEqual([row[0] for row in self.calls],['phy'])
        wait=s.read(self.root/'interrupted_actual_wait.json')
        self.assertTrue(wait['actual_wait']);self.assertEqual(wait['returncode'],0)
        self.assertFalse(wait['terminated']);self.assertEqual(wait['stage'],'phy')
        self.assertTrue((self.root/'owner_failure.json').exists())

    def test_visual_projection_excludes_source_paths_and_selection(self):
        master={k:{} for k in ('plan_id','caps','numeric_flags','resource_policy','runtime','code_bindings','models','device_receipt')}
        master.update(automatic_retry=False,automatic_R2=False,training_updates=0,additional_qualification_frames=0,
                      bootstrap_calls=0,expected_physical_frames=1,expected_logical_var_images=2,
                      pins=dict(common_ledger={'path':'codeonly'},cal200_sources={'path':'SECRET_SOURCE'}),
                      visual_request={'path':'OLD_SOURCE_RECIPE'},sources=['SECRET_TOKENS'],metric_config={})
        projected=builder.visual_projection(master,{'path':'MODEL_ONLY'})
        encoded=json.dumps(projected)
        self.assertNotIn('SECRET_',encoded);self.assertNotIn('OLD_SOURCE_RECIPE',encoded)
        self.assertEqual(set(projected['pins']),{'common_ledger'})

    def test_unclosed_r0_rejected_before_actual_audits_or_nomination(self):
        with mock.patch.object(s,'readpin',return_value={'status':'NOT_COMPLETE'}), \
             mock.patch.object(builder.r0_finalize,'audit_bler') as audit, self.assertRaises(ValueError):
            builder.authenticated_r0({'path':'synthetic','sha256':'a'*64})
        audit.assert_not_called()

    def test_no_scientific_runtime_imported(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('sionna',sys.modules)


if __name__=='__main__':unittest.main()
