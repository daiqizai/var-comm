"""Synthetic CPU metadata tests only; no model, channel, quality or live receipts."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_score_ancestry_test', HERE/'score_provenance.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
WORKSPACE = HERE.parents[2]
ORIGINAL = WORKSPACE/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2/score_selection.py'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def pin(path):
    return dict(path=str(path), sha256=a.sha(path), bytes=path.stat().st_size)


class AncestryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name); self.oldroot=base/'original'; self.newroot=base/'recovery'
        self.original_path=base/'old_master.json'
        runtime=dict(root=str(self.oldroot), deadline_unix=1234567890., max_seconds=43200,
            python_cpu='/original/cpu', python_visual='/original/visual', target_gpu=3,
            cpu_affinity=[0,1], phy_cpu_affinity=[2,3])
        runtime.update({stage+'_directory':str(self.oldroot/stage) for stage in ('phy','projection','visual','score','selection')})
        self.old={k:'fixed_'+k for k in a.SCIENTIFIC_FIELDS}
        self.old.update(schema='LAYERED_RAW_R1_EXECUTION_REGISTRATION_V1',status='REGISTERED_R1_EXECUTION_V1',
            expected_physical_frames=32400,expected_logical_var_images=43200,runtime=runtime,
            pins={k:dict(path='/frozen/'+k,sha256='b'*64) for k in a.PIN_FIELDS},
            phy_registration=dict(path='/frozen/phy.json',sha256='c'*64),
            visual_registration=dict(path='/frozen/old_visual.json',sha256='d'*64))
        write(self.original_path,self.old);self.original_pin=pin(self.original_path)
        self.reg=copy.deepcopy(self.old);self.reg['recovery']=dict(original_registration=self.original_pin)
        self.reg['runtime']['root']=str(self.newroot)
        for stage in ('visual','score','selection'):self.reg['runtime'][stage+'_directory']=str(self.newroot/stage)
        self.reg['visual_registration']=dict(path='/frozen/new_visual.json',sha256='e'*64)
        self.newpin=dict(path='/frozen/new_master.json',sha256='f'*64)
        self.addCleanup(patch.stopall)
        patch.object(a,'ORIGINAL_REGISTRATION',{k:self.original_pin[k] for k in ('path','sha256')}).start()
        patch.object(a,'OLD_ROOT',self.oldroot).start()
        self.module=types.ModuleType('_private_original_scorer');self.module.__file__=str(ORIGINAL)
        self.sentinel=lambda: 'unchanged'
        self.module.__dict__.update(require=a.require,Path=Path,HERE=ORIGINAL.parent,read=read,pin=pin,
            same_pin=a.same_pin,pair_key=self.sentinel,score_rows=self.sentinel,
            source_pixels=self.sentinel,quality_summaries=self.sentinel)
        a.install(self.module)

    def dispatch(self,stage,old=False):
        root=self.oldroot if old else self.newroot;regpin=a.ORIGINAL_REGISTRATION if old else self.newpin
        child=dict(pid=111 if old else 222,start_ticks=777 if old else 888)
        parent=dict(pid=11 if old else 22,start_ticks=77 if old else 88)
        owner=dict(registration=regpin,process=parent)
        if stage=='score':
            script=ORIGINAL;action='score';flag='--registration-sha256';sub=regpin
            args=['--parent',str(parent['pid'])]
        else:
            script=ORIGINAL.parent/'receiver_visual.py';action='project' if stage=='projection' else 'visual'
            flag='--expected-registration';sub=regpin if stage=='projection' else self.reg['visual_registration']
            args=['--parent-pid',str(parent['pid']),'--parent-ticks',str(parent['start_ticks'])]
        argv=['python','-B','-u','/new/explicit_runner.py','--original',str(script),action,
            '--registration',sub['path'],flag,sub['sha256'],*args]
        launch=dict(stage=stage,registration=regpin,process=child,argv=argv)
        wait=dict(actual_wait=True,returncode=0,stage=stage,pid=child['pid'],start_ticks=child['start_ticks'],
                  stopped=False,terminated=False,killed=False,timed_out=False)
        write(root/'owner_launch.json',owner);write(root/(stage+'_launch.json'),launch)
        write(root/(stage+'_actual_wait.json'),wait);write(root/stage/'actual_wait.json',wait)
        return root,launch,wait

    def test_only_two_provenance_functions_replaced(self):
        for name in ('pair_key','score_rows','source_pixels','quality_summaries'):
            self.assertIs(getattr(self.module,name),self.sentinel)
        proof=self.module.RECOVERY_SCORE_PROVENANCE_PROOF
        self.assertEqual(proof['original_AST_sha256'],proof['inverse_restored_AST_sha256'])
        self.assertEqual(len(proof['substitutions']),3)
        self.assertIn('ThreeMetrics',proof['unchanged_functions_and_classes'])

    def test_projection_wait_uses_real_ancestor_and_keeps_failed_owner_scene(self):
        root,_,_=self.dispatch('projection',old=True)
        write(root/'owner_failure.json',dict(status='FAILED_STOP_NO_RETRY',completed_stages=['phy','projection']))
        write(root/'original_owner_actual_wait.json',dict(returncode=1,actual_wait=True))
        before={p.name:p.read_bytes() for p in root.glob('*.json')}
        result=self.module.successful_wait(self.reg,self.newpin,'projection',process_probe=lambda _:None)
        self.assertEqual(result,pin(root/'projection/actual_wait.json'))
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.glob('*.json')})
        self.assertEqual(a.projection_root(self.reg),self.oldroot)
        self.assertEqual(a.projection_registration(self.reg),a.ORIGINAL_REGISTRATION)

    def test_new_visual_and_score_keep_new_dispatch_identity(self):
        for stage in ('visual','score'):
            root,_,_=self.dispatch(stage)
            result=self.module.successful_wait(self.reg,self.newpin,stage,process_probe=lambda _:None)
            self.assertEqual(result,pin(root/stage/'actual_wait.json'))

    def test_wrong_projection_registration_rejected(self):
        root,launch,_=self.dispatch('projection',old=True)
        launch['registration']=self.newpin;write(root/'projection_launch.json',launch)
        with self.assertRaises(RuntimeError):self.module.successful_wait(self.reg,self.newpin,'projection',process_probe=lambda _:None)

    def test_wait_failure_and_child_generation_mismatch_rejected(self):
        for field,value in [('returncode',1),('timed_out',True),('start_ticks',999)]:
            root,_,wait=self.dispatch('projection',old=True)
            wait[field]=value;write(root/'projection/actual_wait.json',wait);write(root/'projection_actual_wait.json',wait)
            with self.assertRaises(RuntimeError):self.module.successful_wait(self.reg,self.newpin,'projection',process_probe=lambda _:None)

    def test_live_child_rejected_and_pid_reuse_allowed(self):
        _,launch,_=self.dispatch('projection',old=True)
        live=dict(launch['process'],state='R')
        with self.assertRaises(RuntimeError):self.module.successful_wait(self.reg,self.newpin,'projection',process_probe=lambda _:live)
        reused=dict(live,start_ticks=123)
        self.module.successful_wait(self.reg,self.newpin,'projection',process_probe=lambda _:reused)

    def test_unexpected_ancestor_pin_or_modified_bytes_rejected(self):
        bad=copy.deepcopy(self.reg);bad['recovery']['original_registration']['sha256']='0'*64
        with self.assertRaises(RuntimeError):a.original_master(bad)
        self.original_path.write_text('{}',encoding='utf8')
        with self.assertRaises(RuntimeError):a.original_master(self.reg)

    def test_optional_ancestor_bytes_reject_bool_or_wrong_count(self):
        for value in (True,1):
            bad=copy.deepcopy(self.reg);bad['recovery']['original_registration']['bytes']=value
            with self.assertRaises(RuntimeError):a.original_master(bad)

    def test_changed_science_source_or_deadline_rejected(self):
        for field in ('metric_config','numeric_flags','expected_physical_frames'):
            bad=copy.deepcopy(self.reg);bad[field]='changed'
            with self.assertRaises(RuntimeError):a.original_master(bad)
        bad=copy.deepcopy(self.reg);bad['pins']['cal200_sources']['sha256']='0'*64
        with self.assertRaises(RuntimeError):a.original_master(bad)
        bad=copy.deepcopy(self.reg);bad['runtime']['deadline_unix']+=1
        with self.assertRaises(RuntimeError):a.original_master(bad)

    def test_relocated_completed_stage_rejected(self):
        for stage in ('phy','projection'):
            bad=copy.deepcopy(self.reg);bad['runtime'][stage+'_directory']=str(self.newroot/stage)
            with self.assertRaises(RuntimeError):a.original_master(bad)

    def test_changed_source_shape_rejected_and_closed_visual_exact_route(self):
        source=ORIGINAL.read_text(encoding='utf8')
        with self.assertRaises(RuntimeError):a.adapted_source(source.replace(a.SUBSTITUTIONS[0][0],'pass'))
        changed,_=a.adapted_source(source)
        tree=ast.parse(changed);closed=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='closed_visual')
        text=ast.unparse(closed)
        self.assertIn('_recovery_projection_registration(reg)',text)
        self.assertIn('_recovery_projection_root(reg)',text)
        self.assertIn("successful_wait(reg, registration, 'visual')",text)
        self.assertIn('receiver_visual.audit_visual(reg)',text)


if __name__=='__main__':unittest.main()
