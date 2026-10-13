"""No-model resource thresholds, inverse source proof and outer wait tests."""
import ast
import contextlib
import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import shared as s
import resource_revision as revision
import dispatch_once as dispatch

HERE = Path(__file__).parent
RES = s.ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/h800_alternate_gpu_resource_v3.py'
GiB = 1 << 30
Busy = type('ResourceBusy', (Exception,), {})


def resource_stub():
    tree = ast.parse(RES.read_text(encoding='utf-8-sig'))
    ns = dict(g=types.SimpleNamespace(require=s.require), os=os, sys=sys, math=math, time=s.time,
              GiB=GiB, ResourceBusy=Busy, policy=lambda index: dict(index=index))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('assess_snapshot', 'own_allocator_observation')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RES), 'exec'), ns)
    original = types.SimpleNamespace(__file__=str(RES), **ns)
    assessor, observer = s.resource_adapter_sources(original)
    exec(compile(assessor+'\n'+observer, str(RES), 'exec'), ns)
    return ns


def snapshot(free=15985, utilization=0):
    return dict(gpu=dict(index=3, free_MiB=free, utilization_percent=utilization),
        cpu=dict(effective_cpu_cores=2, allowed_cpus=[0, 1], host_busy_percent=0,
                 available_memory_bytes=8*GiB), personal_filesystem_free_bytes=20*GiB)


def own(reserved=0, initialized=True):
    return dict(pid=os.getpid(), initialized=initialized, identity_verified=True,
                actual_identity={'index':3}, reserved_bytes=reserved)


class RevisionTests(unittest.TestCase):
    def test_exact_inverse_all_five_files(self):
        proof = revision.verify()
        self.assertEqual(len(proof['source_records']), 5)
        for row in proof['source_records'].values():
            self.assertEqual(row['original_AST_sha256'], row['inverse_restored_AST_sha256'])
        self.assertFalse(proof['source_free_RX_projection_changed'])
        self.assertFalse(proof['timing_v6_resource_changed'])

    def test_undeclared_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); new = base/HERE.name; old = base/'r1_execution_v1'
            new.mkdir(); old.mkdir()
            for name in revision.BASELINE_SHA:
                shutil.copyfile(HERE/name, new/name)
                shutil.copyfile(HERE.parent/'r1_execution_v1'/name, old/name)
            (new/'receiver_visual.py').write_text((new/'receiver_visual.py').read_text()+'\nBAD=True\n')
            with self.assertRaisesRegex(RuntimeError, 'Undeclared change'):
                revision.verify(new)

    def test_caps_before_model_loading(self):
        for name, marker in [('receiver_visual.py', "vae,var=ns['load_models']"),
                             ('score_selection.py', "self.perceptual = ledger.call")]:
            text = (HERE/name).read_text()
            self.assertLess(text.index('set_per_process_memory_fraction'), text.index(marker))
            tree = ast.parse(text)
            setters = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr=='set_per_process_memory_fraction']
            self.assertEqual(len(setters), 1)
            self.assertEqual(ast.dump(setters[0].args[0].left), ast.dump(ast.parse('12 << 30', mode='eval').body))

    def test_same_authoritative_paths_math_and_owner(self):
        self.assertEqual(s.TMP.name, 'r1_execution_v1')
        self.assertEqual(s.OUT.name, 'r1_execution_v1')
        self.assertEqual(s.sha(HERE/'owner.py'), revision.BASELINE_SHA['owner.py'])
        self.assertEqual(s.FLAGS, dict(matmul_tf32=False, cudnn_tf32=False, precision='highest',
            cudnn_benchmark=False, cudnn_deterministic=True, deterministic=True, threads=6, interop_threads=2))
        self.assertEqual(s.RESOURCE, dict(prelaunch_free_GiB=15, runtime_free_plus_own_GiB=14,
            allocator_cap_GiB=12, absolute_free_GiB=2, math_threads=6, interop_threads=2,
            scheduling_CPUs=2, shared_environment=True, exclusive_host_required=False, timing_priority=True))

    def test_prelaunch_boundary(self):
        f=resource_stub()['assess_snapshot']
        f(snapshot(15360), own(initialized=False), own(initialized=False), True)
        with self.assertRaises(Busy): f(snapshot(15359), own(), own(), True)
        with self.assertRaises(Busy): f(snapshot(15985,80), own(), own(), True)

    def test_runtime_floor_and_total(self):
        f=resource_stub()['assess_snapshot']
        f(snapshot(2048), own(12*GiB), own(12*GiB), False)
        with self.assertRaises(Busy): f(snapshot(2047), own(12*GiB), own(12*GiB), False)
        with self.assertRaises(Busy): f(snapshot(2048), own(11*GiB), own(11*GiB), False)

    def test_only_verified_current_minimum_credit(self):
        f=resource_stub()['assess_snapshot']; other=own(12*GiB); other['pid']+=1
        with self.assertRaises(ValueError): f(snapshot(2048),other,other,False)
        with self.assertRaises(Busy): f(snapshot(2048),own(12*GiB),own(11*GiB),False)
        with self.assertRaises(Busy): f(snapshot(2048),own(12*GiB,False),own(12*GiB),False)
        with self.assertRaises(ValueError): f(snapshot(2048),own(13*GiB),own(13*GiB),False)

    def test_resource_functions_inverse_ast(self):
        tree=ast.parse(RES.read_text(encoding='utf-8-sig'))
        originals={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        ns=dict(g=types.SimpleNamespace(require=s.require),os=os,sys=sys,math=math,time=s.time,
                GiB=GiB,ResourceBusy=Busy,policy=lambda index:dict(index=index))
        exec(compile(ast.Module(body=[originals['assess_snapshot'],originals['own_allocator_observation']],
            type_ignores=[]),str(RES),'exec'),ns)
        a,o=s.resource_adapter_sources(types.SimpleNamespace(__file__=str(RES),**ns))
        a=a.replace('free>=15*GiB if prelaunch else free>=2*GiB','free>=20*GiB if prelaunch else free>=4*GiB')
        a=a.replace('free+own>=14*GiB','free+own>=20*GiB').replace('0<=own<=12*GiB','0<=own<=16*GiB')
        o=o.replace('<=12*GiB','<=16*GiB').replace('reviewed12GiB cap','frozen16GiB cap')
        for name,source in [('assess_snapshot',a),('own_allocator_observation',o)]:
            self.assertEqual(ast.dump(ast.parse(source).body[0],include_attributes=False),
                             ast.dump(originals[name],include_attributes=False))

    def test_observer_cap_applies_to_reserved_peak(self):
        ns=resource_stub();f=ns['own_allocator_observation']
        device=types.SimpleNamespace(index=3,creator_pid=os.getpid(),check_cuda_environment=lambda:None,
                                     validate_cuda_device_identity=lambda *a:{'index':3})
        ns['original_math']=types.SimpleNamespace(__file__='stub',cuda_driver_pci_bus_id=lambda:'pci')
        ns['MATH_SHA']='original';ns['g'].sha=lambda p:'original'
        cuda=types.SimpleNamespace(is_initialized=lambda:True,device_count=lambda:1,current_device=lambda:0,
            get_device_properties=lambda i:types.SimpleNamespace(uuid='uuid'),memory_reserved=lambda i:12*GiB,
            memory_allocated=lambda i:GiB,max_memory_reserved=lambda i:12*GiB)
        self.assertEqual(f(device,types.SimpleNamespace(cuda=cuda))['reserved_bytes'],12*GiB)
        cuda.max_memory_reserved=lambda i:12*GiB+1
        with self.assertRaises(ValueError):f(device,types.SimpleNamespace(cuda=cuda))

    def test_host_guards_unchanged(self):
        f=resource_stub()['assess_snapshot']
        for mutation in [lambda x:x['cpu'].update(host_busy_percent=96),
                         lambda x:x['cpu'].update(available_memory_bytes=GiB),
                         lambda x:x.update(personal_filesystem_free_bytes=9*GiB)]:
            sample=snapshot(); mutation(sample)
            with self.assertRaises(Busy): f(sample,own(),own(),True)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.rt=self.base/'runtime';self.out=self.base/'results'
        self.d=self.base/'dispatch';self.calls=[]
        self.reg=dict(schema=s.MASTER_SCHEMA, code_bindings={str(Path(dispatch.__file__)):s.sha(dispatch.__file__),
            str(dispatch.ENTRY):s.sha(dispatch.ENTRY)},runtime=dict(deadline_unix=s.time.time()+300))

    def patches(self, child_class):
        stack=contextlib.ExitStack();self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(dispatch,'DISPATCH',self.d))
        stack.enter_context(mock.patch.object(s,'TMP',self.rt))
        stack.enter_context(mock.patch.object(s,'OUT',self.out))
        stack.enter_context(mock.patch.object(s,'validate_registration',return_value=self.reg))
        stack.enter_context(mock.patch.object(s,'process_identity',side_effect=lambda pid:dict(pid=pid,start_ticks=1,state='S')))
        stack.enter_context(mock.patch.dict(sys.modules,{'fcntl':types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)}))
        stack.enter_context(mock.patch.object(dispatch.subprocess,'Popen',child_class))
        return stack

    def child(self, code=0, timeout=False, started=False):
        outer=self
        class FakeChild:
            pid=456
            def __init__(self, argv, **kw):
                outer.calls.append((argv,kw));self.waits=0
                if started:
                    outer.rt.mkdir();s.write(outer.rt/'owner_launch.json',dict(process=dict(pid=self.pid,start_ticks=1)))
            def wait(self,timeout=None):
                self.waits+=1
                if timeout is not None and timeout is not False and timeout and self.waits==1 and should_timeout:
                    raise subprocess.TimeoutExpired('fake',timeout)
                return code
            def poll(self):return code
        should_timeout=timeout
        return FakeChild

    def test_wait_exit_is_not_science(self):
        self.patches(self.child())
        self.assertEqual(dispatch.run('a'*64),0)
        record=s.read(self.d/'actual_wait.json')
        self.assertTrue(record['actual_wait']);self.assertFalse(record['scientific_owner_launch_exists'])
        self.assertFalse(record['scientific_completion']);self.assertFalse((self.rt/'owner_launch.json').exists())
        argv,kw=self.calls[0]
        self.assertEqual(argv[3],str(dispatch.ENTRY));self.assertIn('--registration-sha256',argv)
        self.assertTrue(kw['start_new_session']);self.assertEqual(len(kw['pass_fds']),1)

    def test_true_attempt_exit_failure_recorded_no_retry(self):
        self.patches(self.child(1,started=True));self.assertEqual(dispatch.run('a'*64),1)
        record=s.read(self.d/'actual_wait.json');self.assertEqual(record['returncode'],1)
        self.assertTrue(record['scientific_owner_launch_exists'])
        with self.assertRaises(ValueError):dispatch.run('a'*64)
        self.assertEqual(len(self.calls),1)

    def test_existing_markers_refuse(self):
        self.patches(self.child());self.rt.mkdir();(self.rt/'score').mkdir()
        with self.assertRaises(ValueError):dispatch.run('a'*64)
        self.assertEqual(self.calls,[])

    def test_deadline_refuse(self):
        self.reg['runtime']['deadline_unix']=s.time.time()-1;self.patches(self.child())
        with self.assertRaises(ValueError):dispatch.run('a'*64)
        self.assertEqual(self.calls,[])

    def test_identity_record_failure_still_waits_dead_child(self):
        self.patches(self.child(0))
        original=s.write
        def fail(path,value):
            if Path(path).name=='owner_identity.json':raise OSError('fixture metadata failure')
            return original(path,value)
        with mock.patch.object(s,'write',side_effect=fail),self.assertRaises(OSError):dispatch.run('a'*64)
        record=s.read(self.d/'interrupted_actual_wait.json')
        self.assertTrue(record['actual_wait']);self.assertEqual(record['returncode'],0)
        self.assertTrue((self.d/'failure.json').exists())


if __name__=='__main__':unittest.main()
