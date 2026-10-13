"""No-GPU tests for the explicitly reviewed quality-only memory revision."""
import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('r0_clean_resource_v2.py')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load(P,'resource_v2_under_test')
oldtests=load(P.with_name('test_r0_clean.py'),'old_clean_cpu_tests')
oldtests.m=m;oldtests.P=P
ROOT=P.resolve().parents[2]
RES=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/h800_alternate_gpu_resource_v3.py'
MATH=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/leo_whole_math_v1.py'
ENG=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/leo_single_rx_diagnostic_v2.py'
GiB=1<<30
ResourceBusy=type('ResourceBusy',(Exception,),{})

def resource_stub():
    tree=ast.parse(RES.read_text(encoding='utf-8-sig'))
    ns=dict(g=types.SimpleNamespace(require=m.require),os=os,sys=sys,math=math,time=m.time,GiB=GiB,
        ResourceBusy=ResourceBusy,policy=lambda index:dict(index=index))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('assess_snapshot','own_allocator_observation')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RES),'exec'),ns)
    stub=types.SimpleNamespace(__file__=str(RES),**ns)
    assess,observe=m.resource_adapter_sources(stub)
    exec(compile(assess+'\n'+observe,str(RES),'exec'),ns)
    return ns

def snapshot(free=15985):
    return dict(gpu=dict(index=3,free_MiB=free,utilization_percent=0),
        cpu=dict(effective_cpu_cores=2,allowed_cpus=[178,179],host_busy_percent=0,available_memory_bytes=8*GiB),
        personal_filesystem_free_bytes=20*GiB)

def own(reserved=0,initialized=True):
    return dict(pid=os.getpid(),initialized=initialized,identity_verified=True,actual_identity={'index':3},reserved_bytes=reserved)

def engine():
    tree=ast.parse(ENG.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='adapt_backend')
    ns=dict(ast=ast,copy=copy,Path=Path,MATH_SHA=None,
        g=types.SimpleNamespace(require=m.require,sha=m.sha,hashlib=hashlib))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ENG),'exec'),ns)
    return ns

class Tests(oldtests.Tests):
    def test_original_source_still_frozen(self):
        self.assertEqual(m.sha(P.with_name('r0_clean.py')),'f34c6495d0ce0988188a0946d538a46c37bc21ddc7a07595206b324ee73cebed')
    def test_only_declared_functions_changed(self):
        old=ast.parse(P.with_name('r0_clean.py').read_text(encoding='utf-8-sig'))
        new=ast.parse(P.read_text(encoding='utf-8-sig'))
        a={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
        b={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
        changed={k for k in a if a[k]!=b[k]}
        self.assertEqual(changed,{'draft','review_gate','registration','resource_device','worker','score_stage','_owner_locked'})
        for name in ('plan','render_one','validate_source','validate_image','token_count','guard_for','owner','wait_child','main'):
            self.assertEqual(a[name],b[name],name)
        oldscore=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='score_stage')
        restored=m.inspect.getsource(m.score_stage).replace('(12<<30)/props.total_memory','(16<<30)/props.total_memory')
        self.assertEqual(ast.dump(oldscore,include_attributes=False),ast.dump(ast.parse(restored).body[0],include_attributes=False))
        oldworker=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        restored=m.inspect.getsource(m.worker).replace('adapt_backend_limited(module,device,visual.engine())','visual.engine().adapt_backend(module,device)')
        self.assertEqual(ast.dump(oldworker,include_attributes=False),ast.dump(ast.parse(restored).body[0],include_attributes=False))
    def test_policy_is_exact(self):
        self.assertEqual(m.RESOURCE_POLICY,dict(prelaunch_free_GiB=15,runtime_free_plus_own_GiB=14,
            allocator_cap_GiB=12,absolute_free_GiB=2,math_threads=6,interop_threads=2,scheduling_CPUs=2,overlap_with_v6_timing=False))
    def test_current15985_admitted_but_less15_rejected(self):
        f=resource_stub()['assess_snapshot'];f(snapshot(),own(initialized=False),own(initialized=False),True)
        self.assertRaises(ResourceBusy,f,snapshot(15359),own(initialized=False),own(initialized=False),True)
    def test_runtime_absolute_and_sum_guards(self):
        f=resource_stub()['assess_snapshot']
        f(snapshot(2048),own(12*GiB),own(12*GiB),False)
        self.assertRaises(ResourceBusy,f,snapshot(2047),own(12*GiB),own(12*GiB),False)
        self.assertRaises(ResourceBusy,f,snapshot(2048),own(11*GiB),own(11*GiB),False)
    def test_only_current_process_credit_and_minimum(self):
        f=resource_stub()['assess_snapshot'];other=own(12*GiB);other['pid']+=1
        self.assertRaises(RuntimeError,f,snapshot(2048),other,other,False)
        self.assertRaises(ResourceBusy,f,snapshot(2048),own(12*GiB),own(11*GiB),False)
        self.assertRaises(ResourceBusy,f,snapshot(2048),own(12*GiB,False),own(12*GiB),False)
        self.assertRaises(RuntimeError,f,snapshot(2048),own(13*GiB),own(13*GiB),False)
    def test_no_cap_or_math_change_inherited(self):
        original=load(P.with_name('r0_clean.py'),'original_compare')
        for name in ('CAPS','MODELS','FLAGS','SIZES','OFFSETS','MISSING_STATES','TMP','OUT'):
            self.assertEqual(getattr(m,name),getattr(original,name),name)
    def test_real_backend_inverse_ast_and_allocation_before_model(self):
        ns=engine();ns['MATH_SHA']=m.sha(MATH);mod=m.load(MATH,'actual_math_cpu_only')
        dev=types.SimpleNamespace(device_identity_binding=lambda:{'test_only':True})
        backend,proof=m.adapt_backend_limited(mod,dev,types.SimpleNamespace(adapt_backend=ns['adapt_backend']))
        self.assertEqual(proof['original_AST_sha256'],proof['inverse_restored_AST_sha256'])
        self.assertEqual(proof['allocator_cap_bytes'],12*GiB)
        # Frozen assignment executes before CUDA allocator setup, which precedes model construction.
        source=MATH.read_text(encoding='utf-8-sig')
        self.assertLess(source.index('cap=16*(1<<30)'),source.index('torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,0)'))
        self.assertLess(source.index('torch.cuda.set_per_process_memory_fraction(cap/props.total_memory,0)'),source.index("vae,var=ns['load_models'](paths,device)"))
        self.assertIs(backend.__init__.__globals__['_explicit_device_adapter'],dev)
        self.assertIn(12*GiB,backend.__init__.__code__.co_consts)
        self.assertNotIn(16*GiB,backend.__init__.__code__.co_consts)
    def test_limited_constructor_sets_cap_before_stub_model(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.py'
            p.write_text('class Backend:\n def __init__(self):\n  import leo_whole_gate_v1 as g\n  cap=16*(1<<30)\n  g.set_cap(cap)\n  g.model()\n',encoding='utf8')
            ns=engine();ns['MATH_SHA']=m.sha(p);mod=m.load(p,'stub_backend_fixture');events=[]
            dev=types.SimpleNamespace(device_identity_binding=lambda:{},set_cap=lambda c:events.append(('cap',c)),model=lambda:events.append(('model_stub',None)))
            backend,_=m.adapt_backend_limited(mod,dev,types.SimpleNamespace(adapt_backend=ns['adapt_backend']))
            backend();self.assertEqual(events,[('cap',12*GiB),('model_stub',None)])
    def retirement(self):
        return dict(status='RETIRED_ZERO_SCIENCE',owner_invocations=0,scientific_owner_launch_count=0,scientific_calls=0,
            process=dict(pid=17,start_ticks=12),observed_gone=True,pidfd_exit_observed=True)
    def test_retirement_original_process_must_be_gone(self):
        r=dict(resource_revision=dict(retirement={'path':'unused'}))
        with patch.object(m,'checked',return_value=Path('unused')),patch.object(m,'read',return_value=self.retirement()):
            with patch.object(m,'process_identity',return_value=dict(pid=17,start_ticks=12,state='Z')):
                self.assertRaises(RuntimeError,m.retirement_gate,r)
            with patch.object(m,'process_identity',return_value=None):m.retirement_gate(r)
            with patch.object(m,'process_identity',return_value=dict(pid=17,start_ticks=99)):m.retirement_gate(r)
    def test_retirement_unknown_or_nonzero_refused(self):
        r=dict(resource_revision=dict(retirement={'path':'unused'}))
        for key,value in [('scientific_calls',1),('owner_invocations',1),('pidfd_exit_observed',False),('observed_gone',False),('status','UNKNOWN')]:
            rec=self.retirement();rec[key]=value
            with patch.object(m,'checked',return_value=Path('unused')),patch.object(m,'read',return_value=rec),patch.object(m,'process_identity',return_value=None):
                self.assertRaises(RuntimeError,m.retirement_gate,r)
    def test_prelaunch_markers_only_block_prelaunch_not_owned_worker(self):
        r=dict(resource_revision=dict(retirement={'path':'unused'}))
        with tempfile.TemporaryDirectory() as d,patch.object(m,'TMP',Path(d)/'tmp'),patch.object(m,'OUT',Path(d)/'out'),patch.object(m,'checked',return_value=Path('unused')),patch.object(m,'read',return_value=self.retirement()),patch.object(m,'process_identity',return_value=None):
            m.TMP.mkdir();m.OUT.mkdir();m.retirement_gate(r,True)
            (m.TMP/'render').mkdir()
            self.assertRaises(RuntimeError,m.retirement_gate,r,True)
            m.retirement_gate(r,False)
    def test_no_new_empty_cache_or_numerical_settings(self):
        self.assertNotIn('empty_cache',P.read_text(encoding='utf8'))
        self.assertEqual(m.FLAGS,dict(matmul_tf32=False,cudnn_tf32=False,precision='highest',cudnn_benchmark=False,
            cudnn_deterministic=True,deterministic=True,threads=6,interop_threads=2))

if __name__=='__main__':unittest.main()
