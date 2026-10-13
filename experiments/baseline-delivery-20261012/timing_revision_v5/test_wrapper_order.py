"""Exercise the actual wrapper ordering with inert CPU-only components."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE/'timing_revision_v4'),str(BASE/'timing_revision_v3')]
import constructor_compat
from live_globals import adapt_build_live
from module_aliases import ExactByteAliases,sha


RUNTIME_SOURCE='''from types import SimpleNamespace
import sys

def visual_backend(*args):
    events.append('visual_original_or_patched')
    return SimpleNamespace(models={'vae':model}, t=torch_stub,
        resource=dict(shared_GPU=False,timing_measurement=True,
            scheduling_amendment='Original timing6/2 and six-CPU affinity; exclusive admission checked by owner'))

def build(registration,reservations,out,boundary):
    import t1_phy,ep_phy
    for loader in (t1_phy.load_exact,ep_phy.load_exact):
        for row in registration['module_alias_pairs']:
            assert loader(row['requested_path'],row['module_name']) is sys.modules[row['module_name']]
    events.append('aliases_effective')
    backend=visual_backend()
    s=SimpleNamespace(native=SimpleNamespace(multiple_snr=[1,4]))
    s.native.args.multiple_snr=','.join(str(v) for v in range(1,20))
    events.append('base_build_completed')
    return SimpleNamespace(backend=backend,endpoints={'P100000':SimpleNamespace(model=model),'Swin200000':SimpleNamespace(model=model)},snrs=s.native.multiple_snr)
'''


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def worker_function(path,namespace,stop_after_old_worker=False):
    source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    if stop_after_old_worker:
        index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='old' and n.value.func.attr=='worker')
        # Only omit the subsequent completed-science CSV export, which is not
        # part of wrapper installation/construction or first-call dispatch.
        node.body=node.body[:index+1]
    tree=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(tree)
    exec(compile(tree,str(path),'exec'),namespace);return namespace['worker']


class WrapperOrder(unittest.TestCase):
    def fixture(self,base):
        events=[];module_path=base/'stub_runtime.py';module_path.write_text(RUNTIME_SOURCE)
        runtime=load(module_path,'_synthetic_runtime_for_v5')
        model=SimpleNamespace(parameters=lambda:[SimpleNamespace(numel=lambda:3,requires_grad=False)])
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i:SimpleNamespace(total_memory=80*1024**3),set_per_process_memory_fraction=lambda *a:events.append('private_memory_cap')))
        runtime.events,runtime.model,runtime.torch_stub=events,model,torch
        runtime.ES=base;runtime.__file__=str(module_path)
        pairs=[];mods={}
        for name in ('main_raw64_plan_only','main_raw64_packet_adapter'):
            a=base/'old'/(name+'.py');b=base/'canonical'/(name+'.py');a.parent.mkdir(exist_ok=True);b.parent.mkdir(exist_ok=True)
            a.write_text('sentinel = 1\n');b.write_bytes(a.read_bytes())
            pairs.append(dict(module_name=name,loaded_path=str(a),requested_path=str(b),sha256=sha(a)))
            mods[name]=SimpleNamespace(__file__=str(a))
        mods['main_raw64_packet_adapter'].p=mods['main_raw64_plan_only']
        def strict(path,name):
            module=sys.modules[name]
            if Path(module.__file__).resolve()!=Path(path).resolve():raise RuntimeError('strict path mismatch')
            return module
        t1,ep=SimpleNamespace(load_exact=strict),SimpleNamespace(load_exact=strict)
        endpoint_module=SimpleNamespace(fresh_case=lambda:events.append('unadapted_case'))
        mods.update(timing_runtime=runtime,timing_endpoints=endpoint_module,t1_phy=t1,ep_phy=ep,torch=torch)
        return runtime,events,pairs,mods,strict

    def run_order(self,adapter):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);runtime,events,pairs,mods,strict=self.fixture(base);saved={k:sys.modules.get(k) for k in mods};sys.modules.update(mods)
            writes=[];built=[]
            def write(path,value):writes.append((Path(path).name,value))
            def audit_install(system,endpoints):
                events.append('audit_installed')
                endpoints.fresh_case=lambda:events.append('audited_first_case_stub')
                return dict(audit_hook=True)
            def old_worker():
                system=runtime.build(dict(module_alias_pairs=pairs),None,base,lambda:None);built.append(system)
                # This is an inert dispatch sentinel, not a timing frame/model.
                mods['timing_endpoints'].fresh_case()
            common=dict(Path=Path,sys=sys,ROOT=base,OUT=base,RUN=base,write=write,pin=lambda p:{'path':str(p)})
            v2=SimpleNamespace()
            v2.worker=worker_function(BASE/'timing_revision_v2/timing_owner_v2.py',dict(common,old=SimpleNamespace(worker=old_worker),audit=SimpleNamespace(install=audit_install)),True)
            v3=SimpleNamespace()
            v3.worker=worker_function(BASE/'timing_revision_v3/timing_owner_v3.py',dict(common,v2=v2,ExactByteAliases=ExactByteAliases))
            v4_worker=worker_function(BASE/'timing_revision_v4/timing_owner_v4.py',dict(common,v3=v3,adapt_build=adapter))
            try:
                v4_worker()
                self.assertIs(mods['t1_phy'].load_exact,strict);self.assertIs(mods['ep_phy'].load_exact,strict)
                self.assertIn('aliases_effective',events);self.assertEqual(events[-2:],['audit_installed','audited_first_case_stub'])
                self.assertEqual(built[0].snrs,list(range(1,20)))
                self.assertTrue(next(v for name,v in writes if name=='actual_module_aliases.json')['original_loaders_restored_before_first_case'])
                return built[0].backend.resource,writes
            finally:
                for k,value in saved.items():
                    if value is None:sys.modules.pop(k,None)
                    else:sys.modules[k]=value

    def test_v4_copied_globals_reproduces_lost_shared_visual_hook(self):
        resource,_=self.run_order(constructor_compat.adapt_build)
        self.assertFalse(resource['shared_GPU'])
        self.assertIn('exclusive admission',resource['scheduling_amendment'])

    def test_v5_full_nested_order_preserves_alias_shared_visual_and_audit(self):
        resource,writes=self.run_order(adapt_build_live)
        self.assertTrue(resource['shared_GPU']);self.assertTrue(resource['timing_measurement'])
        self.assertIn('user-authorized shared-H800',resource['scheduling_amendment'])
        proof=next(v for name,v in writes if name=='actual_constructor_compatibility.json')
        self.assertTrue(proof['runtime_globals_identity_preserved'])
        self.assertTrue(proof['compiled_code_object_unchanged'])

    def test_live_function_resolves_later_replacements_and_keeps_code_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime,events,pairs,mods,strict=self.fixture(Path(temp))
            live,proof=adapt_build_live(runtime)
            self.assertIs(live.__globals__,vars(runtime));self.assertTrue(proof['inverse_AST_equal'])
            replacement=object();runtime.visual_backend=replacement
            self.assertIs(live.__globals__['visual_backend'],replacement)
            self.assertEqual(live.__defaults__,None);self.assertEqual(live.__closure__,None)

    def test_actual_registered_runtime_compilation_retains_original_globals(self):
        runtime=load(BASE/'timing_runtime.py','_real_runtime_definition_only_v5')
        live,proof=adapt_build_live(runtime)
        self.assertIs(live.__globals__,vars(runtime))
        self.assertTrue(proof['inverse_AST_equal']);self.assertEqual(live.__name__,'build')
        # No invocation of this real constructor or any imported model occurs.


if __name__=='__main__':unittest.main(verbosity=2)
