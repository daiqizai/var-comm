"""Original four metric computations on actually closed leo GPU5 raw images.

The population remains the original common500 first100 diagnostic subset. The
new leo native/device identity is explicit, all old score reuse is excluded,
and paired three-noise arithmetic has no bootstrap or policy selection.
"""
from __future__ import annotations
import ast
import copy
import inspect
import os
from pathlib import Path
import time
import types
import h800_mismatch_metric_owner_v1 as base
import leo_mismatch_visual_v3 as visual

BASE_SHA='55c3957d2b3de73051f831e243b3028f7558d802e5da43fcdd4194635c5d68ff'
VISUAL_SHA='807228a03279922f323123e43332b14762d076b9dc36bbb9884fe384fb757df2'
SCHEMA='LEO_RAW100_ONE_BIN_MISMATCH_METRIC_OWNER_V3_GPU5'
PASS='PASS_LEO_RAW100_MISMATCH_FOUR_METRICS_GPU5'
g=visual.g;gate=visual.gate;raw=visual.raw;metric=base.metric;CAPS=metric.CAPS
NATIVE_PATH=visual.RT/'qualification/leo_metric_native_import_v1/binding.json'
NATIVE_SHA='fb2c2080302eaf3a4d476ba58430d287013881320cf20930ff561ddd4c7f0e91'
OUT=visual.RT/'qualification/leo_mismatch_metrics_v3_attempt1/registered/run'
TOOLS=visual.TOOLS+('leo_mismatch_visual_v3.py','h800_mismatch_metric_owner_v1.py',
    'h800_ep_pilot_metric_owner_v1.py','h800_ep_pilot_metrics_v1.py','h800_mismatch_metrics_v1.py')


def native_binding():
    pin=dict(path=str(NATIVE_PATH),sha256=NATIVE_SHA);doc=gate.readpin(pin)
    g.require(doc['schema']=='LEO_RAW_METRIC_NATIVE_BINDING_V1' and doc['cuda_initialized'] is False and
        doc['numerical_qualification'] is False,'Actual leo CPU-hidden metric import observation required')
    waited=gate.readpin(doc['actual_child_wait_pin']);observation=gate.readpin(doc['observation'])
    g.require(waited==doc['actual_child_wait'] and observation['schema']=='LEO_METRIC_NATIVE_IMPORT_OBSERVATION_V1' and
        observation['external_non_driver_libraries']==doc['files'] and observation['cuda_initialized'] is False and
        observation['dino_implementation_receipt']['sha256']==metric.original.DINO_RECEIPT_SHA and
        observation['runtime_projection']['sha256']==gate.source.host.PROJECTION_SHA,'Actual leo metric import closure changed')
    maps=observation['maps'];g.require(g.sha(g.inside(maps['path']))==maps['sha256'],'Actual leo loaded-library maps changed')
    # Reuse the strict canonical external-file/actual-wait/zero-call validator.
    # Its internal schema spelling is adapted; no H800 byte equality is claimed.
    projected=copy.deepcopy(doc);projected['schema']='H800_NEW_HOST_NATIVE_BINDING_V1'
    return pin,gate.source.host.validate_native_document(projected)


class DeviceAdapter(visual.DeviceAdapter):
    def resource_snapshot(self,shared,prelaunch=False):
        try:return visual.original_device.DeviceAdapter.resource_snapshot(self,shared,prelaunch)
        except visual.original_device.ResourceBusy as error:
            # Metric refusals belong only to this attempt, never visual receipts.
            if hasattr(error,'resource_snapshot') and OUT.is_dir():
                dest=g.inside(OUT/'resource_refusals');dest.mkdir(exist_ok=True)
                g.write(dest/f'{os.getpid()}-{time.time_ns()}.json',dict(error=str(error),snapshot=error.resource_snapshot,
                    pid=os.getpid(),prelaunch=prelaunch,thresholds_unchanged=True))
            raise


# Existing numerical environment and configuration functions have unchanged
# code objects, with only actual current-host dependencies in this namespace.
original_ns=dict(vars(base.original));original_ns.update(g=g,native_binding=native_binding)
for name in ('metric_environment','runtime_identity','configuration','worker'):
    original_ns[name]=gate.source.host.clone_function(getattr(base.original,name),original_ns)
original=types.SimpleNamespace(**original_ns)
ns=dict(vars(base));ns.update(g=g,visual=visual,original=original,__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,
    TOOLS=TOOLS,DeviceAdapter=DeviceAdapter,metric_environment=original.metric_environment,runtime_identity=original.runtime_identity)
for name,value in vars(base).items():
    if isinstance(value,types.FunctionType) and value.__module__==base.__name__:
        ns[name]=gate.source.host.clone_function(value,ns)


def adapted_control(name,ints_expected,paths_expected,prior_expected,device_expected,cpu_expected):
    tree=ast.parse(inspect.getsource(getattr(base,name)));changed=dict(ints=0,paths=0,prior=0,device=0,cpu=0)
    class Admission(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value==2:node.value=5;changed['ints']+=1
            elif node.value=='qualification/h800_mismatch_visual_v1_attempt1':
                node.value='qualification/leo_mismatch_visual_v3_attempt1';changed['paths']+=1
            elif node.value=='Registered single GPU2 required':node.value='Registered single leo GPU5 required'
            return node
        def visit_Call(self,node):
            if ast.unparse(node.func)=='visual.original.source.prerequisites':
                node.func=ast.parse('visual.failed_attempt',mode='eval').body;changed['prior']+=1
            elif ast.unparse(node.func)=='gate.source.host.DeviceAdapter':
                node.func=ast.Name(id='DeviceAdapter',ctx=ast.Load());changed['device']+=1
            elif ast.unparse(node.func)=='visual.cpu_closure':
                node.func=ast.parse('visual.base.cpu_closure',mode='eval').body;changed['cpu']+=1
            return self.generic_visit(node)
    tree=Admission().visit(tree)
    g.require(changed==dict(ints=ints_expected,paths=paths_expected,prior=prior_expected,device=device_expected,cpu=cpu_expected),
        'Frozen raw metric admission control changed: '+name)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),ns)


adapted_control('visual_closure',0,1,0,1,1)
adapted_control('registration',0,0,1,0,0)
adapted_control('prepare',2,0,1,1,0)
adapted_control('main',1,0,0,0,0)
base_closure=ns['visual_closure'];base_registration=ns['registration'];base_prepare=ns['prepare']
def visual_closure(pins):
    result=base_closure(pins);request=result[0]
    g.require(request['target_index']==5 and request['recovery_budget']==visual.failed_attempt()[1],
        'Actual leo visual cumulative recovery budget changed')
    return result
def registration(path,digest,runner):
    result=base_registration(path,digest,runner)
    g.require(result['target_index']==5,'GPU5-only leo metric identity required')
    return result
def prepare(a,runner):
    g.require(g.inside(a.out)==OUT.parent,'Exact independent leo metric output required')
    return base_prepare(a,runner)


def engine():
    g.require(g.sha(base.__file__)==BASE_SHA and g.sha(visual.__file__)==VISUAL_SHA,'Frozen raw metric/leo visual source changed')
    namespace=dict(vars(visual.engine()));namespace.update(g=g,DeviceAdapter=DeviceAdapter,
        __file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        namespace[name]=gate.source.host.clone_function(namespace[name],namespace)
    return types.SimpleNamespace(**namespace)


ns.update(visual_closure=visual_closure,registration=registration,prepare=prepare,engine=engine)
for name,value in ns.items():
    if not name.startswith('__'):globals()[name]=value


if __name__=='__main__':main()
