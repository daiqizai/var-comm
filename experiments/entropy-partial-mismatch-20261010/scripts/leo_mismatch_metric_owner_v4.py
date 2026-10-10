"""Original four metric computations on actually closed leo GPU5 raw images.

The only engineering repair merges identical native file identities while
preserving lexical-path provenance. Actual binary differences still fail.
The population remains the original common500 first100 diagnostic subset. The
new leo native/device identity is explicit, all old score reuse is excluded,
and paired three-noise arithmetic has no bootstrap or policy selection.
"""
from __future__ import annotations
import ast
import copy
import inspect
import os
import re
from pathlib import Path, PurePosixPath
import time
import types
import h800_mismatch_metric_owner_v1 as base
import leo_mismatch_visual_v3 as visual

BASE_SHA='55c3957d2b3de73051f831e243b3028f7558d802e5da43fcdd4194635c5d68ff'
VISUAL_SHA='807228a03279922f323123e43332b14762d076b9dc36bbb9884fe384fb757df2'
SCHEMA='LEO_RAW100_ONE_BIN_MISMATCH_METRIC_OWNER_V4_GPU5'
PASS='PASS_LEO_RAW100_MISMATCH_FOUR_METRICS_GPU5_V4'
g=visual.g;gate=visual.gate;raw=visual.raw;metric=base.metric;CAPS=metric.CAPS
NATIVE_PATH=visual.RT/'qualification/leo_metric_native_import_v1/binding.json'
NATIVE_SHA='fb2c2080302eaf3a4d476ba58430d287013881320cf20930ff561ddd4c7f0e91'
OUT=visual.RT/'qualification/leo_mismatch_metrics_v4_attempt1/registered/run'
TOOLS=visual.TOOLS+('leo_mismatch_metric_owner_v3.py','leo_mismatch_visual_v3.py','h800_mismatch_metric_owner_v1.py',
    'h800_ep_pilot_metric_owner_v1.py','h800_ep_pilot_metrics_v1.py','h800_mismatch_metrics_v1.py')
PREVIOUS_SHA='db6e009ada8266424f9afa81609f8f65f1dc7503cf5affa7f12aaecfc0cfb521'
FAILED_ROOT=visual.RT/'qualification/leo_mismatch_metrics_v3_attempt1'
FAILED_PINS={'prepare_actual_wait.json':'280474b5caf3e1a3ebcd4533a443892fb3c024641fd5b260efa5f6726400960f',
    'prepare.log':'aa5d75bf1944d3153ab0ba370e5f0601e2a77cc7d80dbf0621ec5a853ea4ed76',
    'prepare_started.json':'ee7a09501f619dfc6eaa2ffa29591072298439c8b1f25d82557a75f9fe46ff7b',
    'launch_registration.json':'addab866aa600a0cc31259a69c83671d57534d0bacce720e8a48c632c9bf982b'}


def prepare_failure():
    pins={name:dict(path=str(FAILED_ROOT/name),sha256=h) for name,h in FAILED_PINS.items()}
    for p in pins.values():g.require(g.sha(g.inside(p['path']))==p['sha256'],'Preserved zero-call metric prepare failure changed')
    wait=gate.readpin(pins['prepare_actual_wait.json'])
    g.require(wait['phase']=='prepare' and wait['actual_wait'] is True and wait['returncode']==1 and wait['timeout'] is False and
        wait['log_sha256']==FAILED_PINS['prepare.log'] and not (FAILED_ROOT/'registered').exists() and
        not (FAILED_ROOT/'run_owner_started.json').exists(),'Prior metric attempt must remain prepare-only, no model/score calls')
    return dict(pins=pins,prior_prepare_exit=1,prior_model_constructions=0,prior_metric_scores=0,prior_bootstrap_calls=0,
        scientific_budget_unchanged=True,automatic_retry=False)


def merge_native_facts(original_rows,new_rows):
    """Preserve old evidence; identical canonical byte identities may co-exist."""
    combined=copy.deepcopy(original_rows);existing={}
    for row in combined:existing.setdefault(row['path'],[]).append(row)
    seen=set();base=PurePosixPath('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu')
    for row in new_rows:
        g.require(set(row)=={'path','sha256','bytes'} and type(row['path']) is str and
            type(row['bytes']) is int and row['bytes']>=0 and g.SHA_RE.fullmatch(row['sha256']), 'Exact metric native file identity required')
        p=PurePosixPath(row['path'])
        g.require(p.is_absolute() and str(p)==row['path'] and '..' not in p.parts and not p.is_relative_to(base) and
            re.fullmatch(r'.+\.so(?:\.[0-9]+)*',p.name) and not re.match(r'lib(?:cuda|nvidia).*\.so',p.name),
            'Metric additions must be canonical external non-driver native libraries')
        g.require(row['path'] not in seen,'Duplicate metric native file');seen.add(row['path'])
        for old in existing.get(row['path'],[]):
            g.require(set(old) in ({'path','sha256','bytes'},{'path','sha256','bytes','observed_lexical_path'}),
                'Unknown native provenance fields cannot be silently discarded')
            g.require(all(old[k]==row[k] for k in ('path','bytes','sha256')),
                'Conflicting actual native file path/size/SHA: '+row['path'])
            if 'observed_lexical_path' in old:
                g.require(type(old['observed_lexical_path']) is str and PurePosixPath(old['observed_lexical_path']).is_absolute(),
                    'Original observed lexical path must remain explicit')
        if row['path'] not in existing:combined.append(copy.deepcopy(row))
    return combined


def metric_environment(spec,verify_environment=True):
    result=g.validate_spec(spec,verify_environment=verify_environment);environment=result[1]
    _,rows=native_binding();original_rows=copy.deepcopy(environment.rows)
    # native_binding independently validates canonical resolved paths and hashes
    # every actual external file, even during metadata-only preparation.
    environment.host_native=merge_native_facts(environment.host_native,rows)
    g.require(environment.rows==original_rows,'Original PFS runtime map must stay exact')
    if verify_environment:
        for row in rows:
            p=Path(row['path']);g.require(p.stat().st_size==row['bytes'] and g.sha(p)==row['sha256'],'Metric native bytes changed')
    return result


def runtime_identity(spec,device):
    result=base_runtime_identity(spec,device)
    result.update(native_fact_merge=dict(adapter_source=raw.pin(__file__),
        identity_fields=['path','bytes','sha256'],classification='canonical external non-driver shared libraries',
        original_lexical_provenance_preserved=True,original_PFS_rows_unchanged=True),
        preserved_zero_call_prepare_failure=prepare_failure())
    return result


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
for name in ('runtime_identity','configuration','worker'):
    original_ns[name]=gate.source.host.clone_function(getattr(base.original,name),original_ns)
original_ns['metric_environment']=metric_environment
base_runtime_identity=original_ns['runtime_identity']
original_ns['runtime_identity']=runtime_identity
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
    g.require(g.sha(base.__file__)==BASE_SHA and g.sha(visual.__file__)==VISUAL_SHA and
        g.sha(Path(__file__).with_name('leo_mismatch_metric_owner_v3.py'))==PREVIOUS_SHA,'Frozen raw metric/leo visual source changed')
    namespace=dict(vars(visual.engine()));namespace.update(g=g,DeviceAdapter=DeviceAdapter,
        __file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        namespace[name]=gate.source.host.clone_function(namespace[name],namespace)
    return types.SimpleNamespace(**namespace)


ns.update(visual_closure=visual_closure,registration=registration,prepare=prepare,engine=engine)
for name,value in ns.items():
    if not name.startswith('__'):globals()[name]=value


if __name__=='__main__':main()
