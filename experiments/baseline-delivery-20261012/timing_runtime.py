"""Explicit resource-only adapters for a single frozen timing process.

No code below runs on import. Historical model/PHY code stays byte-identical;
adapters record inverse-AST proofs and never qualify a completed timing result.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
RT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
ES=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def replace_checked(source, substitutions, namespace, name):
    """Exact text edits plus inverse AST equality; fail on any source drift."""
    changed=source
    for old,new in substitutions:
        assert changed.count(old)==1, old
        changed=changed.replace(old,new,1)
    inverse=changed
    for old,new in reversed(substitutions):
        assert inverse.count(new)==1, new
        inverse=inverse.replace(new,old,1)
    before=ast.dump(ast.parse(source),include_attributes=False)
    restored=ast.dump(ast.parse(inverse),include_attributes=False)
    assert before==restored
    exec(compile(changed,namespace['__file__'],'exec'),namespace)
    return namespace[name],dict(original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        inverse_restored_AST_sha256=hashlib.sha256(restored.encode()).hexdigest(),
        changes=[dict(original=a,adapted=b) for a,b in substitutions],original_files_modified=False)


def assert_timing_flags(torch):
    assert torch.get_num_threads()==6 and torch.get_num_interop_threads()==2
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cudnn.benchmark and torch.are_deterministic_algorithms_enabled()
    assert torch.get_float32_matmul_precision()=='highest'


def adapt_raw_factory(module):
    source=Path(module.__file__).read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='create_runtime')
    selected=ast.get_source_segment(source,function)
    substitutions=[
        ("require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only receiver required')",
         "require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,''),'Explicit single-device timing process required')"),
        ("require(os.environ.get(name)=='2','Frozen original two-thread PHY environment required')",
         "require(os.environ.get(name)=='6','Registered unified six-thread timing environment required')"),
        ('packet.runtime_flags(old.torch)','_assert_timing_flags(old.torch)'),
        ("require(not old.torch.cuda.is_initialized(),'Raw PHY initialized CUDA')",
         "require(old.device=='cpu','Raw LDPC must still execute on CPU')")]
    namespace=dict(vars(module));namespace['_assert_timing_flags']=assert_timing_flags
    return replace_checked(selected,substitutions,namespace,'create_runtime')


class ModelCalls:
    """In-memory accounting only; per-case reservation already exists on disk."""
    def __init__(self):
        self.counts={}

    def call(self,name,callback,**unused):
        self.counts[name]=self.counts.get(name,0)+1
        return callback()


def visual_backend(request, out, boundary):
    import h800_ep_new100_visual_v3 as visual
    runner=visual.engine();original_device=runner.device_from_request(request)
    class TimingDevice:
        def __getattr__(self,name):
            return getattr(original_device,name)
        def inside(self,path,*args,**kwargs):
            p=Path(path).absolute()
            # Only this explicitly SHA-bound private package is added to paths.
            if p.is_relative_to(TMP/'timing_private_sionna_v1'):
                return p
            return original_device.inside(path,*args,**kwargs)
    device=TimingDevice()
    spec=copy.deepcopy(request['spec'])
    # Refresh only the model-loader's finite admission window. This does not
    # replay its historical scientific owner, sources or call ledger.
    spec['deadline_unix']=time.time()+1800;spec['max_seconds']=1800
    resolver,environment,_,_,extra=visual.g.validate_spec(spec)
    module=load(ES/'leo_whole_math_v1.py','_timing_frozen_visual_math')
    assert sha(module.__file__)=='db4349f936b2cd0c267932cc286e2661e1f7d56dd10d1c4e4a5d7d90ec279252'
    source=Path(module.__file__).read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Backend')
    selected=ast.get_source_segment(source,node)
    substitutions=[
        ('import leo_whole_gate_v1 as g','g = _explicit_timing_device'),
        ("g.require(len(os.sched_getaffinity(0))==2,'Two-CPU scheduling mask required')",
         "g.require(len(os.sched_getaffinity(0))==6,'Original timing six-CPU scheduling mask required')"),
        ('shared_GPU=True,timing_measurement=False','shared_GPU=False,timing_measurement=True'),
        ("scheduling_amendment='Original6/2 retained; CPU affinity restricted to2'",
         "scheduling_amendment='Original timing6/2 and six-CPU affinity; exclusive admission checked by owner'")]
    namespace=dict(vars(module));namespace['_explicit_timing_device']=device
    Backend,proof=replace_checked(selected,substitutions,namespace,'Backend')
    (out/'visual_resource_adapter.json').write_text(json.dumps(proof,indent=2)+'\n')
    calls=ModelCalls()
    backend=calls.call('model_load',lambda:Backend(resolver,environment,spec,extra,calls,boundary,out))
    return backend,module,resolver,environment,calls,visual.g


def private_imports():
    parent=TMP/'timing_private_sionna_v1'
    for name in ['binding.json','dependency_binding.json']:
        for row in read(parent/name)['entries'].values():
            assert sha(row['source'])==sha(row['copy'])==row['sha256']
    assert 'sionna' not in sys.modules
    sys.path.insert(0,str(parent))


def register_native_addendum(registration, environment, out):
    """Admit one existing, pre-pinned header library; never scan-and-trust."""
    item=registration['native_header']
    for field in ['source','descriptor','library']:
        assert sha(item[field]['path'])==item[field]['sha256'],field
    descriptor=read(item['descriptor']['path'])
    assert descriptor['source_sha256']==item['source']['sha256']
    assert descriptor['library_sha256']==item['library']['sha256']
    library=Path(item['library']['path']).resolve()
    row=dict(path=str(library),sha256=item['library']['sha256'],bytes=library.stat().st_size)
    # This is a private in-memory extension, not a change to the historical
    # runtime inventory or an assertion that the old host loaded this file.
    environment.host_native=list(environment.host_native)+[row]
    proof=dict(schema='TIMING_PREBOUND_NATIVE_ADDENDUM_V1',library=row,
        source=item['source'],descriptor=item['descriptor'],
        original_inventory_modified=False,old_host_byte_identity_claimed=False)
    (out/'native_header_addendum.json').write_text(json.dumps(proof,indent=2)+'\n')


def build(registration, reservations, out, boundary):
    """Construct once after exclusive admission; never invoked by prepare."""
    sys.path[:0]=[str(ES),str(ROOT/'experiments/wcl-evidence-closure-20261009/scripts'),
                  str(ROOT/'experiments/baseline-strength-20261011')]
    import timing_endpoints as endpoints
    import ep_plan, ep_source_codec, t1_phy, ep_phy
    import ep_t1_portable_qualification_v1 as portable
    import h800_mismatch_phy_core_v1 as raw
    import baseline_runtime as baseline
    import torch
    private_imports()
    visual_request=read(registration['visual_request']['path'])
    backend,module,resolver,environment,calls,g=visual_backend(visual_request,out,boundary)
    native=backend.native
    register_native_addendum(registration,environment,out)
    codec=ep_source_codec.PartialSourceCodec(ROOT)
    # Instantiate the exact independent provider without TX-CDF tracing or
    # TX-token comparisons. Each encode/decode context is created afresh.
    ns=dict(np=endpoints.np,time=time)
    driver=resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py')
    assert sha(driver)=='b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce'
    module.definitions(driver,{'IndependentProvider'},ns)
    codec.native=native
    codec.provider_factory=lambda:ns['IndependentProvider'](native,SimpleNamespace(cdf_function=codec.entropy.probability_cdf))
    phy_request=read(registration['phy_request']['path'])
    charges=endpoints.PacketCharges(reservations)
    factory,proof=adapt_raw_factory(raw)
    (out/'raw_resource_adapter.json').write_text(json.dumps(proof,indent=2)+'\n')
    raw_runtime=factory(phy_request['raw_binding'],charges,boundary)
    whole=portable.create_runtime(phy_request['T1_original241_relocation'],ROOT,t1_phy,
        phy_request['providers']['whole']['CPU_environment'],configure_threads=False)
    partial=ep_phy.create_runtime(ROOT,phy_request['providers']['partial']['qualification']['path'],configure_threads=False)
    assert_timing_flags(torch)
    for runtime in [raw_runtime,whole,partial]:
        runtime.header.s.ROOT=TMP/'native_build'
        runtime.header.s.load_native()  # Build/load once outside online windows.
    planner_path=Path(phy_request['raw_binding']['roles']['plan_module']).resolve()
    planner_hash=sha(planner_path)
    def verified_planner_hash(path):
        assert Path(path).resolve()==planner_path
        return planner_hash
    # Several frozen loaders create their own module instance. Patch only the
    # audit function on those exact instances, never metadata/codec operations.
    components=[raw_runtime.original,raw_runtime.rx,raw_runtime.packet,
                whole.mapping,partial.mapping]
    planners=[v for component in components for v in vars(component).values()
              if getattr(v,'__file__',None) and Path(v.__file__).resolve()==planner_path]
    assert planners
    for planner in planners:planner.sha=verified_planner_hash
    development=load(registration['development_module']['path'],'_timing_original_development')
    points=registration['policies'];result={}
    for method in ('RAW_WHOLE','RAW_PARTIAL'):
        for point in points[method].values():
            p=raw_runtime.cat.entry(point['profile_id'])
            assert p['wire_key']==point['wire_key'] and (p['m'],p['K'])==(point['m'],point['K'])
        result[method]=endpoints.RawEndpoint(native,raw_runtime,backend.render_function,
            points[method],g.OFFSETS,development.standard_noise)
    for method,runtime,allowed in [('EC_VAR_WHOLE',whole,False),('EC_VAR_PARTIAL',partial,True)]:
        result[method]=endpoints.EntropyEndpoint(native,codec,runtime,ep_plan,backend.render_function,
            points[method],allowed,charges,t1_phy.standard_noise,ep_source_codec.InvalidSourceStream)
    ns,_=baseline.p_math()
    payload=torch.load(registration['P_checkpoint']['path'],map_location='cpu',weights_only=True)
    # Use the selected communication state, not its 40k parent and no LPIPS net.
    state=payload['model'];p=ns['BudgetContinuous'](state['scale'].flatten(),1024)
    p.load_state_dict(state,strict=True);p=p.cuda().eval().requires_grad_(False)
    result['P100000']=endpoints.ContinuousEndpoint(native,p,backend.models['decoder'],ns['seeded_noise'])
    sm,sp,sr,_=baseline.swin_modules()
    s=sm.build_official(baseline.TMP/'vendor/SwinJSCC')
    payload=torch.load(registration['Swin_checkpoint']['path'],map_location='cpu',weights_only=True)
    s.load_state_dict(payload['model'],strict=True);s=s.eval().requires_grad_(False)
    s.native.args.multiple_snr=','.join(str(v) for v in range(1,20))
    result['Swin200000']=endpoints.SwinEndpoint(torch,s,sr,sp)
    assert_timing_flags(torch)
    linkage=module.native_linkage(environment,backend.g,previous=backend.native_start)
    (out/'native_linkage_before_first_case.json').write_text(json.dumps(linkage,indent=2)+'\n')
    return SimpleNamespace(endpoints=result,backend=backend,source_calls=calls,charges=charges,
                           resolver=resolver,environment=environment,development=development,
                           audit_hashes={str(planner_path):planner_hash})
