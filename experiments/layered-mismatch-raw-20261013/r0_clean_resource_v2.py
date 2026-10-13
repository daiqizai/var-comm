"""Finite clean raster-prefix supplement. No scientific work on import/prepare.

Only the predeclared 5400 missing (calibration source, m, K) pairs are admitted.
The original qualified VAR receiver and DINO-L implementation remain unchanged.
An owner waits for resources, then runs render and DINO-only in separate processes.
No implicit recovery, extra qualification image, tokenizer, channel or training.
"""
import argparse
import ast
import copy
import collections
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
RT = BASE/'code/liulu/var_comm_runtime_20261010'
TMP = BASE/'liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013/r0_clean_supplement_v1'
OUT = ROOT/'results/layered_mismatch_raw_20261013/r0_clean_supplement_v1'
PREP = ROOT/'results/layered_mismatch_raw_20261013/r0_supplement_preparation'
TIMING_LOCK = ROOT/'results/baseline_strength_20261011/timing/six_method_v6/owner.lock'
REUSE = ROOT/'results/layered_mismatch_raw_20261013/r0_reuse'
ES = ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'
TS = ROOT/'experiments/wcl-evidence-closure-20261009/scripts'
COMMON = ROOT/'experiments/entropy-main-common500-20261013/common.py'
PYTHON = RT/'envs/frozen_visual_py310/bin/python'
SIZES = (1,2,3,4,5,6,8,10,13,16)
OFFSETS = tuple(sum(x*x for x in SIZES[:m]) for m in range(11))
MISSING_STATES = ((1,0),(1,1),(1,2),(1,3),(2,0),(2,2),(2,4),(2,6),
 (3,0),(3,4),(3,8),(3,12),(4,6),(4,12),(4,18),(4,21),(5,23),(5,27),
 (6,27),(6,32),(6,40),(7,25),(7,50),(8,42),(8,62),(8,84),(8,102))
MODELS = dict(vae='6830fc533a34d52c8f765a7b212782a5b5ef6138992099e8a817ce920fe6fe2b',
 var='b82cb0855b24e1d0d8d0aacebbb2d2d6728c2dcc6509b667dfa697c9330347d5',
 decoder='bf1d64bf8ff7eeda416032daa2ada1654fb37de2553235e85d364211ddfefad1')
CAPS = {'render':dict(model_load=1,var_render=5400,prior_scale=54000,decoder_forward=5400),
 'score':dict(dino_model_load=1,source_embedding=200,reconstruction_dino=5400,image_score=5400)}
FLAGS = dict(matmul_tf32=False,cudnn_tf32=False,precision='highest',cudnn_benchmark=False,
 cudnn_deterministic=True,deterministic=True,threads=6,interop_threads=2)
SCREENING_PINS = {
 'screening_contract.json':'6380037674b15ba2a7346c1af6153832177a38837ab769a5d7558ad91c783d24',
 'screening_approval_addendum.json':'22b3bb973dc965b14c654c14446ba30a3e0b7fe80892ec38eec905e5e1500507',
 'server_input_relocation.json':'00b82102d73f615f0e9338fdb7f77570fe2b0d1b227f801a7264e42e87c9342d'}

def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(obj): return hashlib.sha256(canonical(obj).encode('utf8')).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def pin(path):
    p=Path(path);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def checked(p):
    f=Path(p['path']);require(f.is_file() and sha(f)==p['sha256'],'Changed bound file: '+str(f))
    if 'bytes' in p:require(f.stat().st_size==p['bytes'],'Changed bound file size')
    return f
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:f.write(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def status(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(canonical(obj)+'\n',encoding='utf8');os.replace(temp,p)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def libraries():
    sys.path[:0]=[str(ES),str(TS)]
    return load(COMMON,'_r0_clean_shared_utilities')
def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def token_count(m,K):
    require(type(m) is int and type(K) is int and 1<=m<=8 and 0<=K<SIZES[m]**2,'Illegal raster prefix')
    return OFFSETS[m]+K

def plan(cal,missing,historical):
    records=cal['records'];require(len(records)==200 and len({r['source_id'] for r in records})==200,'Fixed unique cal200 required')
    require([r['selection_slot'] for r in records]==list(range(200)),'Selection order changed')
    require(cal['selection_domain']=='UEP_MISMATCH_CAL200_V1' and cal['uses_quality'] is False,'Source selection changed')
    byid={r['source_id']:r for r in records};keys=[]
    for r in missing['missing_pairs']:
        require(r['source_id'] in byid and r['source_index']==byid[r['source_id']]['source_index'],'Missing-pair source identity')
        token_count(r['m'],r['K']);keys.append((r['source_id'],r['m'],r['K']))
    expected={(r['source_id'],m,K) for r in records for m,K in MISSING_STATES}
    require(len(keys)==5400 and len(set(keys))==5400 and set(keys)==expected,'Exact predeclared 5400 missing cells required')
    old=[(r['source_id'],int(r['m']),int(r['K'])) for r in historical]
    require(len(old)==3400 and len(set(old))==3400 and not(set(old)&expected),'Historical clean cells must be disjoint and complete')
    allstates={tuple(x) for x in missing['all_candidate_terminal_states']}
    require(len(allstates)==44 and set(old)|expected=={(r['source_id'],m,K) for r in records for m,K in allstates},'44 by 200 joint domain does not close')
    return [dict(cell_index=i*27+j,selection_slot=i,source_index=r['source_index'],source_id=r['source_id'],m=m,K=K,T=token_count(m,K))
            for i,r in enumerate(records) for j,(m,K) in enumerate(MISSING_STATES)]

def draft(gpu,max_seconds,draft_name='clean_registration_draft_v2.json'):
    require(type(gpu) is int and gpu in range(4) and 1<=max_seconds<=21600,'GPU or finite six-hour ceiling invalid')
    cp=REUSE/'cal200_sources.json';mp=REUSE/'clean_quality_missing_pairs.json';hp=REUSE/'historical_raw_intact_cal200.csv'
    with hp.open(encoding='utf8',newline='') as f:old=list(csv.DictReader(f))
    cal=read(cp);schedule=plan(cal,read(mp),old)
    origin=ROOT/'results/entropy_main_common500_20261013/registration_v4.json';prior=read(origin)
    bindings=dict(prior['code_bindings'])
    for p in (Path(__file__),Path(__file__).with_name('test_r0_clean_resource_v2.py'),COMMON):bindings[str(p)]=sha(p)
    for p,h in bindings.items():require(sha(p)==h,'Inherited qualified code changed: '+p)
    require(prior['native_identity']['model_state_identity']==MODELS,'Frozen visual model identity differs')
    screening={name:pin(PREP/name) for name in SCREENING_PINS}
    require(all(screening[name]['sha256']==h for name,h in SCREENING_PINS.items()),'Reviewed screening metadata differs')
    source=ROOT/'experiments/layered-mismatch-raw-20261013/r0_screening.py'
    require(sha(source)=='a1968d3196c5ebb4a733b31c7de4bbc1b316e562756b23300723cf6f4b66d7c0','Reviewed screening source differs')
    require(Path(draft_name).name==draft_name and draft_name.startswith('clean_registration_draft') and draft_name.endswith('.json'),'Draft basename required')
    require(len(os.sched_getaffinity(0))>=2,'Two available CPU slots required')
    r=dict(schema='R0_RAW_CLEAN5400_V1',status='DRAFT_REVIEW_REQUIRED',created_unix=time.time(),
        target_gpu=gpu,max_seconds=max_seconds,source_manifest=pin(cp),missing_pairs=pin(mp),historical_clean=pin(hp),
        CPU_slots=sorted(os.sched_getaffinity(0))[-2:],
        sources=cal['records'],schedule=schedule,schedule_sha256=digest(schedule),code_bindings=bindings,
        visual_request=prior['visual_request'],metric_config=prior['metric_config'],native_identity=prior['native_identity'],
        qualified_composition=pin(origin),existing_qualification_reused=True,models=MODELS,
        caps=CAPS,terminal_cells=5400,source_embedding_cap=200,source_embedding_is_not_reconstruction_cell=True,
        screening_bindings=screening,screening_source=pin(source),
        tokenizer_calls=0,training_updates=0,channel_calls=0,packet_calls=0,additional_qualification_cells=0,
        metric='dinov2_vitl14_cosine',requested_metric_models=['dinov2_vitl14'],numeric_flags=FLAGS,
        conditioning='class1000 unconditional; raw raster prefix only; deterministic argmax',
        clean_scope='Original calibration only; valid clean source prefix, not a channel-receiver isolation test',
        old_3400_usage='Historical4090D exact-clean screening evidence; unchanged and not represented as H800 reruns',
        process_order=['render','score'],batch_size=1,output_root=str(OUT),runtime_root=str(TMP),automatic_retry=False,
        resource_policy=dict(RESOURCE_POLICY),
        timing_exclusion=dict(lock_path=str(TIMING_LOCK),lock_mode='shared nonblocking; existing v6 uses exclusive',
            held_until_actual_child_wait=True,inherited_by_children=True,old_timing_code_changed=False),
        authorization='Human approved independent R0 supplement: at most5400 missing clean VAR plus DINO-L pairs; max200 source embeddings separately accounted; no R1 or development selection launched here')
    write(PREP/draft_name,r)
    write(PREP/(Path(draft_name).stem+'_notes.json'),dict(registration=pin(PREP/draft_name),
        science_calls=0,renderer='Unmodified SHA-bound h_source_driver.render_received(native,prefix,m,K)',
        metric='Unmodified MetricEvaluator requested=(dinov2_vitl14,); original prepare_reference/score; private feature wrapper only counts/hashes existing outputs',
        source_reference_calls=200,reconstruction_calls=5400,extra_determinism_rerenders=0,
        determinism_qualification='Reuse original H800 qualification; verify original flags/model state and hash each scheduled result. No new bitwise repeatability claim without repeated calls.',
        review_then_run='Pass exact approved registration SHA and --reviewed; WAIT admission before any child/model; no existing launch is retried'))
    return pin(PREP/draft_name)

def review_gate(r):
    require(r.get('status')=='REGISTERED_REVIEWED' and 'review_approval' in r,'Drafts cannot run; registered independent approval required')
    a=read(checked(r['review_approval']))
    require(a.get('approved') is True and a.get('entry_sha256')==r['code_bindings'][str(Path(__file__))]
        and a.get('test_sha256')==r['code_bindings'][str(Path(__file__).with_name('test_r0_clean_resource_v2.py'))]
        and a.get('source_manifest_sha256')==r['source_manifest']['sha256'] and a.get('schedule_sha256')==r['schedule_sha256']
        and a.get('clean_VAR_cap')==5400 and a.get('dino_reconstruction_cap')==5400 and a.get('source_DINO_cap')==200
        and a.get('scientific_calls_during_review')==0,'Independent review scope/binding differs')

RESOURCE_POLICY = dict(prelaunch_free_GiB=15,runtime_free_plus_own_GiB=14,allocator_cap_GiB=12,absolute_free_GiB=2,
 math_threads=6,interop_threads=2,scheduling_CPUs=2,overlap_with_v6_timing=False)


def retirement_gate(r,prelaunch=False):
    """Verify the pinned retired zero-science predecessor; no process mutation."""
    revision=r['resource_revision'];receipt=read(checked(revision['retirement']))
    require(receipt.get('status')=='RETIRED_ZERO_SCIENCE','Original waiter has not retired without science')
    require(all(type(receipt.get(k)) is int and receipt[k]==0 for k in
        ('owner_invocations','scientific_owner_launch_count','scientific_calls')),'Retirement is not zero science')
    process=receipt['process']
    require(type(process['pid']) is int and process['pid']>0 and type(process['start_ticks']) is int
        and process['start_ticks']>0,'Retired waiter identity absent')
    actual=process_identity(process['pid'])
    require(actual is None or actual['start_ticks']!=process['start_ticks'],'Original waiter identity is still present')
    require(receipt.get('observed_gone') is True and receipt.get('pidfd_exit_observed') is True,'Original waiter exit was not observed')
    if prelaunch:
        forbidden=(TMP/'owner_launch.json',TMP/'render',TMP/'score',TMP/'owner_failure.json',
            TMP/'render_actual_wait.json',TMP/'score_actual_wait.json',OUT/'completion.json',OUT/'clean_quality.csv')
        require(not any(p.exists() for p in forbidden),'Original clean science was already attempted')
    return receipt


def registration(path,expected):
    require(sha(path)==expected,'Registration SHA differs from reviewed bytes');r=read(path)
    review_gate(r)
    require(r['resource_policy']==RESOURCE_POLICY,'Exact reviewed resource revision required')
    require(r['target_gpu']==3 and r['CPU_slots']==[178,179],'Frozen GPU3/CPU scheduling changed')
    retirement_gate(r)
    require(r['schema']=='R0_RAW_CLEAN5400_V1' and r['terminal_cells']==5400 and r['caps']==CAPS and r['models']==MODELS,'Scope/budget/model changed')
    require(r['numeric_flags']==FLAGS and r['requested_metric_models']==['dinov2_vitl14'],'Metric scope changed')
    require(all(r[k]==0 for k in ('tokenizer_calls','training_updates','channel_calls','packet_calls','additional_qualification_cells')),'Unregistered scientific call')
    require(r['output_root']==str(OUT) and r['runtime_root']==str(TMP) and r['automatic_retry'] is False,'Storage/retry boundary changed')
    require(0<=r['target_gpu']<=3 and 1<=r['max_seconds']<=21600,'Resource or wall ceiling changed')
    require(len(r['CPU_slots'])==2 and len(set(r['CPU_slots']))==2 and all(type(x) is int and x>=0 for x in r['CPU_slots']),'Exactly two registered scheduling CPUs')
    for p,h in r['code_bindings'].items():require(sha(p)==h,'Bound code changed: '+p)
    for name,h in SCREENING_PINS.items():require(r['screening_bindings'][name]['sha256']==h and sha(checked(r['screening_bindings'][name]))==h,'Reviewed screening pin changed')
    checked(r['screening_source'])
    with checked(r['historical_clean']).open(encoding='utf8',newline='') as f:old=list(csv.DictReader(f))
    cal=read(checked(r['source_manifest']));s=plan(cal,read(checked(r['missing_pairs'])),old)
    require(s==r['schedule'] and digest(s)==r['schedule_sha256'] and cal['records']==r['sources'],'Registered schedule/source changed')
    checked(r['visual_request']);return r

def process_identity(pid):
    try:
        p=Path('/proc')/str(pid);stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
        return dict(pid=int(pid),start_ticks=int(tail[19]),state=tail[0],argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace'))
    except (OSError,ValueError):return None

def active_timing_processes():
    found=[]
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            rec=process_identity(int(p.name))
            if rec and rec['state']!='Z' and 'baseline-delivery-20261012/timing_' in rec['argv'] and any(x in rec['argv'] for x in ('v6.py','six_method_v6')):found.append(rec)
    return found

def resource_adapter_sources(resource):
    """Only scheduling and allocator limits differ from the original guard."""
    assessor=inspect.getsource(resource.assess_snapshot)
    replacements={
        'free>=20*GiB if prelaunch else free>=4*GiB':'free>=15*GiB if prelaunch else free>=2*GiB',
        'free+own>=20*GiB':'free+own>=14*GiB',
        '0<=own<=16*GiB':'0<=own<=12*GiB'}
    for old,new in replacements.items():
        require(assessor.count(old)==1,'Resource adapter shape changed: '+old)
        assessor=assessor.replace(old,new)
    observer=inspect.getsource(resource.own_allocator_observation)
    require(observer.count('<=16*GiB')==1,'Allocator observation adapter shape changed')
    observer=observer.replace('<=16*GiB','<=12*GiB').replace('frozen16GiB cap','reviewed12GiB cap')
    return assessor,observer


def resource_device(r,out):
    libraries();import h800_alternate_gpu_resource_v3 as resource
    ns=dict(vars(resource));policy=dict(resource.POLICY_BASE)
    policy.update(schema='R0_CLEAN_SHARED_RESOURCE_V2',prelaunch_free_bytes=15<<30,
        runtime_free_plus_own_reserved_bytes=14<<30,runtime_free_bytes=2<<30,own_allocator_cap_bytes=12<<30)
    ns['POLICY_BASE']=policy;f=resource.policy
    ns['policy']=types.FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
    assessor,observer=resource_adapter_sources(resource)
    exec(compile(assessor+'\n'+observer,resource.__file__,'exec'),ns)
    f=resource.DeviceAdapter.resource_snapshot;replacement=types.FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
    def bind_evidence(self,directory,request_sha):
        p=resource.g.inside(directory,base=TMP);require(p.is_dir() and resource.g.SHA_RE.fullmatch(request_sha),'Evidence path')
        self.evidence_directory=p;self.request_sha=request_sha
    cls=type('R0CleanSharedDevice',(resource.DeviceAdapter,),dict(resource_snapshot=replacement,bind_resource_evidence=bind_evidence))
    old=read(checked(r['visual_request']));d=cls(old['device_receipt'],r['target_gpu']);d.bind_resource_evidence(out,digest(r));return d,resource.g


def adapt_backend_limited(module,device,engine):
    """Keep the reviewed device injection and limit allocation before model load."""
    baseline,prior_proof=engine.adapt_backend(module,device)
    path=Path(module.__file__);tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Backend')
    changed=copy.deepcopy(original)
    init=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    imports=[(i,n) for i,n in enumerate(init.body) if isinstance(n,ast.Import) and len(n.names)==1
        and n.names[0].name=='leo_whole_gate_v1' and n.names[0].asname=='g']
    require(len(imports)==1,'Exactly one original gate injection required')
    i,old_import=imports[0]
    init.body[i]=ast.copy_location(ast.Assign(targets=[ast.Name(id='g',ctx=ast.Store())],
        value=ast.Name(id='_explicit_device_adapter',ctx=ast.Load())),old_import)
    caps=[(j,n) for j,n in enumerate(init.body) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='cap']
    require(len(caps)==1 and ast.dump(caps[0][1].value,include_attributes=False)==
        ast.dump(ast.parse('16*(1<<30)',mode='eval').body,include_attributes=False),'Original allocator assignment changed')
    j,old_cap=caps[0];old_cap=copy.deepcopy(old_cap)
    init.body[j].value=ast.copy_location(ast.parse('12*(1<<30)',mode='eval').body,init.body[j].value)
    restored=copy.deepcopy(changed)
    restored_init=next(n for n in restored.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    restored_init.body[i]=copy.deepcopy(old_import);restored_init.body[j]=old_cap
    before=ast.dump(original,include_attributes=False);after=ast.dump(restored,include_attributes=False)
    require(before==after,'Resource adaptation changed original mathematics')
    require(hashlib.sha256(before.encode()).hexdigest()==prior_proof['original_AST_sha256'],
        'Resource and original device-adapter source differ')
    ns=dict(baseline.__init__.__globals__)
    require(ns.get('_explicit_device_adapter') is device,'Original explicit adapter was not retained')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(path),'exec'),ns)
    proof=dict(schema='R0_CLEAN_ALLOCATOR_CAP_ONLY_V2',device_adapter=prior_proof,
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        inverse_restored_AST_sha256=hashlib.sha256(after.encode()).hexdigest(),allocator_cap_bytes=12<<30,
        only_additional_change='Backend.__init__ cap assignment 16GiB to12GiB before any model construction',
        mathematics_changed=False,extra_model_calls=0)
    return ns['Backend'],proof

def admission(r,gate):
    active=active_timing_processes()
    if active:return dict(status='WAIT',reason='Registered v6 timing process is active',timing_processes=active,model_calls=0)
    device,g=resource_device(r,gate)
    try:
        snap,_=device.resource_snapshot(g.helper(),prelaunch=True)
        return dict(status='READY',snapshot=snap,model_calls=0)
    except Exception as e:
        if type(e).__name__!='ResourceBusy':raise
        return dict(status='WAIT',reason=str(e),model_calls=0)

def validate_source(record):
    import numpy as np
    with np.load(checked(record['archive']),allow_pickle=False) as z:pixels=z['pixels'].copy();tokens=z['tokens'].copy()
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256),'Original source pixel type/shape')
    require(tokens.dtype==np.int64 and tokens.shape==(680,) and np.all((tokens>=0)&(tokens<4096)),'Original680 tokens required')
    require(hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],'Source pixel identity changed')
    require(hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==record['tokens_sha256'],'Source token identity changed')
    checked(record['checkpoint']);return pixels,tokens

def validate_image(image):
    import numpy as np
    require(isinstance(image,np.ndarray) and image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and image.min()>=0 and image.max()<=1,'Original float32 RGB image required')

def render_one(b,tokens,cell,ledger):
    prefix=tokens[:token_count(cell['m'],cell['K'])].copy();before=array_sha(prefix)
    b.source_index=cell['source_index']
    with b.t.no_grad():image=ledger.call('var_render',lambda:b.render_function(b.native,prefix,cell['m'],cell['K']),cell=cell['cell_index'])
    require(array_sha(prefix)==before,'Renderer modified its accepted prefix');validate_image(image)
    return image,before

def guard_for(r,parent,stage):
    launch=read(TMP/'owner_launch.json');expected=launch['process'];start=time.monotonic()
    def guard():
        actual=process_identity(parent)
        require(os.getppid()==parent and actual is not None and actual['start_ticks']==expected['start_ticks'],'Registered owner lost')
        require(time.time()<launch['deadline_unix'] and time.monotonic()-start<r['max_seconds'],'Finite execution deadline')
        require(not any(p.exists() for p in (OUT/'STOP',TMP/'STOP',stage/'STOP')),'Requested STOP')
    return guard

def worker(path,expected,stage,parent):
    r=registration(path,expected);out=TMP/stage;out.mkdir();guard=guard_for(r,parent,out);util=libraries()
    device,g=resource_device(r,out);device.check_cuda_environment();last=0.;history=[];ledger=None;b=None
    def boundary(force=False):
        nonlocal last
        guard()
        if force or time.monotonic()-last>=5:
            require(not active_timing_processes(),'Timing became active; no further scientific calls')
            snap,_=device.resource_snapshot(g.helper());history.append(snap);last=time.monotonic();status(out/'resource_latest.json',snap)
    try:
        # Recheck in the owned child before any model/ledger reservation. A race
        # with a new resident job remains WAIT, never a partially launched model.
        while True:
            guard();ready=admission(r,out)
            if ready['status']=='READY':break
            status(OUT/(stage+'_pre_model_wait.json'),dict(ready,observed_unix=time.time()));time.sleep(30)
        ledger=util.Ledger(out/'calls.sqlite',CAPS[stage],guard)
        write(out/'launch.json',dict(process=process_identity(os.getpid()),registration=pin(path),stage=stage))
        if stage=='render':
            import h800_ep_new100_visual_v3 as visual
            old=read(checked(r['visual_request']));spec=dict(old['spec']);spec.update(deadline_unix=time.time()+900,max_seconds=900)
            resolver,environment,_,_,extra=visual.g.validate_spec(spec)
            module=load(ES/'leo_whole_math_v1.py','_r0_clean_frozen_backend')
            Backend,proof=adapt_backend_limited(module,device,visual.engine());write(out/'resource_adapter_AST.json',proof)
            b=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
            require(dict(b.g.MODELS)==MODELS,'Actual model state identity changed')
            rows=[]
            for record in r['sources']:
                boundary();_,tokens=validate_source(record)
                for cell in r['schedule'][record['selection_slot']*27:(record['selection_slot']+1)*27]:
                    boundary();image,prefix_sha=render_one(b,tokens,cell,ledger)
                    ip=out/'images'/f"{cell['cell_index']:04d}.npz";ip.parent.mkdir(exist_ok=True)
                    with ip.open('xb') as f:__import__('numpy').savez(f,image=image)
                    row=dict(cell,source_archive=record['archive'],source_preprocessing_id=record['preprocessing_id'],source_tokens_sha256=record['tokens_sha256'],
                        prefix_array_sha256=prefix_sha,image_array_sha256=array_sha(image),image_archive=pin(ip),model_identity=digest(MODELS),conditioning='unconditional_class1000',order='raw_raster')
                    write(out/'rows'/f"{cell['cell_index']:04d}.json",row);rows.append(row)
                status(OUT/'render_status.json',dict(completed=len(rows),total=5400,counts=ledger.summary()))
            b.close();resolver.reverify();environment.reverify();write(out/'frames.json',rows)
            require(ledger.completed['var_render']==5400 and ledger.completed['decoder_forward']==5400 and ledger.completed['prior_scale']==54000,'Complete clean render budget')
            result=dict(frames=pin(out/'frames.json'),cell_count=5400,models=MODELS,extra_qualification_calls=0)
        else:result=score_stage(r,out,ledger,boundary)
        boundary(True);counts=ledger.summary();require(counts['unresolved']==0,'Incomplete science calls')
        write(out/'resource_history.json',history);write(out/'completion.json',dict(status='COMPLETE',stage=stage,counts=counts,resource_history=pin(out/'resource_history.json'),**result));ledger.close()
    except BaseException as e:
        if b is not None:b.failure_evidence()
        write(out/'resource_history_at_failure.json',history)
        write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

def score_stage(r,out,ledger,boundary):
    import numpy as np
    require(read(TMP/'render/actual_wait.json')['returncode']==0,'Render actual exit0 required')
    done=read(TMP/'render/completion.json');frames=read(checked(done['frames']))
    require([{k:row[k] for k in r['schedule'][0]} for row in frames]==r['schedule'],'Actual render grid differs')
    import h800_ep_pilot_metrics_v1 as metric
    assets=metric.bound_assets(r['metric_config']);import torch
    torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    device,g=resource_device(r,out);device.check_cuda_environment();torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction((12<<30)/props.total_memory)
    module=load(ES/'leo_whole_math_v1.py','_r0_clean_metric_device')
    identity=device.validate_cuda_device_identity(str(props.uuid),module.cuda_driver_pci_bus_id())
    import platform
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    require(runtime==device.EXPECTED_RUNTIME and sorted(os.sched_getaffinity(0))==r['CPU_slots'],'Qualified metric runtime/affinity differs')
    flags=metric.flags_function(r['metric_config']['sources']['step0_reference_metrics.py']['path'])
    require(flags(torch)==FLAGS,'Original numerical metric settings differ')
    original=metric.load_module(r['metric_config']['sources']['metric_models.py'],'_r0_clean_original_dino')
    boundary(True)
    with original.no_network():evaluator=ledger.call('dino_model_load',lambda:original.MetricEvaluator(assets['dinov2_manifest'],device='cuda:0',requested=('dinov2_vitl14',)))
    require(set(evaluator.models)=={'dinov2_vitl14'},'Additional metric model constructed')
    evaluator.metadata['metric_selection_usage']='Original calibration-only clean-Q supplement; no holdout/development selection'
    context={};feature_original=original.dinov2_vitl14_features
    def observed_feature(model,image):
        feature=ledger.call(context['kind'],lambda:feature_original(model,image),cell=context.get('cell'),source_id=context['source_id'])
        values=feature.detach().cpu().numpy();require(values.dtype==np.float32 and values.shape==(1,1024) and np.isfinite(values).all(),'DINO-L feature shape')
        context['feature_sha256']=array_sha(values);return feature
    original.dinov2_vitl14_features=observed_feature
    meta=dict(original_source=r['metric_config']['sources']['metric_models.py'],DINO_L_metadata=evaluator.metadata,
        metric_identity=evaluator.identity(),numeric_flags=flags(torch),GPU=identity,numerical_runtime=runtime,CPU_slots=r['CPU_slots'],source_feature_preparations_cap=200,
        original_reference_and_score_code_unchanged=True,feature_wrapper='Private module wrapper records count/hash of actual original feature return; no additional forward',
        extra_metric_models=0,batch_size=1,source_and_reconstruction_preprocessing_identical=True)
    write(out/'metric_identity.json',meta);rows=[];refs=[];prepared=None;record=None
    for row in frames:
        boundary();require(flags(torch)==FLAGS,'Numerical flags changed')
        if record is None or record['source_id']!=row['source_id']:
            record=r['sources'][row['selection_slot']];pixels,_=validate_source(record)
            target=pixels.astype(np.float32)/255;validate_image(target);x=torch.as_tensor(target[None],dtype=torch.float32,device='cuda:0')
            context.clear();context.update(kind='source_embedding',source_id=record['source_id'])
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):prepared=evaluator.prepare_reference(x)
            fp=out/'source_features'/f"{record['selection_slot']:03d}.npz";fp.parent.mkdir(exist_ok=True)
            with fp.open('xb') as f:np.savez(f,feature=prepared['dinov2_vitl14'].numpy())
            refs.append(dict(source_id=record['source_id'],preprocessing_id=record['preprocessing_id'],input_array_sha256=array_sha(target),
                feature_sha256=context['feature_sha256'],feature_archive=pin(fp),metric_identity=evaluator.identity(),cache_key=digest([record['preprocessing_id'],evaluator.identity()])))
        with np.load(checked(row['image_archive']),allow_pickle=False) as z:image=z['image'].copy()
        validate_image(image);require(array_sha(image)==row['image_array_sha256'],'Actual reconstruction bytes changed')
        y=torch.as_tensor(image[None],dtype=torch.float32,device='cuda:0');context.clear()
        context.update(kind='reconstruction_dino',source_id=row['source_id'],cell=row['cell_index'])
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            value=ledger.call('image_score',lambda:evaluator.score(x,y,[record['evaluation_class_index']],label_conditioned=False,prepared=prepared),cell=row['cell_index'])[0]
        require(set(value)=={'dinov2_vitl14_cosine','label_conditioned'} and value['label_conditioned'] is False,'Unexpected metric output')
        q=float(value['dinov2_vitl14_cosine']);require(math.isfinite(q) and -1.000001<=q<=1.000001,'Invalid DINO cosine')
        public={k:row[k] for k in ('cell_index','selection_slot','source_index','source_id','m','K','T','source_preprocessing_id','source_tokens_sha256','prefix_array_sha256','image_array_sha256','model_identity')}
        public.update(dinov2_vitl14_cosine=q,metric_identity=evaluator.identity(),source_feature_sha256=refs[-1]['feature_sha256'],
            reconstructed_feature_sha256=context['feature_sha256'],origin='H800_NEW_CLEAN_RASTER',image_path=row['image_archive']['path'])
        write(out/'rows'/f"{row['cell_index']:04d}.json",public);rows.append(public)
        if len(rows)%27==0:status(OUT/'score_status.json',dict(completed=len(rows),total=5400,counts=ledger.summary()))
    require(len(rows)==5400 and ledger.completed['source_embedding']==200 and ledger.completed['reconstruction_dino']==5400 and ledger.completed['image_score']==5400,'Complete finite DINO budget')
    with (OUT/'clean_quality.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'source_embeddings.json',refs);torch.cuda.synchronize()
    return dict(cell_count=5400,clean_quality=pin(OUT/'clean_quality.csv'),source_embeddings=pin(out/'source_embeddings.json'),metric_identity=pin(out/'metric_identity.json'),source_embeddings_actual=200,bootstrap_calls=0)

def wait_child(child,deadline):
    timed_out=False;terminated=False;killed=False
    try:code=child.wait(timeout=max(0.001,deadline-time.time()))
    except subprocess.TimeoutExpired:
        timed_out=True;terminated=True;child.terminate()
        try:code=child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();killed=True;code=child.wait()
    return dict(returncode=code,timed_out=timed_out,terminated=terminated,killed=killed,actual_wait=True,pid=child.pid)

def owner(path,expected,reviewed):
    require(reviewed,'Explicit root review required before scientific owner')
    registration(path,expected);TMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    # Original v6's innermost owner takes an exclusive lock on this exact file.
    # Shared lock lets the two independent R0 supplement owners coexist while
    # excluding timing. Children inherit the descriptor across exec, so owner
    # death cannot release exclusion while a model child is still alive.
    with TIMING_LOCK.open('a+') as timing_lock,(TMP/'owner.lock').open('a+') as own_lock:
        try:fcntl.flock(timing_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:
            result=dict(status='WAIT',reason='Existing timing owner lock held',model_calls=0)
            status(OUT/'resource_wait.json',dict(result,observed_unix=time.time()));return result
        try:
            fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return _owner_locked(path,expected,reviewed,(timing_lock.fileno(),own_lock.fileno()))
        finally:
            # Both children have actually exited before this scope returns.
            # On interruption _owner_locked also performs actual termination wait.
            fcntl.flock(timing_lock,fcntl.LOCK_UN)

def _owner_locked(path,expected,reviewed,lock_fds):
    require(reviewed,'Explicit root review required before scientific owner')
    r=registration(path,expected);TMP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    require(not (TMP/'owner_launch.json').exists(),'Existing attempt requires separate reviewed recovery; no automatic retry')
    retirement_gate(r,prelaunch=True)
    gate=TMP/'prelaunch';gate.mkdir(exist_ok=True);ready=admission(r,gate);status(OUT/'resource_wait.json',dict(ready,observed_unix=time.time()))
    if ready['status']=='WAIT':return ready
    write(TMP/'owner_launch.json',dict(process=process_identity(os.getpid()),registration=pin(path),root_reviewed=True,deadline_unix=time.time()+r['max_seconds']))
    child=None;receipts=[]
    try:
        for stage in ('render','score'):
            require(not (TMP/stage).exists(),'Stage already attempted')
            while True:
                require(time.time()<read(TMP/'owner_launch.json')['deadline_unix'] and not (OUT/'STOP').exists(),'Owner wait deadline/STOP')
                ready=admission(r,gate);status(OUT/'resource_wait.json',dict(ready,stage=stage,observed_unix=time.time()))
                if ready['status']=='READY':break
                time.sleep(30)
            env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(r['target_gpu']),TMPDIR=str(TMP/'cache'))
            (TMP/'cache').mkdir(exist_ok=True)
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='6'
            cpus=r['CPU_slots'];require(set(cpus).issubset(os.sched_getaffinity(0)),'Registered CPU slots unavailable');start=time.monotonic()
            command=[str(PYTHON),str(Path(__file__)),'worker','--registration',str(path),'--expected-registration',expected,'--stage',stage,'--parent',str(os.getpid())]
            with (TMP/(stage+'.stdout.log')).open('xb') as stdout,(TMP/(stage+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=ROOT,env=env,pass_fds=lock_fds,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
                actual=wait_child(child,read(TMP/'owner_launch.json')['deadline_unix'])
            code=actual['returncode'];actual.update(elapsed_seconds=time.monotonic()-start,stage=stage);child=None
            write(TMP/(stage+'_actual_wait.json'),actual)
            if (TMP/stage).is_dir():write(TMP/stage/'actual_wait.json',actual)
            require(code==0 and not actual['timed_out'],'Actual child failure/timeout: '+stage)
            completion=read(TMP/stage/'completion.json');require(completion['status']=='COMPLETE' and completion['cell_count']==5400,'Stage did not close5400')
            receipts.append(dict(stage=stage,actual_wait=actual,completion=pin(TMP/stage/'completion.json')))
        write(OUT/'completion.json',dict(status='COMPLETE_CLEAN5400',registration=pin(path),stages=receipts,all_actual_exit_zero=True,
            clean_quality=pin(OUT/'clean_quality.csv'),clean_VAR_actual=5400,reconstruction_DINO_actual=5400,source_DINO_actual=200,
            extra_qualification_reconstructions=0,training_updates=0,channel_calls=0,packet_calls=0,tokenizer_calls=0,
            historical_3400_unchanged=r['historical_clean'],complete_joint_terminal_domain=8800,selection_or_R1_started=False))
        return dict(status='COMPLETE_CLEAN5400')
    except BaseException as e:
        if child is not None and child.poll() is None:
            child.terminate();write(TMP/'interrupted_actual_wait.json',wait_child(child,time.time()+15))
        write(TMP/'owner_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),completed_stages=receipts));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker'));p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--max-seconds',type=int,default=21600);p.add_argument('--registration',type=Path);p.add_argument('--expected-registration')
    p.add_argument('--draft-name',default='clean_registration_draft_v2.json')
    p.add_argument('--reviewed',action='store_true');p.add_argument('--stage',choices=('render','score'));p.add_argument('--parent',type=int)
    a=p.parse_args()
    if a.action=='prepare':print(json.dumps(draft(a.gpu,a.max_seconds,a.draft_name)))
    elif a.action=='run':print(json.dumps(owner(a.registration,a.expected_registration,a.reviewed)))
    else:
        require(a.stage is not None and a.parent is not None,'Owned worker arguments required');worker(a.registration,a.expected_registration,a.stage,a.parent)

if __name__=='__main__':main()
