"""Exact actual-input native cache, independent native workers, post-exit closure."""
from analysis_common import *
import argparse, collections, ctypes
import copy, ast

NATIVE_SECONDS=43200
NEST_ID_FIELDS=('model_state_identity','numeric_settings','native_binding',
                'received_render_core_SHA','source_codec_SHA','source_math_SHA')

def canonical_arpc_runtime(v):
    x=copy.deepcopy(v)
    for k in ('pid','gpu_name','gpu_uuid','started_unix'):x.pop(k,None)
    x['native_arguments'].pop('cache_dir',None)
    return x

def canonical_arpc_config(v):
    keys=('code','upstream','environment','weights_root','preset','input_shape','endpoints',
          'seed','cfg','tau','native_float_tolerance','required_exact_uint8','wireless_frames')
    x={k:v[k] for k in keys};x['resource']={k:a for k,a in v['resource'].items() if k!='gpu'}
    return x

def check_nest_identity(v,routes):
    q=routes['native_qualification']
    assert v['models']==q['model_state_identity']
    assert v['runtime']==q['numeric_settings']['runtime']
    assert v['numeric_flags']==routes['numeric_flags']
    assert v['sampling']=='unconditional_class1000_greedy_no_CFG' and v['training_updates']==0
    assert v['source_mathematics']=='Unchanged SHA-bound original function/class definitions'

def same_native_registration(a,b):
    for k in ('model_routes','static','weights'):assert a[k]==b[k],('BLOCKED_NATIVE_REGISTRATION_CHANGED',k)

def strict_wait(v):
    assert type(v['returncode']) is int and v['returncode']==0
    for k in ('timed_out','terminated','killed'):assert v.get(k,False) is False
    assert v.get('metadata_error') is None

def native_lineage(b1):
    """Closed metadata only. Original scientific output and source pins stay intact."""
    full=BASE/'arpc_n2048_validation_20261019_v1';fo=R/'results/arpc_n2048_validation_20261019_v1'
    fr=read(fo/'registration.json');assert fr['model_routes']==b1['model_routes'] and fr['static']==b1['static']
    routes=load(b1['model_routes']);template=read(full/'formal/arpc_resource_v1/public.json')
    reference=read(full/'formal/arpc_resource_v1/rx/runtime.json')
    proofs={};members={}
    for name in ('a2_clean_v1','a2b_rx_v1','a3_rx_v1'):
        rr=OLD_L/name;oo=OLD_O/name;reg=read(oo/'registration.json');cl=read(oo/'root_closure_review.json')
        assert cl['all_processes_gone'] is True
        assert cl.get('unresolved',0)==0 and not cl.get('new_failure_files',[])
        if 'registration' in cl:assert load(cl['registration'])==reg
        else:assert cl['registration_sha256']==sha(oo/'registration.json')
        for d in cl.get('controls',cl.get('receipts',[])):
            if 'actual_wait' in Path(d['path']).name:strict_wait(load(d))
        assert reg['weights']==fr['weights']
        for p in (OLD_E/'a1_native.py',OLD_E/'wire.py',OLD_E/'cuda_recovery_v1/runtime.py'):
            verify(next(d for d in reg['source_files'] if d['path']==str(p)))
        cfg=load(reg['public']);runtime=read(rr/'rx/runtime.json')
        assert canonical_arpc_config(cfg)==canonical_arpc_config(template)
        assert canonical_arpc_runtime(runtime)==canonical_arpc_runtime(reference)
        if name=='a2_clean_v1':
            files=read(oo/'closed_files.json')['files'];sp=oo/'clean_states.json'
            for p in (rr/'rx/runtime.json',rr/'rx/completion.json'):
                d=next(x for x in files if x['path']==str(p));verify(d)
            completion=load(next(x for x in cl['receipts'] if x['path']==str(oo/'completion.json')))
            rows=load(next(x for x in completion['outputs'] if x['path']==str(sp)))['states']
        else:
            sp=oo/'actual_states.json';d=next(x for x in cl['outputs'] if x['path']==str(sp));rows=load(d)
        proofs[name]=dict(registration=stable_pin(oo/'registration.json'),closure=stable_pin(oo/'root_closure_review.json'),
            configuration=reg['public'],runtime=stable_pin(rr/'rx/runtime.json'),states=stable_pin(sp))
        if name=='a2_clean_v1':
            # Its 700 states include 24 already-closed A1 outputs. Bind that
            # lineage too; A1 requested only a subset of the same endpoints.
            for d in reg['prior']:
                verify(d)
                if 'actual_wait' in Path(d['path']).name:strict_wait(load(d))
            a1reg=load(next(d for d in reg['prior'] if d['path'].endswith('a1_registration.json')))
            assert a1reg['weights']==fr['weights']
            a1cfg=load(a1reg['public']);a=canonical_arpc_config(a1cfg);b=canonical_arpc_config(template)
            assert set(a.pop('endpoints'))<=set(b.pop('endpoints')) and a==b
            a1p=OLD_L/'a1_cuda_recovery_v1/rx/runtime.json'
            closed=load(next(d for d in reg['prior'] if d['path'].endswith('a1_closed_files.json')))
            if isinstance(closed,dict):closed=closed['files']
            verify(next(d for d in closed if d['path']==str(a1p)))
            assert canonical_arpc_runtime(read(a1p))==canonical_arpc_runtime(reference)
            proofs[name]['inherited_A1']=dict(prior=reg['prior'],configuration=a1reg['public'],runtime=stable_pin(a1p))
        members[name]={json.dumps(x,sort_keys=True,separators=(',',':')) for x in rows}
    for name,codec in (('nest',0),('arpc_resource_v1',1)):
        run=full/'formal'/name;kind='nest' if codec==0 else 'arpc';cl=read(fo/'formal'/(kind+'_process_closure.json'))
        assert cl['status']=='GONE'
        # The formal closure pins the actual wait, supervisor and completion.
        descriptors=[]
        def collect(v):
            if isinstance(v,dict):
                if {'path','sha256'}<=set(v):descriptors.append(v)
                else:
                    for x in v.values():collect(x)
            elif isinstance(v,list):
                for x in v:collect(x)
        collect(cl)
        cp=run/'rx/completion.json';cd=next(x for x in descriptors if x['path']==str(cp));comp=load(cd)
        for d in descriptors:
            if 'actual_wait' in Path(d['path']).name:strict_wait(load(d))
        cfg=read(run/'public.json')
        ip=run/'rx'/('runtime.json' if codec else 'actual_native_identity.json')
        if codec:
            assert canonical_arpc_config(cfg)==canonical_arpc_config(template)
            assert canonical_arpc_runtime(read(ip))==canonical_arpc_runtime(reference)
        else:
            assert cfg['model_routes']==b1['model_routes'] and cfg['static']==b1['static'];check_nest_identity(read(ip),routes)
        key='N2048_FULL_RX_'+str(codec)
        proofs[key]=dict(registration=stable_pin(fo/'registration.json'),closure=stable_pin(fo/'formal'/(kind+'_process_closure.json')),
            configuration=stable_pin(run/'public.json'),runtime=stable_pin(ip),completion=cd)
        members[key]={json.dumps(x,sort_keys=True,separators=(',',':')) for x in comp['outputs']}
    for name,run in [('qualification',old.L/'qualification/native'),('cal100',old.L/'cal100_clean_v1')]:
        cfg=read(run/'public.json');assert cfg['model_routes']==b1['model_routes'] and cfg['static']==b1['static']
        check_nest_identity(read(run/'rx/actual_native_identity.json'),routes)
        control=old.L/'controls'/('qualification_native' if name=='qualification' else 'cal100_clean_native')
        strict_wait(read(control/'actual_wait.json'))
        regpath=old.O/('qualification/review.json' if name=='qualification' else 'cal100_clean_registration.json')
        saved=read(regpath)
        if name=='qualification':assert saved['status']=='PASS' and saved['actual_opened'] is True
        else:assert load(saved['public'])==cfg and load(saved['original_registration'])==b1
        proofs[name]=dict(registration=stable_pin(regpath),configuration=stable_pin(run/'public.json'),
            runtime=stable_pin(run/'rx/actual_native_identity.json'),actual_wait=stable_pin(control/'actual_wait.json'),completion=stable_pin(run/'rx/completion.json'))
    return dict(proofs=proofs,arpc_reference_runtime=stable_pin(full/'formal/arpc_resource_v1/rx/runtime.json'),
        environment_policy=stable_pin(OLD_L/'environment_preparation_v1/policy.json'),
        environment_completion=stable_pin(OLD_L/'environment_preparation_v1/completion.json')),members

def require_member(c,lineage,members):
    s,meta=c['source'],c['cache'];origin=meta['origin']
    if origin=='ORIGINAL_ENTROPY_RX':return
    name=('N2048_FULL_RX_'+str(s['codec']) if origin=='N2048_FULL_RX' else
          'a2_clean_v1' if origin=='ARPC_CLEAN_INDEPENDENT_RX' else origin)
    assert name in lineage['proofs'] and name in members,('UNBOUND_NATIVE_ORIGIN',origin)
    assert json.dumps(meta['image']['evidence'],sort_keys=True,separators=(',',':')) in members[name]
    c['native_lineage']=lineage['proofs'][name]

# Run by the selected /usr/bin/python3, with the same independent overlay as RX.
# Only stdlib and installed distribution metadata are read; no tensor/model import.
METADATA_PROBE=r'''
import sys,os,json,hashlib,ast,importlib.metadata,importlib.machinery
from pathlib import Path
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
policy=json.loads(Path(sys.argv[1]).read_text());overlay=Path(policy['target']).resolve()
assert sys.executable==policy['base_python']=='/usr/bin/python3'
packages=[]
for spec in policy['packages']+['flash-attn==2.8.3.post2']:
 name,version=spec.split('==');d=importlib.metadata.distribution(name)
 assert d.version==version,(name,d.version,version)
 meta=Path(d._path).resolve();assert meta.is_relative_to(overlay),(name,str(meta))
 packages.append(dict(name=name,version=d.version,METADATA=pin(meta/'METADATA'),RECORD=pin(meta/'RECORD')))
d=importlib.metadata.distribution('torch');origin=importlib.machinery.PathFinder.find_spec('torch',sys.path).origin
vp=Path(origin).with_name('version.py');values={}
for n in ast.parse(vp.read_text()).body:
 if isinstance(n,(ast.Assign,ast.AnnAssign)):
  names=n.targets if isinstance(n,ast.Assign) else [n.target]
  for t in names:
   if isinstance(t,ast.Name) and t.id in ('__version__','cuda','git_version'):
    values[t.id]=ast.literal_eval(n.value)
assert values['__version__']==d.version
assert not any(k.split('.')[0] in ('torch','transformers','tensorflow','sionna') for k in sys.modules)
print(json.dumps(dict(executable=sys.executable,executable_binary=pin(sys.executable),python=sys.version,
 torch=d.version,torch_version_file=pin(vp),torch_init=pin(origin),torch_build=values,
 overlay=str(overlay),packages=packages,model_imports=[])),flush=True)
'''

def validate_runtime_metadata(v,reference,policy):
    assert v['executable']==policy['base_python']=='/usr/bin/python3'
    assert v['python']==reference['python'] and v['torch']==reference['torch']
    assert v['torch_build']['__version__']==reference['torch'] and v['model_imports']==[]
    assert v['overlay']==policy['target'] and policy['preserve_base_torch'] is True
    expected=dict(x.split('==') for x in policy['packages']+['flash-attn==2.8.3.post2'])
    assert {x['name']:x['version'] for x in v['packages']}==expected
    for k in ('executable_binary','torch_version_file','torch_init'):verify(v[k])
    for x in v['packages']:
        for k in ('METADATA','RECORD'):
            assert Path(x[k]['path']).is_relative_to(Path(policy['target']));verify(x[k])

def runtime_metadata(phase,root,lineage):
    policy=load(lineage['environment_policy'])
    assert load(lineage['environment_completion'])['status']=='ENVIRONMENT_ENGINEERING_READY'
    out=root/'runtime_metadata_control';out.mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(OLD_L/'environment_v1')+':'+str(OLD_E))
    wait_child(out,['/usr/bin/python3','-B','-c',METADATA_PROBE,lineage['environment_policy']['path']],env,120)
    v=json.loads((out/'stdout').read_text())
    validate_runtime_metadata(v,load(lineage['arpc_reference_runtime']),policy)
    put(root/'runtime_metadata.json',dict(metadata=v,actual_wait=stable_pin(out/'actual_wait.json'),
        launch=stable_pin(out/'launch.json'),stdout=stable_pin(out/'stdout'),stderr=stable_pin(out/'stderr'),
        environment_policy=lineage['environment_policy'],reference_runtime=lineage['arpc_reference_runtime'],
        historical_binary_equality_claimed=False,scope='This execution binary and overlay metadata pinned; historical Python/Torch and actual numerical runtime must match.'))
    return stable_pin(root/'runtime_metadata.json')

def public_job(j,d,codec):
    assert set(j)=={'key','codec','endpoint','bit_length','input','generation_seed'}
    assert j['codec']==codec and j['generation_seed']==0
    item={k:j[k] for k in ('key','codec','endpoint','bit_length','generation_seed')}
    item['input']=d
    if codec:item.update(path=d['path'],paid_profile=j['endpoint'])
    return item

def cache_candidates(lineage):
    """Index saved input identities. Pixel validity is checked only when reused."""
    b0=read(old.O/'b0_completion.json');assert b0['scientific_calls']==0
    for d in b0['runtime_files']:
        if Path(d['path']).name in ('cache_catalogue.json','source_catalogue.json','sources.json'):verify(d)
    streams=read(old.L/'b0/source_catalogue.json')
    sources={(s['population'],s['source_id'],s['codec'],s['endpoint']):s for s in streams}
    index={}
    for c in read(old.L/'b0/cache_catalogue.json'):
        if c['image'] is None:continue
        s=sources[c['population'],c['source_id'],c['codec'],c['endpoint']]
        index.setdefault(source_key(s),[]).append(dict(kind='B0_EXACT_INPUT',source=s,cache=c))
    # These jobs were independently received after B0; exactly two qualified inputs
    # and 1,791 cal100 gap inputs are added without repeating successful calls.
    for run,jobpath in [
        (old.L/'qualification/native',old.L/'qualification/native/public.json'),
        (old.L/'cal100_clean_v1',old.L/'cal100_clean_v1/public.json')]:
        cfg=read(jobpath);jobs=load(cfg['jobs']);out=read(run/'rx/completion.json')
        assert out['status']=='COMPLETE' and out['ledger']['unresolved']==0
        cfgpin,completionpin=stable_pin(jobpath),stable_pin(run/'rx/completion.json')
        proof=lineage['proofs']['qualification' if run==old.L/'qualification/native' else 'cal100']
        images={r['source_input_key']:r for r in out['outputs']}
        for j in jobs:
            assert j['key'] in images
            index.setdefault(exact_key(j),[]).append(dict(kind='SAVED_INDEPENDENT_RX',
                job=j,output=images[j['key']],configuration=cfgpin,
                completion=completionpin,native_lineage=proof))
    return index

def verified_candidate(c,routes,model_routes,static,lineage,members):
    if c['kind']=='B0_EXACT_INPUT':
        require_member(c,lineage,members)
        s,meta=c['source'],c['cache'];im=meta['image']
        wire=verify(s['wire'])
        if s['codec']==0:
            import numpy as np
            with np.load(wire,allow_pickle=False) as z:bits=z[s['array_key']].copy()
            assert bits.dtype==np.uint8 and bits.ndim==1 and len(bits)==s['bit_length']
            assert np.isin(bits,(0,1)).all()
            assert hashlib.sha256(bits.tobytes()).hexdigest()==s['bits_sha256']
            assert hashlib.sha256(protocol.pack_bits(bits)).hexdigest()==s['object_sha256']
            if meta['origin']=='ORIGINAL_ENTROPY_RX':
                ident=im['original_visual_identity']
                for k in NEST_ID_FIELDS:assert ident[k]==routes['native_qualification'][k],('NATIVE_IDENTITY_CHANGED',k)
        else:
            assert hashlib.sha256(wire.read_bytes()).hexdigest()==s['object_sha256']
            ev=im['evidence'];received=ev.get('input',ev.get('wire'))
            if received is not None:assert verify(received).read_bytes()==wire.read_bytes()
            else:assert meta['origin']=='N2048_FULL_RX'
        d=stable_pin(verify(im['image_archive']));key=im.get('image_key','image')
        a,h=image(d,key)
        declared=im.get('image_sha256')
        if declared:
            if s['codec']==0 and meta['origin']=='ORIGINAL_ENTROPY_RX':
                expected=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
            else:expected=h
            assert declared==expected
        return dict(image=d,array_key=key,image_float_sha256=h,receiver_status='EXACT_CLOSED_NATIVE_CACHE',cache_evidence=c)
    assert c['kind']=='SAVED_INDEPENDENT_RX'
    cfg=load(c['configuration']);assert cfg['model_routes']==model_routes and cfg['static']==static
    check_nest_identity(load(c['native_lineage']['runtime']),routes)
    v=c['output'];assert not v['source_truth_used'] and not v['TX_CDF_used']
    assert v['status'] in ('SOURCE_DECODED','SOURCE_PARSE_FAIL')
    image(v['image'],'image',v['float_sha256'])
    return dict(image=v['image'],array_key='image',image_float_sha256=v['float_sha256'],receiver_status=v['status'],cache_evidence=c)

def prepare(phase,transport_sha):
    assert phase in transport_plan.CAPS
    tpath=TO/phase/'registration.json';assert sha(tpath)==transport_sha;t=read(tpath)
    tcpath=TO/phase/'transport_completion.json';tc=read(tcpath)
    assert tc['status']=='TRANSPORT_COMPLETE_NATIVE_AND_SCORE_SEPARATE'
    assert tc['registration']==stable_pin(tpath) and tc['logical']==transport_plan.CAPS[phase]['logical_objects']
    for role in ('tx','rx','audit'):check_wait(TL/'controls'/(phase+'_'+role)/'actual_wait.json',True)
    jobs=load(tc['source_jobs']);frames=load(tc['logical_frames'])
    assert len({j['key'] for j in jobs})==len(jobs)
    assert {j['key'] for j in jobs}=={f['source_input_key'] for f in frames if f['accepted']}
    root=L/phase;out=O/phase;root.mkdir(parents=True);out.mkdir(parents=True)
    b1=read(old.REG);routes=load(b1['model_routes']);lineage,members=native_lineage(b1)
    put(root/'native_identity_compatibility.json',lineage)
    candidates=cache_candidates(lineage);reuse={};missing={0:[],1:[]};proofs=[]
    # B3 may reuse B2 actual received objects, including retained wrong inputs.
    if phase=='B3':
        prior=read(O/'B2/native_completion.json');assert prior['status']=='COMPLETE'
        preg=load(prior['registration']);full=read(R/'results/arpc_n2048_validation_20261019_v1/registration.json')
        same_native_registration(preg,dict(model_routes=b1['model_routes'],static=b1['static'],weights=full['weights']))
        for d in preg['code']:verify(d)
        load(preg['native_identity_compatibility'])
        for d in prior['closure_observations']:check_observation(d)
        for x in load(prior['actual_native_objects']):
            image(x['image'],x['array_key'],x['image_float_sha256'])
            k=exact_key(x['actual_job'])
            candidates.setdefault(k,[]).append(dict(kind='B2_CLOSED_ACTUAL_NATIVE',output=x,
                completion=stable_pin(O/'B2/native_completion.json'),registration=prior['registration']))
    for j in jobs:
        k=exact_key(j);options=candidates.get(k,[]);got=[]
        for c in options:
            if c['kind']=='B2_CLOSED_ACTUAL_NATIVE':
                v=c['output'];v=dict(v,cache_evidence=c)
            else:v=verified_candidate(c,routes,b1['model_routes'],b1['static'],lineage,members)
            got.append(v)
        if got:
            assert len({v['image_float_sha256'] for v in got})==1,('BLOCKED_NATIVE_CACHE_CONFLICT',j['key'])
            v=got[0];reuse[j['key']]=dict(v,actual_job=j)
        else:missing[j['codec']].append(j)
    # Independent views contain actual RX bytes and public profile only.
    for codec,name in ((0,'nest'),(1,'arpc')):
        run=root/name;run.mkdir();inputs=run/'wire_inputs';inputs.mkdir();selected=[]
        for j in missing[codec]:
            data=verify(j['input']).read_bytes();p=inputs/(j['key']+('.arc' if codec else '.bin'))
            with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            item=public_job(j,stable_pin(p),codec)
            selected.append(item)
        put(inputs/'jobs.json',selected)
    put(root/'cached_native_objects.json',list(reuse.values()))
    originals=[old.E/'nest_prefix_rx.py',old.E/'shared.py',old.E/'adaptive_plan.py',
        OLD_E/'a2b_rx_v1/native.py',OLD_E/'a1_native.py',OLD_E/'wire.py',OLD_E/'cuda_recovery_v1/runtime.py']
    code=[stable_pin(E/n) for n in FILES]+[stable_pin(p) for p in originals]+t['code']
    assert sha(OLD_E/'a1_native.py')=='29cf445d424c7f26bf8578ded1ba30354c81846f44a18abeb4e3f1dc0074a03c'
    fullreg=read(R/'results/arpc_n2048_validation_20261019_v1/registration.json')
    put(out/'registration.json',dict(schema='N2048_B2B3_NATIVE_ANALYSIS_V1',phase=phase,
        transport_registration=stable_pin(tpath),transport_completion=stable_pin(tcpath),
        code=code,model_routes=b1['model_routes'],static=b1['static'],
        native_identity_compatibility=stable_pin(root/'native_identity_compatibility.json'),
        weights=fullreg['weights'],prior_model_registration=stable_pin(R/'results/arpc_n2048_validation_20261019_v1/registration.json'),
        arpc_template=stable_pin(BASE/'arpc_n2048_validation_20261019_v1/formal/arpc_resource_v1/public.json'),
        b0_source_catalogue=stable_pin(old.L/'b0/source_catalogue.json'),
        b0_cache_catalogue=stable_pin(old.L/'b0/cache_catalogue.json'),
        cached_native_objects=stable_pin(root/'cached_native_objects.json'),
        jobs={name:stable_pin(root/name/'wire_inputs/jobs.json') for name in ('nest','arpc')},
        new_native_calls={name:len(missing[c]) for c,name in ((0,'nest'),(1,'arpc'))},
        reused_native_objects=len(reuse),received_unique_objects=len(jobs),
        logical_objects=len(frames),new_TX=0,new_channel=0,new_training=0,
        engineering_seconds=NATIVE_SECONDS,automatic_retry=False,
        receiver_view='Only actual received bits, paid endpoint, shared frozen models. No source IDs, TX truth, or CDF.'))
    print(json.dumps(dict(registration=stable_pin(out/'registration.json'),new_native={k:len(v) for k,v in missing.items()},reuse=len(reuse))),flush=True)

def configure(phase,name,expected,gpu):
    r=registration(phase,expected);assert name in ('nest','arpc')
    root=L/phase/name;assert not(root/'public.json').exists()
    lineage=load(r['native_identity_compatibility'])
    metapin=runtime_metadata(phase,root,lineage) if name=='arpc' else None
    # Metadata-only device admission on the actual execution host.
    vals=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    g=dict(index=int(vals[0]),uuid=vals[1].strip(),name=vals[2].strip(),pci_bus_id=vals[3].strip().lower(),total_mib=int(vals[4]),free_mib=int(vals[5]),utilization=int(vals[6]))
    n=r['new_native_calls'][name]
    if name=='nest':
        assert g['free_mib']>=15360
        cfg=dict(run=str(root),model_routes=r['model_routes'],static=r['static'],jobs=r['jobs'][name],
            gpu_index=gpu,gpu_pci=g['pci_bus_id'],gpu_uuid=g['uuid'],new_jobs_cap=n,max_seconds=NATIVE_SECONDS)
    else:
        cfg=load(r['arpc_template']);assert cfg['resource']['threads']==4 and cfg['resource']['interop_threads']==2
        assert n<=6300 and g['free_mib']*(1<<20)>=cfg['resource']['start_free_bytes']
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu))
        lib=ctypes.CDLL('libcuda.so.1');assert lib.cuInit(0)==0;dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
        pci=ctypes.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(pci,64,dev)==0
        uuid=(ctypes.c_ubyte*16)();assert lib.cuDeviceGetUuid(ctypes.byref(uuid),dev)==0
        assert pci.value.decode().lower().lstrip('0')==g['pci_bus_id'].lstrip('0')
        cfg.update(run=str(root),output=str(O/phase),schema='N2048_B2B3_ACTUAL_ARPC_RX_V1',new_job_count=n)
        cfg['resource']['gpu']=g
        cfg['cuda_admission']=dict(visible_index=gpu,pci_bus_id=pci.value.decode().lower(),cuda_uuid_hex=bytes(uuid).hex(),nvml_uuid=g['uuid'])
    put(root/'public.json',cfg)
    put(root/'configuration.json',dict(host=os.uname().nodename,phase=phase,name=name,registration=stable_pin(O/phase/'registration.json'),public=stable_pin(root/'public.json'),actual_gpu=g,model_calls=0,runtime_metadata=metapin))

def run(phase,name,expected):
    r=registration(phase,expected);root=L/phase/name
    load(r['jobs'][name]);control=L/'controls'/(phase+'_'+name);control.mkdir(parents=True)
    if r['new_native_calls'][name]==0:
        put(control/'zero_call_completion.json',dict(status='NO_MISSING_NATIVE_JOBS',new_model_calls=0,registration=stable_pin(O/phase/'registration.json')))
        return
    cfg=read(root/'public.json');conf=read(root/'configuration.json')
    assert conf['host']==os.uname().nodename and conf['registration']['sha256']==expected
    verify(conf['public'])
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(conf['actual_gpu']['index']),PYTHONDONTWRITEBYTECODE='1')
    if name=='nest':argv=[str(PY),'-B',str(old.E/'nest_prefix_rx.py'),str(root/'public.json')]
    else:
        for d in r['weights']:verify(d)
        v=load(conf['runtime_metadata']);lineage=load(r['native_identity_compatibility'])
        validate_runtime_metadata(v['metadata'],load(lineage['arpc_reference_runtime']),load(lineage['environment_policy']))
        argv=['/usr/bin/python3','-B',str(OLD_E/'a2b_rx_v1/native.py'),'rx','--public',str(root/'public.json')]
        env['PYTHONPATH']=str(OLD_L/'environment_v1')+':'+str(OLD_E)
    import fcntl
    with (old.L/'adaptive_execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_child(control,argv,env,NATIVE_SECONDS+300)

def observe(phase,name,expected):
    r=registration(phase,expected);control=L/'controls'/(phase+'_'+name)
    if r['new_native_calls'][name]==0:
        assert read(control/'zero_call_completion.json')['new_model_calls']==0;return
    w,launch=check_wait(control/'actual_wait.json',True)
    assert not(L/phase/name/'rx/failure.json').exists()
    put(control/'closure_observation.json',dict(status='CLOSED_ON_EXECUTION_HOST',host=os.uname().nodename,
        child=w['process'],parent=launch['parent_process'],child_gone=True,parent_gone=True,
        observed_unix=time.time(),not_before_unix=launch['started_unix']+w['seconds'],
        actual_wait=stable_pin(control/'actual_wait.json'),launch=stable_pin(control/'launch.json')))

def audit_new_native(objects,frames,source_catalogue):
    """Truth is opened only by this post-exit audit, never by a receiver."""
    import numpy as np
    from adaptive_plan import NEST_ENDPOINTS,SIZES
    catalogue={(s['source_id'],s['codec'],s['endpoint']):s for s in load(source_catalogue)}
    rows=[];seen=set()
    for f in frames:
        if not f['exact_post_exit'] or f['source_input_key'] in seen:continue
        seen.add(f['source_input_key']);obj=objects[f['source_input_key']]
        if 'native_output' not in obj:continue  # Previously closed, bound cache audit.
        v=obj['native_output'];j=obj['actual_job'];s=catalogue[f['source_id'],j['codec'],j['endpoint']]
        assert obj['receiver_status']!='SOURCE_PARSE_FAIL',('BLOCKED_VALID_SOURCE_PARSE_FAILURE',j['key'])
        if j['codec']==0:
            with np.load(verify(s['source']),allow_pickle=False) as z:truth=z['tokens'].copy()
            m,k=NEST_ENDPOINTS[j['endpoint']];n=sum(x*x for x in SIZES[:m])+k
            assert np.array_equal(truth[:n],np.asarray(v['received_tokens'])),('BLOCKED_SOURCE_CDF_MODEL_MISMATCH',j['key'])
            rows.append(dict(key=j['key'],codec=0,source_id=f['source_id'],endpoint=j['endpoint'],tokens_exact=True,source=s['source']))
        else:
            wire=verify(s['wire']);stem=wire.stem.rsplit('_e',1)[0]
            wp=wire.parent/(stem+'_source.json');tp=wire.parent/(stem+'_truth.npz');w=read(wp)
            assert w['source_id']==f['source_id']
            assert w['caption_bytes_hex']==v['caption_bytes_hex'] and w['cdf_hashes'][:j['endpoint']]==v['cdf_hashes'],('BLOCKED_ARPC_CDF_MISMATCH',j['key'])
            assert next(q['payload_sha256'] for q in w['lengths'] if q['endpoint']==j['endpoint'])==s['object_sha256']
            dp=Path(v['path']).with_name(j['key']+'_decoded.npz')
            with np.load(tp,allow_pickle=False) as truth,np.load(dp,allow_pickle=False) as got:
                assert set(got.files)=={f's{i}' for i in range(1,j['endpoint']+1)}
                assert all(np.array_equal(truth[k],got[k]) for k in got.files),('BLOCKED_ARPC_TOKEN_MISMATCH',j['key'])
            rows.append(dict(key=j['key'],codec=1,source_id=f['source_id'],endpoint=j['endpoint'],tokens_exact=True,cdfs_exact=True,witness=stable_pin(wp),source_truth=stable_pin(tp),received_tokens=stable_pin(dp)))
    return rows

def seal(phase,expected):
    r=registration(phase,expected);root=L/phase;tc=load(r['transport_completion'])
    lineage=load(r['native_identity_compatibility']);routes=load(r['model_routes'])
    jobs={j['key']:j for j in load(tc['source_jobs'])};objects={j['actual_job']['key']:j for j in load(r['cached_native_objects'])}
    observations=[];waits=[];manifest=[r['transport_registration'],r['transport_completion'],r['cached_native_objects']]
    for name in ('nest','arpc'):
        control=L/'controls'/(phase+'_'+name);newjobs=load(r['jobs'][name])
        if not newjobs:
            assert read(control/'zero_call_completion.json')['new_model_calls']==0
            manifest.append(stable_pin(control/'zero_call_completion.json'));continue
        op=stable_pin(control/'closure_observation.json');obs=check_observation(op);observations.append(op);waits.append(obs['actual_wait'])
        cp=root/name/'rx/completion.json';comp=read(cp);assert comp['status']=='COMPLETE'
        assert not(root/name/'rx/failure.json').exists()
        if name=='nest':
            assert comp['ledger']['unresolved']==0
            ip=root/name/'rx/actual_native_identity.json';check_nest_identity(read(ip),routes)
        else:
            assert not comp['blocked_reads']
            ip=root/name/'rx/runtime.json'
            assert canonical_arpc_runtime(read(ip))==canonical_arpc_runtime(load(lineage['arpc_reference_runtime']))
            conf=read(root/name/'configuration.json');meta=load(conf['runtime_metadata'])
            validate_runtime_metadata(meta['metadata'],read(ip),load(lineage['environment_policy']))
            manifest.extend([conf['runtime_metadata'],stable_pin(root/name/'configuration.json')])
        manifest.append(stable_pin(ip))
        output={v.get('source_input_key',v.get('key')):v for v in comp['outputs']}
        assert set(output)=={j['key'] for j in newjobs}
        for j in newjobs:
            v=output[j['key']];status=v.get('status','SOURCE_DECODED')
            if status=='SOURCE_PARSE_FAIL':
                d=None;arraykey=None;_,h=image(None,None)
            else:
                d=v.get('image') or stable_pin(v['path']);arraykey='image' if name=='nest' else 'float_output'
                _,h=image(d,arraykey,v['float_sha256'])
            if name=='nest':assert v['source_truth_used'] is False and v['TX_CDF_used'] is False
            objects[j['key']]=dict(actual_job=jobs[j['key']],image=d,array_key=arraykey,image_float_sha256=h,receiver_status=status,native_output=v,native_completion=stable_pin(cp))
        manifest.extend([stable_pin(cp),obs['actual_wait'],obs['launch'],op])
    assert set(objects)==set(jobs)
    for obj in objects.values():image(obj['image'],obj['array_key'],obj['image_float_sha256'])
    frames=load(tc['logical_frames'])
    audit=audit_new_native(objects,frames,r['b0_source_catalogue'])
    put(root/'post_exit_native_audit.json',dict(status='PASS',new_exact_inputs_audited=len(audit),rows=audit,truth_read_only_after_all_native_processes_gone=True,accepted_wrong_outputs_never_replaced=True))
    manifest.append(stable_pin(root/'post_exit_native_audit.json'))
    pixels=[];gray_hash=image(None,None)[1]
    for f in frames:
        if f['accepted']:v=objects[f['source_input_key']]
        else:v=dict(image=None,array_key=None,image_float_sha256=gray_hash,receiver_status='CHANNEL_OR_SOURCE_NULL_GRAY')
        pixels.append(dict(f,method=METHODS[f['codec']],image=v['image'],array_key=v['array_key'],
            image_float_sha256=v['image_float_sha256'],receiver_status=v['receiver_status'],accepted_wrong_post_exit=f['accepted_wrong']))
    assert len(pixels)==transport_plan.CAPS[phase]['logical_objects']
    put(root/'actual_native_objects.json',list(objects.values()));put(root/'actual_logical_pixels.json',pixels)
    outputs=[stable_pin(root/'actual_native_objects.json'),stable_pin(root/'actual_logical_pixels.json')]
    put(root/'native_manifest.json',dict(inputs=manifest,outputs=outputs))
    put(O/phase/'native_completion.json',dict(status='COMPLETE',phase=phase,logical_objects=len(pixels),
        logical_pixels=outputs[1],actual_native_objects=outputs[0],
        manifest=stable_pin(root/'native_manifest.json'),inputs=manifest,outputs=outputs,
        native_actual_waits=waits,closure_observations=observations,
        process_closure_status='ALL_NATIVE_PROCESSES_GONE',process_closure_evidence=observations,
        registration=stable_pin(O/phase/'registration.json'),transport_completion=r['transport_completion'],
        rejected_and_wrong_retained=True,source_truth_in_native=False,new_TX=0,new_channel=0,new_training=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['prepare','configure','run','observe','seal'])
    ap.add_argument('--phase',choices=['B2','B3'],required=True);ap.add_argument('--expected-registration',required=True)
    ap.add_argument('--name',choices=['nest','arpc']);ap.add_argument('--gpu',type=int)
    a=ap.parse_args()
    try:
        if a.role=='prepare':prepare(a.phase,a.expected_registration)
        elif a.role in ('run','observe'):globals()[a.role](a.phase,a.name,a.expected_registration)
        elif a.role=='configure':configure(a.phase,a.name,a.expected_registration,a.gpu)
        else:seal(a.phase,a.expected_registration)
    except BaseException:fail(L/a.phase);raise
