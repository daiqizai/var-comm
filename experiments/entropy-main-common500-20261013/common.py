"""Finite common500 composition utilities; no scientific work on import."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
RT = BASE/'code/liulu/var_comm_runtime_20261010'
TMP = BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011'
ART = TMP/'entropy_main_common500_20261013_v1'
OUT = ROOT/'results/entropy_main_common500_20261013'
ES = ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'
TS = ROOT/'experiments/wcl-evidence-closure-20261009/scripts'
sys.path[:0] = [str(ES), str(TS)]
SNRS = (4, 10, 19)
SEEDS = (6201, 6202, 6203)
METRICS = ('psnr_db', 'lpips_alex', 'dinov2_vitl14_cosine', 'convnext_top1_source_prediction')

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def descriptor(path):
    p=Path(path);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def checked(pin):
    p=Path(pin['path']);require(p.is_file() and sha(p)==pin['sha256'],'Bound input changed: '+str(p))
    if 'bytes' in pin:require(p.stat().st_size==pin['bytes'],'Bound size changed: '+str(p))
    return p
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def status(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,sort_keys=True)+'\n');os.replace(tmp,p)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def counter(i,s,n):
    require(type(i) is int and 0<=i<500 and s in SNRS and n in SEEDS,'Unregistered frame')
    return ([1,4,7,10,13,19].index(s)*500+i)*3+SEEDS.index(n)

def registration(path):
    r=read(path)
    require(r['schema']=='ENTROPY_MAIN_COMMON500_V1' and r['logical_frames']==4500 and r['SNRs']==list(SNRS)
            and r['noise_seeds']==list(SEEDS) and r['training_updates']==0 and r['E4_allowed'] is False,'Frozen scope changed')
    for p,h in r['code_bindings'].items():require(sha(p)==h,'Bound code changed: '+p)
    require(len(r['sources'])==500 and [x['source_index'] for x in r['sources']]==list(range(500))
            and len({x['source_id'] for x in r['sources']})==500,'Fixed500 required')
    return r

class Ledger:
    """Durable nested-call reservations, finite caps, no implicit replay."""
    def __init__(self,path,caps,guard):
        require(not Path(path).exists(),'Fresh scientific ledger required')
        self.db=sqlite3.connect(path);self.db.execute('CREATE TABLE calls (id INTEGER PRIMARY KEY, kind TEXT, metadata TEXT, state TEXT)')
        self.db.commit();self.caps=dict(caps);self.guard=guard;self.reserved=collections.Counter();self.completed=collections.Counter()
    def call(self,kind,fn,**metadata):
        self.guard();require(kind in self.caps and self.reserved[kind]<self.caps[kind],'Scientific cap: '+kind)
        self.reserved[kind]+=1
        cur=self.db.execute('INSERT INTO calls(kind,metadata,state) VALUES(?,?,?)',(kind,canonical(metadata),'RESERVED'));self.db.commit();i=cur.lastrowid
        try:result=fn()
        except BaseException:
            self.db.execute('UPDATE calls SET state=? WHERE id=?',('FAILED',i));self.db.commit();raise
        self.db.execute('UPDATE calls SET state=? WHERE id=?',('COMPLETE',i));self.db.commit();self.completed[kind]+=1;return result
    def summary(self):
        return dict(caps=self.caps,reserved={k:self.reserved[k] for k in self.caps},completed={k:self.completed[k] for k in self.caps},
                    unresolved=sum(self.reserved.values())-sum(self.completed.values()))
    def close(self):
        require(self.summary()['unresolved']==0,'Unresolved scientific call');self.db.close()

def backend(r,out,ledger,boundary):
    """Reuse the already-qualified H800 visual provider and resource adapter."""
    import h800_ep_new100_visual_v3 as visual
    old=read(checked(r['visual_request']));runner=visual.engine();device,_=resource_device(r,out)
    device.check_cuda_environment()
    spec=dict(old['spec']);spec['deadline_unix']=time.time()+900;spec['max_seconds']=900
    resolver,environment,_,_,extra=visual.g.validate_spec(spec)
    module=load(ES/'leo_whole_math_v1.py','_emain_frozen_visual')
    Backend,proof=runner.adapt_backend(module,device);write(out/'resource_adapter_AST.json',proof)
    b=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
    return b,module,resolver,environment,device,visual.g

def resource_device(r,out):
    import h800_alternate_gpu_resource_v3 as resource
    import inspect
    import types
    # Explicit shared-host admission amendment, retaining the16GiB private
    # allocator cap,4GiB absolute free margin and verified own-PID credit.
    # Only the prelaunch/free+own reserve changes20->18GiB. CUDA/library
    # allocation outside PyTorch is otherwise falsely charged as another job.
    ns=dict(vars(resource));policy=dict(resource.POLICY_BASE)
    policy.update(schema='EMAIN_SHARED_RESOURCE_POLICY_V1',prelaunch_free_bytes=18*(1<<30),runtime_free_plus_own_reserved_bytes=18*(1<<30))
    ns['POLICY_BASE']=policy
    fn=resource.policy
    ns['policy']=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
    source=inspect.getsource(resource.assess_snapshot)
    require(source.count('20*GiB')==2,'Original resource-only adapter changed')
    adapted=source.replace('20*GiB','18*GiB')
    exec(compile(adapted,resource.__file__,'exec'),ns)
    method=resource.DeviceAdapter.resource_snapshot
    replacement=types.FunctionType(method.__code__,ns,method.__name__,method.__defaults__,method.__closure__)
    def bind_evidence(self,directory,request_sha):
        # New logs follow the user's tmp storage rule. Keep original canonical
        # path and no-symlink checks, explicitly rooted in this task's tmp tree.
        p=resource.g.inside(directory,base=ART)
        require(p.is_dir() and resource.g.SHA_RE.fullmatch(request_sha),'Explicit EMAIN evidence binding')
        self.evidence_directory=p;self.request_sha=request_sha
    Device=type('ExplicitSharedEMAINDevice',(resource.DeviceAdapter,),dict(resource_snapshot=replacement,bind_resource_evidence=bind_evidence))
    old=read(checked(r['visual_request']));d=Device(old['device_receipt'],0)
    d.bind_resource_evidence(out,digest(r));return d,resource.g

def guard_for(r,out,parent):
    start=time.monotonic()
    def guard():
        require(os.getppid()==parent,'Owner lost')
        require(time.time()<r['deadline_unix'] and time.monotonic()-start<r['stage_max_seconds'],'Finite execution deadline')
        require(not (OUT/'STOP').exists() and not (ART/'STOP').exists() and not (out/'STOP').exists(),'Requested STOP')
    return guard
