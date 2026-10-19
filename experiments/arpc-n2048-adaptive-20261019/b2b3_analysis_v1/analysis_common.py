"""Independent analysis controls; original native and metric functions stay frozen."""
from pathlib import Path
import sys, os, json, hashlib, math, time, traceback, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared as old
from shared import R, RT, BASE, OLD_E, OLD_O, OLD_L, CPY, PY
from shared import pin, verify, read, put, sha, gone, identity, wait_child, Ledger, cache
sys.path.insert(0, str(HERE.parent / 'b2b3_revision_v1'))
import plan as transport_plan
import pair_protocol as protocol
E, O, L = old.E/'b2b3_analysis_v1', old.O/'b2b3_analysis_v1', old.L/'b2b3_analysis_v1'
TO, TL = old.O/'b2b3_revision_v1', old.L/'b2b3_revision_v1'
METHODS = {0:'NEST_ADAPTIVE_N2048', 1:'ARPC_ADAPTIVE_N2048'}
FILES = ('analysis_common.py','native.py','score.py','dino_worker.py','test_analysis.py','README.md')

def descriptor(path, expected, size):
    return dict(path=str(path), sha256=expected, bytes=int(size))

def load(d):
    return read(verify(d))

def stable_pin(path):
    p=Path(path);a=p.stat();d=pin(p);b=p.stat()
    assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
    verify(d);return d

def image(d, key, expected=None):
    import numpy as np
    if d is None:a=np.full((3,256,256),.5,np.float32)
    else:
        with np.load(verify(d),allow_pickle=False) as z:a=z[key].copy()
    assert a.dtype==np.float32 and a.shape==(3,256,256)
    assert np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
    h=hashlib.sha256(a.tobytes()).hexdigest()
    if expected is not None:assert h==expected
    return a,h

def exact_key(job):
    data=verify(job['input']).read_bytes()
    assert job['codec'] in (0,1) and job['generation_seed']==0
    assert protocol.input_key(job['codec'],job['endpoint'],job['bit_length'],data)==job['key']
    assert len(data)==(job['bit_length']+7)//8
    return (job['codec'],job['endpoint'],job['bit_length'],hashlib.sha256(data).hexdigest())

def source_key(row):
    return (row['codec'],row['endpoint'],row['bit_length'],row['object_sha256'])

def check_wait(path, require_parent=False):
    w=read(path);assert type(w['returncode']) is int and w['returncode']==0
    for flag in ('timed_out','terminated','killed'):assert w.get(flag,False) is False
    assert w.get('metadata_error') is None and gone(w['process'])
    launch=read(Path(path).with_name('launch.json'))
    assert launch['process']==w['process']
    if require_parent:assert gone(launch['parent_process'])
    return w,launch

def check_observation(d):
    x=load(d);assert x['status']=='CLOSED_ON_EXECUTION_HOST'
    w=load(x['actual_wait']);launch=load(x['launch'])
    assert type(w['returncode']) is int and w['returncode']==0
    assert w['process']==launch['process']==x['child']
    assert launch['parent_process']==x['parent']
    assert x['host']==x['child']['host']==x['parent']['host']
    assert x['child_gone'] is True and x['parent_gone'] is True
    assert x['observed_unix']>=x['not_before_unix']
    for flag in ('timed_out','terminated','killed'):assert w.get(flag,False) is False
    assert w.get('metadata_error') is None
    return x

def registration(phase, expected):
    assert phase in transport_plan.CAPS
    p=O/phase/'registration.json';assert sha(p)==expected;r=read(p)
    assert r['phase']==phase
    for d in r['code']:verify(d)
    load(r['transport_completion']);load(r['transport_registration'])
    return r

def fail(root):
    root=Path(root)
    if root.exists() and not(root/'failure.json').exists():
        put(root/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False))

def dino_identity(metadata):
    x=dict(metadata);x.pop('metric_selection_usage',None);return x

def source_means(rows):
    """Exactly three registered noises per source, then equal source weights."""
    buckets={}
    for x in rows:
        k=(x['codec'],x['snr_db'],x['policy_id'],x['source_id'])
        b=buckets.setdefault(k,{})
        assert x['noise_seed'] not in b and math.isfinite(x['dino'])
        b[x['noise_seed']]=x['dino']
    bycell={};source_rows=[]
    for (codec,snr,policy,sid),values in sorted(buckets.items()):
        assert set(values)==set(transport_plan.SEEDS['B2'])
        v=sum(values[n] for n in transport_plan.SEEDS['B2'])/3
        source_rows.append(dict(codec=codec,snr_db=snr,policy_id=policy,source_id=sid,DINO_source_mean=v))
        bycell.setdefault((codec,snr,policy),[]).append(v)
    assert set(bycell)==set(transport_plan.B2_CELLS)
    means=[]
    for (codec,snr,policy),values in sorted(bycell.items()):
        assert len(values)==1000
        means.append(dict(codec=codec,snr_db=snr,policy_id=policy,calibration_mean_DINO=sum(values)/1000,source_mean_count=1000,noise_per_source=3))
    return source_rows,means
