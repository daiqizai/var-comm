"""Exactly one new clean raster endpoint on the original construction200.

Register is metadata/file verification only. Worker is one explicit GPU process;
an external caller must wait it and bind its closed log/exit receipt. This script
does not launch children, select policies, invoke PHY or read development.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

STATES = ((8,142,397),)
SCHEMA='MAIN_RAW64_CLEAN_MISSING_REQUEST_V1'
REG='MAIN_RAW64_CLEAN_MISSING_REGISTERED_V1'
DONE='MAIN_RAW64_CLEAN_MISSING200_COMPLETE_V1'
STOP=False
TEST_COUNT=9

def require(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def verify(pins):
    for p,h in pins.items():require(Path(p).is_absolute() and sha(p)==h,'Changed/unbound file '+p)
def merge(*maps):
    out={}
    for m in maps:
        for p,h in m.items():require(p not in out or out[p]==h,'Conflicting SHA '+p);out[p]=h
    return out
def bind(pins,p,expected=None):
    p=str(Path(p).absolute());h=sha(p)
    require(expected is None or h==expected,'Original output SHA differs '+p)
    require(p not in pins or pins[p]==h,'Conflicting input '+p);pins[p]=h;return read(p)
def module(p,name):
    p=str(Path(p).absolute())
    if name in sys.modules:
        require(str(Path(sys.modules[name].__file__).absolute())==p,'Unexpected imported '+name)
        return sys.modules[name]
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def proc(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),uid=p.stat().st_uid,argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
def gone(i):
    try:n=proc(i['pid'])
    except FileNotFoundError:return True
    require((n['start_ticks'],n['uid'])!=(i['start_ticks'],i['uid']),'Original process not reaped');return True
def stop(*_):
    global STOP
    STOP=True

def endpoint_gap(old_rows,profiles):
    old={(x['m'],x['K'],x['token_count']) for x in old_rows}
    require(len(old_rows)==len(old)==395 and {t for _,_,t in old}==set(range(1,396)),
            'Exact original395 clean endpoints required')
    wanted={(x['m'],x['K'],x['token_count']) for x in profiles}
    require(wanted-old==set(STATES) and max(t for _,_,t in wanted)==397,
            'Only actual new T397 endpoints may be supplemented')
    return [list(x) for x in STATES]

def qualification(request,sources,inputs):
    p=request['prepared_qualification'];require(inputs.get(p)==sha(p),'Actual CPU qualification must be explicitly input-bound')
    q=read(p)
    require(q['status']=='MAIN_RAW64_CLEAN_MISSING_CPU_TESTS_PASS' and q['tests_run']==TEST_COUNT
        and q['test_modules']==['test_clean_missing_r1'] and q['exit_code']==0 and q['process_waited'] is True
        and q['new_model_calls']==q['new_packet_decodes']==0 and q['python']==request['python'], 'Actual CPU interface qualification differs')
    for n in ('clean_missing_r1.py','test_clean_missing_r1.py'):
        f=str(Path(__file__).with_name(n).absolute());require(q['source_bindings'].get(f)==sources.get(f)==sha(f),'Qualification source coverage differs')
    verify(q['input_bindings']);verify(q['outputs']);log=q['log']
    require(q['outputs'].get(log)==sha(log) and f'Ran {TEST_COUNT} tests' in Path(log).read_text()
        and Path(log).read_text().rstrip().endswith('OK'), 'Closed qualification log missing')
    inputs.update(merge(inputs,q['input_bindings'],q['outputs']))

def inspect(request, *, check_old_process=True):
    require(request['schema']==SCHEMA and request['states']==[list(x) for x in STATES]
        and request['source_count']==200 and request['images']==200 and request['new_packet_decodes']==0
        and request['selection'] is False and request['development_used'] is False and request['holdout_used'] is False,
        'Exact clean200 construction-only scope required')
    sources=dict(request['source_bindings']);inputs=dict(request['input_bindings']);verify(sources);verify(inputs)
    for n in ('clean_missing_r1.py','test_clean_missing_r1.py'):
        p=str(Path(__file__).with_name(n).absolute());require(sources.get(p)==sha(p),'Both new execution/test sources must be bound')
    for k in ('source_driver_module','quality_module','closure_module','environment_module'):
        require(sources.get(request[k])==sha(request[k]),'Required execution source unbound '+k)
    qualification(request,sources,inputs)
    oc=bind(inputs,request['old_source_config'],inputs.get(request['old_source_config']))
    oldreg=bind(inputs,oc['registration'],inputs.get(oc['registration']))
    require(oldreg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and oldreg['input_bindings'].get(request['old_source_config'])==sha(request['old_source_config'])
        and oldreg['source_bindings'].get(request['source_driver_module'])==sha(request['source_driver_module']),
        'Original clean source config/module registration differs')
    require(oc['root']==request['root'] and oc['uep_runtime']==str(Path(request['quality_module']).parent),
            'Original native loader path differs')
    inherited=merge(oldreg['source_bindings'],oldreg['input_bindings'])
    ids=bind(inputs,oc['source200'],inherited.get(oc['source200']))['source_ids']
    require(len(ids)==len(set(ids))==200,'Original construction hash200 only')
    cal=bind(inputs,oc['calibration_registration'],inherited.get(oc['calibration_registration']))
    require(cal['stage']==cal['calibration_or_development']=='m1_calibration'
        and len(cal['source_ids'])==1000 and set(ids)<=set(cal['source_ids']),'Calibration population only')
    num=bind(inputs,request['native_qualification'],inputs.get(request['native_qualification']))
    require(num['status']=='REAL_NATIVE_QUALIFICATION_PASS' and num['synthetic'] is False
        and num['frozen_identity']==cal['identity'] and num['numerical_runtime']['threads']==6
        and num['numerical_runtime']['interop_threads']==2,'Original actual FP32 native qualification required')
    nativepins=dict(request['native_source_bindings']);verify(nativepins);sources=merge(sources,nativepins)
    for p,h in request['model_bindings'].items():require(inputs.get(p)==h==sha(p),'Frozen model file not directly bound')
    require(request['model_bindings'],'Original model asset map required')
    clean=bind(inputs,request['old_clean_completion'],inputs.get(request['old_clean_completion']))
    require(clean['status']=='H_CLEAN_QUALITY200_COMPLETE' and clean['source_count']==200
        and clean['registration_sha256']==sha(oc['registration']) and clean['development_used'] is False
        and clean['holdout_used'] is False,'Original clean200 completion differs')
    launch=bind(inputs,request['old_clean_launch'],inputs.get(request['old_clean_launch']))
    ex=bind(inputs,request['old_clean_exit'],inputs.get(request['old_clean_exit']))
    log=request['old_clean_log'];require(inputs.get(log)==ex['closed_log_sha256']==sha(log),'Original closed clean log required')
    argv=[request['python'],'-B',request['source_driver_module'],'--config',request['old_source_config'],'--stage','clean-quality200']
    require(ex['identity']==launch['identity'] and ex['exit_code']==0 and ex['identity']['argv']==argv
        and launch['argv']==argv and ex['completion']==request['old_clean_completion']
        and ex['completion_sha256']==sha(request['old_clean_completion'])
        and ex['registration_sha256']==launch['registration_sha256']==sha(oc['registration']), 'Original clean worker not normally closed')
    if check_old_process:gone(ex['identity'])
    # Bind source assets from the original completion. No tensor shard import.
    assets=Path(oc['S1'])/'export-assets';adone=bind(inputs,assets/'completion.json',inputs.get(str(assets/'completion.json')))
    require(adone['status']=='S1_EXPORT_ASSETS_COMPLETE','Original S1 raw assets incomplete')
    profile_data=bind(inputs,request['new_profiles'],inputs.get(request['new_profiles']))
    profiles=profile_data if isinstance(profile_data,list) else profile_data['profiles']
    records=[];cleanroot=Path(request['old_clean_completion']).parent
    for i,sid in enumerate(ids):
        cp_path=assets/'source_checkpoints'/f'{i:04d}.json';cp=bind(inputs,cp_path,adone['outputs'].get(str(cp_path)))
        require(str(cp_path) in adone['outputs'] and cp['source_index']==i and cp['source_id']==sid
            and cp['payload_sha256']==digest({k:v for k,v in cp.items() if k!='payload_sha256'}), 'S1 source checkpoint differs')
        j=cp['original_calibration_index'];require(cal['source_ids'][j]==sid and cal['preprocessing_ids'][j]==cp['preprocessing_id'], 'Original calibration index/preprocess differs')
        archive=cp['archive'];require(adone['outputs'].get(archive)==cp['outputs'].get(archive)==sha(archive),'Raw source archive differs');inputs[archive]=sha(archive)
        oldcp_path=cleanroot/'source_checkpoints'/f'{i:04d}.json';oldcp=bind(inputs,oldcp_path,clean['outputs'].get(str(oldcp_path)))
        require(str(oldcp_path) in clean['outputs'] and oldcp['source_id']==sid and oldcp['source_index']==i
            and oldcp['clean_states']==395 and oldcp['registration_sha256']==sha(oc['registration']),'Old clean source identity differs')
        rp=cleanroot/'sources'/f'{i:04d}.json';require(oldcp['outputs'].get(str(rp))==clean['outputs'].get(str(rp))==sha(rp),'Old clean source rows unbound')
        rows=bind(inputs,rp,sha(rp));endpoint_gap(rows,profiles)
        records.append(dict(source_index=i,source_id=sid,original_calibration_index=j,checkpoint=str(cp_path),archive=archive,preprocessing_id=cp['preprocessing_id']))
    require(request['resources']==dict(gpu_device=0,threads=6,interop_threads=2,affinity=[4,5,6,7,8,9],nice=15)
        and 0<request['max_seconds']<=14400 and request['visual_lock']==str(Path(request['root'])/'outputs/CONTENT-REAL-64QAM-20261006/shared_visual.lock'), 'Original singleGPU envelope required')
    require(inputs.get(request['nvidia_smi'])==sha(request['nvidia_smi']),'GPU query binary must be bound')
    return dict(request=request,sources=sources,inputs=inputs,bound=merge(sources,inputs),old=oc,
        source_ids=ids,records=records,native_identity=cal['identity'],flags=num['numerical_runtime'])

def register(path):
    path=str(Path(path).absolute());r=read(path);ctx=inspect(r);e,out=Path(r['execution_dir']),Path(r['out'])
    require(e.is_absolute() and out.is_absolute() and e!=out and not e.exists() and not out.exists(), 'Fresh independent execution/output required')
    e.mkdir(parents=True);cfg=dict(request=path,registration=str(e/'registration.json'),out=str(out))
    try:
        save(e/'config.json',cfg);inputs=merge(ctx['inputs'],{path:sha(path),str(e/'config.json'):sha(e/'config.json')})
        verify(ctx['sources']);verify(inputs)
        reg=dict(status=REG,config_sha256=sha(e/'config.json'),source_bindings=ctx['sources'],input_bindings=inputs,
            source_ids=ctx['source_ids'],records=ctx['records'],states=r['states'],images=200,new_packet_decodes=0,
            frozen_identity=ctx['native_identity'],numerical_runtime=ctx['flags'],model_bindings=r['model_bindings'],
            native_source_bindings=r['native_source_bindings'],external_wait_required=True)
        save(e/'registration.json',reg);save(e/'registration_completion.json',dict(status='REGISTERED_NOT_LAUNCHED',registration_sha256=sha(e/'registration.json'),config=str(e/'config.json')))
        return reg
    except BaseException:save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()));raise

def load_registered(path):
    path=str(Path(path).absolute());cfg=read(path);reg=read(cfg['registration']);verify(reg['source_bindings']);verify(reg['input_bindings'])
    require(reg['status']==REG and reg['config_sha256']==sha(path)==reg['input_bindings'].get(path)
        and reg['input_bindings'].get(cfg['request'])==sha(cfg['request']),'Execution seal differs')
    ctx=inspect(read(cfg['request']));require(ctx['source_ids']==reg['source_ids'] and ctx['records']==reg['records']
        and ctx['flags']==reg['numerical_runtime'] and ctx['native_identity']==reg['frozen_identity'],'Registered scope changed')
    return dict(ctx,cfg=cfg,reg=reg,regsha=sha(cfg['registration']),config_path=path)

@contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def idle(nvidia_smi):
    def q(kind,fields):return subprocess.run([nvidia_smi,'--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15).stdout.strip()
    require(len(q('gpu','index').splitlines())==1 and not q('compute-apps','pid'),'Exactly one idle visual GPU required')

def read_source(rec):
    cp=read(rec['checkpoint']);require(cp['source_id']==rec['source_id'],'Source identity changed')
    with np.load(rec['archive'],allow_pickle=False) as z:tokens=z['tokens'].copy();pixels=z['pixels'].copy()
    require(tokens.shape==(680,) and np.issubdtype(tokens.dtype,np.integer) and ((tokens>=0)&(tokens<4096)).all(), 'Original680 rawtokens required')
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256) and hashlib.sha256(pixels.tobytes()).hexdigest()==rec['preprocessing_id'], 'Original uint8 target/preprocess differs')
    return tokens.astype(np.int64,copy=False),pixels.astype(np.float32)/np.float32(255)

def evaluate(tokens,target,render,score,boundary):
    rows=[]
    for m,K,T in STATES:
        boundary();im=np.asarray(render(tokens[:T].copy(),m,K))
        require(im.dtype==np.float32 and im.shape==(3,256,256) and np.isfinite(im).all() and ((im>=0)&(im<=1)).all(),'Original float32 RGB required')
        metrics=score(im,target);require(set(metrics)=={'mse','psnr_db'} and all(np.isfinite(x) for x in metrics.values()),'Original finite pixel scores required')
        rows.append(dict(m=m,K=K,token_count=T,**metrics,image_sha256=hashlib.sha256(str(im.dtype).encode()+str(im.shape).encode()+im.tobytes()).hexdigest()))
    return rows

def worker(path):
    require(sys.platform.startswith('linux'),'Registered Linux worker only');ctx=load_registered(path);r=ctx['request'];out=Path(r['out'])
    require(not out.exists(),'Fresh worker output; no automatic resume');out.mkdir(parents=True)
    started=time.monotonic();native=None
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
    try:
        require(str(Path(sys.executable).absolute())==r['python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Original UM interpreter/GPU visibility required')
        require(set(os.sched_getaffinity(0))==set(r['resources']['affinity']) and os.getpriority(os.PRIO_PROCESS,0)==15,'Registered6cores/nice15 required')
        sys.path[:0]=list(dict.fromkeys([str(Path(r['environment_module']).parent),str(Path(r['source_driver_module']).parent),ctx['old']['uep_runtime'],ctx['old']['native_runtime'],str(Path(r['root'])/'src')]))
        guard=module(r['environment_module'],'original_preflight_environment_r1')
        def boundary():
            require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-started<r['max_seconds']
                and not any(Path(p).exists() for p in r['stop_files']) and not (out/'STOP').exists(),'STOP/deadline; preserve partial output')
            guard.check(ctx['bound'],r['root'])
            if native is not None:require(not quality.boundary(native),'Original native safety stop')
        with lock(r['visual_lock']):
            boundary();idle(r['nvidia_smi']);identity=proc(os.getpid());save(out/'worker_identity.json',dict(identity=identity,registration_sha256=ctx['regsha'],config_sha256=sha(path)))
            closure=module(r['closure_module'],'clean_missing_native_closure')
            graph=closure.collect_bindings(r['root'],ctx['old']['native_runtime'],r['var_source'],r['dino_source'],ctx['old']['uep_runtime'])
            require(graph==r['native_source_bindings'],'Full current native source graph changed')
            quality=module(r['quality_module'],'quality_driver');source=module(r['source_driver_module'],'clean_missing_frozen_h_source')
            native=quality.build_native(Path(r['root']),ctx['old']['native_runtime'],stop)
            require(native.flags==ctx['flags'] and native.loaded['identity']==ctx['native_identity'],'Actual model/precision identity differs')
            for p,h in native.driver_bindings.items():require(ctx['bound'].get(p)==h==sha(p),'Actual loaded dependency missing '+p)
            boundary();outputs={};flat=[]
            with native.torch.no_grad():
                for rec in ctx['records']:
                    boundary();tokens,target=read_source(rec)
                    rows=evaluate(tokens,target,lambda a,m,K:source.render_received(native,a,m,K),source.pixel_scores,boundary)
                    rows=[dict(source_index=rec['source_index'],source_id=rec['source_id'],original_calibration_index=rec['original_calibration_index'],population='original_calibration_construction_hash200',**v) for v in rows]
                    p=out/'sources'/f"{rec['source_index']:04d}.json";save(p,rows);outputs[str(p)]=sha(p);flat.extend(rows)
                    cp=out/'source_checkpoints'/f"{rec['source_index']:04d}.json";save(cp,dict(status='CLEAN_ONE_ENDPOINT_COMPLETE',registration_sha256=ctx['regsha'],**rec,outputs={str(p):sha(p)},images=1,new_packet_decodes=0));outputs[str(cp)]=sha(cp)
            require(len(flat)==200,'All200 clean outputs required');table=out/'clean_missing_per_source.csv'
            with table.open('x',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
            outputs[str(table)]=sha(table);boundary();native.frozen();verify(ctx['reg']['source_bindings']);verify(ctx['reg']['input_bindings'])
            result=dict(status=DONE,registration_sha256=ctx['regsha'],config_sha256=sha(path),source_count=200,source_ids=ctx['source_ids'],images=200,
                states=r['states'],outputs=outputs,source_bindings=ctx['sources'],input_bindings=ctx['reg']['input_bindings'],frozen_identity=native.loaded['identity'],
                numerical_runtime=native.flags,new_packet_decodes=0,development_used=False,holdout_used=False,selection=False,
                metric_scope='Original clean raster float64 MSE/PSNR only; no DINO or real channel evaluation',external_wait_required=True,elapsed_seconds=time.monotonic()-started)
            save(out/'completion.json',result);return result
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc(),pid=os.getpid(),new_packet_decodes=0));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['register','worker'],required=True);p.add_argument('--request');p.add_argument('--config');a=p.parse_args()
    result=register(a.request) if a.stage=='register' else worker(a.config);print(result['status'])
