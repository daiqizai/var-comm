"""Bounded raw telemetry reuse; original STOP, peer and thermal checks remain."""
from __future__ import annotations
import atexit
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/home/liulu/projects/VAR_COMM')
OUT = ROOT / 'outputs/M2-TELEMETRY-20261002'
R4 = ROOT / 'outputs/METRIC-CONCURRENT-R4-20261002'
ENTRY = R4 / 'runtime/scheduled_process.py'
ENV = 'M2_TELEMETRY_MANIFEST'
TTL = 1.0
COMPUTE_QUERY = ('nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits')
THERMAL_QUERY = ('nvidia-smi','--id=0','--query-gpu=temperature.gpu,clocks_event_reasons.hw_thermal_slowdown','--format=csv,noheader,nounits')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sources(folder):
    paths = sorted(p for p in Path(folder).resolve().iterdir() if p.suffix in ('.py','.md'))
    if not paths or any(p.is_symlink() or not p.is_file() for p in paths): raise RuntimeError('Invalid telemetry sources')
    return {str(p): sha(p) for p in paths}
def verify(mapping):
    if not isinstance(mapping,dict) or not mapping: raise RuntimeError('Missing telemetry bindings')
    for path,digest in mapping.items():
        if not re.fullmatch('[0-9a-f]{64}',str(digest)) or sha(path)!=digest: raise RuntimeError('Telemetry bound file changed: '+path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


class QueryCache:
    def __init__(self, original=None, approved_pids=None, *, clock=time.monotonic, ttl=TTL, enabled=True, record=None):
        if isinstance(ttl,bool) or not isinstance(ttl,(int,float)) or not math.isfinite(ttl) or not 0 < ttl <= TTL:
            raise ValueError('Telemetry TTL must be positive and at most one second')
        self.original=original or subprocess.check_output;self.approved_pids=approved_pids or (lambda:{})
        self.clock,self.ttl,self.enabled,self.record=clock,float(ttl),bool(enabled),record
        self.entries={};self.first=True;self.last_record=None;self.reset_stats()
    def reset_stats(self):
        self.counters=dict(fresh_calls=0,cache_hits=0,invalidations=0,query_seconds=0.,by_query={})
    def stats(self): return json.loads(json.dumps(self.counters))
    def clear(self):
        self.entries.clear();self.counters['invalidations']+=1
    def report(self):
        now=self.clock()
        if self.first or self.last_record is None or now-self.last_record>=30:
            status='FIRST_QUERY_RECORDED' if self.first else 'RUNNING_QUERY_STATS'
            self.first=False;self.last_record=now
            if self.record is not None:self.record(status,self.stats())
    def checked(self,key,value):
        try:return self.valid(key,value)
        except BaseException:self.clear();raise
    def valid(self,key,value):
        if not isinstance(value,str): return None
        if key==THERMAL_QUERY:
            fields=value.strip().split(',')
            if len(fields)!=2 or not re.fullmatch('[0-9]+',fields[0].strip()): return None
            if not 0 <= int(fields[0]) < 75 or fields[1].strip().lower()!='not active': return None
            return {}
        allowed=self.approved_pids()
        if not isinstance(allowed,dict) or any(type(p) is not int or p<=1 or not str(t).isdigit() for p,t in allowed.items()):
            raise RuntimeError('Invalid admitted process identities')
        token={}
        for line in value.splitlines():
            if not line.strip(): continue
            fields=line.split(',')
            if len(fields)!=3 or not fields[0].strip().isdigit() or not fields[1].strip(): return None
            pid=int(fields[0].strip())
            try: memory=float(fields[2].strip())
            except ValueError: return None
            if pid not in allowed or pid in token or not math.isfinite(memory) or memory<0: return None
            token[pid]=str(allowed[pid])
        return token
    def __call__(self,*args,**kwargs):
        if (len(args)!=1 or type(args[0]) not in (list,tuple) or any(type(v) is not str for v in args[0])
                or tuple(args[0]) not in (COMPUTE_QUERY,THERMAL_QUERY) or set(kwargs)!={'text'} or kwargs['text'] is not True):
            return self.original(*args,**kwargs)
        key=tuple(args[0]);kind='compute' if key==COMPUTE_QUERY else 'thermal'
        stats=self.counters['by_query'].setdefault(kind,dict(fresh_calls=0,cache_hits=0))
        now=self.clock();cached=self.entries.get(key)
        if self.enabled and cached is not None and 0 <= now-cached['started'] < self.ttl:
            token=self.checked(key,cached['value'])
            if token is not None and token==cached['token']:
                self.counters['cache_hits']+=1;stats['cache_hits']+=1
                self.report()
                return cached['value']
            self.clear()
        started=self.clock();self.counters['fresh_calls']+=1;stats['fresh_calls']+=1
        try: value=self.original(*args,**kwargs)
        except BaseException:
            self.clear();raise
        finally: self.counters['query_seconds']+=self.clock()-started
        if self.enabled:
            token=self.checked(key,value)
            if token is None:self.clear()
            else:self.entries[key]=dict(started=started,value=value,token=token)
        self.report()
        return value
    def __enter__(self):
        if getattr(subprocess.check_output,'_m2_telemetry_cache',False): raise RuntimeError('Telemetry cache already installed')
        self.previous=subprocess.check_output
        def wrapped(*a,**k):return self(*a,**k)
        wrapped._m2_telemetry_cache=True;self.wrapped=wrapped;subprocess.check_output=wrapped;return self
    def __exit__(self,*args):
        if subprocess.check_output is not self.wrapped:raise RuntimeError('Telemetry wrapper changed unexpectedly')
        subprocess.check_output=self.previous


Cache=QueryCache


def make_approved_pids(root,guard,admission_path):
    root=Path(root);out=root/'outputs/METRIC-CONCURRENT-R4-20261002'
    admission=read(admission_path);digest=sha(admission_path)
    me=guard.identity(guard.process(os.getpid()))
    def approved():
        if sha(admission_path)!=digest:raise RuntimeError('Telemetry admission changed')
        current=guard.process(me['pid'])
        if not guard.alive(me,current):raise RuntimeError('Telemetry self identity changed')
        answer={me['pid']:me['start_ticks']}
        peer=guard.approved_peer(out,admission,digest,'m2')
        if peer is not None:
            actual=guard.process(peer['pid'])
            # The original guard retains its cleanup grace; a raw value is
            # cached only while the peer is genuinely alive with this identity.
            if guard.alive(peer,actual):answer[peer['pid']]=peer['start_ticks']
        return answer
    return approved


def qualification(path,digest,own):
    path=Path(path).resolve()
    if path!=(OUT/'telemetry_qualification.json').resolve() or sha(path)!=digest:
        raise RuntimeError('Telemetry qualification path/hash differs')
    value=read(path)
    payload={k:v for k,v in value.items() if k!='payload_sha256'}
    payload_sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    speed=value.get('speedup')
    if (value.get('status')!='QUALIFIED' or value.get('selected_implementation')!='cached'
            or value.get('strict_equality_passed') is not True or type(speed) not in (int,float)
            or not math.isfinite(speed) or speed<1.10 or value.get('payload_sha256')!=payload_sha
            or value.get('source_bindings')!=own or value.get('engineering_fixture') is not True
            or value.get('real_model_weights') is not True or value.get('scientific_result') is not False
            or value.get('development_read') is not False or value.get('training_updates')!=0
            or value.get('policy_selection_updates')!=0 or value.get('original_scientific_outputs_written') is not False):
        raise RuntimeError('Real exact-parity telemetry qualification is not deployment eligible')
    verify(value.get('inputs'))
    return value


def validate(path,argv,here):
    path=Path(path).resolve();here=Path(here).resolve()
    if path!=OUT/'manifest.json' or here!=OUT/'runtime':raise RuntimeError('Unexpected telemetry manifest/runtime')
    if (len(argv)!=7 or Path(argv[0]).resolve()!=ENTRY or argv[1:3]!=['--root',str(ROOT)]
            or argv[3:5]!=['--admission',str(R4/'admission.json')] or argv[5]!='--stage'
            or argv[6] not in ('calibration','evaluation','actual','timing')):raise RuntimeError('Telemetry execution scope differs')
    value=read(path);own=sources(here);original=value.get('original_source_bindings')
    if (value.get('status')!='REGISTERED_M2_TELEMETRY_EXTENSION' or value.get('entrypoint')!=str(ENTRY)
            or value.get('runtime_source_bindings')!=own or value.get('ttl_seconds')!=TTL
            or value.get('training_updates')!=0 or value.get('policy_selection_updates')!=0
            or value.get('inference_functions_changed') is not False):raise RuntimeError('Telemetry manifest scope differs')
    verify(own);verify(original)
    required={str(ROOT/'experiments/scale-causal-partial-residual-20261002'/n) for n in ('common.py','m2_runner.py')}
    required.add(str(ROOT/'experiments/metric-speed-20261002/wrapper.py'));required.add(str(R4/'admission.json'))
    r4=sources(R4/'runtime');admission=read(R4/'admission.json');required.update(admission['original_guard_bindings'])
    if not required.issubset(original) or any(original.get(p)!=h for p,h in r4.items()):raise RuntimeError('Original guards/full R4/M2 are not pinned')
    for kind in ('manifest','sitecustomize'):
        selected=Path(value.get('bool_'+kind+'_path','')).resolve()
        expected=ROOT/'outputs/M2-JSON-RECOVERY-20261002'/('manifest.json' if kind=='manifest' else 'runtime/sitecustomize.py')
        if selected!=expected or sha(selected)!=value.get('bool_'+kind+'_sha256'):raise RuntimeError('Pinned Boolean startup recovery differs')
    if os.environ.get('M2_JSON_RECOVERY_MANIFEST')!=value['bool_manifest_path']:raise RuntimeError('Original Boolean manifest environment differs')
    qualification(value.get('qualification_path',''),value.get('qualification_sha256'),own)
    return value


def install(path,argv,here):
    manifest=validate(path,argv,here)
    # Explicitly chain the already frozen Boolean recovery once; adding a new
    # leading sitecustomize directory must not shadow that existing repair.
    module('_m2_telemetry_pinned_boolean_startup',manifest['bool_sitecustomize_path'])
    guard=module('_m2_telemetry_frozen_scheduling_guard',R4/'runtime/scheduling_guard.py')
    me=guard.identity(guard.process(os.getpid()));destination=OUT/'executions'/f'{me["pid"]}_{me["start_ticks"]}.json'
    bound={str(Path(path).resolve()):sha(path),**manifest['runtime_source_bindings'],**manifest['original_source_bindings']}
    for key in ('bool_manifest','bool_sitecustomize'):bound[manifest[key+'_path']]=manifest[key+'_sha256']
    bound[manifest['qualification_path']]=manifest['qualification_sha256']
    enabled=argv[6]!='timing'
    base=dict(**me,stage=argv[6],entrypoint=str(ENTRY),inputs=bound,runtime_source_bindings=manifest['runtime_source_bindings'],
        manifest_sha256=sha(path),enabled=enabled,ttl_seconds=TTL,max_detection_delay_seconds=TTL if enabled else 0.,
        raw_command_protocol=[list(COMPUTE_QUERY),list(THERMAL_QUERY)],exact_kwargs={'text':True},
        stop_and_original_guards_called_every_boundary=True,original_peer_identity_checked_each_boundary=True,
        inference_functions_changed=False,training_updates=0,policy_selection_updates=0,scientific_completion_claimed=False)
    def record(status,stats):
        verify(bound);destination.parent.mkdir(parents=True,exist_ok=True)
        tmp=destination.with_name(destination.name+'.tmp')
        tmp.write_text(json.dumps(dict(base,status=status,stats=stats,time=time.time()),sort_keys=True,indent=2)+'\n',encoding='utf-8')
        os.replace(tmp,destination)
    cache=QueryCache(approved_pids=make_approved_pids(ROOT,guard,R4/'admission.json'),enabled=enabled,record=record)
    # Formal timing preserves the exact raw subprocess function object. Even a
    # pass-through wrapper or periodic statistics would add measured RX work.
    if enabled:cache.__enter__()
    record('REGISTERED' if enabled else 'DISABLED_FOR_FORMAL_TIMING',cache.stats())
    atexit.register(lambda:record('PROCESS_EXIT_RECORDED',cache.stats()))
    return cache
