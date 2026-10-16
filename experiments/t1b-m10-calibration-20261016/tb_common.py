"""T1-B only: immutable metadata and reuse utilities; no launch side effects."""
from pathlib import Path
import sys, json, hashlib, os, time, math
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT=R.parent.parent/'var_comm_runtime_20261010'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
E=R/'experiments/t1b-m10-calibration-20261016'
O=R/'results/t1b_m10_calibration_20261016_v1'
T=B/'t1b_m10_calibration_20261016_v1'
FIRST=R/'experiments/submission-closure-20261016'
FIRST_O=R/'results/submission_closure_20261016_v1'
FIRST_T=B/'submission_closure_20261016_v1'
S=R/'experiments/entropy-partial-mismatch-20261010/scripts'
Q=RT/'qualification'
sys.path.insert(0,str(FIRST));sys.path.insert(0,str(S))
from common import sha,pin,read,checked,save,csvsave,asha,pid_identity,gone,Ledger,MODELS,ENDS,OFFSETS,SIZES
HEAD='3818fe6b05121347dbfa96f5d708818d8c0d21c8'
SNRS=(4,10,19);SEEDS=(4101,4102,4103)
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
_parents=set()
def checkread(v):
 p=Path(v['path']);assert p.is_absolute() and '..' not in p.parts
 for q in (p,*p.parents):
  if q not in _parents:assert not q.is_symlink();_parents.add(q)
 a=p.stat();b=p.read_bytes();z=p.stat()
 assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns) and ('bytes' not in v or len(b)==v['bytes']) and hashlib.sha256(b).hexdigest()==v['sha256'],str(p)
 return json.loads(b)
def registration(expected):
 p=O/'registration.json';assert sha(p)==expected
 r=read(p)
 for d in r['code']:checked(d)
 assert r['scope']=='T1-B_ONLY_STOP_AFTER_DELIVERY' and r['T1C'] is False
 return r
def counter(i,s,seed):return ([1,4,7,10,13,19].index(s)*1000+i)*3+SEEDS.index(seed)
def progress(out,v):
 with (out/'progress.jsonl').open('a',encoding='utf8')as f:f.write(canonical(dict(observed_unix=time.time(),**v))+'\n');f.flush()
def runtime_cache(out):
 import tempfile
 for n in ('tmp','inductor','triton'):(out/'runtime_cache'/n).mkdir(parents=True,exist_ok=False)
 os.environ.update(TMPDIR=str(out/'runtime_cache/tmp'),TMP=str(out/'runtime_cache/tmp'),TEMP=str(out/'runtime_cache/tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache/inductor'),TRITON_CACHE_DIR=str(out/'runtime_cache/triton'))
 tempfile.tempdir=os.environ['TMPDIR']
def stage_start(name,expected):
 r=registration(expected);out=T/name;out.mkdir(exist_ok=False);runtime_cache(out)
 os.sched_setaffinity(0,{0,1});save(out/'launch.json',dict(process=pid_identity(os.getpid()),parent_process=pid_identity(os.getppid()),argv=sys.argv,registration=pin(O/'registration.json')))
 return r,out
def source_key(rx):
 b=rx['body'];p=rx['rx_profile']
 if not rx['header']['header_ok'] or b is None or not b['crc_accepted'] or not b['parser_accepted']:return digest({'kind':'FIXED_GRAY'})
 return digest(dict(kind='ACTUAL_SOURCE_INPUT',family=p['family'],m=p['m'],K=p['K'],payload=b['payload']))
def event_lookup(i,s,seed,profile_id,payload_sha):return canonical([i,s,seed,counter(i,s,seed),profile_id,payload_sha])
def old_closed(name):
 p=Q/name;run=p/'registered/run';done=read(run/'completion.json');worker=checkread(done['worker_completion']);wait=read(p/'run_owner_actual_wait.json')
 assert done['actual_children_waited'] and done['worker_exit_codes']==[0] and done['actual_wait']['success'] and wait['actual_wait'] and type(wait['returncode'])is int and wait['returncode']==0
 request=p/'registered/request.json';assert sha(request)==done['request_sha256']==worker['request_sha256']
 return read(request),worker,[pin(request),pin(run/'completion.json'),pin(run/'worker_completion.json'),pin(p/'run_owner_actual_wait.json')]
