"""Exact metadata-only compatibility boundary; no scientific implementation."""
from pathlib import Path
import ast, hashlib, json, os, sys, time

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
B=BASE/'baseline_strength_20261011'
E=R/'experiments/arpc-n2048-adaptive-20261019'
O=R/'results/arpc_n2048_adaptive_20261019_v1'
L=BASE/'arpc_n2048_adaptive_20261019_v1'
A=E/'b2b3_analysis_v1'; AO=O/'b2b3_analysis_v1'; AL=L/'b2b3_analysis_v1'
HERE=E/'b2b3_metadata_compatibility_v1'
CONTROL=L/'b2b3_metadata_compatibility_v1'
PUBLIC=O/'b2b3_metadata_compatibility_v1'
CPY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
FILES=('metadata.py','lineage.py','runner.py','test_metadata.py','README.md')
ORIGINAL={
 'analysis_common.py':(4780,'54ed47a43c4495daac6344fbbe7b0a38ac4d22dd8f60e63a93d4b1a75bce8fb6'),
 'dino_worker.py':(2939,'7c4866ede98eb5d340879307b0fd4d98ffdd5f71481df4533780f0f5436e7990'),
 'native.py':(34279,'d02813fe2cdb3be74129c9a7cd7afd0474307278daa75f5aab644ad38088a889'),
 'score.py':(17100,'eada10787de7cfd5580b1a89a73db7643656c88290b60e9576e6f00d9ae6a4a0'),
 'test_analysis.py':(16140,'1a957d009770004026b9b415bfa04b92f03ed6fa10f264e0ef1273855e561153'),
 'README.md':(11377,'4a8c8f157503456dc07b5c3da6c6163ab55df683f01a02b2c6f992a29cee2bdf')}
FAILURE_AUDIT=dict(path=str(O/'b2b3_analysis_metadata_failure_v1/failure_audit.json'),bytes=6556,
 sha256='d2b8a044152987359dca74e2489e52c62c332a38f0a761b91301dadfeb1575c8')
RESIDUE={
 'failure.json':(999,'5bb9c1abfa1c1b656d916e4cca384af224c7b732547e8246e8bcced76efaa656'),
 'native_identity_compatibility.json':(14906,'671ed778678c265e62f373f774ed41d5d74f146a55d03789cf6d5ae06b11b8c7')}
NEST_FIELDS=('model_state_identity','numeric_settings','native_binding','received_render_core_SHA','source_codec_SHA','source_math_SHA')

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def safe(p):
    p=Path(p);assert p.is_absolute() and '..' not in p.parts
    for q in (p,*p.parents):assert not q.is_symlink(),str(q)
    return p
def pin(p):
    p=safe(p);a=p.stat();b=p.read_bytes();z=p.stat()
    assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def verify(d):
    assert set(d)=={'path','bytes','sha256'} and type(d['bytes']) is int
    assert pin(d['path'])==d,d['path'];return Path(d['path'])
def read(p):return json.loads(safe(p).read_bytes())
def load(d):return read(verify(d))
def put(p,v):
    p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'w',encoding='utf8',newline='\n')as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def expected(p,n,h):return dict(path=str(p),bytes=n,sha256=h)
def identity(pid):
    try:
        x=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(x[19]),state=x[0],host=os.uname().nodename)
    except FileNotFoundError:return None
def gone(v,historical_hostless=False):
    if 'host' in v:assert v['host']==os.uname().nodename,'OTHER_HOST_IS_NOT_GONE'
    else:assert historical_hostless
    x=identity(v['pid']);return x is None or x['start_ticks']!=v['start_ticks'] or x['state']=='Z'
def same_process(a,b):
    return all(a[k]==b[k] for k in ('pid','start_ticks','host'))
def check_wait(w,returncode=0,hostless=False):
    assert type(w['returncode'])is int and w['returncode']==returncode
    for flag in ('timed_out','terminated','killed'):assert w[flag]is False
    assert w['metadata_error']is None and gone(w['process'],hostless)

def original_pins():return [expected(A/n,*ORIGINAL[n])for n in sorted(ORIGINAL)]
def validate_authorization(d):
    v=load(d);assert v['schema']=='B2B3_METADATA_COMPATIBILITY_AUTHORIZATION_V1'
    assert v['status']=='AUTHORIZED_METADATA_ONLY_REVISION'
    assert v['original_code']==original_pins() and v['failure_audit']==FAILURE_AUDIT
    assert v['scientific_math_changed'] is False and v['new_scientific_calls_authorized_by_revision']==0
    assert v['allowed_prepare_phases']==['B2','B3'] and v['automatic_retry'] is False
    assert {x['path']for x in v['adapter_code']}=={str(HERE/n)for n in FILES}
    for item in v['original_code']+v['adapter_code']+[v['failure_audit'],v['diagnosis'],v['independent_review'],v['qualification']]:verify(item)
    return v

def require_prepare_host(diagnosis,actual_host):
    assert diagnosis['status']=='METADATA_DIAGNOSIS_PASS_NO_PIXELS'
    assert diagnosis['scientific_calls']==diagnosis['pixel_files_opened']==0
    assert diagnosis['model_imports']==[] and diagnosis['source_files_changed']is False
    assert diagnosis['historical_no_host_PID_schema_preserved']is True
    assert actual_host==diagnosis['observer_host'],'HOSTLESS_HISTORY_MUST_BE_CHECKED_ON_PINNED_MAIN_HOST'

def check_failure_audit():
    v=load(FAILURE_AUDIT)
    assert v['status']=='METADATA_PREPARATION_FAILED_BEFORE_NATIVE_REGISTRATION'
    assert v['processes_gone']is True and v['registration_exists']is False and v['automatic_retry']is False
    assert all(type(v[k])is int and v[k]==0 for k in ('new_native_calls','new_metric_calls','new_channel_calls','scientific_success_repeated'))
    assert sorted(v['source'],key=lambda d:d['path'])==original_pins()
    for q in v['copies']:
        verify(q['original']);verify(q['copy']);assert q['original']['sha256']==q['copy']['sha256']
    w=load(v['actual_wait']);check_wait(w,1)
    launch=read(Path(v['actual_wait']['path']).with_name('launch.json'))
    assert same_process(launch['process'],w['process']) and gone(launch['parent_process'])
    assert launch['argv'][:4]==[str(CPY),'-B',str(A/'native.py'),'prepare']
    return v

def check_residue(root,out,initial):
    assert root==AL/'B2' and out==AO/'B2'
    safe(root);safe(out)
    for n,(size,h)in RESIDUE.items():verify(expected(root/n,size,h))
    if initial:
        assert {p.name for p in root.iterdir()}==set(RESIDUE)
        assert out.is_dir() and not list(out.iterdir())

REPLACEMENTS=(
 ('                ident=im[\'original_visual_identity\']\n                for k in NEST_ID_FIELDS:assert ident[k]==routes[\'native_qualification\'][k],(\'NATIVE_IDENTITY_CHANGED\',k)',
  '                revision_assert_entropy(c,routes)'),
 ('    root=L/phase;out=O/phase;root.mkdir(parents=True);out.mkdir(parents=True)',
  '    root=L/phase;out=O/phase;revision_directories(phase,root,out)'),
 ("    put(root/'native_identity_compatibility.json',lineage)",
  "    revision_lineage(phase,root,lineage)"))
def transformed(source):
    # Only the two metadata functions are compiled. The inverse must recover
    # every source character; all unaffected function ASTs are identical.
    original=source;changed=source
    for old,new in REPLACEMENTS:
        assert changed.count(old)==1 and new not in changed;changed=changed.replace(old,new)
    inverse=changed
    for old,new in reversed(REPLACEMENTS):inverse=inverse.replace(new,old)
    assert inverse==original
    aa={n.name:n for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    bb={n.name:n for n in ast.parse(changed).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert set(aa)==set(bb)
    assert all(ast.dump(aa[n])==ast.dump(bb[n]) for n in aa if n not in ('verified_candidate','prepare'))
    selected=ast.Module(body=[bb['verified_candidate'],bb['prepare']],type_ignores=[])
    return selected,dict(original_text_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_text_sha256=hashlib.sha256(changed.encode()).hexdigest(),inverse_exact=True,
        changed_functions=['verified_candidate','prepare'],all_other_functions_AST_equal=True)

def assert_closed_observation(v,wait,launch,completion,now_host):
    assert v['status']=='CLOSED_ON_PREPARE_HOST' and v['child_gone']is True and v['parent_gone']is True
    assert v['host']==v['child']['host']==v['parent']['host']
    assert same_process(v['child'],wait['process']) and same_process(v['child'],launch['process'])
    assert same_process(v['parent'],launch['parent_process'])
    assert same_process(v['child'],completion['process']) and same_process(v['parent'],completion['parent_process'])
    assert type(wait['returncode'])is int and wait['returncode']==0 and wait['metadata_error']is None
    for k in ('terminated','killed','timed_out'):assert wait[k]is False
    assert v['observed_unix']>=completion['finished_unix']
    assert v['observed_unix']>=launch['started_unix']+wait['seconds']-1
    assert launch['argv']==completion['argv']
    if now_host==v['host']:assert gone(v['child'])and gone(v['parent'])
    # A different host consumes the SHA-bound observation. It never interprets
    # absent local /proc entries for a remote PID as evidence of termination.

def reject_unknown_failure(phase):
    roots=[AL/phase,CONTROL/phase,AL/phase/'dino',AL/phase/'dino/worker']
    for name in ('nest','arpc'):
        roots.extend([AL/phase/name,AL/phase/name/'rx',AL/phase/name/'runtime_metadata_control'])
    roots.extend(AL/'controls'/(phase+'_'+name)for name in ('nest','arpc','dino'))
    allowed=AL/'B2/failure.json'
    for root in roots:
        p=root/'failure.json'
        if not p.exists():continue
        if p==allowed:verify(expected(p,*RESIDUE['failure.json']))
        else:raise AssertionError(('UNKNOWN_FAILURE',str(p)))
