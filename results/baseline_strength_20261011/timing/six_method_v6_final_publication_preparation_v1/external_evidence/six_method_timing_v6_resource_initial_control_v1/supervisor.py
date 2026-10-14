"""Explicit once-only outer supervisor. No science occurs when this is imported."""
from pathlib import Path
import argparse,contextlib,csv,fcntl,hashlib,json,os,signal,stat,subprocess,sys,time,traceback

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
B=T/'baseline_strength_20261011';L=T/'layered_mismatch_raw_20261013'
E=R/'experiments/baseline-delivery-20261012/timing_resource_revision_v1'
ENTRY=E/'timing_resource_owner.py'
O=R/'results/baseline_strength_20261011/timing';OLD=O/'six_method_v6'
REG=O/'six_method_v6_resource_v1/registration.json'
CONTROL=B/'six_method_timing_v6_resource_v1'
D=B/'six_method_timing_v6_resource_dispatch_v1';I=B/'six_method_timing_v6_resource_initial_control_v1'
PY=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python')
OLD_SHA='54aafaca38b177266ef2c21cac23fce4dee2a0658e9bd9a0cfc5d2836baa269a'
OUTER_SECONDS=15169  # Original 13069 worker budget + original 1800 owner grace + 300 outer grace.
SOURCES=('resource_adapter.py','timing_resource_owner.py','test_resource_adapter.py','README.md')
LOCKS=tuple(O/('six_method_v%d'%v)/'owner.lock' for v in range(1,7))+tuple(L/p for p in (
 'r1_execution_v1/owner.lock','r1_resource_dispatch_v2/dispatch.lock','r1_cache_recovery_v1/owner.lock',
 'r1_cache_recovery_dispatch_v1/dispatch.lock','r1_cache_recovery_delivery_dispatch_v1/dispatch.lock'))

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def checked(p):
    require(set(p)=={'path','sha256','bytes'} and type(p['bytes']) is int,'Exact byte pin required')
    q=Path(p['path']);require(q.resolve()==q and stat.S_ISREG(q.lstat().st_mode),'Nonregular metadata path')
    require(pin(q)==p,'Pinned metadata differs: '+str(q));return q
def put(p,v):
    data=(json.dumps(v,indent=2)+'\n').encode('utf8')
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def proc(pid):
    try:
        q=Path('/proc')/str(pid);text=(q/'stat').read_text();a=text[text.rfind(')')+2:].split()
        return dict(pid=int(pid),start_ticks=int(a[19]),state=a[0],pgrp=int(a[2]),session=int(a[3]),
            argv=(q/'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError,ProcessLookupError):return None
def same(a,b):return a is not None and (a['pid'],a['start_ticks'])==(b['pid'],b['start_ticks'])

def metadata(expected):
    require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'Explicit registration SHA required')
    require(sha(REG)==expected,'Supplemental registration changed');reg=read(REG)
    require(reg['schema']=='TIMING_V6_STARTUP_RESOURCE_REVISION_V1' and reg['status']=='REGISTERED_NOT_EXECUTED','Wrong revision')
    wanted=dict(startup_free_MiB=8192,prior_startup_free_MiB=18432,GPU_index=0,
        initial_GPU_utilization_max_percent=25,allocator_cap_GiB=16,runtime_free_floor_MiB=2048,
        inherited_complete_calls=576,remaining_complete_calls=1152,remaining_seconds=13069,
        scientific_math_changed=False,numeric_runtime_changed=False,automatic_retries=False)
    for k,v in wanted.items():require(type(reg[k]) is type(v) and reg[k]==v,'Different registered scope: '+k)
    require(set(reg['sources'])==set(SOURCES),'Four reviewed sources required')
    for name,p in reg['sources'].items():require(checked(p)==E/name,'Unexpected wrapper source')
    require(checked(reg['original_registration'])==OLD/'registration.json' and reg['original_registration']['sha256']==OLD_SHA,'Different original registration')
    old=read(OLD/'registration.json');require(old['max_seconds']==13069 and old['remaining_complete_frames']==1152,'Original budget changed')
    auth=read(checked(reg['authorization']))
    require(auth['schema']=='TIMING_V6_STARTUP_RESOURCE_AUTHORIZATION_V1' and auth['status']=='AUTHORIZED_18_TO_8_GIB'
        and auth['user_instruction']=='计时放开一下门槛' and type(auth['scientific_calls']) is int and auth['scientific_calls']==0,'Missing exact user authorization')
    require(auth['expected_source_sha256']=={n:p['sha256'] for n,p in reg['sources'].items()} and auth['review_evidence'],'Review source bindings differ')
    for p in auth['review_evidence']:checked(p)
    release=O/'six_method_v2/coordination_release.json';value=read(release)
    require(value['status']=='TIMING_WINDOW_RELEASED' and value['quality_jobs_stopped'] is True
        and value['no_quality_science_until_timing_actual_exit'] is True and value['evidence'],'Original coordination not released')
    for p in value['evidence']:
        require(Path(p['path']).is_file() and sha(p['path'])==p['sha256'],'Original coordination evidence changed')
    return dict(resource_registration=pin(REG),original_registration=reg['original_registration'],
        sources=reg['sources'],authorization=reg['authorization'],coordination_release=pin(release),
        scientific_seconds=13069,original_owner_grace_seconds=1800,outer_only_grace_seconds=300)

def no_attempt(allow_initial=False):
    forbidden=[B/'six_method_timing_v6',B/'six_method_timing_v6_owner',B/'six_method_timing_v6_dispatch_v1',CONTROL,D]
    forbidden += [OLD/n for n in ('dispatch.json','launch.json','worker_launch.json','actual_wait.json','completion.json')]
    if not allow_initial:forbidden.append(I)
    require(not any(os.path.lexists(p) for p in forbidden),'Existing attempt/control; preserve and inspect, never retry')
    return [str(p) for p in forbidden]

def availability():
    handles=[];blocked=[]
    try:
        for p in LOCKS:
            require(p.resolve()==p and p.is_file(),'Missing or symlinked original coordination lock')
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);handles.append(fd)
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:blocked.append(str(p))
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        selected=[r for r in csv.reader(raw.splitlines()) if r and int(r[0])==0];require(len(selected)==1,'GPU0 absent/ambiguous')
        row=selected[0];gpu=dict(index=0,uuid=row[1].strip(),name=row[2].strip(),free_MiB=float(row[3]),utilization_percent=float(row[4]))
        return dict(observed_unix=time.time(),GPU0=gpu,locks_checked=[str(p) for p in LOCKS],blocked_locks=blocked,
            admitted=not blocked and gpu['free_MiB']>=8192 and gpu['utilization_percent']<=25,
            scientific_calls=0,original_readiness_or_preflight_called=False)
    finally:
        for fd in reversed(handles):os.close(fd)

def group_members(group):
    rows=[]
    for q in Path('/proc').iterdir():
        if q.name.isdigit():
            try:
                text=(q/'stat').read_text();a=text[text.rfind(')')+2:].split()
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if int(a[2])==group and int(a[3])==group:
                x=proc(int(q.name))
                if x is not None:rows.append(x)
    return rows

def stop_owned_group(child,identity,flags):
    current=proc(identity['pid'])
    require(current is None or same(current,identity),'Owned leader PID reused; no signal sent')
    members=group_members(identity['pid']);flags['cleanup_group_members_before']=members
    if members:
        os.killpg(identity['pid'],signal.SIGTERM);flags['terminated']=True
    try:code=child.wait(timeout=30)
    except subprocess.TimeoutExpired:code=None
    live=[x for x in group_members(identity['pid']) if x['state']!='Z']
    if live:
        current=proc(identity['pid']);require(current is None or same(current,identity),'Leader generation changed before kill')
        os.killpg(identity['pid'],signal.SIGKILL);flags['killed']=True
    if code is None:code=child.wait(timeout=30)
    flags['cleanup_group_members_after']=group_members(identity['pid'])
    return code

def supervise(expected,helper_sha):
    require(sha(Path(__file__))==helper_sha,'Supervisor source differs')
    deadline=time.monotonic()+15
    while not (I/'launch.json').exists() and time.monotonic()<deadline:time.sleep(.05)
    launch=read(I/'launch.json');me=proc(os.getpid())
    require(same(launch['process'],me) and launch['argv']==me['argv'] and me['pgrp']==me['pid']==me['session'],'Supervisor generation/session/argv differs')
    require(launch['resource_registration']['sha256']==expected,'Outer registration changed')
    bindings=metadata(expected);absent=no_attempt(allow_initial=True);snapshot=availability()
    require(snapshot['admitted'],'Resources changed after initial observation; preserve this attempted control, no retry')
    D.mkdir(mode=0o700)
    lock=os.open(D/'dispatch.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(D/'prepared.json',dict(schema='TIMING_V6_RESOURCE_OUTER_V1',process=me,parent_process=proc(os.getppid()),
        initial_launcher=launch['parent_process'],bindings=bindings,observation=snapshot,absence=absent,outer_max_seconds=OUTER_SECONDS))
    child=None;identity=None;error=None;code=None;flags=dict(timed_out=False,terminated=False,killed=False)
    argv=[str(PY),'-B',str(ENTRY),'run','--expected-resource-registration',expected]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)  # Original run selects GPU0 for its worker.
    start=time.monotonic()
    try:
        with (D/'stdout.txt').open('xb') as out,(D/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(argv,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            identity=proc(child.pid);require(identity is not None and identity['argv']==argv and identity['pgrp']==child.pid==identity['session'],'Actual wrapper leader differs')
            put(D/'launch.json',dict(schema='TIMING_V6_RESOURCE_OUTER_V1',process=identity,parent_process=me,argv=argv,
                bindings=bindings,outer_max_seconds=OUTER_SECONDS,scientific_seconds_unchanged=13069,initial_control=pin(I/'launch.json')))
            try:code=child.wait(timeout=OUTER_SECONDS)
            except subprocess.TimeoutExpired:
                flags['timed_out']=True;code=stop_owned_group(child,identity,flags)
                raise RuntimeError('Outer watchdog expired; no additional scientific budget or retry')
            if code!=0:
                stop_owned_group(child,identity,flags)
                raise RuntimeError('Original resource wrapper failed; preserve all original evidence, no retry')
    except BaseException:
        error=traceback.format_exc()
        if child is not None and child.poll() is None:
            identity=identity or proc(child.pid)
            require(identity is not None,'Started child generation unavailable')
            code=stop_owned_group(child,identity,flags)
    finally:
        if child is not None:
            if code is None:code=child.wait(timeout=30)
            put(D/'actual_wait.json',dict(actual_wait=True,returncode=code,child=identity,parent_process=me,argv=argv,
                elapsed_seconds=time.monotonic()-start,outer_max_seconds=OUTER_SECONDS,scientific_seconds_unchanged=13069,
                resource_registration=bindings['resource_registration'],metadata_error=error,**flags,
                stdout=pin(D/'stdout.txt'),stderr=pin(D/'stderr.txt')))
        os.close(lock)
    if error is not None:
        put(D/'failure.json',dict(status='STOPPED_NO_RETRY',error=error,science_completion_not_claimed=True));raise RuntimeError(error)
    require(code==0 and not group_members(identity['pid']),'Owned wrapper group still present after wait')
    metadata(expected)
    require(not list(CONTROL.glob('*failure.json')),'Resource wrapper retained failure')
    require(read(OLD/'actual_wait.json')==dict(actual_child_waited=True,exit_code=0,timed_out=False),'Original real worker wait differs')
    require(read(OLD/'completion.json')['status']=='COMPLETE_FIXED16_SIX_METHOD_TIMING','Original timing completion missing')
    for role in ('owner','worker'):
        receipt=read(CONTROL/(role+'_resource_execution.json'));returned=read(CONTROL/(role+'_returned.json'))
        require(returned['status']=='ORIGINAL_V6_RETURNED_SUCCESS' and returned['process']==receipt['process'],'Missing actual wrapper return')
        x=receipt['process'];current=proc(x['pid']);require(current is None or int(current['start_ticks'])!=int(x['start_ticks']),'Original generation still present')
    put(D/'completion.json',dict(status='OUTER_ACTUAL_WAIT_ZERO_ORIGINAL_TIMING_COMPLETE',resource_registration=pin(REG),
        original_completion=pin(OLD/'completion.json'),original_actual_wait=pin(OLD/'actual_wait.json'),
        actual_wait=pin(D/'actual_wait.json'),requires_supervisor_generation_gone=True,postprocessing_not_run=True))

def launch(expected):
    bindings=metadata(expected);absent=no_attempt();snapshot=availability()
    if not snapshot['admitted']:
        print(json.dumps(dict(status='WAIT_NO_OUTER_ATTEMPT_CREATED',snapshot=snapshot)));return
    I.mkdir(mode=0o700);me=proc(os.getpid());source=Path(__file__).read_bytes();snapshot_path=I/'supervisor.py'
    with snapshot_path.open('xb') as f:f.write(source);f.flush();os.fsync(f.fileno())
    digest=hashlib.sha256(source).hexdigest()
    put(I/'prepared.json',dict(process=me,bindings=bindings,observation=snapshot,absence=absent,source=pin(snapshot_path)))
    argv=[str(PY),'-B',str(snapshot_path),'supervise','--expected-resource-registration',expected,'--expected-helper',digest]
    child=None;identity=None
    try:
        with (I/'stdout.txt').open('xb') as out,(I/'stderr.txt').open('xb') as err:
            child=subprocess.Popen(argv,cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'),
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        identity=proc(child.pid);require(identity is not None and identity['argv']==argv,'Actual supervisor identity differs')
        value=dict(status='OUTER_SUPERVISOR_LAUNCHED_NOT_SCIENTIFIC_COMPLETION',process=identity,parent_process=me,argv=argv,
            resource_registration=bindings['resource_registration'],source=pin(snapshot_path),outer_max_seconds=OUTER_SECONDS)
        put(I/'launch.json',value);print(json.dumps(value))
    except BaseException:
        error=traceback.format_exc();flags=dict(terminated=False,killed=False)
        if child is not None:
            identity=identity or proc(child.pid)
            require(identity is not None,'Supervisor generation unavailable after Popen')
            code=stop_owned_group(child,identity,flags)
            put(I/'actual_wait_after_launch_failure.json',dict(actual_wait=True,returncode=code,process=identity,**flags))
        put(I/'failure.json',dict(status='STOPPED_NO_RETRY',error=error));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=('launch','supervise'))
    p.add_argument('--expected-resource-registration',required=True);p.add_argument('--expected-helper');a=p.parse_args()
    require(os.name=='posix' and Path('/proc').is_dir() and Path(__file__).is_file(),'Deploy explicit helper to personal Linux temporary area')
    if a.role=='launch':launch(a.expected_resource_registration)
    else:
        try:supervise(a.expected_resource_registration,a.expected_helper)
        except BaseException:
            if I.exists() and not (I/'supervisor_failure.json').exists():put(I/'supervisor_failure.json',dict(status='STOPPED_NO_RETRY',error=traceback.format_exc()))
            raise

if __name__=='__main__':main()
