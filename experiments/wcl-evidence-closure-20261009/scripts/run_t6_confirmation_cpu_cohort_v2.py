"""Own one finite T6 confirmation CPU cohort: four raw + four entropy workers.

Only the registered ledger-init and cpu-worker commands can be launched.
All eight workers use one request and its existing 5400-packet ledger.
No GPU stage or automatic retry is launched, including after interrupted starts.
"""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path

SCHEMA='T6_CONFIRMATION_CPU_COHORT_REGISTRATION_V1'
REQUEST_SCHEMA='WCL_T6_CONFIRMATION100_RENDER_EXECUTION_V1'
WORKERS=4

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(desc):
    require(sha(desc['path'])==desc['sha256'],'Changed sealed dependency: '+desc['path']);return read(desc['path'])

def validate(r):
    require(r['schema']==REQUEST_SCHEMA and r['N']==2048 and r['source_count']==100 and r['frame_count']==2700
        and r['packet_cap']==5400 and r['workers']==WORKERS,'Fixed confirmation request requires --workers 4 per branch, not total8')
    require(r['snrs']==[4,10,19]and r['noise_seeds']==[9201,9202,9203]and len(r['records'])==100
        and len(set(r['source_ids']))==100 and[(x['source_index'],x['source_id'])for x in r['records']]==list(enumerate(r['source_ids'])),
        'Fixed confirmation population and noise seeds required')
    require(not r['source_reselection']and not r['policy_reselection'],'No confirmation selection')
    require(len(r['schedule'])==9 and sum(x['branch']=='raw'for x in r['schedule'])==6
        and sum(x['branch']=='entropy'for x in r['schedule'])==3,'Exactly six raw and three entropy method/SNR policies')

def prepare(request,record_dir):
    request=Path(request).resolve();r=read(request);validate(r)
    runner=Path(__file__).with_name('t6_confirmation_render.py').resolve();record=Path(record_dir).resolve();out=Path(r['out']).resolve()
    require(record.is_relative_to(out)and record!=out and not record.exists(),'Fresh cohort directory below the registered output required')
    require(not(out/'cpu_cohort_launch_claim.json').exists(),'A cohort was already attempted; inspect it instead of restarting')
    require(r['source_bindings'].get(str(runner))==sha(runner),'Exact frozen confirmation runner required')
    for p,h in r['source_bindings'].items():require(sha(p)==h,'Changed registered science dependency: '+p)
    require(Path(r['python_cpu']).is_file()and Path(r['root']).is_dir(),'Registered interpreter and repository required')
    python=os.path.abspath(os.path.expanduser(r['python_cpu']));argv=[python,'-B',str(runner)]
    jobs=[dict(job_id=f'{branch}_{i}',branch=branch,index=i,argv=argv+['cpu-worker','--request',str(request),'--branch',branch,'--index',str(i)])
        for i in range(WORKERS)for branch in('raw','entropy')]
    registration=dict(schema=SCHEMA,request=pin(request),runner=pin(runner),owner=pin(__file__),record_dir=str(record),root=r['root'],
        output_dir=str(out),python_cpu=python,workers_per_branch=4,total_workers=8,packet_cap=5400,
        ledger_path=str(out/'packet_ledger.sqlite'),ledger_init=dict(job_id='ledger_init',argv=argv+['ledger-init','--request',str(request)]),
        jobs=jobs,deadline_unix=r['deadline_unix'],stop_files=r['stop_files'],launch_stagger_seconds=.3,poll_seconds=.2,
        terminate_timeout_seconds=30,source_bindings=r['source_bindings'],global_claim=str(out/'cpu_cohort_launch_claim.json'),
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),automatic_successor=False,automatic_retry=False)
    record.mkdir(parents=True);save(record/'registration.json',registration)
    return dict(status='T6_CONFIRMATION_CPU_COHORT_REGISTERED_NOT_RUN',registration=pin(record/'registration.json'),total_workers=8,new_packet_calls=0)

def guard(reg):
    require(time.time()<reg['deadline_unix'],'Registered deadline reached; no successor or retry')
    require(not any(Path(p).exists()for p in reg['stop_files']),'Registered STOP file observed')

def children_stage(reg,jobs,phase):
    """Standard-library subprocess scheduler; tests use real synthetic children."""
    record=Path(reg['record_dir']);entries=[];failure=None;started=time.time();env=dict(os.environ,**reg['environment'])
    def healthy():
        guard(reg)
        require(not any(e['process'].poll()not in(None,0)for e in entries),'Owned worker failed; later launches stopped')
    try:
        for job in jobs:
            healthy();base=record/job['job_id'];base.mkdir()
            save(base/'reservation.json',dict(phase=phase,job=job,owner_pid=os.getpid(),reserved_unix=time.time()))
            log=(base/'child.log').open('xb')
            try:child=subprocess.Popen(job['argv'],cwd=reg['root'],env=env,stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            entry=dict(process=child,log=log,base=base,job=job,started_unix=time.time());entries.append(entry)
            save(base/'launch.json',dict(job=job,argv=job['argv'],cwd=reg['root'],owner_pid=os.getpid(),child_pid=child.pid,
                started_unix=entry['started_unix'],request=reg.get('request'),phase=phase))
            until=time.monotonic()+reg['launch_stagger_seconds']
            while time.monotonic()<until:
                healthy();time.sleep(min(reg['poll_seconds'],max(0,until-time.monotonic())))
        while any(e['process'].poll()is None for e in entries):healthy();time.sleep(reg['poll_seconds'])
        healthy()
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc())
        for e in entries:
            if e['process'].poll()is None:e['process'].terminate()
    finally:
        exits=[]
        for e in entries:
            child=e['process'];forced=False
            try:code=child.wait(timeout=reg['terminate_timeout_seconds'])
            except subprocess.TimeoutExpired:child.kill();forced=True;code=child.wait()
            e['log'].close();value=dict(child_pid=child.pid,job_id=e['job']['job_id'],exit_code=code,actual_child_waited=True,
                forced_termination=forced,elapsed_seconds=time.time()-e['started_unix'])
            save(e['base']/'exit.json',value);exits.append(value)
    success=failure is None and len(exits)==len(jobs)and all(x['exit_code']==0 for x in exits)
    result=dict(phase=phase,status='COMPLETE'if success else'FAILED',actual_children_waited=True,expected_children=len(jobs),
        launched_children=len(entries),worker_exit_codes=[x['exit_code']for x in exits],exits=exits,failure=failure,
        elapsed_seconds=time.time()-started,unlaunched_jobs=[j['job_id']for j in jobs[len(entries):]])
    save(record/(phase+'_exit.json'),result);return result

def run(registration_path):
    rp=Path(registration_path).resolve();reg=read(rp);require(reg['schema']==SCHEMA,'Registered finite cohort required')
    require(reg['owner']==pin(__file__),'Cohort owner changed after registration');r=checked(reg['request']);validate(r)
    require(sha(reg['runner']['path'])==reg['runner']['sha256'],'Registered scientific runner changed')
    require(Path(reg['record_dir']).resolve()==rp.parent and reg['source_bindings']==r['source_bindings'],'Cohort identity or dependencies changed')
    out=Path(r['out']).resolve()
    require(reg['root']==r['root']and reg['output_dir']==str(out)and rp.parent.is_relative_to(out)
        and reg['global_claim']==str(out/'cpu_cohort_launch_claim.json')and reg['ledger_path']==str(out/'packet_ledger.sqlite')
        and reg['deadline_unix']==r['deadline_unix']and reg['stop_files']==r['stop_files'],'One exact output namespace, ledger, deadline and no-repeat claim')
    require(reg['environment']==dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),'Registered CPU-only environment required')
    for p,h in reg['source_bindings'].items():require(sha(p)==h,'Changed sealed dependency: '+p)
    require(reg['workers_per_branch']==4 and reg['total_workers']==8 and reg['packet_cap']==5400,'One finite eight-worker cohort')
    runner=reg['runner']['path'];base=[os.path.abspath(os.path.expanduser(r['python_cpu'])),'-B',runner]
    require(reg['ledger_init']['argv']==base+['ledger-init','--request',reg['request']['path']],'Only registered ledger initialization allowed')
    require([(j['branch'],j['index'])for j in reg['jobs']]==[(b,i)for i in range(4)for b in('raw','entropy')]
        and all(j['argv']==base+['cpu-worker','--request',reg['request']['path'],'--branch',j['branch'],'--index',str(j['index'])]for j in reg['jobs']),
        'Exact separate raw/entropy CPU worker argv required')
    guard(reg);save(reg['global_claim'],dict(registration=pin(rp),owner_pid=os.getpid(),started_unix=time.time(),no_automatic_retry=True))
    started=time.time();save(rp.parent/'owner_started.json',dict(pid=os.getpid(),registration=pin(rp)))
    def stop(*_):
        for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
        raise InterruptedError('Owned cohort interrupted')
    previous={sig:signal.signal(sig,stop)for sig in(signal.SIGINT,signal.SIGTERM)};result=None
    try:
        initialized=children_stage(reg,[reg['ledger_init']],'ledger_init')
        require(initialized['status']=='COMPLETE','Ledger initialization failed; no workers launched')
        ld=read(Path(r['out'])/'ledger_initialized.json')
        require(ld['request_sha256']==reg['request']['sha256']and ld['packet_cap']==5400,'Exact shared ledger admission required')
        cpu=children_stage(reg,reg['jobs'],'workers')
        require(cpu['status']=='COMPLETE','CPU cohort failed; no retry or GPU successor')
        result=dict(status='T6_CONFIRMATION_CPU_COHORT_COMPLETE',exit_code=0,actual_children_waited=True,
            worker_exit_codes=cpu['worker_exit_codes'],ledger_init_exit_codes=initialized['worker_exit_codes'])
    except BaseException as exc:
        result=dict(status='T6_CONFIRMATION_CPU_COHORT_FAILED',exit_code=1,error=repr(exc),traceback=traceback.format_exc(),
            actual_children_waited=True,no_automatic_retry=True)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    result.update(registration=pin(rp),request_sha256=reg['request']['sha256'],elapsed_seconds=time.time()-started,
        GPU_launched=False,automatic_successor=False,shared_packet_cap=5400,ledger_path=reg['ledger_path'])
    result['outputs']={str(p):sha(p)for p in rp.parent.rglob('*.json')}
    save(rp.parent/'exit.json',result);print(json.dumps(result,sort_keys=True),flush=True);return result['exit_code']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--request',required=True);q.add_argument('--record-dir',required=True)
    q=sub.add_parser('run');q.add_argument('--registration',required=True);a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.request,a.record_dir)),flush=True)
    else:raise SystemExit(run(a.registration))
