"""Resource-only recovery after proven pre-model CUDA initialization failure."""
import argparse,copy,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
sys.path.insert(0,str(E))
import a1_owner as original
L=original.L;O=original.O;RUN=L/'a1_cuda_recovery_v1';RO=O/'a1_cuda_recovery_v1';HERE=E/'cuda_recovery_v1'
OLD_SHA='8069e57bd466f91132f330349da645e4fb7225b67b0722343c7c19bdc9f09824'
MAPPING={'visible_index':4,'pci_bus_id':'0000:A1:00.0','cuda_uuid_hex':'e3133767ae42afba35150873255753fc','nvml_uuid':'GPU-535fda57-9774-5902-7375-bcab46fb9895'}
ABSOLUTE_DEADLINE=1792354447.8323371+7500

def guard_failed_origin():
    p=O/'a1_registration.json';assert original.sha(p)==OLD_SHA
    old=json.loads(p.read_text())
    for d in old['source_files']+[old['public'],old['sources'],old['scope']]:original.verify(d)
    assert not (L/'a1_native_v1/tx/calls').exists()
    assert not (L/'a1_native_v1/rx').exists() and not (L/'a1_native_v1/wire_inputs').exists()
    for c,n in [('a1_native_v1','tx.actual_wait.json'),('a1_dispatch_v1','actual_wait.json'),('a1_outer_execution_v1','actual_wait.json')]:
        w=json.loads((L/c/n).read_text());assert type(w['returncode']) is int and w['returncode']==1 and not w['timed_out'] and original.gone(w['process'])
    for c,n in [('a1_outer_execution_v1','supervisor.json'),('a1_dispatch_v1','launch.json'),('a1_native_v1','tx.launch.json')]:
        a=json.loads((L/c/n).read_text())
        assert original.gone(a['process'])
    error=(L/'a1_native_v1/tx.stderr').read_text()
    assert 'torch.cuda.set_device(0)' in error and 'RuntimeError: No CUDA GPUs are available' in error
    return old

def prepare():
    assert not RUN.exists() and not RO.exists()
    old=guard_failed_origin()
    auth=O/'a1_cuda_recovery_preparation_v1/authorization.json'
    a=json.loads(auth.read_text());assert a['status']=='AUTHORIZED_RESOURCE_ONLY_ZERO_SCIENCE_RECOVERY'
    assert a['original_registration_sha256']==OLD_SHA and a['mapping']==MAPPING
    for d in a['evidence']+a['source_files']:original.verify(d)
    assert a['absolute_deadline_unix']==ABSOLUTE_DEADLINE
    for d in old['weights']+old['preparation_receipts']:original.verify(d)
    now=next(x for x in original.gpus() if x['uuid']==MAPPING['nvml_uuid'])
    assert now['index']==MAPPING['visible_index'] and now['free_mib']>=65536 and now['utilization']<=25
    public=json.loads(pathlib.Path(old['public']['path']).read_text());public['run']=str(RUN);public['cuda_admission']=MAPPING
    remaining=ABSOLUTE_DEADLINE-time.time();assert remaining>300
    RUN.mkdir();original.put(RUN/'public.json',public)
    reg=copy.deepcopy(old);reg.update(schema='ARPC_A1_CUDA_RESOURCE_RECOVERY_V1',created_unix=time.time(),
        public=original.desc(RUN/'public.json'),global_wall_seconds=remaining,absolute_deadline_unix=ABSOLUTE_DEADLINE,
        original_registration=original.desc(O/'a1_registration.json'),recovery_authorization=original.desc(auth))
    reg['source_files']+=a['source_files'];reg['preparation_receipts']+=a['evidence']+[original.desc(auth)]
    RO.mkdir();original.put(RO/'a1_registration.json',reg);print(json.dumps(original.desc(RO/'a1_registration.json')))

def adjusted_stage(name,argv,env,seconds,runner,now):
    remain=ABSOLUTE_DEADLINE-now();assert remain>0,'Original A1 deadline expired'
    if name in ['tx','rx']:
        assert argv[2]==str(E/'a1_native.py') and env['CUDA_VISIBLE_DEVICES']==MAPPING['nvml_uuid']
        argv=list(argv);argv[2]=str(HERE/'runtime.py');env=dict(env,CUDA_VISIBLE_DEVICES=str(MAPPING['visible_index']))
    else:assert name=='audit' and argv[2]==str(E/'a1_audit.py') and env['CUDA_VISIBLE_DEVICES']==''
    return runner(name,argv,env,min(seconds,remain))

def run(expected):
    guard_failed_origin()
    p=RO/'a1_registration.json';assert original.sha(p)==expected
    r=json.loads(p.read_text());original.verify(r['recovery_authorization'])
    assert r['absolute_deadline_unix']==ABSOLUTE_DEADLINE and time.time()<ABSOLUTE_DEADLINE
    original.O=RO;original.RUN=RUN
    stage=original.run_stage
    original.run_stage=lambda name,argv,env,seconds:adjusted_stage(name,argv,env,seconds,stage,time.time)
    # Original registration verification, locks, numerical runtime, serial stage
    # ordering, real waits, limits, failure preservation and completion stay intact.
    original.run(expected)

def dispatch(expected):
    d=L/'a1_cuda_recovery_dispatch_v1';assert not d.exists() and not (RUN/'owner.launch.json').exists()
    assert original.sha(RO/'a1_registration.json')==expected
    remaining=ABSOLUTE_DEADLINE-time.time();assert remaining>0
    d.mkdir();argv=[sys.executable,'-B',str(HERE/'recovery.py'),'run','--expected-registration',expected]
    start=time.time();timed=terminated=killed=False
    with (d/'stdout.log').open('xb') as o,(d/'stderr.log').open('xb') as e:
        child=subprocess.Popen(argv,stdout=o,stderr=e,start_new_session=True);ident=original.identity(child.pid)
        original.put(d/'launch.json',{'process':ident,'parent_process':original.identity(os.getpid()),'argv':argv,'started_unix':start,'absolute_science_deadline_unix':ABSOLUTE_DEADLINE,'timeout_seconds':remaining+300})
        try:rc=child.wait(timeout=remaining+300)
        except subprocess.TimeoutExpired:
            timed=terminated=True
            for n in ['tx','rx','audit']:
                p=RUN/(n+'.launch.json')
                if p.exists():
                    a=json.loads(p.read_text())['process']
                    if not original.gone(a):os.killpg(a['pid'],signal.SIGTERM)
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=30)
            except subprocess.TimeoutExpired:killed=True;os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
            for n in ['tx','rx','audit']:
                p=RUN/(n+'.launch.json')
                if p.exists():
                    a=json.loads(p.read_text())['process']
                    if not original.gone(a):killed=True;os.killpg(a['pid'],signal.SIGKILL)
        original.put(d/'actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-start,'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':None})
    if rc!=0 or timed:original.put(d/'failure.json',{'returncode':rc,'timed_out':timed,'retry_authorized':False});raise SystemExit(1)
    assert original.gone(ident)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run','dispatch']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='prepare':prepare()
    elif a.role=='run':run(a.expected_registration)
    else:dispatch(a.expected_registration)
