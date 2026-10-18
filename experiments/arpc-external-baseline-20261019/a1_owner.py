"""Once-only A1 registration and serial TX/RX/audit with real wait receipts."""
import argparse,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
E=R/'experiments/arpc-external-baseline-20261019'
O=R/'results/arpc_external_baseline_20261019_v1'
L=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_external_20261019_v1')
K=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/arpc_external_20261019_v1')
RUN=L/'a1_native_v1'
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def desc(p):
    p=pathlib.Path(p);assert p.is_file() and not p.is_symlink()
    a=p.stat();h=sha(p);b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
    return {'path':str(p),'bytes':a.st_size,'sha256':h}
def verify(d):
    assert desc(d['path'])==d,d['path']
def put(p,d):
    with pathlib.Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def identity(pid):
    t=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'start_ticks':int(t[19]),'host':os.uname().nodename}
def gone(d):
    assert d['host']==os.uname().nodename
    p=pathlib.Path(f"/proc/{d['pid']}/stat")
    return not p.exists() or int(p.read_text().split(') ',1)[1].split()[19])!=d['start_ticks']
def gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    out=[]
    for l in rows.strip().splitlines():
        i,u,n,t,f,v=[x.strip() for x in l.split(',')]
        out.append({'index':int(i),'uuid':u,'name':n,'total_mib':int(t),'free_mib':int(f),'utilization':int(v)})
    return out
def prerequisites():
    for control in ['environment_preparation_v1','asset_acquisition_v3','cpu_qualification_v1','runtime_import_v1']:
        d=json.loads((L/control/'completion.json').read_text())
        assert d['status'] in ['PASS','ENVIRONMENT_ENGINEERING_READY']
        assert not list((L/control).glob('*failure.json')),control
    wait=json.loads((L/'asset_acquisition_v3/actual_wait.json').read_text())
    assert type(wait['returncode']) is int and wait['returncode']==0 and not wait['timed_out'] and gone(wait['process'])
    # Compilation and imports must have genuinely exited before GPU work.
    for name in ['dependencies','flash_build','cpu_import']:
        w=json.loads((L/'environment_preparation_v1'/f'{name}.actual_wait.json').read_text())
        assert type(w['returncode']) is int and w['returncode']==0 and not w['timed_out'] and gone(w['process'])
def prepare():
    prerequisites();assert not RUN.exists() and not (O/'a1_registration.json').exists()
    paths=sorted(p for p in E.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert paths and not any(p.is_symlink() for p in paths)
    assets=json.loads((L/'asset_acquisition_v3/completion.json').read_text())['assets']
    weightpins=[]
    for a in assets:
        d=desc(a['path']);assert d['sha256']==a['actual_sha256'] and d['bytes']==a['bytes'];weightpins.append(d)
    sources=desc(O/'sources.json');scope=desc(O/'execution_scope.json')
    source_data=json.loads(pathlib.Path(sources['path']).read_text());assert len(source_data['a1_first8'])==8
    for s in source_data['a1_first8']:verify(s['archive'])
    candidates=[x for x in gpus() if x['total_mib']>=65536 and x['free_mib']>=65536 and x['utilization']<=25]
    assert candidates,'No currently qualified GPU; preparation has not reserved any scientific call'
    gpu=max(candidates,key=lambda x:(x['free_mib'],-x['index']))
    public={'schema':'ARPC_A1_PUBLIC_V1','run':str(RUN),'output':str(O),'code':str(E),'upstream':str(E/'upstream'),
        'environment':str(L/'environment_v1'),'weights_root':str(K),'resource':{'gpu':gpu,'start_free_bytes':64*1024**3,
        'running_free_bytes':8*1024**3,'allocator_max_bytes':70*1024**3,'threads':4,'interop_threads':2},
        'preset':'0.06M','input_shape':[256,256],'endpoints':[1,4,7],'seed':0,'cfg':3,'tau':0.5,
        'native_float_tolerance':1e-6,'required_exact_uint8':True,'wireless_frames':0}
    # All data inside public is image-independent. Source IDs/CDFs/TX tensors are absent.
    RUN.mkdir();put(RUN/'public.json',public)
    registration={'schema':'ARPC_A1_REGISTRATION_V1','created_unix':time.time(),'host':os.uname().nodename,
        'repository_head':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
        'source_files':[desc(p) for p in paths],'weights':weightpins,'sources':sources,'scope':scope,'public':desc(RUN/'public.json'),
        'preparation_receipts':[desc(L/c/'completion.json') for c in ['asset_acquisition_v3','environment_preparation_v1','cpu_qualification_v1','runtime_import_v1']],
        'limits':{'caption':8,'tokenizer':8,'tx_probability':8,'prefix_roundtrip':24,'reference_images':24,'rx_images':24,'wireless_frames':0,'training_updates':0},
        'stage_wall_seconds':{'tx':3600,'rx':3600,'audit':300},'global_wall_seconds':7500,
        'automatic_successor':'A1 audit only; root visual/resource review is required before A2',
        'failure_policy':'Preserve all attempts; no automatic retry or completed-call replay.'}
    put(O/'a1_registration.json',registration);print(json.dumps(desc(O/'a1_registration.json')))
def run_stage(name,argv,env,seconds):
    start=time.time();timed=terminated=killed=False;metadata_error=None
    with (RUN/f'{name}.stdout').open('xb') as o,(RUN/f'{name}.stderr').open('xb') as e:
        p=subprocess.Popen(argv,env=env,stdout=o,stderr=e,start_new_session=True)
        try:ident=identity(p.pid)
        except Exception:
            metadata_error=traceback.format_exc();p.terminate();p.wait();raise
        put(RUN/f'{name}.launch.json',{'process':ident,'parent_process':identity(os.getpid()),'argv':argv,'started_unix':start,'timeout_seconds':seconds})
        try:rc=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed=terminated=True;os.killpg(p.pid,signal.SIGTERM)
            try:rc=p.wait(timeout=30)
            except subprocess.TimeoutExpired:killed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
        put(RUN/f'{name}.actual_wait.json',{'process':ident,'returncode':rc,'seconds':time.time()-start,
            'timed_out':timed,'terminated':terminated,'killed':killed,'metadata_error':metadata_error})
        assert type(rc) is int and rc==0 and not timed and gone(ident),f'{name} failed'
def run(expected):
    registration=O/'a1_registration.json';assert sha(registration)==expected
    r=json.loads(registration.read_text());assert r['host']==os.uname().nodename
    for d in r['source_files']+r['weights']+r['preparation_receipts']+[r['sources'],r['scope'],r['public']]:verify(d)
    assert not (RUN/'owner.launch.json').exists() and not (RUN/'failure.json').exists()
    cfg=json.loads(pathlib.Path(r['public']['path']).read_text());gpu=cfg['resource']['gpu']
    now=next(x for x in gpus() if x['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time(),'gpu_snapshot':now})
    started=time.monotonic()
    env=dict(os.environ,PYTHONPATH=str(L/'environment_v1')+':'+str(E)+':'+str(E/'upstream'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    try:
        for name in ['tx','rx','audit']:
            if name=='audit':argv=[sys.executable,'-B',str(E/'a1_audit.py'),str(RUN/'public.json')];stage_env=dict(env,CUDA_VISIBLE_DEVICES='')
            else:
                argv=[sys.executable,'-B',str(E/'a1_native.py'),name,'--public',str(RUN/'public.json')]
                if name=='tx':argv+=['--sources',r['sources']['path']]
                stage_env=env
            remain=r['global_wall_seconds']-(time.monotonic()-started);assert remain>0
            run_stage(name,argv,stage_env,min(r['stage_wall_seconds'][name],remain))
        put(RUN/'completion.json',{'status':'A1_ROUNDTRIPS_DONE_REVIEW_PENDING','seconds':time.monotonic()-started,'registration_sha256':expected,'ended_unix':time.time()})
    except BaseException:
        put(RUN/'failure.json',{'error':traceback.format_exc(),'ended_unix':time.time(),'registration_sha256':expected});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='prepare':prepare()
    else:assert a.expected_registration;run(a.expected_registration)
if __name__=='__main__':main()
