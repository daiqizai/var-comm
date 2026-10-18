"""One finite A2a TX92/RX676/audit run; no metric or channel successor."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,desc,verify,put,identity,gone,gpus
CODE=E/'a2_clean_v1';BASE=old.O;O=BASE/'a2_clean_v1';RUN=L/'a2_clean_v1';A1=L/'a1_cuda_recovery_v1'
REG=O/'registration.json'
REVIEW_SHA='f9d6f73b28d2ffe6de3d27a6c6e77436f0168f0fb658ef052912d133a055c6a8'
A1_REG_SHA='3cda84a89fa42e65eeadf40218feba6b24d9a7390fc1547cdcd07266bcdc98ec'
def success(path):
    w=json.loads(path.read_text());assert type(w['returncode']) is int and w['returncode']==0
    assert not w['timed_out'] and gone(w['process']);return w
def prior_closed():
    review=desc(BASE/'actual_visual_resource_review.json');assert review['sha256']==REVIEW_SHA
    assert json.loads(pathlib.Path(review['path']).read_text())['status']=='READY_FOR_CALIBRATION'
    reg=BASE/'a1_cuda_recovery_v1/a1_registration.json';assert desc(reg)['sha256']==A1_REG_SHA
    for n in ['tx','rx','audit']:success(A1/f'{n}.actual_wait.json')
    success(L/'a1_cuda_recovery_dispatch_v1/actual_wait.json')
    for p in [A1/'owner.launch.json',L/'a1_cuda_recovery_dispatch_v1/launch.json']:
        a=json.loads(p.read_text());assert gone(a['process'])
        if 'parent_process' in a:assert gone(a['parent_process'])
    assert not (A1/'failure.json').exists()
    return json.loads(reg.read_text())
def prepare():
    previous=prior_closed();assert not O.exists() and not RUN.exists()
    q=L/'a2_clean_qualification_v1';success(q/'actual_wait.json')
    qualification=json.loads((q/'completion.json').read_text());assert qualification['status']=='PASS' and qualification['scientific_calls']==0
    for p in previous['source_files']+previous['weights']:verify(p)
    sources=desc(BASE/'sources.json');a=json.loads(pathlib.Path(sources['path']).read_text())
    assert len(a['cal100'])==100 and a['cal100'][:8]==a['a1_first8']
    assert [s['rank'] for s in a['cal100']]==list(range(100))
    for s in a['cal100']:verify(s['archive'])
    public=json.loads((A1/'public.json').read_text())
    gpu=public['resource']['gpu'];current=next(g for g in gpus() if g['uuid']==gpu['uuid'])
    assert current['free_mib']>=65536 and current['utilization']<=25
    public.update(schema='ARPC_A2_CLEAN_PUBLIC_V1',run=str(RUN),output=str(O),endpoints=list(range(1,8)))
    # No source/CDF/model truth or source-result map enters public receiver metadata.
    RUN.mkdir();O.mkdir();put(RUN/'public.json',public)
    bindings=previous['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]
    prior=[desc(BASE/n) for n in ['actual_visual_resource_review.json','a1_closure.json','a1_closed_files.json','codec_roundtrip.csv']]
    prior += [desc(BASE/'a1_cuda_recovery_v1/a1_registration.json'),desc(A1/'tx/completion.json'),desc(A1/'rx/completion.json')]
    prior += [desc(A1/f'{n}.actual_wait.json') for n in ['tx','rx','audit']]
    prior += [desc(L/'a1_cuda_recovery_dispatch_v1/actual_wait.json')]
    reg={'schema':'ARPC_A2_CLEAN_REGISTRATION_V1','created_unix':time.time(),'host':os.uname().nodename,
      'repository_head':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
      'source_files':bindings,'weights':previous['weights'],'sources':sources,'public':desc(RUN/'public.json'),'prior':prior,
      'qualification':[desc(q/'actual_wait.json'),desc(q/'completion.json')],
      'limits':{'total_sources':100,'reused_source_passes':8,'new_caption':92,'new_tokenizer':92,'new_tx_probability':92,
        'total_clean_states':700,'reused_clean_states':24,'new_rx_prefix':676,'new_rx_render':676,'new_reference_render':0,
        'metric_calls':0,'wireless_frames':0,'training_updates':0},
      'stage_wall_seconds':{'tx':1200,'rx':3600,'audit':600},'global_wall_seconds':5400,
      'resource_basis':'A1 TX80.69s/8 sources and RX53.90s/24 images including model loading; conservative finite 90-minute cap.',
      'successor':'None. Separate frozen-metric scoring and reliability screening required before A2b.',
      'failure_policy':'Retain all failures and successful calls; no automatic retry.'}
    put(REG,reg);print(json.dumps(desc(REG)))
def assemble_jobs():
    """Source-independent RX jobs: actual streams and paid endpoints only."""
    wires=RUN/'wire_inputs';jobs=json.loads((wires/'new_jobs.json').read_text());assert len(jobs)==644
    assert {j['key'] for j in jobs}=={f'{i:02d}_e{e}' for i in range(8,100) for e in range(1,8)}
    closed=json.loads((BASE/'a1_closed_files.json').read_text());pins={d['path']:d for d in closed['files']}
    inherited=[]
    for i in range(8):
        for e in [2,3,5,6]:
            key=f'{i:02d}_e{e}';source=A1/'tx'/(key+'.arc');d=pins[str(source)];verify(d)
            dest=wires/(key+'.arc');data=source.read_bytes()
            with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            assert desc(dest)['sha256']==d['sha256']
            jobs.append({'key':key,'path':str(dest),'paid_profile':e,'generation_seed':0})
            inherited.append({'source':d,'destination':desc(dest)})
    jobs.sort(key=lambda x:x['key']);assert len(jobs)==676
    assert len({j['key'] for j in jobs})==676
    put(wires/'jobs.json',jobs);put(RUN/'inherited_streams.json',inherited)
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    prior_closed()
    for d in reg['source_files']+reg['weights']+reg['prior']+reg['qualification']+[reg['sources'],reg['public']]:verify(d)
    assert not (RUN/'owner.launch.json').exists() and not (RUN/'failure.json').exists()
    cfg=json.loads((RUN/'public.json').read_text());gpu=cfg['resource']['gpu']
    current=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert current['free_mib']>=65536 and current['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time(),'gpu_snapshot':current})
    old.RUN=RUN;start=time.monotonic()
    env=dict(os.environ,PYTHONPATH=str(L/'environment_v1')+':'+str(E),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
      CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    try:
        for name in ['tx','rx','audit']:
            if name=='rx':assemble_jobs()
            if name=='audit':argv=[sys.executable,'-B',str(CODE/'audit.py'),'--registration',str(REG),'--expected-registration',expected];stage_env=dict(env,CUDA_VISIBLE_DEVICES='')
            else:
                argv=[sys.executable,'-B',str(CODE/'native.py'),name,'--public',str(RUN/'public.json')];stage_env=env
                if name=='tx':argv+=['--sources',reg['sources']['path']]
            remain=reg['global_wall_seconds']-(time.monotonic()-start);assert remain>0
            old.run_stage(name,argv,stage_env,min(reg['stage_wall_seconds'][name],remain))
        put(RUN/'completion.json',{'status':'A2_CLEAN_CLOSED_AWAITING_SEPARATE_METRICS','registration_sha256':expected,'seconds':time.monotonic()-start,'ended_unix':time.time()})
    except BaseException:
        put(RUN/'failure.json',{'error':traceback.format_exc(),'registration_sha256':expected,'ended_unix':time.time()});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='prepare':prepare()
    else:assert a.expected_registration;run(a.expected_registration)
if __name__=='__main__':main()
