"""Score the closed actual cal1000 receiver, then freeze without test-set access."""
import argparse,csv,fcntl,hashlib,json,os,pathlib,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone,gpus
CODE=E/'a2b_metrics_v1';RUN=L/'a2b_metrics_v1';OUT=O/'a2b_metrics_v1';REG=OUT/'registration.json';QUAL=L/'a2b_metrics_qualification_v1'
PY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
def success(p):
    a=json.loads(p.read_text());assert type(a['returncode']) is int and a['returncode']==0 and gone(a['process']) and not any(a.get(k) for k in ['timed_out','terminated','killed','metadata_error']);return a
def qualify():
    assert not QUAL.exists();QUAL.mkdir();old.RUN=QUAL
    code="""import importlib.abc,sys,unittest,json
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','transformers','infinity'}:
   blocked.append(fullname);raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard());r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.',pattern='test_metrics.py'))
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==6 and not blocked
"""
    old.run_stage('tests',[str(PY),'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':6,'science':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def prepare():
    import numpy as np
    assert not RUN.exists() and not OUT.exists();success(QUAL/'tests.actual_wait.json');q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    rx=O/'a2b_rx_v1';review=rx/'root_closure_review.json';closed=json.loads(review.read_text());assert closed['status']=='ACTUAL_RX_CLOSED' and closed['all_processes_gone'] and closed['unresolved']==0
    for d in closed['controls']+closed['outputs']:verify(d)
    rows=json.loads((rx/'logical_receivers.json').read_text());assert len(rows)==18000
    sources=json.loads((O/'a2b_source_v1/sources.json').read_text())['full_sources'];assert len(sources)==1000
    grayhash=hashlib.sha256(np.full((3,256,256),.5,dtype=np.float32).tobytes()).hexdigest();groups={s['source_id']:{} for s in sources}
    fields=['key','source_id','source_index','snr_db','candidate_id','noise_seed','target_endpoint','endpoint','actual_status','receiver_status']
    for row in rows:
        sha=row['image_float_sha256'] if row['image'] is not None else grayhash;group=groups[row['source_id']]
        if sha not in group:group[sha]={'state':sha,'image':row['image'],'image_float_sha256':sha,'frames':[]}
        group[sha]['frames'].append({k:row[k] for k in fields})
    plan={'schema':'ARPC_ACTUAL_CALIBRATION_PAIRS_V1','sources':[]}
    for s in sorted(sources,key=lambda s:s['source_index']):
        verify(s['archive']);items=list(groups[s['source_id']].values());assert sum(len(x['frames']) for x in items)==18
        for item in items:
            if item['image'] is not None:verify(item['image'])
        plan['sources'].append({'source_id':s['source_id'],'source_index':s['source_index'],'archive':s['archive'],'pairs':items})
    assert sum(len(s['pairs']) for s in plan['sources'])<=8000
    prior=O/'a2_metrics_v1';priorreg=json.loads((prior/'registration.json').read_text());pc=json.loads((prior/'completion.json').read_text());assert pc['unresolved']==0 and pc['unique_pairs']==800
    success(L/'a2_metrics_v1/score.actual_wait.json');success(L/'a2_metrics_dispatch_v1/actual_wait.json')
    assert gone(json.loads((L/'a2_metrics_dispatch_v1/supervisor.json').read_text())['process'])
    for d in priorreg['source_files']+pc['outputs']:verify(d)
    priorfiles=[]
    with (prior/'clean_quality.csv').open() as f:oldrows=list(csv.DictReader(f))
    for key in sorted({r['pair_key'] for r in oldrows}):
        pin=desc(L/'a2_metrics_v1'/(key+'.json'));r=json.loads(pathlib.Path(pin['path']).read_text());assert r['pair_key']==key and r['metric_identity']==pc['metric_identity'];priorfiles.append({'pair_key':key,'file':pin})
    refs=[]
    for s in json.loads((O/'sources.json').read_text())['cal100']:refs.append({'source_id':s['source_id'],'source':s['archive'],'file':desc(L/'a2_metrics_v1'/f"{s['rank']:02d}_reference.npz")})
    window=L/'a2b_source_v1/shared_window.json';w=json.loads(window.read_text());assert time.time()<w['absolute_deadline_unix'];verify(w['budget'])
    resource=priorreg['resource'];gpu=next(g for g in gpus() if g['uuid']==resource['gpu']['uuid']);assert gpu['free_mib']>=8192 and gpu['utilization']<=25
    RUN.mkdir();OUT.mkdir();put(OUT/'pair_plan.json',plan)
    inputs=[desc(p) for p in [review,rx/'completion.json',rx/'logical_receivers.json',O/'a2b_source_v1/sources.json',O/'a2_screen_v1/nomination.json',prior/'registration.json',prior/'completion.json',prior/'metric_identity.json',prior/'clean_quality.csv',QUAL/'completion.json',QUAL/'tests.actual_wait.json',window,OUT/'pair_plan.json']]
    reg={'schema':'ARPC_A2B_CAL1000_METRICS_REGISTRATION_V1','host':os.uname().nodename,'code_root':str(E),'created_unix':time.time(),
      'source_files':priorreg['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],'inputs':inputs,'original_score':desc(E/'a2_metrics_v1/score.py'),
      'metric_config':priorreg['metric_config'],'assets':priorreg['assets'],'resource':resource,'pair_plan':desc(OUT/'pair_plan.json'),
      'prior_metadata':desc(prior/'metric_identity.json'),'prior_metric_identity':pc['metric_identity'],'prior_pair_files':priorfiles,'prior_references':refs,
      'nomination':desc(O/'a2_screen_v1/nomination.json'),'absolute_deadline_unix':w['absolute_deadline_unix'],'shared_budget':w['budget'],
      'limits':{'new_source_references':900,'new_pairs':7200,'source_population':1000,'logical_frames':18000,'new_PHY':0,'ARPC':0,'bootstrap':0},
      'stage_seconds':{'score':2400,'freeze':300},'phase_wall_seconds':2700,'selection':'DINO source-first noise means only; exact tie lexicographic candidateID.'}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inputs']:verify(d)
    for d in reg['prior_pair_files']:verify(d['file'])
    for d in reg['prior_references']:verify(d['file'])
    assert not (RUN/'owner.launch.json').exists();gpu=reg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=8192 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();assert start<reg['absolute_deadline_unix'];put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'gpu':now,'started_unix':start});old.RUN=RUN
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');env.pop('PYTHONPATH',None)
    try:
        for role in ['score','freeze']:
            if role=='freeze':success(RUN/'score.actual_wait.json')
            remain=min(reg['phase_wall_seconds']-(time.time()-start),reg['absolute_deadline_unix']-time.time());assert remain>0
            argv=[str(PY),'-B',str(CODE/'worker.py'),'--registration',str(REG),'--sha256',expected] if role=='score' else [str(PY),'-B',str(CODE/'freeze.py'),'--expected-registration',expected]
            old.run_stage(role,argv,env if role=='score' else dict(env,CUDA_VISIBLE_DEVICES=''),min(remain,reg['stage_seconds'][role]))
        put(RUN/'completion.json',{'status':'CALIBRATION_POLICY_FROZEN','output':desc(OUT/'frozen_policies.json'),'seconds':time.time()-start})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
