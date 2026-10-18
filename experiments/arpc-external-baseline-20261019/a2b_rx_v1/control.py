"""Bind actual source inputs/cache after PHY closure; inherit A2b finite budget."""
import argparse,fcntl,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone,gpus
from cache import match
CODE=E/'a2b_rx_v1';RUN=L/'a2b_rx_v1';OUT=O/'a2b_rx_v1';REG=OUT/'registration.json';QUAL=L/'a2b_rx_qualification_v1'
def success(p):
    a=json.loads(p.read_text());assert type(a['returncode']) is int and a['returncode']==0 and not any(a.get(k) for k in ['timed_out','terminated','killed','metadata_error']) and gone(a['process']);return a
def qualify():
    assert not QUAL.exists();QUAL.mkdir();old.RUN=QUAL
    code="""import importlib.abc,sys,unittest,json
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','transformers','infinity'}:
   blocked.append(fullname);raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard());r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.',pattern='test_rx.py'))
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==8 and not blocked
"""
    old.run_stage('tests',[sys.executable,'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':8,'science':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def prepare():
    assert not RUN.exists() and not OUT.exists();success(QUAL/'tests.actual_wait.json')
    q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    phy=O/'a2b_link_v1';review=phy/'root_closure_review.json';c=json.loads(review.read_text())
    assert c['status']=='PHY18000_CLOSED' and c['all_processes_gone'] and c['unresolved']==0
    for d in c['controls']+c['outputs']:verify(d)
    pc=json.loads((phy/'completion.json').read_text());assert pc['logical_frames']==18000 and pc['unresolved']==0
    for d in pc['outputs']:verify(d)
    jobsfile=L/'a2b_link_v1/rx/actual_source_jobs.json';jobs=json.loads(jobsfile.read_text())
    oldstates=O/'a2_clean_v1/clean_states.json';states=json.loads(oldstates.read_text())['states'];assert len(states)==700
    cleanreg=json.loads((O/'a2_clean_v1/registration.json').read_text())
    for d in cleanreg['source_files']+cleanreg['weights']:verify(d)
    prior=O/'a2_clean_v1/root_closure_review.json';assert desc(prior)['sha256']=='7b3b3ebdf3a8f9be7e2f1236621e16dff9826f63423b17a62af4cc6b6c68029c'
    original=json.loads((L/'a2_clean_v1/public.json').read_text());cfg=dict(original)
    for j in jobs:verify(j['input']);assert j['path']==j['input']['path']
    for s in states:verify(s['wire']);verify(s['image'])
    reused,new=match(jobs,states)
    window=L/'a2b_source_v1/shared_window.json';w=json.loads(window.read_text());assert time.time()<w['absolute_deadline_unix']
    budget=O/'a2b_source_v1/resource_budget.json';verify(w['budget']);assert w['budget']==desc(budget)
    cfg.update(run=str(RUN),output=str(OUT),schema='ARPC_A2B_ACTUAL_SOURCE_RX_V1',new_job_count=len(new))
    assert {k:v for k,v in cfg.items() if k not in ['run','output','schema','new_job_count']}=={k:v for k,v in original.items() if k not in ['run','output','schema','new_job_count']}
    gpu=cfg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    RUN.mkdir();OUT.mkdir();wires=RUN/'wire_inputs';wires.mkdir();received=[]
    for j in new:
        p=wires/(j['key']+'.arc');raw=pathlib.Path(j['path']).read_bytes()
        with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        pin=desc(p);assert all(pin[k]==j['input'][k] for k in ['bytes','sha256'])
        received.append({'key':j['key'],'path':str(p),'input':pin,'paid_profile':j['paid_profile'],'generation_seed':0})
    put(wires/'jobs.json',received);put(RUN/'public.json',cfg);put(OUT/'reused_states.json',reused)
    inputs=[desc(p) for p in [review,phy/'registration.json',phy/'completion.json',phy/'logical_frames.json',jobsfile,oldstates,prior,O/'a2_clean_v1/registration.json',L/'a2_clean_v1/public.json',window,budget,QUAL/'completion.json',QUAL/'tests.actual_wait.json',wires/'jobs.json',RUN/'public.json',OUT/'reused_states.json']]
    reg={'schema':'ARPC_A2B_ACTUAL_SOURCE_RX_REGISTRATION_V1','host':os.uname().nodename,'created_unix':time.time(),
      'source_files':cleanreg['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],'weights':cleanreg['weights'],'inputs':inputs,
      'public':desc(RUN/'public.json'),'new_jobs':len(new),'reused_jobs':len(reused),'actual_unique_jobs':len(jobs),'absolute_deadline_unix':w['absolute_deadline_unix'],
      'new_rx_state_cap':6300,'phase_wall_seconds':6600,'stage_seconds':{'rx':6300,'audit':300},
      'cache_identity':'Exact full wire bytes/hash, paid endpoint, fixed seed0, same original native engine, full public numerical/model settings and all weight/source pins; independent RX images only.',
      'failure_policy':'Native canonical SourceParseError yields gray after actual decode; unknown error retains failure. No TX truth or source-ID lookup at RX.'}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['weights']+reg['inputs']:verify(d)
    assert not (RUN/'owner.launch.json').exists();cfg=json.loads((RUN/'public.json').read_text());gpu=cfg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();assert start<reg['absolute_deadline_unix'];put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'gpu':now,'started_unix':start});old.RUN=RUN
    env=dict(os.environ,PYTHONPATH=str(L/'environment_v1')+':'+str(E),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    try:
        for role in ['rx','audit']:
            if role=='audit':success(RUN/'rx.actual_wait.json')
            remain=min(reg['phase_wall_seconds']-(time.time()-start),reg['absolute_deadline_unix']-time.time());assert remain>0
            argv=[sys.executable,'-B',str(CODE/'native.py'),'rx','--public',str(RUN/'public.json')] if role=='rx' else [sys.executable,'-B',str(CODE/'audit.py'),'--expected-registration',expected]
            old.run_stage(role,argv,env if role=='rx' else dict(env,CUDA_VISIBLE_DEVICES=''),min(remain,reg['stage_seconds'][role]))
        put(RUN/'completion.json',{'status':'ACTUAL_SOURCE_RX_CLOSED_PENDING_METRICS','output':desc(OUT/'completion.json'),'seconds':time.time()-start})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
