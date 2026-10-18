"""Gated fixed external_common100 source coding, finite evaluation budget."""
import argparse,fcntl,json,os,pathlib,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone,gpus
from population import evaluate_sources,policies
CODE=E/'a3_source_v1';RUN=L/'a3_source_v1';OUT=O/'a3_source_v1';REG=OUT/'registration.json';QUAL=L/'a3_source_qualification_v1'
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
sys.meta_path.insert(0,Guard());r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.',pattern='test_source.py'))
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==4 and not blocked
"""
    old.run_stage('tests',[sys.executable,'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':4,'science':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def prepare():
    assert not RUN.exists() and not OUT.exists();success(QUAL/'tests.actual_wait.json');q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    cal=O/'a2b_metrics_v1';review=cal/'root_closure_review.json';c=json.loads(review.read_text());assert c['status']=='CAL1000_FROZEN_CLOSED' and c['all_processes_gone'] and c['unresolved']==0
    for d in c['controls']+c['outputs']:verify(d)
    frozen=cal/'frozen_policies.json';choices=policies(json.loads(frozen.read_text()))
    public_sources=O/'sources.json';assert desc(public_sources)['sha256']=='4a6ca6950158529bedff8524345b69cde88248499920e7b06a70c4db85e87f5d'
    data=json.loads(public_sources.read_text());cal_sources=O/'a2b_source_v1/sources.json'
    sources=evaluate_sources(data['external_common100_v1'],json.loads(cal_sources.read_text())['full_sources'])
    for s in sources:verify(s['archive'])
    oldreg=json.loads((O/'a2b_source_v1/registration.json').read_text())
    for d in oldreg['source_files']+oldreg['weights']:verify(d)
    cfg=json.loads((L/'a2b_source_v1/public.json').read_text());cfg.update(schema='ARPC_A3_FIXED_EVALUATION_PUBLIC_V1',run=str(RUN),output=str(OUT))
    gpu=cfg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    original_eval=R/'results/entropy_main_common500_20261013/registration_v4.json';assert desc(original_eval)['sha256']=='2551cd1492e4f8354e8ee882a189b9fd498786f46868ab1b20a5638bf5791956'
    noise=json.loads(original_eval.read_text());assert noise['noise_seeds']==[6201,6202,6203] and noise['N']==1024
    RUN.mkdir();OUT.mkdir();put(RUN/'public.json',cfg);put(OUT/'sources.json',{'new_sources':sources,'population':'external_common100_v1','calibration_overlap':0,'selection_uses_metrics':False})
    budget={'schema':'ARPC_A3_FINITE_BUDGET_V1','sources':100,'caption_tokenizer_TX_each':100,'new_reference_renders':0,'logical_frames':900,'header_decodes':900,'body_decodes':900,
      'new_unique_RX_state_cap':700,'unique_score_pair_cap':800,'source_metric_references_cap':100,'seven_metrics':True,'noise_seeds':[6201,6202,6203],'SNRs':[4,10,19],
      'frozen_policies':choices,'window_seconds':7200,'basis':'Closed A1/A2 measured native and PHY cost, at most100TX+700RX+800sevenmetric pairs; finite conservative7200s window, not timing comparison.',
      'bootstrap_replicates':10000,'bootstrap_seed':2026100701,'bootstrap_draw_matrix_calls':1,'comparison':'Source-paired NeST-Com minus ARPC only; five same-source descriptive means, pointwise intervals.',
      'training':0,'policy_or_checkpoint_selection':False,'no_additional_calibration':True}
    put(OUT/'resource_budget.json',budget)
    put(OUT/'frozen_nomination.json',{'decisions':{s:{'proposed_candidates':[cid]} for s,cid in choices.items()},'frozen_calibration':desc(frozen),'selection_changed':False})
    inputs=[desc(p) for p in [review,frozen,public_sources,cal_sources,original_eval,O/'a2b_source_v1/registration.json',L/'a2b_source_v1/public.json',QUAL/'completion.json',QUAL/'tests.actual_wait.json',RUN/'public.json',OUT/'sources.json',OUT/'resource_budget.json',OUT/'frozen_nomination.json']]
    reg={'schema':'ARPC_A3_SOURCE100_REGISTRATION_V1','host':os.uname().nodename,'created_unix':time.time(),'source_files':oldreg['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],
      'weights':oldreg['weights'],'inputs':inputs,'public':desc(RUN/'public.json'),'sources':desc(OUT/'sources.json'),'frozen_calibration':desc(frozen),'shared_budget':desc(OUT/'resource_budget.json'),
      'limits':{'caption':100,'tokenizer':100,'TX_probability':100,'reference_images':0,'wireless':0,'metrics':0},'stage_seconds':{'tx':900,'audit':120},'phase_wall_seconds':1200}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['weights']+reg['inputs']:verify(d)
    assert not (RUN/'owner.launch.json').exists();cfg=json.loads((RUN/'public.json').read_text());gpu=cfg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();put(RUN/'shared_window.json',{'started_unix':start,'absolute_deadline_unix':start+7200,'budget':reg['shared_budget']})
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'gpu':now,'started_unix':start});old.RUN=RUN
    env=dict(os.environ,PYTHONPATH=str(L/'environment_v1')+':'+str(E),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    try:
        old.run_stage('tx',[sys.executable,'-B',str(CODE/'native.py'),'tx','--public',str(RUN/'public.json'),'--sources',reg['sources']['path']],env,900)
        old.run_stage('audit',[sys.executable,'-B',str(CODE/'audit.py'),'--expected-registration',expected],dict(env,CUDA_VISIBLE_DEVICES=''),120)
        put(RUN/'completion.json',{'status':'EVAL100_SOURCE_READY','output':desc(OUT/'completion.json'),'seconds':time.time()-start})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
