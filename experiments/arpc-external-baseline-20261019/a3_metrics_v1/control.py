"""Gated scoring and one final paired analysis under the inherited A3 deadline."""
import argparse,csv,fcntl,json,os,pathlib,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone,gpus
CODE=E/'a3_metrics_v1';RUN=L/'a3_metrics_v1';OUT=O/'a3_metrics_v1';REG=OUT/'registration.json';QUAL=L/'a3_metrics_qualification_v1'
PY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
BASELINE_PINS={
 'main':{'path':str(R/'results/entropy_main_common500_20261013/analysis_v1/source_means.csv'),'bytes':9219976,'sha256':'f905169eef0f9e63863e5740c94f4c75631cbb006a874978a0862e523b534fc3'},
 'supplement':{'path':str(R/'results/quality_metrics_supplement_20261014_v1/common500/source_noise_means.csv'),'bytes':1417402,'sha256':'f8fb6e16754d25cd0fb801359dfe534f591df258f5008f9f0136db044a0702da'},
 't1c':{'path':str(R/'results/t1c_m10_test_20261016_v1/delivery/source_means_19dB.csv'),'bytes':618329,'sha256':'5f8ec2824dd34d97e35e8870e98c206bd264d9a7afdaae9c1f92ec7e190ecde1'}}
def success(p):
    x=json.loads(p.read_text());assert type(x['returncode']) is int and x['returncode']==0 and gone(x['process']) and not any(x.get(k) for k in ['timed_out','terminated','killed','metadata_error']);return x
def qualify():
    assert not QUAL.exists();QUAL.mkdir();(QUAL/'fixtures').mkdir();old.RUN=QUAL
    code="""import importlib.abc,sys,unittest,json
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','transformers','infinity'}:
   blocked.append(fullname);raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard());r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.',pattern='test_metrics.py'))
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==8 and not blocked
"""
    old.run_stage('tests',[str(PY),'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',ARPC_SYNTHETIC_ROOT=str(QUAL/'fixtures')),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':8,'science':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def prepare():
    import hashlib,numpy as np
    from data import group,baseline_rows
    assert not RUN.exists() and not OUT.exists();success(QUAL/'tests.actual_wait.json');q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for pin in q['sources']:verify(pin)
    rx=O/'a3_rx_v1';review=rx/'root_closure_review.json';closed=json.loads(review.read_text());assert closed['status']=='EVAL_ACTUAL_RX_CLOSED' and closed['all_processes_gone'] and closed['unresolved']==0
    for pin in closed['controls']+closed['outputs']:verify(pin)
    rows=json.loads((rx/'logical_receivers.json').read_text());sources=json.loads((O/'a3_source_v1/sources.json').read_text())['new_sources']
    gray=hashlib.sha256(np.full((3,256,256),.5,dtype=np.float32).tobytes()).hexdigest();plan={'sources':group(rows,sources,gray)}
    for s in plan['sources']:
        verify(s['archive'])
        for pair in s['pairs']:
            if pair['image']:verify(pair['image'])
    for pin in BASELINE_PINS.values():verify(pin)
    baseline=baseline_rows({k:v['path'] for k,v in BASELINE_PINS.items()},sources)
    prior=O/'a2_metrics_v1/registration.json';pr=json.loads(prior.read_text())
    for pin in pr['source_files']:verify(pin)
    original=R/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py'
    from metric_adapter import SHA
    assert desc(original)['sha256']==SHA
    supplemental=R/'results/quality_metrics_supplement_20261014_v1/common500/registration.json';sr=json.loads(supplemental.read_text());assert desc(supplemental)['sha256']=='191f10cce3ac20000583f70edfbe74e83ec0d22272e403ea0b683eed484cfb02'
    for k in ['loader','metric_assets','dependency_overlay']:verify(sr[k])
    assert sr['loader']['sha256']=='dbc6f6c78c92ffd8a6e914d79008bbee8fc8b485a776884de45c6bdce1703066'
    assert sr['metric_assets']['sha256']=='1dc00388cc91e86ffb3296e794ea23229d945a25e5a2a47619cf13c6b8c02362'
    assert sr['dependency_overlay']['sha256']=='c3372713ca0237ade062a3a286b9ce7a531858d607f247a97a3d793e8c9bee4d'
    window=L/'a3_source_v1/shared_window.json';w=json.loads(window.read_text());verify(w['budget']);assert time.time()<w['absolute_deadline_unix']
    resource=pr['resource'];gpu=next(g for g in gpus() if g['uuid']==resource['gpu']['uuid']);assert gpu['free_mib']>=8192 and gpu['utilization']<=25
    RUN.mkdir();OUT.mkdir();put(OUT/'pair_plan.json',plan)
    with (OUT/'baseline_source_means.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(baseline[0]));writer.writeheader();writer.writerows(baseline);f.flush();os.fsync(f.fileno())
    inputs=[desc(p) for p in [review,rx/'completion.json',rx/'logical_receivers.json',O/'a3_source_v1/sources.json',O/'a3_source_v1/resource_budget.json',prior,original,supplemental,window,QUAL/'completion.json',QUAL/'tests.actual_wait.json',OUT/'pair_plan.json',OUT/'baseline_source_means.csv']]+list(BASELINE_PINS.values())+[sr[k] for k in ['loader','metric_assets','dependency_overlay']]
    reg={'schema':'ARPC_A3_SEVEN_METRICS_FINAL_STATISTICS_V1','host':os.uname().nodename,'created_unix':time.time(),
      'source_files':pr['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],'inputs':inputs,'original_four_metrics':desc(original),
      'metric_config':pr['metric_config'],'resource':resource,'actual_metric_resource':{'startup_free_mib':8192,'running_free_mib':2048,'allocator_gib':12,'numeric_flags':'original6/2 FP32 TF32off, four and supplement models sequential'},
      'pair_plan':desc(OUT/'pair_plan.json'),'baseline_source_means':desc(OUT/'baseline_source_means.csv'),'supplemental_registration':desc(supplemental),
      'supplemental_overlay_root':str(L.parent/'baseline_strength_20261011/quality_metrics_supplement_20261014_v1/dependency_overlay_v1/sitepackages'),
      'absolute_deadline_unix':w['absolute_deadline_unix'],'shared_budget':w['budget'],'limits':{'source_references':100,'new_pairs':800,'logical_frames':900,'new_baseline_scores':0,'training':0,'bootstrap_draws':1},
      'stage_seconds':{'score':2400,'statistics':300},'phase_wall_seconds':2700,'selection':False,'bootstrap':{'replicates':10000,'seed':2026100701,'unit':'source','paired':'NeST-Com minus ARPC'}}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for pin in reg['source_files']+reg['inputs']:verify(pin)
    assert not (RUN/'owner.launch.json').exists();gpu=reg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=8192 and now['utilization']<=25
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();assert start<reg['absolute_deadline_unix'];put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'gpu':now,'started_unix':start});old.RUN=RUN
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6');env.pop('PYTHONPATH',None)
    try:
        for role,filename in [('score','score.py'),('statistics','final_stats.py')]:
            if role=='statistics':success(RUN/'score.actual_wait.json')
            remain=min(reg['phase_wall_seconds']-(time.time()-start),reg['absolute_deadline_unix']-time.time());assert remain>0
            old.run_stage(role,[str(PY),'-B',str(CODE/filename),'--expected-registration',expected],env if role=='score' else dict(env,CUDA_VISIBLE_DEVICES=''),min(remain,reg['stage_seconds'][role]))
        put(RUN/'completion.json',{'status':'EVALUATION_ANALYSIS_CLOSED','output':desc(OUT/'completion.json'),'seconds':time.time()-start})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
