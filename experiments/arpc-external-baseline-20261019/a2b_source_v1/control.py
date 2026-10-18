"""Finite missing-source preparation; successors require separately bound owners."""
import argparse,fcntl,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone,gpus
from population import missing
CODE=E/'a2b_source_v1';RUN=L/'a2b_source_v1';OUT=O/'a2b_source_v1';REG=OUT/'registration.json';QUAL=L/'a2b_source_qualification_v1'
def success(path):
    a=json.loads(path.read_text());assert type(a['returncode']) is int and a['returncode']==0 and not any(a.get(k) for k in ['timed_out','terminated','killed','metadata_error']) and gone(a['process']);return a
def prepare():
    assert not RUN.exists() and not OUT.exists()
    success(QUAL/'tests.actual_wait.json');q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    review=O/'a2_phy_engineering_v1/root_closure_review.json';assert desc(review)['sha256']=='ffbeb279191426afa326a01984c06a4069951771ec12404748dee674dded6bd0'
    phy=json.loads(review.read_text());assert phy['status']=='READY_FOR_REGISTERED_CAL1000' and phy['all_recorded_processes_gone']
    for d in phy['controls']+[phy['completion']]:verify(d)
    nominal=O/'a2_screen_v1/nomination.json';n=json.loads(nominal.read_text());assert n['status']=='A2A_NOMINATION_READY_FOR_REGISTERED_LINK_QUALIFICATION'
    assert all(len(x['proposed_candidates'])==2 and not x['uncovered_possible_candidates'] for x in n['decisions'].values())
    source_manifest=R/'results/submission_closure_20261016_v1/source_registration.json'
    assert desc(source_manifest)['sha256']=='13c9057b42cdc9d36a8934b2479e8a9385073d5205402852a355e32803c60845'
    full=json.loads(source_manifest.read_text())['sources'];old100=json.loads((O/'sources.json').read_text())['cal100'];new=missing(full,old100)
    for s in full:verify(s['archive'])
    sourcecfg={'schema':'ARPC_CAL1000_EXACT_SOURCE_POPULATION_V1','full_sources':[{k:s[k] for k in ['source_id','source_index','archive','preprocessing_id']} for s in full],
      'new_sources':new,'reused_sources':old100,'selection_uses_metrics':False}
    previous=json.loads((O/'a2_clean_v1/registration.json').read_text())
    for d in previous['source_files']+previous['weights']:verify(d)
    cfg=json.loads((L/'a2_clean_v1/public.json').read_text());gpu=cfg['resource']['gpu'];available=next(g for g in gpus() if g['uuid']==gpu['uuid'])
    assert available['free_mib']>=65536 and available['utilization']<=25
    cfg.update(schema='ARPC_A2B_TX_ONLY_PUBLIC_V1',run=str(RUN),output=str(OUT))
    RT=R.parent.parent/'var_comm_runtime_20261010';oldreq=RT/'qualification/h800_ep_full_phy_v1_attempt1/registered/request.json'
    assert desc(oldreq)['sha256']=='81ae17e2dea3569424323d919cd84a089ed967a970a7a4cbc189bd8e48b5f2a5'
    noise=json.loads(oldreq.read_text());assert sorted({f['noise_seed'] for f in noise['frames']})==[4101,4102,4103]
    RUN.mkdir();OUT.mkdir();put(RUN/'public.json',cfg);put(OUT/'sources.json',sourcecfg)
    budget={'schema':'ARPC_A2B_FINITE_RESOURCE_BUDGET_V1','calibration_sources':1000,'new_source_passes':900,'reused_source_passes':100,
      'SNRs':[4,10,19],'noise_seeds':[4101,4102,4103],'candidates_per_SNR':{s:d['proposed_candidates'] for s,d in n['decisions'].items()},
      'logical_frame_cap':18000,'header_decoder_cap':18000,'body_decoder_cap':18000,'total_clean_state_planning_bound':7000,
      'existing_clean_states':700,'new_unique_receiver_state_cap':6300,'received_wrong_but_accepted_state':'Use actual data; counts against same6300 cap. Stop with resource re-estimate if cap insufficient.',
      'total_scoring_pairs_bound':8000,'existing_cal100_clean_gray_pairs':800,'new_scoring_pair_cap':7200,
      'window_seconds':14400,'clock':'One absolute deadline recorded when firstA2b source owner starts; later stages inherit without resetting.',
      'basis':'Measured A2a TX92=128.9032s, RX676=587.0199s; max900TX~1261s +6300RX~5471s; engineering24TX+RX30.73s includes two backend initializations; conservative CPU PHY allowance3600s and metric/closure3068s.',
      'execution':'This owner only missing900TX and source audit. Actual PHY/RX/metrics need their own source-pinned subordinate registration inside this shared cap/window.',
      'A3_allowed_before_frozen_calibration':False,'new_training':0,'new_clean_reference_images':0,'scope':desc(O/'execution_scope.json')}
    put(OUT/'resource_budget.json',budget)
    inputs=[desc(p) for p in [review,nominal,source_manifest,oldreq,O/'sources.json',O/'a2_clean_v1/clean_states.json',O/'a2_clean_v1/root_closure_review.json',QUAL/'completion.json',QUAL/'tests.actual_wait.json',OUT/'sources.json',OUT/'resource_budget.json',RUN/'public.json']]
    reg={'schema':'ARPC_A2B_MISSING_SOURCE900_REGISTRATION_V1','created_unix':time.time(),'host':os.uname().nodename,
      'repository_head':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
      'source_files':previous['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],
      'weights':previous['weights'],'inputs':inputs,'public':desc(RUN/'public.json'),'sources':desc(OUT/'sources.json'),
      'limits':{'caption':900,'tokenizer':900,'tx_probability':900,'reference_render':0,'rx_render':0,'wireless_frames':0,'metric_calls':0},
      'stage_seconds':{'tx':3600,'audit':300},'source_window_seconds':3900,'shared_budget':desc(OUT/'resource_budget.json')}
    put(REG,reg);print(json.dumps(desc(REG)))
def qualify():
    assert not QUAL.exists();QUAL.mkdir();old.RUN=QUAL
    code="""import importlib.abc,sys,unittest,json
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','tensorflow','infinity'}:
   blocked.append(fullname);raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard());suite=unittest.defaultTestLoader.discover('.',pattern='test_source.py');r=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==6 and not blocked
"""
    old.run_stage('tests',[sys.executable,'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':6,'scientific_calls':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['weights']+reg['inputs']:verify(d)
    cfg=json.loads((RUN/'public.json').read_text());gpu=cfg['resource']['gpu'];now=next(g for g in gpus() if g['uuid']==gpu['uuid']);assert now['free_mib']>=65536 and now['utilization']<=25
    assert not (RUN/'owner.launch.json').exists();lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    glock=(L/f"gpu_{gpu['uuid']}.lock").open('a+');fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();put(RUN/'shared_window.json',{'started_unix':start,'absolute_deadline_unix':start+14400,'budget':reg['shared_budget']})
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'gpu_snapshot':now,'started_unix':start});old.RUN=RUN
    env=dict(os.environ,PYTHONPATH=str(L/'environment_v1')+':'+str(E),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(cfg['cuda_admission']['visible_index']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    try:
        old.run_stage('tx',[sys.executable,'-B',str(CODE/'native.py'),'tx','--public',str(RUN/'public.json'),'--sources',reg['sources']['path']],env,3600)
        old.run_stage('audit',[sys.executable,'-B',str(CODE/'audit.py'),'--expected-registration',expected],dict(env,CUDA_VISIBLE_DEVICES=''),300)
        put(RUN/'completion.json',{'status':'SOURCE1000_READY_NO_WIRELESS_YET','registration':desc(REG),'seconds':time.time()-start,'output':desc(OUT/'completion.json')})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
