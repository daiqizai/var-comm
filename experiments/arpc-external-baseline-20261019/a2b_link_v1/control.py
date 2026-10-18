"""Subordinate PHY registration inheriting the unchanged A2b absolute deadline."""
import argparse,fcntl,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019');sys.path.insert(0,str(E));sys.path.insert(0,str(E/'phy_v1'))
import link,a1_owner as old
from a1_owner import R,L,O,desc,verify,put,identity,gone
CODE=E/'a2b_link_v1';RUN=L/'a2b_link_v1';OUT=O/'a2b_link_v1';REG=OUT/'registration.json';QUAL=L/'a2b_link_qualification_v1'
CPY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
def success(p):
    a=json.loads(p.read_text());assert type(a['returncode']) is int and a['returncode']==0 and not any(a.get(k) for k in ['timed_out','terminated','killed','metadata_error']) and gone(a['process']);return a
def qualify():
    assert not QUAL.exists();QUAL.mkdir();old.RUN=QUAL
    code="""import importlib.abc,sys,unittest,json
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','sionna','transformers','infinity'}:
   blocked.append(fullname);raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard());r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.',pattern='test_link.py'))
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'science':0}));assert r.wasSuccessful() and r.testsRun==6 and not blocked
"""
    old.run_stage('tests',[str(CPY),'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    put(QUAL/'completion.json',{'status':'PASS','tests':6,'scientific_calls':0,'sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]})
def prepare():
    assert not RUN.exists() and not OUT.exists();success(QUAL/'tests.actual_wait.json');qual=json.loads((QUAL/'completion.json').read_text());assert qual['status']=='PASS'
    for d in qual['sources']:verify(d)
    source=O/'a2b_source_v1';review=source/'root_closure_review.json';closed=json.loads(review.read_text());assert closed['status']=='SOURCE1000_CLOSED' and closed['all_processes_gone'] and closed['unresolved']==0
    for d in closed['controls']+closed['outputs']:verify(d)
    complete=json.loads((source/'completion.json').read_text());assert complete['status']=='SOURCE1000_READY_NO_WIRELESS_YET' and complete['new_source_passes']==900
    for d in complete['outputs']:verify(d)
    window=L/'a2b_source_v1/shared_window.json';w=json.loads(window.read_text());assert time.time()<w['absolute_deadline_unix']
    budget=json.loads((source/'resource_budget.json').read_text());assert budget['logical_frame_cap']==18000 and budget['noise_seeds']==[4101,4102,4103]
    assert desc(source/'resource_budget.json')==w['budget']
    phy=O/'a2_phy_engineering_v1/root_closure_review.json';assert desc(phy)['sha256']=='ffbeb279191426afa326a01984c06a4069951771ec12404748dee674dded6bd0'
    oldreg=json.loads((O/'a2_phy_engineering_v1/registration.json').read_text());recovery=json.loads((O/'a2_phy_rx_environment_v2/registration.json').read_text())
    for d in oldreg['source_files']+oldreg['inputs']+recovery['source_files']+recovery['inherited_closed']:verify(d)
    n=O/'a2_screen_v1/nomination.json';nom=json.loads(n.read_text());assert {s:v['proposed_candidates'] for s,v in nom['decisions'].items()}==budget['candidates_per_SNR']
    inputs=[desc(p) for p in [review,source/'registration.json',source/'completion.json',source/'stream_catalogue.json',source/'resource_budget.json',window,n,phy,QUAL/'completion.json',QUAL/'tests.actual_wait.json',O/'a2_phy_engineering_v1/registration.json',O/'a2_phy_rx_environment_v2/registration.json']]
    sourcefiles=oldreg['source_files']+recovery['source_files']+[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()]
    RUN.mkdir();OUT.mkdir();reg={'schema':'ARPC_A2B_PHY18000_REGISTRATION_V1','host':os.uname().nodename,'created_unix':time.time(),
      'source_files':sourcefiles,'inputs':inputs,'streams':desc(source/'stream_catalogue.json'),'nomination':desc(n),
      'backend_identity':oldreg['backend_identity'],'shared_absolute_deadline_unix':w['absolute_deadline_unix'],'shared_budget':w['budget'],
      'limits':{'logical_frames':18000,'header':18000,'body':18000,'neural_calls':0,'scores':0},
      'stage_seconds':{'tx':1000,'rx':2600,'audit':300},'phase_wall_seconds':3600,
      'source_input':'Source streams only used by TX. RX receives observations, SNR and public counter, full96profile catalogue; no source identity/desired profile/TX cache.',
      'failure_policy':'All CRC/header/parser failures remain in logical denominator. Unknown error preserves state and stops without replay.'}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inputs']:verify(d)
    assert not (RUN/'owner.launch.json').exists() and time.time()<reg['shared_absolute_deadline_unix'];lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time()})
    old.RUN=RUN;start=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        for name in ['tx','rx','audit']:
            if name=='audit':success(RUN/'tx.actual_wait.json');success(RUN/'rx.actual_wait.json')
            remain=min(reg['phase_wall_seconds']-(time.time()-start),reg['shared_absolute_deadline_unix']-time.time());assert remain>0
            argv=[str(CPY),'-B',str(CODE/'worker.py'),name]
            if name!='rx':argv+=['--expected-registration',expected]
            old.run_stage(name,argv,env,min(reg['stage_seconds'][name],remain))
        put(RUN/'completion.json',{'status':'PHYSICAL_CALIBRATION_CLOSED_PENDING_SOURCE_RX','registration':desc(REG),'output':desc(OUT/'completion.json'),'seconds':time.time()-start})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
