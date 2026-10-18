"""Once-only explicit recovery after diagnosed zero-RX-call import failure."""
import argparse,fcntl,json,os,pathlib,subprocess,sys,time,traceback
from adapter import E,ORIGINAL,SHA
sys.path.insert(0,str(E/'phy_v1'));sys.path.insert(0,str(E))
import owner as original
from a1_owner import L,O,desc,put,verify,identity,gone
old=original.old;success=original.success;CPY=original.CPY
CODE=E/'phy_rx_environment_v2';ROOT=L/'a2_phy_rx_environment_v2';RUN=ROOT;REG=O/'a2_phy_rx_environment_v2/registration.json'
def failed_closed():
    assert desc(original.REG)['sha256']=='5251dc25420f4da924ad6e7fc795c0027ec56fc0edac3d820f3e4c01dad35554'
    success(original.RUN/'tx.actual_wait.json')
    for p in [original.RUN/'rx.actual_wait.json',L/'a2_phy_dispatch_control_v1/child.actual_wait.json']:
        w=json.loads(p.read_text());assert type(w['returncode']) is int and w['returncode']==1 and not w['timed_out'] and gone(w['process'])
    for p in [original.RUN/'owner.launch.json',L/'a2_phy_dispatch_control_v1/supervisor.json']:assert gone(json.loads(p.read_text())['process'])
    assert {p.name for p in (original.RUN/'rx').iterdir()}=={'runtime_cache'}
    assert not (original.RUN/'rx_environment_v1').exists() and not (original.OUT/'completion.json').exists()
    assert 'frozen_python_base_v1/lib/python3.11/csv.py' in (original.RUN/'rx.stderr').read_text()
    reg=json.loads(original.REG.read_text())
    for d in reg['source_files']+reg['inputs']:verify(d)
    assert desc(ORIGINAL)['sha256']==SHA
    return reg
def qualify():
    failed_closed();assert not ROOT.exists();ROOT.mkdir();old.RUN=ROOT
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    old.run_stage('tests',[str(CPY),'-B','-m','unittest','-v','test_adapter'],env,120)
    old.run_stage('probe',[str(CPY),'-B',str(CODE/'worker.py'),'probe'],env,180)
    p=json.loads((ROOT/'constructor_probe/completion.json').read_text());assert not p['calls'] and not p['blocked'] and p['profiles']==96
    put(ROOT/'qualification.json',{'status':'PASS','sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],
      'tests':2,'packets':0,'neural_calls':0,'outputs':[desc(ROOT/n) for n in ['tests.actual_wait.json','probe.actual_wait.json','constructor_probe/completion.json']]})
def prepare():
    original_reg=failed_closed();success(ROOT/'tests.actual_wait.json');success(ROOT/'probe.actual_wait.json')
    v1=L/'a2_phy_rx_environment_v1';success(v1/'tests.actual_wait.json')
    for p in [v1/'probe.actual_wait.json',L/'a2_phy_rx_environment_qualify_control_v1/child.actual_wait.json']:
        w=json.loads(p.read_text());assert type(w['returncode']) is int and w['returncode']==1 and gone(w['process'])
    q=json.loads((ROOT/'qualification.json').read_text());assert q['status']=='PASS'
    for d in q['sources']+q['outputs']:verify(d)
    oldpins=[desc(original.REG),desc(original.RUN/'tx/completion.json')]
    oldpins += [desc(p) for p in sorted((original.RUN/'received_inputs').iterdir()) if p.is_file()]
    oldpins += [desc(p) for p in [original.RUN/'failure.json',original.RUN/'rx.actual_wait.json',original.RUN/'rx.stderr',original.RUN/'tx.actual_wait.json',L/'a2_phy_dispatch_control_v1/child.actual_wait.json']]
    assert json.loads((ROOT/'constructor_probe/completion.json').read_text())['backend_identity']==original_reg['backend_identity']
    oldpins += [desc(p) for p in sorted((E/'phy_rx_environment_v1').iterdir()) if p.is_file()]
    oldpins += [desc(p) for p in [v1/'tests.actual_wait.json',v1/'probe.actual_wait.json',v1/'probe.stderr',L/'a2_phy_environment_diagnostic_v1/diagnose.actual_wait.json',L/'a2_phy_environment_diagnostic_v1/work/constructor_probe/diagnostic.json']]
    assert not REG.parent.exists();REG.parent.mkdir()
    reg={'schema':'ARPC_RX_ENVIRONMENT_RECOVERY_V1','created_unix':time.time(),'host':os.uname().nodename,'original_registration':desc(original.REG),
      'source_files':q['sources'],'qualification':desc(ROOT/'qualification.json'),'inherited_closed':oldpins,
      'limits':{'new_transmits':0,'new_header_decodes':24,'new_body_decodes':24,'model_calls':0,'training':0},
      'scope':'Two exact Python environment roots only; old RX zero decoder calls. Preserve original failure and all24 TX observations.',
      'original_scientific_source_sha256':SHA,'stage_seconds':{'rx':300,'audit':120},'global_seconds':420}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    failed_closed();assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inherited_closed']+[reg['qualification']]:verify(d)
    assert not (ROOT/'owner.launch.json').exists();lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(ROOT/'owner.launch.json',{'process':identity(os.getpid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time()});old.RUN=ROOT;t=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        for name in ['rx','audit']:
            if name=='audit':success(ROOT/'rx.actual_wait.json')
            argv=[str(CPY),'-B',str(CODE/'worker.py'),name]
            if name=='audit':argv+=['--expected-registration',reg['original_registration']['sha256']]
            old.run_stage(name,argv,env,min(reg['stage_seconds'][name],reg['global_seconds']-(time.monotonic()-t)))
        put(ROOT/'completion.json',{'status':'RECOVERED_RX_AND_AUDIT_COMPLETE','registration':desc(REG),'output':desc(original.OUT/'completion.json'),'new_transmits':0,'original_failure_preserved':True,'seconds':time.monotonic()-t})
    except BaseException:put(ROOT/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:run(a.expected_registration)
if __name__=='__main__':main()
