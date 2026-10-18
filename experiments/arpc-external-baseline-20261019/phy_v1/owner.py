"""Bounded registration and serial TX/RX/post-exit audit for 24 paid frames."""
import argparse,fcntl,json,os,pathlib,re,subprocess,sys,time,traceback
import link
E=link.R/'experiments/arpc-external-baseline-20261019';sys.path.insert(0,str(E))
import a1_owner as old
from a1_owner import L,O,desc,put,verify,identity,gone
CODE=E/'phy_v1';RUN=L/'a2_phy_engineering_v1';OUT=O/'a2_phy_engineering_v1';REG=OUT/'registration.json'
RT=link.R.parent.parent/'var_comm_runtime_20261010';CPY=RT/'envs/frozen_ldpc_py311/bin/python'
QUAL=L/'a2_phy_synthetic_qualification_v1'
def success(path):
    w=json.loads(path.read_text());assert type(w['returncode']) is int and w['returncode']==0
    assert not any(w.get(k) for k in ['timed_out','terminated','killed','metadata_error']) and gone(w['process']);return w
def read_verified(d):
    verify(d);return json.loads(pathlib.Path(d['path']).read_text())
def prepare():
    assert not OUT.exists() and not RUN.exists()
    success(QUAL/'tests.actual_wait.json');q=json.loads((QUAL/'completion.json').read_text());assert q['status']=='PASS'
    for d in q['sources']:verify(d)
    success(L/'a2_screen_execution_v1/actual_wait.json');success(L/'a2_metrics_dispatch_v1/actual_wait.json')
    nomination=O/'a2_screen_v1/nomination.json';a=json.loads(nomination.read_text())
    assert a['status']=='A2A_NOMINATION_READY_FOR_REGISTERED_LINK_QUALIFICATION'
    for d in a['inputs']:verify(d)
    states=O/'a2_clean_v1/clean_states.json';s=json.loads(states.read_text())['states']
    source=O/'sources.json';sources=json.loads(source.read_text())['cal100']
    by={(x['rank'],x['endpoint']):x for x in s};assert len(by)==700
    candidates=[]
    for snr in [4,10,19]:
        n=a['decisions'][str(snr)];assert not n['uncovered_possible_candidates'] and len(n['proposed_candidates'])==2
        for cid in n['proposed_candidates']:
            m=re.fullmatch(r'e([1-7])_q([246])_r(\d)-(\d)',cid);assert m
            target,qbit,num,den=map(int,m.groups());candidates.append((snr,cid,target,qbit,f'{num}/{den}'))
    frames=[]
    def add(rank,snr,cid,target,qbit,rate,purpose,force_null=False):
        p=next(p for p in link.templates() if p['endpoint']==target and p['q']==qbit and p['nominal_rate']==rate)
        lengths={e:8*by[rank,e]['wire']['bytes'] for e in range(1,8)}
        endpoint=0 if force_null else link.fallback(lengths,target,p['source_capacity_bits'])
        wire=None if endpoint==0 else by[rank,endpoint]['wire']
        if wire:verify(wire)
        frames.append({'key':f'engineering_{len(frames):02d}','rank':rank,'source_id':sources[rank]['source_id'],
          'snr_db':snr,'candidate_id':cid,'target_endpoint':target,'endpoint':endpoint,'q':qbit,'nominal_rate':rate,
          'wire':wire,'noise_seed':4101,'counter':1000000+len(frames),'purpose':purpose})
    for snr,cid,target,qbit,rate in candidates:
        for rank in range(3):add(rank,snr,cid,target,qbit,rate,'candidate_operating_point')
    mc=sorted({(qbit,rate) for _,_,_,qbit,rate in candidates});assert len(mc)==5
    for qbit,rate in mc:add(3,19,f'e7_q{qbit}_r{rate.replace("/","-")}',7,qbit,rate,'19dB_selected_MCS_connectivity')
    add(4,4,'explicit_null_q2_r1-2',7,2,'1/2','explicit_null_full_frame',True);assert len(frames)==24
    pilot=RT/'qualification/h800_ep_pilot_phy_v1_attempt1/registered/request.json'
    assert desc(pilot)['sha256']=='34a3538a57650dcc54c00abcf9ca45b3cde2dfc11e02d125b96e42555accd478'
    historical=json.loads(pilot.read_text());assert 4101 in {f['noise_seed'] for f in historical['frames']}
    phy=historical['phy_identity'];proof=phy['qualification'];assert link.ep.sha(proof['path'])==proof['sha256']
    completed=json.loads(pathlib.Path(proof['path']).read_text());assert completed['status']=='PASS' and completed['packet_decode_count']==457
    link.ep.verify_bindings(completed['source_bindings']);link.ep.verify_bindings(completed['input_bindings'])
    cpp=link.R/'src/var_comm/token_trellis.cpp';build=link.R/'outputs/build'/link.ep.sha(cpp)[:16]
    b=json.loads((build/'build.json').read_text());assert b['source_sha256']==link.ep.sha(cpp) and b['library_sha256']==link.ep.sha(build/'token_trellis.so')
    assert not list(build.glob('.build-*')),'Unexpected existing compiler temporary state; do not modify it'
    paths=set(link.ep.source_paths(link.R))|{E/'a1_owner.py',E/'a1_native.py',E/'wire.py',E/'upstream/utils/arithmeticcoding.py'}
    paths.update(p for p in CODE.iterdir() if p.is_file())
    reg={'schema':'ARPC_24_ENGINEERING_FRAMES_V1','created_unix':time.time(),'host':os.uname().nodename,
      'repository_head':subprocess.check_output(['git','-C',str(link.R),'rev-parse','HEAD'],text=True).strip(),
      'source_files':[desc(p) for p in sorted(paths)],'inputs':[desc(p) for p in [nomination,states,source,pilot,pathlib.Path(proof['path']),build/'build.json',build/'token_trellis.so',QUAL/'completion.json',QUAL/'tests.actual_wait.json']],
      'backend_identity':phy['backend_identity'],'frames':frames,'session':link.SESSION,'public_profiles':link.templates(),
      'limits':{'engineering_frames':24,'header_decodes':24,'body_decodes':24,'neural_calls':0,'calibration_logical_frames':0,'evaluation_frames':0},
      'resources':{'device':'CPU','CUDA_visible':False,'threads':2,'interop':1,'global_seconds':600,'stage_seconds':{'tx':180,'rx':300,'audit':120}},
      'rules':'Actual received paid header selects profile. CRC/length/padding/source grammar only. No TX truth in RX; truth audit after RX exit. No automatic replay.',
      'engineering_only':'18 candidate/source fixed frames, five selected-MCS connectivity frames at19dB, one explicit full-length null frame; never included in quality statistics.'}
    RUN.mkdir();OUT.mkdir();put(REG,reg);print(json.dumps(desc(REG)))
def qualify():
    assert not QUAL.exists();QUAL.mkdir();put(QUAL/'supervisor.json',{'process':identity(os.getpid()),'argv':sys.argv})
    code="""import importlib.abc,json,sys,unittest
blocked=[]
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','sionna','transformers','infinity'}:
   blocked.append(fullname);raise RuntimeError('Synthetic test imported model/PHY runtime '+fullname)
sys.meta_path.insert(0,Guard())
suite=unittest.defaultTestLoader.discover('.',pattern='test_*.py')
r=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests':r.testsRun,'blocked':blocked,'neural_calls':0,'packets':0}))
assert r.testsRun==15 and r.wasSuccessful() and not blocked
"""
    previous=old.RUN;old.RUN=QUAL
    try:old.run_stage('tests',[str(CPY),'-B','-c',code],dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),120)
    finally:old.RUN=previous
    put(QUAL/'completion.json',{'status':'PASS','sources':[desc(p) for p in sorted(CODE.iterdir()) if p.is_file()],
      'tests':15,'packets':0,'neural_calls':0,'actual_wait':desc(QUAL/'tests.actual_wait.json')})
def run(expected):
    assert desc(REG)['sha256']==expected;reg=json.loads(REG.read_text());assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inputs']:verify(d)
    assert not (RUN/'owner.launch.json').exists() and not (RUN/'failure.json').exists()
    lock=(L/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put(RUN/'owner.launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'registration_sha256':expected,'started_unix':time.time()})
    old.RUN=RUN;start=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        for name in ['tx','rx','audit']:
            if name=='audit':success(RUN/'tx.actual_wait.json');success(RUN/'rx.actual_wait.json')
            argv=[str(CPY),'-B',str(CODE/'engineering.py'),name]
            if name!='rx':argv+=['--expected-registration',expected]
            remain=reg['resources']['global_seconds']-(time.monotonic()-start);assert remain>0
            old.run_stage(name,argv,env,min(remain,reg['resources']['stage_seconds'][name]))
        put(RUN/'completion.json',{'status':'PASS','registration_sha256':expected,'seconds':time.monotonic()-start,'output':desc(OUT/'completion.json')})
    except BaseException:put(RUN/'failure.json',{'error':traceback.format_exc(),'retry_authorized':False});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['qualify','prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    if a.role=='qualify':qualify()
    elif a.role=='prepare':prepare()
    else:assert a.expected_registration;run(a.expected_registration)
if __name__=='__main__':main()
