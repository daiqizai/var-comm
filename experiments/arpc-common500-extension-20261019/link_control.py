"""Exactly 3600 new N1024 events, frozen profiles and original counters/noise."""
import argparse,time
from stage_support import *
RUN=L/'link_v1';OUT=O/'link_v1';REG=OUT/'registration.json'
def prepare():
    assert not RUN.exists()and not OUT.exists()
    q=qualification();source=closed('source')
    done=read(O/'source_v1/completion.json');assert done['sources']==400 and done['streams']==2800 and done['unresolved']==0
    old=read(PO/'a3_link_v1/registration.json')
    for d in old['source_files']:verify(d)
    policy=PO/'a2b_metrics_v1/frozen_policies.json'
    registered=read(O/'registration.json')
    for d in registered['inputs']:verify(d)
    # Frozen policies were established and byte-bound by the population inventory.
    expected={k:{'proposed_candidates':[v]}for k,v in POLICIES.items()}
    from plan import events
    sys.path.insert(0,str(PE/'phy_v1'));import link
    frames=events(read(O/'source_v1/stream_catalogue.json')['sources'],expected,link.templates())
    assert len(frames)==3600 and len({r['source_id']for r in frames})==400
    assert all(r['counter']//1==3000000+96*r['source_index']+r['profile_id']for r in frames)
    RUN.mkdir();OUT.mkdir();put(OUT/'frozen_nomination.json',{'decisions':expected,'original_frozen_policies':desc(policy),'selection':False})
    code=[desc(E/n)for n in ('common.py','stage_support.py','execute_stage_v1.py','link_control.py','link_worker.py','plan.py')]
    reg={'schema':'ARPC_COMMON500_REMAINING400_PHY_V1','host':os.uname().nodename,'created_unix':time.time(),'source_files':old['source_files']+code,'inputs':q+[source,desc(O/'registration.json'),desc(policy),desc(O/'source_v1/completion.json'),desc(O/'source_v1/stream_catalogue.json'),desc(OUT/'frozen_nomination.json')],'streams':desc(O/'source_v1/stream_catalogue.json'),'nomination':desc(OUT/'frozen_nomination.json'),'backend_identity':old['backend_identity'],'limits':{'logical_frames':3600,'header':3600,'body':3600,'neural_calls':0,'scores':0},'stage_seconds':{'tx':1200,'rx':3000,'audit':600},'phase_wall_seconds':4800,'N_complex':1024,'noise_seeds':[6201,6202,6203],'original_source_indices':True,'selection':False,'RX_truth':False}
    put(REG,reg);print(json.dumps(desc(REG)))
def run(expected):
    r=read(REG);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');env.pop('PYTHONPATH',None)
    jobs=[(role,[str(CPY),'-B',str(E/'link_worker.py'),role]+([]if role=='rx'else['--expected-registration',expected]))for role in ('tx','rx','audit')]
    run_stages('link',expected,jobs,env,r['stage_seconds'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','run']);p.add_argument('--expected-registration');a=p.parse_args()
    prepare()if a.role=='prepare'else run(a.expected_registration)
