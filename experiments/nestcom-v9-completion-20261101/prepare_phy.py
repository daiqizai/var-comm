"""Prepare the single Kodak Token216 continuation from frozen source streams."""
from paid_null import R,choose
import sys,os,subprocess,time
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,sha,checked,cr,asha,digest,ident
from frame_plan import policy
import numpy as np
from pathlib import Path
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/nestcom-v9-completion-20261101';D=R/'results/nestcom_v9_completion_20261101_v1';L=B/'nestcom_v9_completion_20261101_v1'
CPY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'

def closed(role):
    w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid']) is None
    c=read(L/role/'completion.json');assert c['ledger']['unresolved']==0
    return c

def main():
    closed('token_source')
    w=read(L/'paid_null_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
    q=read(L/'paid_null_regression/completion.json');assert q['status']=='PASS'
    out=L/'token_frame_plan';out.mkdir(exist_ok=False)
    sources=read(L/'sources.json');new={v['source_id']:v for v in read(L/'token_source/outputs.json')};cat=cr(q['catalogue'])
    K=B/'kodak_current_20261026_v1/nest_v1/recovery_v1';oldrows=read(K/'phy/logical_frames.json');old={}
    for v in oldrows:
        f=cr(v['physical_frame']);p=f['actual_RX']['rx_profile']
        if f['actual_RX']['header']['header_ok']and f['actual_RX']['body']['parser_accepted']:
            old[(v['source_id'],v['snr_db'],v['noise_seed'],v['public_frame_counter'],p['profile_id'],asha(np.asarray(f['actual_RX']['body']['payload'],np.uint8)))]=v['physical_frame']
    frozen=pin(R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json');jobs=[];logical=[]
    for s in sources:
        with np.load(checked(new[s['source_id']]['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
        for slot,snr in enumerate((4,10,19)):
            c=policy(1024,snr,'NEST',dict(policy_N1024=frozen))
            p,b,nf,attempts=choose(streams,cat,q=c['q'],rate=c['rate'],target=c['target'],grid='token')
            for repeat,seed in enumerate((2001,2002,2003)):
                counter=(slot*24+s['source_index'])*3+repeat
                identity=dict(N=1024,source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,noise_seed=seed,counter=counter,profile_id=p['profile_id'],payload_sha256=asha(b))
                key=digest(identity);previous=old.get((s['source_id'],snr,seed,counter,p['profile_id'],asha(b)))
                jobs.append(dict(key=key,**identity,payload=b.tolist(),profile=p,old=previous))
                logical.append(dict(physical_key=key,method='NeST-Token',N=1024,snr_db=snr,source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],source_float_sha256=s['reference_sha256'],target=c['target'],q=c['q'],rate=c['rate'],endpoint=[p['m'],p['K']],source_bits=len(b),capacity_bits=p['k']-29,utilization=len(b)/(p['k']-29),noise_seed=seed,repeat=repeat,tx_no_fit=nf,checks=len(attempts)))
    assert len(jobs)==len(logical)==216 and len({j['key']for j in jobs})==216
    save(out/'physical.json',jobs);save(out/'logical.json',logical)
    save(out/'completion.json',dict(status='PREPARED',logical=216,new_at_most=216,reuse_candidates=sum(v['old']is not None for v in jobs),policy=frozen,catalogue=q['catalogue'],source=pin(L/'token_source/completion.json'),NULL_no_fit=sum(v['tx_no_fit']for v in logical)))
    reg=read(R/'results/nestcom_opt_20261030_v1/final_phy_1024_registration.json')
    reg.update(output=str(L/'token_phy'),jobs=pin(out/'physical.json'),catalogue=q['catalogue'],seconds=1800,cpu_affinity=[44,45],scope=pin(D/'scope.json'),paid_null_qualification=pin(L/'paid_null_regression/completion.json'))
    reg['code'] += [pin(E/'worker.py'),pin(E/'prepare_phy.py'),pin(E/'paid_null.py')]
    rp=D/'token_phy_registration.json';save(rp,reg)
    c=dict(control=str(L/'token_phy_control'),command=[str(CPY),'-B',str(E/'worker.py'),'extended_frame_worker_v3',str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=1860,gpu=False,code=reg['code'])
    cp=D/'token_phy_owner_registration.json';save(cp,c)
    with(L/'token_phy_owner.log').open('xb')as log:
        p=subprocess.Popen([str(CPY),'-B',str(R/'experiments/nestcom-opt-20261030/control_once.py'),str(cp),sha(cp)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(D/'token_phy_launch.json',dict(owner=ident(p.pid),registration=pin(rp),created_unix=time.time(),unique_initial_launch=True))
    print(read(out/'completion.json'))
if __name__=='__main__':main()
