"""Deduplicate actual RX inputs; reuse exact historical outputs; one CDF vector."""
from pathlib import Path
import sys,os,time,subprocess
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,checked,cr,asha,digest,ident,sha,OFFSETS
import numpy as np
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');E=R/'experiments/nestcom-v9-completion-20261101';D=R/'results/nestcom_v9_completion_20261101_v1';L=B/'nestcom_v9_completion_20261101_v1';RT=R.parent.parent/'var_comm_runtime_20261010'
def main():
    w=read(L/'token_phy_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
    c=read(L/'token_phy/completion.json');assert c['ledger']['unresolved']==0 and c['physical']==216
    out=L/'token_rx_plan';out.mkdir(exist_ok=False);(out/'repacked').mkdir()
    sys.path.insert(0,str(R/'experiments/nestcom-n512-20261028'));from n512_phy import received_projection
    K=B/'kodak_current_20261026_v1/nest_v1/recovery_v1'
    old_id=read(K/'receiver/actual_native_identity.json');new_id=read(L/'token_source/actual_native_identity.json')
    for key in ['models','runtime','numeric_flags','actual_CUDA_UUID','actual_PCI_bus_id','sampling']:assert old_id[key]==new_id[key],key
    images=read(K/'receiver/images.json');oldprojections=read(K/'phy/rx_projection.json');old={digest(received_projection(v['actual_RX'])):images[v['source_input_key']]for v in oldprojections}
    jobs=read(L/'token_phy/rx_inputs.json');formal_keys=[j['key']for j in jobs];reuse={}
    for j in jobs:
        if j['key']not in old:continue
        v=old[j['key']];e=cr(v['evidence'])['evidence'];assert e['canonical']and not e['receiver_truth_used']and not e['TX_CDF_used']
        with np.load(checked(v['image_archive']),allow_pickle=False)as z:image=z['image'].copy()
        tokens=np.asarray(e['received_tokens'],np.int64);assert asha(image)==v['image_sha256']
        p=out/'repacked'/(j['key']+'.npz')
        with p.open('xb')as f:np.savez(f,image=image,received_tokens=tokens)
        reuse[j['key']]=dict(key=j['key'],image=pin(p),pixel_sha256=asha(image),received_tokens_sha256=asha(tokens),status='ACCEPTED_SOURCE',original_image=v['image_archive'],original_evidence=v['evidence'],same_model_numeric_device=True)
    # One missing whole-scale CDF trace is captured without a new source pass or PHY.
    source=read(L/'sources.json')[0]
    with np.load(checked(source['nest_streams']),allow_pickle=False)as z:bits=z['m4_K0'].copy()
    projection=dict(status='SOURCE_INPUT',codec='NEST',m=4,K=0,family='EC_VAR_WHOLE',bits=bits.tolist());refkey=digest(projection)
    assert refkey not in formal_keys
    jobs.append(dict(key=refkey,projection=projection))
    save(out/'reference_request.json',dict(key=refkey,source_id=source['source_id'],source_index=source['source_index'],scope='One WHOLE missing independent CDF trace; no new PHY; excluded from Kodak216',old_source_streams=source['nest_streams']))
    save(out/'jobs.json',jobs);save(out/'reuse.json',reuse);save(out/'formal_keys.json',formal_keys)
    reg=read(R/'results/nestcom_opt_20261030_v1/final_rx_registration.json')
    for k in ['final_freeze','route','budget_total_seconds','prior_route_device_seconds','prior_route_wait_receipts']:reg.pop(k,None)
    count=len(jobs)-len(reuse)
    reg.update(output=str(L/'token_rx'),jobs=pin(out/'jobs.json'),reuse=pin(out/'reuse.json'),seconds=1800,cpu_affinity=[0,1],pci='0000:61:00.0',scope=pin(D/'scope.json'),reference_request=pin(out/'reference_request.json'),caps=dict(model_load=1,prior_scale=20*count,decoder_forward=count,source_decode=count,completion_forward=count,render=count))
    reg['code'] += [pin(E/'rx_with_trace.py'),pin(E/'prepare_rx.py')]
    rp=D/'token_rx_registration.json';save(rp,reg)
    cp=D/'token_rx_owner_registration.json';ctrl=dict(control=str(L/'token_rx_control'),command=[str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/'rx_with_trace.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0'),hard_seconds=1860,gpu=True,code=reg['code']);save(cp,ctrl)
    free,util=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>=60000 and util<=10
    with(L/'token_rx_owner.log').open('xb')as log:
        p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(R/'experiments/nestcom-opt-20261030/control_once.py'),str(cp),sha(cp)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(D/'token_rx_launch.json',dict(owner=ident(p.pid),registration=pin(rp),created_unix=time.time(),unique_initial_launch=True))
    print(dict(formal_unique=len(formal_keys),formal_reuse=len(reuse),formal_new=count-1,reference_new=1))
if __name__=='__main__':main()
