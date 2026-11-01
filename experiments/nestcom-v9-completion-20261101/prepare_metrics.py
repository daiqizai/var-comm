"""Native-float Kodak scoring plan; old four-metric values are reused exactly."""
from pathlib import Path
import sys,os,time,subprocess,csv
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,checked,cr,asha,digest,ident,sha
import numpy as np
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');E=R/'experiments/nestcom-v9-completion-20261101';D=R/'results/nestcom_v9_completion_20261101_v1';L=B/'nestcom_v9_completion_20261101_v1';RT=R.parent.parent/'var_comm_runtime_20261010'
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
def rows(p):return list(csv.DictReader(Path(p).open()))
def main():
    wait=read(L/'token_rx_control/actual_wait.json');assert wait['returncode']==0 and ident(wait['child']['pid'])is None
    assert read(L/'token_rx/completion.json')['ledger']['unresolved']==0
    out=L/'metric_plan';out.mkdir(exist_ok=False);sources=read(L/'sources.json');ss={s['source_id']:s for s in sources}
    frames=[];pairs={};four_cache={};verified=set()
    def pair(sid,pix,image,image_key='image'):
        s=ss[sid];key=digest([s['reference_sha256'],pix]);j=dict(pair_key=key,source_id=sid,source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=pix,image=image,image_key=image_key)
        if key not in verified:
            if image is None:y=np.full((3,256,256),.5,np.float32)
            else:
                with np.load(checked(image),allow_pickle=False)as z:y=z[image_key].copy()
            assert y.shape==(3,256,256)and y.dtype==np.float32 and asha(y)==pix and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1
            verified.add(key)
        pairs.setdefault(key,j);return key
    baseline=rows(R/'results/kodak_current_20261026_v1/delivery/frame_metrics.csv')
    for v in baseline:
        if v['method']not in ('NEST_WHOLE','NEST_PARTIAL'):continue
        s=ss[v['source_id']];assert s['reference_sha256']==v['reference_array_sha256']
        image=dict(path=v['image_path'],sha256=v['image_file_sha256']);key=pair(v['source_id'],v['image_array_sha256'],image,v['image_key'])
        values={m:float(v[m])for m in METRICS};four_cache.setdefault(key,values);assert four_cache[key]==values
        frames.append(dict(method={'NEST_WHOLE':'WHOLE','NEST_PARTIAL':'Quarter'}[v['method']],N=1024,source_id=v['source_id'],source_index=int(v['source_index']),snr_db=int(v['snr_db']),noise_seed=int(v['noise_seed']),pair_key=key,status=v['rx_status'],four_origin='EXACT_CLOSED_BASELINE_VALUES',**values))
    ap=read(B/'adatoken_response_20261026_v1/c_predclass_kodak24_metric_preparation/three_plan.json');am={(v['source_id'],int(v['snr_db']),int(v['noise_seed'])):v for v in rows(R/'results/adatoken_response_20261026_v1/c_predclass_formal_delivery_v1/kodak24/frame_metrics.csv')}
    for v in ap['frames']:
        assert ss[v['source_id']]['reference_sha256']==v['reference_sha256']
        m=am[(v['source_id'],v['snr_db'],v['noise_seed'])];assert m['pixel_sha256']==v['pixel_sha256']
        key=pair(v['source_id'],v['pixel_sha256'],v['image']);values={k:float(m[k])for k in METRICS};four_cache.setdefault(key,values);assert four_cache[key]==values
        frames.append(dict(method='Ada',N=1024,source_id=v['source_id'],source_index=v['source_index'],snr_db=v['snr_db'],noise_seed=v['noise_seed'],pair_key=key,status=v['source_status'],four_origin='EXACT_CLOSED_BASELINE_VALUES',**values))
    rx={v['key']:v for v in read(L/'token_rx/outputs.json')};physical={v['physical_key']:v for v in read(L/'token_phy/frames.json')};new={}
    for v in read(L/'token_frame_plan/logical.json'):
        p=physical[v['physical_key']];r=rx[p['received_key']];key=pair(v['source_id'],r['pixel_sha256'],r['image'])
        f=dict(v,pair_key=key,status=r['status'],received_key=r['key'])
        if key in four_cache:f.update(four_origin='EXACT_NATIVE_PIXEL_BASELINE_PAIR',**four_cache[key])
        else:f.update(four_origin='NEW_TOKEN_PAIR');new[key]=pairs[key]
        frames.append(f)
    assert len(frames)==864 and len(new)<=72 and len(pairs)<=288
    save(out/'sources.json',sources);save(out/'frames.json',frames);save(out/'dists_pairs.json',list(pairs.values()));save(out/'new_four_plan.json',dict(sources=sources,pairs=list(new.values())))
    save(out/'completion.json',dict(status='REGISTERED_INPUTS',logical_frames=len(frames),new_four_pairs=len(new),new_dists_pairs=len(pairs),baseline_four_logical_reused=648,token_four_logical_reused=sum(f['method']=='NeST-Token'and f['four_origin']!='NEW_TOKEN_PAIR'for f in frames),native_arrays_verified=len(verified),new_reconstruction=0))
    for role,mode,gpu,pci in [('token_three','three',1,'0000:62:00.0'),('token_clip','clip',2,'0000:63:00.0'),('kodak_dists','dists',3,'0000:64:00.0')]:
        if mode=='dists':
            reg=read(R/'results/nestcom_perceptual_followup_20261030_v2/E1_core_dists_registration.json')
            for k in ['priority_pairs','caches','route','budget_total_seconds','prior_route_wait_receipts','prior_route_device_seconds']:reg.pop(k,None)
            reg.update(sources=pin(out/'sources.json'),pairs=pin(out/'dists_pairs.json'),caps=dict(load_dists=1,pair=len(pairs)),caches=[])
            engine='dists_worker';python=RT/'envs/frozen_visual_py310/bin/python'
        else:
            reg=read(R/('results/nestcom_opt_20261030_v1/fit_A_'+mode+'_registration.json'))
            for k in ['route','budget_total_seconds','prior_route_wait_receipts','prior_route_device_seconds']:reg.pop(k,None)
            caps=dict(embedding=24+len(new),pair=len(new));caps.update(dict(load_lpips=1,load_dreamsim=1)if mode=='three'else dict(load_clip=1))
            reg.update(plan=pin(out/'new_four_plan.json'),caches=[],caps=caps)
            engine='metrics';python=RT/'envs/frozen_visual_py310/bin/python'if mode=='three'else Path('/usr/bin/python')
        reg.update(output=str(L/role),pci=pci,cpu_affinity=[8+gpu*2,9+gpu*2],seconds=1800,scope=pin(D/'scope.json'))
        reg['code'] += [pin(E/'worker.py'),pin(E/'prepare_metrics.py')]
        rp=D/(role+'_registration.json');save(rp,reg)
        ctrl=dict(control=str(L/(role+'_control')),command=[str(python),'-B',str(E/'worker.py'),engine,str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0'),hard_seconds=1860,gpu=True,code=reg['code'])
        cp=D/(role+'_owner_registration.json');save(cp,ctrl)
        free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>=60000 and util<=10
        with(L/(role+'_owner.log')).open('xb')as log:
            p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(R/'experiments/nestcom-opt-20261030/control_once.py'),str(cp),sha(cp)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(D/(role+'_launch.json'),dict(owner=ident(p.pid),registration=pin(rp),created_unix=time.time(),unique_initial_launch=True))
    print(read(out/'completion.json'))
if __name__=='__main__':main()
