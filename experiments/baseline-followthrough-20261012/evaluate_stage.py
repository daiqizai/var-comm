"""Finite frozen baseline diagnostics and holdout; no training or model selection."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,json,math,os,sys,time,traceback
ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path[:0]=[str(ROOT/'experiments/baseline-strength-20261011'),str(ROOT/'experiments/entropy-partial-mismatch-20261010/scripts')]
import numpy as np
import torch
import baseline_runtime as b
import run_continue as training
import h800_ep_pilot_metrics_v1 as metric
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
OUT=b.RESULT/'automatic_followthrough_v1'

def checked(pin):
    p=Path(pin['path']);assert b.sha(p)==pin['sha256'],str(p)
    return p

def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

class Ledger:
    def __init__(self,n,frames):self.counts=collections.Counter();self.n=n;self.frames=frames
    def call(self,name,fn,**kwargs):
        caps={'model_constructions':3,'reference_preparations':self.n,'image_scores':self.frames,
              'dinov2_vitl14_reference':self.n,'convnext_reference':self.n,
              'dinov2_vitl14_reconstruction':self.frames,'convnext_reconstruction':self.frames,
              'lpips_pair':self.frames,'lpips_alexnet_backbone_forward':2*self.frames}
        assert self.counts[name]<caps[name],name
        self.counts[name]+=1  # Reserve before execution. Unknown failures never retry.
        return fn()

class FixedPopulation:
    def __init__(self,which,reg):
        self.ids=[];images=[];features=[];self.bindings={}
        for row in reg['populations'][which]['records']:
            p=checked(row['pixels']);self.bindings[str(p)]=row['pixels']['sha256']
            with np.load(p,allow_pickle=False) as z:
                pixels=z['pixels'].copy()
                latent=z['F'].copy() if 'F' in z.files else None
            assert pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['preprocessing_id']
            if latent is None:
                p=checked(row['latent']);self.bindings[str(p)]=row['latent']['sha256']
                with np.load(p,allow_pickle=False) as z:latent=z['F'].copy()
            assert latent.dtype==np.float32 and latent.shape==(32,16,16) and np.isfinite(latent).all()
            images.append(pixels);features.append(latent);self.ids.append(row['source_id'])
        self.images=torch.from_numpy(np.stack(images));self.F=torch.from_numpy(np.stack(features))
        assert len(self.ids)==len(set(self.ids))==reg['populations'][which]['source_count']
    def batch(self,ids,device='cuda:0'):return self.images[ids].to(device=device,dtype=torch.float32)/255
    def __len__(self):return len(self.ids)
    p_batch=b.Population.p_batch

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['milestones','bottleneck','common500','new100']);args=a.parse_args()
    stage=args.stage;start=time.monotonic();reg=b.read(OUT/'registration.json')
    for path,digest in reg['code_bindings'].items():assert b.sha(path)==digest,path
    dest=OUT/stage;dest.mkdir(exist_ok=False)
    lock=(dest/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(reg['gpu'])
    torch.cuda.set_per_process_memory_fraction((16<<30)/torch.cuda.get_device_properties(0).total_memory)
    def guard():
        assert not (OUT/'STOP').exists(),'User stop requested'
        assert time.monotonic()-start<reg['stage_timeout_seconds'],'Finite stage time exhausted'
    n=1000 if stage=='milestones' else 100 if stage in ('bottleneck','new100') else 500
    frames=reg['stage_frames'][stage];ledger=Ledger(n,frames)
    config=reg['metric_config'].copy();config['runtime_identity']={'purpose':'frozen baseline '+stage,'physical_gpu':reg['gpu'],'torch':torch.__version__,'cuda':torch.version.cuda,'old_host_numeric_identity_claimed':False}
    evaluator=metric.FourMetrics(config,ledger,guard)
    evaluator.evaluator.metadata['metric_selection_usage']='Descriptive frozen baseline evaluation; no model selection or stopping use'
    evaluator.metadata.update(population=stage,selection_objective=None,policy_selection=False,DINO_L_metadata=evaluator.evaluator.metadata)
    evaluator.identity=metric.core.digest(evaluator.metadata)
    b.write(dest/'metric_identity.json',{'identity':evaluator.identity,'metadata':evaluator.metadata})
    pop=b.Population('calibration',latent=True,limit=n) if stage in ('milestones','bottleneck') else FixedPopulation(stage,reg)
    references={}
    with torch.inference_mode():
        for i in range(n):
            x=pop.batch([i],device='cpu')[0].numpy();references[i]=(x,evaluator.prepare(x))
    methods=reg['milestone_methods'] if stage=='milestones' else reg['bottleneck_methods'] if stage=='bottleneck' else reg['final_methods']
    reused={}
    if stage=='milestones':
        for method in ('P40000','P80000','Swin80000'):
            p=checked(reg['interim_reuse'][method]);rows=csv_rows(p)
            for row in rows:
                if method.startswith('Swin') and int(row['noise_seed'])!=4101:continue
                reused[method,int(row['source_index']),int(row['snr_db']),int(row['noise_seed'])]=row
    elif stage=='bottleneck':
        for method in ('P40000',reg['final_methods'][0]['name']):
            p=OUT/'milestones'/f'{method}.csv'
            done=b.read(OUT/'milestones/completion.json');assert b.sha(p)==done['outputs'][str(p)]
            for row in csv_rows(p):
                if int(row['source_index'])<100 and int(row['snr_db'])==19:
                    reused[method,int(row['source_index']),19,int(row['noise_seed'])]=row
    all_rows=[];state=None;current=None;reuse_count=0;actual=0;overlap_validation={}
    for spec in methods:
        branch=spec['branch'];method=spec['name'];cp=reg['checkpoints'][spec['checkpoint']]
        if current!=branch:
            if state is not None:del state;torch.cuda.empty_cache()
            state=training.load(branch);current=branch
            if branch=='P':del state['lp'];torch.cuda.empty_cache()
        payload=torch.load(checked(cp),map_location='cpu',weights_only=True)
        state['model'].load_state_dict(payload['model'],strict=True);del payload
        state['model'].eval().requires_grad_(False)
        snrs=spec.get('snrs',[4,10,19] if stage=='new100' else [1,4,7,10,13,19])
        seeds=spec.get('seeds',[9301,9302,9303] if stage=='new100' else [2001,2002,2003] if stage=='common500' else [4101,4102,4103])
        rows=[]
        for snr in snrs:
            holdout_cache={}
            for seed in seeds:
                for first in range(0,n,4):
                    guard();ids=list(range(first,min(first+4,n)));fresh=[]
                    for i in ids:
                        old=reused.get((spec.get('reuse_method',method),i,snr,seed))
                        if old is None:fresh.append(i);continue
                        assert old['source_id']==pop.ids[i]
                        assert old['reference_sha256']==metric.array_sha(references[i][0])
                        row=dict(old);row.update(method=method,branch=branch,checkpoint_step=cp['step'],source_index=i,snr_db=snr,noise_seed=seed,reused=True)
                        rows.append(row);reuse_count+=1
                    if not fresh:continue
                    ids=fresh
                    with torch.inference_mode():
                        if branch=='P':
                            if stage in ('common500','new100'):
                                # Original holdout uses float32 NumPy channel,
                                # source B1 TX/RX and same-source three-noise B3 Dc.
                                for i in ids:
                                    if i in holdout_cache:continue
                                    wave=state['model'].transmit(pop.F[[i]].cuda())[0].cpu().numpy()
                                    assert wave.shape==(1024,2) and wave.dtype==np.float32
                                    energy=float(np.square(wave.astype(np.float64)).sum());assert abs(energy-2048)<.02
                                    zs=[]
                                    for noise_seed in seeds:
                                        noise=state['ns']['seeded_noise']('VAR-CONTINUOUS-1024|'+pop.ids[i],noise_seed,(1024,2)).astype(np.float32)
                                        y=wave+noise*np.float32(10.**(-float(snr)/20))
                                        zs.append(state['model'].receive(torch.as_tensor(y[None],device='cuda:0'),torch.tensor([snr],dtype=torch.float32,device='cuda:0'))[0])
                                    holdout_cache[i]=(state['dc'](torch.stack(zs)).cpu().numpy(),energy)
                                images=np.stack([holdout_cache[i][0][seeds.index(seed)] for i in ids])
                                links=[dict(header_accepted=True,header_crc_accepted=True,header_fields_legal=True,actual_energy=holdout_cache[i][1],fallback='none') for i in ids]
                            else:
                                batch=pop.p_batch(ids,[snr]*len(ids),[seed]*len(ids),state['ns']['seeded_noise'])
                                if spec.get('mode')=='direct_F':z=batch['F'];energy=None
                                else:
                                    if spec.get('mode')=='zero_noise':batch['noise'].zero_()
                                    z,wave=state['model'](batch['F'],batch['snr'],batch['noise'])
                                    assert tuple(wave.shape)==(len(ids),1024,2)
                                    energy=wave.square().sum((1,2));assert torch.allclose(energy,energy.new_full(energy.shape,2048),rtol=1e-5,atol=.02)
                                images=state['dc'](z).cpu().numpy()
                                links=[dict(header_accepted=True,header_crc_accepted=True,header_fields_legal=True,actual_energy=float(energy[j]) if energy is not None else 0.,fallback='none') for j in range(len(ids))]
                        else:
                            images=[];links=[]
                            groups=[[i] for i in ids] if stage in ('common500','new100') else [ids]
                            for group in groups:
                                noises=np.stack([state['sp'].standard_noise(pop.ids[i],seed,1024,snr) for i in group])
                                recon,link=state['sr'].physical_batch(state['model'],pop.batch(group),1024,snr,noises)
                                images.extend(recon.cpu().numpy());links.extend(link)
                    for i,image,link in zip(ids,images,links):
                        target,prepared=references[i];values=evaluator.score(target,image,0,prepared);actual+=1
                        assert all(math.isfinite(float(v)) for v in values.values())
                        rows.append(dict(method=method,branch=branch,checkpoint_step=cp['step'],source_index=i,source_id=pop.ids[i],N=0 if spec.get('mode')=='direct_F' else 1024,snr_db=snr,noise_seed=seed,**values,**link,reference_sha256=metric.array_sha(target),reconstruction_sha256=metric.array_sha(image),reused=False))
                    b.write(dest/'status.json',dict(status='SCORING',pid=os.getpid(),method=method,rows=len(all_rows)+len(rows),total_rows=frames,reused=reuse_count,new_scores=actual,elapsed_seconds=time.monotonic()-start))
        expected=n*len(snrs)*len(seeds);assert len(rows)==expected
        assert len({(int(x['source_index']),int(x['snr_db']),int(x['noise_seed'])) for x in rows})==expected
        if stage=='milestones':
            calibration=b.RESULT/('P_continue' if branch=='P' else 'Swin_1to19')/'calibration'/f"{cp['step']:06d}.csv"
            original={(int(x['source_index']),int(x['snr_db']),int(x['seed'])):x for x in csv_rows(calibration) if int(x['N'])==1024}
            matched=0;max_mse=0.;max_lpips=0.
            for row in rows:
                old=original.get((int(row['source_index']),int(row['snr_db']),int(row['noise_seed'])))
                if old is None:continue
                assert old['source_id']==row['source_id']
                error=abs(10.**(-float(row['psnr_db'])/10)-float(old['mse']));max_mse=max(max_mse,error)
                assert error<1e-6,'Original calibration MSE mismatch'
                assert str(row['header_accepted'])==old['header_accepted'],'Actual header status mismatch'
                if branch=='P':
                    error=abs(float(row['lpips_alex'])-float(old['lpips']));max_lpips=max(max_lpips,error)
                    assert error<2e-5,'Original calibration LPIPS mismatch'
                matched+=1
            assert matched==(15000 if branch=='P' else 6000)
            overlap_validation[method]=dict(matched=matched,max_mse_error=max_mse,max_lpips_error=max_lpips)
        fields=list(rows[0]);assert all(set(x)==set(fields) for x in rows)
        with (dest/f'{method}.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        all_rows.extend(rows);print('COMPLETE_METHOD',method,len(rows),flush=True)
    assert len(all_rows)==frames and actual+reuse_count==frames
    summaries=[]
    for spec in methods:
        own=[r for r in all_rows if r['method']==spec['name']]
        groups=collections.defaultdict(list)
        for row in own:groups[int(row['snr_db']),int(row['source_index'])].append(row)
        for snr in sorted({int(r['snr_db']) for r in own}):
            for key in METRICS:
                values=[math.fsum(float(r[key]) for r in groups[snr,i])/len(groups[snr,i]) for i in range(n)]
                summaries.append(dict(method=spec['name'],snr_db=snr,metric=key,mean=math.fsum(values)/n,source_count=n,noise_count=len([r for r in own if int(r['snr_db'])==snr and int(r['source_index'])==0]),ci_computed=False))
    with (dest/'descriptive_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    for path,digest in reg['code_bindings'].items():assert b.sha(path)==digest
    done=dict(status='COMPLETE_FROZEN_FRAME_SCORING',stage=stage,frames=frames,source_count=n,new_scores=actual,reused_scores=reuse_count,counts=dict(ledger.counts),original_calibration_overlap=overlap_validation,seconds=time.monotonic()-start,training_updates=0,bootstrap_calls=0,selection_or_stopping_input=False,all_failures_retained=True,outputs={str(p):b.sha(p) for p in dest.glob('*.csv')})
    b.write(dest/'completion.json',done);b.write(dest/'status.json',done)

if __name__=='__main__':
    try:main()
    except BaseException:
        b.write(OUT/(sys.argv[1]+'_failure.json'),dict(status='STOPPED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc()));raise
