"""Fixed calibration-only checkpoint comparison, independent of live training."""
from pathlib import Path
import collections, csv, fcntl, hashlib, json, math, os, subprocess, sys, time, traceback

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(ROOT/'experiments/baseline-strength-20261011'))
sys.path.insert(0,str(ROOT/'experiments/entropy-partial-mismatch-20261010/scripts'))
import numpy as np
import torch
import baseline_runtime as b
import run_continue as training
import h800_ep_pilot_metrics_v1 as metric

OUT=ROOT/'results/baseline_strength_20261011/interim_cal100_v1'
TEMP=b.TMP/'interim_cal100_v1'
SNRS=[1,4,7,10,13,19]
SEEDS=[4101,4102,4103]
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
METHODS=[('P40k','P',40000),('P80k','P',80000),('Swin80k','Swin',80000),('Swin150k','Swin',150000)]

def guard():
    if (OUT/'STOP').exists():raise RuntimeError('Evaluation stop requested; training unaffected')
    if time.monotonic()-START>10800:raise RuntimeError('Three-hour finite diagnostic budget exceeded')

class Ledger:
    def __init__(self):self.counts=collections.Counter()
    def call(self,name,fn,**kw):
        caps={'model_constructions':3,'reference_preparations':100,'image_scores':7200,
              'dinov2_vitl14_reference':100,'convnext_reference':100,
              'dinov2_vitl14_reconstruction':7200,'convnext_reconstruction':7200,
              'lpips_pair':7200,'lpips_alexnet_backbone_forward':14400}
        assert self.counts[name]<caps[name],name
        result=fn();self.counts[name]+=1;return result

def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global START
    START=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);TEMP.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'registration.json').exists(),'No duplicate or automatic rerun'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
    free,temp=map(int,gpu.strip().split(','));assert free>20000 and temp<80
    torch.cuda.set_per_process_memory_fraction((16<<30)/torch.cuda.get_device_properties(0).total_memory)
    pins={str(Path(__file__)):b.sha(__file__),str(Path(training.__file__)):b.sha(training.__file__),str(Path(b.__file__)):b.sha(b.__file__),str(Path(metric.__file__)):b.sha(metric.__file__)}
    request=b.read(b.RT/'qualification/h800_ep_new100_metrics_v3_attempt1/registered/request.json')
    config={k:v for k,v in request['metric_config'].items() if k in ['sources','manifest','registration','weight_receipt','dino_receipt','whole_policy']}
    del request
    config['runtime_identity']={'purpose':'calibration-only interim baseline diagnostic','physical_gpu':0,'torch':torch.__version__,'cuda':torch.version.cuda,'old_host_numeric_identity_claimed':False}
    pop=b.Population('calibration',latent=True,limit=100)
    checkpoints={}
    for name,branch,step in METHODS:
        if name=='P40k':p=next((b.CP/'parents/p').glob('*.pt'));sha=b.P_SHA
        elif name=='Swin80k':p=b.CP/'parents/swin/model_080000.pt';sha=b.S_SHA
        else:
            h=b.read(b.RESULT/('P_continue' if branch=='P' else 'Swin_1to19')/'calibration_history.json')
            cp=next(x['checkpoint'] for x in h if x['step']==step);p=Path(cp['path']);sha=cp['sha256']
        assert b.sha(p)==sha;checkpoints[name]={'path':str(p),'sha256':sha,'step':step}
    registration={'schema':'BASELINE_INTERIM_CAL100_V1','population':'original_calibration_first100','source_ids':pop.ids,
                  'source_count':100,'noise_seeds':SEEDS,'snrs_db':SNRS,'N':1024,'method_frames':7200,
                  'checkpoints':checkpoints,'data_bindings':pop.bindings,'source_bindings':pins,
                  'metrics':METRICS,'metric_config':config,'training_updates':0,'bootstrap_calls':0,
                  'holdout_images_read':False,'selection_or_stopping_input':False,'training_processes_paused':False,
                  'reconstruction_batch':4,'metric_batch':1,'source_label_used':False,
                  'label_api_placeholder':0,'label_api_note':'The legacy DINO-only API validates a label argument; it is unused by all four reported metrics and never supplied to reconstruction.',
                  'old_Swin_19dB_outside_training_and_calibration_range':True,
                  'P10dB_diagnostic_only':True,'is_full1000_milestone_diagnostic':False,
                  'physical_gpu':0,'allocator_cap_bytes':16<<30,'started_unix':time.time()}
    b.write(OUT/'registration.json',registration)
    ledger=Ledger();evaluator=metric.FourMetrics(config,ledger,guard)
    evaluator.evaluator.metadata['metric_selection_usage']='Interim calibration-only baseline display; no model selection or stopping use'
    evaluator.metadata.update(population=registration['population'],selection_objective=None,policy_selection=False,
                              DINO_L_metadata=evaluator.evaluator.metadata)
    evaluator.identity=metric.core.digest(evaluator.metadata)
    b.write(OUT/'metric_identity.json',dict(identity=evaluator.identity,metadata=evaluator.metadata))
    references={}
    with torch.inference_mode():
        for i in range(100):
            x=pop.batch([i],device='cpu')[0].numpy();references[i]=(x,evaluator.prepare(x))
    rows=[];state=None;current=None
    for method,branch,step in METHODS:
        if current!=branch:
            if state is not None:del state;torch.cuda.empty_cache()
            state=training.load(branch);current=branch
            state['model'].eval().requires_grad_(False)
            # No training optimizer is created. The legacy P loader supplies a frozen
            # LPIPS used by training; this diagnostic uses the frozen metric provider.
            if branch=='P':del state['lp'];torch.cuda.empty_cache()
        if method not in ('P40k','Swin80k'):
            value=torch.load(checkpoints[method]['path'],map_location='cpu',weights_only=True)
            state['model'].load_state_dict(value['model'],strict=True);del value
        state['model'].eval();started=time.monotonic()
        with torch.inference_mode():
            for snr in SNRS:
                for seed in SEEDS:
                    for first in range(0,100,4):
                        guard();ids=list(range(first,min(first+4,100)))
                        if branch=='P':
                            batch=pop.p_batch(ids,[snr]*len(ids),[seed]*len(ids),state['ns']['seeded_noise'])
                            z,wave=state['model'](batch['F'],batch['snr'],batch['noise'])
                            assert tuple(wave.shape)==(len(ids),1024,2)
                            energy=wave.square().sum((1,2));assert torch.allclose(energy,energy.new_full(energy.shape,2048),rtol=1e-5,atol=.02)
                            images=state['dc'](z).cpu().numpy()
                            ledgers=[dict(header_accepted=True,header_crc_accepted=True,header_fields_legal=True,actual_energy=float(v),fallback='none') for v in energy.cpu()]
                        else:
                            noises=np.stack([state['sp'].standard_noise(pop.ids[i],seed,1024,snr) for i in ids])
                            reconstruction,ledgers=state['sr'].physical_batch(state['model'],pop.batch(ids),1024,snr,noises)
                            images=reconstruction.cpu().numpy()
                        for i,image,link in zip(ids,images,ledgers):
                            target,prepared=references[i];values=evaluator.score(target,image,0,prepared)
                            rows.append(dict(method=method,branch=branch,checkpoint_step=step,source_index=i,source_id=pop.ids[i],N=1024,snr_db=snr,noise_seed=seed,
                                             **values,**link,reference_sha256=metric.array_sha(target),reconstruction_sha256=metric.array_sha(image)))
                    b.write(OUT/'status.json',dict(status='SCORING',pid=os.getpid(),method=method,snr_db=snr,noise_seed=seed,rows=len(rows),target_rows=7200,elapsed_seconds=time.monotonic()-START))
                print('COMPLETE_CELL',method,snr,len(rows),flush=True)
        own=[x for x in rows if x['method']==method];assert len(own)==1800
        write_csv(OUT/(method+'_per_frame.csv'),own)
        b.write(OUT/(method+'_completion.json'),dict(method=method,frames=len(own),seconds=time.monotonic()-started,checkpoint=checkpoints[method],header_failures=sum(not x['header_accepted'] for x in own)))
    assert len(rows)==7200 and len({(x['method'],x['source_index'],x['snr_db'],x['noise_seed']) for x in rows})==7200
    summary=[];by_source={}
    for method,_,_ in METHODS:
        for snr in SNRS:
            for key in METRICS:
                vals=[]
                for i in range(100):
                    a=[float(x[key]) for x in rows if x['method']==method and x['snr_db']==snr and x['source_index']==i];assert len(a)==3
                    value=math.fsum(a)/3;by_source[method,snr,key,i]=value;vals.append(value)
                summary.append(dict(method=method,snr_db=snr,metric=key,mean=math.fsum(vals)/100,source_count=100,noise_count=3,ci_computed=False))
    paired=[]
    for method,reference in [('P80k','P40k'),('Swin150k','Swin80k')]:
        for snr in SNRS:
            for key in METRICS:
                values=[by_source[method,snr,key,i]-by_source[reference,snr,key,i] for i in range(100)]
                paired.append(dict(method=method,reference=reference,snr_db=snr,metric=key,mean=math.fsum(values)/100,source_count=100,noise_count=3,delta_definition='method - reference',ci_computed=False))
    write_csv(OUT/'summary.csv',summary);write_csv(OUT/'paired.csv',paired)
    for p,sha in pins.items():assert b.sha(p)==sha
    for cp in checkpoints.values():assert b.sha(cp['path'])==cp['sha256']
    assert ledger.counts['image_scores']==7200 and ledger.counts['reference_preparations']==100
    done=dict(status='COMPLETE_CALIBRATION_ONLY_INTERIM',frames=7200,source_count=100,counts=dict(ledger.counts),seconds=time.monotonic()-START,
              training_updates=0,bootstrap_calls=0,holdout_images_read=False,selection_or_stopping_input=False,all_failures_retained=True)
    b.write(OUT/'completion.json',done);b.write(OUT/'status.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);b.write(OUT/'failure.json',dict(status='DIAGNOSTIC_FAILED_TRAINING_UNCHANGED',traceback=traceback.format_exc()));raise
