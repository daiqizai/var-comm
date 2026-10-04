"""Score completed N2048 M1 float caches with the frozen unified metric suite.

Calibration never imports this module. A policy and all 100 development sources
must already be sealed. No image reconstruction or model selection is performed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np

METHODS=('whole_policy','raster_policy','random_policy','entropy_policy','oracle_policy',
         'raster_at_entropy','random_at_entropy','oracle_at_entropy')
SNRS=(1,4,7,13,19); SEEDS=(2001,2002,2003); PHYS=('QPSK','16QAM')
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine',
         'dists','dreamsim','ms_ssim','dino_specificity','resnet50_top1_label',
         'resnet50_top1_source_prediction','semantic_error','confidently_wrong')
STOP=False

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def ident(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def seal(path,v):
    if Path(path).exists():require(read(path)==v,'Immutable receipt differs: '+str(path))
    else:write(path,v)
def verify(bindings):
    for p,h in bindings.items():require(sha(p)==h,'Bound input differs: '+str(p))
def pixels(a):
    a=np.asarray(a);require(a.dtype==np.float32 and a.shape==(3,256,256),'RGB precision/shape differs')
    require(np.isfinite(a).all() and a.min()>=0 and a.max()<=1,'Invalid RGB');return np.ascontiguousarray(a)
def rgb_sha(a):return hashlib.sha256(b'float32:3,256,256:RGB\0'+pixels(a).tobytes()).hexdigest()
def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def pause():
    if STOP:raise InterruptedError('Requested pause at a source boundary')
def key(r):return (int(r['source_index']),r['phy_family'],int(r['snr_db']),int(r['noise_seed']),r['method'])

def validate_source(cp,index):
    require(cp['source_index']==index and cp['payload_sha256']==ident({k:v for k,v in cp.items() if k!='payload_sha256'}),'Source checkpoint seal differs')
    rows=cp['rows'];expected={(index,p,s,n,m) for p in PHYS for s in SNRS for n in SEEDS for m in METHODS}
    require(len(rows)==240 and {key(r) for r in rows}==expected,'Development action/seed coverage differs')
    proof=cp['float_reconstructions'];require(sha(proof['path'])==proof['sha256'],'Development archive differs')
    with np.load(proof['path'],allow_pickle=False) as z:
        require(set(z.files)=={'images','source_rgb','row_ids','image_slots'},'Float archive schema differs')
        images=z['images'].copy();target=pixels(z['source_rgb']).copy();slots=z['image_slots'].tolist();ids=z['row_ids'].tolist()
    require(images.dtype==np.float32 and images.ndim==4 and images.shape[1:]==(3,256,256),'Invalid image archive')
    require(len(slots)==240 and ids==[r['replay_row_id'] for r in rows] and len(set(ids))==240,'Archive row mapping differs')
    hashes=[rgb_sha(x) for x in images];ref=rgb_sha(target)
    for r,j in zip(rows,slots):
        require(type(j) is int and 0<=j<len(images),'Invalid image slot')
        require(r['image_sha256']==hashes[j] and r['reference_sha256']==ref and int(r['N'])==2048,'Float reconstruction/reference identity differs')
        require(r.get('label_conditioned',False) is False,'No class-conditioned receiver allowed')
        require(r['source_id']==cp['record']['image_id'] and r['preprocessing_id']==cp['record']['preprocessing_id'],'Source identity differs')
    return rows,images,target,slots

def interval(v,draws):
    v=np.asarray(v,dtype=np.float64);require(v.shape==(100,) and np.isfinite(v).all(),'Incomplete source bootstrap')
    lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
    return dict(mean=float(v.mean()),ci_low=float(lo),ci_high=float(hi),n_sources=100,n_frames=300)

def analyze(rows):
    require(len(rows)==24000 and len({key(r) for r in rows})==24000,'Full 24,000 development rows required')
    groups=defaultdict(list)
    for r in rows:groups[r['phy_family'],int(r['snr_db']),r['method']].append(r)
    draws=np.random.default_rng(20261002).integers(0,100,(10000,100));summary=[];paired=[];means={}
    extra=('E','latent_sq_err_final','zero_erasure_proxy_sq_error','F_sq_error_zero_erasure_proxy')
    for (phy,snr,method),rr in sorted(groups.items()):
        require(len(rr)==300,'Incomplete scored group');by={(int(r['source_index']),int(r['noise_seed'])):r for r in rr}
        require(set(by)=={(i,n) for i in range(100) for n in SEEDS},'Missing paired source/noise frame')
        for metric in METRICS+extra:
            vals=[by[i,n].get(metric) for i in range(100) for n in SEEDS]
            if any(v in (None,'') for v in vals):continue
            values=np.array(vals,dtype=np.float64).reshape(100,3).mean(1)
            means[phy,snr,method,metric]=values
            summary.append(dict(N=2048,phy_family=phy,snr_db=snr,method=method,metric=metric,**interval(values,draws)))
        for field in ('header_ok','prefix_crc_ok','body_crc_ok','partial_used','prefix_false_accept','latent_valid'):
            if not all(field in r for r in rr):continue
            vals=np.array([float(bool(by[i,n][field])) for i in range(100) for n in SEEDS]).reshape(100,3).mean(1)
            summary.append(dict(N=2048,phy_family=phy,snr_db=snr,method=method,metric=field+'_fraction',**interval(vals,draws)))
    contrasts=[(m,'whole_policy') for m in METHODS[1:]]+ [('entropy_policy',m) for m in METHODS[5:]]
    for phy in PHYS:
        for snr in SNRS:
            for a,b in contrasts:
                for met in METRICS+extra:
                    if (phy,snr,a,met) not in means or (phy,snr,b,met) not in means:continue
                    paired.append(dict(N=2048,phy_family=phy,snr_db=snr,method_A=a,method_B=b,metric=met,
                        **interval(means[phy,snr,a,met]-means[phy,snr,b,met],draws),
                        comparison_scope='same_source_and_same_registered_channel_noise_vector',delta='A_minus_B'))
    return summary,paired

def historical_controls(root,targets,records,modelmanifest_sha):
    """Copy completed final P and M1 rows; no historical decoder is rerun."""
    from step0_reference_prepare import admitted
    _,receipt,bindings=admitted(root,'FINAL_P2048_P3060')
    rp=root/'results/historical_metrics_r2_20261003/FINAL_P2048_P3060/registration.json'
    require(receipt['bindings'].get(str(rp))==sha(rp),'Historical P registration changed')
    reg=read(rp);bindings[str(rp)]=sha(rp);rows=[]
    for i in range(100):
        p=root/'outputs/HISTORICAL-METRICS-R2-20261003/FINAL_P2048_P3060/source_checkpoints'/f'{i:04d}.json'
        require(receipt['bindings'].get(str(p))==sha(p),'Historical P source not admitted')
        cp=read(p);bindings[str(p)]=sha(p)
        require(cp['binding']==ident(reg) and cp['payload_sha256']==ident({k:v for k,v in cp.items() if k!='payload_sha256'}),'Historical P source seal differs')
        selected=[]
        for r in cp['rows']:
            meta=json.loads(r['history_metadata_json'])
            if meta.get('method_id')!='P2048':continue
            require(meta['N']==2048 and meta['decoder']=='Dc' and meta['training_seed']==2026092304
                and meta['label_conditioned'] is False,'Different historical P selection')
            require(r['history_source_id']==records[i]['image_id'] and r['history_preprocessing_id']==records[i]['preprocessing_id']
                and r['history_reference_sha256']==rgb_sha(targets[i]) and r['history_modelmanifest_sha256']==modelmanifest_sha,
                'P reference/model population differs; do not mix incompatible metrics')
            row=dict(N=2048,source_index=i,source_id=records[i]['image_id'],preprocessing_id=records[i]['preprocessing_id'],
                snr_db=int(float(r['snr_db'])),noise_seed=int(r['noise_seed']),method='P2048',phy_family='continuous',
                E=float(r['E']),training_seed=2026092304,selected_step=int(r['selected_step']),historical=True,
                reference_sha256=r['history_reference_sha256'],image_sha256=r['history_image_sha256'])
            for m in METRICS:
                field=m if m in ('psnr_db','lpips_alex','dino_cosine') else 'new_'+m
                row[m]=None if r.get(field) in (None,'') else float(r[field])
            row['semantic_error']=1-row['resnet50_top1_source_prediction']
            selected.append(row)
        require(len(selected)==15 and {(x['snr_db'],x['noise_seed']) for x in selected}=={(s,n) for s in SNRS for n in SEEDS},'P2048 historical coverage differs')
        rows.extend(selected)
    # Original N512/N1024 M1 and P: freeze the published source checkpoint inventory.
    folder=root/'results/unified_metrics_20261002';invp=folder/'scoring_inventory.json';regp=folder/'metrics_registration.json'
    completion=root/'outputs/UNIFIED-METRICS-20261002/scoring_completion.json';old_done=read(completion)
    require(old_done['outputs'].get(str(invp))==sha(invp) and old_done['outputs'].get(str(regp))==sha(regp),'Unified source inventory changed')
    inventory=read(invp);oldreg=read(regp)
    for p in (completion,invp,regp):bindings[str(p)]=sha(p)
    for i in range(100):
        p=root/'outputs/UNIFIED-METRICS-20261002/source_checkpoints'/f'{i:03d}.json'
        require(inventory['source_checkpoint_sha256'].get(str(p))==sha(p),'Published old M1 source changed')
        cp=read(p);bindings[str(p)]=sha(p)
        require(cp['binding']==ident(oldreg) and cp['payload_sha256']==ident({k:v for k,v in cp.items() if k!='payload_sha256'}),'Published old metric seal differs')
        for r in cp['rows']:
            if r.get('experiment') not in ('M1','N512','N1024'):continue
            n=int(float(r.get('N',0) or 0))
            if n not in (512,1024) or r.get('snr_db') not in SNRS+tuple(str(s) for s in SNRS):continue
            if r.get('experiment')=='M1' and r.get('method') not in ('entropy_policy','whole_policy'):continue
            if r.get('experiment')!='M1' and r.get('method')!='P'+str(n):continue
            require(r['reference_sha256']==rgb_sha(targets[i]) and r['source_id']==records[i]['image_id']
                and r['preprocessing_id']==records[i]['preprocessing_id'],'Old M1/P reference population differs')
            require(r['decoder_id']=='Dc' and str(r['label_conditioned']).lower()=='false','Incompatible historical decoder/class information')
            row=dict(N=n,source_index=i,source_id=r['source_id'],preprocessing_id=r['preprocessing_id'],
                snr_db=int(r['snr_db']),noise_seed=int(r['noise_seed']),method=r['method'],
                phy_family=r['phy_family'] if r['experiment']=='M1' else 'continuous',historical=True,
                E=float(r['E']),reference_sha256=r['reference_sha256'],image_sha256=r['image_sha256'])
            for m in METRICS:row[m]=None if r.get(m) in (None,'') else float(r[m])
            if row['dino_specificity'] is None and r.get('dino_mismatched') not in (None,''):
                row['dino_specificity']=row['dino_cosine']-float(r['dino_mismatched'])
            row['semantic_error']=1-row['resnet50_top1_source_prediction'];rows.append(row)
    return rows,bindings

def resource_analysis(rows,history):
    groups=defaultdict(list)
    for r in rows+history:
        if not r.get('historical') and r['method'] not in ('entropy_policy','whole_policy'):continue
        groups[int(r['N']),r['phy_family'],int(r['snr_db']),r['method']].append(r)
    draws=np.random.default_rng(20261002).integers(0,100,(10000,100));summary=[];means={};paired=[]
    for g,rr in sorted(groups.items()):
        by={(int(r['source_index']),int(r['noise_seed'])):r for r in rr}
        require(len(rr)==len(by)==300 and set(by)=={(i,n) for i in range(100) for n in SEEDS},'Incomplete resource curve point')
        for m in METRICS+('E',):
            vals=[by[i,n].get(m) for i in range(100) for n in SEEDS]
            if any(v in (None,'') for v in vals):continue
            arr=np.array(vals,dtype=np.float64).reshape(100,3).mean(1);means[g,m]=arr
            summary.append(dict(N=g[0],phy_family=g[1],snr_db=g[2],method=g[3],metric=m,**interval(arr,draws)))
    for n in (512,1024,2048):
        for phy in PHYS:
            for snr in SNRS:
                for method in ('entropy_policy','whole_policy'):
                    a=(n,phy,snr,method);b=(n,'continuous',snr,'P'+str(n))
                    for m in METRICS:
                        if (a,m) not in means or (b,m) not in means:continue
                        paired.append(dict(N=n,phy_family=phy,snr_db=snr,method_A=method,method_B=b[3],metric=m,
                            **interval(means[a,m]-means[b,m],draws),delta='A_minus_B',
                            comparison_scope='same_source_and_nominal_noise_seed_distinct_channel_noise_namespace',
                            energy_scope='QPSK_and_P_fixed_2N' if phy=='QPSK' else '16QAM_actual_E_separately_reported'))
    return summary,paired

def score(root,out):
    root=Path(root).resolve();out=Path(out).resolve();metrics=out/'metrics';metrics.mkdir(exist_ok=True)
    runtime=root/'outputs/EXTERNAL-COMPARISON-20261004/runtime';shared=root/'experiments/unified-metrics-20261002'
    for p in (runtime,shared):sys.path.insert(0,str(p))
    donepath=out/'development_completion.json';devregpath=out/'development_registration.json'
    require(donepath.exists(),'Complete development is required before metric evaluation')
    done=read(donepath);devreg=read(devregpath)
    require(done['status']=='COMPLETE' and done['sources']==100 and done['method_rows']==24000
        and done['registration_sha256']==sha(out/'registration.json')
        and done['outputs'].get(str(devregpath))==sha(devregpath)
        and done['policy_sha256']==devreg['policy_sha256']==sha(out/'m1_policy.json'),
        'Development completion or frozen policy/registration differs')
    inputs={str(donepath):sha(donepath),str(devregpath):sha(devregpath),str(Path(__file__).resolve()):sha(__file__)}
    checkpoints=[];targets=[];records=[]
    for i in range(100):
        p=out/'development/source_checkpoints'/f'{i:04d}.json';cp=read(p)
        require(done['outputs'].get(str(p))==sha(p),'Unbound development checkpoint')
        require(cp['binding']==ident(devreg),'Development registration binding differs')
        raw,images,target,slots=validate_source(cp,i);del images
        checkpoints.append(p);targets.append(target);records.append(cp['record']);inputs[str(p)]=sha(p)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from step0_reference_metrics import load_suite,numeric_flags
    import torch
    import runner
    import replay
    from batch_speed import qualify_and_select_batch,qualified_chunks
    evaluator,metric_module,native,lpips,dino,model_bindings,flags=load_suite(root,'cuda:0')
    manifest=root/'outputs/UNIFIED-METRICS-20261002/modelmanifest.json';manifest_sha=sha(manifest)
    batch=qualify_and_select_batch(evaluator,torch.device('cuda:0'),manifest,metrics,
        runtime_context=dict(study='M1_N2048_FULL_20261004',development_sha256=sha(donepath),model_bindings=model_bindings,numerical_runtime=flags))
    registration=dict(version='M1_N2048_UNIFIED_METRICS_R1',sources=100,rows=24000,methods=list(METHODS),snrs=list(SNRS),
        phys=list(PHYS),noise_seeds=list(SEEDS),input_bindings=inputs,model_bindings=model_bindings,numerical_runtime=flags,
        metric_evaluator_identity=evaluator.identity(),modelmanifest_sha256=manifest_sha,
        source_records=records,metric_batch_qualification_sha256=sha(metrics/'metric_batch_qualification.json'),
        batch_size=batch['chosen_batch_size'],qualified_batch_sizes=batch['qualified_batch_sizes'],
        bootstrap_unit='source_mean_over_three_noise_repeats',bootstrap_seed=20261002,bootstrap_replicates=10000,
        DINO='DINOv2 ViT-S/14',additional_DINO='DINOv2 ViT-L/14',selection_used_new_metrics=False,
        training_updates=0,holdout_access=False,KID='DEFERRED_HOLDOUT',FID='NOT_EVALUATED')
    seal(metrics/'registration.json',registration);seal(metrics/'model_metadata.json',evaluator.metadata);binding=ident(registration)
    features=[]
    for target in targets:
        pause()
        with torch.no_grad():features.append(native.dino_features(dino,torch.from_numpy(target[None]).to('cuda:0'))[0].cpu().numpy())
    features=np.stack(features);mismatch=replay.ReplayEngine._derangement();allrows=[];outputs={};started=time.monotonic()
    for i,(p,record) in enumerate(zip(checkpoints,records)):
        pause();require(sha(p)==inputs[str(p)] and numeric_flags(torch)==flags,'Input or numeric runtime changed')
        cp=read(p);raw,images,target,slots=validate_source(cp,i);dest=metrics/'source_checkpoints'/f'{i:04d}.json'
        if dest.exists():
            saved=read(dest)
            require(saved['binding']==binding and saved['input_checkpoint_sha256']==sha(p)
                and saved['payload_sha256']==ident({k:v for k,v in saved.items() if k!='payload_sha256'}),'Metric resume receipt differs')
        else:
            truth=int(record['class_index']);target_tensor=torch.from_numpy(target[None]);prepared=evaluator.prepare_reference(target_tensor)
            prediction=int(prepared['resnet50'][0]);legacy,observed,embeddings=native.quality_metrics(target,list(images),lpips,dino,torch.device('cuda:0'))
            np.testing.assert_allclose(observed,features[i],rtol=1e-6,atol=1e-6)
            with torch.no_grad():
                negative=torch.from_numpy(features[mismatch[i]][None]).to('cuda:0').expand(len(images),-1)
                mismatched=torch.nn.functional.cosine_similarity(torch.from_numpy(embeddings).to('cuda:0'),negative,dim=1).cpu().tolist()
            scorer=runner.PendingMetricScorer(evaluator,torch,target_tensor,prepared,truth,batch['chosen_batch_size'],batch['qualified_batch_sizes'])
            rows=[]
            for r,j in zip(raw,slots):
                value=dict(r,**legacy[j],dino_mismatched=mismatched[j],dino_specificity=legacy[j]['dino_cosine']-mismatched[j],
                    native_quality_json=json.dumps({k:r.get(k) for k in ('psnr_db','lpips_alex','dino_cosine','dino_mismatched')},sort_keys=True),
                    mismatch_source_id=records[mismatch[i]]['image_id'],modelmanifest_sha256=manifest_sha)
                require(type(r['latent_valid']) is bool,'Explicit latent availability required')
                value['F_sq_error_zero_erasure_proxy']=float(r['latent_sq_err_final'] if r['latent_valid'] else r['zero_erasure_proxy_sq_error'])
                cachekey=replay.metric_cache_key(images[j],target,evaluator.identity());rows.append(value);scorer.add(cachekey,images[j],value)
            scorer.flush();confidence=[];offset=0
            for count in qualified_chunks(len(images),batch['chosen_batch_size'],batch['qualified_batch_sizes']):
                with torch.inference_mode():
                    rgb=torch.from_numpy(images[offset:offset+count]).to('cuda:0')
                    prob,pred=evaluator.models['resnet50'](metric_module.preprocess_resnet50(rgb)).float().softmax(-1).max(-1)
                    confidence.extend(zip(prob.cpu().tolist(),pred.cpu().tolist()))
                offset+=count
            for r,j in zip(rows,slots):
                prob,pred=confidence[j];require(pred==r['resnet50_prediction'],'Classifier confidence forward differs')
                r.update(resnet50_top1_probability=prob,semantic_error=int(pred!=prediction),confidently_wrong=int(prob>=.5 and pred!=prediction))
                require(all(np.isfinite(float(r[k])) for k in METRICS),'Missing registered metric')
            saved=dict(binding=binding,source_index=i,input_checkpoint_sha256=sha(p),rows=rows,unique_images=len(images))
            saved['payload_sha256']=ident(saved);seal(dest,saved)
        require([key(r) for r in saved['rows']]==[key(r) for r in raw],'Scored source mapping differs')
        allrows.extend(saved['rows']);outputs[str(dest)]=sha(dest)
        write(out/'score_status.json',dict(status='RUNNING',pid=os.getpid(),sources=i+1,rows=len(allrows),elapsed_seconds=time.monotonic()-started))
    summary,paired=analyze(allrows)
    history,historical_bindings=historical_controls(root,targets,records,manifest_sha)
    resources,against_p=resource_analysis(allrows,history)
    seal(metrics/'historical_input_bindings.json',historical_bindings)
    for filename,rows in [('metrics_per_frame.csv.gz',allrows),('metrics_summary.csv',summary),('metrics_paired_intervals.csv',paired),
            ('historical_controls.csv.gz',history),('resource_summary.csv',resources),('m1_vs_P_paired.csv',against_p)]:
        dest=metrics/filename;csv_write(dest,rows);outputs[str(dest)]=sha(dest)
    verify(inputs);verify(model_bindings);verify(historical_bindings)
    for name in ('registration.json','model_metadata.json','historical_input_bindings.json','metric_batch_qualification.json'):
        p=metrics/name;outputs[str(p)]=sha(p)
    complete=dict(status='M1_N2048_METRICS_COMPLETE',sources=100,rows=24000,outputs=outputs,registration_sha256=sha(metrics/'registration.json'),
        input_bindings=inputs,elapsed_seconds=time.monotonic()-started,synthetic=False,training_updates=0)
    seal(out/'score_completion.json',complete);write(out/'score_status.json',dict(status='COMPLETE',sources=100,rows=24000));return complete

def main():
    global STOP
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    import fcntl
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);lock=(out/'score.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:score(a.root,out)
    except BaseException as e:
        write(out/'score_failure.json',dict(error=repr(e),automatic_retry=False));raise
    finally:lock.close()
if __name__=='__main__':main()
