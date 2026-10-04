"""Frozen metric reference levels; explicit CPU/CUDA execution, one row/source."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from step0_cache_export import read, require, sha, seal, identity, pixel_hash
from step0_reference_prepare import CONDITIONS, admitted


def update(path,value):
    import json
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,p)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def verify(bindings):
    for p,h in bindings.items():require(sha(p)==h,'Frozen reference dependency changed: '+str(p))


def numeric_flags(torch):
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())


def load_suite(root,device):
    import torch
    root=Path(root).resolve()
    queue,legacy,bindings=admitted(root,'LEGACY_N3060_FINAL')
    shared=queue['shared_metric_bindings'];verify(shared)
    bindings.update(shared)
    source=root/'experiments/unified-metrics-20261002/metric_models.py'
    require(shared.get(str(source))==sha(source),'Metric implementation is not the frozen unified source')
    regpath=root/'results/unified_metrics_20261002/metrics_registration.json'
    registered=read(regpath)
    manifest=root/'outputs/UNIFIED-METRICS-20261002/modelmanifest.json'
    qualification_path=root/'outputs/UNIFIED-METRICS-20261002/models_qualification.json'
    qualification=read(qualification_path)
    require(registered['synthetic'] is False and registered['modelmanifest_sha256']==sha(manifest)
        and qualification['status']=='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
        and qualification['modelmanifest_sha256']==sha(manifest),'Unified metric model qualification differs')
    verify(qualification['source_bindings']);bindings.update(qualification['source_bindings'])
    bindings.update({str(p):sha(p) for p in (regpath,manifest,qualification_path)})
    flags=registered['numerical_runtime']
    torch.set_num_interop_threads(flags['interop_threads']);torch.set_num_threads(flags['threads'])
    torch.set_float32_matmul_precision(flags['precision'])
    torch.backends.cuda.matmul.allow_tf32=flags['matmul_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_tf32']
    torch.backends.cudnn.benchmark=flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic'])
    require(numeric_flags(torch)==flags,'Unable to establish original metric numerical flags')
    metric=module(source,'_registered_reference_metric_models')
    evaluator=metric.MetricEvaluator(manifest,device=device)
    require(all(x['status']=='READY' for x in evaluator.metadata['metrics'].values()),'A registered metric is unavailable')
    # CPU is a separately registered backend; same assets/preprocessing, no claim
    # that its numerical output is bitwise equal to prior CUDA values.
    cp=root/'outputs/HISTORICAL-METRICS-20261003/cached_studies.json'
    require(legacy['bindings'].get(str(cp))==sha(cp),'Legacy quality specification changed')
    bindings[str(cp)]=sha(cp)
    quality=read(cp)['studies']['LEGACY_N3060_FINAL']['quality']
    def path(value):
        p=Path(value)
        return (p if p.is_absolute() else root/p).resolve()
    quality_path=path(quality['module_path']);paths={k:str(path(v)) for k,v in quality['paths'].items()}
    require(legacy['bindings'].get(str(quality_path))==sha(quality_path),'Original quality implementation is not bound')
    for p,h in legacy['bindings'].items():
        resolved=Path(p).resolve()
        if (str(resolved) in (str(quality_path),paths['alexnet_checkpoint'],paths['dino_checkpoint'])
                or Path(paths['dino_source']).resolve() in resolved.parents):
            require(sha(p)==h,'Legacy quality weight/source differs');bindings[p]=h
    require(all(p in bindings for p in (str(quality_path),paths['alexnet_checkpoint'],paths['dino_checkpoint'])),
            'Missing original LPIPS/S14 weight proof')
    native=module(quality_path,'_registered_reference_native_quality')
    lpips,dino,linear=native.load_quality_models(paths,torch.device(device))
    require(sha(linear)==quality['lpips_linear_sha256'],'LPIPS linear weights differ')
    bindings[str(Path(linear).resolve())]=sha(linear)
    require(numeric_flags(torch)==flags,'Metric construction changed numerical flags')
    return evaluator,metric,native,lpips,dino,bindings,flags


def clean_metrics(native,new,probability):
    result=dict(new)
    require(type(probability) in (int,float) and math.isfinite(probability) and 0<=probability<=1,'Invalid top1 probability')
    for key in ('lpips_alex','dino_cosine'):
        require(math.isfinite(native[key]),'Nonfinite original reference metric')
        result[key]=float(native[key])
    psnr=float(native['psnr_db'])
    require(math.isfinite(psnr) or psnr==math.inf,'Invalid PSNR')
    result.update(psnr_db=psnr if math.isfinite(psnr) else None,psnr_infinite=psnr==math.inf,
        resnet50_top1_probability=float(probability),semantic_error=1-int(new['resnet50_top1_source_prediction']),
        confidently_wrong=int(probability>=.5 and new['resnet50_prediction']!=new['resnet50_source_prediction']))
    return result


def validate_checkpoint(cp,registration,entry):
    require(cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'}), 'Reference checkpoint checksum differs')
    require(cp['registration_sha256']==identity(registration) and cp['source_index']==entry['source_index']
            and cp['input_cache_sha256']==entry['cache']['sha256'],'Reference checkpoint source/runtime differs')
    rows=cp['rows']
    require(len(rows)==6 and [r['condition'] for r in rows]==list(CONDITIONS),'Reference condition coverage differs')
    for row,digest in zip(rows,entry['cache']['image_sha256']):
        require(row['source_id']==entry['source_id'] and row['reference_sha256']==entry['reference_sha256']
                and row['image_sha256']==digest and row['class_index']==entry['class_index'],'Reference scientific identity differs')
        require(row['N'] is None and row['snr_db'] is None and row['noise_seed'] is None,'Reference cannot masquerade as channel result')
        require(row['semantic_error']==int(row['resnet50_prediction']!=row['resnet50_source_prediction'])
            and row['confidently_wrong']==int(row['resnet50_top1_probability']>=.5 and row['semantic_error']),
            'Semantic reference metric differs')
        require(row['psnr_infinite']==(row['condition']=='identity') and (row['psnr_db'] is None)==row['psnr_infinite'],
                'Identity PSNR infinity must be explicit')
    return rows


def analyze(rows):
    require(len(rows)==600,'Final reference report needs all 600 measured rows')
    summary=[];boot=np.random.default_rng(20261002).integers(0,100,(10000,100))
    metrics=('psnr_db','lpips_alex','dino_cosine','clip_image_cosine','dists','dreamsim','dinov2_vitl14_cosine',
        'ms_ssim','resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')
    for condition in CONDITIONS:
        group=sorted([r for r in rows if r['condition']==condition],key=lambda r:r['source_index'])
        require([r['source_index'] for r in group]==list(range(100)),'Source reference coverage differs')
        for metric in metrics:
            if condition=='identity' and metric=='psnr_db':
                summary.append(dict(condition=condition,metric=metric,mean=None,ci_low=None,ci_high=None,
                                    positive_infinity=True,sources=100));continue
            values=np.asarray([r[metric] for r in group],dtype=np.float64)
            require(np.isfinite(values).all(),'Nonfinite reference summary metric')
            means=values[boot].mean(1);lo,hi=np.quantile(means,[.025,.975])
            summary.append(dict(condition=condition,metric=metric,mean=float(values.mean()),ci_low=float(lo),ci_high=float(hi),
                                positive_infinity=False,sources=100))
    return summary


def write_csv(path,rows):
    require(rows,'No reference rows')
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(root,prepared_path,out,device,limit_sources=100):
    require(device=='cpu' or device.startswith('cuda:'),'Explicit cpu or cuda:index device required')
    require(1<=limit_sources<=100,'Limit sources must be 1..100')
    root,out=Path(root).resolve(),Path(out).resolve()
    allowed=root/'outputs/EXTERNAL-COMPARISON-20261004'
    require(out!=root and (root not in out.parents or out==allowed or allowed in out.parents),'Reference output must be separate from historical inputs')
    out.mkdir(parents=True,exist_ok=True)
    prepared=read(prepared_path)
    require(prepared['status']=='SIX_REFERENCE_INPUTS_COMPLETE' and prepared['rows']==600
            and prepared['conditions']==list(CONDITIONS) and len(prepared['caches'])==100,'Reference input coverage differs')
    verify(prepared['input_bindings']);verify(prepared['source_bindings'])
    import torch
    if device.startswith('cuda:'):
        require(torch.cuda.is_available(),'Explicit CUDA requested but unavailable')
    started=time.time()
    evaluator,metric,native,lpips,dino,bindings,flags=load_suite(root,device)
    own={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('step0_*.py')}
    registration=dict(status='REGISTERED_SIX_REFERENCE_METRICS',target_sources=100,rows=600,
        conditions=list(CONDITIONS),prepared_sha256=sha(prepared_path),model_bindings=bindings,
        evaluator_identity=evaluator.identity(),evaluator_metadata=evaluator.metadata,numerical_flags=flags,
        source_bindings=own,device=str(device),torch_version=torch.__version__,batch_size=6,
        source_reference_batch=1,model_inference_only=True,holdout_access=False,noise_triplication=False,
        label_conditioned=False,reference_only=True,
        note='Same-class donor uses source label for reference selection; no reference is a deployed communication method.',
        numerical_note='Frozen weights/preprocessing/FP32 flags; CPU backend is explicitly distinct from prior CUDA execution.')
    seal(out/'registration.json',registration)
    timing=dict(model_load_seconds=time.time()-started,device=device,threads=torch.get_num_threads())
    allrows=[];checkpoints=[]
    for entry in prepared['caches'][:limit_sources]:
        index=entry['source_index'];p=out/'source_checkpoints'/f'{index:04d}.json'
        if p.exists():
            cp=read(p);allrows.extend(validate_checkpoint(cp,registration,entry));checkpoints.append(p);continue
        cache=entry['cache'];require(sha(cache['path'])==cache['sha256'],'Reference RGB cache changed')
        with np.load(cache['path'],allow_pickle=False) as archive:images=archive['images']
        require(images.shape==(6,3,256,256) and [pixel_hash(x) for x in images]==cache['image_sha256'],'Reference RGB pixels differ')
        target=images[0]
        require(pixel_hash(target)==entry['reference_sha256'],'Reference target differs')
        begin=time.time()
        with torch.inference_mode():
            x=torch.from_numpy(target[None].copy())
            prepared_features=evaluator.prepare_reference(x)
            repeated,features=evaluator.expand_reference(x,prepared_features,6)
            newer=evaluator.score(repeated,torch.from_numpy(images),[entry['class_index']]*6,False,features)
            older=native.quality_metrics(target,list(images),lpips,dino,torch.device(device))[0]
            logits=evaluator.models['resnet50'](metric.preprocess_resnet50(torch.from_numpy(images).to(device)))
            probability,prediction=logits.float().softmax(-1).max(-1)
            probabilities=probability.cpu().tolist();predictions=prediction.cpu().tolist()
        require(predictions==[r['resnet50_prediction'] for r in newer],'R50 confidence forward prediction differs')
        require(numeric_flags(torch)==flags,'Metric scoring changed numerical flags')
        require(abs(older[0]['lpips_alex'])<1e-6 and abs(newer[0]['clip_image_cosine']-1)<1e-5
                and abs(newer[0]['dinov2_vitl14_cosine']-1)<1e-5
                and newer[0]['resnet50_top1_source_prediction']==1,'Identity reference sanity check failed')
        rows=[]
        for condition,old,new,prob,digest in zip(CONDITIONS,older,newer,probabilities,cache['image_sha256']):
            values=clean_metrics(old,new,float(prob))
            rows.append(dict(source_index=index,source_id=entry['source_id'],class_index=entry['class_index'],
                condition=condition,N=None,snr_db=None,noise_seed=None,reference_only=True,
                image_sha256=digest,reference_sha256=entry['reference_sha256'],**values))
        cp=dict(registration_sha256=identity(registration),source_index=index,input_cache_sha256=cache['sha256'],
                rows=rows,elapsed_seconds=time.time()-begin)
        cp['payload_sha256']=identity(cp);validate_checkpoint(cp,registration,entry);seal(p,cp)
        allrows.extend(rows);checkpoints.append(p)
        update(out/'status.json',dict(status='SCORING',sources=index+1,target_sources=100,rows=len(allrows),
                                     last_source_seconds=cp['elapsed_seconds'],device=device))
        print('REFERENCE_SOURCE_COMPLETE',index,len(rows),round(cp['elapsed_seconds'],3),flush=True)
    timing['sources']=len(checkpoints);timing['rows']=len(allrows)
    timing['source_seconds']=[read(p)['elapsed_seconds'] for p in checkpoints]
    timing['median_source_seconds']=float(np.median(timing['source_seconds']))
    timing['estimated_100_source_seconds']=100*timing['median_source_seconds']
    update(out/'performance.json',timing)
    if limit_sources<100:
        update(out/'status.json',dict(status='PARTIAL_BENCHMARK',sources=len(checkpoints),rows=len(allrows),target_rows=600,
                                     device=device,completion_claim=False))
        return timing
    verify(bindings);verify(own)
    summary=analyze(allrows)
    write_csv(out/'per_source.csv',allrows);write_csv(out/'summary.csv',summary)
    text=['# Metric reference levels','',
        '100 fixed development originals, six conditions, one paired observation per source. No channel or noise triplication.',
        'Same-class donors come only from the original training/calibration populations. All donor identities were selected before scoring.',
        'These values illustrate the scale of each metric; they are not tuned thresholds or communication results.',
        f'Execution device: {device}. Weights and preprocessing match the frozen suite; backend identity is recorded separately.',
        '', '| Condition | PSNR | LPIPS | DINOv2-L | Semantic error | Confidently wrong |',
        '|---|---:|---:|---:|---:|---:|']
    lookup={(r['condition'],r['metric']):r for r in summary}
    for condition in CONDITIONS:
        values=[]
        for name in ('psnr_db','lpips_alex','dinov2_vitl14_cosine','semantic_error','confidently_wrong'):
            value=lookup[condition,name]
            values.append('+infinity (identical pixels)' if value['positive_infinity'] else
                          f"{value['mean']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]")
        text.append('| '+condition+' | '+' | '.join(values)+' |')
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    completion=dict(status='SIX_REFERENCE_METRICS_COMPLETE',sources=100,rows=600,conditions=list(CONDITIONS),
        registration_sha256=sha(out/'registration.json'),prepared_sha256=sha(prepared_path),
        outputs={str(p):sha(p) for p in [*checkpoints,out/'per_source.csv',out/'summary.csv',out/'report.md']},
        bootstrap_replicates=10000,bootstrap_seed=20261002,statistical_unit='source_image',noise_triplication=False,
        holdout_access=False,training_updates=0,policy_selection_updates=0)
    seal(out/'completion.json',completion);update(out/'status.json',dict(status='COMPLETE',sources=100,rows=600,device=device))
    return completion


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',required=True,help='Explicit cpu or cuda:index; never auto-selects a GPU')
    p.add_argument('--limit-sources',type=int,default=100)
    args=p.parse_args()
    run(args.root,args.prepared,args.out,args.device,args.limit_sources)
