"""Add only frozen ResNet50 probabilities to the selected 6,600 history rows.

CPU-only, six threads. Every inferred argmax must equal the existing historical
measurement. Float reconstructions, source identities and original metrics are
read-only. A mismatch stops the run; it never triggers an alternative image,
backend, preprocessing, or batch retry.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from step0_cache_export import (read,sha,seal,require,identity,pixel_hash,
    verified_arrays,STUDIES)
from step0_reference_prepare import admitted
from step0_statistics import select

HERE=Path(__file__).resolve().parent
BOOTSTRAP_SEED=20261002
THRESHOLD=.5


class SafePause(Exception): pass


def update(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def verify(bindings):
    for path,digest in bindings.items():
        require(Path(path).is_file() and sha(path)==digest,'Bound confidence input changed: '+path)


def seal_payload(path,value):
    value=dict(value,payload_sha256=identity(value));seal(path,value);return value


def check_payload(value):
    require(value.get('payload_sha256')==identity({k:v for k,v in value.items() if k!='payload_sha256'}),
        'Confidence checkpoint checksum differs')


def serialized(row):
    return {k:'' if v is None else str(v) for k,v in row.items()}


def load_scope(root,statistics_out):
    statistics_out=Path(statistics_out).resolve()
    donepath=statistics_out/'completion.json';done=read(donepath)
    require(done.get('status')=='STEP0_STATISTICS_COMPLETE' and done.get('rows')==6600
        and done.get('cells')==22 and done.get('model_inference') is False,'Completed selected statistics required')
    source=HERE/'step0_statistics.py'
    require(done['code_sha256']==sha(source),'Frozen history selection implementation differs')
    verify(done['input_bindings'])
    csvpath=statistics_out/'per_frame.csv'
    require(done['outputs'].get('per_frame.csv')==sha(csvpath),'Selected history CSV changed')
    with csvpath.open(encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    require(len(rows)==6600 and len({r['history_row_id'] for r in rows})==6600,'Historical selection coverage differs')
    require(collections.Counter(int(r['source_index']) for r in rows)=={i:66 for i in range(100)},
        'Each selected source must have exactly 66 registered rows')
    expected={r['history_row_id']:r for r in rows}
    inputs={**done['input_bindings'],str(donepath):sha(donepath),str(csvpath):sha(csvpath)}
    studies={};canonical=None;queue=None
    for study in STUDIES:
        queue,receipt,bindings=admitted(root,study);inputs.update(bindings)
        item=queue['inherited_studies'][study]
        folder=root/'outputs/HISTORICAL-METRICS-R2-20261003'/study
        result=root/'results/historical_metrics_r2_20261003'/study
        require(item['out']==str(folder) and item['result']==str(result),'Historical source folder differs')
        rp=result/'registration.json'
        require(receipt['bindings'].get(str(rp))==sha(rp),'Historical score registration changed')
        inputs[str(rp)]=sha(rp);registration=read(rp)
        require(registration.get('synthetic') is False and len(registration['source_identity'])==100,
            'Real complete source registration is required')
        if canonical is None: canonical=registration['source_identity']
        require([(r['image_id'],r['class_index']) for r in registration['source_identity']]
            ==[(r['image_id'],r['class_index']) for r in canonical],'Paired source/class identities differ')
        studies[study]=dict(folder=folder,receipt=receipt,registration=registration,registration_path=rp)
    return expected,studies,canonical,queue,inputs


def model_inputs(root,queue,studies):
    source=root/'experiments/unified-metrics-20261002/metric_models.py'
    manifest=root/'outputs/UNIFIED-METRICS-20261002/modelmanifest.json'
    qualified=root/'outputs/UNIFIED-METRICS-20261002/models_qualification.json'
    require(queue['shared_metric_bindings'].get(str(source))==sha(source),'Unregistered metric source')
    value=read(manifest);qualification=read(qualified)
    require(qualification.get('status')=='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
        and qualification['modelmanifest_sha256']==sha(manifest),'Original evaluator qualification differs')
    for spec in studies.values():
        require(spec['registration']['modelmanifest_sha256']==sha(manifest),'Historical evaluator weights differ')
    weight=value['resnet50']['weights'];path=Path(weight['path']).resolve()
    require(weight['sha256'].startswith('11ad3fa6') and sha(path)==weight['sha256'],
        'Only the registered ResNet50 ImageNet1K V2 weights are admitted')
    bindings={str(p):sha(p) for p in (source,manifest,qualified,path)}
    return source,value,bindings


def load_resnet(source,manifest):
    # Set before importing Torch; this process cannot claim or initialize a GPU.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(6);torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(),'CPU confidence process already initialized CUDA')
    spec=importlib.util.spec_from_file_location('_historical_confidence_metric_models',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    # Deliberately do not construct MetricEvaluator or any other metric model.
    model=module.load_resnet50(manifest['resnet50'],torch.device('cpu'))
    require(not model.training and all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters()),
        'ResNet50 must remain frozen on CPU')
    import torchvision
    source_bindings={}
    for name in ('torchvision.models.resnet','torchvision.transforms._presets',
        'torchvision.transforms.functional','torchvision.transforms._functional_tensor'):
        imported=sys.modules.get(name)
        require(imported is not None and Path(imported.__file__).is_file(),'Missing actual classifier implementation: '+name)
        path=Path(imported.__file__).resolve();source_bindings[str(path)]=sha(path)
    runtime=dict(device='cpu',threads=6,interop_threads=2,precision='float32',AMP=False,
        TF32=False,deterministic=True,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,
        preprocessing='existing metric_models.preprocess_resnet50; tensor RGB without quantization',
        numerical_note='CPU backend; no bitwise equivalence claim to old CUDA logits. Every argmax must equal the old measured prediction.')
    return torch,model,module.preprocess_resnet50,runtime,source_bindings


def source_files(index,studies):
    bindings={}
    for spec in studies.values():
        for path in (spec['folder']/'source_checkpoints'/f'{index:04d}.json',spec['folder']/'reconstructions'/f'{index:04d}.npz'):
            digest=sha(path)
            require(spec['receipt']['bindings'].get(str(path))==digest,'Admitted source cache changed: '+str(path))
            bindings[str(path)]=digest
    return bindings


def historical_predictions(index,studies,expected):
    result={}
    for study,spec in studies.items():
        checkpoint=read(spec['folder']/'source_checkpoints'/f'{index:04d}.json')
        for row in checkpoint['rows']:
            chosen=select(row,study)
            if chosen is None: continue
            require(chosen['history_row_id'] in expected and serialized(chosen)==expected[chosen['history_row_id']],
                'Source checkpoint differs from completed selected statistics')
            result[chosen['history_row_id']]=dict(prediction=int(row['new_resnet50_prediction']),
                source_prediction=int(row['new_resnet50_source_prediction']),
                image_sha256=row['history_image_sha256'],reference_sha256=row['history_reference_sha256'])
    require(len(result)==66,'Historical prediction coverage differs')
    return result


def selected_source(index,studies,canonical,expected):
    images={};selected=[]
    reference=dict(source_index=index,**canonical[index])
    for study,spec in studies.items():
        cp=read(spec['folder']/'source_checkpoints'/f'{index:04d}.json')
        cache=spec['folder']/'reconstructions'/f'{index:04d}.npz'
        target,array,slots=verified_arrays(cp,spec['registration'],reference,study,cache)
        targetsha=pixel_hash(target)
        images.setdefault(targetsha,target.copy())
        for old in cp['rows']:
            chosen=select(old,study)
            if chosen is None: continue
            require(chosen['history_row_id'] in expected and serialized(chosen)==expected[chosen['history_row_id']],
                'Actual cached row differs from the selected completed statistics')
            digest=old['history_image_sha256']
            require(targetsha==old['history_reference_sha256'],'Original reference hash differs')
            if digest not in images: images[digest]=array[slots[old['history_row_id']]].copy()
            selected.append(dict(selected=chosen,history_study=study,image_sha256=digest,reference_sha256=targetsha,
                expected_prediction=int(old['new_resnet50_prediction']),
                expected_source_prediction=int(old['new_resnet50_source_prediction']),
                class_index=int(old['history_true_class_index'])))
        del target,array
    wanted={key for key,row in expected.items() if int(row['source_index'])==index}
    require(len(selected)==len(wanted)==66 and {r['selected']['history_row_id'] for r in selected}==wanted,
        'Selected source does not exactly cover its 66 history rows')
    return sorted(selected,key=lambda r:r['selected']['history_row_id']),images


def check_probability(record,registration_sha,digest):
    check_payload(record)
    require(record.get('registration_sha256')==registration_sha and record.get('image_sha256')==digest,
        'Cached classifier runtime or float-image identity differs')
    require(type(record.get('prediction')) is int and 0<=record['prediction']<1000
        and type(record.get('probability')) in (float,int) and math.isfinite(record['probability'])
        and 0<=record['probability']<=1,'Invalid classifier probability')
    return record


def enrich(row,prediction,original):
    require(prediction['prediction']==row['expected_prediction'],
        'CPU reconstruction argmax differs from old metrics: '+row['selected']['history_row_id'])
    require(original['prediction']==row['expected_source_prediction'],
        'CPU original-image argmax differs from old metrics: '+row['selected']['history_row_id'])
    for value in (prediction['probability'],original['probability']):
        require(math.isfinite(value) and 0<=value<=1,'Invalid softmax probability')
    error=int(prediction['prediction']!=original['prediction'])
    require(error==row['selected']['semantic_error'],'Original semantic-error label differs')
    return dict(**row['selected'],history_study=row['history_study'],
        image_sha256=row['image_sha256'],reference_sha256=row['reference_sha256'],class_index=row['class_index'],
        resnet50_prediction=prediction['prediction'],resnet50_source_prediction=original['prediction'],
        resnet50_top1_probability=float(prediction['probability']),
        resnet50_source_top1_probability=float(original['probability']),
        confidently_wrong=int(prediction['probability']>=THRESHOLD and error),
        confidence_threshold=THRESHOLD,argmax_matches_historical=True,classification_main_eligible=not row['selected']['true_class_paid'])


def check_source(record,registration_sha,index,bindings,expected,historical):
    check_payload(record)
    require(record['registration_sha256']==registration_sha and record['source_index']==index
        and record['input_bindings']==bindings,'Confidence source checkpoint identity differs')
    rows=record['rows'];wanted={k for k,r in expected.items() if int(r['source_index'])==index}
    require(len(rows)==66 and {r['history_row_id'] for r in rows}==wanted,'Confidence checkpoint rows differ')
    for row in rows:
        original=expected[row['history_row_id']]
        predictions=historical[row['history_row_id']]
        require(serialized({key:row[key] for key in original})==original,'Copied historical value changed')
        require(row['argmax_matches_historical'] is True and row['confidence_threshold']==THRESHOLD
            and row['semantic_error']==int(row['resnet50_prediction']!=row['resnet50_source_prediction'])
            and row['confidently_wrong']==int(row['resnet50_top1_probability']>=THRESHOLD and row['semantic_error']),
            'Cached semantic confidence metric differs')
        require(row['resnet50_prediction']==predictions['prediction']
            and row['resnet50_source_prediction']==predictions['source_prediction']
            and row['image_sha256']==predictions['image_sha256']
            and row['reference_sha256']==predictions['reference_sha256'],'Cached predictions or image hashes differ from old metrics')
        for key in ('resnet50_top1_probability','resnet50_source_top1_probability'):
            require(math.isfinite(row[key]) and 0<=row[key]<=1,'Invalid cached probability')
    return rows


def analyze(rows):
    require(len(rows)==6600 and len({r['history_row_id'] for r in rows})==6600,'All 6600 rows are required')
    groups=collections.defaultdict(list)
    for row in rows: groups[row['group'],row['N'],row['snr_db']].append(row)
    require(len(groups)==22,'Historical confidence group scope differs')
    boot=np.random.default_rng(BOOTSTRAP_SEED).integers(0,100,(10000,100))
    metrics=('confidently_wrong','semantic_error','resnet50_top1_probability','resnet50_source_top1_probability')
    summary=[];means={}
    for key,cell in sorted(groups.items()):
        require(len(cell)==300 and {(r['source_index'],r['noise_seed']) for r in cell}
            =={(i,s) for i in range(100) for s in (2001,2002,2003)},'Missing paired source/noise coverage')
        for metric in metrics:
            values=np.array([math.fsum(r[metric] for r in cell if r['source_index']==i)/3 for i in range(100)])
            require(np.isfinite(values).all(),'Nonfinite source-level metric')
            means[key,metric]=values;lo,hi=np.quantile(values[boot].mean(1),[.025,.975])
            summary.append(dict(group=key[0],N=key[1],snr_db=key[2],metric=metric,mean=float(values.mean()),
                ci_low=float(lo),ci_high=float(hi),sources=100,frames=300,confidence_threshold=THRESHOLD,
                decoder=cell[0]['decoder'],true_class_paid=cell[0]['true_class_paid'],
                classification_main_eligible=not cell[0]['true_class_paid'],training_seed=cell[0]['training_seed']))
    contrasts=[]
    for group,N,snr in sorted(groups):
        if group=='P': continue
        for metric in metrics:
            delta=means[(group,N,snr),metric]-means[('P',N,snr),metric]
            lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
            contrasts.append(dict(method=group,reference='P',N=N,snr_db=snr,metric=metric,
                mean_difference=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),sources=100,frames_per_arm=300,
                contrast='method minus P; source-paired bootstrap',true_class_paid=True,
                classification_main_eligible=False,same_decoder=not group.startswith('Legacy'),
                interpretation='descriptive paid-class/protocol comparison; no causal bandwidth attribution'))
    return summary,contrasts


def write_csv(path,rows):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def run(args):
    import fcntl
    root,out=Path(args.root).resolve(),Path(args.out).resolve()
    allowed=root/'outputs/EXTERNAL-COMPARISON-20261004'
    require(allowed in out.parents,'Confidence output must be its own subdirectory of the new external run')
    require(out!=Path(args.statistics_out).resolve(),'Never overwrite existing statistics')
    require(type(args.batch_size) is int and 1<=args.batch_size<=16,'Explicit CPU batch size1..16 required')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=dict(root=str(root),statistics_out=str(Path(args.statistics_out).resolve()),out=str(out),batch_size=args.batch_size)
    if (out/'completion.json').exists():
        done=read(out/'completion.json');require(done.get('request')==request,'Completed confidence request differs')
        verify(done['input_bindings']);verify(done['source_bindings']);verify(done['outputs']);verify(done['probability_bindings'])
        print('HISTORICAL_CONFIDENCE_COMPLETE verified',flush=True);return
    require(not (out/'failure.json').exists(),'Previous confidence failure requires review, not automatic retry')
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    last_complete=-1
    try:
        expected,studies,canonical,queue,inputs=load_scope(root,Path(args.statistics_out))
        source,manifest,model_bindings=model_inputs(root,queue,studies);inputs.update(model_bindings)
        torch,model,preprocess,runtime,implementation=load_resnet(source,manifest)
        own={str(HERE/name):sha(HERE/name) for name in ('external_history_confidence.py',
            'step0_cache_export.py','step0_reference_prepare.py','step0_reference_protocol.py','step0_statistics.py')}
        own.update(implementation)
        registration=dict(status='REGISTERED_HISTORICAL_CONFIDENCE',request=request,input_bindings=inputs,
            source_bindings=own,model='torchvision ResNet50_Weights.IMAGENET1K_V2',runtime=runtime,
            rows=6600,cells=22,sources=100,noise_seeds=[2001,2002,2003],snrs=[7,13],
            confidence_threshold=THRESHOLD,bootstrap=10000,bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_unit='average three noises per source before paired resampling of 100 sources',
            only_model_loaded='resnet50',argmax_mismatch_policy='stop; no alternate image/backend/batch retry',
            original_inputs_modified=False,training_updates=0,policy_selection_updates=0,synthetic=False)
        seal(out/'registration.json',registration);regsha=identity(registration)
        parameters=tuple(model.parameters());versions=tuple(p._version for p in parameters)
        allrows=[];checkpoint_paths=[];probability_paths={};inference_count=0;all_source_inputs={}
        for index in range(100):
            if stop[0]: raise SafePause()
            cp_path=out/'source_checkpoints'/f'{index:04d}.json'
            bindings=source_files(index,studies)
            all_source_inputs.update(bindings)
            historical=historical_predictions(index,studies,expected)
            if cp_path.exists():
                cp=read(cp_path);rows=check_source(cp,regsha,index,bindings,expected,historical)
                verify(cp['probability_bindings']);probability_paths.update(cp['probability_bindings'])
                cached={}
                for path in cp['probability_bindings']:
                    digest=Path(path).stem;cached[digest]=check_probability(read(path),regsha,digest)
                for row in rows:
                    require(cached[row['image_sha256']]['prediction']==row['resnet50_prediction']
                        and cached[row['image_sha256']]['probability']==row['resnet50_top1_probability']
                        and cached[row['reference_sha256']]['prediction']==row['resnet50_source_prediction']
                        and cached[row['reference_sha256']]['probability']==row['resnet50_source_top1_probability'],
                        'Saved source rows differ from the qualified probability cache')
                allrows.extend(rows);checkpoint_paths.append(cp_path);last_complete=index;continue
            chosen,images=selected_source(index,studies,canonical,expected)
            probabilities={};paths={}
            for digest in images:
                path=out/'probabilities'/digest[:2]/(digest+'.json')
                if path.exists(): probabilities[digest]=check_probability(read(path),regsha,digest)
                paths[digest]=path
            missing=[digest for digest in images if digest not in probabilities]
            for begin in range(0,len(missing),args.batch_size):
                if stop[0]: raise SafePause()
                keys=missing[begin:begin+args.batch_size]
                with torch.inference_mode():
                    logits=model(preprocess(torch.from_numpy(np.stack([images[key] for key in keys]))))
                    require(tuple(logits.shape)==(len(keys),1000) and bool(torch.isfinite(logits).all()),'Invalid R50 logits')
                    probs,preds=logits.float().softmax(-1).max(-1)
                    values=list(zip(preds.tolist(),probs.tolist()))
                for digest,(prediction,probability) in zip(keys,values):
                    value=dict(registration_sha256=regsha,image_sha256=digest,prediction=int(prediction),
                        probability=float(probability),device='cpu',model='ResNet50_IMAGENET1K_V2')
                    probabilities[digest]=seal_payload(paths[digest],value)
                inference_count+=len(keys)
            rows=[enrich(row,probabilities[row['image_sha256']],probabilities[row['reference_sha256']]) for row in chosen]
            # Confirm inputs stayed fixed while the CPU classifier ran.
            verify(bindings)
            proof={str(path):sha(path) for path in paths.values()}
            cp=seal_payload(cp_path,dict(status='HISTORICAL_SOURCE_CONFIDENCE_COMPLETE',registration_sha256=regsha,
                source_index=index,source_id=canonical[index]['image_id'],input_bindings=bindings,
                rows=rows,unique_float_images=len(images),probability_bindings=proof,argmax_all_match=True,synthetic=False))
            check_source(cp,regsha,index,bindings,expected,historical)
            allrows.extend(rows);checkpoint_paths.append(cp_path);probability_paths.update(proof);last_complete=index
            require(tuple(p._version for p in parameters)==versions and all(p.grad is None for p in parameters)
                and not model.training and not torch.cuda.is_initialized(),'Frozen CPU classifier state changed')
            update(out/'status.json',dict(status='RUNNING',source_complete=index+1,sources=100,rows=len(allrows),
                new_unique_inferences_this_process=inference_count,device='cpu',threads=6,safe_pause_handler_installed=True,pid=os.getpid()))
            print('HISTORICAL_CONFIDENCE_SOURCE',index+1,len(rows),'unique',len(images),flush=True)
            del images,chosen
        summary,contrasts=analyze(allrows)
        files={'per_frame.csv':allrows,'summary.csv':summary,'paired_vs_P.csv':contrasts}
        for name,rows in files.items(): write_csv(out/name,rows)
        references={}
        for row in allrows:
            key=row['source_index'],row['reference_sha256']
            item=dict(source_index=row['source_index'],source_id=row['source_id'],class_index=row['class_index'],
                reference_sha256=row['reference_sha256'],resnet50_source_prediction=row['resnet50_source_prediction'],
                resnet50_source_top1_probability=row['resnet50_source_top1_probability'],
                source_top1_label=int(row['resnet50_source_prediction']==row['class_index']))
            require(key not in references or references[key]==item,'Shared original-image confidence differs')
            references[key]=item
        write_csv(out/'source_confidence.csv',list(references.values()))
        text=['# Historical classification confidence','',
            'These are the same 6,600 measured frames selected in Step 0. Only frozen ResNet-50 ImageNet1K V2 softmax inference was added, on CPU with six threads.',
            'Confident disagreement means reconstruction top-1 probability >= 0.5 and reconstruction prediction differs from the classifier prediction on the original image. Softmax confidence is not a calibrated probability of correctness.',
            'Every original and reconstruction argmax exactly matched its historical measurement. Any mismatch stops the run; no image, preprocessing, batch or backend substitution is allowed.',
            'First average the three registered noise repetitions for each source. Confidence intervals and paired differences use the same 10,000 bootstrap resamples of 100 sources (seed 20261002).',
            'Paid-class digital rows are descriptive: the generation pipeline receives the true class. D0 and Dc remain separate protocols. These comparisons do not isolate a causal effect of bandwidth.',
            'Source confidence is reported for each exact historical float reference. Original uint8 images match; legacy normalization may retain its registered 1-ULP float difference.',
            '', '| Method | N | SNR | Confident disagreement [95% CI] | Original top-1 confidence |',
            '|---|---:|---:|---:|---:|']
        lookup={(r['group'],r['N'],r['snr_db'],r['metric']):r for r in summary}
        for row in summary:
            if row['metric']!='confidently_wrong': continue
            reference=lookup[row['group'],row['N'],row['snr_db'],'resnet50_source_top1_probability']
            text.append(f"| {row['group']} | {row['N']} | {row['snr_db']} | {100*row['mean']:.1f}% [{100*row['ci_low']:.1f}, {100*row['ci_high']:.1f}] | {reference['mean']:.4f} |")
        (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        verify(inputs);verify(all_source_inputs);verify(own)
        artifacts=[out/name for name in (*files,'source_confidence.csv','report.md','registration.json')]
        outputs={str(path):sha(path) for path in [*artifacts,*checkpoint_paths]}
        seal(out/'completion.json',dict(status='HISTORICAL_CONFIDENCE_COMPLETE',request=request,rows=6600,cells=22,sources=100,
            argmax_all_match=True,confidence_threshold=THRESHOLD,bootstrap=10000,bootstrap_source_unit=True,
            input_bindings={**inputs,**all_source_inputs},source_bindings=own,outputs=outputs,probability_bindings=probability_paths,
            unique_float_images_scored=len(probability_paths),only_model_loaded='resnet50',device='cpu',threads=6,
            original_inputs_modified=False,training_updates=0,policy_selection_updates=0,synthetic=False))
        update(out/'status.json',dict(status='HISTORICAL_CONFIDENCE_COMPLETE',rows=6600,sources=100,device='cpu'))
    except SafePause:
        update(out/'status.json',dict(status='PAUSED',completed_source_index=last_complete,device='cpu',safe_to_resume=True))
        raise SystemExit(75)
    except BaseException as error:
        update(out/'failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),completed_source_index=last_complete,
            no_automatic_retry=True,no_image_or_backend_substitution=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--statistics-out',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--batch-size',type=int,default=8)
    run(parser.parse_args())
