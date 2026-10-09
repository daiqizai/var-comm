"""Only missing fixed-group HiFi/Swin ConvNeXt predictions from frozen float RGB.

No PSNR/LPIPS/DINO recomputation, reconstruction, channel, or statistics.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def pinned(d):
    require(sha(d['path'])==d['sha256'],'Pinned dependency changed: '+d['path']);return read(d['path'])
def load(d,name):
    require(sha(d['path'])==d['sha256'],'Executable changed')
    spec=importlib.util.spec_from_file_location(name,d['path']);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def csv_read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def run(s):
    import numpy as np
    core=load(s['core_module'],'_a5_frozen_core');r=pinned(s['main_request'])
    require(s['core_module']['sha256']==r['worker_sha256'],'Core/request revision mismatch')
    out=Path(r['out']);dest=out/'missing_convnext';dest.mkdir(parents=True,exist_ok=True)
    collection=read(out/'collection.json');oldplan=pinned(r['old_plan']);freeze=pinned(r['original_FINAL_FREEZE'])
    require(sha(out/'original_five_columns_metrics.csv')==collection['old_metrics_csv_sha256'],'Original metrics changed')
    require(sha(r['old_provenance']['path'])==r['old_provenance']['sha256'],'Original provenance changed')
    original=csv_read(out/'original_five_columns_metrics.csv');provenance=csv_read(r['old_provenance']['path'])
    missing=[v for v in original if v['metric_status'].startswith('MISSING')]
    require(0<len(missing)<=8 and len(missing)==collection['missing_original_metric_cells']
        and all(v['metric']=='convnext_top1_source_prediction' and v['method'] in ('HiFiDiffCom','SwinJSCC80k')
                and int(v['source_index']) in core.SELECTED and int(v['snr_db'])==13 and int(v['N'])==1024
                and int(v['noise_seed'])==2001 and v['value']=='' for v in missing),'Only eight-or-fewer explicit HiFi/Swin ConvNeXt gaps are allowed')
    keys={(v['source_index'],v['method'],v['metric']) for v in missing}
    require(len(keys)==len(missing),'Duplicate missing cells')
    if (dest/'completion.json').exists():
        completed=read(dest/'completion.json')
        require(completed['status']=='COMPLETE' and completed['original_metrics_sha256']==collection['old_metrics_csv_sha256'],'Wrong previous supplement')
        for p,h in completed['outputs'].items():require(sha(p)==h,'Completed supplement changed')
        print('Missing-only ConvNeXt already complete; no new calls');return
    require(not (out/'completion.json').exists(),'Main figure export already finalized; do not silently revise it')
    require(sha(out/'bpg16_float_inputs.npz')==collection['bpg_float_archive_sha256'],'Source float inputs changed')
    with np.load(out/'bpg16_float_inputs.npz',allow_pickle=False) as z:references=z['references'].copy()
    require(references.shape==(16,3,256,256) and references.dtype==np.float32,'Wrong source references')
    hdir=Path(r['root'])/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
    hd=read(hdir/'reconstruction_completion.json');metricdone=read(hdir/'completion.json')
    require(hd['status']=='HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE' and metricdone['status']=='HIFI_FIXED16_EVALUATION_COMPLETE','Actual historical completions required')
    inputs=[];source_ids={}
    for v in missing:
        index=int(v['source_index']);target=references[core.FIXED.index(index)]
        p=core.one([p for p in provenance if p['source_index']==v['source_index'] and p['method']==v['method']],'display provenance cell')
        require(p['source_id']==v['source_id'] and p['image_sha256']==v['image_sha256'],'Missing score does not describe displayed float RGB')
        cp=hdir/'source_checkpoints'/f'{index:04d}.json'
        require(hd['outputs'].get(str(cp))==oldplan['source_bindings'].get(str(cp))==sha(cp),'Original reconstruction checkpoint changed')
        hc=read(cp);arc=hc['float_reconstructions'];ap=p['archive']
        require(ap==arc['path'] and hd['outputs'].get(ap)==oldplan['source_bindings'].get(ap)==arc['sha256']==sha(ap),'Original float archive changed')
        expected_method={'HiFiDiffCom':'HiFiDiffCom_SwinJSCC','SwinJSCC80k':'SwinJSCC_new_shared'}[v['method']]
        row=core.one([x for x in hc['rows'] if x['method']==expected_method and int(x['N'])==1024 and int(x['snr_db'])==13
                      and int(x['noise_seed'])==2001 and x['source_id']==v['source_id'] and x['image_sha256']==v['image_sha256']], 'original shared-observation row')
        with np.load(ap,allow_pickle=False) as z:
            slot=int(p['image_slot']);image=z[p['key']][slot].copy();source=z['source_rgb'].copy()
            require(z['row_ids'].tolist()==[x['replay_row_id'] for x in hc['rows']]
                and int(z['image_slots'][hc['rows'].index(row)])==slot,'Historical image slot mismatch')
        require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all()
            and image.min()>=0 and image.max()<=1 and core.baseline_sha(image)==v['image_sha256']
            and np.array_equal(source,target) and core.baseline_sha(target)==row['reference_sha256'],'Float source/reconstruction identity mismatch')
        original_metric=Path(v['metric_source'])
        require(metricdone['outputs'].get(str(original_metric))==v['metric_source_sha256']==sha(original_metric),'Original missing-field metric CP changed')
        source_ids[index]=v['source_id'];inputs.append(dict(csv_row=v,image=image,archive=ap,archive_sha256=sha(ap)))
    require(len(source_ids)<=4,'Reference prediction cap exceeded')
    start=time.monotonic()
    def guard():
        require(time.monotonic()-start<s['max_seconds'],'Missing-only time budget exhausted')
        require(not any(Path(p).exists() for p in r['stop_files']),'Stop requested')
    guard()
    lock=Path(r['visual_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own=(dest/'worker.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Original CUBLAS setting required')
    import torch
    flags=freeze['holdout_metrics']['numerical_runtime']
    torch.set_num_threads(flags['threads']);torch.set_num_interop_threads(flags['interop_threads'])
    torch.set_float32_matmul_precision(flags['precision'])
    torch.backends.cuda.matmul.allow_tf32=flags['matmul_tf32'];torch.backends.cudnn.allow_tf32=flags['cudnn_tf32']
    torch.backends.cudnn.benchmark=flags['cudnn_benchmark'];torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic'])
    paths=pinned(r['metric_assets'])['paths'];bound=r['metric_bindings'];vpath=r['validation_module'];weights=paths['convnext_weights']
    require(bound.get(weights)=='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
            and bound.get(vpath)==sha(vpath),'Original classifier implementation/weight binding absent')
    validation=load(dict(path=vpath,sha256=bound[vpath]),'_a5_frozen_convnext_only')
    policy=r['original_frozen_policy']
    classifier=validation.ConvNeXtValidation(weights,bound[weights],device='cuda:0',stage='holdout',
        policy_path=policy['path'],policy_sha256=policy['sha256'])
    require(digest(classifier.identity)==digest(freeze['holdout_metrics']['metadata']['independent']),'Independent classifier identity changed')
    # The constructor retains its original admission namespace. The actual
    # sources and reported population here are development, never holdout.
    replacements={};ref_new_total=0;ref_reused_total=0;recon_new_total=0
    calls_now=0;references_now=0;checkpoint_reuse=0
    for index in core.SELECTED:
        if index not in source_ids:continue
        guard();target=references[core.FIXED.index(index)];selected=[x for x in inputs if int(x['csv_row']['source_index'])==index]
        cp=dest/'source_checkpoints'/f'{index:04d}.json'
        identity=dict(source_id=source_ids[index],source_index=index,reference_sha256=core.baseline_sha(target),
            images={x['csv_row']['method']:x['csv_row']['image_sha256'] for x in selected},
            independent_classifier_identity=digest(classifier.identity),original_metrics_sha256=collection['old_metrics_csv_sha256'])
        if cp.exists():
            result=read(cp);require(result['identity']==identity,'Existing missing-only checkpoint mismatch');checkpoint_reuse+=1
        else:
            reservation=cp.with_suffix('.reserved.json');require(not reservation.exists(),'Unresolved missing-only prediction reservation; do not duplicate')
            bpgcp=out/'bpg_metric_sources'/f'{index:04d}.json';source_prediction=None;source_origin='NEW_REFERENCE_PREDICTION';source_binding={}
            if bpgcp.exists():
                bpg=read(bpgcp);bi=bpg['identity']
                require(bi['source_id']==source_ids[index] and bi['reference_sha256']==core.baseline_sha(target)
                    and bi['metric_evaluator_identity']==freeze['holdout_metrics']['metadata_identity'],'BPG source prediction evaluator/reference mismatch')
                source_prediction=bpg['source_prediction'];source_origin='REUSED_A5_BPG_REFERENCE_PREDICTION'
                source_binding=dict(path=str(bpgcp),sha256=sha(bpgcp))
            core.write(reservation,dict(identity=identity,max_reconstruction_predictions=len(selected),max_reference_predictions=int(source_prediction is None)))
            reference_calls=int(source_prediction is None)
            if source_prediction is None:source_prediction=classifier.predict(target);references_now+=1
            require(type(source_prediction)==int and 0<=source_prediction<1000,'Bad reference prediction')
            values=[]
            for item in selected:
                guard();prediction=classifier.predict(item['image']);calls_now+=1
                values.append(dict(method=item['csv_row']['method'],image_sha256=item['csv_row']['image_sha256'],
                    reconstruction_prediction=prediction,convnext_top1_source_prediction=int(prediction==source_prediction),
                    archive=item['archive'],archive_sha256=item['archive_sha256']))
            result=dict(identity=identity,source_prediction=source_prediction,reference_origin=source_origin,
                reference_prediction_binding=source_binding,reference_predictions=reference_calls,
                reused_reference_predictions=1-reference_calls,reconstruction_predictions=len(values),rows=values,
                actual_population='development',used_for_selection=False)
            core.write(cp,result)
        ref_new_total+=result['reference_predictions'];ref_reused_total+=result['reused_reference_predictions'];recon_new_total+=result['reconstruction_predictions']
        for value in result['rows']:
            row=dict(core.one([x['csv_row'] for x in selected if x['csv_row']['method']==value['method']], 'missing metric cell'))
            row.update(value=str(value['convnext_top1_source_prediction']),metric_status='NEW_MISSING_ONLY_CONVNEXT',
                metric_source=str(cp),metric_source_sha256=sha(cp),metric_evaluator_identity=digest(classifier.identity))
            replacements[(row['source_index'],row['method'],row['metric'])]=row
        print(json.dumps(dict(source_index=index,reconstruction_predictions_this_invocation=calls_now,reference_predictions_this_invocation=references_now)),flush=True)
    require(set(replacements)==keys and recon_new_total==len(missing)<=8 and ref_new_total<=4,'Exact missing-only cap/count mismatch')
    completed=[replacements.get((row['source_index'],row['method'],row['metric']),row) for row in original]
    require(all(a==b for a,b in zip(original,completed) if (a['source_index'],a['method'],a['metric']) not in keys),'An existing metric was changed')
    csvpath=dest/'original_five_columns_metrics_completed.csv';core.write_csv(csvpath,completed)
    outputs={str(p):sha(p) for p in sorted((dest/'source_checkpoints').glob('*.json'))};outputs[str(csvpath)]=sha(csvpath)
    core.write(dest/'completion.json',dict(status='COMPLETE',schema='A5_MISSING_ONLY_CONVNEXT_COMPLETE_V1',
        original_metrics_sha256=collection['old_metrics_csv_sha256'],collection_sha256=sha(out/'collection.json'),
        resolved_missing_cells=len(missing),new_reconstruction_predictions=recon_new_total,new_reference_predictions=ref_new_total,
        reused_A5_BPG_reference_predictions=ref_reused_total,reconstruction_predictions_this_invocation=calls_now,
        reference_predictions_this_invocation=references_now,reused_source_checkpoints_this_invocation=checkpoint_reuse,
        actual_population='development',sources=list(source_ids),metric='convnext_top1_source_prediction',
        classifier_identity=classifier.identity,numerical_runtime=flags,existing_metrics_recomputed=0,
        new_reconstructions=0,new_PHY_calls=0,new_bootstrap=0,policy_selection=False,training_updates=0,
        elapsed_seconds=time.monotonic()-start,outputs=outputs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();s=read(a.request)
    require(s['schema']=='A5_MISSING_ONLY_CONVNEXT_REQUEST_V1' and s['worker_sha256']==sha(__file__),'Frozen missing-only script/request mismatch')
    require(s['max_seconds']<=600 and s['max_reconstruction_predictions']==8 and s['max_reference_predictions']==4,'Missing-only budget changed')
    run(s)


if __name__=='__main__':main()
