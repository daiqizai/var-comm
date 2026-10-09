"""Score only the actual fixed16 adaptive outputs, then export a separate figure.

The native256 six-column figure and every previously available metric are reused
read-only. Neither model reconstruction nor PHY is invoked by this script.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

METHOD='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'


def setup(path):
    r=json.loads(Path(path).read_text(encoding='utf-8-sig'));d=r['core_module']
    import hashlib
    h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if h(d['path'])!=d['sha256'] or h(__file__)!=r['worker_sha256']:raise RuntimeError('Frozen display worker changed')
    spec=importlib.util.spec_from_file_location('_a5_frozen_figure_core',d['path']);c=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=c;spec.loader.exec_module(c)
    c.require(r['schema']=='A5_ADAPTIVE_CACHED_FIGURE_REQUEST_V1' and r['max_quality_calls']==16
        and r['max_reference_feature_preparations']==16 and r['max_seconds']==1800,'Finite cached-score scope changed')
    return r,c


def collect(r,c):
    import numpy as np
    out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    c.require(not (out/'collection.json').exists(),'Collection already finalized; run score/render/export')
    receive_request=c.pinned(r['receive_request']);received=c.read(r['receive_completion_path'])
    c.require(received['status']=='A5_ADAPTIVE_FIXED16_ACTUAL_CHILDREN_WAIT_ZERO'
        and received['request_sha256']==r['receive_request']['sha256'] and received['actual_children_waited']
        and received['worker_exit_codes']==[0,0] and received['source_indices']==c.FIXED
        and received['source_count']==received['frame_count']==16 and received['SNRs']==[13]
        and received['noise_seeds']==[2001] and received['profile_id']==3009
        and received['independent_ledger']['unresolved']==0 and received['independent_ledger']['total']<=32,
        'Actual completed finite adaptive16 reception required')
    native_done=c.pinned(r['native_figure_completion']);native_collection=c.pinned(r['native_collection'])
    native_score=c.pinned(r['native_metrics_completion']);native_dir=Path(r['native_collection']['path']).parent
    c.require(native_done['status']=='COMPLETE' and native_done['remaining_missing_metric_cells']==0
        and native_collection['selected']==c.SELECTED and native_score['source_count']==16,'Completed native figure including missing-only supplement required')
    old_display=native_dir/'display_images.npz';old_float=native_dir/'bpg16_float_inputs.npz';old_metrics=native_dir/'per_image_metrics.csv'
    for p in (old_display,old_float,old_metrics):c.require(native_done['outputs'].get(str(p))==c.sha(p),'Completed native artifact changed')
    with np.load(old_display,allow_pickle=False) as z:old_images=z['images'].copy()
    with np.load(old_float,allow_pickle=False) as z:old_references=z['references'].copy()
    c.require(old_images.shape==(4,6,256,256,3) and old_images.dtype==np.uint8,'Original display schema differs')
    with old_metrics.open(encoding='utf-8',newline='') as f:previous=[v for v in csv.DictReader(f) if v['method'] in c.OLD]
    c.require(len(previous)==80 and sum(v['metric_status']=='REFERENCE_NOT_SCORED' for v in previous)==16
        and not any(v['metric_status'].startswith('MISSING') for v in previous),'Old five-column metric grid is not complete')
    references=[];images=[];metadata=[];bindings={}
    for ordinal,index in enumerate(c.FIXED):
        record=receive_request['records'][ordinal];cp=Path(receive_request['out'])/'sources'/f'{index:04d}.json'
        row=c.verify_output(cp,received)
        c.require(row['status']=='A5_ADAPTIVE_FIXED_SOURCE_COMPLETE_V1' and row['request_sha256']==received['request_sha256']
            and row['method']==METHOD and row['source_id']==record['source_id'] and row['source_index']==index
            and row['N']==1024 and row['snr_db']==13 and row['noise_seed']==2001 and row['profile_id']==3009
            and not row['policy_selection'] and not row['source_truth_used_by_receiver'], 'Frozen new adaptive frame differs')
        desc=row['float_reconstruction'];ap=desc['path']
        c.require(row['outputs'].get(ap)==received['outputs'].get(ap)==desc['sha256']==c.sha(ap),'Completed adaptive float archive changed')
        with np.load(ap,allow_pickle=False) as z:
            c.require(set(z.files)=={'rgb','source_rgb'},'Unexpected adaptive float schema')
            image=z['rgb'].copy();reference=z['source_rgb'].copy()
        c.require(image.dtype==reference.dtype==np.float32 and image.shape==reference.shape==(3,256,256)
            and np.isfinite(image).all() and image.min()>=0 and image.max()<=1
            and c.baseline_sha(image)==row['image_sha256'] and c.baseline_sha(reference)==row['reference_sha256']
            and np.array_equal(reference,old_references[ordinal]),'Exact adaptive source/RGB differs')
        if row['source_unfit']:c.require(row['link_status']=='SOURCE_UNFIT' and not row['actual_link_executed'] and np.all(image==np.float32(.5)),'Unfit source requires explicit no-frame gray')
        native_cp=native_dir/'bpg_metric_sources'/f'{index:04d}.json'
        c.require(native_score['outputs'].get(str(native_cp))==c.sha(native_cp),'Original native per-source metric changed')
        old=c.read(native_cp)
        c.require(old['identity']['source_id']==row['source_id'] and old['identity']['reference_sha256']==row['reference_sha256']
            and old['identity']['metric_evaluator_identity']==native_score['metric_evaluator_identity'], 'Native reference prediction mismatch')
        references.append(reference);images.append(image);metadata.append(dict(row,source_checkpoint=str(cp),source_checkpoint_sha256=c.sha(cp),
            native_metric_checkpoint=str(native_cp),native_metric_checkpoint_sha256=c.sha(native_cp)))
        bindings[str(cp)]=c.sha(cp);bindings[ap]=c.sha(ap);bindings[str(native_cp)]=c.sha(native_cp)
    np.savez_compressed(out/'adaptive16_float_inputs.npz',references=np.stack(references),images=np.stack(images))
    new=np.stack([np.rint(images[c.FIXED.index(i)]*255).astype(np.uint8).transpose(1,2,0) for i in c.SELECTED])
    np.savez_compressed(out/'display_images.npz',images=np.concatenate([old_images[:,:5],new[:,None]],axis=1))
    c.write_csv(out/'original_five_columns_metrics.csv',previous)
    c.write(out/'collection.json',dict(status='ACTUAL_ADAPTIVE16_FLOAT_RGB_COLLECTED',fixed16=c.FIXED,selected=c.SELECTED,
        method_order=c.OLD+[METHOD],rows=metadata,input_bindings=bindings,
        receive_completion_sha256=c.sha(r['receive_completion_path']),receive_request=r['receive_request'],
        native_figure_completion=r['native_figure_completion'],native_collection=r['native_collection'],
        adaptive_float_sha256=c.sha(out/'adaptive16_float_inputs.npz'),display_sha256=c.sha(out/'display_images.npz'),
        old_metrics_sha256=c.sha(out/'original_five_columns_metrics.csv'),old_five_columns_pixel_identical=True,
        new_reconstructions=0,new_PHY_calls=0,new_metric_calls=0,new_bootstrap=0))
    print(json.dumps(dict(stage='collect',sources=16,display_sources=c.SELECTED,old_method_metric_cells=64)),flush=True)


def score(r,c):
    import fcntl
    import numpy as np
    import torch
    out=Path(r['out']);plan=c.read(out/'collection.json');cpdone=out/'metrics_completion.json'
    if cpdone.exists():
        done=c.read(cpdone);c.require(done['status']=='COMPLETE' and done['collection_sha256']==c.sha(out/'collection.json'),'Different completed metrics')
        for p,h in done['outputs'].items():c.require(c.sha(p)==h,'Metric output changed')
        print('Adaptive metrics already complete; zero new calls');return
    c.require(c.sha(out/'adaptive16_float_inputs.npz')==plan['adaptive_float_sha256'],'Adaptive metric bundle changed')
    lock=Path(r['visual_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    own=(out/'score.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Frozen GPU environment required')
    start=time.monotonic()
    def guard():
        c.require(time.monotonic()-start<1800 and not any(Path(p).exists() for p in r['stop_files']),'Adaptive metric deadline/STOP')
    d=r['existing_score_module'];c.require(c.sha(d['path'])==d['sha256'],'Original backend loader changed')
    spec=importlib.util.spec_from_file_location('_a5_adaptive_frozen_backend',d['path']);old=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=old;spec.loader.exec_module(old)
    guard();backend=old.build_backend(r,c.pinned(r['original_FINAL_FREEZE']))
    with np.load(out/'adaptive16_float_inputs.npz',allow_pickle=False) as z:references=z['references'];images=z['images']
    rows=[];fresh_calls=0;fresh_features=0;native_reuses=0;checkpoint_reuses=0;conv_reuses=0
    for j,index in enumerate(c.FIXED):
        guard();meta=plan['rows'][j];image=images[j];target=references[j]
        cp=out/'metric_sources'/f'{index:04d}.json'
        identity=dict(source_id=meta['source_id'],source_index=index,reference_sha256=c.baseline_sha(target),
            image_sha256=c.baseline_sha(image),metric_evaluator_identity=backend.identity,population='development',N=1024,snr_db=13,method=METHOD)
        c.require(c.sha(meta['native_metric_checkpoint'])==meta['native_metric_checkpoint_sha256'],'Native metric source changed')
        previous=c.read(meta['native_metric_checkpoint'])
        c.require(previous['identity']['source_id']==meta['source_id'] and previous['identity']['reference_sha256']==identity['reference_sha256']
            and previous['identity']['metric_evaluator_identity']==backend.identity,'Cached native source prediction identity changed')
        if cp.exists():
            result=c.read(cp);c.require(result['identity']==identity,'Existing adaptive metric CP differs');checkpoint_reuses+=1
        elif previous['identity']['image_sha256']==identity['image_sha256']:
            result=dict(identity=identity,values=previous['values'],quality_calls=0,reference_feature_preparations=0,
                reference_classifier_predictions=0,reused_native_metric=True,reused_reference_classifier_prediction=False,
                source_prediction=previous['source_prediction'],source_metric_origin=meta['native_metric_checkpoint'],
                source_metric_origin_sha256=meta['native_metric_checkpoint_sha256'])
            c.write(cp,result);native_reuses+=1
        else:
            reservation=cp.with_suffix('.reserved.json');c.require(not reservation.exists(),'Unresolved adaptive metric attempt; do not repeat')
            c.write(reservation,dict(identity=identity,max_quality_calls=1,max_reference_feature_preparations=1,reference_classifier_prediction_reused=True))
            # Same frozen SuiteBackend.prepare feature calls; its ConvNeXt source
            # prediction is taken from the already-scored identical reference.
            backend.check();x=torch.from_numpy(target[None])
            with torch.inference_mode(),backend.metric.no_network(),torch.autocast(device_type=backend.evaluator.device.type,enabled=False):
                features=backend.evaluator.prepare_reference(x)
                embedding=backend.native.dino_features(backend.dino,x.to(backend.evaluator.device))[0].cpu().numpy()
            prepared=dict(reference_tensor=x,prepared=features,dino_feature=np.asarray(embedding).copy(),
                convnext_prediction=previous['source_prediction'],resnet50_prediction=int(features['resnet50'][0]))
            fresh_features+=1;conv_reuses+=1;backend.check();guard()
            with torch.inference_mode(),backend.metric.no_network(),torch.autocast(device_type=backend.evaluator.device.type,enabled=False):
                legacy,source_feature,_=backend.native.quality_metrics(target,[image],backend.lpips,backend.dino,backend.evaluator.device)
                c.require(np.array_equal(np.asarray(source_feature),prepared['dino_feature']),'Original reference DINO feature mismatch')
                newer=backend.evaluator.score(x,torch.from_numpy(image[None]),[meta['evaluation_class_index']],label_conditioned=False,prepared=features)
                independent=backend.classifier.score(image,true_label=meta['evaluation_class_index'],source_prediction=prepared['convnext_prediction'])
            backend.check();c.require(len(legacy)==len(newer)==1,'Original batch-one metrics required');fresh_calls+=1
            mse=float(np.square(image.astype(np.float64)-target.astype(np.float64)).mean());c.require(mse>0,'Unexpected infinite PSNR')
            values=dict(psnr_db=-10*math.log10(mse),lpips_alex=float(legacy[0]['lpips_alex']),
                dinov2_vitl14_cosine=float(newer[0]['dinov2_vitl14_cosine']),convnext_top1_source_prediction=bool(independent['convnext_top1_source_prediction']))
            result=dict(identity=identity,values=values,quality_calls=1,reference_feature_preparations=1,reference_classifier_predictions=0,
                reused_native_metric=False,reused_reference_classifier_prediction=True,source_prediction=previous['source_prediction'],
                source_prediction_origin=meta['native_metric_checkpoint'],source_prediction_origin_sha256=meta['native_metric_checkpoint_sha256'])
            c.write(cp,result)
        c.require(result['values']['psnr_db']==meta['psnr_db'],'Actual adaptive float64 PSNR differs')
        common=dict(source_index=index,source_id=meta['source_id'],population='development',N=1024,snr_db=13,method=METHOD,
            method_label='BPG + LDPC (adaptive downsampling)',noise_seed=2001,noise_namespace=meta['noise_namespace'],
            image_sha256=meta['image_sha256'],source_unfit=meta['source_unfit'],display_role=meta['display_role'])
        current=c.metric_rows(common,result['values'],str(cp),backend.identity)
        for v in current:v['metric_status']='REUSED_NATIVE_IDENTICAL_RGB' if result['reused_native_metric'] else 'NEW_ADAPTIVE_CACHED_RGB_SCORE'
        rows+=current
        print(json.dumps(dict(completed_sources=j+1,new_quality_calls=fresh_calls,new_reference_features=fresh_features,native_quality_reuses=native_reuses)),flush=True)
    c.write_csv(out/'adaptive16_per_image_metrics.csv',rows)
    checkpoints=[c.read(out/'metric_sources'/f'{index:04d}.json') for index in c.FIXED]
    total=sum(v['quality_calls'] for v in checkpoints);features=sum(v['reference_feature_preparations'] for v in checkpoints)
    c.require(total<=16 and features<=16 and len(rows)==64,'Finite adaptive metric budget exceeded')
    outputs={str(p):c.sha(p) for p in sorted((out/'metric_sources').glob('*.json'))}
    outputs[str(out/'adaptive16_per_image_metrics.csv')]=c.sha(out/'adaptive16_per_image_metrics.csv')
    c.write(cpdone,dict(status='COMPLETE',source_count=16,frame_count=16,collection_sha256=c.sha(out/'collection.json'),
        new_quality_calls_total=total,new_reference_feature_preparations_total=features,new_reference_classifier_predictions_total=0,
        native_identical_image_reuses_total=sum(v['reused_native_metric'] for v in checkpoints),
        reused_native_reference_classifier_predictions_total=sum(v['reused_reference_classifier_prediction'] for v in checkpoints),
        quality_calls_this_invocation=fresh_calls,reference_feature_preparations_this_invocation=fresh_features,source_checkpoint_reuses=checkpoint_reuses,
        metric_evaluator_identity=backend.identity,metric_metadata=backend.metadata,outputs=outputs,elapsed_seconds=time.monotonic()-start,
        old_metrics_recomputed=0,new_reconstructions=0,new_PHY_calls=0,new_bootstrap=0,policy_selection=False,training_updates=0))


def render(r,c):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(r['out']);plan=c.read(out/'collection.json');c.require(c.sha(out/'display_images.npz')==plan['display_sha256'],'Display pixels changed')
    with np.load(out/'display_images.npz',allow_pickle=False) as z:images=z['images']
    labels=c.LABELS[:5]+['BPG + LDPC\n(adaptive downsampling)']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(4,6,figsize=(7.2,5.25),gridspec_kw={'wspace':.025,'hspace':.045})
    fig.subplots_adjust(left=.008,right=.992,bottom=.025,top=.90);metadata={x['source_index']:x for x in plan['rows']}
    for j,index in enumerate(c.SELECTED):
        for k in range(6):
            ax=axes[j,k];ax.imshow(images[j,k],interpolation='nearest');ax.set_axis_off()
            if j==0:ax.set_title(labels[k],fontsize=7.1,pad=7,linespacing=1.12)
            if k==5 and metadata[index]['source_unfit']:
                ax.text(.5,.5,'SOURCE_UNFIT\nNo received frame',ha='center',va='center',transform=ax.transAxes,fontsize=6.9,color='white',
                    linespacing=1.35,bbox=dict(facecolor='#343434',edgecolor='none',alpha=.88,pad=3))
    stem=out/'six_methods_adaptive_BPG_N1024_13dB_development'
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600)
    plt.close(fig)
    caption=(r'\caption{Fixed development examples at $N=1024$ and SNR $13$ dB, including the adaptive-downsampling BPG + LDPC baseline. '
        'The source images are the unchanged second group in the historical fixed-sixteen order. The original five columns are reused pixel for pixel. '
        'Adaptive BPG uses the source rule and the 13-dB modulation/coding choice frozen on the original 100 calibration sources: 16-QAM with nominal rate 5/6. '
        'Only one predetermined noise realization (seed 2001) is used, with no selection by reconstruction quality. '
        'Adaptive BPG and the native256 BPG supplement use the A3\\_DEVELOPMENT\\_TIMING noise namespace; their transmitted waveforms differ. '
        'The historical digital method uses seed 6201, while the other baselines use seed 2001; only SwinJSCC and HiFi-DiffCom share their actual received observation. '
        'Any SOURCE\\_UNFIT panel denotes a gray source-overflow fallback with no received frame. '
        'The native256 BPG comparison remains a separate supplementary figure. Scores use the original float reconstructions and are descriptive per-image development results.}'+'\n')
    (out/'captions.tex').write_text(caption,encoding='utf-8')
    print(json.dumps(dict(stage='render',png=str(stem.with_suffix('.png')))))


def export(r,c):
    out=Path(r['out']);plan=c.read(out/'collection.json');done=c.read(out/'metrics_completion.json')
    c.require(done['status']=='COMPLETE' and done['collection_sha256']==c.sha(out/'collection.json'),'Actual new scoring completion required')
    for p,h in done['outputs'].items():c.require(c.sha(p)==h,'Adaptive metric output changed')
    c.require(c.sha(out/'original_five_columns_metrics.csv')==plan['old_metrics_sha256'],'Old five-column scores changed')
    with (out/'original_five_columns_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    with (out/'adaptive16_per_image_metrics.csv').open(encoding='utf-8',newline='') as f:rows += [v for v in csv.DictReader(f) if int(v['source_index']) in c.SELECTED]
    c.require(len(rows)==96 and len({(v['source_index'],v['method'],v['metric']) for v in rows})==96,'Incomplete or duplicate six-column metric grid')
    c.require(not any(v['metric_status'].startswith('MISSING') for v in rows),'Unexpected missing metric')
    c.write_csv(out/'per_image_metrics.csv',rows)
    stem='six_methods_adaptive_BPG_N1024_13dB_development'
    c.require(all((out/(stem+'.'+ext)).exists() for ext in ['png','pdf','svg']),'Render before final export')
    text='# Fixed development examples with adaptive BPG\n\n'
    text+='This is a separate six-column figure for sources `[4,21,24,29]`, N1024, 13 dB. The first five columns and all 64 existing method-metric cells are reused unchanged, including the previously filled eight ConvNeXt values; the 16 source-reference cells are not scored. The original native256 BPG six-column figure is retained.\n\n'
    text+='The new baseline follows the existing adaptive source rule and the modulation/coding choice frozen on 100 calibration sources. A separate fixed16, single-SNR, single-noise receive job generated its actual float RGB under a 32-PHY-call cap. This figure script does not reconstruct or simulate a channel. All failures remain present.\n\n'
    text+=f'New quality calls: {done["new_quality_calls_total"]}; new reference feature preparations: {done["new_reference_feature_preparations_total"]}; original-reference ConvNeXt predictions reused: {done["reused_native_reference_classifier_predictions_total"]}; exact native image metric reuses: {done["native_identical_image_reuses_total"]}. No original scores or bootstrap were recomputed. The source feature count is separate and does not claim to remove legacy reference computations internal to the frozen scorer.\n\n'
    text+='The four core fields are PSNR (dB), LPIPS-Alex, DINOv2 ViT-L/14 cosine, and ConvNeXt source-prediction agreement (per-image 0/1, not label accuracy). Every metric uses original float RGB. Same source and SNR do not imply the same received observation; see the caption and per-image metadata. These examples do not estimate holdout success rates.\n\n'
    text+='Reproduction: use the prepared request with `adaptive_cached_examples.py --request <request.json> --mode collect`, then scheduled `--mode score`, CPU `--mode render`, and `--mode export`. The read-only dependencies and exact receive completion SHA are in `collection.json`.\n'
    (out/'README.md').write_text(text,encoding='utf-8')
    outputs={str(p):c.sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.npz','.tex','.md','.png','.pdf','.svg') and p.name!='completion.json'}
    c.write(out/'completion.json',dict(status='COMPLETE',source_count_scored=16,display_sources=c.SELECTED,population='development',
        N=1024,snr_db=13,adaptive_BPG_new_quality_calls=done['new_quality_calls_total'],missing_metric_cells=0,
        old_native_figure_modified=False,old_five_columns_modified=False,new_bootstrap=0,outputs=outputs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--mode',choices=['collect','score','render','export'],required=True)
    a=p.parse_args();r,c=setup(a.request);globals()[a.mode](r,c)
