"""Append native256 BPG to the historical fixed group02; never reconstruct a model.

collect/render/export are CPU readers. score is an explicitly scheduled GPU job
for exactly the sixteen new BPG 13-dB cached outputs, with no bootstrap or PHY.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

FIXED = [0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
SELECTED = [4,21,24,29]
METRICS = ['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
OLD = ['source','SwinJSCC80k','P1024','HiFiDiffCom','RAW64_PARTIAL_VAR_COMPLETION']
BPG = 'BPG_LDPC_native256'
LABELS = ['Original image','SwinJSCC image\ntransmission','Continuous latent-\nspace JSCC',
          'HiFi-DiffCom\ndiffusion-prior recovery','Partial-scale digital\n+ VAR (Proposed)',
          'BPG + LDPC\n(native256)']


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def pinned(d):
    require(sha(d['path'])==d['sha256'], 'Pinned input changed: '+d['path'])
    return read(d['path'])


def write(path, value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists(): require(p.read_bytes()==data, 'Immutable output differs: '+str(p)); return
    with p.open('xb') as f: f.write(data)


def write_csv(path, rows):
    import io
    stream=io.StringIO(newline=''); w=csv.DictWriter(stream,list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);p=Path(path);data=stream.getvalue().encode()
    if p.exists(): require(p.read_bytes()==data,'Immutable CSV differs: '+str(p));return
    with p.open('xb') as f:f.write(data)


def raw_sha(a):
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def baseline_sha(a):return hashlib.sha256(b'float32:3,256,256:RGB\0'+a.tobytes()).hexdigest()


def one(rows, what):
    require(len(rows)==1, f'Expected exactly one {what}; got {len(rows)}');return rows[0]


def verify_output(path, done):
    p=str(path);require(done['outputs'].get(p)==sha(p), 'Completed output not bound: '+p)
    return read(p)


def metric_rows(common, values, origin, identity='', missing='MISSING_FIELD_IN_ORIGINAL_CACHE'):
    result=[]
    for metric in METRICS:
        v=values.get(metric)
        if isinstance(v,str) and not v.strip():v=None
        if v is not None:
            # The continuous baseline's legacy CSV-imported PSNR and LPIPS are
            # numeric strings. Validate their number, retain their exact text.
            require(isinstance(v,(int,float,bool,str)), 'Invalid cached metric type')
            number=float(v)
            require(math.isfinite(number), 'Invalid cached metric')
            if metric=='convnext_top1_source_prediction':require(number in (0,1),'Agreement must be 0/1')
        result.append(dict(common,metric=metric,value=v if v is not None else '',
            unit='fraction_0_or_1' if metric=='convnext_top1_source_prediction' else 'dB' if metric=='psnr_db' else 'unitless',
            metric_status='REUSE' if v is not None else missing,metric_source=origin,
            metric_source_sha256=sha(origin) if origin and Path(origin).exists() else '',metric_evaluator_identity=identity))
    return result


def collect(r):
    import numpy as np
    out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    require(not (out/'collection.json').exists(), 'Collection exists; use score/render/export to resume')
    root=Path(r['root']);n=root/'outputs/MAIN-RAW64-20261007'
    a3=pinned(r['a3_request']);done=pinned(r['a3_completion'])
    require(done['status']=='COMPLETE' and done['method']==BPG and done['source_count']==16
            and done['display_archives']==48 and done['fresh_source_attempts']==192
            and done['request_sha256']==r['a3_request']['sha256'], 'Actual completed A3 native BPG required')
    require(a3['source_indices']==FIXED and not a3['holdout_used'] and not a3['policy_selection']
            and done['noise_namespace']=='A3_DEVELOPMENT_TIMING' and done['noise_seed']==2001,'A3 scope changed')
    records={x['source_index']:x for x in a3['records']}
    old=pinned(r['old_plan']);require([x['source_index'] for x in old['samples']]==SELECTED
        and old['array_order'][:5]==OLD and old['population']=='development' and old['N']==1024 and old['snr_db']==13,'Wrong historical group')
    require(sha(r['old_images']['path'])==r['old_images']['sha256']==old['display_images_sha256'],'Old display changed')
    require(sha(r['old_provenance']['path'])==r['old_provenance']['sha256'],'Old provenance changed')
    with np.load(r['old_images']['path'],allow_pickle=False) as z:old_images=z['images'].copy()
    require(old_images.shape==(4,7,256,256,3) and old_images.dtype==np.uint8,'Historical display shape changed')
    with Path(r['old_provenance']['path']).open(encoding='utf-8',newline='') as f: provenance=list(csv.DictReader(f))
    cases=[]
    for path in sorted(done['outputs']):
        if Path(path).name.startswith('case_') and Path(path).suffix=='.json':
            value=verify_output(path,done)
            if value['phase']=='measured' and value['repetition']==1 and value['snr_db']==13:
                cases.append(dict(value,case_path=path,case_sha256=done['outputs'][path]))
    require(len(cases)==16 and {x['source_index'] for x in cases}==set(FIXED),'Incomplete or duplicate 13-dB first repetition')
    cases={x['source_index']:x for x in cases};targets=[];images=[];metadata=[]
    for index in FIXED:
        row=cases[index];rec=records[index]
        require(row['source_id']==rec['source_id'] and row['method']==BPG,'BPG source identity mismatch')
        require(sha(rec['archive'])==rec['archive_sha256'] and sha(rec['checkpoint'])==rec['checkpoint_sha256'],'Original source binding changed')
        with np.load(rec['archive'],allow_pickle=False) as z:pixels=z[rec['pixels_key']].copy()
        require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
            and hashlib.sha256(pixels.tobytes()).hexdigest()==rec['preprocessing_id'],'Original preprocessing differs')
        require(done['outputs'].get(row['display_archive'])==row['display_archive_sha256']==sha(row['display_archive']),'A3 float archive changed')
        with np.load(row['display_archive'],allow_pickle=False) as z:
            require(z.files==['rgb'],'Unexpected A3 archive schema');rgb=z['rgb'].copy()
        require(rgb.dtype==np.float32 and rgb.shape==(3,256,256) and np.isfinite(rgb).all()
            and rgb.min()>=0 and rgb.max()<=1 and hashlib.sha256(rgb.tobytes()).hexdigest()==row['output_sha256'],'A3 actual float pixels changed')
        if row['source_unfit']:
            require(row['display_role']=='source_unfit_gray_no_received_frame' and not row['actual_link_executed']
                and row['link_status']=='SOURCE_UNFIT' and np.all(rgb==np.float32(.5)), 'Unfit source must be explicit no-frame gray')
        else:require(row['actual_link_executed'] and row['display_role']=='actual_first_measured_received_output','Missing actual BPG received provenance')
        target=np.ascontiguousarray(pixels.astype(np.float32)/np.float32(255))
        if index in SELECTED:
            j=SELECTED.index(index);s=old['samples'][j]
            require(s['source_id']==rec['source_id'] and s['preprocessing_id']==rec['preprocessing_id']
                and np.array_equal(old_images[j,0],pixels.transpose(1,2,0)), 'Old five columns and BPG use different originals')
        targets.append(target);images.append(rgb)
        metadata.append(dict(row,preprocessing_id=rec['preprocessing_id'],evaluation_class_index=rec['evaluation_class_index'],
            reference_sha256=baseline_sha(target),raw_reference_sha256=raw_sha(target),float_image_sha256=baseline_sha(rgb)))
    np.savez_compressed(out/'bpg16_float_inputs.npz',references=np.stack(targets),images=np.stack(images))
    appended=np.stack([np.rint(images[FIXED.index(i)]*255).astype(np.uint8).transpose(1,2,0) for i in SELECTED])
    np.savez_compressed(out/'display_images.npz',images=np.concatenate([old_images[:,:5],appended[:,None]],axis=1))
    reused=[];missing=[]
    hdir=root/'outputs/EXTERNAL-COMPARISON-20261004/fixed80k_revision/hifi_fixed16_release/evaluation'
    pdir=root/'outputs/CONTENT-REAL-64QAM-20261006/H/p600_replay_v1'
    rdir=n/'development_metrics_mixed_v1'
    for index in SELECTED:
        for method in OLD:
            prov=one([p for p in provenance if int(p['source_index'])==index and p['method']==method],'original figure cell')
            require(prov['source_id']==records[index]['source_id'],'Old metric/source identity differs')
            common=dict(source_index=index,source_id=prov['source_id'],population='development',N=1024,snr_db=13,
                method=method,method_label=LABELS[OLD.index(method)].replace('\n',' '),noise_seed=prov['noise_seed'],
                noise_namespace='historical_frozen_cache',image_sha256=prov['image_sha256'],
                source_unfit='',display_role='reference' if method=='source' else 'cached_reconstruction')
            if method=='source':
                reused+=metric_rows(common,{},'',missing='REFERENCE_NOT_SCORED');continue
            values={};origin='';identity='';reason='MISSING_CACHE'
            base=hdir if method in ['SwinJSCC80k','HiFiDiffCom'] else pdir if method=='P1024' else rdir
            cp=base/('metrics/source_checkpoints' if base==hdir else 'source_checkpoints')/f'{index:04d}.json'
            if (base/'completion.json').exists() and cp.exists():
                completion=read(base/'completion.json');checkpoint=verify_output(cp,completion)
                if base==hdir:
                    rows=checkpoint['rows'];origin=str(cp)
                else:
                    origin=str(base/'sources'/f'{index:04d}.json');rows=verify_output(origin,checkpoint)
                want_method={'SwinJSCC80k':'SwinJSCC_new_shared','HiFiDiffCom':'HiFiDiffCom_SwinJSCC','P1024':'P1024'}.get(method)
                rows=[v for v in rows if int(v['snr_db'])==13 and int(v['noise_seed'])==int(prov['noise_seed'])
                      and v['source_id']==prov['source_id'] and v['image_sha256']==prov['image_sha256']
                      and (v.get('method')==want_method if want_method else
                           'PARTIAL' in v.get('family_memberships',[]) and v.get('decoder_arm')=='VAR_completion')]
                value=one(rows,'cached per-image score matching displayed float hash')
                expected_ref=metadata[FIXED.index(index)]['raw_reference_sha256' if base==rdir else 'reference_sha256']
                require(value['reference_sha256']==expected_ref and int(value['N'])==1024,'Cached score reference/N mismatch')
                values={k:value[k] for k in METRICS if k in value}
                identity=value.get('metric_evaluator_identity',completion.get('metric_evaluator_identity',completion.get('metadata_identity','')))
                reason='MISSING_FIELD_IN_ORIGINAL_CACHE'
            got=metric_rows(common,values,origin,identity,reason);reused+=got
            missing += [x for x in got if x['metric_status'].startswith('MISSING')]
    write_csv(out/'original_five_columns_metrics.csv',reused)
    write(out/'missing_original_metrics.json',missing)
    write(out/'collection.json',dict(schema='A5_NATIVE_BPG_FIXED_GROUP02_V1',status='COLLECTED_ACTUAL_CACHED_RGB',
        a3_request=r['a3_request'],a3_completion=r['a3_completion'],old_plan=r['old_plan'],
        selected=SELECTED,fixed16=FIXED,method_order=OLD+[BPG],bpg_rows=metadata,
        bpg_float_archive_sha256=sha(out/'bpg16_float_inputs.npz'),display_archive_sha256=sha(out/'display_images.npz'),
        old_metrics_csv_sha256=sha(out/'original_five_columns_metrics.csv'),missing_original_metric_cells=len(missing),
        old_five_columns_pixel_identical=True,new_reconstruction_calls=0,new_PHY_calls=0,new_bootstrap=0,
        bpg_source_unfit_count=sum(x['source_unfit'] for x in metadata),all_methods_same_observation=False,
        swin_hifi_same_observation=True,noise_note='A3 BPG seed2001 namespace A3_DEVELOPMENT_TIMING; historical digital seed6201; other baselines seed2001. No best-noise selection.'))
    print(json.dumps(dict(stage='collect',bpg_sources=16,unfit=sum(x['source_unfit'] for x in metadata),missing_original_metric_cells=len(missing))))


def score(r, max_seconds):
    import fcntl
    import numpy as np
    import torch
    out=Path(r['out']);plan=read(out/'collection.json')
    require(max_seconds<=1800 and max_seconds>0,'A5 GPU budget must be at most 1800 seconds')
    require(sha(out/'bpg16_float_inputs.npz')==plan['bpg_float_archive_sha256'],'Collected float inputs changed')
    if (out/'bpg_metrics_completion.json').exists():
        done=read(out/'bpg_metrics_completion.json')
        require(done['status']=='COMPLETE' and done['source_count']==16,'Bad prior score completion')
        for p,h in done['outputs'].items():require(sha(p)==h,'Prior A5 metric output changed')
        print('Already complete; no new GPU work');return
    lock=Path(r['visual_lock']).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    joblock=(out/'score.lock').open('a+');fcntl.flock(joblock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    def guard():
        require(time.monotonic()-start<max_seconds,'A5 metric time budget exhausted')
        require(not any(Path(p).exists() for p in r['stop_files']),'A5 stop requested')
    guard()
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Original CUBLAS setting required')
    # The original load_suite sets all frozen flags, including interop threads,
    # exactly once. Setting them here first would make PyTorch reject its call.
    d=r['existing_score_module'];require(sha(d['path'])==d['sha256'],'Original backend loader changed')
    spec=importlib.util.spec_from_file_location('_a5_original_bpg_score',d['path']);old=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=old;spec.loader.exec_module(old)
    backend=old.build_backend(r,pinned(r['original_FINAL_FREEZE']))
    with np.load(out/'bpg16_float_inputs.npz',allow_pickle=False) as z:references=z['references'];images=z['images']
    require(references.shape==images.shape==(16,3,256,256) and references.dtype==images.dtype==np.float32,'Wrong fixed16 metric inputs')
    metrics=[];prepared_calls=0;new_calls=0;reused_calls=0
    for j,index in enumerate(FIXED):
        guard();row=plan['bpg_rows'][j];require(row['source_index']==index,'Fixed source order changed')
        target,image=references[j],images[j];cp=out/'bpg_metric_sources'/f'{index:04d}.json'
        identity=dict(source_index=index,source_id=row['source_id'],reference_sha256=baseline_sha(target),image_sha256=baseline_sha(image),
                      metric_evaluator_identity=backend.identity,population='development',N=1024,snr_db=13,method=BPG)
        if cp.exists():
            result=read(cp);require(result['identity']==identity,'Prior A5 metric checkpoint mismatch');reused_calls+=1
        else:
            reserved=cp.with_suffix('.reserved.json')
            require(not reserved.exists(),'Unresolved GPU attempt exists; do not duplicate: '+str(reserved))
            write(reserved,dict(identity=identity,reserved_reference_preparations=1,reserved_quality_calls=1))
            prepared=backend.prepare(target);prepared_calls+=1;backend.check();guard()
            # This is the original SuiteBackend's core four-metric path, with
            # unrelated mismatch/confidence calculations omitted, not relabelled.
            with torch.inference_mode(),backend.metric.no_network(),torch.autocast(device_type=backend.evaluator.device.type,enabled=False):
                legacy,source_feature,_=backend.native.quality_metrics(target,[image],backend.lpips,backend.dino,backend.evaluator.device)
                require(np.array_equal(np.asarray(source_feature),prepared['dino_feature']),'Original reference feature changed')
                newer=backend.evaluator.score(prepared['reference_tensor'],torch.from_numpy(image[None]),
                    [row['evaluation_class_index']],label_conditioned=False,prepared=prepared['prepared'])
                independent=backend.classifier.score(image,true_label=row['evaluation_class_index'],source_prediction=prepared['convnext_prediction'])
            backend.check();new_calls+=1
            require(len(legacy)==len(newer)==1,'Original batch-one scoring required')
            mse=float(np.mean((target.astype(np.float64)-image.astype(np.float64))**2))
            require(mse>0,'Unexpected perfect BPG frame; represent infinite PSNR explicitly before export')
            values=dict(psnr_db=-10*math.log10(mse),lpips_alex=float(legacy[0]['lpips_alex']),
                dinov2_vitl14_cosine=float(newer[0]['dinov2_vitl14_cosine']),
                convnext_top1_source_prediction=bool(independent['convnext_top1_source_prediction']))
            result=dict(identity=identity,values=values,mse=mse,source_unfit=row['source_unfit'],display_role=row['display_role'],
                source_prediction=int(prepared['convnext_prediction']),reconstruction_prediction=independent.get('convnext_prediction'),
                scoring_scope='New cached BPG RGB only; original FP32 batch-one FullSuite and independent ConvNeXt core path',
                reference_preparations=1,quality_calls=1,used_for_selection=False)
            write(cp,result)
        common=dict(source_index=index,source_id=row['source_id'],population='development',N=1024,snr_db=13,
            method=BPG,method_label='BPG + LDPC (native256)',noise_seed=2001,noise_namespace='A3_DEVELOPMENT_TIMING',
            image_sha256=row['float_image_sha256'],source_unfit=row['source_unfit'],display_role=row['display_role'])
        current=metric_rows(common,result['values'],str(cp),backend.identity)
        for v in current:v['metric_status']='NEW_CACHED_RGB_SCORE'
        metrics+=current
        print(json.dumps(dict(stage='score',completed_sources=j+1,new_quality_calls=new_calls,reused_quality_calls=reused_calls)),flush=True)
    write_csv(out/'bpg16_per_image_metrics.csv',metrics)
    outputs={str(p):sha(p) for p in sorted((out/'bpg_metric_sources').glob('*.json'))}
    outputs[str(out/'bpg16_per_image_metrics.csv')]=sha(out/'bpg16_per_image_metrics.csv')
    write(out/'bpg_metrics_completion.json',dict(status='COMPLETE',population='development',source_count=16,frame_count=16,
        new_quality_calls_this_invocation=new_calls,new_reference_preparations_this_invocation=prepared_calls,
        reused_sources_this_invocation=reused_calls,total_quality_calls=16,total_reference_preparations=16,
        metric_evaluator_identity=backend.identity,metric_metadata=backend.metadata,elapsed_seconds=time.monotonic()-start,
        input_collection_sha256=sha(out/'collection.json'),outputs=outputs,new_PHY_calls=0,new_bootstrap=0,
        reconstruction_model_calls=0,training_updates=0,policy_selection=False))


def render(r):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(r['out']);plan=read(out/'collection.json')
    require(plan['selected']==SELECTED and plan['method_order']==OLD+[BPG]
        and sha(out/'display_images.npz')==plan['display_archive_sha256'],'Display collection differs')
    with np.load(out/'display_images.npz',allow_pickle=False) as z:images=z['images']
    require(images.shape==(4,6,256,256,3) and images.dtype==np.uint8,'Six-column display shape differs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(4,6,figsize=(7.2,5.25),gridspec_kw={'wspace':.025,'hspace':.045})
    fig.subplots_adjust(left=.008,right=.992,bottom=.025,top=.90)
    rows={v['source_index']:v for v in plan['bpg_rows']}
    for j,index in enumerate(SELECTED):
        for k in range(6):
            ax=axes[j,k];ax.imshow(images[j,k],interpolation='nearest');ax.set_axis_off()
            if j==0:ax.set_title(LABELS[k],fontsize=7.1,pad=7,linespacing=1.12)
            if k==5 and rows[index]['source_unfit']:
                ax.text(.5,.5,'SOURCE_UNFIT\nNo received frame',ha='center',va='center',transform=ax.transAxes,
                    fontsize=6.9,color='white',linespacing=1.35,bbox=dict(facecolor='#343434',edgecolor='none',alpha=.88,pad=3))
    stem=out/'six_methods_N1024_13dB_development'
    for ext in ['pdf','svg','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=600)
    plt.close(fig)
    caption=(r'\caption{Fixed development examples at $N=1024$ and SNR $13$ dB. '
        'The four source images are the second consecutive group in the historical fixed-sixteen order, '
        'selected without image-quality or best-noise ranking. The original five columns are reused pixel for pixel. '
        'BPG + LDPC (native256) is the original native-resolution digital baseline, distinct from the adaptive-downsampling baseline. '
        'Its first measured repetition is shown; SOURCE\\_UNFIT denotes a gray fallback for a source bitstream that does not fit, with no transmitted or received frame. '
        'SwinJSCC and HiFi-DiffCom share their historical received observation. BPG uses seed 2001 in the A3\\_DEVELOPMENT\\_TIMING namespace; '
        'the historical digital method uses seed 6201, and the other baselines use seed 2001. Different method waveforms do not imply identical observations. '
        'These development examples do not estimate holdout success rates. Scores use original float RGB, not display PNGs; missing cached metrics are explicitly marked.}'+'\n')
    p=out/'captions.tex'
    if p.exists():require(p.read_text(encoding='utf-8')==caption,'Caption differs')
    else:p.write_text(caption,encoding='utf-8')
    print(json.dumps(dict(stage='render',png=str(stem.with_suffix('.png')))))


def export(r):
    out=Path(r['out']);plan=read(out/'collection.json');done=read(out/'bpg_metrics_completion.json')
    require(done['status']=='COMPLETE' and done['input_collection_sha256']==sha(out/'collection.json'),'Actual BPG scoring completion required')
    for p,h in done['outputs'].items():require(sha(p)==h,'Metric artifact changed')
    require(sha(out/'original_five_columns_metrics.csv')==plan['old_metrics_csv_sha256'],'Old CSV changed')
    old_metrics=out/'original_five_columns_metrics.csv';supplement=out/'missing_convnext/completion.json'
    extra=dict(new_reconstruction_predictions=0,new_reference_predictions=0,resolved_missing_cells=0)
    if supplement.exists():
        extra_done=read(supplement)
        require(extra_done['status']=='COMPLETE' and extra_done['original_metrics_sha256']==plan['old_metrics_csv_sha256'], 'Missing-only supplement does not match originals')
        for p,h in extra_done['outputs'].items():require(sha(p)==h,'Missing-only supplement changed')
        old_metrics=out/'missing_convnext/original_five_columns_metrics_completed.csv'
        extra={k:extra_done[k] for k in extra}
    with old_metrics.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    with (out/'bpg16_per_image_metrics.csv').open(encoding='utf-8',newline='') as f:rows += [v for v in csv.DictReader(f) if int(v['source_index']) in SELECTED]
    require(len(rows)==96 and len({(v['source_index'],v['method'],v['metric']) for v in rows})==96,'Per-image six-column grid is incomplete/duplicate')
    write_csv(out/'per_image_metrics.csv',rows)
    description='# Fixed development examples with native256 BPG\n\n'
    description+='Sources `[4, 21, 24, 29]`, N1024, 13 dB. Old five columns are pixel-identical to the historical group02. New BPG scores cover all fixed16 sources at this one SNR and first measured repetition; the combined CSV contains the displayed four sources.\n\n'
    description+='`SOURCE_UNFIT` is a gray fallback with no received frame. It remains in the per-image quality table. Native256 BPG and adaptive-downsampling BPG are distinct baselines. No same-observation claim is made across methods.\n\n'
    description+='Core fields: `psnr_db`, `lpips_alex`, `dinov2_vitl14_cosine`, `convnext_top1_source_prediction`. Agreement is a per-image 0/1 value, not classification accuracy. Existing PSNR values retain the precision and computation of their original cache. Source/reference cells are `REFERENCE_NOT_SCORED`; missing method metrics remain empty with explicit `MISSING` reasons, never zero.\n\n'
    remaining=plan['missing_original_metric_cells']-extra['resolved_missing_cells']
    description+=f'Originally missing metric cells: {plan["missing_original_metric_cells"]}; remaining: {remaining}. See `missing_original_metrics.json` for the original inventory and `missing_convnext/completion.json` when filled. No existing metric, reconstruction, channel simulation, bootstrap, or policy was rerun. The BPG job scored 16 cached outputs, with 16 source preparations counted separately. The independent missing-only ConvNeXt supplement adds {extra["new_reconstruction_predictions"]} reconstruction predictions and {extra["new_reference_predictions"]} source predictions; it does not recompute existing PSNR, LPIPS, or DINO-L.\n\n'
    description+='Reproduce with the frozen request: `python experiments/paper_supplement_20261008/a5_bpg_examples/a5_cached_examples.py --request <request.json> --mode collect`, then scheduled `--mode score`, `--mode render`, and `--mode export`. Only score needs the GPU. The score stage does not rerun completed source checkpoints.\n'
    (out/'README.md').write_text(description,encoding='utf-8')
    outputs={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.png','.svg','.pdf','.tex','.md','.npz') and p.name!='completion.json'}
    require(all((out/('six_methods_N1024_13dB_development.'+ext)).exists() for ext in ['png','pdf','svg']),'Render the figure before export')
    write(out/'completion.json',dict(status='COMPLETE_WITH_EXPLICIT_MISSING_OLD_METRICS' if remaining else 'COMPLETE',
        new_BPG_scored_sources=16,display_sources=SELECTED,missing_original_metric_cells=plan['missing_original_metric_cells'],
        remaining_missing_metric_cells=remaining,missing_only_supplement=extra,
        population='development',holdout_used=False,policy_selection=False,old_images_changed=False,outputs=outputs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--mode',required=True,choices=['collect','score','render','export'])
    p.add_argument('--max-seconds',type=int,default=1800);a=p.parse_args();r=read(a.request)
    require(r['schema']=='A5_NATIVE_BPG_CACHED_EXAMPLES_REQUEST_V1' and sha(__file__)==r['worker_sha256'],'Frozen A5 executable/request mismatch')
    if a.mode=='score':score(r,a.max_seconds)
    else:globals()[a.mode](r)


if __name__=='__main__':main()
