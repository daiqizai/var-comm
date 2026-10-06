"""Prepared, CPU-only H/P normalized view. No CLI, model, metric or statistics run.

Original rows remain immutable. Actual callers must separately register the
metric-identity admission and prove normal completed owners before loading rows.
MAIN has no adapter here and cannot be silently substituted or filled in.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

H_AGGREGATE_SHA='bb820536b7ae8bddb7100a966b3c85171c1743d5ac9721c0c95c90323236c8a9'
P_DRIVER_SHA='60211f97135f391dfe94a4f6dacf4e78e113f928e59ae86e3161f8e0fe000d86'
STATISTICS_SHA='4e25eac1ed26201242101a4d0dad14ad00054752ab837a4688dd583806d8b342'
NOISE={'H':(6201,6202,6203),'P':(2001,2002,2003)}
METRICS=('mse','psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists',
    'dreamsim','ms_ssim','resnet50_top1_label','resnet50_top1_source_prediction','resnet50_top1_probability',
    'semantic_error','confidently_wrong','convnext_top1_label','convnext_top1_source_prediction',
    'convnext_source_prediction_agreement','convnext_source_correct_to_wrong','convnext_source_wrong_to_correct',
    'dino_mismatched','dino_specificity')
P_MISSING={'mse','resnet50_top1_probability','confidently_wrong'}
P_DERIVED={'semantic_error':'int(resnet50_prediction != resnet50_source_prediction)',
    'dino_specificity':'float(dino_cosine) - float(dino_mismatched)'}
BOOL_METRICS={'resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong',
    'convnext_top1_label','convnext_top1_source_prediction','convnext_source_prediction_agreement',
    'convnext_source_correct_to_wrong','convnext_source_wrong_to_correct'}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(mapping):
    for p,s in mapping.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)


def module(path,name):
    sys.path.insert(0,str(Path(path).parent))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def integer(v):
    require(type(v) is int or (type(v) is str and re.fullmatch(r'-?(0|[1-9][0-9]*)',v)), 'Exact integer field required')
    return int(v)


def numeric(v):
    require(type(v) in (int,float,bool,str) and (type(v) is not str or v.strip()==v and v), 'Numeric scalar required')
    x=float(v);require(math.isfinite(x),'Missing/nonfinite scalar cannot be imputed');return x


def value(row,metric,branch):
    if branch=='P' and metric in P_MISSING:raise ValueError('Old P scalar is unavailable: '+metric)
    if branch=='P' and metric=='semantic_error':return int(integer(row['resnet50_prediction'])!=integer(row['resnet50_source_prediction']))
    if branch=='P' and metric=='dino_specificity':return numeric(row['dino_cosine'])-numeric(row['dino_mismatched'])
    x=numeric(row[metric])
    if metric in BOOL_METRICS:
        require(x in (0,1),'Boolean metric is not zero or one');return bool(x)
    return x


def float_pixels(array):
    x=np.asarray(array)
    require(x.dtype==np.uint8 and x.shape==(3,256,256),'Original uint8 CHW pixels required; no resampling')
    return np.ascontiguousarray(x.astype(np.float32)/np.float32(255.))


def target_hashes(pixels):
    x=float_pixels(pixels);raw=np.ascontiguousarray(pixels).tobytes()
    return dict(preprocessing_sha256=hashlib.sha256(raw).hexdigest(),
        H=hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest(),
        P=hashlib.sha256(b'float32:3,256,256:RGB\0'+x.tobytes()).hexdigest(),
        common=hashlib.sha256(b'VARCOMM:EXACT_SHARED_TARGET:uint8:CHW:3,256,256\0'+raw).hexdigest())


def target_record(index,sid,h_pixels,p_pixels,h_pre,p_pre,h_hash,p_hash):
    """Called only on independently bound branch source pixels, never reconstructions."""
    require(type(index)is int and 0<=index<100 and type(sid)is str and sid,'Original source identity required')
    hh=target_hashes(h_pixels);ph=target_hashes(p_pixels)
    require(np.array_equal(h_pixels,p_pixels),'H/P target pixels differ, even if their labels agree')
    require(hh['preprocessing_sha256']==h_pre and ph['preprocessing_sha256']==p_pre
        and hh['H']==h_hash and ph['P']==p_hash,'Original preprocessing/reference hash domain differs')
    require(hh['common']==ph['common'],'Common target identity differs')
    return dict(source_index=index,source_id=sid,reference_sha256=hh['common'],
        common_hash_domain='VARCOMM:EXACT_SHARED_TARGET:uint8:CHW:3,256,256\\0',
        original_reference_sha256={'H':h_hash,'P':p_hash},
        original_hash_domains={'H':'str(float32)+str((3,256,256))+contiguous_float32_bytes',
            'P':'float32:3,256,256:RGB\\0+contiguous_float32_bytes'},
        preprocessing_sha256={'H':h_pre,'P':p_pre},pixel_equality='EXACT_UINT8_AND_FLOAT32_CONVERSION',
        shape=[3,256,256],channel_order='RGB_CHW',new_metric_calls=0)


def load_target(manifest,completion,population,index):
    """Narrow source-pixel loader. No tokens or reconstructed images are extracted."""
    ids=population['source_ids'];require(len(ids)==len(set(ids))==100 and 0<=index<100,'Original100 source list required')
    require(population['stage']==population['calibration_or_development']=='m1_development'
        and manifest['source_ids']==ids and len(manifest['records'])==100,'Wrong split/source order')
    require(manifest['status']==completion['status']=='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE',
        'Original source-only asset receipt required')
    item=manifest['records'][index];cp=Path(item['checkpoint'])
    require(completion['outputs'].get(str(cp))==item['checkpoint_sha256']==sha(cp),'Source checkpoint not sealed')
    c=read(cp);sid=ids[index];pre=population['preprocessing_ids'][index]
    for r in (item,c):
        require(r['source_index']==index and r['source_id']==sid and r['preprocessing_id']==pre
            and r['original_development_data_binding']==population['data_bindings'][index],'Wrong original source asset')
    require(item['archive']==c['archive'] and item['evaluation_class_index']==c['evaluation_class_index'],'Original source archive/label differs')
    verify(c['outputs']);ap=Path(c['archive'])
    require(c['outputs'].get(str(ap))==completion['outputs'].get(str(ap))==sha(ap),'Pixel archive unsealed')
    with np.load(ap,allow_pickle=False) as z:pixels=z['pixels'].copy()
    require(target_hashes(pixels)['preprocessing_sha256']==pre,'Original pixel bytes changed')
    return pixels,dict(source_id=sid,source_index=index,preprocessing_id=pre,true_class=c['evaluation_class_index'],
        input_bindings={str(cp):sha(cp),str(ap):sha(ap)})


def bridge_targets(h,p):
    """Actually compare the two independently admitted original pixel arrays."""
    require(h['normal_owner_success'] is True and p['normal_owner_success'] is True
        and h['source_ids']==p['source_ids'],'Closed identical source populations required')
    references={}
    for branch,data in (('H',h),('P',p)):
        refs={}
        for row in data['rows']:
            i=integer(row['source_index']);v=row['reference_sha256']
            require(i not in refs or refs[i]==v,'Branch source reference changed between frames');refs[i]=v
        require(set(refs)==set(range(100)),'All100 original source references required');references[branch]=refs
    records=[]
    for i,sid in enumerate(h['source_ids']):
        hp,hi=load_target(h['manifest'],h['assets'],h['population'],i)
        pp,pi=load_target(p['manifest'],p['assets'],p['population'],i)
        require(hi['source_id']==pi['source_id']==sid and hi['true_class']==pi['true_class'],'Original labels differ')
        for branch,info in (('H',hi),('P',pi)):
            for path,s in info['input_bindings'].items():
                require({'H':h,'P':p}[branch]['bindings'].get(path)==s,'Pixel proof is not bound to that branch')
        r=target_record(i,sid,hp,pp,hi['preprocessing_id'],pi['preprocessing_id'],references['H'][i],references['P'][i])
        r.update(true_class=hi['true_class'],branch_input_bindings={'H':hi['input_bindings'],'P':pi['input_bindings']})
        records.append(r)
    return records


def load_h(request,aggregate_module):
    require(sha(aggregate_module)==H_AGGREGATE_SHA,'Frozen H aggregate entry differs')
    a=module(aggregate_module,'normalized_original_h_aggregate')
    ctx,closed=a.normal_metrics_closed(request);rows,extra=a.load_rows(ctx,closed)
    return dict(branch='H',normal_owner_success=True,rows=rows,source_ids=ctx['source_ids'],schedule=ctx['schedule'],
        policy_sha256=sha(ctx['group']['cfg']['finalized']),completion=closed['done'],
        population=ctx['population'],manifest=ctx['manifest'],assets=ctx['assets'],
        bindings=a.merge(ctx['bound'],closed['bindings'],closed['done']['outputs'],extra),
        metric_provenance=closed['done']['metric_metadata'],context=ctx)


def load_p(spec,driver_module):
    """Future CPU admission after actual P owner closure; never call model constructors."""
    require(set(spec)=={'config','owner_config','registration','launch','completion'}
        and sha(driver_module)==P_DRIVER_SHA,'Exact completed P replay batch and driver required')
    d=module(driver_module,'normalized_original_p_replay');ctx=d.load_registered(spec['config']);cfg=ctx['cfg']
    require(cfg['registration']==spec['registration'] and cfg['visual_owner_config']==spec['owner_config'],'P lineage changed')
    expected=dict(status=d.DONE,source_count=100,frame_count=600,P_policy_snr_points=2,N=1024,
        snrs_db=[13,19],noise_seeds=list(NOISE['P']),RGB_exact_parity=True,legacy_scalars_preserved=True,
        ConvNeXt_scored=True,source_predictions=100,reconstruction_predictions=600,new_packet_decodes=0,
        new_noise_draws=0,budget_writes=0,policy_selection=False,development_used=True,holdout_used=False,
        MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False)
    original=ctx['prior']['context'];api=ctx['owner']
    closed=original['cpu'].closed_batch(api,original['wait'],dict(config=spec['owner_config'],registration=spec['registration'],
        launch=spec['launch'],completion=spec['completion']),cfg['owner_module'],expected,api.raw_process_state)
    old=closed['owner'];done=closed['done'];job=old['stages'][0]['jobs'][0]
    require(len(old['stages'])==1 and old['stages'][0]['id']=='development' and old['stages'][0]['resource']=='gpu' and len(old['stages'][0]['jobs'])==1
        and job['id']=='p600_received_replay' and job['argv'][1:]==['-B',str(driver_module),'--config',spec['config']]
        and job['out']==cfg['out'],'Wrong normal P replay command')
    require(done['budget_before']==done['budget_after']==ctx['before']==closed['owner_done']['budget']
        and done['registration_sha256']==sha(spec['registration']) and done['config_sha256']==sha(spec['config'])
        and done['source_ids']==ctx['corectx']['source_ids'],'P budget/config/source identity differs')
    for k in ('source_bindings','input_bindings','outputs'):verify(done[k])
    rows=[];base=Path(cfg['out'])
    for i,sid in enumerate(done['source_ids']):
        cp=base/'source_checkpoints'/f'{i:04d}.json';rp=base/'sources'/f'{i:04d}.json'
        for p in (cp,rp):require(done['outputs'].get(str(p))==sha(p),'P source output not sealed')
        c=read(cp);verify(c['outputs']);verify(c['input_bindings'])
        require(c['status']=='P600_RECEIVED_REPLAY_SOURCE_COMPLETE' and c['source_index']==i and c['source_id']==sid
            and c['frame_count']==c['RGB_parity_frames']==c['ConvNeXt_frames']==6
            and c['registration_sha256']==done['registration_sha256'] and c['outputs'].get(str(rp))==sha(rp),
            'P source checkpoint differs')
        part=read(rp);require(len(part)==6,'Incomplete P source');rows.extend(part)
    fp=base/'frame_metrics.json';require(done['outputs'].get(str(fp))==sha(fp) and rows==read(fp),'P flat rows differ')
    ctx['core'].validate_complete(ctx['corectx'],rows)
    return dict(branch='P',normal_owner_success=True,rows=rows,source_ids=done['source_ids'],
        policy_sha256=ctx['corectx']['selected_P_sha256'],completion=done,
        population=ctx['population'],manifest=ctx['manifest'],assets=ctx['assets'],
        bindings=d.merge(ctx['bound'],closed['bindings'],done['outputs']),
        metric_provenance=dict(original_metrics_registration=read(cfg['metrics_registration']),
            classifier_identity=done['classifier_identity'],replay_numerical_runtime=done['numerical_runtime']),context=ctx)


def admit_metrics(contract,branches):
    """Validate a separately frozen, source-reviewed metric identity admission.

    Identity dictionaries are deliberate preregistration assertions, with exact
    original evidence pins. This does not infer equivalent models from names.
    """
    require(contract['status']=='FROZEN_H_P_COMMON_METRIC_ADMISSION' and contract['used_for_selection'] is False
        and set(contract['metrics'])==set(METRICS),'Explicit21-metric admission inventory required')
    verify(contract['input_bindings']);identities={};availability=[]
    for m in METRICS:
        entry=contract['metrics'][m];status=entry['status']
        if m in P_MISSING:
            require(status=='MISSING_IN_P','Missing original P values cannot be inferred or filled')
        if status!='ADMITTED':
            require(status in ('MISSING_IN_P','UNRESOLVED_IDENTITY'),'Unknown missing/identity disposition')
            availability.append(dict(metric=m,status=status,reason=entry['reason']));continue
        require(set(entry['branches'])=={'H','P'},'Both branch metric provenance records required')
        a,b=entry['branches']['H'],entry['branches']['P']
        for branch,proof in entry['branches'].items():
            require(set(proof['identity'])=={'definition','models','preprocessing'} and proof['identity']['definition']
                and proof['identity']['preprocessing'] and type(proof['identity']['models'])is dict,
                'Definition/model weights/preprocessing identity must be explicit')
            require(all(type(k)is str and k and type(v)is str and re.fullmatch('[0-9a-f]{64}',v)
                for k,v in proof['identity']['models'].items()),'Explicit model weight SHA256 required')
            require(proof['evidence'] and type(proof['precision'])is str and proof['precision']
                and type(proof['batch_execution'])is str and proof['batch_execution'],'Numerical and batch provenance absent')
            for p,s in proof['evidence'].items():
                require(branches[branch]['bindings'].get(p)==contract['input_bindings'].get(p)==s,
                    'Metric evidence is not from that admitted branch')
            if m=='ms_ssim':
                require(proof['identity']['models']=={},'MS-SSIM has no learned model weights')
                implementation=proof.get('implementation',{})
                require(implementation and all(proof['evidence'].get(p)==s for p,s in implementation.items()),
                    'MS-SSIM implementation must be explicitly bound as source, not model weights')
            elif m!='psnr_db':require(proof['identity']['models'],'Neural metric/classifier weight identity absent')
        require(a['identity']==b['identity'],'Metric definition/model/preprocessing mismatch: '+m)
        if m=='ms_ssim':require(a['implementation']==b['implementation'],'MS-SSIM implementation differs')
        if m=='psnr_db':
            require(a['precision']=='H_float64_MSE_then_minus10_log10'
                and b['precision']=='P_retained_original_float32_Torch_PSNR_scalar'
                and entry['numerical_difference_admitted'] is True,'Original PSNR precision difference must be explicit')
        elif (a['precision'],a['batch_execution'])!=(b['precision'],b['batch_execution']):
            require(entry['numerical_difference_admitted'] is True,'Undeclared numerical/batch provenance difference')
        require(entry['numerical_difference_note'] and entry['definition_reviewed_before_comparison'] is True,
            'Prior identity review and numerical interpretation required')
        identities[m]=identity(dict(metric=m,identity=a['identity'],both_branch_numerics={k:{j:v[j] for j in
            ('precision','batch_execution')} for k,v in entry['branches'].items()},
            admission_sha256=identity(contract)))
        availability.append(dict(metric=m,status='ADMITTED',value_origin_P='DERIVED_FROM_RETAINED_SCALARS' if m in P_DERIVED else 'ORIGINAL_RETAINED_VALUE',
            derivation=P_DERIVED.get(m),metric_identity=identities[m]))
    require(identities,'No common metric has completed identity admission')
    return identities,availability


def normalize(h,p,targets,contract):
    """Return only a new view; do not compute means, choose pairs or alter old rows."""
    require(h['branch']=='H' and p['branch']=='P' and h['normal_owner_success'] is True and p['normal_owner_success'] is True,
        'Both normally completed branches required; pending P cannot be imputed')
    ids=h['source_ids'];require(ids==p['source_ids'] and len(ids)==len(set(ids))==100 and len(targets)==100,'Exact original100 required')
    metrics,availability=admit_metrics(contract,{'H':h,'P':p});points={};out=[];seen={};truth={};mismatch_truth={}
    for i,t in enumerate(targets):
        require(t['source_index']==i and t['source_id']==ids[i] and t['pixel_equality']=='EXACT_UINT8_AND_FLOAT32_CONVERSION',
            'Exact target pixel bridge missing')
    for branch,data in (('H',h),('P',p)):
        require(len(data['rows'])==(5400 if branch=='H' else 600),'Full H18/P2 frame grid required')
        for row in data['rows']:
            i=integer(row['source_index']);n=integer(row['noise_seed']);snr=integer(row['snr_db']);N=integer(row['N'])
            require(0<=i<100 and row['source_id']==ids[i] and n in NOISE[branch] and snr in (13,19) and N==1024,'Source/grid/noise differs')
            if branch=='H':
                slot=integer(row['development_slot']);require(0<=slot<18 and row['point_id']==f'H18_SLOT_{slot:02d}'
                    and row['used_for_selection'] is False and row['holdout_used'] is False,'H declared point changed')
                point=row['point_id'];label=integer(row['true_class'])
            else:
                require(row['method']=='P1024' and row['p_replay_exact_RGB'] is True
                    and row['p_replay_new_noise_draws']==row['p_replay_new_packet_decodes']==0,'P original replay proof absent')
                point=f'P1024_SNR_{snr}';label=integer(row['true_class_index'])
            target=targets[i];require(row['reference_sha256']==target['original_reference_sha256'][branch],'Original reference hash differs')
            predictions=(label,integer(row['resnet50_source_prediction']),integer(row['convnext_source_prediction']))
            require(label==target['true_class'] and all(0<=v<1000 for v in predictions)
                and (i not in truth or truth[i]==predictions),'Target labels/source predictions differ')
            if 'dino_mismatched' in metrics or 'dino_specificity' in metrics:
                negative=row['mismatch_source_id' if branch=='H' else 'replay_mismatch_source_id']
                require(negative in ids and negative!=ids[i] and (i not in mismatch_truth or mismatch_truth[i]==negative),
                    'Original DINO mismatch permutation differs between branches')
                mismatch_truth[i]=negative
            truth[i]=predictions;key=(point,i,n);require(key not in seen,'Duplicate normalized source/noise');seen[key]=True
            spec=dict(branch=branch,snr_db=snr,noise_seeds=list(NOISE[branch]),policy_sha256=data['policy_sha256'],metric_identity=metrics)
            require(point not in points or points[point]==spec,'Frozen point identity changed');points[point]=spec
            values={m:value(row,m,branch) for m in metrics}
            out.append(dict(values,point_id=point,source_index=i,source_id=ids[i],branch=branch,N=N,snr_db=snr,noise_seed=n,
                policy_sha256=data['policy_sha256'],metric_identity=metrics,reference_sha256=target['reference_sha256'],
                true_class=predictions[0],resnet50_source_prediction=predictions[1],convnext_source_prediction=predictions[2],
                used_for_selection=False,holdout_used=False,original_row_sha256=identity(row),
                original_reference_sha256=row['reference_sha256'],original_reference_hash_domain=target['original_hash_domains'][branch],
                normalization_only=True,original_noise_seed_preserved=True,frame_level_common_noise_claimed=False))
    expectedpoints={f'H18_SLOT_{i:02d}' for i in range(18)}|{'P1024_SNR_13','P1024_SNR_19'}
    require(set(points)==expectedpoints and set(seen)=={(point,i,n) for point in points for i in range(100) for n in points[point]['noise_seeds']},
        'Missing point/source/noise cannot be silently intersected')
    return dict(status='H_P_NORMALIZED_VIEW_NOT_STATISTICAL_RESULT',rows=out,points=points,metrics=list(metrics),source_ids=ids,
        availability=availability,target_bridge=targets,common_metric_admission_sha256=identity(contract),
        original_H_rows_sha256=identity(h['rows']),original_P_rows_sha256=identity(p['rows']),
        MAIN_complete=False,P_complete=True,overall_system_conclusion=False,policy_selection=False,
        frame_level_noise_pairing_claimed=False,metrics_recomputed=False,comparisons_selected=False,
        statistics_contract=dict(module='h_p_main_source_statistics.py',sha256=STATISTICS_SHA,
            requires='Explicit predeclared same-SNR comparisons; average each branch original3 noises per source, then bootstrap10000 seed2026100605'))
