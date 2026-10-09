"""Score only new adaptive-BPG cached RGB and new source-paired comparisons.

The unchanged original FullSuite/ConvNeXt construction and bootstrap functions
are imported by hash. Old means, intervals, and same-image metrics are read-only.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

SNRS=[1,4,7,10,13,19]
SEEDS=[2001,2002,2003]
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
METHOD='BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'
STOP=False


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def descriptor(d):
    require(sha(d['path'])==d['sha256'],'Pinned input changed: '+d['path']);return read(d['path'])


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists():require(p.read_bytes()==data,'Immutable output differs: '+str(p));return
    tmp=p.with_suffix(p.suffix+'.pending')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def load_module(descriptor_,name):
    require(sha(descriptor_['path'])==descriptor_['sha256'],'Original executable changed')
    spec=importlib.util.spec_from_file_location(name,descriptor_['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def verify_outputs(value):
    for path,expected in value['outputs'].items():require(sha(path)==expected,'Completed output changed: '+path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def csv_write(path,rows):
    import io
    f=io.StringIO(newline='');w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    p=Path(path);data=f.getvalue().encode()
    if p.exists():require(p.read_bytes()==data,'Existing CSV differs');return
    with p.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def authenticate(r):
    request=descriptor(r['adaptive_request']);normal=descriptor(r['adaptive_holdout_normal'])
    freeze_normal=descriptor(r['adaptive_freeze_normal']);freeze=descriptor(r['adaptive_freeze_worker'])
    for completion,stage,count in [(normal,'holdout',8),(freeze_normal,'freeze',1)]:
        require(completion['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1' and completion['stage']==stage
                and completion['request_sha256']==r['adaptive_request']['sha256']
                and completion['actual_children_waited'] and completion['worker_exit_codes']==[0]*count
                and completion['ledger']['unresolved']==0
                and completion['root_budget_before']==completion['root_budget_after'], 'Actual closed adaptive stage required')
    require(normal['frame_count']==9000 and freeze_normal['outputs'].get(r['adaptive_freeze_worker']['path'])==r['adaptive_freeze_worker']['sha256'], 'Actual adaptive holdout/freeze binding missing')
    require(freeze['request_sha256']==r['adaptive_request']['sha256'] and freeze['calibration_source_count']==100
            and freeze['noise_count']==3 and not freeze['holdout_used_for_selection'], 'Calibration-only adaptive freeze required')
    require(request['SNRs']==SNRS and request['noise_seeds']==SEEDS and not request['holdout_used_for_selection'], 'Frozen new baseline scope differs')
    final=descriptor(r['original_FINAL_FREEZE']);ids=descriptor(r['source_manifest'])['source_ids']
    require(ids==final['population']['source_ids']==[x['source_id'] for x in request['records']['holdout']]
            and len(ids)==len(set(ids))==500,'Exact original500 population/order required')
    # Only finite stage-control files here; each consumed source/image is checked below.
    for path,expected in normal['outputs'].items():
        if '/attempt_' in path or Path(path).name.startswith('worker_'):
            require(sha(path)==expected,'Actual holdout wait/log changed')
    return request,normal,freeze,ids,final


def source(np,old,request,normal,freeze,index,sid):
    path=request['out']+'/holdout/sources/%04d.json'%index
    require(normal['outputs'].get(path)==sha(path),'Completed adaptive source checkpoint changed')
    cp=read(path);record=request['records']['holdout'][index];archive=cp['float_reconstructions']
    require(cp['status']=='ADAPTIVE_BPG_SOURCE_RX_COMPLETE_V1' and cp['method']==METHOD
            and cp['source_index']==index and cp['source_id']==sid==record['source_id']
            and cp['request_sha256']==normal['request_sha256'] and len(cp['rows'])==18
            and cp['preprocessing_id']==record['preprocessing_id']
            and cp['evaluation_class_index']==record['evaluation_class_index'], 'Adaptive source provenance differs')
    require(sha(record['checkpoint'])==record['checkpoint_sha256'] and sha(record['archive'])==record['archive_sha256'], 'Original source archive/checkpoint changed')
    require(archive['dtype']=='float32' and archive['layout']=='CHW' and archive['lossless'] and archive['rows']==18
            and normal['outputs'].get(archive['path'])==cp['outputs'].get(archive['path'])==archive['sha256']==sha(archive['path']), 'Float reconstruction binding differs')
    with np.load(archive['path'],allow_pickle=False) as z:
        require(set(z.files)=={'images','source_rgb','row_ids','image_slots'},'Unexpected cached image schema')
        images,target,rowids,slots=[z[k].copy() for k in ('images','source_rgb','row_ids','image_slots')]
    require(images.shape==(archive['unique_images'],3,256,256) and images.dtype==np.float32
            and slots.dtype==np.int64 and slots.shape==(18,) and rowids.shape==(18,)
            and rowids.tolist()==archive['row_ids']==[x['frame_id'] for x in cp['rows']]
            and slots.tolist()==archive['image_slots']==[x['image_slot'] for x in cp['rows']]
            and set(slots.tolist())==set(range(len(images)))
            and old.rgb_sha(np,target)==archive['reference_sha256']==cp['reference_sha256'], 'Image slots/reference identity differs')
    with np.load(record['archive'],allow_pickle=False) as z:pixels=z['pixels'].copy()
    require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
            and hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id']
            and np.array_equal(target,pixels.astype(np.float32)/np.float32(255)), 'Wrong original source pixels')
    require([old.rgb_sha(np,x) for x in images]==archive['image_sha256']
            and len(set(archive['image_sha256']))==len(images),'Exact per-source reconstruction dedup required')
    for j,row in enumerate(cp['rows']):
        require((row['snr_db'],row['noise_seed'])==(SNRS[j//3],SEEDS[j%3])
                and row['source_id']==sid and row['source_index']==index
                and row['profile_id']==freeze['selected_profile_ids'][str(row['snr_db'])]
                and row['image_sha256']==archive['image_sha256'][slots[j]]
                and row['reference_sha256']==cp['reference_sha256'] and row['allocated_symbols']==1024
                and row['quality_selection'] is False,'Wrong frozen same-source frame')
    return dict(cp=cp,target=target,images=images,path=path,sha256=sha(path),record=record)


def prepared_cache(np,torch,backend,r,record,target,index,guard):
    """Losslessly persist the original prepare result; never serialize executable pickle."""
    root=Path(r['source_feature_cache']);root.mkdir(parents=True,exist_ok=True)
    identity=dict(source_id=record['source_id'],source_archive_sha256=record['archive_sha256'],
                  preprocessing_id=record['preprocessing_id'],reference_rgb_sha256=r['_rgb_sha'](np,target),
                  evaluator_identity=backend.identity)
    key=digest(identity);cp=root/(key+'.json');archive=root/(key+'.npz');reservation=root/(key+'.reserved.json')
    if cp.exists():
        done=read(cp);require(done['identity']==identity and sha(archive)==done['archive_sha256'],'Source feature cache identity differs')
        with np.load(archive,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        def restore(node):
            kind=node['kind']
            if kind=='tensor':return torch.from_numpy(arrays[node['key']]).to(node['device'])
            if kind=='ndarray':return arrays[node['key']]
            if kind=='dict':return {k:restore(v) for k,v in node['items'].items()}
            if kind in ('list','tuple'):
                values=[restore(v) for v in node['items']];return tuple(values) if kind=='tuple' else values
            require(kind=='scalar','Unknown prepared-cache schema');return node['value']
        return restore(done['tree']),False,{str(cp):sha(cp),str(archive):sha(archive)}
    require(not reservation.exists() and not archive.exists(),'Unresolved source feature preparation; no automatic repeat')
    save(reservation,dict(identity=identity,status='RESERVED'));guard()
    prepared=backend.prepare(target);arrays={}
    def freeze(node):
        if torch.is_tensor(node) or isinstance(node,np.ndarray):
            key='a%04d'%len(arrays);tensor=torch.is_tensor(node)
            arrays[key]=node.detach().cpu().numpy() if tensor else np.asarray(node)
            require(arrays[key].dtype.kind!='O','Object array cannot be a frozen source feature')
            return dict(kind='tensor' if tensor else 'ndarray',key=key,**({'device':str(node.device)} if tensor else {}))
        if isinstance(node,dict):
            require(all(isinstance(k,str) for k in node),'Nonstring prepared feature key')
            return dict(kind='dict',items={k:freeze(v) for k,v in node.items()})
        if isinstance(node,(list,tuple)):return dict(kind='tuple' if isinstance(node,tuple) else 'list',items=[freeze(v) for v in node])
        if isinstance(node,np.generic):node=node.item()
        require(node is None or type(node) in (str,int,float,bool),'Unsupported prepared cache value')
        return dict(kind='scalar',value=node)
    tree=freeze(prepared)
    with archive.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    save(cp,dict(identity=identity,archive_sha256=sha(archive),tree=tree,status='ORIGINAL_PREPARE_COMPLETE'))
    return prepared,True,{str(cp):sha(cp),str(archive):sha(archive)}


def metric_cache(path,identity,request_hash=None):
    reserved,complete={},{}
    for line in Path(path).read_text().splitlines():
        event=json.loads(line);key=event['key']
        require(event['evaluator_identity']==identity and (request_hash is None or event['request_sha256']==request_hash),'Quality cache evaluator/request differs')
        if event['status']=='RESERVED':require(key not in reserved,'Duplicate reservation');reserved[key]=event
        else:
            require(event['status']=='COMPLETE' and key in reserved and key not in complete,'Invalid quality cache completion')
            complete[key]=event['metrics']
    require(set(reserved)==set(complete),'Unresolved quality call must not be repeated')
    return reserved,complete


def old_tables(r,ids,backend_identity):
    main=descriptor(r['original_statistics_completion']);native=descriptor(r['native_BPG_metrics_completion'])
    require(main['source_ids']==native['source_ids']==ids and main['scientific_statistics_completed'] is True
            and main['final_freeze_sha256']==r['original_FINAL_FREEZE']['sha256']
            and native['metric_evaluator_identity']==backend_identity, 'Old source statistics/metric identity differ')
    result={};summaries=[]
    for origin,normal,means_descriptor,summary_descriptor in [
        ('published_common500',main,r['original_source_means'],r['original_summary']),
        ('native256_BPG',native,r['native_BPG_source_means'],r['native_BPG_summary'])]:
        for d in [means_descriptor,summary_descriptor]:require(normal['outputs'].get(d['path'])==d['sha256']==sha(d['path']),'Bound original table changed')
        if means_descriptor['path'].endswith('.gz'):
            with gzip.open(means_descriptor['path'],'rt') as f:means=json.load(f)
            summary=read(summary_descriptor['path'])
        else:means=csv_rows(means_descriptor['path']);summary=csv_rows(summary_descriptor['path'])
        prefixes=set(r['paired_reference_prefixes'])
        for row in means:
            point=row['point_id'];metric=row['metric']
            if metric not in METRICS or not any(point==prefix+'_SNR_'+str(snr) for prefix in prefixes for snr in SNRS):continue
            i=int(row['source_index']);require(row['source_id']==ids[i],'Old source mean ID/order differs')
            key=(point,metric);result.setdefault(key,{})
            require(i not in result[key],'Duplicate old source mean');result[key][i]=float(row['mean'])
        for row in summary:
            if row['metric'] in METRICS and any(row['point_id']==prefix+'_SNR_'+str(snr) for prefix in prefixes for snr in SNRS):
                summaries.append(dict(point_id=row['point_id'],snr_db=int(row.get('snr_db',row['point_id'].rsplit('_',1)[1])),metric=row['metric'],
                                      mean=float(row['mean']),ci_low=float(row['ci_low']),ci_high=float(row['ci_high']),
                                      source_count=int(row['source_count']),noise_count=int(row['noise_count']),provenance=origin+'_UNCHANGED'))
    for prefix in r['paired_reference_prefixes']:
        for snr in SNRS:
            for metric in METRICS:require(set(result.get((prefix+'_SNR_'+str(snr),metric),{}))==set(range(500)),'Missing500 original source means')
    return result,summaries,native


def run(path):
    r=read(path);request_hash=sha(path)
    require(r['schema']=='ADAPTIVE_BPG_CACHED_FOUR_METRIC_REQUEST_V1' and r['metrics']==METRICS
            and r['new_quality_call_cap']==9000 and r['reference_prepare_cap']==500
            and r['new_packet_decodes']==0 and r['batch_size']==1 and r['training_updates']==0,'Finite new-baseline metric scope differs')
    require(sha(__file__)==r['worker_sha256'] and sys.platform=='linux'
            and os.path.abspath(sys.executable)==r['python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
            and sorted(os.sched_getaffinity(0))==[4,5,6,7,8,9]
            and os.getpriority(os.PRIO_PROCESS,0)==15,'Original UM FP32/B1 numeric environment required')
    def stopped(*_):
        global STOP
        STOP=True
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopped)
    started=time.monotonic();out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
    def guard():require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-started<r['max_seconds']
                        and not any(Path(p).exists() for p in r['stop_files']) and not (out/'STOP').exists(),'STOP/deadline reached')
    old=load_module(r['original_scoring_module'],'_original_native_BPG_scoring')
    import numpy as np
    try:
        with Path(r['visual_lock']).open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
            request,normal,freeze,ids,final=authenticate(r)
            before=old.root_snapshot(request['original_root_ledger_readonly'])
            require(before==normal['root_budget_after'],'Original root budget changed')
            if (out/'completion.json').exists():
                done=read(out/'completion.json');require(done['request_sha256']==request_hash,'Completed request differs');verify_outputs(done)
                return {'status':'ACTUAL_COMPLETE_REUSED_NO_NEW_CALLS','completion':str(out/'completion.json')}
            backend=old.build_backend(r,final);import torch
            reference_values,old_summary,native=old_tables(r,ids,backend.identity)
            native_ledger=r['native_BPG_quality_ledger'];require(native['outputs'].get(native_ledger['path'])==native_ledger['sha256']==sha(native_ledger['path']),'Original same-image quality cache changed')
            _,inherited=metric_cache(native_ledger['path'],backend.identity,native['request_sha256'])
            stat=load_module(r['statistics_module'],'_unchanged_source_bootstrap')
            require(stat.SOURCE_COUNT==500 and stat.REPLICATES==10000 and stat.BOOTSTRAP_SEED==2026100701,'Original bootstrap contract differs')
            permutation=final['holdout_metrics']['mismatch']['permutation']
            require(sorted(permutation)==list(range(500)) and all(i!=p for i,p in enumerate(permutation)), 'Original fixed mismatch mapping differs')
            outputs={};references=[];source_preparations=0;r['_rgb_sha']=old.rgb_sha
            for i,sid in enumerate(ids):
                guard();value=source(np,old,request,normal,freeze,i,sid)
                prepared,fresh,pins=prepared_cache(np,torch,backend,r,value['record'],value['target'],i,guard)
                source_preparations+=fresh;outputs.update(pins)
                references.append(dict(target=value['target'],prepared=prepared))
                require(source_preparations<=500,'Reference preparation cap exceeded')
            ledger=out/'quality_calls.jsonl'
            reserved,cache=metric_cache(ledger,backend.identity,request_hash) if ledger.exists() else ({},{})
            def event(value):
                with ledger.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows=[];fresh_calls=0;inherited_reuses=0
            for i,sid in enumerate(ids):
                guard();value=source(np,old,request,normal,freeze,i,sid);scores={}
                for j,image in enumerate(value['images']):
                    guard();ih=value['cp']['float_reconstructions']['image_sha256'][j]
                    key=digest(dict(source_id=sid,reference_sha256=value['cp']['reference_sha256'],image_sha256=ih,
                                    evaluator_identity=backend.identity,mismatch_source_id=ids[permutation[i]]))
                    if key in inherited:
                        scores[j]=inherited[key];inherited_reuses+=1;continue
                    if key not in cache:
                        require(len(reserved)<9000,'New unique quality cap exhausted')
                        base=dict(key=key,request_sha256=request_hash,evaluator_identity=backend.identity,
                                  source_index=i,source_id=sid,image_sha256=ih)
                        event(dict(base,status='RESERVED'));reserved[key]=base
                        result=backend.score(value['target'],image,value['cp']['evaluation_class_index'],references[i]['prepared'],references[permutation[i]]['prepared']['dino_feature'])
                        mse=float(np.square(image.astype(np.float64)-value['target'].astype(np.float64)).mean())
                        require(mse>0 and math.isfinite(mse),'Infinite PSNR requires explicit statistical treatment')
                        metric=dict(psnr_db=float(-10*math.log10(mse)),lpips_alex=float(result['lpips_alex']),
                                    dinov2_vitl14_cosine=float(result['dinov2_vitl14_cosine']),convnext_top1_source_prediction=result['convnext_top1_source_prediction'])
                        require(all(math.isfinite(float(metric[m])) for m in METRICS) and type(metric['convnext_top1_source_prediction']) is bool
                                and result['label_conditioned'] is result['convnext_used_for_selection'] is False,'Original four metrics invalid')
                        event(dict(base,status='COMPLETE',metrics=metric));cache[key]=metric;fresh_calls+=1
                    scores[j]=cache[key]
                current=[]
                for raw in value['cp']['rows']:
                    metric=scores[raw['image_slot']];require(metric['psnr_db']==raw['psnr_db'],'Received float64 PSNR differs')
                    row=dict(raw,**metric,point_id=METHOD+'_SNR_'+str(raw['snr_db']),metric_evaluator_identity=backend.identity,metric_batch_size=1)
                    rows.append(row);current.append(row)
                cp=out/'source_checkpoints'/('%04d.json'%i)
                save(cp,dict(status='ADAPTIVE_BPG_FOUR_METRIC_SOURCE_COMPLETE',request_sha256=request_hash,source_id=sid,source_index=i,
                             rows=current,original_received_CP_sha256=value['sha256'],metric_evaluator_identity=backend.identity,frame_count=18))
                outputs[str(cp)]=sha(cp)
                print(json.dumps({'sources_complete':i+1,'new_unique_quality_charged':len(reserved),'old_same_image_reuses':inherited_reuses}),flush=True)
            require(len(rows)==9000 and set(reserved)==set(cache),'Incomplete fixed500x6x3 quality results')
            guard();samples=stat.draws();summaries=[];means=[];pairs=[]
            for snr in SNRS:
                group=[x for x in rows if x['snr_db']==snr];require(len(group)==1500,'Missing fixed SNR rows')
                for metric in METRICS:
                    values=np.asarray([np.mean([x[metric] for x in group if x['source_index']==i],dtype=np.float64) for i in range(500)],dtype=np.float64)
                    point=METHOD+'_SNR_'+str(snr)
                    summaries.append(dict(point_id=point,snr_db=snr,metric=metric,**stat.interval(values,samples)))
                    means.extend(dict(point_id=point,metric=metric,source_index=i,source_id=sid,mean=float(values[i])) for i,sid in enumerate(ids))
                    for prefix in r['paired_reference_prefixes']:
                        reference=prefix+'_SNR_'+str(snr)
                        previous=np.asarray([reference_values[reference,metric][i] for i in range(500)],dtype=np.float64)
                        pairs.append(dict(method=point,reference=reference,snr_db=snr,metric=metric,delta_definition='method minus reference',
                                          frame_level_noise_pairing_claimed=False,**stat.interval(values-previous,samples)))
            combined=old_summary+[dict(point_id=x['point_id'],snr_db=x['snr_db'],metric=x['metric'],mean=x['mean'],ci_low=x['ci_low'],ci_high=x['ci_high'],
                                      source_count=x['source_count'],noise_count=x['noise_count'],provenance='ADAPTIVE_BPG_NEW_BASELINE') for x in summaries]
            for name,table in [('summary',summaries),('source_means',means),('paired',pairs),('failure_breakdown',old.breakdown(rows)),('comparison_summary',combined)]:
                p=out/(name+'.csv');csv_write(p,table);outputs[str(p)]=sha(p)
            save(out/'rows.json',rows);outputs[str(out/'rows.json')]=sha(out/'rows.json')
            if ledger.exists():outputs[str(ledger)]=sha(ledger)
            backend.check();guard();after=old.root_snapshot(request['original_root_ledger_readonly']);require(after==before,'Root budget changed')
            done=dict(status='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1',request_sha256=request_hash,worker_sha256=sha(__file__),
                      source_ids=ids,source_count=500,frame_count=9000,metrics=METRICS,summary_rows=24,paired_rows=len(pairs),source_mean_cells=12000,
                      metric_evaluator_identity=backend.identity,metric_metadata=backend.metadata,metric_batch_size=1,
                      new_quality_call_cap=9000,new_unique_quality_calls_total=len(reserved),new_quality_calls_this_run=fresh_calls,
                      inherited_native256_same_image_reuses=inherited_reuses,new_reference_preparations_this_run=source_preparations,
                      reference_cache_records=500,reference_prepare_cap=500,reference_cache_prior_audit='No persisted old source feature cache; first preparation is explicitly new',
                      bootstrap_replicates=10000,bootstrap_seed=2026100701,bootstrap_unit='source after original3-noise mean',
                      old_summary_recomputed=False,old_bootstrap_recomputed=False,new_packet_decodes=0,training_updates=0,policy_selection=False,
                      root_budget_before=before,root_budget_after=after,outputs=outputs,external_wait_required=True)
            save(out/'completion.json',done)
            return {'status':done['status'],'completion':str(out/'completion.json'),'sha256':sha(out/'completion.json')}
    except BaseException:
        save(out/('failure_'+str(time.time_ns())+'.json'),dict(status='FAILED_PRESERVE_NO_AUTO_RETRY',request_sha256=request_hash,traceback=traceback.format_exc(),new_packet_decodes=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    print(json.dumps(run(parser.parse_args().request)))
