"""Frozen four-metric scoring plus same-RX CRC-DROP gray diagnostics.

Reuses the actual original SuiteBackend, source feature caches, and completed
same-source image metrics. It performs no source encoding, PHY, or VAR render.
"""
import argparse,csv,gzip,hashlib,importlib.util,json,math,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from t1_entropy_core import read,require,sha,verify,write,csv_write
from t1_holdout_metadata import FAMILIES,SNRS,SEEDS,RAW_DROP,pinned,pin
from t1_holdout_statistics import METRICS,bound_output
import t2_pilot as shared

def load_module(d,name):
    verify(d['path'],d['sha256']);s=importlib.util.spec_from_file_location(name,d['path'])
    value=importlib.util.module_from_spec(s);sys.modules[name]=value;s.loader.exec_module(value);return value

def prepare(args):
    rendered=read(args.render_completion)
    require(rendered['status']=='T1_POSTHOC_COMMON500_RENDER_COMPLETE' and rendered['source_count']==500
        and rendered['frame_count']==9000,'Actual closed entropy render grid required')
    source=pinned(rendered['source_request']);policy=pinned(source['freeze'])
    require(rendered['freeze']==source['freeze'] and rendered['original_statistics_completion']==source['original_statistics_completion'],
        'Renderer must preserve original frozen source/statistics bindings')
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and not policy['holdout_used_for_selection'],'Frozen calibration-only policies required')
    template=read(args.metric_template);original=pinned(template['original_FINAL_FREEZE'])
    require(original['population']['source_ids']==rendered['source_ids']==source['source_ids'],'Original common500 order differs')
    require(sha(args.original_scoring_module)==template['worker_sha256'],'Original completed metric worker implementation differs')
    adaptive=read(args.adaptive_metrics_completion)
    require(adaptive['status']=='ADAPTIVE_BPG_FOUR_METRICS_NEW_PAIRED_STATISTICS_COMPLETE_V1'
        and adaptive['source_ids']==source['source_ids'] and sha(args.adaptive_scoring_module)==adaptive['worker_sha256'],
        'Completed original source-feature/image cache and its implementation required')
    native_path=Path(template['out'])/'completion.json';native=read(native_path)
    require(native['status']=='BPG_CACHED_FOUR_METRICS_AND_NEW_SOURCE_STATISTICS_COMPLETE_V1'
        and native['source_ids']==source['source_ids'] and native['request_sha256']==sha(args.metric_template)
        and native['worker_sha256']==template['worker_sha256']
        and native['metric_evaluator_identity']==adaptive['metric_evaluator_identity'],
        'Completed original native-BPG inherited image cache required')
    diagnostic=read(args.raw_diagnostic_completion)
    require(diagnostic['status']=='T1_RAW_CRC_DROP_METADATA_COMPLETE_NOT_SCORED' and diagnostic['diagnostic_frame_count']==9000,
        'Exact original same-RX diagnostic mapping required')
    oldstat=source['original_statistics_completion']
    require(diagnostic['input_bindings'].get(oldstat['path'])==oldstat['sha256'],
        'Raw diagnostic must bind the same original statistics completion')
    for path,digest in diagnostic['outputs'].items():verify(path,digest)
    out=Path(args.out).resolve();root=Path(source['root'])
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh WCL metric output required')
    request=dict(schema='T1_POSTHOC_COMMON500_METRIC_REQUEST_V1',out=str(out),root=str(root),
        render_completion=pin(args.render_completion),source_request=rendered['source_request'],freeze=source['freeze'],
        original_statistics_completion=source['original_statistics_completion'],
        metric_template=pin(args.metric_template),original_scoring_module=pin(args.original_scoring_module),
        native_metrics_completion=pin(native_path),
        adaptive_scoring_module=pin(args.adaptive_scoring_module),adaptive_metrics_completion=pin(args.adaptive_metrics_completion),
        raw_diagnostic_completion=pin(args.raw_diagnostic_completion),source_ids=source['source_ids'],source_count=500,
        metrics=METRICS,frame_count=18000,entropy_frames=9000,raw_diagnostic_frames=9000,
        source_reference_prepare_cap=500,new_unique_quality_cap=9000+diagnostic['gray_sources'],
        old_metric_intervals_modified=False,policy_selection=False,post_hoc_supplement=True,
        deadline_unix=args.deadline_unix,max_seconds=args.max_seconds,
        stop_files=[str(root/'STOP'),str(out/'STOP')],worker_sha256=sha(__file__))
    request['source_bindings']={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n))
        for n in ('t1_holdout_metrics.py','t1_holdout_metadata.py','t1_holdout_statistics.py','t1_entropy_core.py','t2_pilot.py')}
    out.mkdir(parents=True);write(out/'request.json',request);return request

def run(path):
    r=read(path);rh=sha(path);require(r['schema']=='T1_POSTHOC_COMMON500_METRIC_REQUEST_V1' and sha(__file__)==r['worker_sha256'],'Frozen metric request/worker required')
    for p,h in r['source_bindings'].items():verify(p,h)
    template=pinned(r['metric_template']);require(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and os.path.abspath(sys.executable)==template['python'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
        'Original metric environment required')
    require(sorted(os.sched_getaffinity(0))==[4,5,6,7,8,9] and os.getpriority(os.PRIO_PROCESS,0)==15,
        'Original admitted metric CPU affinity/priority required')
    out=Path(r['out']);require(not (out/'completion.json').exists() and not (out/'owner_started.json').exists(),'Do not repeat an existing metric owner attempt')
    started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    def guard():shared.guard(r,started)
    counts=dict(new_reference_preparations=0,old_reference_reuses=0,new_quality_calls=0,old_quality_reuses=0,
        native_BPG_quality_reuses=0,original_common500_quality_reuses=0)
    try:
        with shared.lock(template['visual_lock']),shared.lock(out/'gpu.lock'):
            guard();shared.save(out/'owner_started.json',dict(pid=os.getpid(),request_sha256=rh,started_unix=time.time()))
            rendered=pinned(r['render_completion']);source=pinned(r['source_request']);ids=r['source_ids']
            original=pinned(template['original_FINAL_FREEZE']);diagnostic=pinned(r['raw_diagnostic_completion'])
            previous=pinned(r['adaptive_metrics_completion'])
            native_previous=pinned(r['native_metrics_completion'])
            old=load_module(r['original_scoring_module'],'_wcl_original_four_metric_runtime')
            helper=load_module(r['adaptive_scoring_module'],'_wcl_completed_adaptive_metric_helpers')
            backend=old.build_backend(template,original);import torch
            require(backend.identity==previous['metric_evaluator_identity']==original['holdout_metrics']['metadata_identity'],
                'Actual original four-metric identity differs')
            previous_ledger=bound_output(previous,'quality_calls.jsonl')
            _,inherited=helper.metric_cache(previous_ledger,backend.identity,previous['request_sha256'])
            native_ledger=bound_output(native_previous,'quality_calls.jsonl')
            _,native_inherited=helper.metric_cache(native_ledger,backend.identity,native_previous['request_sha256'])
            for key in set(inherited).intersection(native_inherited):
                require(inherited[key]==native_inherited[key],'Same completed source/image has conflicting inherited metrics')
            permutation=original['holdout_metrics']['mismatch']['permutation']
            require(sorted(permutation)==list(range(500)) and all(i!=j for i,j in enumerate(permutation)),'Original mismatch pairing differs')
            references=[];outputs={};cache={};reserved=set();ledger=out/'quality_calls.jsonl'
            def event(value):
                with ledger.open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            for record in source['records']:
                guard();i=record['source_index'];verify(record['archive'],record['archive_sha256'])
                with np.load(record['archive'],allow_pickle=False) as z:pixels=z['pixels'].copy()
                require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
                    and hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],'Original reference pixels differ')
                target=pixels.astype(np.float32)/np.float32(255)
                identity=dict(source_id=record['source_id'],source_archive_sha256=record['archive_sha256'],
                    preprocessing_id=record['preprocessing_id'],reference_rgb_sha256=old.rgb_sha(np,target),evaluator_identity=backend.identity)
                key=helper.digest(identity)
                candidates=[p for p in previous['outputs'] if p.endswith('/source_feature_cache/'+key+'.json')]
                if candidates:
                    require(len(candidates)==1,'Ambiguous completed reference cache');cp=Path(candidates[0]);ap=cp.with_suffix('.npz')
                    verify(cp,previous['outputs'][str(cp)]);verify(ap,previous['outputs'][str(ap)])
                    feature_root=cp.parent
                else:feature_root=out/'source_feature_cache'
                cfg=dict(source_feature_cache=str(feature_root),_rgb_sha=old.rgb_sha)
                prepared,fresh,pins=helper.prepared_cache(np,torch,backend,cfg,record,target,i,guard)
                require(not candidates or not fresh,'Existing scientific source cache must never be written')
                counts['new_reference_preparations']+=int(fresh);counts['old_reference_reuses']+=int(not fresh)
                if fresh:outputs.update(pins)
                require(counts['new_reference_preparations']<=r['source_reference_prepare_cap'],'Reference cap exceeded')
                references.append(dict(target=target,prepared=prepared,record=record,reference_sha256=old.rgb_sha(np,target),
                    common_reference_sha256=shared.image_sha(target)))
            original_stats=pinned(r['original_statistics_completion'])
            original_rows_path=bound_output(original_stats,'normalized_rows.json.gz')
            points=read(bound_output(original_stats,'points.json'))
            metric_identity={m:points['RAW64_PARTIAL_VAR_COMPLETION_SNR_4']['metric_identity'][m] for m in METRICS}
            original_cache={}
            with gzip.open(original_rows_path,'rt',encoding='utf-8') as f:original_rows=json.load(f)
            for previous_row in original_rows:
                i=previous_row['source_index']
                require(previous_row['source_id']==ids[i] and previous_row['mismatch_source_id']==ids[permutation[i]]
                    and previous_row['common_reference_sha256']==references[i]['common_reference_sha256']
                    and previous_row['original_metric_evaluator_identity']==backend.identity
                    and all(previous_row['metric_identity'][m]==metric_identity[m] for m in METRICS),
                    'Original source/image metric cache identity differs')
                key=(i,previous_row['original_image_sha256']);metric={m:previous_row['metrics'][m] for m in METRICS}
                if key in original_cache:require(original_cache[key]==metric,'Same original image has inconsistent completed metrics')
                original_cache[key]=metric
            del original_rows
            def score(index,image):
                guard();ref=references[index]
                require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all(),'Actual float32 image required')
                ih=old.rgb_sha(np,image)
                key=helper.digest(dict(source_id=ids[index],reference_sha256=ref['reference_sha256'],image_sha256=ih,
                    evaluator_identity=backend.identity,mismatch_source_id=ids[permutation[index]]))
                if key in cache:return cache[key]
                original_key=(index,shared.image_sha(image))
                if original_key in original_cache:
                    counts['original_common500_quality_reuses']+=1;cache[key]=original_cache[original_key];return cache[key]
                if key in native_inherited:
                    counts['native_BPG_quality_reuses']+=1;cache[key]=native_inherited[key];return cache[key]
                if key in inherited:
                    counts['old_quality_reuses']+=1;cache[key]=inherited[key];return cache[key]
                require(len(reserved)<r['new_unique_quality_cap'],'New quality call cap exhausted')
                base=dict(key=key,request_sha256=rh,evaluator_identity=backend.identity,source_index=index,source_id=ids[index],image_sha256=ih)
                event(dict(base,status='RESERVED'));reserved.add(key)
                result=backend.score(ref['target'],image,ref['record']['evaluation_class_index'],ref['prepared'],references[permutation[index]]['prepared']['dino_feature'])
                mse=float(np.square(image.astype(np.float64)-ref['target'].astype(np.float64)).mean())
                require(mse>0 and math.isfinite(mse),'Infinite PSNR needs explicit statistics handling')
                value=dict(psnr_db=float(-10*math.log10(mse)),lpips_alex=float(result['lpips_alex']),
                    dinov2_vitl14_cosine=float(result['dinov2_vitl14_cosine']),convnext_top1_source_prediction=result['convnext_top1_source_prediction'])
                require(all(math.isfinite(float(value[m])) for m in METRICS) and type(value['convnext_top1_source_prediction']) is bool
                    and result['label_conditioned'] is result['convnext_used_for_selection'] is False,'Invalid original four metrics')
                counts['new_quality_calls']+=1;event(dict(base,status='COMPLETE',metrics=value));cache[key]=value;return value
            rows_path=bound_output(rendered,'per_frame.csv')
            with rows_path.open(newline='',encoding='utf-8') as f:entropy_rows=list(csv.DictReader(f))
            require(len(entropy_rows)==9000,'Complete actual entropy frames required')
            entropy_rows.sort(key=lambda x:(int(x['source_index']),int(x['snr_db']),int(x['noise_seed']),x['family']))
            rows=[];last_index=None;image_archives={}
            for row in entropy_rows:
                guard();i=int(row['source_index']);require(row['source_id']==ids[i] and row['family'] in FAMILIES,'Actual entropy source differs')
                image_path=row['image_path']
                require(rendered['outputs'].get(image_path)==row['image_file_sha256'],'Image archive not sealed by render completion')
                if i!=last_index:image_archives={};last_index=i
                if image_path not in image_archives:
                    verify(image_path,row['image_file_sha256'])
                    with np.load(image_path,allow_pickle=False) as z:image_archives[image_path]={k:z[k].copy() for k in z.files}
                array=image_archives[image_path][row['image_key']];image=array if array.ndim==3 else array[int(row['image_slot'])]
                require(shared.image_sha(image)==row['image_sha256'],'Actual reconstruction image changed')
                value=score(i,image)
                rows.append(dict(family=row['family'],point_id=row['family']+'_SNR_'+str(int(row['snr_db'])),
                    source_index=i,source_id=ids[i],snr_db=int(row['snr_db']),noise_seed=int(row['noise_seed']),
                    metric_evaluator_identity=backend.identity,**value))
            plan_path=bound_output(diagnostic,'frame_plan.json');plan=read(plan_path);gray=np.full((3,256,256),.5,dtype=np.float32)
            for item in plan:
                i=item['source_index'];require(item['source_id']==ids[i] and item['metric_evaluator_identity']==backend.identity,'Original raw diagnostic metric identity differs')
                value=score(i,gray) if item['gray_requested'] else item['reused_metrics']
                rows.append(dict(family=item['family'],point_id=item['point_id'],source_index=i,source_id=ids[i],
                    snr_db=item['snr_db'],noise_seed=item['noise_seed'],metric_evaluator_identity=backend.identity,**value))
            expected={(f,s,i,n) for f in FAMILIES+RAW_DROP for s in SNRS for i in range(500) for n in SEEDS}
            require(len(rows)==18000 and {(x['family'],x['snr_db'],x['source_index'],x['noise_seed']) for x in rows}==expected,
                'Full entropy plus raw diagnostic grid required')
            backend.check();guard();csv_write(out/'per_frame.csv',rows);outputs[str(out/'per_frame.csv')]=sha(out/'per_frame.csv')
            if ledger.exists():outputs[str(ledger)]=sha(ledger)
            done=dict(status='T1_POSTHOC_COMMON500_FOUR_METRICS_COMPLETE',request_sha256=rh,source_count=500,source_ids=ids,frame_count=18000,
                metric_evaluator_identity=backend.identity,metric_identity=metric_identity,freeze=r['freeze'],
                original_statistics_completion=r['original_statistics_completion'],request_path=str(Path(path).resolve()),counts=counts,outputs=outputs,
                new_packet_decodes=0,new_VAR_renders=0,post_hoc_supplement=True,holdout_used_for_selection=False)
            shared.save(out/'completion.json',done);return done
    except BaseException:
        shared.save(out/'owner_failed.json',dict(counts=counts,traceback=traceback.format_exc(),no_auto_retry=True));raise
    finally:
        shared.save(out/'owner_exited.json',dict(pid=os.getpid(),counts=counts,completed=(out/'completion.json').exists()))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for n in ('render-completion','metric-template','original-scoring-module','adaptive-scoring-module','adaptive-metrics-completion','raw-diagnostic-completion','out'):a.add_argument('--'+n,required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=float,default=14400)
    b=sub.add_parser('run');b.add_argument('--request',required=True)
    args=p.parse_args();value=prepare(args) if args.command=='prepare' else run(args.request);print(value.get('status',value.get('schema')))

if __name__=='__main__':main()
