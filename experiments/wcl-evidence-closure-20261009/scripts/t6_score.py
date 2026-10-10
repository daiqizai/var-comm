"""Frozen N2048 confirmation100 scoring; no PHY, source encoder or VAR render.

prepare reads metadata; run invokes the existing frozen SuiteBackend. Every new
reference comes from this confirmation manifest, never the old common500 cache.
close authenticates run_recorded_child.py's actual wait/exit evidence.
"""
import argparse, contextlib, csv, hashlib, importlib.util, json, math, os
from pathlib import Path
import signal, sys, time, traceback
import numpy as np

COUNT=100
N=2048
SNRS=[4,10,19]
SEEDS=[9201,9202,9203]
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']
RAW_WHOLE='N2048_RAW_WHOLE_VAR_COMPLETION'
RAW_PARTIAL='N2048_RAW_PARTIAL_VAR_COMPLETION'
OBSERVER_SHA='dfeba4b30aa7cc3febe67253d584c18c6e1ede11f41dbbe84f6a7fdfdd829d89'
STOP=False

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def verify(path,expected):require(sha(path)==expected,'Changed bound input: '+str(path))
def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def pinned(value):verify(value['path'],value['sha256']);return read(value['path'])
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def csv_read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def bound_output(done,name):
    found=[(p,h) for p,h in done['outputs'].items() if Path(p).name==name]
    require(len(found)==1,'One sealed output required: '+name);p,h=found[0];verify(p,h);return Path(p)
def image_sha(image):
    a=np.ascontiguousarray(image)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def image_check(image):
    require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all()
        and image.min()>=0 and image.max()<=1,'Actual finite float32 CHW RGB256 required')
def methods(family):
    require(family in ('EC_STATIC_WHOLE','EC_VAR_WHOLE'),'One preselected entropy family required')
    return [RAW_WHOLE,RAW_PARTIAL,'N2048_'+family+'_VAR_COMPLETION']
def comparisons(family):
    whole,partial,entropy=methods(family)
    return [[partial,whole],[entropy,whole],[partial,entropy]]
def point(method,snr):return method+'_SNR_'+str(snr)
def load_module(descriptor,name):
    verify(descriptor['path'],descriptor['sha256'])
    spec=importlib.util.spec_from_file_location(name,descriptor['path'])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def stop(*_):
    global STOP;STOP=True
@contextlib.contextmanager
def lock(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def metadata(rendered):
    """Authenticate the independently fixed source, policy and completed images."""
    require(rendered['status']=='T6_CONFIRMATION100_RENDER_COMPLETE' and rendered['source_count']==COUNT
        and rendered['noise_count']==3 and rendered['frame_count']==2700 and rendered['N']==N
        and rendered['noise_seeds']==SEEDS and rendered['actual_children_waited'] is True
        and rendered['worker_exit_codes'] and all(x==0 for x in rendered['worker_exit_codes']),
        'Closed actual N2048 confirmation render with real child exits required')
    source=pinned(rendered['source_manifest']);frozen=pinned(rendered['calibration_freeze'])
    selected=pinned(rendered['entropy_family_selection'])
    require(source['schema']=='T6_CONFIRMATION100_SOURCE_MANIFEST_V1' and source['source_count']==COUNT
        and len(source['records'])==COUNT and source['source_ids']==rendered['source_ids']
        and len(set(source['source_ids']))==COUNT,'Exact new confirmation100 source order required')
    registration=pinned(source['confirmation_registration'])
    require(registration['schema']=='WCL_N2048_CONFIRMATION100_METADATA_V1'
        and registration['status']=='SOURCE_IDS_FIXED_BEFORE_ANY_NEW_SOURCE_IMAGE_OR_SCORE_ACCESS'
        and registration['source_count']==COUNT
        and [r['source_id'] for r in registration['records']]==source['source_ids'],
        'Sources must match prior metadata-only selection, never reselected by score')
    duplicate=pinned(source['content_duplicate_check_completion'])
    require(duplicate['status']=='T6_CONFIRMATION100_CONTENT_DUPLICATES_CHECKED_PASS'
        and duplicate['source_count']==COUNT and duplicate['duplicate_source_id_count']==0
        and duplicate['duplicate_preprocessing_count']==0
        and duplicate['confirmation_registration']==source['confirmation_registration']
        and duplicate['scope']=='all_available_audited_source_content_hashes'
        and type(duplicate['available_reference_hash_count']) is int
        and type(duplicate['unavailable_reference_hash_count']) is int,
        'New-source content duplicate audit, including its coverage limitation, required')
    require(frozen['status']=='T6_N2048_POLICIES_FROZEN_CALIBRATION_ONLY_V1'
        and frozen['N']==N and frozen['source_count']==1000 and frozen['noise_count']==3
        and frozen['selection_used_confirmation'] is False and frozen['holdout_used_for_selection'] is False
        and frozen['entropy_family_selection']==rendered['entropy_family_selection'],
        'Full-calibration frozen policies required before confirmation scoring')
    require(selected['holdout_used_for_selection'] is False and selected['holdout_files_read'] is False,
        'Entropy family must be selected on original full calibration only')
    family=selected['family'];methods(family)
    require(set(frozen['policies'])=={'RAW_WHOLE','RAW_PARTIAL','ENTROPY_WHOLE'}
        and all(set(map(int,v))==set(SNRS) for v in frozen['policies'].values()),'All three frozen SNRs and methods required')
    for i,record in enumerate(source['records']):
        require(record['source_index']==i and record['source_id']==source['source_ids'][i]
            and type(record['evaluation_class_index']) is int and 0<=record['evaluation_class_index']<1000,
            'Contiguous new-source records and actual class index required')
        verify(record['checkpoint'],record['checkpoint_sha256'])
        cp=read(record['checkpoint'])
        require(cp['source_id']==record['source_id'] and cp['preprocessing_id']==record['preprocessing_id']
            and cp['outputs'].get(record['archive'])==record['archive_sha256'],
            'Source checkpoint must authenticate exact pixels and archive')
    return source,frozen,selected,duplicate

def frame_grid(rows,ids,family,scored=False):
    expected={(m,s,i,n) for m in methods(family) for s in SNRS for i in range(COUNT) for n in SEEDS}
    keyed={}
    for row in rows:
        key=row['method_id'],int(row['snr_db']),int(row['source_index']),int(row['noise_seed'])
        require(key in expected and key not in keyed and row['source_id']==ids[key[2]]
            and row['point_id']==point(key[0],key[1]),'Duplicate, missing or misidentified N2048 frame')
        require(int(row['N'])==N,'No N1024 or other-budget frame may enter T6')
        require(row.get('status') not in (None,''),'Actual receive/failure state must remain in every row')
        if scored:
            for metric in METRICS:require(math.isfinite(metric_value(row,metric)),'Every failed frame needs finite scored output')
            require(metric_value(row,METRICS[-1]) in (0.,1.),'ConvNeXt agreement is a binary source-prediction event')
        keyed[key]=row
    require(len(rows)==2700 and set(keyed)==expected,'All2700 fixed method/source/SNR/noise frames required, including failures')
    return keyed

def metric_value(row,metric):
    x=row[metric]
    if metric==METRICS[-1] and x in ('True','False'):return float(x=='True')
    return float(x)

def prepare(args):
    rendered=read(args.render_completion);source,freeze,selected,dup=metadata(rendered)
    template=read(args.metric_template);original=pinned(template['original_FINAL_FREEZE'])
    verify(args.original_scoring_module,template['worker_sha256'])
    require(set(source['source_ids']).isdisjoint(original['population']['source_ids']),
        'New confirmation sources must be disjoint from the original500 identities')
    original_done=read(Path(template['out'])/'completion.json')
    require(original_done['status']=='BPG_CACHED_FOUR_METRICS_AND_NEW_SOURCE_STATISTICS_COMPLETE_V1'
        and original_done['request_sha256']==sha(args.metric_template)
        and original_done['metric_evaluator_identity']==original['holdout_metrics']['metadata_identity'],
        'Previously completed original frozen SuiteBackend template required')
    rows_path=bound_output(rendered,'per_frame.csv');rows=csv_read(rows_path)
    frame_grid(rows,source['source_ids'],selected['family'])
    for row in rows:
        require(rendered['outputs'].get(row['image_path'])==row['image_file_sha256'],
            'Every actual reconstruction must be sealed in the final render closure')
    root=Path(template['root']).resolve();out=Path(args.out).resolve()
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh isolated WCL score output required')
    request=dict(schema='T6_N2048_CONFIRMATION100_SCORE_REQUEST_V1',root=str(root),out=str(out),N=N,
        source_count=COUNT,noise_count=3,source_ids=source['source_ids'],snrs=SNRS,noise_seeds=SEEDS,frame_count=2700,
        render_completion=pin(args.render_completion),source_manifest=rendered['source_manifest'],
        calibration_freeze=rendered['calibration_freeze'],entropy_family_selection=rendered['entropy_family_selection'],
        confirmation_registration=source['confirmation_registration'],content_duplicate_check_completion=source['content_duplicate_check_completion'],
        metric_template=pin(args.metric_template),original_scoring_module=pin(args.original_scoring_module),
        original_metric_completion=pin(Path(template['out'])/'completion.json'),
        evaluator_identity=original['holdout_metrics']['metadata_identity'],family=selected['family'],methods=methods(selected['family']),
        comparisons=comparisons(selected['family']),metrics=METRICS,source_reference_prepare_cap=100,new_unique_quality_cap=2700,
        reference_policy='Only new confirmation100 uint8 pixels / float32(255); no old reference feature cache',
        quality_reuse='Only within this run, same source and exact float RGB SHA plus frozen evaluator',
        mismatch_permutation=[(i+1)%COUNT for i in range(COUNT)],mismatch_diagnostics_reported=False,
        selection_used_confirmation=False,holdout_used_for_selection=False,policy_selection=False,
        content_duplicate_audit_scope=dup['scope'],available_reference_hash_count=dup['available_reference_hash_count'],
        unavailable_reference_hash_count=dup['unavailable_reference_hash_count'],
        deadline_unix=args.deadline_unix,max_seconds=args.max_seconds,
        stop_files=[str(root/'STOP'),str(out/'STOP')],worker_sha256=sha(__file__),
        source_bindings={str(Path(__file__).resolve()):sha(__file__),
            str(Path(__file__).with_name('t2_ledger.py').resolve()):sha(Path(__file__).with_name('t2_ledger.py'))})
    out.mkdir(parents=True);write(out/'request.json',request);return request

def run(path):
    path=Path(path).resolve();r=read(path);rh=sha(path)
    require(r['schema']=='T6_N2048_CONFIRMATION100_SCORE_REQUEST_V1' and sha(__file__)==r['worker_sha256'],'Frozen scoring request/worker required')
    for p,h in r['source_bindings'].items():verify(p,h)
    template=pinned(r['metric_template'])
    require(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        and os.path.abspath(sys.executable)==template['python'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
        'Actual original UM metric environment required')
    require(sorted(os.sched_getaffinity(0))==[4,5,6,7,8,9] and os.getpriority(os.PRIO_PROCESS,0)==15,
        'Original metric affinity and priority required')
    out=Path(r['out']);require(not (out/'owner_started.json').exists(),'Never repeat an existing scoring attempt')
    start=time.monotonic();refs_ledger=quality_ledger=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def guard():
        require(not STOP and time.time()<r['deadline_unix'] and time.monotonic()-start<r['max_seconds']
            and not any(Path(p).exists() for p in r['stop_files']),'Stopped/deadline; no automatic retry')
    counts=dict(new_reference_preparations=0,new_quality_calls=0,within_run_exact_image_reuses=0,
        old_reference_cache_reads=0,old_quality_cache_reads=0)
    try:
        with lock(template['visual_lock']),lock(out/'gpu.lock'):
            guard();write(out/'owner_started.json',dict(pid=os.getpid(),request_sha256=rh,request_path=str(path),started_unix=time.time()))
            from t2_ledger import Ledger
            refs_ledger=Ledger(out/'reference_calls.sqlite',rh,100);quality_ledger=Ledger(out/'quality_calls.sqlite',rh,2700)
            rendered=pinned(r['render_completion']);source,freeze,selected,dup=metadata(rendered)
            require(rendered['source_manifest']==r['source_manifest'] and rendered['calibration_freeze']==r['calibration_freeze']
                and rendered['entropy_family_selection']==r['entropy_family_selection'] and source['source_ids']==r['source_ids']
                and selected['family']==r['family'],'Prepared source/policy bindings changed')
            original=pinned(template['original_FINAL_FREEZE']);old=load_module(r['original_scoring_module'],'_t6_original_frozen_suite_factory')
            backend=old.build_backend(template,original)
            require(backend.identity==r['evaluator_identity'],'Frozen original SuiteBackend numerical/model identity differs')
            references=[];outputs={};source_evidence=[]
            for i,record in enumerate(source['records']):
                guard();verify(record['archive'],record['archive_sha256'])
                with np.load(record['archive'],allow_pickle=False) as z:pixels=z['pixels'].copy()
                require(pixels.dtype==np.uint8 and pixels.shape==(3,256,256)
                    and hashlib.sha256(pixels.tobytes()).hexdigest()==record['preprocessing_id'],'New source exact pixel hash differs')
                target=pixels.astype(np.float32)/np.float32(255);image_check(target);holder={}
                def prepare_reference():
                    holder['prepared']=backend.prepare(target)
                    return dict(source_id=record['source_id'],reference_rgb_sha256=image_sha(target),evaluator_identity=backend.identity)
                evidence=refs_ledger.call(dict(event_id='reference:'+str(i)),
                    dict(source_id=record['source_id'],archive_sha256=record['archive_sha256'],preprocessing_id=record['preprocessing_id'],
                        evaluator_identity=backend.identity),prepare_reference)
                require('prepared' in holder,'No existing reference call may silently restore from an old source cache')
                counts['new_reference_preparations']+=1
                references.append(dict(target=target,prepared=holder['prepared'],record=record,reference_sha256=image_sha(target)))
                source_evidence.append(dict(source_index=i,**evidence,source_archive_sha256=record['archive_sha256'],preprocessing_id=record['preprocessing_id']))
            require(counts['new_reference_preparations']==100,'Every new confirmation reference must be actually prepared')
            actual_rows=csv_read(bound_output(rendered,'per_frame.csv'));frame_grid(actual_rows,r['source_ids'],r['family'])
            actual_rows.sort(key=lambda x:(int(x['source_index']),int(x['snr_db']),int(x['noise_seed']),x['method_id']))
            scored=[];cache={};archives={};last=None
            for row in actual_rows:
                guard();i=int(row['source_index']);reference=references[i]
                if i!=last:archives={};last=i
                p=row['image_path'];require(rendered['outputs'].get(p)==row['image_file_sha256'],'Unsealed actual image')
                if p not in archives:
                    verify(p,row['image_file_sha256'])
                    with np.load(p,allow_pickle=False) as z:archives[p]={k:z[k].copy() for k in z.files}
                arrays=archives[p];a=arrays[row['image_key']];image=a if a.ndim==3 else a[int(row['image_slot'])]
                image_check(image);require(image_sha(image)==row['image_sha256'],'Actual received reconstruction changed')
                if 'source_rgb' in arrays:require(np.array_equal(arrays['source_rgb'],reference['target']),'Reconstruction archive source differs from new confirmation pixels')
                key=digest(dict(source_id=r['source_ids'][i],reference_sha256=reference['reference_sha256'],
                    image_sha256=row['image_sha256'],evaluator_identity=backend.identity))
                if key in cache:values=cache[key];counts['within_run_exact_image_reuses']+=1
                else:
                    def score_image():
                        result=backend.score(reference['target'],image,reference['record']['evaluation_class_index'],reference['prepared'],
                            references[r['mismatch_permutation'][i]]['prepared']['dino_feature'])
                        mse=float(np.square(image.astype(np.float64)-reference['target'].astype(np.float64)).mean())
                        require(mse>0 and math.isfinite(mse),'Infinite PSNR requires explicit statistics handling, never silently drop it')
                        values=dict(psnr_db=float(-10*math.log10(mse)),lpips_alex=float(result['lpips_alex']),
                            dinov2_vitl14_cosine=float(result['dinov2_vitl14_cosine']),
                            convnext_top1_source_prediction=result['convnext_top1_source_prediction'])
                        require(all(math.isfinite(float(values[m])) for m in METRICS)
                            and type(values[METRICS[-1]]) is bool and result['label_conditioned'] is False
                            and result['convnext_used_for_selection'] is False,'Invalid frozen four-metric output')
                        return values
                    values=quality_ledger.call(dict(event_id=key),dict(source_index=i,source_id=r['source_ids'][i],
                        image_sha256=row['image_sha256'],reference_sha256=reference['reference_sha256'],evaluator_identity=backend.identity),score_image)
                    cache[key]=values;counts['new_quality_calls']+=1
                item=dict(row);item.update(values);item.update(source_count=100,noise_count=3,N=N,
                    metric_evaluator_identity=backend.identity,reference_sha256=reference['reference_sha256'],
                    reference_preprocessing_id=reference['record']['preprocessing_id'],quality_cache_key=key,
                    selection_used_confirmation=False)
                scored.append(item)
            frame_grid(scored,r['source_ids'],r['family'],scored=True);backend.check();guard()
            budgets=dict(references=refs_ledger.snapshot(),quality=quality_ledger.snapshot())
            require(all(x['unresolved']==0 for x in budgets.values()),'Every actual metric reservation must be resolved')
            refs_ledger.close();refs_ledger=None;quality_ledger.close();quality_ledger=None
            csv_write(out/'per_frame.csv',scored);write(out/'reference_evidence.json',source_evidence)
            for name in ('per_frame.csv','reference_evidence.json','reference_calls.sqlite','quality_calls.sqlite'):
                outputs[str(out/name)]=sha(out/name)
            done=dict(status='T6_CONFIRMATION100_FOUR_METRICS_WORKER_COMPLETE',request_sha256=rh,request_path=str(path),
                source_count=100,source_ids=r['source_ids'],noise_count=3,noise_seeds=SEEDS,snrs=SNRS,N=N,frame_count=2700,
                family=r['family'],methods=r['methods'],comparisons=r['comparisons'],metric_evaluator_identity=backend.identity,
                metric_metadata=backend.metadata,metrics=METRICS,source_manifest=r['source_manifest'],
                calibration_freeze=r['calibration_freeze'],entropy_family_selection=r['entropy_family_selection'],
                confirmation_registration=r['confirmation_registration'],content_duplicate_check_completion=r['content_duplicate_check_completion'],
                counts=counts,budgets=budgets,outputs=outputs,selection_used_confirmation=False,holdout_used_for_selection=False,
                content_duplicate_audit_scope=r['content_duplicate_audit_scope'],available_reference_hash_count=r['available_reference_hash_count'],
                unavailable_reference_hash_count=r['unavailable_reference_hash_count'],new_packet_decodes=0,new_VAR_renders=0,
                new_source_encodes=0,new_bootstrap_calls=0,elapsed_seconds=time.monotonic()-start)
            write(out/'worker_complete.json',done);return done
    except BaseException:
        write(out/'owner_failed.json',dict(pid=os.getpid(),counts=counts,traceback=traceback.format_exc(),no_auto_retry=True));raise
    finally:
        if refs_ledger is not None:refs_ledger.close()
        if quality_ledger is not None:quality_ledger.close()
        write(out/'owner_exited.json',dict(pid=os.getpid(),request_sha256=rh,counts=counts,worker_complete=(out/'worker_complete.json').exists()))

def actual_wait(wait_record,request,started):
    launch_path=Path(wait_record)/'launch.json';exit_path=Path(wait_record)/'exit.json'
    launch=read(launch_path);exited=read(exit_path);argv=launch['argv']
    require(launch['owner_script_sha256']==OBSERVER_SHA and launch['child_pid']==started['pid']
        and exited['actual_child_waited'] is True and exited['exit_code']==0,
        'Pinned actual parent must have waited for this precise worker and observed exit0')
    rp=str(Path(request).resolve());require(argv[-3:]==['run','--request',rp]
        and str(Path(__file__).resolve()) in argv,'Actual child argv must bind exact score script and request')
    return pin(launch_path),pin(exit_path)

def close(args):
    r=read(args.request);rh=sha(args.request);out=Path(r['out'])
    require(sha(__file__)==r['worker_sha256'] and not (out/'completion.json').exists(),'One immutable final metric closure only')
    started=read(out/'owner_started.json');done=read(out/'worker_complete.json');exited=read(out/'owner_exited.json')
    require(started['request_sha256']==done['request_sha256']==exited['request_sha256']==rh
        and exited['pid']==started['pid'] and exited['worker_complete']
        and done['status']=='T6_CONFIRMATION100_FOUR_METRICS_WORKER_COMPLETE','Actual complete metric worker required')
    launch,observed=actual_wait(args.wait_record,args.request,started)
    for p,h in done['outputs'].items():verify(p,h)
    frame_grid(csv_read(bound_output(done,'per_frame.csv')),done['source_ids'],done['family'],scored=True)
    result=dict(done,status='T6_CONFIRMATION100_FOUR_METRICS_COMPLETE',actual_children_waited=True,
        worker_exit_codes=[0],parent_launch=launch,parent_exit=observed,worker_completion=pin(out/'worker_complete.json'))
    write(out/'completion.json',result);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('render-completion','metric-template','original-scoring-module','out'):a.add_argument('--'+name,required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=float,default=14400)
    a=sub.add_parser('run');a.add_argument('--request',required=True)
    a=sub.add_parser('close');a.add_argument('--request',required=True);a.add_argument('--wait-record',required=True)
    args=parser.parse_args();value=prepare(args) if args.command=='prepare' else run(args.request) if args.command=='run' else close(args)
    print(value.get('status',value.get('schema')))
if __name__=='__main__':main()
