"""Add metrics to every sealed development row, with source-level resume."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs/UNIFIED-METRICS-20261002'
RESULT=ROOT/'results/unified_metrics_20261002'
PARENT=ROOT/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed: '+path)

def source_bindings():return {str(p):sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md')}

def runtime_flags(torch):
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic=torch.are_deterministic_algorithms_enabled(),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())

def parent_receipts():
    completion=PARENT/'completion.json';status=PARENT/'supervisor_status.json'
    c=read(completion);s=read(status)
    if c.get('status')!='AUTHORIZED_TWO_METHODS_COMPLETE' or s.get('status')!='COMPLETE':
        raise RuntimeError('Original experiments are not complete')
    pub=c.get('publication',{})
    if pub.get('status')!='PUSHED' or pub.get('commit')!=pub.get('remote_commit'):
        raise RuntimeError('Original final result publication not verified')
    return {str(p):sha(p) for p in (completion,status)}

def normalized(row,study,meta):
    """Keep all original fields, adding unambiguous comparison scopes."""
    value=dict(row,**meta)
    # Method-level access stays conditioned even when this frame's header fails.
    conditioned=bool(meta['replay_semantic_side_information'])
    reference=bool(meta['replay_reference_only'])
    oracle=study=='M2_ORACLE'
    dense=study=='M1_RATE'
    paid_oracle=bool(meta.get('replay_paid_oracle_mask',False))
    scope=('historical_development' if study in ('N512','N1024') else
           'method1_development' if study=='M1' else
           'source_only_rate_curve' if dense else
           'oracle_'+str(row['stage']) if oracle else 'actual_link')
    value.update(experiment=study,scope=scope,N='' if oracle or dense else row.get('N',''),
        phy_family=row.get('phy_family','') or ('oracle' if oracle else 'continuous'),
        method=row.get('method') or row.get('control'),projection=row.get('projection',''),
        control=row.get('control',''),output_role='reference' if reference else 'source_only_reference' if dense else 'oracle' if oracle else 'paid_oracle_reference' if paid_oracle else 'label_conditioned' if conditioned else 'main',
        decoder_id=meta['replay_decoder_id'],label_conditioned=conditioned,
        is_main_conclusion=not(conditioned or reference or oracle or paid_oracle or dense),
        dino_model_id='DINOv2_ViT-S/14',lpips_model_id='LPIPS_Alex',
        dinov2_vitl14_model_id='DINOv2_ViT-L/14',
        classification_interpretation='label_conditioned_descriptive_only' if conditioned else 'independent_classifier')
    return value

def write_csv(path,rows):
    if not rows:raise ValueError('Cannot publish empty scoring table')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temp=Path(str(path)+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(temp,path)

def checkpoint_valid(payload,binding,expected_ids):
    if payload.get('binding')!=binding:raise RuntimeError('Checkpoint identity differs')
    actual=[r['replay_row_id'] for r in payload.get('rows',[])]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected_ids):
        raise RuntimeError('Checkpoint row coverage differs')
    if not all(r.get('replay_parity_passed') is True for r in payload['rows']):
        raise RuntimeError('Checkpoint contains unverified replay')
    if payload.get('payload_sha256')!=identity({k:v for k,v in payload.items() if k!='payload_sha256'}):
        raise RuntimeError('Checkpoint checksum differs')
    return payload


class PendingMetricScorer:
    """One source's bounded RGB queue and permanent scalar score cache.

    Rows stay in their original list order. Pending entries retain row objects,
    so all duplicates receive the one measured value when their batch finishes.
    """
    def __init__(self,evaluator,torch,reference,prepared,truth,batch_size,qualified_batch_sizes):
        from batch_speed import qualified_chunks
        qualified_chunks(0,batch_size,qualified_batch_sizes)
        self.evaluator=evaluator;self.torch=torch;self.reference=reference;self.prepared=prepared
        self.truth=truth;self.batch_size=batch_size;self.qualified_batch_sizes=tuple(qualified_batch_sizes)
        self.cache={};self.pending={};self.max_pending_unique_images=0;self.batch_sizes_used=[]

    @staticmethod
    def metric_values(values):
        from batch_speed import FLOAT_FIELDS,EXACT_FIELDS
        import math
        if set(values)!=set(FLOAT_FIELDS+EXACT_FIELDS) or values['label_conditioned'] is not False:
            raise RuntimeError('Metric batch output fields or conditioning differ')
        if any(isinstance(values[k],bool) or not isinstance(values[k],(int,float)) or
               not math.isfinite(values[k]) for k in FLOAT_FIELDS):
            raise RuntimeError('Nonfinite metric batch output')
        # Method-level label access is owned by normalized(), including erased
        # class headers. The score never changes that scientific row annotation.
        return {k:int(v) if isinstance(v,bool) else v for k,v in values.items() if k!='label_conditioned'}

    def add(self,key,rgb,row):
        import numpy as np
        if key in self.cache:
            row.update(self.cache[key]);return
        if key in self.pending:
            self.pending[key]['rows'].append(row);return
        pixels=np.asarray(rgb)
        if pixels.dtype!=np.float32 or pixels.shape!=(3,256,256):
            raise ValueError('Metric batch requires frozen float32 CHW RGB')
        self.pending[key]=dict(rgb=np.array(pixels,copy=True,order='C'),rows=[row])
        self.max_pending_unique_images=max(self.max_pending_unique_images,len(self.pending))
        if len(self.pending)==self.batch_size:self.flush()

    def flush(self):
        import numpy as np
        from batch_speed import qualified_chunks
        for batch in qualified_chunks(len(self.pending),self.batch_size,self.qualified_batch_sizes):
            keys=list(self.pending)[:batch]
            images=np.stack([self.pending[key]['rgb'] for key in keys],axis=0)
            reference,prepared=self.evaluator.expand_reference(self.reference,self.prepared,batch)
            values=self.evaluator.score(reference,self.torch.from_numpy(images),
                self.torch.tensor([self.truth]*batch,dtype=self.torch.int64),
                label_conditioned=False,prepared=prepared)
            if len(values)!=batch:raise RuntimeError('Metric batch returned the wrong row count')
            outputs=[self.metric_values(value) for value in values]
            # Do not cache a partial batch if any row failed validation.
            for key,output in zip(keys,outputs):
                self.cache[key]=output
                for row in self.pending[key]['rows']:row.update(output)
                del self.pending[key]
            self.batch_sizes_used.append(batch)


def registered_contrasts(expected):
    from analysis import contrasts,context
    pairs={(a,b):name for a,b,name in contrasts(expected,{})}
    for a in expected:
        for b in expected:
            if (a[0]=='M2_ACTUAL' and b[0] in ('N512','N1024') and
                a[2]==b[2] and a[4]==b[4] and b[5]=='P'+b[2] and b[9]=='Dc'):
                pairs[a,b]='actual_link_minus_same_budget_P'
            am='P' if a[5]=='P1024' else a[5]
            bm='P' if b[5]=='P512' else b[5]
            if (a[0]=='N1024' and b[0]=='N512' and am==bm and
                a[3:5]==b[3:5] and a[6:]==b[6:]):
                pairs[a,b]='N1024_minus_N512_matched_method'
    return [dict(group_A=context(a),group_B=context(b),name=name) for (a,b),name in sorted(pairs.items())]

def main():
    import torch
    import replay
    from metric_models import MetricEvaluator
    from batch_speed import qualify_and_select_batch
    from analysis import GROUP_FIELDS,group,NEW_METRICS
    OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    start=time.time();own=source_bindings();parent=parent_receipts()
    pub=read(OUT/'source_publication.json')
    if pub.get('status')!='PUSHED' or pub.get('commit')!=pub.get('remote_commit'):
        raise RuntimeError('Metric source publication missing')
    verify(pub['source_bindings'])
    manifest=OUT/'modelmanifest.json';manifestsha=sha(manifest)
    qualification=read(OUT/'models_qualification.json')
    if qualification.get('status')!='REAL_MODEL_WEIGHTS_QUALIFICATION_PASS' or qualification.get('modelmanifest_sha256')!=manifestsha:
        raise RuntimeError('Metric model qualification missing or stale')
    verify(qualification['source_bindings'])
    # Frozen setup establishes the original numerical flags before evaluator load.
    engine=replay.create_engine(ROOT)
    numerical_runtime=runtime_flags(torch)
    evaluator=MetricEvaluator(manifest,device=engine.loaded['device'])
    if runtime_flags(torch)!=numerical_runtime:raise RuntimeError('Metric loader changed the frozen reconstruction runtime')
    if any(v['status']!='READY' for v in evaluator.metadata['metrics'].values()):
        raise RuntimeError('Every registered metric must be ready')
    metricid=evaluator.identity();bound=engine.setup_bound_files()
    batch_qualification=qualify_and_select_batch(evaluator,engine.loaded['device'],manifest,OUT,
        runtime_context=dict(frozen_replay_bindings=bound,numerical_runtime=numerical_runtime))
    if runtime_flags(torch)!=numerical_runtime:raise RuntimeError('Metric batch qualification changed the frozen reconstruction runtime')
    batch_path=OUT/'metric_batch_qualification.json';batch_sha=sha(batch_path)
    if read(batch_path)!=batch_qualification:raise RuntimeError('Metric batch receipt changed')
    batch_size=batch_qualification['chosen_batch_size']
    qualified_batch_sizes=batch_qualification['qualified_batch_sizes']
    tables={str(engine.layouts[s].rows):sha(engine.layouts[s].rows) for s in engine.rows}
    expected={};allids=[]
    for study,rows in engine.rows.items():
        for row in rows:
            value=normalized(row,study,engine.metadata(row,study))
            expected[group(value)]={name:value[name] for name in GROUP_FIELDS}
            allids.append(replay.row_id(study,row))
    if len(allids)!=len(set(allids)):raise RuntimeError('Frozen row identity duplicated')
    registration=dict(schema_version=1,synthetic=False,training_updates=0,policy_selection_updates=0,
        original_pipeline_complete=True,pipeline_completion_bindings=parent,input_tables=tables,
        numerical_runtime=numerical_runtime,
        frozen_policy_bindings=bound,source_bindings=own,source_publication=pub,
        source_ids=[r['image_id'] for r in engine.records],modelmanifest_path=str(manifest),
        modelmanifest_sha256=manifestsha,metric_evaluator_identity=metricid,
        metric_batch_size=batch_size,metric_qualified_batch_sizes=qualified_batch_sizes,
        metric_batch_qualification_path=str(batch_path),metric_batch_qualification_sha256=batch_sha,
        expected_groups=list(expected.values()),expected_frames=len(allids),expected_rows_sha256=identity(allids),
        contrasts=registered_contrasts(expected),
        metric_availability={k:'READY' for k in NEW_METRICS},replay_parity_passed=True,
        KID='DEFERRED_HOLDOUT',FID='NOT_EVALUATED',new_metrics_used_for_selection=False,
        old_metric_values_retained=True,original_replay_inventory=engine.manifest())
    regpath=RESULT/'metrics_registration.json'
    if regpath.exists() and read(regpath)!=registration:raise RuntimeError('Scoring registration changed on resume')
    write(regpath,registration);write(RESULT/'model_metadata.json',evaluator.metadata)
    write(RESULT/'modelmanifest.json',read(manifest));write(RESULT/'model_qualification.json',qualification)
    # Preserve the exact receipt bytes, so its registration hash also verifies
    # the small artifact placed under results for publication.
    import shutil
    shutil.copyfile(batch_path,RESULT/'metric_batch_qualification.json')
    binding=identity(registration);rows_out=[];baselines=[];unique_count=0
    checkpoints=OUT/'source_checkpoints';checkpoints.mkdir(exist_ok=True)
    for index,record in enumerate(engine.records):
        verify(own);verify(parent)
        if runtime_flags(torch)!=numerical_runtime:raise RuntimeError('Frozen reconstruction runtime changed')
        target=engine.target(index);targetsha=replay.rgb_fingerprint(target)
        target_tensor=torch.from_numpy(target).unsqueeze(0)
        truth=int(record['class_index'])
        ids=[replay.row_id(s,r) for s in engine.rows for r in engine.by_source[s].get(index,[])]
        checkpoint=checkpoints/f'{index:03}.json'
        if checkpoint.exists():
            done=checkpoint_valid(read(checkpoint),binding,ids)
        else:
            prepared=evaluator.prepare_reference(target_tensor)
            pred=int(prepared['resnet50'][0])
            baseline=dict(source_id=record['image_id'],source_index=index,preprocessing_id=record['preprocessing_id'],
                reference_sha256=targetsha,true_class_index=truth,resnet50_source_prediction=pred,
                resnet50_source_top1_label=int(pred==truth))
            scorer=PendingMetricScorer(evaluator,torch,target_tensor,prepared,truth,batch_size,qualified_batch_sizes)
            scored=[];parities=[];last_status=0
            for study in engine.rows:
                for row,rgb,parity in engine.iterate_source(study,index):
                    if not parity['replay_parity_passed'] or parity['synthetic']:raise RuntimeError('Actual replay parity required')
                    key=replay.metric_cache_key(rgb,target,metricid)
                    value=normalized(row,study,engine.metadata(row,study))
                    value.update(image_sha256=parity['rgb_sha256'],reference_sha256=targetsha,
                        true_class_index=truth,modelmanifest_sha256=manifestsha,replay_parity_passed=True,
                        replay_metric_deltas=json.dumps(parity['metric_deltas'],sort_keys=True),
                        metric_cache_key=key)
                    scored.append(value);parities.append(parity)
                    scorer.add(key,rgb,value)
                    if time.time()-last_status>30:
                        write(OUT/'scoring_status.json',dict(status='RUNNING',study=study,source_index=index,
                            sources_complete=index,source_frames=len(scored),total_sources=100,
                            frames_complete=len(rows_out),expected_frames=len(allids),elapsed_seconds=time.time()-start))
                        last_status=time.time()
            scorer.flush()
            done=dict(binding=binding,source_index=index,rows=scored,baseline=baseline,
                unique_images=len(scorer.cache),parity=parities,
                metric_batch_sizes_used=scorer.batch_sizes_used,
                max_pending_unique_images=scorer.max_pending_unique_images)
            done['payload_sha256']=identity(done);checkpoint_valid(done,binding,ids);write(checkpoint,done)
        rows_out.extend(done['rows']);baselines.append(done['baseline']);unique_count+=done['unique_images']
    if len(rows_out)!=len(allids) or {r['replay_row_id'] for r in rows_out}!=set(allids):
        raise RuntimeError('Completed replay row coverage differs')
    engine.verify_frozen();verify(own);verify(parent);verify(bound)
    if runtime_flags(torch)!=numerical_runtime:raise RuntimeError('Frozen reconstruction runtime changed')
    if sha(manifest)!=manifestsha:raise RuntimeError('Metric manifest changed')
    if sha(batch_path)!=batch_sha:raise RuntimeError('Metric batch qualification changed')
    write_csv(RESULT/'metrics_per_frame.csv',rows_out);write_csv(RESULT/'source_baseline.csv',baselines)
    write(RESULT/'scoring_inventory.json',dict(frames=len(rows_out),sources=100,unique_image_reference_pairs=unique_count,
        metric_batch_size=batch_size,metric_batch_qualification_sha256=batch_sha,
        expected_frames=len(allids),parity_passed=True,studies={s:len(r) for s,r in engine.rows.items()},
        source_checkpoint_sha256={str(p):sha(p) for p in sorted(checkpoints.glob('*.json'))}))
    outputs={str(p):sha(p) for p in RESULT.iterdir() if p.name in ['metrics_per_frame.csv','source_baseline.csv',
        'scoring_inventory.json','metrics_registration.json','model_metadata.json','modelmanifest.json','model_qualification.json',
        'metric_batch_qualification.json']}
    write(OUT/'scoring_completion.json',dict(status='COMPLETE',synthetic=False,training_updates=0,
        policy_selection_updates=0,parity_passed=True,frames=len(rows_out),inputs={**bound,**parent,str(batch_path):batch_sha},outputs=outputs,
        metric_batch_size=batch_size,metric_qualified_batch_sizes=qualified_batch_sizes,
        metric_batch_qualification_sha256=batch_sha,metric_evaluator_identity=metricid,
        source_bindings=own,modelmanifest_sha256=manifestsha,elapsed_seconds=time.time()-start))
    write(OUT/'scoring_status.json',dict(status='COMPLETE',frames=len(rows_out),sources_complete=100,
        elapsed_seconds=time.time()-start))

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'scoring_status.json',dict(status='FAILED',error=str(error),time=time.time()))
        raise
