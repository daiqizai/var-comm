"""Finalize verified cached rows and replay every uncached row without fake RGB."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import cache_assembly as assembly


def local_sources(cache):
    return {str(p.resolve()):cache.sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.md')}


def publication(root, directory, runtime_sources, cache, canonical_name):
    path=directory/'extension_source_publication.json'
    record=cache.read(path)
    expected={str(root/'experiments'/canonical_name/Path(p).name):digest for p,digest in runtime_sources.items()}
    if (record.get('status')!='PUSHED' or record.get('checks')!='PASS'
            or not re.fullmatch('[0-9a-f]{40}',str(record.get('commit','')))
            or record.get('commit')!=record.get('remote_commit')
            or record.get('runtime_source_bindings')!=runtime_sources or record.get('source_bindings')!=expected):
        raise RuntimeError('Final assembly source publication missing or different')
    cache.verify(runtime_sources);cache.verify(expected)
    return {str(path):cache.sha(path),**runtime_sources,**expected}


def verify_real_qualification(extension, snapshot, cache):
    path=extension/'real_cached_row_qualification.json'
    proof=cache.read(path)
    if (proof.get('status')!='REAL_CACHED_ROW_ASSEMBLY_PASS'
            or proof.get('assembler_sha256')!=cache.sha(Path(assembly.__file__).resolve())
            or any(proof.get(k) is not True for k in ('prior_actual_replay_parity_preserved','scalar_values_exact',
                'original_fields_exact','classification_boundaries_preserved'))
            or any(proof.get(k) is not False for k in ('fresh_reconstruction','fresh_metric_inference','torch_imported'))):
        raise RuntimeError('Real cached-row qualification is missing or stale')
    indexes=proof.get('source_indices')
    if (not isinstance(indexes,list) or not indexes or any(type(i) is not int or i not in snapshot.paths for i in indexes)
            or indexes!=sorted(set(indexes))
            or proof.get('rows')!=sum(snapshot.registration['sources'][i]['frames'] for i in indexes)):
        raise RuntimeError('Real cached-row qualification coverage differs')
    expected={str(snapshot.paths[i]):snapshot.bindings[str(snapshot.paths[i])] for i in indexes}
    if proof.get('checked_checkpoint_bindings')!=expected:
        raise RuntimeError('Real cached-row qualification checkpoint identity differs')
    sources=proof.get('source_bindings',{})
    if (len(sources)!=4 or {Path(p).name for p in sources}!={'cache_assembly.py','cache_handoff.py','replay.py','runner.py'}
            or not any(Path(p).name=='cache_assembly.py' and digest==proof['assembler_sha256'] for p,digest in sources.items())):
        raise RuntimeError('Real cached-row qualification source inventory differs')
    if any(snapshot.registration['source_bindings'].get(p)!=digest for p,digest in sources.items()
           if Path(p).name!='cache_assembly.py'):
        raise RuntimeError('Real cached-row qualification used different scientific source')
    tables=proof.get('original_tables',{})
    if len(tables)!=4 or any(snapshot.registration['frozen_inputs'].get(p)!=digest for p,digest in tables.items()):
        raise RuntimeError('Real cached-row qualification original tables differ')
    for bindings in (sources,tables,expected):cache.verify(bindings)
    return {str(path):cache.sha(path),**sources,**tables,**expected},proof


def run_final(snapshot, runner, replay, cache, extension_bindings):
    ROOT, OUT, RESULT = runner.ROOT, runner.OUT, runner.RESULT
    sha, write, identity, verify = runner.sha, runner.write, runner.identity, runner.verify
    normalized, write_csv = runner.normalized, runner.write_csv
    checkpoint_valid, runtime_flags = runner.checkpoint_valid, runner.runtime_flags
    PendingMetricScorer, registered_contrasts = runner.PendingMetricScorer, runner.registered_contrasts
    parent_receipts = runner.parent_receipts
    def read(path):
        value = runner.read(path)
        return cache.JsonComparableRecord(value) if Path(path) == RESULT/'metrics_registration.json' else value
    def source_bindings():
        result=dict(runner.source_bindings())
        for extra in (cache.source_bindings(),extension_bindings):
            for path,digest in extra.items():
                if path in result and result[path]!=digest:raise RuntimeError('Conflicting scientific source binding')
                result[path]=digest
        return result
    provenance = dict(snapshot=snapshot.provenance(), execution=assembly.execution_protocol(snapshot))
    provenance['execution'].update(
        prior_snapshot_final_replay_requirement='historical R4 plan, superseded only by this verified final assembly',
        historical_rows_require_fresh_final_replay=False, historical_actual_replay_proofs_preserved=True,
        current_original_full_row_hash_join_required=True, uncached_rows_require_fresh_final_replay=True)
    qualification_paths=[Path(p) for p in snapshot.bindings if Path(p).name=='real_cached_row_qualification.json']
    if len(qualification_paths)>1:raise RuntimeError('Ambiguous real cached-row qualification')
    if qualification_paths:
        provenance['real_cached_row_qualification']=dict(path=str(qualification_paths[0]),
            sha256=cache.sha(qualification_paths[0]),receipt=cache.read(qualification_paths[0]))
    provenance_path = RESULT/'cache_assembly_provenance.json'

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
    previous_inventory=snapshot.registration['original_replay_inventory']
    current_inventory=engine.manifest()
    for field in ('model_identity','source_ids','preprocessing_ids','parity_tolerances'):
        if not cache.same_json(previous_inventory[field],current_inventory[field]):
            raise RuntimeError('Final assembly original replay identity differs: '+field)
    cache.verify(snapshot.bindings)
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
        old_metric_values_retained=True,original_replay_inventory=engine.manifest(),
        cache_assembly=provenance, metric_batch_execution=provenance['execution'])
    regpath=RESULT/'metrics_registration.json'
    if regpath.exists() and read(regpath)!=registration:raise RuntimeError('Scoring registration changed on resume')
    cache.seal(provenance_path,provenance)
    write(regpath,registration);write(RESULT/'model_metadata.json',evaluator.metadata)
    write(RESULT/'modelmanifest.json',read(manifest));write(RESULT/'model_qualification.json',qualification)
    # Preserve the exact receipt bytes, so its registration hash also verifies
    # the small artifact placed under results for publication.
    import shutil
    shutil.copyfile(batch_path,RESULT/'metric_batch_qualification.json')
    if qualification_paths:
        shutil.copyfile(qualification_paths[0],RESULT/'real_cached_row_qualification.json')
    binding=identity(registration);rows_out=[];baselines=[];unique_count=0
    total_origins={assembly.HISTORICAL:0,assembly.FRESH:0}
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
            cached=assembly.SourceAssembly(snapshot,cache,replay,runner,engine,index,reference_sha=targetsha,
                evaluator_identity=metricid,numerical_runtime=numerical_runtime,truth=truth,prediction=pred,manifest_sha=manifestsha)
            for key,values in cached.scores.items():
                scorer.cache[key]=scorer.metric_values(dict(values,label_conditioned=False))
            scored=[];parities=[];last_status=0
            for study in engine.rows:
                if cached.available(study):
                    for value,parity in cached.assemble(study):
                        scored.append(value);parities.append(parity)
                    continue
                for row,rgb,parity in engine.iterate_source(study,index):
                    if not parity['replay_parity_passed'] or parity['synthetic']:raise RuntimeError('Actual replay parity required')
                    parity=dict(parity,parity_origin=assembly.FRESH)
                    key=replay.metric_cache_key(rgb,target,metricid)
                    value=normalized(row,study,engine.metadata(row,study))
                    value.update(image_sha256=parity['rgb_sha256'],reference_sha256=targetsha,
                        true_class_index=truth,modelmanifest_sha256=manifestsha,replay_parity_passed=True,
                        replay_metric_deltas=json.dumps(parity['metric_deltas'],sort_keys=True),
                        metric_cache_key=key,parity_origin=assembly.FRESH)
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
                parity_origin_counts=assembly.origin_counts(scored,parities),
                metric_batch_sizes_used=scorer.batch_sizes_used,
                max_pending_unique_images=scorer.max_pending_unique_images)
            done['payload_sha256']=identity(done);checkpoint_valid(done,binding,ids);write(checkpoint,done)
        counts=assembly.origin_counts(done['rows'],done['parity'])
        if done.get('parity_origin_counts')!=counts:raise RuntimeError('Final checkpoint assembly counts differ')
        for origin,count in counts.items():total_origins[origin]+=count
        rows_out.extend(done['rows']);baselines.append(done['baseline']);unique_count+=done['unique_images']
    if len(rows_out)!=len(allids) or {r['replay_row_id'] for r in rows_out}!=set(allids):
        raise RuntimeError('Completed replay row coverage differs')
    engine.verify_frozen();verify(own);verify(parent);verify(bound);cache.verify(snapshot.bindings)
    if runtime_flags(torch)!=numerical_runtime:raise RuntimeError('Frozen reconstruction runtime changed')
    if sha(manifest)!=manifestsha:raise RuntimeError('Metric manifest changed')
    if sha(batch_path)!=batch_sha:raise RuntimeError('Metric batch qualification changed')
    write_csv(RESULT/'metrics_per_frame.csv',rows_out);write_csv(RESULT/'source_baseline.csv',baselines)
    write(RESULT/'scoring_inventory.json',dict(frames=len(rows_out),sources=100,unique_image_reference_pairs=unique_count,
        metric_batch_size=batch_size,metric_batch_qualification_sha256=batch_sha,
        expected_frames=len(allids),parity_passed=True,studies={s:len(r) for s,r in engine.rows.items()},
        parity_origin_counts=total_origins,cache_assembly=provenance,
        source_checkpoint_sha256={str(p):sha(p) for p in sorted(checkpoints.glob('*.json'))}))
    outputs={str(p):sha(p) for p in RESULT.iterdir() if p.name in ['metrics_per_frame.csv','source_baseline.csv',
        'scoring_inventory.json','metrics_registration.json','model_metadata.json','modelmanifest.json','model_qualification.json',
        'metric_batch_qualification.json','cache_assembly_provenance.json','real_cached_row_qualification.json']}
    write(OUT/'scoring_completion.json',dict(status='COMPLETE',synthetic=False,training_updates=0,
        policy_selection_updates=0,parity_passed=True,frames=len(rows_out),inputs={**bound,**parent,**snapshot.bindings,str(batch_path):batch_sha,str(provenance_path):sha(provenance_path)},outputs=outputs,
        metric_batch_size=batch_size,metric_qualified_batch_sizes=qualified_batch_sizes,
        metric_batch_qualification_sha256=batch_sha,metric_evaluator_identity=metricid,
        source_bindings=own,modelmanifest_sha256=manifestsha,cache_assembly=provenance,
        parity_origin_counts=total_origins,elapsed_seconds=time.time()-start))
    write(OUT/'scoring_status.json',dict(status='COMPLETE',frames=len(rows_out),sources_complete=100,
        elapsed_seconds=time.time()-start))


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--cache-dir',required=True)
    args=parser.parse_args()
    root,directory=Path(args.root).resolve(),Path(args.cache_dir).resolve()
    extension=root/'outputs/METRIC-FINAL-CACHE-20261002'
    if directory!=root/'outputs/METRIC-CONCURRENT-R4-20261002' or Path(__file__).resolve().parent!=extension/'runtime':
        raise RuntimeError('Unexpected final assembly runtime/cache directory')
    locks=[]
    for folder in (extension,directory):
        lock=(folder/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    runtime=directory/'runtime'
    sys.path.insert(0,str(runtime))
    import cache_handoff as cache
    import concurrent_runner
    for module in (cache,concurrent_runner):
        if Path(module.__file__).resolve().parent!=runtime:raise RuntimeError('Unexpected immutable cache module origin')
    snapshot=cache.CacheSnapshot(directory)
    status=cache.read(directory/'status.json')
    if len(snapshot.paths)!=100 or status.get('status')!='SEALED_STUDIES_COMPLETE':
        raise RuntimeError('All100 concurrent sources must be sealed before final assembly')
    snapshot.bindings[str(directory/'status.json')]=cache.sha(directory/'status.json')
    own=local_sources(cache)
    snapshot.bindings.update(publication(root,directory,cache.source_bindings(),cache,'metric-concurrent-20261002'))
    snapshot.bindings.update(publication(root,extension,own,cache,'metric-final-cache-20261002'))
    qualification_bindings,_=verify_real_qualification(extension,snapshot,cache)
    snapshot.bindings.update(qualification_bindings)
    runner,replay,_,_=concurrent_runner.imports(root)
    if not cache.same_json(snapshot.registration.get('replay_compatibility'),replay._historical_latent_alias_receipt):
        raise RuntimeError('Final assembly compatibility differs')
    run_final(snapshot,runner,replay,cache,own)


if __name__=='__main__':
    main()
