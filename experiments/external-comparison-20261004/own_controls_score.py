"""Score 5,400 registered own reconstructions on the external common target.

Run in the frozen unified-metrics environment, with an exclusively held GPU:
  python own_controls_score.py --root ROOT --launch-id TOKEN
No training, policy updates, image replay or holdout access occurs here.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
import own_controls_score_common as own
from own_controls_score_common import load_source_images
from external_eval_common import (sha, read, write, seal, identity, verify, pixels,
                                  rgb_sha, write_csv, PauseRequested, gpu_available)

HERE = Path(__file__).resolve().parent
EXTRA_METRICS = ('semantic_error','confidently_wrong')


def code_bindings():
    names = ('own_controls_score.py','own_controls_score_common.py','own_controls_score_tests.py',
             'own_controls_common.py','external_eval_common.py','step0_reference_metrics.py',
             'step0_reference_prepare.py','step0_cache_export.py')
    return {str(HERE/name):sha(HERE/name) for name in names}


def classification(row, probability, prediction, source_prediction):
    own.require(np.isfinite(probability) and 0 <= probability <= 1
        and type(prediction) is int and 0 <= prediction < 1000
        and prediction == row['resnet50_prediction'], 'Confidence forward changed classifier identity')
    row.update(resnet50_top1_probability=probability,semantic_error=int(prediction != source_prediction),
        confidently_wrong=int(probability >= .5 and prediction != source_prediction))


def expected_groups(analysis, example_rows):
    groups = {analysis.group(row):analysis.context(analysis.group(row)) for row in example_rows}
    own.require(len(groups) == 18, 'Expected exactly three own methods at two budgets and three SNRs')
    contrasts = []
    for n,methods in own.METHODS.items():
        for snr in own.SNRS:
            keys = {key[5]:key for key in groups if int(key[2]) == n and int(key[4]) == snr}
            own.require(set(keys) == set(methods), 'Registered own group differs')
            for a,b,name in ((methods[1],methods[0],'D_U_minus_same_budget_P'),
                             (methods[2],methods[0],'M1_minus_same_budget_P'),
                             (methods[2],methods[1],'M1_minus_whole_D_U')):
                contrasts.append(dict(name=name,group_A=analysis.context(keys[a]),group_B=analysis.context(keys[b])))
    return groups,contrasts


def summarize(analysis, allrows, baselines, registration):
    """Original 10,000 source-paired bootstrap, with no image-level resampling."""
    own.require(len(allrows) == own.ROWS and [r['replay_row_id'] for r in allrows] ==
                [rid for i in range(100) for rid in own.expected_ids(i)], 'Incomplete own metric population')
    groups = defaultdict(list)
    for row in allrows: groups[analysis.group(row)].append(row)
    wanted = {analysis.group(x) for x in registration['expected_groups']}
    own.require(set(groups) == wanted and len(groups) == 18, 'Own method/SNR inventory differs')
    sources,identities = analysis.original_sources(baselines)
    own.require(sources == registration['source_ids'], 'Metric source order differs')
    for key,frames in groups.items():
        own.require(len(frames) == 300 and {(r['source_id'],r['noise_seed']) for r in frames}
            == {(s,seed) for s in sources for seed in own.SEEDS}, 'Own group is not complete source/noise grid')
        for row in frames:
            base = identities[row['source_id']]
            own.require(row['reference_sha256'] == base['reference_sha256']
                and row['preprocessing_id'] == base['preprocessing_id']
                and row['resnet50_source_prediction'] == base['prediction']
                and row['resnet50_top1_label'] == int(row['resnet50_prediction'] == base['truth'])
                and row['resnet50_top1_source_prediction'] == int(row['resnet50_prediction'] == base['prediction']),
                'Classifier/reference comparison identity differs')
    availability = set(analysis.NEW_METRICS); flags = {key:True for key in groups}
    bootstrap = analysis.Bootstrap()
    summary,per_source,means = analysis.summarize(groups,sources,availability,flags,bootstrap)
    paired = analysis.paired(groups,sources,registration,means,flags,bootstrap)
    for row in paired:
        row['comparison_scope'] = 'same_source_nominal_noise_seed_and_common_float_reference_distinct_physical_noise_namespaces'
    for name in EXTRA_METRICS:
        for key,frames in groups.items():
            values,observations = analysis.source_values(frames,sources,name,availability)
            means[key,name] = values
            meta = dict(**analysis.context(key),group_id=analysis.group_id(key),is_main_conclusion=True)
            summary.append(dict(**meta,metric=name,direction='lower',**bootstrap.interval(values),
                n_frames=300,n_expected_frames=300,status='EVALUATED',
                aggregation='mean_noise_within_source_then_equal_source_mean'))
            per_source.extend(dict(**meta,source_id=sid,source_index=i,metric=name,
                value=float(values[i]),n_noise=len(observations[sid])) for i,sid in enumerate(sources))
        for a,b,contrast in analysis.contrasts(groups,registration):
            exemplar = next(row for row in paired if row['group_A'] == analysis.group_id(a)
                            and row['group_B'] == analysis.group_id(b))
            paired.append(dict(exemplar,metric=name,direction='lower',**bootstrap.interval(means[a,name]-means[b,name])))
    return summary,per_source,paired


def score(root, progress, stopped):
    root = Path(root).resolve(); paths = own.c.locations(root); out,result = paths['out'],paths['result']
    result.mkdir(parents=True,exist_ok=True); metric_dir = out/'metrics'; metric_dir.mkdir(parents=True,exist_ok=True)
    sources = code_bindings(); complete_path = out/'score_completion.json'
    if complete_path.exists():
        done = read(complete_path)
        own.require(done.get('status') == 'OWN_CONTROLS_METRICS_COMPLETE' and done.get('sources') == 100
            and done.get('rows') == own.ROWS and done.get('source_bindings') == sources,
            'Completed own metric scope/runtime changed')
        verify(done['bindings']); verify(done['outputs']); progress('COMPLETE',sources_complete=100,rows=own.ROWS); return done
    if stopped(): raise PauseRequested('Stop requested before input admission')
    progress('VERIFYING_COMPLETED_FLOAT_CACHES')
    admission = own.admit_inputs(root)
    # Freeze before loading any model; immutable on restart.
    admission_path = out/'score_input_admission.json'; seal(admission_path,admission)
    if stopped(): raise PauseRequested('Stop requested before metric model loading')
    gpu_available(); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from step0_reference_metrics import load_suite,numeric_flags
    evaluator,metric_module,native,lpips,dino,model_bindings,flags = load_suite(root,'cuda:0')
    shared = root/'experiments/unified-metrics-20261002'; sys.path.insert(0,str(shared))
    import runner
    import replay
    import analysis
    from batch_speed import qualify_and_select_batch,qualified_chunks
    original_path = root/'results/unified_metrics_20261002/metrics_registration.json'
    original = read(original_path)
    own.require(evaluator.identity() == original['metric_evaluator_identity'], 'Frozen metric evaluator differs')
    manifest = root/'outputs/UNIFIED-METRICS-20261002/modelmanifest.json'; manifest_sha = sha(manifest)
    batch = qualify_and_select_batch(evaluator,torch.device('cuda:0'),manifest,metric_dir,
        runtime_context=dict(study=own.VERSION,numerical_runtime=flags,
            score_input_admission_sha256=sha(admission_path),model_bindings=model_bindings))
    own.require(numeric_flags(torch) == flags,'Metric batch qualification changed numerical flags')
    _,example,_,_ = own.load_source_payload(root,0,admission)
    expected,contrasts = expected_groups(analysis,example)
    registration = dict(**admission,schema_version=1,status='OWN_CONTROLS_METRICS_REGISTERED',synthetic=False,
        source_ids=[r['image_id'] for r in admission['source_identity']],source_count=100,expected_frames=own.ROWS,
        own_methods_by_N={str(n):list(ms) for n,ms in own.METHODS.items()},
        expected_groups=list(expected.values()),contrasts=contrasts,source_bindings=sources,
        score_input_admission_sha256=sha(admission_path),training_updates=0,policy_selection_updates=0,
        new_metrics_used_for_selection=False,selection_uses_development=False,holdout_access=False,
        modelmanifest_path=str(manifest),modelmanifest_sha256=manifest_sha,
        metric_evaluator_identity=evaluator.identity(),model_bindings=model_bindings,numerical_runtime=flags,
        metric_batch_size=batch['chosen_batch_size'],metric_qualified_batch_sizes=batch['qualified_batch_sizes'],
        metric_batch_qualification_sha256=sha(metric_dir/'metric_batch_qualification.json'),
        metric_availability={k:'READY' for k in analysis.NEW_METRICS},replay_parity_passed=True,
        reference_targets='unchanged_admitted_FINAL_P2048_P3060_float_RGB',
        native_metric_policy='original_native_scores_preserved_separately_no_metric_offset',
        source_dino_feature_batch=1,native_quality_batch='one_source_all_unique_reconstructions',
        mismatch_derangement_seed=20260930,classifier_confidence_threshold=.5,
        bootstrap_seed=analysis.SEED,bootstrap_replicates=analysis.REPLICATES,
        KID='DEFERRED_HOLDOUT',FID='NOT_EVALUATED')
    reg_path = result/'metrics_registration.json'; seal(reg_path,registration)
    seal(result/'model_metadata.json',evaluator.metadata); binding = identity(registration)
    evaluation_identity = dict(modelmanifest_sha256=manifest_sha,metric_evaluator_identity=evaluator.identity(),
        numerical_runtime=flags,batch_size=batch['chosen_batch_size'],qualified_batch_sizes=batch['qualified_batch_sizes'],
        batch_qualification_sha256=sha(metric_dir/'metric_batch_qualification.json'))
    records = registration['source_identity']; mismatch = replay.ReplayEngine._derangement()
    targets,features = [],[]
    for index in range(100):
        if stopped(): raise PauseRequested('Stop requested while preparing fixed source features')
        spec = admission['cache_inventory'][index]['stages']['P2048']
        own.require(sha(spec['archive']) == spec['archive_sha256'],'Reference archive changed')
        with np.load(spec['archive'],allow_pickle=False) as archive: target = pixels(archive['source_rgb']).copy()
        own.require(rgb_sha(target) == admission['source_float_sha256'][index],'Fixed common target changed')
        targets.append(target)
        with torch.no_grad():
            features.append(native.dino_features(dino,torch.from_numpy(target[None]).to('cuda:0'))[0].cpu().numpy())
    features = np.stack(features)
    started = time.monotonic(); allrows,baselines,outputs = [],[],{}
    for index,record in enumerate(records):
        if stopped(): raise PauseRequested('Stop requested at committed metric source boundary')
        gpu_available(); verify(sources)
        own.require(numeric_flags(torch) == flags,'Frozen metric numerical flags changed')
        target,rawrows,mapping,input_bindings = own.load_source_payload(root,index,registration)
        own.require(np.array_equal(target,targets[index]),'Common scoring target changed')
        checkpoint = metric_dir/'source_checkpoints'/f'{index:04d}.json'
        if checkpoint.exists():
            done = own.validate_scored_checkpoint(read(checkpoint),binding,index,input_bindings,evaluation_identity)
        else:
            source_started = time.monotonic(); images,slots = own.deduplicate(rawrows,mapping)
            target_tensor = torch.from_numpy(target[None].copy()); truth = int(record['class_index'])
            prepared = evaluator.prepare_reference(target_tensor); prediction = int(prepared['resnet50'][0])
            baseline = dict(source_id=record['image_id'],source_index=index,preprocessing_id=record['preprocessing_id'],
                reference_sha256=rgb_sha(target),true_class_index=truth,resnet50_source_prediction=prediction,
                resnet50_source_top1_label=int(prediction == truth))
            legacy,observed_feature,embeddings = native.quality_metrics(target,list(images),lpips,dino,torch.device('cuda:0'))
            np.testing.assert_allclose(observed_feature,features[index],rtol=1e-6,atol=1e-6)
            with torch.no_grad():
                negative = torch.from_numpy(features[mismatch[index]][None]).to('cuda:0').expand(len(images),-1)
                mismatched = torch.nn.functional.cosine_similarity(torch.from_numpy(embeddings).to('cuda:0'),negative,dim=1).cpu().tolist()
            scorer = runner.PendingMetricScorer(evaluator,torch,target_tensor,prepared,truth,
                batch['chosen_batch_size'],batch['qualified_batch_sizes'])
            rows = []
            for raw,slot in zip(rawrows,slots):
                value = dict(raw,**legacy[slot],dino_mismatched=mismatched[slot],
                    dino_specificity=legacy[slot]['dino_cosine']-mismatched[slot],
                    mismatch_source_id=records[mismatch[index]]['image_id'],mismatch_source_index=mismatch[index],
                    modelmanifest_sha256=manifest_sha)
                cache_key = replay.metric_cache_key(images[slot],target,evaluator.identity())
                value['metric_cache_key'] = cache_key; rows.append(value); scorer.add(cache_key,images[slot],value)
            scorer.flush()
            confidence = []; offset = 0
            for count in qualified_chunks(len(images),batch['chosen_batch_size'],batch['qualified_batch_sizes']):
                with torch.inference_mode():
                    rgb = torch.from_numpy(images[offset:offset+count]).to('cuda:0')
                    logits = evaluator.models['resnet50'](metric_module.preprocess_resnet50(rgb))
                    probability,predicted = logits.float().softmax(-1).max(-1)
                    confidence.extend(zip(probability.cpu().tolist(),predicted.cpu().tolist()))
                offset += count
            for row,slot in zip(rows,slots):
                classification(row,*confidence[slot],prediction)
                for name in (*analysis.METRICS,*EXTRA_METRICS,'dino_mismatched','resnet50_top1_probability'):
                    own.require(np.isfinite(row[name]),'Missing/nonfinite registered metric: '+name)
            done = dict(binding=binding,source_index=index,rows=rows,baseline=baseline,input_bindings=input_bindings,
                evaluation_identity=evaluation_identity,unique_images=len(images),
                metric_batch_sizes_used=scorer.batch_sizes_used,mismatch_source_index=mismatch[index],
                confidence_threshold=.5,elapsed_seconds=time.monotonic()-source_started)
            done['payload_sha256'] = identity(done)
            own.validate_scored_checkpoint(done,binding,index,input_bindings,evaluation_identity); seal(checkpoint,done)
        # Hash-bound old rows are retained; no metric-model recomputation is
        # needed on resume, but their original float-cache receipts are rechecked.
        own.require([(r['replay_row_id'],r['image_sha256'],r['original_row_sha256']) for r in done['rows']]
            == [(r['replay_row_id'],r['image_sha256'],r['original_row_sha256']) for r in rawrows],
            'Resumed score rows differ from the immutable native outputs')
        outputs[str(checkpoint)] = sha(checkpoint); allrows.extend(done['rows']); baselines.append(done['baseline'])
        progress('RUNNING',sources_complete=index+1,rows=len(allrows),elapsed_seconds=time.monotonic()-started)
    summary,per_source,paired = summarize(analysis,allrows,baselines,registration)
    for name,values in (('metrics_per_frame.csv',allrows),('source_baseline.csv',baselines),
                        ('metrics_summary.csv',summary),('metrics_per_source.csv',per_source),
                        ('metrics_paired_intervals.csv',paired)):
        path = result/name; write_csv(path,values); outputs[str(path)] = sha(path)
    for path in (reg_path,result/'model_metadata.json',metric_dir/'metric_batch_qualification.json',admission_path):
        outputs[str(path)] = sha(path)
    verify(sources); verify(model_bindings); verify(admission['input_bindings'])
    done = dict(status='OWN_CONTROLS_METRICS_COMPLETE',synthetic=False,sources=100,rows=own.ROWS,metric_groups=18,
        own_methods_by_N={str(n):list(ms) for n,ms in own.METHODS.items()},
        metrics_registration_sha256=sha(reg_path),score_input_admission_sha256=sha(admission_path),
        metric_evaluator_identity=evaluator.identity(),numerical_runtime=flags,
        metric_batch_qualification_sha256=sha(metric_dir/'metric_batch_qualification.json'),
        metric_qualified_batch_sizes=batch['qualified_batch_sizes'],metric_batch_size=batch['chosen_batch_size'],
        paired_bootstrap_unit='source_mean_after_three_noise_repeats',bootstrap_seed=analysis.SEED,
        bootstrap_replicates=analysis.REPLICATES,source_bindings=sources,
        bindings={**sources,**model_bindings,**admission['input_bindings']},outputs=outputs,
        selection_uses_development=False,new_metrics_used_for_selection=False,training_updates=0,
        policy_selection_updates=0,holdout_access=False,native_scores_preserved=True,new_metric_offset_applied=False,
        elapsed_metric_seconds=time.monotonic()-started)
    seal(complete_path,done); progress('COMPLETE',sources_complete=100,rows=own.ROWS); return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--launch-id',default=None)
    args = parser.parse_args(); paths = own.c.locations(args.root); out = paths['out']; out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (out/'run.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = [False]; signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True)); launch_id = args.launch_id or str(uuid.uuid4())
    def progress(status,**fields):
        write(out/'score_status.json',dict(status=status,stage='score',pid=os.getpid(),launch_id=launch_id,
            safe_pause_handler_installed=True,updated=time.time(),**fields))
    progress('STARTING'); failure = out/'score_failure.json'
    try:
        own.require(not failure.exists(),'Previous score failure requires review before resume')
        score(args.root,progress,lambda:stop[0])
    except PauseRequested as error:
        progress('PAUSED',reason=str(error)); return 75
    except Exception as error:
        if not failure.exists():
            write(failure,dict(status='FAILED_REQUIRES_REVIEW',stage='score',error=repr(error),
                launch_id=launch_id,automatic_restart_allowed=False))
        progress('FAILED_REQUIRES_REVIEW',error=repr(error)); raise
    finally:
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
