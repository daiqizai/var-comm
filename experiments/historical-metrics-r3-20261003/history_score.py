"""Score an unchanged native historical adapter, with one checkpoint per source.

This process never invokes a training, calibration or original delivery entry.
Original row fields are retained; every newly computed metric uses new_ prefix.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import importlib
import json
from pathlib import Path
import sys
import time
from history_common import (read, write, sha, verify, identity, row_ids, checkpoint_valid,
    write_csv, source_bindings, canonical, normalize_metadata, native_interop_setup, metric_numerical_context)
from history_float_cache import SourceImages, verify_saved


def ordered_payload(rows, parities, ids):
    values = {r['history_row_id']: r for r in rows}
    proofs = {p['history_row_id']: p for p in parities}
    if len(values) != len(rows) or len(proofs) != len(parities) or set(values) != set(ids) or set(proofs) != set(ids):
        raise RuntimeError('Incomplete/duplicate original output identity')
    return [values[k] for k in ids], [proofs[k] for k in ids]


def first_source_receipt(out, checkpoint, regpath, done):
    if (done.get('source_index') != 0 or not done.get('rows')
            or len(done.get('parity', [])) != len(done['rows'])
            or not all(p.get('replay_parity_passed') is True and p.get('synthetic') is False
                       for p in done['parity'])):
        raise RuntimeError('First-source receipt requires a complete actual source parity checkpoint')
    value = dict(status='REAL_FIRST_SOURCE_PARITY_PASS', scientific_result=True, synthetic=False,
        parity_passed=True, source_index=0, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        registration_sha256=sha(regpath), rows=len(done['rows']), original_values_preserved=True,
        evaluation_identity=done['evaluation_identity'],
        original_rows_sha256=identity([r['history_original_row_sha256'] for r in done['rows']]),
        training_updates=0, policy_selection_updates=0)
    value['proof_types'] = dict(
        original_scalar_replay=sum(p.get('original_scalar_parity_available', True) is True for p in done['parity']),
        new_deterministic_reference=sum(p.get('original_scalar_parity_available') is False for p in done['parity']))
    path = out/'first_source_qualification.json'
    if path.exists() and read(path) != value:
        raise RuntimeError('Existing first-source qualification differs')
    write(path, value)


def score(root, module_name, study, queue_registration):
    root = Path(root).resolve()
    queue_registration = Path(queue_registration).resolve()
    from history_controller import validate_queue, parent_gate, gpu_pids, published
    queue = validate_queue(root, Path(__file__).parent, queue_registration)
    if study in queue.get('inherited_studies', {}):
        raise RuntimeError('Completed R2 studies are inherited read-only and must not be rescored')
    parent_proof = parent_gate(root)
    if gpu_pids():
        raise RuntimeError('Historical scoring requires an available GPU')
    source_pub_path = root/'outputs/HISTORICAL-METRICS-R3-20261003/source_publication.json'
    source_publication = read(source_pub_path); published(source_publication)
    if source_publication.get('runtime_source_bindings') != queue['source_bindings']:
        raise RuntimeError('Historical scoring source is not the published runtime')
    verify(source_publication['source_bindings']); verify(source_publication['published_files'])
    if {'adapter': module_name, 'study': study} not in queue['jobs']:
        raise RuntimeError('Historical job is not registered')
    out = root / 'outputs/HISTORICAL-METRICS-R3-20261003' / study
    result = root / 'results/historical_metrics_r3_20261003' / study
    out.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    original = root / 'experiments/unified-metrics-20261002'
    sys.path.insert(0, str(original))
    import torch
    import replay
    import runner
    from metric_models import MetricEvaluator
    from batch_speed import qualify_and_select_batch
    manifest = root / 'outputs/UNIFIED-METRICS-20261002/modelmanifest.json'
    original_registration_path = root/'results/unified_metrics_20261002/metrics_registration.json'
    original_registration = read(original_registration_path)
    original_registration_sha = sha(original_registration_path)
    metric_runtime = original_registration['numerical_runtime']
    if (original_registration.get('synthetic') is not False
            or original_registration.get('modelmanifest_sha256') != sha(manifest)):
        raise RuntimeError('Original metric numerical/model registration differs')
    model_qualification = root/'outputs/UNIFIED-METRICS-20261002/models_qualification.json'
    qualification = read(model_qualification)
    if (qualification.get('status') != 'REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
            or qualification.get('modelmanifest_sha256') != sha(manifest)):
        raise RuntimeError('Original real metric model qualification is missing or stale')
    verify(qualification['source_bindings'])
    own = source_bindings(Path(__file__).parent)
    verify(queue['source_bindings'])
    verify(queue['shared_metric_bindings'])
    if own != queue['source_bindings']:
        raise RuntimeError('Queue does not bind this exact runtime')
    adapter = importlib.import_module(module_name).create_adapter(root, study)
    with native_interop_setup(torch, metric_runtime['interop_threads']):
        adapter.setup()
    verify(adapter.bindings)
    runtime = runner.runtime_flags(torch)
    device = torch.device('cuda:0')
    with metric_numerical_context(torch, metric_runtime, runner.runtime_flags), torch.random.fork_rng(devices=[device.index]):
        evaluator = MetricEvaluator(manifest, device=device)
    if runner.runtime_flags(torch) != runtime:
        raise RuntimeError('Metric construction changed native numerical flags')
    if any(x['status'] != 'READY' for x in evaluator.metadata['metrics'].values()):
        raise RuntimeError('Every requested metric must be ready')
    if evaluator.identity() != original_registration['metric_evaluator_identity']:
        raise RuntimeError('Historical and original six-study metric evaluators differ')
    with metric_numerical_context(torch, metric_runtime, runner.runtime_flags), torch.random.fork_rng(devices=[device.index]):
        batch = qualify_and_select_batch(evaluator, device, manifest, out,
            runtime_context=dict(study=study, native_bindings=adapter.bindings, numerical_runtime=metric_runtime,
                                 native_numerical_runtime=runtime))
    if runner.runtime_flags(torch) != runtime:
        raise RuntimeError('Batch qualification changed native numerical flags')
    batchpath = out / 'metric_batch_qualification.json'
    expected = [list(adapter.expected_rows(i)) for i in range(len(adapter.records))]
    if not expected or any(not rows for rows in expected):
        raise RuntimeError('Historical adapter omitted a registered source')
    ids = [row_ids(study, i, rows) for i, rows in enumerate(expected)]
    registered = dict(status='REGISTERED', study=study, adapter=module_name, synthetic=False,
        training_updates=0, policy_selection_updates=0, source_bindings=own,
        original_bindings=dict(adapter.bindings), queue_registration_sha256=sha(queue_registration),
        parent_completion_bindings=parent_proof['inputs'], source_publication_sha256=sha(source_pub_path),
        model_qualification_sha256=sha(model_qualification),
        modelmanifest_sha256=sha(manifest), evaluator_identity=evaluator.identity(),
        numerical_runtime=metric_runtime, native_numerical_runtime=runtime,
        metric_numerical_runtime=metric_runtime, metric_runtime_restored_after_each_call=True,
        interop_setup='original metric value; duplicate identical native request only',
        original_metric_registration_sha256=original_registration_sha,
        metric_batch_size=batch['chosen_batch_size'],
        batch_qualification_sha256=sha(batchpath), source_count=len(expected),
        frame_count=sum(map(len, expected)), expected_ids_sha256=identity(ids),
        source_ids=[r['image_id'] for r in adapter.records],
        source_identity=[{k:r[k] for k in ('image_id','class_index','preprocessing_id')} for r in adapter.records],
        metric_qualified_batch_sizes=batch['qualified_batch_sizes'],
        original_rows_retained=True, new_metric_prefix='new_', new_metrics_used_for_selection=False,
        float_reconstructions_retained=True,
        KID='DEFERRED_HOLDOUT', FID='NOT_EVALUATED')
    registered = canonical(registered)
    regpath = result / 'registration.json'
    if regpath.exists() and read(regpath) != registered:
        raise RuntimeError('Historical registration changed on resume')
    write(regpath, registered)
    write(result / 'model_metadata.json', evaluator.metadata)
    binding = identity(registered)
    evaluation_identity = {k: registered[k] for k in ('modelmanifest_sha256', 'evaluator_identity',
        'numerical_runtime', 'metric_batch_size', 'batch_qualification_sha256', 'metric_qualified_batch_sizes')}
    started = time.time()
    allrows, baselines = [], []
    for index, record in enumerate(adapter.records):
        verify(own)
        checkpoint = out / 'source_checkpoints' / f'{index:04d}.json'
        checkpoint_context = dict(expected_rows=expected[index], study=study, source_index=index,
            evaluation_identity=evaluation_identity, qualified_batch_sizes=batch['qualified_batch_sizes'])
        if checkpoint.exists():
            done = checkpoint_valid(read(checkpoint), binding, ids[index], **checkpoint_context)
            verify_saved(done['float_reconstructions'], done['rows'])
        else:
            waiting = defaultdict(deque)
            for row, rid in zip(expected[index], ids[index]):
                waiting[identity(row)].append(rid)
            scored, parities, score_rows = [], [], []
            scorer = None
            image_cache = None
            targetsha = None
            last_status = 0
            truth = int(record['class_index'])
            for row, rgb, target, parity in adapter.iterate_source(index):
                if parity.get('replay_parity_passed') is not True or parity.get('synthetic') is not False:
                    raise RuntimeError('Actual historical replay parity is required')
                rawhash = identity(row)
                if not waiting[rawhash]:
                    raise RuntimeError('Unexpected or repeated historical row')
                rid = waiting[rawhash].popleft()
                rgb = replay.rgb_array(rgb)
                target = replay.rgb_array(target)
                fingerprint = replay.rgb_fingerprint(target)
                if scorer is None:
                    image_cache = SourceImages(target)
                    targetsha = fingerprint
                    target_tensor = torch.from_numpy(target).unsqueeze(0)
                    with metric_numerical_context(torch, metric_runtime, runner.runtime_flags), torch.random.fork_rng(devices=[device.index]):
                        prepared = evaluator.prepare_reference(target_tensor)
                    pred = int(prepared['resnet50'][0])
                    baseline = dict(study=study, source_index=index, source_id=record['image_id'],
                        preprocessing_id=record['preprocessing_id'], reference_sha256=targetsha,
                        true_class_index=truth, resnet50_source_prediction=pred, resnet50_source_top1_label=int(pred == truth))
                    scorer = runner.PendingMetricScorer(evaluator, torch, target_tensor, prepared, truth,
                        batch['chosen_batch_size'], batch['qualified_batch_sizes'])
                elif fingerprint != targetsha:
                    raise RuntimeError('Source target changed between historical methods')
                metadata = normalize_metadata(row, adapter.metadata(row))
                if (not isinstance(metadata, dict)
                        or any(type(metadata.get(k)) is not bool for k in ('label_conditioned','reference_only','oracle'))):
                    raise RuntimeError('Historical scientific metadata missing')
                value = dict(row)
                if any(k.startswith(('history_', 'new_')) for k in value):
                    raise RuntimeError('Original row collides with additive metric namespace')
                value.update(history_row_id=rid, history_study=study, history_source_index=index,
                    history_source_id=record['image_id'], history_preprocessing_id=record['preprocessing_id'],
                    history_original_row_sha256=rawhash, history_metadata_json=json.dumps(metadata, sort_keys=True),
                    history_modelmanifest_sha256=registered['modelmanifest_sha256'], history_replay_parity_passed=True,
                    history_image_sha256=replay.rgb_fingerprint(rgb), history_reference_sha256=targetsha,
                    history_true_class_index=truth)
                if parity.get('original_scalar_parity_available') is False:
                    native = parity.get('native_metrics', {})
                    if set(native) != {'psnr_db','lpips_alex','dino_cosine'}:
                        raise RuntimeError('New deterministic reference requires all three actual native metrics')
                    value.update({'new_native_'+k: v for k,v in native.items()})
                image_cache.add(rid, rgb)
                holder = {}
                key = replay.metric_cache_key(rgb, target, registered['evaluator_identity'])
                with metric_numerical_context(torch, metric_runtime, runner.runtime_flags), torch.random.fork_rng(devices=[device.index]):
                    scorer.add(key, rgb, holder)
                scored.append(value)
                score_rows.append(holder)
                parities.append(dict(parity, history_row_id=rid))
                if time.time() - last_status > 30:
                    write(out / 'status.json', dict(status='RUNNING', study=study, source_index=index,
                        sources_complete=index, sources_total=len(expected), current_source_frames=len(scored),
                        frames_complete=len(allrows), expected_frames=registered['frame_count'], elapsed_seconds=time.time()-started))
                    last_status = time.time()
            if any(waiting.values()) or scorer is None:
                raise RuntimeError('Historical native replay did not cover all original rows')
            with metric_numerical_context(torch, metric_runtime, runner.runtime_flags), torch.random.fork_rng(devices=[device.index]):
                scorer.flush()
            for value, scores in zip(scored, score_rows):
                value.update({'new_' + key: val for key, val in scores.items()})
            scored, parities = ordered_payload(scored, parities, ids[index])
            done = dict(binding=binding, source_index=index, rows=scored, parity=parities, baseline=baseline,
                evaluation_identity=evaluation_identity,
                metric_batch_sizes_used=scorer.batch_sizes_used, unique_images=len(scorer.cache),
                float_reconstructions=image_cache.save(out/'reconstructions'/f'{index:04d}.npz', ids[index]))
            done['payload_sha256'] = identity(done)
            checkpoint_valid(done, binding, ids[index], **checkpoint_context)
            write(checkpoint, done)
        if (done['baseline']['source_id'] != record['image_id']
                or done['baseline']['true_class_index'] != int(record['class_index'])
                or done['baseline']['preprocessing_id'] != record['preprocessing_id']):
            raise RuntimeError('Source classification baseline identity differs')
        if index == 0:
            first_source_receipt(out, checkpoint, regpath, done)
        if runner.runtime_flags(torch) != runtime:
            raise RuntimeError('Native runtime changed during scoring')
        allrows.extend(done['rows'])
        baselines.append(done['baseline'])
    verify(own)
    verify(adapter.bindings)
    if adapter.bindings != registered['original_bindings']:
        raise RuntimeError('Native adapter input inventory changed after registration')
    verify(queue['shared_metric_bindings'])
    if sha(manifest) != registered['modelmanifest_sha256']:
        raise RuntimeError('Frozen metric weights manifest changed')
    if sha(original_registration_path) != original_registration_sha:
        raise RuntimeError('Original metric numerical registration changed')
    if len(allrows) != registered['frame_count']:
        raise RuntimeError('Historical study is incomplete')
    write_csv(result / 'metrics_per_frame.csv', allrows)
    write_csv(result / 'source_baseline.csv', baselines)
    outputs = {str(p): sha(p) for p in result.iterdir() if p.is_file()}
    write(out / 'completion.json', dict(status='HISTORICAL_STUDY_METRICS_COMPLETE', study=study,
        frames=len(allrows), sources=len(expected), synthetic=False, parity_passed=True,
        training_updates=0, policy_selection_updates=0,
        inputs={**adapter.bindings, **queue['shared_metric_bindings'], **parent_proof['inputs'],
                str(manifest):sha(manifest), str(model_qualification):sha(model_qualification),
                str(original_registration_path):original_registration_sha,
                str(source_pub_path):sha(source_pub_path), str(queue_registration):sha(queue_registration)}, source_bindings=own,
        outputs=outputs, elapsed_seconds=time.time()-started))
    write(out / 'status.json', dict(status='COMPLETE', study=study, sources_complete=len(expected), frames=len(allrows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--queue-registration', required=True)
    args = p.parse_args()
    try:
        score(args.root, args.adapter, args.study, args.queue_registration)
    except Exception as error:
        write(Path(args.root) / 'outputs/HISTORICAL-METRICS-R3-20261003' / args.study / 'status.json',
            dict(status='FAILED', error=str(error), time=time.time()))
        raise


if __name__ == '__main__':
    main()
