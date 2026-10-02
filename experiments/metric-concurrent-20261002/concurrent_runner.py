"""Score only completed studies beside M2; original sources/outputs stay read-only."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
import cache_handoff as cache
from scheduling_guard import SchedulingPause


class PauseRequested(RuntimeError):
    pass


def original_resource_pause(error):
    runtime = sys.modules.get('latent_enhancement.runtime')
    dedicated = getattr(runtime, 'ResourceBusy', None)
    return isinstance(dedicated, type) and isinstance(error, dedicated)


def timed_replay(iterator, timing):
    """Host wall time only; no new GPU synchronization or numerical operations."""
    iterator = iter(iterator)
    while True:
        started = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            timing['replay_seconds'] += time.perf_counter() - started
            return
        timing['replay_seconds'] += time.perf_counter() - started
        yield item


def imports(root):
    source = root / 'experiments/unified-metrics-20261002'
    sys.path.insert(0, str(source))
    import runner, replay, metric_models, batch_speed
    for module in (runner, replay, metric_models, batch_speed):
        if Path(module.__file__).resolve().parent != source.resolve():
            raise RuntimeError('Unexpected frozen metric module origin')
    from replay_compat import install
    install(replay, root)
    return runner, replay, metric_models, batch_speed


def qualify_scalar(evaluator, torch, model_module, batch_module, directory, binding, device):
    path = directory / 'concurrent_scalar_qualification.json'
    if path.exists():
        receipt = cache.read(path)
        if (receipt.get('binding') != binding or receipt.get('status') != 'CONCURRENT_SCALAR_QUALIFICATION_PASS'
                or receipt.get('payload_sha256') != cache.identity({k: v for k, v in receipt.items() if k != 'payload_sha256'})):
            raise RuntimeError('Concurrent scalar qualification differs')
        return path
    with torch.random.fork_rng(devices=[device.index or 0]), model_module.no_network():
        source, images = batch_module.synthetic_fixtures(torch)
        prepared = evaluator.prepare_reference(source)
        truth = int(prepared['resnet50'][0])
        def score():
            return [evaluator.score(source, images[i:i+1], torch.tensor([truth], dtype=torch.int64),
                                    prepared=prepared, label_conditioned=False)[0] for i in range(len(images))]
        first, second = score(), score()
        comparison = batch_module.compare_rows(first, second)
        if not comparison['passed']:
            raise RuntimeError('Concurrent GPU scalar repeatability failed')
    receipt = dict(status='CONCURRENT_SCALAR_QUALIFICATION_PASS', binding=binding, comparison=comparison,
                   synthetic_images=True, scientific_result=False, source_or_development_images_used=False,
                   metric_batch_size=1, rng_state_preserved=True, scalar_scores=first,
                   fixture_source_sha256=model_module.image_digest(source),
                   fixture_reconstructions_sha256=model_module.image_digest(images))
    receipt['payload_sha256'] = cache.identity(receipt)
    cache.write(path, receipt)
    return path


def run(args):
    import fcntl
    root, directory = Path(args.root).resolve(), Path(args.cache_dir).resolve()
    expected = root / 'outputs/METRIC-CONCURRENT-R4-20261002'
    if directory != expected:
        raise RuntimeError('Concurrent cache must use its separate registered output directory')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    if args.previous_cache_dir:
        previous_dir = Path(args.previous_cache_dir).resolve()
        if previous_dir != root / 'outputs/METRIC-CONCURRENT-R3-20261002':
            raise RuntimeError('Only the sealed direct R3 snapshot may be inherited')
        previous_lock = (previous_dir / 'owner.lock').open('a')
        fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = cache.CacheSnapshot(previous_dir)
    start = time.time()
    status = directory / 'status.json'
    def update(state, **extra):
        cache.write(status, dict(status=state, pid=os.getpid(), time=time.time(), elapsed_seconds=time.time()-start, **extra))
    registration = None
    def check_pause():
        path = directory / 'pause_requested.json'
        if path.exists():
            request = cache.read(path)
            if request.get('status') != 'PAUSE_REQUESTED':
                raise RuntimeError('Invalid concurrent pause marker')
            raise PauseRequested(str(request.get('reason', 'controller requested pause')))
    def pause_exit(error):
        if registration is not None:
            cache.snapshot_receipt(directory, registration)
        update('PAUSED_AT_FRAME_BOUNDARY', reason=str(error), unfinished_source_discarded=True,
               sources_complete=len(list((directory / 'source_checkpoints').glob('*.json'))))
        raise SystemExit(75) from error
    try:
        check_pause()
        update('LOADING_VERIFIED_MODELS')
        from scheduling_guard import install_admission
        scheduling = install_admission(root, Path(args.admission).resolve(), role='metrics')
        admitted = {key: scheduling[key] for key in ('admission_path', 'admission_sha256', 'guard_source_bindings')}
        runner, replay, models, batch = imports(root)
        own = dict(runner.source_bindings(), **cache.source_bindings())
        assets = runner.OUT / 'assets_complete.json'
        manifest = runner.OUT / 'modelmanifest.json'
        qualification = runner.OUT / 'models_qualification.json'
        ar, qr = cache.read(assets), cache.read(qualification)
        msha = cache.sha(manifest)
        if (ar.get('status') != 'ASSETS_READY' or ar.get('manifest_sha256') != msha
                or qr.get('status') != 'REAL_MODEL_WEIGHTS_QUALIFICATION_PASS'
                or qr.get('modelmanifest_sha256') != msha or qr.get('real_model_weights') is not True):
            raise RuntimeError('Verified complete assets and real-model qualification required')
        cache.verify(qr['source_bindings'])
        m1path = runner.PARENT / 'm1_complete.json'
        m1 = cache.read(m1path)
        pub = m1.get('publication', {})
        if (m1.get('status') != 'M1_COMPLETE' or pub.get('status') != 'PUSHED'
                or not pub.get('commit') or pub.get('commit') != pub.get('remote_commit')):
            raise RuntimeError('Sealed, pushed M1 results required')
        import torch
        if not torch.cuda.is_available() or not 0 < args.gpu_memory_fraction <= .45:
            raise RuntimeError('Visible CUDA and at most45 percent own memory allocation required')
        device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        free, total = torch.cuda.mem_get_info(device)
        if free < total * args.gpu_memory_fraction:
            raise RuntimeError('Insufficient free VRAM for the bounded concurrent worker')
        engine = replay.create_engine(root, studies=list(cache.STUDIES))
        if tuple(engine.rows) != cache.STUDIES:
            raise RuntimeError('Unexpected active/unsealed study included')
        flags = runner.runtime_flags(torch)
        evaluator = models.MetricEvaluator(manifest, device=engine.loaded['device'])
        if flags != runner.runtime_flags(torch) or any(v['status'] != 'READY' for v in evaluator.metadata['metrics'].values()):
            raise RuntimeError('Metric model setup changed frozen runtime or omitted an asset')
        metricid = evaluator.identity()
        bound = engine.setup_bound_files()
        compatibility = replay._historical_latent_alias_receipt
        bound.update(compatibility['source_bindings'])
        bound[str(m1path)] = cache.sha(m1path)
        for study in cache.STUDIES:
            bound[str(engine.layouts[study].rows)] = cache.sha(engine.layouts[study].rows)
        asset_inputs = {str(p): cache.sha(p) for p in (assets, manifest, qualification)}
        asset_inputs[admitted['admission_path']] = admitted['admission_sha256']
        scalar_binding = dict(metric_evaluator_identity=metricid, modelmanifest_sha256=msha,
                              numerical_runtime=flags, source_bindings=own,
                              gpu_name=torch.cuda.get_device_name(device), gpu_memory_fraction=args.gpu_memory_fraction)
        qp = qualify_scalar(evaluator, torch, models, batch, directory, scalar_binding, device)
        asset_inputs[str(qp)] = cache.sha(qp)
        sources = []
        for i, record in enumerate(engine.records):
            inventory = cache.row_inventory(engine, replay, i)
            sources.append(dict(source_index=i, source_id=record['image_id'], preprocessing_id=record['preprocessing_id'],
                reference_sha256=replay.rgb_fingerprint(engine.target(i)), true_class_index=int(record['class_index']),
                row_inventory_sha256=cache.identity(inventory), frames=len(inventory)))
        registration = dict(schema_version=1, studies=list(cache.STUDIES), synthetic=False, training_updates=0,
            policy_selection_updates=0, original_m2_read_or_changed=False, metric_batch_size=1,
            metric_evaluator_identity=metricid, modelmanifest_sha256=msha, numerical_runtime=flags,
            source_bindings=own, frozen_inputs=bound, asset_inputs=asset_inputs, sources=sources,
            scheduling_admission=admitted,
            replay_compatibility=compatibility,
            original_replay_inventory=engine.manifest(), gpu_memory_fraction=args.gpu_memory_fraction)
        if previous is not None:
            registration['inherited_cache'] = cache.inheritance_contract(previous, registration)
            registration['asset_inputs'].update(previous.bindings)
        cache.seal(directory / 'concurrent_registration.json', registration)
        cache.seal(directory / 'model_metadata.json', evaluator.metadata)
        checkpoints = directory / 'source_checkpoints'
        checkpoints.mkdir(exist_ok=True)
        if previous is not None:
            cache.import_snapshot(previous, directory, registration)
            cache.snapshot_receipt(directory, registration)
        newly_completed = 0
        for index, record in enumerate(engine.records):
            cache.verify(own); cache.verify(bound); cache.verify(asset_inputs)
            if runner.runtime_flags(torch) != flags:
                raise RuntimeError('Frozen replay runtime changed')
            path = checkpoints / f'{index:03}.json'
            if path.exists():
                cache.validate_checkpoint(cache.read(path), registration, index)
                continue
            target = engine.target(index)
            reference = torch.from_numpy(target).unsqueeze(0)
            source_started = time.perf_counter()
            reference_started = time.perf_counter()
            prepared = evaluator.prepare_reference(reference)
            timing = dict(reference_seconds=time.perf_counter()-reference_started, replay_seconds=0.,
                          scorer_seconds=0., unique_metric_images=0, metric_cache_hit_frames=0,
                          host_wall_time=True, added_cuda_synchronization=False, scientific_result=False)
            truth, pred = int(record['class_index']), int(prepared['resnet50'][0])
            baseline = dict(source_id=record['image_id'], preprocessing_id=record['preprocessing_id'],
                reference_sha256=sources[index]['reference_sha256'], true_class_index=truth,
                resnet50_source_prediction=pred, resnet50_source_top1_label=int(pred == truth))
            scorer = runner.PendingMetricScorer(evaluator, torch, reference, prepared, truth, 1, [1])
            rows, images, last_status = [], {}, 0
            for study in cache.STUDIES:
                originals = {replay.row_id(study, row): replay.source_row_hash(row)
                             for row in engine.by_source[study].get(index, [])}
                for row, rgb, parity in timed_replay(engine.iterate_source(study, index), timing):
                    check_pause()
                    rowid = replay.row_id(study, row)
                    key = replay.metric_cache_key(rgb, target, metricid)
                    if parity.get('synthetic') is not False or parity.get('replay_parity_passed') is not True:
                        raise RuntimeError('Real original replay parity required')
                    values = {}
                    was_cached = key in scorer.cache
                    metric_started = time.perf_counter()
                    scorer.add(key, rgb, values)
                    timing['scorer_seconds'] += time.perf_counter() - metric_started
                    timing['metric_cache_hit_frames'] += int(was_cached)
                    timing['unique_metric_images'] += int(not was_cached)
                    cache.validate_scores(values, truth, pred)
                    rows.append(dict(study=study, row_id=rowid, source_row_sha256=originals[rowid],
                                     metric_cache_key=key, parity=parity))
                    if key in images and images[key] != parity['rgb_sha256']:
                        raise RuntimeError('Exact image cache key collision')
                    images[key] = parity['rgb_sha256']
                    if time.time() - last_status > 20:
                        update('SCORING_SEALED_STUDIES', source_index=index, study=study, frames_in_source=len(rows),
                               sources_complete=len(list(checkpoints.glob('*.json'))), total_sources=100,
                               own_reserved_bytes=torch.cuda.memory_reserved(device), gpu_memory_fraction=args.gpu_memory_fraction,
                               operational_timing=dict(timing, source_elapsed_seconds=time.perf_counter()-source_started))
                        last_status = time.time()
            metric_started = time.perf_counter()
            scorer.flush()
            timing['scorer_seconds'] += time.perf_counter() - metric_started
            timing['source_elapsed_seconds'] = time.perf_counter() - source_started
            done = dict(status='SEALED_SOURCE_METRICS_COMPLETE', synthetic=False, metric_batch_size=1,
                registration_sha256=cache.identity(registration), source_index=index, source=sources[index],
                baseline=baseline, rows=rows, operational_timing=timing,
                scores={key: dict(image_sha256=images[key], metrics=value) for key, value in scorer.cache.items()})
            done['payload_sha256'] = cache.identity(done)
            cache.validate_checkpoint(done, registration, index)
            cache.verify(own); cache.verify(bound); cache.verify(asset_inputs)
            cache.write(path, done)
            newly_completed += 1
            torch.cuda.empty_cache()
            cache.snapshot_receipt(directory, registration)
            check_pause()
            if args.max_sources and newly_completed >= args.max_sources:
                update('PILOT_COMPLETE', sources_complete=len(list(checkpoints.glob('*.json'))), total_sources=100,
                       reusable_by_final_pipeline=True)
                return
        engine.verify_frozen(); cache.verify(own); cache.verify(bound); cache.verify(asset_inputs)
        cache.snapshot_receipt(directory, registration)
        update('SEALED_STUDIES_COMPLETE', sources_complete=100, total_sources=100, reusable_by_final_pipeline=True)
    except (PauseRequested, SchedulingPause) as error:
        pause_exit(error)
    except BaseException as error:
        if original_resource_pause(error):
            pause_exit(error)
        update('FAILED', error=str(error)); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--previous-cache-dir', help='Sealed direct R3 snapshot; keep identical on every resume')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.45)
    parser.add_argument('--max-sources', type=int, default=0, help='New source limit; zero completes remaining100')
    args = parser.parse_args()
    if args.max_sources < 0:
        parser.error('max-sources must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
