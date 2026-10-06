"""Prepared actual-RX development adapter; no model or job is constructed here.

Only the frozen eighteen H declarations on the original development100 are
accepted. Reconstruction uses the existing receiver and its actual-wire cache.
Target pixels are passed separately to scoring. This module is not a GPU
launcher, registrar, metric-suite runner, or proof that evaluation has run.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

COUNT = 100
SLOTS = 18
SEEDS = (6201, 6202, 6203)
FRAMES_PER_SOURCE = 54
FRAMES = 5400
FINAL = {'RAW_SOURCE_DECODED', 'ARITHMETIC_SOURCE_DECODED',
         'WIRE_REJECT_GRAY', 'ARITHMETIC_SOURCE_INVALID_GRAY'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def merge(*maps):
    result = {}
    for values in maps:
        for path, value in values.items():
            require(path not in result or result[path] == value, 'Conflicting bound SHA: ' + path)
            result[path] = value
    return result


def verify(values):
    for path, value in values.items():
        require(Path(path).is_absolute() and sha(path) == value, 'Changed bound file: ' + path)


def schedule_rows(schedule):
    require(len(schedule) == SLOTS, 'Exactly eighteen H policy declarations required')
    rows = {r['development_slot']: r for r in schedule}
    require(set(rows) == set(range(SLOTS)), 'Duplicate/missing H slot or reserved MAIN slot')
    for slot, row in rows.items():
        expected_role = ('H_WHOLE_SYSTEM' if slot < 8 else
                         'H_RAW_PARTIAL_SYSTEM' if slot < 10 else 'H_FIXED_M7_ATTRIBUTION')
        require(row['status'] == 'FROZEN_POLICY_READY',
                'Declared H point is not executable; no replacement or fabricated frame permitted')
        require(row['phase'] == 'development' and row['role'] == expected_role,
                'Frozen H development role differs')
        require(row['candidate']['snr_db'] in (13, 19), 'Only the two registered H SNRs allowed')
    return rows


def verify_cpu_closed(spec, driver, owner, wait):
    """Recheck normal CPU closure, actual trace/ledger equality, and MAIN reserve.

    `spec` uses driver config plus separate owner config. The existing development
    driver provides the explicitly development-aware closure checker; the older
    calibration-only wait.verify_batch must not be weakened or called instead.
    Caller must bind this returned provenance, this module and the supplied
    frozen modules in a separate visual execution registration before GPU use.
    """
    require(set(spec) == {'config', 'owner_config', 'registration', 'launch', 'completion'},
            'Unexpected CPU predecessor schema')
    ctx = driver.load_registered(spec['config'])
    cfg, reg = ctx['cfg'], ctx['reg']
    require(cfg['registration'] == spec['registration'] and cfg['owner_config'] == spec['owner_config']
            and reg['source_stage_scope'] == 'H_DEVELOPMENT_18_POINTS_CPU_ONLY'
            and reg['allowed_stage_ids'] == ['development', 'report']
            and Path(spec['completion']) == Path(cfg['out']) / 'completion.json',
            'Development CPU predecessor scope differs')
    oc = read(spec['owner_config'])
    require([s['id'] for s in oc['stages']] == ['development', 'report']
            and all(s['resource'] == 'cpu' for s in oc['stages']), 'Predecessor is not H-only CPU receive/merge')
    expected = dict(status='H_DEVELOPMENT_CPU_RECEIVE_COMPLETE', source_count=COUNT,
                    frame_count=FRAMES, worker_index=None, workers=2,
                    declared_policy_snr_points=SLOTS, executable_policy_snr_points=SLOTS,
                    maximum_H_packet_calls=10800, MAIN_reserved_packet_calls=2400,
                    images_scored=False, source_decode_complete=False,
                    arithmetic_source_decode_complete=False, GPU_used=False,
                    development_used=True, holdout_used=False, policy_selection=False,
                    MAIN_complete=False, overall_development_complete=False,
                    H_full_delivery_claimed=False, visual_stage_started=False,
                    requires_owner_worker_exit_receipt_before_visual_stage=True)
    closure_spec = dict(spec, config=spec['owner_config'])
    closure_spec.pop('owner_config')
    closure = driver.closed_batch(owner, wait, closure_spec, cfg['owner_module'], expected, owner.raw_process_state)
    done = closure['done']
    for key, value in expected.items():
        require(done.get(key) == value, 'Development CPU completion differs: ' + key)
    require(done['registration_sha256'] == sha(spec['registration'])
            and done['driver_config_sha256'] == sha(spec['config'])
            and done['source_completion_sha256'] == sha(cfg['source_completion'])
            and done['finalized_policies_sha256'] == sha(cfg['finalized'])
            and done['budget_registration_sha256'] == sha(cfg['budget_registration']),
            'Completed CPU inputs changed')
    for values in (done['outputs'], done['source_bindings'], done['input_bindings']):
        verify(values)
    rows = schedule_rows(ctx['rows'])
    ids = ctx['ids']
    require(len(ids) == len(set(ids)) == COUNT and done['source_ids'] == ids
            and done['source_indices'] == list(range(COUNT)), 'Original development100 population differs')
    require(len(done['worker_identities']) == 2, 'Two actual packet-worker exits required')
    for identity in done['worker_identities']:
        require(any(owner.same_identity(identity, child) for child in closure['children'])
                and wait.exited(identity, owner.raw_process_state), 'Packet worker is unclosed or absent from owner proof')
    paths = [Path(cfg['out']) / f'worker_{i % 2}' / 'traces' / f'{i:04d}.json' for i in range(COUNT)]
    require(all(done['outputs'].get(str(p)) == sha(p) for p in paths), 'All100 actual source traces must be sealed')
    for i, path in enumerate(paths):
        cp_path = Path(cfg['out']) / f'worker_{i % 2}' / 'source_checkpoints' / f'{i:04d}.json'
        require(done['outputs'].get(str(cp_path)) == sha(cp_path), 'All100 source checkpoints must be sealed before model loading')
        cp = read(cp_path)
        require(cp['status'] == 'H_DEVELOPMENT_CPU_SOURCE_COMPLETE' and cp['source_id'] == ids[i]
                and cp['source_index'] == i and cp['registration_sha256'] == done['registration_sha256']
                and cp['outputs'] == {str(path): sha(path)} and cp['frame_count'] == FRAMES_PER_SOURCE,
                'Completed development source checkpoint differs')
    events, inventory = driver.trace_inventory(ctx, paths)
    audit = driver.audit_ledger(cfg['ledger'], events)
    require(audit == done['ledger_audit'] and inventory['source_count'] == COUNT
            and inventory['frame_count'] == FRAMES, 'Actual trace/ledger coverage changed')
    current = owner.budget_snapshot(cfg['ledger'], sha(cfg['budget_registration']), driver.PHASES, quiescent=True)
    driver.budget_admission(reg['budget_before'], current, events)
    require(current == done['budget_after'] and done['budget_before'] == reg['budget_before']
            and reg['budget_before']['charged'] == 153960
            and current['phase_charged']['development'] == audit['paid_events']
            and current['development_remaining'] >= 2400, 'Frozen CPU budget or reserved MAIN allocation changed')
    require(closure['bindings'].get(spec['completion']) == sha(spec['completion']), 'CPU receipt not sealed by successful owner')
    bindings = merge(closure['bindings'], done['source_bindings'], done['input_bindings'],
                     {p: sha(p) for p in spec.values()})
    return dict(context=ctx, cfg=cfg, done=done, closure=closure, bindings=bindings,
                source_ids=ids, schedule=list(rows.values()), core=ctx['core'],
                catalogue=ctx['context']['catalogue'], before=current,
                population=ctx['args']['development_registration'])


def load_source(ctx, index):
    require(type(index) is int and 0 <= index < COUNT, 'Only original development100 indices allowed')
    sid = ctx['source_ids'][index]
    base = Path(ctx['cfg']['out']) / f'worker_{index % 2}'
    cp_path, trace_path = base / 'source_checkpoints' / f'{index:04d}.json', base / 'traces' / f'{index:04d}.json'
    for path in (cp_path, trace_path):
        require(ctx['done']['outputs'].get(str(path)) == sha(path), 'Actual development source evidence changed')
    cp, source = read(cp_path), read(trace_path)
    require(cp['status'] == 'H_DEVELOPMENT_CPU_SOURCE_COMPLETE'
            and source['status'] == 'H_DEVELOPMENT_CPU_SOURCE_TRACES'
            and cp['source_id'] == source['source_id'] == sid and cp['source_index'] == source['source_index'] == index
            and cp['registration_sha256'] == source['registration_sha256'] == ctx['done']['registration_sha256']
            and cp['outputs'] == {str(trace_path): sha(trace_path)}
            and cp['frame_count'] == len(source['frames']) == FRAMES_PER_SOURCE, 'Development source trace identity differs')
    verify(source['source_bindings'])
    old, pop = ctx['context'], ctx['population']
    row = old['manifest']['records'][index]
    ap = Path(row['checkpoint'])
    require(old['assets']['outputs'].get(str(ap)) == sha(ap) == row['checkpoint_sha256'], 'Target checkpoint unsealed')
    asset = read(ap)
    require(asset['source_index'] == index and asset['source_id'] == sid
            and asset['preprocessing_id'] == pop['preprocessing_ids'][index]
            and asset['original_development_data_binding'] == pop['data_bindings'][index]
            and asset['archive'] == row['archive'], 'Target is not original registered development image')
    verify(asset['outputs'])
    require(all(old['assets']['outputs'].get(p) == s for p, s in asset['outputs'].items()), 'Target archive is unsealed')
    # Deliberately extract only pixels. Encoded truth/source tokens are not read
    # by this adapter and never reach Receiver or CachedReceiver.
    with np.load(asset['archive'], allow_pickle=False) as archive:
        target = archive['pixels'].copy()
    require(target.dtype == np.uint8 and target.shape == (3, 256, 256)
            and hashlib.sha256(target.tobytes()).hexdigest() == asset['preprocessing_id'], 'Target pixels/preprocessing changed')
    bindings = merge(source['source_bindings'], {str(p): sha(p) for p in (cp_path, trace_path, ap)}, asset['outputs'])
    return source, target, asset['preprocessing_id'], bindings


def validate_frame_grid(source, schedule, core):
    require(source['status'] == 'H_DEVELOPMENT_CPU_SOURCE_TRACES', 'Actual development source trace required')
    sid, index = source['source_id'], source['source_index']
    require(type(index) is int and 0 <= index < COUNT, 'Development index outside original100')
    rows = schedule_rows(schedule)
    expected = {(slot, seed) for slot in rows for seed in SEEDS}
    seen, frames = set(), []
    for frame in source['frames']:
        slot, seed = frame['development_slot'], frame['noise_seed']
        require((slot, seed) in expected and (slot, seed) not in seen, 'Duplicate, unregistered, or reserved MAIN frame')
        seen.add((slot, seed))
        entry, c = rows[slot], rows[slot]['candidate']
        require(frame['status'] == 'H_DEVELOPMENT_CPU_FRAME_TRACE'
                and frame['stage'] == frame['noise_stage'] == 'development'
                and frame['source_id'] == sid and frame['source_index'] == index
                and frame['role'] == entry['role'] and frame['candidate_id'] == c['candidate_id']
                and frame['arm'] == c['arm'] and frame['snr_db'] == c['snr_db']
                and frame['public_frame_counter'] == core.frame_counter(slot, index, seed)
                and frame['event_id'] == core.frame_event(slot, index, seed)
                and frame['scrambling_session'] == core.SESSION
                and frame['execution_registration_sha256'] == source['registration_sha256']
                and frame['total_symbols'] == 1024 and frame['frame_normalized'] is False
                and frame['arithmetic_source_decode_run'] is False and frame['neural_metrics_run'] is False
                and frame['policy_selection'] is False and frame['MAIN_PHY_run'] is False,
                'Actual development frame/public schedule identity differs')
        frames.append(frame)
    require(seen == expected, 'Incomplete eighteen-slot development source grid')
    return sorted(frames, key=lambda f: (f['development_slot'], f['noise_seed']))


def render_source(source, target, target_sha, schedule, core, receiver, rx_api, cache_api, cache_identity, boundary):
    frames = validate_frame_grid(source, schedule, core)
    entries = schedule_rows(schedule)
    cache = cache_api.CachedReceiver(receiver, cache_identity)
    rows, images, bysha = [], {}, {}
    for frame in frames:
        boundary()
        c = entries[frame['development_slot']]['candidate']
        # The reconstruction boundary contains only actual received header/body
        # and its paid public profile, including legally accepted wrong content.
        result = cache.reconstruct(rx_api.receiver_view(frame), frame['event_id'])
        summary = result['summary']
        require(summary['source_decode_complete'] is True and summary['source_status'] in FINAL,
                'Pending arithmetic state cannot be scored or silently made gray')
        scores = rx_api.score_reconstruction(result, target, expected_preprocessing_sha256=target_sha)
        diagnostic = rx_api.evaluation_diagnostics(result, frame['evaluation_only'])
        image = result['image']
        require(image.dtype == np.float32 and image.shape == (3, 256, 256)
                and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1, 'Only finite float32 CHW RGB01 accepted')
        ih = summary['image_sha256']
        if ih not in bysha:
            key = f'image_{len(images):04d}'
            bysha[ih] = key
            images[key] = image.copy()
        else:
            key = bysha[ih]
            require(np.array_equal(images[key], image), 'RGB hash collision')
        rows.append(dict(phase='development', development_slot=frame['development_slot'], role=frame['role'],
                         candidate_id=c['candidate_id'], arm=c['arm'], target_m=c['target_m'], K=c.get('K', 0),
                         q=c['q'], nominal_rate=c['nominal_rate'], snr_db=c['snr_db'], source_id=frame['source_id'],
                         source_index=frame['source_index'], noise_seed=frame['noise_seed'], event_id=frame['event_id'],
                         public_frame_counter=frame['public_frame_counter'], noise_stage='development', **scores,
                         source_status=summary['source_status'], gray=summary['gray'], image_key=key,
                         receiver_view_sha256=summary['receiver_view_sha256'], received_m=summary['received_m'],
                         received_K=summary['received_K'], received_mode=summary['received_mode'],
                         received_profile_id=summary['received_profile_id'], rx_summary=summary, evaluation_only=diagnostic))
    cost = dict(frames=len(rows), unique_images=len(images), receiver_cache_hits=cache.hits, receiver_cache_misses=cache.misses,
                arithmetic_canonical_calls=sum(r['rx_summary']['arithmetic_canonical_executed_this_frame'] for r in rows),
                suffix_renderer_calls=sum(r['rx_summary']['suffix_renderer_executed_this_frame'] for r in rows),
                gray_frames=sum(r['gray'] for r in rows), new_packet_decodes=0, policy_selection=False, MAIN_rendered=False)
    return rows, images, cost


def validate_complete_rows(rows, source_ids):
    require(len(source_ids) == len(set(source_ids)) == COUNT and len(rows) == FRAMES, 'Incomplete development100/5400 result coverage')
    expected = {(i, slot, seed) for i in range(COUNT) for slot in range(SLOTS) for seed in SEEDS}
    seen = set()
    for row in rows:
        key = row['source_index'], row['development_slot'], row['noise_seed']
        require(key in expected and key not in seen and row['source_id'] == source_ids[key[0]], 'Duplicate/wrong source or reserved MAIN result')
        require(row['phase'] == 'development' and row['source_status'] in FINAL
                and row['rx_summary']['source_decode_complete'] is True, 'Unresolved actual received state in final rows')
        seen.add(key)
    require(seen == expected, 'Missing development frame')
    return dict(source_count=COUNT, frame_count=FRAMES, H_policy_snr_points=SLOTS, MAIN_frames=0)


def write_source(out, index, source_id, rows, images, cost, inputs, registration_sha256, cache_api, rx_api):
    """Exclusive float RGB output; no complete-stage claim or unsafe overwrite."""
    require(type(index) is int and 0 <= index < COUNT and len(rows) == FRAMES_PER_SOURCE
            and all(r['source_index'] == index and r['source_id'] == source_id for r in rows), 'Wrong development source write scope')
    require({(r['development_slot'], r['noise_seed']) for r in rows} == {(s, n) for s in range(SLOTS) for n in SEEDS},
            'Missing/duplicate H slot at source write')
    require(cost['frames'] == FRAMES_PER_SOURCE and cost['new_packet_decodes'] == 0
            and cost['policy_selection'] is False and cost['MAIN_rendered'] is False, 'Wrong output cost scope')
    verify(inputs)
    base = Path(out)
    archive = base / 'images' / f'{index:04d}.npz'
    values = base / 'sources' / f'{index:04d}.json'
    checkpoint = base / 'source_checkpoints' / f'{index:04d}.json'
    require(not any(p.exists() for p in (archive, values, checkpoint)), 'Existing development source preserved; no retry/overwrite')
    for p in (archive, values, checkpoint):
        p.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as f:
        np.savez(f, **images)
    # Keep caller rows intact: the stored copies carry archive references.
    stored = [dict(r, image_archive=str(archive)) for r in rows]
    cache_api.validate_source_archive(stored, archive, rx_api)
    with values.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(stored, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    outputs = {str(p): sha(p) for p in (archive, values)}
    receipt = dict(status='H_DEVELOPMENT_RX_SOURCE_COMPLETE', registration_sha256=registration_sha256,
                   source_index=index, source_id=source_id, frame_count=FRAMES_PER_SOURCE,
                   input_bindings=inputs, outputs=outputs, images_scored=True, scored_metrics=['mse', 'psnr_db'],
                   source_decode_complete=True, arithmetic_source_decode_complete=True,
                   development_used=True, holdout_used=False, overall_development_complete=False,
                   H_full_delivery_claimed=False, **cost)
    with checkpoint.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    return merge(outputs, {str(checkpoint): sha(checkpoint)}), archive.stat().st_size
