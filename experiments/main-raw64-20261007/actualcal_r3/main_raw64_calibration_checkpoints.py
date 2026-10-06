"""Source/point checkpoints and exact historical reuse for raw64 calibration.

This module is callable preparation, not a stage launcher or ledger authority.
The registered caller authenticates normal predecessor receipts once and gives
this module explicit immutable pins and actual paid event records. It must not
treat the earlier header-domain audit as a complete reuse admission.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import os
import numpy as np

SNRS = (1, 4, 7, 10, 13, 19)
NOISE = (4101, 4102, 4103)
NAMESPACE = 'MAIN_CALIBRATION_V1'
POINT_STATUS = 'MAIN_RAW64_CALIBRATION_POINT_COMPLETE_V1'
PRIMARY = 'dinov2_vitl14_cosine'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    """Read current bytes every time; filesystem metadata is not content identity.

    No extension, size or stat tuple admits a cached digest. Large source NPZs
    remain verified by the original driver once per source, outside this helper.
    """
    path = Path(path).absolute()
    def key():
        s=path.stat(); return (str(path),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    before=key()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    require(key()==before,'File changed while hashing: '+str(path))
    return h.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def verify_files(bindings):
    require(isinstance(bindings, dict) and bindings, 'Explicit nonempty immutable pins required')
    for path, expected in bindings.items():
        require(Path(path).is_absolute() and file_sha(path) == expected, 'Changed or unbound file: '+path)


def schedule(profiles, shortlist):
    """Validate the already frozen shortlist. Never rank or add a candidate."""
    pool = {p['candidate_id']: p for p in profiles}
    require(len(profiles) == len(pool) == 433 and [p['profile_id'] for p in profiles] == list(range(433)), 'Complete registered433 representative catalogue required')
    require(shortlist['selection_role'] == 'calibration' and shortlist['modulation_coverage_quota'] is False, 'Original cross-modulation shortlist required')
    cells = {(c['family'], c['snr_db']): c for c in shortlist['cells']}
    require(len(cells) == len(shortlist['cells']) == 12 and set(cells) == {(f, s) for f in ('WHOLE', 'PARTIAL') for s in SNRS}, 'Frozen six-SNR family grid required')
    for snr in SNRS:
        w, p = cells['WHOLE', snr], cells['PARTIAL', snr]
        a, b = w['candidate_ids'], p['candidate_ids']
        require(len(a) == len(set(a)) == len(b) == len(set(b)) == 3 and set(a+b) <= set(pool), 'Each family has three distinct admitted candidates')
        require(all(pool[k]['K'] == 0 for k in a) and b[0] == a[0] == p['fixed_whole_fallback'] and all(pool[k]['K'] > 0 for k in b[1:]), 'Fixed global-best K0 plus two positiveK; no modulation quota')
        require(len({pool[k]['wire_key'] for k in a+b}) == len(set(a+b)) <= 5, 'Shared K0 is one actual wire observation')
    points = sorted({(s, k) for (_, s), c in cells.items() for k in c['candidate_ids']})
    return [dict(slot=i, snr_db=s, candidate_id=k, profile_id=pool[k]['profile_id'], wire_key=pool[k]['wire_key'], family_memberships=[f for f in ('WHOLE', 'PARTIAL') if k in cells[f, s]['candidate_ids']]) for i, (s, k) in enumerate(points)]


def make_plan(profiles, shortlist, source_ids, *, science_sha256, shortlist_sha256, source_manifest_sha256):
    require(len(source_ids) == len(set(source_ids)) == 1000 and all(isinstance(s, str) and s for s in source_ids), 'Original ordered calibration1000 required')
    for v in (science_sha256, shortlist_sha256, source_manifest_sha256):
        require(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v), 'Frozen input SHA required')
    points = schedule(profiles, shortlist)
    return dict(schema='MAIN_RAW64_ACTUAL_CALIBRATION_PLAN_V1', status='PREPARED_REQUIRES_STAGE_REGISTRATION', population_role='calibration', source_ids=list(source_ids), source_manifest_sha256=source_manifest_sha256, science_sha256=science_sha256, shortlist_sha256=shortlist_sha256, catalogue_digest=digest(profiles), schedule=points, noise_seeds=list(NOISE), source_count=1000, frame_count=1000*3*len(points), maximum_new_packet_calls=2*1000*3*len(points), phase='actual_calibration', development_used=False, holdout_used=False, policy_selection=False, GPU_used=False)


def frame_counter(index, snr, seed):
    require(type(index) is int and 0 <= index < 1000 and snr in SNRS and seed in NOISE, 'Original source/SNR/noise identity required')
    return (SNRS.index(snr)*1000+index)*3+NOISE.index(seed)


def standard_noise(source_id, seed):
    require(isinstance(source_id, str) and source_id and seed in NOISE, 'Original source/noise seed required')
    v = int.from_bytes(hashlib.sha256(canonical([NAMESPACE, 1024, source_id, seed]).encode()).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(v)).standard_normal((1024, 2))


def prepare_frame(record, point, seed, payload, waveform, transmission):
    """Simulation TX shell only; no original tokens/target enter an RX call."""
    counter = frame_counter(record['source_index'], point['snr_db'], seed)
    wave = np.asarray(waveform)
    require(wave.dtype == np.float64 and wave.shape == (1024, 2) and np.isfinite(wave).all(), 'Full original float64 waveform required')
    require(transmission['N'] == 1024 and transmission['header_uses'] == 68 and transmission['body_uses']+transmission['idle_uses'] == 956 and transmission['profile_id'] == point['profile_id'] and transmission['frame_counter'] == counter and transmission['waveform_sha256'] == array_sha(wave), 'Original fully paid waveform identity differs')
    require(transmission['E'] == float(np.square(wave).sum()), 'Actual frame energy differs')
    b = np.asarray(payload)
    require(b.dtype == np.uint8 and b.ndim == 1 and np.isin(b, (0, 1)).all(), 'Raw actual TX payload required')
    z = standard_noise(record['source_id'], seed)
    received = wave + z*10**(-float(point['snr_db'])/20)
    expected = dict(source_index=record['source_index'], source_id=record['source_id'], source_tokens_sha256=record['tokens_sha256'], candidate_id=point['candidate_id'], profile_id=point['profile_id'], wire_key=point['wire_key'], snr_db=point['snr_db'], noise_seed=seed, population_role='calibration', public_frame_counter=counter, noise_sha256=array_sha(z), transmitted_payload_sha256=array_sha(b), received_sha256=array_sha(received), transmission=deepcopy(transmission))
    return expected, received


def event_request(row, event, kind, catalogue):
    if 'request' in event:
        require(isinstance(event['request'], str), 'SQLite request must be canonical JSON text')
        return json.loads(event['request'])
    # Original MAIN_R2 stores only request_sha. Reconstruct exactly its frozen
    # audit_frame request and compare the paid digest; this is not a new request.
    require(row['status'] == 'MAIN_RAW_KEEP_RECEPTION_COMPLETE', 'New paid events must retain request JSON')
    request = dict(schema='MAIN_ACTUAL_RX_REQUEST_V1', received_sha256=row['received_sha256'], codebook_sha256=row['codebook_sha256'], aliases_sha256=row['aliases_sha256'], snr_db=float(row['snr_db']), public_frame_counter=row['public_frame_counter'], N=1024, body_session='actual-body', body_group=0, component=kind)
    if kind == 'body':
        request.update(actual_profile_id=row['rx_profile_id'], actual_wire_key=catalogue.entry(row['rx_profile_id'])['wire_key'], header_result_sha256=digest(row['header']), body_received_float32_sha256=row['body_received_float32_sha256'])
    return request


def _paid(row, events, catalogue):
    records = row['packet_events']
    require(len(records) == row['logical_packet_calls'] == (2 if row['header_ok'] else 1) and row['packet_event_ids'] == [r['event_id'] for r in records], 'Actual header/body charge inventory differs')
    for i, item in enumerate(records):
        kind = 'header' if i == 0 else 'body'
        value = row['header'] if i == 0 else row['body']
        event = events.get(item['event_id'])
        require(event is not None and event['status'] == 'COMPLETE' and event['phase'] == 'actual_calibration' and event['kind'] == item['kind'] == kind and event['phy_key'] == item['phy_key'], 'Missing original COMPLETE paid event')
        require(isinstance(event['result'], str), 'Original SQLite result JSON text required')
        request = event_request(row, event, kind, catalogue); result = json.loads(event['result'])
        require(digest(request) == event['request_sha'] == item['request_sha256'] and digest(result) == event['result_sha'] == item['result_sha256'] and result == value, 'Actual paid request/result differs from frame')
        require(request['received_sha256'] == row['received_sha256'] and request['public_frame_counter'] == row['public_frame_counter'] and request['snr_db'] == row['snr_db'] and request['N'] == 1024 and request['body_session'] == 'actual-body' and request['body_group'] == 0 and request['component'] == kind and request['codebook_sha256'] == row['codebook_sha256'] and request['aliases_sha256'] == row['aliases_sha256'], 'Original event does not describe this exact frame')
        if i:
            require(request['actual_profile_id'] == row['rx_profile_id'] and request['header_result_sha256'] == digest(row['header']) and request['body_received_float32_sha256'] == row['body_received_float32_sha256'], 'Wrong received-ID body request')


def _frame(row, expected, catalogue, present_actual, events):
    require(all(row[k] == v for k, v in expected.items()), 'Source/payload/waveform/noise/frame identity changed')
    require(row['status'] in ('MAIN_RAW_KEEP_RECEPTION_COMPLETE', 'MAIN_RAW64_KEEP_RECEPTION_COMPLETE') and row['source_decode_complete'] is True and row['image_reconstruction_complete'] is False and row['quality_scored'] is False and row['new_metric_calls'] == 0, 'Complete raw KEEP CPU trace required')
    presented = present_actual(row['header'], row['body'], catalogue)
    require(all(row[k] == v for k, v in presented.items()), 'Actual RX hard bits/KEEP state changed')
    _paid(row, events, catalogue)


def admit_trace(row, expected, point, received, *, catalogue, present_actual, paid_events, evidence, legacy_catalogue=None):
    """Caller binds evidence to a normal completion and exact ledger audit.

    Historical evidence has a trace pin plus normal_receipts pins. Newly paid
    frames instead require the current live owner admission; their eventual
    normal closure is the later supervising parent's responsibility.
    Historical bytes stay in place; the projected row explicitly records the
    old catalogue and scheduler metadata. This never relabels an old paid call
    as a newly paid call.
    """
    legacy = legacy_catalogue is not None
    require(evidence['paid_results_verified'] is True, 'Actual ledger audit admission required')
    if legacy:
        require(evidence['normal_exit_verified'] is True, 'Historical normal parent/worker admission required')
        receipts = evidence['normal_receipts']
    else:
        require(evidence['live_owner_admission_verified'] is True, 'New receipt requires current live registered parent admission')
        receipts = evidence['admission_receipts']
    verify_files(evidence['bindings'])
    trace_path = evidence['trace_path']
    require(trace_path in evidence['bindings'] and receipts and all(p in evidence['bindings'] for p in receipts), 'Trace and actual admission receipts must be explicitly pinned')
    original_rows = json.loads(Path(trace_path).read_text(encoding='utf-8'))
    require(sum(digest(r) == digest(row) for r in original_rows) == 1, 'Exact frame must occur once in pinned original trace')
    if legacy:
        require(row['codebook_sha256'] == legacy_catalogue.digest and row['aliases_sha256'] == legacy_catalogue.aliases_digest, 'Wrong original public catalogue')
        require(row['header']['header_crc_ok'] is False or row['header']['header_ok'] is True, 'CRC-accepted old unknown ID needs new header/body reception')
        require(legacy_catalogue.entry(row['profile_id']) == catalogue.entry(row['profile_id']), 'Transmitted public ID semantics changed')
        if row['header_ok']:
            require(legacy_catalogue.entry(row['rx_profile_id']) == catalogue.entry(row['rx_profile_id']), 'Actual received public ID semantics changed')
    else:
        require(row['codebook_sha256'] == catalogue.digest and row['aliases_sha256'] == catalogue.aliases_digest, 'Wrong current public catalogue')
    y = np.asarray(received)
    require(y.dtype == np.float64 and y.shape == (1024, 2) and array_sha(y) == expected['received_sha256'], 'Exact newly prepared receive waveform required for reuse')
    if row['header_ok']:
        profile = catalogue.entry(row['rx_profile_id'])
        require(array_sha(y[68:68+profile['groups'][0]['symbols']].astype(np.float32)) == row['body_received_float32_sha256'], 'Actual received-ID body slice changed')
        require(event_request(row, paid_events[row['packet_events'][1]['event_id']], 'body', catalogue)['actual_wire_key'] == profile['wire_key'], 'Paid body uses a different received wire')
    _frame(row, expected, catalogue, present_actual, paid_events)
    result = deepcopy(row)
    result.update(slot=point['slot'], family_memberships=point['family_memberships'], effective_catalogue_digest=catalogue.digest, effective_aliases_digest=catalogue.aliases_digest)
    result['execution_provenance'] = dict(status='MAIN_RAW64_EXACT_RX_PROVENANCE_V1', origin='HISTORICAL_EXACT_REUSE' if legacy else 'NEW_PAID_RECEIVE', receiver_interpretation_unchanged=legacy, original_row_sha256=digest(row), original_trace={'path': trace_path, 'sha256': evidence['bindings'][trace_path]}, original_slot=row.get('slot'), original_family_memberships=row.get('family_memberships'), original_catalogue_digest=row['codebook_sha256'], normal_receipts=list(receipts) if legacy else [], admission_receipts=[] if legacy else list(receipts), input_bindings=dict(evidence['bindings']), paid_event_ids=list(row['packet_event_ids']), newly_charged_packet_calls=0 if legacy else row['logical_packet_calls'])
    return result


def visual_reuse(cpu_row, visual_row, image, *, old_visual_identity, required_visual_identity, old_metric_identity, required_metric_identity, source_record):
    """VAR RGB/DINO reuse only. Direct-Dc is a separate same-RX rendering arm."""
    require(canonical(old_visual_identity) == canonical(required_visual_identity) and canonical(old_metric_identity) == canonical(required_metric_identity), 'Visual/model/numeric/metric identity changed')
    provenance = cpu_row['execution_provenance']
    require(provenance['origin'] == 'HISTORICAL_EXACT_REUSE', 'Historical visual requires an admitted historical CPU frame')
    for k, v in cpu_row.items():
        if k in ('execution_provenance', 'effective_catalogue_digest', 'effective_aliases_digest', 'slot', 'family_memberships', 'image_reconstruction_complete'):
            continue
        require(visual_row[k] == v, 'Visual row changes original CPU fact: '+k)
    require(visual_row['cpu_image_reconstruction_complete'] is False and visual_row['image_reconstruction_complete'] is True and visual_row['synthetic'] is False and visual_row['primary_metric'] == PRIMARY, 'Actual original VAR visual row required')
    require(visual_row['preprocessing_id'] == source_record['preprocessing_id'] and visual_row['source_id'] == source_record['source_id'] and visual_row['source_index'] == source_record['source_index'], 'Target/source identity changed')
    a = np.asarray(image)
    require(a.dtype == np.float32 and a.shape == (3, 256, 256) and np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() and array_sha(a) == visual_row['image_sha256'], 'Exact original FLOAT32 RGB bytes required')
    require(type(visual_row.get(PRIMARY)) in (int, float) and math.isfinite(visual_row[PRIMARY]) and -1 <= visual_row[PRIMARY] <= 1, 'Original finite DINO-L result required')
    return dict(status='MAIN_RAW64_VAR_VISUAL_REUSE_ELIGIBLE_PENDING_NORMAL_EVIDENCE', render_arm='VAR', cpu_original_row_sha256=provenance['original_row_sha256'], visual_row_sha256=digest(visual_row), image_sha256=visual_row['image_sha256'], image_archive=visual_row['image_archive'], image_slot=visual_row['image_slot'], metric_identity_sha256=digest(required_metric_identity), visual_identity_sha256=digest(required_visual_identity), dinov2_vitl14_cosine=visual_row[PRIMARY], new_GPU_calls=0, direct_Dc_qualified=False)


def point_relative_path(index, point):
    require(type(index) is int and 0 <= index < 1000, 'Original source index required')
    key = point['candidate_id']
    require(isinstance(key, str) and len(key) == 64 and all(c in '0123456789abcdef' for c in key) and point['snr_db'] in SNRS, 'Exact representative candidate path required')
    return Path('sources')/f'{index:04d}'/f'snr{point["snr_db"]}'/(key+'.json')


def write_point(root, plan, index, point, rows, registration_sha256):
    require(isinstance(registration_sha256, str) and len(registration_sha256) == 64, 'Actual stage registration SHA required')
    require(point in plan['schedule'] and len(rows) == 3 and [r['noise_seed'] for r in rows] == list(NOISE), 'Exact frozen point three-noise checkpoint required')
    inputs = {}
    for row in rows:
        require(row['source_index'] == index and row['source_id'] == plan['source_ids'][index] and all(row[k] == point[k] for k in ('candidate_id', 'profile_id', 'wire_key', 'slot', 'snr_db', 'family_memberships')), 'Wrong point/source checkpoint')
        require(row['effective_catalogue_digest'] == plan['catalogue_digest'] and row['population_role'] == 'calibration', 'Wrong accepted catalogue/population')
        provenance = row['execution_provenance']
        require(provenance['origin'] in ('HISTORICAL_EXACT_REUSE', 'NEW_PAID_RECEIVE') and provenance['newly_charged_packet_calls'] == (0 if provenance['origin'] == 'HISTORICAL_EXACT_REUSE' else row['logical_packet_calls']), 'New charge accounting differs')
        require(provenance['status'] == 'MAIN_RAW64_EXACT_RX_PROVENANCE_V1' and (provenance['origin'] != 'HISTORICAL_EXACT_REUSE' or provenance['receiver_interpretation_unchanged'] is True), 'Exact received interpretation proof required')
        for path, h in provenance['input_bindings'].items():
            require(path not in inputs or inputs[path] == h, 'Conflicting point provenance pin')
            inputs[path] = h
    verify_files(inputs)
    result = dict(status=POINT_STATUS, registration_sha256=registration_sha256, plan_sha256=digest(plan), source_index=index, source_id=plan['source_ids'][index], point=deepcopy(point), noise_seeds=list(NOISE), frame_count=3, newly_charged_packet_calls=sum(r['execution_provenance']['newly_charged_packet_calls'] for r in rows), reused_frames=sum(r['execution_provenance']['origin'] == 'HISTORICAL_EXACT_REUSE' for r in rows), frames=rows, input_bindings=inputs, GPU_used=False, quality_ranked=False, development_used=False, holdout_used=False)
    target = Path(root)/point_relative_path(index, point)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as f:
        f.write(canonical(result)+'\n'); f.flush(); os.fsync(f.fileno())
    return {'path': str(target.absolute()), 'sha256': file_sha(target)}


def read_point(pin, plan, index, point, registration_sha256):
    require(file_sha(pin['path']) == pin['sha256'], 'Point checkpoint byte hash changed')
    value = json.loads(Path(pin['path']).read_text(encoding='utf-8'))
    require(value['status'] == POINT_STATUS and value['registration_sha256'] == registration_sha256 and value['plan_sha256'] == digest(plan) and value['source_index'] == index and value['source_id'] == plan['source_ids'][index] and value['point'] == point and value['noise_seeds'] == list(NOISE) and value['frame_count'] == len(value['frames']) == 3, 'Checkpoint belongs to another registered source/point')
    require([r['noise_seed'] for r in value['frames']] == list(NOISE) and value['newly_charged_packet_calls'] == sum(r['execution_provenance']['newly_charged_packet_calls'] for r in value['frames']), 'Checkpoint frame/charge summary differs')
    return value
