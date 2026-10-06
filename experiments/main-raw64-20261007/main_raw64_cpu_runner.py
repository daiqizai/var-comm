"""One registered CPU run: new-wire qualification, missing proxy cells, merge.

The pinned adapter prepares waveforms and returns a decoder callback. Only the
ledger calls that callback, after the durable charge. This runner never reads
image data and never continues into calibration, GPU or holdout stages.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from main_raw64_ledger import Ledger, canonical, digest, require

SCHEMA = 'MAIN_RAW64_CPU_REQUEST_V1'
ENV_THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify(mapping):
    require(isinstance(mapping, dict), 'Bindings must be a map')
    for p, h in mapping.items():
        require(Path(p).is_absolute() and sha(p) == h, 'Changed or relative input: ' + p)


def combine(*maps):
    result = {}
    for values in maps:
        for p, h in values.items():
            require(p not in result or result[p] == h, 'Conflicting source/input binding')
            result[p] = h
    return result


def build_plan(materials, target):
    """Translate an actual audited schedule, preserving full historic row counts.

    materials contains explicit source paths. It is safe on both local and
    remote layouts; no path is inferred from a downloaded basename.
    """
    m = read(materials)
    verify(m['bindings'])
    for k in ('schedule', 'catalogue', 'science_registration', 'legacy_bler', 'legacy_MAIN_proxy'):
        require(m['bindings'].get(m[k]) == sha(m[k]), 'Unbound plan material: ' + k)
    schedule, cat = read(m['schedule']), read(m['catalogue'])
    require(schedule['schema'] == 'MAIN_RAW64_EXACT_PHY_SCHEDULE_V1' and schedule['status'] == 'PREPARED_FOR_REGISTRATION', 'Exact root schedule required')
    require(schedule['science_registration_sha256'] == sha(m['science_registration']) and schedule['new_scientific_calls_performed'] == 0 and schedule['image_quality_read'] is False, 'Prospective no-quality schedule required')
    require(schedule['qualification_trials'] == '4noiseless +4highSNR60dB per new body; header8 exact paid IDs including invalid4095 to test rejection; no new quality', 'Qualification noise rule changed')
    require(schedule['blocks_per_missing_cell'] == 256 and schedule['proxy_randomness'].startswith('new fixed namespace MAIN_RAW64_PROXY_V1;'), 'Registered proxy rule changed')
    source_hashes = {name: sha(m[name]) for name in ('legacy_bler', 'legacy_MAIN_proxy')}
    legacy = read(m['legacy_bler']); newer = read(m['legacy_MAIN_proxy'])
    require(legacy['status'] == 'REFINEMENT_COMPLETE' and legacy['synthetic'] is False, 'Real completed legacy table required')
    original = {'legacy_bler': {(x['phy_key'], x['snr_db']): x for x in legacy['rows']}, 'legacy_MAIN_proxy': {(x['phy_key'], x['snr_db']): x for x in newer}}
    proof = {tuple(x['cell']): x for x in schedule['reuse_proof']}
    require(len(proof) == len(schedule['reuse_proof']), 'Duplicate reuse proof')
    reused, headers = [], []
    for row in schedule['reused_cells']:
        k = (row['phy_key'], row['snr_db']); witness = proof[k]; origin = witness['origin']
        full = original[origin][k]
        require(digest(full) == witness['row_sha256'] and row['reuse_origin'] == origin, 'Historical whole row differs from actual audit')
        old_counts = [full[x] for x in (('n_blocks', 'n_correct', 'n_reject', 'n_undetected') if origin == 'legacy_bler' else ('n_blocks', 'correct', 'rejected', 'undetected'))]
        require(old_counts == [row[x] for x in ('n_blocks', 'n_correct', 'n_reject', 'n_undetected')], 'Actual historic sample counts changed')
        v = dict(row, row_sha256=witness['row_sha256'], evidence={'path': m[origin], 'sha256': source_hashes[origin]})
        (headers if row['kind'] == 'header' else reused).append(v)
    require(len(headers) == 6 and {x['snr_db'] for x in headers} == {1, 4, 7, 10, 13, 19}, 'All six original headerCorrect cells required')
    groups = {p['groups'][0]['phy_key']: p['groups'][0] for p in cat['profiles']}
    qual = schedule['qualification']; h = [x for x in qual if x['kind'] == 'header']; bodies = [x for x in qual if x['kind'] == 'body']
    require(len(h) == 1 and h[0]['trials'] == len(h[0]['profile_ids']) == 8 and all(x['trials'] == 8 and x['group'] == groups[x['phy_key']] for x in bodies), 'Actual qualification domain/layout differs')
    missing = schedule['missing_cells']
    require(all(x['kind'] == 'body' and x['group'] == groups[x['phy_key']] and x['n_blocks'] == 256 for x in missing), 'Actual missing physical layout differs')
    newkeys = sorted(x['phy_key'] for x in bodies)
    plan = {'schema': 'MAIN_RAW64_CPU_PLAN_V1', 'status': 'FROZEN', 'namespace': 'MAIN_RAW64_CPU_V1', 'catalogue_digest': cat['catalogue_digest'], 'snrs_db': [1, 4, 7, 10, 13, 19], 'source_images_read': False, 'quality_ranked': False, 'all_body_phy_keys': sorted(groups), 'header_phy_key': h[0]['phy_key'],
        'qualification': {'mode': '4_noiseless_4_awgn60', 'noise_namespace': 'MAIN_RAW64_QUALIFICATION_V1', 'body_cases': 8, 'body_phy_keys': newkeys, 'reused_body_phy_keys': sorted(set(groups) - set(newkeys)), 'header_profile_ids': h[0]['profile_ids'], 'snr_db': 60},
        'proxy': {'random_namespace': 'MAIN_RAW64_PROXY_V1', 'blocks_per_cell': 256, 'cells': [{'phy_key': x['phy_key'], 'snr_db': x['snr_db']} for x in missing], 'reused_cells': reused, 'reused_header_correct_cells': headers},
        'phase_caps': schedule['phase_caps'], 'source_schedule': {'path': m['schedule'], 'sha256': sha(m['schedule'])}, 'input_bindings': m['bindings']}
    validate_plan(plan)
    require(schedule['wanted_cell_count'] == len(groups) * 6 + 6 and schedule['reused_cell_count'] == len(reused) + 6 and schedule['missing_cell_count'] == len(missing), 'Actual schedule count differs')
    save(target, plan)
    return {'path': str(Path(target).absolute()), 'sha256': sha(target), 'phase_caps': plan['phase_caps'], 'new_body_qualification': len(newkeys), 'missing_proxy_cells': len(missing), 'reused_body_cells': len(reused), 'reused_header_cells': len(headers)}


def build_request(materials, target):
    """Fill hashes from explicitly supplied paths; never infer legacy locations."""
    m = read(materials); r = dict(m['request'])
    require(r['schema'] == SCHEMA, 'Explicit full request fields required')
    sources = {p: sha(p) for p in m['source_paths']}
    inputs = combine({p: sha(p) for p in m['input_paths']}, read(r['plan'])['input_bindings'])
    for p in (r['plan'], r['reuse_admission'], r['science_registration'], r['adapter_config']['catalogue'], r['adapter_config']['legacy_qualification']):
        inputs = combine(inputs, {p: sha(p)})
    # This enumerates only the header's current small source tree; no RGB/model tree.
    src = Path(r['project_root']) / 'src' / 'var_comm'
    for p in sorted(src.rglob('*')):
        if p.is_file() and p.suffix in ('.py', '.cpp'):
            sources = combine(sources, {str(p): sha(p)})
    for k, p in r['adapter_config'].items():
        if k not in ('catalogue', 'legacy_qualification'):
            sources = combine(sources, {p: sha(p)})
    for p in (r['adapter_module'], str(Path(__file__).absolute()), str(Path(__file__).with_name('main_raw64_ledger.py').absolute())):
        sources = combine(sources, {p: sha(p)})
    inputs = combine(inputs, {str(Path(materials).absolute()): sha(materials)})
    verify(combine(sources, inputs)); r.update(source_bindings=sources, input_bindings=inputs)
    validate_plan(read(r['plan'])); save(target, r)
    return {'path': str(Path(target).absolute()), 'sha256': sha(target), 'source_count': len(sources), 'input_count': len(inputs), 'registration_created': False}


def validate_plan(plan):
    require(plan['schema'] == 'MAIN_RAW64_CPU_PLAN_V1' and plan['status'] == 'FROZEN', 'Actual enumerated/frozen plan required')
    require(plan['snrs_db'] == [1, 4, 7, 10, 13, 19] and plan['source_images_read'] is False and plan['quality_ranked'] is False, 'Metadata-only six-SNR plan required')
    q, p = plan['qualification'], plan['proxy']
    require(q['mode'] == '4_noiseless_4_awgn60' and q['noise_namespace'] == 'MAIN_RAW64_QUALIFICATION_V1' and p['random_namespace'] == 'MAIN_RAW64_PROXY_V1', 'Prospective payload/noise rules differ')
    require(isinstance(plan['catalogue_digest'], str) and len(plan['catalogue_digest']) == 64, 'Exact extended public catalogue required')
    require(q['body_cases'] == 8 and p['blocks_per_cell'] == 256, 'Prospectively registered8/256 CPU experiment required')
    require(q['snr_db'] == 60 and len(q['body_phy_keys']) == len(set(q['body_phy_keys'])), 'Distinct noiseless qualification layouts required')
    require(len(q['header_profile_ids']) == len(set(q['header_profile_ids'])) and all(type(i) is int and 0 <= i < 4096 for i in q['header_profile_ids']), 'Fixed12-bit header cases required')
    require(len(q['header_profile_ids']) > 0, 'Extended catalogue needs known/unknown header qualification')
    keys = set(plan['all_body_phy_keys'])
    require(len(keys) == len(plan['all_body_phy_keys']) and set(q['body_phy_keys']) <= keys, 'Qualification layouts must belong to complete domain')
    cells = [(r['phy_key'], r['snr_db']) for r in p['cells']]
    old = [(r['phy_key'], r['snr_db']) for r in p['reused_cells']]
    require(len(cells) == len(set(cells)) and len(old) == len(set(old)) and not set(cells) & set(old), 'Missing/reused cells overlap')
    require(set(cells) | set(old) == {(k, s) for k in keys for s in plan['snrs_db']}, 'Full domain must be covered before any cell can be dropped')
    require(set(q['body_phy_keys']) | set(q['reused_body_phy_keys']) == keys and not set(q['body_phy_keys']) & set(q['reused_body_phy_keys']), 'Every wire needs new or exact historical qualification')
    for row in p['reused_cells']:
        n, c, reject, undetected = (row[k] for k in ('n_blocks', 'n_correct', 'n_reject', 'n_undetected'))
        require(all(type(x) is int and x >= 0 for x in (n, c, reject, undetected)) and n > 0 and c + reject + undetected == n and row['evidence']['path'] and row['evidence']['sha256'] and row['row_sha256'], 'Reused cell preserves complete actual sample count and evidence')
    caps = {'qualification': len(q['body_phy_keys']) * 8 + len(q['header_profile_ids']), 'proxy': len(cells) * 256}
    require(all(plan['phase_caps'][p] == n for p, n in caps.items()), 'Caps must equal actual finite plan, not an old allowance')
    from main_raw64_ledger import valid_caps
    valid_caps(plan['phase_caps'])
    return plan['phase_caps']


def points(plan, phase):
    if phase == 'qualification':
        q = plan[phase]
        return ([{'kind': 'body', 'phy_key': k, 'snr_db': q['snr_db'], 'blocks': q['body_cases']} for k in sorted(q['body_phy_keys'])] +
                [{'kind': 'header', 'phy_key': plan['header_phy_key'], 'snr_db': q['snr_db'], 'blocks': 1, 'profile_id': p} for p in q['header_profile_ids']])
    require(phase == 'proxy', 'Only qualification/proxy are implemented')
    return [dict(r, kind='body', blocks=plan['proxy']['blocks_per_cell']) for r in sorted(plan['proxy']['cells'], key=lambda r: (r['phy_key'], r['snr_db']))]


def events(plan, phase, point):
    for i in range(point['blocks']):
        suffix = ('header/' + str(point['profile_id']) if point['kind'] == 'header' else 'body/' + point['phy_key'] + '/snr' + str(point['snr_db']) + '/block' + str(i))
        yield dict(point, phase=phase, index=i, noiseless=(phase == 'qualification' and (point['kind'] == 'header' or i < 4)), event_id=plan['namespace'] + '/' + phase + '/' + suffix)


def result_valid(event, value):
    require(isinstance(value, dict), 'Actual decoder must return JSON object')
    if event['kind'] == 'body':
        for k in ('correct', 'rejected', 'undetected', 'crc_accept', 'payload_equal'):
            require(type(value[k]) is bool, 'Boolean actual decoder outcomes required')
        require(sum(value[k] for k in ('correct', 'rejected', 'undetected')) == 1 and value['correct'] == (value['crc_accept'] and value['payload_equal']) and value['rejected'] == (not value['crc_accept']) and value['undetected'] == (value['crc_accept'] and not value['payload_equal']), 'Incorrect C/R/U partition')
        if event['phase'] == 'qualification':
            require(value['correct'], 'Noiseless actual body roundtrip failed')
    else:
        require(value['header_crc_ok'] is True and type(value['header_fields_legal']) is bool and value['header_ok'] == value['header_fields_legal'], 'Header qualification failed')
        require(value['profile_id'] == (event['profile_id'] if value['header_ok'] else None), 'Header actual received ID differs')


def register(request):
    r = read(request)
    require(r['schema'] == SCHEMA, 'Explicit new CPU request required')
    caps = validate_plan(read(r['plan']))
    bound = combine(r['source_bindings'], r['input_bindings'])
    verify(bound)
    require(all(bound.get(p) == h for p, h in read(r['plan']).get('input_bindings', {}).items()), 'Plan material omitted from actual request bindings')
    for p in (r['plan'], r['adapter_module'], r['science_registration'], str(Path(__file__).absolute()), str(Path(__file__).with_name('main_raw64_ledger.py').absolute())):
        require(bound.get(p) == sha(p), 'Direct execution dependency not bound: ' + p)
    science = read(r['science_registration'])
    require(science['schema'] == 'MAIN_RAW12_RASTER_KEEP_64QAM_SCIENCE_V1' and science['status'] == 'FROZEN_BEFORE_NEW_PHY_QUALITY_OR_SELECTION', 'Frozen new science contract required')
    require(science['scope']['N'] == 1024 and science['scope']['SNRs'] == [1, 4, 7, 10, 13, 19] and science['scope']['order'] == 'raster' and science['scope']['failure_rule'] == 'KEEP', 'Protocol scope differs')
    require(science['resources']['proxy_samples_per_missing_cell'] == 256 and science['resources']['qualification_per_new_wire_body_calls'] == 8, 'CPU sampling must equal science registration')
    catpath = r['adapter_config']['catalogue']; require(bound.get(catpath) == sha(catpath), 'Actual extended catalogue file must be bound')
    require(read(catpath)['catalogue_digest'] == read(r['plan'])['catalogue_digest'], 'CPU plan/catalogue digest differs')
    for point in read(r['plan'])['proxy']['reused_cells']:
        evidence = point['evidence']; require(bound.get(evidence['path']) == evidence['sha256'], 'Reused table outside admitted evidence')
    require(bound.get(r['reuse_admission']) == sha(r['reuse_admission']), 'Normal historical reuse admission must be explicit')
    reuse = read(r['reuse_admission'])
    require(reuse['status'] == 'MAIN_RAW64_REUSE_ADMITTED' and reuse['normal_owners_verified'] is True and reuse['source_images_read'] is False, 'Reuse requires completed original provenance admission')
    require(reuse['plan_sha256'] == sha(r['plan']), 'Reuse admission must cover this exact plan')
    require(r['resources']['workers'] == 2 and r['resources']['threads'] == 2 and r['resources']['nice'] == 15, 'Two CPU workers with two threads/nice15 required')
    affinities = r['resources']['affinities']
    require(len(affinities) == 2 and all(len(x) == len(set(x)) == 2 for x in affinities) and len(set(sum(affinities, []))) == 4, 'Two disjoint registered CPU pairs required')
    ceiling = science['resources']['wall_clock_ceiling_days'] * 86400
    require(Path(r['python']).is_absolute() and time.time() < r['deadline_unix'] <= time.time() + ceiling and 0 < r['max_seconds'] <= ceiling, 'Explicit interpreter and new finite deadline required')
    root, e, out = (Path(r[k]).absolute() for k in ('root', 'execution_dir', 'out'))
    require(root != e and root in e.parents and root != out and root in out.parents and e != out and e not in out.parents and out not in e.parents, 'Separate new-root execution/output directories required')
    require(not e.exists() and not out.exists() and not Path(r['ledger']).exists(), 'Fresh execution/output/independent ledger only')
    require(root in Path(r['ledger']).absolute().parents and root in Path(r['stop_file']).absolute().parents, 'New root owns ledger and STOP')
    require(not Path(r['stop_file']).exists(), 'STOP exists')
    e.mkdir(parents=True)
    try:
        cfg = dict(r, _path=str(e / 'config.json'), registration=str(e / 'registration.json'))
        save(e / 'config.json', cfg)
        registration = {'status': 'MAIN_RAW64_CPU_REGISTERED_V1', 'config_sha256': sha(e / 'config.json'), 'request_path': str(Path(request).absolute()), 'request_sha256': sha(request), 'source_bindings': r['source_bindings'], 'input_bindings': r['input_bindings'], 'plan_sha256': sha(r['plan']), 'phase_caps': caps, 'scope': ['qualification', 'proxy', 'merge'], 'automatic_calibration': False, 'GPU_used': False}
        save(e / 'registration.json', registration)
        return str(e / 'registration.json')
    except BaseException:
        save(e / 'registration_failure.json', {'traceback': traceback.format_exc()})
        raise


def load(registration):
    p = Path(registration).absolute(); reg = read(p); cfg = read(p.parent / 'config.json')
    require(reg['status'] == 'MAIN_RAW64_CPU_REGISTERED_V1' and cfg['registration'] == str(p) and reg['config_sha256'] == sha(cfg['_path']), 'Registration/config mismatch')
    verify(combine(reg['source_bindings'], reg['input_bindings']))
    require(reg['source_bindings'].get(str(Path(__file__).absolute())) == sha(__file__), 'Unbound runner entry')
    plan = read(cfg['plan']); require(validate_plan(plan) == reg['phase_caps'] and sha(cfg['plan']) == reg['plan_sha256'], 'Changed plan')
    return {'cfg': cfg, 'reg': reg, 'regsha': sha(p), 'plan': plan}


def process(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text(); rest = stat[stat.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': int(rest[19]), 'ppid': int(rest[1]), 'uid': p.stat().st_uid, 'argv': [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x], 'state': rest[0]}


def same(a, b):
    return all(a[k] == b[k] for k in ('pid', 'start_ticks', 'uid', 'argv'))


def capture(child, argv):
    until = time.monotonic() + 10
    while time.monotonic() < until:
        require(child.poll() is None, 'Child exited before identity admission')
        me = process(child.pid)
        if me['argv'] == argv and me['uid'] == os.getuid():
            return me
        time.sleep(.02)
    raise ValueError('Child identity capture timed out')


def check_stop(cfg):
    require(not Path(cfg['stop_file']).exists() and time.time() < cfg['deadline_unix'], 'Registered STOP/deadline')


def environment(cfg):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(cfg['pythonpath']))
    env.update({k: '2' for k in ENV_THREADS})
    return env


def set_resources(affinity):
    os.sched_setaffinity(0, set(affinity)); os.setpriority(os.PRIO_PROCESS, 0, 15)


def require_resources(cfg, index):
    require(sys.platform.startswith('linux') and str(Path(sys.executable).absolute()) == cfg['python'], 'Original exact LDPC interpreter path required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in ENV_THREADS), 'CPU-only environment required')
    require(set(os.sched_getaffinity(0)) == set(cfg['resources']['affinities'][index]) and os.getpriority(os.PRIO_PROCESS, 0) == 15, 'Worker resource contract changed')


def module(path):
    spec = importlib.util.spec_from_file_location('main_raw64_registered_packet_adapter', path)
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
    return value


def worker(registration, phase, index):
    ctx = load(registration); cfg = ctx['cfg']; require_resources(cfg, index); check_stop(cfg)
    e = Path(cfg['execution_dir']); proofpath = e / phase / ('admission_' + str(index) + '.json')
    until = time.monotonic() + 15
    while not proofpath.exists():
        check_stop(cfg); require(time.monotonic() < until and os.getppid() > 1, 'No live registered parent admission'); time.sleep(.02)
    proof = read(proofpath); me = process(os.getpid()); parent = process(os.getppid())
    require(proof['registration_sha256'] == ctx['regsha'] and same(proof['worker'], me) and same(proof['owner'], parent), 'Foreign worker/parent admission')
    nworkers = 1 if phase == 'qualification' else 2
    require(0 <= index < nworkers, 'Wrong phase worker')
    owned = list(enumerate(points(ctx['plan'], phase)))[index::nworkers]
    out = Path(cfg['out']) / phase / ('worker_' + str(index)); out.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(cfg['ledger'], ctx['regsha'], ctx['reg']['phase_caps'])
    try:
        adapter = module(cfg['adapter_module']).create(cfg)
        outputs = {}; count = 0
        for point_index, point in owned:
            rows = []
            for event in events(ctx['plan'], phase, point):
                check_stop(cfg); require(same(proof['owner'], process(os.getppid())), 'Registered owner disappeared')
                require_resources(cfg, index)
                request, callback = adapter.prepare(event)
                def decoded():
                    value = callback(); result_valid(event, value); return value
                value = ledger.decode_once(event, request, decoded, me)
                rows.append({'event': event, 'request_sha256': digest(request), 'result_sha256': digest(value), 'result': value}); count += 1
            target = out / ('point_' + str(point_index).zfill(6) + '.json')
            save(target, {'point_index': point_index, 'point': point, 'events': rows}); outputs[str(target)] = sha(target)
        done = {'status': 'MAIN_RAW64_CPU_WORKER_COMPLETE_V1', 'phase': phase, 'worker_index': index, 'worker_identity': me, 'registration_sha256': ctx['regsha'], 'config_sha256': sha(cfg['_path']), 'packet_calls': count, 'outputs': outputs, 'GPU_used': False, 'source_images_read': False}
        save(out / 'completion.json', done)
    except BaseException:
        save(out / 'failure.json', {'traceback': traceback.format_exc(), 'budget': ledger.snapshot(), 'worker_identity': me})
        raise


@contextmanager
def owner_lock(cfg):
    import fcntl
    with (Path(cfg['root']) / 'cpu_runner.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def request_stop(cfg, reason):
    if not Path(cfg['stop_file']).exists():
        save(cfg['stop_file'], {'reason': reason, 'time': time.time()})


def wait_closed(child, log, logpath, identity, exitpath):
    rc = child.wait(); log.flush(); os.fsync(log.fileno()); log.close()
    save(exitpath, {'process_waited': True, 'exit_code': rc, 'identity': identity, 'log': str(logpath), 'log_sha256': sha(logpath)})
    return rc


def drain_children(children):
    errors = {}; pending = list(enumerate(children))
    while pending:
        remaining = []
        for index, (child, log, logpath, identity, exitpath) in pending:
            try:
                rc = child.wait()
            except BaseException:
                errors[str(index) + '_wait'] = traceback.format_exc()
                remaining.append((index, (child, log, logpath, identity, exitpath)))
                continue
            try:
                if not log.closed:
                    try:
                        log.flush(); os.fsync(log.fileno())
                    finally:
                        log.close()
                if not Path(exitpath).exists():
                    save(exitpath, {'process_waited': True, 'exit_code': rc, 'identity': identity, 'log': str(logpath), 'log_sha256': sha(logpath)})
            except BaseException:
                # Evidence I/O failure must never skip another live child.
                errors[str(index) + '_evidence'] = traceback.format_exc()
        pending = remaining
        if pending:
            # Retain the owner's lock and logs until every wait has returned.
            time.sleep(.05)
    return errors


def run_phase(ctx, phase):
    cfg = ctx['cfg']; e = Path(cfg['execution_dir']) / phase; e.mkdir()
    owner = process(os.getpid()); children = []; count = 1 if phase == 'qualification' else 2
    try:
        for i in range(count):
            argv = [cfg['python'], '-B', str(Path(__file__).absolute()), '--registration', cfg['registration'], '--worker', phase, '--index', str(i)]
            logpath = e / ('worker_' + str(i) + '.log'); log = logpath.open('xb')
            try:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment(cfg), preexec_fn=lambda i=i: set_resources(cfg['resources']['affinities'][i]))
            except BaseException:
                log.close(); raise
            row = [child, log, logpath, {'pid': child.pid, 'argv': argv, 'identity_verified': False}, e / ('exit_' + str(i) + '.json')]
            children.append(row)
            save(e / ('spawn_' + str(i) + '.json'), row[3])
            identity = capture(child, argv); row[3] = identity
            save(e / ('admission_' + str(i) + '.json'), {'registration_sha256': ctx['regsha'], 'owner': owner, 'worker': identity})
        start = time.monotonic()
        while any(row[0].poll() is None for row in children):
            check_stop(cfg); require(time.monotonic() - start < cfg['max_seconds'], 'Phase wall-time exceeded')
            require(all(row[0].poll() in (None, 0) for row in children), 'Worker failed')
            time.sleep(.2)
        for row in children:
            require(wait_closed(*row) == 0, 'Worker exit must be zero')
        return merge(ctx, phase, [row[4] for row in children])
    except BaseException:
        detail = traceback.format_exc()
        try:
            request_stop(cfg, phase + ' failure')
        except BaseException:
            detail += '\nSTOP_WRITE_FAILED\n' + traceback.format_exc()
        # Parent retains lock and open log handles until every child is reaped.
        errors = drain_children(children)
        save(e / 'failure.json', {'traceback': detail, 'all_children_waited': True, 'drain_evidence_errors': errors})
        raise


def merge(ctx, phase, exitpaths):
    cfg = ctx['cfg']; ledger = Ledger(cfg['ledger'], ctx['regsha'], ctx['reg']['phase_caps']); before = ledger.quiescent()
    nworkers = 1 if phase == 'qualification' else 2; require(len(exitpaths) == nworkers, 'Exact normal workers required')
    expected = points(ctx['plan'], phase); actual = ledger.events(phase); seen = set(); outputs = {}; table = []
    for i, exitpath in enumerate(exitpaths):
        closed = read(exitpath); require(closed['process_waited'] is True and closed['exit_code'] == 0 and sha(closed['log']) == closed['log_sha256'], 'Missing normal wait/closed log')
        out = Path(cfg['out']) / phase / ('worker_' + str(i)); cp = out / 'completion.json'; done = read(cp)
        require(done['status'] == 'MAIN_RAW64_CPU_WORKER_COMPLETE_V1' and done['phase'] == phase and done['worker_index'] == i and done['registration_sha256'] == ctx['regsha'] and done['config_sha256'] == sha(cfg['_path']) and same(done['worker_identity'], closed['identity']), 'Wrong completion identity')
        verify(done['outputs']); owned = list(enumerate(expected))[i::nworkers]
        require(set(done['outputs']) == {str(out / ('point_' + str(j).zfill(6) + '.json')) for j, _ in owned}, 'Worker omitted or added a point')
        subtotal = 0
        for j, point in owned:
            value = read(out / ('point_' + str(j).zfill(6) + '.json')); evs = list(events(ctx['plan'], phase, point))
            require(value['point'] == point and value['point_index'] == j and [x['event'] for x in value['events']] == evs, 'Exact frozen point event order required')
            for row in value['events']:
                event = row['event']; a = actual.get(event['event_id']); require(event['event_id'] not in seen and a is not None, 'Missing/duplicate paid event'); seen.add(event['event_id'])
                require(a['status'] == 'COMPLETE' and a['kind'] == event['kind'] and a['phy_key'] == event['phy_key'] and a['request_sha'] == row['request_sha256'] and digest(json.loads(a['request'])) == row['request_sha256'], 'Point/request disagrees with paid ledger')
                require(json.loads(a['result']) == row['result'] and a['result_sha'] == row['result_sha256'] == digest(row['result']) and same(json.loads(a['worker']), done['worker_identity']), 'Point/result/owner disagrees with paid ledger')
                result_valid(event, row['result']); subtotal += 1
            table.append({'point': point, 'packet_calls': len(evs), **({k: sum(row['result'][k] for row in value['events']) for k in ('correct', 'rejected', 'undetected')} if point['kind'] == 'body' else {})})
        require(subtotal == done['packet_calls'], 'Completion charge count differs')
        outputs.update(done['outputs']); outputs.update({str(cp): sha(cp), str(exitpath): sha(exitpath), closed['log']: closed['log_sha256']})
    require(set(actual) == seen and len(seen) == ctx['reg']['phase_caps'][phase], 'Exact registered packet grid/cap must complete')
    require(ledger.quiescent() == before, 'Ledger changed during merge')
    done = {'status': 'MAIN_RAW64_CPU_PHASE_COMPLETE_V1', 'phase': phase, 'registration_sha256': ctx['regsha'], 'packet_calls': len(seen), 'budget': before, 'points': table, 'outputs': outputs, 'workers_waited': nworkers, 'GPU_used': False, 'quality_ranked': False, 'source_images_read': False}
    target = Path(cfg['out']) / phase / 'completion.json'; save(target, done); return target


def run(registration):
    ctx = load(registration); cfg = ctx['cfg']; require(sys.platform.startswith('linux'), 'Linux actual owner required')
    require(str(Path(sys.executable).absolute()) == cfg['python'], 'Registered interpreter path required')
    check_stop(cfg); set_resources(sum(cfg['resources']['affinities'], []))
    with owner_lock(cfg):
        e = Path(cfg['execution_dir']); save(e / 'attempt.json', {'identity': process(os.getpid()), 'registration_sha256': ctx['regsha']})
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        def interrupted(signum, frame):
            # Further signals must not interrupt the locked child-drain path.
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            raise InterruptedError('Owner received signal ' + str(signum))
        for s in previous:
            signal.signal(s, interrupted)
        try:
            Ledger.create(cfg['ledger'], ctx['regsha'], ctx['reg']['phase_caps']); Path(cfg['out']).mkdir(parents=True, exist_ok=False)
            paths = [run_phase(ctx, 'qualification'), run_phase(ctx, 'proxy')]
            budget = Ledger(cfg['ledger'], ctx['regsha'], ctx['reg']['phase_caps']).quiescent()
            done = {'status': 'MAIN_RAW64_QUALIFICATION_PROXY_NORMALLY_COMPLETE_V1', 'registration_sha256': ctx['regsha'], 'owner_identity': process(os.getpid()), 'outputs': {str(p): sha(p) for p in paths}, 'budget': budget, 'automatic_successor_started': False, 'calibration_implemented': False, 'GPU_used': False, 'source_images_read': False}
            save(e / 'completion.json', done); return done
        except BaseException:
            detail = traceback.format_exc()
            try:
                request_stop(cfg, 'CPU owner failure')
            except BaseException:
                detail += '\nSTOP_WRITE_FAILED\n' + traceback.format_exc()
            save(e / 'failure.json', {'traceback': detail, 'all_children_waited': True}); raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)


def launch(registration):
    ctx = load(registration); cfg = ctx['cfg']; e = Path(cfg['execution_dir']); check_stop(cfg)
    save(e / 'launch_claim.json', {'registration_sha256': ctx['regsha'], 'time': time.time()})
    argv = [cfg['python'], '-B', str(Path(__file__).absolute()), '--registration', cfg['registration'], '--run']
    # Root's durable outer supervisor must wait this owner and seal its log.
    # This command intentionally runs in the foreground so an exit cannot be lost.
    return argv


def main():
    p = argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--build-plan'); p.add_argument('--plan-out'); p.add_argument('--build-request'); p.add_argument('--request-out'); p.add_argument('--registration'); p.add_argument('--run', action='store_true'); p.add_argument('--launch-command', action='store_true'); p.add_argument('--worker', choices=('qualification', 'proxy')); p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.build_request:
        require(a.request_out and not any((a.request, a.build_plan, a.registration, a.run, a.launch_command, a.worker)), 'Request preparation is one metadata action'); print(canonical(build_request(a.build_request, a.request_out)), flush=True)
    elif a.build_plan:
        require(a.plan_out and not any((a.request, a.registration, a.run, a.launch_command, a.worker)), 'Plan preparation is one explicit metadata action'); print(canonical(build_plan(a.build_plan, a.plan_out)), flush=True)
    elif a.request:
        require(not any((a.registration, a.run, a.launch_command, a.worker)), 'Register is one explicit action'); print(register(a.request), flush=True)
    elif a.launch_command:
        print(canonical(launch(a.registration)), flush=True)
    elif a.run:
        print(run(a.registration)['status'], flush=True)
    else:
        require(a.registration and a.worker and a.index is not None, 'Explicit registered worker required'); worker(a.registration, a.worker, a.index)


if __name__ == '__main__':
    main()
