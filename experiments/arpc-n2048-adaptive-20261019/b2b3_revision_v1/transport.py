"""Actual two-fragment TX/RX and post-exit truth audit, in separate processes."""
from common import *
from plan import CAPS, public_job, validate_public_job
import argparse
import collections
import hashlib
import math


def fsync_binary(path, writer):
    with Path(path).open('xb') as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())


def runtime_identity(rt):
    t = rt.torch
    result = dict(torch=t.__version__, python=sys.version, device='cpu',
        cpu_intra_threads=t.get_num_threads(), cpu_interop_threads=t.get_num_interop_threads(),
        matmul_tf32=t.backends.cuda.matmul.allow_tf32, cudnn_tf32=t.backends.cudnn.allow_tf32,
        deterministic_algorithms=t.are_deterministic_algorithms_enabled(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cpu_affinity=sorted(os.sched_getaffinity(0)), backend_identity=rt.catalogue['backend_identity'])
    assert result['cpu_intra_threads'] == 2 and result['cpu_interop_threads'] == 1
    assert not result['matmul_tf32'] and not result['cudnn_tf32']
    assert result['deterministic_algorithms'] is True and result['cuda_visible_devices'] == ''
    return result


class RecordedLedger:
    """Every completed physical decoder result persists before ledger COMPLETE."""
    def __init__(self, ledger, directory, allowed_keys):
        self.raw = ledger
        self.directory = Path(directory)
        self.allowed = set(allowed_keys)

    def call(self, kind, key, fn):
        assert kind in ('header', 'body') and key in self.allowed
        target = self.directory / (key + '_' + kind + '.json')
        assert not target.exists()
        def saved():
            value = fn()
            put(target, value)
            return value
        return self.raw.call(kind, key, saved)


def pair_result(j, received, observed_sha256, wire_module, protocol):
    """Only actual observations/paid metadata enter this pure assembly step."""
    validate_public_job(j)
    result = dict(key=j['key'], received=received, observed_sha256=observed_sha256,
        source_truth_used=False, status='PAIR_REJECT', source_input_key=None)
    data = None
    try:
        if any(r['status'] != 'FRAGMENT_PARSED' for r in received):
            reasons = [r['status'] for r in received]
            raise protocol.Reject('fragment statuses: ' + ','.join(reasons))
        obj = protocol.reassemble([(bytes.fromhex(r['body']['payload_hex']), r['profile']) for r in received])
        data = obj.pop('data')
        if obj['codec'] == 1:
            try:
                wire_module.unpack(data, obj['endpoint'])
            except wire_module.SourceParseError as ex:
                raise protocol.Reject('native container: ' + str(ex)) from ex
        key = protocol.input_key(obj['codec'], obj['endpoint'], obj['bit_length'], data)
        result.update(status='PAIR_ASSEMBLED', source_input_key=key, actual_metadata=obj,
                      source_object_sha256=hashlib.sha256(data).hexdigest())
    except protocol.Reject as ex:
        result['reason'] = str(ex)
        data = None
    return result, data


def load_object(asset, protocol, np):
    path = verify(asset['wire'])
    if asset['codec'] == 0:
        with np.load(path, allow_pickle=False) as z:
            bits = z[asset['array_key']].copy()
        assert bits.dtype == np.uint8 and bits.ndim == 1 and np.isin(bits, (0, 1)).all()
        assert len(bits) == asset['bit_length']
        assert hashlib.sha256(bits.tobytes()).hexdigest() == asset['bits_sha256']
        data = protocol.pack_bits(bits)
    else:
        assert asset['codec'] == 1
        data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == asset['object_sha256']
    protocol.validate_object(data, asset['bit_length'], asset['codec'], asset['endpoint'])
    return data


def tx(phase, expected):
    r = registration(phase, expected)
    jobs = read(verify(r['private_jobs']))
    assets = read(verify(r['private_assets']))
    assert len(jobs) == CAPS[phase]['logical_objects']
    out = L / phase / 'tx'
    out.mkdir(); cache(out)
    incoming = L / phase / 'received'; incoming.mkdir()
    records_dir = out / 'logical_records'; records_dir.mkdir()
    tx_checkpoints = out / 'transmit_checkpoints'; tx_checkpoints.mkdir()
    import numpy as np
    import pair_protocol as p
    from pair_link import Runtime, noise, ep
    rt = Runtime()
    assert rt.catalogue['backend_identity'] == r['backend_identity']
    assert rt.catalogue == read(verify(r['backend_catalogue']))
    put(out/'runtime.json', runtime_identity(rt))
    ledger = Ledger(out / 'calls.sqlite', dict(transmit=CAPS[phase]['subframes']))
    put(incoming / 'catalogue.json', rt.catalogue)
    pub, records = [], []
    object_cache = {}
    for position, j in enumerate(jobs):
        ak = j['asset_key']
        if ak is None:
            assert j['source_no_fit'] and j['actual_endpoint'] is None
            q = p.profile(j['codec'], None, j['layout_id'], None)
            fragments = [(b'', q), (b'', q)]
            length = 0; data = None; source_key = None
        else:
            assert not j['source_no_fit']
            asset = assets[ak]
            assert (asset['source_id'], asset['codec'], asset['endpoint']) == (j['source_id'], j['codec'], j['actual_endpoint'])
            if ak not in object_cache:
                object_cache[ak] = load_object(asset, p, np)
            data = object_cache[ak]; length = asset['bit_length']
            fragments = p.fragment(data, length, j['codec'], asset['endpoint'], j['pair_id'], j['layout_id'])
            source_key = p.input_key(j['codec'], asset['endpoint'], length, data)
        frames, transmissions = [], []
        for i, (payload, profile) in enumerate(fragments):
            key = j['key'] + '_' + str(i)
            def sent(payload=payload, profile=profile, i=i, key=key):
                frame, receipt = rt.transmit(payload, profile['profile_id'], 2 * j['pair_id'] + i)
                assert frame.shape == (1024, 2) and np.isfinite(frame).all()
                assert receipt['N'] == 1024 and receipt['Es'] == 2
                assert math.isfinite(receipt['energy']) and receipt['energy'] > 0
                put(tx_checkpoints / (key + '.json'), receipt)
                return frame, receipt
            frame, receipt = ledger.call('transmit', key, sent)
            frames.append(frame); transmissions.append(receipt)
        frame = np.concatenate(frames)
        z = noise(j['source_id'], j['noise_seed'])
        assert z.shape == frame.shape == (2048, 2)
        observed = frame + z * 10 ** (-j['snr_db'] / 20)
        path = incoming / (j['key'] + '.npy')
        fsync_binary(path, lambda f: np.save(f, observed, allow_pickle=False))
        public = public_job(j['key'], j['snr_db'], [2*j['pair_id'], 2*j['pair_id']+1], pin(path))
        pub.append(public)
        record = dict(**j, bit_length=length,
            source_object_sha256=None if data is None else hashlib.sha256(data).hexdigest(),
            source_input_key=source_key, transmissions=transmissions,
            noise_sha256=ep.array_sha(z), observed_sha256=ep.array_sha(observed),
            actual_total_energy=sum(t['energy'] for t in transmissions),
            N=2048, nominal_Es=2, observation=public['observation'])
        rp = records_dir / (j['key'] + '.json'); put(rp, record); records.append(pin(rp))
        if (position + 1) % 300 == 0:
            print(json.dumps(dict(phase=phase, role='tx', logical=position+1, total=len(jobs))), flush=True)
    summary = ledger.summary()
    assert summary['unresolved'] == 0
    assert summary['counts'] == [dict(kind='transmit', status='COMPLETE', count=CAPS[phase]['subframes'])]
    put(incoming / 'jobs.json', pub)
    put(out / 'records.json', records)
    # The receiver gets only source/static implementation pins, no model routes,
    # population map, TX jobs, source objects, or desired codec/endpoint/identity.
    code = [d for d in r['code'] if Path(d['path']).is_relative_to(R/'experiments') or Path(d['path']).is_relative_to(R/'src')]
    public = dict(schema='N2048_B2B3_RX_PUBLIC_V1', phase=phase, caps=CAPS[phase], code=code,
        catalogue=pin(incoming/'catalogue.json'), jobs=pin(incoming/'jobs.json'),
        registration_sha256=expected, backend_identity=r['backend_identity'])
    put(incoming / 'public.json', public)
    put(out / 'completion.json', dict(status='COMPLETE', logical=len(jobs),
        subframes=2*len(jobs), ledger=summary, records=pin(out/'records.json'),
        public=pin(incoming/'public.json'), source_probability_calls=0, tokenizer_calls=0))


def rx(public_path, expected, size):
    cfg = public_configuration(public_path, expected, size)
    phase = cfg['phase']; out = L / phase / 'rx'
    out.mkdir(); cache(out)
    incoming = L / phase / 'received'
    for d in cfg['code']:
        verify(d)
    for k in ('catalogue', 'jobs'):
        verify(cfg[k])
    import numpy as np
    import pair_protocol as p
    # Resolve pure parsers/imports before any decoder invocation. In particular,
    # permit the exact old protocol directory including optional bytecode lookup.
    p.legacy()
    sys.path.insert(0, str(OLD_E)); sys.path.insert(0, str(OLD_E/'upstream'))
    import wire
    from pair_link import Runtime, ep, legacy
    from a1_native import install_rx_boundary
    allow = [OLD_ADAPTIVE_E, OLD_E, p.OLD_PATH.parent, legacy.OLD,
        R/'experiments/wcl-evidence-closure-20261009/scripts',
        R/'experiments/prior-aware-grouped-mcs-v1', R/'experiments/main-raw64-20261007',
        R/'src', R/'outputs/build', R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json',
        RT/'envs/frozen_ldpc_py311', RT/'envs/frozen_python_base_v1/lib/python3.11',
        RT/'envs/frozen_python_base_v1/lib/python311.zip',
        RT/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',
        incoming, out]
    put(out/'boundary.json', dict(allowed=[str(x) for x in allow],
        source_truth_inputs=[], registration_read=False, actual_paid_metadata_only=True))
    blocked = install_rx_boundary(allow)
    rt = Runtime()
    assert rt.catalogue == read(cfg['catalogue']['path'])
    assert rt.catalogue['backend_identity'] == cfg['backend_identity']
    put(out/'runtime.json', runtime_identity(rt))
    jobs = read(cfg['jobs']['path'])
    assert len(jobs) == CAPS[phase]['logical_objects']
    assert len({j['key'] for j in jobs}) == len(jobs)
    for j in jobs:
        validate_public_job(j)
        require_contained(j['observation']['path'], incoming)
        assert Path(j['observation']['path']) == incoming / (j['key'] + '.npy')
    raw = Ledger(out/'calls.sqlite', dict(header=CAPS[phase]['header'], body=CAPS[phase]['body']))
    checkpoints = out/'physical_checkpoints'; checkpoints.mkdir()
    ledger = RecordedLedger(raw, checkpoints, [j['key']+'_'+str(i) for j in jobs for i in (0, 1)])
    wires = out/'source_inputs'; wires.mkdir()
    outputs_dir = out/'logical_records'; outputs_dir.mkdir()
    outputs, sourcejobs = [], {}
    for position, j in enumerate(jobs):
        verify(j['observation'])
        y = np.load(j['observation']['path'], allow_pickle=False)
        assert y.dtype == np.float64 and y.shape == (2048, 2) and np.isfinite(y).all()
        rs = [rt.receive(y[i*1024:(i+1)*1024], float(j['snr_db']), j['counters'][i], ledger,
                         j['key']+'_'+str(i)) for i in (0, 1)]
        result, data = pair_result(j, rs, ep.array_sha(y), wire, p)
        if data is not None:
            obj = result['actual_metadata']; key = result['source_input_key']
            if key not in sourcejobs:
                ip = wires/(key+'.bin'); fsync_binary(ip, lambda f: f.write(data))
                sourcejobs[key] = dict(key=key, codec=obj['codec'], endpoint=obj['endpoint'],
                    bit_length=obj['bit_length'], input=pin(ip), generation_seed=0)
            else:
                assert sourcejobs[key]['input']['sha256'] == hashlib.sha256(data).hexdigest()
        rp = outputs_dir/(j['key']+'.json'); put(rp, result); outputs.append(pin(rp))
        if (position + 1) % 300 == 0:
            print(json.dumps(dict(phase=phase, role='rx', logical=position+1, total=len(jobs),
                accepted_unique=len(sourcejobs))), flush=True)
    put(out/'blocked_reads.json', blocked)
    summary = raw.summary()
    assert not blocked and summary['unresolved'] == 0
    counts = {x['kind']: x['count'] for x in summary['counts']}
    assert counts['header'] == CAPS[phase]['header'] and counts.get('body', 0) <= CAPS[phase]['body']
    put(out/'source_jobs.json', list(sourcejobs.values()))
    put(out/'outputs.json', outputs)
    put(out/'completion.json', dict(status='COMPLETE', logical=len(jobs),
        subframes=2*len(jobs), header_decodes=counts['header'], body_decodes=counts.get('body', 0),
        decoder_attempts=sum(counts.values()), ledger=summary, blocked_reads=blocked,
        outputs=pin(out/'outputs.json'), source_jobs=pin(out/'source_jobs.json'),
        source_image_reads=0, source_token_reads=0, TX_object_reads=0, model_calls=0))


def audit(phase, expected):
    r = registration(phase, expected)
    closed_wait(L/'controls'/(phase+'_tx')/'actual_wait.json')
    closed_wait(L/'controls'/(phase+'_rx')/'actual_wait.json')
    root = L/phase
    txc, rxc = read(root/'tx/completion.json'), read(root/'rx/completion.json')
    assert txc['status'] == rxc['status'] == 'COMPLETE'
    assert txc['logical'] == rxc['logical'] == CAPS[phase]['logical_objects']
    assert txc['ledger']['unresolved'] == rxc['ledger']['unresolved'] == 0
    assert not rxc['blocked_reads'] and rxc['decoder_attempts'] <= CAPS[phase]['decoder_attempts']
    records = read(verify(txc['records'])); outputs = read(verify(rxc['outputs']))
    jobs = read(verify(r['private_jobs']))
    assert len(records) == len(outputs) == len(jobs)
    rows, physical_counts = [], collections.Counter()
    for j, td, rd in zip(jobs, records, outputs):
        t, x = read(verify(td)), read(verify(rd))
        assert j['key'] == t['key'] == x['key']
        assert t['observed_sha256'] == x['observed_sha256'] and x['source_truth_used'] is False
        for i, received in enumerate(x['received']):
            key = j['key']+'_'+str(i)
            assert read(root/'rx/physical_checkpoints'/(key+'_header.json')) == received['header']
            physical_counts['header'] += 1
            body_path = root/'rx/physical_checkpoints'/(key+'_body.json')
            if received['body'] is None:
                assert received['status'] == 'HEADER_REJECT' and not body_path.exists()
            else:
                assert read(body_path) == received['body']
                physical_counts['body'] += 1
            assert read(root/'tx/transmit_checkpoints'/(key+'.json')) == t['transmissions'][i]
        accepted = x['status'] == 'PAIR_ASSEMBLED'
        exact = accepted and x['source_input_key'] == t['source_input_key'] and x['source_object_sha256'] == t['source_object_sha256']
        actual = x.get('actual_metadata', {})
        row = dict(key=x['key'], phase=phase, policy_id=j['policy_id'], codec=j['codec'],
            snr_db=j['snr_db'], noise_seed=j['noise_seed'], source_id=j['source_id'],
            source_index=j['source_index'], layout_id=j['layout_id'], target=j['target'],
            sent_endpoint=j['actual_endpoint'], endpoint=j['actual_endpoint'], source_no_fit=j['source_no_fit'], object_bits=t['bit_length'],
            status=x['status'], accepted=accepted, exact_post_exit=exact, accepted_wrong=accepted and not exact,
            failure_reason=x.get('reason', ''), source_input_key=x['source_input_key'],
            actual_received_codec=actual.get('codec'), actual_received_endpoint=actual.get('endpoint'),
            header_statuses=[v['status'] if v['body'] is None else 'HEADER_OK' for v in x['received']],
            fragment_statuses=[v['status'] for v in x['received']],
            actual_total_energy=t['actual_total_energy'], N=2048, nominal_Es=2,
            tx_record=td, rx_record=rd)
        rows.append(row)
    assert physical_counts['header'] == rxc['header_decodes']
    assert physical_counts['body'] == rxc['body_decodes']
    sourcejobs = read(verify(rxc['source_jobs']))
    assert {s['key'] for s in sourcejobs} == {r['source_input_key'] for r in rows if r['accepted']}
    for j in sourcejobs:
        verify(j['input'])
    out = O/phase
    put(root/'logical_frames.json', rows)
    summary = []
    for c in r['cells']:
        rr = [x for x in rows if (x['codec'], x['snr_db'], x['policy_id']) == (c['codec'], c['snr_db'], c['policy_id'])]
        summary.append(dict(**c, logical=len(rr), accepted=sum(x['accepted'] for x in rr),
            correct=sum(x['exact_post_exit'] for x in rr), accepted_wrong=sum(x['accepted_wrong'] for x in rr),
            source_no_fit=sum(x['source_no_fit'] for x in rr),
            reasons=dict(collections.Counter(x['failure_reason'] for x in rr if not x['accepted']))))
    put(out/'transport_completion.json', dict(status='TRANSPORT_COMPLETE_NATIVE_AND_SCORE_SEPARATE',
        phase=phase, logical=len(rows), subframes=2*len(rows), transmitted_subframes=txc['subframes'],
        header_decodes=rxc['header_decodes'], body_decodes=rxc['body_decodes'],
        decoder_attempts=rxc['decoder_attempts'], cells=summary,
        logical_frames=pin(root/'logical_frames.json'), source_jobs=rxc['source_jobs'],
        unique_received_objects=len(sourcejobs), registration=pin(out/'registration.json'),
        TX_truth_only_post_RX_exit=True, native_recovery_calls=0, metric_calls=0,
        rejected_objects_remain_in_denominator=True, automatic_retry=False,
        actual_waits=[pin(L/'controls'/(phase+'_'+s)/'actual_wait.json') for s in ('tx','rx')]))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('role', choices=['tx','rx','audit'])
    a.add_argument('--phase', choices=['B2','B3'])
    a.add_argument('--expected-registration')
    a.add_argument('--public'); a.add_argument('--public-sha256'); a.add_argument('--public-bytes', type=int)
    x = a.parse_args()
    target = None
    try:
        if x.role == 'rx':
            assert x.phase is None and x.expected_registration is None
            cfg = public_configuration(x.public, x.public_sha256, x.public_bytes)
            target = L/cfg['phase']/'rx'
            rx(x.public, x.public_sha256, x.public_bytes)
        else:
            assert x.phase is not None and x.expected_registration is not None and x.public is None
            target = L/x.phase/x.role
            if x.role == 'audit':
                target.mkdir()
            globals()[x.role](x.phase, x.expected_registration)
    except BaseException:
        if target is not None:
            fail(target)
        raise
