"""H initial_true200 payload core; no CLI, scheduling, GPU, or model loading.

This module requires the frozen h64_* modules on the import path. The caller
must seal ENGINEERING_CHOICES and all input/source files before real execution.
Returned payloads/decoded bits are private receiver inputs, not public reports.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from h64_catalog import PROTOCOL, SIZES, RATES, digest, require, token_count
import h64_phy as phy

PHASE = 'initial_true200'
SESSION = 'H_INITIAL_TRUE200_V1'
ARMS = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
SNRS = (13, 19)
SEEDS = (6101, 6102, 6103)
ENGINEERING_CHOICES = dict(
    schema='H_INITIAL_TRUE200_CPU_CHOICES_V1', phase=PHASE,
    noise_identity=[PROTOCOL, PHASE, 'source_id', 'integer_snr_db', 'integer_noise_seed'],
    json='ensure_ascii=True,sort_keys=True,separators=(comma,colon),allow_nan=False; UTF8',
    seed='SHA256 first16 bytes interpreted little-endian unsigned integer',
    generator='numpy.random.Generator(numpy.random.PCG64(seed))',
    noise='one standard_normal((1024,2)) float64 draw; no candidate/arm/profile fields',
    received='(float64 transmitted_wave + 10**(-snr_db/20)*float64 noise).astype(float32)',
    source_order='exact frozen shortlist source_ids; 200 entries',
    candidate_slot='frozen whole_candidates.slot, global0..15; source-independent',
    counter='((slot*200+source_index)*3+SEEDS.index(noise_seed))',
    scrambling_session=SESSION, scrambling_group=0,
    event='H:initial_true200:<candidate_id>:snr<SNR>:src<index4digits>:seed<seed>',
    codebook='all admitted frozen catalogue profiles, original paid profile_id unchanged',
    awgn_shared='same underlying Gaussian, not same received waveform',
    cpu_boundary='frame CRC/L/padding parsing; raw bits to tokens; arithmetic source decoding deferred',
    ledger='existing H ledger; every actual header/body callback uses initial_true200',
    source_cache='read only m6_bits..m9_bits from bound source200 NPZ; no image arrays loaded',
    execution='no real decoder call until a new contract binds these choices and this module')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def split_raw_tokens(flat):
    a = np.asarray(flat)
    require(a.shape == (sum(s*s for s in SIZES),) and np.issubdtype(a.dtype, np.integer)
            and np.all((a >= 0) & (a < 4096)), 'Invalid S1 source token vector')
    offsets = np.cumsum([0] + [s*s for s in SIZES])
    return [a[offsets[i]:offsets[i+1]].astype(np.int64, copy=True) for i in range(len(SIZES))]


def checked_file(path, expected):
    require(sha_string(expected) and file_sha(path) == expected, 'Bound source asset changed')


def load_source_assets(source_checkpoint, s1_checkpoint, *, source_checkpoint_sha256,
                       s1_checkpoint_sha256, source_id, source_index, source_registration_sha256):
    """Load only TX source data. Checkpoint hashes must come from sealed inputs.

    The completion/registration gate is the caller's responsibility. This local
    loader additionally verifies each referenced archive against its checkpoint.
    It never reads images, pixels, models, receiver CDFs, or development data.
    """
    require(type(source_index) is int and 0 <= source_index < 200, 'Source index outside hash200')
    checked_file(source_checkpoint, source_checkpoint_sha256)
    checked_file(s1_checkpoint, s1_checkpoint_sha256)
    cp, old = read_json(source_checkpoint), read_json(s1_checkpoint)
    require(cp['source_id'] == old['source_id'] == source_id, 'Source identity mismatch')
    require(cp['source_index'] == source_index, 'Source200 index changed')
    if 'source_index' in old:
        require(old['source_index'] == source_index, 'S1 source index changed')
    require(cp['registration_sha256'] == source_registration_sha256
            and cp.get('independent_roundtrip') is True, 'Source roundtrip not independently verified')
    archives = list(cp['outputs'])
    require(len(archives) == 1 and Path(archives[0]).suffix == '.npz', 'Expected one bound source200 archive')
    archive = archives[0]
    checked_file(archive, cp['outputs'][archive])
    raw_archive = old['archive']
    require(raw_archive in old['outputs'], 'S1 archive absent from original output bindings')
    checked_file(raw_archive, old['outputs'][raw_archive])
    lengths = {r['m']: r for r in cp['lengths']}
    require(set(lengths) == {6, 7, 8, 9} and len(cp['lengths']) == 4, 'Missing whole-prefix source lengths')
    with np.load(archive, allow_pickle=False) as z:
        streams = {m: phy.binary(z[f'm{m}_bits']) for m in (6, 7, 8, 9)}
    with np.load(raw_archive, allow_pickle=False) as z:
        scales = split_raw_tokens(z['tokens'])
    for m, bits in streams.items():
        r = lengths[m]
        require(len(bits) >= 2 and len(bits) == r['arithmetic_bits']
                and r['raw_bits'] == 12*token_count(m) and r['zero_extension_reads'] == 30,
                'Source stream lengths/terminal lookahead changed')
    return dict(source_id=source_id, source_index=source_index, scales=scales,
                arithmetic_bits=streams, source_bindings={str(source_checkpoint): source_checkpoint_sha256,
                    str(s1_checkpoint): s1_checkpoint_sha256, archive: cp['outputs'][archive],
                    raw_archive: old['outputs'][raw_archive]})


def validate_candidate(candidate):
    require(candidate['arm'] in ARMS, 'Only four registered whole arms enter initial_true200')
    require(type(candidate['target_m']) is int and candidate['target_m'] in (6, 7, 8, 9)
            and candidate.get('K', 0) == 0, 'Only whole m6..m9 candidates are registered here')
    require(type(candidate['snr_db']) is int and candidate['snr_db'] in SNRS, 'Only main SNRs')
    q = 4 if candidate['arm'].startswith('H16-') else 6
    require(candidate['q'] == q and candidate['nominal_rate'] in RATES, 'Arm/resource mismatch')
    require(isinstance(candidate['candidate_id'], str) and candidate['candidate_id'], 'Missing frozen candidate ID')
    require(type(candidate['slot']) is int and 0 <= candidate['slot'] < 16, 'Invalid public candidate slot')
    return q


def public_codebook(catalogue):
    require(catalogue['protocol'] == PROTOCOL and catalogue['header_id_bits'] == 12,
            'Different public catalogue protocol')
    profiles = catalogue['profiles']
    require(0 < len(profiles) == catalogue['profile_count'] <= 4096, 'Invalid public catalogue size')
    book = {}
    for p in profiles:
        pid = p['profile_id']
        require(type(pid) is int and 0 <= pid < 4096 and str(pid) not in book, 'Duplicate/invalid public ID')
        require(p['admission'] == 'ADMITTED' and p['n'] == 956*p['q']
                and p['k']-29 == p['source_capacity'] and p.get('layout_id'), 'Unqualified public profile')
        book[str(pid)] = dict(p)
    return book


def select_payload(candidate, catalogue, scales, arithmetic_bits):
    """TX-only finite fallback. No quality, receiver observation, or labels used."""
    q = validate_candidate(candidate)
    book = public_codebook(catalogue)
    matches = [b for b in catalogue['buckets'] if b['q'] == q and b['nominal_rate'] == candidate['nominal_rate']]
    require(len(matches) == 1 and matches[0]['admission'] == 'ADMITTED', 'Requested bucket is unavailable')
    capacity = matches[0]['source_capacity']
    require(capacity >= 12*token_count(6), 'Registered m6 raw fallback does not fit')
    attempts = []
    for m in range(candidate['target_m'], 5, -1):
        raw = phy.raw_payload(scales, m)
        arithmetic = None
        if candidate['arm'].endswith('-A'):
            require(m in arithmetic_bits, 'Missing frozen arithmetic stream; do not silently use raw')
            arithmetic = phy.binary(arithmetic_bits[m])
            require(len(arithmetic) >= 2, 'Invalid frozen arithmetic stream')
        use_arithmetic = arithmetic is not None and len(arithmetic) < len(raw)
        mode, payload = ('arithmetic', arithmetic) if use_arithmetic else ('raw', raw)
        fits = len(payload) <= capacity
        attempts.append(dict(m=m, raw_bits=len(raw), arithmetic_bits=None if arithmetic is None else len(arithmetic),
                             selected_mode=mode, selected_bits=len(payload), capacity=capacity, fits=fits))
        if not fits:
            continue
        profiles = [p for p in book.values() if p['q'] == q and p['nominal_rate'] == candidate['nominal_rate']
                    and p['m'] == m and p['K'] == 0 and p['mode'] == mode]
        require(len(profiles) == 1, 'Actual chosen state/mode absent from frozen paid public catalogue')
        return dict(profile=profiles[0], payload=payload.copy(), attempts=attempts,
                    target_m=candidate['target_m'], actual_m=m, fell_back=m != candidate['target_m'])
    raise ValueError('No whole prefix fits; no uncharged alternative is permitted')


def standard_noise(source_id, snr_db, noise_seed):
    require(isinstance(source_id, str) and source_id, 'Missing source identity for paired noise')
    require(type(snr_db) is int and snr_db in SNRS, 'Unregistered integer SNR')
    require(type(noise_seed) is int and noise_seed in SEEDS, 'Unregistered initial calibration noise seed')
    identity = [PROTOCOL, PHASE, source_id, snr_db, noise_seed]
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf8')
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1024, 2))


def frame_counter(slot, source_index, noise_seed):
    require(type(slot) is int and 0 <= slot < 16 and type(source_index) is int and 0 <= source_index < 200,
            'Public slot/source index outside fixed schedule')
    require(type(noise_seed) is int and noise_seed in SEEDS, 'Noise slot outside public schedule')
    return (slot*200+source_index)*3+SEEDS.index(noise_seed)


def validate_contract(protocol, catalogue, shortlist, contract, candidate, source_id, source_index):
    """An explicit seal avoids turning a draft core into an unregistered run."""
    require(contract.get('status') == 'H_PAYLOAD_CPU_ENGINEERING_SEALED', 'Engineering choices are not sealed')
    require(sha_string(contract.get('execution_registration_sha256')), 'Execution registration SHA required')
    for key, value in [('protocol_canonical_sha256', protocol), ('catalogue_canonical_sha256', catalogue),
                       ('shortlist_canonical_sha256', shortlist), ('engineering_choices_sha256', ENGINEERING_CHOICES)]:
        require(contract.get(key) == digest(value), 'Sealed input/engineering choices changed: '+key)
    require(contract.get('core_source_sha256') == file_sha(__file__), 'Core source absent/changed in seal')
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['N'] == 1024
            and protocol['header_symbols'] == 68 and protocol['body_symbols'] == 956
            and protocol['Es'] == 2 and protocol['channel'] == 'AWGN', 'Different frozen H wire/channel')
    require(shortlist['status'] == 'H_EXPECTED_PSNR_SHORTLIST_FROZEN'
            and shortlist['ready_for_real_calibration'] is True, 'Shortlist not ready/frozen')
    ids = shortlist['source_ids']
    require(len(ids) == len(set(ids)) == 200 and type(source_index) is int
            and 0 <= source_index < 200 and ids[source_index] == source_id, 'Hash200 source order changed')
    seen, identities, counts, matches = set(), set(), {}, []
    for row in shortlist['whole_candidates']:
        validate_candidate(row)
        require(row['slot'] not in seen, 'Duplicate frozen candidate slot')
        seen.add(row['slot'])
        identity = (row['candidate_id'], row['snr_db'])
        require(identity not in identities, 'Duplicate candidate identity at one SNR')
        identities.add(identity)
        key = (row['arm'], row['snr_db'])
        counts[key] = counts.get(key, 0)+1
        require(counts[key] <= 2, 'More than two frozen candidates per arm/SNR')
        if row['slot'] == candidate['slot']:
            matches.append(row)
    require(len(matches) == 1 and digest(matches[0]) == digest(candidate), 'Candidate is not the exact frozen row')


def run_frame(*, protocol, catalogue, shortlist, contract, candidate, source_id, source_index,
              noise_seed, scales, arithmetic_bits, backend, header, ledger):
    """Produce one paid actual AWGN trace, without any neural inference.

    header/LDPC implementations and the existing ledger are injected. Only
    receive_frame invokes metered decoders. No exception is retried or refunded.
    Arithmetic acceptance here means wire-parser acceptance, not canonical source
    acceptance: subsequent RX must independently call h64_source.decode_prefix.
    """
    validate_contract(protocol, catalogue, shortlist, contract, candidate, source_id, source_index)
    require(ledger.branch == 'H' and 0 < ledger.limits.get(PHASE, 0) <= 20000,
            'Existing H initial_true200 ledger quota required')
    require(getattr(backend, 'device', 'cpu') == 'cpu', 'Physical payload backend must use CPU')
    noise = standard_noise(source_id, candidate['snr_db'], noise_seed)
    counter = frame_counter(candidate['slot'], source_index, noise_seed)
    tx = select_payload(candidate, catalogue, scales, arithmetic_bits)
    book = public_codebook(catalogue)
    profile = tx['profile']
    header_wave = np.asarray(header.transmit(profile['profile_id']))
    require(header_wave.shape == (68, 2) and np.isfinite(header_wave).all(), 'Invalid actual paid header waveform')
    body_wave, body_meta = phy.transmit_body(backend, tx['payload'], profile, counter, SESSION)
    transmitted = np.concatenate((header_wave, body_wave)).astype(np.float64)
    require(transmitted.shape == (1024, 2) and np.isfinite(transmitted).all(), 'Wrong transmitted frame shape')
    snr = candidate['snr_db']
    received = (transmitted+noise*10**(-snr/20)).astype(np.float32)
    event = f'H:{PHASE}:{candidate["candidate_id"]}:snr{snr}:src{source_index:04d}:seed{noise_seed}'
    # The receiver sees the actual observation, public frame number and public
    # catalogue only. Neither TX profile nor source tokens are an RX argument.
    rx = phy.receive_frame(backend, header, received, snr, counter, book, ledger, PHASE, event, SESSION)
    parsed = bool(rx.get('body') and rx['body'].get('parser_accepted'))
    rx_profile = book[str(rx['header']['profile_id'])] if rx['header']['header_ok'] else None
    rx_tokens = None
    if not parsed:
        source_status, gray = 'WIRE_REJECT_GRAY', True
    elif rx_profile['mode'] == 'raw':
        rx_tokens = phy.raw_tokens(rx['body']['payload'], rx_profile).tolist()
        source_status, gray = 'RAW_SOURCE_DECODED', False
    else:
        source_status, gray = 'ARITHMETIC_CANONICAL_RX_REQUIRED', None
    correct_wire = bool(parsed and rx_profile['profile_key'] == profile['profile_key']
                        and rx['body']['payload'] == tx['payload'].tolist())
    return dict(status='H_INITIAL_CPU_FRAME_TRACE', stage=PHASE, source_id=source_id, source_index=source_index,
        candidate_id=candidate['candidate_id'], arm=candidate['arm'], snr_db=snr, noise_seed=noise_seed,
        candidate_slot=candidate['slot'], public_frame_counter=counter, scrambling_session=SESSION, event_id=event,
        execution_registration_sha256=contract['execution_registration_sha256'],
        engineering_choices_sha256=digest(ENGINEERING_CHOICES),
        noise_sha256=phy.array_sha(noise), transmitted_sha256=phy.array_sha(transmitted),
        received_sha256=phy.array_sha(received), total_symbols=1024,
        header_energy=float(np.square(transmitted[:68]).sum()), body_energy=float(np.square(transmitted[68:]).sum()),
        total_energy=float(np.square(transmitted).sum()), frame_normalized=False,
        tx=dict(profile_id=profile['profile_id'], profile_key=profile['profile_key'], target_m=tx['target_m'],
                actual_m=tx['actual_m'], mode=profile['mode'], fell_back=tx['fell_back'], attempts=tx['attempts'],
                payload_sha256=phy.array_sha(tx['payload']), phy=body_meta),
        rx=rx, rx_profile=rx_profile, rx_source_status=source_status, gray=gray, received_raw_tokens=rx_tokens,
        evaluation_only=dict(header_correct=bool(rx['header']['header_ok'] and rx['header']['profile_id'] == profile['profile_id']),
                             parsed_wire_matches_transmission=correct_wire,
                             accepted_wire_mismatch=bool(parsed and not correct_wire)),
        logical_packet_events=1+int(rx.get('body') is not None),
        packet_event_ids=[event+':header']+([event+':body'] if rx.get('body') is not None else []),
        cost_scope='Logical event references only; actual charged calls come from the existing ledger, not these fields. Reuse adds no decoder call.',
        gpu_used=False, arithmetic_source_decode_run=False, neural_metrics_run=False)
