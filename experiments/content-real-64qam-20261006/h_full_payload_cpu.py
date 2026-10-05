"""Prepared full-calibration H CPU wire core; no invocation on import.

The original codecs/PHY remain unchanged. Whole and partial populations share
Gaussian noise, but have separate immutable ledger phases. Arithmetic parsing
here is only wire parsing; canonical receiver inference remains a later stage.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from h64_catalog import PROTOCOL, SIZES, RATES, digest, require, token_count, normalize_state
import h64_phy as phy
import h_payload_cpu as initial

COUNT = 1000
SEEDS = (6101, 6102, 6103)
SNRS = (13, 19)
ARMS = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
PARTIAL_ARM = 'H64-RAW-COMPLETE-STATE-CONTROL'
PHASE_CAPS = {'whole_calibration': 48000, 'partial_calibration': 36000}
NOISE_STAGE = 'full_calibration'
SESSION = 'H_FULL_CALIBRATION_V1'
ENGINEERING_CHOICES = dict(
    schema='H_FULL_CALIBRATION_CPU_ENGINEERING_V1', source_count=COUNT,
    phases=PHASE_CAPS, noise_stage=NOISE_STAGE, seeds=list(SEEDS), snrs=list(SNRS),
    noise='SHA256 canonical ASCII JSON [H protocol,full_calibration,source_id,SNR,seed]; first16 bytes little-endian PCG64; float64 standard_normal((1024,2))',
    received='(float64 waveform + 10**(-SNR/20)*shared_standard_noise).astype(float32)',
    full_slots='Whole winners sorted by original initial slot -> 0..7; partial frozen references sorted by original slot -> 8..13',
    frame_counter='((full_slot*1000 + original_full_calibration_source_index)*3 + seed_index)',
    scrambling_session=SESSION, public_catalogue='All original admitted paid header IDs unchanged',
    initial_noise_reused=False, initial_physical_results_substituted=False,
    whole_fallback='Unchanged original h_payload_cpu.select_payload',
    partial_fallback='None: exact frozen raw raster m/K/profile, including whole K0 if shortlisted',
    receiver='Actual paid header determines parsing; no TX truth repairs; arithmetic canonical validation deferred',
    scheduling='Closed full1000 source-codec owner -> wholeCPU owner closed -> partialCPU owner closed -> separately registered GPU receiver; never overlap metered PHY with budget-unchanged GPU stages',
    source_prerequisite='All1000 independent frozen-source encodings or exact proven source200 reuse, never incomplete CPU assets')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def candidate_key(row):
    return row['candidate_id'], row['snr_db']


def schedule(selected, partial_reference, shortlist, catalogue, source_ids):
    """Validate frozen families, then derive only a public counter schedule.

    No ranking or quality comparison is performed. The source documents and
    completed selector receipt must also be byte-bound by the execution driver.
    """
    require(len(source_ids) == len(set(source_ids)) == COUNT
            and all(isinstance(s, str) and s for s in source_ids), 'Original1000 source IDs required')
    require(shortlist['status'] == 'H_EXPECTED_PSNR_SHORTLIST_FROZEN'
            and shortlist['ready_for_real_calibration'] is True, 'Final prescreen shortlist required')
    require(selected['status'] == 'H_INITIAL_TRUE200_SELECTION_COMPLETE'
            and selected['source_count'] == 200 and selected['selected_count'] == 8
            and selected['candidate_count'] == 16 and selected['measured_frames'] == 9600
            and selected['new_candidate_search'] is False and selected['development_used'] is False
            and selected['holdout_used'] is False and selected['full1000_calibration_complete'] is False,
            'Complete actual initial200 whole selection required')
    ids = shortlist['source_ids']
    require(len(ids) == len(set(ids)) == 200 and selected['source_ids'] == ids
            and set(ids) <= set(source_ids), 'Initial200 must map by exact ID into original1000')
    require(partial_reference['status'] == 'FROZEN_PRESCREEN_REFERENCE_ONLY_NOT_RESELECTED'
            and partial_reference['candidate_count'] == 6 and partial_reference['partial_reselected'] is False
            and partial_reference['actual_partial_calibration_complete'] is False
            and partial_reference['partial_candidates'] == shortlist['partial_candidates'],
            'Six unchanged prescreen references required; they are not whole winners')
    original = {candidate_key(r): r for r in shortlist['whole_candidates']}
    require(len(original) == 16, 'Original whole shortlist coverage differs')
    winners = selected['selected_candidates']
    require(len(winners) == 8 and {(r['arm'], r['snr_db']) for r in winners}
            == {(a, s) for a in ARMS for s in SNRS}, 'Exactly one winner for every arm/SNR required')
    seen = set()
    for row in winners:
        initial.validate_candidate(row)
        require(candidate_key(row) in original and row['slot'] not in seen, 'Winner identity/slot differs')
        seen.add(row['slot'])
        old = original[candidate_key(row)]
        # Selection adds summaries/rank fields, but cannot alter the wire policy.
        for key in ('candidate_id', 'arm', 'q', 'nominal_rate', 'target_m', 'K', 'slot', 'snr_db', 'policy_key'):
            require(row[key] == old[key], 'Selected whole policy changed: '+key)
        require(row['selection_rank'] == 1, 'Selected whole row is not an actual winner')
    book = initial.public_codebook(catalogue)
    parts = partial_reference['partial_candidates']
    require(len(parts) == 6 and {r['slot'] for r in parts} == set(range(6))
            and len({candidate_key(r) for r in parts}) == 6
            and all(sum(r['snr_db'] == s for r in parts) == 3 for s in SNRS), 'Partial six-slot/SNR coverage differs')
    for row in parts:
        require(row['arm'] == PARTIAL_ARM and row['q'] == 6 and row['snr_db'] in SNRS
                and row['nominal_rate'] in RATES and (row['m'], row['K']) == normalize_state(row['m'], row['K'])
                and row['target_m'] == row['m'], 'Noncanonical partial reference')
        p = book.get(str(row['profile_id']))
        require(p is not None and p['mode'] == 'raw' and p['q'] == 6, 'Partial profile absent from original paid catalogue')
        for key in ('m', 'K', 'nominal_rate', 'profile_key', 'source_capacity', 'layout_id'):
            require(row[key] == p[key], 'Partial reference changed profile: '+key)
        require(row['policy_key'] == p['profile_key'] and row['token_count'] == token_count(p['m'], p['K']),
                'Partial token accounting differs')
    result = []
    for phase, rows in (('whole_calibration', sorted(winners, key=lambda r: r['slot'])),
                        ('partial_calibration', sorted(parts, key=lambda r: r['slot']))):
        for row in rows:
            result.append(dict(full_slot=len(result), phase=phase, candidate=copy.deepcopy(row)))
    return result


def prepare(protocol, catalogue, selected, partial_reference, shortlist, source_ids, contract):
    require(contract.get('status') == 'H_FULL_PAYLOAD_CPU_ENGINEERING_SEALED'
            and initial.sha_string(contract.get('execution_registration_sha256')), 'Independent full-calibration seal required')
    for name, value in [('protocol', protocol), ('catalogue', catalogue), ('selected', selected),
                        ('partial_reference', partial_reference), ('shortlist', shortlist), ('source_ids', source_ids),
                        ('engineering_choices', ENGINEERING_CHOICES)]:
        require(contract.get(name+'_canonical_sha256') == digest(value), 'Sealed input changed: '+name)
    require(contract.get('core_source_sha256') == file_sha(__file__)
            and contract.get('initial_core_source_sha256') == file_sha(initial.__file__), 'Frozen/new core source differs')
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['N'] == 1024
            and protocol['header_symbols'] == 68 and protocol['body_symbols'] == 956
            and protocol['Es'] == 2 and protocol['channel'] == 'AWGN', 'Frozen H channel/wire differs')
    rows = schedule(selected, partial_reference, shortlist, catalogue, source_ids)
    require(contract.get('public_schedule_canonical_sha256') == digest(rows), 'Public fourteen-policy schedule differs')
    return dict(protocol=copy.deepcopy(protocol), catalogue=copy.deepcopy(catalogue),
                book=initial.public_codebook(catalogue), source_ids=tuple(source_ids), schedule=rows,
                rows_by_slot={r['full_slot']: r for r in rows}, contract=copy.deepcopy(contract))


def standard_noise(source_id, snr_db, noise_seed):
    require(isinstance(source_id, str) and source_id and type(snr_db) is int and snr_db in SNRS
            and type(noise_seed) is int and noise_seed in SEEDS, 'Unregistered full calibration noise identity')
    text = json.dumps([PROTOCOL, NOISE_STAGE, source_id, snr_db, noise_seed],
                      sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    seed = int.from_bytes(hashlib.sha256(text).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1024, 2))


def frame_counter(full_slot, source_index, noise_seed):
    require(type(full_slot) is int and 0 <= full_slot < 14 and type(source_index) is int
            and 0 <= source_index < COUNT and type(noise_seed) is int and noise_seed in SEEDS,
            'Public full-calibration counter outside registered schedule')
    return (full_slot*COUNT+source_index)*3+SEEDS.index(noise_seed)


def select_payload(ctx, entry, scales, arithmetic_bits):
    c = entry['candidate']
    if entry['phase'] == 'whole_calibration':
        return initial.select_payload(c, ctx['catalogue'], scales, arithmetic_bits)
    require(entry['phase'] == 'partial_calibration', 'No other ledger phase allowed')
    p = ctx['book'][str(c['profile_id'])]
    payload = phy.raw_payload(scales, p['m'], p['K'])
    require(len(payload) == 12*c['token_count'] <= p['source_capacity'], 'Partial exact resource accounting differs')
    return dict(profile=p, payload=payload, target_m=c['m'], actual_m=c['m'], fell_back=False,
                attempts=[dict(m=c['m'], K=c['K'], raw_bits=len(payload), arithmetic_bits=None,
                               selected_mode='raw', selected_bits=len(payload), capacity=p['source_capacity'], fits=True)])


def run_frame(*, ctx, full_slot, source_id, source_index, noise_seed, scales, arithmetic_bits, backend, header, ledger):
    require(type(source_index) is int and 0 <= source_index < COUNT
            and ctx['source_ids'][source_index] == source_id, 'Original full1000 source order differs')
    require(full_slot in ctx['rows_by_slot'], 'Unregistered public full slot')
    entry = ctx['rows_by_slot'][full_slot]
    phase, c = entry['phase'], entry['candidate']
    require(ledger.branch == 'H' and all(ledger.limits.get(k) == v for k, v in PHASE_CAPS.items()),
            'Original separate whole/partial quotas required; no reallocation')
    require(getattr(backend, 'device', 'cpu') == 'cpu', 'CPU backend required')
    noise = standard_noise(source_id, c['snr_db'], noise_seed)
    counter = frame_counter(full_slot, source_index, noise_seed)
    tx = select_payload(ctx, entry, scales, arithmetic_bits)
    p, book = tx['profile'], ctx['book']
    head_wave = np.asarray(header.transmit(p['profile_id']))
    require(head_wave.shape == (68, 2) and np.isfinite(head_wave).all(), 'Invalid paid header waveform')
    body_wave, body_meta = phy.transmit_body(backend, tx['payload'], p, counter, SESSION)
    sent = np.concatenate((head_wave, body_wave)).astype(np.float64)
    require(sent.shape == (1024, 2) and np.isfinite(sent).all(), 'Wrong full frame shape')
    received = (sent+noise*10**(-c['snr_db']/20)).astype(np.float32)
    event = f'H:{phase}:{c["candidate_id"]}:snr{c["snr_db"]}:src{source_index:04d}:seed{noise_seed}'
    # RX arguments contain no source tokens, TX profile, or clean arithmetic CDF.
    rx = phy.receive_frame(backend, header, received, c['snr_db'], counter, book, ledger, phase, event, SESSION)
    parsed = bool(rx.get('body') and rx['body'].get('parser_accepted'))
    rx_profile = book[str(rx['header']['profile_id'])] if rx['header']['header_ok'] else None
    tokens = None
    if not parsed:
        status, gray = 'WIRE_REJECT_GRAY', True
    elif rx_profile['mode'] == 'raw':
        tokens = phy.raw_tokens(rx['body']['payload'], rx_profile).tolist()
        status, gray = 'RAW_SOURCE_DECODED', False
    else:
        status, gray = 'ARITHMETIC_CANONICAL_RX_REQUIRED', None
    correct_wire = bool(parsed and rx_profile['profile_key'] == p['profile_key']
                        and rx['body']['payload'] == tx['payload'].tolist())
    return dict(status='H_FULL_CPU_FRAME_TRACE', stage=phase, source_id=source_id, source_index=source_index,
        candidate_id=c['candidate_id'], arm=c['arm'], snr_db=c['snr_db'], noise_seed=noise_seed,
        candidate_slot=c['slot'], full_slot=full_slot, public_frame_counter=counter, scrambling_session=SESSION,
        noise_stage=NOISE_STAGE, event_id=event, execution_registration_sha256=ctx['contract']['execution_registration_sha256'],
        engineering_choices_sha256=digest(ENGINEERING_CHOICES), noise_sha256=phy.array_sha(noise),
        transmitted_sha256=phy.array_sha(sent), received_sha256=phy.array_sha(received), total_symbols=1024,
        header_energy=float(np.square(sent[:68]).sum()), body_energy=float(np.square(sent[68:]).sum()),
        total_energy=float(np.square(sent).sum()), frame_normalized=False,
        tx=dict(profile_id=p['profile_id'], profile_key=p['profile_key'], target_m=tx['target_m'],
                actual_m=tx['actual_m'], K=p['K'], mode=p['mode'], fell_back=tx['fell_back'], attempts=tx['attempts'],
                payload_sha256=phy.array_sha(tx['payload']), phy=body_meta),
        rx=rx, rx_profile=rx_profile, rx_source_status=status, gray=gray, received_raw_tokens=tokens,
        evaluation_only=dict(header_correct=bool(rx['header']['header_ok'] and rx['header']['profile_id'] == p['profile_id']),
            parsed_wire_matches_transmission=correct_wire, accepted_wire_mismatch=bool(parsed and not correct_wire)),
        logical_packet_events=1+int(rx.get('body') is not None),
        packet_event_ids=[event+':header']+([event+':body'] if rx.get('body') is not None else []),
        gpu_used=False, arithmetic_source_decode_run=False, neural_metrics_run=False,
        initial_physical_results_reused=False)
