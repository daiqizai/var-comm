"""Prepared H development wire core. Importing it runs no science.

Only frozen H policies are mapped to a public transmission schedule. A future
driver must verify the normal selection/source owners, all original byte SHA
bindings, and the unchanged ledger before constructing this sealed context.
MAIN has a separate reserved allocation and is deliberately not implemented here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from h64_catalog import PROTOCOL, RATES, digest, require, token_count, normalize_state
import h64_phy as phy
import h_payload_cpu as initial
import h_full_payload_cpu as calibrated
import h_full_calibration_select as selection_rules

COUNT = 100
SEEDS = (6201, 6202, 6203)
SNRS = (13, 19)
ARMS = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
PHASE = 'development'
SESSION = 'H_DEVELOPMENT_V1'
H_SLOTS = 18
MAIN_SLOTS = (18, 19, 20, 21)
H_MAX_CALLS = 10800
MAIN_RESERVED_CALLS = 2400
TOTAL_CALLS = 13200
ENGINEERING_CHOICES = dict(
    schema='H_DEVELOPMENT_CPU_ENGINEERING_V1', source_count=COUNT,
    seeds=list(SEEDS), snrs_db=list(SNRS), phase=PHASE,
    H_policy_snr_points=18, MAIN_reserved_policy_snr_points=4,
    H_maximum_frames=5400, H_maximum_packet_calls=H_MAX_CALLS,
    MAIN_reserved_packet_calls=MAIN_RESERVED_CALLS, maximum_packet_calls=TOTAL_CALLS,
    slots='0..7 unchanged whole sorted original slot;8..9 partial sorted SNR;10..17 fixedm7 sorted SNR then H16-R,H16-A,H64-R,H64-A;18..21 MAIN reserved',
    counter='(development_slot*100+original_development_source_index)*3+noise_index',
    synchronization='Ideal public frame boundaries and schedule counters; counter carries no source content or received profile',
    noise='SHA256 canonical ASCII JSON [H protocol,development,source_id,SNR,seed],first16 little-endian PCG64,float64 standard_normal((1024,2)); same for all matched H and MAIN methods',
    received='(float64 waveform+10**(-SNR/20)*shared_standard_noise).astype(float32)',
    scrambling_session=SESSION, source_selection=False, policy_selection=False,
    catalogue='Entire original paid profile ID catalogue unchanged; actual received ID controls RX',
    fixed_source='m7 K0 rate1/2 only; no fallback, raw155-token payload must fit; strict-shorter arithmetic/raw-on-tie unchanged',
    unavailable='Retain declared NOT_FEASIBLE slot and reason; never substitute candidate or transfer unused calls',
    arithmetic_receiver='Wire parse only in CPU; every CRC-accepted arithmetic payload awaits independent canonical RX and actual rendering',
    automatic_launch=False, MAIN_PHY_implemented=False)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population(registration):
    require(registration.get('stage') == registration.get('calibration_or_development') == 'm1_development',
            'Original M1 development registration required, never calibration or holdout')
    ids, pp = registration['source_ids'], registration['preprocessing_ids']
    require(len(ids) == len(set(ids)) == len(pp) == COUNT and all(isinstance(s, str) and s for s in ids),
            'Original100 source identities/order required')
    require(all(initial.sha_string(v) for v in pp), 'Original preprocessing SHA pins required')
    bindings = registration['data_bindings']
    # The original development registration binds100 cached source NPZs as a
    # list; calibration1000 uses a different dictionary/shard schema.
    require(isinstance(bindings,list) and len(bindings) == COUNT, 'Original development100 data binding list required')
    for i, row in enumerate(bindings):
        require(row['index'] == i and initial.sha_string(row['rgb_sha256'])
                and initial.sha_string(row['source_npz_sha256']), 'Development cache index/SHA pin missing')
    return tuple(ids)


def schedule(finalized, selected, partial_reference, shortlist, catalogue, protocol):
    """Map completed choices; do not compare scores or run selection."""
    expected = dict(status='H_FULL1000_CALIBRATION_POLICIES_FINALIZED', schema='H_FULL1000_SELECTION_V1',
        source_count=1000, measured_frames=42000, whole_policy_reselected=False, new_candidate_search=False,
        new_packet_decodes=0, new_visual_inference=0, development_used=False, holdout_used=False)
    for k, value in expected.items():
        require(finalized.get(k) == value, 'Incomplete/changed full1000 selection: '+k)
    old = calibrated.schedule(selected, partial_reference, shortlist, catalogue, finalized['source_ids'])
    require(finalized['whole_candidates'] == selected['selected_candidates'], 'Whole policies must remain exact initial200 winners')
    require(len(finalized['selected_partial']) == 2 and {r['candidate']['snr_db'] for r in finalized['selected_partial']} == set(SNRS),
            'Exactly two full1000 selected partial policies required')
    for row in finalized['selected_partial']:
        require(type(row['full_slot']) is int and 8 <= row['full_slot'] < 14
                and row['candidate'] == old[row['full_slot']]['candidate']
                and row['selection_scope'] == 'ACTUAL_FULL1000_THREE_FROZEN_PARTIAL_CANDIDATES',
                'Partial policy not an exact member of original six calibrated candidates')
    measured = {(r['candidate']['policy_key'], r['candidate']['snr_db']) for r in old}
    controls = selection_rules.fixed_controls(catalogue, protocol, measured)
    require(finalized['fixed_source_controls'] == controls, 'Fixed m7 controls changed or were selected from data')
    result = []
    def add(role, candidate, available=True):
        result.append(dict(development_slot=len(result), role=role, phase=PHASE,
            status='FROZEN_POLICY_READY' if available else 'NOT_FEASIBLE',
            candidate=copy.deepcopy(candidate), missing_reason=candidate.get('reason') if not available else None))
    for row in sorted(finalized['whole_candidates'], key=lambda r:r['slot']):
        add('H_WHOLE_SYSTEM', row)
    for row in sorted(finalized['selected_partial'], key=lambda r:r['candidate']['snr_db']):
        add('H_RAW_PARTIAL_SYSTEM', row['candidate'])
    for row in controls:
        require(row['status'] in ('EXISTING_FIXED_POINT_FEASIBLE_NOT_EVALUATED', 'NOT_FEASIBLE'), 'Unknown fixed control status')
        add('H_FIXED_M7_ATTRIBUTION', row, row['status'] != 'NOT_FEASIBLE')
    require(len(result) == H_SLOTS, 'Frozen eighteen-point H schedule incomplete')
    return result


def prepare(*, protocol, catalogue, finalized, selected, partial_reference, shortlist, development_registration, contract):
    require(contract.get('status') == 'H_DEVELOPMENT_CPU_ENGINEERING_SEALED'
            and initial.sha_string(contract.get('execution_registration_sha256')), 'Independent development registration seal required')
    values = dict(protocol=protocol, catalogue=catalogue, finalized=finalized, selected=selected,
                  partial_reference=partial_reference, shortlist=shortlist,
                  development_registration=development_registration, engineering_choices=ENGINEERING_CHOICES)
    for name, value in values.items():
        require(contract.get(name+'_canonical_sha256') == digest(value), 'Sealed development input changed: '+name)
    modules = dict(core=__file__, initial_core=initial.__file__, calibrated_core=calibrated.__file__,
                   fixed_control_rules=selection_rules.__file__, phy=phy.__file__)
    for name, path in modules.items():
        require(contract.get(name+'_source_sha256') == file_sha(path), 'Required frozen source changed: '+name)
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['status'] == 'FROZEN_BEFORE_DATA'
        and (protocol['N'], protocol['header_symbols'], protocol['body_symbols'], protocol['Es'], protocol['channel']) == (1024,68,956,2,'AWGN')
        and protocol['main_snrs_db'] == list(SNRS)
        and protocol['population']['development_noise_seeds'] == list(SEEDS)
        and protocol['development']['maximum_packet_calls'] == TOTAL_CALLS
        and (protocol['matched_attribution']['m'],protocol['matched_attribution']['K'],
             protocol['matched_attribution']['nominal_rate']) == (7,0,'1/2')
        and protocol['matched_attribution']['arms'] == list(ARMS)
        and protocol['calibration']['no_development_selection'] is True, 'Frozen H development protocol differs')
    ids = population(development_registration)
    rows = schedule(finalized, selected, partial_reference, shortlist, catalogue, protocol)
    require(contract.get('public_schedule_canonical_sha256') == digest(rows), 'Eighteen-point public schedule changed')
    return dict(protocol=copy.deepcopy(protocol), catalogue=copy.deepcopy(catalogue),
        book=initial.public_codebook(catalogue), source_ids=ids, schedule=rows,
        rows_by_slot={r['development_slot']:r for r in rows}, contract=copy.deepcopy(contract))


def standard_noise(source_id, snr_db, noise_seed):
    require(isinstance(source_id,str) and source_id and type(snr_db) is int and snr_db in SNRS
            and type(noise_seed) is int and noise_seed in SEEDS, 'Unregistered development noise identity')
    encoded = json.dumps([PROTOCOL, PHASE, source_id, snr_db, noise_seed], sort_keys=True,
                         separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1024,2))


def frame_counter(slot, source_index, noise_seed):
    require(type(slot) is int and 0 <= slot < 22 and type(source_index) is int and 0 <= source_index < COUNT
            and type(noise_seed) is int and noise_seed in SEEDS, 'Outside public22-slot development schedule')
    return (slot*COUNT+source_index)*3+SEEDS.index(noise_seed)


def frame_event(slot, source_index, noise_seed):
    frame_counter(slot, source_index, noise_seed)
    require(slot < H_SLOTS, 'MAIN reserve cannot be consumed by H core')
    return f'H:development:Hslot{slot:02d}:src{source_index:04d}:seed{noise_seed}'


class FrameLedger:
    """Only the two prepaid events of this H frame can reach the atomic ledger.

    Across all18 slots and100x3 draws this finite namespace has at most10,800
    distinct calls. Original duplicate/result/failure handling is not replaced.
    MAIN must use its own four slots; no unused-call transfer is permitted.
    """
    def __init__(self, ledger, slot, source_index, noise_seed):
        require(ledger.branch == 'H' and ledger.limits.get(PHASE) == TOTAL_CALLS, 'Original development13200 quota required')
        self.ledger, self.event = ledger, frame_event(slot, source_index, noise_seed)

    def decode_once(self, phase, event_id, kind, phy_key, request, decode):
        require(phase == PHASE and kind in ('header','body') and event_id == self.event+':'+kind,
                'Decode outside exact H frame; MAIN reserve cannot be spent here')
        return self.ledger.decode_once(phase,event_id,kind,phy_key,request,decode)

    def register_configuration(self, key, definition):
        return self.ledger.register_configuration(key,definition)


def select_payload(ctx, entry, scales, arithmetic_bits):
    c = entry['candidate']
    require(entry['status'] == 'FROZEN_POLICY_READY', 'Unavailable control cannot be replaced or evaluated')
    if entry['role'] in ('H_WHOLE_SYSTEM', 'H_FIXED_M7_ATTRIBUTION'):
        # Fixed controls have no old shortlist slot. This local validation index
        # is not used for noise, counters, candidate identity, or policy choice.
        wire = dict(c, slot=c.get('slot', entry['development_slot']-10))
        tx = initial.select_payload(wire, ctx['catalogue'], scales, arithmetic_bits)
        if entry['role'] == 'H_FIXED_M7_ATTRIBUTION':
            require((c['target_m'], c['K'], c['nominal_rate']) == (7,0,'1/2')
                    and tx['actual_m'] == 7 and not tx['fell_back'] and len(tx['attempts']) == 1,
                    'Fixed m7 source must not fallback or truncate')
        return tx
    require(entry['role'] == 'H_RAW_PARTIAL_SYSTEM', 'Only the three registered H roles are executable')
    p = ctx['book'][str(c['profile_id'])]
    require((p['m'],p['K']) == normalize_state(p['m'],p['K']) and p['mode'] == 'raw' and p['q'] == 6,
            'Partial profile is not exact canonical raster raw64')
    payload = phy.raw_payload(scales,p['m'],p['K'])
    require(len(payload) == 12*c['token_count'] <= p['source_capacity'], 'Partial source budget differs')
    return dict(profile=p,payload=payload,target_m=p['m'],actual_m=p['m'],fell_back=False,
        attempts=[dict(m=p['m'],K=p['K'],raw_bits=len(payload),arithmetic_bits=None,
                       selected_mode='raw',selected_bits=len(payload),capacity=p['source_capacity'],fits=True)])


def run_frame(*, ctx, development_slot, source_id, source_index, noise_seed, scales, arithmetic_bits, backend, header, ledger):
    require(type(source_index) is int and 0 <= source_index < COUNT and ctx['source_ids'][source_index] == source_id,
            'Original100 development source order differs')
    require(type(development_slot) is int and development_slot in ctx['rows_by_slot'], 'Unregistered H development slot')
    require(getattr(backend,'device','cpu') == 'cpu', 'CPU PHY required')
    entry = ctx['rows_by_slot'][development_slot];c = entry['candidate']
    bounded = FrameLedger(ledger,development_slot,source_index,noise_seed)
    noise = standard_noise(source_id,c['snr_db'],noise_seed)
    counter = frame_counter(development_slot,source_index,noise_seed)
    tx = select_payload(ctx,entry,scales,arithmetic_bits);p = tx['profile']
    head_wave = np.asarray(header.transmit(p['profile_id']))
    require(head_wave.shape == (68,2) and np.isfinite(head_wave).all(), 'Invalid paid header waveform')
    body_wave,body_meta = phy.transmit_body(backend,tx['payload'],p,counter,SESSION)
    sent = np.concatenate((head_wave,body_wave)).astype(np.float64)
    require(sent.shape == (1024,2) and np.isfinite(sent).all(), 'Wrong full paid frame')
    received = (sent+noise*10**(-c['snr_db']/20)).astype(np.float32)
    rx = phy.receive_frame(backend,header,received,c['snr_db'],counter,ctx['book'],bounded,PHASE,bounded.event,SESSION)
    parsed = bool(rx.get('body') and rx['body'].get('parser_accepted'))
    rp = ctx['book'][str(rx['header']['profile_id'])] if rx['header']['header_ok'] else None
    tokens = None
    if not parsed:status,gray = 'WIRE_REJECT_GRAY',True
    elif rp['mode'] == 'raw':
        tokens = phy.raw_tokens(rx['body']['payload'],rp).tolist();status,gray = 'RAW_SOURCE_DECODED',False
    else:status,gray = 'ARITHMETIC_CANONICAL_RX_REQUIRED',None
    correct = bool(parsed and rp['profile_key'] == p['profile_key'] and rx['body']['payload'] == tx['payload'].tolist())
    return dict(status='H_DEVELOPMENT_CPU_FRAME_TRACE',stage=PHASE,role=entry['role'],development_slot=development_slot,
        source_id=source_id,source_index=source_index,candidate_id=c['candidate_id'],arm=c['arm'],
        snr_db=c['snr_db'],noise_seed=noise_seed,public_frame_counter=counter,scrambling_session=SESSION,
        noise_stage=PHASE,event_id=bounded.event,execution_registration_sha256=ctx['contract']['execution_registration_sha256'],
        engineering_choices_sha256=digest(ENGINEERING_CHOICES),noise_sha256=phy.array_sha(noise),
        transmitted_sha256=phy.array_sha(sent),received_sha256=phy.array_sha(received),total_symbols=1024,
        header_energy=float(np.square(sent[:68]).sum()),body_energy=float(np.square(sent[68:]).sum()),
        total_energy=float(np.square(sent).sum()),frame_normalized=False,
        tx=dict(profile_id=p['profile_id'],profile_key=p['profile_key'],target_m=tx['target_m'],actual_m=tx['actual_m'],
            K=p['K'],mode=p['mode'],fell_back=tx['fell_back'],attempts=tx['attempts'],payload_sha256=phy.array_sha(tx['payload']),phy=body_meta),
        rx=rx,rx_profile=rp,rx_source_status=status,gray=gray,received_raw_tokens=tokens,
        evaluation_only=dict(header_correct=bool(rx['header']['header_ok'] and rx['header']['profile_id'] == p['profile_id']),
            parsed_wire_matches_transmission=correct,accepted_wire_mismatch=bool(parsed and not correct)),
        logical_packet_events=1+int(rx.get('body') is not None),
        packet_event_ids=[bounded.event+':header']+([bounded.event+':body'] if rx.get('body') is not None else []),
        gpu_used=False,arithmetic_source_decode_run=False,neural_metrics_run=False,policy_selection=False,
        calibration_physical_results_reused=False,MAIN_PHY_run=False)
