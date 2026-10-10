"""Finite full1000 PHY grid and exact-event reuse, without model execution."""
from __future__ import annotations
import copy
from pathlib import Path
import ep_plan as plan
import h800_ep_pilot_core_v1 as pilot

SCHEMA = 'H800_EP_ORIGINAL1000_FULL_PHY_V1'
LOGICAL_CAP = 36000
PACKET_CAP = 72000
SEEDS = (4101, 4102, 4103)
require = pilot.require


def shortlist(metric_rows, actual_finalists, policy):
    """Recompute from all43200 actual four-metric rows, never a top3-only table."""
    ids = policy['calibration_source_ids']
    require(len(ids) == 1000 and len(set(ids)) == 1000 and
        policy['status'] == 'T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and
        policy['holdout_used_for_selection'] is False, 'Original full1000 calibration policy required')
    for row in metric_rows:
        i = row['source_index']
        require(type(i) is int and 0 <= i < 100 and row['source_id'] == ids[i],
            'Pilot metrics must use the exact original first100 IDs')
    expected = pilot.finalists(metric_rows, policy)
    require(actual_finalists == expected, 'Actual complete-pilot finalists or original whole winner differ')
    return expected


def admitted_finalists(value, policy):
    require(value['status'] == 'PILOT_SHORTLIST_ONLY_FULL_CALIBRATION_REQUIRED' and
        value['holdout_used_for_selection'] is False and value['final_strategy_frozen'] is False and
        value['automatic_successor'] is False and set(value['snrs']) == {'4', '10', '19'},
        'Complete original pilot shortlist required, not a frozen final strategy')
    result = {}
    for snr in plan.SNRS:
        row = value['snrs'][str(snr)]
        whole = policy['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id']
        expected = list(plan.full_shortlist(row['ranking'], whole))
        require(row['finalists'] == expected and len(expected) in (3, 4) and
            row['original_final_whole_winner'] == whole and row['whole_winner_retained'] is True,
            'Each SNR must retain pilot top3 union original whole winner')
        result[snr] = expected
    return result


def counter(index, snr, seed):
    require(type(index) is int and 0 <= index < 1000 and snr in plan.SNRS and seed in SEEDS,
        'Original1000, registered SNR and seed required')
    return ([1, 4, 7, 10, 13, 19].index(snr) * 1000 + index) * 3 + SEEDS.index(seed)


def frames(records, finalists, policy):
    require(len(records) == 1000 and [r['source_index'] for r in records] == list(range(1000)) and
        [r['source_id'] for r in records] == policy['calibration_source_ids'] and
        len({r['source_id'] for r in records}) == 1000, 'Exact original1000 order required')
    chosen = admitted_finalists(finalists, policy)
    candidates = {r['candidate_id']: r for r in plan.candidates()}
    result = []
    for source in records:
        for snr in plan.SNRS:
            for seed in SEEDS:
                for cid in chosen[snr]:
                    i = source['source_index']
                    result.append(dict(candidates[cid], source_index=i, source_id=source['source_id'],
                        snr_db=snr, noise_seed=seed, public_frame_counter=counter(i, snr, seed),
                        event_id=f'{SCHEMA}/source{i}/snr{snr}/seed{seed}/{cid}'))
    require(len(result) == 1000 * 3 * sum(map(len, chosen.values())) <= LOGICAL_CAP and
        len({r['event_id'] for r in result}) == len(result), 'Finite complete full-calibration grid required')
    return result


def complete_rows(rows, records, finalists, policy):
    require(rows == frames(records, finalists, policy),
        'Missing, duplicate, reordered or changed full-calibration logical frame')


def pilot_cache(certificate, identity, read):
    """Admit actual closed pilot packets, including its separately closed48 reuse."""
    require(certificate['schema'] == 'CLOSED_ORIGINAL100_PILOT_PHY_REUSE_V1' and
        certificate['phy_identity'] == identity and certificate['previous_call_counts_preserved'] is True,
        'Complete closed pilot and identical qualified physical environment required')
    owner = read(certificate['owner_completion']); wait = read(certificate['owner_actual_wait'])
    worker = read(owner['worker_completion']); request = read(certificate['request'])
    require(wait['actual_wait'] is True and wait['returncode'] == 0 and owner['actual_wait']['success'] is True and
        owner['actual_children_waited'] is True and owner['worker_exit_codes'] == [0] and
        owner['status'] == worker['status'] == 'PASS_H800_ORIGINAL100_PILOT_PHY_ONLY' and
        owner['request_sha256'] == worker['request_sha256'] == certificate['request']['sha256'],
        'Actual completed pilot owner/worker chain required')
    require(request['schema'] == 'H800_EP_ORIGINAL100_PILOT_PHY_V1' and request['phy_identity'] == identity and
        worker['CUDA_initialized'] is False and len(worker['results']['logical_frames']) == 43200 and
        worker['results']['packet_ledger']['unresolved'] == 0 and
        worker['results']['packet_ledger']['total'] <= 86400, 'Complete bounded pilot PHY proof required')
    pilot.complete_rows(request['frames'])
    cache = pilot.external_packets(certificate['closed48'], identity)
    expected = {r['event_id']: r for r in request['frames']}
    for pin in worker['results']['actual_new_physical']:
        packet = read(pin)
        require(packet['schema'] == pilot.SCHEMA and packet['external_packet_reuse'] is False and
            packet['logical_event'] == expected.get(packet['logical_event']['event_id']),
            'Pilot reusable packet must belong to an actual admitted event')
        key = pilot.physical_key(packet, identity)
        require(packet['physical_key'] == key, 'Pinned pilot physical identity differs')
        if key in cache:
            require(cache[key]['actual_RX'] == packet['actual_RX'], 'Same physical event has inconsistent RX')
        cache[key] = dict(pin=pin, actual_RX=copy.deepcopy(packet['actual_RX']), origin='CLOSED_PILOT')
    return cache


def cpu_frames(runtime, rows, records, finalists, policy, streams, partial, phy, ledger, boundary,
               out, identity, external, read):
    """Hash each new actual observation before reusing a whole actual RX outcome."""
    complete_rows(rows, records, finalists, policy)
    require(len(runtime.profiles) == 360 and identity['profile_count'] == 360 and
        identity['catalogue_sha256'] == runtime.catalogue['catalogue_sha256'] and
        identity['backend_identity'] == runtime.backend.identity, 'Qualified360-profile PHY identity required')
    require(set(streams) == set(range(1000)), 'Exactly1000 original fresh H800 stream sets required')
    out = Path(out); out.mkdir(); (out / 'physical').mkdir(); (out / 'logical').mkdir()
    cache = pilot_cache(external, identity, read)
    lookup = {(p['family'], p['m'], p['K'], p['q'], p['nominal_rate']): p for p in runtime.profiles.values()}
    pins, unique = [], []
    counts = dict(actual_new_physical=0, reused_closed_pilot=0, reused_closed48=0, reused_this_full=0)
    for index, row in enumerate(rows):
        boundary(); source = streams[row['source_index']]
        selected = plan.select_stream({key: len(bits) for key, bits in source.items()}, row)
        m, K = selected['m'], selected['K']; family = partial.PARTIAL_FAMILY if K else partial.WHOLE_FAMILY
        profile = lookup[family, m, K, row['q'], row['nominal_rate']]; bits = source[m, K].copy()
        wave, tx = runtime.transmit(profile['profile_id'], bits, row['public_frame_counter'])
        noise = phy.standard_noise(row['source_id'], row['noise_seed']) * 10 ** (-row['snr_db'] / 20)
        packet = dict(schema=SCHEMA, logical_event=row, fallback=selected, transmission=tx,
            payload_sha256=phy.array_sha(bits), noise_sha256=phy.array_sha(noise),
            observation_sha256=phy.array_sha(wave + noise), full_public_receive_catalogue=True,
            source_truth_supplied_to_RX=False)
        key = pilot.physical_key(packet, identity)
        if key not in cache:
            rx = runtime.receive(wave + noise, row['snr_db'], row['public_frame_counter'], runtime.profiles,
                ledger, row['event_id'], phase='entropy_partial_original1000_full')
            packet.update(actual_RX=rx, physical_key=key, external_packet_reuse=False)
            path = out / 'physical' / f'{len(unique):05d}.json'; pilot.save(path, packet)
            pin = pilot.descriptor(path); unique.append(pin)
            cache[key] = dict(pin=pin, actual_RX=rx, origin='THIS_FULL')
            counts['actual_new_physical'] += 1; reuse = 'ACTUAL_NEW'
        else:
            reuse = cache[key]['origin']
            field = {'CLOSED_EP48': 'reused_closed48', 'CLOSED_PILOT': 'reused_closed_pilot',
                'THIS_FULL': 'reused_this_full'}[reuse]
            counts[field] += 1
        path = out / 'logical' / f'{index:05d}.json'
        pilot.save(path, dict(frame_index=index, logical_event=row, physical_key=key,
            physical_frame=cache[key]['pin'], reuse=reuse))
        pins.append(pilot.descriptor(path))
    snapshot = ledger.snapshot()
    require(snapshot['unresolved'] == 0 and snapshot['total'] <= PACKET_CAP and
        sum(counts.values()) == len(rows) <= LOGICAL_CAP, 'Full PHY ledger did not close within bound')
    return dict(logical_frames=pins, actual_new_physical=unique, counts=counts,
        packet_ledger=snapshot, previous_pilot_and_gate_packet_counts_erased=False,
        neural_model_calls=0, metric_calls=0)
