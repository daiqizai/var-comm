"""Finite original-calibration pilot operations; importing performs no science.

The owner must bind its inputs, environment, completed gates and execution window
before calling these operations. No training, holdout reads or automatic retry.
Existing source arithmetic, PHY, received-token rendering and metrics are reused.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import ep_plan as plan
import h800_ep48_link_core_v1 as link

SCHEMA = 'H800_EP_ORIGINAL100_PILOT_V1'
LOGICAL_FRAMES = 43200
PACKET_CAP = 86400
SOURCE_CAPS = dict(model_load=1, encoder=0, source_tx=68, source_rx=0,
                   var_render=0, prior_scale=680, decoder_forward=0)
VISUAL_CAPS = dict(model_load=1, encoder=0, source_tx=0, source_rx=43200,
                   var_render=43200, prior_scale=864000, decoder_forward=43200)
METRIC_CAPS = dict(suite_constructions=1, reference_preparations=100,
                   image_scores=43200)
METRICS = plan.METRICS
MANIFEST_SHA = '49bb6b8dfb45ac9fec5968481fe7f1691959a80964972bad1c0bde4e468e3cae'
SOURCE_COMPLETION_SHA = '9760c3f946a10abb1dd7fdc050cfc96832b3e2a31f359f86e7fb2504393a335b'
OLD_ASSET = '/home/liulu/projects/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/full1000_assets/'
require = link.require
sha = link.sha
save = link.save
descriptor = link.descriptor


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def checked(pin):
    path = Path(pin['path'])
    require(sha(path) == pin['sha256'], 'Frozen pilot file changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def input_records(manifest_pin, completion_pin, relocation, inside):
    """Exact original100 metadata/archive closure, without reading image arrays."""
    require(manifest_pin['sha256'] == MANIFEST_SHA and completion_pin['sha256'] == SOURCE_COMPLETION_SHA,
            'Pinned original1000 calibration metadata required')
    manifest = checked(manifest_pin); completion = checked(completion_pin)
    require(manifest['source_count'] == completion['source_count'] == 1000 and
            len(manifest['records']) == len(manifest['source_ids']) == 1000,
            'Original calibration population changed')
    rows = manifest['records'][:100]; result = []
    require([r['source_index'] for r in rows] == list(range(100)) and
            [r['source_id'] for r in rows] == manifest['source_ids'][:100] and
            len({r['source_id'] for r in rows}) == 100, 'Original calibration first100 ordering required')
    for i, row in enumerate(rows):
        old_cp = OLD_ASSET + f'source_checkpoints/{i:04d}.json'
        old_npz = OLD_ASSET + f'sources/{i:04d}.npz'
        require(row['checkpoint'] == old_cp and row['archive'] == old_npz,
                'Unexpected original calibration input path')
        cp_pin = relocation[old_cp]; archive = relocation[old_npz]
        require(cp_pin['sha256'] == row['checkpoint_sha256'] == completion['outputs'][old_cp] and
                archive['sha256'] == completion['outputs'][old_npz], 'Original relocation digest differs')
        inside(cp_pin['path']); npz_path = inside(archive['path']); cp = checked(cp_pin)
        require(npz_path.is_file() and sha(npz_path) == archive['sha256'], 'Restored original source bytes differ')
        require(cp['original_calibration_index'] == i and cp['outputs'][old_npz] == archive['sha256'] and all(
                cp[k] == row[k] for k in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256', 'archive')),
                'Original checkpoint identity or asset closure differs')
        require(type(cp['evaluation_class_index']) is int and 0 <= cp['evaluation_class_index'] < 1000,
                'Original evaluation class metadata malformed')
        result.append(dict(source_index=i, source_id=row['source_id'], preprocessing_id=row['preprocessing_id'],
            tokens_sha256=row['tokens_sha256'], evaluation_class_index=cp['evaluation_class_index'],
            archive=dict(path=str(npz_path), sha256=archive['sha256'], bytes=npz_path.stat().st_size),
            checkpoint=cp_pin))
    return result


def reference_pixels(record):
    pin = record['archive']
    require(sha(pin['path']) == pin['sha256'], 'Original reference archive changed')
    with np.load(pin['path'], allow_pickle=False) as archive: pixels = archive['pixels'].copy()
    require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256) and
            hashlib.sha256(pixels.tobytes()).hexdigest() == record['preprocessing_id'],
            'Original calibration preprocessing/pixels differ')
    return pixels.astype(np.float32) / 255


def counter(index, snr):
    require(type(index) is int and 0 <= index < 100 and snr in plan.SNRS,
            'Original first100 and three registered SNRs required')
    return ([1, 4, 7, 10, 13, 19].index(snr) * 1000 + index) * 3


def frames(records, policy):
    require(len(records) == 100 and [r['source_index'] for r in records] == list(range(100)),
            'Complete original first100 in original order required')
    ids = [r['source_id'] for r in records]
    require(len(set(ids)) == 100 and ids == policy['calibration_source_ids'][:100],
            'Pilot source IDs differ from frozen original calibration')
    require(policy['status'] == 'T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1' and
            policy['holdout_used_for_selection'] is False and policy['source_count'] == 1000,
            'Original final whole policy required')
    result = []
    for row in records:
        for snr in plan.SNRS:
            for candidate in plan.candidates():
                result.append(dict(candidate, source_index=row['source_index'],
                    source_id=row['source_id'], snr_db=snr, noise_seed=4101,
                    public_frame_counter=counter(row['source_index'], snr),
                    event_id=f"{SCHEMA}/source{row['source_index']}/snr{snr}/seed4101/{candidate['candidate_id']}"))
    require(len(result) == LOGICAL_FRAMES and len({r['event_id'] for r in result}) == LOGICAL_FRAMES,
            'Exactly43200 distinct logical pilot frames required')
    return result


def complete_rows(rows):
    """Reject missing, repeated, altered or holdout rows before expensive work."""
    candidates = {c['candidate_id']: c for c in plan.candidates()}
    expected = {(i, snr, cid) for i in range(100) for snr in plan.SNRS for cid in candidates}
    seen = set()
    source_ids = {}
    require(len(rows) == LOGICAL_FRAMES, 'Exactly43200 logical rows required')
    for row in rows:
        key = row['source_index'], row['snr_db'], row['candidate_id']
        require(key in expected and key not in seen, 'Missing/duplicate/unregistered pilot cell')
        seen.add(key)
        candidate = candidates[row['candidate_id']]
        require(all(row[k] == v for k, v in candidate.items()), 'Frozen candidate changed')
        require(row['noise_seed'] == 4101 and row['public_frame_counter'] == counter(*key[:2]),
                'Registered noise/counter changed')
        require(source_ids.setdefault(key[0], row['source_id']) == row['source_id'], 'Source alias changed')
    require(seen == expected and len(set(source_ids.values())) == 100, 'Complete original100 grid required')


def source68(record, backend, codec, partial, ledger, load_tokens, out):
    """Generate only missing H800 TX streams; no RX, encoder, render or scoring."""
    index = record['source_index']
    require(type(index) is int and 32 <= index < 100, 'Only original calibration0032..0099 TX permitted')
    out = Path(out); out.mkdir()
    backend.source_index = index; backend.current = ('TX', 10)
    backend.traces = []; backend.tx_cdfs = {}; backend.received = {}
    backend.boundary()
    tokens = load_tokens(record)
    encoded = ledger.call('source_tx', lambda: codec.encode_endpoints(tokens, partial.all_endpoints()), source=index)
    del tokens
    trace = copy.deepcopy(backend.traces)
    require(len(trace) == 10 and [r['scale'] for r in trace] == list(range(10)) and all(
        r['source'] == index and r['role'] == 'TX' and r['m'] == 10 for r in trace),
        'Ten actual fresh TX priors required')
    endpoints = partial.all_endpoints()
    require(set(encoded) == set(endpoints) and len(endpoints) == 24, 'Exact24 fresh endpoints required')
    arrays = {}; metadata = []
    for m, K in endpoints:
        item = encoded[m, K]; bits = item['bits']
        require(bits.dtype == np.uint8 and bits.ndim == 1 and 2 <= len(bits) <= 1048576 and
                np.isin(bits, (0, 1)).all() and item['arithmetic_bits'] == len(bits) and
                item['m'] == m and item['K'] == K, 'Actual endpoint bits malformed')
        arrays[f'm{m}_K{K}'] = bits.copy()
        metadata.append(dict(m=m, K=K, payload_bits=len(bits), bits_sha256=backend.g.image_sha(bits)))
    archive = out / 'actual_streams.npz'
    with archive.open('xb') as stream: np.savez(stream, **arrays)
    actual = dict(source_index=index, archive=descriptor(archive), endpoints=metadata,
                  CDF_trace=trace, old_host_bits_used=False, old_host_CDF_used=False)
    path = out / 'actual_TX.json'; save(path, actual)
    result = dict(source_index=index, source_id=record['source_id'], metadata=descriptor(path),
                  archive=actual['archive'], actual_new_TX=1, actual_new_prior=10,
                  independent_RX_calls=0, status='PILOT_FRESH_TX_COMPLETE_NO_RX_CLAIM')
    save(out / 'completion.json', result)
    backend.traces = []; backend.tx_cdfs = {}; backend.received = {}; backend.boundary()
    return result


def physical_key(packet, phy_identity):
    """Exact received experiment including full catalogue and numerical backend."""
    row = packet['logical_event']; tx = packet['transmission']
    require(packet['full_public_receive_catalogue'] is True and packet['source_truth_supplied_to_RX'] is False,
            'Full actual public received event required')
    require(phy_identity['profile_count'] == 360, 'Full360 receive catalogue required')
    return digest(dict(phy_identity=phy_identity, snr_db=row['snr_db'],
        source_id=row['source_id'], noise_seed=row['noise_seed'],
        public_frame_counter=row['public_frame_counter'], transmission=tx,
        payload_sha256=packet['payload_sha256'], noise_sha256=packet['noise_sha256'],
        observation_sha256=packet['observation_sha256']))


def external_packets(certificate, current_identity):
    """Owner-pinned closed48 evidence only; unqualified oldT1 is never admitted."""
    require(certificate['schema'] == 'CLOSED_EP48_EXACT_EVENT_REUSE_V1' and
            certificate['phy_identity'] == current_identity and
            certificate['already_consumed_counts_preserved'] is True,
            'Independent closed48 provenance and identical PHY environment required')
    wait = checked(certificate['owner_actual_wait'])
    done = checked(certificate['owner_completion'])
    request = checked(certificate['request'])
    require(wait['actual_wait'] is True and wait['returncode'] == 0 and
            done['status'] == link.PASS and done['actual_wait']['success'] is True and
            done['request_sha256'] == certificate['request']['sha256'] and request['schema'] == link.SCHEMA,
            'EP48 owner must actually close before pilot event reuse')
    gpu = checked(done['worker_completion']); cpu_owner = checked(gpu['CPU_completion'])
    cpu = checked(cpu_owner['worker_completion'])
    require(cpu_owner['actual_wait']['success'] is True and cpu['actual_frames'] == 48 and
            cpu['packet_ledger']['unresolved'] == 0 and cpu['frames'] == certificate['physical_packets'] and
            gpu['request_sha256'] == cpu['request_sha256'] == done['request_sha256'],
            'External packets must belong to actual completed48 owner/worker chain')
    result = {}
    require(len(certificate['physical_packets']) == 48, 'Complete pinned48 receipt set required')
    for pin in certificate['physical_packets']:
        packet = checked(pin)
        require(packet['schema'] == link.SCHEMA and packet['external_packet_reuse'] is False,
                'Only actual original EP48 packet receipts are reusable')
        key = physical_key(packet, current_identity)
        if key in result:
            require(result[key]['actual_RX'] == packet['actual_RX'], 'Same physical event has inconsistent RX')
        result[key] = dict(pin=pin, actual_RX=copy.deepcopy(packet['actual_RX']), origin='CLOSED_EP48')
    return result


def cpu_frames(runtime, rows, streams, partial, phy, ledger, boundary, out, identity, external=None):
    """Receive exact unique physical events; preserve every logical candidate row."""
    complete_rows(rows)
    require(len(runtime.profiles) == 360 and identity['profile_count'] == 360 and
            identity['catalogue_sha256'] == runtime.catalogue['catalogue_sha256'] and
            identity['backend_identity'] == runtime.backend.identity,
            'Current qualified360-profile PHY identity required')
    out = Path(out); out.mkdir(); (out / 'physical').mkdir(); (out / 'logical').mkdir()
    cache = external_packets(external, identity) if external is not None else {}
    lookup = {(p['family'], p['m'], p['K'], p['q'], p['nominal_rate']): p for p in runtime.profiles.values()}
    pins = []; unique = []; counts = dict(actual_new_physical=0, reused_closed48=0, reused_this_pilot=0)
    for index, row in enumerate(rows):
        boundary(); source = streams[row['source_index']]
        selected = plan.select_stream({key:len(bits) for key, bits in source.items()}, row)
        m, K = selected['m'], selected['K']; family = partial.PARTIAL_FAMILY if K else partial.WHOLE_FAMILY
        profile = lookup[family, m, K, row['q'], row['nominal_rate']]; bits = source[m, K].copy()
        wave, tx = runtime.transmit(profile['profile_id'], bits, row['public_frame_counter'])
        noise = phy.standard_noise(row['source_id'], row['noise_seed']) * 10 ** (-row['snr_db'] / 20)
        packet = dict(schema=SCHEMA, logical_event=row, fallback=selected, transmission=tx,
            payload_sha256=phy.array_sha(bits), noise_sha256=phy.array_sha(noise),
            observation_sha256=phy.array_sha(wave + noise), full_public_receive_catalogue=True,
            source_truth_supplied_to_RX=False)
        key = physical_key(packet, identity)
        if key not in cache:
            rx = runtime.receive(wave + noise, row['snr_db'], row['public_frame_counter'], runtime.profiles,
                                 ledger, row['event_id'], phase='entropy_partial_original100_pilot')
            packet.update(actual_RX=rx, physical_key=key, external_packet_reuse=False)
            path = out / 'physical' / f'{len(unique):05d}.json'; save(path, packet)
            pin = descriptor(path); unique.append(pin)
            cache[key] = dict(pin=pin, actual_RX=rx, origin='THIS_PILOT')
            counts['actual_new_physical'] += 1; reuse = 'ACTUAL_NEW'
        else:
            reuse = cache[key]['origin']
            counts['reused_closed48' if reuse == 'CLOSED_EP48' else 'reused_this_pilot'] += 1
        entry = dict(frame_index=index, logical_event=row, physical_key=key,
                     physical_frame=cache[key]['pin'], reuse=reuse)
        path = out / 'logical' / f'{index:05d}.json'; save(path, entry); pins.append(descriptor(path))
    snapshot = ledger.snapshot()
    require(snapshot['unresolved'] == 0 and snapshot['total'] <= PACKET_CAP and
            sum(counts.values()) == LOGICAL_FRAMES, 'Pilot PHY counts not closed within frozen bound')
    return dict(logical_frames=pins, actual_new_physical=unique, counts=counts, packet_ledger=snapshot,
                prior_gate_packet_counts_erased=False)


def source_input(actual_rx, visual_identity):
    """Cache the decoder's actual inputs; TX target/truth never enters this key."""
    body = actual_rx['body']; profile = actual_rx['rx_profile']
    if not actual_rx['header']['header_ok'] or body is None or not body['crc_accepted'] or not body['parser_accepted']:
        return dict(visual_identity=visual_identity, kind='FIXED_GRAY')
    require(profile is not None and profile['family'] in ('EC_STATIC_WHOLE', 'EC_VAR_WHOLE', 'EC_VAR_PARTIAL'),
            'Unregistered accepted source family')
    return dict(visual_identity=visual_identity, kind='ACTUAL_SOURCE_INPUT', family=profile['family'],
                m=profile['m'], K=profile['K'], payload=body['payload'])


def gpu_frames(logical_pins, codec, backend, partial, ledger, boundary, out, visual_identity):
    """One original independent decode/render per identical actual source input."""
    require(len(logical_pins) == LOGICAL_FRAMES, 'Exact complete pilot physical outcome set required')
    require(visual_identity and visual_identity['device_identity'] and visual_identity['native_binding'] and
            visual_identity['model_state_identity'] and visual_identity['numeric_settings'],
            'Explicit actual visual identity required for reconstruction reuse')
    out = Path(out); out.mkdir(); (out / 'images').mkdir(); (out / 'logical').mkdir()
    cache = {}; pins = []; physical = {}
    for index, pin in enumerate(logical_pins):
        boundary(); row = checked(pin); require(row['frame_index'] == index, 'Pilot logical order changed')
        pp = row['physical_frame']; pk = canonical(pp)
        if pk not in physical: physical[pk] = checked(pp)
        packet = physical[pk]; actual_rx = packet['actual_RX']
        key = digest(source_input(actual_rx, visual_identity))
        if key not in cache:
            backend.source_index = row['logical_event']['source_index']
            image, evidence = link.recover_and_render(actual_rx, codec, backend, partial, ledger, boundary)
            image_path = out / 'images' / f'{len(cache):05d}.npz'
            with image_path.open('xb') as stream: np.savez(stream, image=image)
            cache[key] = dict(image_archive=descriptor(image_path), image_sha256=backend.g.image_sha(image),
                actual_source_input_key=key, evidence=evidence, first_physical_frame=pp)
        result = dict(frame_index=index, logical_event=row['logical_event'], physical_frame=pp,
                      actual_RX_status=actual_rx['status'], actual_received_profile=actual_rx['rx_profile'],
                      actual_header_ok=actual_rx['header']['header_ok'],
                      actual_crc_accepted=None if actual_rx['body'] is None else actual_rx['body']['crc_accepted'],
                      actual_parser_accepted=None if actual_rx['body'] is None else actual_rx['body']['parser_accepted'],
                      reconstruction=cache[key],
                      reconstruction_evidence_describes_first_cached_decode=True)
        path = out / 'logical' / f'{index:05d}.json'; save(path, result); pins.append(descriptor(path))
    counts = ledger.summary()
    require(counts['unresolved'] == 0 and counts['completed'] == counts['reserved'] and all(
        0 <= counts['completed'][k] <= cap for k, cap in VISUAL_CAPS.items()), 'Pilot visual calls exceed frozen caps')
    require(counts['completed']['model_load'] == 1 and
            counts['completed']['decoder_forward'] == counts['completed']['var_render'],
            'One visual model and one Dc per actual render required')
    return dict(logical_frames=pins, unique_source_inputs=len(cache), counts=counts)


def finalists(metric_rows, policy):
    """DINOv2-L pilot top3 union unchanged original full-calibration whole winner."""
    candidates = {c['candidate_id'] for c in plan.candidates()}
    expected = {(i, s, c) for i in range(100) for s in plan.SNRS for c in candidates}
    seen = set(); grouped = {(s, c):[] for s in plan.SNRS for c in candidates}
    require(len(metric_rows) == LOGICAL_FRAMES, 'Complete pilot metrics required before ranking')
    for row in metric_rows:
        key = row['source_index'], row['snr_db'], row['candidate_id']
        require(key in expected and key not in seen, 'Missing or duplicate pilot metric cell')
        seen.add(key)
        require(row['noise_seed'] == 4101 and all(type(row[m]) in (int, float) and math.isfinite(row[m]) for m in METRICS[:3]) and
                type(row[METRICS[3]]) in (bool, int, float),
                'All four original metrics at seed4101 required')
        require(row['convnext_top1_source_prediction'] in (0, 1), 'Per-image prediction agreement must be binary')
        grouped[key[1], key[2]].append(row['dinov2_vitl14_cosine'])
    require(seen == expected, 'Incomplete pilot cannot choose finalists')
    result = {}
    for snr in plan.SNRS:
        means = {cid:math.fsum(grouped[snr, cid]) / 100 for cid in candidates}
        ranked = sorted(candidates, key=lambda cid:(-means[cid], cid))
        whole = policy['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id']
        chosen = plan.full_shortlist(ranked, whole)
        result[str(snr)] = dict(ranking=ranked, dinov2_vitl14_means=means, finalists=list(chosen),
                               original_final_whole_winner=whole, whole_winner_retained=True)
    return dict(status='PILOT_SHORTLIST_ONLY_FULL_CALIBRATION_REQUIRED', snrs=result,
                holdout_used_for_selection=False, convnext_used_for_selection=False,
                final_strategy_frozen=False, automatic_successor=False)
