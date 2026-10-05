"""Independent bounded H refinement. No cell selection, source image or GPU work.

Execution needs a new immutable registration; this module never schedules itself.
Only complete, paid decode events can be merged. Failed evidence is never erased.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

from h_coarse_driver import (atomic, canonical, cpu_configure, csv_write, digest,
                            exclusive, make_ledger, read, require, sha,
                            summarize_rows, verify)
from h_prescreen import validate_body, validate_catalog

ADDITIONAL = 6144
ORIGINAL = 2048
SNRS = (13, 19)
SEED = 2026100602
SESSION = 'H_REFINE_R1'
STOP = False
COUNT_FIELDS = ('correct', 'rejected', 'undetected', 'crc_accepted', 'parser_invalid')


def stop(*_):
    global STOP
    STOP = True


def generator(protocol, layout_id, snr, index, stream):
    import hashlib
    import numpy as np
    require(snr in SNRS and type(index) is int and ORIGINAL <= index < ORIGINAL+ADDITIONAL
            and stream in ('payload', 'noise'), 'Unregistered refinement random identity')
    seed = int.from_bytes(hashlib.sha256(canonical([protocol, 'refine', SEED, str(layout_id),
                                int(snr), index, stream]).encode()).digest()[:16], 'little')
    return np.random.Generator(np.random.PCG64(seed))


def public_counter(layout_index, snr, index):
    require(type(layout_index) is int and 0 <= layout_index < 8 and snr in SNRS
            and type(index) is int and ORIGINAL <= index < ORIGINAL+ADDITIONAL,
            'Invalid public refinement counter')
    return SEED*100000+100000+layout_index*len(SNRS)*ADDITIONAL+SNRS.index(snr)*ADDITIONAL+(index-ORIGINAL)


def event_id(layout_id, snr, index):
    return f'HREFINE:{layout_id}:{snr}:{index}'


def measure_cell(backend, ledger, bucket, layout_index, snr, boundary,
                 on_packet=lambda row: None, blocks=ADDITIONAL):
    """Only blocks is injectable for unit tests; production never configures it."""
    import numpy as np
    from h64_catalog import PROTOCOL, make_profile
    from h64_phy import transmit_body, receive_body, pack_body, array_sha
    require(type(blocks) is int and 0 < blocks <= ADDITIONAL, 'Bad refinement test block count')
    p = make_profile(bucket, 6, 0, 'arithmetic', ['REFINE_RANDOM_MESSAGE_PHY_ONLY'])
    lid = bucket['layout']['layout_id']; rows = []
    for index in range(ORIGINAL, ORIGINAL+blocks):
        boundary()
        payload = generator(PROTOCOL, lid, snr, index, 'payload').integers(0, 2, p['source_capacity'], dtype=np.uint8)
        counter = public_counter(layout_index, snr, index)
        wave, meta = transmit_body(backend, payload, p, counter, SESSION)
        noise = generator(PROTOCOL, lid, snr, index, 'noise').standard_normal(wave.shape).astype(np.float32)
        received = (wave+noise*(10**(-float(snr)/20))).astype(np.float32)
        eid = event_id(lid, snr, index)
        event = receive_body(backend, received, p, snr, counter, ledger, 'refine', eid, SESSION)
        actual = np.asarray(event['decoded_bits'], dtype=np.uint8)
        accepted = bool(event['crc_accepted'])
        correct = bool(accepted and np.array_equal(actual, pack_body(payload, p)))
        row = dict(index=index, event_id=eid, public_frame_counter=counter,
                   correct=int(correct), rejected=int(not accepted), undetected=int(accepted and not correct),
                   crc_accepted=int(accepted), parser_invalid=int(accepted and not event['parser_accepted']),
                   body_energy=meta['actual_energy'], noise_sha256=array_sha(noise),
                   received_sha256=array_sha(received), decoded_sha256=array_sha(actual))
        on_packet(row); rows.append(row)
    return rows, dict(layout_id=lid, layout_index=layout_index, phy_profile_key=p['profile_key'],
        resource_id=bucket['resource_id'], q=bucket['q'], nominal_rate=bucket['nominal_rate'],
        k=bucket['k'], n=bucket['n'], body_symbols=956, snr_db=snr,
        source_capacity=p['source_capacity'], seed=SEED, stage_namespace='refine',
        first_index=ORIGINAL, last_index=ORIGINAL+blocks-1, **summarize_rows(rows))


def validate_requests(request, cat, coarse):
    require(request['status'] == 'H_REFINEMENT_REQUEST_FROZEN', 'Request was not frozen')
    rows = request['requests']
    require(1 <= len(rows) <= 2 and request['requested_body_calls'] == len(rows)*ADDITIONAL
            and request['maximum_additional_body_calls'] == 12288, 'Wrong bounded refinement request')
    require(len({(x['layout_id'], x['snr_db']) for x in rows}) == len(rows), 'Duplicate requested cell')
    buckets = {b['layout']['layout_id']: (i, b) for i, b in enumerate(cat['buckets'])}
    answer = []
    for rank, row in enumerate(rows, 1):
        require(row['refinement_rank'] == rank and row['layout_id'] in buckets and row['snr_db'] in SNRS,
                'Unknown layout/SNR/rank in request')
        i, b = buckets[row['layout_id']]
        require(row['q'] == b['q'] and row['nominal_rate'] == b['nominal_rate']
                and row['additional_body_calls'] == ADDITIONAL and row['original_trials'] == ORIGINAL
                and row['required_merged_trials'] == ORIGINAL+ADDITIONAL, 'Request changed physical resource or count')
        old = coarse[row['layout_id'], row['snr_db']]
        require(.01 <= old['empirical_BLER'] <= .99, 'Requested cell violates registered eligibility')
        answer.append(dict(request=row, layout_index=i, bucket=b, old=old))
    return answer


def load_registered(config_path):
    cfg = read(config_path); reg = read(cfg['registration'])
    require(reg['status'] == 'H_EXECUTION_REVISION_REGISTERED' and reg['branch'] == 'H', 'Refinement is not registered')
    verify(reg['input_bindings']); verify(reg['source_bindings'])
    bound = dict(reg['input_bindings'], **reg['source_bindings'])
    require(bound.get(str(Path(config_path).absolute())) == sha(config_path), 'Config not registered')
    for name in ('h_refine_driver.py', 'h_coarse_driver.py', 'h_prescreen.py', 'h64_catalog.py', 'h64_backend.py', 'h64_phy.py'):
        path = str(Path(__file__).absolute().with_name(name))
        require(bound.get(path) == sha(path), 'Executing module not bound: '+path)
    for key in ('protocol', 'engineering_contract', 'budget_registration', 'catalogue', 'ledger_module',
                'reference_qualification', 'qualification_completion', 'prescreen_completion',
                'refinement_requests', 'coarse_completion'):
        require(bound.get(cfg[key]) == sha(cfg[key]), 'Required input missing binding: '+key)
    old_backend = str(Path(cfg['legacy_runtime'])/'ldpc_backend.py')
    require(bound.get(old_backend) == sha(old_backend), 'Original adapter not bound')
    protocol = read(cfg['protocol']); contract = read(cfg['engineering_contract'])
    require(protocol['status'] == 'FROZEN_BEFORE_DATA' and protocol['main_snrs_db'] == list(SNRS)
            and protocol['population']['coarse_seed'] == SEED and protocol['N'] == 1024,
            'Wrong original frozen protocol')
    require(contract['schema'] == 'H_REFINEMENT_ENGINEERING_CONTRACT_V1'
            and contract['additional_packets_per_cell'] == ADDITIONAL
            and contract['old_packets_per_cell'] == ORIGINAL and contract['random_seed'] == SEED
            and contract['public_scrambling_session'] == SESSION,
            'Engineering contract differs from executing refinement')
    budget = read(cfg['budget_registration'])
    require(budget['status'] == 'FROZEN' and budget['branch'] == 'H' and budget['total_cap'] == 200000
            and budget['phase_limits']['refine'] == 12288, 'Wrong immutable H refine budget')
    qual = read(cfg['qualification_completion']); verify(qual['outputs']); verify(qual['source_bindings'])
    require(qual['status'] == 'H_PHY_QUALIFICATION_PASS' and qual['budget_registration_sha256'] == sha(cfg['budget_registration'])
            and qual['outputs'].get(cfg['catalogue']) == sha(cfg['catalogue']), 'Qualified catalogue/budget mismatch')
    cat = read(cfg['catalogue']); validate_catalog(cat)
    coarse_done = read(cfg['coarse_completion']); verify(coarse_done['outputs']); verify(coarse_done['source_bindings'])
    require(coarse_done['status'] == 'H_COARSE_COMPLETE' and coarse_done['body_decode_events'] == 49152
            and coarse_done['budget_registration_sha256'] == sha(cfg['budget_registration'])
            and coarse_done['qualification_completion_sha256'] == sha(cfg['qualification_completion']),
            'Original coarse evidence incomplete or incompatible')
    table_path = str(Path(cfg['coarse_completion']).parent/'bler_counts.json')
    require(coarse_done['outputs'].get(table_path) == sha(table_path), 'Original coarse counts not bound')
    coarse = validate_body(read(table_path), cat)
    previous = read(cfg['prescreen_completion']); verify(previous['outputs']); verify(previous['input_bindings']); verify(previous['source_bindings'])
    require(previous['status'] == 'H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED' and previous['phase'] == 'coarse'
            and previous['ready_for_real_calibration'] is False
            and previous['outputs'].get(cfg['refinement_requests']) == sha(cfg['refinement_requests']),
            'Refinement requires a completed provisional coarse prescreen')
    for name, key in (('protocol', 'protocol'), ('catalogue', 'catalogue'), ('budget_registration', 'budget_registration'),
                      ('qualification_completion', 'qualification_completion'), ('coarse_dir_completion', 'coarse_completion')):
        require(previous['prescreen_basis'][name] == dict(path=cfg[key], sha256=sha(cfg[key])),
                'Refinement changed the previous screening basis: '+name)
    requests = read(cfg['refinement_requests'])
    require(requests['registration_sha256'] == previous['registration_sha256'], 'Request registration differs from prescreen')
    cells = validate_requests(requests, cat, coarse)
    affinity = cfg['cpu_affinity']
    require(len(affinity) == len(cells) and all(len(x) == len(set(x)) == 2 for x in affinity)
            and len(set(c for pair in affinity for c in pair)) == 2*len(cells), 'Each requested cell needs one separate2-core worker')
    require(0 < cfg['max_worker_seconds'] <= 7200, 'Unbounded refinement timeout')
    return cfg, reg, budget, qual, cat, coarse_done, cells


def read_packets(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('index', 'public_frame_counter')+COUNT_FIELDS:
            row[key] = int(row[key])
        row['body_energy'] = float(row['body_energy'])
        require(row['correct']+row['rejected']+row['undetected'] == 1
                and row['crc_accepted'] == row['correct']+row['undetected']
                and all(row[k] in (0, 1) for k in COUNT_FIELDS)
                and row['parser_invalid'] <= row['undetected'], 'Invalid actual packet classification')
    return rows


def merge_cell(original_rows, added_rows, old, additional, *, original_count=ORIGINAL, additional_count=ADDITIONAL):
    """Keep original rows, compute exact combined counts and energy quantiles."""
    from h_coarse_driver import public_counter as coarse_counter
    lid, snr, i = old['layout_id'], old['snr_db'], old['layout_index']
    require(len(original_rows) == original_count and len(added_rows) == additional_count, 'Wrong old/new sample count')
    for j, row in enumerate(original_rows):
        require(row['index'] == j and row['event_id'] == f'HCOARSE:{lid}:{snr}:{j}'
                and row['public_frame_counter'] == coarse_counter(i, snr, j), 'Original sample identity differs')
    for j, row in enumerate(added_rows, ORIGINAL):
        require(row['index'] == j and row['event_id'] == event_id(lid, snr, j)
                and row['public_frame_counter'] == public_counter(i, snr, j), 'Additional sample identity differs')
    require(not ({r['event_id'] for r in original_rows} & {r['event_id'] for r in added_rows}), 'Old observations reused as new')
    old_summary, added_summary = summarize_rows(original_rows), summarize_rows(added_rows)
    for fields, summary, label in ((old, old_summary, 'Original'), (additional, added_summary, 'Additional')):
        require(all(fields[k] == summary[k] for k in ('trials',)+COUNT_FIELDS), label+' CSV and sealed counts differ')
        require(abs(fields['body_energy']['mean']-summary['body_energy']['mean']) < 1e-10
                and abs(fields['body_energy']['sd_population']-summary['body_energy']['sd_population']) < 1e-10,
                label+' energy evidence differs')
    require(all(old[k] == additional[k] for k in ('layout_id', 'layout_index', 'resource_id', 'q', 'nominal_rate',
             'k', 'n', 'snr_db', 'source_capacity')), 'Refinement altered the physical layout')
    result = {k: v for k, v in old.items() if k not in old_summary}
    result.update(summarize_rows(original_rows+added_rows))
    result.update(original_counts={k: old[k] for k in ('trials', 'correct', 'rejected', 'undetected')},
                  additional_counts={k: additional[k] for k in ('trials', 'correct', 'rejected', 'undetected')},
                  original_samples_unchanged=True, original_observation_indices=[0, original_count-1],
                  additional_observation_indices=[ORIGINAL, ORIGINAL+additional_count-1],
                  additional_random_namespace='refine', public_scrambling_session=SESSION)
    return result


def audit_ledger(path, cells, selected_index=None, blocks=ADDITIONAL):
    """A worker audits only itself; merge rejects any unrequested refinement event."""
    chosen = cells if selected_index is None else [cells[selected_index]]
    prefixes = tuple(f'HREFINE:{x["old"]["layout_id"]}:{x["old"]["snr_db"]}:' for x in chosen)
    wanted = {event_id(x['old']['layout_id'], x['old']['snr_db'], j)
              for x in chosen for j in range(ORIGINAL, ORIGINAL+blocks)}
    by_event = {event_id(x['old']['layout_id'], x['old']['snr_db'], j): (x, j)
                for x in chosen for j in range(ORIGINAL, ORIGINAL+blocks)}
    seen = set(); proofs = {}
    with contextlib.closing(sqlite3.connect(str(path), timeout=60)) as conn:
        for eid, phase, kind, status, encoded, checksum in conn.execute(
                "SELECT event_id,phase,kind,status,result,result_sha FROM events WHERE phase='refine'"):
            if selected_index is not None and not eid.startswith(prefixes):
                continue
            require(eid in wanted and eid not in seen and (phase, kind, status) == ('refine', 'body', 'COMPLETE'),
                    'Unexpected/unresolved refinement ledger event')
            value = json.loads(encoded)
            require(digest(value) == checksum, 'Actual decoded ledger evidence changed')
            seen.add(eid)
            # Compare measured packet records with the paid actual decoded vector.
            import numpy as np
            actual = np.asarray(value['decoded_bits'], dtype=np.uint8)
            from h64_catalog import PROTOCOL, make_profile
            from h64_phy import array_sha, pack_body
            cell, index = by_event[eid]
            profile = make_profile(cell['bucket'], 6, 0, 'arithmetic', ['REFINE_RANDOM_MESSAGE_PHY_ONLY'])
            payload = generator(PROTOCOL, cell['old']['layout_id'], cell['old']['snr_db'], index, 'payload').integers(
                0, 2, profile['source_capacity'], dtype=np.uint8)
            correct = bool(value['crc_accepted'] and np.array_equal(actual, pack_body(payload, profile)))
            proofs[eid] = (bool(value['crc_accepted']), bool(value['parser_accepted']), array_sha(actual), correct)
    require(seen == wanted, 'Refinement ledger coverage incomplete')
    return dict(event_count=len(seen), event_ids_sha256=digest(sorted(seen)), all_complete=True), proofs


def check_packet_proofs(rows, proofs):
    require(set(proofs) == {r['event_id'] for r in rows}, 'Packet/ledger identities differ')
    for row in rows:
        accepted, parsed, decoded, correct = proofs[row['event_id']]
        require(bool(row['crc_accepted']) == accepted and row['decoded_sha256'] == decoded
                and bool(row['parser_invalid']) == (accepted and not parsed)
                and row['correct'] == int(correct) and row['undetected'] == int(accepted and not correct)
                and row['rejected'] == int(not accepted), 'Packet record disagrees with actual paid result')


def run_worker(config_path, worker):
    cfg, reg, budget, qual, cat, old_done, cells = load_registered(config_path)
    require(type(worker) is int and 0 <= worker < len(cells), 'Worker not in the frozen cell set')
    cell = cells[worker]; b = cell['bucket']; i = cell['layout_index']; snr = cell['old']['snr_db']
    out = Path(cfg['out'])/f'worker_{worker}'; out.mkdir(parents=True, exist_ok=True)
    regsha = sha(cfg['registration']); start = time.monotonic()
    require(not (out/'failure.json').exists(), 'Existing refinement failure requires independent diagnosis')
    with exclusive(out/'worker.lock'):
        try:
            if (out/'completion.json').exists():
                done = read(out/'completion.json'); verify(done['outputs'])
                require(done['status'] == 'H_REFINEMENT_WORKER_COMPLETE' and done['registration_sha256'] == regsha
                        and done['worker_index'] == worker, 'Wrong prior completion')
                return done
            # No implicit resume after a signal/kill between paid calls and final seal.
            require(not (out/'attempt.json').exists(), 'Partial attempt preserved; independent recovery registration required')
            cpu_configure(cfg['cpu_affinity'][worker], True)
            ledger = make_ledger(cfg, budget)
            from h64_backend import H64Backend
            backend = H64Backend(reference_qualification=read(cfg['reference_qualification']),
                                 legacy_backend_path=Path(cfg['legacy_runtime'])/'ldpc_backend.py')
            require(backend.identity == qual['backend_identity'] and backend.plan(b['k'], b['n'], b['q']) == b['layout'],
                    'Current backend/layout differs from qualification')
            atomic(out/'attempt.json', dict(status='STARTED', registration_sha256=regsha,
                   worker_index=worker, cell=cell['request'], automatic_retry=False))
            def boundary():
                require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),
                        'Safe packet-boundary STOP requested')
                require(time.monotonic()-start < cfg['max_worker_seconds'], 'Registered refinement time exhausted')
            with (out/'packets.journal.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
                def record(row):
                    journal.write(canonical(row)+'\n')
                    count = row['index']-ORIGINAL+1
                    if count % 64 == 0:
                        journal.flush(); os.fsync(journal.fileno())
                        atomic(out/'status.json', dict(status='RUNNING', completed_packets=count,
                               total_packets=ADDITIONAL, worker_index=worker, registration_sha256=regsha,
                               elapsed_seconds=time.monotonic()-start))
                rows, summary = measure_cell(backend, ledger, b, i, snr, boundary, record)
                journal.flush(); os.fsync(journal.fileno())
            proof, actual = audit_ledger(cfg['ledger'], cells, worker)
            check_packet_proofs(rows, actual)
            csv_write(out/'packets.csv', rows); atomic(out/'additional_summary.json', summary)
            verify(reg['input_bindings']); verify(reg['source_bindings'])
            done = dict(status='H_REFINEMENT_WORKER_COMPLETE', registration_sha256=regsha,
                        budget_registration_sha256=sha(cfg['budget_registration']), worker_index=worker,
                        workers=len(cells), layout_id=b['layout']['layout_id'], layout_index=i, snr_db=snr,
                        additional_body_decode_events=ADDITIONAL, header_decode_events=0,
                        refinement_requests_sha256=sha(cfg['refinement_requests']), ledger_audit=proof,
                        first_index=ORIGINAL, last_index=ORIGINAL+ADDITIONAL-1,
                        GPU_used=False, source_images_read=False, development_used=False,
                        source_bindings=reg['source_bindings'],
                        outputs={str(out/n): sha(out/n) for n in ('packets.csv', 'additional_summary.json', 'packets.journal.jsonl')})
            atomic(out/'completion.json', done); atomic(out/'status.json', done)
            return done
        except BaseException as exc:
            if not (out/'failure.json').exists():
                atomic(out/'failure.json', dict(status='FAILED', registration_sha256=regsha,
                       worker_index=worker, error=repr(exc), traceback=traceback.format_exc(),
                       existing_measurements_preserved=True, ledger_never_refunded=True, automatic_retry=False))
            raise


def run_merge(config_path):
    cfg, reg, budget, qual, cat, coarse_done, cells = load_registered(config_path)
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True); regsha = sha(cfg['registration'])
    require(not (out/'merge_failure.json').exists(), 'Prior merge failure retained')
    with exclusive(out/'merge.lock'):
        try:
            if (out/'completion.json').exists():
                done = read(out/'completion.json'); verify(done['outputs'])
                require(done['status'] == 'H_REFINEMENT_COMPLETE' and done['registration_sha256'] == regsha,
                        'Existing refinement completion differs')
                return done
            cpu_configure(cfg['cpu_affinity'][0], False)
            require(not STOP and not Path(cfg['stop_file']).exists(), 'STOP before merge')
            output_bindings = {}; input_receipts = {}; merged = []
            for worker, cell in enumerate(cells):
                directory = out/f'worker_{worker}'
                require(not (directory/'failure.json').exists(), 'Failed worker cannot be merged')
                done_path = directory/'completion.json'; done = read(done_path); verify(done['outputs'])
                i, old = cell['layout_index'], cell['old']; snr = old['snr_db']
                require(done['status'] == 'H_REFINEMENT_WORKER_COMPLETE' and done['registration_sha256'] == regsha
                        and done['worker_index'] == worker and done['workers'] == len(cells)
                        and done['layout_id'] == old['layout_id'] and done['snr_db'] == snr
                        and done['additional_body_decode_events'] == ADDITIONAL and done['header_decode_events'] == 0
                        and done['refinement_requests_sha256'] == sha(cfg['refinement_requests']), 'Worker receipt differs')
                old_directory = Path(cfg['coarse_completion']).parent/f'worker_{i//4}'/'cells'/f'layout{i}_snr{snr}'
                old_packets, old_summary = old_directory/'packets.csv', old_directory/'summary.json'
                for p in (old_packets, old_summary):
                    require(coarse_done['outputs'].get(str(p)) == sha(p), 'Original cell output not sealed by coarse completion')
                    input_receipts[str(p)] = sha(p)
                require(read(old_summary) == old, 'Original merged coarse table and cell summary differ')
                original_rows, added_rows = read_packets(old_packets), read_packets(directory/'packets.csv')
                proof, actual = audit_ledger(cfg['ledger'], cells, worker)
                check_packet_proofs(added_rows, actual)
                value = merge_cell(original_rows, added_rows, old, read(directory/'additional_summary.json'))
                value.update(original_packets_sha256=sha(old_packets), original_summary_sha256=sha(old_summary),
                             additional_packets_sha256=sha(directory/'packets.csv'),
                             additional_summary_sha256=sha(directory/'additional_summary.json'))
                merged.append(value); output_bindings.update(done['outputs'])
                output_bindings[str(done_path)] = sha(done_path); input_receipts[str(done_path)] = sha(done_path)
            audit, _ = audit_ledger(cfg['ledger'], cells)
            ledger = make_ledger(cfg, budget); snapshot = ledger.assert_quiescent()
            require(snapshot['phase_charged']['refine'] == len(cells)*ADDITIONAL,
                    'Refine paid count differs from the frozen cell set')
            table = out/'refined_cells.json'; atomic(table, merged)
            output_bindings[str(table)] = sha(table)
            flat = [{k: r[k] for k in ('layout_id', 'layout_index', 'q', 'nominal_rate', 'snr_db', 'trials',
                       'correct', 'rejected', 'undetected', 'crc_accepted', 'parser_invalid', 'empirical_BLER')} for r in merged]
            csv_write(out/'refined_cells.csv', flat); output_bindings[str(out/'refined_cells.csv')] = sha(out/'refined_cells.csv')
            verify(reg['input_bindings']); verify(reg['source_bindings'])
            done = dict(status='H_REFINEMENT_COMPLETE', registration_sha256=regsha,
                        budget_registration_sha256=sha(cfg['budget_registration']),
                        refinement_requests_sha256=sha(cfg['refinement_requests']),
                        prescreen_completion_sha256=sha(cfg['prescreen_completion']),
                        coarse_completion_sha256=sha(cfg['coarse_completion']),
                        qualification_completion_sha256=sha(cfg['qualification_completion']),
                        cells=len(cells), additional_body_decode_events=len(cells)*ADDITIONAL,
                        original_body_observations=len(cells)*ORIGINAL, merged_trials_per_cell=8192,
                        header_decode_events=0, ledger=snapshot, ledger_audit=audit,
                        GPU_used=False, source_images_read=False, development_used=False,
                        outputs=output_bindings, input_receipts=input_receipts,
                        source_bindings=reg['source_bindings'], input_bindings=reg['input_bindings'],
                        old_coarse_evidence_modified=False, automatic_further_refinement=False)
            atomic(out/'completion.json', done)
            return done
        except BaseException as exc:
            if not (out/'merge_failure.json').exists():
                atomic(out/'merge_failure.json', dict(status='FAILED', registration_sha256=regsha,
                       error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
            raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('worker', 'merge'), required=True)
    p.add_argument('--worker-index', type=int)
    args = p.parse_args(); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if args.stage == 'worker':
        require(args.worker_index is not None, 'Worker index required')
        value = run_worker(args.config, args.worker_index)
    else:
        require(args.worker_index is None, 'Merge has no worker index')
        value = run_merge(args.config)
    print(json.dumps({k: value[k] for k in ('status', 'registration_sha256')}))


if __name__ == '__main__':
    main()
