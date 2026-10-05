"""Registered two-worker H initial_true200 CPU reception, never image scoring.

No invocation occurs on import. Real use requires an independently sealed stage
registration, frozen final shortlist, completed source caches and qualified PHY.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import time
import traceback

PHASE = 'initial_true200'
WORKERS = 2
STOP = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf8')
    os.replace(temporary, path)


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Changed bound file: '+str(path))


def stop(*_):
    global STOP
    STOP = True


def source_indices(worker):
    require(type(worker) is int and 0 <= worker < WORKERS, 'Worker must be0 or1')
    return list(range(worker, 200, WORKERS))


def checked_receipt(path, status, *, verify_inputs=False):
    receipt = read(path)
    require(receipt['status'] == status, 'Completion status differs: '+str(path))
    verify(receipt['outputs'])
    if verify_inputs:
        verify(receipt['input_bindings'])
        verify(receipt['source_bindings'])
    return receipt


def check_shortlist(shortlist):
    require(shortlist['status'] == 'H_EXPECTED_PSNR_SHORTLIST_FROZEN'
            and shortlist['ready_for_real_calibration'] is True, 'Final ready shortlist required')
    ids = shortlist['source_ids']
    require(len(ids) == len(set(ids)) == 200 and all(isinstance(x, str) and x for x in ids), 'Wrong fixed source200')
    rows = shortlist['whole_candidates']
    arms = ('H16-R', 'H16-A', 'H64-R', 'H64-A')
    expected = {(arm, snr) for arm in arms for snr in (13, 19)}
    counts, identities, slots = {}, set(), set()
    for row in rows:
        key = row['arm'], row['snr_db']
        require(key in expected and type(row['slot']) is int and 0 <= row['slot'] < 16,
                'Candidate arm/SNR/slot outside registration')
        require((row['candidate_id'], row['snr_db']) not in identities and row['slot'] not in slots,
                'Duplicate candidate/event schedule')
        identities.add((row['candidate_id'], row['snr_db']))
        slots.add(row['slot'])
        counts[key] = counts.get(key, 0)+1
    require(set(counts) == expected and all(1 <= n <= 2 for n in counts.values()),
            'All four arms/two SNR need one or two frozen policies each')
    require(len(rows)*200*3*2 <= 19200, 'Initial payload packet upper bound exceeds19200')
    return sorted(rows, key=lambda row: row['slot'])


def load_registered(config_path):
    """Read-only complete original-byte binding checks, before PHY imports."""
    cfg = read(config_path)
    reg = read(cfg['registration'])
    require(reg['status'] == 'H_EXECUTION_REVISION_REGISTERED' and reg['branch'] == 'H', 'Unregistered H stage')
    require(PHASE in reg['allowed_stage_ids'], 'This execution revision does not authorize initial_true200')
    verify(reg['input_bindings'])
    verify(reg['source_bindings'])
    bound = dict(reg['input_bindings'], **reg['source_bindings'])
    require(bound.get(str(Path(config_path).absolute())) == sha(config_path), 'Config is not registration-bound')
    for name in ('h_payload_driver.py', 'h_payload_cpu.py'):
        p = str(Path(__file__).absolute().with_name(name))
        require(bound.get(p) == sha(p), 'Payload source missing/changed in registration')
    for name in ('h64_catalog.py', 'h64_phy.py', 'h64_backend.py', 'h64_source.py'):
        p = str(Path(cfg['runtime_dir'])/name)
        require(bound.get(p) == sha(p), 'Frozen H source missing/changed')
    dependencies = [str(Path(cfg['legacy_runtime'])/name) for name in ('ldpc_backend.py', 'uep_phy.py', 'uep_common.py')]
    dependencies += [str(Path(cfg['root'])/'src/var_comm'/name) for name in ('scale_channel.py', 'token_trellis.cpp')]
    for p in dependencies:
        require(bound.get(p) == sha(p), 'Header/backend dependency not bound: '+p)
    for key in ('protocol', 'engineering_contract', 'budget_registration', 'ledger_module', 'catalogue',
                'reference_qualification', 'qualification_completion', 'shortlist', 'prescreen_completion',
                'source_completion', 'S1_completion', 'S1_assets_completion', 'source200'):
        require(bound.get(cfg[key]) == sha(cfg[key]), 'Input absent from execution binding: '+key)
    budget = read(cfg['budget_registration'])
    require(budget['status'] == 'FROZEN' and budget['branch'] == 'H' and budget['total_cap'] == 200000,
            'Wrong original H budget')
    require(budget['phase_limits'] == cfg['phase_limits'] == reg['phase_limits']
            and sum(budget['phase_limits'].values()) == 200000
            and budget['phase_limits'][PHASE] == 20000, 'Full immutable phase limits differ')
    require(cfg['budget_registration_sha256'] == sha(cfg['budget_registration']), 'Budget raw SHA differs')
    require(Path(cfg['ledger']).is_file(), 'Original existing H ledger required; never create replacement')
    protocol = read(cfg['protocol'])
    require(protocol['status'] == 'FROZEN_BEFORE_DATA' and protocol['schema'] == 'H_CODEC_PROTOCOL_V1', 'Wrong frozen protocol')
    qual = checked_receipt(cfg['qualification_completion'], 'H_PHY_QUALIFICATION_PASS')
    verify(qual['source_bindings'])
    require(qual['budget_registration_sha256'] == sha(cfg['budget_registration'])
            and qual['outputs'].get(cfg['catalogue']) == sha(cfg['catalogue']), 'Qualification/catalogue/budget mismatch')
    catalogue = read(cfg['catalogue'])
    prescreen = read(cfg['prescreen_completion'])
    require(prescreen['status'] == 'H_PRESCREEN_COMPLETE_FINAL'
            and prescreen['ready_for_real_calibration'] is True, 'Prescreen is provisional/incomplete')
    verify(prescreen['outputs'])
    verify(prescreen['input_bindings'])
    verify(prescreen['source_bindings'])
    require(prescreen['outputs'].get(cfg['shortlist']) == sha(cfg['shortlist']), 'Final shortlist not sealed')
    shortlist = read(cfg['shortlist'])
    require(shortlist['registration_sha256'] == prescreen['registration_sha256'], 'Shortlist/prescreen identity differs')
    candidates = check_shortlist(shortlist)
    source = checked_receipt(cfg['source_completion'], 'H_SOURCE200_COMPLETE')
    require(source['registration_sha256'] == qual['registration_sha256'] and source['source_count'] == 200
            and source['development_used'] is False and source['holdout_used'] is False, 'Source cache scope differs')
    s1 = checked_receipt(cfg['S1_completion'], 'S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION')
    verify(s1['input_bindings'])
    verify(s1['source_bindings'])
    assets = checked_receipt(cfg['S1_assets_completion'], 'S1_EXPORT_ASSETS_COMPLETE')
    require(assets['registration_sha256'] == s1['registration_sha256'] and assets['source_count'] == 200
            and s1['outputs'].get(cfg['S1_assets_completion']) == sha(cfg['S1_assets_completion']),
            'S1 assets belong to a different execution or root receipt')
    ids = read(cfg['source200'])['source_ids']
    require(ids == shortlist['source_ids'], 'Different source200 order')
    for i in range(200):
        for receipt, directory in ((source, Path(cfg['source_completion']).parent),
                                   (assets, Path(cfg['S1_assets_completion']).parent)):
            cp_path = str(directory/'source_checkpoints'/f'{i:04d}.json')
            require(receipt['outputs'].get(cp_path) == sha(cp_path), 'Source checkpoint missing from completed scope')
            cp = read(cp_path)
            require(cp['source_id'] == ids[i] and cp['source_index'] == i, 'Source checkpoint identity differs')
            verify(cp['outputs'])
    for name, path in (('protocol', cfg['protocol']), ('catalogue', cfg['catalogue']),
                       ('budget_registration', cfg['budget_registration']), ('source200', cfg['source200']),
                       ('qualification_completion', cfg['qualification_completion']),
                       ('source_dir_completion', cfg['source_completion'])):
        require(prescreen['prescreen_basis'][name] == dict(path=path, sha256=sha(path)),
                'Prescreen used different registered basis: '+name)
    contract = read(cfg['engineering_contract'])
    require(contract['status'] == 'H_PAYLOAD_CPU_ENGINEERING_SEALED'
            and 'execution_registration_sha256' not in contract, 'Contract must precede registration without a hash cycle')
    runtime_contract = dict(contract, execution_registration_sha256=sha(cfg['registration']))
    affinity = cfg['cpu_affinities']
    require(len(affinity) == 2 and all(len(pair) == len(set(pair)) == 2 for pair in affinity)
            and len(set(x for pair in affinity for x in pair)) == 4, 'Two disjoint two-core workers required')
    require(0 < cfg['max_worker_seconds'] <= 21600 and cfg['workers'] == WORKERS, 'Unbounded worker scope')
    return dict(cfg=cfg, reg=reg, budget=budget, protocol=protocol, qualification=qual, catalogue=catalogue,
                shortlist=shortlist, candidates=candidates, source=source, assets=assets, contract=runtime_contract)


def configure_cpu(cores, torch_required):
    require(sys.platform.startswith('linux'), 'Registered real payload execution requires Linux')
    require(set(cores).issubset(os.sched_getaffinity(0)), 'Registered affinity unavailable')
    os.sched_setaffinity(0, set(cores))
    os.setpriority(os.PRIO_PROCESS, 0, 15)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    if torch_required:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)


@contextlib.contextmanager
def exclusive(path):
    import fcntl
    with Path(path).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def make_ledger(ctx):
    cfg = ctx['cfg']
    spec = importlib.util.spec_from_file_location('h_initial_registered_budget', cfg['ledger_module'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ledger = module.BudgetLedger(cfg['ledger'], sha(cfg['budget_registration']), 'H', ctx['budget']['phase_limits'])
    require(ledger.binding == sha(cfg['budget_registration']) and ledger.limits == cfg['phase_limits'], 'Ledger binding differs')
    return ledger


def build_runtime(ctx):
    cfg = ctx['cfg']
    for path in (cfg['runtime_dir'], cfg['legacy_runtime'], str(Path(cfg['root'])/'src')):
        sys.path.insert(0, path)
    import h_payload_cpu as core
    from h64_backend import H64Backend
    from uep_phy import Header
    # Validate the core/contract before constructing any real PHY.
    core.validate_contract(ctx['protocol'], ctx['catalogue'], ctx['shortlist'], ctx['contract'],
                           ctx['candidates'][0], ctx['shortlist']['source_ids'][0], 0)
    backend = H64Backend(reference_qualification=read(cfg['reference_qualification']),
                         legacy_backend_path=Path(cfg['legacy_runtime'])/'ldpc_backend.py')
    require(backend.identity == ctx['qualification']['backend_identity'], 'Backend identity differs from qualified implementation')
    for bucket in ctx['catalogue']['buckets']:
        require(bucket['admission'] == 'ADMITTED' and backend.plan(bucket['k'], bucket['n'], bucket['q']) == bucket['layout'],
                'Current actual layout differs from qualification')
    return core, backend, Header(cfg['root'])


def collect_source(ctx, core, backend, header, ledger, source_index, boundary, record):
    cfg = ctx['cfg']
    sid = ctx['shortlist']['source_ids'][source_index]
    source_cp = str(Path(cfg['source_completion']).parent/'source_checkpoints'/f'{source_index:04d}.json')
    s1_cp = str(Path(cfg['S1_assets_completion']).parent/'source_checkpoints'/f'{source_index:04d}.json')
    assets = core.load_source_assets(source_cp, s1_cp,
        source_checkpoint_sha256=ctx['source']['outputs'][source_cp],
        s1_checkpoint_sha256=ctx['assets']['outputs'][s1_cp], source_id=sid, source_index=source_index,
        source_registration_sha256=ctx['source']['registration_sha256'])
    frames = []
    for candidate in ctx['candidates']:
        for seed in (6101, 6102, 6103):
            boundary()
            trace = core.run_frame(protocol=ctx['protocol'], catalogue=ctx['catalogue'], shortlist=ctx['shortlist'],
                contract=ctx['contract'], candidate=candidate, source_id=sid, source_index=source_index,
                noise_seed=seed, scales=assets['scales'], arithmetic_bits=assets['arithmetic_bits'],
                backend=backend, header=header, ledger=ledger)
            record(trace)
            frames.append(trace)
    return dict(status='H_INITIAL_CPU_SOURCE_TRACES', registration_sha256=ctx['contract']['execution_registration_sha256'],
                source_id=sid, source_index=source_index, frames=frames, source_bindings=assets['source_bindings'])


def trace_inventory(ctx, paths, worker=None):
    """Streaming trace validation; keep event checksums, not all decoded vectors."""
    import h_payload_cpu as core
    wanted_sources = source_indices(worker) if worker is not None else list(range(200))
    seen_sources, events, energy, source_count = set(), {}, [], 0
    candidates = {row['slot']: row for row in ctx['candidates']}
    frame_count, gray, pending = 0, 0, 0
    regsha = ctx['contract']['execution_registration_sha256']
    for path in paths:
        source = read(path)
        i = source['source_index']
        require(i in wanted_sources and i not in seen_sources and source['source_id'] == ctx['shortlist']['source_ids'][i]
                and source['registration_sha256'] == regsha and source['status'] == 'H_INITIAL_CPU_SOURCE_TRACES', 'Trace source changed')
        seen_sources.add(i)
        verify(source['source_bindings'])
        keys = set()
        for f in source['frames']:
            slot, seed = f['candidate_slot'], f['noise_seed']
            require(slot in candidates and seed in (6101, 6102, 6103) and (slot, seed) not in keys, 'Duplicate/unknown frame')
            keys.add((slot, seed))
            c = candidates[slot]
            event = f'H:{PHASE}:{c["candidate_id"]}:snr{c["snr_db"]}:src{i:04d}:seed{seed}'
            require(f['source_id'] == source['source_id'] and f['source_index'] == i
                    and f['execution_registration_sha256'] == regsha and f['event_id'] == event
                    and f['arm'] == c['arm'] and f['candidate_id'] == c['candidate_id'] and f['snr_db'] == c['snr_db']
                    and f['public_frame_counter'] == core.frame_counter(slot, i, seed)
                    and f['stage'] == PHASE and f['total_symbols'] == 1024 and f['frame_normalized'] is False,
                    'Frame identity/resource changed')
            names = ['header'] + (['body'] if f['rx']['body'] is not None else [])
            require(f['logical_packet_events'] == len(names) and f['packet_event_ids'] == [event+':'+name for name in names],
                    'Logical event references changed')
            for name in names:
                eid = event+':'+name
                require(eid not in events, 'Duplicate actual event reference')
                events[eid] = dict(kind=name, result_sha256=digest(f['rx'][name]))
            energy.append(float(f['total_energy']))
            gray += int(f['gray'] is True)
            pending += int(f['rx_source_status'] == 'ARITHMETIC_CANONICAL_RX_REQUIRED')
            frame_count += 1
        require(keys == {(slot, seed) for slot in candidates for seed in (6101, 6102, 6103)}, 'Incomplete source frame coverage')
        source_count += 1
    require(seen_sources == set(wanted_sources), 'Source coverage incomplete')
    require(frame_count == len(wanted_sources)*len(candidates)*3, 'Incorrect logical frame count')
    import numpy as np
    a = np.asarray(energy, dtype=np.float64)
    require(np.isfinite(a).all() and np.all(a > 0), 'Invalid actual frame energy')
    summary = dict(source_count=source_count, frame_count=frame_count, logical_packet_events=len(events),
                   wire_gray_frames=gray, arithmetic_pending_frames=pending,
                   total_symbols_per_frame=1024, energy=dict(mean=float(a.mean()), sd_population=float(a.std()),
                       min=float(a.min()), max=float(a.max()), quantiles={str(q): float(np.quantile(a, q)) for q in (0.01, .5, .99)}))
    return events, summary


def audit_ledger(path, expected, worker=None):
    seen = set()
    with contextlib.closing(sqlite3.connect(str(path), timeout=60)) as conn:
        for eid, kind, status, result, checksum in conn.execute(
                'SELECT event_id,kind,status,result,result_sha FROM events WHERE phase=?', (PHASE,)):
            if worker is not None and eid not in expected:
                match = re.search(r':src(\d{4}):seed\d+:', eid)
                require(match is not None and int(match.group(1)) % 2 != worker, 'Unexpected event in worker source partition')
                continue
            require(eid in expected and eid not in seen and status == 'COMPLETE', 'Unexpected/incomplete paid event')
            require(kind == expected[eid]['kind'] and checksum == expected[eid]['result_sha256']
                    and digest(json.loads(result)) == checksum, 'Trace differs from actual paid receiver result')
            seen.add(eid)
        require(seen == set(expected), 'Missing paid events for trace')
        if worker is None:
            count = conn.execute('SELECT charged FROM counters WHERE phase=?', (PHASE,)).fetchone()
            require(count is not None and count[0] == len(expected) <= 19200, 'Paid counter differs or exceeds registered frame bound')
    return dict(status='CPU_LEDGER_TRACE_MATCH', paid_events=len(seen), event_ids_sha256=digest(sorted(seen)),
                all_complete=True, actual_charges_source='original immutable H ledger')


def run_worker(config_path, worker):
    indices = source_indices(worker)
    ctx = load_registered(config_path)
    cfg = ctx['cfg']
    out = Path(cfg['out'])/f'worker_{worker}'
    out.mkdir(parents=True, exist_ok=True)
    regsha = ctx['contract']['execution_registration_sha256']
    require(not (out/'failure.json').exists(), 'Previous failure preserved; no automatic retry')
    with exclusive(out/'worker.lock'):
        try:
            if (out/'completion.json').exists():
                done = checked_receipt(out/'completion.json', 'H_INITIAL_TRUE200_CPU_WORKER_COMPLETE')
                require(done['registration_sha256'] == regsha and done['worker_index'] == worker, 'Wrong completed worker')
                return done
            require(not (out/'attempt.json').exists(), 'Partial attempt requires independent recovery registration')
            configure_cpu(cfg['cpu_affinities'][worker], True)
            core, backend, header = build_runtime(ctx)
            ledger = make_ledger(ctx)
            started = time.monotonic()
            atomic(out/'attempt.json', dict(status='STARTED', registration_sha256=regsha, worker_index=worker,
                                          source_indices=indices, worker_identity=ledger.worker, automatic_retry=False))
            outputs, trace_paths = {}, []
            def boundary():
                require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(), 'STOP at packet boundary')
                require(time.monotonic()-started <= cfg['max_worker_seconds'], 'Registered worker time exhausted')
            with (out/'traces.journal.jsonl').open('x', encoding='utf8', newline='\n') as journal:
                def record(trace):
                    journal.write(canonical(trace)+'\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                for i in indices:
                    source = collect_source(ctx, core, backend, header, ledger, i, boundary, record)
                    path = out/'traces'/f'{i:04d}.json'
                    atomic(path, source)
                    checkpoint = out/'source_checkpoints'/f'{i:04d}.json'
                    atomic(checkpoint, dict(status='H_INITIAL_CPU_SOURCE_COMPLETE', registration_sha256=regsha,
                        source_index=i, source_id=source['source_id'], frame_count=len(source['frames']),
                        outputs={str(path): sha(path)}, images_scored=False))
                    outputs.update({str(path): sha(path), str(checkpoint): sha(checkpoint)})
                    trace_paths.append(path)
                    atomic(out/'status.json', dict(status='RUNNING', registration_sha256=regsha, worker_index=worker,
                        completed_sources=len(trace_paths), total_sources=100, elapsed_seconds=time.monotonic()-started))
            expected, summary = trace_inventory(ctx, trace_paths, worker)
            audit = audit_ledger(cfg['ledger'], expected, worker)
            verify(ctx['reg']['input_bindings'])
            verify(ctx['reg']['source_bindings'])
            outputs[str(out/'traces.journal.jsonl')] = sha(out/'traces.journal.jsonl')
            done = dict(status='H_INITIAL_TRUE200_CPU_WORKER_COMPLETE', registration_sha256=regsha,
                worker_index=worker, workers=2, source_indices=indices, budget_registration_sha256=sha(cfg['budget_registration']),
                worker_identity=ledger.worker, driver_config_sha256=sha(config_path),
                shortlist_sha256=sha(cfg['shortlist']), ledger_audit=audit, **summary, outputs=outputs,
                input_bindings=ctx['reg']['input_bindings'], source_bindings=ctx['reg']['source_bindings'],
                images_scored=False, source_decode_complete=False, arithmetic_source_decode_complete=False, GPU_used=False,
                development_used=False, scope='CPU actual reception only; arithmetic canonical decoding still required')
            atomic(out/'completion.json', done)
            atomic(out/'status.json', done)
            return done
        except BaseException as error:
            if not (out/'failure.json').exists():
                atomic(out/'failure.json', dict(status='FAILED', registration_sha256=regsha, worker_index=worker,
                    error=repr(error), traceback=traceback.format_exc(), evidence_preserved=True,
                    ledger_never_refunded=True, automatic_retry=False))
            raise


def run_merge(config_path):
    ctx = load_registered(config_path)
    cfg = ctx['cfg']
    out = Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'merge_failure.json').exists(), 'Prior merge failure preserved')
    regsha = ctx['contract']['execution_registration_sha256']
    with exclusive(out/'merge.lock'):
        try:
            if (out/'completion.json').exists():
                done = checked_receipt(out/'completion.json', 'H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE')
                require(done['registration_sha256'] == regsha, 'Different completed registration')
                return done
            configure_cpu(cfg['cpu_affinities'][0], False)
            sys.path.insert(0, cfg['runtime_dir'])
            require(not STOP and not Path(cfg['stop_file']).exists(), 'STOP before merge')
            paths, outputs, identities = [], {}, []
            for worker in range(WORKERS):
                directory = out/f'worker_{worker}'
                require(not (directory/'failure.json').exists(), 'Failed worker cannot be merged')
                cp = directory/'completion.json'
                done = checked_receipt(cp, 'H_INITIAL_TRUE200_CPU_WORKER_COMPLETE')
                require(done['registration_sha256'] == regsha and done['worker_index'] == worker and done['workers'] == 2
                        and done['source_indices'] == source_indices(worker)
                        and done['budget_registration_sha256'] == sha(cfg['budget_registration'])
                        and done['driver_config_sha256'] == sha(config_path)
                        and done['shortlist_sha256'] == sha(cfg['shortlist'])
                        and done['images_scored'] is False, 'Worker scope/binding changed')
                for i in source_indices(worker):
                    path = directory/'traces'/f'{i:04d}.json'
                    require(done['outputs'].get(str(path)) == sha(path), 'Required trace not sealed')
                    paths.append(path)
                outputs.update(done['outputs'])
                outputs[str(cp)] = sha(cp)
                identities.append(done['worker_identity'])
            expected, summary = trace_inventory(ctx, paths)
            audit = audit_ledger(cfg['ledger'], expected)
            ledger = make_ledger(ctx)
            snapshot = ledger.assert_quiescent()
            require(snapshot['phase_charged'][PHASE] == len(expected), 'Counter changed during merge')
            verify(ctx['reg']['input_bindings'])
            verify(ctx['reg']['source_bindings'])
            done = dict(status='H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE', registration_sha256=regsha,
                budget_registration_sha256=sha(cfg['budget_registration']), shortlist_sha256=sha(cfg['shortlist']),
                driver_config_sha256=sha(config_path), worker_identities=identities,
                source_completion_sha256=sha(cfg['source_completion']), workers=2, **summary,
                maximum_packet_calls=19200, ledger=snapshot, ledger_audit=audit, outputs=outputs,
                input_bindings=ctx['reg']['input_bindings'], source_bindings=ctx['reg']['source_bindings'],
                images_scored=False, source_decode_complete=False, arithmetic_source_decode_complete=False, GPU_used=False,
                workers_exit_verified=False, requires_owner_worker_exit_receipt_before_visual_stage=True,
                development_used=False, scope='CPU actual reception only; no image score or H system conclusion')
            atomic(out/'completion.json', done)
            return done
        except BaseException as error:
            if not (out/'merge_failure.json').exists():
                atomic(out/'merge_failure.json', dict(status='FAILED', registration_sha256=regsha,
                    error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('worker', 'merge'), required=True)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.stage == 'worker':
        require(args.worker_index is not None, 'Worker index required')
        result = run_worker(args.config, args.worker_index)
    else:
        require(args.worker_index is None, 'Merge has no worker index')
        result = run_merge(args.config)
    print(json.dumps({key: result[key] for key in ('status', 'registration_sha256')}))


if __name__ == '__main__':
    main()
