"""Read-only finite shortlist from the registered CPU table and clean200.

No PHY, GPU, development or holdout access. Original artifacts remain intact.
"""
import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import raw64_selection as selection


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def verify(bindings):
    require(bool(bindings), 'Empty provenance')
    for path, pin in bindings.items():
        require(sha(path) == pin, 'Changed input: ' + path)


def identity(pid):
    p = Path('/proc') / str(pid)
    s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(s[19]), uid=p.stat().st_uid,
                argv=[x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x])


def gone(old):
    p = Path('/proc') / str(old['pid'])
    require(not p.exists() or identity(old['pid'])['start_ticks'] != old['start_ticks'],
            'Predecessor process has not exited')


def same_process(a, b):
    """Compare stable execution identity; state and PPID are observation fields."""
    keys = ('pid', 'start_ticks', 'uid', 'argv')
    return all(k in a and k in b and a[k] == b[k] for k in keys)


def admit_normal_cpu(request):
    """Validate the original normal parent/child closure, not status counters."""
    root = Path(request['new_root'])
    e, q = root / 'cpu_execution_v1', root / 'cpu_sequence_v1'
    require(not (root / 'STOP').exists() and not (e / 'failure.json').exists()
            and not (q / 'failure.json').exists(), 'Predecessor failure must be diagnosed')
    outer, owner, receipt = read(q / 'completion.json'), read(e / 'completion.json'), read(q / 'owner_exit.json')
    require(outer['status'] == 'RAW64_CPU_SEQUENCE_COMPLETE' and outer['original_owner_success'] is True
            and owner['status'] == 'MAIN_RAW64_QUALIFICATION_PROXY_NORMALLY_COMPLETE_V1', 'Normal CPU closure required')
    require(outer['owner_completion_sha256'] == sha(e / 'completion.json')
            and outer['exit_receipt_sha256'] == sha(q / 'owner_exit.json'), 'Unbound CPU exit')
    require(receipt['exit_code'] == 0 and receipt['process_waited'] is True
            and sha(receipt['log']) == receipt['log_sha256']
            and same_process(receipt['identity'], owner['owner_identity']), 'CPU owner was not normally reaped')
    gone(receipt['identity'])
    gone(read(q / 'identity.json'))
    reg = read(e / 'registration.json')
    require(owner['registration_sha256'] == sha(e / 'registration.json')
            and reg['config_sha256'] == sha(e / 'config.json'), 'Different CPU registration')
    verify(reg['source_bindings'])
    verify(reg['input_bindings'])
    verify(owner['outputs'])
    for path in owner['outputs']:
        phase = read(path)
        require(phase['status'] == 'MAIN_RAW64_CPU_PHASE_COMPLETE_V1'
                and phase['registration_sha256'] == sha(e / 'registration.json'), 'Invalid phase closure')
        verify(phase['outputs'])
    # Query the existing ledger read-only. The already sealed original merge
    # authenticated every per-call request/result hash and worker identity.
    with sqlite3.connect('file:' + str(root / 'cpu_budget_v1.sqlite') + '?mode=ro', uri=True) as db:
        counts = dict(db.execute('SELECT phase,charged FROM phase_counts'))
        statuses = dict(db.execute('SELECT status,count(*) FROM events GROUP BY status'))
    expected = dict(reg['phase_caps'])
    require(counts == {k: v if k in ('qualification', 'proxy') else 0 for k, v in expected.items()},
            'Actual phase already started or wrong CPU charges')
    require(set(statuses) == {'COMPLETE'} and statuses['COMPLETE'] == sum(counts.values()), 'Ledger not quiescent')
    return {str(p): sha(p) for p in [e / 'completion.json', q / 'completion.json', q / 'owner_exit.json',
            e / 'registration.json', e / 'config.json', *map(Path, owner['outputs'])]}


def probability_map(plan, proxy):
    require(proxy['phase'] == 'proxy' and proxy['packet_calls'] == plan['phase_caps']['proxy'], 'Incomplete proxy phase')
    result = {}
    for row in plan['proxy']['reused_cells'] + plan['proxy']['reused_header_correct_cells']:
        key = ('header' if row['kind'] == 'header' else row['phy_key'], row['snr_db'])
        n, c, r, u = (row[x] for x in ('n_blocks', 'n_correct', 'n_reject', 'n_undetected'))
        require(all(type(v) is int and v >= 0 for v in (n, c, r, u)) and n > 0 and c+r+u == n and key not in result, 'Invalid reused counts')
        result[key] = c/n
    expected = {(r['phy_key'], r['snr_db']) for r in plan['proxy']['cells']}
    seen = set()
    for row in proxy['points']:
        point = row['point']; key = point['phy_key'], point['snr_db']
        n, c, r, u = (row[x] for x in ('packet_calls', 'correct', 'rejected', 'undetected'))
        require(point['kind'] == 'body' and key in expected and key not in result and key not in seen,
                'Missing, duplicate or unregistered physical cell')
        require(n == 256 and all(type(v) is int and v >= 0 for v in (n, c, r, u)) and c+r+u == n, 'Invalid measured counts')
        result[key] = c/n; seen.add(key)
    wanted = {(k, s) for k in plan['all_body_phy_keys'] + ['header'] for s in selection.SNRS}
    require(seen == expected and set(result) == wanted, 'Incomplete exact probability grid')
    return result


def psnr(row):
    m, p = row['mse'], row['psnr_db']
    require(type(m) in (int, float) and type(p) in (int, float) and 0 < m <= 1
            and math.isfinite(p) and abs(p + 10*math.log10(m)) < 1e-9, 'Inconsistent clean PSNR')
    return p


def clean_data(request):
    """Use only original per-source scalar tables and authenticated checkpoints."""
    old = Path(request['old_root']); new = Path(request['new_root'])
    cfg = read(request['old_clean_compatibility_config'])
    require(cfg['compatibility']['same_raster_render_received_semantics_verified'] is True
            and cfg['compatibility']['same_source_tokens_verified'] is True
            and cfg['compatibility']['same_frozen_visual_identity_and_numeric_flags_verified'] is True,
            'Original exact clean compatibility proof required')
    # The normal completed old screening used this exact config and the scalar
    # table/checkpoint hashes in this proof. Do not reopen any image archives.
    bindings = cfg['compatibility']['input_bindings']
    source_completion = read(old / 'H/source200/completion.json')
    clean_completion = read(old / 'H/clean-quality200/completion.json')
    new_completion = read(new / 'clean_missing_v1/completion.json')
    new_registration = read(new / 'clean_missing_execution_v1/registration.json')
    require(new_completion['registration_sha256'] == sha(new / 'clean_missing_execution_v1/registration.json')
            and new_completion['config_sha256'] == new_registration['config_sha256']
            == sha(new / 'clean_missing_execution_v1/config.json'), 'Different endpoint config/registration')
    require(new_completion['frozen_identity'] == new_registration['frozen_identity']
            == cfg['compatibility']['original_visual_identity']
            and new_completion['numerical_runtime'] == new_registration['numerical_runtime'],
            'Endpoint changed visual or numeric identity')
    new_outer = read(new / 'clean_sequence_v1/completion.json')
    require(new_completion['status'] == 'MAIN_RAW64_CLEAN_MISSING200_COMPLETE_V1'
            and new_outer['status'] == 'RAW64_CLEAN200_SEQUENCE_COMPLETE'
            and new_outer['scientific_completion_sha256'] == sha(new / 'clean_missing_v1/completion.json')
            and new_outer['worker_exit_sha256'] == sha(new / 'clean_sequence_v1/worker_exit.json'), 'New endpoint not normally closed')
    # Explicitly validate the normal wait receipt of the original GPU worker.
    ex = read(new / 'clean_sequence_v1/worker_exit.json')
    require(ex['exit_code'] == 0 and ex['process_waited'] is True and sha(ex['log']) == ex['log_sha256'], 'Endpoint worker not reaped')
    gone(ex['identity'])
    ids = cfg['source_ids']; require(ids == new_completion['source_ids'] and len(ids) == len(set(ids)) == 200, 'Different construction population')
    gray, clean = {}, {}
    for i, sid in enumerate(ids):
        entry = cfg['record_paths'][i]
        require(entry['source_index'] == i and entry['source_id'] == sid, 'Wrong original source order')
        sc, cc, cr = (Path(entry[k]) for k in ('source_checkpoint', 'clean_checkpoint', 'clean_rows'))
        require(bindings.get(str(sc)) == sha(sc) and bindings.get(str(cc)) == sha(cc)
                and bindings.get(str(cr)) == sha(cr), 'Original scalar checkpoint changed')
        require(source_completion['outputs'].get(str(sc)) == sha(sc)
                and clean_completion['outputs'].get(str(cc)) == sha(cc), 'Scalar tables not sealed by source/clean completion')
        s, c = read(sc), read(cc)
        require(s['source_index'] == c['source_index'] == i and s['source_id'] == c['source_id'] == sid
                and s['independent_roundtrip'] is True and c['clean_states'] == 395
                and c['outputs'].get(str(cr)) == sha(cr), 'Original clean/source identity differs')
        rows = read(cr); require(len(rows) == 395, 'Original states missing')
        values = {(r['m'], r['K']): psnr(r) for r in rows}
        require(len(values) == 395, 'Duplicate original state')
        ncp = new / 'clean_missing_v1/source_checkpoints' / f'{i:04d}.json'
        require(new_completion['outputs'].get(str(ncp)) == sha(ncp), 'Unsealed new endpoint source')
        nc = read(ncp); require(nc['source_id'] == sid and nc['source_index'] == i, 'New endpoint identity differs')
        verify(nc['outputs'])
        nrpath = new / 'clean_missing_v1/sources' / f'{i:04d}.json'
        require(str(nrpath) in nc['outputs'], 'Endpoint row path unbound')
        nr = read(nrpath)
        require(len(nr) == 1 and (nr[0]['m'], nr[0]['K'], nr[0]['token_count']) == (8, 142, 397), 'Wrong new endpoint')
        values[8, 142] = psnr(nr[0]); gray[sid] = psnr(s['gray']); clean[sid] = values
    return ids, gray, clean


def compute(request):
    verify(request['bindings'])
    closure = admit_normal_cpu(request)
    root = Path(request['new_root']); cat = read(root / 'assemble_v1/completion.json')
    profiles = cat['profiles']; require(len(profiles) == 433 and len({p['candidate_id'] for p in profiles}) == 433, '433 wire representatives required')
    plan = read(root / 'prepared_v1/cpu_plan_v1.json')
    require(cat['catalogue_digest'] == plan['catalogue_digest'], 'Different catalogue')
    probs = probability_map(plan, read(root / 'cpu_results_v1/proxy/completion.json'))
    ids, gray, clean = clean_data(request)
    scores = selection.score_proxy(profiles, ids, gray, clean, probs)
    shortlist = selection.shortlist(profiles, scores)
    shortlist.update(catalogue_digest=cat['catalogue_digest'], catalogue_sha256=sha(root / 'assemble_v1/completion.json'),
        science_registration_sha256=sha(root / 'SCIENCE_REGISTRATION_V1.json'),
        original_construction_source_ids=ids, new_PHY_calls=0, GPU_used=False,
        development_used=False, holdout_used=False, final_objective=selection.PRIMARY)
    return shortlist, scores, closure


def waited_worker(out, argv):
    """Always reap and seal the log, including admission/receipt exceptions."""
    out = Path(out); log = out / 'worker.log'; ident = None; capture_error = None
    with log.open('xb') as f:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        try:
            save(out / 'worker_pending.json', dict(pid=child.pid, expected_argv=argv))
            until = time.monotonic() + 10
            while True:
                ident = identity(child.pid)
                if ident['argv'] == argv:
                    break
                require(child.poll() is None and time.monotonic() < until, 'Worker argv admission failed')
                time.sleep(.01)
            save(out / 'worker_launch.json', dict(identity=ident, expected_argv=argv))
        except BaseException:
            capture_error = traceback.format_exc()
        finally:
            rc = child.wait()
    save(out / 'worker_exit.json', dict(pid=child.pid, identity=ident, expected_argv=argv,
        exit_code=rc, process_waited=True, log=str(log), log_sha256=sha(log), capture_error=capture_error))
    require(capture_error is None, 'Worker admission failed after normal wait: ' + str(capture_error))
    require(rc == 0, 'Prescreen worker failed')
    return ident


def worker(request_path):
    request = read(request_path); out = Path(request['out'])
    verify(request['bindings']); require(time.time() < request['deadline_unix'], 'Registered deadline reached')
    os.sched_setaffinity(0, set(request['affinity'])); os.setpriority(os.PRIO_PROCESS, 0, 15)
    shortlist, scores, closure = compute(request)
    save(out / 'shortlist.json', shortlist)
    save(out / 'proxy_scores.json', {str(s): v for s, v in scores.items()})
    verify(request['bindings'])
    save(out / 'science_completion.json', dict(status='RAW64_SIX_SNR_FINITE_SHORTLIST_COMPLETE',
        request_sha256=sha(request_path), worker_identity=identity(os.getpid()),
        source_count=200, wire_count=433, SNRs=list(selection.SNRS),
        outputs={str(p): sha(p) for p in (out / 'shortlist.json', out / 'proxy_scores.json')},
        predecessor_completion_bindings=closure, new_PHY_calls=0, GPU_used=False,
        final_calibration_complete=False, automatic_successor=False))


def main(request_path):
    request = read(request_path); out = Path(request['out']); out.mkdir(exist_ok=False)
    try:
        verify(request['bindings'])
        save(out / 'registration.json', dict(status='FROZEN_READ_ONLY_PRESCREEN_EXECUTION', request_sha256=sha(request_path),
            source_bindings=request['bindings'], deadline_unix=request['deadline_unix'], automatic_calibration=False))
        save(out / 'owner_identity.json', identity(os.getpid()))
        root = Path(request['new_root'])
        while not (root / 'cpu_sequence_v1/completion.json').exists():
            require(time.time() < request['deadline_unix'], 'Deadline while waiting for CPU table')
            require(not any((root / p).exists() for p in ('STOP', 'cpu_sequence_v1/failure.json', 'cpu_execution_v1/failure.json')), 'CPU predecessor failed')
            time.sleep(30)
        # Outer sequence may write its final receipt just before it exits.
        previous = read(root / 'cpu_sequence_v1/identity.json')
        while Path('/proc', str(previous['pid'])).exists() and identity(previous['pid'])['start_ticks'] == previous['start_ticks']:
            require(time.time() < request['deadline_unix'], 'CPU sequence did not exit')
            time.sleep(1)
        predecessor = admit_normal_cpu(request)
        save(out / 'release_gate.json', predecessor)
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(Path(request_path).resolve())]
        ident = waited_worker(out, argv)
        done = read(out / 'science_completion.json')
        require(done['worker_identity'] == ident
                and done['request_sha256'] == sha(request_path), 'Prescreen result identity mismatch')
        verify(done['outputs']); verify(request['bindings']); verify(predecessor)
        save(out / 'completion.json', dict(status='RAW64_PRESCREEN_NORMAL_COMPLETE', original_worker_success=True,
            science_completion_sha256=sha(out / 'science_completion.json'), exit_receipt_sha256=sha(out / 'worker_exit.json'),
            new_PHY_calls=0, GPU_used=False, automatic_calibration=False))
    except BaseException:
        save(out / 'failure.json', dict(status='FAILED_PRESERVE_NO_RETRY', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request'); parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main(args.request)
