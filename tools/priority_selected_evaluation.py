"""Prioritize frozen C models, preserving the original serialized GPU queue.

This independent tool changes scheduling only. It imports the original Runner,
evaluates a separately registered 14-model population, then resumes the original
entry point. It never edits active experiment sources or historical receipts.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923'
DEST = OUT / 'C_priority_selected_v1'
MAIN = OUT / 'delivery_chain_v1'
SHORT = ROOT / 'outputs/SHORT-PREFIX-20260923'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, p)

def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=stat[19], state=stat[0],
                    cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None

def same_process(actual, expected):
    return bool(actual and actual['state'] != 'Z' and all(
        actual[k] == expected[k] for k in ('pid', 'start_ticks', 'cmdline')))

def check_bindings(bindings):
    for name, expected in bindings.items():
        if sha(name) != expected:
            raise RuntimeError('dependency changed: ' + name)

def add(bindings, path):
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in bindings and bindings[str(path)] != digest:
        raise RuntimeError('conflicting dependency: ' + str(path))
    bindings[str(path)] = digest

def inventory():
    entries = []
    for group, arms in [('m6', ['H6-V', 'H6-P']), ('m7', ['H7-V', 'H7-P']),
                        ('m8', ['H8-V']), ('pure', ['P4084'])]:
        entries.append((SHORT / 'training' / f'{group}_seed2026092304',
                        arms, 4084, 2026092304))
    entries.append((SHORT / 'scoped_N3060/training/m6_seed2026092304',
                    ['H6-V', 'H6-P'], 3060, 2026092304))
    for group, arms in [('m6', ['H6-V', 'H6-P']), ('m8', ['H8-V']), ('pure', ['P4084'])]:
        base = SHORT / ('scoped_N4084/training' if group == 'pure' else 'training')
        entries.append((base / f'{group}_seed2026092404', arms, 4084, 2026092404))
    entries.append((SHORT / 'training/m6_seed2026092504',
                    ['H6-V', 'H6-P'], 4084, 2026092504))
    return entries

def require_final(done, decision, arms):
    step = done['state']['step']
    if step < 20000 or step % 10000 or set(done['state']['updates']) != set(arms):
        raise RuntimeError('invalid completed training boundary')
    if any(v != step for v in done['state']['updates'].values()):
        raise RuntimeError('unequal updates')
    if decision['step'] != step or decision['extend'] is not False:
        raise RuntimeError('training has not stopped by calibration')
    if decision['development_used'] is not False:
        raise RuntimeError('development selected training')

def prepare():
    if (DEST / 'registration.json').exists():
        raise RuntimeError('registration already exists; inspect it, do not overwrite')
    bound = {}
    models = []
    for folder, arms, n, seed in inventory():
        regpath = folder / 'registration.json'
        donepath = folder / 'completion.json'
        done = read(donepath)
        decisionpath = folder / 'delivery_decisions' / f"at_{done['state']['step']:05d}.json"
        decision = read(decisionpath)
        require_final(done, decision, arms)
        reg = read(regpath)
        check_bindings(reg['bindings'])
        check_bindings(decision['evidence_bindings'])
        bound.update(reg['bindings'])
        bound.update(decision['evidence_bindings'])
        for p in [regpath, donepath, decisionpath]:
            add(bound, p)
        for arm in arms:
            selectedpath = folder / f'selected_{arm}.json'
            selected = read(selectedpath)
            if selected != done['selected'][arm] or selected['registration_sha256'] != sha(regpath):
                raise RuntimeError('selected/completion/registration mismatch')
            cp = Path(selected['checkpoint'])
            if not cp.is_absolute():
                cp = ROOT / cp
            if sha(cp) != selected['checkpoint_sha256']:
                raise RuntimeError('selected checkpoint mismatch')
            for p in [selectedpath, cp]:
                add(bound, p)
            models.append(dict(method=f'{arm}_N{n}_seed{seed}', training=str(folder),
                               arm=arm, N=n, training_seed=seed))
    if len(models) != 14 or len({m['method'] for m in models}) != 14:
        raise RuntimeError('priority model inventory')
    candidate = OUT / 'C_followups/candidate.json'
    if read(candidate)['m'] != 6 or read(candidate)['development_used'] is not False:
        raise RuntimeError('frozen candidate')
    add(bound, candidate)
    mainreg = read(MAIN / 'registration.json')
    check_bindings(mainreg['bindings'])
    bound.update(mainreg['bindings'])
    add(bound, MAIN / 'registration.json')
    add(bound, __file__)
    add(bound, ROOT / 'tools/run_priority_selected_evaluation.sh')
    spec = dict(run_id='SHORT-PREFIX-20260923-priority-selected-v1', models=models,
                population='development', new_holdout=False,
                output=str(DEST / 'selected_grid'), candidate_sha256=sha(candidate))
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST / 'spec.json', spec)
    add(bound, DEST / 'spec.json')
    launch = read(MAIN / 'launch_receipt.json')
    expected = launch['process']
    if not same_process(identity(expected['pid']), expected):
        raise RuntimeError('original main process changed before registration')
    reg = dict(status='REGISTERED_PRIORITY_EVALUATION', bindings=bound, models=models,
               original_main=expected, original_registration_sha256=sha(MAIN / 'registration.json'),
               population='development', new_holdout=False, content_selector=False,
               expected_frame_rows=21000, expected_timed_calls=1400,
               user_request='Evaluate completed models first; GPU0 remains serial.',
               subsequent_full_grid='Original final 16-model context remains separate; no false reuse claim.',
               time=time.time())
    write(DEST / 'registration.json', reg)
    print(json.dumps({'status': reg['status'], 'models': len(models), 'bindings': len(bound)}))

def lock(path):
    h = Path(path).open('a')
    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return h

def verify_completion(done, reg):
    if done['status'] != 'REAL_C_SELECTED_GRID_COMPLETE':
        raise RuntimeError('evaluation did not complete')
    if done['frame_rows'] != reg['expected_frame_rows'] or done['timed_calls'] != reg['expected_timed_calls']:
        raise RuntimeError('incomplete registered population')
    if done.get('synthetic') is not False or done.get('new_holdout') is not False:
        raise RuntimeError('evaluation scope')

def run():
    from token_efficiency import delivery_chain
    from token_efficiency.retired_gate import acquire_predecessor_lock
    from latent_followup.run_identity import checked_checkpoint
    ownlock = lock(DEST / 'worker.lock')
    reg = read(DEST / 'registration.json')
    check_bindings(reg['bindings'])
    if (DEST / 'completion.json').exists() or (DEST / 'handoff.json').exists():
        raise RuntimeError('existing handoff/completion requires explicit evidence review')
    expected = reg['original_main']
    actual = identity(expected['pid'])
    if not same_process(actual, expected):
        raise RuntimeError('main identity changed; refuse signal')
    status = read(MAIN / 'status.json')
    if status.get('pid') != expected['pid'] or time.time() - status['time'] > 40:
        raise RuntimeError('main status is not fresh')
    archive = DEST / 'original_queue_before_pause'
    archive.mkdir()
    for name in ['status.json', 'launch_receipt.json', 'registration.json']:
        shutil.copyfile(MAIN / name, archive / name)
    delivery_chain.CHAIN = DEST
    runner = delivery_chain.Runner()
    runner.status('PAUSING_VERIFIED_ORIGINAL_OUTERMOST_QUEUE', original_main=actual)
    if not same_process(identity(expected['pid']), expected):
        raise RuntimeError('main identity changed immediately before signal')
    os.kill(expected['pid'], signal.SIGTERM)
    write(DEST / 'pause_request.json', dict(process=actual, signal='SIGTERM',
                                          time=time.time(), reason='user priority evaluation request'))
    while same_process(identity(expected['pid']), expected):
        runner.check_stop()
        time.sleep(2)
    stopped = read(MAIN / 'status.json')
    if stopped['status'] != 'STOPPED_BY_REQUEST' or stopped.get('returncode') != 75:
        raise RuntimeError('unexpected original queue termination')
    locks = [lock(MAIN / 'chain.lock'), lock(OUT / 'C_followups/followups.lock'),
             acquire_predecessor_lock()]
    live = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        proc = identity(int(p.name))
        if proc and proc['state'] != 'Z' and ('-m short_prefix.train ' in proc['cmdline'] or
               '-m token_efficiency.C_train ' in proc['cmdline']):
            live.append(proc)
    if live:
        raise RuntimeError('training process still alive')
    training = SHORT / 'training/m8_seed2026092504'
    latest = read(training / 'latest.json')
    checked_checkpoint(latest, ROOT)
    check_bindings(read(training / 'registration.json')['bindings'])
    write(DEST / 'handoff.json', dict(status='ORIGINAL_QUEUE_SAFELY_PAUSED', stopped=stopped,
          training=str(training), latest=latest, time=time.time(),
          original_main=expected, original_registration_sha256=reg['original_registration_sha256']))
    runner.status('PRIORITY_EVALUATION_OWNS_ORIGINAL_LOCKS', checkpoint=latest)
    done = runner.run('C_priority_selected_development',
                      ['token_efficiency.C_evaluate', '--spec', str(DEST / 'spec.json')],
                      DEST / 'selected_grid/completion.json')
    verify_completion(done, reg)
    check_bindings(reg['bindings'])
    runner.check_stop()
    write(DEST / 'evaluation_verified.json', dict(status='REAL_PRIORITY_SELECTED_GRID_VERIFIED',
          completion=done, completion_sha256=sha(DEST / 'selected_grid/completion.json'), time=time.time()))
    for h in reversed(locks):
        h.close()
    if same_process(identity(expected['pid']), expected):
        raise RuntimeError('old main unexpectedly alive before resume')
    script = ROOT / 'experiments/token_channel_efficiency_20260923/scripts/run_delivery_chain.sh'
    with (DEST / 'resumed_original_console.log').open('a') as log:
        child = subprocess.Popen(['bash', str(script)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    proc = identity(child.pid)
    if not proc or proc['state'] == 'Z' or ' -m token_efficiency.delivery_chain ' not in proc['cmdline']:
        raise RuntimeError('resumed original main identity missing; do not relaunch blindly')
    receipt = dict(status='ORIGINAL_QUEUE_RESUMED_AFTER_PRIORITY_EVALUATION',
                   process=proc, pid=proc['pid'], start_ticks=proc['start_ticks'],
                   cmdline=proc['cmdline'], session_id=os.getsid(child.pid),
                   registration_sha256=sha(MAIN / 'registration.json'),
                   git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   previous_receipts=str(archive), priority_evaluation=str(DEST), time=time.time())
    write(MAIN / 'launch_receipt.json', receipt)
    write(DEST / 'resumed_original_launch.json', receipt)
    write(DEST / 'completion.json', dict(status='PRIORITY_EVALUATION_COMPLETE_ORIGINAL_QUEUE_RESUMED',
          original_queue=proc, evaluation_completion_sha256=sha(DEST / 'selected_grid/completion.json'),
          time=time.time(), full_delivery_complete=False))
    runner.status('PRIORITY_EVALUATION_COMPLETE_ORIGINAL_QUEUE_RESUMED', original_queue=proc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except Exception:
            write(DEST / 'failure.json', dict(status='FAILED_PRIORITY_EVALUATION_NO_AUTOMATIC_TRAINING_RESUME',
                  traceback=traceback.format_exc(), time=time.time(), process=identity(os.getpid())))
            raise

if __name__ == '__main__':
    main()
