"""Restore two protected historical waiters after the priority interlude.

Only the two recorded requested-stop propagation exits are eligible. This is
CPU-only orchestration; the unchanged original waiters retain all GPU gates.
"""
import argparse
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from tools.priority_selected_evaluation import ROOT, OUT, DEST, MAIN, read, sha, write, identity, same_process, check_bindings, lock

HERE = DEST / 'historical_waiter_recovery'
WORKERS = [
    ('author_native_timing_v1', 'run_author_reference_timing.sh',
     'tools.replay_author_reference_timing',
     'FAILED_AUTHOR_PARITY_TIMING_WORKER',
     "RuntimeError('upstream stopped/failed; preserve state for repair')",
     'WAITING_FOR_N4084_AND_ALL_PREDECESSORS'),
    ('n3060_native_timing_v1', 'run_n3060_reference_timing.sh',
     'tools.replay_n3060_reference_timing',
     'FAILED_N3060_TIMING_WORKER',
     "RuntimeError('upstream failure or requested stop requires review')",
     'WAITING_FOR_AUTHOR_AND_ALL_PREDECESSORS'),
]

def eligible(status, expected_status, expected_error, pause_time):
    return (status.get('status') == expected_status and status.get('error') == expected_error
            and status.get('time', 0) >= pause_time)

def prepare():
    if (HERE / 'registration.json').exists():
        raise RuntimeError('recovery already registered')
    pause = read(DEST / 'pause_request.json')
    items = []
    bindings = {str(Path(__file__).resolve()): sha(__file__),
                str(ROOT / 'tools/priority_selected_evaluation.py'): sha(ROOT / 'tools/priority_selected_evaluation.py')}
    for name, script, module, failure, error, waiting in WORKERS:
        folder = OUT / name
        status = read(folder / 'worker_status.json')
        old = read(folder / 'launch_receipt.json')
        if not eligible(status, failure, error, pause['time']):
            raise RuntimeError('unreviewed failure: ' + name)
        if same_process(identity(old['process']['pid']), old['process']):
            raise RuntimeError('original worker still live')
        handle = lock(folder / 'worker.lock')
        handle.close()
        workerreg = read(folder / 'worker_registration.json')
        check_bindings(workerreg['bindings'])
        bindings.update(workerreg['bindings'])
        scriptpath = ROOT / 'tools' / script
        bindings[str(scriptpath)] = sha(scriptpath)
        archive = HERE / 'previous' / name
        archive.mkdir(parents=True)
        saved = {}
        for leaf in ['worker_status.json', 'launch_receipt.json', 'worker_registration.json', 'detached_console.log']:
            src = folder / leaf
            if src.exists():
                dst = archive / leaf
                shutil.copyfile(src, dst)
                saved[leaf] = sha(dst)
        items.append(dict(name=name, script=str(scriptpath), module=module, waiting=waiting,
                          old_process=old['process'], archive=str(archive), archived_sha256=saved,
                          worker_registration_sha256=sha(folder / 'worker_registration.json')))
    write(HERE / 'registration.json', dict(status='REVIEWED_REQUESTED_STOP_PROPAGATION_ONLY',
          bindings=bindings, workers=items, pause_time=pause['time'], time=time.time(),
          GPU_work='NONE; unchanged workers resume their original wait gates'))
    print('RECOVERY_REGISTERED', len(items), flush=True)

def main_is_resumed():
    if not (DEST / 'completion.json').exists():
        return False
    done = read(DEST / 'completion.json')
    if done['status'] != 'PRIORITY_EVALUATION_COMPLETE_ORIGINAL_QUEUE_RESUMED':
        raise RuntimeError('unexpected priority outcome')
    original = read(MAIN / 'launch_receipt.json')['process']
    if original != done['original_queue'] or not same_process(identity(original['pid']), original):
        raise RuntimeError('resumed main identity mismatch')
    status = read(MAIN / 'status.json')
    if 'FAIL' in status['status'] or status['status'] == 'STOPPED_BY_REQUEST':
        return False
    return status.get('pid') == original['pid'] and time.time() - status['time'] < 40

def run():
    ownlock = lock(HERE / 'worker.lock')
    reg = read(HERE / 'registration.json')
    check_bindings(reg['bindings'])
    if (HERE / 'completion.json').exists() or (HERE / 'launch_intents').exists():
        raise RuntimeError('previous recovery attempt must be inspected before retry')
    while not main_is_resumed():
        if (DEST / 'failure.json').exists():
            raise RuntimeError('priority evaluation failed; do not restore waiters')
        write(HERE / 'status.json', dict(status='WAITING_FOR_PRIORITY_COMPLETION_AND_ORIGINAL_MAIN',
              process=identity(os.getpid()), time=time.time()))
        time.sleep(10)
    for item in reg['workers']:
        check_bindings(reg['bindings'])
        if not main_is_resumed():
            raise RuntimeError('original main no longer healthy')
        folder = OUT / item['name']
        if sha(folder / 'worker_registration.json') != item['worker_registration_sha256']:
            raise RuntimeError('historical registration changed')
        if same_process(identity(item['old_process']['pid']), item['old_process']):
            raise RuntimeError('old worker still live')
        # The lock proves no replacement waiter owns this registration.
        handle = lock(folder / 'worker.lock')
        handle.close()
        intent = HERE / 'launch_intents' / (item['name'] + '.json')
        write(intent, dict(status='ONE_LAUNCH_AUTHORIZED', worker=item['name'], time=time.time()))
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        with (folder / 'detached_console.log').open('a') as log:
            log.write('\nRESTORE after reviewed priority-evaluation requested-stop propagation\n')
            log.flush()
            p = subprocess.Popen(['bash', item['script']], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        deadline = time.time() + 90
        while True:
            actual = identity(p.pid)
            if p.poll() is not None:
                raise RuntimeError('restored worker exited; preserve evidence')
            status = read(folder / 'worker_status.json')
            if (actual and ('-m ' + item['module'] + ' --wait ') in actual['cmdline']
                    and status.get('pid') == p.pid and status.get('start_ticks') == actual['start_ticks']
                    and status['status'] == item['waiting']):
                break
            if time.time() > deadline:
                raise RuntimeError('restored waiter did not register; inspect real process before retry')
            time.sleep(2)
        receipt = dict(status='RESTORED_REVIEWED_PRIORITY_STOP_PROPAGATION', process=actual,
            pid=actual['pid'], start_ticks=actual['start_ticks'], cmdline=actual['cmdline'],
            session_id=os.getsid(p.pid), registration_sha256=item['worker_registration_sha256'],
            previous_receipts=item['archive'], time=time.time(), GPU_forward='NOT_RUN_IN_RESTORATION')
        write(folder / 'launch_receipt.json', receipt)
        write(HERE / (item['name'] + '_restored.json'), receipt)
    write(HERE / 'completion.json', dict(status='TWO_ORIGINAL_HISTORICAL_WAITERS_RESTORED',
          workers=[i['name'] for i in reg['workers']], time=time.time(), GPU_work='NONE'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare()
    else:
        try:
            run()
        except Exception:
            write(HERE / 'failure.json', dict(status='FAILED_REVIEWED_WAITER_RESTORATION',
                  traceback=traceback.format_exc(), time=time.time()))
            raise
