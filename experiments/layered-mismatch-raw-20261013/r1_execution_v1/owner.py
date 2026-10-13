"""One finite serial R1 owner; never registers, retries, or launches R2."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import shared as s

STAGES = ('phy', 'projection', 'visual', 'score', 'selection')
ENV_THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def actual_success(wait):
    s.require(wait.get('actual_wait') is True and wait['returncode'] == 0
              and not any(wait.get(key, False) for key in ('timed_out', 'terminated', 'killed')),
              'Actual child did not complete successfully')


def live_identity(value):
    if not isinstance(value, dict):
        return False
    identities = [value] + [v for v in value.values() if isinstance(v, dict)]
    for identity in identities:
        pid = identity.get('pid', identity.get('child_pid'))
        if pid is None:
            continue
        actual = s.process_identity(pid)
        ticks = identity.get('start_ticks', identity.get('child_start_ticks'))
        if actual and actual['state'] != 'Z' and (ticks is None or str(ticks) == str(actual['start_ticks'])):
            return True
    return False


def timing_state():
    evidence = {}; present = False
    for name in ('dispatch.json', 'launch.json', 'worker_launch.json'):
        path = s.TIMING_OUT/name
        if path.exists():
            evidence[name] = s.read(path); present = True
    wait_path = s.TIMING_OUT/'actual_wait.json'
    waited = s.read(wait_path) if wait_path.exists() else None
    live = any(live_identity(value) for value in evidence.values())
    waiting_on_actual = (present or s.TIMING_RUN.exists()) and not (
        waited and waited.get('actual_child_waited', waited.get('actual_wait', False)))
    return dict(active=live or waiting_on_actual,
                eligible_to_start=not present and not s.TIMING_RUN.exists(), evidence=evidence, actual_wait=waited)


def timing_gpu_eligible(query=subprocess.run, timing_registration=None):
    registration = timing_registration if timing_registration is not None else s.readpin(
        dict(path=str(s.TIMING_OUT/'registration.json'), sha256=s.TIMING_REGISTRATION_SHA))
    target = registration['GPU_index']
    free_required = (registration['GPU_memory_fraction_cap_GiB'] + registration['free_safety_GiB']) * 1024
    utilization_max = registration['initial_GPU_utilization_max_percent']
    result = query(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                   capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, free, utilization = [int(x.strip()) for x in line.split(',')]
        rows.append(dict(index=index, free_MiB=free, utilization_percent=utilization))
    s.require(len(rows) == 4, 'Cannot assess original four-GPU timing priority')
    selected = [r for r in rows if r['index'] == target]
    s.require(len(selected) == 1, 'Registered timing device absent')
    return selected[0]['free_MiB'] >= free_required and selected[0]['utilization_percent'] <= utilization_max, rows


def acquire_window(reg, stage, flock):
    lock = (s.TIMING_OUT/'owner.lock').open('a+')
    try:
        flock.flock(lock, flock.LOCK_SH | flock.LOCK_NB)
        timing = timing_state()
        ready = not timing['active']; wait = dict(reason='Timing actual owner/child active', timing=timing)
        if ready and timing['eligible_to_start']:
            eligible, gpus = timing_gpu_eligible()
            if eligible:
                ready = False; wait = dict(reason='Registered v6 timing has priority on an eligible GPU', gpus=gpus)
        if ready and stage in ('visual', 'score'):
            gate = Path(reg['runtime']['root'])/'prelaunch'/stage; gate.mkdir(parents=True, exist_ok=True)
            wait = s.admission(reg, gate)
            ready = wait['status'] == 'READY'
    except BlockingIOError:
        ready = False; wait = dict(reason='Existing timing exclusive lock held')
    except BaseException:
        lock.close(); raise
    if ready:
        return lock, dict(status='READY')
    lock.close()
    return None, dict(wait, status='WAIT', stage=stage, observed_unix=time.time())


def stage_command(reg, registration, stage, parent, parent_ticks):
    root = Path(__file__).parent; runtime = reg['runtime']
    if stage == 'phy':
        value = reg['phy_registration']
        return [runtime['python_cpu'], '-B', '-u', str(s.EXPERIMENT/'r1_phy.py'), 'run',
                '--registration', value['path'], '--registration-sha256', value['sha256']]
    if stage in ('projection', 'visual'):
        value = reg['visual_registration'] if stage == 'visual' else registration
        return [runtime['python_visual'] if stage == 'visual' else runtime['python_cpu'], '-B', '-u',
                str(root/'receiver_visual.py'), 'visual' if stage == 'visual' else 'project',
                '--registration', value['path'], '--expected-registration', value['sha256'],
                '--parent-pid', str(parent), '--parent-ticks', str(parent_ticks)]
    s.require(stage in ('score', 'selection'), 'Unknown stage')
    return [runtime['python_visual'] if stage == 'score' else runtime['python_cpu'], '-B', '-u',
            str(root/'score_selection.py'), 'score' if stage == 'score' else 'select',
            '--registration', registration['path'], '--registration-sha256', registration['sha256'],
            '--parent', str(parent)]


def stage_environment(reg, stage):
    env = os.environ.copy(); gpu = stage in ('visual', 'score')
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=str(reg['runtime']['target_gpu']) if gpu else '',
               CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               TMPDIR=str(Path(reg['runtime']['root'])/'cache'))
    env.update({key: '6' if gpu else '2' for key in ENV_THREADS})
    return env


def wait_child(child, deadline, stop_paths, clock=time.time):
    timed_out = terminated = killed = stopped = False
    while True:
        try:
            code = child.wait(timeout=max(.001, min(1., deadline-clock())))
            break
        except subprocess.TimeoutExpired:
            stopped = any(Path(p).exists() for p in stop_paths)
            timed_out = clock() >= deadline
            if stopped or timed_out:
                terminated = True; child.terminate()
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    killed = True; child.kill(); code = child.wait()
                break
    return dict(actual_wait=True, pid=child.pid, returncode=code, timed_out=timed_out,
                stopped=stopped, terminated=terminated, killed=killed)


def closed_stage(reg, registration, stage):
    directory = Path(reg['runtime'][stage+'_directory'])
    waited = s.read(directory/'actual_wait.json'); actual_success(waited)
    s.require(not (directory/'failure.json').exists(), 'Failed stage requires diagnosis, not continuation')
    done = s.read(directory/'completion.json')
    expected_status = dict(phy='COMPLETE_R1_PHY_ONLY', projection='COMPLETE_R1_RX_PROJECTION',
                           visual='COMPLETE_R1_VISUAL', score='COMPLETE_R1_SCORE', selection='COMPLETE_R1_CALIBRATION_SELECTION')[stage]
    s.require(done['status'] == expected_status, 'Missing actual stage completion: '+stage)
    expected = reg['phy_registration'] if stage == 'phy' else (reg['visual_registration'] if stage == 'visual' else registration)
    s.require(done['registration']['path'] == expected['path'] and done['registration']['sha256'] == expected['sha256'],
              'Completed stage has a different registration')
    if stage == 'phy':
        ledger = done['ledger']
        s.require(ledger['physical_complete'] == ledger['physical_reserved'] == reg['expected_physical_frames']
                  and ledger['packet_decodes_complete'] == ledger['packet_decodes_reserved'] <= 108000
                  and ledger['unresolved_packets'] == 0, 'Physical ledger did not close')
        s.checked(done['physical_frames']); s.checked(done['logical_frames'])
    return dict(stage=stage, actual_wait=s.pin(directory/'actual_wait.json'), completion=s.pin(directory/'completion.json'))


def run(registration):
    import fcntl
    reg = s.validate_registration(registration)
    s.require(reg['schema'] == s.MASTER_SCHEMA, 'Owner requires reviewed complete-chain registration')
    root = Path(reg['runtime']['root']); root.mkdir(parents=True, exist_ok=True)
    s.require(not (root/'owner_launch.json').exists() and not (root/'owner_failure.json').exists()
              and not any(Path(reg['runtime'][stage+'_directory']).exists() for stage in STAGES),
              'Previously attempted chain cannot automatically replay')
    deadline = min(reg['runtime']['deadline_unix'], time.time()+reg['runtime']['max_seconds'])
    s.require(deadline > time.time(), 'Expired registration; no deadline reset')
    s.require(not any(p.exists() for p in (Path(reg['runtime']['stop_file']), root/'STOP')),
              'Registered STOP before launch')
    identity = s.process_identity(os.getpid())
    s.require(identity is not None, 'Actual Linux owner identity required')
    owner_lock = (root/'owner.lock').open('a+')
    try:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root/'cache').mkdir(exist_ok=True); (root/'logs').mkdir(exist_ok=True)
        timing_lock, window = acquire_window(reg, 'phy', fcntl)
    except BaseException:
        owner_lock.close()
        raise
    if timing_lock is None:
        s.status(s.OUT/'resource_wait.json', dict(window, scientific_calls=0, dispatched=False,
                                                 registration=registration, deadline_unix=deadline))
        owner_lock.close()
        return dict(window, scientific_calls=0, dispatched=False)
    stages = []; child = None
    try:
        s.write(root/'owner_launch.json', dict(process=identity, registration=registration,
            deadline_unix=deadline, automatic_retry=False, automatic_R2=False))
        for stage in STAGES:
            directory = Path(reg['runtime'][stage+'_directory'])
            # Reacquire between stages, so the already registered timing can
            # take its priority window while no R1 child is running.
            while timing_lock is None:
                s.require(time.time() < deadline and not any(p.exists() for p in
                    (Path(reg['runtime']['stop_file']), root/'STOP')), 'Owner STOP/deadline')
                timing_lock, wait = acquire_window(reg, stage, fcntl)
                if timing_lock is not None:
                    break
                s.status(s.OUT/'resource_wait.json', wait)
                time.sleep(min(30, max(.01, deadline-time.time())))
            s.validate_registration(registration)
            s.require(time.time() < deadline and not any(p.exists() for p in
                (Path(reg['runtime']['stop_file']), root/'STOP', directory/'STOP')), 'Owner STOP/deadline')
            s.require(not directory.exists(), 'Scientific stage has already been attempted')
            argv = stage_command(reg, registration, stage, identity['pid'], identity['start_ticks'])
            cpus = reg['runtime']['cpu_affinity' if stage in ('visual','score') else 'phy_cpu_affinity']
            s.require(set(cpus) <= os.sched_getaffinity(0), 'Registered CPU slots unavailable')
            started = time.monotonic()
            with (root/'logs'/(stage+'.stdout.log')).open('xb') as stdout, (root/'logs'/(stage+'.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(argv, cwd=s.ROOT, env=stage_environment(reg, stage), stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, pass_fds=(owner_lock.fileno(), timing_lock.fileno()),
                    preexec_fn=lambda: os.sched_setaffinity(0, cpus))
                launch = dict(stage=stage, process=s.process_identity(child.pid), argv=argv, registration=registration,
                              cpu_affinity=cpus, deadline_unix=deadline)
                s.require(launch['process'] is not None, 'Actual child start identity unavailable')
                s.write(root/(stage+'_launch.json'), launch)
                waited = wait_child(child, deadline, [reg['runtime']['stop_file'], root/'STOP', directory/'STOP'])
                waited.update(stage=stage, elapsed_seconds=time.monotonic()-started)
                child = None
            directory.mkdir(parents=True, exist_ok=True)
            s.write(directory/'actual_wait.json', waited)
            s.write(root/(stage+'_actual_wait.json'), waited)
            actual_success(waited)
            stages.append(closed_stage(reg, registration, stage))
            timing_lock.close(); timing_lock = None
        s.validate_registration(registration)
        return s.write(s.OUT/'completion.json', dict(status='COMPLETE_R1_CHAIN', registration=registration,
            stages=stages, actual_exit_zero_all=True, frozen_policy_completion=stages[-1]['completion'],
            R2='NOT_REGISTERED_NOT_RUN', automatic_R2=False))
    except BaseException:
        if child is not None:
            terminated = child.poll() is None
            if terminated:
                child.terminate()
            waited = wait_child(child, time.time()+15, [])
            waited.update(terminated=terminated or waited['terminated'], stage=stage,
                          elapsed_seconds=time.monotonic()-started)
            s.write(root/'interrupted_actual_wait.json', waited)
        s.write(root/'owner_failure.json', dict(status='FAILED_STOP_NO_RETRY', registration=registration,
            completed_stages=stages, traceback=traceback.format_exc(), remaining_stages_not_launched=True))
        raise
    finally:
        if timing_lock is not None:
            timing_lock.close()
        owner_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True); parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    print(s.canonical(run(dict(path=args.registration,sha256=args.registration_sha256))))
