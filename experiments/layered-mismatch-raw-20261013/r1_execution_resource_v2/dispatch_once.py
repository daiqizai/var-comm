"""One persistent parent for the already registered R1 resource-v2 owner.

No registration, retry, GPU probe, model construction or automatic successor.
The owner itself retains the frozen finite deadline and timing coordination.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import shared as s

ENTRY = Path(__file__).with_name('owner.py')
REG = s.OUT / 'registered/registration.json'
DISPATCH = s.TMP.parent / 'r1_resource_dispatch_v2'


def no_prior_attempt():
    markers = ('owner_launch.json', 'owner_failure.json', 'completion.json',
               'interrupted_actual_wait.json', 'phy', 'projection', 'visual', 'score', 'selection')
    s.require(not any((s.TMP/name).exists() for name in markers), 'Existing R1 attempt; no automatic replay')
    s.require(not (s.OUT/'completion.json').exists(), 'R1 result already exists')
    s.require(not any((DISPATCH/name).exists() for name in
        ('launch.json', 'owner_identity.json', 'actual_wait.json', 'failure.json', 'interrupted_actual_wait.json')),
        'Existing resource-v2 dispatch; no implicit repeat')


def terminate_owned(child, grace=15):
    """Only our still-live process group; always collect our actual child."""
    terminated = killed = False
    if child.poll() is None:
        s.require(os.getpgid(child.pid) == child.pid, 'Owned group identity differs')
        os.killpg(child.pid, signal.SIGTERM); terminated = True
        try:
            code = child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); killed = True; code = child.wait()
    else:
        code = child.wait()
    return dict(actual_wait=True, pid=child.pid, returncode=code, terminated=terminated, killed=killed)


def run(expected):
    registration = dict(path=str(REG), sha256=expected)
    reg = s.validate_registration(registration)
    s.require(reg['schema'] == s.MASTER_SCHEMA, 'Master R1 registration required')
    s.require(reg['code_bindings'][str(Path(__file__))] == s.sha(__file__)
              and reg['code_bindings'][str(ENTRY)] == s.sha(ENTRY), 'Dispatcher or owner binding differs')
    s.require(time.time() < reg['runtime']['deadline_unix'], 'Registered deadline already exhausted')
    import fcntl
    DISPATCH.mkdir(parents=True, exist_ok=True)
    with (DISPATCH/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        no_prior_attempt()
        argv = [s.PY_CPU, '-B', '-u', str(ENTRY), '--registration', str(REG),
                '--registration-sha256', expected]
        s.write(DISPATCH/'launch.json', dict(process=s.process_identity(os.getpid()), registration=registration,
            entry=s.pin(ENTRY), argv=argv, created_unix=time.time(), automatic_retry=False,
            scientific_completion=False, deadline_unix=reg['runtime']['deadline_unix']))
        child = None; start = time.monotonic(); waited = False
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                   CUDA_VISIBLE_DEVICES='')
        try:
            with (DISPATCH/'owner.stdout.log').open('xb') as stdout, (DISPATCH/'owner.stderr.log').open('xb') as stderr:
                child = subprocess.Popen(argv, cwd=s.ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(lock.fileno(),))
                identity = s.process_identity(child.pid)
                s.require(identity is not None, 'Actual dispatched owner identity unavailable')
                s.write(DISPATCH/'owner_identity.json', dict(process=identity, registration=registration))
                timed_out = False
                try:
                    code = child.wait(timeout=max(.001, reg['runtime']['deadline_unix']-time.time()+120))
                    record = dict(actual_wait=True, pid=child.pid, returncode=code, terminated=False, killed=False)
                except subprocess.TimeoutExpired:
                    timed_out = True; record = terminate_owned(child); code = record['returncode']
                record.update(timed_out=timed_out, elapsed_seconds=time.monotonic()-start,
                    identity_at_launch=identity, registration=registration, automatic_retry=False,
                    scientific_owner_launch_exists=(s.TMP/'owner_launch.json').exists(),
                    scientific_completion=False)
                s.write(DISPATCH/'actual_wait.json', record); waited = True
            return code
        except BaseException:
            error = traceback.format_exc()
            if child is not None and not waited:
                record = terminate_owned(child)
                record.update(automatic_retry=False, registration=registration)
                s.write(DISPATCH/'interrupted_actual_wait.json', record)
            s.write(DISPATCH/'failure.json', dict(status='STOPPED_NO_RETRY', traceback=error,
                registration=registration, automatic_retry=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-registration', required=True)
    args = parser.parse_args()
    sys.exit(run(args.expected_registration))
