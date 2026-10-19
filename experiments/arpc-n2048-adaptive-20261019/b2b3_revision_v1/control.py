"""One explicitly requested bounded stage; no polling, retry or implicit B3."""
from common import *
import argparse
import fcntl


def stage(phase, role, expected):
    r = registration(phase, expected)
    assert role in ('tx','rx','audit')
    previous = {'rx':'tx','audit':'rx'}.get(role)
    if previous:
        closed_wait(L/'controls'/(phase+'_'+previous)/'actual_wait.json')
    lock_path = OLD_ADAPTIVE_L/'execution.lock'
    no_symlink_ancestors(lock_path); no_symlink_ancestors(L)
    lock = os.fdopen(os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW), 'r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (L/'controls').mkdir(exist_ok=True)
    out = L/'controls'/(phase+'_'+role)
    out.mkdir()
    put(out/'supervisor.json', dict(process=identity(os.getpid()), parent_process=identity(os.getppid()), argv=sys.argv))
    argv = [str(CPY), '-B', str(E/'transport.py'), role]
    if role == 'rx':
        tx = read(L/phase/'tx/completion.json')
        public = tx['public']; verify(public)
        argv += ['--public', public['path'], '--public-sha256', public['sha256'], '--public-bytes', str(public['bytes'])]
    else:
        argv += ['--phase', phase, '--expected-registration', expected]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    try:
        wait_child(out, argv, env, r['stage_seconds'][role])
        put(out/'completion.json', dict(status='COMPLETE', actual_wait=pin(out/'actual_wait.json'),
            no_automatic_next_stage=True))
    except BaseException:
        fail(out)
        raise


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('phase', choices=['B2','B3'])
    a.add_argument('role', choices=['tx','rx','audit'])
    a.add_argument('--expected-registration', required=True)
    x = a.parse_args()
    stage(x.phase, x.role, x.expected_registration)
