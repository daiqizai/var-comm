"""Durable continuation supervisor, exactly one child and one actual wait."""
import os
import signal
import subprocess
import sys
import time
from adapter import *


def main(expected):
    r, rec = bind(expected)
    CONTROL.mkdir()
    d = CONTROL/'dispatch'
    d.mkdir()
    save(d/'supervisor.json', dict(process=process(os.getpid()),
        parent_process=process(os.getppid()), recovery_registration_sha256=expected))
    argv = [CPY, '-B', str(HERE/'owner.py'), expected]
    start = time.monotonic()
    timed = False
    with (d/'stdout.log').open('xb') as stdout, (d/'stderr.log').open('xb') as stderr:
        p = subprocess.Popen(argv, cwd=E, stdout=stdout, stderr=stderr, start_new_session=True)
        ident = process(p.pid)
        save(d/'launch.json', dict(process=ident, parent_process=process(os.getpid()), argv=argv))
        maximum = sum(r['stage_wall_seconds'][role]+300 for _, role in stages())+300
        try:
            rc = p.wait(timeout=maximum)
        except subprocess.TimeoutExpired:
            timed = True
            for path in (CONTROL/'control').glob('*/launch.json'):
                child = read(path)['process']
                if not gone(child):
                    os.killpg(child['pid'], signal.SIGTERM)
            os.killpg(p.pid, signal.SIGTERM)
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait(timeout=30)
    save(d/'actual_wait.json', dict(returncode=rc, seconds=time.monotonic()-start,
        child=ident, timed_out=timed, child_gone=gone(ident)))
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
