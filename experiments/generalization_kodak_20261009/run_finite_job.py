"""Run an explicitly bound finite command sequence and record actual child exits."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(path):
    request = json.loads(Path(path).read_text())
    assert request['schema'] == 'KODAK24_FINITE_JOB_V1'
    assert request['runner_sha256'] == sha(__file__)
    assert request['cwd'] == '/home/liulu/projects/VAR_COMM'
    assert 1 <= len(request['steps']) <= 4
    out = Path(request['out'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'owner.lock').open('ab') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out/'start.json').exists(), 'Existing job retained; no automatic duplicate or retry'
        for entry in request['inputs']:
            assert sha(entry['path']) == entry['sha256'], entry['path']
        save(out/'start.json', dict(pid=os.getpid(), request_sha256=sha(path), time=time.time()))
        children = []
        active = None
        def stop(signum, frame):
            raise InterruptedError('Finite owner received signal '+str(signum))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        try:
            for index, step in enumerate(request['steps']):
                assert time.time() < request['deadline_unix']
                for entry in step.get('inputs', []):
                    assert sha(entry['path']) == entry['sha256'], entry['path']
                with (out/f'step_{index:02d}.log').open('xb') as log:
                    active = subprocess.Popen(step['argv'], cwd=request['cwd'],
                        env=dict(os.environ, **request.get('environment', {})),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    child = dict(index=index, pid=active.pid, argv=step['argv'], started_unix=time.time())
                    save(out/f'step_{index:02d}_launch.json', child)
                    remaining = request['deadline_unix']-time.time()
                    code = active.wait(timeout=max(0.1, remaining))
                    child.update(exit_code=code, ended_unix=time.time())
                    children.append(child)
                    log.flush(); os.fsync(log.fileno())
                    save(out/f'step_{index:02d}_wait.json', child)
                    active = None
                    assert code == 0, 'Child failed; no successor or automatic retry'
            save(out/'completion.json', dict(status='FINITE_JOB_ACTUAL_WAIT_ZERO',
                request_sha256=sha(path), children=children, completed_unix=time.time()))
        except BaseException:
            if active is not None and active.poll() is None:
                os.killpg(active.pid, signal.SIGTERM)
                try:
                    active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
            save(out/'failure.json', dict(status='FAILED_NO_AUTOMATIC_SUCCESSOR',
                request_sha256=sha(path), children=children, traceback=traceback.format_exc(),
                active_exit_code=active.returncode if active is not None else None))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    run(args.request)
