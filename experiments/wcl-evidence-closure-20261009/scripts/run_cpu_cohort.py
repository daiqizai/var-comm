"""Own one finite CPU cohort; never admit GPU work or retry failed calls."""
import argparse, fcntl, hashlib, json, os
from pathlib import Path
import signal, subprocess, time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--runner', required=True)
    p.add_argument('--record-dir', required=True)
    a = p.parse_args()
    request = Path(a.request).resolve()
    r = json.loads(request.read_text())
    runner = Path(a.runner).resolve()
    record = Path(a.record_dir).resolve()
    assert runner.is_file() and record.is_relative_to(Path(r['out']).resolve())
    record.mkdir(parents=True, exist_ok=False)
    lock = (record/'owner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rh = hashlib.sha256(request.read_bytes()).hexdigest()
    children, handles = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    def interrupted(*_):
        raise InterruptedError('Owned cohort interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.time()
    try:
        for i in range(r['workers']):
            log = (record/f'worker_{i:02d}.log').open('xb')
            handles.append(log)
            child = subprocess.Popen([r['python_cpu'], '-B', str(runner), 'cpu-worker',
                '--request', str(request), '--index', str(i)], cwd=r['root'], env=env,
                stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        save(record/'launch.json', dict(pid=os.getpid(), request_sha256=rh,
            runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            children=[c.pid for c in children], started_unix=started))
        while any(c.poll() is None for c in children):
            if time.time() > r['deadline_unix']:
                raise RuntimeError('Registered deadline reached')
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError('Worker failed; preserved evidence; no retry')
            time.sleep(2)
        assert all(c.returncode == 0 for c in children)
        result = dict(status='CPU_COHORT_COMPLETE', exit_code=0)
    except BaseException as e:
        for c in children:
            if c.poll() is None:
                c.send_signal(signal.SIGTERM)
        result = dict(status='CPU_COHORT_FAILED', exit_code=1, error=repr(e))
    finally:
        for c in children:
            c.wait()
        for h in handles:
            h.close()
    result.update(request_sha256=rh, child_exit_codes=[c.returncode for c in children],
                  elapsed_seconds=time.time()-started, GPU_launched=False)
    save(record/'exit.json', result)
    print(json.dumps(result), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
