"""Detached sequential owner of the explicitly authorized N512 study."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs/EXTREME-BW-20260930-R1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary, path)


def receipt_complete(stage, receipt):
    if not receipt.exists():
        return False
    record = json.loads(receipt.read_text())
    if record.get('status') != stage['complete_status']:
        return False
    if any(record.get(k) != v for k, v in stage.get('required_fields', {}).items()):
        return False
    if stage['name'] == 'publication':
        return bool(record.get('commit')) and record['commit'] == record.get('remote_commit') and record.get('checks') == 'PASS'
    return True


def main():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    stages = json.loads((HERE / 'stages.json').read_text())
    if protocol['N'] != 512 or protocol['automatic_N1024']:
        raise RuntimeError('Only the explicitly authorized N512 stage may run')
    bindings = {str(p): sha(p) for p in HERE.iterdir()
                if p.suffix in ('.py', '.json', '.md', '.csv') and p.name != 'README.md'}
    launch = OUT / 'supervisor_registration.json'
    record = dict(N=512, automatic_N1024=False, stages=stages, source_bindings=bindings)
    if launch.exists():
        if json.loads(launch.read_text()) != record:
            raise RuntimeError('Existing supervisor registration changed')
    else:
        write(launch, record)

    def status(**fields):
        write(OUT / 'supervisor_status.json', dict(pid=os.getpid(), time=time.time(), **fields))

    def verify_sources():
        for path, expected in bindings.items():
            if sha(path) != expected:
                raise RuntimeError('Bound source changed: ' + path)

    try:
        status(status='STARTING')
        for stage in stages:
            verify_sources()
            receipt = ROOT / stage['receipt']
            if receipt_complete(stage, receipt):
                status(status='RECEIPT_PRESENT', stage=stage['name'], receipt=str(receipt))
                continue
            attempt = 0
            while True:
                verify_sources()
                attempt += 1
                log = OUT / f"{stage['name']}_attempt_{attempt:03d}_{time.time_ns()}.log"
                with log.open('a') as handle:
                    worker = subprocess.Popen([sys.executable, '-u', str(HERE / stage['script']),
                                               *stage.get('args', [])], cwd=ROOT,
                                              stdin=subprocess.DEVNULL, stdout=handle,
                                              stderr=subprocess.STDOUT)
                    status(status='RUNNING', stage=stage['name'], worker_pid=worker.pid,
                           attempt=attempt, log=str(log))
                    code = worker.wait()
                if code == 75:
                    status(status='WAITING_FOR_SAFE_RESOURCE', stage=stage['name'], attempt=attempt)
                    time.sleep(30)
                    continue
                if code != 0 or not receipt_complete(stage, receipt):
                    status(status='FAILED', stage=stage['name'], exit_code=code,
                           receipt_exists=receipt.exists(), log=str(log))
                    raise RuntimeError('Stage failed: ' + stage['name'])
                break
        verify_sources()
        status(status='COMPLETE', stages=[s['name'] for s in stages])
    except Exception as error:
        status(status='FAILED', error=str(error))
        raise


if __name__ == '__main__':
    main()
