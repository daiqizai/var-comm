"""Detached owner of the authorized B training followed by frozen evaluation."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs/RX-POSTERIOR-STEP2-B-20260930-R1'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in HERE.iterdir() if p.suffix in ('.py','.json','.md') and p.name!='README.md'}


def status(**kw):
    path=OUT/'supervisor_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),**kw),indent=2));os.replace(tmp,path)


status(status='STARTING',bindings=bindings)
stages=[('qualification','qualification.py',OUT/'training/qualification_resumed_parent.json'),
        ('training','train.py',OUT/'training/completion.json'),
        ('evaluation','evaluate.py',OUT/'evaluation/completion.json'),
        ('publication','publish.py',OUT/'publication.json')]
for stage,script,receipt in stages:
    if receipt.exists():continue
    attempt=0
    while True:
        for path,expected in bindings.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                status(status='FAILED',stage=stage,reason='bound source changed',path=path)
                raise RuntimeError('bound source changed '+path)
        attempt+=1
        with (OUT/f'{stage}_attempt_{attempt:03d}_{time.time_ns()}.log').open('a') as output:
            worker=subprocess.Popen([sys.executable,'-u',str(HERE/script)],stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT)
            status(status='RUNNING',stage=stage,worker_pid=worker.pid,attempt=attempt,bindings=bindings)
            code=worker.wait()
        if code==75:
            status(status='WAITING_FOR_SAFE_RESOURCE',stage=stage,attempt=attempt)
            time.sleep(30);continue
        if code!=0 or not receipt.exists():
            status(status='FAILED',stage=stage,exit_code=code,attempt=attempt,receipt_exists=receipt.exists())
            raise SystemExit(code or 1)
        break
status(status='COMPLETE',stages=[s[0] for s in stages],bindings=bindings)
