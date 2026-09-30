"""Detached supervisor for this inference study only; retries safe resource pauses."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs/RX-POSTERIOR-STEP2-A-20260930-R1'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.md')}
def status(**kw):
    p=OUT/'supervisor_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),**kw),indent=2));os.replace(tmp,p)
status(status='STARTING',bindings=bindings)
attempt=0
while True:
    for p,h in bindings.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('bound source changed '+p)
    attempt+=1
    with (OUT/f'worker_attempt_{attempt:03d}.log').open('a') as output:
        worker=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py')],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        status(status='RUNNING',worker_pid=worker.pid,attempt=attempt,bindings=bindings)
        code=worker.wait()
    if code==75:
        status(status='WAITING_FOR_SAFE_RESOURCE',attempt=attempt)
        time.sleep(30)
        continue
    status(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,attempt=attempt)
    raise SystemExit(code)
