"""Durable, fail-closed queue for the first registered training milestone only."""
import fcntl,os,signal,subprocess,sys,time
from pathlib import Path
from latent_enhancement.runtime import digest,write_json,verify_snapshot
from .common import OUT,ROOT,read
from .protocol import EXP,config

CHILD=None

def stop(signum,frame):
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)

def main():
    global CHILD
    cfg=config();OUT.mkdir(parents=True,exist_ok=True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    # Record exact source files for the whole queue before launching dependencies.
    files=[EXP/'protocol.json',*sorted((EXP/'src').rglob('*.py')),EXP/'scripts/run_first_matrix.sh']
    queue_identity={str(p):digest(p) for p in files}
    old=OUT/'queue_identity.json'
    if old.exists():
        if read(old)!=queue_identity:raise RuntimeError('queue code changed; inspect checkpoint before registering a new queue')
    else:write_json(old,queue_identity)
    commands=[]
    # Always rerun the real-weight shared-path checks before launching bulk work.
    for m in (6,7,8):commands.append(['short_prefix.train','--group',f'm{m}','--qualification-only','--cache-from-preflight'])
    commands.append(['short_prefix.qualify_execution'])
    for m in (6,7,8):
        for role in ('calibration','train'):commands.append(['short_prefix.cache','--role',role,'--m',str(m)])
        commands.append(['short_prefix.train','--group',f'm{m}','--until','20000'])
    commands.append(['short_prefix.train','--group','pure','--until','20000'])
    for number,command in enumerate(commands):
        verify_snapshot(queue_identity)
        path=logs/(f'queue_{number:02d}_{time.time_ns()}.log')
        write_json(OUT/'status.json',{'status':'RUNNING_FIRST_MATRIX','controller_pid':os.getpid(),'stage':number,'stages':len(commands),'command':command,'log':str(path),'timestamp':time.time(),'new_test':'NOT_OPENED'})
        print('START',number,command,flush=True)
        with path.open('w') as log:
            CHILD=subprocess.Popen([sys.executable,'-u','-m',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            code=CHILD.wait()
        if code:
            write_json(OUT/'status.json',{'status':'PAUSED' if code==75 else 'FAILED','stage':number,'returncode':code,'log':str(path),'timestamp':time.time(),'remaining':'Do not bypass failed qualification; inspect and resume exact checkpoint.'})
            raise SystemExit(code)
        print('COMPLETE',number,flush=True)
    write_json(OUT/'status.json',{'status':'FIRST_20K_MILESTONES_COMPLETE_PENDING_CALIBRATION_DECISION','timestamp':time.time(),'pending':['calibration-only extension decision','selected development and timing','N3060 confirmation','candidate/control training-seed repeats','content-selection diagnostic and conditional predictor','500 new communication-test sources only after freeze']})
if __name__=='__main__':main()
