"""Bound CPU concurrency without changing shards, blocks or measured outcomes."""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import time
from uep_common import read,write,require
STOP=False

def stop(*_):
    global STOP
    STOP=True

def alive(job):
    path=Path('/proc')/str(job['pid'])/'stat'
    return path.exists() and path.read_text().split()[21]==job['proc_start_ticks'] and path.read_text().split()[2]!='Z'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--concurrency',type=int,default=4);a=p.parse_args();require(1<=a.concurrency<=8,'CPU concurrency out of scope')
    import fcntl
    lock=(a.out/'coarse_dispatch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    launch=read(a.out/'coarse_launch.json');jobs=launch['jobs'];code=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        VAR_COMM_ROOT=str(a.root),PYTHONPATH=str(code)+':'+str(a.root/'src'),PYTHONDONTWRITEBYTECODE='1')
    completed=lambda j:(a.out/'phy_shards'/str(j['shard'])/'coarse_completion.json').exists()
    active=[j for j in jobs if alive(j) and not completed(j)]
    paused=[]
    for j in active[a.concurrency:]:
        os.kill(j['pid'],signal.SIGTERM);paused.append(dict(shard=j['shard'],pid=j['pid']))
    write(a.out/'coarse_concurrency_revision.json',dict(status='SCHEDULING_ONLY',concurrency=a.concurrency,threads_per_worker=2,
        original_concurrency=8,reason='Keep CPU LDPC off the existing GPU inference critical path; source/block/noise identities unchanged',
        checkpoint_pause_requests=paused,time=time.time()))
    while True:
        if STOP or (a.out/'STOP').exists():
            for j in jobs:
                if alive(j):os.kill(j['pid'],signal.SIGTERM)
            write(a.out/'coarse_dispatch_status.json',dict(status='PAUSED_CHECKPOINT_REQUESTED',time=time.time()));return
        done=[j for j in jobs if completed(j)]
        if len(done)==8:break
        active=[j for j in jobs if alive(j) and not completed(j)]
        for j in jobs:
            folder=a.out/'phy_shards'/str(j['shard'])
            require(not (folder/'failure.json').exists(),'Real PHY failure requires review')
            if completed(j) or alive(j) or len(active)>=a.concurrency:continue
            status=read(folder/'status.json') if (folder/'status.json').exists() else {}
            require(status.get('status')=='PAUSED','Worker exited unexpectedly; no automatic failure retry')
            with (folder/'worker.log').open('ab') as f:
                child=subprocess.Popen(j['args'],cwd=a.root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            os.setpriority(os.PRIO_PROCESS,child.pid,10)
            j['previous_pid']=j['pid'];j['pid']=child.pid;j['proc_start_ticks']=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21]
            active.append(j);launch['jobs']=jobs;write(a.out/'coarse_launch.json',launch)
        write(a.out/'coarse_dispatch_status.json',dict(status='RUNNING',completed_shards=len(done),active_pids=[j['pid'] for j in active],
            concurrency=a.concurrency,time=time.time()))
        time.sleep(15)
    write(a.out/'coarse_dispatch_status.json',dict(status='ALL_COARSE_SHARDS_COMPLETE',time=time.time()))

if __name__=='__main__':main()
