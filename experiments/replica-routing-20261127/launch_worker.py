"""Detached job owner. A launch creates its unique directory and records the real child wait."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def write(path,data):
    with path.open("x") as f:
        json.dump(data,f,indent=2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["mechanism","development","formal","timing"])
    for n in ("scratch","study","parent","python"):
        p.add_argument(n,type=Path)
    p.add_argument("--gpu",type=int,default=3)
    p.add_argument("--own",action="store_true")
    a=p.parse_args()
    job=a.scratch/("job_"+a.stage)
    if not a.own:
        job.mkdir(exist_ok=False)
        argv=[sys.executable,"-B",__file__,*sys.argv[1:],"--own"]
        with (job/"owner.log").open("x") as f:
            owner=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"owner":owner.pid,"role":a.stage}))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),STUDY_SCRATCH=str(a.scratch),STUDY_PARENT=str(a.parent),
            MPLCONFIGDIR=str(a.scratch/("mpl_"+a.stage)),PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1")
    script="mechanism.py" if a.stage=="mechanism" else "routing_study.py"
    cmd=[str(a.python),"-B",str(a.study/script)]
    if a.stage!="mechanism":
        cmd.append(a.stage)
    write(job/"registration.json",dict(role=a.stage,argv=cmd,protocol_sha256=hashlib.sha256((a.study/"protocol.json").read_bytes()).hexdigest(),
          start_unix=time.time(),gpu=a.gpu,limit_s=3600))
    with (job/"worker.log").open("x") as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        write(job/"pid.json",{"owner":os.getpid(),"worker":child.pid,"start_ticks":Path(f"/proc/{child.pid}/stat").read_text().split()[21]})
        try:
            ret=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            child.terminate()
            try: ret=child.wait(15)
            except subprocess.TimeoutExpired: child.kill();ret=child.wait()
        write(job/"wait.json",{"actual_wait":ret,"end_unix":time.time()})

if __name__=="__main__":
    main()
