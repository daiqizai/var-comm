"""Launch one bounded role and keep independent process, registration, and wait receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
TEMP=ROOT/"liberai-tmp/liulu/liulu-var-comm/pnoma_matched_20261127_v1"
PY=ROOT/"code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"

def record(path, data):
    with path.open("x") as f:
        json.dump(data,f,indent=2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("role",choices=("train","evaluate"))
    p.add_argument("--gpu",type=int,default=3)
    p.add_argument("--owner",action="store_true")
    a=p.parse_args()
    job=TEMP/("job_"+a.role)
    if not a.owner:
        if a.role=="evaluate":
            wait=json.loads((TEMP/"job_train/wait.json").read_text())
            assert wait["actual_wait"]==0 and (HERE/"train/freeze.json").is_file()
        job.mkdir(exist_ok=False)
        with (job/"owner.log").open("x") as f:
            proc=subprocess.Popen([sys.executable,"-B",__file__,*sys.argv[1:],"--owner"],stdin=subprocess.DEVNULL,
                    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(owner=proc.pid,role=a.role)))
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8",
        MPLCONFIGDIR=str(TEMP/("mpl_"+a.role)))
    begin=time.time()
    script=HERE/(a.role+".py")
    limit=10800 if a.role=="train" else 3600
    record(job/"registration.json",dict(role=a.role,started=begin,gpu=a.gpu,seconds_limit=limit,
        argv=[str(PY),"-B",str(script)],source={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
        for n in [script.name,"common.py","adapter.py","protocol.json"]}))
    with (job/"worker.log").open("x") as f:
        child=subprocess.Popen([str(PY),"-B",str(script)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        ticks=Path(f"/proc/{child.pid}/stat").read_text().split()[21]
        record(job/"pid.json",dict(owner=os.getpid(),child=child.pid,start_ticks=ticks))
        try:
            rc=child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                rc=child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                rc=child.wait()
        record(job/"wait.json",dict(actual_wait=rc,ended=time.time(),elapsed_s=time.time()-begin))
if __name__=="__main__":
    main()
