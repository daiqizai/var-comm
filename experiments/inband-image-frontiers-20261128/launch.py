"""Launch each independent registered role once and retain the worker's actual wait status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path("/mnt/pfs/pfs-yc2F4O/modelTeam")
TEMP = BASE / "liberai-tmp/liulu/liulu-var-comm/inband_image_frontiers_20261128_v1"
PY = BASE / "code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python"
ROLES = {"activity":"activity.py","separation":"separation.py","frontend":"frontend.py","drift":"drift.py"}


def save(path,data):
    with path.open("x") as f:
        json.dump(data,f,indent=2)


def execute(role,gpu,owner=False):
    job=TEMP/("job_"+role)
    if not owner:
        job.mkdir(parents=True,exist_ok=False)
        with (job/"owner.log").open("x") as log:
            p=subprocess.Popen([sys.executable,"-B",__file__,role,str(gpu),"--owner"],
                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(role,"owner",p.pid)
        return
    begin=time.time()
    deps=["protocol.json","study.py","launch.py",ROLES[role]]
    if role in ("frontend","drift"):
        deps.append("single_link.py")
    if role in ("separation","activity"):
        deps.append("access.py")
    hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in deps}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
          OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8",
          MPLCONFIGDIR=str(TEMP/("mpl_"+role)))
    save(job/"registration.json",{"role":role,"gpu":gpu,"limit":5400,"sources":hashes,"started":begin})
    with (job/"worker.log").open("x") as log:
        child=subprocess.Popen([str(PY),"-Bu",str(HERE/ROLES[role])],cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        ticks=Path(f"/proc/{child.pid}/stat").read_text().split()[21]
        save(job/"process.json",{"owner":os.getpid(),"child":child.pid,"start_ticks":ticks})
        try:
            rc=child.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            child.terminate()
            try: rc=child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                rc=child.wait()
        save(job/"wait.json",{"actual_wait":rc,"wall_seconds":time.time()-begin,"end":time.time()})
        if rc: sys.exit(rc)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("role",choices=ROLES)
    p.add_argument("gpu",type=int)
    p.add_argument("--owner",action="store_true")
    args=p.parse_args()
    execute(args.role,args.gpu,args.owner)

