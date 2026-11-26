"""Run the independent secondary freeze after the primary GPU's real timing exit."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from continue_gpu import wait_job

def main():
    scratch,study,parent,py=map(Path,sys.argv[1:])
    # Continuation may still be running its preceding formal role when this owner starts.
    deadline=time.monotonic()+7350
    while not (scratch/"job_timing/registration.json").exists():
        if (scratch/"continuation_exit.json").exists() or time.monotonic()>deadline:
            raise RuntimeError("Primary timing role unavailable")
        time.sleep(5)
    assert wait_job(scratch/"job_timing")==0 and wait_job(scratch/"job_formal")==0
    job=scratch/"job_secondary"
    job.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="3",STUDY_SCRATCH=str(scratch),STUDY_PARENT=str(parent),
            PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1",MPLCONFIGDIR=str(scratch/"mpl_secondary"))
    cmd=[str(py),"-B",str(study/"secondary.py")]
    (job/"registration.json").write_text(json.dumps({"start_unix":time.time(),"argv":cmd,"limit_s":3600}))
    with (job/"worker.log").open("x") as f:
        p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        (job/"pid.json").write_text(json.dumps({"owner":os.getpid(),"worker":p.pid,"start_ticks":Path(f"/proc/{p.pid}/stat").read_text().split()[21]}))
        try: code=p.wait(3600)
        except subprocess.TimeoutExpired: p.terminate();code=p.wait(30)
        (job/"wait.json").write_text(json.dumps({"actual_wait":code,"end_unix":time.time()}))
    return code

if __name__=="__main__":
    sys.exit(main())
