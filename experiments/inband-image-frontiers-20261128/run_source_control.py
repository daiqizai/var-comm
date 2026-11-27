"""Launch the bounded secondary source-amplitude control after the first-round audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from launch import HERE,PY,TEMP,save

if __name__ == "__main__":
    gpu=int(sys.argv[1])
    job=TEMP/"job_source_predictor"
    if len(sys.argv)==2:
        assert (HERE/"ledger.json").is_file()
        job.mkdir(exist_ok=False)
        with (job/"owner.log").open("x") as f:
            p=subprocess.Popen([sys.executable,"-B",__file__,str(gpu),"owner"],cwd=HERE,start_new_session=True,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=f)
        print("source-amplitude control owner",p.pid)
    else:
        start=time.time()
        names=("protocol.json","source_predictor_protocol.json","source_predictor.py","analyze.py","single_link.py",
               "study.py","frontend.py","launch.py","run_source_control.py")
        hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
        save(job/"registration.json",dict(gpu=gpu,limit=1800,started=start,sources=hashes))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                 OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8",MPLCONFIGDIR=str(TEMP/"mpl_source_predictor"))
        with (job/"worker.log").open("x") as f:
            proc=subprocess.Popen([str(PY),"-Bu",str(HERE/"source_predictor.py")],cwd=HERE,env=env,
                                  stdin=subprocess.DEVNULL,stdout=f,stderr=f)
            ticks=Path(f"/proc/{proc.pid}/stat").read_text().split()[21]
            save(job/"process.json",dict(owner=os.getpid(),child=proc.pid,start_ticks=ticks))
            try:
                rc=proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:rc=proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc=proc.wait()
        save(job/"wait.json",dict(actual_wait=rc,wall_seconds=time.time()-start,end=time.time()))
        sys.exit(rc)

