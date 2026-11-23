"""Run the preregistered 4B receiver comparison after a device becomes available."""
import os
import subprocess
import sys
import time
from after_training import wait_for_roles
from common import PY, STUDY, TEMP, digest, include_sources, read, record
from worker import admission


def main(gpu):
    root=STUDY/"pic"
    assert read(root/"protocol.json")["selection"]["coefficient_grid"]==[0.,.25,.5,.75,1.]
    wait_for_roles(["feedback"])
    time.sleep(10)
    samples=admission(gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
             PYTHONPATH=include_sources(),CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="4",
             OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
    with (TEMP/"logs/pic.log").open("xb") as log:
        cmd=[str(PY),"-B",str(STUDY/"pic.py"),"run"]
        start=time.time()
        p=subprocess.Popen(cmd,cwd=STUDY,env=env,stdout=log,stderr=subprocess.STDOUT)
        with open(f"/proc/{p.pid}/stat") as f:
            ticks=f.read().split(") ")[1].split()[19]
        record(root/"start.json",{"pid":p.pid,"start_ticks":ticks,"started_unix":start,"command":cmd,"admission":samples,
               "physical_gpu":gpu,"code_sha256":digest(STUDY/"pic.py"),"protocol_sha256":digest(root/"protocol.json")})
        try:
            rc=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: rc=p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                rc=p.wait()
        record(root/"wait.json",{"pid":p.pid,"actual_wait":rc,"device_seconds":time.time()-start})
        if rc: raise RuntimeError(f"Receiver comparison exited {rc}")
    subprocess.run([PY,"-B",STUDY/"pic_stats.py"],cwd=STUDY,env=dict(env,CUDA_VISIBLE_DEVICES=""),check=True)


if __name__=="__main__":
    main(int(sys.argv[1]))
