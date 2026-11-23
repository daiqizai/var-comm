"""Calibrate the 2-bit ADC and run the two frozen RF receivers on CPU."""
import os
import resource
import subprocess
import sys
import time
from common import PY, STUDY, TEMP, digest, include_sources, record


def main():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=include_sources(),
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="12",
               OPENBLAS_NUM_THREADS="12", MKL_NUM_THREADS="12")
    root = STUDY/"rf"
    subprocess.run([PY,"-B",STUDY/"rf.py","register"],env=env,cwd=STUDY,check=True)
    for stage,limit in (("calibrate",900),("test",7200)):
        cmd = [str(PY),"-B",str(STUDY/"rf.py"),stage,"cpu"]
        with (TEMP/"logs"/("rf_"+stage+".log")).open("xb") as log:
            start = time.time()
            cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
            p = subprocess.Popen(cmd,env=env,cwd=STUDY,stdout=log,stderr=subprocess.STDOUT)
            with open(f"/proc/{p.pid}/stat") as status:
                ticks = status.read().split(") ")[1].split()[19]
            record(root/(stage+"_start.json"),{"pid":p.pid,"ticks":ticks,"unix":start,"command":cmd,
                   "code_sha256":digest(STUDY/"rf.py"),"protocol_sha256":digest(root/"protocol.json")})
            try:
                rc = p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    rc = p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait()
            end = resource.getrusage(resource.RUSAGE_CHILDREN)
            record(root/(stage+"_wait.json"),{"pid":p.pid,"actual_wait":rc,"wall_s":time.time()-start,
                   "cpu_process_s":end.ru_utime+end.ru_stime-cpu.ru_utime-cpu.ru_stime})
            if rc:
                raise RuntimeError(f"{stage} exited {rc}")


if __name__ == "__main__":
    main()
