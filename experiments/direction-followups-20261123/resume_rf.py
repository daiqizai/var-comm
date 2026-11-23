"""Run the CPU-placement correction after the recorded pre-inference failure."""
import os
import resource
import subprocess
import time
from common import PY, STUDY, TEMP, digest, include_sources, read, record


def main():
    root = STUDY/"rf"
    assert read(root/"calibrate_wait.json")["actual_wait"] == 1
    assert not (root/"calibration.json").exists() and not (root/"test_start.json").exists()
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=include_sources(),PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="12",MKL_NUM_THREADS="12",OPENBLAS_NUM_THREADS="12")
    subprocess.run([PY,"-B",STUDY/"rf_cpu.py","prepare"],env=env,cwd=STUDY,check=True)
    for stage,tag,limit in (("calibrate","calibrate_cpu",900),("test","test",7200)):
        cmd = [str(PY),"-B",str(STUDY/"rf_cpu.py"),stage]
        with (TEMP/"logs"/("rf_"+tag+".log")).open("xb") as log:
            start = time.time()
            first = resource.getrusage(resource.RUSAGE_CHILDREN)
            p = subprocess.Popen(cmd,cwd=STUDY,env=env,stdout=log,stderr=subprocess.STDOUT)
            with open(f"/proc/{p.pid}/stat") as f:
                ticks = f.read().split(") ")[1].split()[19]
            record(root/(tag+"_start.json"),{"pid":p.pid,"ticks":ticks,"unix":start,"command":cmd,
                   "code_sha256":digest(STUDY/"rf_cpu.py"),"study_sha256":digest(STUDY/"rf.py"),
                   "protocol_sha256":digest(root/"protocol.json")})
            try:
                rc = p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                p.terminate()
                try: rc = p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait()
            end = resource.getrusage(resource.RUSAGE_CHILDREN)
            record(root/(tag+"_wait.json"),{"pid":p.pid,"actual_wait":rc,"wall_s":time.time()-start,
                   "cpu_process_s":end.ru_utime+end.ru_stime-first.ru_utime-first.ru_stime})
            if rc:
                raise RuntimeError(f"{tag} exited {rc}")


if __name__ == "__main__":
    main()
