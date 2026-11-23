"""Continue each evaluation exactly once after the corresponding training subprocess exits."""
import os
import subprocess
import sys
import time
from common import PY, STUDY, TEMP, digest, gpu_snapshot, include_sources, read, record
from worker import run


def wait_for_roles(roles):
    deadline=time.monotonic()+5500
    while time.monotonic()<deadline:
        waits=[STUDY/r/"wait.json" for r in roles]
        if any(p.exists() and read(p)["actual_wait"]!=0 for p in waits):
            raise RuntimeError("A prerequisite failed")
        if all(p.exists() for p in waits):
            for r in roles:
                start,done=read(STUDY/r/"start.json"),read(STUDY/r/"wait.json")
                assert start["pid"]==done["pid"] and done["actual_wait"]==0
            return
        time.sleep(10)
    raise TimeoutError(roles)


def feedback(gpu,env):
    root=STUDY/"feedback"
    assert read(root/"calibrate_wait.json")["actual_wait"]==0
    with (TEMP/"logs/feedback_test.log").open("xb") as log:
        start=time.time()
        cmd=[str(PY),"-B",str(STUDY/"feedback.py"),"test"]
        p=subprocess.Popen(cmd,cwd=STUDY,env=env,stdout=log,stderr=subprocess.STDOUT)
        with open(f"/proc/{p.pid}/stat") as f:
            ticks=f.read().split(") ")[1].split()[19]
        record(root/"start.json",{"pid":p.pid,"start_ticks":ticks,"started_unix":start,"command":cmd,
               "physical_gpu":gpu,"code_sha256":digest(STUDY/"feedback.py"),
               "protocol_sha256":digest(root/"protocol.json"),"estimator_sha256":digest(root/"estimator.json")})
        try:
            rc=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.terminate()
            try: rc=p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                rc=p.wait()
        record(root/"wait.json",{"pid":p.pid,"actual_wait":rc,"device_seconds":time.time()-start})
        if rc:
            raise RuntimeError(f"Feedback evaluation exited {rc}")


def main(stage,gpu):
    if stage not in ("feedback","wave_test"):
        raise ValueError(stage)
    prereqs=["no_spread"] if stage=="feedback" else ["no_spread","spread"]
    wait_for_roles(prereqs)
    if (STUDY/stage/"start.json").exists():
        raise RuntimeError("Role has already started")
    # Allow monitoring to pass the preceding job's last utilization window.
    time.sleep(8)
    deadline=time.monotonic()+1200
    observations=[]
    while time.monotonic()<deadline:
        a=gpu_snapshot()[gpu]
        observations.append(a)
        if len(observations)>=5 and all(z["util"]<=25 and z["free_mib"]>=48000 and z["temperature_c"]<76
                                        for z in observations[-5:]):
            break
        time.sleep(2)
    else:
        raise RuntimeError("No admission slot on the registered shared A800 after training")
    record(TEMP/"owners"/(stage+"_admission.json"),observations[-5:])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=include_sources(),PYTHONUNBUFFERED="1",
             PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
             CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if stage=="feedback":
        feedback(gpu,env)
        command=[PY,"-B",STUDY/"feedback_stats.py"]
    else:
        run("wave_test",gpu)
        command=[PY,"-B",STUDY/"statistics.py","wave_test"]
    subprocess.run(command,cwd=STUDY,env=dict(env,CUDA_VISIBLE_DEVICES=""),check=True)


if __name__=="__main__":
    main(sys.argv[1],int(sys.argv[2]))
