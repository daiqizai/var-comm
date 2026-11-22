"""Own one registered evaluation, accounting for its pair of real training exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from study import PYTHON, SPEC, STUDY, TEMP, dump_once, regular_path


def run(track):
    q = SPEC[track]
    root = TEMP/"runs"/track
    regular_path(root)
    worker = STUDY/"evaluate_pairs.py"
    with (root/"evaluation_claim.json").open("x") as f:
        json.dump({"owner":os.getpid(),"time":time.time(),
                  "worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest()},f)
    spent = 0.
    training_owner = json.loads((root/"claim.json").read_text())["pid"]
    for arm in q["arms"]:
        path = root/f"{arm}_wait.json"
        while not path.exists():
            if not Path(f"/proc/{training_owner}").exists():
                raise RuntimeError(f"Missing actual training exit for {arm}")
            time.sleep(15)
        receipt = json.loads(path.read_text())
        if receipt["actual_wait"] != 0:
            raise RuntimeError(f"Training failed: {arm}: {receipt}")
        spent += receipt["elapsed_s"]
    remaining = 6*3600-spent
    if remaining<=0:
        raise RuntimeError("Training used this track's six-device-hour reservation")
    cache = TEMP/"cache"/track/"evaluation"
    regular_path(cache)
    cache.mkdir(parents=True,exist_ok=False)
    env = {**os.environ,"CUDA_VISIBLE_DEVICES":str(q["gpu"]),"CUDA_DEVICE_ORDER":"PCI_BUS_ID",
           "PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","HF_HUB_OFFLINE":"1",
           "TRANSFORMERS_OFFLINE":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8",
           "TMPDIR":str(cache),"XDG_CACHE_HOME":str(cache/"xdg"),"CUDA_CACHE_PATH":str(cache/"cuda"),
           "OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"}
    start = time.time()
    with (root/"evaluation.log").open("x") as log:
        p = subprocess.Popen([str(PYTHON),"-B",str(worker),track],env=env,cwd=STUDY,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        dump_once(root/"evaluation_start.json",{"pid":p.pid,
                  "start_ticks":Path(f"/proc/{p.pid}/stat").read_text().split(") ")[1].split()[19],
                  "start_unix":start,"gpu":q["gpu"],"available_seconds":remaining})
        try:
            rc = p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.terminate()
            rc = p.wait(timeout=45)
        dump_once(root/"evaluation_wait.json",{"pid":p.pid,"actual_wait":rc,"elapsed_s":time.time()-start,
                  "measured_training_s":spent,"end_unix":time.time()})
    sys.exit(rc)


if __name__=="__main__":
    run(sys.argv[1])
