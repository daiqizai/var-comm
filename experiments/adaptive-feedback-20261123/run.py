"""Single-owner stage runner. CUDA is visible only in the scored image-reconstruction stage."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "direction-followups-20261123"))
from common import PY, TEMP_BASE, USER, gpu_snapshot, record

ROOT = USER / "analysis/var_comm_adaptive_feedback_20261123"
TMP = TEMP_BASE / "adaptive_feedback_exploration_20261123_v1"


def idle_device():
    samples = []
    for _ in range(5):
        samples.append(gpu_snapshot())
        time.sleep(1)
    good = [i for i in samples[-1] if all(s[i]["free_mib"] >= 49152 and s[i]["util"] <= 8
            and s[i]["temperature_c"] < 78 for s in samples)]
    if not good:
        return None, samples
    chosen = max(good, key=lambda i:min(s[i]["free_mib"] for s in samples))
    return samples[-1][chosen], samples


def run():
    record(ROOT / "owner.json", {"pid":os.getpid(), "start_ticks":Path("/proc/self/stat").read_text().split()[21],
           "started_epoch":time.time(), "host":os.uname().nodename, "python":str(PY)})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for stage in ("register","calibrate","prepare","score","analyze"):
        device = None
        if stage=="score":
            while device is None:
                device, samples = idle_device()
                if device is None:
                    print("Awaiting a free device for image scoring", flush=True)
                    time.sleep(30)
            record(ROOT / "device.json", {"chosen":device, "samples":samples})
            env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        start = time.time()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        argv = [str(PY), "-B", str(HERE / ("analyze.py" if stage=="analyze" else "study.py"))]
        if stage != "analyze":
            argv.append(stage)
        with (TMP / (stage+".log")).open("xb") as log:
            sub = subprocess.Popen(argv, env=env, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            record(ROOT / (stage+"_start.json"),{"pid":sub.pid,"epoch":start,"argv":argv,
                   "start_ticks":Path(f"/proc/{sub.pid}/stat").read_text().split()[21]})
            try:
                code = sub.wait(timeout=2700 if device else 1800)
            except subprocess.TimeoutExpired:
                sub.terminate()
                code = sub.wait()
        end = time.time()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record(ROOT / (stage+"_wait.json"), {"returncode":code,"start_epoch":start,"end_epoch":end,
               "wall_s":end-start,"gpu_device_s":end-start if device else 0.,
               "cpu_process_s":(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)})
        if code:
            raise RuntimeError(f"{stage} exited with {code}")
    record(ROOT / "done.json", {"epoch":time.time(), "stages":5})


if __name__=="__main__":
    try:
        run()
    except Exception:
        record(ROOT / "failure.json", {"epoch":time.time(),"exception":traceback.format_exc()})
        raise
