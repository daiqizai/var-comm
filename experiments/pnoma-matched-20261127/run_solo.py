"""Run the registered diagnostic only after the separate primary timing worker has exited."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from launch import HERE, TEMP, PY, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--owner", action="store_true")
    args = parser.parse_args()
    job = TEMP / "job_solo"
    if not args.owner:
        assert json.loads((TEMP/"job_evaluate/wait.json").read_text())["actual_wait"] == 0
        assert (HERE/"evaluate/done.json").is_file()
        job.mkdir(exist_ok=False)
        with (job/"owner.log").open("xb") as f:
            p = subprocess.Popen([sys.executable, "-B", __file__, *sys.argv[1:], "--owner"],
                stdout=f, stderr=f, stdin=subprocess.DEVNULL, start_new_session=True)
        print("solo owner", p.pid)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLCONFIGDIR=str(TEMP/"mpl_solo"))
    start = time.time()
    names = ["solo.py", "run_solo.py", "solo_protocol.json", "protocol.json", "adapter.py", "common.py", "analyze.py"]
    record(job/"registration.json", dict(started=start, gpu=args.gpu, seconds_limit=3600,
        source={n: hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}))
    with (job/"worker.log").open("xb") as f:
        p = subprocess.Popen([str(PY), "-B", str(HERE/"solo.py")], env=env,
                             stdout=f, stderr=f, stdin=subprocess.DEVNULL)
        record(job/"pid.json", dict(owner=os.getpid(), child=p.pid,
               start_ticks=Path(f"/proc/{p.pid}/stat").read_text().split()[21]))
        try:
            rc = p.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
        record(job/"wait.json", dict(actual_wait=rc, ended=time.time(), elapsed_s=time.time()-start))

if __name__ == "__main__":
    main()
