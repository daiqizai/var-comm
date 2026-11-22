"""Run each finite matched training pair once and write observed wait status."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from study import PYTHON, SPEC, STUDY, SWIN, TEMP, WEIGHTS, dump_once, regular_path


def main(track):
    q = SPEC[track]
    assert (STUDY / "qualification.json").is_file()
    ledger_root = TEMP / "runs" / track
    regular_path(ledger_root)
    ledger_root.mkdir(parents=True, exist_ok=True)
    # Persistent claim prevents accidental restarts after either success or failure.
    claim = ledger_root / "claim.json"
    with claim.open("x") as out:
        json.dump({"pid": os.getpid(), "start": time.time(), "protocol_sha256":
                   hashlib.sha256((STUDY / "protocol.json").read_bytes()).hexdigest()}, out)
    for arm in q["arms"]:
        outdir = WEIGHTS / track / arm
        resultdir = STUDY / "training" / track / arm
        if outdir.exists() or resultdir.exists():
            raise FileExistsError((outdir, resultdir))
        cache = TEMP / "cache" / track / arm
        regular_path(cache)
        cache.mkdir(parents=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(q["gpu"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "TMPDIR":str(cache), "XDG_CACHE_HOME":str(cache/"xdg"), "CUDA_CACHE_PATH":str(cache/"cuda"),
               "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4"}
        args = [str(PYTHON), "-B", str(TEMP/"source/experiments"/q["project"]/"scripts"/q["train"]),
                "--track", "noma" if track=="noma" else "holorate", "--method", q["method"], "--run-tag", arm,
                "--total-steps", str(q["steps"]), "--seed",str(q["seed"]), "--batch-size","16",
                "--lr","0.0003","--eval-interval","500","--physical-gpu",str(q["gpu"]),
                "--ckpt-dir",str(outdir),"--summary-dir",str(resultdir),"--backbone-ckpt",str(SWIN)]
        start = time.time()
        with (ledger_root/f"{arm}.log").open("x") as log:
            child = subprocess.Popen(args, env=env, cwd=STUDY, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            dump_once(ledger_root/f"{arm}_start.json",{"pid":child.pid,"proc_start_ticks":
                      Path(f"/proc/{child.pid}/stat").read_text().split(") ")[1].split()[19],
                      "command":args,"start_unix":start,"physical_gpu":q["gpu"]})
            try:
                rc = child.wait(timeout=3*3600)
            except subprocess.TimeoutExpired:
                child.terminate()
                rc = child.wait(timeout=45)
            end = time.time()
            dump_once(ledger_root/f"{arm}_wait.json",{"pid":child.pid,"actual_wait":rc,
                      "elapsed_s":end-start, "ended_unix":end})
            if rc:
                raise SystemExit(rc)
    print("Both training workers exited successfully.",flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
