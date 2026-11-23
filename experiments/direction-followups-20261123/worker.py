"""Run a single registered role with device admission, immutable claim and actual child wait."""
import argparse
import os
import subprocess
import sys
import time
import traceback

from common import (
    CKPT, D5, PY, SEED, SHARED, STEPS, STUDY, TEMP, WAVE_ARMS, digest,
    gpu_snapshot, include_sources, path_check, read, record,
)


def admission(gpu):
    samples = []
    for n in range(5):
        samples.append(gpu_snapshot()[gpu])
        if n < 4:
            time.sleep(1)
    if not all("A800" in s["name"] and s["free_mib"] >= 48000 and s["temperature_c"] < 76 for s in samples):
        raise RuntimeError(f"Insufficient memory or thermal headroom: {samples}")
    if max(s["util"] for s in samples) > 25:
        raise RuntimeError(f"Shared device in active use: {samples}")
    return samples


def run(role, gpu):
    cfg = read(STUDY / "protocol.json")
    for rel, ids in cfg["references"]["source_files"].items():
        if digest(TEMP / rel) != ids["after"]:
            raise ValueError(rel)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=include_sources(), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    if role in WAVE_ARMS:
        assert (STUDY / "qualify_cache/wait.json").exists() and read(STUDY/"qualify_cache/wait.json")["actual_wait"] == 0
        command = [PY, TEMP / "source" / D5 / "scripts/train_dir5_otfs_genvar.py",
                   "--mode", "train", "--track", "otfs", "--method", WAVE_ARMS[role], "--run-tag", role,
                   "--physical-gpu", str(gpu), "--total-steps", str(STEPS), "--eval-every", "700",
                   "--batch-size", "28", "--seed", str(SEED), "--shared-cache-dir", SHARED,
                   "--out-dir", STUDY / "training" / role, "--ckpt-dir", CKPT / role]
        limit = 5400
    else:
        file = {"qualify_cache":"qualify.py","noma":"noma.py","wave_test":"wave_test.py"}[role]
        if role == "wave_test":
            assert all(read(STUDY / arm / "wait.json")["actual_wait"] == 0 for arm in WAVE_ARMS)
        command, limit = [PY, STUDY / file, str(gpu)], 1800
    out = path_check(STUDY / role)
    if out.exists():
        raise FileExistsError(out)
    samples = admission(gpu)
    out.mkdir(exist_ok=False)
    log = path_check(TEMP / "logs" / (role + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("xb") as stdout:
        p = subprocess.Popen([str(a) for a in command], cwd=STUDY, env=env, stdout=stdout, stderr=subprocess.STDOUT)
        ticks = open(f"/proc/{p.pid}/stat").read().split(") ")[1].split()[19]
        record(out / "start.json", {"host":os.uname().nodename, "pid":p.pid, "start_ticks":ticks,
               "started_unix":start,"command":[str(a) for a in command],"protocol_sha256":digest(STUDY/"protocol.json"),
               "script_sha256":digest(command[1]),"physical_gpu":gpu,"admission":samples})
        try:
            status = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                status = p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                status = p.wait()
        record(out / "wait.json", {"pid":p.pid,"actual_wait":status,"device_seconds":time.time()-start})
        if status:
            raise RuntimeError(f"{role} failed: {status}; retained {log}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=[*WAVE_ARMS,"qualify_cache","noma","wave_test"])
    p.add_argument("gpu",type=int)
    args = p.parse_args()
    run(args.role,args.gpu)
