"""Explicit one-shot owner; no retries, old controllers, or periodic monitor."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path

BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
ROOT=BASE/'code/liulu/code/VAR_COMM'
TMP=BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011'
PY=BASE/'code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['P','Swin']);p.add_argument('mode',choices=['preflight','train']);p.add_argument('--gpu',required=True,type=int);p.add_argument('--attempt',default='v1');a=p.parse_args()
    assert a.gpu in range(4) and a.attempt.replace('_','').isalnum()
    run=TMP/'jobs'/f'{a.branch}_{a.mode}_{a.attempt}';run.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(a.gpu),BASELINE_GPU_ID=str(a.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6',TMPDIR=str(TMP/'tmp'),XDG_CACHE_HOME=str(TMP/'cache'),TORCH_HOME=str(TMP/'torch'),HF_HOME=str(TMP/'huggingface'),PYTHONNOUSERSITE='1')
    for key in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','HF_HOME']:Path(env[key]).mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),str(ROOT/'experiments/baseline-strength-20261011/run_continue.py'),a.branch,'--mode',a.mode]
    receipt=dict(branch=a.branch,mode=a.mode,attempt=a.attempt,gpu=a.gpu,owner_pid=os.getpid(),command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    os.sched_setaffinity(0,{7,8} if a.branch=='P' else {9,10});os.nice(10)
    start=time.monotonic()
    with (run/'worker.log').open('xb') as log:
        child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        receipt['worker_pid']=child.pid;(run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        code=child.wait()
    receipt.update(exit_code=code,elapsed_seconds=time.monotonic()-start,actual_child_waited=True)
    (run/'exit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
