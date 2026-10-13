"""Single-attempt subprocess owner; preserves actual exits, never retries science."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generalization_kodak_final_20261013'
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
RT=BASE/'code/liulu/var_comm_runtime_20261010'
ASSETS=BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/kodak_final_20261013_v1'
PY=RT/'envs/frozen_visual_py310/bin/python'
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(stage):
 lock=(OUT/f'owner_{stage}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/f'owner_{stage}.json').exists(),'Existing owner: inspect actual wait; no repeated call'
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',BASELINE_GPU_ID='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
 e=Path(__file__).parent
 commands={'inference':[str(PY),str(e/'run_final.py'),'run'],
 'scoring':[str(PY),str(e/'score_final.py'),'--registration',str(OUT/'registration.json'),'--frames',str(OUT/'frames.json'),'--out',str(OUT/'metrics_v1')],
 'postprocess':[str(PY),str(e/'postprocess.py'),'--rows',str(OUT/'metrics_v1/rows.csv'),'--completion',str(OUT/'metrics_v1/completion.json'),'--old-root',str(ROOT/'results/generalization_kodak_20261009'),'--reuse-audit',str(OUT/'anchor_reuse_audit.json'),'--out',str(OUT/'analysis_v1')]}
 cmd=commands[stage];started=time.time()
 with (ASSETS/f'{stage}.log').open('xb') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save(OUT/f'owner_{stage}.json',dict(owner_pid=os.getpid(),worker_pid=child.pid,command=cmd,started_unix=started))
  code=child.wait()
 save(OUT/f'actual_wait_{stage}.json',dict(returncode=code,owner_pid=os.getpid(),worker_pid=child.pid,started_unix=started,ended_unix=time.time(),automatic_retry=False,log=str(ASSETS/f'{stage}.log')))
 raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inference','scoring','postprocess']);run(p.parse_args().stage)
