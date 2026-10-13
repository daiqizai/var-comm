"""Audited zero-call metadata recovery, then one statistics/figure delivery.

The original inference and failed scorer registration are immutable. The first
scorer failed on a missing metadata alias before launch/model/reference/score.
This owner never retries attempted scientific calls.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/generalization_kodak_final_20261013'
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam');RT=BASE/'code/liulu/var_comm_runtime_20261010'
ASSETS=BASE/'liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/kodak_final_20261013_v1'
PY=RT/'envs/frozen_visual_py310/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(stage):
 lock=(OUT/f'owner_{stage}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/f'owner_{stage}.json').exists(),'Do not repeat an attempted stage'
 e=Path(__file__).parent
 if stage=='scoring_recovery':
  failed=read(OUT/'metrics_v1/failure.json');wait=read(OUT/'actual_wait_scoring.json')
  assert wait['returncode']==1 and failed['counts']=={} and failed['completed_metric_rows']==0
  assert "KeyError: 'original_score_completion'" in failed['traceback']
  assert not (OUT/'metrics_v1/launch.json').exists()
  assert read(OUT/'actual_wait_inference.json')['returncode']==0
  reg=read(OUT/'registration.json')
  for p,h in reg['code_bindings'].items():assert sha(p)==h,p
  recovery=OUT/'scoring_recovery_v1';recovery.mkdir(exist_ok=False)
  reg['original_score_completion']=reg['old_score_completion']
  reg['code_bindings'][str(Path(__file__).resolve())]=sha(__file__)
  reg['recovery_origin']=dict(original_registration_sha256=sha(OUT/'registration.json'),failed_metadata_sha256=sha(OUT/'metrics_v1/failure.json'),actual_exit_sha256=sha(OUT/'actual_wait_scoring.json'),diagnosis='Missing original_score_completion metadata alias; zero model/reference/score calls',scientific_retry=False,additional_scientific_budget=0)
  save(recovery/'registration.json',reg)
  cmd=[str(PY),str(e/'score_final.py'),'--registration',str(recovery/'registration.json'),'--frames',str(OUT/'frames.json'),'--out',str(OUT/'metrics_final_v1')]
 else:
  assert read(OUT/'actual_wait_scoring_recovery.json')['returncode']==0
  assert read(OUT/'metrics_final_v1/completion.json')['metric_rows']==1728
  cmd=[str(PY),str(e/'postprocess.py'),'--rows',str(OUT/'metrics_final_v1/rows.csv'),'--completion',str(OUT/'metrics_final_v1/completion.json'),'--old-root',str(ROOT/'results/generalization_kodak_20261009'),'--reuse-audit',str(OUT/'anchor_reuse_audit.json'),'--out',str(OUT/'analysis_v1')]
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',BASELINE_GPU_ID='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
 started=time.time()
 with (ASSETS/f'{stage}.log').open('xb') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  save(OUT/f'owner_{stage}.json',dict(owner_pid=os.getpid(),worker_pid=child.pid,command=cmd,started_unix=started))
  code=child.wait()
 save(OUT/f'actual_wait_{stage}.json',dict(returncode=code,worker_pid=child.pid,owner_pid=os.getpid(),started_unix=started,ended_unix=time.time(),automatic_retry=False,log=str(ASSETS/f'{stage}.log')))
 raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['scoring_recovery','postprocess']);run(p.parse_args().stage)
