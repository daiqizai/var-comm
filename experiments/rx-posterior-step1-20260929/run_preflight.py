"""Use the existing serial GPU thermal guard for a new isolated probe."""
import os,sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'experiments/var-latent-enhancement-20260917'
paths=[Path(__file__).parent,ROOT/'src',E/'src',E/'phase_b/src',E/'evaluation/src',E/'followup/src',E/'research/src',E/'mechanisms/src',ROOT/'experiments/var-short-prefix-hybrid-20260923/src',ROOT/'experiments/token_channel_efficiency_20260923/src']
os.environ['PYTHONPATH']=':'.join(map(str,paths))
os.environ['OMP_NUM_THREADS']='6';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for p in paths:sys.path.insert(0,str(p))
from token_efficiency import delivery_chain
from latent_enhancement.runtime import write_json
OUT=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929'
delivery_chain.CHAIN=OUT/'supervisor'
if __name__=='__main__':
 runner=delivery_chain.Runner()
 try:
  result=runner.run('calibration_engineering_preflight',['probe','--stage','preflight'],OUT/'preflight_complete.json')
  write_json(OUT/'supervisor_completion.json',dict(status='PREFLIGHT_COMPLETE_AWAITING_NEXT_AUTHORIZED_STAGE',result=result,time=time.time()))
 except Exception:
  if (OUT/'blocked.json').exists():
   write_json(OUT/'supervisor_completion.json',dict(status='ENGINEERING_GATE_STOP_NO_DEVELOPMENT_EVALUATION',time=time.time(),blocked=str(OUT/'blocked.json')))
  else:
   write_json(OUT/'supervisor_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
