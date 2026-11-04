from pathlib import Path
import json,os,subprocess,sys
from owned_process import run_role
L=Path(sys.argv[1]);S=Path(__file__).resolve().parent
assert json.loads((L/'p0_audit_v2/actual_wait.json').read_text())['returncode']==0
uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e'
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S))
run_role(L,'p0_quality_v2',[sys.executable,'-u',str(S/'p0_quality_v2.py'),'--root',str(L)],env,[2,3],[uuid],1800,json.loads((L/'p0_quality_bindings_v2.json').read_text()))
