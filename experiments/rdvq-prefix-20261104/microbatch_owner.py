from pathlib import Path
import json,os,subprocess,sys
from owned_process import run_role
L=Path(sys.argv[1]);S=Path(__file__).resolve().parent;uuid='GPU-a9818dee-2885-eacd-d344-8b1f35b82798'
assert json.loads((L/'training_qualification_v2/actual_wait.json').read_text())['returncode']==0
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S))
run_role(L,'microbatch4_qualification_v1',[sys.executable,'-u',str(S/'microbatch_qualification.py'),str(L)],env,[0,1],[uuid],1800,json.loads((L/'microbatch4_bindings.json').read_text()))
