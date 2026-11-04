from pathlib import Path
import json,os,sys,hashlib,time,subprocess
from owned_process import run_role
L=Path(sys.argv[1]);S=Path(__file__).resolve().parent
assert json.loads((L/'p0_audit_v2/actual_wait.json').read_text())['returncode']==0
assert json.loads((L/'p0_chain_v2/complete.json').read_text())['passed']
bindings=json.loads((L/'training_qualification_bindings.json').read_text())
assert not (L/'training_qualification_v1').exists()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
assert 'GPU-a9818dee-2885-eacd-d344-8b1f35b82798' not in apps
U=L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONNOUSERSITE='1',
    PYTHONPATH=str(L/'environment/site')+':'+str(U)+':'+str(S),CUBLAS_WORKSPACE_CONFIG=':4096:8')
command=[sys.executable,'-u',str(S/'qualify_training.py'),'--root',str(L),'--output',str(L/'training_qualification_v1')]
run_role(L,'training_qualification_v1',command,env,[0,1],['GPU-a9818dee-2885-eacd-d344-8b1f35b82798'],2400,bindings)
