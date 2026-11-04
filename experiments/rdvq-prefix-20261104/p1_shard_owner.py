from pathlib import Path
import json,os,sys,subprocess,time
from owned_process import run_role
request=Path(sys.argv[1]);r=json.loads(request.read_text());L=Path(r['root']);S=Path(__file__).resolve().parent;O=Path(r['output_root'])
assert not O.exists();O.mkdir(parents=True)
uuid=r['gpu_uuid'];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert uuid not in apps
identity=subprocess.check_output(['nvidia-smi','-i',str(r['gpu_index']),'--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True).strip().split(',')
assert identity[0].strip()==uuid and identity[1].strip().lower()==r['gpu_pci'].lower()
assert json.loads((L/'p0_gate.json').read_text())['passed']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(r['gpu_index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
 PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S),
 TORCH_EXTENSIONS_DIR=str(L/'extensions_headers_v2'),CPATH=str(L/'dependencies/python310_headers/Include')+':'+str(L/'dependencies/python310_headers'))
ledger=[]
for mode in ['tx','rx','audit']:
 e=env.copy()
 if mode=='audit':e['CUDA_VISIBLE_DEVICES']=''
 # TX manifest is absent from the RX-specific request. Only public inputs and
 # a deny-list of forbidden source/private roots are supplied.
 rp=request
 if mode=='rx':rp=Path(r['rx_request'])
 command=[sys.executable,'-u',str(S/'prefix_evaluation.py'),mode,'--request',str(rp)]
 result=run_role(O,mode,command,e,r['cpus'],[] if mode=='audit' else [uuid],300 if mode=='audit' else r[mode+'_seconds'],r['bindings']);ledger.append(result)
 (O/'resource_ledger.json').write_text(json.dumps(ledger,indent=2))
(O/'complete.json').write_text(json.dumps(dict(passed=True,finished=time.time(),gpu_hours=sum(x['charged_gpu_device_hours'] for x in ledger)),indent=2))
