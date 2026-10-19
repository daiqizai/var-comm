"""Copy only closed actual-RX byte objects to independent model-worker views."""
from shared import *
def main(phase,expected):
 r=registration(expected);assert read(O/phase/'phy_completion.json')['status'] in ['PASS_QUALIFICATION_PHY','FORMAL_PHY_COMPLETE']
 w=read(L/'controls'/(phase+'_rx')/'actual_wait.json');assert w['returncode']==0 and gone(w['process'])
 jobs=read(L/phase/'rx/source_jobs.json')
 for codec,name in [(0,'nest'),(1,'arpc')]:
  root=L/phase/name;root.mkdir();inputs=root/'wire_inputs';inputs.mkdir();selected=[]
  for j in jobs:
   if j['codec']!=codec:continue
   data=verify(j['input']).read_bytes();path=inputs/(j['key']+('.arc' if codec else '.bin'));path.open('xb').write(data)
   selected.append(dict(key=j['key'],path=str(path),input=pin(path),paid_profile=7 if codec else 10,bit_length=j['bit_length'],generation_seed=0))
  if phase=='qualification':assert len(selected)==6
  put(inputs/'jobs.json',selected)
  if codec:
   cfg=read(OLD_L/'a3_rx_v1/public.json');cfg.update(run=str(root),output=str(O/phase),schema='N2048_ACTUAL_ARPC_E7_RX',new_job_count=len(selected))
   # Existing same-host native engine, mapping and model-numerical settings.
   assert cfg['cuda_admission']['visible_index']==4 and cfg['resource']['gpu']['index']==4
  else:
   gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=pci.bus_id,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
   assert int(gpu[1])>=15360
   cfg=dict(run=str(root),model_routes=r['model_routes'],static=r['static'],jobs=pin(inputs/'jobs.json'),gpu_index=0,gpu_pci=gpu[0].strip(),max_seconds=r['stage_seconds'][phase+'_nest'])
  put(root/'public.json',cfg)
 put(O/phase/'native_preparation.json',dict(status='PREPARED_ACTUAL_RX_ONLY',jobs=pin(L/phase/'rx/source_jobs.json'),public=[pin(L/phase/n/'public.json') for n in ['nest','arpc']],TX_truth_in_public=False))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
