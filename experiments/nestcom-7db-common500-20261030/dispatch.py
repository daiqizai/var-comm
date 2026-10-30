"""Exclusive stage registration and one launch; no retries or grid expansion."""
from g import *
def dependency(role):
 closed(role);return pin(L/(role+'_control/actual_wait.json'))
def gpu(index):
 line=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid,pci.bus_id,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
 assert int(line[2])>=45000 and int(line[3])<=80,line
 return dict(gpu=True,gpu_index=index,gpu_uuid=line[0],pci=line[1].lower().replace('00000000:','0000:'))
def launch(r,script,python=None):
 assert not(D/(r['role']+'_registration.json')).exists()and not(L/(r['role']+'_control')).exists()
 codes={}
 for name in ['cal_phy_nest','cal_rx_nest','cal_rx_ada','cal_rx_arpc_environment_v2','dev_bpg_tx','dev_p100000_tx','dev_scores']:
  for d in read(OLD/(name+'_registration.json'))['code']:codes[d['path']]=d
 for p in E.glob('*.py'):codes[str(p)]=pin(p)
 r.update(code=list(codes.values()),protocol_sha256=sha(D/'protocol.json'),output=str(L/r['role']),argv=[python or str(RT/'envs/frozen_visual_py310/bin/python'),'-B',str(E/script)])
 for c in r['code']:checked(c)
 p=D/(r['role']+'_registration.json');save(p,r);o=L/(r['role']+'_dispatch');o.mkdir()
 with(o/'stdout.log').open('xb')as so,(o/'stderr.log').open('xb')as se:q=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control.py'),str(p),sha(p)],stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True)
 save(o/'launch.json',dict(process=ident(q.pid),registration=pin(p)));return read(o/'launch.json')
def dispatch(kind,method,phase=None):
 r=dict(gpu=False,cpu_affinity=list(range(6)),dependencies=[],timing_config=pin(OLD/'timing_config.json'),policy=pin(OLD/'frozen_policies.json'),max_seconds=1800)
 cfg=read(OLD/'timing_config.json');python=None
 if kind=='phy':
  role='phy_'+method.lower();r.update(method=method,stage='formal',sources=pin(L/'sources.json'),frozen_mcs=[cr(r['policy'])['methods'][method][k]for k in ['q','rate']],phy_qualification=read(OLD/'cal_phy_nest_registration.json')['phy_qualification']);script='phy.py';python=str(RT/'envs/frozen_ldpc_py311/bin/python')
 elif kind=='rx':
  role='rx_'+method.lower();dep='phy_'+method.lower();d=closed(dep);jobs=cr(d['inputs']);p=L/(role+'_inputs.json');save(p,jobs)
  old=read(OLD/('cal_rx_'+method.lower()+('_environment_v2'if method=='ARPC'else'')+'_registration.json'))
  for k in ['codec','model_routes','static','cpu_affinity','weights','upstream_code','scope','revision']:
   if k in old:r[k]=old[k]
  index={'NEST':1,'ADA':2,'ARPC':3}[method];r.update(gpu(index));r.update(inputs=pin(p),dependencies=[dependency(dep)],caps=dict(model_load=1,receive_render=len(jobs),prior_scale=len(jobs)*20,decoder_forward=len(jobs)),max_seconds=7200 if method=='ADA'else 3600)
  script='arpc_entry_v2.py'if method=='ARPC'else'rx_native.py'
  if method=='ARPC':python='/usr/bin/python3'
 elif kind=='aux':
  role=method.lower()+'_'+phase;r.update(method=method,phase=phase,max_seconds=7200 if method=='BPG'and phase=='tx'else 3600);script='aux.py'
  if phase=='tx':r['inputs']=pin(L/'sources.json')
  else:
   dep=method.lower()+('_phy'if method=='BPG'and phase=='rx'else'_tx');r['inputs']=closed(dep)['outputs'];r['dependencies']=[dependency(dep)]
  if method!='BPG':r.update(gpu(5 if method=='P100000'else 7))
  else:python=str(RT/'envs/frozen_ldpc_py311/bin/python')
 else:raise ValueError(kind)
 r['role']=role
 return launch(r,script,python)
if __name__=='__main__':print(json.dumps(dispatch(*sys.argv[1:])))
