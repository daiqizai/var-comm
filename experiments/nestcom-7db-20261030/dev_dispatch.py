"""Exclusive first starts only, after the calibration policy is frozen."""
from g import *

def dependency(role):
 p=L/(role+'_control/actual_wait.json');a=read(p);assert a['returncode']==0 and gone(a['process']) and gone(read(p.parent/'started.json')['process']);return pin(p)

def dispatch(kind,method=None,phase=None):
 policy=pin(D/'frozen_policies.json');assert cr(policy)['status']=='FROZEN_BEFORE_DEVELOPMENT';cfg=read(D/'timing_config.json');codes={}
 for p in D.glob('*registration.json'):
  for c in read(p)['code']:codes[c['path']]=c
 for p in E.glob('*.py'):codes[str(p)]=pin(p)
 for p in [R/'experiments/nestcom-n512-20261028/nest_source_worker.py',R/'experiments/five-method-timing-20261020/bpg_threads_recovery_v1/resume.py']:codes[str(p)]=pin(p)
 r=dict(gpu=True,gpu_index=1,gpu_uuid=cfg['resource']['gpu_uuid'],pci=cfg['resource']['gpu_pci'],timing_config=pin(D/'timing_config.json'),policy=policy,protocol_sha256=sha(D/'protocol.json'),code=list(codes.values()),cpu_affinity=list(range(6)),max_seconds=300,dependencies=[dependency('cal_dino')],model_routes=cfg['model_routes'],static=cfg['static']);python=str(RT/'envs/frozen_visual_py310/bin/python')
 if kind=='source':
  role='dev_nest_source';inputs=[dict(s,population='development')for s in read(D/'development_sources.json')];p=D/'dev_nest_source_inputs.json';save(p,inputs)
  r.update(inputs=pin(p),cpu_affinity=[0,1],caps=dict(model_load=1,source_probability_pass=2,prior_scale=40,decoder_forward=0));script='source_entry.py'
 elif kind=='phy':
  role='dev_phy_'+method.lower();r.update(gpu=False,method=method,stage='development',sources=pin(D/'development_sources_ready.json'),frozen_mcs=[cr(policy)['methods'][method][x]for x in ['q','rate']],phy_qualification=read(D/'cal_phy_nest_registration.json')['phy_qualification']);script='phy.py';python=str(RT/'envs/frozen_ldpc_py311/bin/python')
 elif kind=='rx':
  role='dev_rx_'+method.lower();names=['nest','whole']if method=='NEST'else[method.lower()];jobs={}
  for name in names:
   r['dependencies'].append(dependency('dev_phy_'+name))
   for j in cr(read(L/('dev_phy_'+name)/'completion.json')['inputs']):jobs.setdefault(j['key'],j)
  p=D/(role+'_inputs.json');save(p,list(jobs.values()));donor=read(D/('cal_rx_'+method.lower()+('_environment_v2'if method=='ARPC'else'')+'_registration.json'))
  r.update({k:donor[k]for k in ['codec','gpu_index','gpu_uuid','pci','cpu_affinity']});r.update(inputs=pin(p),caps=dict(model_load=1,receive_render=len(jobs),prior_scale=len(jobs)*20,decoder_forward=len(jobs)));script='arpc_entry_v2.py'if method=='ARPC'else'rx_native.py'
  if method=='ADA':r.update({k:donor[k]for k in ['weights','upstream_code','scope','revision']})
  if method=='ARPC':python='/usr/bin/python3'
 elif kind=='aux':
  role='dev_'+method.lower()+'_'+phase;r.update(method=method,phase=phase,snr_db=7,gpu=method!='BPG');script='dev_aux.py'
  if phase=='tx':r['inputs']=pin(D/'development_sources.json')
  else:
   dep='dev_'+method.lower()+('_phy'if method=='BPG'and phase=='rx'else'_tx');r['dependencies'].append(dependency(dep));r['inputs']=read(L/dep/'completion.json')['outputs']
  if method=='BPG':python=str(RT/'envs/frozen_ldpc_py311/bin/python');r['max_seconds']=1800
 elif kind=='scores':
  role='dev_scores';script='dev_scores.py';donor=read(R/'results/nestcom_fig3_4db_20261029_v1/scores_registration.json')
  r.update({k:donor[k]for k in ['development_scope','metric_donor']});r['jobs']=pin(D/'development_score_jobs.json');r['actual_waits']=[dependency(n)for n in ['dev_rx_nest','dev_rx_ada','dev_rx_arpc','dev_p100000_rx','dev_swin200000_rx','dev_bpg_rx']]
 else:raise ValueError(kind)
 r.update(role=role,output=str(L/role),argv=[python,'-B',str(E/script)])
 for c in r['code']:checked(c)
 p=D/(role+'_registration.json');save(p,r);o=L/(role+'_dispatch');o.mkdir()
 with(o/'stdout.log').open('xb')as so,(o/'stderr.log').open('xb')as se:q=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control.py'),str(p),sha(p)],stdout=so,stderr=se,stdin=subprocess.DEVNULL,start_new_session=True)
 save(o/'launch.json',dict(process=ident(q.pid),registration=pin(p)));print(read(o/'launch.json'))
if __name__=='__main__':dispatch(*sys.argv[1:])
