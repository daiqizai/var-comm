"""Prepare projection-only jobs and exact original-receiver cache reuse."""
from opt_common import *

def baseline_cache():
 w=read(L/'fit_A_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 return {x['key']:x for x in read(L/'fit_A/outputs.json')if x['variant']=='original'}

def from_frames(plan_role,frame_roles,output_role):
 out=L/output_role;out.mkdir(exist_ok=False);jobs={};frames={}
 for role in frame_roles:
  w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
  assert read(L/role/'completion.json')['ledger']['unresolved']==0
  for row in read(L/role/'rx_inputs.json'):jobs.setdefault(row['key'],row)
  for row in read(L/role/'frames.json'):frames[row['physical_key']]=row
 logical=read(L/plan_role/'logical.json')
 for row in logical:row['received_key']=frames[row['physical_key']]['received_key']
 cache=baseline_cache();reuse={k:cache[k]for k in jobs if k in cache}
 save(out/'inputs.json',list(jobs.values()));save(out/'reuse.json',reuse);save(out/'logical.json',logical)
 save(out/'completion.json',dict(status='PREPARED',logical=len(logical),unique_inputs=len(jobs),reused=len(reuse),new_decode=sum(j['projection']['status']=='SOURCE_INPUT'and k not in reuse for k,j in jobs.items()),source_ground_truth_in_jobs=False))
 print(output_role,read(out/'completion.json'))

def tx_simulation():
 out=L/'fit_E_tx_plan';out.mkdir(exist_ok=False);data=read(L/'fit_frame_plan/tx_candidates.json');jobs={};choices=[]
 for row in data:
  c=[]
  for i,v in enumerate(row['candidates']):
   m,K=v['endpoint'];p=dict(status='SOURCE_INPUT',codec='NEST',m=m,K=K,family=v['family'],bits=v['payload']);key=digest(p)
   jobs.setdefault(key,dict(key=key,projection=p));c.append(dict(rank=i,received_key=key,endpoint=v['endpoint']))
  choices.append({**{k:v for k,v in row.items()if k!='candidates'},'candidates':c})
 cache=baseline_cache();reuse={k:cache[k]for k in jobs if k in cache and cache[k]['status']=='ACCEPTED_SOURCE'}
 save(out/'inputs.json',list(jobs.values()));save(out/'reuse.json',reuse);save(out/'choices.json',choices)
 save(out/'completion.json',dict(status='PREPARED',source_workpoints=len(choices),unique_inputs=len(jobs),reused=len(reuse),new_decode=len(jobs)-len(reuse),selection_has_no_actual_RX_status_or_noise=True,reference_access='TX scoring only; not present in receiver jobs'))
 print(read(out/'completion.json'))

if __name__=='__main__':
 from_frames('fit_D_frame_plan',['fit_D_phy_1024'],'fit_D_rx_plan')
 tx_simulation()
