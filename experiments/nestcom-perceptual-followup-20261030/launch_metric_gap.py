"""Unique first launch of one named missing-metric stage; no implicit retries."""
from followup_common import *
from launch import launch
def main(role,mode,plan_path):
 assert role in ['E1_external_clip','E4_three','E4_clip','E4_dists']
 plan=read(plan_path);pairs=plan['pairs'];n=len(pairs);assert n>0
 if mode=='dists':
  old=read(D/'E1_external_dists_registration.json')
  r={k:old[k]for k in ['sources','assets','loader','torch']}
  r.update(pairs=pin(L/'metric_gaps/E4_new_pairs.json'),code=old['code']+[pin(E/'launch_metric_gap.py')],cpu_affinity=[22,23],caps=dict(load_dists=1,pair=n),priority_pairs=[],caches=[pin(L/x/'completion.json')for x in ['E1_core_dists','E1_external_dists']])
  worker=E/'dists_worker.py';python=RT/'envs/frozen_visual_py310/bin/python'
 else:
  old=read(OLD/('final_new_'+mode+'_registration.json'));caches=[pin(OL/x/'completion.json')for x in ['fit_A_'+mode,'confirm_D_'+mode,'final_new_'+mode]]
  if mode=='clip':caches.append(pin(OL/'final_historical_missing_clip/completion.json'))
  r=dict(code=old['code']+[pin(E/v)for v in ['followup_common.py','control_once.py','launch.py','metric_adapter.py','launch_metric_gap.py']],plan=pin(plan_path),metric_registration=old['metric_registration'],mode=mode,cpu_affinity=[22,23],caches=caches,caps=dict(embedding=n+len(plan['sources']),pair=n,**(dict(load_clip=1)if mode=='clip'else dict(load_lpips=1,load_dreamsim=1))))
  worker=E/'metric_adapter.py';python=Path('/usr/bin/python')if mode=='clip'else RT/'envs/frozen_visual_py310/bin/python'
 launch(role,r,worker,python,7200,gpu=True,required_uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e')
if __name__=='__main__':main(*sys.argv[1:])
