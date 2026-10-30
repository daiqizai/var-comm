"""First-only metric launch for a finite prepared gap, on the registered device."""
from followup_common import *
from launch import launch

def main(experiment,mode):
 assert experiment in ['E2','E3','E6']and mode in ['three','clip','dists']
 p=L/(experiment+'_metric_plan');plan=read(p/'plan.json');n=len(plan['pairs']);assert n>0 and read(p/'completion.json')['status']=='PREPARED'
 role=experiment+'_'+mode
 own=[pin(E/v)for v in ['followup_common.py','control_once.py','launch.py','launch_followup_metrics.py']]
 if mode=='dists':
  old=read(D/'E1_external_dists_registration.json');r={k:old[k]for k in ['sources','assets','loader','torch']}
  names=['E1_core_dists','E1_external_dists','E4_dists']
  names += [x+'_dists'for x in ['E2','E3','E6']if x!=experiment and(L/(x+'_dists')/'completion.json').exists()]
  for x in names:closed(x)
  r.update(pairs=pin(p/'pairs.json'),code=old['code']+own,cpu_affinity=[22,23],caps=dict(load_dists=1,pair=n),priority_pairs=[],caches=[pin(L/x/'completion.json')for x in names]);worker=E/'dists_worker.py';python=RT/'envs/frozen_visual_py310/bin/python'
 else:
  old=read(OLD/('final_new_'+mode+'_registration.json'));caches=[pin(OL/(x+'_'+mode)/'completion.json')for x in ['fit_A','confirm_D','final_new']]
  if mode=='clip':caches.append(pin(OL/'final_historical_missing_clip/completion.json'))
  names=['E4_'+mode]+[x+'_'+mode for x in ['E2','E3','E6']if x!=experiment and(L/(x+'_'+mode)/'completion.json').exists()]
  for x in names:closed(x);caches.append(pin(L/x/'completion.json'))
  r=dict(code=old['code']+own+[pin(E/'metric_adapter.py')],plan=pin(p/'plan.json'),metric_registration=old['metric_registration'],mode=mode,cpu_affinity=[22,23],caches=caches,caps=dict(embedding=n+len(plan['sources']),pair=n,**(dict(load_clip=1)if mode=='clip'else dict(load_lpips=1,load_dreamsim=1))))
  worker=E/'metric_adapter.py';python=Path('/usr/bin/python')if mode=='clip'else RT/'envs/frozen_visual_py310/bin/python'
 launch(role,r,worker,python,14400,gpu=True,required_uuid='GPU-f5047136-08e5-9416-3c86-87786d0b355e')
if __name__=='__main__':main(*sys.argv[1:])
