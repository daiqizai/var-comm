"""One frozen five-metric route, exact cache identities, no new metric variants."""
from study_common import *

def start(prefix,mode,plan_path):
 p=Path(plan_path);plan=read(p/'plan.json');n=len(plan['pairs']);ns=len(plan['sources']);assert n
 old=read(F/('E3_'+mode+'_registration.json'))
 if mode=='dists':
  r={k:old[k]for k in ['assets','loader','torch','code','caches']}
  r.update(sources=pin(p/'sources.json'),pairs=pin(p/'pairs.json'),cpu_affinity=[32,34],caps=dict(load_dists=1,pair=n),priority_pairs=[]);kind='dists';py=VPY
 else:
  r={k:old[k]for k in ['code','metric_registration','mode','caches']}
  r.update(plan=pin(p/'plan.json'),cpu_affinity=[32,34],caps=dict(embedding=n+ns,pair=n,**(dict(load_clip=1)if mode=='clip'else dict(load_lpips=1,load_dreamsim=1))));kind='metrics';py=APY if mode=='clip'else VPY
 r['caches']=r['caches']+[pin(FL/('E3_'+mode)/'completion.json'),pin(L/('low_'+mode)/'completion.json')]
 if prefix!='compiled32'and(L/('compiled32_'+mode)/'completion.json').exists():
  closed('compiled32_'+mode);r['caches'].append(pin(L/('compiled32_'+mode)/'completion.json'))
 r['code'].append(pin(E/'five_metrics.py'))
 launch(prefix+'_'+mode,r,kind,py,1800 if prefix=='compiled32'else 7200,gpu=1)

def gather(prefix):
 scores={}
 for mode in ['three','clip','dists']:
  c=closed(prefix+'_'+mode)
  for key,v in cr(c['scores']).items():scores.setdefault(key,{}).update(v)
 for key,row in scores.items():
  for m in METRICS:assert np.isfinite(float(row[m])),(key,m)
 return scores
