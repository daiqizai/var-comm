"""Pure test-scope validation and source-paired statistics; no model imports."""
import collections,json,hashlib
import numpy as np
POPS={'common500':(500,(6201,6202,6203)),'new100':(100,(9301,9302,9303))}
def validate_grid(rows):
 seen=set();events={};changed=set();reuse=0;ids={}
 for r in rows:
  pop=r['population'];n,seeds=POPS[pop];i=r['source_index'];seed=r['noise_seed'];fam=r['strategy_family']
  assert r['snr_db']==19 and fam in ('WHOLE','PARTIAL') and type(i)is int and 0<=i<n and seed in seeds
  ident=(pop,i,seed,fam);assert ident not in seen;seen.add(ident)
  assert r['source_global_index']==i+(500 if pop=='new100' else 0)
  assert r['public_frame_counter']==((5 if pop=='common500' else 2)*n+i)*3+seeds.index(seed)
  assert r['candidate_id']==f'm10_{fam.lower()}_q6_r5-6' and 2<=r['arithmetic_bits']<=4751
  assert r['m']in range(4,11) and (r['m']!=10 or r['K']==0)
  assert fam!='WHOLE' or r['K']==0
  k=(pop,i);assert k not in ids or ids[k]==r['source_id'];ids[k]=r['source_id']
  ek=r['event_key'];signature=(pop,i,seed,r['public_frame_counter'],r['profile_id'],r['profile_key'],r['payload_sha256'],r['m'],r['K'],r['arithmetic_bits'],r['reuse'])
  assert ek not in events or events[ek]==signature;events[ek]=signature
  if r['reuse']:reuse+=1
  else:changed.add(ek)
 assert len(seen)==3600 and len(ids)==len(set(ids.values()))==600
 return dict(logical=3600,reused=reuse,changed=3600-reuse,new_physical=len(changed),packet_cap=2*len(changed))
def closed_wait(v,gone):
 assert v['actual_wait'] is True and type(v['returncode']) is int and v['returncode']==0
 assert all(v[k] is False for k in ('terminated','killed','timed_out')) and v['metadata_error'] is None
 assert gone(v['process']) and gone(v['parent_process'])
def source_means(rows,pop,family,metric,field='metrics'):
 n,seeds=POPS[pop];groups=collections.defaultdict(dict)
 for r in rows:
  if r['population']!=pop or r['strategy_family']!=family:continue
  i=r['source_index'];s=r['noise_seed'];assert s not in groups[i];groups[i][s]=float(r[field][metric])
 assert set(groups)==set(range(n)) and all(set(x)==set(seeds) for x in groups.values())
 a=np.asarray([np.mean([groups[i][s] for s in seeds],dtype=np.float64) for i in range(n)],dtype=np.float64);assert np.isfinite(a).all();return a
def interval(values,indices):
 a=np.asarray(values,dtype=np.float64);assert a.ndim==1 and np.isfinite(a).all()
 assert indices.ndim==2 and indices.shape[1]==len(a) and np.issubdtype(indices.dtype,np.integer) and indices.min()>=0 and indices.max()<len(a)
 samples=a[indices].mean(axis=1);lo,hi=np.quantile(samples,[.025,.975]);return dict(mean=float(a.mean()),ci_low=float(lo),ci_high=float(hi))
