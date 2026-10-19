"""Pure invariants; no scientific or model imports."""
import collections,hashlib,json
import numpy as np
def closed_wait(v,gone):
 assert v['actual_wait'] is True and type(v['returncode']) is int and v['returncode']==0
 assert all(v[k] is False for k in ('terminated','killed','timed_out')) and v['metadata_error'] is None
 assert gone(v['process']) and gone(v['parent_process'])
def frame_key(v):return (v['source_index'],int(v['snr_db']),v['noise_seed'],v.get('strategy_family',v.get('family')))
def validate_repair_grid(rows):
 assert len(rows)==2421
 keys=set();events=set();groups=collections.Counter()
 for e in rows:
  assert e['population']=='common500' and e['strategy_family']=='WHOLE' and e['reuse'] is False
  i=e['source_index'];snr=e['snr_db'];seed=e['noise_seed']
  assert type(i)is int and 0<=i<500 and e['source_global_index']==i and snr in (4,10) and seed in (6201,6202,6203)
  assert e['public_frame_counter']==(([1,4,7,10,13,19].index(snr)*500+i)*3+seed-6201)
  assert e['K']==0 and 4<=e['m']<=(7 if snr==4 else 9)
  assert 2<=e['arithmetic_bits']<=(1245 if snr==4 else 2520)
  assert e['candidate_id']==('m7_q2_r2-3' if snr==4 else 'm9_q4_r2-3')
  k=frame_key(e);assert k not in keys and e['event_key'] not in events
  keys.add(k);events.add(e['event_key']);groups[snr]+=1
 assert dict(groups)=={4:1491,10:930}
 return dict(logical=2421,reused=0,changed=2421,new_physical=2421,packet_cap=4842)
def validate_full_grid(rows):
 keys=[frame_key(x) for x in rows]
 expected={(i,s,n,f) for i in range(500) for s in (4,10,19) for n in (6201,6202,6203) for f in ('WHOLE','PARTIAL')}
 assert len(rows)==9000 and len(set(keys))==9000 and set(keys)==expected
 assert all(x['population']=='common500' for x in rows)
def receiver_reuse(cache,key,rx,source_key):
 assert key==source_key(rx)
 if key not in cache:return None
 item=cache[key];assert item['source_input_key']==key
 assert item['validated_unique_fixed_pixel'] is True and item['source_truth_used'] is False
 return item['reconstruction']
def source_means(rows,snr,family,metric):
 groups=collections.defaultdict(dict)
 for r in rows:
  if int(r['snr_db'])!=snr or r.get('strategy_family',r.get('family'))!=family:continue
  i=r['source_index'];n=r['noise_seed'];assert n not in groups[i];groups[i][n]=float(r['metrics'][metric])
 assert set(groups)==set(range(500)) and all(set(x)=={6201,6202,6203} for x in groups.values())
 a=np.asarray([np.mean([groups[i][n] for n in (6201,6202,6203)],dtype=np.float64) for i in range(500)],dtype=np.float64)
 assert np.isfinite(a).all();return a
def interval(values,indices):
 a=np.asarray(values,dtype=np.float64);assert a.shape==(500,) and np.isfinite(a).all()
 assert indices.shape==(10000,500) and indices.dtype==np.int64 and indices.min()>=0 and indices.max()<500
 lo,hi=np.quantile(a[indices].mean(axis=1),[.025,.975]);return dict(mean=float(a.mean()),ci_low=float(lo),ci_high=float(hi))
