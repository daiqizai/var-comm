"""Frozen confirmation/final frame plans, with explicit public repeat counters."""
from opt_common import *
from frame_plan import policy,nt,CELLS,QUARTER
from grid_codec import endpoints

def main(request):
 r=read(request);sources=cr(r['sources']);dataset=r['dataset'];assert dataset in ('confirm100','common500')
 a=read(D/'protocol_resolved_p0.json');grid=r['grid'];assert grid in ('quarter','token','eighth')
 streams={}
 if grid!='quarter':
  root=Path(r['grid_source_output']);w=read(root.with_name(root.name+'_control')/'actual_wait.json')
  assert w['returncode']==0 and ident(w['child']['pid'])is None
  assert read(root/'completion.json')['ledger']['unresolved']==0
  streams={s['source_id']:s for s in read(root/'outputs.json')}
 cats={N:cr(r['catalogues'][str(N)])for N in (512,1024)}
 old={}
 for v in r.get('old_frame_indices',[]):
  for f in cr(v):old[f['physical_key']]=f['frame']
 legacy512=read(B/'nestcom_n512_20261028_v1'/('full_rx_plan'if dataset=='confirm100'else'formal_rx_plan')/'frames.json') if grid=='quarter' else {}
 old_protocol=cr(a['policy_N512'])['protocol_sha256'];physical={};logical=[]
 for s in sources:
  sp=s['nest_streams']if grid=='quarter'else streams[s['source_id']]['streams']
  if grid!='quarter':assert streams[s['source_id']]['source_pixels_sha256']==s['preprocessing_id']
  with np.load(checked(sp),allow_pickle=False)as z:bits={k:z[k].copy()for k in z.files}
  for N,snr in r.get('cells',CELLS):
   assert (N,snr) in CELLS
   for method in r['methods']:
    p=policy(N,snr,method,a);profiles=[v for v in cats[N]['profiles']if(v['q'],v['nominal_rate'])==(p['q'],p['rate'])];cap=profiles[0]['k']-29
    es=QUARTER if grid=='quarter'else endpoints(grid)
    ends=sorted({e for e in es if nt(e)<=nt(p['target'])and(method=='NEST'or e[1]==0)}|{tuple(p['target'])},key=nt,reverse=True)
    feasible=[e for e in ends if len(bits['m%d_K%d'%e])<=min(cap,8191)]
    if feasible:
     e=feasible[0];b=bits['m%d_K%d'%e];pr=next(v for v in profiles if(v['m'],v['K'],v['family'])==(*e,'EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE'))
    else:
     assert N==512;pr=next(v for v in profiles if v['family']=='EXPLICIT_NULL_FIXEDFRAME');b=np.zeros(2,np.uint8)
    for repeat in range(3):
     seed=(4101 if dataset=='confirm100'else 6201)+repeat;idx=s['source_index']
     if N==512:counter=(85000000 if dataset=='confirm100'else 90000000)+idx*3+repeat;public=None
     elif snr==7:
      # Confirm extends the public calibration base by a repeat block; final
      # preserves the original common500 recipe, never the calibration base.
      public=17000000+4096*idx+4096000*repeat if dataset=='confirm100'else 26000000+16384*idx+4096*repeat
      counter=public+pr['profile_id']
     else:counter=([1,4,7,10,13,19].index(snr)*(1000 if dataset=='confirm100'else 500)+idx)*3+repeat;public=None
     identity=dict(N=N,source_id=s['source_id'],source_index=idx,snr_db=snr,noise_seed=seed,counter=counter,profile_id=pr['profile_id'],payload_sha256=asha(b));key=digest(identity)
     oldpin=old.get(key)
     if oldpin is None and N==512 and grid=='quarter':
      historical={k:v for k,v in identity.items()if k!='N'};historical.update(session='NESTCOM_N512_20261028_V1',protocol_sha256=old_protocol)
      oldpin=legacy512.get(digest(historical));assert oldpin,'An already successful frozen N512 event must be reused, not repeated'
     physical.setdefault(key,dict(key=key,**identity,payload=b.tolist(),profile=pr,old=oldpin,public_base=public))
     logical.append(dict(physical_key=key,method=method,variant=r['variant'],N=N,snr_db=snr,source_id=s['source_id'],source_index=idx,source_pixels_sha256=s['preprocessing_id'],source_float_sha256=s['reference_sha256'],target=p['target'],q=p['q'],rate=p['rate'],endpoint=[pr['m'],pr['K']],source_bits=len(b),capacity_bits=cap,utilization=len(b)/cap,noise_seed=seed,channel_repeat=repeat,grid=grid,feasible_count=len(feasible),length_checks=len(ends)))
 out=L/r['output'];out.mkdir(exist_ok=False)
 for N in (512,1024):save(out/f'physical_{N}.json',[v for v in physical.values()if v['N']==N])
 save(out/'logical.json',logical);save(out/'completion.json',dict(status='PREPARED',registration=pin(request),sources=r['sources'],logical=len(logical),unique_physical=len(physical),old_candidates=sum(v['old']is not None for v in physical.values()),no_recalibration=True,channel_recipe=r['channel_recipe']))
 print(read(out/'completion.json'))

if __name__=='__main__':main(*sys.argv[1:])
