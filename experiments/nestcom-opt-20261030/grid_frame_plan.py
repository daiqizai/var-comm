"""Longest feasible grid selection under the existing frozen policy ceiling."""
from opt_common import *
from frame_plan import policy,nt
from grid_codec import endpoints

def main(cells=((1024,7),(1024,10)),role='fit_D_frame_plan',grids=('quarter','eighth','token')):
 out=L/role;out.mkdir(exist_ok=False);a=read(D/'protocol_resolved_p0.json');sources=read(L/'inputs/fit100.json')
 w=read(L/'fit_D_source_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 streams={s['source_id']:s for s in read(L/'fit_D_source/outputs.json')}
 cats={N:read(D/'grid_catalogues'/f'N{N}.json')for N in {c[0]for c in cells}}
 old={x['physical_key']:x['frame']for N in (512,1024)for x in read(L/f'fit_phy_{N}/frames.json')}
 physical={};logical=[]
 for s in sources:
  z=streams[s['source_id']];assert z['source_pixels_sha256']==s['preprocessing_id']
  with np.load(checked(z['streams']),allow_pickle=False)as f:bits={k:f[k].copy()for k in f.files}
  for N,snr in cells:
   p=policy(N,snr,'NEST',a);profiles=[x for x in cats[N]['profiles']if(x['q'],x['nominal_rate'])==(p['q'],p['rate'])];cap=profiles[0]['k']-29
   for grid in grids:
    ends=sorted({e for e in endpoints(grid)if nt(e)<=nt(p['target'])}|{tuple(p['target'])},key=nt,reverse=True)
    feasible=[e for e in ends if len(bits['m%d_K%d'%e])<=min(cap,8191)]
    if not feasible:
     assert N==512
     pr=next(x for x in profiles if x['family']=='EXPLICIT_NULL_FIXEDFRAME');b=np.zeros(2,np.uint8)
    else:
     e=feasible[0];b=bits['m%d_K%d'%e];pr=next(x for x in profiles if(x['m'],x['K'],x['family'])==(*e,'EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE'))
    counter=85000000+s['source_index']*3 if N==512 else(17000000+4096*s['source_index']+pr['profile_id']if snr==7 else([1,4,7,10,13,19].index(snr)*1000+s['source_index'])*3)
    identity=dict(N=N,source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,noise_seed=4101,counter=counter,profile_id=pr['profile_id'],payload_sha256=asha(b));key=digest(identity)
    physical.setdefault(key,dict(key=key,**identity,payload=b.tolist(),profile=pr,old=old.get(key),public_base=counter-pr['profile_id']if snr==7 else None))
    logical.append(dict(physical_key=key,method='NEST',variant={'quarter':'original','eighth':'D_eighth','token':'D_token'}[grid],N=N,snr_db=snr,source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],source_float_sha256=s['reference_sha256'],target=p['target'],q=p['q'],rate=p['rate'],endpoint=[pr['m'],pr['K']],source_bits=len(b),capacity_bits=cap,utilization=len(b)/cap,noise_seed=4101,grid=grid,feasible_count=len(feasible),length_checks=len(ends)))
 for N in cats:save(out/f'physical_{N}.json',[v for v in physical.values()if v['N']==N])
 save(out/'logical.json',logical);save(out/'completion.json',dict(status='PREPARED',logical=len(logical),unique_physical=len(physical),old_exact_candidates=sum(bool(v['old'])for v in physical.values()),same_frozen_ceiling=True,monotonic_length_assumed=False,catalogue=[pin(D/'grid_catalogues'/f'N{N}.json')for N in cats]))
 print(read(out/'completion.json'))

if __name__=='__main__':main()
