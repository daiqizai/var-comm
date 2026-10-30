"""Matched ceiling/MCS/grid comparison using the exact D 10 dB noise events."""
from followup_common import *
from opt_common import OFFSETS
from grid_codec import endpoints

def main():
 ss=cr(read(OLD/'frozen_final.json')['common500_sources'])[:100];assert[s['source_id']for s in ss]==read(D/'source_sets.json')['control100']
 baseline=read(OLD/'final_phy_1024_registration.json');cat=cr(baseline['catalogue']);profiles=[p for p in cat['profiles']if(p['q'],p['nominal_rate'])==(4,'2/3')]
 original={v['key']:v for v in cr(baseline['jobs'])};frameidx={v['physical_key']:v['frame']for v in read(OL/'historical_regression_inputs/physical_index.json')}
 frameidx.update({v['physical_key']:v['frame']for v in read(OL/'final_phy_1024/frames.json')})
 core=read(L/'inventory_core/logical.json');canonical={(v['source_id'],int(v['channel_repeat'])):original[v['physical_key']]for v in core if v['method']=='final_NeST'and int(v['N'])==1024 and int(v['snr_db'])==10}
 streams={v['source_id']:v for v in read(OL/'final_D_source/outputs.json')};logical=[];physical={};target=(8,126);limit=OFFSETS[8]+126;cap=profiles[0]['k']-29
 for s in ss:
  v=streams[s['source_id']];assert v['source_pixels_sha256']==s['preprocessing_id']
  with np.load(checked(v['streams']),allow_pickle=False)as z:bits={k:z[k].copy()for k in z.files}
  for grid in ['whole','quarter','eighth','token']:
   gridends=[e for e in endpoints('token'if grid=='whole'else grid)if OFFSETS[e[0]]+e[1]<=limit and(grid!='whole'or e[1]==0)]
   gridends.sort(key=lambda e:OFFSETS[e[0]]+e[1],reverse=True)
   checks=0;found=None
   for e in gridends:
    checks+=1;b=bits['m%d_K%d'%e]
    if len(b)<=cap:found=e;break
   assert found is not None
   pr=next(p for p in profiles if(p['m'],p['K'],p['family'])==(*found,'EC_VAR_PARTIAL'if found[1]else'EC_VAR_WHOLE'))
   for repeat in range(3):
    c=canonical[(s['source_id'],repeat)];assert c['snr_db']==10 and c['noise_seed']==6201+repeat
    identity={k:c[k]for k in ['N','source_id','source_index','snr_db','noise_seed','counter']};identity.update(profile_id=pr['profile_id'],payload_sha256=asha(b));key=digest(identity)
    old=frameidx.get(key)
    if old:
     rec=cr(old);assert rec['actual_RX']['header']['header_ok'],'Do not rerun an existing completed rejected event'
    job=dict(key=key,**identity,payload=b.tolist(),profile=pr,old=old,public_base=None)
    if grid=='token':assert all(job[k]==c[k]for k in ['key','payload','profile','counter','noise_seed'])and old is not None
    physical.setdefault(key,job)
    logical.append(dict(source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],N=1024,snr_db=10,method=grid,channel_repeat=repeat,noise_seed=c['noise_seed'],physical_key=key,m=pr['m'],K=pr['K'],target=list(target),q=4,rate='2/3',source_bits=len(b),capacity_bits=cap,utilization=len(b)/cap,candidate_count=len(gridends),actual_length_comparisons=checks,new_independent_terminations=0,new_prior_forwards_for_fitting=0,selection='descending actual independently-terminated length; stop at first feasible'))
 assert len(logical)==1200
 out=L/'E3_plan';out.mkdir(exist_ok=False);save(out/'physical.json',list(physical.values()));save(out/'logical.json',logical)
 save(out/'completion.json',dict(status='PREPARED',logical=1200,unique_physical=len(physical),exact_old_candidates=sum(j['old']is not None for j in physical.values()),new_physical=sum(j['old']is None for j in physical.values()),catalogue=baseline['catalogue'],qualification=baseline['qualification'],old_registered_sources=100,source_traversals=0,model_calls=0))
 print(read(out/'completion.json'))
if __name__=='__main__':main()
