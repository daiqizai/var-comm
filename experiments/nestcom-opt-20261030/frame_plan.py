"""Frozen-policy frames and exact historical event reuse, no model calls."""
from opt_common import *
import re

CELLS=((1024,7),(1024,10),(1024,4),(1024,19),(512,4),(512,10),(512,19))
QUARTER=tuple((m,k)for m in range(4,10)for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3))))+((10,0),)
def nt(e):return OFFSETS[e[0]]+e[1]
def policy(N,snr,method,resolved):
 if N==512:return dict(cr(resolved['policy_N512'])['policies'][str(snr)][method])
 if snr==7:
  p=cr(resolved['policy_7db'])['methods'][method]
  return dict(method=method,target=[10,0],q=p['q'],rate=p['rate'])
 winner=cr(resolved['policy_N1024'])['policies'][str(snr)]['PARTIAL'if method=='NEST'else'WHOLE']['winner']
 match=re.fullmatch(r'm(\d+)(?:_K(\d+)|_(?:partial|whole))?_q(\d+)_r(\d+)-(\d+)',winner);assert match,winner
 m,k,q,a,b=match.groups();return dict(method=method,target=[int(m),int(k or 0)],q=int(q),rate=a+'/'+b)

def main():
 out=L/'fit_frame_plan';out.mkdir(exist_ok=False)
 resolved=read(D/'protocol_resolved_p0.json');sources=cr(cr(resolved['sources'])['FIT'])
 assert len(sources)==100
 cats={n:cr(resolved['catalogue_N'+str(n)])for n in(512,1024)}
 old512=read(B/'nestcom_n512_20261028_v1/full_rx_plan/frames.json')
 old1024=read(B/'baseline_strength_20261011/t1b_m10_calibration_20261016_v1/preparation/old_packet_index.json')
 old_proto=cr(resolved['policy_N512'])['protocol_sha256'];physical={};logical=[];tx_candidates=[]
 for s in sources:
  with np.load(checked(s['nest_streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files}
  for N,snr in CELLS:
   for method in ('NEST','WHOLE'):
    c=policy(N,snr,method,resolved);profiles=[p for p in cats[N]['profiles']if(p['q'],p['nominal_rate'])==(c['q'],c['rate'])]
    cap=profiles[0]['k']-29;ends=sorted((e for e in QUARTER if nt(e)<=nt(c['target'])and(method=='NEST'or e[1]==0)),key=nt,reverse=True)
    feasible=[e for e in ends if len(streams['m%d_K%d'%e])<=min(cap,8191)]
    if not feasible:
     assert N==512,'N1024 no-fit is an execution error, not an invented NULL'
     p=next(p for p in profiles if p['family']=='EXPLICIT_NULL_FIXEDFRAME');bits=np.zeros(2,np.uint8)
    else:
     e=feasible[0];p=next(p for p in profiles if(p['m'],p['K'],p['family'])==(*e,'EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE'));bits=streams['m%d_K%d'%e]
    seed=4101;counter=85000000+s['source_index']*3 if N==512 else (17000000+4096*s['source_index']+p['profile_id']if snr==7 else ([1,4,7,10,13,19].index(snr)*1000+s['source_index'])*3)
    identity=dict(N=N,source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,noise_seed=seed,counter=counter,profile_id=p['profile_id'],payload_sha256=asha(bits))
    key=digest(identity);old=None
    if N==512:
     oi={k:v for k,v in identity.items()if k!='N'};oi.update(session='NESTCOM_N512_20261028_V1',protocol_sha256=old_proto)
     old=old512.get(digest(oi));assert old,'Frozen N512 calibration identity must exist'
    elif snr!=7:
     lookup=json.dumps([s['source_index'],snr,seed,counter,p['profile_id'],asha(bits)],separators=(',',':'));old=old1024.get(lookup)
    physical.setdefault(key,dict(key=key,**identity,payload=bits.tolist(),profile=p,old=old,public_base=counter-p['profile_id']if snr==7 else None))
    logical.append(dict(physical_key=key,method=method,N=N,snr_db=snr,source_id=s['source_id'],source_index=s['source_index'],source_pixels_sha256=s['preprocessing_id'],source_float_sha256=s['reference_sha256'],target=c['target'],q=c['q'],rate=c['rate'],endpoint=[p['m'],p['K']],source_bits=len(bits),capacity_bits=cap,utilization=len(bits)/cap,noise_seed=seed))
    if method=='NEST'and N==1024 and snr in(7,10):
     tx_candidates.append(dict(source_id=s['source_id'],source_index=s['source_index'],N=N,snr_db=snr,reference=s['archive'],source_float_sha256=s['reference_sha256'],candidates=[dict(endpoint=list(e),payload=streams['m%d_K%d'%e].tolist(),family='EC_VAR_PARTIAL'if e[1]else'EC_VAR_WHOLE')for e in feasible[:3]]))
 assert len(logical)==1400
 save(out/'logical.json',logical);save(out/'tx_candidates.json',tx_candidates)
 for N in(512,1024):save(out/f'physical_{N}.json',[v for v in physical.values()if v['N']==N])
 save(out/'completion.json',dict(status='PREPARED',source_count=100,logical_count=len(logical),unique_physical=len(physical),old_exact_lookup=sum(v['old']is not None for v in physical.values()),new_count=sum(v['old']is None for v in physical.values()),candidate_strategy_frozen=True,source_numerics='Pinned qualified PRO FP32 independently finished streams; historical events reused only after full waveform and noise equality. No H800-output identity assumed.',sources=pin(L/'inputs/fit100.json'),logical=pin(out/'logical.json')))
 print(json.dumps(read(out/'completion.json')))
if __name__=='__main__':main()
