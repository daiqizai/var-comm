"""Fixed per-budget additive directory, unchanged paid 12-bit header."""
from opt_common import *
from grid_codec import endpoints
from frame_plan import CELLS,policy,nt
def extend(N,base,resolved):
 profiles=[dict(p)for p in base['profiles']];old={p['profile_id']:dict(p)for p in profiles}
 # BPG uses a separate source protocol; conservatively leave its paid range
 # unassigned in the new N1024 NeST extension. Existing N512 IDs remain intact.
 reserved=set(range(3000,3012))if N==1024 else set()
 occupied=set(old)|reserved;nextid=max(old)+1;needed={}
 for n,snr in CELLS:
  if n!=N:continue
  c=policy(N,snr,'NEST',resolved);pair=(c['q'],c['rate']);needed[pair]=max(needed.get(pair,0),nt(c['target']))
 for (q,rate),ceiling in sorted(needed.items()):
  basep=next(p for p in profiles if p['family']=='EC_VAR_PARTIAL'and(p['q'],p['nominal_rate'])==(q,rate))
  for m,K in endpoints('token'):
   if nt((m,K))>ceiling:continue
   if any((p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==('EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',m,K,q,rate)for p in profiles):continue
   while nextid in occupied:nextid+=1
   if nextid>=4096:raise ValueError('D2 catalogue exceeds paid 12-bit space; keep D1 separately')
   p={k:v for k,v in basep.items()if k!='profile_key'};p.update(profile_id=nextid,family='EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',m=m,K=K,token_count=nt((m,K)))
   p['profile_key']=digest(p);profiles.append(p);occupied.add(nextid);nextid+=1
 assert all(next(p for p in profiles if p['profile_id']==i)==v for i,v in old.items())
 result={k:v for k,v in base.items()if k!='catalogue_sha256'};result.update(protocol='NESTCOM_OPT_FIXED_TOKEN_GRID_N'+str(N),profiles=profiles,extension=dict(old_profiles=len(old),new_profiles=len(profiles)-len(old),reserved_ids=sorted(reserved),old_meanings_preserved=True,shared_N=True,per_source_directory=False,needed_MCS_ceiling={str(k):v for k,v in needed.items()},header_bits=12))
 result['catalogue_sha256']=digest(result);return result
def main():
 a=read(D/'protocol_resolved_p0.json');out=D/'grid_catalogues';out.mkdir(exist_ok=False)
 base1024=read(B/'nestcom_7db_20261030_v1/cal_phy_nest/catalogue.json')
 assert base1024['profiles'][:372]==cr(a['catalogue_N1024'])['profiles']
 for N,base in [(1024,base1024),(512,cr(a['catalogue_N512']))]:
  cat=extend(N,base,a);save(out/f'N{N}.json',cat);print('CATALOGUE',N,len(cat['profiles']),max(p['profile_id']for p in cat['profiles']),cat['extension'])
 save(out/'scope.json',dict(status='PREPARED',N1024_inherits='Complete frozen 7dB NeST/Ada paid directory including the unchanged original372 NeST IDs.',external_protocols='Historical ARPC uses its own explicitly distinct session/directory (IDs0..95); it is not reinterpreted. Historical external Ada comparison keeps its own frozen directory. This NeST-only grid change neither alters nor reruns them.',N512_inherits='All3384 original IDs, including Ada and NULL, unchanged.',BPG='N1024 IDs3000..3011 conservatively reserved, no assignment or modification.',no_header_expansion=True))
if __name__=='__main__':main()
