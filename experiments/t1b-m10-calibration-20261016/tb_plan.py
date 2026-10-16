"""Symmetric m10 strategy candidates and paid-directory extension, no model calls."""
from fractions import Fraction
import hashlib,json
SIZES=(1,2,3,4,5,6,8,10,13,16)
ENDS=tuple((m,k)for m in range(4,10)for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3))))
RATES=('1/2','2/3','3/4','5/6')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def nt(m,k):
 assert (m,k)in ENDS or (m,k)==(10,0)
 return sum(x*x for x in SIZES[:m])+k
def fallback(family):
 assert family in ('WHOLE','PARTIAL')
 return ((10,0),)+ (tuple((m,0)for m in range(9,3,-1)) if family=='WHOLE'else tuple(sorted(ENDS,key=lambda p:nt(*p),reverse=True)))
def choose(lengths,family,capacity):
 for p in fallback(family):
  n=lengths[p];assert type(n)is int and n>=2
  if n<=min(capacity,8191):return p
 raise ValueError('No actual stream fits; preserve and stop without inventing a packet')
def candidates():
 rows=[]
 for family in ('WHOLE','PARTIAL'):
  for q in (2,4,6):
   for rate in RATES:
    f=Fraction(rate);n=956*q;k=n*f.numerator//f.denominator
    rows.append(dict(candidate_id=f'm10_{family.lower()}_q{q}_r{rate.replace("/","-")}',strategy_family=family,target_m=10,target_K=0,q=q,nominal_rate=rate,k=k,n=n,source_capacity_bits=k-29,legacy_alias=f'm9'+('_K192'if family=='PARTIAL'else'')+f'_q{q}_r{rate.replace("/","-")}'))
 return rows
def extend_catalogue(old):
 assert len(old['profiles'])==360 and len({p['profile_id']for p in old['profiles']})==360
 profiles=[dict(p)for p in old['profiles']]
 for q in (2,4,6):
  for rate in RATES:
   oldp=next(p for p in old['profiles']if(p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==('EC_VAR_WHOLE',9,0,q,rate))
   p={k:v for k,v in oldp.items()if k!='profile_key'};p.update(profile_id=len(profiles),m=10,K=0,token_count=680)
   p['profile_key']=digest(p);profiles.append(p)
 assert profiles[:360]==old['profiles'] and [p['profile_id']for p in profiles]==list(range(372))
 v=dict(old,protocol='NESTCOM_T1B_M10_PUBLIC_DIRECTORY_V1',profiles=profiles,requested_profiles=372,status='M10_DIRECTORY_EXTENSION_INHERITS_12_CLOSED_LAYOUTS',old_catalogue_sha256=old['catalogue_sha256'])
 v.pop('catalogue_sha256');v['catalogue_sha256']=digest(v);return v
def directory_unchanged_on_old_event(header):
 # The unchanged paid-header decoder only consults the public directory after CRC.
 # If CRC passed but the old ID was illegal, raw ID was discarded; do not infer it.
 return not header['header_crc_ok'] or header['header_fields_legal']
def select_winner(means,legacy_ids):
 # Exact ties retain the original lexical rule among legacy candidates, then new IDs.
 return min(means,key=lambda c:(-means[c],c not in legacy_ids,c))
