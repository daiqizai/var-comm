"""Finite source/MCS policy and charged two-fragment geometry, without models."""
from fractions import Fraction
SIZES=(1,2,3,4,5,6,8,10,13,16)
NEST_ENDPOINTS=tuple((m,k) for m in range(4,10) for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3))))+((10,0),)
RATES=('1/2','2/3','3/4','5/6')
def layouts():
 out=[]
 for q in (2,4,6):
  for rate in RATES:
   r=Fraction(rate);n=956*q;k=n*r.numerator//r.denominator;b=(k-13-16-96)//8
   out.append(dict(layout_id=f'q{q}_r{rate.replace("/","-")}',q=q,nominal_rate=rate,k=k,n=n,actual_rate=k/n,fragment_bytes=b,object_bits=16*b,header_symbols_per_fragment=68,body_symbols_per_fragment=956,length_bits_per_fragment=13,crc_bits_per_fragment=16,reassembly_bits_per_fragment=96,whole_crc_copies=2,body_slack_bits=k-125-8*b,total_complex_symbols=2048,nominal_symbol_energy=2))
 return out
def nest_family(endpoint):
 m,k=NEST_ENDPOINTS[endpoint]
 return 'EC_VAR_WHOLE' if k==0 else 'EC_VAR_PARTIAL'
def fallback(codec,target):
 if codec==1:
  assert 1<=target<=7
  return tuple(range(target,0,-1))
 assert codec==0 and 0<=target<25
 m,k=NEST_ENDPOINTS[target]
 # Original m4..m9 whole targets retain their whole-only fallback. The new
 # m10 target belongs to the current NeST partial interface (T1-B PARTIAL),
 # whose frozen fallback visits all24 original endpoints in token order.
 if k==0 and m<10:return tuple(i for i in range(target,-1,-1) if NEST_ENDPOINTS[i][1]==0)
 return tuple(range(target,-1,-1))
def choose(streams,codec,target,layout):
 for endpoint in fallback(codec,target):
  s=streams[endpoint]
  if (s['bit_length']+7)//8<=2*layout['fragment_bytes']:return endpoint
 return None
def policies(codec):
 return [dict(policy_id=('nest' if codec==0 else 'arpc')+f'_t{target}_'+g['layout_id'],codec=codec,target=target,layout_id=g['layout_id']) for target in (range(25) if codec==0 else range(1,8)) for g in layouts()]
