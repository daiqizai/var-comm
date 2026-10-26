"""Additive paid profiles on the unchanged N1024 LDPC/QAM/header engine."""
from common import *
import types,numpy as np
MCS=((2,'1/2'),(2,'2/3'),(4,'2/3'),(4,'3/4'),(6,'5/6'))
CANDIDATES=((4,2,'1/2'),(4,2,'2/3'),(10,4,'2/3'),(10,4,'3/4'),(19,4,'2/3'),(19,6,'5/6'))
LLAMA='LLAMAGEN_16K_PREFIX';NULL='EXPLICIT_NULL_FIXEDFRAME'
def isolated_engine():
 p=R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py';sys.path.insert(0,str(p.parent))
 ns=dict(__file__=str(p),__name__='_abc_bound_ep_phy');exec(compile(p.read_bytes(),str(p),'exec'),ns)
 assert ns['FAMILIES']==('EC_STATIC_WHOLE','EC_VAR_WHOLE','EC_VAR_PARTIAL')
 # Extend only the legal arithmetic family directory. The body/framing/LDPC math is unchanged.
 ns['FAMILIES']=(*ns['FAMILIES'],LLAMA,NULL)
 return types.SimpleNamespace(**ns)
def catalogue(old,branch):
 sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'));from tb_plan import extend_catalogue
 cat=extend_catalogue(old);profiles=list(cat['profiles'])
 def append(base,**changes):
  p={k:v for k,v in base.items()if k!='profile_key'};p.update(profile_id=len(profiles),**changes);p['profile_key']=digest(p);profiles.append(p)
 pairs=MCS if branch=='C'else((2,'2/3'),(4,'2/3'),(6,'5/6'))
 for q,rate in pairs:
  base=next(p for p in profiles if(p['family'],p['m'],p['K'],p['q'],p['nominal_rate'])==('EC_VAR_WHOLE',9,0,q,rate))
  if branch=='C':
   for count in range(1,257):append(base,family=LLAMA,m=count,K=0,token_count=count,prefix_count=count,source_syntax='LLAMAGEN_VOCAB16384_INTEGER24_ARITHMETIC32_PREFIX_V1')
  else:
   for m,K in grids()['8']:
    if(m,K)not in grids()['4']:append(base,family='EC_VAR_PARTIAL',m=m,K=K,token_count=OFFSETS[m]+K)
  append(base,family=NULL,m=0,K=0,token_count=0,prefix_count=0,null_payload=[0,0],null_rule='Explicit paid null header and body. No source decode; fixed gray; denominator retained.')
 assert profiles[:360]==old['profiles']and len(profiles)<4096
 cat={k:v for k,v in cat.items()if k!='catalogue_sha256'};cat.update(protocol='NEST_ABC_ADDITIVE_FIXEDFRAME_'+branch+'_V1',profiles=profiles,requested_profiles=len(profiles),original360_unchanged=True)
 cat['catalogue_sha256']=digest(cat);return cat
def runtime(reg):
 phy=isolated_engine();rt=phy.create_runtime(R,str(checked(reg['phy_qualification'])))
 current=catalogue(rt.catalogue,reg['branch']);assert current==cr(reg['catalogue'])
 rt.catalogue=current;rt.profiles={str(p['profile_id']):p for p in current['profiles']}
 return phy,rt
def choose(streams,profiles,branch,q,rate,grid=None):
 if branch=='C':candidates=[(i,'p%03d'%i)for i in range(256,0,-1)]
 else:candidates=[(e,'m%d_K%d'%e)for e in reversed(grids()[str(grid)])]
 eligible=[p for p in profiles if p['q']==q and p['nominal_rate']==rate];capacity=eligible[0]['source_capacity_bits'];attempts=[]
 for endpoint,key in candidates:
  bits=streams[key];assert bits.dtype==np.uint8 and bits.ndim==1 and len(bits)>=2;fits=len(bits)<=min(capacity,8191);attempts.append(dict(endpoint=endpoint,arithmetic_bits=len(bits),fits=bool(fits)))
  if fits:
   if branch=='C':p=next(p for p in eligible if p['family']==LLAMA and p['prefix_count']==endpoint)
   else:p=next(p for p in eligible if p['family']in('EC_VAR_WHOLE','EC_VAR_PARTIAL')and(p['m'],p['K'])==endpoint)
   return p,bits,attempts
 p=next(p for p in eligible if p['family']==NULL);return p,np.array([0,0],np.uint8),attempts
