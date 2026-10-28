"""Public fixed-frame protocol. No image-dependent choice or model inference."""
from fractions import Fraction
from block_common import *
VERSION='SAME_SCALE_CAB_N1024_V1'
RATES=('1/4','1/3','1/2','2/3')
ALLOC=((478,239,239),(598,179,179),(718,119,119))
CONFIGS=((4,12),(4,25),(5,18),(5,36))
SIZES=(1,2,3,4,5,6,8,10,13,16)
OFFSETS=tuple(sum(x*x for x in SIZES[:i])for i in range(11))

def endpoints():
 sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
 from ep_source_codec import all_endpoints
 e=tuple(all_endpoints())+((10,0),)
 assert len(e)==25 and len(set(e))==25
 return e

def geometry(N,rate):
 f=2*N*Fraction(rate);k=f.numerator//f.denominator
 return dict(N=N,k=k,n=2*N,q=2,rate=rate,capacity=k-29)

def catalogue():
 profiles={};identities={};candidates=[]
 def add(p):
  p=dict(version=VERSION,**p);key=digest(p)
  if key not in identities:
   i=len(profiles);assert i<4096;identities[key]=i;profiles[str(i)]=dict(profile_id=i,**p)
  return identities[key]
 for m,K in CONFIGS:
  for Ns in ALLOC:
   for rc in RATES:
    for re in RATES:
     if Fraction(rc)>Fraction(re):continue
     layouts=[geometry(n,r)for n,r in zip(Ns,(rc,re,re))]
     Ks=[K,SIZES[m]**2//2,0]if K==SIZES[m]**2 else[K,0]
     ids=[add(dict(kind='THREE',m=m,K=k,positions_A=list(range(k//2)),positions_B=list(range(k//2,k)),layouts=layouts,null=False,empty_enhancement=k==0))for k in Ks]
     ids.append(add(dict(kind='THREE',m=m,K=0,positions_A=[],positions_B=[],layouts=layouts,null=True,empty_enhancement=True)))
     candidates.append(dict(candidate_id=f'T_m{m}_K{K}_N{Ns[0]}_r{rc.replace("/", "-")}_{re.replace("/", "-")}',kind='THREE',target_m=m,target_K=K,uniform=rc==re,profiles=ids))
 for m,K in endpoints():
  for rate in RATES:
   # Current whole-only targets retain whole-only fallback; partial targets admit all shorter quarters.
   allowed=[e for e in endpoints()if OFFSETS[e[0]]+e[1]<=OFFSETS[m]+K and(K>0 or e[1]==0)]
   ids=[add(dict(kind='SINGLE',m=a,K=b,layouts=[geometry(956,rate)],null=False))for a,b in reversed(allowed)]
   ids.append(add(dict(kind='SINGLE',m=4,K=0,layouts=[geometry(956,rate)],null=True)))
   candidates.append(dict(candidate_id=f'S_m{m}_K{K}_r{rate.replace("/", "-")}',kind='SINGLE',target_m=m,target_K=K,rate=rate,profiles=ids))
 assert len(candidates)==220
 return dict(version=VERSION,profiles=profiles,candidates=candidates)

def choose(candidate,profiles,lengths):
 for pid in candidate['profiles']:
  p=profiles[str(pid)]
  if p['null']:return p
  if p['kind']=='SINGLE':b=[lengths['single'][f'm{p["m"]}_K{p["K"]}']]
  else:
   b=[lengths['C'][str(p['m'])]]+[lengths[f'm{p["m"]}_K{p["K"]}'][z]if p['K']else 0 for z in ('A','B')]
  if all(x<=g['capacity']for x,g in zip(b,p['layouts'])):return p
 raise AssertionError('Registered fallback must end at paid NULL')

def packet_module():
 import ast,types
 p=R/'experiments/layered-entropy-feasibility-20261014/packet_two_group.py'
 names={'binary','intbits','bitsint','crc16','pack_group','parse_group'}
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in nodes}==names
 ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 return types.SimpleNamespace(**{n:ns[n]for n in names})

def state_identity(m,coarse,positions,values,preprocessing,route):
 assert len(coarse)==OFFSETS[m] and len(positions)==len(values)
 assert 4<=m<=10 and positions==sorted(set(positions))
 assert (all(0<=i<SIZES[m]**2 for i in positions)if m<10 else not positions)
 # Normalize complete boundary to the next complete scale, without changing the wire profile.
 if m<10 and positions==list(range(SIZES[m]**2)):
  coarse=list(coarse)+list(values);m+=1;positions=[];values=[]
 return dict(m=m,coarse=list(map(int,coarse)),positions=positions,values=list(map(int,values)),preprocessing=preprocessing,route=route,completion='class1000_greedy_no_CFG_FP32_no_grad_TF32off')

def replay_states(c,a,b,enabled=True):
 if not c:return dict(ALL='GRAY',SEQ='GRAY',SEL='GRAY')
 if not enabled:return dict(ALL='C',SEQ='C',SEL='C')
 return dict(ALL='CAB'if a and b else'GRAY',SEQ='CAB'if a and b else'CA'if a else'C',SEL='CAB'if a and b else'CA'if a else'CB'if b else'C')
