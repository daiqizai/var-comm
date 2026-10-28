"""Independent N512 namespace; exact old math, only frame dimensions adapted.

The old file is never edited or monkey-patched. Every transformed AST node is
enumerated and reverse-checked. A new paid catalogue and new real qualification
are required; historical N1024 qualification cannot qualify N512 packets.
"""
from n512_common import *
import ast,copy,types,math
from fractions import Fraction
from functools import lru_cache
SIZES=(1,2,3,4,5,6,8,10,13,16)
ENDS=tuple((m,k) for m in range(4,10) for k in (0,*(SIZES[m]**2*j//4 for j in (1,2,3))))+((10,0),)
RATES=('1/2','2/3','3/4','5/6');MCS=tuple((q,r) for q in (2,4,6) for r in RATES)
NEST_W='EC_VAR_WHOLE';NEST_P='EC_VAR_PARTIAL';ADA='ADA_PREDICTED_CLASS_PREFIX';NULL='EXPLICIT_NULL_FIXEDFRAME'
WIRE='NESTCOM_N512_20261028_V1'
def nt(e):return sum(n*n for n in SIZES[:e[0]])+e[1]
def candidate_id(method,e,q,r):return f'{method}_m{e[0]:03d}_K{e[1]:03d}_q{q}_r{r.replace("/","-")}'
def candidates(admitted):
 rows=[]
 for method in ('NEST','WHOLE','ADA'):
  for e in (((256,0),) if method=='ADA' else ENDS if method=='NEST' else tuple(e for e in ENDS if e[1]==0)):
   for q,r in admitted:rows.append(dict(method=method,target=list(e),q=q,rate=r,candidate_id=candidate_id(method,e,q,r)))
 return rows
def choose(streams,profiles,candidate,predicted_class=None):
 c=candidate;method=c['method'];q,r=c['q'],c['rate'];target=tuple(c['target'])
 eligible=[p for p in profiles if (p['q'],p['nominal_rate'])==(q,r)]
 assert eligible
 cap=eligible[0]['k']-(39 if method=='ADA' else 29)
 chain=range(target[0],0,-1) if method=='ADA' else sorted((e for e in ENDS if nt(e)<=nt(target) and (method=='NEST' or e[1]==0)),key=nt,reverse=True)
 for e in chain:
  bits=streams[f'p{e:03d}' if method=='ADA' else f'm{e[0]}_K{e[1]}']
  assert bits.dtype==np.uint8 and bits.ndim==1 and len(bits)>=2 and np.isin(bits,(0,1)).all()
  if len(bits)>cap:continue
  if method=='ADA':
   assert type(predicted_class)is int and 0<=predicted_class<1000
   payload=np.concatenate((np.array([(predicted_class>>j)&1 for j in range(9,-1,-1)],np.uint8),bits))
   p=next(p for p in eligible if p['family']==ADA and p['m']==e)
  else:
   payload=bits.copy();p=next(p for p in eligible if p['family']==(NEST_P if e[1] else NEST_W) and (p['m'],p['K'])==e)
  return p,payload,False
 return next(p for p in eligible if p['family']==NULL),np.zeros(2,np.uint8),True

def engine():
 p=R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'
 sys.path.insert(0,str(p.parent))
 ns=dict(__file__=str(p),__name__='_n512_original_dependency_namespace')
 exec(compile(p.read_bytes(),str(p),'exec'),ns)
 wanted={'Backend','transmit_body','transmit_frame','receive_body','receive_frame','standard_noise'}
 tree=ast.parse(p.read_text());selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
 assert {n.name for n in selected}==wanted
 old=ast.Module(body=selected,type_ignores=[]);new=copy.deepcopy(old);changes=[]
 for node in ast.walk(new):
  if isinstance(node,ast.Constant) and type(node.value)is int and node.value in (956,1024):
   changes.append(dict(line=node.lineno,old=node.value,new={956:444,1024:512}[node.value]));node.value={956:444,1024:512}[node.value]
  elif isinstance(node,ast.Constant) and isinstance(node.value,str):node.value=node.value.replace('956','444').replace('1024','512')
 # Reverse transformation proves absence of any other numerical/control edit.
 rev=copy.deepcopy(new)
 for node in ast.walk(rev):
  if isinstance(node,ast.Constant) and type(node.value)is int and node.value in (444,512):node.value={444:956,512:1024}[node.value]
  elif isinstance(node,ast.Constant) and isinstance(node.value,str):node.value=node.value.replace('444','956').replace('512','1024')
 assert ast.dump(rev,include_attributes=False)==ast.dump(old,include_attributes=False)
 assert len(changes)==8,changes
 ns.update(N=512,BODY_SYMBOLS=444,HEADER_SYMBOLS=68,PROTOCOL=WIRE,WIRE_PROTOCOL=WIRE,FAMILIES=(NEST_W,NEST_P,ADA,NULL))
 exec(compile(new,str(p)+'::N512_DIMENSIONS','exec'),ns)
 return types.SimpleNamespace(**ns),dict(original=pin(p),nodes=sorted(wanted),changes=changes,reverse_AST_equal=True)

def runtime():
 phy,proof=engine();rt=phy.Runtime(R);layouts=[];rejected=[];profiles=[]
 for q,r in MCS:
  n=444*q;k=int(Fraction(r)*n)
  try:layout=rt.backend.plan(k,n,q);rt.backend.codecs(k,n,q)
  except phy.UnsupportedConfiguration as exc:rejected.append(dict(q=q,rate=r,reason=str(exc)));continue
  layouts.append(dict(q=q,rate=r,layout=layout))
  endpoint_rows=[(NEST_P if K else NEST_W,m,K,nt((m,K))) for m,K in ENDS]+[(ADA,i,0,i) for i in range(1,257)]+[(NULL,0,0,0)]
  for family,m,K,count in endpoint_rows:
   v=dict(profile_id=len(profiles),family=family,mode='arithmetic',m=m,K=K,token_count=count,q=q,nominal_rate=r,k=k,n=n,N=512,header_symbols=68,body_symbols=444,frame_padding_symbols=0,length_bits=13,crc_bits=16,class_bits=10 if family==ADA else 0,source_capacity_bits=k-29,arithmetic_capacity_bits=k-(39 if family==ADA else 29),layout=layout,layout_id=layout['layout_id'])
   v['profile_key']=digest(v);profiles.append(v)
 assert profiles and len(profiles)<4096 and len({p['profile_id'] for p in profiles})==len(profiles)
 cat=dict(protocol=WIRE,profiles=profiles,layouts=layouts,rejected_layouts=rejected,requested_layouts=12,AST_proof=proof);cat['catalogue_sha256']=digest(cat)
 rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in profiles}
 return phy,rt,cat

class PacketLedger:
 def __init__(self,ledger,out):self.ledger,self.out=ledger,out
 def call(self,event,request,fn):
  kind=event['kind'];key=digest([event,request])
  def perform():
   result=fn();save(self.out/(key+'.json'),dict(event=event,request=request,result=result));return result
  return self.ledger.call(kind,key,perform)

def received_projection(rx):
 """No transmitter identity or truth is accepted by this function."""
 if not rx['header']['header_ok']:return dict(status='FIXED_GRAY',reason='HEADER_REJECT')
 p,b=rx['rx_profile'],rx['body']
 if not b['crc_accepted'] or not b['parser_accepted']:return dict(status='FIXED_GRAY',reason=b['status'])
 bits=np.asarray(b['payload'],np.uint8)
 if p['family']==NULL:return dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE' if np.array_equal(bits,[0,0]) else 'INVALID_NULL_PAYLOAD')
 if p['family']==ADA:
  if len(bits)<12:return dict(status='FIXED_GRAY',reason='CLASS_CONTAINER_REJECT')
  cls=sum(int(v)<<(9-j) for j,v in enumerate(bits[:10]))
  if cls>=1000:return dict(status='FIXED_GRAY',reason='CLASS_CONTAINER_REJECT')
  return dict(status='SOURCE_INPUT',codec='ADA',m=p['m'],K=0,received_class=cls,bits=bits[10:].tolist())
 if p['family'] in (NEST_W,NEST_P):return dict(status='SOURCE_INPUT',codec='NEST',m=p['m'],K=p['K'],family=p['family'],bits=bits.tolist())
 return dict(status='FIXED_GRAY',reason='UNKNOWN_FAMILY')
