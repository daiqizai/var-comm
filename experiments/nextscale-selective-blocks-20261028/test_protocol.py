"""CPU engineering checks only: real AC syntax with synthetic conditional probabilities."""
from block_common import *
from protocol import *
from contextlib import contextmanager,nullcontext

def test_paid_geometry_and_public_catalogue():
 c=catalogue();assert len(c['candidates'])==220
 assert sum(x['kind']=='THREE'and x['uniform']for x in c['candidates'])==48
 assert sum(x['kind']=='THREE'and not x['uniform']for x in c['candidates'])==72
 expected={478:[210,289,449,608],598:[270,369,569,768],718:[330,449,689,928],239:[90,130,210,289],179:[60,90,150,209],119:[30,50,90,129],956:[449,608,927,1245]}
 for n,a in expected.items():assert [geometry(n,r)['capacity']for r in RATES]==a
 assert len(c['profiles'])<4096
 for p in c['profiles'].values():assert sum(x['N']for x in p['layouts'])==956

def test_rejections_even_with_repaired_crc():
 p=packet_module();info=p.pack_group([1,0],59)
 assert p.parse_group(info,59)['bits'].tolist()==[1,0]
 bad=info.copy();bad[14]^=1;assert p.parse_group(bad,59)['status']=='CRC_REJECT'
 for field,status in [('length','DECLARED_LENGTH_REJECT'),('padding','KNOWN_PADDING_REJECT')]:
  bad=info.copy()
  if field=='length':bad[:13]=p.intbits(31,13)
  else:bad[15]=1
  bad[-16:]=p.intbits(p.crc16(bad[:-16]),16)
  assert p.parse_group(bad,59)['status']==status
 empty=p.pack_group([],59,True);assert p.parse_group(empty,59,True)['accepted']
 assert not p.parse_group(empty,59,False)['accepted']

def test_hole_identity_and_replay():
 a=state_identity(4,[0]*30,list(range(6)),[1]*6,'x','r')
 b=state_identity(4,[0]*30,list(range(6,12)),[1]*6,'x','r')
 assert digest(a)!=digest(b)
 assert replay_states(True,False,True)==dict(ALL='GRAY',SEQ='C',SEL='CB')
 assert set(replay_states(False,True,True).values())=={'GRAY'}
 full=state_identity(4,[0]*30,list(range(25)),[1]*25,'x','r')
 assert full==state_identity(5,[0]*30+[1]*25,[],[],'x','r')

def test_independent_arithmetic_and_missing_condition():
 from same_scale_codec import SameScaleCodec
 codec=SameScaleCodec(R)
 calls=[]
 class Provider:
  def __init__(self):self.scale=0;self.previous=[]
  def cdf(self):
   calls.append(self.scale);n=SIZES[self.scale]**2
   logits=np.zeros((n,4096),np.float64)
   offset=sum(self.previous)%4096
   logits[np.arange(n),(np.arange(n)+offset)%4096]=1
   return codec.entropy.probability_cdf(logits)
  def advance(self,v):self.previous+=list(map(int,v));self.scale+=1
 @contextmanager
 def context(family):yield Provider()
 codec._context=context;codec._inference=lambda f:nullcontext()
 tokens=(np.arange(680,dtype=np.int64)*37)%4096
 streams,meta,trace=codec.encode_boundary(tokens)
 for m,K in CONFIGS:
  C=streams['C'+str(m)];A=streams[f'm{m}_K{K}_A'];B=streams[f'm{m}_K{K}_B']
  whole=codec.receive(m,K,C,A,B);hole=codec.receive(m,K,C,None,B)
  assert whole['C']==tokens[:OFFSETS[m]].tolist()==hole['C']
  assert whole['B']==hole['B']==tokens[OFFSETS[m]+K//2:OFFSETS[m]+K].tolist()
  before=len(calls);missing=codec.receive(m,K,None,None,B)
  assert len(calls)==before and missing['B']is None and missing['status_B']=='CONDITION_UNAVAILABLE'
  assert len(whole['trace'])==m+1
