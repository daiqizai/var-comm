"""Three independently terminated streams, with boundary probabilities conditioned only on C."""
from block_common import *
from protocol import CONFIGS,SIZES,OFFSETS
import ast
sys.path[:0]=[str(R/'experiments/kodak-current-20261026/nest_v1'),str(R/'experiments/entropy-partial-mismatch-20261010/scripts'),str(R/'experiments/wcl-evidence-closure-20261009/scripts')]
from ep_source_codec import PartialSourceCodec,WHOLE_FAMILY
from t1_codec_runtime import InvalidSourceStream
from t1_entropy_core import source_tokens
ns=dict(PartialSourceCodec=PartialSourceCodec,WHOLE_FAMILY=WHOLE_FAMILY,InvalidSourceStream=InvalidSourceStream,np=np,asha=asha,OFFSETS=OFFSETS)
p=R/'experiments/layered-entropy-feasibility-20261014/source_split_codec.py'
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='SplitSourceCodec')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)

class SameScaleCodec(ns['SplitSourceCodec']):
 def encode_boundary(self,tokens):
  tokens=source_tokens(tokens);whole=self.Encoder();streams={};trace=[];meta={}
  def finish(name,enc):
   b=np.asarray(enc.finish(),dtype=np.uint8);streams[name]=b;meta[name]=dict(bits=len(b),finish_bits=len(b)-len(enc.bits),array_sha256=asha(b))
  with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as p:
   for s in range(6):
    if s in(4,5):finish(f'C{s}',whole)
    tab=self._table(p,s,trace);v=tokens[OFFSETS[s]:OFFSETS[s+1]]
    if s in(4,5):
     for m,K in CONFIGS:
      if m!=s:continue
      for label,lo,hi in [('A',0,K//2),('B',K//2,K)]:
       e=self.Encoder();e.encode(v[lo:hi],tab[lo:hi]);finish(f'm{m}_K{K}_{label}',e)
    whole.encode(v,tab);p.advance(v.copy())
  return streams,meta,trace

 def decode_slice(self,bits,table):
  b=np.asarray(bits)
  if b.ndim!=1 or len(b)<2 or not np.isin(b,(0,1)).all():raise InvalidSourceStream('Invalid binary source')
  d=self._decoder(b);canonical=self.Encoder()
  try:v=d.decode(table)
  except InvalidSourceStream:raise
  except ValueError as e:raise InvalidSourceStream(str(e))from e
  canonical.encode(v,table)
  if not np.array_equal(canonical.finish(),b)or d.position-len(b)!=30:raise InvalidSourceStream('Noncanonical independent termination')
  return v.astype(np.int64,copy=False)

 def receive(self,m,K,C,A,B):
  """Only public m/K and actual validated source streams. Never accepts source pixels/tokens/CDF."""
  assert (m,K)in CONFIGS or(m in(4,5)and K==0)
  result=dict(m=m,K=K,C=None,A=None,B=None,trace=[],status_C='MISSING',status_A='CONDITION_UNAVAILABLE',status_B='CONDITION_UNAVAILABLE')
  if C is None:return result
  with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as p:
   try:result['C']=self._decode_group(p,np.asarray(C,np.uint8),0,m,0,result['trace']).tolist();result['status_C']='ACCEPTED'
   except InvalidSourceStream as e:result['status_C']='AC_REJECT:'+str(e);return result
   if K==0:result.update(status_A='PAID_EMPTY',status_B='PAID_EMPTY');return result
   tab=self._table(p,m,result['trace'])
   for label,bits,lo,hi in [('A',A,0,K//2),('B',B,K//2,K)]:
    if bits is None:result['status_'+label]='MISSING';continue
    try:result[label]=self.decode_slice(bits,tab[lo:hi]).tolist();result['status_'+label]='ACCEPTED'
    except InvalidSourceStream as e:result['status_'+label]='AC_REJECT:'+str(e)
  return result
