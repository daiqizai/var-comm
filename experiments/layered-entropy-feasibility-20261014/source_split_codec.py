"""Independent terminated arithmetic streams; only VAR conditional state spans groups."""
from pathlib import Path
import sys,numpy as np
from common import R,ENDS,OFFSETS,SIZES,asha
sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
from ep_source_codec import PartialSourceCodec,WHOLE_FAMILY
from t1_codec_runtime import InvalidSourceStream
from t1_entropy_core import source_tokens
from packet_two_group import lookup,parse_group
class SplitSourceCodec(PartialSourceCodec):
 def _table(self,p,scale,trace):
  table=self._cdf(WHOLE_FAMILY,p,scale);trace.append(dict(scale=scale,cdf_sha256=asha(table)));return table
 def encode_all(self,tokens):
  """Exactly one full conditional TX traversal for all24 endpoints/68 edges.

  The enhancement encoders really start at j. finish() is called on each
  independent state; no stream slicing or subtractive enhancement length.
  """
  tokens=source_tokens(tokens);single=self.Encoder();enh={};result=dict(single={},base={},enhancement={},cdf_trace=[])
  def finished(enc):
   bits=np.asarray(enc.finish(),dtype=np.uint8)
   return dict(bits=bits,arithmetic_bits=len(bits),finish_bits=len(bits)-len(enc.bits))
  def flush(m,K):
   result['single'][(m,K)]=finished(single)
   if K==0 and m in (4,5,6,7):result['base'][m]=finished(single)
   for j,enc in enh.items():
    if OFFSETS[m]+K>OFFSETS[j]:result['enhancement'][(j,m,K)]=finished(enc)
  with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY) as p:
   for scale in range(10):
    values=tokens[OFFSETS[scale]:OFFSETS[scale+1]];table=self._table(p,scale,result['cdf_trace']);start=0
    for K in sorted(K for m,K in ENDS if m==scale and K>0):
     single.encode(values[start:K],table[start:K])
     for enc in enh.values():enc.encode(values[start:K],table[start:K])
     flush(scale,K);start=K
    if scale<9:
     single.encode(values[start:],table[start:])
     for enc in enh.values():enc.encode(values[start:],table[start:])
     p.advance(values.copy())
     if (scale+1,0) in ENDS:flush(scale+1,0)
     if scale+1 in (4,5,6,7):enh[scale+1]=self.Encoder()
  assert len(result['single'])==24 and len(result['base'])==4 and len(result['enhancement'])==68
  return result
 def _decoder(self,bits):
  class Bounded(self.Decoder):
   def read_bit(inner):
    if inner.position>=len(inner.bits)+30:raise InvalidSourceStream('Terminal lookahead exceeded')
    return super().read_bit()
  return Bounded(bits)
 def _decode_group(self,p,bits,start,m,K,trace):
  decoder=self._decoder(bits);canonical=self.Encoder();parts=[]
  for scale in range(start,m+int(K>0)):
   table=self._table(p,scale,trace);cdf=table if scale<m else table[:K]
   try:values=decoder.decode(cdf)
   except InvalidSourceStream:raise
   except ValueError as e:raise InvalidSourceStream(str(e)) from e
   canonical.encode(values,cdf);parts.append(values.copy())
   if scale<m:p.advance(values.copy())
  if not np.array_equal(canonical.finish(),bits):raise InvalidSourceStream('Noncanonical independent termination')
  if decoder.position-len(bits)!=30:raise InvalidSourceStream('Terminal lookahead mismatch')
  return np.concatenate(parts).astype(np.int64,copy=False)
 def decode_received(self,received_header_bits,base_information,enhancement_information,catalogue):
  """Actual paid ID and actual received information only. No source ID/truth/CDF."""
  profile=lookup(received_header_bits,catalogue);trace=[]
  gray=lambda reason:dict(status='GRAY',reason=reason,received_tokens=np.empty(0,np.int64),m=None,K=None,cdf_trace=trace,receiver_truth_used=False,TX_CDF_used=False)
  if profile is None:return gray('HEADER_PROFILE_REJECT')
  if profile['null_profile']:return gray('TX_SOURCE_UNENCODABLE_PAID_NULL')
  B=parse_group(base_information,profile['k_B']);j=profile['j']
  if not B['accepted']:return gray('BASE_'+B['status'])
  with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY) as p:
   try:base=self._decode_group(p,B['bits'],0,j,0,trace)
   except InvalidSourceStream as e:return gray('BASE_SOURCE_'+str(e))
   retained=dict(status='BASE_ONLY',received_tokens=base.copy(),m=j,K=0,cdf_trace=trace,receiver_truth_used=False,TX_CDF_used=False)
   E=parse_group(enhancement_information,profile['k_E'],profile['explicit_empty_enhancement'])
   if not E['accepted']:return dict(retained,reason='ENHANCEMENT_'+E['status'])
   if profile['explicit_empty_enhancement']:return dict(retained,reason='EXPLICIT_PAID_EMPTY_ENHANCEMENT')
   try:extra=self._decode_group(p,E['bits'],j,profile['m'],profile['K'],trace)
   except InvalidSourceStream as e:return dict(retained,reason='ENHANCEMENT_SOURCE_'+str(e))
   return dict(retained,status='FULL_RECEIVED_PREFIX',received_tokens=np.concatenate((base,extra)),m=profile['m'],K=profile['K'],reason='TWO_INDEPENDENT_CANONICAL_STREAMS')
 def receive_image(self,received_header_bits,base_information,enhancement_information,catalogue,render_received):
  """Renderer sees only fixed actual RX tokens; failed enhancement retains base."""
  rx=self.decode_received(received_header_bits,base_information,enhancement_information,catalogue)
  if rx['status']=='GRAY':return np.full((3,256,256),.5,np.float32),rx
  with self._inference(WHOLE_FAMILY):im=render_received(self.native,rx['received_tokens'].copy(),rx['m'],rx['K'])
  return im,rx
