"""Source-only m10 audit; existing 24 endpoint implementation stays unchanged."""
from common import R,OFFSETS,ENDS,asha
import sys,numpy as np
sys.path.insert(0,str(R/'experiments/entropy-partial-mismatch-20261010/scripts'))
from ep_source_codec import PartialSourceCodec,WHOLE_FAMILY
from t1_entropy_core import source_tokens
from t6_source_codec import SourceCodec10
class AuditCodec(PartialSourceCodec):
 def encode_all(self,tokens):
  tokens=source_tokens(tokens);encoder=self.Encoder();out={};trace=[]
  def flush(m,K):
   bits=np.asarray(encoder.finish(),dtype=np.uint8)
   out[m,K]=dict(bits=bits,finish_bits=len(bits)-len(encoder.bits))
  with self._inference(WHOLE_FAMILY),self._context(WHOLE_FAMILY)as provider:
   for scale in range(10):
    values=tokens[OFFSETS[scale]:OFFSETS[scale+1]];cdf=self._cdf(WHOLE_FAMILY,provider,scale)
    trace.append(dict(scale=scale,cdf_sha256=asha(cdf)));start=0
    for K in sorted(K for m,K in ENDS if m==scale and K>0):
     encoder.encode(values[start:K],cdf[start:K]);flush(scale,K);start=K
    encoder.encode(values[start:],cdf[start:]);provider.advance(values.copy())
    if (scale+1,0)in ENDS or scale==9:flush(scale+1,0)
  assert set(out)==set(ENDS)|{(10,0)}
  return out,trace
 def decode_m10(self,received_bits):
  # Reuse the existing independent m10 receiver, with only received bits.
  return SourceCodec10.decode(self,WHOLE_FAMILY,received_bits,10)
