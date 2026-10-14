"""Independent constructor-only grouped LDPC design, not a packet backend.

Original ep_phy.Backend.plan remains unchanged and cannot plan N!=956.
No encoder/decoder forward or channel API is exposed here.
"""
from common import *
from fractions import Fraction

class GroupLayoutPlanner:
 def __init__(self,encoder,identity):
  self.Encoder=encoder;self.identity=identity;self.cache={};self.constructor_calls=0
 def plan(self,N,q,rate):
  key=(int(N),int(q),str(rate))
  if key in self.cache:return self.cache[key]
  N,q,rate=key;f=Fraction(rate);n=N*q;k=n*f.numerator//f.denominator
  row=dict(N=N,q=q,nominal_rate=rate,n=n,k=k,source_capacity_bits=k-29,length_field_bits=13,crc_bits=16,layout_status='UNSUPPORTED',backend_identity=self.identity)
  assert 0<N<=956 and q in (2,4,6) and k>=31 and n>=k and n%q==0
  self.constructor_calls+=1
  try:e=self.Encoder(k,n,num_bits_per_symbol=q,device='cpu',precision='single')
  except (ValueError,AssertionError) as exc:
   row['constructor_error']=type(exc).__name__+': '+str(exc);self.cache[key]=row;return row
  assert e.k==k and e.n==n and e.k_ldpc-e.k_filler==k
  if n>e.n_cb_comp:
   row['constructor_error']='RV0 repetition outside original pinned encoder support';self.cache[key]=row;return row
  permutation=e.out_int.cpu().numpy();assert sorted(permutation.tolist())==list(range(n))
  row.update(layout_status='ACTUAL_CONSTRUCTOR_ADMITTED_NOT_PACKET_QUALIFIED',k_ldpc=int(e.k_ldpc),k_filler=int(e.k_filler),n_cb=int(e.n_cb),n_cb_comp=int(e.n_cb_comp),bg=str(e._bg),z=int(e.z),code_blocks=1,mother_bits=int(e.n_ldpc),puncturing_bits=int(e.n_ldpc-e.k_filler-n),puncturing_first_2Z=int(2*e.z),puncturing_remaining=int(e.n_cb_comp-n),shortening_bits=int(e.k_filler),repetition_bits=0,modulation_padding_bits=0,interleaver_sha256=asha(permutation),interleaver_raw_bytes_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(),actual_effective_rate=k/n,decoder_config=self.identity['decoder'],actual_constructor_used=True)
  row['geometry_id']=digest({k:row[k] for k in ('k','n','q','bg','z','k_ldpc','k_filler','mother_bits','puncturing_first_2Z','puncturing_remaining','interleaver_raw_bytes_sha256')})
  self.cache[key]=row;return row
