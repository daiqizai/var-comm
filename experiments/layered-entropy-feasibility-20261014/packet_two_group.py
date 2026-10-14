"""New paid two-stream information syntax. No LDPC or channel is executed."""
import numpy as np
from common import ENDS,OFFSETS
def binary(a):
 a=np.asarray(a);assert a.ndim==1 and np.isin(a,(0,1)).all();return a.astype(np.uint8,copy=True)
def intbits(n,width):
 assert type(n) is int and 0<=n<1<<width
 return np.asarray([(n>>i)&1 for i in range(width-1,-1,-1)],dtype=np.uint8)
def bitsint(a):
 n=0
 for b in binary(a):n=(n<<1)|int(b)
 return n
def crc16(a):
 c=65535
 for b in binary(a):
  f=(c>>15)^int(b);c=((c<<1)^(0x1021 if f else 0))&65535
 return c
def pack_group(bits,k,explicit_empty=False):
 a=binary(bits);assert type(k) is int and k>=31 and len(a)<=min(k-29,8191)
 assert (len(a)==0 and explicit_empty) or (len(a)>=2 and not explicit_empty)
 body=np.concatenate((intbits(len(a),13),a,np.zeros(k-29-len(a),np.uint8)))
 return np.concatenate((body,intbits(crc16(body),16)))
def parse_group(info,k,explicit_empty=False):
 try:a=binary(info)
 except AssertionError:return dict(accepted=False,status='MALFORMED_BINARY')
 if len(a)!=k:return dict(accepted=False,status='INFORMATION_LENGTH_REJECT')
 if crc16(a[:-16])!=bitsint(a[-16:]):return dict(accepted=False,status='CRC_REJECT')
 L=bitsint(a[:13]);allowed=(L==0 if explicit_empty else 2<=L<=min(k-29,8191))
 if not allowed:return dict(accepted=False,status='DECLARED_LENGTH_REJECT')
 if np.any(a[13+L:-16]):return dict(accepted=False,status='KNOWN_PADDING_REJECT')
 return dict(accepted=True,status='PAID_EMPTY_ENHANCEMENT' if explicit_empty else 'ARITHMETIC_BITS_PARSED',bits=a[13:13+L].copy(),declared_length=L)
def proof_catalogue():
 """All fallback endpoints paid, at one explicit source-proof layout.

 N_B=N_E=478,q=6,r=5/6,k_B=k_E=2390. The information syntax tests and
 independent source proofs do not qualify this layout's wireless success.
 Offline feasibility candidates do not silently receive these profile IDs.
 """
 profiles={};i=0
 for j in (4,5,6,7):
  for m,K in ((j,0),*(x for x in ENDS if OFFSETS[x[0]]+x[1]>OFFSETS[j])):
   profiles[i]=dict(profile_id=i,j=j,m=m,K=K,explicit_empty_enhancement=(m==j and K==0),null_profile=False,
       k_B=2390,k_E=2390,N_B=478,N_E=478,q_B=6,q_E=6,nominal_rate_B='5/6',nominal_rate_E='5/6',
       length_bits_per_group=13,crc_bits_per_group=16,header_symbols=68,body_symbols=956,N=1024,nominal_Es=2,
       status='SOURCE_PROOF_DIRECTORY_NOT_WIRELESS_QUALIFICATION')
   i+=1
 profiles[i]=dict(profile_id=i,null_profile=True,N=1024,header_symbols=68,body_symbols=956,nominal_Es=2,status='PAID_NULL_PROFILE_KNOWN_FILLER')
 assert len(profiles)==73 and max(profiles)<4096
 return profiles
def lookup(received_header_bits,catalogue):
 try:a=binary(received_header_bits)
 except AssertionError:return None
 if len(a)!=12:return None
 return catalogue.get(bitsint(a))
def choose_fallback(base_lengths,enhancement_lengths,j_target,total_target,C_B,C_E):
 assert j_target in (6,7) and total_target in ENDS and OFFSETS[total_target[0]]+total_target[1]>OFFSETS[j_target]
 j=next((x for x in range(j_target,3,-1) if base_lengths[x]<=C_B),None)
 if j is None:return dict(status='TX_SOURCE_UNENCODABLE',null_profile=True,j=None,m=None,K=None)
 ceiling=OFFSETS[total_target[0]]+total_target[1]
 ends=sorted((x for x in ENDS if OFFSETS[j]<OFFSETS[x[0]]+x[1]<=ceiling),key=lambda x:OFFSETS[x[0]]+x[1],reverse=True)
 end=next((x for x in ends if enhancement_lengths[(j,*x)]<=C_E),(j,0))
 return dict(status='BASE_ONLY' if end==(j,0) else 'TWO_NONEMPTY_GROUPS',null_profile=False,j=j,m=end[0],K=end[1])
