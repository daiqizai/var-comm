"""Two unchanged single956 LDPC blocks, new paid transport protocol only."""
import sys,math,hashlib
import numpy as np
from shared import R,OLD_E
import pair_protocol as p
sys.path.insert(0,str(OLD_E/'phy_v1'))
import link as legacy
ep=legacy.ep
class Runtime:
 def __init__(self):
  prior=R/'experiments/prior-aware-grouped-mcs-v1';raw=R/'experiments/main-raw64-20261007'
  ep.load_exact(prior/'uep_common.py','uep_common');ep.load_exact(prior/'profiles.py','profiles')
  old=ep.load_exact(prior/'ldpc_backend.py','_pair_original_ldpc')
  self.legacy=ep.load_exact(prior/'uep_phy.py','uep_phy')
  ep.load_exact(raw/'main_raw64_plan_only.py','main_raw64_plan_only')
  self.mapping=ep.load_exact(raw/'main_raw64_packet_adapter.py','main_raw64_packet_adapter')
  ref=R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json';assert ep.sha(ref)==ep.LEGACY_QUALIFICATION_SHA
  self.backend=ep.Backend(old.SionnaBackend('cpu',str(ref)));self.torch=self.backend.torch;t=self.torch
  t.set_num_threads(2);t.set_num_interop_threads(1);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
  t.backends.cudnn.benchmark=False;t.use_deterministic_algorithms(True)
  self.header=self.legacy.Header(R);self.profiles=p.profiles();layout=self.backend.plan(p.K,p.N,p.Q)
  self.catalogue=dict(protocol=p.SESSION,profiles=self.profiles,layout=layout,backend_identity=self.backend.identity)
 def transmit(self,payload,pid,counter):
  profile=self.profiles[str(pid)];info=np.asarray(p.pack_body(payload,profile),np.uint8);b=self.backend;t=self.torch
  encoded=b.encode(info[None],p.N,p.Q)
  mask=t.as_tensor(ep.scramble_mask(p.N,counter,p.SESSION)[None],device='cpu');coded=(encoded+mask)%2
  body=ep.as_numpy(self.mapping.modulate(coded,p.Q,t,self.legacy))[0].astype(np.float32)
  head=np.asarray(self.header.transmit(pid),np.float64);assert body.shape==(956,2) and head.shape==(68,2)
  frame=np.concatenate([head,body.astype(np.float64)])
  return frame,dict(profile_id=pid,counter=counter,information_sha256=ep.array_sha(info),frame_sha256=ep.array_sha(frame),energy=float(np.square(frame).sum()),N=1024,Es=2)
 def receive(self,y,snr,counter,ledger,key):
  y=np.asarray(y,np.float64);assert y.shape==(1024,2) and np.isfinite(y).all() and math.isfinite(snr)
  h=ledger.call('header',key,lambda:self.header.receive(y[:68],snr,self.profiles))
  if not h['header_ok']:return dict(header=h,status='HEADER_REJECT',profile=None,body=None)
  profile=self.profiles[str(h['profile_id'])]
  def decode():
   t=self.torch;x=t.as_tensor(y[68:][None],dtype=t.float32,device='cpu');logits=self.mapping.demap(x,snr,p.Q,t,self.legacy)
   mask=t.as_tensor(ep.scramble_mask(p.N,counter,p.SESSION)[None],device='cpu');logits=logits*(1-2*mask.to(logits.dtype))
   hard=ep.as_numpy(self.backend.decode(logits,p.K,p.N,p.Q))[0].astype(np.uint8)
   result=dict(hard_bits_sha256=ep.array_sha(hard),decoded_bits=hard.tolist())
   try:payload=p.parse_body(hard,profile);result.update(status='NULL_PROFILE' if profile['mode']=='null' else 'FRAGMENT_PARSED',payload_hex=payload.hex())
   except p.Reject as e:result.update(status='BODY_REJECT',reason=str(e))
   return result
  body=ledger.call('body',key,decode);return dict(header=h,profile=profile,body=body,status=body['status'])
def noise(source_id,seed):
 key=hashlib.sha256((p.SESSION+'|'+source_id+'|'+str(seed)).encode()).digest()
 return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16],'big'))).standard_normal((2048,2)).astype(np.float64)
