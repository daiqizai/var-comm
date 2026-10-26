"""Original main-paper model loader, tokenizer, encoder and RX functions."""
from kn_common import *
import copy,platform,types
def load(routes,ledger,boundary,out,pci):
 ns=dict(Path=Path,R=R,os=os,sys=sys,types=types,platform=platform,copy=copy,hashlib=hashlib,time=time,SIZES=SIZES,OFFSETS=OFFSETS,MODELS=MODELS,definitions=definitions,checked=checked,save=save)
 definitions(R/'experiments/t1b-m10-calibration-20261016/tb_native.py',{'build'},ns)
 return ns['build'](routes,ledger,boundary,out,pci)
def source_codec(native):
 import numpy as np
 from ep_source_codec import PartialSourceCodec,WHOLE_FAMILY
 from t1_entropy_core import source_tokens
 from t6_source_codec import SourceCodec10
 ns=dict(R=R,OFFSETS=OFFSETS,ENDS=ENDS,asha=asha,np=np,PartialSourceCodec=PartialSourceCodec,WHOLE_FAMILY=WHOLE_FAMILY,source_tokens=source_tokens,SourceCodec10=SourceCodec10)
 definitions(R/'experiments/submission-closure-20261016/m10_codec.py',{'AuditCodec'},ns)
 return ns['AuditCodec'](R,native=native)
def receiver_codec(native):
 from ep_source_codec import PartialSourceCodec
 from t6_source_codec import SourceCodec10
 ns=dict(PartialSourceCodec=PartialSourceCodec,SourceCodec10=SourceCodec10)
 definitions(R/'experiments/t1c-m10-test-20261016/tc_receiver.py',{'ReceiverCodec'},ns)
 return ns['ReceiverCodec'](R,native=native)
def tokenizer(native,pixels):
 import numpy as np
 def require(ok,msg):
  if not ok:raise ValueError(msg)
 ns=dict(np=np,require=require)
 definitions(R/'experiments/content-real-64qam-20261006/h_online_components.py',{'native_encoder'},ns)
 with native.torch.no_grad():return ns['native_encoder'](native,pixels)
def choose(streams,family,snr):
 ns=dict(SIZES=SIZES)
 definitions(R/'experiments/n1024-consistency-20261019/audit_v2.py',{'nt','choose'},ns)
 return ns['choose'](streams,family,snr)
