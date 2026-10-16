"""Exact CUDA UUID spelling compatibility; no device or numeric substitution."""
from tb_common import *
import uuid
def cuda_uuid_equal(observed,registered):
 def parse(v):
  assert isinstance(v,str)
  return uuid.UUID(v.removeprefix('GPU-'))
 return parse(observed)==parse(registered)
def verify_metric_recovery(expected):
 p=O/'metric_device_recovery_v2/registration.json';assert sha(p)==expected
 x=read(p);registration(x['original_registration_sha256'])
 for d in x['code']+x['evidence']:checked(d)
 for identity in x['retired_processes']:assert gone(identity)
 assert x['new_PHY_calls']==x['new_RX_calls']==0 and x['automatic_retry']is False
 assert not(T/'STOP').exists()
 return x
