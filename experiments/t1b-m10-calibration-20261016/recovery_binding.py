"""Verify closed zero-model initialization failure before its narrow recovery."""
from tb_common import *
REG='6186a4c9c01daf9101cf0ac8478152a1a7d1c60978a3b4465213b04201307a63'
def verify_recovery(expected):
 p=O/'runtime_import_recovery_v1/registration.json';assert sha(p)==expected
 x=read(p);assert x['original_registration_sha256']==REG and x['new_PHY_calls']==0 and x['automatic_retry']is False
 registration(REG)
 for d in x['code']+x['preserved_failure_evidence']+[x['PHY_completion'],x['qualification']]:checked(d)
 for ident in x['retired_processes']:assert gone(ident)
 assert not(T/'STOP').exists()
 return x
