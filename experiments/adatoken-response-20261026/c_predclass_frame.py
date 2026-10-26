"""C-only class container over the unchanged, already qualified LDPC/QAM math."""
from common import *
from fixed_frame import catalogue as base_catalogue,isolated_engine,CANDIDATES,LLAMA,NULL
from c_predclass_protocol import source_capacity,container,split_container
import numpy as np
FAMILY='ADA_PREDICTED_CLASS_PREFIX'
def catalogue(old):
 c=base_catalogue(old,'C');profiles=[]
 for p0 in c['profiles']:
  p=dict(p0)
  if p['family']==LLAMA:
   p.pop('profile_key');p.update(family=FAMILY,source_syntax='PAID_CLASS10_CONDITIONAL_ARITHMETIC_PREFIX_V1',class_bits=10,arithmetic_capacity_bits=source_capacity(p['k']),length_semantics='class_plus_finished_arithmetic_bits',generation='CFG4_T1_K2000_P1_FP32');p['profile_key']=digest(p)
  profiles.append(p)
 assert profiles[:360]==old['profiles'];c.pop('catalogue_sha256');c.update(protocol='ADA_PREDICTED_CLASS_FIXEDFRAME_V1',profiles=profiles);c['catalogue_sha256']=digest(c);return c
def runtime(reg):
 phy=isolated_engine();ns=phy.pack_body.__globals__;assert ns['FAMILIES'][-2:]==(LLAMA,NULL);ns['FAMILIES']=(*ns['FAMILIES'][:-2],FAMILY,NULL)
 rt=phy.create_runtime(R,str(checked(reg['phy_qualification'])));cat=catalogue(rt.catalogue);assert cat==cr(reg['catalogue']);rt.catalogue=cat;rt.profiles={str(p['profile_id']):p for p in cat['profiles']};return phy,rt
def choose(streams,profiles,q,rate,predicted_class):
 available=[p for p in profiles if p['q']==q and p['nominal_rate']==rate];cap=source_capacity(available[0]['k']);checks=[]
 for count in range(256,0,-1):
  bits=streams['p%03d'%count];fits=len(bits)<=cap;checks.append(dict(prefix=count,arithmetic_bits=len(bits),class_bits=10,container_bits=10+len(bits),fits=bool(fits)))
  if fits:
   p=next(p for p in available if p['family']==FAMILY and p['prefix_count']==count);return p,container(predicted_class,bits),checks
 p=next(p for p in available if p['family']==NULL);return p,np.array([0,0],np.uint8),checks
def source_projection(rx):
 p=rx['rx_profile'];b=rx['body']
 if not rx['header']['header_ok']or b is None or not b['crc_accepted']or not b['parser_accepted']:return dict(status='FIXED_GRAY',reason=rx['status'])
 if p['family']==NULL:return dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE')
 if p['family']!=FAMILY:return dict(status='FIXED_GRAY',reason='UNSUPPORTED_SOURCE_FAMILY')
 try:class_id,bits=split_container(np.array(b['payload'],np.uint8))
 except ValueError as e:return dict(status='FIXED_GRAY',reason='CLASS_CONTAINER_REJECT',detail=str(e))
 count=p['m']
 if not 1<=count<=256 or p['K']!=0:return dict(status='FIXED_GRAY',reason='INVALID_PREFIX_PROFILE')
 return dict(status='SOURCE_INPUT',prefix=count,received_class=class_id,bits=bits.tolist(),payload_sha256=asha(bits))
