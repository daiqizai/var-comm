"""Real paid fields and CRCs, error-free information transfer without FEC."""
from followup_common import *
from opt_common import definitions
from grid_codec import endpoints

def primitives():
 ns=dict(np=np,array_sha=asha,FAMILIES=('EC_VAR_WHOLE','EC_VAR_PARTIAL','ADA_PREDICTED_CLASS_PREFIX','EXPLICIT_NULL_FIXEDFRAME'))
 definitions(R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py',{'require','binary','integer_bits','bits_integer','crc16','append_crc','crc_ok','pack_body','parse_body'},ns)
 return ns

def catalogue():
 p=[dict(profile_id=i,codec='NEST',m=m,K=k,family='EC_VAR_PARTIAL'if k else'EC_VAR_WHOLE')for i,(m,k)in enumerate(endpoints('token'))]
 p += [dict(profile_id=len(p)+i-1,codec='ADA',m=i,K=0,family='ADA_PREDICTED_CLASS_PREFIX')for i in range(1,257)]
 p += [dict(profile_id=len(p),codec='NULL',m=0,K=0,family='EXPLICIT_NULL_FIXEDFRAME')]
 assert len(p)==908 and len({v['profile_id']for v in p})==908
 return p

def transfer(bits,profile,B,predicted_class=None):
 f=primitives();a=np.asarray(bits,np.uint8);p=dict(profile,mode='arithmetic',k=B-28)
 if p['codec']=='ADA':
  assert type(predicted_class)is int and 0<=predicted_class<1000
  a=np.concatenate((f['integer_bits'](predicted_class,10),a))
 h=f['append_crc'](f['integer_bits'](p['profile_id'],12));body=f['pack_body'](a,p)
 wire=np.concatenate((h,body));assert len(wire)==B and f['crc_ok'](wire[:28])
 rid=f['bits_integer'](wire[:12]);rp=catalogue()[rid];assert rp==profile
 parsed=f['parse_body'](wire[28:],dict(rp,mode='arithmetic',k=B-28));assert parsed['crc_accepted']and parsed['parser_accepted']
 payload=np.array(parsed['payload'],np.uint8)
 if rp['codec']=='NULL':projection=dict(status='FIXED_GRAY',reason='EXPLICIT_NULL_PROFILE')
 elif rp['codec']=='NEST':projection=dict(status='SOURCE_INPUT',codec='NEST',m=rp['m'],K=rp['K'],family=rp['family'],bits=payload.tolist())
 else:
  cls=f['bits_integer'](payload[:10]);assert cls<1000
  projection=dict(status='SOURCE_INPUT',codec='ADA',m=rp['m'],K=0,received_class=cls,bits=payload[10:].tolist())
 return projection,dict(wire_sha256=asha(wire),wire_bits=B,profile_id=rid,header_bits=28,body_bits=B-28,payload_bits=len(a),arithmetic_bits=len(bits),class_bits=10 if rp['codec']=='ADA'else 0,padding_bits=B-57-len(a),header_crc_ok=True,body_crc_ok=True,actual_error_free_parse=True,FEC=False,PHY=False)

def generation_seed(source_id,anchor):
 return int.from_bytes(hashlib.sha256(('NEST_FOLLOWUP_E2_GENERATION_V1|'+source_id+'|'+str(anchor)).encode()).digest()[:8],'little')%(1<<31)
