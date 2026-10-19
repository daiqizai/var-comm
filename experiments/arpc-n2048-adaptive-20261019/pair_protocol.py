"""Paid adaptive equal-protection fragments. Pure byte/bit transformations.

The public profile carries actual endpoint and MCS; no TX target or source truth
is an RX argument. IDs0..5 delegate to the exact original version1 codec.
"""
from pathlib import Path
import hashlib,importlib.util,struct,zlib
from adaptive_plan import NEST_ENDPOINTS,layouts,nest_family
OLD_PATH=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-n2048-validation-20261019/pair_protocol.py')
VERSION=2
SESSION='NEST_ARPC_N2048_ADAPTIVE_EQUAL_PROTECTION_V2'
NOISE_NAMESPACE='NEST_ARPC_N2048_EQUAL_PROTECTION_V1'
META_BITS=96
_old=None
class Reject(ValueError):pass
def legacy():
 global _old
 if _old is None:
  p=OLD_PATH
  if not p.exists():p=Path(__file__).parents[1]/'arpc_n2048_execution_20261019/pair_protocol.py'
  spec=importlib.util.spec_from_file_location('_n2048_original_protocol',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);_old=m
 return _old
def integer_bits(v,n):
 if type(v)is not int or not 0<=v<1<<n:raise Reject('integer range')
 return [(v>>i)&1 for i in range(n-1,-1,-1)]
def bits_integer(bits):
 v=0
 for b in bits:
  if b not in (0,1):raise Reject('non-binary')
  v=(v<<1)|int(b)
 return v
def pack_bits(bits):
 bits=list(bits);bits_integer(bits)
 return bytes(bits_integer((bits+[0]*7)[i:i+8]) for i in range(0,len(bits),8))
def unpack_bits(data):return [b for x in data for b in integer_bits(x,8)]
def profiles():
 out={k:dict(v) for k,v in legacy().profiles().items()}
 for codec in (0,1):
  for endpoint in (range(25) if codec==0 else range(1,8)):
   for g in layouts():
    for index in (0,1):
     pid=len(out);row=dict(profile_id=pid,version=VERSION,codec=codec,endpoint=endpoint,index=index,mode='data',q=g['q'],k=g['k'],n=g['n'],layout_id=g['layout_id'])
     if codec==0:row.update(m=NEST_ENDPOINTS[endpoint][0],K=NEST_ENDPOINTS[endpoint][1],family=nest_family(endpoint))
     out[str(pid)]=row
 for codec in (0,1):
  for g in layouts():
   pid=len(out);out[str(pid)]=dict(profile_id=pid,version=VERSION,codec=codec,endpoint=None,index=None,mode='null',q=g['q'],k=g['k'],n=g['n'],layout_id=g['layout_id'])
 assert len(out)==798 and max(map(int,out))<4096
 return out
_directory=None
def directory():
 global _directory
 if _directory is None:_directory=profiles()
 return _directory
def checked_profile(p):
 if not isinstance(p,dict) or p!=directory().get(str(p.get('profile_id'))):raise Reject('unknown or modified public profile')
 return p
def profile(codec,endpoint,layout_id,index):
 rows=[p for p in directory().values() if p.get('version')==VERSION and p['codec']==codec and p['endpoint']==endpoint and p['layout_id']==layout_id and p['index']==index]
 if len(rows)!=1:raise Reject('profile lookup')
 return rows[0]
def fragment_capacity(p):return (p['k']-125)//8
def validate_object(data,length,codec,endpoint,capacity=None):
 if not isinstance(data,bytes) or type(length)is not int or not 0<length<16384:raise Reject('object type/length')
 if codec==0:
  if type(endpoint)is not int or not 0<=endpoint<25:raise Reject('NeST endpoint')
 elif codec==1:
  if type(endpoint)is not int or not 1<=endpoint<=7 or length%8:raise Reject('ARPC endpoint/alignment')
 else:raise Reject('codec')
 if len(data)!=(length+7)//8 or len(data)<2:raise Reject('object size')
 if capacity is not None and len(data)>capacity:raise Reject('object capacity')
 if length%8 and data[-1]&((1<<(8-length%8))-1):raise Reject('nonzero object pad')
def object_crc(data,length,codec,endpoint):
 return zlib.crc32(b'N2048PAIR2'+struct.pack('>BBH',codec,endpoint,length)+data)&0xffffffff
def fragment(data,length,codec,endpoint,pair_id,layout_id):
 p0=profile(codec,endpoint,layout_id,0);cap=fragment_capacity(p0)
 validate_object(data,length,codec,endpoint,2*cap);integer_bits(pair_id,32)
 split=(len(data)+1)//2;crc=object_crc(data,length,codec,endpoint);out=[]
 for index,part in enumerate((data[:split],data[split:])):
  fields=((VERSION,4),(codec,2),(endpoint,5),(index,1),(1,1),(pair_id,32),(length,14),(0,5),(crc,32))
  meta=[b for v,n in fields for b in integer_bits(v,n)];assert len(meta)==96 and 0<len(part)<=cap
  out.append((pack_bits(meta)+part,profile(codec,endpoint,layout_id,index)))
 return out
def crc16(bits):
 crc=65535
 for bit in bits:
  top=((crc>>15)&1)^int(bit);crc=(crc<<1)&65535
  if top:crc^=0x1021
 return crc
def pack_body(payload,p):
 checked_profile(p)
 if p['profile_id']<6:
  try:return legacy().pack_body(payload,p)
  except legacy().Reject as e:raise Reject(str(e)) from e
 bits=unpack_bits(payload);k=p['k']
 if p['mode']=='null':
  if bits:raise Reject('null contains data')
 elif not 96<len(bits)<=k-29:raise Reject('fragment capacity')
 protected=integer_bits(len(bits),13)+bits+[0]*(k-29-len(bits))
 return protected+integer_bits(crc16(protected),16)
def parse_body(bits,p):
 checked_profile(p)
 if p['profile_id']<6:
  try:return legacy().parse_body(bits,p)
  except legacy().Reject as e:raise Reject(str(e)) from e
 bits=list(map(int,bits));k=p['k']
 if len(bits)!=k:raise Reject('decoded size')
 bits_integer(bits)
 if crc16(bits[:-16])!=bits_integer(bits[-16:]):raise Reject('CRC16')
 length=bits_integer(bits[:13])
 if length>k-29 or length%8 or any(bits[13+length:-16]):raise Reject('length/padding')
 if p['mode']=='null':
  if length:raise Reject('null length')
  return b''
 if length<=96:raise Reject('empty fragment')
 return pack_bits(bits[13:13+length])
def read_fragment(payload,p):
 checked_profile(p)
 if p['profile_id']<6:raise Reject('version1 handled only as complete original pair')
 if p['mode']!='data' or not 12<len(payload)<=12+fragment_capacity(p):raise Reject('fragment size/mode')
 bits=unpack_bits(payload[:12]);pos=0;values=[]
 for width in (4,2,5,1,1,32,14,5,32):values.append(bits_integer(bits[pos:pos+width]));pos+=width
 version,codec,endpoint,index,count,pair_id,length,reserved,crc=values
 if (version,count,reserved)!=(VERSION,1,0):raise Reject('version/count/reserved')
 if (codec,endpoint,index)!=(p['codec'],p['endpoint'],p['index']):raise Reject('profile metadata')
 return dict(version=version,codec=codec,endpoint=endpoint,index=index,pair_id=pair_id,bit_length=length,crc=crc,data=payload[12:],q=p['q'],k=p['k'],n=p['n'])
def reassemble(received):
 if len(received)!=2:raise Reject('two complete fragments required')
 for _,p in received:checked_profile(p)
 if all(p['profile_id']<6 for _,p in received):
  try:v=legacy().reassemble(received)
  except legacy().Reject as e:raise Reject(str(e)) from e
  v['version']=1;v['endpoint']=24 if v['codec']==0 else v['endpoint'];return v
 if any(p['profile_id']<6 for _,p in received):raise Reject('mixed versions')
 fs=[read_fragment(data,p) for data,p in received]
 if {f['index'] for f in fs}!={0,1}:raise Reject('duplicate fragment')
 fs.sort(key=lambda f:f['index']);a,b=fs
 if any(a[k]!=b[k] for k in ('version','codec','endpoint','pair_id','bit_length','crc','q','k','n')):raise Reject('pair metadata mismatch')
 data=a['data']+b['data'];validate_object(data,a['bit_length'],a['codec'],a['endpoint'],2*((a['k']-125)//8))
 if len(a['data'])!=(len(data)+1)//2 or len(b['data'])!=len(data)//2:raise Reject('prescribed split')
 if object_crc(data,a['bit_length'],a['codec'],a['endpoint'])!=a['crc']:raise Reject('whole CRC32')
 return {k:a[k] for k in ('version','codec','endpoint','pair_id','bit_length','q','k','n')}|dict(data=data)
def input_key(codec,endpoint,length,data):
 validate_object(data,length,codec,endpoint)
 return hashlib.sha256(struct.pack('>BBH',codec,endpoint,length)+data).hexdigest()
def scrambling_session(p):return NOISE_NAMESPACE if p['profile_id']<6 else SESSION
