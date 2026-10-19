"""Paid two-fragment transport. No model, channel, or target-truth access."""
import hashlib, struct, zlib

VERSION=1
K=4780
N=5736
Q=6
META_BITS=96
FRAGMENT_BYTES=(K-13-16-META_BITS)//8
MAX_BYTES=2*FRAGMENT_BYTES
SESSION='NEST_ARPC_N2048_EQUAL_PROTECTION_V1'
ENDPOINTS={0:10,1:7}

class Reject(ValueError): pass

def integer_bits(v,n):
    if type(v) is not int or not 0<=v<1<<n:raise Reject('integer range')
    return [(v>>i)&1 for i in range(n-1,-1,-1)]

def bits_integer(bits):
    v=0
    for b in bits:
        if b not in (0,1):raise Reject('non-binary')
        v=(v<<1)|int(b)
    return v

def pack_bits(bits):
    bits=list(bits);length=len(bits);bits_integer(bits)
    return bytes(bits_integer((bits+[0]*7)[i:i+8]) for i in range(0,length,8))

def unpack_bits(data):return [b for x in data for b in integer_bits(x,8)]

def validate_object(data,length,codec):
    if codec not in ENDPOINTS or type(length) is not int or not 0<length<16384:raise Reject('object metadata')
    if len(data)!=(length+7)//8 or len(data)>MAX_BYTES:raise Reject('object size')
    if codec==1 and length%8:raise Reject('ARPC alignment')
    if length%8 and data[-1]&((1<<(8-length%8))-1):raise Reject('nonzero final pad')

def object_crc(data,length,codec):
    return zlib.crc32(b'N2048PAIR1'+struct.pack('>BBH',codec,ENDPOINTS[codec],length)+data)&0xffffffff

def profiles():
    rows=[]
    for codec,endpoint in ENDPOINTS.items():
        for index in (0,1):rows.append(dict(profile_id=2*codec+index,codec=codec,endpoint=endpoint,index=index,mode='data',q=Q,k=K,n=N))
    for codec,endpoint in ENDPOINTS.items():rows.append(dict(profile_id=4+codec,codec=codec,endpoint=endpoint,index=None,mode='null',q=Q,k=K,n=N))
    return {str(p['profile_id']):p for p in rows}

def fragment(data,length,codec,pair_id):
    validate_object(data,length,codec);integer_bits(pair_id,32)
    split=(len(data)+1)//2;parts=(data[:split],data[split:]);crc=object_crc(data,length,codec)
    result=[]
    for i,part in enumerate(parts):
        meta=[]
        for v,n in [(VERSION,4),(codec,2),(ENDPOINTS[codec],4),(i,1),(1,1),(pair_id,32),(length,14),(0,6),(crc,32)]:meta+=integer_bits(v,n)
        assert len(meta)==96 and len(part)<=FRAGMENT_BYTES
        result.append(bytes(pack_bits(meta))+part)
    return result

def crc16(bits):
    crc=65535
    for bit in bits:
        top=((crc>>15)&1)^int(bit);crc=(crc<<1)&65535
        if top:crc^=0x1021
    return crc

def pack_body(payload,profile):
    bits=unpack_bits(payload)
    if profile['mode']=='null':
        if bits:raise Reject('null contains data')
    elif not META_BITS<len(bits)<=K-29:raise Reject('fragment capacity')
    protected=integer_bits(len(bits),13)+bits+[0]*(K-29-len(bits))
    return protected+integer_bits(crc16(protected),16)

def parse_body(bits,profile):
    bits=list(map(int,bits))
    if len(bits)!=K:raise Reject('decoded size')
    bits_integer(bits)
    if crc16(bits[:-16])!=bits_integer(bits[-16:]):raise Reject('CRC16')
    length=bits_integer(bits[:13])
    if length>K-29 or length%8 or any(bits[13+length:-16]):raise Reject('length/padding')
    if profile['mode']=='null':
        if length:raise Reject('null length')
        return b''
    if length<=96:raise Reject('empty fragment')
    return pack_bits(bits[13:13+length])

def read_fragment(payload,profile):
    if profile['mode']!='data' or not 12<len(payload)<=12+FRAGMENT_BYTES:raise Reject('fragment size/mode')
    bits=unpack_bits(payload[:12]);values=[];pos=0
    for width in (4,2,4,1,1,32,14,6,32):values.append(bits_integer(bits[pos:pos+width]));pos+=width
    version,codec,endpoint,index,count,pair_id,length,reserved,crc=values
    if (version,count,reserved)!=(VERSION,1,0):raise Reject('version/count/reserved')
    if (codec,endpoint,index)!=(profile['codec'],profile['endpoint'],profile['index']):raise Reject('profile metadata')
    return dict(codec=codec,endpoint=endpoint,index=index,pair_id=pair_id,length=length,crc=crc,data=payload[12:])

def reassemble(received):
    if len(received)!=2:raise Reject('two complete fragments required')
    fs=[read_fragment(payload,p) for payload,p in received]
    if {f['index'] for f in fs}!={0,1}:raise Reject('duplicate index')
    fs.sort(key=lambda f:f['index']);a,b=fs
    if any(a[k]!=b[k] for k in ('codec','endpoint','pair_id','length','crc')):raise Reject('pair metadata mismatch')
    data=a['data']+b['data'];validate_object(data,a['length'],a['codec'])
    if len(a['data'])!=(len(data)+1)//2 or len(b['data'])!=len(data)//2:raise Reject('unequal prescribed split')
    if object_crc(data,a['length'],a['codec'])!=a['crc']:raise Reject('whole CRC32')
    return dict(codec=a['codec'],endpoint=a['endpoint'],pair_id=a['pair_id'],bit_length=a['length'],data=data)

def input_key(codec,length,data):
    validate_object(data,length,codec)
    return hashlib.sha256(struct.pack('>BH',codec,length)+data).hexdigest()
