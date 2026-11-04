"""New paid dynamic-MCS frame using existing pinned LDPC/QAM primitives.

No NeST profile catalogue or 68/29-bit accounting is reused. The protected
control header identifies MCS and packet-byte length; endpoint is paid inside
SourcePacket. Ideal shared CSI, synchronized public frame counter, one frame,
no retransmission, no hidden transmitter truth in receive().
"""
from pathlib import Path
from functools import lru_cache
from fractions import Fraction
import hashlib,importlib.util,json,sys
import numpy as np
from packet import SourcePacket

MCS=tuple((q,r) for q in (2,4,6) for r in ('1/2','2/3','3/4','5/6'))
VERSION=1
HEADER_INFORMATION_BITS=38 # version2 + MCS4 + source-byte-length16 + CRC16
HEADER_CODE_BITS=128
HEADER_SYMBOLS=64
PROTOCOL='RDVQ_DYNAMIC_FRAME_V1'

def uintbits(n,width):
    assert 0<=n<2**width;return ((n>>np.arange(width-1,-1,-1))&1).astype(np.uint8)
def asint(b):return sum(int(x)<<(len(b)-1-i) for i,x in enumerate(b))
def crc16(b):
    x=65535
    for v in b:
        feedback=(x>>15)^int(v);x=((x<<1)^(0x1021 if feedback else 0))&65535
    return x
def mask(n,counter,part):
    seed=int.from_bytes(hashlib.sha256(json.dumps([PROTOCOL,counter,part],separators=(',',':')).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed)).integers(0,2,n,dtype=np.uint8)
def noise(N,source_id,seed):
    key=int.from_bytes(hashlib.sha256(json.dumps([PROTOCOL,N,str(source_id),seed],separators=(',',':')).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(key)).standard_normal((N,2))

class PHY:
    def __init__(self,repository):
        R=Path(repository);p=R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py';sys.path.insert(0,str(p.parent))
        spec=importlib.util.spec_from_file_location('_rdvq_pinned_phy_dependencies',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rt=module.Runtime(R);self.rt=rt;self.torch=rt.torch;self.E=rt.backend.Encoder;self.D=rt.backend.Decoder;self.decoder=rt.backend.decoder
    @lru_cache(maxsize=40)
    def codecs(self,k,n,q):
        e=self.E(k,n,num_bits_per_symbol=q,precision='single',device='cpu')
        assert e.k==k and e.n==n and e.k_ldpc-e.k_filler==k and n<=e.n_cb_comp
        d=self.D(e,**{k:v for k,v in self.decoder.items() if k!='early_stop'},device='cpu');e.eval();d.eval()
        return e,d
    def layout(self,N,mcs):
        assert N in (512,1024,2048) and 0<=mcs<len(MCS);q,r=MCS[mcs];n=(N-HEADER_SYMBOLS)*q;k=int(Fraction(r)*n);e,_=self.codecs(k,n,q)
        return dict(N=N,mcs=mcs,q=q,rate=r,k=k,n=n,body_symbols=N-HEADER_SYMBOLS,header_symbols=HEADER_SYMBOLS,source_capacity_bits=(k//8)*8,
            header_information_bits=HEADER_INFORMATION_BITS,header_code_bits=HEADER_CODE_BITS,body_extra_crc_bits=0,source_wrapper_bits=72,k_filler=int(e.k_filler),bg=str(e._bg),z=int(e.z),mother_bits=int(e.n_ldpc),n_cb_comp=int(e.n_cb_comp),repetition_bits=0)
    def coded(self,info,n,q,counter,part):
        t=self.torch;e,_=self.codecs(len(info),n,q)
        with t.inference_mode():code=e(t.as_tensor(info[None],dtype=t.float32))
        scrambled=(code+t.as_tensor(mask(n,counter,part)[None]))%2
        return self.rt.mapping.modulate(scrambled,q,t,self.rt.legacy)[0].numpy().astype(np.float64)
    def decoded(self,observed,k,q,snr,counter,part):
        t=self.torch;n=len(observed)*q;_,d=self.codecs(k,n,q)
        llr=self.rt.mapping.demap(t.as_tensor(observed[None],dtype=t.float32),float(snr),q,t,self.rt.legacy)
        llr*=1-2*t.as_tensor(mask(n,counter,part)[None],dtype=t.float32)
        with t.inference_mode():out=d(llr)[0].numpy().astype(np.uint8)
        assert out.shape==(k,);return out
    def transmit(self,wire,N,mcs,counter):
        SourcePacket.from_bytes(wire);p=self.layout(N,mcs);assert len(wire)*8<=p['k'] and len(wire)<65536
        fields=np.concatenate([uintbits(VERSION,2),uintbits(mcs,4),uintbits(len(wire),16)]);header=np.concatenate([fields,uintbits(crc16(fields),16)])
        assert len(header)==38
        body=np.zeros(p['k'],np.uint8);body[:len(wire)*8]=np.unpackbits(np.frombuffer(wire,dtype=np.uint8))
        tx=np.concatenate([self.coded(header,128,2,counter,'header'),self.coded(body,p['n'],p['q'],counter,'body')]);assert tx.shape==(N,2)
        energy=float(np.square(tx).sum())
        return tx,dict(**p,packet_bits=len(wire)*8,known_padding_bits=p['k']-len(wire)*8,frame_energy=energy,nominal_Es=2,rho=energy/(2*N))
    def receive(self,observed,snr,counter):
        # All dispatch uses decoded header fields; caller cannot supply TX MCS.
        y=np.asarray(observed,np.float64);N=len(y);assert y.shape==(N,2) and N in (512,1024,2048)
        h=self.decoded(y[:64],38,2,snr,counter,'header');f=h[:-16]
        if asint(h[-16:])!=crc16(f):return dict(status='HEADER_CRC_REJECT',wire=None)
        version=asint(f[:2]);mcs=asint(f[2:6]);length=asint(f[6:22])
        if version!=VERSION or mcs>=12:return dict(status='HEADER_FIELDS_REJECT',wire=None)
        p=self.layout(N,mcs)
        if not 9<=length<=p['k']//8:return dict(status='LENGTH_REJECT',wire=None)
        b=self.decoded(y[64:],p['k'],p['q'],snr,counter,'body')
        if b[length*8:].any():return dict(status='BODY_PADDING_REJECT',wire=None)
        wire=np.packbits(b[:length*8]).tobytes()
        try:source=SourcePacket.from_bytes(wire)
        except ValueError as e:return dict(status='SOURCE_PACKET_REJECT',wire=None,reason=str(e))
        return dict(status='ACCEPTED_PACKET',wire=wire,received_mcs=mcs,received_slices=source.slices,null=source.slices==0)
