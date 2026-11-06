"""Original arithmetic primitives with a new, checkpoint-bound model context.

12-bit endpoint profile; body length13, terminated stream, CRC16. No new PHY
claim: training uses the clean recovered body. The 68-symbol protected header
and existing LDPC/QAM are unchanged and not rerun in this training tranche.
"""
import binascii
import numpy as np
import torch
from runtime import R,SIZES,setup
setup()
from t1_entropy_core import load_integer_codec
ENT,Encoder,Decoder=load_integer_codec(R)
def uint_bits(n,width):
    assert 0<=n<2**width
    return np.array([(n>>k)&1 for k in reversed(range(width))],dtype=np.uint8)
def integer(a):
    n=0
    for b in a:n=2*n+int(b)
    return n
def crc(a):return binascii.crc_hqx(np.packbits(a).tobytes(),0xffff)
def pack(n,bits):
    bits=np.asarray(bits,dtype=np.uint8);assert n==0 or 30<=n<=680
    profile=0 if n==0 else n-29
    body=np.concatenate([uint_bits(len(bits),13),bits])
    return np.concatenate([uint_bits(profile,12),body,uint_bits(crc(body),16)])
def unpack(packet):
    a=np.asarray(packet);assert a.ndim==1 and np.all((a==0)|(a==1)) and len(a)>=43
    n=integer(a[:12]);assert n<=651;n=0 if n==0 else n+29
    length=integer(a[12:25]);assert len(a)==41+length
    assert crc(a[12:-16])==integer(a[-16:]),'CRC failure'
    if n==0:assert np.array_equal(a[25:-16],[0,0])
    return n,a[25:-16]
@torch.no_grad()
def encode_fit(model,ids,capacity):
    assert len(ids)==10 and ids[0].shape[0]==1
    pr=model.Prior(model.vae,model.prior);enc=Encoder();best_n=0;best=np.array([0,0],dtype=np.uint8);count=0;previous=0;cdf_hash=__import__('hashlib').sha256()
    try:
        for s,pn in enumerate(SIZES):
            table=ENT.probability_cdf(torch.log_softmax(pr.logits(s),-1)[0].cpu().numpy())
            cdf_hash.update(table.tobytes());q=ids[s][0].cpu().numpy()
            for j,tok in enumerate(q):
                enc.encode(np.array([tok]),table[j:j+1]);count+=1
                length=len(enc.bits)+enc.pending+2
                assert length>=previous;previous=length
                if count>=30 and length<=capacity:best=enc.finish().copy();best_n=count;assert len(best)==length
            pr.advance(q,s)
        return pack(best_n,best),dict(tokens=best_n,arithmetic_bits=len(best),full_arithmetic_bits=previous,source_capacity=capacity,body_information_bits=len(best)+29,null=best_n==0,tx_cdf_sha256=cdf_hash.hexdigest())
    finally:pr.close()
@torch.no_grad()
def decode(model,packet):
    n,bits=unpack(packet)
    if n==0:return np.empty(0,dtype=np.int64)
    pr=model.Prior(model.vae,model.prior);dec=Decoder(bits);canonical=Encoder();values=[]
    try:
        for s,pn in enumerate(SIZES):
            count=min(n-len(values),pn*pn)
            if count<=0:break
            table=ENT.probability_cdf(torch.log_softmax(pr.logits(s),-1)[0].cpu().numpy())[:count]
            q=dec.decode(table);canonical.encode(q,table);values.extend(q.tolist())
            if count<pn*pn:break
            pr.advance(q,s)
        assert len(values)==n and np.array_equal(canonical.finish(),bits),'Noncanonical stream / numerical CDF mismatch'
        return np.array(values,dtype=np.int64)
    finally:pr.close()
