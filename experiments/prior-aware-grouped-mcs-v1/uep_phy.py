"""CRC, fixed power mapping, and actual LDPC packets; receiver never accepts truth."""
from __future__ import annotations
import math
import numpy as np
from uep_common import require,rng,array_sha

def _crc_table():
    result=[]
    for b in range(256):
        x=b<<8
        for _ in range(8):x=((x<<1)^ (0x1021 if x&0x8000 else 0))&65535
        result.append(x)
    return np.asarray(result,dtype=np.uint16)
CRC_TABLE=_crc_table()

def binary_batch(bits):
    a=np.asarray(bits);one=a.ndim==1
    if one:a=a[None]
    require(a.ndim==2 and np.isin(a,(0,1)).all(),'Binary [batch,bits] required')
    return a.astype(np.uint8),one

def crc16_batch(bits):
    a,_=binary_batch(bits);v=np.full(len(a),65535,dtype=np.uint16);full=a.shape[1]//8
    packed=np.packbits(a[:,:full*8],axis=1,bitorder='big')
    for b in packed.T:v=((v<<8)^CRC_TABLE[(v>>8)^b]).astype(np.uint16)
    for b in a[:,full*8:].T:
        feedback=(v>>15)^b;v=((v<<1)^np.where(feedback,0x1021,0)).astype(np.uint16)
    return v

def append_crc_batch(bits):
    a,one=binary_batch(bits);v=crc16_batch(a);crc=((v[:,None]>>np.arange(15,-1,-1))&1).astype(np.uint8)
    result=np.concatenate((a,crc),axis=1);return result[0] if one else result

def crc_accept_batch(bits):
    a,one=binary_batch(bits);require(a.shape[1]>=16,'CRC missing')
    expected=(a[:,-16:].astype(np.uint32)*(1<<np.arange(15,-1,-1))).sum(1)
    result=crc16_batch(a[:,:-16])==expected;return bool(result[0]) if one else result

def mask_batch(n,counters,group=0,session='body'):
    return np.stack([rng('scramble',session,int(c),int(group)).integers(0,2,int(n),dtype=np.uint8) for c in counters])

def modulate_torch(bits,q):
    import torch
    require(bits.ndim==2 and bits.shape[1]%q==0 and q in (2,4),'Aligned codewords required')
    if q==2:return (1-2*bits).reshape(len(bits),-1,2)
    pam=torch.tensor([-3.,-1.,3.,1.],dtype=bits.dtype,device=bits.device)/math.sqrt(5)
    pairs=bits.reshape(len(bits),-1,2).long()
    return pam[2*pairs[:,:,0]+pairs[:,:,1]].reshape(len(bits),-1,2)

def demap_torch(y,snr,q):
    import torch
    gamma=10**(float(snr)/10)
    # Decoder logits are log P(bit=1)/P(bit=0), unlike the original Viterbi evidence.
    if q==2:return (-2*gamma*y).reshape(len(y),-1)
    pam=torch.tensor([-3.,-1.,3.,1.],dtype=y.dtype,device=y.device)/math.sqrt(5)
    lp=-.5*gamma*(y.reshape(len(y),-1,1)-pam)**2
    a=torch.logsumexp(lp[:,:,[2,3]],-1)-torch.logsumexp(lp[:,:,[0,1]],-1)
    b=torch.logsumexp(lp[:,:,[1,3]],-1)-torch.logsumexp(lp[:,:,[0,2]],-1)
    return torch.stack((a,b),-1).reshape(len(y),-1)

def transmit_packet(backend,payload,n,q,counters,group=0,session='body'):
    a,_=binary_batch(payload);info=append_crc_batch(a)
    require(len(counters)==len(a),'One public frame counter per payload')
    code=backend.encode(info,n,q);mask=backend.torch.as_tensor(mask_batch(n,counters,group,session),device=backend.device)
    wave=modulate_torch((code+mask)%2,q)
    return wave,dict(k=info.shape[1],n=n,q=q,source_bits=a.shape[1],crc_bits=16,tail_bits=0,
        actual_E=wave.square().sum((1,2)).cpu().tolist())

def receive_packet(backend,received,k,n,q,snr,counters,group=0,session='body'):
    # Only public format, noisy symbols and public session counter are accepted here.
    logits=demap_torch(received,snr,q)
    mask=backend.torch.as_tensor(mask_batch(n,counters,group,session),device=backend.device)
    logits=logits*(1-2*mask.to(logits.dtype))
    decoded=backend.decode(logits,k,n,q).cpu().numpy().astype(np.uint8)
    return decoded[:,:-16],np.asarray(crc_accept_batch(decoded)),decoded

def simulate_packets(backend,payload,n,q,snr,counters,group=0,session='body',noise_namespace='BLER'):
    a,_=binary_batch(payload);wave,meta=transmit_packet(backend,a,n,q,counters,group,session)
    # Noise is generated only in the channel; the receiver gets received symbols.
    noise=np.stack([rng(noise_namespace,float(snr),int(c),int(group)).standard_normal((n//q,2)).astype(np.float32) for c in counters])
    noise=backend.torch.as_tensor(noise,device=backend.device)
    received=wave+noise*(10**(-float(snr)/20))
    values,accepted,decoded=receive_packet(backend,received,a.shape[1]+16,n,q,snr,counters,group,session)
    correct=accepted & np.all(values==a,axis=1)
    meta.update(correct=correct,accepted=accepted,rejected=~accepted,undetected=accepted&~correct,
                decoded=values,codeword_correct=np.all(decoded==append_crc_batch(a),axis=1))
    return meta

def continuous_prefix(group_payloads,accepted):
    """Discard first failed group and all later groups, regardless of later CRC."""
    require(len(group_payloads)==len(accepted) and len(accepted) in (1,2),'G=1 or2')
    kept=[]
    for p,ok in zip(group_payloads,accepted):
        if not bool(ok):break
        kept.append(np.asarray(p,dtype=np.uint8))
    return kept

class Header:
    """Original paid 68-use convolutional header carrying one public profile ID."""
    def __init__(self,root):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(root)/'src'))
        from var_comm import scale_channel
        self.s=scale_channel

    def transmit(self,profile_id):
        require(type(profile_id) is int and 0<=profile_id<4096,'12-bit public ID')
        bits=self.s.indices_to_bits([profile_id],12)
        return self.s.encode_packet(bits,68)['symbols']

    def receive(self,y,snr,codebook=None):
        require(np.asarray(y).shape==(68,2),'Exactly 68 received header symbols')
        mapping=self.s.rate_match_indices(68,136)
        evidence=np.bincount(mapping,weights=(np.asarray(y)*10**(float(snr)/10)).ravel(),minlength=68)
        bits,score=self.s.decode_map(evidence)
        crc=bool(self.s.crc_accepts(bits[:-6]));pid=int(self.s.bits_to_indices(bits[:12],12)[0])
        legal=codebook is None or str(pid) in codebook or pid in codebook
        return dict(profile_id=pid if crc and legal else None,header_crc_ok=crc,
                    header_fields_legal=legal,header_ok=bool(crc and legal),score=float(score))
