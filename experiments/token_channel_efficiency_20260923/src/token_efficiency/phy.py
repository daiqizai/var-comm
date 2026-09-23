"""Versioned m6--m10 digital PHY; actual soft16QAM and paid robust header."""
import numpy as np
from var_comm.entropy import validate_bits
from var_comm.scale_channel import (indices_to_bits,bits_to_indices,append_crc,crc_accepts,convolutional_encode,rate_match_indices,decode_map,encode_packet,channel_evidence)
from var_comm.progressive import split_prefix
from var_comm.study import seeded_noise
from .codec import SIZES

PROTOCOL='token-efficiency-digital-v1'
PAM=np.array([-3.,-1.,3.,1.])/np.sqrt(5.) # binary00,01,10,11 -> Gray levels
LABELS=((np.arange(4)[:,None] >> np.array([1,0])) & 1).astype(np.uint8)

def modulate(bits,mcs):
    bits=validate_bits(bits)
    if mcs=='QPSK':
        if len(bits)%2:raise ValueError('QPSK complete symbols only')
        return (1.-2.*bits.astype(float)).reshape(-1,2)
    if mcs!='16QAM' or len(bits)%4:raise ValueError('16QAM complete symbols only')
    pairs=bits.reshape(-1,2);indices=2*pairs[:,0]+pairs[:,1]
    return PAM[indices].reshape(-1,2)

def soft_half_llr(observed,snr,mcs):
    y=np.asarray(observed,dtype=np.float64)
    if y.ndim!=2 or y.shape[1]!=2 or not np.isfinite(y).all() or not np.isfinite(snr):raise ValueError('finite complex observation and SNR required')
    gamma=10.**(float(snr)/10)
    if not np.isfinite(gamma) or gamma<=0:raise ValueError('noise precision')
    if mcs=='QPSK':return (y*gamma).ravel()
    if mcs!='16QAM':raise ValueError('MCS')
    logp=-(y.ravel()[:,None]-PAM[None,:])**2*gamma/2
    # decode_map uses half log(P(bit=0|y)/P(bit=1|y)); preserve full soft mixtures.
    result=np.stack([.5*(np.logaddexp.reduce(logp[:,LABELS[:,bit]==0],axis=1)-np.logaddexp.reduce(logp[:,LABELS[:,bit]==1],axis=1)) for bit in range(2)],axis=1).ravel()
    if not np.isfinite(result).all():raise FloatingPointError('soft demodulation')
    return result

def dimensions(N,family,mcs):
    if N not in (2048,3060,4084) or family not in ('raw','arithmetic') or mcs not in ('QPSK','16QAM'):raise ValueError('unregistered cell')
    header=70 if family=='raw' else 96
    return header,N-header,(2 if mcs=='QPSK' else 4)*(N-header)

def packet(bits,uses,mcs):
    bits=validate_bits(bits);information=np.concatenate((append_crc(bits),np.zeros(6,dtype=np.uint8)))
    slots=uses*(2 if mcs=='QPSK' else 4)
    if len(information)>slots:raise ValueError('information plus CRC/tail exceeds coded slots')
    mother=convolutional_encode(information);mapping=rate_match_indices(len(mother),slots)
    return modulate(mother[mapping],mcs)

def decode_packet(y,length,snr,mcs):
    info=int(length)+22;slots=len(y)*(2 if mcs=='QPSK' else 4)
    if length<1 or info>slots:raise ValueError('invalid received length')
    mapping=rate_match_indices(2*info,slots);evidence=np.bincount(mapping,weights=soft_half_llr(y,snr,mcs),minlength=2*info)
    decoded,score=decode_map(evidence)
    return decoded[:length],bool(crc_accepts(decoded[:-6])),float(score)

def transmit(encoded,label,requested_m,N,family,mcs):
    nh,nd,slots=dimensions(N,family,mcs)
    if requested_m not in range(6,11) or not isinstance(label,(int,np.integer)) or not 0<=label<1000:raise ValueError('mode/class')
    actual=requested_m;payload=None;length=0
    for m in ([requested_m] if family=='raw' else range(requested_m,5,-1)):
        candidate=encoded[m]['raw' if family=='raw' else 'payload']
        if len(candidate)+22<=slots:
            payload=validate_bits(candidate);actual=m;length=0 if family=='raw' else int(encoded[m]['length_field']);break
    if payload is None and family=='raw':raise ValueError('raw cell not encodable: no truncation')
    erasure=payload is None
    # Code7 is a paid, predeclared source-overflow erasure. Still transmit the full N using symbols of the registered constellation.
    mode_code=7 if erasure else actual-6
    header=np.concatenate((indices_to_bits([label],10),indices_to_bits([mode_code],3),indices_to_bits([length],13) if family=='arithmetic' else np.empty(0,dtype=np.uint8)))
    body=modulate(np.zeros(slots,dtype=np.uint8),mcs) if erasure else packet(payload,nd,mcs)
    wave=np.concatenate((encode_packet(header,nh)['symbols'],body))
    if wave.shape!=(N,2) or not np.isfinite(wave).all():raise RuntimeError('waveform resource mismatch')
    return wave,{'protocol':PROTOCOL,'N':N,'N_header':nh,'N_data':nd,'m_requested':requested_m,'m_actual':None if erasure else actual,'source_overflow_erasure':erasure,'overflow_lower_m':not erasure and actual!=requested_m,'payload_bits':0 if erasure else len(payload),'mother_bits':0 if erasure else 2*(len(payload)+22),'coded_slots':slots,'length_field':length,'mcs':mcs,'E':float(np.sum(wave**2)),'energy_constraint':'per_frame_2N' if mcs=='QPSK' else 'fixed_constellation_average_2_per_symbol'}

def receive(observed,snr,N,family,mcs):
    nh,nd,slots=dimensions(N,family,mcs);y=np.asarray(observed,dtype=float)
    if y.shape!=(N,2) or not np.isfinite(y).all():raise ValueError('RX actual complete finite waveform')
    header_length=13 if family=='raw' else 26
    header,crc,_=decode_packet(y[:nh],header_length,snr,'QPSK')
    label=int(bits_to_indices(header[:10],10)[0]);code=int(bits_to_indices(header[10:13],3)[0]);length=0 if family=='raw' else int(bits_to_indices(header[13:],13)[0])
    erasure=code==7;mode=code+6;legal=label<1000 and (0<=code<=4 or (family=='arithmetic' and erasure))
    result={'header_ok':bool(crc and legal and not erasure),'header_crc_ok':crc,'header_fields_legal':legal,'source_overflow_erasure':bool(crc and legal and erasure),'decoded_label':None,'decoded_mode':None,'body_crc_ok':False,'prefix':None,'source_bits':None,'length_field':length}
    if not result['header_ok']:return result
    length=12*sum(s*s for s in SIZES[:mode]) if length==0 else length
    if length+22>slots or (result['length_field'] and length<2):result['header_ok']=False;result['header_fields_legal']=False;return result
    bits,body_crc,score=decode_packet(y[nh:],length,snr,mcs)
    result.update(decoded_label=label,decoded_mode=mode,source_bits=bits,body_crc_ok=body_crc,body_score=score)
    if family=='raw' or result['length_field']==0:result['prefix']=split_prefix(bits_to_indices(bits),mode)
    return result

def channel(wave,snr,source_id,seed,N,family,mcs):
    dimensions(N,family,mcs);wave=np.asarray(wave,dtype=float)
    if wave.shape!=(N,2) or not np.isfinite(wave).all() or not np.isfinite(snr):raise ValueError('channel inputs')
    noise=seeded_noise(f'{PROTOCOL}/{family}/{mcs}/N{N}|{source_id}',int(seed),wave.shape)
    return wave+noise*10.**(-float(snr)/20)
