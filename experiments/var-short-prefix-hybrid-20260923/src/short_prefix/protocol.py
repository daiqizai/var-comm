"""An explicit v1 header domain; legacy 7/8/9 archives are not changed."""
import json
from pathlib import Path
import numpy as np
from var_comm.scale_channel import (indices_to_bits,bits_to_indices,encode_packet,
    rate_match_indices,channel_evidence,decode_map,crc_accepts)
from var_comm.progressive import split_prefix
from var_comm.study import seeded_noise

EXP=Path(__file__).resolve().parents[2]
ROOT=EXP.parents[1]
CONFIG=EXP/'protocol.json'
SIZES=(1,2,3,4,5,6,8,10,13,16)

def config():return json.loads(CONFIG.read_text())

def allocation(m,N=4084,nd=None):
    if isinstance(m,bool) or m not in (6,7,8):raise ValueError('v1 mode must be6/7/8')
    if N not in (3060,4084):raise ValueError('unregistered total budget')
    nd={6:1200,7:1968,8:2992}[m] if nd is None else nd
    if nd not in ({6:(1200,1456),7:(1968,2224),8:(2992,)}[m]):raise ValueError('unregistered ND')
    na=N-68-nd
    if na<=0 or 2*na%256:raise ValueError('no registered spatial continuous allocation')
    return {'m':m,'N':N,'E':2*N,'NH':68,'ND':nd,'NA':na,'protocol_id':'short-prefix-v1'}

def transmit_digital(scales,label,ledger):
    m=ledger['m']
    if not isinstance(label,(int,np.integer)) or not 0<=label<1000:raise ValueError('class field')
    if len(scales)<m or any(np.asarray(s).size!=SIZES[i]**2 for i,s in enumerate(scales[:m])):raise ValueError('official prefix lengths')
    header=np.concatenate((indices_to_bits([label],10),indices_to_bits([m-6],2)))
    payload=indices_to_bits(np.concatenate(scales[:m]))
    wave=np.concatenate((encode_packet(header,68)['symbols'],encode_packet(payload,ledger['ND'])['symbols']))
    assert wave.shape==(68+ledger['ND'],2) and np.all(np.abs(wave)==1)
    return wave

def decode_frame(observed,snr,N=4084,allowed_modes=(6,7,8),digital_allocations=None):
    """RX sees only waveform and pre-shared configuration, never TX fields."""
    y=np.asarray(observed)
    if y.shape!=(N,2) or not np.isfinite(y).all() or not np.isfinite(snr):raise ValueError('invalid RX waveform/SNR')
    bits,_=decode_map(channel_evidence(y[:68],rate_match_indices(68,136),34,snr))
    label=int(bits_to_indices(bits[:10],10)[0]);m=6+int(bits_to_indices(bits[10:12],2)[0])
    crc=bool(crc_accepts(bits[:-6]));legal=label<1000 and m in allowed_modes and m in (6,7,8)
    r={'header_ok':bool(crc and legal),'header_crc_ok':crc,'header_fields_legal':bool(legal),'decoded_label':label,'decoded_mode':m,'body_crc_ok':False,'prefix':[]}
    if not r['header_ok']:return r
    ledger=allocation(m,N,None if digital_allocations is None else digital_allocations[m])
    length=12*sum(s*s for s in SIZES[:m]);info=length+22
    decoded,_=decode_map(channel_evidence(y[68:68+ledger['ND']],rate_match_indices(2*info,2*ledger['ND']),info,snr))
    r.update(ledger=ledger,prefix=split_prefix(bits_to_indices(decoded[:length]),m),body_crc_ok=bool(crc_accepts(decoded[:-6])),observation=y[68+ledger['ND']:].copy())
    return r

def standard_noise(image_id,seed,ledger,segment):
    if segment=='digital':length=68+ledger['ND'];field='ND';budget=ledger['ND']
    elif segment=='continuous':length=ledger['NA'];field='NA';budget=ledger['NA']
    else:raise ValueError('noise segment')
    name=f"SHORT-PREFIX-v1/N{ledger['N']}/m{ledger['m']}/{field}{budget}/{segment}|{image_id}"
    return seeded_noise(name,int(seed),(length,2))

def channel(digital,continuous,image_id,seed,snr,ledger):
    if digital.shape!=(68+ledger['ND'],2) or continuous.shape!=(ledger['NA'],2):raise ValueError('TX budget')
    wave=np.concatenate((digital,continuous))
    if not np.isfinite(wave).all() or not np.isfinite(snr):raise ValueError('nonfinite TX')
    if abs(float(np.square(wave,dtype=np.float64).sum())-ledger['E'])>.02:raise ValueError('TX energy')
    sigma=10**(-float(snr)/20)
    return np.concatenate((digital+sigma*standard_noise(image_id,seed,ledger,'digital'),continuous+sigma*standard_noise(image_id,seed,ledger,'continuous')))
