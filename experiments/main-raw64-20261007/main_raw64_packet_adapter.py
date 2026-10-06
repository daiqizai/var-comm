"""Registered raw12 MAIN packet extension: original q2/q4 and appended q6.

No source images or quality enter this module. prepare performs TX/channel
work; its one-use callback alone invokes RX, after the caller charges its ledger.
RX returns CRC-failed hard information; proxy receipts retain its exact SHA.
This is not the H L13/padding format or an actual-image receiver receipt.
"""
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import main_raw64_plan_only as p

HEADER_KEY='uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2'
EXTRA_SOURCE_SHA={
    'original_phy':'1f1fad07942e30cbc844e0cc3c2d7da7c747b568639d27115dc33eaec49cc65b',
    'original_lookup':'d28e0bb2175c8a7ffa3a5c69502954b60d7f91022d95b6f032587713f0debdcb'}
Q={'QPSK':2,'16QAM':4,'64QAM':6}
PROXY_NAMESPACE='MAIN_RAW64_PROXY_V1'
QUALIFICATION_NAMESPACE='MAIN_RAW64_QUALIFICATION_V1'

def runtime_flags(torch):
    # Exact original MAIN qualification/proxy CPU numerical setup.
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)

def proxy_payloads(common,g,snr,index):
    material=json.dumps([PROXY_NAMESPACE,'counter',g['phy_key'],int(snr),int(index)],sort_keys=True,separators=(',',':')).encode()
    counter=int.from_bytes(hashlib.sha256(material).digest()[:8],'little')&((1<<63)-1)
    payload=common.rng(PROXY_NAMESPACE,'payload',g['phy_key'],int(snr),int(index)).integers(0,2,(1,g['source_bits']),dtype=np.uint8)
    return payload,counter

def binary(value,length):
    a=np.asarray(value)
    p.require(a.shape==(length,) and np.isin(a,(0,1)).all(),'Actual hard bits shape/range changed')
    return a.astype(np.uint8)

def modulate(bits,q,torch,legacy):
    if q in (2,4):return legacy.modulate_torch(bits,q)
    p.require(q==6 and bits.ndim==2 and bits.shape[1]%6==0,'Aligned64QAM bits required')
    pam=torch.tensor([-7.,-5.,-1.,-3.,7.,5.,1.,3.],dtype=bits.dtype,device=bits.device)/math.sqrt(21)
    b=bits.reshape(len(bits),-1,3).long()
    return pam[4*b[:,:,0]+2*b[:,:,1]+b[:,:,2]].reshape(len(bits),-1,2)

def demap(y,snr,q,torch,legacy):
    if q in (2,4):return legacy.demap_torch(y,snr,q)
    p.require(q==6 and y.ndim==3 and y.shape[-1]==2,'64QAM received symbol shape differs')
    pam=torch.tensor([-7.,-5.,-1.,-3.,7.,5.,1.,3.],dtype=y.dtype,device=y.device)/math.sqrt(21)
    lp=-.5*(10**(float(snr)/10))*(y.reshape(len(y),-1,1)-pam)**2
    logits=[]
    for bit in (2,1,0):
        one=[i for i in range(8) if (i>>bit)&1];zero=[i for i in range(8) if not (i>>bit)&1]
        logits.append(torch.logsumexp(lp[:,:,one],-1)-torch.logsumexp(lp[:,:,zero],-1))
    return torch.stack(logits,-1).reshape(len(y),-1)

def transmit_packet(backend,legacy,payload,n,q,counters,group=0,session='body'):
    if q in (2,4):return legacy.transmit_packet(backend,payload,n,q,counters,group,session)
    a,_=legacy.binary_batch(payload);info=legacy.append_crc_batch(a)
    p.require(q==6 and a.shape[1]>0 and a.shape[1]%12==0 and len(counters)==len(a),'Raw64 source/counters invalid')
    code=backend.encode(info,n,q)
    mask=backend.torch.as_tensor(legacy.mask_batch(n,counters,group,session),device=backend.device)
    wave=modulate((code+mask)%2,q,backend.torch,legacy)
    return wave,dict(k=info.shape[1],n=n,q=q,source_bits=a.shape[1],crc_bits=16,tail_bits=0,
        actual_E=wave.square().sum((1,2)).cpu().tolist())

def receive_packet(backend,legacy,received,k,n,q,snr,counters,group=0,session='body'):
    if q in (2,4):return legacy.receive_packet(backend,received,k,n,q,snr,counters,group,session)
    p.require(q==6 and k>=28 and (k-16)%12==0 and n%6==0,'MAIN raw64 public dimensions differ')
    logits=demap(received,snr,q,backend.torch,legacy)
    mask=backend.torch.as_tensor(legacy.mask_batch(n,counters,group,session),device=backend.device)
    logits=logits*(1-2*mask.to(logits.dtype))
    decoded=backend.decode(logits,k,n,q).cpu().numpy().astype(np.uint8)
    return decoded[:,:-16],np.asarray(legacy.crc_accept_batch(decoded)),decoded

class Backend:
    """Actual encoder/decoder are unchanged old Sionna objects, including cache."""
    def __init__(self,old):
        self.old=old;self.torch=old.torch;self.device=old.device
        self.identity64=p.new_identity(old.identity,p.sha(p.__file__))
        self.identity=old.identity
    def plan(self,k,n,num_bits_per_symbol):
        if num_bits_per_symbol in (2,4):return self.old.plan(k,n,num_bits_per_symbol)
        return p.encoder_layout(self.old.Encoder,k,n,num_bits_per_symbol,self.identity64)
    def encode(self,*a,**kw):return self.old.encode(*a,**kw)
    def decode(self,*a,**kw):return self.old.decode(*a,**kw)

def validate_catalogue(cat,old_identity):
    p.require(cat['status']=='PROPOSED_EXTENDED_PUBLIC_CATALOGUE_REQUIRES_REGISTRATION'
        and cat['legacy_profile_count']==270 and cat['catalogue_digest']==p.identity(cat['profiles']),
        'Exact complete appended catalogue required')
    entries=cat['profiles'];p.require([x['profile_id'] for x in entries]==list(range(len(entries)))
        and 270<len(entries)<=4096,'Paid IDs must append to original270')
    groups={};newidentity=p.new_identity(old_identity,p.sha(p.__file__))
    for row in entries:
        p.require(row['N']==1024 and row['G']==1 and row['j'] is None and row['order']=='raster'
            and row['receiver_rule']=='KEEP_actual_hard_prefix_and_raster_partial_v1'
            and row['header_phy_key']==HEADER_KEY and row['header_symbols']==68,'Frozen MAIN raw raster KEEP frame required')
        g=row['groups'][0];q=Q[g['modulation']];source_bits=12*(p.PREFIX[row['m']]+row['K'])
        p.require(g['source_bits']==source_bits and g['crc_bits']==16 and g['tail_bits']==0
            and g['information_bits']==source_bits+16 and g['transmitted_bits']==q*g['symbols']
            and g['symbols']+row['idle_symbols']==956,'MAIN raw/CRC/budget differs')
        p.require((row['profile_id']<270)==(q in (2,4)),'Original/new ID modulation domain changed')
        implementation=old_identity if q in (2,4) else newidentity
        physical=dict(backend=implementation,source_bits=source_bits,CRC=16,layout=g['layout'],modulation=g['modulation'],modulation_padding_bits=0)
        p.require(g['layout']['implementation']==implementation and g['phy_key']==p.identity(physical),'Actual body physical identity changed')
        key=g['phy_key']
        # Nominal-rate aliases can share a physical key. Compare only wire fields.
        definition={k:g[k] for k in ('phy_key','source_bits','crc_bits','tail_bits','information_bits','transmitted_bits','symbols','modulation','layout')}
        p.require(key not in groups or groups[key]==definition,'One phy key names different bodies');groups[key]=definition
    return groups

class Adapter:
    def __init__(self,backend,legacy,common,lookup,header,catalogue,plan):
        self.backend=backend;self.legacy=legacy;self.common=common;self.lookup=lookup;self.header=header
        self.catalogue=catalogue;self.plan=plan;self.groups=validate_catalogue(catalogue,backend.identity)
        p.require(plan['qualification']['mode']=='4_noiseless_4_awgn60'
            and plan['qualification']['noise_namespace']==QUALIFICATION_NAMESPACE
            and plan['proxy']['random_namespace']==PROXY_NAMESPACE,'Registered qualification/proxy randomness differs')
        self.header_map={str(r['profile_id']):r for r in catalogue['profiles']}
        self.validated=set()
    def prepare(self,event):
        p.require(event['phase'] in ('qualification','proxy') and event['kind'] in ('body','header'),'No other science stages here')
        if event['kind']=='header':return self.header_case(event)
        phase=event['phase'];key=event['phy_key'];index=event['index'];snr=event['snr_db']
        p.require(type(index) is int and key in self.groups,'Unknown exact body layout')
        if phase=='qualification':p.require(snr==60 and event['blocks']==8 and 0<=index<8 and event['noiseless'] is (index<4)
            and key in self.plan['qualification']['body_phy_keys'],'Qualification event not in frozen plan')
        else:p.require(snr in self.plan['snrs_db'] and event['blocks']==256 and 0<=index<256
            and dict(phy_key=key,snr_db=snr) in self.plan['proxy']['cells'],'Proxy event not in frozen missing cells')
        g=self.groups[key];q=Q[g['modulation']];k,n=g['information_bits'],g['transmitted_bits']
        if key not in self.validated:
            p.require(self.backend.plan(k,n,q)==g['layout'],'Current actual encoder plan differs from sealed metadata');self.validated.add(key)
        if phase=='qualification':
            payload=self.common.rng('qualify',g['source_bits'],n,q).integers(0,2,(8,g['source_bits']),dtype=np.uint8)[index:index+1]
            counter=index;session='qualification'
        else:
            payload,counter=proxy_payloads(self.common,g,snr,index);session='UEP_BLER_public_counter_v1'
        wave,meta=transmit_packet(self.backend,self.legacy,payload,n,q,[counter],0,session)
        if phase=='qualification' and index<4:received=wave;noise_sha=None
        else:
            generator=(self.common.rng(QUALIFICATION_NAMESPACE,'noise',key,index) if phase=='qualification'
                else self.common.rng(PROXY_NAMESPACE,'noise',key,int(snr),index))
            noise=generator.standard_normal((n//q,2)).astype(np.float32)
            received=wave+self.backend.torch.as_tensor(noise[None],device=self.backend.device)*10**(-float(snr)/20)
            noise_sha=self.common.array_sha(noise)
        request=dict(phase=phase,phy_key=key,snr_db=snr,index=index,counter=counter,session=session,group=0,
            catalogue_digest=self.catalogue['catalogue_digest'],payload_sha256=self.common.array_sha(payload[0]),
            received_sha256=self.common.array_sha(received[0].cpu().numpy()),noise_sha256=noise_sha,
            random_namespace=QUALIFICATION_NAMESPACE if phase=='qualification' else PROXY_NAMESPACE,
            noiseless=phase=='qualification' and index<4,actual_E=float(meta['actual_E'][0]),
            physical_definition_sha256=p.identity(g),source_bits=g['source_bits'],k=k,n=n,q=q)
        used=False
        def decode():
            nonlocal used
            p.require(not used,'Decode callback is single-use; failed calls are not retryable');used=True
            hard,accepted,decoded=receive_packet(self.backend,self.legacy,received,k,n,q,snr,[counter],0,session)
            p.require(np.asarray(hard).shape==(1,g['source_bits']) and np.asarray(decoded).shape==(1,k)
                and np.asarray(accepted).shape==(1,) and np.asarray(accepted).dtype==np.dtype(bool),'Actual decoder shape differs')
            actual=binary(hard[0],g['source_bits']);full=binary(decoded[0],k);accept=bool(accepted[0])
            p.require(np.array_equal(actual,full[:-16]) and bool(self.legacy.crc_accept_batch(full))==accept,'Hard/CRC contradiction')
            equal=bool(np.array_equal(actual,payload[0]));correct=accept and equal
            if phase=='qualification':p.require(correct,'Registered qualification body roundtrip failed')
            return dict(correct=correct,rejected=not accept,undetected=accept and not equal,payload_equal=equal,
                crc_accept=accept,actual_hard_sha256=self.common.array_sha(actual),
                decoded_sha256=self.common.array_sha(full),actual_E=request['actual_E'],event_id=event['event_id'])
        return request,decode
    def header_case(self,event):
        pid=event['profile_id'];p.require(event['phase']=='qualification' and event['phy_key']==HEADER_KEY
            and event['snr_db']==60 and event['blocks']==1 and event['index']==0
            and pid in self.plan['qualification']['header_profile_ids'],'Only registered header qualification')
        wave=np.asarray(self.header.transmit(pid),dtype=np.float64);p.require(wave.shape==(68,2),'Paid header shape changed')
        request=dict(profile_id=pid,received_sha256=self.common.array_sha(wave),snr_db=60,
            catalogue_digest=self.catalogue['catalogue_digest'],actual_E=float(np.square(wave).sum()))
        used=False
        def decode():
            nonlocal used
            p.require(not used,'Single-use header callback');used=True
            v=self.header.receive(wave,60,self.header_map);known=str(pid) in self.header_map
            p.require(v['header_crc_ok'] is True and v['header_fields_legal'] is known and v['header_ok'] is known
                and v['profile_id']==(pid if known else None),'Exact known/unknown header roundtrip failed')
            return dict(v,event_id=event['event_id'],actual_E=request['actual_E'])
        return request,decode

def create(cfg):
    """Called by the registered CPU worker, before any charged callback."""
    a=cfg['adapter_config'];bound={}
    for values in (cfg['source_bindings'],cfg['input_bindings']):
        for path,h in values.items():
            p.require(path not in bound or bound[path]==h,'Conflicting bound dependency');bound[path]=h
    def check(path,expected=None):
        p.require(bound.get(path)==p.sha(path) and (expected is None or bound[path]==expected),'Unbound/changed packet dependency: '+path)
    check(str(Path(__file__).absolute()));check(str(Path(p.__file__).absolute()))
    p.require(a['plan_module']==str(Path(p.__file__).absolute()),'Imported exact resource adapter differs')
    for k in ('original_common','original_planner','original_backend'):check(a[k],p.FROZEN_SOURCES[k])
    for k,h in EXTRA_SOURCE_SHA.items():check(a[k],h)
    check(a['legacy_qualification'],p.OLD_QUALIFICATION_SHA);check(a['catalogue']);check(cfg['plan'])
    cat=p.read(a['catalogue']);plan=p.read(cfg['plan'])
    p.require(plan['catalogue_digest']==cat['catalogue_digest'],'Registered plan/public catalogue mismatch')
    p.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2'
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only two-thread worker required')
    src=Path(cfg['project_root'])/'src/var_comm'
    header_sources=sorted(x for x in src.rglob('*') if x.suffix in ('.py','.cpp') and x.is_file())
    p.require(header_sources and any(x.name=='scale_channel.py' for x in header_sources)
        and any(x.name=='token_trellis.cpp' for x in header_sources),'Original paid-header source closure incomplete')
    for path in header_sources:check(str(path))
    common=p.load(a['original_common'],'uep_common');p.load(a['original_planner'],'profiles')
    legacy=p.load(a['original_phy'],'uep_phy');lookup=p.load(a['original_lookup'],'main64_original_lookup')
    module=p.load(a['original_backend'],'main64_actual_original_backend')
    old=module.SionnaBackend('cpu',a['legacy_qualification']);runtime_flags(old.torch)
    p.require(not old.torch.cuda.is_initialized(),'CUDA initialized unexpectedly')
    backend=Backend(old);header=legacy.Header(cfg['project_root'])
    return Adapter(backend,legacy,common,lookup,header,cat,plan)
