"""Independent paid N2048 entropy PHY around frozen T1 bit format and backend.

Only actual constructor-admitted layouts enter the public received-header
catalogue. Unsupported queries are recorded before quality access. All m4..10
headers remain receiver-visible for every admitted MCS in the frozen family.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import math,os
import numpy as np
import t1_phy as old
import t2_pilot as h
from t6_raw_phy import standard_noise

PROTOCOL='WCL_T6_N2048_ENTROPY_ACTUAL_V1'
QUAL_SCHEMA='WCL_T6_N2048_ENTROPY_REAL_PHY_QUALIFICATION_V1'
N=2048;BODY_SYMBOLS=1980;HEADER_SYMBOLS=68
require=old.require;canonical=old.canonical;digest=old.digest;sha=old.sha;read=old.read
array_sha=old.array_sha;as_numpy=old.as_numpy;binary=old.binary;pack_body=old.pack_body
scramble_mask=old.scramble_mask;UnsupportedConfiguration=old.UnsupportedConfiguration
checked_profile=old.checked_profile;select_received_profile=old.select_received_profile

def parse_body(decoded,profile):
    value=old.parse_body(decoded,profile)
    value['source_capacity_bits']=min(profile['k']-29,8191)
    return value

def profile_templates(family):
    require(family in old.FAMILIES,'One N1024-frozen entropy family required')
    rows=[]
    for m in range(4,11):
        for q in(2,4,6):
            for rate in old.RATES:
                fraction=Fraction(rate);n=1980*q;k=n*fraction.numerator//fraction.denominator
                rows.append(dict(profile_id=len(rows),family=family,mode='arithmetic',m=m,K=0,
                    token_count=sum(x*x for x in old.SIZES[:m]),q=q,nominal_rate=rate,k=k,n=n,N=2048,
                    header_symbols=68,body_symbols=1980,frame_padding_symbols=0,length_bits=13,crc_bits=16,
                    body_tail_bits=0,source_capacity_bits=min(k-29,8191),layout_status='UNQUALIFIED_METADATA_ONLY'))
    require(len(rows)==84,'Seven actual prefix lengths by12 MCS resource queries')
    return rows

class Backend(old.Backend):
    def __init__(self,original):
        super().__init__(original)
        self.identity=dict(self.identity,implementation='Sionna_5G_LDPC_WCL_T6_N2048_ENTROPY_ADAPTER_V1',
            paid_body_symbols=1980,length_field_bits=13,source_capacity_rule='min(k-29,8191)')
    @lru_cache(maxsize=12)
    def plan(self,k,n,q):
        k,n,q=int(k),int(n),int(q)
        if q not in (2,4,6) or n!=1980*q or k<31 or n<k:
            raise UnsupportedConfiguration('N2048 full1980-body single-codeblock dimensions invalid')
        try:e=self.Encoder(k,n,num_bits_per_symbol=q,device='cpu',precision='single')
        except (ValueError,AssertionError) as exc:
            raise UnsupportedConfiguration(type(exc).__name__+': '+str(exc)) from exc
        require(e.k==k and e.n==n and e.k_ldpc-e.k_filler==k,'Actual encoder information dimensions differ')
        if n>e.n_cb_comp:raise UnsupportedConfiguration('RV0 repetition beyond pinned encoder unsupported')
        permutation=e.out_int.cpu().numpy()
        require(sorted(permutation.tolist())==list(range(n)),'Actual output interleaver is not a permutation')
        layout=dict(k=k,n=n,q=q,k_ldpc=int(e.k_ldpc),k_filler=int(e.k_filler),n_cb=int(e.n_cb),
          n_cb_comp=int(e.n_cb_comp),bg=str(e._bg),z=int(e.z),code_blocks=1,mother_bits=int(e.n_ldpc),
          puncturing_bits=int(e.n_ldpc-e.k_filler-n),puncturing_first_2Z=int(2*e.z),
          puncturing_remaining=int(e.n_cb_comp-n),shortening_bits=int(e.k_filler),repetition_bits=0,
          modulation_padding_bits=0,interleaver_sha256=array_sha(permutation),
          actual_effective_rate=k/n,decoder_config=self.decoder,implementation=self.identity,
          actual_constructor_used=True)
        layout['layout_id']=digest(layout);return layout

class Runtime(old.Runtime):
    def __init__(self,root,family,reference_qualification=None,configure_threads=True):
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only entropy PHY')
        super().__init__(root,reference_qualification,configure_threads)
        require(family in old.FAMILIES,'Frozen single-family context required')
        self.family=family;self.backend=Backend(self.backend.old)
        require(not self.torch.cuda.is_initialized(),'Entropy PHY must not initialize CUDA')
    def build_catalogue(self):
        layouts={};profiles=[];rejected=[]
        for p in profile_templates(self.family):
            key=(p['k'],p['n'],p['q'])
            if key not in layouts:
                try:layouts[key]=self.backend.plan(*key)
                except UnsupportedConfiguration as exc:
                    layouts[key]=None;rejected.append(dict(k=key[0],n=key[1],q=key[2],reason=str(exc)))
            if layouts[key]is None:continue
            row=dict(p,layout=layouts[key],layout_id=layouts[key]['layout_id'],layout_status='ACTUAL_CONSTRUCTOR_ADMITTED')
            row['profile_key']=digest(row);profiles.append(row)
        value=dict(protocol=PROTOCOL,N=2048,family=self.family,profiles=profiles,
            layouts=[x for x in layouts.values()if x is not None],rejected_layouts=rejected,
            requested_profiles=84,requested_layouts=12,status='ACTUAL_CONSTRUCTOR_ADMISSION_COMPLETE_NOT_PACKET_QUALIFIED',
            admission_rule='Only actual frozen single-codeblock Encoder constructor legality; no image/quality access; do not truncate k')
        value['catalogue_sha256']=digest(value);self.catalogue=value;self.profiles={str(p['profile_id']):p for p in profiles};return value
    def use_catalogue(self,value):
        current=self.build_catalogue();require(current==value,'Actual N2048 entropy constructors/catalogue changed')
        require(len(current['profiles'])==7*len(current['layouts'])and current['profiles'],'Every admitted MCS must expose m4..10')
    def transmit(self,profile_id,payload_bits,counter,session=PROTOCOL):
        require(self.qualified,'New actual N2048 entropy qualification required')
        return transmit_frame(self,payload_bits,profile_id,counter,session)
    def receive(self,observed,snr,counter,codebook,ledger,event,phase='calibration',session=PROTOCOL):
        require(self.qualified and codebook==self.profiles,'Use complete qualified RX catalogue, never filter to TX m')
        event_id=event['event_id']if isinstance(event,dict)else event
        return receive_frame(self,observed,snr,counter,ledger,event_id,phase,session)

def source_bindings(root):
    bindings=old.collect_source_bindings(root)
    for name in('t6_entropy_phy.py','t6_qualify_entropy_phy.py','t6_raw_phy.py','t6_plan.py','t6_calibration_plan.py','t2_pilot.py'):
        p=str(Path(__file__).with_name(name).resolve());bindings[p]=sha(p)
    return bindings

def create_runtime(root,qualification,configure_threads=True):
    done=read(qualification);require(done['schema']==QUAL_SCHEMA and done['status']=='PASS','Completed N2048 entropy qualification first')
    r=h.checked(done['request']);require(r['root']==str(Path(root).resolve()),'Qualified root differs')
    old.verify_bindings(r['source_bindings']);old.verify_bindings(r['input_bindings']);old.verify_bindings(done['outputs'])
    require(done['packet_decode_count']==r['packet_cap']and done['ledger']==dict(total=r['packet_cap'],unresolved=0,cap=r['packet_cap']),
        'Every admitted body/header qualification must complete')
    rt=Runtime(root,r['family'],configure_threads=configure_threads);rt.use_catalogue(r['catalogue'])
    require(rt.backend.identity==r['backend_identity']==done['backend_identity'],'Current qualified backend differs')
    require(rt.catalogue['catalogue_sha256']==done['catalogue_sha256'],'Qualified catalogue differs')
    rt.qualified=True;rt.qualification=h.desc(qualification);return rt

def transmit_body(runtime,payload,profile,frame_counter,session=PROTOCOL):
    checked_profile(runtime,profile);info=pack_body(payload,profile);backend=runtime.backend
    code=backend.encode(info[None],profile['n'],profile['q'])
    require(tuple(code.shape)==(1,profile['n']) and np.isin(as_numpy(code),(0,1)).all(),'Actual LDPC encoder output invalid')
    mask=backend.torch.as_tensor(scramble_mask(profile['n'],frame_counter,session)[None],device='cpu')
    scrambled=(code+mask)%2
    body=as_numpy(runtime.mapping.modulate(scrambled,profile['q'],backend.torch,runtime.legacy))[0].astype(np.float32)
    require(body.shape==(1980,2) and np.isfinite(body).all(),'Fixed1980-symbol body required')
    energy=float(np.square(body.astype(np.float64)).sum())
    return body,dict(profile_id=profile['profile_id'],profile_key=profile['profile_key'],family=profile['family'],
      m=profile['m'],K=0,N=2048,k=profile['k'],n=profile['n'],q=profile['q'],
      length_field_bits=13,arithmetic_bits=len(payload),crc_bits=16,information_bits=len(info),
      known_information_padding_bits=profile['k']-29-len(payload),frame_padding_symbols=0,
      body_symbols=1980,body_energy=energy,information_sha256=array_sha(info),codeword_sha256=array_sha(as_numpy(code)[0]),
      transmitted_body_sha256=array_sha(body),public_frame_counter=frame_counter,session=session)

def transmit_frame(runtime,payload,profile_id,frame_counter,session=PROTOCOL):
    require(runtime.profiles is not None and str(profile_id) in runtime.profiles,'Unknown paid TX profile')
    p=runtime.profiles[str(profile_id)];body,meta=transmit_body(runtime,payload,p,frame_counter,session)
    header=np.asarray(runtime.header.transmit(int(profile_id)),dtype=np.float64)
    require(header.shape==(68,2),'Paid68 header required')
    frame=np.concatenate((header,body.astype(np.float64)))
    E=float(np.square(frame).sum());meta.update(header_symbols=68,total_symbols=2048,
      header_energy=float(np.square(header).sum()),E_frame=E,rho=E/(2*2048),transmitted_frame_sha256=array_sha(frame))
    return frame,meta

def receive_header(runtime,observed,snr_db,ledger,event_id,phase='qualification'):
    y=np.asarray(observed,dtype=np.float64)
    require(y.shape==(68,2) and np.isfinite(y).all() and math.isfinite(float(snr_db)),'Finite paid68 header observations required')
    require(runtime.profiles is not None,'Registered public catalogue required')
    req=dict(protocol=PROTOCOL,received_sha256=array_sha(y),snr_db=float(snr_db),
             catalogue_sha256=runtime.catalogue['catalogue_sha256'],header_symbols=68)
    event=dict(event_id=event_id,phase=phase,kind='header',protocol=PROTOCOL)
    return ledger.call(event,req,lambda:runtime.header.receive(y,float(snr_db),runtime.profiles))

def select_received_profile(runtime,header_outcome):
    if not header_outcome['header_ok']:return None
    p=runtime.profiles.get(str(header_outcome['profile_id']))
    require(p is not None,'Header was accepted with an unknown profile')
    return p

def receive_body(runtime,observed,profile,snr_db,frame_counter,ledger,event_id,phase='qualification',session=PROTOCOL):
    """Profile is selected from paid RX header. No payload truth is accepted."""
    checked_profile(runtime,profile);y=np.asarray(observed,dtype=np.float32)
    require(y.shape==(1980,2) and np.isfinite(y).all() and math.isfinite(float(snr_db)),'Finite1980 received body required')
    backend=runtime.backend
    req=dict(protocol=PROTOCOL,received_sha256=array_sha(y),snr_db=float(snr_db),
       public_frame_counter=frame_counter,session=session,profile_key=profile['profile_key'],
       layout_id=profile['layout_id'],catalogue_sha256=runtime.catalogue['catalogue_sha256'])
    event=dict(event_id=event_id,phase=phase,kind='body',protocol=PROTOCOL)
    def decode():
        t=backend.torch;rx=t.as_tensor(y[None],dtype=t.float32,device='cpu')
        logits=runtime.mapping.demap(rx,float(snr_db),profile['q'],t,runtime.legacy)
        mask=t.as_tensor(scramble_mask(profile['n'],frame_counter,session)[None],device='cpu')
        logits=logits*(1-2*mask.to(logits.dtype))
        hard=as_numpy(backend.decode(logits,profile['k'],profile['n'],profile['q']))
        require(hard.shape==(1,profile['k']) and np.isin(hard,(0,1)).all(),'Actual LDPC hard decoder output invalid')
        return dict(parse_body(hard[0].astype(np.uint8),profile),profile_id=profile['profile_id'],
                    profile_key=profile['profile_key'],layout_id=profile['layout_id'])
    return ledger.call(event,req,decode)

def receive_frame(runtime,observed,snr_db,frame_counter,ledger,event_id,phase='calibration',session=PROTOCOL):
    """Every error-accepted header selects its actual RX family/m/q/r; never TX."""
    y=np.asarray(observed)
    require(y.shape==(2048,2) and np.isfinite(y).all(),'Exactly2048 finite received symbols')
    header=receive_header(runtime,y[:68],snr_db,ledger,event_id+':header',phase)
    p=select_received_profile(runtime,header)
    if p is None:return dict(header=header,body=None,rx_profile=None,status='HEADER_REJECT')
    body=receive_body(runtime,y[68:],p,snr_db,frame_counter,ledger,event_id+':body',phase,session)
    return dict(header=header,body=body,rx_profile={k:p[k] for k in ('profile_id','family','mode','m','K','q','nominal_rate','k','n','token_count','profile_key')},status=body['status'])
