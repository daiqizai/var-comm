"""Paid N1024 entropy partial extension; derived from immutable T1 PHY.

No image, entropy probability, source token or TX truth enters receive_frame.
Source decoding/canonical checks belong to the caller after physical parsing.
"""
from __future__ import annotations
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

OLD_SCRIPTS=Path(__file__).resolve().parents[2]/'wcl-evidence-closure-20261009/scripts'
sys.path.insert(0,str(OLD_SCRIPTS))
from ep_plan import ks

PROTOCOL='WCL_EP_ENTROPY_PARTIAL_PHY_20261010_V1'
WIRE_PROTOCOL='WCL_T1_ENTROPY_WHOLE_PHY_20261009_V1'
FAMILIES=('EC_STATIC_WHOLE','EC_VAR_WHOLE','EC_VAR_PARTIAL')
RATES=('1/2','2/3','3/4','5/6')
SIZES=(1,2,3,4,5,6,8,10,13,16)
N=1024; HEADER_SYMBOLS=68; BODY_SYMBOLS=956; LENGTH_BITS=13; CRC_BITS=16
MAPPING='original_MAIN_q2_q4_and_q6_Gray_I_then_Q;nominal_Es2;no_frame_normalization'
SCRAMBLING='WCL_T1_public_PCG64_SHA256_protocol_session_counter_after_RV0_v1'
LEGACY_QUALIFICATION_SHA='d23d9dc2c9c08855f2b539db90f51b83d034b61d97a3e0bf19093db95a232dd7'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def binary(bits):
    a=np.asarray(bits)
    require(a.ndim==1 and np.isin(a,(0,1)).all(),'One binary vector required')
    return a.astype(np.uint8,copy=True)
def integer_bits(value,width):
    require(type(value) is int and 0<=value<1<<width,'Integer field overflow')
    return ((value>>np.arange(width-1,-1,-1))&1).astype(np.uint8)
def bits_integer(bits):
    value=0
    for bit in binary(bits):value=2*value+int(bit)
    return value
def crc16(bits):
    value=65535
    for bit in binary(bits):
        feedback=(value>>15)^int(bit)
        value=((value<<1)^(0x1021 if feedback else 0))&65535
    return value
def append_crc(bits):
    a=binary(bits);return np.concatenate((a,integer_bits(crc16(a),16)))
def crc_ok(bits):
    a=binary(bits);return bool(len(a)>=16 and crc16(a[:-16])==bits_integer(a[-16:]))

def profile_templates():
    """Public IDs only. These integer templates are not LDPC admission evidence."""
    rows=[]
    for family in FAMILIES[:2]:
        for m in range(4,10):
            for q in (2,4,6):
                for rate in RATES:
                    r=Fraction(rate);n=BODY_SYMBOLS*q;k=n*r.numerator//r.denominator
                    rows.append(dict(profile_id=len(rows),family=family,mode='arithmetic',m=m,K=0,
                      token_count=sum(x*x for x in SIZES[:m]),q=q,nominal_rate=rate,k=k,n=n,N=N,
                      header_symbols=HEADER_SYMBOLS,body_symbols=BODY_SYMBOLS,frame_padding_symbols=0,
                      length_bits=LENGTH_BITS,crc_bits=CRC_BITS,body_tail_bits=0,
                      source_capacity_bits=k-LENGTH_BITS-CRC_BITS,
                      layout_status='UNQUALIFIED_METADATA_ONLY'))
    # Exact old 0..143 records remain unchanged. New m/K is paid in ID144..359.
    for m in range(4,10):
        for K in ks(m):
            for q in (2,4,6):
                for rate in RATES:
                    base=next(r for r in rows[:144] if r['family']=='EC_VAR_WHOLE'
                        and r['m']==m and r['q']==q and r['nominal_rate']==rate)
                    rows.append(dict(base,profile_id=len(rows),family='EC_VAR_PARTIAL',
                        K=K,token_count=base['token_count']+K))
    require(len(rows)==360 and len({r['profile_id'] for r in rows})==360,'Fixed360 paid profiles')
    return rows

def pack_body(arithmetic_bits,profile):
    a=binary(arithmetic_bits);capacity=profile['k']-29
    require(profile['mode']=='arithmetic' and profile['family'] in FAMILIES,'Pure arithmetic profile required')
    require(2<=len(a)<=min(capacity,8191),'Actual arithmetic stream does not fit paid body')
    protected=np.concatenate((integer_bits(len(a),13),a,np.zeros(capacity-len(a),dtype=np.uint8)))
    result=append_crc(protected);require(len(result)==profile['k'],'Paid information length differs')
    return result

def parse_body(decoded,profile):
    """CRC/length/known-padding only; never recognizes errors using TX truth."""
    a=binary(decoded);require(len(a)==profile['k'],'Actual hard decoder length differs')
    L=bits_integer(a[:13]);base=dict(decoded_bits=a.tolist(),hard_bits_sha256=array_sha(a),
        declared_length=L,source_capacity_bits=profile['k']-29,
        canonical_source_checked=False,payload=None)
    if not crc_ok(a):return dict(base,crc_accepted=False,parser_accepted=False,status='CRC_REJECT')
    reason=None
    if profile['mode']!='arithmetic' or profile['family'] not in FAMILIES:reason='UNKNOWN_ENTROPY_FAMILY'
    elif not 2<=L<=profile['k']-29:reason='L_OUTSIDE_PROFILE_CAPACITY'
    elif np.any(a[13+L:-16]):reason='NONZERO_KNOWN_PADDING'
    if reason:return dict(base,crc_accepted=True,parser_accepted=False,status='DECODE_INVALID',invalid_reason=reason)
    base['payload']=a[13:13+L].tolist()
    return dict(base,crc_accepted=True,parser_accepted=True,status='PAYLOAD_PARSED',known_padding_zero=True)

def scramble_mask(n,frame_counter,session):
    require(type(frame_counter) is int and frame_counter>=0 and isinstance(session,str) and bool(session),'Public counter/session required')
    seed=int.from_bytes(hashlib.sha256(canonical([WIRE_PROTOCOL,'scramble',session,frame_counter]).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed)).integers(0,2,int(n),dtype=np.uint8)

def standard_noise(source_id,noise_seed):
    """Same full-frame standard variates for every q/m/family at a source/seed."""
    require(isinstance(source_id,(str,int)) and not isinstance(source_id,bool),'Stable public source identity required')
    require(type(noise_seed) is int and noise_seed>=0,'Public integer noise seed required')
    seed=int.from_bytes(hashlib.sha256(canonical([WIRE_PROTOCOL,'paired_standard_AWGN',str(source_id),noise_seed]).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((N,2))

class UnsupportedConfiguration(ValueError):pass

class Backend:
    """General k/n wrapper around the unchanged original actual Sionna classes."""
    def __init__(self,old):
        self.old=old;self.torch=old.torch;self.device=old.device
        self.Encoder=old.Encoder;self.Decoder=old.Decoder
        self.decoder=dict(old.identity['decoder'])
        self.identity=dict(implementation='Sionna_5G_LDPC_WCL_T1_ADAPTER_V1',
            original_backend_identity=old.identity,decoder=self.decoder,bit_mapping=MAPPING,
            scrambling=SCRAMBLING,power_protocol='fixed_constellation_average_Es2_no_per_frame_normalization',
            rate_matching=old.identity['rate_matching'],precision='float32',device='cpu')
    @lru_cache(maxsize=12)
    def plan(self,k,n,q):
        k,n,q=int(k),int(n),int(q)
        if q not in (2,4,6) or n!=956*q or k<31 or n<k:
            raise UnsupportedConfiguration('N1024 full956-body single-codeblock dimensions invalid')
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
    @lru_cache(maxsize=12)
    def codecs(self,k,n,q):
        self.plan(k,n,q)
        encoder=self.Encoder(int(k),int(n),num_bits_per_symbol=int(q),precision='single',device='cpu')
        decoder=self.Decoder(encoder,**{k:v for k,v in self.decoder.items() if k!='early_stop'},device='cpu')
        encoder.eval();decoder.eval();return encoder,decoder
    def encode(self,bits,n,q):
        t=self.torch;x=t.as_tensor(bits,dtype=t.float32,device='cpu')
        with t.inference_mode():return self.codecs(x.shape[-1],n,q)[0](x)
    def decode(self,logits,k,n,q):
        t=self.torch;x=t.as_tensor(logits,dtype=t.float32,device='cpu')
        with t.inference_mode():return self.codecs(k,n,q)[1](x)

def source_paths(root):
    root=Path(root).resolve();prior=root/'experiments/prior-aware-grouped-mcs-v1';raw=root/'experiments/main-raw64-20261007'
    paths=[prior/n for n in ('uep_common.py','profiles.py','ldpc_backend.py','uep_phy.py')]
    paths.extend(raw/n for n in ('main_raw64_plan_only.py','main_raw64_packet_adapter.py'))
    paths.extend(p for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h','.hpp'))
    paths.extend([Path(__file__).resolve(),(OLD_SCRIPTS/'t2_ledger.py').resolve(),
                  Path(__file__).with_name('ep_qualify_phy.py').resolve(),Path(__file__).with_name('ep_plan.py').resolve()])
    require(any(p.name=='scale_channel.py' for p in paths) and any(p.name=='token_trellis.cpp' for p in paths),'Complete paid-header source closure required')
    return sorted(set(paths))
def collect_source_bindings(root):return {str(p):sha(p) for p in source_paths(root)}
def verify_bindings(bindings):
    for path,h in bindings.items():require(sha(path)==h,'Changed bound source/input: '+path)
def load_exact(path,name):
    path=Path(path).resolve()
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve()==path,'Unexpected imported module: '+name)
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

class Runtime:
    def __init__(self,root,reference_qualification=None,configure_threads=True):
        self.root=Path(root).resolve();prior=self.root/'experiments/prior-aware-grouped-mcs-v1';raw=self.root/'experiments/main-raw64-20261007'
        common=load_exact(prior/'uep_common.py','uep_common');load_exact(prior/'profiles.py','profiles')
        oldmodule=load_exact(prior/'ldpc_backend.py','_wcl_ep_original_ldpc')
        self.legacy=load_exact(prior/'uep_phy.py','uep_phy')
        load_exact(raw/'main_raw64_plan_only.py','main_raw64_plan_only')
        self.mapping=load_exact(raw/'main_raw64_packet_adapter.py','main_raw64_packet_adapter')
        reference=Path(reference_qualification or self.root/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json')
        require(sha(reference)==LEGACY_QUALIFICATION_SHA,'Original pinned backend qualification changed')
        old=oldmodule.SionnaBackend('cpu',str(reference));self.backend=Backend(old);self.torch=old.torch
        if configure_threads:
            self.torch.set_num_threads(2);self.torch.set_num_interop_threads(1)
            self.torch.backends.cuda.matmul.allow_tf32=False;self.torch.backends.cudnn.allow_tf32=False
            self.torch.backends.cudnn.benchmark=False;self.torch.use_deterministic_algorithms(True)
        self.header=self.legacy.Header(self.root);self.reference_qualification=str(reference)
        # Source implementations of metadata CRC must agree; no packet decoder is called.
        probe=np.random.default_rng(20261009031).integers(0,2,101,dtype=np.uint8)
        require(np.array_equal(append_crc(probe),self.legacy.append_crc_batch(probe)),'Body CRC convention differs from original')
        self.catalogue=None;self.profiles=None;self.qualified=False;self.qualification=None
    def build_catalogue(self):
        layouts={};profiles=[];rejected=[]
        for p in profile_templates():
            key=(p['k'],p['n'],p['q'])
            if key not in layouts:
                try:layouts[key]=self.backend.plan(*key)
                except UnsupportedConfiguration as exc:
                    rejected.append(dict(k=key[0],n=key[1],q=key[2],reason=str(exc)));layouts[key]=None
            if layouts[key] is None:continue
            row=dict(p,layout=layouts[key],layout_id=layouts[key]['layout_id'],layout_status='ACTUAL_CONSTRUCTOR_ADMITTED')
            row['profile_key']=digest(row);profiles.append(row)
        result=dict(protocol=PROTOCOL,profiles=profiles,layouts=[v for v in layouts.values() if v is not None],
                    rejected_layouts=rejected,requested_profiles=360,requested_layouts=12,
                    status='ACTUAL_LAYOUTS_ADMITTED_NOT_YET_PACKET_QUALIFIED' if not rejected else 'UNSUPPORTED_LAYOUTS_STOP')
        result['catalogue_sha256']=digest(result)
        self.catalogue=result;self.profiles={str(p['profile_id']):p for p in profiles};return result
    def use_catalogue(self,catalogue):
        current=self.build_catalogue();require(current==catalogue,'Actual current constructors/catalogue differ from frozen request')
        require(not current['rejected_layouts'] and len(current['profiles'])==360,'All360/12 required layouts must be admitted')

    def transmit(self,profile_id,payload_bits,counter,session=WIRE_PROTOCOL):
        require(self.qualified,'A real completed 457-call qualification must bind formal use')
        return transmit_frame(self,payload_bits,profile_id,counter,session)

    def receive(self,observed,snr,counter,codebook,ledger,event,phase='calibration',session=WIRE_PROTOCOL):
        require(self.qualified,'A real completed 457-call qualification must bind formal use')
        require(codebook==self.profiles,'Receive requires the whole qualified public catalogue, never a TX-filtered codebook')
        if isinstance(event,dict):
            event_id=event['event_id'];phase=event.get('phase',phase)
        else:event_id=event
        require(isinstance(event_id,str) and bool(event_id),'Stable event identity required')
        return receive_frame(self,observed,snr,counter,ledger,event_id,phase,session)

def create_runtime(root,qualification,reference_qualification=None,configure_threads=True):
    """Formal factory: exact source/input/output bindings and actual constructors."""
    cp=Path(qualification).resolve()
    if cp.is_dir():cp=cp/'completion.json'
    c=read(cp)
    require(c['status']=='PASS' and c['schema']=='WCL_EP_REAL_PHY_QUALIFICATION_20261010_V1','Completed real qualification required')
    require(c['actual_constructor_validation'] is True and c['profile_count']==360 and c['actual_layout_count']==12,'Full actual qualification grid required')
    require(c['actual_body_decodes']==96 and c['actual_header_decodes']==361 and c['packet_decode_count']==457,'Full 457 actual decoder calls required')
    require(c['ledger']==dict(total=457,unresolved=0,cap=457),'Qualification independent ledger is incomplete')
    rp=Path(c['request_path']);r=read(rp)
    require(sha(rp)==c['request_sha256'] and Path(r['root']).resolve()==Path(root).resolve(),'Qualification request/root differs')
    require(r['source_bindings']==c['source_bindings'] and r['input_bindings']==c['input_bindings'],'Qualification binding sets differ')
    verify_bindings(c['source_bindings']);verify_bindings(c['input_bindings']);verify_bindings(c['outputs'])
    require(collect_source_bindings(root)==c['source_bindings'],'Current full source closure differs')
    rt=Runtime(root,reference_qualification,configure_threads);rt.use_catalogue(r['catalogue'])
    require(rt.backend.identity==c['backend_identity']==r['backend_identity'],'Actual current backend differs from qualified backend')
    require(rt.catalogue['catalogue_sha256']==c['catalogue_sha256'],'Actual public catalogue differs from qualification')
    rt.qualified=True;rt.qualification=dict(path=str(cp),sha256=sha(cp),request_sha256=c['request_sha256'])
    return rt

def checked_profile(runtime,profile):
    require(runtime.profiles is not None,'Load the registered public catalogue')
    expected=runtime.profiles.get(str(profile['profile_id']))
    require(expected==profile,'Profile is not the registered received public profile')
    require(runtime.backend.plan(profile['k'],profile['n'],profile['q'])==profile['layout'],'Actual LDPC layout changed')
    return profile
def as_numpy(value):return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def transmit_body(runtime,payload,profile,frame_counter,session=WIRE_PROTOCOL):
    checked_profile(runtime,profile);info=pack_body(payload,profile);backend=runtime.backend
    code=backend.encode(info[None],profile['n'],profile['q'])
    require(tuple(code.shape)==(1,profile['n']) and np.isin(as_numpy(code),(0,1)).all(),'Actual LDPC encoder output invalid')
    mask=backend.torch.as_tensor(scramble_mask(profile['n'],frame_counter,session)[None],device='cpu')
    scrambled=(code+mask)%2
    body=as_numpy(runtime.mapping.modulate(scrambled,profile['q'],backend.torch,runtime.legacy))[0].astype(np.float32)
    require(body.shape==(956,2) and np.isfinite(body).all(),'Fixed956-symbol body required')
    energy=float(np.square(body.astype(np.float64)).sum())
    return body,dict(profile_id=profile['profile_id'],profile_key=profile['profile_key'],family=profile['family'],
      m=profile['m'],K=profile['K'],N=1024,k=profile['k'],n=profile['n'],q=profile['q'],
      length_field_bits=13,arithmetic_bits=len(payload),crc_bits=16,information_bits=len(info),
      known_information_padding_bits=profile['k']-29-len(payload),frame_padding_symbols=0,
      body_symbols=956,body_energy=energy,information_sha256=array_sha(info),codeword_sha256=array_sha(as_numpy(code)[0]),
      transmitted_body_sha256=array_sha(body),public_frame_counter=frame_counter,session=session)

def transmit_frame(runtime,payload,profile_id,frame_counter,session=WIRE_PROTOCOL):
    require(runtime.profiles is not None and str(profile_id) in runtime.profiles,'Unknown paid TX profile')
    p=runtime.profiles[str(profile_id)];body,meta=transmit_body(runtime,payload,p,frame_counter,session)
    header=np.asarray(runtime.header.transmit(int(profile_id)),dtype=np.float64)
    require(header.shape==(68,2),'Paid68 header required')
    frame=np.concatenate((header,body.astype(np.float64)))
    E=float(np.square(frame).sum());meta.update(header_symbols=68,total_symbols=1024,
      header_energy=float(np.square(header).sum()),E_frame=E,rho=E/(2*1024),transmitted_frame_sha256=array_sha(frame))
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

def receive_body(runtime,observed,profile,snr_db,frame_counter,ledger,event_id,phase='qualification',session=WIRE_PROTOCOL):
    """Profile is selected from paid RX header. No payload truth is accepted."""
    checked_profile(runtime,profile);y=np.asarray(observed,dtype=np.float32)
    require(y.shape==(956,2) and np.isfinite(y).all() and math.isfinite(float(snr_db)),'Finite956 received body required')
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

def receive_frame(runtime,observed,snr_db,frame_counter,ledger,event_id,phase='calibration',session=WIRE_PROTOCOL):
    """Every error-accepted header selects its actual RX family/m/q/r; never TX."""
    y=np.asarray(observed)
    require(y.shape==(1024,2) and np.isfinite(y).all(),'Exactly1024 finite received symbols')
    header=receive_header(runtime,y[:68],snr_db,ledger,event_id+':header',phase)
    p=select_received_profile(runtime,header)
    if p is None:return dict(header=header,body=None,rx_profile=None,status='HEADER_REJECT')
    body=receive_body(runtime,y[68:],p,snr_db,frame_counter,ledger,event_id+':body',phase,session)
    return dict(header=header,body=body,rx_profile={k:p[k] for k in ('profile_id','family','mode','m','K','q','nominal_rate','k','n','token_count','profile_key')},status=body['status'])
