"""ARPC paid profile/container adapter over the unchanged single956 PHY.

RX takes observations, public counter/session and the full public catalogue.
It has no source image, desired endpoint, transmitted payload or error oracle.
"""
from fractions import Fraction
import hashlib,math,pathlib,sys
import numpy as np
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
OLD=R/'experiments/entropy-partial-mismatch-20261010/scripts';sys.path.insert(0,str(OLD));import ep_phy as ep
SESSION='ARPC_PAID_NATIVE256_SINGLE956_V1'
SIDES=(1,2,4,6,8,12,16)
def templates():
    rows=[]
    for endpoint in range(8):
        for q in [2,4,6]:
            for rate in ep.RATES:
                r=Fraction(rate);n=956*q;k=n*r.numerator//r.denominator
                rows.append({'profile_id':len(rows),'family':'ARPC_NATIVE256','mode':'null' if endpoint==0 else 'native_prefix',
                  'endpoint':endpoint,'q':q,'nominal_rate':rate,'k':k,'n':n,'N':1024,'header_symbols':68,'body_symbols':956,
                  'source_capacity_bits':k-29,'length_bits':13,'crc_bits':16,'frame_padding_symbols':0,
                  'token_blocks':sum(s*s for s in SIDES[:endpoint])})
    assert len(rows)==96 and all(p['profile_id']<4096 for p in rows);return rows
def pack(payload,profile):
    assert isinstance(payload,bytes)
    bits=np.unpackbits(np.frombuffer(payload,dtype=np.uint8));capacity=profile['k']-29
    assert (profile['mode']=='null' and len(bits)==0) or (profile['mode']=='native_prefix' and 0<len(bits)<=capacity)
    assert len(bits)<=8191
    protected=np.concatenate([ep.integer_bits(len(bits),13),bits,np.zeros(capacity-len(bits),np.uint8)])
    out=ep.append_crc(protected);assert len(out)==profile['k'];return out
def parse(decoded,profile):
    a=ep.binary(decoded);assert len(a)==profile['k'];length=ep.bits_integer(a[:13]);capacity=profile['k']-29
    base={'decoded_bits':a.tolist(),'hard_bits_sha256':ep.array_sha(a),'declared_length':length,'crc_accepted':ep.crc_ok(a),'parser_accepted':False,'payload':None}
    if not base['crc_accepted']:return dict(base,status='CRC_REJECT')
    if length>capacity or length%8 or np.any(a[13+length:-16]):return dict(base,status='BODY_INVALID')
    if profile['mode']=='null':
        return dict(base,status='NULL_PROFILE',parser_accepted=True,payload=b'') if length==0 else dict(base,status='BODY_INVALID')
    if profile['mode']!='native_prefix' or length==0:return dict(base,status='BODY_INVALID')
    return dict(base,status='PAYLOAD_PARSED',parser_accepted=True,payload=np.packbits(a[13:13+length]).tobytes())
def fallback(lengths,target,capacity):
    assert type(target) is int and 1<=target<=7 and set(lengths)==set(range(1,8))
    return max([e for e in range(1,target+1) if lengths[e]<=capacity],default=0)
class Runtime:
    def __init__(self):
        # The old Runtime's360-entry catalogue/457-call factory is deliberately not used.
        # Only its unchanged backend, mapping and convolutional header are composed.
        prior=R/'experiments/prior-aware-grouped-mcs-v1';raw=R/'experiments/main-raw64-20261007'
        ep.load_exact(prior/'uep_common.py','uep_common');ep.load_exact(prior/'profiles.py','profiles')
        old=ep.load_exact(prior/'ldpc_backend.py','_arpc_original_ldpc')
        self.legacy=ep.load_exact(prior/'uep_phy.py','uep_phy')
        ep.load_exact(raw/'main_raw64_plan_only.py','main_raw64_plan_only')
        self.mapping=ep.load_exact(raw/'main_raw64_packet_adapter.py','main_raw64_packet_adapter')
        reference=R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json'
        assert ep.sha(reference)==ep.LEGACY_QUALIFICATION_SHA
        self.backend=ep.Backend(old.SionnaBackend('cpu',str(reference)));self.torch=self.backend.torch
        t=self.torch;t.set_num_threads(2);t.set_num_interop_threads(1)
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;t.backends.cudnn.benchmark=False;t.use_deterministic_algorithms(True)
        self.header=self.legacy.Header(R);rows=[]
        for p in templates():
            layout=self.backend.plan(p['k'],p['n'],p['q']);row=dict(p,layout=layout,layout_id=layout['layout_id'])
            row['profile_key']=ep.digest(row);rows.append(row)
        self.catalogue={'protocol':SESSION,'profiles':rows,'backend_identity':self.backend.identity}
        self.catalogue['catalogue_sha256']=ep.digest(self.catalogue);self.profiles={str(p['profile_id']):p for p in rows}
    def profile(self,endpoint,q,rate):
        a=[p for p in self.profiles.values() if (p['endpoint'],p['q'],p['nominal_rate'])==(endpoint,q,rate)];assert len(a)==1;return a[0]
    def transmit(self,payload,profile_id,counter):
        p=self.profiles[str(profile_id)];info=pack(payload,p);b=self.backend;t=b.torch
        encoded=b.encode(info[None],p['n'],p['q']);assert tuple(encoded.shape)==(1,p['n'])
        mask=t.as_tensor(ep.scramble_mask(p['n'],counter,SESSION)[None],device='cpu');coded=(encoded+mask)%2
        body=ep.as_numpy(self.mapping.modulate(coded,p['q'],t,self.legacy))[0].astype(np.float32)
        head=np.asarray(self.header.transmit(profile_id),dtype=np.float64);assert body.shape==(956,2) and head.shape==(68,2)
        frame=np.concatenate([head,body.astype(np.float64)]);energy=float(np.square(frame).sum())
        return frame,{'profile_id':profile_id,'profile_key':p['profile_key'],'endpoint':p['endpoint'],'q':p['q'],'k':p['k'],'n':p['n'],
          'payload_bits':8*len(payload),'information_sha256':ep.array_sha(info),'frame_sha256':ep.array_sha(frame),'counter':counter,
          'N':1024,'header_symbols':68,'body_symbols':956,'Es':2,'energy':energy,'rho':energy/2048,'session':SESSION}
    def receive(self,observed,snr,counter,ledger,key):
        y=np.asarray(observed,dtype=np.float64);assert y.shape==(1024,2) and np.isfinite(y).all() and math.isfinite(float(snr))
        header=ledger.call('header',key,lambda:self.header.receive(y[:68],float(snr),self.profiles))
        if not header['header_ok']:return {'header':header,'profile':None,'body':None,'status':'HEADER_REJECT'}
        p=self.profiles[str(header['profile_id'])]
        def decode():
            t=self.torch;x=t.as_tensor(y[68:][None],dtype=t.float32,device='cpu')
            logits=self.mapping.demap(x,float(snr),p['q'],t,self.legacy)
            mask=t.as_tensor(ep.scramble_mask(p['n'],counter,SESSION)[None],device='cpu');logits=logits*(1-2*mask.to(logits.dtype))
            hard=ep.as_numpy(self.backend.decode(logits,p['k'],p['n'],p['q']));assert hard.shape==(1,p['k'])
            return parse(hard[0].astype(np.uint8),p)
        body=ledger.call('body',key,decode)
        return {'header':header,'profile':{k:p[k] for k in ['profile_id','profile_key','endpoint','q','nominal_rate','k','n','mode']},'body':body,'status':body['status']}
