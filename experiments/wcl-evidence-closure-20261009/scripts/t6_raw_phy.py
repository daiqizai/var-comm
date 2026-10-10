"""N2048 raw12 KEEP frame shell around unchanged original actual PHY.

The original Sionna classes, packet functions, paid Header and raw state parser
are loaded by exact source SHA. This module never loads source images or neural
models. New N2048 catalogue/layout qualification is mandatory for formal use.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,math,os
import numpy as np
import t2_pilot as h
from t6_plan import N,HEADER,BODY,MODULATIONS

PROTOCOL='WCL_T6_N2048_RAW_ACTUAL_V1'
QUAL_SCHEMA='WCL_T6_N2048_RAW_REAL_PHY_QUALIFICATION_V1'
WIRE_SHA='9e9f84c577996cd16d7aba22c0e3bb7eca9e0e50f2bfe2b9ddfd77a27359968d'
array_sha=h.image_sha


def standard_noise(source_id,seed):
    material=h.canonical([PROTOCOL,'standard_noise',str(source_id),int(seed)]).encode()
    key=int.from_bytes(hashlib.sha256(material).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(key)).standard_normal((N,2))


class PublicCatalogue:
    def __init__(self,value,original):
        h.require(value['status']=='T6_ACTUAL_CONSTRUCTOR_ACTION_CATALOGUE_NOT_PHY_QUALIFIED'
            and value['catalogue_sha256']==h.digest(value['profiles'])
            and value['profile_namespace']=='WCL_T6_N2048_RAW_V1','New actual N2048 catalogue required')
        self._profiles=deepcopy(value['profiles']);self.aliases=deepcopy(value['aliases']);self.original=original
        self.digest=value['catalogue_sha256'];self.aliases_digest=h.digest(self.aliases)
        # The unchanged Planner uses its injected backend identity for PHY keys.
        # T6 metadata injected the raw64 adapter for all Q, while q2/q4 retain
        # their original layout implementation identity inside each layout.
        identities={h.digest(p['groups'][0]['layout']['implementation']):p['groups'][0]['layout']['implementation']
            for p in self._profiles if p['groups'][0]['modulation']=='64QAM'}
        h.require(len(identities)==1,'One registered raw64 planning backend identity required')
        self.backend_identity=deepcopy(next(iter(identities.values())))
        h.require(0<len(self._profiles)<4096 and [p['profile_id']for p in self._profiles]==list(range(len(self._profiles))),
            'New consecutive paid profile namespace and one unknown test ID required')
        h.require(len({p['wire_key']for p in self._profiles})==len(self._profiles),'Unique actual wires required')
        ids=set()
        for row in self.aliases:
            h.require(row['candidate_id']not in ids,'Duplicate alias');ids.add(row['candidate_id'])
            p=self.entry(row['profile_id']);h.require(row['wire_key']==p['wire_key']and row['candidate_id']in p['alias_candidate_ids'],'Alias wire differs')
        for p in self._profiles:
            g=p['groups'][0];q=MODULATIONS[g['modulation']]
            h.require(p['N']==2048 and p['G']==1 and p['j']is None and p['body_symbols']==1980
                and p['header_symbols']==68 and p['header_transmitted_bits']==136 and p['header_information_bits']==12
                and p['header_crc_bits']==16 and p['header_tail_bits']==6 and p['header_phy_key']==original.HEADER_KEY,
                'Paid N2048 header/body layout differs')
            h.require(original.canonical(p['m'],p['K'])==(p['m'],p['K']) and p['order']=='raster'
                and p['token_count']==original.PREFIX[p['m']]+p['K'] and g['source_bits']==12*p['token_count']
                and g['information_bits']==g['source_bits']+16 and g['crc_bits']==16 and g['tail_bits']==0,
                'Canonical actual raw12 plus CRC16 required')
            h.require(g['transmitted_bits']==q*g['symbols'] and p['used_body_symbols']==g['symbols']
                and p['idle_symbols']>=0 and p['idle_symbols']+g['symbols']==1980,'Exact body symbol accounting required')
            layout=g['layout'];h.require(layout['k']==g['information_bits']and layout['n']==g['transmitted_bits']
                and layout['layout_id']==h.digest({k:v for k,v in layout.items()if k!='layout_id'})
                and layout['code_blocks']==1 and layout['repetition_bits']==layout['modulation_padding_bits']==0,
                'Actual qualified single-codeblock layout required')
            h.require(p['backend_id']==h.digest(self.backend_identity)
                and h.digest(original.physical_definition(g,self.backend_identity))==g['phy_key'],'Body physical identity differs')
            wire=dict(N=2048,G=1,m=p['m'],K=p['K'],j=None,group_phy_keys=[g['phy_key']],header_phy_key=p['header_phy_key'],
                order='raster',receiver_rule='KEEP_actual_hard_prefix_and_raster_partial_v1',idle_symbols=p['idle_symbols'],
                idle_rule='public_known_QPSK_Es2_no_source_information')
            h.require(p['wire_key']==h.digest(wire) and p['receiver_rule']==wire['receiver_rule'] and p['idle_rule']==wire['idle_rule'],
                'N2048 wire/KEEP protocol differs')
            h.require(('WHOLE'in p['family_memberships'])==(p['K']==0)and'PARTIAL_WITH_WHOLE_FALLBACK'in p['family_memberships'],
                'PARTIAL must include whole and whole may use full-budget protection')
    def verify(self):h.require(h.digest(self._profiles)==self.digest and h.digest(self.aliases)==self.aliases_digest,'Public catalogue mutated')
    def entry(self,pid):
        h.require(type(pid)is int and 0<=pid<len(self._profiles),'Unknown received paid profile ID');return deepcopy(self._profiles[pid])
    def header_map(self):return {str(p['profile_id']):deepcopy(p)for p in self._profiles}


class Runtime:
    def __init__(self,metadata_request,catalogue,bindings):
        self.metadata_request=metadata_request;r=metadata_request;env=h.checked(r['environment_request']);cc=env['cpu_config'];self.root=r['root']
        h.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only N2048 PHY')
        for p,v in bindings.items():h.require(h.sha(p)==v,'Bound source changed: '+p)
        for name,path in [('uep_common',r['original_common']),('profiles',r['original_planner']),
            ('main_raw64_plan_only',r['raw64_planner']),('main_action_space',r['action_core'])]:h.load(path,name,bindings)
        old=h.load(r['original_backend'],'_t6_actual_original_backend',bindings)
        self.legacy=h.load(cc['adapter_config']['original_phy'],'uep_phy',bindings)
        self.packet=h.load(cc['adapter_module'],'main_raw64_packet_adapter',bindings)
        h.require(bindings[cc['original_receiver_module']]==WIRE_SHA,'Original raw receiver/parser SHA differs')
        self.original=h.load(cc['original_receiver_module'],'_t6_original_raw_state_parser',bindings)
        base=old.SionnaBackend('cpu',r['qualification']['path']);self.packet.runtime_flags(base.torch)
        self.backend=self.packet.Backend(base);self.torch=base.torch;self.header=self.legacy.Header(r['root'])
        h.require(not self.torch.cuda.is_initialized(),'Raw qualification/formal PHY may not initialize CUDA')
        self.catalogue=PublicCatalogue(catalogue,self.original);self.profiles=self.catalogue.header_map();self.qualified=False
        h.require(self.catalogue.backend_identity==self.backend.identity64,'Actual raw64 adapter and planning PHY identity differ')
        self.validated=set()
        for p in self.catalogue._profiles:self.check_profile(p)
    def check_profile(self,p):
        h.require(p==self.catalogue.entry(p['profile_id']),'Only public received-profile layouts are legal');g=p['groups'][0]
        if g['phy_key']not in self.validated:
            h.require(self.backend.plan(g['information_bits'],g['transmitted_bits'],MODULATIONS[g['modulation']])==g['layout'],
                'Current real encoder constructor differs from N2048 metadata')
            self.validated.add(g['phy_key'])
        return g
    def transmit(self,profile_id,source_scales,counter):
        h.require(self.qualified,'Complete independent N2048 actual qualification required');p=self.catalogue.entry(profile_id)
        payload=self.original.serialize_raw(source_scales,p)
        return transmit_frame(self,p,payload,counter)
    def receive(self,received,snr,counter,ledger,event,phase):
        h.require(self.qualified,'Complete independent N2048 actual qualification required')
        return receive_frame(self,received,snr,counter,ledger,event,phase)


def transmit_body(rt,p,payload,counter,session=PROTOCOL):
    g=rt.check_profile(p);bits=rt.packet.binary(payload,g['source_bits']);q=MODULATIONS[g['modulation']]
    wave,meta=rt.packet.transmit_packet(rt.backend,rt.legacy,bits[None],g['transmitted_bits'],q,[counter],0,session)
    body=wave[0].cpu().numpy().astype(np.float32)
    h.require(body.shape==(g['symbols'],2)and np.isfinite(body).all(),'Actual body waveform differs')
    return body,dict(meta,source_payload_sha256=array_sha(bits),body_energy=float(np.square(body.astype(np.float64)).sum()),
        body_waveform_sha256=array_sha(body))


def transmit_frame(rt,p,payload,counter):
    h.require(type(counter)is int and counter>=0,'Registered public frame counter required');rt.catalogue.verify()
    header=np.asarray(rt.header.transmit(p['profile_id']),dtype=np.float64);h.require(header.shape==(68,2),'Paid header shape differs')
    body,detail=transmit_body(rt,p,payload,counter);idle=np.ones((p['idle_symbols'],2),np.float64)
    wave=np.concatenate((header,body.astype(np.float64),idle));h.require(wave.shape==(2048,2),'Exactly2048 paid symbols')
    energy=float(np.square(wave).sum());g=p['groups'][0]
    return wave,dict(N=2048,profile_id=p['profile_id'],candidate_id=p['candidate_id'],m=p['m'],K=p['K'],token_count=p['token_count'],
        header_symbols=68,body_symbols=g['symbols'],padding_symbols=p['idle_symbols'],source_bits=g['source_bits'],k=g['information_bits'],
        n=g['transmitted_bits'],q=MODULATIONS[g['modulation']],actual_code_rate=g['information_bits']/g['transmitted_bits'],
        E_frame=energy,rho=energy/4096,header_energy=float(np.square(header).sum()),body_energy=detail['body_energy'],
        padding_energy=float(np.square(idle).sum()),waveform_sha256=array_sha(wave),body=detail,public_frame_counter=counter,session=PROTOCOL)


def receive_header(rt,received,snr,ledger,event,phase):
    y=np.asarray(received,dtype=np.float64);h.require(y.shape==(68,2)and np.isfinite(y).all(),'Actual68 received header required')
    request=dict(protocol=PROTOCOL,received_sha256=array_sha(y),snr_db=float(snr),catalogue_sha256=rt.catalogue.digest)
    return ledger.call(dict(event_id=event,phase=phase,kind='header'),request,
        lambda:rt.original.validate_header(rt.header.receive(y,snr,rt.catalogue.header_map()),rt.catalogue))


def receive_body(rt,received,p,snr,counter,ledger,event,phase,session=PROTOCOL):
    g=rt.check_profile(p);y=np.asarray(received,dtype=np.float32);q=MODULATIONS[g['modulation']]
    h.require(y.shape==(g['symbols'],2)and np.isfinite(y).all(),'Actual received-profile body shape differs')
    request=dict(protocol=PROTOCOL,received_sha256=array_sha(y),snr_db=float(snr),counter=counter,session=session,
        actual_profile_id=p['profile_id'],actual_wire_key=p['wire_key'],phy_key=g['phy_key'],catalogue_sha256=rt.catalogue.digest)
    def decode():
        b=rt.backend;tensor=b.torch.as_tensor(y[None],dtype=b.torch.float32,device=b.device)
        hard,accepted,full=rt.packet.receive_packet(b,rt.legacy,tensor,g['information_bits'],g['transmitted_bits'],q,snr,[counter],0,session)
        payload=rt.original.bits(hard[0],g['source_bits']);information=rt.original.bits(full[0],g['information_bits'])
        h.require(np.array_equal(payload,information[:-16])and bool(rt.legacy.crc_accept_batch(information))==bool(accepted[0]),'Actual hard bits and CRC differ')
        return dict(profile_id=p['profile_id'],phy_key=g['phy_key'],hard_payload=payload.tolist(),decoded_bits=information.tolist(),crc_accept=bool(accepted[0]))
    return ledger.call(dict(event_id=event,phase=phase,kind='body'),request,decode)


def receive_frame(rt,received,snr,counter,ledger,event,phase):
    rt.catalogue.verify();h.require(type(counter)is int and counter>=0 and math.isfinite(snr),'Finite public channel metadata required')
    y=np.asarray(received,dtype=np.float64);h.require(y.shape==(2048,2)and np.isfinite(y).all(),'Full2048 received observation required')
    header=receive_header(rt,y[:68],snr,ledger,event+'.header',phase);body=None
    if header['header_ok']:
        p=rt.catalogue.entry(header['profile_id']);g=p['groups'][0]
        body=receive_body(rt,y[68:68+g['symbols']],p,snr,counter,ledger,event+'.body',phase)
    state=rt.original.present_actual(header,body,rt.catalogue)
    return dict(state,status='T6_N2048_RAW_KEEP_RECEPTION_COMPLETE',N=2048,header=header,body=body,
        received_sha256=array_sha(y),public_frame_counter=counter,catalogue_sha256=rt.catalogue.digest,
        logical_packet_calls=1+int(header['header_ok']),header_event=event+'.header',body_event=event+'.body'if header['header_ok']else None)


def create_runtime(qualification):
    done=h.read(qualification);h.require(done['status']=='PASS'and done['schema']==QUAL_SCHEMA,'Actual new N2048 PHY qualification required')
    r=h.checked(done['request']);h.require(done['packet_decode_count']==r['packet_cap']and done['ledger']==dict(total=r['packet_cap'],unresolved=0,cap=r['packet_cap']),
        'Complete finite paid header/body qualification required')
    for p,v in done['outputs'].items():h.require(h.sha(p)==v,'Actual qualification output changed')
    meta=h.checked(r['metadata_request']);cat=h.checked(r['candidate_catalogue']);rt=Runtime(meta,cat,r['source_bindings'])
    h.require(rt.catalogue.digest==done['catalogue_sha256']and rt.backend.identity==done['original_backend_identity']
        and rt.backend.identity64==done['raw64_backend_identity'],'Qualified actual backend/catalogue differs')
    rt.qualified=True;return rt
