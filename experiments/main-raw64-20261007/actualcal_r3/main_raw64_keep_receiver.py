"""Prepared mixed catalogue/KEEP receive core; no channel, launcher or ledger.

Only received samples/public counters enter RX. A separately registered caller
must supply the charge(event,request,callback) gate for each actual header/body.
The current qualification/proxy ledger deliberately does not implement these
actual-image phases and cannot be silently substituted for that future gate.
"""
from copy import deepcopy
import math
import numpy as np
import main_raw64_plan_only as p
import main_raw64_packet_adapter as packet

PHASES=('actual_calibration','development','raw_holdout','online_PHY')

class MixedPublicCatalogue:
    def __init__(self,completion,legacy_profiles,legacy_aliases,*,legacy_backend_identity,original_receiver):
        p.require(len(completion['profiles'])==433 and len(completion['aliases'])==523,'Registered433/523 domain required')
        p.require(completion['profiles'][:270]==legacy_profiles and completion['aliases'][:329]==legacy_aliases,
            'Original270 public IDs and329 aliases must remain exact')
        # The original class independently validates all legacy shapes/aliases.
        original_receiver.PublicCatalogue(legacy_profiles,legacy_aliases,catalogue_digest=p.identity(legacy_profiles),
            aliases_digest=p.identity(legacy_aliases),backend_identity=legacy_backend_identity)
        packet.validate_catalogue(completion,legacy_backend_identity)
        profiles=completion['profiles'];aliases=completion['aliases']
        self._profiles=deepcopy(profiles);self._aliases=deepcopy(aliases)
        self.digest=p.identity(profiles);self.aliases_digest=p.identity(aliases)
        self.legacy_backend_identity=deepcopy(legacy_backend_identity);self.original=original_receiver
        p.require(len({r['wire_key'] for r in profiles})==433,'Duplicate physical public wire')
        p.require([r['wire_key'] for r in profiles[270:]]==sorted(r['wire_key'] for r in profiles[270:]),'Appended wire IDs must use frozen lexical order')
        mapped={r['wire_key']:[] for r in profiles};seen=set()
        for row in aliases:
            pid=row['profile_id'];aid=row['candidate_id']
            p.require(type(pid)is int and 0<=pid<433 and original_receiver.sha(aid) and aid not in seen,'Alias identity/range changed')
            seen.add(aid);profile=profiles[pid]
            p.require(row['wire_key']==profile['wire_key'] and row['representative_candidate_id']==profile['candidate_id']
                and row['modulation']==profile['groups'][0]['modulation'],'Alias no longer names its exact wire')
            mapped[profile['wire_key']].append(aid)
        for profile in profiles:
            m,K=profile['m'],profile['K'];g=profile['groups'][0]
            p.require(type(m)is int and 1<=m<=10 and original_receiver.canonical(m,K)==(m,K),'Full supported canonical source domain required')
            layout=g['layout'];q=packet.Q[g['modulation']]
            p.require(profile['token_count']==p.PREFIX[m]+K and layout['k']==g['information_bits']
                and layout['n']==g['transmitted_bits']==q*g['symbols'] and layout['code_blocks']==1
                and layout['k_ldpc']-layout['k_filler']==layout['k'] and layout['modulation_padding_bits']==0
                and layout['power_protocol']=='fixed_constellation_average_Es2','Public raw message/codeblock differs')
            p.require(profile['backend_id']==p.identity(layout['implementation']) and profile['body_symbols']==956
                and profile['used_body_symbols']==g['symbols'] and type(profile['idle_symbols'])is int
                and profile['idle_symbols']>=0 and profile['header_information_bits']==12 and profile['header_crc_bits']==16
                and profile['header_tail_bits']==6 and profile['header_transmitted_bits']==136,'Paid body/header identity differs')
            wire=dict(N=1024,G=1,m=m,K=K,j=None,group_phy_keys=[g['phy_key']],header_phy_key=packet.HEADER_KEY,
                order='raster',receiver_rule='KEEP_actual_hard_prefix_and_raster_partial_v1',idle_symbols=profile['idle_symbols'],
                idle_rule='public_known_QPSK_Es2_no_source_information')
            p.require(profile['wire_key']==p.identity(wire) and profile['idle_rule']==wire['idle_rule'],'Public waveform definition differs')
            ids=sorted(mapped[profile['wire_key']]);p.require(ids==sorted(profile['alias_candidate_ids'])
                and profile['candidate_id']==profile['stable_id']==min(ids),'Alias representative changed')
    def verify(self):
        p.require(p.identity(self._profiles)==self.digest and p.identity(self._aliases)==self.aliases_digest,'Catalogue mutated after admission')
    def entry(self,pid):
        p.require(type(pid)is int and 0<=pid<433,'Unknown actual received profile')
        return deepcopy(self._profiles[pid])
    def header_map(self):return {str(x['profile_id']):deepcopy(x) for x in self._profiles}
    def definition(self,pid):
        g=self._profiles[pid]['groups'][0]
        return self.original.physical_definition(g,g['layout']['implementation'])

class Receiver:
    def __init__(self,catalogue,*,backend,legacy_phy,header,charge):
        p.require(isinstance(catalogue,MixedPublicCatalogue),'Complete mixed public catalogue required')
        p.require(backend.identity==catalogue.legacy_backend_identity and backend.identity64==p.new_identity(backend.identity,p.sha(p.__file__))
            and str(backend.device)=='cpu','Exact legacy/new CPU backend identity required')
        p.require(callable(charge) and callable(header.receive),'Registered metering gate and original paid header required')
        self.catalogue=catalogue;self.backend=backend;self.legacy_phy=legacy_phy;self.header=header;self.charge=charge
        self.original=catalogue.original;self.validated=set()
    def receive(self,received,snr_db,public_counter,*,phase,event_prefix):
        self.catalogue.verify();p.require(phase in PHASES and isinstance(event_prefix,str) and event_prefix,'Explicit actual-image phase/event')
        p.require(self.backend.identity==self.catalogue.legacy_backend_identity and self.backend.identity64==p.new_identity(self.backend.identity,p.sha(p.__file__)),
            'Admitted backend identity changed')
        p.require(type(snr_db)in(int,float) and math.isfinite(snr_db) and type(public_counter)is int and public_counter>=0,'Public SNR/counter invalid')
        y=np.asarray(received,dtype=np.float64);p.require(y.shape==(1024,2) and np.isfinite(y).all(),'Full1024 float64 received frame required')
        base=dict(schema='MAIN_RAW64_ACTUAL_RX_REQUEST_V1',received_sha256=self.original.array_sha(y),
            codebook_sha256=self.catalogue.digest,aliases_sha256=self.catalogue.aliases_digest,
            snr_db=float(snr_db),public_frame_counter=public_counter,N=1024,body_session='actual-body',body_group=0)
        records=[]
        def metered(kind,key,request,callback):
            event=dict(phase=phase,kind=kind,phy_key=key,event_id=event_prefix+'.'+kind)
            used=False
            def once():
                nonlocal used
                p.require(not used,'Actual decode callback is single-use');used=True
                return callback()
            result=self.charge(event,request,once)
            records.append(dict(event_id=event['event_id'],kind=kind,phy_key=key,request_sha256=p.identity(request),result_sha256=p.identity(result)))
            return result
        def header_decode():return self.original.validate_header(self.header.receive(y[:68],snr_db,self.catalogue.header_map()),self.catalogue)
        header=metered('header',packet.HEADER_KEY,dict(base,component='header'),header_decode)
        self.original.validate_header(header,self.catalogue);body=None
        if header['header_ok']:
            profile=self.catalogue.entry(header['profile_id']);g=profile['groups'][0];q=packet.Q[g['modulation']]
            if g['phy_key'] not in self.validated:
                p.require(self.backend.plan(g['information_bits'],g['transmitted_bits'],q)==g['layout'],
                    'Actual received-ID encoder layout differs from sealed metadata')
                self.validated.add(g['phy_key'])
            request=dict(base,component='body',actual_profile_id=profile['profile_id'],actual_wire_key=profile['wire_key'],
                header_result_sha256=p.identity(header),physical_definition_sha256=p.identity(self.catalogue.definition(profile['profile_id'])),
                body_received_float32_sha256=self.original.array_sha(y[68:68+g['symbols']].astype(np.float32)))
            def body_decode():
                b=self.backend;tensor=b.torch.as_tensor(y[None,68:68+g['symbols']],dtype=b.torch.float32,device=b.device)
                hard,accepted,decoded=packet.receive_packet(b,self.legacy_phy,tensor,g['information_bits'],g['transmitted_bits'],q,
                    snr_db,[public_counter],0,'actual-body')
                p.require(np.asarray(hard).shape==(1,g['source_bits']) and np.asarray(decoded).shape==(1,g['information_bits'])
                    and np.asarray(accepted).shape==(1,) and np.asarray(accepted).dtype==np.dtype(bool),'Actual decoder return schema differs')
                payload=self.original.bits(hard[0],g['source_bits']);full=self.original.bits(decoded[0],g['information_bits'])
                p.require(np.array_equal(payload,full[:-16]) and bool(self.legacy_phy.crc_accept_batch(full))==bool(accepted[0]),'Returned hard bits/CRC disagree')
                value=dict(profile_id=profile['profile_id'],phy_key=g['phy_key'],hard_payload=payload.tolist(),decoded_bits=full.tolist(),crc_accept=bool(accepted[0]))
                self.original.present_actual(header,value,self.catalogue);return value
            body=metered('body',g['phy_key'],request,body_decode)
        state=self.original.present_actual(header,body,self.catalogue)
        return dict(state,status='MAIN_RAW64_KEEP_RECEPTION_COMPLETE',N=1024,header=header,body=body,
            received_sha256=base['received_sha256'],codebook_sha256=self.catalogue.digest,aliases_sha256=self.catalogue.aliases_digest,
            public_frame_counter=public_counter,packet_events=records,packet_event_ids=[r['event_id'] for r in records],
            logical_packet_calls=len(records),image_reconstruction_complete=False,actual_new_call_count_authority='new_registered_MAIN_RAW64_ledger')

def transmit_frame(catalogue,backend,legacy_phy,header,profile_id,payload,public_counter):
    """TX shell only; no randomness. Original caller supplies matched noise later."""
    catalogue.verify();p.require(type(public_counter)is int and public_counter>=0,'Public counter invalid')
    p.require(backend.identity==catalogue.legacy_backend_identity and backend.identity64==p.new_identity(backend.identity,p.sha(p.__file__)),
        'Exact admitted TX backend identity required')
    profile=catalogue.entry(profile_id);g=profile['groups'][0]
    payload=packet.binary(payload,g['source_bits']);q=packet.Q[g['modulation']]
    p.require(backend.plan(g['information_bits'],g['transmitted_bits'],q)==g['layout'],'TX encoder layout differs from sealed metadata')
    head=np.asarray(header.transmit(profile_id),dtype=np.float64);p.require(head.shape==(68,2),'Paid header shape changed')
    body,detail=packet.transmit_packet(backend,legacy_phy,payload[None],g['transmitted_bits'],q,[public_counter],0,'actual-body')
    waves=[head,body[0].cpu().numpy().astype(np.float64)]
    if profile['idle_symbols']:waves.append(np.ones((profile['idle_symbols'],2),dtype=np.float64))
    wave=np.concatenate(waves);p.require(wave.shape==(1024,2),'Exactly1024 paid symbols including idle')
    return wave,dict(N=1024,E=float(np.square(wave).sum()),header_uses=68,body_uses=g['symbols'],idle_uses=profile['idle_symbols'],
        groups=[detail],waveform_sha256=catalogue.original.array_sha(wave),profile_id=profile_id,frame_counter=public_counter)
