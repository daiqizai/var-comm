"""Original raw whole/partial online paths, sharing the T4 native runtime.

No image/output cache; no old ledger writer. Only bound audit file hashes are
memoized after verification outside timing. The original PHY/KEEP is retained.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import t1_phy as phy

class RawEndpoint:
    def __init__(self,r,native,helper,baseline,method,reservations):
        phy.require(method in ('RAW64_WHOLE','RAW64_PARTIAL'),'Original raw anchor only')
        self.r,self.native,self.t,self.method=r,native,native.torch,method
        self.points=r['raw_points'][method];bound=r['legacy']['bindings'];load=helper.load
        cfg=helper.descriptor(r['legacy']['raw_config']);a=cfg['adapter_config']
        self.plan=load(a['plan_module'],'main_raw64_plan_only',bound)
        self.packet=load(cfg['adapter_module'],'main_raw64_packet_adapter',bound)
        load(a['original_common'],'uep_common',bound);load(a['original_planner'],'profiles',bound)
        self.legacy=load(a['original_phy'],'uep_phy',bound)
        old=load(a['original_backend'],'_wcl_t4_raw_ldpc',bound).SionnaBackend('cpu',a['legacy_qualification'])
        phy.require(old.qualified,'Original qualified raw LDPC required')
        self.backend=self.packet.Backend(old);self.header=self.legacy.Header(r['root'])
        original=load(cfg['original_receiver_module'],'_wcl_t4_raw_wire',bound)
        self.rxmod=load(cfg['receiver_module'],'_wcl_t4_raw64_wire',bound)
        self.catalogue=self.rxmod.MixedPublicCatalogue(helper.read(a['catalogue']),helper.read(cfg['profiles']),
            helper.read(cfg['aliases']),legacy_backend_identity=self.backend.identity,original_receiver=original)
        self.source=load(r['legacy']['source_driver'],'_wcl_t4_raw_source',bound)
        self.timing=load(r['legacy']['timing_module'],'_wcl_t4_raw_encoder',bound)
        self.noise_module=load(r['legacy']['development_plan_module'],'_wcl_t4_raw_noise',bound)
        pair=load(r['legacy']['pair_module'],'_wcl_t4_raw_pair',bound)
        self.pair=pair;self.renderer=pair.NativePairBackend(native,self.source.render_received,baseline['P_native_runtime'])
        self.receiver=self.rxmod.Receiver(self.catalogue,backend=self.backend,legacy_phy=self.legacy,
            header=self.header,charge=reservations.call)
        self.original=original
        # The unchanged raw shell hashes its planner file inside every TX/RX.
        # Verify it here; the timed path uses exactly that already checked value.
        # No numerical, metadata, receiver, or codec operation is memoized.
        self.audit_hash_cache={str(Path(self.plan.__file__).resolve()):phy.sha(self.plan.__file__)}
        def verified_hash(path):
            key=str(Path(path).resolve())
            phy.require(key in self.audit_hash_cache,'Unregistered timed audit-file read: '+key)
            return self.audit_hash_cache[key]
        self.plan.sha=verified_hash
        for snr,p in self.points.items():
            actual=self.catalogue.entry(p['profile_id'])
            phy.require(actual['m']==p['m'] and actual['K']==p['K'] and actual['wire_key']==p['wire_key'],'Original raw policy differs')
            if method=='RAW64_WHOLE':phy.require(actual['K']==0,'Whole anchor must remain complete-scale')
        self.source_attempts=[];self.last_tokens=None;self.last_tx=None;self.last_rx=None
    def counter_for(self,snr,index):
        return self.noise_module.frame_counter(self.points[str(snr)]['development_slot'],index,6201)
    def noise(self,wave,source_id,snr):
        return wave+self.noise_module.standard_noise(source_id,snr,6201)*10**(-float(snr)/20)
    def encode(self,pixels,snr,counter,meter):
        self.counter=counter;p=self.catalogue.entry(self.points[str(snr)]['profile_id'])
        with self.t.no_grad():
            tokens=meter.call('TX_visual_encoding',lambda:self.timing.native_encoder(self.native,pixels))
            payload=meter.call('TX_raw12',lambda:self.original.serialize_raw(self.source.split_tokens(tokens),p))
            wave,meta=meter.call('TX_header_LDPC_modulation',lambda:self.rxmod.transmit_frame(
                self.catalogue,self.backend,self.legacy,self.header,p['profile_id'],payload,counter))
        g=p['groups'][0];self.last_tokens=tokens
        self.last_tx=dict(meta,m=p['m'],K=p['K'],raw_source_bits=g['source_bits'],k=g['information_bits'],
            n=g['transmitted_bits'],q=self.packet.Q[g['modulation']],known_information_padding_bits=0,
            frame_padding_symbols=p['idle_symbols'],header_symbols=68,body_symbols=g['symbols'],
            E_frame=meta['E'],rho=meta['E']/2048,wire_key=p['wire_key'])
        return wave
    def receive(self,observed,snr,meter,event_id):
        with self.t.no_grad():
            actual=meter.call('RX_PHY_KEEP',lambda:self.receiver.receive(observed,float(snr),self.counter,
                phase='online_PHY',event_prefix=event_id))
            state=self.pair.received_view(actual);self.gray=actual['gray']
            p=None if self.gray else self.catalogue.entry(actual['rx_profile_id'])
            self.last_rx=dict(status=actual['source_status'],rx_profile=None if p is None else
                {k:p[k] for k in ('profile_id','m','K','wire_key')},header=actual['header'],body=actual['body'],
                body_crc_accept=actual['body_crc_accept'],receiver_state=state)
            if self.gray:return np.full((3,256,256),.5,np.float32)
            # KEEP uses the actual hard token candidates even when CRC rejects.
            return meter.call('RX_VAR_and_Dc',lambda:self.renderer.var(deepcopy(state)))
    def finish(self):
        for path,expected in self.audit_hash_cache.items():
            phy.require(phy.sha(path)==expected,'Memoized audit source changed during timing')
