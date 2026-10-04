"""Original frozen own receivers, with TX strictly outside the timed callable."""
from __future__ import annotations
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import numpy as np
import own_controls_common as old
import own_controls_cost_common as k
import external_eval_common as c


class OwnReceivers:
    def __init__(self,root,handler):
        self.root=Path(root);sys.path.insert(0,str(self.root/'experiments/var-latent-enhancement-20260917/mechanisms/src'))
        from own_controls import Runner
        runner=Runner(root,old.HERE/'own_controls_protocol.json')
        self.bindings={**runner.bindings,**runner.require_stage('calibrate')}
        # Only provenance JSON is read here; no development RGB or predictions.
        inventory=old.old_source_inventory(root);self.bindings.update(inventory['bindings'])
        reg=c.read(self.root/'results/unified_metrics_20261002/metrics_registration.json')
        prior=reg['frozen_policy_bindings'];c.verify(prior);self.bindings.update(prior)
        from own_controls_native import Native
        self.native=Native(root,handler);v=self.native;self.torch=v.torch;self.loaded=v.loaded
        self.data=v.assets.load_calibration()
        original,bindings=old.original_registration(root,'calibration');self.bindings.update(bindings)
        self.population_proof=old.data_proof(self.data,original,'calibration')
        # Pixel and latent cache bindings are nested in the original registration.
        def add_nested(value):
            for p,h in value.items():
                if isinstance(h,dict):add_nested(h)
                else:self.bindings[p]=h
        add_nested(self.population_proof['data_bindings'])
        self.sources=[]
        for i,r in enumerate(self.data['records'][:4]):
            rgb=np.asarray(r['pixels'],dtype=np.float32)/np.float32(255)
            self.sources.append(dict(k.source_record(i,r,rgb),rgb=rgb))
        self.new_policy=c.read(runner.result/'N2048_policy.json')
        self.old_policy=c.read(runner.paths['original_result']/'m1_policy.json')
        self.digital_policy=c.read(self.root/'results/extreme_bandwidth_20261001_R1_N1024/digital_selected_policy.json')
        load=v.replay.load_file_module
        self.oldphy=load('_cost_original_partial_phy',runner.paths['source']/'partial_phy.py')
        self.oldrx=load('_cost_original_partial_rx',runner.paths['source']/'partial_receiver.py',{'partial_phy':self.oldphy})
        self.oldm1=load('_cost_original_m1',runner.paths['source']/'m1_runner.py',
                       {'common':v.common,'partial_phy':self.oldphy,'partial_receiver':self.oldrx})
        from latent_enhancement_b.common import scale_statistics
        from token_efficiency import execution
        self.execution=execution;scale=scale_statistics(self.loaded['device'])
        train=load('_cost_p1024_train',self.root/'experiments/extreme-bandwidth-20261001-N1024/train.py')
        p1024=self.root/'outputs/EXTREME-BW-20261001-R1-N1024/training/p1024_2026093001/selected_P1024.json'
        k.require(prior.get(str(p1024))==c.sha(p1024),'P1024 differs from original selected replay')
        from step0_reference_prepare import admitted
        _,receipt,proof=admitted(root,'FINAL_P2048_P3060');self.bindings.update(proof)
        rp=self.root/'results/historical_metrics_r2_20261003/FINAL_P2048_P3060/registration.json'
        k.require(receipt['bindings'].get(str(rp))==c.sha(rp),'Selected P2048 admission changed')
        self.bindings[str(rp)]=c.sha(rp);pb=c.read(rp)['original_bindings'];c.verify(pb);self.bindings.update(pb)
        p2048=self.root/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/training/P2048_seed2026092304/selected_P2048.json'
        k.require(pb.get(str(p2048))==c.sha(p2048),'P2048 selected seed/model changed')
        self.models={};self.model_identity={}
        for n,p,loader in ((1024,p1024,train.load_for_evaluation),(2048,p2048,execution.load_selected_budget)):
            model,meta=loader(p,scale,self.loaded['device'])
            k.require(meta['decoder_sha256']==self.loaded['identity']['models']['decoder'],'Continuous P decoder differs from frozen Dc')
            self.models[n]=model;self.model_identity[n]=v.assets.old.b.state_sha256(model)
            self.bindings[str(p)]=c.sha(p);self.bindings[meta['checkpoint']]=c.sha(meta['checkpoint'])
        # Legacy imports install handlers; restore this owner's boundary handler.
        signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
        c.verify(self.bindings)

    def action(self,n,s,method):
        if method.startswith('P'):return None
        if n==1024 and method=='D_U_QPSK':
            return self.native.common.legacy.phy.Action(int(self.digital_policy['levels'][str(s)]['QPSK']['U']['action_m']),'QPSK')
        if n==1024:
            rows=[r for r in self.old_policy['cells'] if r['N']==1024 and r['phy_family']=='QPSK'
                  and r['snr_db']==s and r['method']=='entropy']
            k.require(len(rows)==1,'Frozen N1024 M1 policy missing')
            return self.oldphy.Action(**rows[0]['action'])
        rows=[r for r in self.new_policy['cells'] if r['snr_db']==s and r['method']==method]
        k.require(len(rows)==1,'Registered new N2048 policy missing')
        return self.native.action(rows[0]['action'])

    def prepare(self,spec):
        """Return waveform and public-policy metadata; never pass TX truth to RX."""
        i,n,s,method=spec['source_index'],spec['N'],spec['snr_db'],spec['method']
        record=self.data['records'][i];v=self.native;l=self.loaded;a=self.action(n,s,method)
        with self.torch.no_grad():
            if method.startswith('P'):
                wave=self.models[n].transmit(self.data['F'][i:i+1].to(l['device']))[0].cpu().numpy()
                observed=self.execution.apply_channel(wave,s,record['image_id'],4101,SimpleNamespace(family='continuous',N=n))
                policy=dict(m='',K='',order='continuous')
            else:
                scales=v.common.split_tokens(self.data['T'][i].numpy())
                if n==1024 and method=='D_U_QPSK':
                    phy=v.common.legacy.phy
                    wave,_=phy.transmit(scales,int(record['class_index']),a)
                    observed=phy.apply_channel(wave,s,record['image_id'],4101,'QPSK')
                    policy=dict(m=a.m,K=0,order='whole',paid_class_bits=10,class_used_at_receiver=False)
                else:
                    m1=self.oldm1 if n==1024 else v.m1
                    wave,_=m1.waveform(l,scales,a,{})
                    observed=wave+m1.noise_for(record,n,4101)*10.**(-float(s)/20)
                    policy=dict(m=a.m,K=a.q,order=a.order,paid_class_bits=0,class_used_at_receiver=False)
        return np.asarray(wave),np.asarray(observed),policy

    def receive(self,observed,n,s,method):
        """Only actual observation and public budget/SNR/method enter this boundary."""
        v=self.native;l=self.loaded
        with self.torch.no_grad():
            if method.startswith('P'):
                image,event=self.execution.receive(observed,s,SimpleNamespace(family='continuous',N=n),
                    l['vae'],l['var'],l['decoder'],l['device'],self.models[n])
                return image,dict(header_accepted='not_applicable',body_crc_accepted='not_applicable',partial_used=False)
            if n==1024 and method=='D_U_QPSK':
                event=v.common.legacy.phy.receive(observed,s,'QPSK')
                latent=None if not event['header_ok'] else v.common.legacy.complete_latent(l['vae'],l['var'],event['prefix'],1000,l['device'])
                partial=False
            else:
                phy,rx=(self.oldphy,self.oldrx) if n==1024 else (v.phy,v.receiver)
                event=phy.receive(observed,s,n,'QPSK')
                result=rx.complete_partial(l['vae'],l['var'],event,l['device'])
                latent=result['fhat'];partial=bool(result['diagnostics'].get('partial_used',False))
            image=np.full((3,256,256),.5,np.float32) if latent is None else l['decoder'](latent.to(l['device']))[0].cpu().numpy()
            return image,dict(header_accepted=bool(event['header_ok']),body_crc_accepted=bool(event['body_crc_ok']),partial_used=partial)

    def check(self):
        self.native.common.check();self.native.frozen()
        for n,model in self.models.items():
            k.require(self.native.assets.old.b.state_sha256(model)==self.model_identity[n]
                      and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen continuous P changed')
