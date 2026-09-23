"""Complete online CPU endpoint execution of unchanged selected N3060 systems."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml
from tools.audit_n3060_reference_phy import ROOT,HISTORY,read,array_sha


def install_paths():
    from var_comm.frozen_timing_adapters import install_frozen_paths
    install_frozen_paths()
    p=str(ROOT/'experiments/wetok-reencoding-vector-control-r3/src')
    if p not in sys.path:sys.path.append(p)


class Systems:
    def __init__(self,device):
        install_paths()
        from latent_enhancement.runtime import model_paths
        from var_comm.next_scale_prior import load_models
        from wetok_comm.deep_support import FrozenDeepSupport,load_deep_support
        from wetok_comm.native import FrozenWeTok
        from latent_enhancement.runtime import digest
        from vector_control.evaluation import model_registry
        self.device=torch.device(device)
        self.vae,self.var=load_models(model_paths(),self.device)
        self.deep=FrozenDeepSupport(load_deep_support(),self.device)
        self.native=FrozenWeTok(self.device,'both')
        config=yaml.safe_load((ROOT/'experiments/wetok-reencoding-vector-control-r3/configs/study.yaml').read_text())
        reference=yaml.safe_load((ROOT/'experiments/wetok-innovation-r1/configs/study.yaml').read_text())
        base=yaml.safe_load((ROOT/'experiments/wetok-comm-v2-20260912/configs/study.yaml').read_text())
        milestone=read(ROOT/'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-TRAINING/milestones/step_0010000.json')
        # Inference-specific lineage: exact completed R3 source/config snapshot,
        # selected checkpoint and parent assets. Old training settings recursively
        # require a pre-migration Deep helper unrelated to R3 inference.
        for relative,expected in milestone['source_hashes'].items():
            if digest(ROOT/relative.removeprefix('VAR_COMM/'))!=expected:raise RuntimeError('R3 inference source changed: '+relative)
        for key in ('parent_model','parent_milestone'):
            asset=ROOT/'outputs'/reference['parent_training']/reference[key]
            if digest(asset)!=reference[key+'_sha256']:raise RuntimeError('original R3 parent lineage changed')
        unused,models,choices=model_registry(config,reference,base,milestone,self.device)
        self.r3=models['r3__full_grid_prediction_features'].eval().requires_grad_(False)
        self.selected=choices['r3__full_grid_prediction_features'];del unused
        self.models={'vae':self.vae,'var':self.var,'deep':self.deep.model,'native':self.native.codec,'r3__full_grid_prediction_features':self.r3}
        self.policy=read(HISTORY/'POLICIES_001/policies.json')['actions']
        self.before=self.state_hashes()
        digital=read(HISTORY/'DEVELOPMENT_001/completion.json')['frozen_before']
        deep=read(ROOT/'outputs/VAR-PREFIX-JSCC-EVAL-001/completion.json')['frozen_before']
        r3=read(ROOT/'outputs/WETOK-REENCODING-VECTOR-CONTROL-R3-EVALUATION/quality_0010000/completion.json')['frozen_models_before']
        for name,expected in [('vae',digital['vae']),('var',digital['var']),('deep',deep['deep']),('native',r3['native']),('r3__full_grid_prediction_features',r3['r3__full_grid_prediction_features'])]:
            if self.before[name]!=expected:raise RuntimeError('loaded historical model state identity: '+name)

    def state_hashes(self):
        from wetok_comm.training import module_sha256
        from var_comm.next_scale_prior import state_sha256
        for m in self.models.values():
            if m.training or any(p.requires_grad for p in m.parameters()):raise RuntimeError('frozen eval models required')
        return {n:(module_sha256(m) if n in ('native','r3__full_grid_prediction_features') else state_sha256(m)) for n,m in self.models.items()}

    @torch.no_grad()
    def transmit(self,pixels,label,snr,method):
        from var_comm.progressive import transmit_whole
        from var_comm.whole_entropy import encode_prefixes,transmit
        from wetok_comm.native import indices_to_features
        d=self.device
        if method.endswith('_adaptive'):
            family=method.split('_')[0];mode=self.policy[family]['quality'][str(float(snr))]
            x=torch.from_numpy(pixels[None]).to(d).float().div(127.5).sub(1)
            source=[v[0].cpu().numpy() for v in self.vae.img_to_idxBl(x)]
            if family=='raw':return transmit_whole(source,label,mode)
            payload=encode_prefixes(self.vae,self.var,source,label,d,modes=(mode,))[mode]
            return transmit(payload,label,mode)[0]
        cond=torch.full((1,),float(snr),device=d)
        if method=='perceptual_deepjscc':
            x=torch.from_numpy(pixels[None]).float().div(127.5).sub(1).add(1).mul(.5).to(d)
            encoded=self.deep.model.encode(x,cond);normalized,_=self.deep.model.normalize_channel_input(encoded)
            signal=normalized.flatten(1).index_select(1,self.deep.model.active_real_indices).reshape(1,3060,2)
        elif method=='wetok_r3':
            x=torch.from_numpy(pixels[None]).to(d).float().div(255)
            indices=self.native.encode(x)
            signal=self.r3.transmit(indices_to_features(indices),cond)
        else:raise ValueError('unregistered historical method')
        return signal[0].cpu().numpy()

    @torch.no_grad()
    def receive(self,observed,snr,method):
        from var_comm.progressive import receive_whole,complete_image
        from var_comm.whole_entropy import decode_phy,decode_source
        if method=='raw_adaptive':
            result=receive_whole(observed,snr)
            return complete_image(self.vae,self.var,result['prefix'],result['label'],self.device)
        if method=='arithmetic_adaptive':return decode_source(decode_phy(observed,snr),self.vae,self.var,self.device)['image']
        wave=torch.from_numpy(observed[None]).to(self.device);cond=torch.full((1,),float(snr),device=self.device)
        if method=='perceptual_deepjscc':
            flat=wave.new_zeros((1,self.deep.model.native_real_symbols))
            latent=flat.index_copy(1,self.deep.model.active_real_indices,wave.reshape(1,6120))
            image=self.deep.model.decode(latent.reshape(1,*self.deep.layout),cond).clamp(0,1)
        elif method=='wetok_r3':image=self.native.decode(self.r3.receive(wave,cond)['receiver_features'])
        else:raise ValueError('unregistered historical receiver')
        return image[0].cpu().numpy()


@torch.no_grad()
def execute(systems,pixels,label,source_id,snr,seed,method):
    from var_comm.study import seeded_noise
    torch.cuda.synchronize();start=time.perf_counter()
    wave=systems.transmit(pixels,label,snr,method)
    torch.cuda.synchronize();tx_ms=1000*(time.perf_counter()-start)
    energy=float(np.square(wave,dtype=np.float64).sum())
    if wave.shape!=(3060,2) or not np.isfinite(wave).all() or abs(energy-6120)>.01:raise RuntimeError('N3060 actual waveform resource contract')
    noise=seeded_noise(source_id,seed,(3060,2))
    if method.endswith('_adaptive'):observed=wave+noise/np.sqrt(10**(snr/10))
    else:
        # Reproduce original FP32 device arithmetic outside RX; return actual CPU observation.
        x=torch.from_numpy(wave).to(systems.device);z=torch.tensor(noise,device=systems.device,dtype=torch.float32)
        condition=torch.full((1,),float(snr),device=systems.device)
        observed=(x+z*torch.pow(10.,condition/10).rsqrt()).cpu().numpy()
    torch.cuda.synchronize();start=time.perf_counter()
    image=systems.receive(observed,snr,method)
    torch.cuda.synchronize();rx_ms=1000*(time.perf_counter()-start)
    if image.shape!=(3,256,256) or not np.isfinite(image).all() or image.min()<0 or image.max()>1:raise RuntimeError('finite CPU RGB output required')
    return image,wave,observed,{'N':3060,'E':energy,'tx_ms':tx_ms,'rx_ms':rx_ms,'total_ms':tx_ms+rx_ms,
        'standard_noise_sha256':array_sha(noise),'tx_sha256':array_sha(wave),'rx_sha256':array_sha(observed),
        'timing_endpoints':'CPU uint8 RGB -> CPU I/Q; actual CPU noisy I/Q -> CPU float32 RGB; channel outside RX',
        'noise_scope':'same source/seed full N3060 standard noise; original FP32 learned versus float64 digital arithmetic'}
