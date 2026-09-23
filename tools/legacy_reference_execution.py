"""Adapters for existing N4084 methods; shared real quality/timing execution."""
import math
import time
import numpy as np

METHODS=('full_tx_control','light_tx','original_1024_Dc','raw_adaptive_m789_Dc','arithmetic_adaptive_m789_Dc','calibration_frozen_resource_lookup')

def selected_endpoint(method,snr,policy,digital):
    if method not in METHODS:raise ValueError('unregistered legacy method')
    if method=='calibration_frozen_resource_lookup':
        endpoint=policy['actions'][str(float(snr))]
        if endpoint not in ('m8_plus_latent_1024','m8_plus_latent_512_fold_N4084'):raise ValueError('new lookup action requires explicit adapter')
        return endpoint
    if method.endswith('_adaptive_m789_Dc'):
        family=method.split('_')[0];mode=digital['actions'][family]['4084']['Dc'][str(float(snr))]['quality']
        if mode not in (7,8,9):raise ValueError('legacy policy mode domain')
        return f'{family}_N4084_m{mode}_Dc'
    return 'm8_plus_latent_1024' if method=='original_1024_Dc' else method

def validate_waveform(wave):
    wave=np.asarray(wave)
    if wave.shape!=(4084,2) or not np.isfinite(wave).all():raise ValueError('actual N4084 finite waveform required')
    energy=float(np.square(wave,dtype=np.float64).sum())
    if abs(energy-8168)>.01:raise ValueError('strict per-frame E8168 required')
    return energy

def folded_execute(record,snr,seed,vae,var,decoder,arm,device):
    import torch
    from latent_followup.timing_clean import source_prepare
    from latent_enhancement.latent import complete_latent,enhancement_noise
    from latent_enhancement_eval.runner import raw_transmit_budget,raw_receive_budget
    from latent_enhancement_b.model import render_received
    from var_comm.study import seeded_noise
    with torch.no_grad():
        torch.cuda.synchronize();started=time.perf_counter()
        image=torch.from_numpy(record['pixels'][None].astype(np.float32)/127.5-1).to(device)
        f,source=source_prepare(vae,image);label=int(record['target']['class_index'])
        tx_base=complete_latent(vae,var,source[:8],label,device)
        base,_=raw_transmit_budget(source,label,8,3572)
        wave=arm.encoder(f-tx_base,tx_base)[0].cpu().numpy()
        torch.cuda.synchronize();tx_ms=(time.perf_counter()-started)*1000
        signal=np.concatenate((base,wave));energy=validate_waveform(signal)
        sid=record['target']['image_id'];sigma=math.sqrt(10**(snr/10))
        # Each disjoint segment receives noise once, outside RX timing.
        observed_base=base+seeded_noise(sid,seed,base.shape)/sigma
        observed_wave=wave+enhancement_noise(sid,seed,512)/sigma
        started=time.perf_counter()
        phy=raw_receive_budget(observed_base,snr,3572);ok=phy['label'] is not None
        rx_base=complete_latent(vae,var,phy['prefix'],phy['label'],device) if ok else torch.zeros((1,32,16,16),device=device)
        status=torch.tensor([[int(ok),int(phy['body_crc_accepted']),int(phy['mode'] or 0)]],device=device)
        z=arm.receiver(torch.as_tensor(observed_wave[None],device=device,dtype=torch.float32),rx_base,torch.tensor([snr],device=device),status)
        image=render_received(decoder,z,status)[0].cpu().numpy();torch.cuda.synchronize();rx_ms=(time.perf_counter()-started)*1000
    return image,{'tx_ms':tx_ms,'rx_ms':rx_ms,'total_ms':tx_ms+rx_ms,'N':4084,'E':energy,'header_ok':int(ok),'body_crc_ok':int(phy['body_crc_accepted'])}

def execute(record,method,snr,seed,vae,var,decoder,models,arms,device,policy,digital):
    from latent_research.evaluate import execute as research_execute
    from latent_followup.timing_clean import execute_method
    endpoint=selected_endpoint(method,snr,policy,digital)
    if method in ('full_tx_control','light_tx'):
        image,info=research_execute(record,method,models[method],snr,seed,vae,var,decoder,device)
        # The bound historical engine checks actual shape and energy before RX.
        info={**info,'N':4084,'E':8168.,'resource_verification':'historical execute actual waveform assertion'}
    elif endpoint=='m8_plus_latent_512_fold_N4084':
        image,info=folded_execute(record,snr,seed,vae,var,decoder,arms['enhancement512'],device)
    else:
        got=execute_method(record,endpoint,snr,seed,vae,var,decoder,arms,device)
        info={k:got[k] for k in ('tx_ms','rx_ms','total_ms')}
        info.update(N=4084,E=validate_waveform(got['encoded']['signal']))
        image=got['image']
    if image.shape!=(3,256,256) or not np.isfinite(image).all():raise ValueError('invalid actual received image')
    return image,{**info,'endpoint':endpoint,'timing_endpoints':'CPU uint8 RGB -> CPU waveform; CPU observation -> CPU float RGB; noise outside RX'}
