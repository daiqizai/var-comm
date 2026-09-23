"""Shared online CPU RGB -> CPU waveform / CPU observation -> CPU RGB.

No RX argument contains transmitter truth, unsent class, source tokens or energy
recovery gain. Timing and quality call this same engine. Memoized/offline quality
runners must call its component functions and cannot use memoized online timing.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,time
import numpy as np
import torch
from latent_enhancement.latent import complete_latent
from latent_enhancement.runtime import digest,verify_snapshot
from latent_followup.run_identity import checked_checkpoint
from var_comm.scale_channel import indices_to_bits
from var_comm.study import seeded_noise
from var_comm.whole_entropy import decode_source
from .codec import encode_all,cumulative
from . import phy
from .models import BudgetContinuous
from .common import ROOT,read

@dataclass(frozen=True)
class Cell:
    family:str
    N:int
    m:int|None=None
    mcs:str='QPSK'
    def __post_init__(self):
        if self.N not in (2048,3060,4084):raise ValueError('registered N only')
        if self.family=='continuous':
            if self.m is not None or self.mcs!='QPSK':raise ValueError('continuous cell has no digital mode/MCS choice')
        elif self.family in ('raw','arithmetic'):
            if self.m not in range(6,11):raise ValueError('source mode')
            phy.dimensions(self.N,self.family,self.mcs)
        else:raise ValueError('family')
    @property
    def name(self):
        return f'P{self.N}' if self.family=='continuous' else f'{phy.PROTOCOL}/{self.family}/{self.mcs}/N{self.N}/m{self.m}'

def waveform_sha(wave):
    return hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()

def source_pixels(pixels):
    pixels=np.asarray(pixels)
    if pixels.dtype!=np.uint8 or pixels.shape!=(3,256,256):raise ValueError('CPU RGB uint8 [3,256,256] required, no implicit lossy cast')
    return pixels

@torch.no_grad()
def transmit(pixels,true_class,cell,vae,var,device,model=None):
    pixels=source_pixels(pixels)
    F=vae.quant_conv(vae.encoder(torch.as_tensor(pixels[None],device=device,dtype=torch.float32)/127.5-1))
    if cell.family=='continuous':
        if model is None or model.uses!=cell.N:raise ValueError('model trained at exact N required')
        wave=model.transmit(F)[0].cpu().numpy()
        ledger={'protocol':f'VAR-CONTINUOUS-{cell.N}','N':cell.N,'N_header':0,'N_data':0,'N_continuous':cell.N,'E':float(np.sum(wave.astype(np.float64)**2)),'energy_constraint':'per_frame_2N','m_requested':None,'m_actual':None,'mcs':'continuous','payload_bits':None,'source_overflow_erasure':False}
    else:
        tokens=vae.quantize.f_to_idxBl_or_fhat(F,to_fhat=False)
        scales=[v[0].cpu().numpy() for v in tokens]
        if cell.family=='raw':encoded={cell.m:{'raw':indices_to_bits(np.concatenate(scales[:cell.m]))}}
        else:encoded=encode_all(vae,var,scales,true_class,device)
        wave,ledger=phy.transmit(encoded,true_class,cell.m,cell.N,cell.family,cell.mcs)
        ledger['N_continuous']=0
    if wave.shape!=(cell.N,2) or not np.isfinite(wave).all():raise ValueError('complete finite TX waveform')
    if cell.family=='continuous' or cell.mcs=='QPSK':
        if not np.isclose(ledger['E'],2*cell.N,rtol=1e-5,atol=.02):raise RuntimeError('actual energy differs from registered constraint')
    return wave,ledger

def apply_channel(wave,snr,source_id,seed,cell):
    wave=np.asarray(wave)
    if wave.shape!=(cell.N,2) or not np.isfinite(wave).all() or not np.isfinite(snr):raise ValueError('finite full-length channel input')
    if cell.family!='continuous':return phy.channel(wave,snr,source_id,seed,cell.N,cell.family,cell.mcs)
    # Identical namespace and float32 arithmetic as actual continuous training.
    noise=seeded_noise(f'VAR-CONTINUOUS-{cell.N}|'+str(source_id),int(seed),(cell.N,2)).astype(np.float32)
    return wave+noise*np.float32(10.**(-float(snr)/20))

@torch.no_grad()
def receive(observed,snr,cell,vae,var,decoder,device,model=None):
    observed=np.asarray(observed)
    if observed.shape!=(cell.N,2) or not np.isfinite(observed).all() or not np.isfinite(snr):raise ValueError('finite complete RX observation and nominal SNR')
    if cell.family=='continuous':
        if model is None or model.uses!=cell.N:raise ValueError('RX model budget')
        z=model.receive(torch.as_tensor(observed[None],dtype=torch.float32,device=device),torch.tensor([snr],dtype=torch.float32,device=device))
        event={'header_ok':'not_applicable','body_crc_ok':'not_applicable','source_complete':'not_applicable','decoded_label':None,'decoded_mode':None,'source_error':'','decoder': 'Dc'}
    else:
        event=phy.receive(observed,snr,cell.N,cell.family,cell.mcs)
        if not event['header_ok']:
            return np.full((3,256,256),.5,np.float32),{**event,'source_complete':False,'source_error':'header_failure_or_paid_source_erasure','decoder':'Dc'}
        prefix=event['prefix'];padding=0;error='';z=None
        if prefix is None:
            # Preserve the established legal partial-candidate rule even after
            # body CRC failure. No source truth is consulted by decode_source.
            decoded=decode_source({'label':event['decoded_label'],'mode':event['decoded_mode'],'header':{'length_field':event['length_field']},'payload':event['source_bits']},vae,var,device,render=False,return_latent=True)
            prefix=decoded['prefix'];padding=decoded['arithmetic_padding_reads'];error=decoded['source_error'];z=decoded['latent']
        if z is None:z=cumulative(vae,prefix,10) if len(prefix)==10 else complete_latent(vae,var,prefix,event['decoded_label'],device)
        event={**event,'source_complete':len(prefix)==event['decoded_mode'] and not error,'source_error':error,'arithmetic_padding_reads':padding,'decoder':'Dc'}
    image=decoder(z)[0].cpu().numpy()
    if image.shape!=(3,256,256) or not np.isfinite(image).all():raise FloatingPointError('actual RX output')
    return image,event

def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)

@torch.no_grad()
def execute(record,cell,snr,seed,vae,var,decoder,device,model=None):
    synchronize(device);started=time.perf_counter()
    wave,ledger=transmit(record['pixels'],record.get('class_index'),cell,vae,var,device,model)
    synchronize(device);tx_ms=(time.perf_counter()-started)*1000
    observed=apply_channel(wave,snr,record['image_id'],seed,cell)
    synchronize(device);started=time.perf_counter()
    image,event=receive(observed,snr,cell,vae,var,decoder,device,model)
    synchronize(device);rx_ms=(time.perf_counter()-started)*1000
    return image,{'ledger':ledger,'rx':event,'tx_ms':tx_ms,'rx_ms':rx_ms,'total_ms':tx_ms+rx_ms,'waveform':wave,'observation':observed,'waveform_sha256':waveform_sha(wave),'observation_sha256':waveform_sha(observed),'timing_endpoints':'CPU uint8 RGB -> CPU waveform; CPU observation -> CPU float RGB; noise outside RX'}

def load_selected_budget(selected_path,scale,device):
    """Load only the actually selected low-budget checkpoint and verified arm."""
    selected_path=Path(selected_path);rec=read(selected_path);folder=selected_path.parent;registration=read(folder/'registration.json')
    regsha=digest(folder/'registration.json')
    if rec.get('registration_sha256')!=regsha or rec.get('N')!=registration['N']:raise RuntimeError('selected registration/budget mismatch')
    verify_snapshot(registration['bindings'])
    checkpoint=checked_checkpoint(rec,ROOT);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['registration_sha256']!=regsha or state['state']['step']!=rec['step']:raise RuntimeError('selected checkpoint identity/step')
    name=f"P{registration['N']}"
    if rec['arm_key']!=name or state['state']['updates'][name]!=rec['total_updates']:raise RuntimeError('selected arm/update count')
    model=BudgetContinuous(scale.cpu(),registration['N'])
    model.load_state_dict({k[len(name)+1:]:v for k,v in state['models'].items() if k.startswith(name+'.')},strict=True)
    return model.to(device).eval().requires_grad_(False),{'selected':str(selected_path),'selected_sha256':digest(selected_path),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'N':registration['N'],'step':rec['step'],'decoder_sha256':registration['decoder_sha256']}
