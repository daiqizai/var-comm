"""Shared actual CPU RGB -> waveform -> CPU RGB quality/timing path."""
import time
import numpy as np,torch
from latent_enhancement.latent import complete_latent
from latent_enhancement_b.model import render_received
from .models import prefix_latent
from .protocol import transmit_digital,decode_frame,channel

@torch.no_grad()
def transmit(pixels,true_class,model,vae,device):
    F=vae.quant_conv(vae.encoder(torch.as_tensor(pixels[None],device=device,dtype=torch.float32)/127.5-1))
    tokens=vae.quantize.f_to_idxBl_or_fhat(F,to_fhat=False)
    B=prefix_latent(vae,tokens,model.ledger['m'])
    digital=transmit_digital([x[0].cpu().numpy() for x in tokens],int(true_class),model.ledger)
    continuous=model.transmit(F,B)[0].cpu().numpy()
    return digital,continuous,{'F':F,'B_TX':B,'tokens':tokens}

@torch.no_grad()
def receive(observed,snr,models,family,vae,var,decoder,device,N=4084):
    # Allowed modes and body lengths are pre-shared, not extracted from sender state.
    by_mode={model.ledger['m']:model for model in models.values()}
    if len(by_mode)!=len(models) or any(m.family!=family or m.ledger['N']!=N for m in models.values()):raise ValueError('RX configuration')
    phy=decode_frame(observed,snr,N,allowed_modes=tuple(by_mode),digital_allocations={m:v.ledger['ND'] for m,v in by_mode.items()})
    if not phy['header_ok']:return np.full((3,256,256),.5,np.float32),{'phy':phy,'latent':None}
    model=by_mode[phy['decoded_mode']]
    tokens=[torch.as_tensor(x[None],device=device) for x in phy['prefix']]
    B=prefix_latent(vae,tokens,phy['decoded_mode'])
    C=complete_latent(vae,var,phy['prefix'],phy['decoded_label'],device) if family=='VAR' else B
    status=torch.tensor([[1.,float(phy['body_crc_ok']),phy['decoded_mode']]],device=device)
    z=model.receive(torch.as_tensor(phy['observation'][None],device=device,dtype=torch.float32),B,C,torch.tensor([phy['decoded_label']],device=device),torch.tensor([snr],device=device),status)
    return render_received(decoder,z,status)[0].cpu().numpy(),{'phy':phy,'latent':z,'B_RX':B,'C_RX':C}

@torch.no_grad()
def execute(record,model,snr,seed,vae,var,decoder,device):
    torch.cuda.synchronize();started=time.perf_counter()
    digital,continuous,tx=transmit(record['pixels'],record['class_index'],model,vae,device)
    torch.cuda.synchronize();tx_ms=(time.perf_counter()-started)*1000
    observed=channel(digital,continuous,record['image_id'],seed,snr,model.ledger)
    torch.cuda.synchronize();started=time.perf_counter()
    pixels,rx=receive(observed,snr,{'fixed':model},model.family,vae,var,decoder,device,model.ledger['N'])
    torch.cuda.synchronize();rx_ms=(time.perf_counter()-started)*1000
    return pixels,{'tx_ms':tx_ms,'rx_ms':rx_ms,'total_ms':tx_ms+rx_ms,'tx':tx,'rx':rx,'waveform':np.concatenate((digital,continuous)),'observation':observed}
