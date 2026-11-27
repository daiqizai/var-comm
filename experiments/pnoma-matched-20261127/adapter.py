"""Matched physical interface and the progressively learned complex user projections."""
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from src.models.nets.deepjsccq2 import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder
from dir6_mb.overloaded_noma_4x2 import Overloaded4x2Modem, physical_grantfree_4x2_mac

def split_qr(width, device):
    q,r=torch.linalg.qr(torch.randn(width,width,device=device,dtype=torch.complex64))
    return q * torch.sgn(r.diagonal())

class ConstellationProjections(nn.Module):
    """P and P' are independent after initialization, as in the author's Rayleigh branch."""
    def __init__(self, dim=12):
        super().__init__()
        ident=torch.eye(dim,dtype=torch.complex64)[None]
        self.tx=nn.Parameter(torch.view_as_real(ident).clone())
        self.rx=nn.Parameter(torch.view_as_real(ident).clone())

    def expand(self):
        old_tx=torch.view_as_complex(self.tx.detach())
        old_rx=torch.view_as_complex(self.rx.detach())
        users,_,width=old_tx.shape
        q=split_qr(2*width,self.tx.device)
        branches=q.reshape(2,width,2*width)
        tx=torch.stack([old_tx[u//2] @ branches[u%2] for u in range(2*users)])
        rx=torch.stack([branches[u%2].mH @ old_rx[u//2] for u in range(2*users)])
        self.tx=nn.Parameter(torch.view_as_real(tx.contiguous()).clone())
        self.rx=nn.Parameter(torch.view_as_real(rx.contiguous()).clone())

class ReferenceReceiver(Overloaded4x2Modem):
    """Reuse the native reference estimator. Only the spatial combiner is replaced."""
    def _demix_single_stage(self, yr,yi,gr,gi,amp,sigma,channel_error,nu,active,latent_ch):
        _,users,nrx,blocks=gr.shape
        payload=1-self.eta_ch-self.eta_sc
        estimate=torch.complex(gr,gi)
        observed=torch.complex(yr,yi)
        power=estimate.abs().square().sum(2)
        uncertainty=sigma[:,None,None]+nrx*channel_error[:,:,None]
        denom=(power+uncertainty/payload).clamp_min(1.e-8)
        matched=(estimate.conj().unsqueeze(-1)*observed).sum(2)/denom.unsqueeze(-1)
        matched=matched * (1-12/384)**.5 / payload.sqrt()
        effective=(10*torch.log10((power*payload/uncertainty.clamp_min(1.e-6)).mean(-1).clamp_min(1.e-5))).clamp(-6,30)
        return [(matched[:,u],effective[:,u]) for u in range(users)]

class AdaptedPNOMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=DeepJSCCQ2Encoder(bw_factor=4,N=256,C=48,num_af_dim=1)
        self.decoder=DeepJSCCQ2Decoder(bw_factor=4,N=256,C=48,num_af_dim=1)
        self.projections=ConstellationProjections()

    @property
    def users(self):
        return self.projections.tx.shape[0]

    def transmit(self, images, modem, mixed=False):
        batch,users,_,height,width=images.shape
        assert users==self.users and height==width and height in (128,256)
        source=F.pixel_unshuffle(images.flatten(0,1),4)
        tx_tag=images.new_full((batch*users,1),7.)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=mixed):
            code=self.encoder((source,tx_tag))
        real,imag=code.float().chunk(2,dim=1)
        z=torch.complex(real,imag).permute(0,2,3,1).reshape(batch,users,-1,12)
        vectors=torch.einsum("bupl,uld->bupd",z,torch.view_as_complex(self.projections.tx))
        # Paper's per-user complex power constraint. No hidden gain must be sent.
        vectors=vectors/vectors.abs().square().mean(dim=(2,3),keepdim=True).clamp_min(1e-9).sqrt()
        reim=torch.cat([vectors.real.flatten(2),vectors.imag.flatten(2)],dim=2)
        modem.num_rbs=vectors.shape[2]*vectors.shape[3]//384
        assert modem.num_rbs*384==vectors.shape[2]*vectors.shape[3]
        return torch.stack([modem.embed_user_tx(reim[:,u,None,:],u) for u in range(users)],dim=1)

    def receive(self, waveform, modem, mixed=False):
        batch,antennas,blocks,length,_=waveform.shape
        assert antennas==2 and length==384
        users=self.users
        site=math.isqrt(blocks*384//(12*users))
        assert site*site*12*users==blocks*384
        modem.num_rbs=blocks
        estimates,_,_=modem.blind_demix_users(waveform,users,96,override_mode="linear_1stage")
        combined=torch.stack([x[0] for x in estimates],dim=1).reshape(batch,users,site*site,12*users)
        csi=torch.stack([x[1] for x in estimates],dim=1).reshape(-1,1)
        z=torch.einsum("bupd,udl->bupl",combined,torch.view_as_complex(self.projections.rx))
        decoded=torch.cat([z.real,z.imag],dim=-1).reshape(batch*users,site,site,24).permute(0,3,1,2)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=mixed):
            out=self.decoder((decoded,csi))
        return F.pixel_shuffle(out.float(),4).reshape(batch,users,3,site*16,site*16)
