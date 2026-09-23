import copy
import torch
from torch import nn
from latent_enhancement.latent import normalize_enhancement
from latent_enhancement_b.model import ResidualBlock

@torch.no_grad()
def prefix_latent(vae, scales, count=8):
    sizes=tuple(vae.quantize.v_patch_nums)
    if len(sizes)!=10 or len(scales)!=10 or count!=8:raise ValueError('preserve official ten-scale map and m8 prefix')
    batch=len(scales[0]);latent=vae.quantize.embedding.weight.new_zeros(batch,vae.Cvae,sizes[-1],sizes[-1])
    for i in range(count):
        embedding=vae.quantize.embedding(scales[i]).transpose(1,2).reshape(batch,vae.Cvae,sizes[i],sizes[i])
        latent,_=vae.quantize.get_next_autoregressive_input(i,len(sizes),latent,embedding)
    return latent

def light_warm_start(parent):
    model=copy.deepcopy(parent)
    # [Wr,Wb] on [F-Fb,Fb] -> [Wr,Wb-Wr] on [F,Fb].
    # Replacing Fb with the true partial prefix is the intentional new input.
    with torch.no_grad():
        c=model.encoder.stem.weight.shape[1]//2
        model.encoder.stem.weight[:,c:].sub_(model.encoder.stem.weight[:,:c])
    return model

class PureContinuous(nn.Module):
    def __init__(self,scale,width=64,blocks=3):
        super().__init__();self.uses=4084
        self.register_buffer('scale',scale.reshape(1,32,1,1).clone())
        self.encode=nn.Sequential(nn.Conv2d(32,width,3,padding=1),*[ResidualBlock(width) for _ in range(blocks)],nn.Conv2d(width,32,3,padding=1))
        self.rx_stem=nn.Conv2d(32,width,3,padding=1)
        self.snr_condition=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.decode=nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)],nn.Conv2d(width,32,3,padding=1))
    def transmit(self,F):
        coordinates=self.encode(F/self.scale).flatten(1)[:,:8168]
        return normalize_enhancement(coordinates.reshape(-1,4084,2))
    def receive(self,y,snr):
        if y.shape[1:]!=(4084,2):raise ValueError('pure RX requires exactly4084 complex symbols')
        padded=torch.nn.functional.pad(y.flatten(1),(0,24)).reshape(-1,32,16,16)
        return self.decode(self.rx_stem(padded)+self.snr_condition(snr[:,None]/20)[:,:,None,None])*self.scale
    def forward(self,F,snr,noise):
        wave=self.transmit(F);return self.receive(wave+noise*torch.pow(10.,-snr/20)[:,None,None],snr),wave

def forward_arm(name,model,batch):
    if name=='pure_continuous':return model(batch['F'],batch['snr_db'],batch['pure_noise'])
    if name=='light_tx':return model.receive_training_residual_sample(batch['F'],batch['F_prefix'],batch['Fb_RX'],batch['snr_db'],batch['rx_status'],batch['standard_noise'])
    return model.receive_training_sample(batch['F'],batch['Fb_TX'],batch['Fb_RX'],batch['snr_db'],batch['rx_status'],batch['standard_noise'])
