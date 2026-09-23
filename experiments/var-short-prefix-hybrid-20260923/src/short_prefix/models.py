import copy
import torch
from torch import nn
from latent_enhancement.latent import normalize_enhancement
from latent_enhancement_b.model import ResidualBlock
from .protocol import SIZES,allocation

@torch.no_grad()
def prefix_latent(vae,scales,m):
    if tuple(vae.quantize.v_patch_nums)!=SIZES or m not in (6,7,8) or len(scales)<m:raise ValueError('official ten-scale mapping required')
    batch=len(scales[0]);latent=vae.quantize.embedding.weight.new_zeros(batch,32,16,16)
    for i in range(m):
        if tuple(scales[i].shape)!=(batch,SIZES[i]**2):raise ValueError('prefix token shape')
        e=vae.quantize.embedding(scales[i]).transpose(1,2).reshape(batch,32,SIZES[i],SIZES[i])
        latent,_=vae.quantize.get_next_autoregressive_input(i,10,latent,e)
    return latent

class Hybrid(nn.Module):
    def __init__(self,m,family,scale,N=4084,nd=None,width=64,blocks=3):
        super().__init__();self.ledger=allocation(m,N,nd);self.family=family;self.uses=self.ledger['NA']
        if family not in ('VAR','PREFIX'):raise ValueError('pre-shared receiver family')
        self.register_buffer('scale',scale.reshape(1,32,1,1).clone())
        coordinates=2*self.uses//256
        self.encoder=nn.Sequential(nn.Conv2d(64,width,3,padding=1),*[ResidualBlock(width) for _ in range(blocks)],nn.Conv2d(width,coordinates,3,padding=1))
        self.prefix_stem=nn.Conv2d(32,width,3,padding=1)
        self.condition_stem=nn.Conv2d(32,width,3,padding=1)
        self.observation_stem=nn.Conv2d(coordinates,width,3,padding=1)
        self.class_condition=nn.Sequential(nn.Embedding(1001,16),nn.Linear(16,width))
        self.status_condition=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,width))
        self.trunk=nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.correction=nn.Conv2d(width,32,3,padding=1)
        nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
    def transmit(self,F,B):
        if F.shape!=B.shape or F.shape[1:]!=(32,16,16):raise ValueError('TX shapes')
        return normalize_enhancement(self.encoder(torch.cat(((F-B)/self.scale,B/self.scale),1)).reshape(-1,self.uses,2))
    def receive(self,y,B,C,received_class,snr,status):
        n=len(B)
        if y.shape!=(n,self.uses,2) or B.shape!=C.shape or B.shape[1:]!=(32,16,16):raise ValueError('RX shape')
        if status.shape!=(n,3) or snr.shape!=(n,) or received_class.shape!=(n,):raise ValueError('RX fields')
        labels=torch.where(status[:,0]>.5,received_class,torch.full_like(received_class,1000))
        if torch.any((labels<0)|(labels>1000)):raise ValueError('received class')
        condition=torch.cat((snr[:,None]/20,status[:,:2],status[:,2:]/10),1)
        hidden=self.prefix_stem(B/self.scale)+self.condition_stem(C/self.scale)+self.observation_stem(y.reshape(n,-1,16,16))
        hidden=hidden+(self.class_condition(labels)+self.status_condition(condition))[:,:,None,None]
        return C+self.correction(self.trunk(hidden))*self.scale
    def forward(self,b):
        wave=self.transmit(b['F'],b['B_TX'])
        y=wave+b['noise']*torch.pow(10.,-b['snr']/20)[:,None,None]
        C=b['G_RX'] if self.family=='VAR' else b['B_RX']
        return self.receive(y,b['B_RX'],C,b['label'],b['snr'],b['status']),wave

def matched_models(m,scale,seed,N=4084,nd=None):
    torch.manual_seed(seed+100*m)
    var=Hybrid(m,'VAR',scale,N,nd);prefix=copy.deepcopy(var);prefix.family='PREFIX'
    return nn.ModuleDict({f'H{m}-V':var,f'H{m}-P':prefix})
