"""Shared two-scale residual logit corrector; targets are never forward inputs."""
import torch
from torch import nn
import math
def position(h,device,dtype):
 y,x=torch.meshgrid(torch.arange(h,device=device,dtype=dtype),torch.arange(h,device=device,dtype=dtype),indexing='ij')
 f=torch.exp(torch.arange(16,device=device,dtype=dtype)*(-math.log(10000)/16))
 return torch.cat([(x.reshape(-1,1)*f).sin(),(x.reshape(-1,1)*f).cos(),(y.reshape(-1,1)*f).sin(),(y.reshape(-1,1)*f).cos()],-1)
class Corrector(nn.Module):
 def __init__(self):
  super().__init__();self.coarse=nn.Linear(32,64);self.base=nn.Linear(4096,64);self.received=nn.Linear(33,64)
  self.scale=nn.Embedding(2,64);self.fraction=nn.Linear(1,64)
  self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(256,8,1024,0.0,'gelu',batch_first=True,norm_first=True)for _ in range(2)])
  self.norm=nn.LayerNorm(256);self.head=nn.Linear(256,4096);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,coarse,base_logits,received_embeddings,known_mask,m,K,no_content=False):
  assert m in (7,8);h=10 if m==7 else 13;L=h*h;B=base_logits.shape[0]
  assert base_logits.shape==(B,L,4096)and coarse.shape==(B,L,32)and received_embeddings.shape==(B,L,32)and known_mask.shape==(B,L)
  assert K.shape==(B,)and bool(((K>0)&(K<L)).all())
  mask=known_mask.to(base_logits.dtype);embedding=received_embeddings*mask.unsqueeze(-1)
  if no_content:embedding=torch.zeros_like(embedding)
  p=position(h,coarse.device,coarse.dtype)[None]+self.scale.weight[m-7][None,None]+self.fraction((K.to(coarse.dtype)/L)[:,None])[:,None]
  x=torch.cat([self.coarse(coarse),self.base(base_logits.log_softmax(-1)),self.received(torch.cat([embedding,mask.unsqueeze(-1)],-1)),p],-1)
  for b in self.blocks:x=b(x)
  delta=self.head(self.norm(x));return base_logits+(1-mask).unsqueeze(-1)*delta
def missing_loss(base,new,target,known):
 missing=(~known).to(base.dtype)
 ce=torch.nn.functional.cross_entropy(new.transpose(1,2),target,reduction='none')
 kl=(base.softmax(-1)*(base.log_softmax(-1)-new.log_softmax(-1))).sum(-1)
 return (((ce+.02*kl)*missing).sum(1)/missing.sum(1)).mean()
