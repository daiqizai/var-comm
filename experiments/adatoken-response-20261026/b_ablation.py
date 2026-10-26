"""A0/B0 share an unoverwritten prediction trajectory; B0 changes only the latent.

This module receives actual decoded prefix tokens, never original source tokens.
It does not recondition remaining positions in the same scale.
"""
from common import SIZES,OFFSETS
import numpy as np
def merge_boundary(predicted,received,K):
 out=predicted.copy();assert 0<=K<=len(out)and len(received)==K
 out[:K]=received;return out
def ablate(native,received,m,K):
 a=np.asarray(received,dtype=np.int64)
 assert 4<=m<=9 and 0<K<SIZES[m]**2 and a.shape==(OFFSETS[m]+K,)
 assert np.all((a>=0)&(a<4096))
 p=native.receiver._Prior(native.loaded['vae'],native.loaded['var'],native.loaded['device'])
 q=native.loaded['vae'].quantize;t=native.torch;direct=p.fhat.clone();offset=0;trajectory=[]
 try:
  for s,size in enumerate(SIZES):
   logits=p.logits(s);pred=logits[0].argmax(-1).cpu().numpy()
   if s<m:
    pred[:]=a[offset:offset+size*size];offset+=size*size
   corrected=merge_boundary(pred,a[offset:offset+K],K)if s==m else pred.copy()
   if s==m:offset+=K
   trajectory.append(dict(scale=s,context_tokens=pred.copy(),latent_tokens=corrected.copy()))
   # A0 always advances with the unoverwritten boundary map. No B0 data can
   # enter p.x, p.fhat, cached attention, or later token predictions.
   p.advance(pred,s)
   idx=t.as_tensor(corrected,dtype=t.long,device=p.device)[None]
   embed=q.embedding(idx).transpose(1,2).reshape(1,native.loaded['var'].Cvae,size,size)
   direct,_=q.get_next_autoregressive_input(s,len(SIZES),direct,embed)
  assert offset==len(a)
  images={}
  for name,latent in [('A0',p.fhat),('B0',direct)]:
   images[name]=native.loaded['decoder'](latent)[0].clamp(0,1).cpu().numpy().astype(np.float32)
  return images,trajectory
 finally:p.close()
