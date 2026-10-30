"""Receiver-only completion from accepted tokens; no source/metric arguments."""
from opt_common import *
import copy

def completed_latent(native,received,m,K,temperature=0.0,seed=None,corrector=None,no_content=False):
 torch=native.torch;a=np.asarray(received,np.int64)
 assert 0<=m<=10 and a.shape==(OFFSETS[m]+K,)and bool(((a>=0)&(a<4096)).all())
 assert m==10 and K==0 or 0<=K<SIZES[m]**2
 p=native.receiver._Prior(native.loaded['vae'],native.loaded['var'],native.loaded['device']);offset=0;nxt=None
 q=native.loaded['vae'].quantize;advance=q.get_next_autoregressive_input
 def capture(*args,**kwargs):
  nonlocal nxt
  f,nxt=advance(*args,**kwargs);return f,nxt
 q.get_next_autoregressive_input=capture
 rng=torch.Generator(device=native.loaded['device']).manual_seed(int(seed))if temperature else None
 calls=0;chosen_maps=[]
 try:
  for s,size in enumerate(SIZES):
   logits=p.logits(s);count=size*size if s<m else(K if s==m else 0)
   if corrector is not None and s==m and m in(7,8)and 0<K<size*size:
    assert nxt.shape==(1,32,size,size)
    coarse=nxt.reshape(1,32,-1).transpose(1,2).contiguous();known=torch.arange(size*size,device=logits.device)[None]<K
    embedding=torch.zeros((1,size*size,32),device=logits.device,dtype=logits.dtype)
    ids=torch.as_tensor(a[offset:offset+K],device=logits.device);embedding[0,:K]=q.embedding(ids)
    logits=corrector(coarse,logits,embedding,known,m,torch.tensor([K],device=logits.device),no_content=no_content);calls+=1
   chosen=logits[0].argmax(-1)
   if temperature and count<size*size:
    probs=(logits[0,count:]/temperature).softmax(-1)
    chosen[count:]=torch.multinomial(probs,1,generator=rng).squeeze(1)
   chosen=chosen.cpu().numpy()
   if count:chosen[:count]=a[offset:offset+count];offset+=count
   chosen_maps.append(chosen.copy());p.advance(chosen,s)
  assert offset==len(a)
  expected=1 if corrector is not None and m in(7,8)and 0<K<SIZES[m]**2 else 0
  assert calls==expected
  return p.fhat.detach().clone(),dict(corrector_forwards=calls,completed_tokens_sha256=asha(np.concatenate(chosen_maps)),received_tokens_unchanged=True)
 finally:q.get_next_autoregressive_input=advance;p.close()

def decode_image(native,latent,anchor=None):
 torch=native.torch
 pred=native.loaded['decoder'](latent)if anchor is None else anchor(latent,torch.zeros(len(latent),device=latent.device,dtype=torch.long))
 out=pred[0].clamp(0,1).cpu().numpy().astype(np.float32)
 assert out.shape==(3,256,256)and np.isfinite(out).all();return out

def load_anchor(native,checkpoint,seed):
 torch=native.torch;sys.path.insert(0,str(R/'experiments/progressive-rd-scaleup-20261022'));from decoder_model import ProfileDecoder
 ns=dict(torch=torch,nn=torch.nn,copy=copy)
 definitions(R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',{'ContinuousDecoder'},ns)
 cp=torch.load(checked(checkpoint),map_location='cpu',weights_only=True)
 assert cp['seed']==seed and cp['arm']=='multiprefix_full_anchor'and cp['global_step']==10000
 model=ProfileDecoder(ns['ContinuousDecoder'](native.loaded['vae']),False).to(native.loaded['device'])
 model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);assert model.conditional is False
 return model
