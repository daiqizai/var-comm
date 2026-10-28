"""Actual position masks update both residual latent and all subsequent scale predictions."""
from block_common import *
from protocol import SIZES,OFFSETS

def render(native,state):
 t=native.torch;m=state['m'];coarse=np.asarray(state['coarse'],np.int64);pos=np.asarray(state['positions'],np.int64);values=np.asarray(state['values'],np.int64)
 assert len(coarse)==OFFSETS[m]and len(pos)==len(values)
 p=native.receiver._Prior(native.loaded['vae'],native.loaded['var'],native.loaded['device']);trace=[]
 try:
  with t.no_grad():
   for scale in range(10):
    logits=p.logits(scale);chosen=t.argmax(logits,dim=-1)[0].cpu().numpy().astype(np.int64)
    if scale<m:chosen[:]=coarse[OFFSETS[scale]:OFFSETS[scale+1]]
    elif scale==m:chosen[pos]=values
    trace.append(dict(scale=scale,merged_tokens_sha256=asha(chosen)))
    p.advance(chosen,scale)
   image=native.loaded['decoder'](p.fhat)[0].clamp(0,1).cpu().numpy().astype(np.float32)
  return image,trace
 finally:p.close()
