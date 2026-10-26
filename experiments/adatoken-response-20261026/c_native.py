"""Frozen public LlamaGen-L/VQ16; null class and shared incremental TX/RX path."""
from common import *
import ctypes,numpy as np
def load_native(reg,out,ledger):
 import torch
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
 assert torch.__version__=='2.11.0+cu128'
 lib=ctypes.CDLL('libcuda.so.1');d=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(d),0)==0;b=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(b,32,d)==0;assert b.value.decode().lower()==reg['pci']
 assert torch.cuda.mem_get_info()[0]>=24<<30;torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 for desc in reg['upstream_code']:checked(desc)
 sys.path.insert(0,str(E/'upstream/llamagen'))
 from autoregressive.models.gpt import GPT_models
 from tokenizer.tokenizer_image.vq_model import VQ_models
 from autoregressive.models.generate import sample
 weights=reg['weights'];torch.manual_seed(2026102601)
 def build():
  vq=VQ_models['VQ-16'](codebook_size=16384,codebook_embed_dim=8).eval()
  vq_state=torch.load(checked(weights['vq']),map_location='cpu',weights_only=True)['model'];vq.load_state_dict(vq_state,strict=True);del vq_state;vq=vq.cuda().float();vq.requires_grad_(False)
  prior=GPT_models['GPT-L'](vocab_size=16384,block_size=256,num_classes=1000,cls_token_num=1,model_type='c2i').eval()
  checkpoint=torch.load(checked(weights['prior']),map_location='cpu',weights_only=True)
  states=[checkpoint[k]for k in('model','module','state_dict')if k in checkpoint];assert len(states)==1
  prior.load_state_dict(states[0],strict=True);del states,checkpoint;prior=prior.cuda().float();prior.requires_grad_(False)
  assert prior.cls_embedding.embedding_table.weight.shape[0]==1001
  return vq,prior
 vq,prior=ledger.call('model_load','official_frozen_pair',build)
 save(out/'numeric_identity.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),pci=b.value.decode(),precision='FP32',TF32=False,AMP=False,threads=6,interop=2,SDPA='math for all TX/RX positions',null_class=1000,cfg_scale=1.0,temperature=1.0,top_k=0,top_p=1.0,weights=weights,upstream=reg['upstream_code']))
 return Native(torch,vq,prior,sample,ledger)
class Native:
 def __init__(self,torch,vq,prior,sample,ledger):self.torch,self.vq,self.prior,self.sample,self.ledger=torch,vq,prior,sample,ledger
 def start(self):
  with self.torch.device('cuda:0'):self.prior.setup_caches(max_batch_size=1,max_seq_length=257,dtype=self.torch.float32)
  self.previous=None
 def logits(self,i):
  t=self.torch;pos=t.tensor([i],dtype=t.int64,device='cuda:0')
  cond=t.tensor([1000],dtype=t.int64,device='cuda:0')if i==0 else None
  idx=None if i==0 else t.tensor([[int(self.previous)]],dtype=t.int64,device='cuda:0')
  with t.nn.attention.sdpa_kernel(t.nn.attention.SDPBackend.MATH):v,_=self.prior(idx,cond,input_pos=pos)
  assert v.shape==(1,1,16384)and v.dtype==t.float32 and t.isfinite(v).all();return v
 def cdf(self,logits,probability_cdf):return probability_cdf(self.torch.log_softmax(logits[0],dim=-1).float().cpu().numpy())
 def tokenize(self,x):
  t=self.torch;xt=t.from_numpy(x[None]).cuda().mul(2).sub(1);_,_,info=self.vq.encode(xt);v=info[2].reshape(-1).cpu().numpy().astype(np.int64);assert v.shape==(256,)and np.all((v>=0)&(v<16384));return v
 def render(self,tokens):
  t=self.torch;v=t.tensor(tokens[None],dtype=t.int64,device='cuda:0');im=self.vq.decode_code(v,[1,8,16,16]).clamp(-1,1).add(1).mul(.5)[0].cpu().numpy();assert im.shape==(3,256,256)and im.dtype==np.float32 and np.isfinite(im).all();return im
