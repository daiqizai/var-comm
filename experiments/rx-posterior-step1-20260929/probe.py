"""Calibration-only receiver posterior probe. No original code is modified."""
import os,sys,json,hashlib,signal,time,traceback,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'experiments/var-latent-enhancement-20260917'
for p in [ROOT/'src',E/'src',E/'phase_b/src',E/'evaluation/src',E/'followup/src',ROOT/'experiments/var-short-prefix-hybrid-20260923/src',ROOT/'experiments/token_channel_efficiency_20260923/src']:
 sys.path.insert(0,str(p))
os.environ['VAR_COMM_DECODER_GATE']=str(ROOT/'outputs/VAR-LATENT-ENHANCEMENT-20260917/stage_B_v2_repaired_20260921/decoder_gate.json')
import numpy as np
import torch
from torch.nn import functional as fn
from latent_enhancement.runtime import model_paths,digest,write_json,require_available,ResourceBusy
from latent_enhancement_b.common import CACHE,load_decoder,load_gate
from token_efficiency.common import configure_runtime
from short_prefix.common import Safety
from var_comm.next_scale_prior import load_models,frequency_log_probs,state_sha256,PATCH_NUMS
OUT=ROOT/'outputs/RX-POSTERIOR-STEP1-20260929'
CONFIG=Path(__file__).with_name('design.json')
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
SAFETY=None
def boundary():
 if STOP or (SAFETY and SAFETY.check()):raise ResourceBusy('probe safe source boundary')
def status(stage,**kw):
 write_json(OUT/'probe_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def read(p):return json.loads(Path(p).read_text())
def noise(source,seed):
 key=f'RX-POSTERIOR-STEP1-20260929/unit-normal/{source}/{seed}'
 g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1))
 return torch.randn((1,32,16,16),generator=g,dtype=torch.float32)
def split(t):
 return list(torch.split(t,[x*x for x in PATCH_NUMS],dim=1))
def distance(q,r,k):
 pn=PATCH_NUMS[k]
 z=(fn.interpolate(r,size=(pn,pn),mode='area') if k<9 else r).permute(0,2,3,1).reshape(-1,32)
 d=z.square().sum(1,keepdim=True)+q.embedding.weight.data.square().sum(1)
 d.addmm_(z,q.embedding.weight.data.T,alpha=-2,beta=1)
 return d.reshape(r.shape[0],pn*pn,4096),z
def embedded(q,t,k):
 pn=PATCH_NUMS[k]
 return q.embedding(t.reshape(-1,pn,pn)).permute(0,3,1,2).contiguous()
def contribution(q,t,k):
 h=embedded(q,t,k)
 if k<9:h=fn.interpolate(h,size=(16,16),mode='bicubic').contiguous()
 return q.quant_resi[k/9](h)
def cumulative(q,tokens):
 f=q.embedding.weight.new_zeros(tokens[0].shape[0],32,16,16)
 for k,t in enumerate(tokens):f,_=q.get_next_autoregressive_input(k,10,f,embedded(q,t,k))
 return f
class Prior:
 def __init__(self,var):
  self.var=var;self.label=torch.tensor([1000],device=next(var.parameters()).device)
  self.cond=var.class_emb(self.label);self.positions=var.lvl_embed(var.lvl_1L)+var.pos_1LC
  self.x=self.cond[:,None].expand(-1,var.first_l,-1)+var.pos_start.expand(1,var.first_l,-1)+self.positions[:,:var.first_l]
  self.offset=0
  for block in var.blocks:block.attn.kv_caching(True)
 def logits(self,k):
  h=self.x;condition=self.var.shared_ada_lin(self.cond)
  for block in self.var.blocks:h=block(x=h,cond_BD=condition,attn_bias=None)
  return self.var.get_logits(h,self.cond).float()
 def advance(self,nextmap,k):
  self.offset+=PATCH_NUMS[k]**2
  if k<9:
   x=nextmap.reshape(1,32,-1).transpose(1,2)
   self.x=self.var.word_embed(x)+self.positions[:,self.offset:self.offset+PATCH_NUMS[k+1]**2]
 def close(self):
  for block in self.var.blocks:block.attn.kv_caching(False)
@torch.no_grad()
def infer(vae,var,Z,truth,prior,mode,variance,static,lam=1,details=False):
 q=vae.quantize;rest=Z.clone();fhat=torch.zeros_like(Z);pr=Prior(var) if prior=='V' else None
 rows=[];chosen=[];logits=[]
 try:
  for k,pn in enumerate(PATCH_NUMS):
   d,_=distance(q,rest,k)
   score=-d/(2*float(variance[k]))
   lp=None
   if prior=='V':
    logits.append(pr.logits(k));lp=fn.log_softmax(logits[-1],dim=-1)
   elif prior=='A2':lp=static[k][None,None]
   if lp is not None:score=score+lam*lp
   idx=score.argmax(-1);chosen.append(idx)
   if details:
    post=fn.log_softmax(score,dim=-1);prob=post.exp();conf=prob.max(-1).values;correct=idx.eq(truth[k])
    bins=torch.clamp((conf*15).long(),max=14)
    rows.append(dict(acc=float(correct.float().mean()),logp_true=float(post.gather(-1,truth[k][...,None]).mean()),entropy=float(-(prob*post).sum(-1).mean()),bins=[dict(count=int((bins==i).sum()),confidence_sum=float(conf[bins==i].sum()),correct_sum=int(correct[bins==i].sum())) for i in range(15)]))
   use=truth[k] if mode=='TF' else idx
   # The official quantizer subtracts each residual contribution in sequence.
   # This is Z - accumulated fhat mathematically, while retaining FP32 ordering.
   h=contribution(q,use,k);rest.sub_(h)
   fhat,nextmap=q.get_next_autoregressive_input(k,10,fhat,embedded(q,use,k))
   if pr:pr.advance(nextmap,k)
 finally:
  if pr:pr.close()
 return fhat,chosen,rows,logits
def load_calibration():
 Fs=[];Fqs=[];Ts=[];ids=[];bound={}
 for p in sorted((CACHE/'calibration').glob('shard_*.pt')):
  rec=read(p.with_suffix('.json'));assert digest(p)==rec['sha256']
  d=torch.load(p,map_location='cpu',weights_only=True)
  assert d['F'].dtype==torch.float32 and d['Fq'].dtype==torch.float32
  Fs.append(d['F']);Fqs.append(d['Fq']);Ts.append(d['full_tokens'].long());ids.extend(d['image_ids']);bound[str(p)]=digest(p)
 assert len(ids)==1000 and len(set(ids))==1000
 return torch.cat(Fs),torch.cat(Fqs),torch.cat(Ts),ids,bound
@torch.no_grad()
def calibration(vae,F,Fq,tokens,ids,bindings,cfg):
 path=OUT/'calibration_statistics.pt'
 if path.exists():
  meta=read(OUT/'calibration_statistics.json');assert digest(path)==meta['sha256']
  assert meta['cache_bindings']==bindings and meta['design_sha256']==digest(CONFIG)
  return torch.load(path,map_location='cpu',weights_only=True)
 mu=F.double().mean(0,keepdim=True);var=F.double().var(0,correction=0,keepdim=True)
 sigma=var.sqrt().clamp_min(cfg['standard_deviation_floor'])
 err=(F.double()-Fq.double())/sigma
 rho=err.var((0,2,3),correction=0,keepdim=True).float()
 counts=[torch.bincount(x.flatten(),minlength=4096) for x in split(tokens)]
 sums=np.zeros(10);sq=np.zeros(10);cnt=np.zeros(10,dtype=np.int64)
 for start in range(0,1000,10):
  boundary();r=F[start:start+10].cuda();truth=split(tokens[start:start+10].cuda())
  for k,pn in enumerate(PATCH_NUMS):
   z=fn.interpolate(r,size=(pn,pn),mode='area') if k<9 else r
   residual=(z-embedded(vae.quantize,truth[k],k)).double()
   sums[k]+=float(residual.sum());sq[k]+=float(residual.square().sum());cnt[k]+=residual.numel()
   r.sub_(contribution(vae.quantize,truth[k],k))
  status('CALIBRATION_CLEAN_RESIDUAL_VARIANCE',sources=start+10)
 s2=sq/cnt-(sums/cnt)**2
 # Unit-eta noise variance estimated only on the frozen first 200 calibration IDs.
 ns=np.zeros(10);nq=np.zeros(10);nc=np.zeros(10,dtype=np.int64)
 for i in range(200):
  boundary()
  for seed in cfg['calibration_noise_seeds']:
   n=(sigma.float()*noise(ids[i],seed)).cuda()
   for k,pn in enumerate(PATCH_NUMS):
    z=(fn.interpolate(n,size=(pn,pn),mode='area') if k<9 else n).double()
    ns[k]+=float(z.sum());nq[k]+=float(z.square().sum());nc[k]+=z.numel()
 stats=dict(mu=mu.float(),sigma=sigma.float(),rho=rho,s2=torch.from_numpy(s2),unit_noise_var=torch.from_numpy(nq/nc-(ns/nc)**2),counts=counts,ids=ids,subset_ids=ids[:200])
 torch.save(stats,path)
 write_json(OUT/'calibration_statistics.json',dict(sha256=digest(path),design_sha256=digest(CONFIG),cache_bindings=bindings,calibration_sources=1000,subset_sources=200,subset_ids=ids[:200],s2=s2.tolist(),unit_noise_variance=(nq/nc-(ns/nc)**2).tolist(),sigma_min=float(sigma.min()),rho_channel=rho.flatten().tolist(),unfloored_sigma_min=float(var.sqrt().min()),source_role='calibration_only'))
 return stats
@torch.no_grad()
def selfcheck(vae,var,F,Fq,tokens,stats,cfg):
 checks=[];q=vae.quantize
 if q.using_znorm:raise RuntimeError('sheet Euclidean likelihood incompatible with normalized-codebook quantizer')
 static=[frequency_log_probs(v,cfg['static_pseudocount']).float().cuda() for v in stats['counts']]
 variance=stats['s2'].clamp_min(cfg['variance_floor'])
 for i in cfg['engineering_calibration_indices']:
  boundary();f=F[i:i+1].cuda();truth=split(tokens[i:i+1].cuda())
  official=q.f_to_idxBl_or_fhat(f,to_fhat=False)
  assert all(torch.equal(a,b) for a,b in zip(official,truth)), 'cached source tokens changed'
  official_fq=q.f_to_idxBl_or_fhat(f,to_fhat=True)[-1]
  predicted,idx,_,_=infer(vae,var,f,truth,'A1','CL',variance,static,lam=0)
  exact=all(torch.equal(a,b) for a,b in zip(idx,official))
  fq_equal=torch.equal(predicted,official_fq)
  reconstructed=cumulative(q,truth)
  assert exact and fq_equal, ('zero-noise nearest-codeword identity failed',i)
  torch.testing.assert_close(reconstructed,Fq[i:i+1].cuda(),atol=2e-5,rtol=1e-5)
  # Compare independent sequential cached execution against official full TF.
  canonical=var(torch.tensor([1000],device='cuda'),q.idxBl_to_var_input(truth)).float()
  _,_,_,sequential=infer(vae,var,f,truth,'V','TF',variance,static,details=True)
  joined=torch.cat(sequential,dim=1)
  maxerr=float((canonical-joined).abs().max())
  assert maxerr<=cfg['official_forward_absolute_tolerance'],('official logits mismatch',maxerr)
  per={}
  for prior in ['A2','V']:
   _,idx,metrics,_=infer(vae,var,f,truth,prior,'CL',variance,static,details=True)
   acc=float(torch.cat(idx,1).eq(torch.cat(truth,1)).float().mean())
   per[prior]=dict(token_accuracy=acc,scale_accuracy=[r['acc'] for r in metrics])
  checks.append(dict(calibration_index=i,zero_noise_no_prior_tokens_exact=exact,Fq_bitwise_equal=fq_equal,official_TF_logits_max_abs=maxerr,zero_noise_lambda1=per))
  write_json(OUT/'engineering_checks_partial.json',checks);print('SELFCHECK',i,per,flush=True)
 aggregate={prior:float(np.mean([r['zero_noise_lambda1'][prior]['token_accuracy'] for r in checks])) for prior in ['A2','V']}
 passed=all(v>=cfg['zero_noise_prior_min_accuracy'] for v in aggregate.values())
 result=dict(status='CALIBRATION_ENGINEERING_CHECKS_PASS' if passed else 'ENGINEERING_CHECK_2_NOT_PASSED_STOP_BEFORE_DEVELOPMENT',checks=checks,mean_zero_noise_lambda1_accuracy=aggregate,threshold=cfg['zero_noise_prior_min_accuracy'],new_development_accessed=False,holdout_accessed=False)
 write_json(OUT/'engineering_checks.json',result)
 if not passed:
  write_json(OUT/'blocked.json',dict(reason='Execution sheet engineering check 2: clean lambda=1 token agreement not close to 100%; do not bypass or tune on development.',result=result,time=time.time()))
  raise SystemExit(3)
 return result
@torch.no_grad()
def main():
 global SAFETY
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight'],default='preflight');args=p.parse_args()
 assert (OUT/'pause_complete.json').exists()
 cfg=read(CONFIG);configure_runtime();require_available();SAFETY=Safety()
 paths=model_paths();vae,var=load_models(paths,torch.device('cuda:0'))
 bound={str(Path(paths[k]).resolve()):digest(paths[k]) for k in ['vae_checkpoint','var_checkpoint']}
 gate=load_gate();bound[gate['selection']['checkpoint']]=gate['selection']['checkpoint_sha256']
 write_json(OUT/'models.json',dict(files=bound,vae_state_sha256=state_sha256(vae),var_state_sha256=state_sha256(var),decoder_gate=gate['selection'],unconditional_label=1000))
 F,Fq,tokens,ids,bindings=load_calibration()
 stats=calibration(vae,F,Fq,tokens,ids,bindings,cfg)
 result=selfcheck(vae,var,F,Fq,tokens,stats,cfg)
 write_json(OUT/'preflight_complete.json',result)
if __name__=='__main__':
 try:main()
 except ResourceBusy as e:
  status('SAFE_PROBE_PAUSE',reason=str(e));raise SystemExit(75)
 except Exception:
  write_json(OUT/'probe_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
