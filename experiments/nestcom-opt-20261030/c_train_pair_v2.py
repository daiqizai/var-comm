"""Matched primary content/control training, independent of historical owners."""
from opt_common import *
import math,traceback,signal,collections,subprocess

def make_schedule(source_count,seed):
 rng=np.random.default_rng(seed);order=[]
 while len(order)<3000*32:order.extend(rng.permutation(source_count).tolist())
 indices=np.asarray(order[:3000*32],np.int64).reshape(3000,4,8)
 K=np.empty_like(indices)
 for step in range(3000):
  for micro,m in enumerate((7,8,7,8)):
   length=100 if m==7 else 169;quarter=np.array([25,50,75]if m==7 else[42,84,126])
   # Exactly half of each scale's examples use Quarter; the other half uniform.
   v=np.concatenate((rng.choice(quarter,4),rng.integers(1,length,4)));rng.shuffle(v);K[step,micro]=v
 return indices,K

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 completion=cr(r['cache_completion']);assert completion['status']=='COMPLETE'and completion['source_scale_pairs']==20000
 cache=cr(completion['index']);sources=cr(r['sources']);assert len(cache)==len(sources)==10000
 byid={x['source_id']:x for x in cache};ordered=[byid[s['source_id']]for s in sources]
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']))
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 start=time.monotonic();stop=False;step=0;arm=None;model=None;optimizer=None
 def request_stop(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 ledger=Ledger(out/'calls.sqlite',dict(model_initialization=2,smoke_update=4,optimizer_update=6000))
 save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  import torch
  from c_model import Corrector,missing_loss
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
  assert torch.cuda.device_count()==1 and torch.__version__=='2.11.0+cu128'
  visible=int(os.environ['CUDA_VISIBLE_DEVICES']);gpu=[x.split(', ')for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],text=True).splitlines()]
  actual=next(g for g in gpu if int(g[0])==visible);assert actual[2].lower()==r['pci'].lower()
  save(out/'numerics.json',dict(torch=str(torch.__version__),CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),GPU_uuid=actual[1],PCI=actual[2],dtype='FP32',TF32=False,attention='math',MHA_fastpath=False))
  torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
  indices,Ks=make_schedule(10000,r['seed']);schedule=out/'schedule.npz'
  with schedule.open('xb')as f:np.savez(f,indices=indices,K=Ks,scale=np.array([7,8,7,8]))
  save(out/'schedule_identity.json',dict(schedule=pin(schedule),source_order=[s['source_id']for s in sources],seed=r['seed'],steps=3000,effective_batch=32,microbatch=8,accumulation=4,scale_balanced=True,quarter_fraction=.5))
  codebook=torch.load(checked(completion['codebook']),weights_only=True,map_location='cpu').float();codebook_hash=asha(codebook.numpy())
  loaded=collections.OrderedDict();verified=set()
  def item(i):
   if i in loaded:loaded.move_to_end(i);return loaded[i]
   p=ordered[i]['item'];path=checked(p)if i not in verified else Path(p['path']);verified.add(i)
   v=torch.load(path,map_location='cpu',weights_only=True);assert v['source_id']==sources[i]['source_id'];loaded[i]=v
   if len(loaded)>128:loaded.popitem(last=False)
   return v
  def batch(step_index,micro,smoke=False):
   m=(7,8,7,8)[micro];ids=(np.arange(micro*8,(micro+1)*8)if smoke else indices[step_index,micro]);k=torch.from_numpy(Ks[step_index,micro]).to('cuda')
   v=[item(int(i))['records'][m-7]for i in ids]
   coarse=torch.stack([x['coarse']for x in v]).to('cuda');base=torch.stack([x['base_logits']for x in v]).to('cuda');target=torch.stack([x['target']for x in v]).to('cuda')
   mask=torch.arange(base.shape[1],device='cuda')[None]<k[:,None]
   # Forward has no label argument. Mask before embedding; unseen IDs never enter it.
   received=torch.zeros((*target.shape,32),device='cuda');book=codebook.to('cuda');received[mask]=book[target[mask]]
   return coarse,base,received,mask,m,k,target
  def checkpoint(tag=None):
   p=out/arm/(tag or f'step_{step:06d}.pt')
   if p.exists():return pin(p)
   ns=np.random.get_state()
   state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,arm=arm,seed=r['seed'],schedule=pin(schedule),source_manifest=r['sources'],cache_identity=r['cache_completion'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_state=dict(generator=ns[0],keys=ns[1].tolist(),position=ns[2],has_gauss=ns[3],cached_gaussian=ns[4]),seconds=time.monotonic()-start,numerics=dict(dtype='FP32',TF32=False,attention='math',torch=str(torch.__version__)),parameter_count=sum(p.numel()for p in model.parameters()))
   torch.save(state,p);return pin(p)
  initial_hash=None;smoke=[];checkpoints=[]
  for arm in ('content','no_content'):
   (out/arm).mkdir();step=0;torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
   model=ledger.call('model_initialization',arm,lambda:Corrector().float().to('cuda'))
   assert sum(p.numel()for p in model.parameters())==2899456
   init={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};ih=digest({k:asha(v.numpy())for k,v in init.items()})
   if initial_hash is None:initial_hash=ih
   else:assert ih==initial_hash
   optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01)
   checkpoints.append(checkpoint());model.train()
   # Fixed 32 training sources only. Reset to exact registered initialization afterwards.
   for u in range(2):
    def smoke_update():
     optimizer.zero_grad(set_to_none=True);losses=[]
     for micro in range(4):
      c,b,e,mask,m,k,target=batch(u,micro,True);new=model(c,b,e,mask,m,k,no_content=arm=='no_content')
      if u==0:assert torch.equal(new,b)
      assert torch.equal(new[mask],b[mask]);loss=missing_loss(b,new,target,mask);assert torch.isfinite(loss)
      (loss/4).backward();losses.append(float(loss.detach()))
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);optimizer.step();return dict(arm=arm,update=u,loss=sum(losses)/4,gradient_norm=float(norm))
    smoke.append(ledger.call('smoke_update',arm+':'+str(u),smoke_update))
   model.load_state_dict(init,strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01)
   torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
   assert digest({k:asha(v.detach().cpu().numpy())for k,v in model.state_dict().items()})==initial_hash
   with(out/arm/'updates.jsonl').open('x')as log:
    for step in range(1,3001):
     if stop or(L/'STOP').exists()or time.monotonic()-start>=r['seconds']:
      step-=1;checkpoints.append(checkpoint());raise BudgetStop('Save current C state; cumulative route allocation exhausted or stop requested')
     lr=3e-4*step/100 if step<=100 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-100)/2900))
     for group in optimizer.param_groups:group['lr']=lr
     def update():
      optimizer.zero_grad(set_to_none=True);losses=[];accuracy=[]
      for micro in range(4):
       c,b,e,mask,m,k,target=batch(step-1,micro);new=model(c,b,e,mask,m,k,no_content=arm=='no_content');loss=missing_loss(b,new,target,mask)
       assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()));accuracy.append(float(((new.argmax(-1)==target)&~mask).sum()/(~mask).sum()))
      norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);optimizer.step()
      return dict(step=step,loss=sum(losses)/4,missing_accuracy=sum(accuracy)/4,gradient_norm=float(norm),lr=lr,seconds=time.monotonic()-start)
     row=ledger.call('optimizer_update',arm+':'+str(step),update);log.write(json.dumps(row)+'\n');log.flush()
     if step%500==0:checkpoints.append(checkpoint());print('C_TRAIN',arm,step,round(row['loss'],6),round(row['seconds'],1),flush=True)
   assert asha(codebook.numpy())==codebook_hash
   save(out/arm/'completion.json',dict(status='COMPLETE',updates=3000,initial_parameters_sha256=initial_hash,source_and_mask_schedule=pin(schedule),frozen_codebook_unchanged=True))
   del model,optimizer;model=optimizer=None;torch.cuda.empty_cache()
  save(out/'smoke.json',dict(status='PASS',fixed_training_sources=[s['source_id']for s in sources[:32]],updates=smoke,initial_parameters_restored=True,not_image_quality_evidence=True))
  save(out/'completion.json',dict(status='COMPLETE',arms=['content','no_content'],seed=r['seed'],updates_per_arm=3000,parameter_count=2899456,checkpoints=checkpoints,ledger=ledger.summary(),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),FIT_evaluation='NOT_RUN_IN_TRAIN_PROCESS'))
 except BaseException as e:
  if model is not None and optimizer is not None:
   try:checkpoint('interrupted_state.pt')
   except BaseException:pass
  save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),arm=arm,step=step,ledger=ledger.summary(),automatic_retry=False,seconds=time.monotonic()-start));raise
if __name__=='__main__':main(*sys.argv[1:])
