"""Frozen metric definitions; cache identity includes the real metric device/runtime."""
from opt_common import *
import importlib.util,traceback,ctypes,tempfile
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 plan=cr(r['plan']);old=cr(r['metric_registration']);cfg=old['config'];out=Path(r['output']);out.mkdir(exist_ok=False)
 for sub in ('features','pairs','tmp'):(out/sub).mkdir()
 tempfile.tempdir=str(out/'tmp');os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');os.sched_setaffinity(0,set(r['cpu_affinity']))
 start=time.monotonic();ledger=Ledger(out/'calls.sqlite',r['caps'])
 save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 try:
  import torch,torch.nn.functional as F
  torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_device(0)
  assert torch.__version__==old['torch'];lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0
  bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower().lstrip('0')==r['pci'].lower().lstrip('0')
  props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction((12<<30)/props.total_memory)
  triple=r['mode']=='three';met=module(checked(cfg['clip']['loader']),'_OPT_FROZEN_METRICS')
  identity=dict(mode=r['mode'],metric_config_sha256=digest(cfg),original_four=old['original_four'],torch=torch.__version__,CUDA=torch.version.cuda,UUID=str(props.uuid),PCI=bus.value.decode().lower(),FP32=True,TF32=False,AMP=False,batch=1)
  features={};scores={};reused_pairs={}
  for desc in r.get('caches',[]):
   cache=cr(desc);ci=cr(cache['identity']);assert ci==identity,('Cache device/runtime/config differs',ci,identity)
   assert cache['status']=='COMPLETE';features.update(cr(cache['features']));reused_pairs.update(cr(cache['scores']))
  save(out/'identity.json',identity)
  if triple:
   sys.path.insert(0,str(Path(old['original_four']['path']).parent));four=module(checked(old['original_four']),'_OPT_FOUR');c=cfg['config']['metric_config'];sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation
   assets=relocation.metric_assets(four,c);import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   psnr,pp=four.psnr_projection(c['sources']['score_bpg_cached_holdout_r2.py']['path']);lp_load,lp_pair,lp_proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=ledger.call('load_lpips','once',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];overlay=cr(sup['dependency_overlay']);[checked(v)for v in overlay['files']];sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=cr(sup['metric_assets']);torch.manual_seed(2026101401)
   with met.no_network():network=ledger.call('load_dreamsim','once',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   preprocess=met.preprocess_dreamsim;save(out/'original_AST.json',dict(psnr=pp,lpips=lp_proof))
  else:
   sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation;assets=relocation.relocate_value(cfg['clip']['clip_assets']);checked(assets['weights'])
   with met.no_network():network=ledger.call('load_clip','once',lambda:met.load_clip(assets,torch.device('cuda:0')))
   preprocess=lambda x:met.preprocess_clip(met.validate_images(x))
  def feature(a,h):
   if h not in features:
    def embed():
     x=torch.from_numpy(a[None]).cuda();v=(network.embed(preprocess(x))if triple else network.encode_image(preprocess(x))).float().cpu().numpy();p=out/'features'/(h+'.npy')
     with p.open('xb')as f:np.save(f,v,allow_pickle=False)
     features[h]=pin(p);return v
    v=ledger.call('embedding',h,embed)
   else:
    with checked(features[h]).open('rb')as f:v=np.load(f,allow_pickle=False)
   return torch.from_numpy(v).cuda()
  sources={s['source_id']:s for s in plan['sources']};last=None;new=0
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   for j in sorted(plan['pairs'],key=lambda j:(j['source_index'],j['pair_key'])):
    if time.monotonic()-start>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Metric route remainder')
    key=j['pair_key'];s=sources[j['source_id']]
    if key in reused_pairs:
     v=reused_pairs[key];assert v['reference_sha256']==s['reference_sha256']and v['pixel_sha256']==j['pixel_sha256'];scores[key]=dict(v,source_id=s['source_id'],source_index=s['source_index'],origin='EXACT_PAIR_REUSE');continue
    if last!=s['source_id']:
     with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
     assert asha(x)==s['reference_sha256'];xt=torch.from_numpy(x[None]).cuda();fx=feature(x,s['reference_sha256']);last=s['source_id']
    if j['image']is None:y=np.full((3,256,256),.5,np.float32)
    else:
     with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
    assert asha(y)==j['pixel_sha256']and y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1
    fy=feature(y,j['pixel_sha256']);yt=torch.from_numpy(y[None]).cuda()
    def score():
     values=dict(psnr_db=float(psnr(x,y)),lpips_alex=float(lp_pair(lp,yt,xt)[0].item()),dreamsim=float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item()))if triple else dict(clip_image_cosine=float(F.cosine_similarity(fx,fy,dim=-1)[0].item()))
     assert all(np.isfinite(v)for v in values.values());v=dict(pair_key=key,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=j['pixel_sha256'],origin='NEW_METRIC_PAIR',**values);save(out/'pairs'/(key+'.json'),v);return v
    scores[key]=ledger.call('pair',key,score);new+=1
    if len(scores)%100==0:print('METRICS',r['mode'],len(scores),len(plan['pairs']),flush=True)
  assert ledger.summary()['unresolved']==0
  save(out/'scores.json',scores);save(out/'features.json',features);save(out/'completion.json',dict(status='COMPLETE',identity=pin(out/'identity.json'),scores=pin(out/'scores.json'),features=pin(out/'features.json'),pairs=len(scores),new_pairs=new,reused_pairs=len(scores)-new,ledger=ledger.summary(),seconds=time.monotonic()-start))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False,seconds=time.monotonic()-start));raise
if __name__=='__main__':main(*sys.argv[1:])
