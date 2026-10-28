"""Frozen four metric definitions on native pixels; no models or channel reruns."""
from evidence_common import *
import importlib.util,traceback,ctypes,tempfile
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(request,expected):
 assert sha(request)==expected;r=read(request);rp=pin(request);[checked(d)for d in r['code']]
 plan=cr(r['plan']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'features').mkdir();(out/'pairs').mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');os.sched_setaffinity(0,set(range(6)))
 import torch,torch.nn.functional as F
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
 assert torch.__version__==r['torch'];lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0;bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower()==r['pci'].lower()
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 triple=r['mode']=='three';ledger=Ledger(out/'calls.sqlite',r['caps']);beg=time.monotonic();cfg=r['config'];met=module(checked(cfg['clip']['loader']),'_N512_FROZEN_METRICS')
 features={};rows={};cache=cr(r['feature_cache'])if r.get('feature_cache')else{};reuse=cr(r['pair_cache'])if r.get('pair_cache')else{}
 try:
  if triple:
   sys.path.insert(0,str(Path(r['original_four']['path']).parent));four=module(checked(r['original_four']),'_N512_FOUR');c=cfg['config']['metric_config'];sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation;assets=relocation.metric_assets(four,c)
   import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   psnr,pp=four.psnr_projection(c['sources']['score_bpg_cached_holdout_r2.py']['path']);lp_load,lp_pair,lp_proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=ledger.call('load_lpips','once',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];overlay=cr(sup['dependency_overlay']);[checked(d)for d in overlay['files']];sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=cr(sup['metric_assets']);torch.manual_seed(2026101401)
   with met.no_network():network=ledger.call('load_dreamsim','once',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   preprocess=met.preprocess_dreamsim;references={}
  else:
   sys.path.insert(0,str(R/'tools'));import checkpoint_relocation_20261028 as relocation;effective_clip_assets=relocation.relocate_value(cfg['clip']['clip_assets']);checked(effective_clip_assets['weights'])
   with met.no_network():network=ledger.call('load_clip','once',lambda:met.load_clip(effective_clip_assets,torch.device('cuda:0')))
   preprocess=lambda x:met.preprocess_clip(met.validate_images(x));old=cr(r['prior_clip_identity']);assert old['fp32']and not old['tf32']and not old['amp']and old['cuda_bus'].lower()==r['pci'].lower();assert old['assets']==cfg['clip']['clip_assets']and old['loader']==cfg['clip']['loader']and old['preprocessing']==cfg['clip']['metric_definition']
   references={v['source_id']:v for v in cr(r['clip_cache'])}
  save(out/'identity.json',dict(request=rp,mode=r['mode'],config=r['frozen_metric_registration'],torch=torch.__version__,pci=bus.value.decode(),FP32=True,TF32=False,AMP=False,batch=1,loader=cfg['clip']['loader'],original_AST=dict(psnr=pp,lpips=lp_proof)if triple else None))
  def feature(a,h):
   if h in cache:
    with checked(cache[h]).open('rb')as f:return torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
   if h not in features:
    def embed():
     x=torch.from_numpy(a[None]).cuda();f=(network.embed(preprocess(x))if triple else network.encode_image(preprocess(x))).float().cpu().numpy();p=out/'features'/(h+'.npy')
     with p.open('xb')as z:np.save(z,f,allow_pickle=False);z.flush();os.fsync(z.fileno())
     features[h]=pin(p);return f
    a=ledger.call('embedding',h,embed)
   else:
    with checked(features[h]).open('rb')as f:a=np.load(f,allow_pickle=False)
   return torch.from_numpy(a).cuda()
  sources={s['source_id']:s for s in plan['sources']};last=None;metrics=['psnr_db','lpips_alex','dreamsim']if triple else['clip_image_cosine']
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   for j in sorted(plan['pairs'],key=lambda j:(j['source_index'],j['pair_key'])):
    assert time.monotonic()-beg<r['max_seconds']and torch.cuda.mem_get_info()[0]>=2<<30 and not(L/'STOP').exists();key=j['pair_key'];s=sources[j['source_id']]
    if key in reuse:rows[key]=dict(reuse[key],origin='EXACT_NATIVE_PAIR_REUSE');continue
    if last!=s['source_id']:
     with np.load(checked(s['archive']),allow_pickle=False)as z:x=z['pixels'].astype(np.float32)/np.float32(255)
     assert asha(x)==s['reference_sha256'];xt=torch.from_numpy(x[None]).cuda();last=s['source_id']
     if not triple:
      ref=references[last];assert ref['pixel_sha256']==s['reference_sha256']
      with checked(ref['feature']).open('rb')as f:fx=torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
     else:fx=feature(x,s['reference_sha256'])
    if j['image']is None:y=np.full((3,256,256),.5,np.float32)
    else:
     with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
    assert asha(y)==j['pixel_sha256']and y.dtype==np.float32 and y.shape==(3,256,256)and np.isfinite(y).all()and 0<=y.min()<=y.max()<=1
    fy=feature(y,j['pixel_sha256']);yt=torch.from_numpy(y[None]).cuda()
    def score():
     values=dict(psnr_db=float(psnr(x,y)),lpips_alex=float(lp_pair(lp,yt,xt)[0].item()),dreamsim=float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item()))if triple else dict(clip_image_cosine=float(F.cosine_similarity(fx,fy,dim=-1)[0].item()))
     assert set(values)==set(metrics)and all(np.isfinite(v)for v in values.values());record=dict(pair_key=key,source_id=s['source_id'],source_index=s['source_index'],reference_sha256=s['reference_sha256'],pixel_sha256=j['pixel_sha256'],origin='NEW_FROZEN_METRIC_PAIR',**values);save(out/'pairs'/(key+'.json'),record);return record
    rows[key]=ledger.call('pair',key,score)
    if len(rows)%100==0:print('FORMAL_PAIR',r['mode'],len(rows),len(plan['pairs']),flush=True)
  save(out/'scores.json',rows);save(out/'features.json',features);save(out/'completion.json',dict(status='COMPLETE_FORMAL_METRICS',request=rp,mode=r['mode'],scores=pin(out/'scores.json'),features=pin(out/'features.json'),ledger=ledger.summary(),count=len(rows),seconds=time.monotonic()-beg,new_reconstruction=0,new_PHY=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
