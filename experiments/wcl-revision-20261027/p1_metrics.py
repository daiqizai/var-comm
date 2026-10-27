"""Only missing scores on existing native outputs or the prescribed gray output."""
from p1_common import *
import importlib.util,ctypes,traceback,tempfile
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(role,expected):
 rp=O/'p1_metrics_registration.json';assert sha(rp)==expected;r=read(rp)
 for d in r['code']:checked(d)
 out=L/('metrics_'+role);out.mkdir();(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
 os.environ.update(TMPDIR=str(out/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'),TRITON_CACHE_DIR=str(out/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(range(6)));save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv))
 import torch,torch.nn.functional as F
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
 rr=r['resources'][role];assert torch.__version__==rr['torch'];lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(dev),0)==0;bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,dev)==0 and bus.value.decode().lower()==rr['pci'];assert torch.cuda.mem_get_info()[0]>=8<<30
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 donor=cr(r['metric_donor']);cfg=donor['config'];met=module(checked(cfg['clip']['loader']),'_frozen_metrics');lc=module(checked(donor['persistent_ledger']),'_persistent_calls');n=r['limits'][role]
 limits={'load_lpips':1,'load_dreamsim':1,'load_clip':1,'image_embedding':2*n+1,'psnr':n,'lpips_pair':n,'cosine_pair':n};ledger=lc.Ledger(out,limits);start=time.monotonic()
 def call(kind,key,fn,feature=False):
  assert time.monotonic()-start<r['max_seconds']and torch.cuda.mem_get_info()[0]>=2<<30;return ledger.call(kind,key,fn,feature)
 def model(kind,fn):
  box={}
  def load():box['v']=fn();return {'loaded':True}
  call(kind,'once',load);return box['v']
 try:
  triple=role=='three';frole='a_three_metrics'if triple else'a_clip_metrics';feature_cache=cr(donor['feature_cache'][frole]);features={};refs=None
  assert donor['resources'][frole]['pci']==rr['pci']and donor['resources'][frole]['torch']==rr['torch']
  if triple:
   sys.path.insert(0,str(Path(donor['original_four']['path']).parent));four=module(checked(donor['original_four']),'_original_four');c=cfg['config']['metric_config'];assets=four.bound_assets(c)
   import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   psnr,pp=four.psnr_projection(c['sources']['score_bpg_cached_holdout_r2.py']['path']);lp_load,lp_pair,lp_proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=model('load_lpips',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];overlay=cr(sup['dependency_overlay'])
   for d in overlay['files']:checked(d)
   sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=cr(sup['metric_assets']);torch.manual_seed(2026101401)
   with met.no_network():network=model('load_dreamsim',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   preprocess=met.preprocess_dreamsim
  else:
   with met.no_network():network=model('load_clip',lambda:met.load_clip(cfg['clip']['clip_assets'],torch.device('cuda:0')))
   preprocess=lambda x:met.preprocess_clip(met.validate_images(x));prior=cr(donor['prior_clip_identity']);assert prior['cuda_bus'].lower()==rr['pci']and prior['fp32']and not prior['tf32']and not prior['amp']and prior['loader']==cfg['clip']['loader']and prior['assets']==cfg['clip']['clip_assets']
   refs={s['source_id']:s for s in cr(donor['clip_cache'])}
  save(out/'metric_identity.json',dict(role=role,registration=pin(rp),donor=r['metric_donor'],device=torch.cuda.get_device_name(0),pci=bus.value.decode(),torch=torch.__version__,fp32=True,tf32=False,batch=1,threads=6,reference_feature_cache=donor['feature_cache'][frole],native_arrays=True))
  def feature(a,h):
   if h in feature_cache:
    with checked(feature_cache[h]).open('rb')as f:return torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
   if h not in features:
    xt=torch.from_numpy(a[None]).cuda();fn=(lambda:network.embed(preprocess(xt)).float().cpu().numpy())if triple else(lambda:network.encode_image(preprocess(xt)).float().cpu().numpy())
    features[h]=call('image_embedding',h,fn,True)
   return torch.from_numpy(features[h]).cuda()
  sources={s['source_id']:s for s in cr(r['sources'])};jobs=cr(r['pairs']);metrics=('psnr_db','lpips_alex','dreamsim')if triple else('clip_image_cosine',);result=[];new=0;sourcecache={}
  with torch.inference_mode():
   for j in jobs:
    if all(m in j['values']for m in metrics):values={m:j['values'][m]for m in metrics};origin='EXACT_PRIOR_NATIVE_PAIR_SCORE'
    else:
     sid=j['source_id'];s=sources[sid]
     if sid not in sourcecache:
      with np.load(checked(s['archive']),allow_pickle=False)as z:raw=z['pixels'].copy()
      assert hashlib.sha256(raw.tobytes()).hexdigest()==s['preprocessing_id'];x=raw.astype(np.float32)/np.float32(255);assert ash(x)==j['reference_sha256']
      if triple:fx=feature(x,j['reference_sha256'])
      else:
       ref=refs[sid];assert ref['pixel_sha256']==j['reference_sha256']
       with checked(ref['feature']).open('rb')as f:fx=torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
      sourcecache[sid]=(x,fx)
     x,fx=sourcecache[sid]
     with np.load(checked(j['image']),allow_pickle=False)as z:y=z[j['image_key']].copy()
     if j['image_slot']is not None:y=y[j['image_slot']]
     assert y.dtype==np.float32 and y.shape==(3,256,256)and ash(y)==j['pixel_sha256']and np.isfinite(y).all()and y.min()>=0 and y.max()<=1
     fy=feature(y,j['pixel_sha256']);key=j['key']
     if triple:
      xt=torch.from_numpy(x[None]).cuda();yt=torch.from_numpy(y[None]).cuda()
      values=dict(psnr_db=call('psnr',key,lambda:float(psnr(x,y))),lpips_alex=call('lpips_pair',key,lambda:float(lp_pair(lp,yt,xt)[0].item())),dreamsim=call('cosine_pair',key,lambda:float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item())))
     else:values=dict(clip_image_cosine=call('cosine_pair',key,lambda:float(F.cosine_similarity(fx,fy,dim=-1)[0].item())))
     origin='NEW_MISSING_PAIR';new+=1
    assert all(np.isfinite(v)for v in values.values());result.append(dict(key=j['key'],source_id=j['source_id'],source_index=j['source_index'],pixel_sha256=j['pixel_sha256'],origin=origin,**values))
  assert new==n and all(st=='COMPLETE'for _,st,_ in ledger.counts());csvsave(out/'scores.csv',result)
  save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,scores=pin(out/'scores.csv'),new_pairs=new,reused_pairs=len(result)-new,counts=ledger.counts(),seconds=time.monotonic()-start,new_VAR=0,new_PHY=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),counts=ledger.counts(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
