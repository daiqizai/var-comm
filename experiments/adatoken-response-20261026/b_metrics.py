"""Score closed actual-prefix ablations using unchanged frozen metric functions."""
from common import *
import argparse,importlib.util,traceback,ctypes
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def metric_inputs(reg):
 plan=cr(reg['plan']);keys=[(v['source_index'],v['snr_db'],v['noise_seed'],v['arm'])for v in plan['frames']]
 assert len(keys)==1800 and len(set(keys))==1800
 assert {k[3]for k in keys}=={'A0','B0','C0'}
 return plan
def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 plan=metric_inputs(r);out=L/role;out.mkdir(parents=True,exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,set(range(6)))
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 import numpy as np,torch
 import torch.nn.functional as F
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
 rr=r['resources'][role];assert torch.__version__==rr['torch']
 lib=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(device),0)==0
 bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,device)==0;assert bus.value.decode().lower()==rr['pci']
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 old_common=module(checked(r['persistent_ledger']),'_persistent_metric_ledger');ledger=old_common.Ledger(out,r['limits'][role]);beg=time.monotonic()
 def call(kind,key,fn,feature=False):
  assert time.monotonic()-beg<r['max_seconds']and torch.cuda.mem_get_info()[0]>=2<<30
  return ledger.call(kind,key,fn,feature)
 def model(kind,fn):
  box={}
  def make():box['v']=fn();return {'loaded':True}
  call(kind,'once',make);return box['v']
 cfg=r['config'];met=module(checked(cfg['clip']['loader']),'_frozen_ablation_metrics')
 triple=role=='b_three_metrics';features={};pairs={};rows=[]
 try:
  if triple:
   sys.path.insert(0,str(Path(r['original_four']['path']).parent));four=module(checked(r['original_four']),'_four');c=cfg['config']['metric_config'];assets=four.bound_assets(c)
   import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   psnr,p_proof=four.psnr_projection(c['sources']['score_bpg_cached_holdout_r2.py']['path']);lp_load,lp_pair,l_proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=model('load_lpips',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];overlay=read(checked(sup['dependency_overlay']))
   for d in overlay['files']:checked(d)
   sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=read(checked(sup['metric_assets']));torch.manual_seed(2026101401)
   with met.no_network():network=model('load_dreamsim',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   preprocess=met.preprocess_dreamsim
  else:
   with met.no_network():network=model('load_clip',lambda:met.load_clip(cfg['clip']['clip_assets'],torch.device('cuda:0')))
   preprocess=lambda x:met.preprocess_clip(met.validate_images(x))
   prior=cr(r['prior_clip_identity']);assert prior['fp32']and not prior['tf32']and not prior['amp']and prior['cuda_bus'].lower()==rr['pci']
   assert prior['assets']==cfg['clip']['clip_assets']and prior['loader']==cfg['clip']['loader']and prior['preprocessing']==cfg['clip']['metric_definition']
   refs={v['source_id']:v for v in cr(r['clip_cache'])}
  def feature(a,h):
   if h not in features:
    xt=torch.from_numpy(a[None]).cuda()
    fn=(lambda:network.embed(preprocess(xt)).float().cpu().numpy())if triple else(lambda:network.encode_image(preprocess(xt)).float().cpu().numpy())
    features[h]=call('image_embedding',h,fn,True)
   return torch.from_numpy(features[h]).cuda()
  save(out/'metric_identity.json',dict(loader=cfg['clip']['loader'],config=r['frozen_metric_registration'],torch=torch.__version__,cuda=torch.version.cuda,pci=bus.value.decode(),fp32=True,tf32=False,batch=1,threads=6,interop=2,original_PSNR_LPIPS_AST={'psnr':p_proof,'lpips':l_proof}if triple else None,reference_cache_reuse=0 if triple else len(refs)))
  with torch.inference_mode():
   for s in plan['sources']:
    with np.load(checked(s['archive']),allow_pickle=False)as z:raw=z['pixels'].copy()
    assert raw.dtype==np.uint8 and raw.shape==(3,256,256)and hashlib.sha256(raw.tobytes()).hexdigest()==s['preprocessing_id'];x=raw.astype(np.float32)/np.float32(255);assert asha(x)==s['reference_sha256'];xt=torch.from_numpy(x[None]).cuda();fx=None
    for j in [v for v in plan['frames']if v['source_id']==s['source_id']]:
     metrics=('psnr_db','lpips_alex','dreamsim')if triple else('clip_image_cosine',)
     if j['reuse_old']:
      values={m:j['old_values'][m]for m in metrics};origin='IDENTICAL_K0_OR_GRAY_EXISTING_OUTPUT'
     else:
      key=s['reference_sha256']+'_'+j['pixel_sha256'];origin='NEW_NATIVE_PAIR'if key not in pairs else'EXACT_PAIR_REUSE'
      if key not in pairs:
       with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
       assert asha(y)==j['pixel_sha256']and y.shape==(3,256,256)and y.dtype==np.float32 and np.isfinite(y).all()and y.min()>=0 and y.max()<=1
       if fx is None:
        if triple:fx=feature(x,s['reference_sha256'])
        else:
         ref=refs[s['source_id']];assert ref['pixel_sha256']==s['reference_sha256']
         with checked(ref['feature']).open('rb')as f:fx=torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
       fy=feature(y,j['pixel_sha256']);yt=torch.from_numpy(y[None]).cuda()
       if triple:values={'psnr_db':call('psnr',key,lambda:float(psnr(x,y))),'lpips_alex':call('lpips_pair',key,lambda:float(lp_pair(lp,yt,xt)[0].item())),'dreamsim':call('cosine_pair',key,lambda:float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item()))}
       else:values={'clip_image_cosine':call('cosine_pair',key,lambda:float(F.cosine_similarity(fx,fy,dim=-1)[0].item()))}
       assert all(np.isfinite(v)for v in values.values());pairs[key]=values
      values=pairs[key]
     rows.append({k:j[k]for k in('source_id','source_index','rank','snr_db','noise_seed','arm','m','K','accepted','pixel_sha256')}|dict(score_origin=origin,**values))
    print(role,len(rows),flush=True)
  assert len(rows)==1800 and all(st=='COMPLETE'for _,st,_ in ledger.counts());csvsave(out/'scores.csv',rows)
  save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,rows=1800,scores=pin(out/'scores.csv'),counts=ledger.counts(),unique_new_pairs=len(pairs),reused_old_rows=sum(v['reuse_old']for v in plan['frames']),elapsed_seconds=time.monotonic()-beg,new_reconstruction=0,new_PHY=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),counts=ledger.counts(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
