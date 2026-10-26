"""Frozen main four-metric algorithms, new Kodak population only; no reconstruction."""
from common import *
import argparse,sys,traceback,importlib.util,tempfile,subprocess,ctypes
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(role,expected):
 assert sha(O/'metric_registration.json')==expected;reg=read(O/'metric_registration.json')
 for d in reg['source_files']+[reg['plan'],reg['population'],reg['parent']]:checked(d)
 out=L/role;out.mkdir();save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv))
 cache=out/'runtime_cache';(cache/'tmp').mkdir(parents=True);tempfile.tempdir=str(cache/'tmp')
 os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(range(6)))
 import numpy as np,torch
 import torch.nn.functional as F
 from PIL import Image
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 torch.cuda.set_device(0);rr=reg['resources'][role]
 assert torch.__version__==rr['torch'] and str(Path(sys.executable).resolve())==str(Path(rr['python']).resolve())
 lib=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(device),0)==0
 bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,device)==0
 assert normalized_pci_bus(bus.value.decode())==normalized_pci_bus(rr['pci_bus_id']);assert str(torch.cuda.get_device_properties(0).uuid).lower()==rr['cuda_uuid'].lower();assert torch.cuda.mem_get_info()[0]>=8<<30
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 ledger=Ledger(out,reg['limits'][role]);start=time.monotonic();cfg=reg['frozen_config'];plan=read(checked(reg['plan']))
 def call(kind,key,fn,feature=False):
  assert time.monotonic()-start<reg['max_seconds'] and torch.cuda.mem_get_info()[0]>=2<<30
  return ledger.call(kind,key,fn,feature)
 def model(kind,key,fn):
  box={}
  def make():box['model']=fn();return dict(loaded=True)
  call(kind,key,make);assert 'model' in box;return box['model']
 def image(j):
  with np.load(checked(j['image']),allow_pickle=False) as z:y=z[j['image_key']].copy()
  assert_array(y,j['pixel_sha256'],'dtype_shape_cbytes');return y
 met=load(checked(cfg['clip']['loader']),'_frozen_kodak_metrics')
 identity=dict(role=role,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),bus=bus.value.decode(),FP32=True,AMP=False,TF32=False,threads=6,interop=2,batch_size=1,loader=cfg['clip']['loader'],registration_sha256=expected,resources=rr)
 try:
  if role=='three_metrics':
   sys.path.insert(0,str(Path(reg['original_four']['path']).parent));four=load(checked(reg['original_four']),'_original_four_projection');c=cfg['config']['metric_config'];assets=four.bound_assets(c)
   import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   psnr,proof_p=four.psnr_projection(checked(c['sources']['score_bpg_cached_holdout_r2.py']))
   lp_load,lp_pair,proof_l=four.lpips_projection(checked(c['sources']['quality.py']),dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=model('load_lpips','alex',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];checked(sup['loader']);overlay=read(checked(sup['dependency_overlay']))
   for d in overlay['files']:checked(d)
   sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=read(checked(sup['metric_assets']));torch.manual_seed(2026101401)
   with met.no_network():dream=model('load_dreamsim','ensemble',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   identity.update(original_metric_config=c,PSNR_AST=proof_p,LPIPS_AST=proof_l,dreamsim=ma['models']['dreamsim'],protocol=sup['protocol']['dreamsim'])
  else:
   with met.no_network():clip=model('load_clip','vit_l14',lambda:met.load_clip(cfg['clip']['clip_assets'],torch.device('cuda:0')))
   identity.update(preprocessing=cfg['clip']['metric_definition'],assets=cfg['clip']['clip_assets'])
  save(out/'metric_identity.json',identity);results=[];features={};pair_values={}
  def feature(x,h,source=False):
   if h not in features:
    xt=torch.from_numpy(x[None]).cuda()
    if role=='three_metrics':fn=lambda:dream.embed(met.preprocess_dreamsim(xt)).float().cpu().numpy()
    else:fn=lambda:clip.encode_image(met.preprocess_clip(met.validate_images(xt))).float().cpu().numpy()
    features[h]=call('image_embedding',h,fn,True)
   return torch.from_numpy(features[h]).cuda()
  with torch.inference_mode():
   for s in plan['sources']:
    p=checked(s['png']);x=np.ascontiguousarray(np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8).transpose(2,0,1)).astype(np.float32)/np.float32(255)
    assert ah(x)==s['reference_sha256'];fx=feature(x,s['reference_sha256'],True);xt=torch.from_numpy(x[None]).cuda()
    for j in [v for v in plan['frames'] if v['source_id']==s['source_id']]:
     key=s['reference_sha256']+'_'+j['pixel_sha256']
     if key not in pair_values:
      y=image(j);fy=feature(y,j['pixel_sha256']);values={}
      if role=='three_metrics':
       yt=torch.from_numpy(y[None]).cuda()
       for metric,kind,fn in [('psnr_db','psnr',lambda:float(psnr(x,y))),('lpips_alex','lpips_pair',lambda:float(lp_pair(lp,yt,xt)[0].item()))]:
        values[metric]=float(j['existing_'+metric]) if j['existing_'+metric]!='' else call(kind,key,fn)
       values['dreamsim']=call('cosine_pair',key,lambda:float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item()))
      else:values={'clip_image_cosine':call('cosine_pair',key,lambda:float(F.cosine_similarity(fx,fy,dim=-1)[0].item()))}
      assert all(np.isfinite(v) for v in values.values());pair_values[key]=values
     results.append(dict(method=j['method'],source_index=j['source_index'],source_id=j['source_id'],snr_db=j['snr_db'],noise_seed=j['noise_seed'],rx_status=j['rx_status'],pixel_sha256=j['pixel_sha256'],**pair_values[key]))
    print(role,len(results),flush=True)
  assert len(results)==1296;counts=ledger.counts();assert all(st=='COMPLETE' for k,st,n in counts)
  csvsave(out/'scores.csv',results);save(out/'completion.json',dict(status='COMPLETE',scores=desc(out/'scores.csv'),identity=desc(out/'metric_identity.json'),rows=len(results),unique_pairs=len(pair_values),unique_image_features=len(features),pair_reuse=len(results)-len(pair_values),counts=counts,elapsed_seconds=time.monotonic()-start,unresolved=0,scientific_zero=['reconstruction','channel','training','caption','tokenizer','bootstrap']))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
 finally:ledger.db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['three_metrics','clip_metrics']);p.add_argument('--registration',required=True);a=p.parse_args();main(a.role,a.registration)
