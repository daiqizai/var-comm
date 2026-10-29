"""Fourteen registered native 4 dB reconstructions; frozen LPIPS/DreamSim."""
from pathlib import Path
import argparse,ctypes,importlib.util,sys,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adatoken-response-20261026'))
from common import *
L=B/'nestcom_fig3_4db_20261029_v1'

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(role,path,expected):
 assert sha(path)==expected;r=read(path)
 for d in r['code']:checked(d)
 scope=cr(r['development_scope']);assert scope['source_indices']==[95,21]
 for d in r['actual_waits']:
  w=cr(d);assert w['returncode']==0 and gone(w['process'])and gone(w['parent_process'])
 jobs=cr(r['jobs']);assert len(jobs)==14 and {j['source_index']for j in jobs}=={95,21} and len({(j['method'],j['source_index'])for j in jobs})==14
 out=L/role;out.mkdir(exist_ok=False);runtime_cache(out);os.sched_setaffinity(0,set(range(6)))
 save(out/'launch.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid()),argv=sys.argv,registration_sha256=expected))
 import numpy as np,torch
 import torch.nn.functional as F
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
 assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
 lib=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(device),0)==0
 bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,device)==0;assert bus.value.decode().lower()==r['pci']
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 ledger=Ledger(out/'calls.sqlite',dict(load_lpips=1,load_dreamsim=1,lpips_pair=14,dreamsim_pair=14));beg=time.monotonic()
 try:
  donor=cr(r['metric_donor']);cfg=donor['config']
  met=module(checked(cfg['clip']['loader']),'_frozen_metric_loader')
  sys.path.insert(0,str(Path(donor['original_four']['path']).parent));four=module(checked(donor['original_four']),'_frozen_four')
  c=cfg['config']['metric_config'];sys.path.insert(0,str(R/'tools'));from checkpoint_relocation_20261028 import metric_assets,relocate_value;assets=metric_assets(four,c)
  import lpips
  assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
  lp_load,lp_pair,proof=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
  with met.no_network():lp=ledger.call('load_lpips','once',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
  sup=cfg['supplement'];overlay=read(checked(sup['dependency_overlay']))
  for d in overlay['files']:checked(d)
  sys.path.insert(0,cfg['config']['supplemental_overlay_root']);ma=relocate_value(read(checked(sup['metric_assets'])));torch.manual_seed(2026101401)
  with met.no_network():network=ledger.call('load_dreamsim','once',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
  save(out/'metric_identity.json',dict(loader=cfg['clip']['loader'],metric_assets=sup['metric_assets'],donor=r['metric_donor'],lpips_AST=proof,fp32=True,tf32=False,batch=1,threads=6,pci=r['pci'],preprocessing='Frozen ensemble DreamSim: float RGB,224 bicubic antialias; existing LPIPS-Alex loader'))
  rows=[]
  with torch.inference_mode():
   for j in jobs:
    assert time.monotonic()-beg<300 and torch.cuda.mem_get_info()[0]>=2<<30
    with np.load(checked(j['source']),allow_pickle=False)as z:raw=z['pixels'].copy()
    x=raw.astype(np.float32)/np.float32(255);assert asha(x)==j['source_array_sha256']
    with np.load(checked(j['image']),allow_pickle=False)as z:y=z['image'].copy()
    assert y.shape==(3,256,256)and y.dtype==np.float32 and asha(y)==j['pixel_sha256']
    xt=torch.from_numpy(x[None]).cuda();yt=torch.from_numpy(y[None]).cuda()
    lpvalue=ledger.call('lpips_pair',j['method']+':'+j['source_id'],lambda:float(lp_pair(lp,yt,xt)[0].item()))
    def dreamsim():
     fx=network.embed(met.preprocess_dreamsim(xt)).float();fy=network.embed(met.preprocess_dreamsim(yt)).float()
     return float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item())
    ds=ledger.call('dreamsim_pair',j['method']+':'+j['source_id'],dreamsim)
    rows.append(dict(source_id=j['source_id'],source_index=j['source_index'],method=j['method'],N=1024,snr_db=4,noise_seed=j['noise_seed'],lpips_alex=lpvalue,dreamsim=ds,source_array_sha256=j['source_array_sha256'],image_path=j['image']['path'],image_file_sha256=j['image']['sha256'],pixel_sha256=j['pixel_sha256'],receive_status=j['receive_status'],received_prefix=j['prefix']))
    rows[-1]['psnr_db']=float(-10*np.log10(np.mean((x.astype(np.float64)-y.astype(np.float64))**2)));save(out/('score_%s_%04d.json'%(j['method'],j['source_index'])),rows[-1])
  csvsave(out/'scores.csv',rows);save(out/'completion.json',dict(status='COMPLETE',registration_sha256=expected,scores=pin(out/'scores.csv'),rows=14,ledger=ledger.summary(),elapsed_seconds=time.monotonic()-beg,new_reconstruction=0,new_PHY=0))
 except BaseException:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role');p.add_argument('--registration',required=True);p.add_argument('--expected-registration',required=True);a=p.parse_args();main(a.role,a.registration,a.expected_registration)
