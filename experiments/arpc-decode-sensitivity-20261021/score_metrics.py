"""Scores only the fixed sensitivity outputs; original algorithms and assets."""
from common import *
import argparse,sys,sqlite3,traceback,importlib.util,tempfile,subprocess
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(role,expected):
 assert sha(O/'metric_registration.json')==expected
 reg=read(O/'metric_registration.json')
 for d in reg['source_files']+[reg['rx_completion'],reg['rx_wait']]:checked(d)
 out=L/role;out.mkdir();save(out/'launch.json',dict(process=process(os.getpid()),parent_process=process(os.getppid()),argv=sys.argv))
 cache=out/'runtime_cache';(cache/'tmp').mkdir(parents=True);tempfile.tempdir=str(cache/'tmp')
 os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
  TRITON_CACHE_DIR=str(cache/'triton'),CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 os.sched_setaffinity(0,set(range(6)))
 import numpy as np,torch,ctypes
 import torch.nn.functional as F
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 torch.cuda.set_device(0)
 expected_torch='2.11.0+cu128' if role=='three_metrics' else '2.10.0+cu130'
 assert torch.__version__==expected_torch
 lib=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();assert lib.cuDeviceGet(ctypes.byref(device),0)==0
 bus=ctypes.create_string_buffer(32);assert lib.cuDeviceGetPCIBusId(bus,32,device)==0
 expected_bus='0000:63:00.0' if role=='three_metrics' else '0000:62:00.0'
 assert bus.value.decode().lower()==expected_bus
 assert torch.cuda.mem_get_info()[0]>=8<<30
 torch.cuda.set_per_process_memory_fraction((12<<30)/torch.cuda.get_device_properties(0).total_memory)
 db=sqlite3.connect(out/'ledger.sqlite');db.execute('CREATE TABLE calls(kind TEXT,id TEXT,status TEXT,PRIMARY KEY(kind,id))');db.commit()
 limits=reg['three_limits' if role=='three_metrics' else 'clip_limits'];start=time.monotonic()
 def call(kind,key,fn):
  assert time.monotonic()-start<reg['max_seconds'] and torch.cuda.mem_get_info()[0]>=2<<30
  assert db.execute('SELECT COUNT(*) FROM calls WHERE kind=?',(kind,)).fetchone()[0]<limits[kind]
  db.execute('INSERT INTO calls VALUES(?,?,?)',(kind,key,'RESERVED'));db.commit()
  try:value=fn()
  except BaseException:
   db.execute('UPDATE calls SET status="FAILED" WHERE kind=? AND id=?',(kind,key));db.commit();raise
  db.execute('UPDATE calls SET status="COMPLETE" WHERE kind=? AND id=?',(kind,key));db.commit();return value
 def source(s):
  checked(s['archive'])
  with np.load(s['archive']['path'],allow_pickle=False) as z:x=z['pixels'].copy()
  assert x.dtype==np.uint8 and x.shape==(3,256,256);return x.astype(np.float32)/np.float32(255)
 def image(j):
  with np.load(checked(j['image']),allow_pickle=False) as z:y=z['float_output'].copy()
  assert y.dtype==np.float32 and y.shape==(3,256,256) and np.isfinite(y).all() and y.min()>=0 and y.max()<=1
  assert hashlib.sha256(y.tobytes()).hexdigest()==j['float_sha256'];return y
 cfg=reg['config'];jobs=read(checked(reg['jobs']));sources=read(checked(reg['sources']))
 met=load(checked(cfg['clip']['loader']),'_frozen_sensitivity_metrics')
 identity=dict(role=role,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
  bus=bus.value.decode(),FP32=True,AMP=False,TF32=False,threads=6,interop=2,batch_size=1,
  loader=cfg['clip']['loader'],registration_sha256=expected)
 try:
  if role=='three_metrics':
   sys.path.insert(0,str(Path(reg['original_four']['path']).parent))
   four=load(checked(reg['original_four']),'_original_four_projection')
   assets=four.bound_assets(cfg['config']['metric_config'])
   import lpips
   assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==four.WEIGHT_SHA['lpips']
   c=cfg['config']['metric_config']
   psnr,proof_p=four.psnr_projection(c['sources']['score_bpg_cached_holdout_r2.py']['path'])
   lp_load,lp_pair,proof_l=four.lpips_projection(c['sources']['quality.py']['path'],dict(Path=Path,lpips=lpips,torch=torch))
   with met.no_network():lp=call('load_lpips','alex',lambda:lp_load({'alexnet_checkpoint':assets['weights']['alexnet']['path']},'cuda:0'))
   sup=cfg['supplement'];checked(sup['loader']);overlay=read(checked(sup['dependency_overlay']))
   for d in overlay['files']:checked(d)
   sys.path.insert(0,cfg['config']['supplemental_overlay_root'])
   ma=read(checked(sup['metric_assets']));torch.manual_seed(2026101401)
   with met.no_network():dream=call('load_dreamsim','ensemble',lambda:met.load_dreamsim(ma['models']['dreamsim'],torch.device('cuda:0')))
   identity.update(original_metric_config=c,PSNR_AST=proof_p,LPIPS_AST=proof_l,dreamsim=ma['models']['dreamsim'],protocol=sup['protocol']['dreamsim'])
  else:
   with met.no_network():clip=call('load_clip','vit_l14',lambda:met.load_clip(cfg['clip']['clip_assets'],torch.device('cuda:0')))
   prior=read(checked(reg['prior_clip_identity']))
   assert prior['fp32'] and not prior['tf32'] and not prior['amp'] and prior['cuda_bus'].lower()==expected_bus
   assert prior['preprocessing']==cfg['clip']['metric_definition'] and prior['loader']==cfg['clip']['loader']
   assert prior['assets']==cfg['clip']['clip_assets']
   identity.update(prior_identity=reg['prior_clip_identity'],preprocessing=cfg['clip']['metric_definition'],assets=cfg['clip']['clip_assets'],source_feature_reuse=100)
   refs={v['source_id']:v for v in read(checked(reg['clip_cache']))}
  save(out/'metric_identity.json',identity)
  (out/'pairs').mkdir();results=[];values_cache={};features={}
  with torch.inference_mode():
   for s in sources:
    x=source(s);xt=torch.from_numpy(x[None]).cuda()
    if role=='three_metrics':fx=call('dreamsim_reference',s['source_id'],lambda:dream.embed(met.preprocess_dreamsim(xt)).float())
    else:
     with checked(refs[s['source_id']]['feature']).open('rb') as f:fx=torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
    for j in [v for v in jobs if v['source_id']==s['source_id']]:
     y=image(j);key=j['reference_sha256']+'_'+j['pixel_sha256']
     if key not in values_cache:
      yt=torch.from_numpy(y[None]).cuda()
      if role=='three_metrics':
       values=dict(psnr_db=call('psnr',key,lambda:float(psnr(x,y))),
        lpips_alex=call('lpips_pair',key,lambda:float(lp_pair(lp,yt,xt)[0].item())))
       fy=call('dreamsim_image',key,lambda:dream.embed(met.preprocess_dreamsim(yt)).float())
       values['dreamsim']=float((1-F.cosine_similarity(fx,fy,dim=-1))[0].item())
      else:
       if j['pixel_sha256'] not in features:
        fy=call('clip_image',j['pixel_sha256'],lambda:clip.encode_image(met.preprocess_clip(met.validate_images(yt))).float())
        features[j['pixel_sha256']]=fy
       values={'clip_image_cosine':call('cosine_pair',key,lambda:float(F.cosine_similarity(fx,features[j['pixel_sha256']],dim=-1)[0].item()))}
      assert all(np.isfinite(v) for v in values.values())
      save(out/'pairs'/(key+'.json'),dict(reference=s['archive'],image=j['image'],values=values))
      values_cache[key]=values
     results.append(dict(setting=j['setting'],key=j['key'],source_id=j['source_id'],source_index=j['source_index'],
       pixel_sha256=j['pixel_sha256'],**values_cache[key]))
    if len(results)%60<6:print(role,len(results),flush=True)
  assert len(results)==592
  counts=db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status').fetchall()
  assert all(st=='COMPLETE' for k,st,n in counts)
  csvsave(out/'scores.csv',results)
  save(out/'completion.json',dict(status='COMPLETE',scores=desc(out/'scores.csv'),identity=desc(out/'metric_identity.json'),
    rows=len(results),unique_pairs=len(values_cache),pair_reuse=len(results)-len(values_cache),counts=counts,
    elapsed_seconds=time.monotonic()-start,unresolved=0))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
 finally:db.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('role',choices=['three_metrics','clip_metrics']);p.add_argument('--registration',required=True)
 a=p.parse_args();main(a.role,a.registration)
