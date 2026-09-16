#!/usr/bin/env python3
"""CPU-only continuation of bounded assets preparation; releases readiness only on success."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys, time
R=pathlib.Path(__file__).resolve().parents[1]
NAME='ei-liulu-xqvar-eval-20260912-v1'
XQ=[('checkpoints/MSVR10P2-4096/best_ckpt.pt',3534394346,'fa349cec5f953e41a30d1877344fba7e485d5d059fb24b0de83d547aa0c70e53'),('checkpoints/VAR-d17-MSVR10P2-4096/ar-ckpt-last.pth',5537116322,'37db545025e1421d91f46e28733d37ef81b0178e2ce229ecbd7c3afcfe4c7c31')]
WETOK=('checkpoints/wetok-imagenet-downsample16/WeTok.ckpt',3763537210,'e40a0bcdefd8509b2201e93a9f5007d38c8fba9423c17a98b0e45f9e8763e696')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=pathlib.Path(p);tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def state(s,**kw):dump(R/'docs/prepare-status.json',{'status':s,'task_name':NAME,'at':datetime.datetime.now().astimezone().isoformat(),'pid':os.getpid(),'gpu_context_created':False,**kw})
def identity(pid):
 try:
  s=pathlib.Path(f'/proc/{pid}/stat').read_text();parts=s[s.rfind(')')+2:].split();return {'start_ticks':int(parts[19]),'state':parts[0]}
 except FileNotFoundError:return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-name',required=True);p.add_argument('--wait-download-pid',type=int,required=True);a=p.parse_args()
 if a.task_name!=NAME:raise ValueError('Wrong owner/task')
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 os.nice(10);subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
 import fcntl
 with (R/'docs/preparation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (R/'docs/assets-ready.json').exists():raise RuntimeError('Already ready; refuse rerun/overwrite')
  initial=identity(a.wait_download_pid);start=time.monotonic()
  state('WAITING_EXISTING_SINGLE_XQ_DOWNLOAD',download_pid=a.wait_download_pid,download_identity=initial)
  while True:
   current=identity(a.wait_download_pid)
   if current is None or current['state']=='Z' or initial is None or current['start_ticks']!=initial['start_ticks']:break
   if time.monotonic()-start>14400:raise TimeoutError('Existing XQ download exceeded4h; no signals sent')
   time.sleep(30)
  for rel,n,want in XQ:
   f=R/rel
   if not f.is_file() or f.stat().st_size!=n:raise RuntimeError('XQ download incomplete; preserve partials for inspection '+rel)
  d=json.loads((R/'docs/xq-download-status.json').read_text())
  if d['status']!='XQ_ASSETS_VERIFIED':raise RuntimeError('XQ download did not produce verified completion')
  rel,n,want=WETOK;f=R/rel;f.parent.mkdir(parents=True,exist_ok=True)
  if not f.exists():
   if os.statvfs(R).f_bavail*os.statvfs(R).f_frsize < 40*1024**3:raise RuntimeError('Less than40GiB free; no more downloads')
   tmp=f.with_name(f.name+'.part');state('DOWNLOADING_WETOK_SINGLE_STREAM',asset=rel,expected_bytes=n)
   url='https://hf-mirror.com/GrayShine/WeTok/resolve/85fc6eb084d458b8d4fa3a32d541379e95b2bf87/ImageNet/downsample16/WeTok.ckpt?download=true'
   with (R/'docs/wetok-download.log').open('a') as log:
    for attempt in range(3):
     result=subprocess.run(['curl','-fL','--connect-timeout','10','--max-time','3600','--speed-time','90','--speed-limit','1024','--limit-rate','8M','--max-filesize',str(n),'-C','-','-o',str(tmp),url+'&attempt='+str(attempt)],stdout=log,stderr=subprocess.STDOUT)
     if result.returncode==0:break
     time.sleep(5)
    if result.returncode:raise RuntimeError('WeTok download failed; retained partial file')
   if tmp.stat().st_size!=n or sha(tmp)!=want:raise RuntimeError('WeTok byte/SHA mismatch')
   tmp.replace(f)
  state('WAITING_IMPLEMENTATION_CPU_TESTS')
  deadline=time.monotonic()+14400
  while not (R/'docs/implementation-ready.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('Implementation was not declared ready within4h')
   time.sleep(20)
  implementation=json.loads((R/'docs/implementation-ready.json').read_text())
  if implementation['status']!='IMPLEMENTATION_CPU_TESTS_PASS':raise RuntimeError('CPU preparation incomplete')
  for p,h in implementation['runtime_source_sha256'].items():
   if sha(p)!=h:raise RuntimeError('Source changed after CPU checks '+p)
  state('VERIFYING_HASHES_AND_CPU_CHECKPOINT_LAYOUT')
  # Import after limiting threads and hiding all CUDA devices. No GPU calls here.
  sys.path.insert(0,str(R/'scripts'))
  import torch
  torch.set_num_threads(2);torch.set_num_interop_threads(2)
  from evaluate_backbones import resolve_config
  from checkpoint_io import load_xq_checkpoint
  config_path=R/'configs/eval.local.json';c=resolve_config(config_path)
  hashes={}
  for rel,n,want in [*XQ,WETOK]:
   f=R/rel
   if f.stat().st_size!=n or sha(f)!=want:raise RuntimeError('Candidate asset checksum failure '+str(f))
   hashes[str(f)]=want
  for key in ('old_vae_checkpoint','old_var_checkpoint','fidelity_checkpoint','dino_checkpoint'):
   f=pathlib.Path(c[key]);want=c[key+'_sha256']
   if sha(f)!=want:raise RuntimeError('Historical asset checksum failure '+key)
   hashes[str(f)]=want
  if sha(c['manifest'])!=c['manifest_sha256']:raise RuntimeError('Manifest changed')
  rows=json.loads(pathlib.Path(c['manifest']).read_text())
  if len(rows)!=100 or len({x['image_id'] for x in rows})!=100:raise RuntimeError('Not the frozen100development')
  images=[{'image_id':x['image_id'],'path':x['path'],'sha256':sha(x['path']),'class_index':x['class_index'],'role':'development'} for x in rows]
  dump(R/'docs/development-images-verified.json',images)
  cpu_layout={}
  for tag,key in [('xq_generator','xq_generator_checkpoint'),('xq_tokenizer','xq_tokenizer_checkpoint'),('wetok','wetok_checkpoint')]:
   # XQ: exact-SHA restricted metadata loader. WeTok: built-in weights-only.
   ck=load_xq_checkpoint(c[key]) if tag.startswith('xq') else torch.load(c[key],map_location='cpu',mmap=True,weights_only=True)
   cpu_layout[tag]={'top_level_keys':list(ck),'unsafe_globals':torch.serialization.get_unsafe_globals_in_checkpoint(c[key])}
   if tag=='xq_generator':
    s=ck['trainer']['var_wo_ddp'];v=ck['trainer']['vae_local']
    if tuple(s['head.weight'].shape)!=(8192,1088) or tuple(s['pos_1LC'].shape)!=(1,286,1088):raise ValueError('Wrong XQ generator')
    if any(tuple(v[f'quantizes.{k}.embedding.weight'].shape)!=(4096,32) for k in range(2)):raise ValueError('Missing/different PQ branch')
    cpu_layout[tag]['head_shape']=list(s['head.weight'].shape);cpu_layout[tag]['position_shape']=list(s['pos_1LC'].shape)
   if tag=='wetok' and not any(k.startswith('model_ema.') for k in ck['state_dict']):raise ValueError('WeTok lacks EMA inference state')
   del ck
  dump(R/'docs/checkpoint-layout-audit.json',cpu_layout)
  lock=json.loads((R/'docs/xq-source-lock.json').read_text())
  for item in lock['files']:
   f=R/'vendor/xqgan-source'/item['path']
   if sha(f)!=item['sha256']:raise RuntimeError('Pinned XQ source changed')
  import importlib.metadata as md
  versions={k:md.version(k) for k in ['torch','torchvision','timm','transformers','peft','einops','lpips','pytorch-msssim','omegaconf','numpy','ruamel.yaml']}
  if torch.cuda.is_initialized():raise RuntimeError('CPU preparation unexpectedly created CUDA context')
  receipt={'status':'ASSETS_VERIFIED_CPU_PREFLIGHT_PASS','at':datetime.datetime.now().astimezone().isoformat(),'task_name':NAME,
   'evaluation_config_sha256':sha(config_path),'runtime_source_sha256':implementation['runtime_source_sha256'],
   'asset_sha256':hashes,'source_manifest_sha256':sha(c['manifest']),'development_images':100,'cpu_layout':cpu_layout,'versions':versions,
   'gpu_context_created':False,'optimizer_updates':0,'source_only':True,'quality_results_exist':False}
  dump(R/'docs/assets-ready.json',receipt);state('ASSETS_VERIFIED_CPU_PREFLIGHT_PASS')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  state('PREPARATION_FAILED_NO_GPU_LAUNCH',error=repr(e));raise
