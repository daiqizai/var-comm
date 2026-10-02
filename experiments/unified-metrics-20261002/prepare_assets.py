"""Acquire pinned public evaluators in an isolated cache; never touch old jobs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/UNIFIED-METRICS-20261002';ASSETS=OUT/'assets'
SOURCES={
 'clip':('openai/CLIP','d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'),
 'dists':('dingkeyan93/DISTS','1267d8cb626c98706db3697422701c56a85ebf2e'),
 'dreamsim':('ssundaram21/dreamsim','db4d16c6948314e36e2e62f0bae2ee23bc974bf3'),
 'ms_ssim':('VainF/pytorch-msssim','b057b072dd869ae3f6b88543786f44d008315f69'),
 'dino':('facebookresearch/dino','7c446df5b9f45747937fb0d72314eb9f7b66930a'),
}
CLIP_SHA='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
DINO2_REVISION='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
WEIGHTS={
 'ViT-L-14.pt':('https://openaipublic.azureedge.net/clip/models/'+CLIP_SHA+'/ViT-L-14.pt',CLIP_SHA),
 'resnet50-11ad3fa6.pth':('https://download.pytorch.org/models/resnet50-11ad3fa6.pth','11ad3fa6'),
 'dino_vitbase16_pretrain.pth':('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',None),
 'dinov2_vitl14_pretrain.pth':('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',None),
 'dreamsim_ensemble_checkpoint.zip':('https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip',None),
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(t,p)

def download(url,path,expected=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 receipt=path.with_suffix(path.suffix+'.download.json')
 if path.exists():
  h=sha(path)
  if expected and not h.startswith(expected):raise RuntimeError('Existing asset hash differs: '+str(path))
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if old['url']!=url or old['sha256']!=h:raise RuntimeError('Frozen download changed')
  else:write(receipt,dict(url=url,sha256=h,bytes=path.stat().st_size))
  return path
 partial=path.with_suffix(path.suffix+'.part')
 for attempt in range(4):
  try:
   size=partial.stat().st_size if partial.exists() else 0
   headers={'User-Agent':'VAR-COMM-registered-metrics/1.0'}
   if size:headers['Range']=f'bytes={size}-'
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
    append=response.status==206
    if append and not response.headers.get('Content-Range','').startswith(f'bytes {size}-'):
     raise RuntimeError('Unverified HTTP resume position')
    mode='ab' if append else 'wb'
    expected_bytes=response.headers.get('Content-Length')
    final_bytes=(size if append else 0)+int(expected_bytes) if expected_bytes is not None else None
    with partial.open(mode) as f:
     for b in iter(lambda:response.read(1024*1024),b''):f.write(b)
    if final_bytes is not None and partial.stat().st_size!=final_bytes:
     raise RuntimeError('Incomplete HTTP asset response')
   h=sha(partial)
   if expected and not h.startswith(expected):raise RuntimeError('Downloaded asset hash differs: '+str(path))
   os.replace(partial,path)
   write(receipt,dict(url=url,sha256=h,bytes=path.stat().st_size,time=time.time()))
   return path
  except Exception:
   if attempt==3:raise
   time.sleep(2*(attempt+1))

def safe_target(folder,relative):
 parts=PurePosixPath(relative).parts
 if PurePosixPath(relative).is_absolute() or '..' in parts or not parts:raise ValueError('Unsafe archive member')
 target=(folder/Path(*parts)).resolve()
 if not target.is_relative_to(folder.resolve()):raise ValueError('Archive escape')
 return target

def extract_tar(archive,folder):
 marker=folder/'_archive_identity.json';identity=sha(archive)
 if marker.exists():
  if json.loads(marker.read_text())['sha256']!=identity:raise RuntimeError('Source archive changed')
  return
 folder.mkdir(parents=True,exist_ok=True)
 with tarfile.open(archive) as tar:
  for item in tar.getmembers():
   parts=PurePosixPath(item.name).parts
   if len(parts)<2:continue
   target=safe_target(folder,str(PurePosixPath(*parts[1:])))
   if item.isdir():target.mkdir(parents=True,exist_ok=True)
   elif item.isfile():
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
   else:raise ValueError('Links and special archive members are not accepted')
 write(marker,dict(sha256=identity))

def extract_zip(archive,folder):
 marker=folder/'_archive_identity.json';identity=sha(archive)
 if marker.exists():
  if json.loads(marker.read_text())['sha256']!=identity:raise RuntimeError('Checkpoint archive changed')
  return
 folder.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for item in z.infolist():
   target=safe_target(folder,item.filename)
   if item.is_dir():target.mkdir(parents=True,exist_ok=True)
   else:
    if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink in checkpoint archive')
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
 write(marker,dict(sha256=identity))

def source_job(name):
 repo,rev=SOURCES[name];archive=download(f'https://codeload.github.com/{repo}/tar.gz/{rev}',ASSETS/'archives'/f'{name}-{rev}.tar.gz')
 folder=ASSETS/'sources'/name;extract_tar(archive,folder)
 return name,dict(path=str(folder),revision=rev,files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*.py'))})

def weight_spec(path):return dict(path=str(path),sha256=sha(path))

def registered_dinov2_source():
 # This exact vendored source already supplies the legacy S/14 evaluator.
 # All 157 Python git blobs were matched to the pinned official upstream tree.
 receipt=json.loads((OUT/'dinov2_source_provenance.json').read_text())
 if receipt.get('upstream_commit')!=DINO2_REVISION or receipt.get('git_blob_match') is not True or receipt.get('mismatches'):
  raise RuntimeError('Original DINOv2 source upstream verification is missing')
 folder=Path(receipt['local_path'])
 files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*.py'))}
 if files!=receipt['python_sha256'] or len(files)!=receipt['compared_python_files']:
  raise RuntimeError('Verified DINOv2 source changed')
 return dict(path=str(folder),revision=DINO2_REVISION,files=files)

def main():
 OUT.mkdir(parents=True,exist_ok=True);write(OUT/'assets_status.json',dict(status='DOWNLOADING',time=time.time()))
 try:
  with ThreadPoolExecutor(max_workers=3) as pool:sources=dict(pool.map(source_job,SOURCES))
  def weight_job(item):
   name,(url,expected)=item;return name,download(url,ASSETS/'weights'/name,expected)
  with ThreadPoolExecutor(max_workers=3) as pool:weights=dict(pool.map(weight_job,WEIGHTS.items()))
  vgg=Path.home()/'.cache/torch/hub/checkpoints/vgg16-397923af.pth'
  if not vgg.exists():vgg=download('https://download.pytorch.org/models/vgg16-397923af.pth',ASSETS/'weights/vgg16-397923af.pth','397923af')
  if not sha(vgg).startswith('397923af'):raise RuntimeError('Existing VGG16 is not official IMAGENET1K_V1')
  cache=ASSETS/'dreamsim_cache';extract_zip(weights['dreamsim_ensemble_checkpoint.zip'],cache)
  entries={p.relative_to(cache).as_posix():sha(p) for p in sorted(cache.rglob('*')) if p.is_file() and p.suffix in ['.pth','.pt','.tar','.bin','.safetensors','.json'] and p.name!='_archive_identity.json'}
  if not entries:raise RuntimeError('DreamSim release has no registered model artifacts')
  manifest=dict(schema_version=1,clip=dict(implementation=sources['clip'],weights=weight_spec(weights['ViT-L-14.pt'])),
   dinov2_vitl14=dict(implementation=registered_dinov2_source(),weights=weight_spec(weights['dinov2_vitl14_pretrain.pth'])),
   dists=dict(implementation=sources['dists'],weights=weight_spec(ASSETS/'sources/dists/DISTS_pytorch/weights.pt'),vgg16_weights=weight_spec(vgg)),
   resnet50=dict(weights=weight_spec(weights['resnet50-11ad3fa6.pth'])),ms_ssim=dict(implementation=sources['ms_ssim']),
   dreamsim=dict(implementation=sources['dreamsim'],cache=dict(path=str(cache),files=entries),dino_implementation=sources['dino'],dino_weights=weight_spec(weights['dino_vitbase16_pretrain.pth'])))
  path=OUT/'modelmanifest.json'
  if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Metric asset manifest changed')
  write(path,manifest)
  env=OUT/'environment';python=env/'bin/python'
  if not python.exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
  subprocess.run([str(python),'-m','pip','install','peft==0.21.0'],check=True)
  freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
  (OUT/'environment_freeze.txt').write_text(freeze)
  write(OUT/'assets_complete.json',dict(status='ASSETS_READY',manifest_sha256=sha(path),environment=str(python),environment_freeze_sha256=sha(OUT/'environment_freeze.txt'),training_updates=0,time=time.time()))
  write(OUT/'assets_status.json',dict(status='READY',time=time.time()))
 except Exception as error:
  write(OUT/'assets_status.json',dict(status='FAILED',error=str(error),time=time.time()));raise

if __name__=='__main__':main()
