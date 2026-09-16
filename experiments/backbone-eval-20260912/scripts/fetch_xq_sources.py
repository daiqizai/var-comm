#!/usr/bin/env python3
"""Read only the pinned official Git tree; verify each file with its Git blob hash."""
import base64,hashlib,json,pathlib,subprocess,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[1];rev='137869c6e60b1c48edc2a33f543a28b565a10632'
tree=json.loads((R/'docs/xq-source-tree.json').read_text())['tree']
prefixes=('models/','tokenizer/tokenizer_image/dino_enc/')
exact=['dist.py','datasets.py','tokenizer/vqgan/cliploss.py','tokenizer/tokenizer_image/quant.py','tokenizer/tokenizer_image/lookup_free_quantize.py','tokenizer/tokenizer_image/latent_perturbation.py','LICENSE','configs/MSVR10P2-4096.yaml','tokenizer/tokenizer_image/xqgan_model.py','inference.py','utils/arg_util.py','environment.yml']
files=[t for t in tree if t['type']=='blob' and (t['path'] in exact or (t['path'].startswith(prefixes) and t['path'].endswith('.py')))]
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def fetch(t):
 p=t['path'];f=R/'vendor/xqgan-source'/p
 if f.exists() and blob(f.read_bytes())==t['sha']:return p
 u='https://api.github.com/repos/lxa9867/ImageFolder/contents/'+p+'?ref='+rev
 for i in range(3):
  r=subprocess.run(['curl','-fsSL','--connect-timeout','6','--max-time','20',u],capture_output=True)
  if r.returncode:continue
  d=json.loads(r.stdout)
  if 'content' not in d:continue
  b=base64.b64decode(d['content']);assert blob(b)==t['sha'],p
  f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);print(p,len(b),flush=True);return p
 raise RuntimeError('cannot fetch pinned source: '+p)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(fetch,files))
(R/'docs/xq-source-lock.json').write_text(json.dumps({'commit':rev,'files':[{'path':t['path'],'git_blob_sha1':t['sha'],'sha256':hashlib.sha256((R/'vendor/xqgan-source'/t['path']).read_bytes()).hexdigest()} for t in files]},indent=2)+'\n')
