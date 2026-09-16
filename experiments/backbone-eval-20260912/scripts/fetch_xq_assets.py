#!/usr/bin/env python3
"""Bounded, single-stream fetch of ONLY the predeclared evaluation assets."""
import hashlib, json, os, pathlib, subprocess, time, datetime
R=pathlib.Path(__file__).resolve().parents[1]
FILES=[('MSVR10P2-4096/best_ckpt.pt',3534394346,'fa349cec5f953e41a30d1877344fba7e485d5d059fb24b0de83d547aa0c70e53'),('VAR-d17-MSVR10P2-4096/ar-ckpt-last.pth',5537116322,'37db545025e1421d91f46e28733d37ef81b0178e2ce229ecbd7c3afcfe4c7c31')]
REV='67d51a379d592d228fe96ddd587aa84682ac53a0'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(d):
 d['at']=datetime.datetime.now().astimezone().isoformat();p=R/'docs/xq-download-status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
os.nice(10)
subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
for name,size,want in FILES:
 p=R/'checkpoints'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.stat().st_size!=size or sha(p)!=want:raise RuntimeError('existing checkpoint mismatch; NOT overwriting '+str(p))
  continue
 tmp=p.with_name(p.name+'.part');write({'status':'DOWNLOADING','asset':name,'total_bytes':size,'transport':'hf-mirror resolver redirects to official HF CAS','revision':REV})
 u='https://hf-mirror.com/qiuk6/XQ-GAN/resolve/'+REV+'/'+name+'?download=true'
 subprocess.run(['curl','-fL','--connect-timeout','10','--max-time','1800','--retry','2','--retry-delay','5','--retry-max-time','3700','--speed-time','90','--speed-limit','1024','--limit-rate','8M','--max-filesize',str(size),'-C','-','-o',str(tmp),u],check=True)
 if tmp.stat().st_size!=size or sha(tmp)!=want:raise RuntimeError('download size/SHA mismatch '+name)
 tmp.replace(p)
write({'status':'XQ_ASSETS_VERIFIED','revision':REV,'files':[{'path':name,'bytes':size,'sha256':s} for name,size,s in FILES]})
