#!/usr/bin/env python3
"""One finite, no-training, sequential evaluation pipeline; gated by idle supervisor."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,required=True);p.add_argument('--task-name',required=True);a=p.parse_args()
 c=json.loads(a.config.read_text());out=pathlib.Path(c['output_root']);out.mkdir(parents=True,exist_ok=True)
 if c['task_name']!=a.task_name:raise ValueError('Named ownership mismatch')
 ready=pathlib.Path(c['assets_ready']);r=json.loads(ready.read_text())
 if r['status']!='ASSETS_VERIFIED_CPU_PREFLIGHT_PASS':raise ValueError('Assets are not ready')
 if r['evaluation_config_sha256']!=sha(a.config):raise ValueError('Config changed after asset verification')
 for f,h in r['runtime_source_sha256'].items():
  if sha(f)!=h:raise ValueError('Runtime source changed after preflight: '+f)
 started=datetime.datetime.now().astimezone().isoformat()
 def status(s,**kw):
  data={'status':s,'at':datetime.datetime.now().astimezone().isoformat(),'started_at':started,'task_name':a.task_name,'optimizer_updates':0,**kw}
  tmp=out/'evaluation-status.pending.json';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/'evaluation-status.json')
 root=pathlib.Path(__file__).resolve().parent
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
 # All inner subprocesses MUST inherit the supervisor-owned process group.
 try:
  for model in ('old-official','old-fidelity','xq','wetok'):
   for smoke in (True,False):
    stage='smoke' if smoke else 'full';status('RUNNING',model=model,stage=stage)
    cmd=[sys.executable,'-u','-B',str(root/'evaluate_backbones.py'),'--config',str(a.config),'--model',model]
    if smoke:cmd.append('--smoke')
    with (out/f'{model}-{stage}.log').open('x') as log:
     subprocess.run(cmd,env=env,cwd=root.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
  status('ANALYZING')
  with (out/'analysis.log').open('x') as log:
   subprocess.run([sys.executable,'-u','-B',str(root/'analyze_backbones.py'),'--config',str(a.config)],env=env,cwd=root.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
  report = out/'analysis/report.md'
  published = root.parents[2]/'reports/backbone_selection_result_2026-09-12.md'
  with published.open('x') as f:
   f.write('# 运行产物位置 / Artifact location\n\n`'+str(out/'analysis')+'`\n\n下列CSV与图片文件名均相对于上述分析目录。\n\n'+report.read_text())
  status('EVALUATION_COMPLETE_NO_TRAINING',decision='SEE_REPORT_NOT_AUTOMATIC_MIGRATION',report=str(published))
 except BaseException as e:
  status('STOPPED_NOT_COMPLETE',error=repr(e));raise
if __name__=='__main__':main()
