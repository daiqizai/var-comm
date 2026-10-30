"""New evaluation namespace; historical scientific files are immutable inputs."""
from pathlib import Path
import sys,os,json,time,hashlib,csv,ast,sqlite3
import numpy as np
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,sha,pin,checked,cr,asha,digest,save,ident,Ledger,BudgetStop
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT=R.parent.parent/'var_comm_runtime_20261010'
E=R/'experiments/nestcom-perceptual-followup-20261030'
D=R/'results/nestcom_perceptual_followup_20261030_v2'
L=B/'nestcom_perceptual_followup_20261030_v2'
OLD=R/'results/nestcom_opt_20261030_v1';OL=B/'nestcom_opt_20261030_v1'
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine','dists')
def rows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def csvout(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for v in data for k in v))
 with p.open('x',newline='',encoding='utf8')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def source_minimal(s):return {k:s[k]for k in ['source_id','source_index','archive','preprocessing_id','reference_sha256']}
def image_spec(v):return None if v in ['',None,'None','null','FIXED_GRAY']else ast.literal_eval(v)if isinstance(v,str)else v
def closed(role):
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
 c=read(L/role/'completion.json');assert c['status']=='COMPLETE';return c
