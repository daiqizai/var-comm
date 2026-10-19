"""Independent N1024 repair metadata; originals and all closed calls are immutable."""
from pathlib import Path
import sys,os,time,json,hashlib
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
from tb_common import sha,pin,read,checked,checkread,save,csvsave,asha,pid_identity,gone,Ledger,MODELS,ENDS,OFFSETS,SIZES,canonical,digest,runtime_cache,source_key
RT=R.parent.parent/'var_comm_runtime_20261010'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
BASE=B.parent/'n1024_consistency_20261019_v1'
E=R/'experiments/n1024-consistency-20261019/repair_v1'
O=R/'results/n1024_consistency_20261019_v1/repair_v1'
T=BASE/'repair_v1'
SCOPE='N1024_COMMON500_WHOLE_4_10_SOURCE_VERSION_REPAIR_2421_ONLY'
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction','dists','dreamsim','ms_ssim')
def registration(name,expected):
 assert name in ('link','receiver') and len(expected)==64
 p=O/'registration.json';assert sha(p)==expected
 r=read(p);assert r['scope']==SCOPE and r['caps']['physical']==2421 and r['caps']['packets']==4842
 for d in r['code']:checked(d)
 checked(r['preparation']);checked(r['frozen_policies'])
 # Only this limited context proceeds into the isolated RX implementation.
 if name=='receiver':
  v=checkread(r['receiver_registration']);assert v['scope']==SCOPE
  assert not(set(v)&{'sources','frames','reused_frames','receiver_cache','legacy','source_truth','tokens'})
  return v
 return r
def progress(out,v):
 with (out/'progress.jsonl').open('a',encoding='utf8') as f:f.write(canonical(dict(unix=time.time(),**v))+'\n');f.flush()
