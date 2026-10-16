"""T1-C test-only immutable metadata, separate from all calibration state."""
import sys
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/t1b-m10-calibration-20261016'))
from tb_common import sha,pin,read,checked,checkread,save,csvsave,asha,pid_identity,gone,Ledger,MODELS,ENDS,OFFSETS,SIZES,canonical,digest,runtime_cache,source_key
import os,time,json,hashlib
RT=R.parent.parent/'var_comm_runtime_20261010'
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')
E=R/'experiments/t1c-m10-test-20261016';O=R/'results/t1c_m10_test_20261016_v1';T=B/'t1c_m10_test_20261016_v1';Q=RT/'qualification'
HEAD='13f0d4d0ffeea36bb88262414fecfd6324cb07d8'
FROZEN=R/'results/t1b_m10_calibration_20261016_v1/delivery/frozen_policies.json'
def registration(name,expected):
 p=O/(name+'_registration.json');assert sha(p)==expected
 r=read(p);assert r['scope']=='T1C_19dB_ONLY_FROZEN_TWO_ENTROPY_FAMILIES'
 for d in r['code']:checked(d)
 checked(r['frozen_policies']);return r
def counter(pop,i,seed):
 seeds=(6201,6202,6203) if pop=='common500' else (9301,9302,9303)
 n=500 if pop=='common500' else 100
 assert pop in ('common500','new100') and 0<=i<n and seed in seeds
 return ((5 if pop=='common500' else 2)*n+i)*3+seeds.index(seed)
def progress(out,v):
 with (out/'progress.jsonl').open('a',encoding='utf8') as f:f.write(canonical(dict(unix=time.time(),**v))+'\n');f.flush()
