"""Independent common500 extension paths and strict, metadata-only primitives."""
from pathlib import Path
import csv, hashlib, json, os, sys
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
E=R/'experiments/arpc-common500-extension-20261019'
O=R/'results/arpc_common500_extension_20261019_v1'
L=BASE/'arpc_common500_extension_20261019_v1'
PE=R/'experiments/arpc-external-baseline-20261019'
PO=R/'results/arpc_external_baseline_20261019_v1'
PL=BASE/'arpc_external_20261019_v1'
sys.path.insert(0,str(PE))
from a1_owner import desc,verify,put,identity,gone,gpus
CPY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python'
PY=R.parent.parent/'var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python'
POLICIES={'4':'e7_q2_r2-3','10':'e7_q4_r2-3','19':'e7_q6_r5-6'}
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction','dists','dreamsim','ms_ssim')
BASELINES=('EC_VAR_PARTIAL','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024')
def read(p):return json.loads(Path(p).read_bytes())
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    assert rows
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
def check_wait(p,live=True):
    x=read(p)
    assert type(x['returncode']) is int and x['returncode']==0,p
    assert not any(x.get(k) for k in ('timed_out','terminated','killed','metadata_error')),p
    if live:assert gone(x['process']),p
    return x
def partition(full,old):
    assert len(full)==500 and len(old)==100
    assert {s['source_index'] for s in full}==set(range(500))
    byid={s['source_id']:s for s in full};assert len(byid)==500
    oldids={s['source_id'] for s in old};assert len(oldids)==100 and oldids<=byid.keys()
    for s in old:
        f=byid[s['source_id']]
        assert all(f[k]==s[k] for k in ('source_index','archive','preprocessing_id'))
    rest=sorted((s for s in full if s['source_id'] not in oldids),key=lambda s:s['source_index'])
    assert len(rest)==400
    return sorted(full,key=lambda s:s['source_index']),rest
