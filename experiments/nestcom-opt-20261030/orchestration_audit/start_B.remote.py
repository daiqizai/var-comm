import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
for role in ['fit_A','fit_D_source']:
 w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
jobs=read(L/'fit_rx_plan/inputs.json');rows=read(L/'fit_A/outputs.json');reuse={x['key']:x for x in rows if x['variant']=='original'and x['image']is not None}
assert len(reuse)==927 and all(j['key']in reuse or j['projection']['status']!='SOURCE_INPUT'for j in jobs)
path=L/'inputs/fit_B_reuse.json';save(path,reuse)
p=read(D/'p0_registration.json');N=len(jobs)
code=[v for v in p['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','receiver_worker.py','control_once.py','launch_registered.py']]
r=dict(code=code,jobs=pin(L/'fit_rx_plan/inputs.json'),reuse=pin(path),model_routes=p['model_routes'],static=p['static'],cpu_affinity=[22,23],caps=dict(model_load=1,prior_scale=N*60,decoder_forward=N*6,source_decode=0,completion_forward=N*6,render=N*6),generation_seeds=[2026103001,2026103002,2026103003],seed_recipe='fixed per-image reset; no hidden source ID; full vocabulary unknown positions only',temperatures=[.3,.7])
launch('fit_B',r,E/'receiver_worker.py',RT/'envs/frozen_visual_py310/bin/python',17000,'B')
