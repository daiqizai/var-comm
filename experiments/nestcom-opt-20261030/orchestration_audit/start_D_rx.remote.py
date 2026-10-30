import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
p=L/'c_cache_control/actual_wait.json'
if not p.exists():print('WAIT_CACHE_ACTUAL_EXIT');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
print('CACHE_CLOSED',read(L/'c_cache/completion.json'))
plan='fit_D_rx_plan';p0=read(D/'p0_registration.json');jobs=read(L/plan/'inputs.json');c=read(L/plan/'completion.json');count=c['new_decode']
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','original_receiver.py','grid_codec.py','control_once.py','launch_registered.py']]
r=dict(code=code,jobs=pin(L/plan/'inputs.json'),reuse=pin(L/plan/'reuse.json'),model_routes=p0['model_routes'],static=p0['static'],cpu_affinity=[20,21],expanded_grid=True,caps=dict(model_load=1,prior_scale=count*20,decoder_forward=count,source_decode=count,completion_forward=count,render=count))
launch('fit_D_rx',r,E/'original_receiver.py',RT/'envs/frozen_visual_py310/bin/python',2500,'D')
