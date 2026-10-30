import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
p=L/'confirm_D_rx_plan';c=read(p/'completion.json');count=c['new_decode'];p0=read(D/'p0_registration.json')
code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','original_receiver.py','grid_codec.py','control_once.py','launch_registered.py']]
r=dict(code=code,jobs=pin(p/'inputs.json'),reuse=pin(p/'reuse.json'),model_routes=p0['model_routes'],static=p0['static'],cpu_affinity=[20,21],expanded_grid=True,confirmation_freeze=pin(D/'D_confirm_frozen.json'),caps=dict(model_load=1,prior_scale=count*20,decoder_forward=count,source_decode=count,completion_forward=count,render=count))
launch('confirm_D_rx',r,E/'original_receiver.py',RT/'envs/frozen_visual_py310/bin/python',3600,'combinations_and_confirm')
