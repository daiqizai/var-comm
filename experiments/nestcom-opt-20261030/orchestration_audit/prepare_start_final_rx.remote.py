import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_receiver_plan_v2 import main
from launch_registered import launch
for role in ['final_phy_512','final_phy_1024']:
 p=L/(role+'_control')/'actual_wait.json'
 if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
rp=D/'final_rx_plan_registration.json';save(rp,dict(frame_roles=['final_phy_512','final_phy_1024'],logical=[pin(L/x/'logical.json')for x in ['final_D_frame_plan','final_WHOLE7_frame_plan']],current_original_roles=['fit_A','fit_D_rx','fit_D_guard_rx','confirm_base_rx','confirm_D_rx'],historical_n512=True,historical_seven=True,output='final_rx_plan',final_freeze=pin(D/'frozen_final.json')));main(rp)
p=L/'final_rx_plan';c=read(p/'completion.json');count=c['new_decode'];p0=read(D/'p0_registration.json');code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','receiver_variants.py','original_receiver.py','grid_codec.py','control_once.py','launch_registered.py']]
r=dict(code=code,jobs=pin(p/'inputs.json'),reuse=pin(p/'reuse.json'),model_routes=p0['model_routes'],static=p0['static'],cpu_affinity=[20,21],expanded_grid=True,final_freeze=pin(D/'frozen_final.json'),caps=dict(model_load=1,prior_scale=count*20,decoder_forward=count,source_decode=count,completion_forward=count,render=count))
launch('final_rx',r,E/'original_receiver.py',RT/'envs/frozen_visual_py310/bin/python',10800,'final')
