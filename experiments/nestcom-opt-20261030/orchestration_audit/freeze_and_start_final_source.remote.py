import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
assert read(D/'confirm_D_delivery/decision.json')['confirm_pass']
for name in ['fit_A_delivery','fit_B_delivery','fit_C_complete_delivery','fit_E_delivery']:assert read(D/name/'decision.json')['status']=='NO_GO'
freeze=D/'frozen_final.json'
save(freeze,dict(status='FROZEN_BEFORE_FINAL_REGRESSION',configuration='original_receiver_plus_D_token',retained_components=['D_token'],receiver='original_Dc_argmax',grid='token',catalogues=[pin(D/'grid_catalogues'/('N%d.json'%n))for n in [512,1024]],FIT=pin(D/'fit_D_complete_delivery/gate.json'),CONFIRM=pin(D/'confirm_D_delivery/gate.json'),excluded_routes={name:pin(D/path/'decision.json')for name,path in [('A','fit_A_delivery'),('B','fit_B_delivery'),('C','fit_C_complete_delivery'),('E','fit_E_delivery')]},combinations=dict(candidate_count=2,candidates=['original','original+D_token'],exact_existing_independent_results_reused=True,new_combination_scientific_calls=0),leave_one_out=dict(remove_D='original',existing_confirm_results_reused=True,new_scientific_calls=0),common500_sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),channel_recipe=pin(D/'confirmation_channel_recipe.json'),NEW200='BLOCKED: auditable unused raw validation pool inaccessible on both registered hosts; do not relabel existing sources',no_reselection=True))
p0=read(D/'p0_registration.json');code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','grid_codec.py','grid_source.py','control_once.py','launch_registered.py']]
r=dict(code=code,sources=pin(B/'nestcom_n512_20261028_v1/qualified_sources/common500.json'),model_routes=p0['model_routes'],cpu_affinity=[20,21],catalogues=[pin(D/'grid_catalogues'/('N%d.json'%n))for n in [512,1024]],final_freeze=pin(freeze))
launch('final_D_source',r,E/'grid_source.py',RT/'envs/frozen_visual_py310/bin/python',3600,'final')
