import sys,subprocess
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from evaluation_frame_plan_v2 import main as plan
from launch_registered import launch
decision=read(D/'fit_D_complete_delivery/decision.json');gate=read(D/'fit_D_complete_delivery/gate.json')['gates']['D_token']
assert decision['selected']=='D_token'and gate['FIT_ready_for_confirmation']
freeze=D/'D_confirm_frozen.json';save(freeze,dict(status='FROZEN_BEFORE_CONFIRM_RESULTS',variant='D_token',grid='token',receiver='original',guard_gate=pin(D/'fit_D_complete_delivery/gate.json'),FIT_selection=pin(D/'D_selected_before_guards.json'),sources=pin(L/'inputs/confirm100.json'),recipe=pin(D/'confirmation_channel_recipe.json'),no_parameter_reselection=True))
a=read(D/'protocol_resolved_p0.json');base=read(D/'fit_phy_1024_registration.json')
rp=D/'confirm_base_plan_registration.json';save(rp,dict(dataset='confirm100',sources=pin(L/'inputs/confirm100.json'),grid='quarter',methods=['NEST','WHOLE'],variant='original',catalogues={str(n):a['catalogue_N'+str(n)]for n in(512,1024)},output='confirm_base_frame_plan',channel_recipe=pin(D/'confirmation_channel_recipe.json')));plan(rp)
for N in [512,1024]:
 role=f'confirm_base_phy_{N}';rp=D/(role+'_registration.json');code=base['code']+[pin(E/'evaluation_frame_plan_v2.py')]
 r=dict(N=N,jobs=pin(L/f'confirm_base_frame_plan/physical_{N}.json'),catalogue=a['catalogue_N'+str(N)],catalogue7=base['catalogue7'],qualification=base['qualification'],code=code,output=str(L/role),cpu_affinity=[24,25]if N==512 else[26,27],seconds=3600,confirmation_freeze=pin(freeze))
 save(rp,r);cp=D/(role+'_control_registration.json');save(cp,dict(code=code,control=str(L/(role+'_control')),command=[str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'frame_worker.py'),str(rp),sha(rp)],environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='0'),hard_seconds=3630,gpu=False))
 with(L/(role+'_owner.log')).open('xb')as log:
  p=subprocess.Popen([str(RT/'envs/frozen_ldpc_py311/bin/python'),'-B',str(E/'control_once.py'),str(cp),sha(cp)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(L/(role+'_dispatch.json'),dict(owner=ident(p.pid),registration=pin(cp),unique_first_launch=True));print(role,ident(p.pid))
p0=read(D/'p0_registration.json');code=[v for v in p0['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','grid_codec.py','grid_source.py','control_once.py','launch_registered.py']]
r=dict(code=code,sources=pin(L/'inputs/confirm100.json'),model_routes=p0['model_routes'],cpu_affinity=[20,21],catalogues=[pin(D/'grid_catalogues'/('N%d.json'%n))for n in[512,1024]],confirmation_freeze=pin(freeze))
launch('confirm_D_source',r,E/'grid_source.py',RT/'envs/frozen_visual_py310/bin/python',1800,'combinations_and_confirm')
