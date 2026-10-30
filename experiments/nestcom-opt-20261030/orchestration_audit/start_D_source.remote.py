import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
w=read(L/'fit_A_clip_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
p=read(D/'p0_registration.json');code=[v for v in p['code']if not v['path'].startswith(str(E))]+[pin(E/x)for x in ['opt_common.py','grid_codec.py','grid_source.py','control_once.py','launch_registered.py']]
r=dict(code=code,sources=pin(L/'inputs/fit100.json'),model_routes=p['model_routes'],cpu_affinity=[22,23],catalogues=[pin(D/'grid_catalogues'/('N%d.json'%n))for n in [512,1024]])
launch('fit_D_source',r,E/'grid_source.py',RT/'envs/frozen_visual_py310/bin/python',5400,'D')
print('SAMPLE',read(L/'fit_A_metric_plan/logical.json')[0])
for role in ['fit_A_three','fit_A_clip']:
 a=read(L/role/'scores.json');print(role,type(a).__name__,str(a)[:1600])
print('BOOTSTRAPS', [str(p)for p in L.rglob('*bank*')])
