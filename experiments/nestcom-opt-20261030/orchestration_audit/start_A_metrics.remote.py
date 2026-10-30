import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from metric_plan import main
from launch_registered import launch
main('fit_A')
mode='three';oldpin=read(D/'protocol_resolved_p0.json')['metrics'];old=cr(oldpin);plan=pin(L/'fit_A_metric_plan/plan.json');p=cr(plan);count=len(p['pairs'])
code=[v for v in old['code']if Path(v['path']).exists()]+[pin(E/x)for x in ['opt_common.py','metrics.py','control_once.py','launch_registered.py']]
r=dict(code=code,plan=plan,metric_registration=oldpin,mode=mode,cpu_affinity=[22,23],caches=[],caps=dict(load_lpips=1,load_dreamsim=1,embedding=count+100,pair=count))
launch('fit_A_three',r,E/'metrics.py',RT/'envs/frozen_visual_py310/bin/python',2400,'A')
