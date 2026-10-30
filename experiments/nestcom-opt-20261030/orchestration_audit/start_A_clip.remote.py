import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from grid_catalogue import main
from launch_registered import launch
main()
w=read(L/'fit_A_three_control/actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None
oldpin=read(D/'protocol_resolved_p0.json')['clip'];old=cr(oldpin);plan=pin(L/'fit_A_metric_plan/plan.json');count=len(cr(plan)['pairs'])
code=[v for v in old['code']if Path(v['path']).exists()]+[pin(E/x)for x in ['opt_common.py','metrics.py','control_once.py','launch_registered.py']]
r=dict(code=code,plan=plan,metric_registration=oldpin,mode='clip',cpu_affinity=[22,23],caches=[],caps=dict(load_clip=1,embedding=count+100,pair=count))
launch('fit_A_clip',r,E/'metrics.py',Path('/usr/bin/python'),1800,'A')
