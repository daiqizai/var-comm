import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
p=L/'fit_E_simulation_control/actual_wait.json'
if not p.exists():print('E_SIMULATION_RUNNING');raise SystemExit(0)
w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
print('E_SIM_CLOSED',read(L/'fit_E_simulation/completion.json'))
oldpin=read(D/'protocol_resolved_p0.json')['metrics'];old=cr(oldpin)
code=[v for v in old['code']if Path(v['path']).exists()]+[pin(E/x)for x in ['opt_common.py','metrics.py','tx_score.py','control_once.py','launch_registered.py']]
r=dict(code=code,choices=pin(L/'fit_E_tx_plan/choices.json'),receiver_outputs=pin(L/'fit_E_simulation/outputs.json'),metric_registration=oldpin,cpu_affinity=[20,21])
launch('fit_E_score',r,E/'tx_score.py',RT/'envs/frozen_visual_py310/bin/python',1200,'E')
