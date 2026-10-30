import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_registered import launch
for role in ['c_cache','fit_E_score']:
 p=L/(role+'_control')/'actual_wait.json'
 if not p.exists():print('DEPENDENCY_RUNNING',role);raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
print('E_SELECTION',read(L/'fit_E_score/completion.json'))
assert read(D/'c_train_preflight_v2.json')['status']=='PASS'
cache=read(L/'c_cache/completion.json');assert cache['count']==10000 and cache['source_scale_pairs']==20000 and cache['ledger']['unresolved']==0
code=[pin(E/x)for x in ['opt_common.py','c_model.py','c_train_pair_v2.py','control_once.py','launch_registered.py']]
r=dict(code=code,cache_completion=pin(L/'c_cache/completion.json'),sources=pin(L/'inputs/train10000.json'),seed=2026103001,cpu_affinity=[20,21],arms=['content','no_content'],steps_per_arm=3000,matched_schedule=True,preflight=pin(D/'c_train_preflight_v2.json'))
launch('c_train_primary',r,E/'c_train_pair_v2.py',RT/'envs/frozen_visual_py310/bin/python',60000,'C')
