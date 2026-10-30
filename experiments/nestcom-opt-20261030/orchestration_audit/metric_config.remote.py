import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
for name in ['three_metrics_registration','clip_metrics_path_v2_registration']:
 v=read(R/'results/nestcom_evidence_20261029_v1'/(name+'.json'));print(name,'KEYS',list(v),'TORCH',v['torch'],'MODE',v['mode']);print('CFG_KEYS',list(v['config']),'ORIGINAL',v['original_four']);print('CONTROL_CONFIG',[(p.name,read(p).get('command'))for p in (R/'results/nestcom_evidence_20261029_v1').glob('*clip*control*')])
 print('CODE', [x['path']for x in v['code']])
print('ENV',[(p.name,str(p/'bin/python'))for p in (RT/'envs').iterdir()if(p/'bin/python').exists()])
for p in [L/'fit_A/failure.json',L/'fit_A/completion.json',L/'fit_A_control/actual_wait.json']:
 if p.exists():print(str(p),read(p))
print('A_TAIL',(L/'fit_A_control/worker.log').read_text()[-700:])
