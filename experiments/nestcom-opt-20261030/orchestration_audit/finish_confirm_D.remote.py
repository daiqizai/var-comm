import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from combined_statistics import main
os.sched_setaffinity(0,{24,25})
for mode in ['three','clip']:
 p=L/('confirm_D_'+mode+'_control')/'actual_wait.json'
 if not p.exists():print('CONFIRM_METRICS_PENDING');raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
rp=D/'confirm_D_statistics_registration.json';save(rp,dict(dataset='confirm100',stage='CONFIRM',output='confirm_D_delivery',segments=[dict(role='confirm_D')],frozen_candidate=pin(D/'D_confirm_frozen.json')))
g=main(rp);d=dict(route='D',status='COMPLETE'if g['D_token']['confirm_pass']else'NO_GO',selected='D_token',confirm_pass=g['D_token']['confirm_pass'],gate=pin(D/'confirm_D_delivery/gate.json'),parameter_reselection=False)
save(D/'confirm_D_delivery/decision.json',d);print(d)
