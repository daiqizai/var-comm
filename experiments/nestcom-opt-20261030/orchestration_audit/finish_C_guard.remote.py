import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from combined_statistics import main
os.sched_setaffinity(0,{24,25})
for mode in ['three','clip']:
 p=L/('fit_C_guard_'+mode+'_control')/'actual_wait.json'
 if not p.exists():print('C_GUARD_METRICS_PENDING');raise SystemExit(0)
 w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
frozen=read(D/'C_selected_before_guards.json');variants=['original',frozen['selected'],frozen['no_content']]
rp=D/'fit_C_complete_statistics_registration.json';save(rp,dict(dataset='fit100',stage='FIT',output='fit_C_complete_delivery',segments=[dict(role='fit_C',variants=variants),dict(role='fit_C_guard'),dict(role='fit_A',variants=['original'])],frozen_checkpoints=pin(D/'C_selected_before_guards.json')))
g=main(rp);passing=g[frozen['selected']]['FIT_ready_for_confirmation']and read(D/'fit_C_delivery/decision.json')['content_increment']
decision=dict(route='C',status='PREPARED'if passing else'NO_GO',selected=frozen['selected'],no_content=frozen['no_content'],guards_complete=True,gate=pin(D/'fit_C_complete_delivery/gate.json'),checkpoint_reselection=False,next_required='Matched second training seed and frozen CONFIRM100; content-specific paired-CI requirement remains'if passing else'Stop C expansion; preserve all primary results',first_round_science_complete=True)
save(D/'fit_C_complete_delivery/decision.json',decision);print(decision)
