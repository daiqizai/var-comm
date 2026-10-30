import sys,os
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from combined_statistics import main as combined
from route_statistics_v2 import main as single
from screen_decision import main as decide
os.sched_setaffinity(0,{24,25})
def ready(role):
 for mode in ('three','clip'):
  p=L/(role+'_'+mode+'_control')/'actual_wait.json'
  if not p.exists():return False
  w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
 return True
if ready('fit_B'):
 if not(D/'fit_B_delivery').exists():single('fit_B');decide('fit_B','B')
 else:print('B_DECISION',read(D/'fit_B_delivery/decision.json'))
else:print('B_METRICS_PENDING')
if ready('fit_C'):
 if not(D/'fit_C_delivery').exists():
  p=D/'fit_C_statistics_registration.json';save(p,dict(dataset='fit100',stage='FIT',output='fit_C_delivery',segments=[dict(role='fit_C'),dict(role='fit_A',variants=['original'])]))
  combined(p)
  from c_screen_decision import main as cdecide
  cdecide()
 else:print('C_DECISION',read(D/'fit_C_delivery/decision.json'))
else:print('C_METRICS_PENDING')
if ready('fit_D_guard'):
 if not(D/'fit_D_complete_delivery').exists():
  p=D/'fit_D_complete_statistics_registration.json';save(p,dict(dataset='fit100',stage='FIT',output='fit_D_complete_delivery',segments=[dict(role='fit_D',variants=['original','D_token']),dict(role='fit_D_guard'),dict(role='fit_A',variants=['original'])],frozen_grid=pin(D/'D_selected_before_guards.json')))
  combined(p);decide('fit_D_complete','D')
 else:print('D_DECISION',read(D/'fit_D_complete_delivery/decision.json'))
else:print('D_GUARD_METRICS_PENDING')
if (L/'fit_D_guard_rx_control/actual_wait.json').exists()and not(D/'fit_D_guard_rx_posthoc_input_audit.json').exists():
 from audit_receiver_inputs import main as audit
 audit('fit_D_guard_rx','fit_D_guard_rx_plan')
