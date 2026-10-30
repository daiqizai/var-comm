import sys
sys.path.insert(0,'/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/nestcom-opt-20261030')
from opt_common import *
from launch_metric_stage import main
w=read(L/'C_guard_metric_queue_control/actual_wait.json');assert w['returncode']==1 and ident(w['child']['pid'])is None
f=read(L/'C_guard_metric_queue/failure.json');assert f['completed']==[]and'No eligible GPU at current resource admission; zero registration/launch'in f['traceback']
for role in ['fit_C_guard_three','fit_C_guard_clip']:assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()and not(L/(role+'_control')).exists()
save(D/'C_guard_unlaunched_metric_admission.json',dict(reason='Independent first launch of two never-registered metric stages after exact resource-admission race; original failed queue is not restarted.',original_failure=pin(L/'C_guard_metric_queue/failure.json'),actual_wait=pin(L/'C_guard_metric_queue_control/actual_wait.json'),unexecuted_roles=['fit_C_guard_three','fit_C_guard_clip'],model_calls_in_failed_queue=0,successful_science_repeated=False,budget_not_reset=True))
main('fit_C_guard','three','C',('fit_A','fit_C'))
