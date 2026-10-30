"""One finite successor owner for final RX -> metric gaps -> paired regression."""
from opt_common import *
from launch_metric_stage_v2 import main as launch_metric
from evaluation_metric_plan import main as prepare
from audit_receiver_inputs_v2 import main as audit
from final_statistics import main as statistics
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']];out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));beg=time.monotonic();save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)));done=[]
 def wait(role):
  control=L/(role+'_control')
  while not(control/'actual_wait.json').exists():
   if time.monotonic()-beg>=r['seconds']or(L/'STOP').exists():raise BudgetStop('Final successor bound')
   if(control/'child.json').exists():
    c=read(control/'child.json')['child'];v=ident(c['pid'])
    # A vanished child may be waiting for its owner to flush actual wait.
    if v is not None:assert v['start_ticks']==c['start_ticks']
   time.sleep(10)
  w=read(control/'actual_wait.json');assert w['returncode']==0 and ident(w['child']['pid'])is None,role
  c=read(L/role/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
  done.append(role);save(out/(role+'_closed.json'),dict(wait=pin(control/'actual_wait.json'),completion=pin(L/role/'completion.json')))
 try:
  wait('final_rx')
  rp=D/'final_new_metric_plan_registration.json';save(rp,dict(role='final_new',sources=r['sources'],segments=[dict(receiver_role='final_rx',logical=pin(L/'final_rx_plan/logical.json'))],frozen=r['freeze']));prepare(rp)
  launch_metric('final_new','three','final',('fit_A','confirm_D'))
  audit('final_rx','final_rx_plan',checked(r['sources']))
  wait('final_new_three')
  launch_metric('final_new','clip','final',('fit_A','confirm_D','final_historical_missing'))
  wait('final_new_clip')
  rp=D/'final_statistics_registration.json';save(rp,dict(final_freeze=r['freeze'],historical_frames=pin(L/'historical_regression_inputs/frames.json'),bank_identity=pin(R/'results/nestcom_7db_common500_20261030_v1/protocol.json'),output='final_common500_delivery',new_metrics=[pin(L/('final_new_'+m)/'completion.json')for m in ['three','clip']],posthoc_audit=pin(D/'final_rx_posthoc_input_audit.json')));statistics(rp)
  save(out/'completion.json',dict(status='COMPLETE',completed=done,statistics=pin(D/'final_common500_delivery/statistics_identity.json'),seconds=time.monotonic()-beg,new_quality_tasks_not_retried=True))
 except BaseException as e:save(out/'failure.json',dict(status='BUDGET_STOP'if isinstance(e,BudgetStop)else'BLOCKED',traceback=traceback.format_exc(),completed=done,automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])
