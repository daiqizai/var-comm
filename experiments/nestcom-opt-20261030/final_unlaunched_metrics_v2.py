"""First launch of missing metrics after an exactly diagnosed zero-Popen refusal."""
from opt_common import *
from launch_metric_stage_v2 import main as launch_metric
from audit_receiver_inputs_v2 import main as audit
from final_statistics import main as statistics
import traceback

def main(request,expected):
 assert sha(request)==expected;r=read(request);[checked(v)for v in r['code']]
 proof=cr(r['continuation_proof']);assert proof['old_completed']==['final_rx'] and proof['zero_metric_Popen']
 out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));start=time.monotonic();done=[];attempts=[]
 save(out/'launch.json',dict(process=ident(os.getpid()),registration=pin(request)))
 def boundary():
  if time.time()>=r['original_absolute_deadline']or(L/'STOP').exists():raise BudgetStop('Original final successor deadline, not reset')
 def launch_first(mode,caches):
  role='final_new_'+mode;beg=time.monotonic()
  while True:
   boundary();assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()
   try:launch_metric('final_new',mode,'final',caches);return
   except AssertionError as e:
    # Only an exact admission refusal before registration/Popen may wait.
    assert str(e)=='No eligible GPU at current resource admission; zero registration/launch'
    assert not(D/(role+'_registration.json')).exists()and not(L/role).exists()
    attempts.append(dict(role=role,unix=time.time(),reason=str(e),registration_and_Popen=0))
    if time.monotonic()-beg>=1200:raise BudgetStop('Bounded waiting for original metric device; no scientific retry')
    time.sleep(10)
 def wait(role):
  p=L/(role+'_control/actual_wait.json')
  while not p.exists():boundary();time.sleep(10)
  w=read(p);assert w['returncode']==0 and ident(w['child']['pid'])is None
  c=read(L/role/'completion.json');assert c['status']=='COMPLETE'and c['ledger']['unresolved']==0
  done.append(role);save(out/(role+'_closed.json'),dict(wait=pin(p),completion=pin(L/role/'completion.json')))
 try:
  assert not(D/'final_rx_posthoc_input_audit.json').exists()
  launch_first('three',('fit_A','confirm_D'))
  audit('final_rx','final_rx_plan',checked(r['sources']))
  wait('final_new_three')
  launch_first('clip',('fit_A','confirm_D','final_historical_missing'))
  wait('final_new_clip')
  rp=D/'final_statistics_registration.json';save(rp,dict(final_freeze=r['freeze'],historical_frames=pin(L/'historical_regression_inputs/frames.json'),bank_identity=pin(R/'results/nestcom_7db_common500_20261030_v1/protocol.json'),output='final_common500_delivery',new_metrics=[pin(L/('final_new_'+m)/'completion.json')for m in ['three','clip']],posthoc_audit=pin(D/'final_rx_posthoc_input_audit.json')))
  statistics(rp)
  save(out/'completion.json',dict(status='COMPLETE',completed=done,statistics=pin(D/'final_common500_delivery/statistics_identity.json'),seconds=time.monotonic()-start,pre_registration_admission_waits=attempts,scientific_calls_retried=0,original_deadline_preserved=True))
 except BaseException:
  save(out/'failure.json',dict(status='BLOCKED',traceback=traceback.format_exc(),completed=done,pre_registration_admission_waits=attempts,automatic_scientific_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])
