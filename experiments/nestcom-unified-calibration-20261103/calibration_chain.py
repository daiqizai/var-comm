"""Single bounded continuation of the already running pilot; never relaunch."""
from study_common import *
from low_finish import wait
import calibration_stages_v2 as stages,calibration_receivers as receivers,calibration_scoring as scoring,calibration_finish as final,five_metrics,traceback

def event(out,name):save(out/(str(time.time_ns())+'_'+name+'.json'),dict(event=name,time=time.time(),process=ident(os.getpid())))

def main():
 out=L/'calibration_chain';out.mkdir(exist_ok=False);save(out/'owner.json',dict(owner=ident(os.getpid()),created=time.time()))
 try:
  for stage in ['pilot','full','formal']:
   assert not(L/'STOP').exists()and time.time()<read(D/'protocol.json')['deadline_unix']
   if stage!='pilot':event(out,stage+'_PHY_start');stages.start_phy(stage)
   for role in read(L/('cal_'+stage+'_phy_dispatch.json'))['roles']:wait(role)
   event(out,stage+'_RX_start');receivers.start(stage)
   for role in read(L/('cal_'+stage+'_rx_plan/completion.json'))['roles']:wait(role)
   if stage!='formal':
    event(out,stage+'_DINO_start');scoring.start_dino(stage);wait('cal_'+stage+'_dino');scoring.select(stage);event(out,stage+'_selected')
   else:
    root=final.prepare()
    for mode in ['three','clip','dists']:event(out,'formal_'+mode+'_start');five_metrics.start('cal_formal',mode,root);wait('cal_formal_'+mode)
    final.finish()
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,other_independent_tasks_not_cancelled=True));raise
if __name__=='__main__':main()
