"""Wait for the already-running missing source pass, then launch pilot once."""
from study_common import *
from low_finish import wait
import calibration_stages_v2,traceback
out=L/'pilot_start';out.mkdir(exist_ok=False);save(out/'owner.json',dict(owner=ident(os.getpid()),code=[pin(E/x)for x in ['start_pilot_once.py','calibration_plan_v3.py','calibration_stages_v2.py']]))
try:
 wait('cal_token_source800');wait('directory_qualification');calibration_stages_v2.start_phy('pilot')
 save(out/'completion.json',dict(status='COMPLETE_DISPATCH',dispatch=pin(L/'cal_pilot_phy_dispatch.json')))
except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
