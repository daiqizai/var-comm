"""One RX-only GPU stage with original thermal guard; never resume stopped training."""
import run_preflight as env
import argparse,fcntl,time,os,traceback
from pathlib import Path
from token_efficiency import delivery_chain as dc
from latent_enhancement.runtime import write_json
OUT=env.OUT/'revision_v3'
STAGES={'A':('rx_v3_task_a','A_completion.json'),'B_calibration':('rx_v3_calibrate','B_calibration_completion.json'),
        'B_evaluation':('rx_v3_evaluate','B_completion.json')}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=list(STAGES),required=True);a=ap.parse_args()
    module,leaf=STAGES[a.stage]
    handles=[]
    for path in [OUT/'rx_stage.lock',env.ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/delivery_chain_v1/chain.lock',
                 env.ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/C_followups/followups.lock',
                 env.ROOT/'outputs/SHORT-PREFIX-20260923/controller.lock']:
        h=path.open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
    dc.CHAIN=OUT/(a.stage+'_supervisor');runner=dc.Runner()
    try:
        done=runner.run(a.stage,[module],OUT/leaf)
        write_json(OUT/(a.stage+'_supervisor_completion.json'),dict(status='RX_STAGE_COMPLETE_ORIGINAL_TRAINING_REMAINS_STOPPED',
                   stage=a.stage,time=time.time(),completion=done['status']))
    except Exception:
        write_json(OUT/(a.stage+'_supervisor_failure.json'),dict(time=time.time(),traceback=traceback.format_exc()));raise
