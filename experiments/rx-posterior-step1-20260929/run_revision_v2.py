"""Run revised calibration under the unchanged serial thermal/resource guard."""
import run_preflight as env
import os,time,traceback,fcntl
from pathlib import Path
from token_efficiency import delivery_chain as dc
from latent_enhancement.runtime import write_json,digest
OUT=env.OUT/'revision_v2'
dc.CHAIN=OUT/'supervisor'
if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'probe.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=dc.Runner()
    try:
        result=runner.run('proper_score_calibration_and_hard_gate',['probe_v2'],OUT/'calibration_completion.json')
        write_json(OUT/'supervisor_completion.json',dict(status=result['status'],passed=result['passed'],
                   completion_sha256=digest(OUT/'calibration_completion.json'),development_accessed=False,time=time.time()))
        runner.status('REVISED_CALIBRATION_DONE' if result['passed'] else 'REVISED_HARD_GATE_STOPPED')
    except Exception:
        write_json(OUT/'supervisor_failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
