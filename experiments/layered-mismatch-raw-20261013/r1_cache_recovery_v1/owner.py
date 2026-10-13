"""One finite missing-stage owner; no entry for completed PHY or projection."""
from __future__ import annotations
import argparse, time
import common as c
import adapter

def run(registration):
    s,_,_,old_owner=adapter.install(registration,'owner')
    reg=s.validate_registration(registration)
    c.require(reg['runtime']['deadline_unix']==c.DEADLINE and time.time()<c.DEADLINE,'Original deadline exhausted')
    # Same real original owner control, locks, 6/2 settings and child.wait.
    # STAGES is restricted by the private adapter to visual/score/selection.
    result=old_owner.run(registration)
    if isinstance(result,dict) and result.get('status')=='WAIT':return result
    complete=c.read(c.OUT/'completion.json')
    c.require(complete['status']=='COMPLETE_R1_CHAIN' and complete['actual_exit_zero_all'] is True
              and [x['stage'] for x in complete['stages']]==list(c.STAGES),'Missing-stage chain did not actually close')
    # New receipt explicitly joins the old successful segment. The old owner
    # failure and failed model-load reservation remain immutable dependencies.
    return c.write(c.OUT/'recovery_completion.json',dict(status='COMPLETE_R1_CACHE_RECOVERY_CHAIN',registration=registration,
        original_registration=c.ORIGINAL,inherited_stages=reg['recovery']['original_closure']['stages'][:2],
        new_stages=complete['stages'],new_segment_completion=c.pin(c.OUT/'completion.json'),
        old_owner_exit=1,new_segment_stage_actual_exits_zero=True,new_owner_actual_wait_required=True,
        failed_initialization_attempts=1,additional_scientific_budget=0,
        R2='NOT_REGISTERED_NOT_RUN',automatic_R2=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--registration-sha256',required=True)
    a=p.parse_args();print(c.json.dumps(run(dict(path=a.registration,sha256=a.registration_sha256))),flush=True)
