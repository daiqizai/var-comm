"""Explicit metadata-only recovery registration; never rebuild model bundles."""
from __future__ import annotations
import argparse, copy, time
from pathlib import Path
import common as c
import adapter

FILES=('common.py','adapter.py','registration.py','runner.py','owner.py','dispatch_once.py','score_provenance.py',
       'test_recovery.py','test_score_provenance.py')

def visual_projection(master,old_visual,projection_control):
    visual=copy.deepcopy(old_visual)
    visual['runtime']=copy.deepcopy(master['runtime']);visual['code_bindings']=copy.deepcopy(master['code_bindings'])
    visual['cache_revision']=dict(schema=c.RECOVERY_SCHEMA,private_cache=str(c.RUN/'visual/runtime_cache'),
                                  source_truth_access_added=False,mathematical_changes=0)
    visual['inherited_projection']=projection_control
    c.require('recovery' not in visual and visual['model_bundle']==old_visual['model_bundle'],'Do not expose CPU recovery descriptor or rebuild bundle')
    return visual

def build(authorization):
    old=c.readpin(c.ORIGINAL);auth=c.validate_authorization(c.readpin(authorization))
    c.require(time.time()<c.DEADLINE,'Original total deadline exhausted; no reset allowed')
    directory=c.OUT/'registered';c.require(not directory.exists() and not c.RUN.exists() and not c.DISPATCH.exists(),'Existing attempt cannot be registered again')
    bindings=dict(old['code_bindings'])
    for name in FILES:
        path=c.HERE/name;c.require(auth['code_bindings'][str(path)]==c.sha(path),'Reviewed recovery file missing: '+name)
        bindings[str(path)]=c.sha(path)
    original_visual=c.readpin(old['visual_registration']);c.checked(original_visual['model_bundle'])
    c.require(old['expected_physical_frames']==32400 and old['expected_logical_var_images']==43200,'Original fixed grid changed')
    master=copy.deepcopy(old);master['runtime']=c.frozen_runtime(old['runtime']);master['code_bindings']=bindings
    master['recovery']=dict(schema=c.RECOVERY_SCHEMA,original_registration=c.ORIGINAL,authorization=authorization,
        failure_audit=c.FAILURE_AUDIT,original_closure=auth['original_closure'],allowed_stages=list(c.STAGES),
        failed_initialization_attempts=1,completed_old_forward_calls=0,additional_scientific_budget=0,
        old_results_unchanged=True,model_bundle_rebuilt=False)
    launch=c.read(c.ANCESTOR/'projection_launch.json');owner=c.read(c.ANCESTOR/'owner_launch.json')
    pure_control=dict(schema='R1_RX_ONLY_INHERITED_PROJECTION_V1',original_master_sha256=c.ORIGINAL['sha256'],
        actual_wait=c.pin(c.ANCESTOR/'projection/actual_wait.json'),projection_process=launch['process'],
        owner_process=owner['process'],source_truth_access_added=False)
    visual=visual_projection(master,original_visual,pure_control)
    adapter.validate_visual_runtime(visual['runtime'])
    # No imports, model tensors, source regeneration or packet calls.
    directory.mkdir(parents=True)
    child=c.write(directory/'visual_registration.json',visual);master['visual_registration']=child
    registered=c.write(directory/'registration.json',master)
    c.write(directory/'registration_receipt.json',dict(status='REGISTERED_RECOVERY_NOT_LAUNCHED',registration=registered,
        visual_registration=child,original_registration=c.ORIGINAL,deadline_unix=c.DEADLINE,scientific_calls=0,
        inherited_phy_and_projection=True,model_bundle_rebuilt=False,automatic_retry=False))
    return registered

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True);p.add_argument('--authorization-sha256',required=True)
    a=p.parse_args();print(c.json.dumps(build(dict(path=a.authorization,sha256=a.authorization_sha256))),flush=True)
