"""Cold control-plane resume; frozen parent science/configuration is untouched."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import swin_early_delivery_common as c


def manual_bypass(original,parent_config,early_config):
    """Retire only the proven exited manual owner, then use the normal launcher."""
    def admitted(config):
        c.require(config==parent_config,'Cold resume was applied to another parent configuration')
        old=original(config);c.require(old is not None,'Expected immutable original manual launch')
        admission=c.snapshot_gate(early_config)
        c.require(any(owner['pid']==old['pid'] and str(owner['start_ticks'])==str(old['start_ticks'])
            for owner in admission['owners']) and not c.live(old),'Original manual owner is not safely retired')
        return None
    return admitted


def run(config_path,token):
    path=Path(config_path).resolve();config=c.validate(c.read(path));p=c.locations(config['root'])
    c.stage_proof(config,'publish');publication=c.read(c.receipt_path(config,'publish'))
    c.verify_publication_blobs(p['root'],publication)
    parent_config=c.original_pause_gate(config);common,parent=c.parent_modules(config)
    # Changing these Python function references affects process scheduling only.
    # New workers still execute the frozen parent file with its original config.
    original=parent.manual_reconstruction
    parent.manual_reconstruction=manual_bypass(original,parent_config,config)
    audit=dict(status='SWIN_PUSHED_ORIGINAL_MANUAL_OWNER_RETIRED',selected_step=80000,
        training_resumed=False,parent_config_sha256=config['parent_config_sha256'],
        early_publication_commit=publication['commit'],
        bindings={str(path):c.sha(path),str(c.receipt_path(config,'publish')):c.sha(c.receipt_path(config,'publish')),
            str(p['child']/'pause_verified.json'):c.sha(p['child']/'pause_verified.json'),
            str(p['child']/'hifi_resume.py'):c.sha(__file__),
            str(p['parent']/'reconstruct_launch.json'):c.sha(p['parent']/'reconstruct_launch.json')},
        only_override='fixed80k_delivery.manual_reconstruction; retired exact safe-paused manual owner returns None',
        model_or_packet_functions_overridden=False,parent_config_modified=False,parent_science_source_modified=False)
    c.seal(p['delivery']/'hifi_resume_admission.json',audit)
    for name in ('controller_launch.json','dispatch.json'):
        owner_path=p['parent_delivery']/name
        if owner_path.exists():c.require(not c.live(c.read(owner_path)),'A parent controller is already alive')
    identity=c.process(os.getpid());c.require(identity is not None,'Resume process identity missing')
    c.seal(p['delivery']/'hifi_resume_launches'/(token+'.json'),dict(identity,launch_id=token,
        parent_config_sha256=config['parent_config_sha256'],early_config_sha256=c.sha(path),
        source_bindings={str(c.HERE/name):c.sha(c.HERE/name) for name in c.OWN_FILES},time=time.time()))
    try:
        code=parent.Controller(config['parent_config']).run()
    finally:parent.manual_reconstruction=original
    if code:return code
    parent_done=p['parent_delivery']/'completion.json';done=c.read(parent_done)
    c.require(done.get('status')=='USER_FIXED80K_MATCHED_EVALUATION_REPORTED_AND_PUSHED'
        and done.get('config_sha256')==config['parent_config_sha256'] and done.get('selected_step')==80000
        and done.get('training_resumed') is False and done.get('full_plan_complete') is True,'Parent did not complete unchanged evaluation')
    parent_pub=p['parent_delivery']/'publication.json';published=c.read(parent_pub)
    c.require(done['publication_sha256']==c.sha(parent_pub) and done['commit']==published['commit'],'Parent completion/publication differs')
    c.verify_publication_blobs(p['root'],published)
    result=dict(status='UNCHANGED_FIXED80K_PARENT_REPORTED_AND_PUSHED',selected_step=80000,training_resumed=False,
        parent_config_sha256=config['parent_config_sha256'],commit=published['commit'],full_plan_complete=True,
        bindings={str(parent_done):c.sha(parent_done),str(parent_pub):c.sha(parent_pub),
            str(p['delivery']/'hifi_resume_admission.json'):c.sha(p['delivery']/'hifi_resume_admission.json')})
    c.seal(p['delivery']/'hifi_resume_completion.json',result);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--launch-id',required=True)
    args=parser.parse_args()
    try:raise SystemExit(run(args.config,args.launch_id))
    except SystemExit:raise
    except BaseException as error:
        path=args.config.parent/'hifi_resume_failure.json'
        if not path.exists():c.write(path,dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False))
        raise
