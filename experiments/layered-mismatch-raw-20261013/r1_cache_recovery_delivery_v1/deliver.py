"""Read-only CPU delivery of the explicitly recovered R1 calibration chain.

Original PHY/projection successes and the diagnosed initialization failure keep
their original bytes. Three new successful stages supply quality. No scientific
model, channel, selection, bootstrap or R2 is executed by this entry.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.dont_write_bytecode = True
ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
TMP = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/layered_mismatch_raw_20261013')
MASTER_SHA = 'ad068190de6aa930a04c7b080aea55f1ad90235dfbddf65b544d30435b5c5c1c'
ORIGINAL_SHA = '70bb0747a749a3d63d10d5fdb8fac1af37a2e37e82310208d31c6823fde0ac47'
VISUAL_SHA = 'c1eeef69de031fd9e370690268c14d06c63810a94ff323b950db0f0ad14f6255'
DELIVERY_SHA = '94b6986e1db8a6082f0a02a6a91855fd79b926a390bb1551a1bc785d06bfd1b3'
FAILURE_SHA = 'daad4d721815607b9b9186ed33ded6b778c9a91b91d169afbfb5000f15add7d4'
AUXILIARY_FAILURE_SHA = '7094c96a7bf5e1e71d4e12cb9601b5075880d106c583c4401c08dacf866a13c0'
AUXILIARY_FAILURE_BYTES = 2466
NAME = 'r1_cache_recovery_delivery_v1'
NEW_STAGES = ('visual', 'score', 'selection')
CODE = {
    'adapter.py':'b8c9b66f455affbb9affa6e0f035db4c1a0b9a339a3b39e1f46cd3e737a4c86f',
    'common.py':'dca6c7fa8ac8c3b914d174b6c01fc08743e514161d9b9a4c99641b1342e2bd40',
    'dispatch_once.py':'7696679b363bfa8f2561b340bcdf445dda66bf470a35309d07a11a58a509fb80',
    'owner.py':'0e879570b2ba8c9da96fd97f6eaf0168320624cf352be90e40d1ca795e814b83',
    'registration.py':'53b517015950763e62806356e929dcdb4f21636314fbc3847dd2b1fc7d862d91',
    'runner.py':'5f823fa7b614148edd37da099ea0598d3491818f3d5bc7d1c91e9ba0040f020a',
    'score_provenance.py':'a47e2617524c1c3fc695881bca35cf3bddc0f4ab0dd7a06f6728f17ffc1d0435',
    'test_recovery.py':'500e05e0e2746e4b2c018f04928271ce5ce86fc9586c85fd1b97ee84ba9ea744',
    'test_score_provenance.py':'c1da8b97d36aebc2f321b444bf27744b0a13a695bb951b900f4941dabc7de34e',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()


def load_original(root):
    path = root/'experiments/layered-mismatch-raw-20261013/r1_delivery_v1/deliver.py'
    require(path.is_file() and not path.is_symlink() and sha(path)==DELIVERY_SHA, 'Frozen original CPU delivery changed')
    spec = importlib.util.spec_from_file_location('_original_r1_cpu_delivery', path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def paths(root, temporary):
    out=root/'results/layered_mismatch_raw_20261013';experiment=root/'experiments/layered-mismatch-raw-20261013'
    return dict(result=out/'r1_cache_recovery_v1', old_result=out/'r1_execution_v1',
        run=temporary/'r1_cache_recovery_v1', old_run=temporary/'r1_execution_v1',
        dispatch=temporary/'r1_cache_recovery_dispatch_v1', old_dispatch=temporary/'r1_resource_dispatch_v2',
        initial=temporary/'r1_cache_recovery_initial_control_v1', old_initial=temporary/'r1_initial_dispatch_control_v1',
        recovery=experiment/'r1_cache_recovery_v1', resource=experiment/'r1_execution_resource_v2',
        failure=out/'r1_cache_failure_audit_v1/failure_audit.json')


def actual_exit(d, waited, identity, expected):
    require(all(type(identity.get(k)) is int and identity[k]>0 for k in ('pid','start_ticks')), 'Positive process identity required')
    require(waited.get('actual_wait') is True and type(waited.get('returncode')) is int
            and waited['returncode']==expected and type(waited.get('pid')) is int
            and waited['pid']==identity['pid'], 'Wrong actual exit or process')
    require(not any(waited.get(k,False) for k in ('terminated','killed','timed_out','timeout','stopped')), 'Interrupted scientific segment')
    if 'start_ticks' in waited:
        require(type(waited['start_ticks']) is int and waited['start_ticks']==identity['start_ticks'], 'Wait process generation differs')


def no_failure_files(base, allowed=()):
    allowed={str(Path(p)) for p in allowed}
    for path in Path(base).rglob('*.json'):
        if 'failure' in path.name or path.name.startswith(('partial_','interrupted_')):
            require(str(path) in allowed, 'Unreviewed failure or partial artifact: '+str(path))


def stage_argv(p, reg, registration, stage, owner, inherited, d):
    runtime=reg['runtime'];cpu,visual=runtime['python_cpu'],runtime['python_visual']
    if stage=='phy':
        script=p['resource'].parent/'r1_phy.py';sub=reg['phy_registration'];py=cpu
        args=['run','--registration',sub['path'],'--registration-sha256',sub['sha256']]
    elif stage in ('projection','visual'):
        script=p['resource']/'receiver_visual.py';sub=reg['visual_registration'] if stage=='visual' else registration
        py=visual if stage=='visual' else cpu
        args=['visual' if stage=='visual' else 'project','--registration',sub['path'],'--expected-registration',sub['sha256'],
            '--parent-pid',str(owner['pid']),'--parent-ticks',str(owner['start_ticks'])]
    else:
        script=p['resource']/'score_selection.py';py=visual if stage=='score' else cpu
        args=['score' if stage=='score' else 'select','--registration',registration['path'],
            '--registration-sha256',registration['sha256'],'--parent',str(owner['pid'])]
    route=d.routed(reg,script,args) if inherited else [str(p['recovery']/'runner.py'),'--original',str(script),*args]
    return [py,'-B','-u',*route]


def verify_stage(d, p, reg, registration, stage, owner, inherited, code, probe):
    run=p['old_run'] if inherited else p['run'];directory=run/stage
    launch=d.read(run/(stage+'_launch.json'));wait=d.read(directory/'actual_wait.json')
    require(launch['stage']==stage and d.same_pin(launch['registration'],registration), 'Stage registration differs')
    require(launch['argv']==stage_argv(p,reg,registration,stage,owner,inherited,d), 'Stage argv differs: '+stage)
    require(launch['deadline_unix']==reg['runtime']['deadline_unix'], 'Stage deadline changed')
    require(launch['cpu_affinity']==reg['runtime']['cpu_affinity' if stage in ('visual','score') else 'phy_cpu_affinity'], 'Stage affinity differs')
    actual_exit(d,wait,launch['process'],code);d.require_gone(launch['process'],probe)
    require(wait==d.read(run/(stage+'_actual_wait.json')) and wait['stage']==stage, 'Stage and owner actual waits differ')
    result=dict(stage=stage,launch=d.pin(run/(stage+'_launch.json')),actual_wait=d.pin(directory/'actual_wait.json'),actual_exit=code)
    if code==0:
        require(not (directory/'failure.json').exists(), 'Completed stage also records failure')
        done=d.read(directory/'completion.json')
        expected=reg['phy_registration'] if stage=='phy' else reg['visual_registration'] if stage=='visual' else registration
        require(done['status']==d.STAGE_STATUS[stage] and d.same_pin(done['registration'],expected), 'Stage completion status or child registration differs')
        result['completion']=d.pin(directory/'completion.json')
    else:
        require(stage=='visual' and not (directory/'completion.json').exists(), 'Only diagnosed old visual failure may be inherited')
    return result


def failure_scope(d, p, original, probe):
    audit=d.readpin(dict(path=str(p['failure']),sha256=FAILURE_SHA))
    require(audit['schema']=='R1_CLOSED_VISUAL_CACHE_FAILURE_AUDIT_V1' and audit['status']=='DIAGNOSED_NO_AUTOMATIC_RETRY'
            and d.same_pin(audit['original_registration'],original) and audit['original_outer_returncode']==1
            and audit['completed_stages']==['phy','projection'] and audit['remaining_unlaunched_stages']==['score','selection'],
            'Different diagnosed failed segment')
    require(audit['failed_calls']==[dict(id=1,kind='model_load',metadata='{}',state='FAILED')]
            and type(audit['extra_failed_initialization_attempts']) is int and audit['extra_failed_initialization_attempts']==1,
            'Original failed initialization scope changed')
    for key in ('completed_model_loads','completed_VAR_renders','completed_prior_scales','completed_decoder_forwards',
                'output_images','partial_logical_frames','scientific_calls_in_this_audit'):
        require(type(audit[key]) is int and audit[key]==0, 'Original failure consumed unregistered forward work')
    evidence=audit['evidence'];require(evidence and all(key==value['path'] for key,value in evidence.items()), 'Malformed failure evidence inventory')
    for item in evidence.values():d.checked(item)
    allowed=(p['old_run']/'owner_failure.json',p['old_run']/'visual/failure.json',
        p['old_run']/'visual/partial_logical_frames.json')
    require(all(str(path) in evidence for path in allowed), 'Known historical failure not exactly bound')
    auxiliary=dict(path=str(p['old_run']/'visual/resource_history_at_failure.json'),
        sha256=AUXILIARY_FAILURE_SHA,bytes=AUXILIARY_FAILURE_BYTES)
    d.checked(auxiliary)  # Independently pinned closed failure attachment; original audit stays unchanged.
    allowed=(*allowed,Path(auxiliary['path']))
    no_failure_files(p['old_run'],allowed);no_failure_files(p['old_dispatch'])
    require(d.read(p['old_run']/'visual/partial_logical_frames.json')==[], 'Failed initialization contains partial frames')
    images=p['old_run']/'visual/images'
    require(images.is_dir() and not images.is_symlink() and not any(images.iterdir()), 'Old failed image directory is not empty')
    ledger=p['old_run']/'visual/calls.sqlite';require(str(ledger) in evidence, 'Failed call ledger unbound')
    db=sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        require(db.execute('PRAGMA quick_check').fetchone()==('ok',), 'Failed initialization ledger corrupt')
        calls=db.execute('SELECT id,kind,metadata,state FROM calls ORDER BY id').fetchall()
    finally:db.close()
    require(calls==[(1,'model_load','{}','FAILED')], 'Failed initialization ledger changed')
    for observed in audit['identities'].values():
        require(observed['original_identity_gone'] is True, 'Failure audit did not establish stopped processes')
        d.require_gone(observed['expected'],probe)
    require(not (p['old_run']/'score').exists() and not (p['old_run']/'selection').exists(), 'Original successors unexpectedly ran')
    return dict(audit=d.pin(p['failure']),all_original_evidence_files_sha_verified=len(evidence),
        auxiliary_failure_attachment=auxiliary,auxiliary_attachment_separate_from_original_audit=True,
        failed_model_load_reservations=1,completed_original_model_loads=0,completed_original_forward_calls=0,
        partial_images=0,original_failure_preserved=True)


def preflight(d, root, temporary, registration, probe=None):
    """Authenticate small closed-process metadata before any quality data read."""
    probe=d.process_identity if probe is None else probe;p=paths(root,temporary)
    require(registration==dict(path=str(p['result']/'registered/registration.json'),sha256=MASTER_SHA), 'Explicit fixed recovery master required')
    reg=d.readpin(registration);original=dict(path=str(p['old_result']/'registered/registration.json'),sha256=ORIGINAL_SHA)
    require(d.same_pin(reg['recovery']['original_registration'],original) and reg['recovery']['schema']=='R1_CACHE_RECOVERY_V1'
            and reg['expected_physical_frames']==32400 and reg['expected_logical_var_images']==43200,
            'Recovery descriptor or fixed grid changed')
    require(reg['runtime']['root']==str(p['run']) and reg['visual_registration']['sha256']==VISUAL_SHA,
            'Recovery runtime or visual child registration changed')
    for stage in ('phy','projection',*NEW_STAGES):
        require(reg['runtime'][stage+'_directory']==str((p['old_run'] if stage in ('phy','projection') else p['run'])/stage), 'Mixed stage path differs')
    for name,expected in CODE.items():
        path=p['recovery']/name
        require(reg['code_bindings'][str(path)]==expected and d.sha(path)==expected, 'Frozen recovery source changed: '+name)
    # The new chain must really close before inspecting old evidence or quality.
    no_failure_files(p['run']);no_failure_files(p['dispatch']);no_failure_files(p['initial'])
    final=d.read(p['result']/'completion.json');recovery=d.read(p['result']/'recovery_completion.json')
    require(final['status']=='COMPLETE_R1_CHAIN' and final['actual_exit_zero_all'] is True
            and final['automatic_R2'] is False and final['R2']=='NOT_REGISTERED_NOT_RUN'
            and d.same_pin(final['registration'],registration) and [s['stage'] for s in final['stages']]==list(NEW_STAGES), 'New segment not complete')
    owner=d.read(p['run']/'owner_launch.json');parent=d.read(p['dispatch']/'launch.json')
    identity=d.read(p['dispatch']/'owner_identity.json');initial=d.read(p['initial']/'dispatch_snapshot.json')
    require(d.same_process(identity['process'],owner['process']) and d.same_process(initial['process'],parent['process']), 'Recovery process chain differs')
    for receipt in (owner,parent,identity,initial):
        require(d.same_pin(receipt['registration'],registration), 'Recovery dispatch master differs');d.require_gone(receipt['process'],probe)
    owner_wait=d.read(p['dispatch']/'actual_wait.json');actual_exit(d,owner_wait,owner['process'],0)
    require(d.same_process(owner_wait['identity_at_launch'],owner['process']) and d.same_pin(owner_wait['registration'],registration)
            and owner_wait['scientific_owner_launch_exists'] is True, 'Recovery owner wait not authenticated')
    cpu=reg['runtime']['python_cpu']
    require(initial['argv']==[cpu,'-B','-u',str(p['recovery']/'dispatch_once.py'),'--expected-registration',registration['sha256']], 'Recovery outer dispatch argv differs')
    require(parent['argv']==[cpu,'-B','-u',str(p['recovery']/'owner.py'),'--registration',registration['path'],
            '--registration-sha256',registration['sha256']], 'Recovery owner argv differs')
    require(owner['deadline_unix']==parent['deadline_unix']==initial['original_deadline_unix']==reg['runtime']['deadline_unix'], 'Recovery extended the deadline')
    new=[]
    for stage,closure in zip(NEW_STAGES,final['stages']):
        item=verify_stage(d,p,reg,registration,stage,owner['process'],False,0,probe)
        require(d.same_pin(closure['actual_wait'],item['actual_wait']) and d.same_pin(closure['completion'],item['completion']), 'New chain closure pins differ')
        new.append(item)
    require(d.same_pin(final['frozen_policy_completion'],new[-1]['completion']), 'New final selection receipt changed')
    require(recovery['status']=='COMPLETE_R1_CACHE_RECOVERY_CHAIN' and d.same_pin(recovery['registration'],registration)
            and d.same_pin(recovery['original_registration'],original) and d.same_pin(recovery['new_segment_completion'],d.pin(p['result']/'completion.json'))
            and recovery['new_stages']==final['stages'] and recovery['old_owner_exit']==1
            and recovery['new_segment_stage_actual_exits_zero'] is True and recovery['new_owner_actual_wait_required'] is True
            and recovery['failed_initialization_attempts']==1 and recovery['additional_scientific_budget']==0
            and recovery['R2']=='NOT_REGISTERED_NOT_RUN' and recovery['automatic_R2'] is False, 'Recovery merge receipt differs')
    old=d.readpin(original);old_owner=d.read(p['old_run']/'owner_launch.json')
    old_parent=d.read(p['old_dispatch']/'launch.json');old_initial=d.read(p['old_initial']/'launch.json')
    old_identity=d.read(p['old_dispatch']/'owner_identity.json')
    require(d.same_process(old_parent['process'],old_initial['process']) and d.same_process(old_identity['process'],old_owner['process']), 'Original process chain differs')
    for receipt in (old_owner,old_parent,old_initial,old_identity):
        require(d.same_pin(receipt['registration'],original), 'Original master was relabelled');d.require_gone(receipt['process'],probe)
    old_wait=d.read(p['old_dispatch']/'actual_wait.json');actual_exit(d,old_wait,old_owner['process'],1)
    require(d.same_process(old_wait['identity_at_launch'],old_owner['process']) and d.same_pin(old_wait['registration'],original), 'Original owner failure wait differs')
    require(old_initial['argv']==[old['runtime']['python_cpu'],'-B','-u',*d.routed(old,p['resource']/'dispatch_once.py',['--expected-registration',original['sha256']])], 'Original dispatcher argv changed')
    require(old_parent['argv']==[old['runtime']['python_cpu'],'-B','-u',*d.routed(old,p['resource']/'owner.py',['--registration',original['path'],'--registration-sha256',original['sha256']])], 'Original owner argv changed')
    inherited=[verify_stage(d,p,old,original,stage,old_owner['process'],True,0,probe) for stage in ('phy','projection')]
    failed=verify_stage(d,p,old,original,'visual',old_owner['process'],True,1,probe)
    inherited_receipts=[dict(stage=s['stage'],launch=s['launch'],actual_wait=s['actual_wait']) for s in inherited]
    require(recovery['inherited_stages']==reg['recovery']['original_closure']['stages'][:2]==inherited_receipts, 'Inherited successful stage pins differ')
    failure=failure_scope(d,p,original,probe)
    d.unheld_locks((p['old_run']/'owner.lock',p['old_dispatch']/'dispatch.lock',p['run']/'owner.lock',p['dispatch']/'dispatch.lock'))
    return reg,dict(recovery_completion=d.pin(p['result']/'recovery_completion.json'),original_registration=original,
        recovery_registration=registration,inherited_successful_stages=inherited,original_failed_visual=failed,
        original_owner_actual_wait=d.pin(p['old_dispatch']/'actual_wait.json'),original_owner_actual_exit=1,
        new_successful_stages=new,new_owner_actual_wait=d.pin(p['dispatch']/'actual_wait.json'),new_owner_actual_exit=0,
        original_failure=failure,all_original_and_new_processes_gone=True,
        detached_initial_dispatchers='PID/start gone observed; no fabricated parent-wait receipt',
        scientific_calls=0,bootstrap_calls=0,reselection_calls=0)


def install_recovery(d,root,registration):
    reg=d.readpin(registration);directory=paths(root,TMP)['recovery']
    for name,expected in CODE.items():
        path=directory/name;require(reg['code_bindings'][str(path)]==expected and sha(path)==expected,'Changed frozen recovery code')
    sys.path.insert(0,str(directory))
    d.load_bound(directory/'common.py','common',reg)
    adapter=d.load_bound(directory/'adapter.py','_delivery_bound_recovery_adapter',reg)
    shared,receiver,score,owner=adapter.install(registration,'selection')
    require(shared.validate_registration(registration)==reg,'Recovery registration changed during CPU audit')
    return shared,receiver,score,owner


def adapt_original(d,root):
    """Keep original math/CSV/publication code; alter only audit routing and output identity."""
    source=Path(d.__file__).read_text(encoding='utf8');tree=ast.parse(source)
    nodes={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    deep=ast.get_source_segment(source,nodes['deep_audit'])
    old="""    for directory in (experiment,resource,experiment/'r1_source_identity_revision_v1'):
        sys.path.insert(0,str(directory))
    adapter=load_bound(experiment/'r1_source_identity_revision_v1/pin_adapter.py','pin_adapter',reg);adapter.install_phy()
    shared=load_bound(resource/'shared.py','shared',reg);require(shared.validate_registration(registration)==reg,'Master changed')
    receiver=load_bound(resource/'receiver_visual.py','receiver_visual',reg)
    score=load_bound(resource/'score_selection.py','score_selection',reg)"""
    replacements=[(old,"    shared,receiver,score,_=_install_recovery(registration)"),
        ("run=temporary/'r1_execution_v1';selection=read(run/'selection/completion.json')",
         "run=Path(reg['runtime']['root']);selection=read(run/'selection/completion.json')"),
        ("projection=read(run/'projection/completion.json');require(readpin(projection['audit'])==phy_audit,'Saved PHY audit differs')",
         "projection=read(Path(reg['runtime']['projection_directory'])/'completion.json');require(readpin(projection['audit'])==phy_audit,'Saved PHY audit differs')")]
    changed=deep
    for before,after in replacements:
        require(changed.count(before)==1,'Original CPU audit routing shape changed');changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(replacements):restored=restored.replace(after,before)
    require(ast.dump(ast.parse(restored))==ast.dump(ast.parse(deep)),'Deep audit inverse AST differs')
    pub=ast.get_source_segment(source,nodes['publish'])
    altered=pub.replace('r1_delivery_v1',NAME).replace("status='DELIVERED_R1_CALIBRATION_ONLY'","status='DELIVERED_R1_RECOVERED_CALIBRATION_ONLY'")
    require(ast.dump(ast.parse(altered.replace(NAME,'r1_delivery_v1').replace("status='DELIVERED_R1_RECOVERED_CALIBRATION_ONLY'","status='DELIVERED_R1_CALIBRATION_ONLY'")))==ast.dump(ast.parse(pub)), 'Publication inverse AST differs')
    d.__dict__['_install_recovery']=lambda registration:install_recovery(d,root,registration)
    exec(compile(changed+'\n\n'+altered,d.__file__,'exec'),d.__dict__)
    original_report=d.report_text
    def report(data):
        return original_report(data)+('\nRecovery provenance: the original master retained its complete PHY and RX projection. '
            'The first visual process and its owner exited with code 1 during a blocked temporary-cache initialization. '
            'The preserved ledger contains one failed model-load reservation, zero completed model loads, zero VAR/prior/decoder forward calls and no output image. '
            'An explicitly registered recovery completed visual reconstruction, scoring and calibration selection within the original deadline and scientific budget. '
            'The failed initialization is disclosed separately and is not an additional completed scientific frame; no original failure or result was overwritten.\n')
    d.report_text=report
    proof=dict(adapter_source=d.pin(Path(__file__)),original_delivery_sha256=DELIVERY_SHA,
        original_deep_audit_AST_sha256=hashlib.sha256(ast.dump(ast.parse(deep)).encode()).hexdigest(),
        restored_deep_audit_AST_sha256=hashlib.sha256(ast.dump(ast.parse(restored)).encode()).hexdigest(),
        deep_audit_metadata_substitutions=[dict(old=x,new=y) for x,y in replacements],
        deep_audit_inverse_AST=True,publication_only_output_identity_changes=True,all_metric_selection_CSV_functions_unchanged=True,
        no_bootstrap=True,no_reselection=True,scientific_calls=0)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-registration',required=True);args=parser.parse_args()
    require(args.expected_registration==MASTER_SHA and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Exact reviewed recovery registration and hidden CUDA required')
    require(sys.platform.startswith('linux'),'Actual CPU delivery requires the original Linux environment')
    d=load_original(ROOT);p=paths(ROOT,TMP);registration=dict(path=str(p['result']/'registered/registration.json'),sha256=args.expected_registration)
    d.require_absent(ROOT/'results/layered_mismatch_raw_20261013'/NAME);d.require_absent(TMP/NAME)
    guard=d.CPUReadOnly();require(not any(x in sys.modules for x in ('torch','tensorflow','sionna','lpips')),'Clean CPU interpreter required')
    sys.meta_path.insert(0,guard);sys.addaudithook(guard.audit)
    reg,closure=preflight(d,ROOT,TMP,registration)
    proof=adapt_original(d,ROOT);data=d.deep_audit(ROOT,TMP,registration,reg)
    data['audit']['recovery_provenance']=closure;data['audit']['CPU_delivery_adapter']=proof
    preflight(d,ROOT,TMP,registration)
    guard.active=False
    try:result=d.publish(ROOT,TMP,registration,data,closure)
    except BaseException as error:
        failure=TMP/(NAME+'_publication_failure.json')
        if not failure.exists():d.write_json(failure,dict(registration=registration,error=repr(error),scientific_closure_passed=True,
            automatic_retry=False,all_original_science_untouched=True))
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
