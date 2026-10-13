"""One reviewed partial recovery; registration/launch require separate release.

Old attempts are immutable. No scheduler and no automatic retry. This source
may be prepared/tested without creating a scientific registration or a model.
"""
import argparse
import copy
import fcntl
import math
from pathlib import Path
import sys
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'timing_revision_v5'))
import timing_owner_v5 as v5
from recovery_adapter import adapt_build,adapt_worker,resumed_reservations,seed_completed,validate_prior,same_numerical_runtime
ROOT,TMP=v5.ROOT,v5.TMP
V5_OUT,V5_RUN=v5.OUT,v5.RUN
OUT=ROOT/'results/baseline_strength_20261011/timing/six_method_v6'
RUN=TMP/'six_method_timing_v6'
AUTH=OUT/'recovery_authorization.json'
read,write,sha,pin=v5.read,v5.write,v5.sha,v5.pin
inherited_verify=v5.verify
v4,v3,v2,old=v5.v4,v5.v4.v3,v5.v4.v3.v2,v5.v4.v3.old
for module in (v5,v4,v3,v2,old):module.OUT,module.RUN=OUT,RUN
old.__file__=__file__


def prior_evidence():
    wait=read(V5_OUT/'actual_wait.json')
    assert wait==dict(actual_child_waited=True,exit_code=1,timed_out=False)
    owner=read(V5_OUT/'launch.json')['owner'];worker=read(V5_OUT/'worker_launch.json')
    assert v3.dead_identity(owner['pid'],owner['start_ticks']) and v3.dead_identity(worker['pid'])
    assert not (V5_OUT/'completion.json').exists()
    trace=read(V5_RUN/'worker_failure.json')['traceback']
    assert trace.endswith("KeyError: 'IndependentProvider'\n") and 'fresh_entropy_choice' in trace
    prior=read(V5_RUN/'case_reservations.json');reg=read(V5_OUT/'registration.json')
    records={r['source_index']:r for r in reg['records']}
    validate_prior(prior,old.fixed_cases(),records)
    files=[V5_OUT/n for n in ('registration.json','launch.json','worker_launch.json','actual_wait.json','running_status.json','actual_constructor_compatibility.json','actual_module_aliases.json','actual_audit_adapter.json')]
    files += [V5_RUN/n for n in ('case_reservations.json','worker_failure.json','failure.json','worker.log','numerical_runtime.json','resource_snapshots.jsonl')]
    return {str(p):sha(p) for p in files}


def authorization():
    value=read(AUTH)
    assert value['status']=='REVIEWED_V6_RECOVERY_AUTHORIZED'
    assert value['inherited_complete_frames']==576 and value['remaining_complete_frames']==1152
    assert value['failed_slot_repeat_authorized'] is True
    assert value['prior_failed_visual_encoder_attempts']==1
    assert value['complete_frame_cap']==1728 and value['measured_frame_cap']==864 and value['packet_cap']==2304
    assert value['known_fresh_visual_attempts_at_success']==1729
    assert value['automatic_retries'] is False
    for key in ('prior_registration','prior_journal'):
        p=value[key];assert sha(p['path'])==p['sha256']
    assert Path(value['prior_registration']['path'])==V5_OUT/'registration.json'
    assert Path(value['prior_journal']['path'])==V5_RUN/'case_reservations.json'
    assert value['authority_evidence'] and value['review_evidence']
    return value


def elapsed_budget():
    # Use enclosing launch-to-actual_wait filesystem intervals, rounded up
    # with5s conservatism, rather than claiming exact GPU/child active time.
    entries=[]
    for version in (2,3,5):
        base=ROOT/f'results/baseline_strength_20261011/timing/six_method_v{version}'
        launch=base/'launch.json';wait=base/'actual_wait.json'
        start=read(launch)['started_unix'];stop=wait.stat().st_mtime
        assert read(wait)['actual_child_waited'] and 0<stop-start<14400
        charged=math.ceil(stop-start)+5
        entries.append(dict(version=version,launch=pin(launch),actual_wait=pin(wait),
            launch_unix=start,actual_wait_mtime_unix=stop,enclosing_seconds=stop-start,
            conservative_charged_seconds=charged))
    spent=sum(x['conservative_charged_seconds'] for x in entries)
    assert 0<spent<14400
    return dict(scope='Conservative cumulative four-hour timing execution envelope across actual attempts',
        original_seconds=14400,attempts=entries,prior_charged_seconds=spent,
        remaining_seconds=14400-spent,v4_actual_execution_seconds=0,
        child_active_time_measured=False,owner_watchdog_grace_seconds=1800,
        watchdog_grace_is_not_additional_scientific_budget=True)


def register_recovery():
    # Root issues authorization only after the no-model CPU tests and review.
    assert not (OUT/'registration.json').exists() and not (OUT/'launch.json').exists() and not RUN.exists()
    authorization();evidence=prior_evidence();budget=elapsed_budget()
    reg=copy.deepcopy(read(V5_OUT/'registration.json'))
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items():assert sha(p)==h,p
    reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY',
        recovery_of_v5=pin(V5_OUT/'registration.json'),v5_failure_evidence_sha256=evidence,
        recovery_authorization=pin(AUTH),prior_journal=pin(V5_RUN/'case_reservations.json'),
        inherited_complete_frames=576,inherited_measured_frames=288,inherited_packet_calls=1152,
        remaining_complete_frames=1152,remaining_measured_frames=576,remaining_packet_cap=1152,
        archived_failed_slot=read(V5_RUN/'case_reservations.json')['cases'][-1],
        incomplete_prior_fresh_visual_encoder_attempts=1,known_fresh_visual_attempts_at_success=1729,
        logical_grid_slots=1728,complete_frame_cap=1728,measured_frame_cap=864,
        physical_endpoint_entries_at_success=1729,
        recovery_is_not_a_repeated_scientific_case=False,
        recovery_scope='Retain576 complete, restore1152 missing complete slots, explicitly repeat one incomplete first entropy warmup',
        additional_subcomputation='One failed fresh VAE encoder/VQ attempt is preserved in addition to1728 completed fresh end-to-end calls',
        prior_failed_attempts=[dict(version='v2',actual_model_construction=True,complete=0,reserved=0,packets=0),
                              dict(version='v3',actual_model_construction=True,complete=0,reserved=0,packets=0),
                              dict(version='v4',launched=False),
                              dict(version='v5',actual_model_construction=True,complete=576,reserved=1,packets=1152)],
        actual_numerical_qualification='Inherited raw96conditions qualified; remaining four methods pending',automatic_retries=False)
    reg['max_seconds']=budget['remaining_seconds'];reg['cumulative_wall_budget']=budget
    visual=read(reg['visual_request']['path']);static=visual['static']
    assert static['fitting_calls']==0
    for item in (static['table'],static['completion']):
        assert sha(item['path'])==item['sha256'];reg['input_sha256'][item['path']]=item['sha256']
    link=ROOT/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep48_link_core_v1.py'
    assert sha(link)=='4ee75147eec3e1008045bc7cfd30eb0cb0ff96d0468097384251a48870deabb4'
    reg['code_sha256'][str(link)]=sha(link)
    reg['actual_received_static_family_support']=dict(binding=static,implementation=pin(link),
        existing_public_profile_counts=[144,360],TX_family_expanded=False,new_static_fitting_calls=0)
    for name in ('timing_owner_v6.py','recovery_adapter.py','test_recovery.py'):
        p=HERE/name;reg['code_sha256'][str(p)]=sha(p)
    write(OUT/'registration.json',reg)
    print('V6_REGISTERED_NO_MODEL_OR_CHANNEL_CALLS')


def verify():
    reg=inherited_verify()
    assert sha(reg['recovery_of_v5']['path'])==reg['recovery_of_v5']['sha256']
    assert sha(reg['recovery_authorization']['path'])==reg['recovery_authorization']['sha256']
    authorization()
    for p,h in reg['v5_failure_evidence_sha256'].items():assert sha(p)==h,p
    assert prior_evidence()==reg['v5_failure_evidence_sha256']
    assert elapsed_budget()==reg['cumulative_wall_budget'] and reg['max_seconds']==reg['cumulative_wall_budget']['remaining_seconds']
    return reg


def worker():
    reg=verify();prior=read(reg['prior_journal']['path'])
    records={r['source_index']:r for r in reg['records']}
    # Install the resume class and loop before building models or entering any
    # endpoint. The inherited frozen calls are never submitted to fresh_case.
    old.CaseReservations=resumed_reservations(old.CaseReservations,prior,records,reg['prior_journal'])
    old._resume_seed=seed_completed
    old.worker,proof=adapt_worker(old)
    write(OUT/'actual_resume_adapter.json',proof)
    def checked_build_adapter(runtime):
        compiled,build_proof=adapt_build(runtime)
        def checked_build(*args,**kwargs):
            system=compiled(*args,**kwargs)
            check=same_numerical_runtime(read(V5_RUN/'numerical_runtime.json'),read(RUN/'numerical_runtime.json'))
            check.update(previous=pin(V5_RUN/'numerical_runtime.json'),current=pin(RUN/'numerical_runtime.json'))
            write(OUT/'actual_inherited_runtime_check.json',check)
            return system
        return checked_build,build_proof
    v5.adapt_build_live=checked_build_adapter
    v5.worker()
    journal=read(RUN/'case_reservations.json')
    assert journal['cases'][:576]==prior['cases'][:576]
    assert len(journal['cases'])==1728 and all(r['state']=='COMPLETE' for r in journal['cases'])
    completion=read(OUT/'completion.json')
    completion.update(inherited_journal=reg['prior_journal'],inherited_complete_frames=576,
        newly_completed_frames=1152,completed_frames_across_attempts=1728,
        failed_partial_visual_attempts=1,known_fresh_visual_attempts_across_attempts=1729,
        physical_endpoint_entries_across_attempts=1729,
        prior_failure_preserved=True,prior_failed_slot_not_counted_as_warmup=True,
        source_calls_scope='Only new v6 process; inherited v5 completed calls are bound by journal',
        inherited_resource_history=pin(V5_RUN/'resource_snapshots.jsonl'),
        inherited_numerical_runtime=pin(V5_RUN/'numerical_runtime.json'),
        recovery_authorization=reg['recovery_authorization'],
        cumulative_wall_budget=reg['cumulative_wall_budget'],
        actual_received_static_family_support=reg['actual_received_static_family_support'])
    write(OUT/'completion.json',completion)


def run():
    verify()
    lock=(V5_OUT/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v5.run()


for module in (v5,v4,v3,v2,old):module.verify=verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register-recovery','verify','run','_worker']);args=p.parse_args()
    try:
        if args.command=='register-recovery':register_recovery()
        elif args.command=='verify':verify();print('V6_BINDINGS_VERIFIED_NO_SCIENCE')
        elif args.command=='run':run()
        else:worker()
    except BaseException:
        if args.command in ('run','_worker') and (OUT/'launch.json').exists():
            write(RUN/('worker_failure.json' if args.command=='_worker' else 'failure.json'),dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
