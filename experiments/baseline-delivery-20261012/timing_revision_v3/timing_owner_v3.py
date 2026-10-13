"""One reviewed recovery after v2 constructor-only namespace failure.

The original v1/v2 attempts, registrations and sources are preserved. No actual
case or packet was consumed by v2, while real model construction is recorded.
register-recovery is metadata/byte verification only; run is a separate action.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
V2=HERE.parent/'timing_revision_v2'
sys.path.insert(0,str(V2))
import timing_owner_v2 as v2
from module_aliases import ExactByteAliases

ROOT,TMP=v2.ROOT,v2.TMP
V2_OUT,V2_RUN=v2.OUT,v2.RUN
OUT=ROOT/'results/baseline_strength_20261011/timing/six_method_v3'
RUN=TMP/'six_method_timing_v3'
read,write,sha,pin=v2.read,v2.write,v2.sha,v2.pin
old=v2.old
v2.OUT,v2.RUN=OUT,RUN
old.OUT,old.RUN,old.__file__=OUT,RUN,__file__


def dead_identity(pid, start_ticks=None):
    path=Path('/proc')/str(pid)
    if not path.exists(): return True
    return start_ticks is not None and (path/'stat').read_text().split()[21] != str(start_ticks)


def recovery_evidence():
    wait=read(V2_OUT/'actual_wait.json')
    assert wait['actual_child_waited'] is True and wait['exit_code']==1 and not wait['timed_out']
    launch=read(V2_OUT/'launch.json'); child=read(V2_OUT/'worker_launch.json')
    assert dead_identity(launch['owner']['pid'],launch['owner']['start_ticks'])
    assert dead_identity(child['pid']), 'Prior child PID currently exists; inspect identity before recovery'
    ledger=read(V2_RUN/'case_reservations.json')
    assert ledger['cases']==[] and ledger['packet_decodes']==0
    assert (ledger['case_cap'],ledger['measured_case_cap'],ledger['packet_cap'])==(1728,864,2304)
    failure=read(V2_RUN/'worker_failure.json')['traceback']
    assert failure.endswith('RuntimeError: Unexpected imported module: main_raw64_plan_only\n')
    assert 'whole=portable.create_runtime' in failure and 'system=runtime.build' in failure
    assert not (V2_OUT/'completion.json').exists()
    return {str(p):sha(p) for p in [V2_OUT/'registration.json',V2_OUT/'launch.json',
        V2_OUT/'worker_launch.json',V2_OUT/'actual_wait.json',V2_RUN/'worker_failure.json',
        V2_RUN/'failure.json',V2_RUN/'worker.log',V2_RUN/'case_reservations.json',
        V2_RUN/'numerical_runtime.json',V2_OUT/'coordination_release.json']}


def register_recovery():
    assert not (OUT/'registration.json').exists(), 'Recovery already registered'
    evidence=recovery_evidence()
    reg=copy.deepcopy(read(V2_OUT/'registration.json'))
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items(): assert sha(p)==h,p
    phy=read(reg['phy_request']['path']);roles=phy['raw_binding']['roles']
    pairs=[]
    for role,name in [('plan_module','main_raw64_plan_only'),('adapter_module','main_raw64_packet_adapter')]:
        a=Path(roles[role]);b=ROOT/'experiments/main-raw64-20261007'/(name+'.py')
        assert a.read_bytes()==b.read_bytes()
        pairs.append(dict(module_name=name,loaded_path=str(a),requested_path=str(b),sha256=sha(a)))
    ExactByteAliases(pairs).verify()
    reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V3_RECOVERY',
        recovery_of=pin(V2_OUT/'registration.json'),recovery_evidence_sha256=evidence,
        recovery_reason='Strict same-name module path conflict between two byte-identical pinned copies during construction',
        module_alias_pairs=pairs,previous_attempt_case_calls=0,previous_attempt_packet_calls=0,
        previous_attempt_real_model_construction=True,
        previous_attempt_inference_or_channel_calls=0,
        recovery_is_not_a_repeated_scientific_case=True,
        automatic_retries=False,actual_numerical_qualification='NOT_RUN')
    for p in [Path(__file__),HERE/'module_aliases.py',HERE/'test_recovery.py']:
        reg['code_sha256'][str(p)]=sha(p)
    write(OUT/'registration.json',reg)
    write(OUT/'recovery_preparation.json',dict(status='RECOVERY_REGISTERED_NOT_LAUNCHED',
        original_attempt_preserved=True,prior_real_model_construction=True,
        completed_or_reserved_timing_cases=0,packet_calls=0,
        remaining_total_case_budget=1728,remaining_measured_budget=864,
        remaining_digital_packet_budget=2304,new_model_or_channel_calls=0))
    print('RECOVERY_REGISTERED_NO_NEW_MODEL_OR_PACKET_CALLS')


def verify():
    reg=read(OUT/'registration.json')
    assert sha(reg['previous_registration']['path'])==reg['previous_registration']['sha256']
    assert sha(reg['recovery_of']['path'])==reg['recovery_of']['sha256']
    for p,h in {**reg['code_sha256'],**reg['input_sha256'],**reg['recovery_evidence_sha256']}.items():
        assert sha(p)==h,p
    recovery_evidence()
    ExactByteAliases(reg['module_alias_pairs']).verify()
    return reg


def worker():
    import timing_runtime as runtime
    original_build=runtime.build
    def construct(registration,reservations,out,boundary):
        sys.path[:0]=[str(runtime.ES),str(ROOT/'experiments/wcl-evidence-closure-20261009/scripts')]
        import t1_phy,ep_phy
        aliases=ExactByteAliases(registration['module_alias_pairs'])
        system=aliases.call(t1_phy,ep_phy,lambda:original_build(registration,reservations,out,boundary))
        assert {(r['loader'],r['module_name']) for r in aliases.events}=={
            (loader,name) for loader in ('t1_phy.load_exact','ep_phy.load_exact')
            for name in ('main_raw64_plan_only','main_raw64_packet_adapter')}
        write(OUT/'actual_module_aliases.json',dict(
            scope='constructor-only exact byte-equivalent path alias',events=aliases.events,
            original_loaders_restored_before_first_case=True,
            original_module_objects_preserved=True,scientific_source_bytes_modified=False,
            source_math_or_threads_modified=False))
        return system
    runtime.build=construct
    v2.worker()


def run():
    verify()
    prior_lock=(V2_OUT/'owner.lock').open('a+')
    fcntl.flock(prior_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v2.run()


v2.verify=verify
old.verify=verify
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['register-recovery','verify','run','_worker'])
    args=parser.parse_args()
    try:
        if args.command=='register-recovery': register_recovery()
        elif args.command=='verify': verify();print('RECOVERY_BINDINGS_VERIFIED_NO_SCIENTIFIC_CALLS')
        elif args.command=='run': run()
        else: worker()
    except BaseException:
        if args.command in ('run','_worker') and (OUT/'launch.json').exists():
            write(RUN/('worker_failure.json' if args.command=='_worker' else 'failure.json'),
                  dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
