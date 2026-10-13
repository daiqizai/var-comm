"""Reviewed v4 recovery after v3 Swin construction field mismatch.

No retry scheduler. Original failed attempts and all scientific source remain
immutable. Registration/verification do not construct a model or run a frame.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import sys
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'timing_revision_v3'))
import timing_owner_v3 as v3
from constructor_compat import adapt_build
ROOT,TMP=v3.ROOT,v3.TMP
V3_OUT,V3_RUN=v3.OUT,v3.RUN
OUT=ROOT/'results/baseline_strength_20261011/timing/six_method_v4'
RUN=TMP/'six_method_timing_v4'
read,write,sha,pin=v3.read,v3.write,v3.sha,v3.pin
inherited_verify=v3.verify
v3.OUT,v3.RUN=OUT,RUN
v3.v2.OUT,v3.v2.RUN=OUT,RUN
v3.old.OUT,v3.old.RUN,v3.old.__file__=OUT,RUN,__file__


def v3_evidence():
    waited=read(V3_OUT/'actual_wait.json')
    assert waited['actual_child_waited'] is True and waited['exit_code']==1 and not waited['timed_out']
    owner=read(V3_OUT/'launch.json')['owner'];worker=read(V3_OUT/'worker_launch.json')
    assert v3.dead_identity(owner['pid'],owner['start_ticks']) and v3.dead_identity(worker['pid'])
    ledger=read(V3_RUN/'case_reservations.json')
    assert ledger['cases']==[] and ledger['packet_decodes']==0
    assert (ledger['case_cap'],ledger['measured_case_cap'],ledger['packet_cap'])==(1728,864,2304)
    trace=read(V3_RUN/'worker_failure.json')['traceback']
    assert trace.endswith("AttributeError: 'SwinJSCC' object has no attribute 'args'\n")
    assert 's.native.args.multiple_snr=' in trace
    assert not (V3_OUT/'completion.json').exists()
    return {str(p):sha(p) for p in [V3_OUT/'registration.json',V3_OUT/'launch.json',V3_OUT/'worker_launch.json',
        V3_OUT/'actual_wait.json',V3_RUN/'worker_failure.json',V3_RUN/'failure.json',V3_RUN/'worker.log',
        V3_RUN/'numerical_runtime.json',V3_RUN/'case_reservations.json']}


def register_recovery():
    assert not (OUT/'registration.json').exists()
    evidence=v3_evidence()
    reg=copy.deepcopy(read(V3_OUT/'registration.json'))
    for p,h in {**reg['code_sha256'],**reg['input_sha256'],**reg['recovery_evidence_sha256']}.items():assert sha(p)==h,p
    reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V4_RECOVERY',
        recovery_of_v3=pin(V3_OUT/'registration.json'),v3_failure_evidence_sha256=evidence,
        construction_revision='One-line native.args.multiple_snr -> native.multiple_snr integer list',
        failed_attempts=[dict(version='v2',actual_exit=1,real_model_construction=True,cases=0,packets=0),
                         dict(version='v3',actual_exit=1,real_model_construction=True,cases=0,packets=0)],
        actual_numerical_qualification='NOT_RUN',automatic_retries=False)
    vendor=TMP/'vendor/SwinJSCC/net/network.py'
    reg['input_sha256'][str(vendor)]=sha(vendor)
    for p in [Path(__file__),HERE/'constructor_compat.py',HERE/'test_constructor_interfaces.py']:
        reg['code_sha256'][str(p)]=sha(p)
    write(OUT/'registration.json',reg)
    write(OUT/'recovery_preparation.json',dict(status='RECOVERY_REGISTERED_NOT_LAUNCHED',
        both_failed_attempts_preserved=True,prior_actual_model_constructions=True,
        prior_cases=0,prior_packets=0,remaining_total_cases=1728,remaining_measured_cases=864,
        remaining_packet_cap=2304,new_model_calls=0,new_channel_calls=0))
    print('V4_REGISTERED_NO_MODEL_OR_CHANNEL_CALLS')


def verify():
    reg=inherited_verify()
    assert sha(reg['recovery_of_v3']['path'])==reg['recovery_of_v3']['sha256']
    for p,h in reg['v3_failure_evidence_sha256'].items():assert sha(p)==h,p
    v3_evidence()
    return reg


def worker():
    import timing_runtime as runtime
    runtime.build,proof=adapt_build(runtime)
    write(OUT/'actual_constructor_compatibility.json',proof)
    v3.worker()


def run():
    verify()
    lock=(V3_OUT/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v3.run()


v3.verify=v3.v2.verify=v3.old.verify=verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register-recovery','verify','run','_worker']);args=p.parse_args()
    try:
        if args.command=='register-recovery':register_recovery()
        elif args.command=='verify':verify();print('V4_BINDINGS_VERIFIED_NO_SCIENCE')
        elif args.command=='run':run()
        else:worker()
    except BaseException:
        if args.command in ('run','_worker') and (OUT/'launch.json').exists():
            write(RUN/('worker_failure.json' if args.command=='_worker' else 'failure.json'),
                  dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
