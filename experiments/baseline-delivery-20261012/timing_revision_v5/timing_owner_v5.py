"""Unlaunched-v4 wrapper-binding correction; no repeated scientific attempt."""
import argparse
import copy
import fcntl
from pathlib import Path
import sys
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'timing_revision_v4'))
import timing_owner_v4 as v4
from live_globals import adapt_build_live
ROOT,TMP=v4.ROOT,v4.TMP
V4_OUT,V4_RUN=v4.OUT,v4.RUN
OUT=ROOT/'results/baseline_strength_20261011/timing/six_method_v5'
RUN=TMP/'six_method_timing_v5'
read,write,sha,pin=v4.read,v4.write,v4.sha,v4.pin
inherited_verify=v4.verify
v4.OUT,v4.RUN=OUT,RUN
v4.v3.OUT,v4.v3.RUN=OUT,RUN
v4.v3.v2.OUT,v4.v3.v2.RUN=OUT,RUN
v4.v3.old.OUT,v4.v3.old.RUN,v4.v3.old.__file__=OUT,RUN,__file__


def unlaunched_v4():
    assert not (V4_OUT/'launch.json').exists() and not (V4_OUT/'actual_wait.json').exists()
    assert not V4_RUN.exists(), 'A v4 runtime attempt exists; inspect before dispatch'
    return {str(p):sha(p) for p in [V4_OUT/'registration.json',V4_OUT/'recovery_preparation.json',V4_OUT/'cpu_tests.json']}


def register_recovery():
    assert not (OUT/'registration.json').exists()
    evidence=unlaunched_v4()
    reg=copy.deepcopy(read(V4_OUT/'registration.json'))
    for p,h in {**reg['code_sha256'],**reg['input_sha256'],**reg['recovery_evidence_sha256'],**reg['v3_failure_evidence_sha256']}.items():assert sha(p)==h,p
    reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V5_RECOVERY',
        unlaunched_v4_registration=pin(V4_OUT/'registration.json'),
        unlaunched_v4_evidence_sha256=evidence,
        wrapper_revision='Corrected build function retains original runtime globals; v2 later shared_visual replacement remains visible',
        v4_model_or_packet_calls=0,actual_numerical_qualification='NOT_RUN',automatic_retries=False)
    for p in [Path(__file__),HERE/'live_globals.py',HERE/'test_wrapper_order.py']:
        reg['code_sha256'][str(p)]=sha(p)
    write(OUT/'registration.json',reg)
    write(OUT/'recovery_preparation.json',dict(status='RECOVERY_REGISTERED_NOT_LAUNCHED',
        v4_preserved_unlaunched=True,v2_v3_actual_model_constructions_preserved=True,
        prior_cases=0,prior_packets=0,remaining_total_cases=1728,remaining_measured_cases=864,
        remaining_packet_cap=2304,new_model_calls=0,new_channel_calls=0))
    print('V5_REGISTERED_NO_MODEL_OR_CHANNEL_CALLS')


def verify():
    reg=inherited_verify()
    assert sha(reg['unlaunched_v4_registration']['path'])==reg['unlaunched_v4_registration']['sha256']
    for p,h in reg['unlaunched_v4_evidence_sha256'].items():assert sha(p)==h,p
    unlaunched_v4()
    return reg


def worker():
    v4.adapt_build=adapt_build_live
    v4.worker()


def run():
    verify()
    lock=(V4_OUT/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v4.run()


v4.verify=v4.v3.verify=v4.v3.v2.verify=v4.v3.old.verify=verify
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register-recovery','verify','run','_worker']);args=p.parse_args()
    try:
        if args.command=='register-recovery':register_recovery()
        elif args.command=='verify':verify();print('V5_BINDINGS_VERIFIED_NO_SCIENCE')
        elif args.command=='run':run()
        else:worker()
    except BaseException:
        if args.command in ('run','_worker') and (OUT/'launch.json').exists():
            write(RUN/('worker_failure.json' if args.command=='_worker' else 'failure.json'),
                  dict(status='STOPPED_NO_RETRY',traceback=traceback.format_exc()))
        raise
