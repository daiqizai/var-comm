"""One explicit GPU0 recovery after the preserved GPU3 startup resource refusal.

The original receiver, normal-frame witness, renderer and resource guard are
unchanged. One failed model-load reservation remains spent and unresolved;
one additional startup is admitted, with no additional scientific frame budget.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import types
import h800_ep_new100_visual_v2 as base

g=base.g;gate=base.gate;source=base.source;core=base.core;resources=base.resources
evaluation=base.evaluation;confirmation=base.confirmation;physical=base.physical
original=base.original;execution=base.execution;CAPS=base.CAPS
SCHEMA='H800_EP_NEW100_FOUR_ARM_VISUAL_V3_RECOVERY_GPU0'
PASS='PASS_H800_EP_NEW100_RECOVERY_GPU0_RECONSTRUCTIONS_ONLY'
BASE_SHA='5ea07c138e555702ed703667a7dee764080017c7e8510d16bedb0302573c161a'
FAILED_ROOT=gate.RT/'qualification/h800_ep_new100_visual_v2_attempt1'
OUT=gate.RT/'qualification/h800_ep_new100_visual_v3_attempt1/registered'
FAILURE_PINS={
 'registered/request.json':'e75aab1c69511d5ed6e1f33ab336907c1fbd6428ad833ed2639c3b5980d466c5',
 'run_owner_actual_wait.json':'266169d43b419f1be40f130c502d8c8fd527398b4835de8ef53438b0793b1b96',
 'registered/run/actual_child_wait.json':'0923eff6767843d1cc6bf76f9f3a0661baef44acea81e0f3b3ad5ab1a7c8696a',
 'registered/run/worker_failure.json':'528bb391f77613be8e0f4003a5d4d439cca8f5b9479524ed67b51cdbea65378e',
 'registered/run/failure.json':'1e423a19717acef521116e4ef156f99acce7973e78345de13a245066628bd721',
 'registered/run/child_started.json':'64a104eff96d186f2bbf24affc68ac048e9ddaa6b43304a965479a5d5486b553',
 'registered/run/calls/0001_model_load.reserved.json':'a11e6abf1a93f153f95f62bf114da6367f52555fa1a2669330dd4eaa2be485af',
 'registered/run/resource_guard_failures/4103632_1791649935351311421.json':'f71df7a39c8233f8872695172d7c3f6c84c66fbf3765fcdbd7a2a258f7b5efac'}
TOOLS=tuple(dict.fromkeys(base.TOOLS+('h800_ep_new100_visual_v2.py',)))
cpu_closure=base.cpu_closure;visual_identity=base.visual_identity;images=base.images


def validate_failure(values):
    request=values['registered/request.json'];owner=values['run_owner_actual_wait.json']
    child=values['registered/run/actual_child_wait.json'];failure=values['registered/run/worker_failure.json']
    counts=failure['counts'];zero=dict.fromkeys(CAPS,0)
    started=values['registered/run/child_started.json'];reservation=values['registered/run/calls/0001_model_load.reserved.json']
    resource=values['registered/run/resource_guard_failures/4103632_1791649935351311421.json']
    g.require(owner['actual_wait'] is True and owner['returncode']==1 and not owner.get('timeout',False) and
        not owner.get('interrupted_or_timeout',False) and child['actual_child_waited'] is True and
        child['child_exit_code']==1 and child['success'] is False and
        counts['caps']==CAPS and counts['completed']==zero and counts['reserved']==dict(zero,model_load=1) and
        counts['unresolved']==1 and failure['request_sha256']==FAILURE_PINS['registered/request.json'] and
        failure['error']=="ResourceBusy('GPU free plus verified own reserved margin')",
        'Exact closed startup-only refusal required; paid scientific calls prohibit recovery')
    g.require(request['schema']==base.SCHEMA and request['caps']==CAPS and request['target_index']==3 and
        started['request_sha256']==failure['request_sha256'] and reservation['kind']=='model_load' and
        reservation['pid']==started['pid']==resource['pid'] and resource['request_sha256']==failure['request_sha256'] and
        resource['error']==failure['error'] and resource['scientific_caps_changed'] is False and
        resource['snapshot']['resource_guard']['policy']==resources.policy(3),
        'Original startup reservation, resource failure and GPU3 request must agree')
    oldfail=values['registered/run/failure.json']
    g.require(oldfail['actual_child_waited'] is True and oldfail['child_exit_code']==1 and oldfail['status']=='STOPPED_NO_RETRY',
        'Failed owner must have actually waited its child')
    return request


def failed_attempt():
    values={rel:g.checked_json(FAILED_ROOT/rel,digest) for rel,digest in FAILURE_PINS.items()}
    request=validate_failure(values);run=FAILED_ROOT/'registered/run'
    g.require({p.name for p in (run/'calls').iterdir()}=={'0001_model_load.reserved.json'} and
        not any((run/n).exists() for n in ('completion.json','worker_completion.json','reconstructions')),
        'Previous attempt must contain only its original unresolved model-load reservation')
    started=values['registered/run/child_started.json']
    g.require(all(not Path('/proc',str(started[k])).exists() for k in ('pid','owner_pid')),'Prior owner or child still exists')
    recovery=dict(failed_attempt_pins={k:dict(path=str(FAILED_ROOT/k),sha256=v) for k,v in FAILURE_PINS.items()},
        previous_model_load_attempts=1,previous_model_load_completed=0,previous_unresolved_model_load=1,
        additional_model_load_attempt_cap=1,cumulative_model_load_attempt_cap=2,
        prior_failed_ledger_preserved=True,previous_scientific_calls=dict.fromkeys((k for k in CAPS if k!='model_load'),0),
        cumulative_scientific_caps={k:v for k,v in CAPS.items() if k!='model_load'},
        resource_guard_changed=False,automatic_retry=False,automatic_successor=False,
        reason='Explicit one-time startup recovery on free same-host GPU0; no prior RX or reconstruction')
    return request,recovery


def registration(path,digest,runner):
    r=_base_registration(path,digest,runner);previous,recovery=failed_attempt()
    g.require(g.inside(path)==OUT/'request.json' and r['target_index']==0 and r['recovery_budget']==recovery and
        r['CPU_closed']==previous['CPU_closed'],'Exact GPU0 recovery and shared cumulative science budget required')
    return r


def prepare(a,runner):
    previous,recovery=failed_attempt();out=g.inside(a.out)
    g.require(out==OUT and not out.exists() and a.target_index==0,'Fresh explicit GPU0 recovery only')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    g.require(pins==previous['CPU_closed'],'Recovery may not change the already closed actual PHY input')
    cpu,worker,logical,prior,static,source_device=cpu_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    g.require(not (gate.RT/'qualification/h800_ep_new100_visual_v1_attempt1/registered').exists(),'Original visual v1 must remain unregistered')
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,0)
    witness=evaluation.select_witness(logical,cpu['sources'],source.readpin,g.inside)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',population=evaluation.POPULATION,caps=CAPS,execution=e,spec=spec,
        CPU_closed=pins,logical_rows=logical,logical_events=cpu['frames'],records=cpu['records'],sources=cpu['sources'],inputs=cpu['inputs'],
        providers=cpu['providers'],raw_binding=cpu['raw_binding'],static=static,visual_identity=visual_identity(device.device_identity_binding()),
        first_PHY_scientific_caps=cpu['complete_scientific_caps'],resource_policy=resources.policy(0),
        resource_guard=core.descriptor(Path(resources.__file__)),device_receipt=dp,target_index=0,source_device_identity=source_device,relocation_witness=witness,
        device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,recovery_budget=recovery,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        encoder_calls=0,source_TX_calls=0,PHY_calls=0,metric_model_calls=0,Direct_calls=0,
        historical_image_reuse_admitted=0,automatic_successor=False,policy_selection_uses_new100=False,
        purpose='One explicit GPU0 startup recovery; same four frozen methods, actual PHY and original600RX/3600render budget')
    out.mkdir(parents=True);path=out/'request.json';g.write(path,r);return core.descriptor(path)


def engine():
    g.require(g.sha(base.__file__)==BASE_SHA,'Frozen v2 receiver/owner implementation changed')
    ns=dict(vars(base.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


_ns=dict(base.run.__globals__);_ns.update(vars(base));_ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS,
    registration=registration,images=images)
_base_registration=gate.source.host.clone_function(base.registration,_ns)
run=gate.source.host.clone_function(base.run,_ns)
worker=gate.source.host.clone_function(base.worker,_ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(0,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
