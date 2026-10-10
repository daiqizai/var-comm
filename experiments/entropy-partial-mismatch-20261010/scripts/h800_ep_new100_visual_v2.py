"""Finite alternate-H800 new100 owner. First normal-frame RX proves its own
compatibility after independent decoding and reuses that result. No extra probe,
no TX, PHY, metric, truth-corrected receiver or automatic retry.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import types
import ep_new100_alternate_visual_core_v2 as evaluation
import h800_ep_new100_phy_v1 as physical
import h800_ep_full_visual_v2 as original
import h800_alternate_gpu_resource_v3 as resources

source=physical.source;gate=source.gate;g=source.g;core=evaluation.pilot
confirmation=evaluation.confirmation
SCHEMA='H800_EP_NEW100_FOUR_ARM_VISUAL_V2'
PASS='PASS_H800_EP_NEW100_ALTERNATE_H800_RECONSTRUCTIONS_ONLY'
CAPS=evaluation.CAPS
ORIGINAL_SHA='6b1c81ef90be96d4c83a509464973e8f75f424ec6dcf62452d0035c2c5e149bd'
PHY_SHA='f6088c2a30506c8f85b2e5aaf9cafa161ffe7491ee22f12bd5f6a3df7dda11ad'
PHY_CORE_SHA='4602e50055d84e9ed12bf9020ccb297490b50ee6a2c6c1e3f565859d7acbbaa5'
RESOURCE_SHA='ce303965e47d2b947707696c392a8f7bb81910bbc19afd8e277ed174fd6132a9'
TOOLS=tuple(dict.fromkeys(physical.TOOLS+original.TOOLS+('h800_ep_full_visual_v2.py','h800_ep_pilot_visual_v1.py',
    'h800_mismatch_visual_v1.py','h800_mismatch_phy_v1.py','ep_new100_visual_core_v1.py','h800_ep_new100_source_v1.py',
    'h800_ep_new100_visual_v1.py','ep_new100_alternate_visual_core_v2.py','h800_alternate_gpu_resource_v3.py')))
execution=original.execution


def cpu_closure(pins):
    root=gate.RT/'qualification/h800_ep_new100_phy_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact new100 actual PHY attempt required')
    done=source.readpin(pins['completion']);outer=source.readpin(pins['owner_actual_wait'])
    worker=source.readpin(done['worker_completion']);request=g.checked_json(root/'registered/request.json',done['request_sha256'])
    g.require(done['schema']==worker['schema']==physical.SCHEMA and done['status']==worker['status']==physical.PASS and
        done['actual_children_waited'] is True and done['worker_exit_codes']==[0,0,0] and
        len(done['actual_waits'])==3 and all(w['success'] is True and w['child_exit_code']==0 for w in done['actual_waits']) and
        outer['actual_wait'] is True and outer['returncode']==0 and not outer.get('timeout',False) and
        not outer.get('interrupted_or_timeout',False) and worker['request_sha256']==done['request_sha256'],
        'Actual successful three-provider owner/worker closure required')
    for group,expected in zip(physical.core.GROUPS,done['actual_waits']):
        path=root/'registered/run'/group/'actual_child_wait.json'
        g.require(source.readpin(core.descriptor(path))==expected,'Provider actual wait differs from final completion')
    sr,sw,records,inputs=physical.source_closure(request['source100_closed'])
    providers,raw,_=physical.provider_bindings(request['T1_original241_relocation'])
    frames=confirmation.frame_grid(source.readpin(inputs['selection']),inputs['selection'],source.readpin(inputs['content_gate']),source.readpin(inputs['policy_bundle']))
    g.require(request['schema']==physical.SCHEMA and request['status']=='REGISTERED_NOT_EXECUTED' and
        request['records']==records and request['sources']==sw['sources'] and request['inputs']==inputs and
        request['frames']==frames and request['providers']==providers and request['raw_binding']==raw and
        request['complete_scientific_caps']==confirmation.scientific_caps() and request['complete_scientific_caps']['visual']==CAPS and
        request['tool_bindings']==physical.bindings() and request['packet_cap']==7200,
        'Closed PHY source, frozen policies, receive domains or original budgets differ')
    result=worker['results'];logical,ledger=physical.core.close_groups(result['groups'],frames,source.readpin)
    g.require(logical==source.readpin(result['logical_rows']) and ledger==result['packet_ledger']==done['packet_ledger'] and
        worker['logical_count']==done['logical_count']==3600 and worker['CUDA_initialized'] is False and
        worker['source_RX_calls']==worker['neural_model_calls']==done['source_RX_calls']==done['neural_model_calls']==0,
        'Complete actual CPU outcomes and finite packet ledger required')
    membership={}
    for result in result['groups']:
        group=result['group'];allowed={confirmation.canonical(p) for p in result['physical_frames']}
        g.require(len(allowed)==result['new_physical_count'] and all(confirmation.canonical(r['physical_frame']) in allowed
            for r in logical if r['provider_group']==group),'Every logical receive result must belong to the actual closed provider')
        membership[group]=allowed
    _,_,static=gate.policy_and_static()
    prior=source.original.prerequisites()[0]
    return request,worker,logical,prior,static,sr['device_identity_binding']


def visual_identity(device):
    return dict(original.visual_identity(device),population=evaluation.POPULATION,
        raw_parser_source_sha256=g.sha(evaluation.raw.__file__),actual_receiver_core_sha256=g.sha(evaluation.__file__),
        raw_semantics='Original full433 hard-token KEEP including body CRC rejection',
        entropy_semantics='Actual received family/m/K/bits; independent original provider; no TX CDF',
        historical_image_reuse_admitted=0)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);resources.validate_policy(r)
    e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    cpu,worker,logical,prior,static,source_device=cpu_closure(r['CPU_closed'])
    identity=visual_identity(runner.device_from_request(r).device_identity_binding())
    g.require(r['source_device_identity']==source_device and r['relocation_witness']==evaluation.select_witness(logical,cpu['sources'],source.readpin,g.inside),
        'Original source identity and first actual normal-frame relocation witness must remain bound')
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['population']==evaluation.POPULATION and
        r['caps']==CAPS and r['execution']==e and r['logical_rows']==logical and r['logical_events']==cpu['frames'] and
        r['records']==cpu['records'] and r['sources']==cpu['sources'] and r['inputs']==cpu['inputs'] and
        r['raw_binding']==cpu['raw_binding'] and r['providers']==cpu['providers'] and r['static']==static and
        r['visual_identity']==identity and r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900) and
        r['first_PHY_scientific_caps']==cpu['complete_scientific_caps'],'Frozen new100 actual RX registration differs')
    runner.check_tools(r['tool_bindings'])
    g.require(r['encoder_calls']==r['source_TX_calls']==r['PHY_calls']==r['metric_model_calls']==r['Direct_calls']==0 and
        r['historical_image_reuse_admitted']==0 and r['automatic_successor'] is False and r['policy_selection_uses_new100'] is False,
        'Only finite actual receive decoding and reconstruction are admitted')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh independent new100 visual registration required')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    cpu,worker,logical,prior,static,source_device=cpu_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    g.require(a.target_index in (0,1,2,3),'Explicit observed same-host H800 required')
    old=gate.RT/'qualification/h800_ep_new100_visual_v1_attempt1/registered'
    g.require(not old.exists(),'Original visual already registered; cumulative science must be separately closed before migration')
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,a.target_index)
    witness=evaluation.select_witness(logical,cpu['sources'],source.readpin,g.inside)
    out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',population=evaluation.POPULATION,caps=CAPS,execution=e,spec=spec,
        CPU_closed=pins,logical_rows=logical,logical_events=cpu['frames'],records=cpu['records'],sources=cpu['sources'],inputs=cpu['inputs'],
        providers=cpu['providers'],raw_binding=cpu['raw_binding'],static=static,visual_identity=visual_identity(device.device_identity_binding()),
        first_PHY_scientific_caps=cpu['complete_scientific_caps'],resource_policy=resources.policy(a.target_index),
        resource_guard=core.descriptor(Path(resources.__file__)),device_receipt=dp,target_index=a.target_index,source_device_identity=source_device,relocation_witness=witness,
        device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        encoder_calls=0,source_TX_calls=0,PHY_calls=0,metric_model_calls=0,Direct_calls=0,
        historical_image_reuse_admitted=0,automatic_successor=False,policy_selection_uses_new100=False,
        purpose='Four frozen methods on same unseen100; explicit alternate H800, first budgeted normal-frame post-RX compatibility check, hard600RX cap')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def images(r,backend,codec,partial,ledger,boundary,out):
    return evaluation.images(r,backend,codec,partial,ledger,boundary,out,g,source.readpin)


def engine():
    g.require(g.sha(original.__file__)==ORIGINAL_SHA and g.sha(physical.__file__)==PHY_SHA and
        g.sha(physical.core.__file__)==PHY_CORE_SHA and g.sha(resources.__file__)==RESOURCE_SHA,
        'Frozen actual owner/PHY/resource implementation changed')
    ns=dict(vars(source.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS,DeviceAdapter=resources.DeviceAdapter)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


# The already used bounded owner and worker retain their original code objects.
# Only the private namespace's input closure, finite caps and output scope change.
_ns=dict(vars(original));_ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS,
    g=g,gate=gate,core=core,evaluation=evaluation,registration=registration,images=images)
run=gate.source.host.clone_function(original.run,_ns)
worker=gate.source.host.clone_function(original.worker,_ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(0,1,2,3),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
