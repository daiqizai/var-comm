"""Finite new100 scoring on the actual closed visual's selected H800.

The four numerical computations and finite scoring loop are unchanged. The
selected device and resource policy are inherited from closed visual v2;
v1 remains unexecuted and immutable. No image, channel, selection or bootstrap.
"""
from __future__ import annotations
import argparse
import inspect
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import types
import numpy as np
import h800_ep_new100_metric_owner_v1 as base
import h800_alternate_gpu_resource_v3 as resources

g=base.g;gate=base.gate;core=base.core;metric=base.metric;original=base.original
SCHEMA='H800_EP_NEW100_FOUR_METRIC_OWNER_V2_ALTERNATE'
PASS='PASS_H800_EP_NEW100_ALTERNATE_FOUR_METRICS_COMPLETE'
CAPS=base.CAPS
BASE_SHA='fe2c954d26f457b076798efe4c3b01c4a5d9966804834977a635d26b032b7a8c'
ORIGINAL_SHA=base.ORIGINAL_SHA
CORE_SHA='47b1dc3f6ab563c16e9779067c95f9c4cd8647ec86a13d1372778d25fb757427'
VISUAL_SHA='5ea07c138e555702ed703667a7dee764080017c7e8510d16bedb0302573c161a'
RESOURCE_SHA='ce303965e47d2b947707696c392a8f7bb81910bbc19afd8e277ed174fd6132a9'
execution=base.execution
metric_environment=base.metric_environment
runtime_identity=base.runtime_identity
TOOLS=tuple(dict.fromkeys(base.TOOLS+('h800_ep_new100_metric_owner_v1.py',
    'h800_ep_new100_visual_v2.py','ep_new100_alternate_visual_core_v2.py','h800_alternate_gpu_resource_v3.py')))


def previous_attempt_unexecuted():
    old=gate.RT/'qualification/h800_ep_new100_metrics_v1_attempt1/registered'
    g.require(not old.exists(),'Original new100 metric v1 already registered; no automatic cumulative-budget replay')


def visual_implementation():
    path=Path(__file__).with_name('h800_ep_new100_visual_v2.py')
    g.require(g.SHA_RE.fullmatch(VISUAL_SHA) is not None and path.is_file() and g.sha(path)==VISUAL_SHA,
        'Final frozen independent new100 visual implementation required; no preparation placeholder may execute')
    return g.import_file(path,'_new100_metric_closed_visual')


def visual_closure(pins):
    root=gate.RT/'qualification/h800_ep_new100_visual_v2_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact new100 visual attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['schema']=='H800_EP_NEW100_FOUR_ARM_VISUAL_V2' and done['status']=='PASS_H800_EP_NEW100_ALTERNATE_H800_RECONSTRUCTIONS_ONLY' and
        done['actual_wait']['success'] is True and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0 and not wait.get('timeout',False) and
        not wait.get('interrupted_or_timeout',False),
        'Complete actual new100 visual owner closure required before metric admission')
    visual=visual_implementation();runner=visual.engine()
    request=visual.registration(root/'registered/request.json',done['request_sha256'],runner)
    worker=gate.readpin(done['worker_completion']);result=worker['results']
    g.require(worker['schema']==done['schema']==visual.SCHEMA and worker['status']==done['status']==visual.PASS and
        worker['request_sha256']==done['request_sha256'] and request['population']==metric.POPULATION and
        request['caps']==visual.CAPS==metric.plan.scientific_caps()['visual'], 'Closed visual schema, population or caps changed')
    child=gate.readpin(dict(path=str(root/'registered/run/actual_child_wait.json'),sha256=g.sha(root/'registered/run/actual_child_wait.json')))
    g.require(child==done['actual_wait'],'Actual visual wait differs from embedded completion')
    metric.plan.check_ledger(result['counts'],visual.CAPS);c=result['counts']['completed']
    metric.plan.complete_grid(request['logical_events'],request['logical_events'])
    g.require(len(result['logical_frames'])==3600 and len(request['records'])==100 and c['model_load']==1 and
        c['encoder']==c['source_tx']==0 and c['source_rx']<=600 and c['var_render']==c['decoder_forward'] and
        worker['quality_scores']==0,'Complete3600 bounded visual outcomes required')
    resources.validate_policy(request)
    g.require(request['target_index'] in (0,1,2,3),'Closed visual must use the explicitly observed same-host H800')
    return request,worker


def registration(path,digest,runner):
    previous_attempt_unexecuted()
    r=g.checked_json(g.inside(path),digest);resources.validate_policy(r);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    vr,vw=visual_closure(r['visual_closed']);device=runner.device_from_request(r)
    spec=runner.inherited_spec(vr,e['deadline_unix'],900)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['spec']==spec and r['records']==vr['records'] and r['inputs']==vr['inputs'] and
        r['reconstruction_results']==vw['results']['logical_frames'] and r['logical_events']==vr['logical_events'] and
        r['metric_config']==configuration(spec,device,vr) and r['population']==metric.POPULATION,
        'Frozen independent new100 metric registration changed')
    g.require(r['target_index']==vr['target_index'] and r['device_identity_binding']==device.device_identity_binding() and
        r['PHY_calls']==r['source_calls']==r['VAR_calls']==r['decoder_calls']==r['training_updates']==r['bootstrap_calls']==0 and
        r['automatic_successor'] is False and r['policy_selection'] is False and r['historical_score_reuse_allowed'] is False,
        'Metric-only confirmation scope changed')
    g.require(r['device_receipt']==vr['device_receipt'] and r['resource_policy']==vr['resource_policy'],
        'Metric must inherit the exact observed device and resource policy of its closed visual')
    runner.check_tools(r['tool_bindings']);metric.bound_assets(r['metric_config']);return r


def prepare(a,runner):
    previous_attempt_unexecuted()
    out=g.inside(a.out);g.require(not out.exists(),'Fresh independent alternate-device metric registration required')
    pins=dict(completion=dict(path=str(g.inside(a.visual_completion)),sha256=a.visual_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.visual_owner_wait)),sha256=a.visual_owner_wait_sha256))
    vr,vw=visual_closure(pins)
    g.require(a.target_index==vr['target_index'],'Metric target must equal the actual closed visual target')
    e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(vr,e['deadline_unix'],900);metric_environment(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,a.target_index)
    g.require(dp==vr['device_receipt'],'The exact closed visual device observation must be retained')
    config=configuration(spec,device,vr);metric.bound_assets(config);visual=visual_implementation()
    names=tuple(dict.fromkeys((*TOOLS,*visual.TOOLS,Path(__file__).name)))
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,visual_closed=pins,
        records=vr['records'],inputs=vr['inputs'],logical_events=vr['logical_events'],reconstruction_results=vw['results']['logical_frames'],
        metric_config=config,population=metric.POPULATION,resource_policy=resources.policy(a.target_index),resource_guard=core.descriptor(Path(resources.__file__)),
        device_receipt=dp,target_index=a.target_index,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in names},PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,
        training_updates=0,bootstrap_calls=0,automatic_successor=False,policy_selection=False,historical_score_reuse_allowed=False,
        purpose='Complete independent new100 four-arm four-metric evaluation; no configuration selection or bootstrap')
    out.mkdir(parents=True);path=out/'request.json';g.write(path,r);return core.descriptor(path)


def engine():
    g.require(g.sha(base.__file__)==BASE_SHA and g.sha(original.__file__)==ORIGINAL_SHA and g.sha(metric.__file__)==CORE_SHA and
        g.sha(resources.__file__)==RESOURCE_SHA,'Frozen metric runtime, computation or resource code changed')
    visual=visual_implementation()
    ns=dict(vars(original.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,
        TOOLS=tuple(dict.fromkeys((*TOOLS,*visual.TOOLS))),DeviceAdapter=resources.DeviceAdapter)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('visual-completion','visual-completion-sha256','visual-owner-wait','visual-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(0,1,2,3),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))

# These numerical/environment/finite-loop functions keep their original code
# objects, with only this module's explicitly selected device/control namespace.
for _name in ('configuration','scores','run','worker'):
    globals()[_name]=gate.source.host.clone_function(getattr(base,_name),globals())
    g.require(globals()[_name].__code__ is getattr(base,_name).__code__,'Original metric computation/loop code changed')


if __name__=='__main__':main()
