"""Finite actual four-metric owner for closed independent new100 reconstructions.

One GPU2 and two CPU cores; model3/reference100/pair3600 hard caps. The original
four computations, weights, native identity and resource controls are retained.
No policy selection, historical score reuse, bootstrap or automatic successor.
"""
from __future__ import annotations
import argparse
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
import h800_ep_full_metric_owner_v2 as original
import h800_shared_gpu_resource_v2 as resources
import ep_new100_metric_core_v1 as metric

g=original.g;gate=original.gate;core=original.core
SCHEMA='H800_EP_NEW100_FOUR_METRIC_OWNER_V1'
PASS='PASS_H800_EP_NEW100_FOUR_METRICS_COMPLETE'
CAPS=metric.CAPS
ORIGINAL_SHA='0ff1e074c8881e4f15eff7c90943431eed092c578e0ac6ae712360396dfa1ebb'
CORE_SHA='47b1dc3f6ab563c16e9779067c95f9c4cd8647ec86a13d1372778d25fb757427'
VISUAL_SHA='c568dc7e72e9b0f8d67b4d4348b8e670df9f58fc762ff8be5e6a5511b38adaa7'
RESOURCE_SHA='0a412ff2705c047ac74426bdb889c55dbad5cb8a669a84bca3b9a8ad728ed7e4'
execution=original.execution
metric_environment=original.metric_environment
runtime_identity=original.runtime_identity
TOOLS=tuple(dict.fromkeys(original.TOOLS+('h800_ep_full_metric_owner_v2.py','ep_new100_confirmation_core_v1.py',
    'ep_new100_metric_core_v1.py','h800_ep_new100_visual_v1.py')))


def visual_implementation():
    path=Path(__file__).with_name('h800_ep_new100_visual_v1.py')
    g.require(g.SHA_RE.fullmatch(VISUAL_SHA) is not None and path.is_file() and g.sha(path)==VISUAL_SHA,
        'Final frozen independent new100 visual implementation required; no preparation placeholder may execute')
    return g.import_file(path,'_new100_metric_closed_visual')


def visual_closure(pins):
    root=gate.RT/'qualification/h800_ep_new100_visual_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact new100 visual attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['schema']=='H800_EP_NEW100_FOUR_ARM_VISUAL_V1' and done['status']=='PASS_H800_EP_NEW100_FOUR_ARM_RECONSTRUCTIONS_ONLY' and
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
    return request,worker


def configuration(spec,device,visual_request):
    result=original.configuration(spec,device)
    result.update(confirmation_population=metric.POPULATION,policy_selection=False,historical_score_reuse_allowed=False,
        confirmation_policy_bundle=visual_request['inputs']['policy_bundle'],
        confirmation_source_manifest=visual_request['inputs']['source_manifest'])
    return result


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);resources.validate_policy(r);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    vr,vw=visual_closure(r['visual_closed']);device=runner.device_from_request(r)
    spec=runner.inherited_spec(vr,e['deadline_unix'],900)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['spec']==spec and r['records']==vr['records'] and r['inputs']==vr['inputs'] and
        r['reconstruction_results']==vw['results']['logical_frames'] and r['logical_events']==vr['logical_events'] and
        r['metric_config']==configuration(spec,device,vr) and r['population']==metric.POPULATION,
        'Frozen independent new100 metric registration changed')
    g.require(r['target_index']==2 and r['device_identity_binding']==device.device_identity_binding() and
        r['PHY_calls']==r['source_calls']==r['VAR_calls']==r['decoder_calls']==r['training_updates']==r['bootstrap_calls']==0 and
        r['automatic_successor'] is False and r['policy_selection'] is False and r['historical_score_reuse_allowed'] is False,
        'Metric-only confirmation scope changed')
    runner.check_tools(r['tool_bindings']);metric.bound_assets(r['metric_config']);return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists() and a.target_index==2,'Fresh new100 metric registration on qualified GPU2 required')
    pins=dict(completion=dict(path=str(g.inside(a.visual_completion)),sha256=a.visual_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.visual_owner_wait)),sha256=a.visual_owner_wait_sha256))
    vr,vw=visual_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(vr,e['deadline_unix'],900);metric_environment(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,a.target_index)
    config=configuration(spec,device,vr);metric.bound_assets(config);visual=visual_implementation()
    names=tuple(dict.fromkeys((*TOOLS,*visual.TOOLS,Path(__file__).name)))
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,visual_closed=pins,
        records=vr['records'],inputs=vr['inputs'],logical_events=vr['logical_events'],reconstruction_results=vw['results']['logical_frames'],
        metric_config=config,population=metric.POPULATION,resource_policy=dict(resources.POLICY),resource_guard=core.descriptor(Path(resources.__file__)),
        device_receipt=dp,target_index=2,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in names},PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,
        training_updates=0,bootstrap_calls=0,automatic_successor=False,policy_selection=False,historical_score_reuse_allowed=False,
        purpose='Complete independent new100 four-arm four-metric evaluation; no configuration selection or bootstrap')
    out.mkdir(parents=True);path=out/'request.json';g.write(path,r);return core.descriptor(path)


def scores(r,evaluator,ledger,boundary,out):
    return metric.scores(r,evaluator,ledger,boundary,out,g,gate.readpin)


def engine():
    g.require(g.sha(original.__file__)==ORIGINAL_SHA and g.sha(metric.__file__)==CORE_SHA and
        g.sha(resources.__file__)==RESOURCE_SHA,'Frozen metric runtime, computation or resource code changed')
    visual=visual_implementation()
    ns=dict(vars(original.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,
        TOOLS=tuple(dict.fromkeys((*TOOLS,*visual.TOOLS))),DeviceAdapter=resources.DeviceAdapter)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual metric owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Metric stage already claimed; no replay');out.mkdir();device.bind_resource_evidence(out,a.request_sha256)
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,execution=e,caps=CAPS))
        try:
            _,_,_,_,extra=metric_environment(spec);snapshot,ad=runner.wait_prelaunch(device,shared,out,e,started,120)
            g.write(out/'prelaunch_resources.json',snapshot);g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
            _,env=shared.controlled_environment(g.BASE,out,6);(out/'home').mkdir()
            env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=extra['environment']['LD_LIBRARY_PATH'],HOME=str(out/'home'),
                LANG='C.UTF-8',LC_ALL='C.UTF-8',VIRTUAL_ENV=str(Path(spec['python']).parent.parent),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                **device.cuda_environment_binding());g.write(out/'controlled_environment.json',env)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Metric deadline expired before launch')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,
                    cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Metric child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'));result=done['results'];counts=result['counts']
            rows=gate.readpin(result['metric_rows']);metric.validate_completion(result,rows,r)
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and result['logical_frames']==3600 and
                result['full_grid_complete'] is True and result['historical_score_reuse']==0 and result['bootstrap_calls']==0 and
                result['policy_selection'] is False and result['actual_unique_pairs']+result['reused_pairs']==3600,
                'Actual complete independent confirmation metric closure differs')
            registration(path,a.request_sha256,runner);metric_environment(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Metric owner final deadline exceeded')
            completion=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False,policy_selection=False,bootstrap_calls=0,statistics_not_computed=True,population=metric.POPULATION)
            g.write(out/'completion.json',completion);return completion
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise

def worker(a,runner):
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    device=runner.device_from_request(r);device.bind_resource_evidence(out,a.request_sha256);device.check_cuda_environment();launch=runner.worker_identity(out,a.request_sha256,a.owner_pid)
    g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    started=time.monotonic();stopped=False;ledger=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'Metric stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Metric deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=10:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'progress.jsonl').open('a') as stream:stream.write(g.json.dumps(dict(resources=snapshot,
                counts=None if ledger is None else ledger.summary(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=metric_environment(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        import torch
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_metric_native_observer')
        g.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and len(os.sched_getaffinity(0))==2,
            'Exactly one registered GPU and two CPU cores required')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        actual=device.validate_cuda_device_identity(str(getattr(props,'uuid','')),module.cuda_driver_pci_bus_id())
        g.write(out/'cuda_device_identity.json',actual)
        g.require(props.total_memory>e['allocator_cap_bytes'],'Insufficient registered device memory')
        torch.cuda.set_per_process_memory_fraction(e['allocator_cap_bytes']/props.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version())
        g.require(observed=={k:v for k,v in g.EXPECTED_RUNTIME.items() if k not in ('threads','interop_threads')},'Original metric framework versions differ')
        for p in (Path(sys.executable),Path(torch.__file__),Path(torch._C.__file__),Path(np.__file__)):environment.path_for_new(p)
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            g.require(os.environ.get(name)=='6','Original six-thread environment required')
        g.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Frozen workspace setting differs')
        before=module.native_linkage(environment,g);g.write(out/'native_before_models.json',before)
        evaluator=metric.FourMetrics(r['metric_config'],ledger,boundary)
        observed.update(threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
        g.require(observed==g.EXPECTED_RUNTIME and evaluator.flags(torch)==metric.FLAGS,'Actual metric runtime flags differ')
        g.require(r['metric_config']['runtime_identity']==runtime_identity(spec,device),'Metric runtime identity is not owner-bound')
        g.write(out/'metric_identity.json',dict(identity=evaluator.identity,metadata=evaluator.metadata,actual_runtime=observed,population=metric.POPULATION,compute_provider_scope=metric.POPULATION,original_four_computations_unchanged=True))
        loaded=module.native_linkage(environment,g,previous=before);g.write(out/'native_after_models.json',loaded)
        results=scores(r,evaluator,ledger,boundary,out/'scores')
        torch.cuda.synchronize();g.write(out/'native_final.json',module.native_linkage(environment,g,previous=loaded))
        metric.bound_assets(r['metric_config']);resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            metric_identity=core.descriptor(out/'metric_identity.json'),PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,
            automatic_successor=False,policy_selection=False,bootstrap_calls=0,statistics_not_computed=True,population=metric.POPULATION,compute_provider_scope=metric.POPULATION,original_four_computations_unchanged=True)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('visual-completion','visual-completion-sha256','visual-owner-wait','visual-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(2,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
