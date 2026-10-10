"""Finite complete full1000 four-metric scoring and DINO-L calibration selection."""
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
import h800_ep_full_visual_v1 as visual
import h800_ep_pilot_metric_owner_v1 as pilot_owner
import h800_ep_pilot_metrics_v1 as metric
import h800_ep_full_evaluation_core_v1 as evaluation
source=visual.source;gate=visual.gate;g=visual.g;core=visual.core;full=visual.full
SCHEMA='H800_EP_ORIGINAL1000_FULL_METRICS_V1'
PASS='PASS_H800_ORIGINAL1000_FULL_FOUR_METRICS_AND_CALIBRATION'
CAPS=evaluation.METRIC_CAPS
TOOLS=tuple(dict.fromkeys(visual.TOOLS+('h800_ep_full_visual_v1.py','h800_ep_pilot_metric_owner_v1.py')))
execution=visual.execution
metric_environment=pilot_owner.metric_environment
runtime_identity=pilot_owner.runtime_identity
configuration=pilot_owner.configuration
VISUAL_SHA='44ed9ab36c2507ad4919aeb20c082701362521058f9c126c2774eb9b353e58a9'
METRIC_SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
PILOT_OWNER_SHA='8f5b04978b23d8b4525b03599d13a9aa1277872481624fa17c5a0627f43b60e1'

def visual_closure(pins):
    root=gate.RT/'qualification/h800_ep_full_visual_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact closed full1000 visual attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait']);worker=gate.readpin(done['worker_completion'])
    g.require(done['schema']==worker['schema']==visual.SCHEMA and done['status']==worker['status']==visual.PASS and
        done['actual_wait']['success'] is True and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0 and worker['request_sha256']==done['request_sha256'],
        'Actual full visual owner/worker must close before metric admission')
    r=g.checked_json(root/'registered/request.json',done['request_sha256'])
    cpu,pw,_,prior,static=visual.cpu_closure(r['full_CPU_closed']);device=gate.source.host.DeviceAdapter(r['device_receipt'],r['target_index'])
    g.require(r['schema']==visual.SCHEMA and r['caps']==visual.CAPS and r['static']==static and r['records']==cpu['records'] and
        r['physical_frames']==pw['results']['logical_frames'] and r['logical_events']==cpu['frames'] and r['finalists']==cpu['finalists'] and
        r['policy']==cpu['policy'] and r['first_full_PHY_scientific_caps']==cpu['complete_scientific_caps'] and
        r['visual_identity']==visual.visual_identity(device.device_identity_binding()) and r['population']==evaluation.POPULATION,
        'Closed full visual original1000 identity changed')
    for name,digest in r['tool_bindings'].items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Closed visual source changed: '+name)
    counts=worker['results']['counts'];evaluation.check_counts(counts,visual.CAPS)
    g.require(len(worker['results']['logical_frames'])==len(cpu['frames']) and counts['completed']['model_load']==1 and
        counts['completed']['decoder_forward']==counts['completed']['var_render'] and worker['quality_scores']==0 and
        worker['population']==evaluation.POPULATION,'Complete full1000 reconstruction closure required')
    return r,worker,cpu,prior


def pilot_pair_certificate(cpu):
    # Re-evaluate actual closed pilot admission without new model or scoring calls.
    policy=gate.readpin(cpu['policy']);visual.physical.metric_closure(cpu['pilot_metrics_closed'],policy)
    done=gate.readpin(cpu['pilot_metrics_closed']['completion']);worker=gate.readpin(done['worker_completion'])
    return dict(owner_closed=cpu['pilot_metrics_closed'],metric_identity=worker['metric_identity'],
        metric_rows=worker['results']['metric_rows'],previous_call_counts_preserved=True)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    _,worker,cpu,prior=visual_closure(r['full_visual_closed']);device=runner.device_from_request(r)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['spec']==spec and r['records']==cpu['records'] and r['reconstruction_results']==worker['results']['logical_frames'] and
        r['logical_events']==cpu['frames'] and r['finalists']==cpu['finalists'] and r['policy']==cpu['policy'] and
        r['metric_config']==configuration(spec,device) and r['reuse_pilot_pairs']==pilot_pair_certificate(cpu) and
        r['first_full_PHY_scientific_caps']==cpu['complete_scientific_caps'] and cpu['complete_scientific_caps']['full_metrics']==CAPS and
        r['population']==evaluation.POPULATION and r['compute_provider_scope']==evaluation.PROVIDER_SCOPE and
        r['provider_reused_unchanged'] is True,'Frozen full1000 metric-only registration changed')
    runner.check_tools(r['tool_bindings']);metric.bound_assets(r['metric_config'])
    g.require(r['PHY_calls']==r['source_calls']==r['VAR_calls']==r['decoder_calls']==r['training_updates']==r['new100_reads']==0 and
        r['automatic_successor'] is False and r['final_strategy_frozen'] is False,'Metric-only finite scope changed')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh full1000 metric registration required')
    pins=dict(completion=dict(path=str(g.inside(a.visual_completion)),sha256=a.visual_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.visual_owner_wait)),sha256=a.visual_owner_wait_sha256))
    _,worker,cpu,prior=visual_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);metric_environment(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=gate.source.host.DeviceAdapter(dp,a.target_index)
    config=configuration(spec,device);metric.bound_assets(config);certificate=pilot_pair_certificate(cpu);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,full_visual_closed=pins,
        records=cpu['records'],logical_events=cpu['frames'],finalists=cpu['finalists'],policy=cpu['policy'],
        reconstruction_results=worker['results']['logical_frames'],metric_config=config,reuse_pilot_pairs=certificate,
        first_full_PHY_scientific_caps=cpu['complete_scientific_caps'],population=evaluation.POPULATION,
        compute_provider_scope=evaluation.PROVIDER_SCOPE,provider_reused_unchanged=True,
        device_receipt=dp,target_index=a.target_index,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,training_updates=0,new100_reads=0,automatic_successor=False,final_strategy_frozen=False,
        purpose='Complete original1000 three-noise four-metric calibration; unchanged DINO-L mean winner rule')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def scores(r,evaluator,ledger,boundary,out):
    return evaluation.scores(r,evaluator,ledger,boundary,out,g,gate.readpin)


def engine():
    g.require(g.sha(visual.__file__)==VISUAL_SHA and g.sha(metric.__file__)==METRIC_SHA and
        g.sha(pilot_owner.__file__)==PILOT_OWNER_SHA,'Frozen full visual/metric providers changed')
    ns=dict(vars(visual.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)

def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual metric owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Metric stage already claimed; no replay');out.mkdir()
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
            rows=gate.readpin(result['metric_rows']);finalists=gate.readpin(result['full_winners']);policy=gate.readpin(r['metric_config']['whole_policy'])
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and counts['unresolved']==0 and
                counts['completed']==counts['reserved'] and finalists==evaluation.winners(rows,r['records'],r['finalists'],policy),'Actual metric closure differs')
            registration(path,a.request_sha256,runner);metric_environment(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Metric owner final deadline exceeded')
            completion=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False,final_strategy_frozen=True,new100_admitted=False,population=evaluation.POPULATION)
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
    device=runner.device_from_request(r);device.check_cuda_environment();launch=runner.worker_identity(out,a.request_sha256,a.owner_pid)
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
        g.write(out/'metric_identity.json',dict(identity=evaluator.identity,metadata=evaluator.metadata,actual_runtime=observed,population=evaluation.POPULATION,compute_provider_scope=evaluation.PROVIDER_SCOPE,provider_reused_unchanged=True))
        loaded=module.native_linkage(environment,g,previous=before);g.write(out/'native_after_models.json',loaded)
        results=scores(r,evaluator,ledger,boundary,out/'scores')
        torch.cuda.synchronize();g.write(out/'native_final.json',module.native_linkage(environment,g,previous=loaded))
        metric.bound_assets(r['metric_config']);resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            metric_identity=core.descriptor(out/'metric_identity.json'),PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,
            automatic_successor=False,final_strategy_frozen=True,new100_admitted=False,population=evaluation.POPULATION,compute_provider_scope=evaluation.PROVIDER_SCOPE,provider_reused_unchanged=True)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('visual-completion','visual-completion-sha256','visual-owner-wait','visual-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(0,2),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
