"""Finite full1000 actual-RX reconstruction owner. Explicit prepare/run only."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import types
import h800_shared_gpu_resource_v2 as resources
import h800_ep_full_phy_v1 as physical
import h800_ep_pilot_visual_v1 as pilot_visual
import h800_ep_full_evaluation_core_v1 as evaluation
source=physical.source;gate=source.gate;g=source.g;core=source.core;full=physical.full
SCHEMA='H800_EP_ORIGINAL1000_FULL_VISUAL_V2'
PASS='PASS_H800_ORIGINAL1000_FULL_RECONSTRUCTIONS_ONLY'
CAPS=evaluation.VISUAL_CAPS
TOOLS=tuple(dict.fromkeys(physical.TOOLS+('h800_ep_full_phy_v1.py','h800_ep_full_evaluation_core_v1.py','h800_ep_full_visual_v1.py','h800_shared_gpu_resource_v2.py')))
execution=pilot_visual.execution
visual_identity=pilot_visual.visual_identity
PHY_SCRIPT_SHA='51217148ec4da7aecb171eb38378380c9f1e33edca31a692f743ba3dbf09a3f0'
EVALUATION_SHA='6ca973dcf0e1fd3fb5ebfd405e913c1e16a6fd4b80e24740b667baed6d6f4f7e'

def cpu_closure(pins):
    root=gate.RT/'qualification/h800_ep_full_phy_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact actual full1000 CPU attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait']);worker=gate.readpin(done['worker_completion'])
    g.require(done['schema']==worker['schema']==physical.SCHEMA and done['status']==worker['status']==physical.PASS and
        done['actual_wait']['success'] is True and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0 and worker['request_sha256']==done['request_sha256'],
        'Actual successful full1000 PHY owner and worker wait required')
    r=g.checked_json(root/'registered/request.json',done['request_sha256'])
    records,streams,sr,prior=physical.source_closure(r['source900_closed'])
    identity,_=physical.physical.phy_binding();policy,pp,static=gate.policy_and_static()
    finalists,materials=physical.metric_closure(r['pilot_metrics_closed'],policy)
    g.require(r['schema']==physical.SCHEMA and r['records']==records and r['streams']==streams and r['policy']==pp and
        r['phy_identity']==identity and r['finalists']==finalists and r['metric_materials']==materials and
        r['frames']==full.frames(records,finalists,policy) and r['complete_scientific_caps']==physical.scientific_caps() and
        r['complete_scientific_caps']['full_visual']==CAPS and r['complete_scientific_caps']['full_metrics']==evaluation.METRIC_CAPS,
        'Closed full PHY original population, finalists or first-request budgets changed')
    for name,digest in r['tool_bindings'].items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Closed full PHY source changed: '+name)
    result=worker['results'];counts=result['packet_ledger']
    g.require(worker['CUDA_initialized'] is False and len(result['logical_frames'])==len(r['frames'])<=36000 and
        counts['unresolved']==0 and counts['total']<=72000,'Complete finite actual full PHY outputs required')
    return r,worker,sr,prior,static


def reusable_pilot(cpu,identity):
    pins=cpu['metric_materials']['pilot_visual_closed']
    r,w,_,_=physical.metric_owner.visual_closure(pins)
    if r['visual_identity']!=identity:return []
    # Full actual pilot closure was validated; this is only a same-identity cache.
    return w['results']['logical_frames']


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);resources.validate_policy(r);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    cpu,worker,_,prior,static=cpu_closure(r['full_CPU_closed']);identity=visual_identity(runner.device_from_request(r).device_identity_binding())
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['population']==evaluation.POPULATION and
        r['caps']==CAPS and r['execution']==e and r['physical_frames']==worker['results']['logical_frames'] and
        r['visual_identity']==identity and r['static']==static and r['reuse_pilot_images']==reusable_pilot(cpu,identity) and
        r['records']==cpu['records'] and r['logical_events']==cpu['frames'] and r['finalists']==cpu['finalists'] and
        r['policy']==cpu['policy'] and r['first_full_PHY_scientific_caps']==cpu['complete_scientific_caps'],
        'Frozen full1000 actual-RX registration changed')
    g.require(r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Historical full visual input closure changed')
    runner.check_tools(r['tool_bindings'])
    g.require(r['source_TX_calls']==r['PHY_calls']==r['metric_model_calls']==r['new100_reads']==0 and
        r['automatic_successor'] is False,'Only finite actual RX and frozen reconstruction admitted')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh full1000 visual registration required')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    cpu,worker,_,prior,static=cpu_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,a.target_index)
    identity=visual_identity(device.device_identity_binding());reuse=reusable_pilot(cpu,identity);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',population=evaluation.POPULATION,caps=CAPS,execution=e,spec=spec,
        full_CPU_closed=pins,physical_frames=worker['results']['logical_frames'],static=static,visual_identity=identity,reuse_pilot_images=reuse,
        records=cpu['records'],logical_events=cpu['frames'],finalists=cpu['finalists'],policy=cpu['policy'],
        first_full_PHY_scientific_caps=cpu['complete_scientific_caps'],resource_policy=dict(resources.POLICY),
        resource_guard=core.descriptor(Path(resources.__file__)),device_receipt=dp,target_index=a.target_index,
        device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        source_TX_calls=0,PHY_calls=0,metric_model_calls=0,new100_reads=0,automatic_successor=False,
        purpose='Complete original1000 actual receive decoding; identical closed pilot images may be reused')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def images(r,backend,codec,partial,ledger,boundary,out):
    return evaluation.images(r,backend,codec,partial,ledger,boundary,out,g,gate.readpin)


def engine():
    g.require(g.sha(physical.__file__)==PHY_SCRIPT_SHA and g.sha(evaluation.__file__)==EVALUATION_SHA,'Frozen full PHY/evaluation code changed')
    ns=dict(vars(source.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS,DeviceAdapter=resources.DeviceAdapter)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)

def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual visual owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Pilot visual owner already claimed; no replay');out.mkdir();device.bind_resource_evidence(out,a.request_sha256)
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,execution=e))
        try:
            _,_,_,_,extra=g.validate_spec(spec);snapshot,ad=runner.wait_prelaunch(device,shared,out,e,started,120)
            g.write(out/'prelaunch_resources.json',snapshot);g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
            _,env=shared.controlled_environment(g.BASE,out,6);(out/'home').mkdir()
            env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=extra['environment']['LD_LIBRARY_PATH'],HOME=str(out/'home'),
                LANG='C.UTF-8',LC_ALL='C.UTF-8',VIRTUAL_ENV=str(Path(spec['python']).parent.parent),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                **device.cuda_environment_binding());g.write(out/'controlled_environment.json',env)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Deadline expired before visual launch')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,
                    cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Visual child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'));counts=done['results']['counts']
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and counts['unresolved']==0 and
                counts['reserved']==counts['completed'] and len(done['results']['logical_frames'])==len(r['logical_events']),'Actual visual completion differs')
            registration(path,a.request_sha256,runner);g.validate_spec(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Owner final deadline exceeded')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False)
            g.write(out/'completion.json',result);return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise

def worker(a,runner):
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    device=runner.device_from_request(r);device.bind_resource_evidence(out,a.request_sha256);device.check_cuda_environment();runner.worker_identity(out,a.request_sha256,a.owner_pid)
    started=time.monotonic();stopped=False;ledger=None;backend=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'Pilot visual stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Pilot visual deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=10:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'progress.jsonl').open('a') as stream:stream.write(g.json.dumps(dict(resources=snapshot,
                counts=None if ledger is None else ledger.summary(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_full_visual_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(spec['project_root']);core.link.bind_static(codec,r['static'])
        driver=resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py')
        core.link.bind_independent_provider(codec,backend,module,driver,g)
        results=images(r,backend,codec,partial,ledger,boundary,out/'reconstructions')
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            model_state_identity=dict(g.MODELS),source_truth_used=False,TX_CDF_used=False,quality_scores=0,automatic_successor=False,population=evaluation.POPULATION)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
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
