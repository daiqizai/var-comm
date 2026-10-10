"""Finite actual-RX-only pilot reconstruction owner; no PHY or metric models.

Identical actual source inputs reuse the closed48 reconstruction only under the
same device, model, native and numerical bindings, or a same-run reconstruction.
Old call counts remain in their original gates; new calls have a separate ledger.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import types
import numpy as np
import h800_ep_pilot_phy_v1 as physical

source=physical.source;gate=source.gate;g=source.g;core=source.core
SCHEMA='H800_EP_ORIGINAL100_PILOT_VISUAL_V1'
PASS='PASS_H800_ORIGINAL100_PILOT_RECONSTRUCTIONS_ONLY'
CAPS=core.VISUAL_CAPS
PHY_SCRIPT_SHA='61683057d883ae6ecd317d7ab998930e9e367fade54ca3ef97d791431f90aaed'
TOOLS=physical.TOOLS+('h800_ep_pilot_phy_v1.py',)


def execution(deadline,seconds):
    g.require(type(seconds) is int and 0<seconds<=21600 and type(deadline) in (int,float) and time.time()<deadline,
        'Independent finite pilot GPU window of at most21600seconds required')
    return dict(deadline_unix=deadline,max_seconds=seconds,GPU_workers=1,CPU_affinity_count=2,allocator_cap_bytes=16*(1<<30))


def cpu_closure(pins):
    root=gate.RT/'qualification/h800_ep_pilot_phy_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact closed pilot CPU attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==physical.PASS and done['actual_wait']['success'] is True and done['actual_children_waited'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,'Actual pilot CPU owner wait required')
    request=g.checked_json(root/'registered/request.json',done['request_sha256']);worker=gate.readpin(done['worker_completion'])
    # Historical CPU windows need not remain live after that owner has closed.
    # Its exact input/source identities are rechecked without admitting another PHY call.
    records,streams,source_request,prior=physical.source_closure(request['source68_closed'])
    identity,_=physical.phy_binding();policy,pp,static=gate.policy_and_static()
    g.require(request['schema']==physical.SCHEMA and request['records']==records and request['streams']==streams and
        request['phy_identity']==identity and request['policy']==pp and request['complete_scientific_caps']==physical.scientific_caps() and
        request['frames']==core.frames(records,policy),'Closed CPU pilot input contract changed')
    for name,digest in request['tool_bindings'].items():g.require(g.sha(Path(__file__).with_name(name))==digest,'CPU pilot source changed: '+name)
    result=worker['results'];snapshot=result['packet_ledger']
    g.require(worker['status']==physical.PASS and worker['request_sha256']==done['request_sha256'] and worker['CUDA_initialized'] is False and
        len(result['logical_frames'])==43200 and snapshot['unresolved']==0 and snapshot['total']<=86400,
        'Complete finite pilot actual receive outputs required')
    return request,worker,source_request,prior,static


def visual_identity(device_binding):
    return dict(device_identity=device_binding,native_binding=g.native_pin(),model_state_identity=dict(g.MODELS),
        numeric_settings=dict(runtime=dict(g.EXPECTED_RUNTIME),FP32=True,TF32=False,deterministic=True,precision='highest',
            cudnn_benchmark=False,cudnn_deterministic=True,batch_size=1),
        source_math_SHA=source.gate.source.SOURCE_PINS['leo_whole_math_v1.py'],
        source_codec_SHA=source.gate.source.SOURCE_PINS['ep_source_codec.py'],
        received_render_core_SHA='4ee75147eec3e1008045bc7cfd30eb0cb0ff96d0468097384251a48870deabb4')


def reusable48(identity):
    root=source.LINK_ROOT;request=g.checked_json(root/'registered/request.json',source.LINK_PINS['registered/request.json'])
    # Device mismatch is a cache miss, not a cross-device numerical claim.
    if identity!=visual_identity(request['device_identity_binding']):return []
    done=g.checked_json(root/'registered/gpu/completion.json',source.LINK_PINS['registered/gpu/completion.json'])
    worker=gate.readpin(done['worker_completion']);gate.core.close_counts(worker['counts'])
    g.require(worker['status']==gate.PASS and len(worker['rendered_frames'])==48 and
        request['actual_host_native_binding']==identity['native_binding'],'Closed48 visual proof changed')
    return worker['rendered_frames']


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    _,worker,_,prior,static=cpu_closure(r['pilot_CPU_closed']);identity=visual_identity(runner.device_from_request(r).device_identity_binding())
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and r['execution']==e and
        r['physical_frames']==worker['results']['logical_frames'] and r['visual_identity']==identity and r['static']==static and
        r['reuse48_images']==reusable48(identity),'Frozen actual-RX-only visual registration changed')
    g.require(r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Historical visual input closure changed')
    runner.check_tools(r['tool_bindings'])
    g.require(r['source_TX_calls']==r['PHY_calls']==r['metric_model_calls']==0 and r['automatic_successor'] is False,
        'Only actual RX and frozen reconstruction are admitted')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh pilot visual registration required')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    _,worker,_,prior,static=cpu_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=gate.source.host.DeviceAdapter(dp,a.target_index)
    identity=visual_identity(device.device_identity_binding());out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,pilot_CPU_closed=pins,
        physical_frames=worker['results']['logical_frames'],static=static,visual_identity=identity,reuse48_images=reusable48(identity),
        device_receipt=dp,target_index=a.target_index,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        source_TX_calls=0,PHY_calls=0,metric_model_calls=0,automatic_successor=False,
        purpose='Original100 actual accepted RX source decoding and frozen VAR completion; exact same-identity image reuse')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def images(r,backend,codec,partial,ledger,boundary,out):
    out=Path(out);out.mkdir();(out/'images').mkdir();(out/'logical').mkdir();cache={};physical_cache={};pins=[]
    reuse=dict(closed48=0,same_run=0,new_actual_source_inputs=0)
    for pin in r['reuse48_images']:
        result=gate.readpin(pin);packet=gate.readpin(result['physical_frame'])
        image_pin=result['image_archive'];g.require(g.sha(g.inside(image_pin['path']))==image_pin['sha256'],'Closed48 image archive changed')
        with np.load(image_pin['path'],allow_pickle=False) as archive:image=archive['image'].copy()
        g.require(image.dtype==np.float32 and image.shape==(3,256,256) and g.image_sha(image)==result['image_sha256'],
            'Closed48 reconstruction bytes changed')
        key=core.digest(core.source_input(packet['actual_RX'],r['visual_identity']))
        entry=dict(image_archive=image_pin,image_sha256=result['image_sha256'],actual_source_input_key=key,evidence=result['evidence'],
            first_physical_frame=result['physical_frame'],origin='CLOSED48',closed48_result=pin)
        if key in cache:g.require(cache[key]['image_sha256']==entry['image_sha256'],'Same actual source input gave inconsistent old images')
        cache[key]=entry
    for index,pin in enumerate(r['physical_frames']):
        boundary();row=gate.readpin(pin);g.require(row['frame_index']==index,'Pilot logical order changed')
        pp=row['physical_frame'];pk=core.canonical(pp)
        if pk not in physical_cache:physical_cache[pk]=gate.readpin(pp)
        packet=physical_cache[pk];rx=packet['actual_RX'];key=core.digest(core.source_input(rx,r['visual_identity']))
        if key not in cache:
            backend.source_index=row['logical_event']['source_index']
            image,evidence=core.link.recover_and_render(rx,codec,backend,partial,ledger,boundary)
            ip=out/'images'/f"{reuse['new_actual_source_inputs']:05d}.npz"
            with ip.open('xb') as stream:np.savez(stream,image=image)
            cache[key]=dict(image_archive=core.descriptor(ip),image_sha256=g.image_sha(image),actual_source_input_key=key,
                evidence=evidence,first_physical_frame=pp,origin='THIS_PILOT')
            reuse['new_actual_source_inputs']+=1;origin='ACTUAL_NEW'
        else:
            origin=cache[key]['origin'];reuse['closed48' if origin=='CLOSED48' else 'same_run']+=1
        result=dict(frame_index=index,logical_event=row['logical_event'],physical_frame=pp,reconstruction=cache[key],reuse=origin,
            actual_RX_status=rx['status'],actual_received_profile=rx['rx_profile'],actual_header_ok=rx['header']['header_ok'],
            actual_crc_accepted=None if rx['body'] is None else rx['body']['crc_accepted'],
            actual_parser_accepted=None if rx['body'] is None else rx['body']['parser_accepted'],
            reconstruction_evidence_describes_first_cached_decode=True)
        path=out/'logical'/f'{index:05d}.json';g.write(path,result);pins.append(core.descriptor(path))
    counts=ledger.summary();g.require(len(pins)==43200 and sum(reuse.values())==43200 and counts['unresolved']==0 and
        counts['completed']==counts['reserved'] and all(counts['completed'][k]<=cap for k,cap in CAPS.items()),'Pilot reconstruction ledger incomplete')
    g.require(counts['completed']['model_load']==1 and counts['completed']['decoder_forward']==counts['completed']['var_render'],
        'Exactly one model construction and one frozen Dc per new render required')
    return dict(logical_frames=pins,reuse=reuse,counts=counts,old48_counts_preserved=True)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual visual owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Pilot visual owner already claimed; no replay');out.mkdir()
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
                counts['reserved']==counts['completed'] and len(done['results']['logical_frames'])==43200,'Actual visual completion differs')
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
    device=runner.device_from_request(r);device.check_cuda_environment();runner.worker_identity(out,a.request_sha256,a.owner_pid)
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
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_pilot_visual_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(spec['project_root']);core.link.bind_static(codec,r['static'])
        driver=resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py')
        core.link.bind_independent_provider(codec,backend,module,driver,g)
        results=images(r,backend,codec,partial,ledger,boundary,out/'reconstructions')
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            model_state_identity=dict(g.MODELS),source_truth_used=False,TX_CDF_used=False,quality_scores=0,automatic_successor=False)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise


def engine():
    g.require(g.sha(physical.__file__)==PHY_SCRIPT_SHA,'Frozen pilot PHY owner changed')
    ns=dict(vars(source.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
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
