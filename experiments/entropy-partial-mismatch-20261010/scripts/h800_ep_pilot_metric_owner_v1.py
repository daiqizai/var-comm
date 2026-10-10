"""Finite four-metric scoring of the actually closed original100 pilot images.

No PHY, source coding, VAR or decoder construction. New metric-native facts are
bound to an actual CUDA-hidden import receipt; original PFS rows stay unchanged.
Only a complete 43200-cell pilot may produce a shortlist, never a final policy.
"""
from __future__ import annotations
import argparse
import copy
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
import h800_ep_pilot_visual_v1 as visual
import h800_ep_pilot_metrics_v1 as metric

source=visual.source;gate=visual.gate;g=visual.g;core=visual.core
SCHEMA='H800_EP_ORIGINAL100_PILOT_METRICS_V1'
PASS='PASS_H800_ORIGINAL100_PILOT_FOUR_METRICS'
CAPS=metric.CAPS
VISUAL_SHA='d790fb8787ceea4f8c200eb9edaacd226b1903a1a5293513adb6f4c1abb46f7f'
METRIC_SHA='6bde19df620a7dbf391f5d406daa3b4c1de2a66d051c990c26ae9a2474c1ec8d'
NATIVE_PATH=gate.RT/'qualification/h800_metric_native_import_v1/binding.json'
NATIVE_SHA='cf4c52b98bbccdd00b50c0539d4da0cb94f7315a8ab905ae20183e7b48904bd3'
CODE_ROOT=gate.RT/'assets/pilot_metric_code_seed_v1'
TOOLS=visual.TOOLS+('h800_ep_pilot_visual_v1.py','h800_ep_pilot_metrics_v1.py')
execution=visual.execution


def native_binding():
    pin=dict(path=str(NATIVE_PATH),sha256=NATIVE_SHA);doc=gate.readpin(pin)
    g.require(doc['schema']=='H800_EP_METRIC_NATIVE_BINDING_V1' and doc['cuda_initialized'] is False and
        doc['numerical_qualification'] is False,'Actual CPU-hidden metric import observation required')
    waited=gate.readpin(doc['actual_child_wait_pin']);observation=gate.readpin(doc['observation'])
    g.require(waited==doc['actual_child_wait'] and observation['schema']=='H800_METRIC_NATIVE_IMPORT_OBSERVATION_V1' and
        observation['external_non_driver_libraries']==doc['files'] and observation['cuda_initialized'] is False and
        observation['dino_implementation_receipt']['sha256']==metric.DINO_RECEIPT_SHA and
        observation['runtime_projection']['sha256']==gate.source.host.PROJECTION_SHA,
        'Actual metric import wait/observation closure changed')
    maps=observation['maps'];g.require(g.sha(g.inside(maps['path']))==maps['sha256'],'Actual loaded-library maps changed')
    projected=copy.deepcopy(doc);projected['schema']='H800_NEW_HOST_NATIVE_BINDING_V1'
    # Only the receipt schema differs. The existing strict external-path, SHA,
    # zero-model scope and actual-wait validator applies unchanged.
    rows=gate.source.host.validate_native_document(projected)
    return pin,rows


def metric_environment(spec,verify_environment=True):
    result=g.validate_spec(spec,verify_environment=verify_environment);environment=result[1]
    pin,rows=native_binding();original_rows=copy.deepcopy(environment.rows)
    existing={r['path']:r for r in environment.host_native}
    for row in rows:
        g.require(row['path'] not in existing or existing[row['path']]==row,'Conflicting actual native import facts')
        existing[row['path']]=row
    environment.host_native=list(existing.values())
    g.require(environment.rows==original_rows,'Original PFS runtime map must stay exact')
    if verify_environment:
        for row in rows:
            p=Path(row['path']);g.require(p.stat().st_size==row['bytes'] and g.sha(p)==row['sha256'],'Metric native bytes changed')
    return result


def runtime_identity(spec,device):
    pin,_=native_binding()
    return dict(original_runtime_projection=spec['environment'],base_host_native=g.native_pin(),
        metric_native_import=pin,device_identity=device.device_identity_binding(),
        expected_runtime=dict(g.EXPECTED_RUNTIME),numeric_flags=dict(metric.FLAGS),
        old_host_numeric_identity_claimed=False,overall_FullSuite_identity_claimed=False)


def configuration(spec,device):
    incoming=gate.RT/'incoming/pilot_metric_weights_seed_v1';_,policy,_=gate.policy_and_static()
    answer=dict(sources={n:dict(path=str(CODE_ROOT/n),sha256=h) for n,h in metric.SOURCE_SHA.items()},
        manifest=dict(path=str(CODE_ROOT/'modelmanifest.json'),sha256=metric.MANIFEST_SHA),
        registration=dict(path=str(CODE_ROOT/'metrics_registration.json'),sha256=metric.REGISTRATION_SHA),
        weight_receipt=dict(path=str(incoming/'hydration_completion.json'),sha256=metric.WEIGHT_RECEIPT_SHA),
        dino_receipt=dict(path=str(incoming/'dino157_hydration_completion.json'),sha256=metric.DINO_RECEIPT_SHA),
        whole_policy=policy,runtime_identity=runtime_identity(spec,device))
    return answer


def visual_closure(pins):
    root=gate.RT/'qualification/h800_ep_pilot_visual_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and
        pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact pilot visual attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==visual.PASS and done['actual_wait']['success'] is True and done['actual_children_waited'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,
        'Actual visual owner must close successfully before metric admission')
    request=g.checked_json(root/'registered/request.json',done['request_sha256']);worker=gate.readpin(done['worker_completion'])
    cpu,physical,_,prior,static=visual.cpu_closure(request['pilot_CPU_closed'])
    device=gate.source.host.DeviceAdapter(request['device_receipt'],request['target_index'])
    g.require(request['schema']==visual.SCHEMA and request['caps']==visual.CAPS and request['static']==static and
        request['physical_frames']==physical['results']['logical_frames'] and
        request['visual_identity']==visual.visual_identity(device.device_identity_binding()),'Closed visual input identity changed')
    for name,digest in request['tool_bindings'].items():
        g.require(g.sha(Path(__file__).with_name(name))==digest,'Closed visual source changed: '+name)
    counts=worker['results']['counts'];frames=worker['results']['logical_frames']
    g.require(worker['status']==visual.PASS and worker['request_sha256']==done['request_sha256'] and len(frames)==43200 and
        counts['caps']==visual.CAPS and counts['unresolved']==0 and counts['completed']==counts['reserved'] and
        all(counts['completed'][k]<=v for k,v in visual.CAPS.items()) and counts['completed']['model_load']==1 and
        counts['completed']['decoder_forward']==counts['completed']['var_render'] and worker['quality_scores']==0,
        'Complete actual reconstruction closure required')
    return request,worker,cpu,prior


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    _,worker,cpu,prior=visual_closure(r['pilot_visual_closed']);device=runner.device_from_request(r)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['spec']==spec and r['records']==cpu['records'] and r['reconstruction_results']==worker['results']['logical_frames'] and
        r['logical_events']==cpu['frames'] and r['metric_config']==configuration(spec,device), 'Frozen metric registration changed')
    runner.check_tools(r['tool_bindings']);metric.bound_assets(r['metric_config'])
    g.require(r['PHY_calls']==r['source_calls']==r['VAR_calls']==r['decoder_calls']==r['training_updates']==0 and
        r['automatic_successor'] is False and r['final_strategy_frozen'] is False,'Metric-only finite scope changed')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh metric registration required')
    pins=dict(completion=dict(path=str(g.inside(a.visual_completion)),sha256=a.visual_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.visual_owner_wait)),sha256=a.visual_owner_wait_sha256))
    _,worker,cpu,prior=visual_closure(pins);e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);metric_environment(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=gate.source.host.DeviceAdapter(dp,a.target_index)
    config=configuration(spec,device);metric.bound_assets(config);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,pilot_visual_closed=pins,
        records=cpu['records'],logical_events=cpu['frames'],reconstruction_results=worker['results']['logical_frames'],metric_config=config,
        device_receipt=dp,target_index=a.target_index,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,training_updates=0,automatic_successor=False,final_strategy_frozen=False,
        purpose='Original100 four-metric pilot; DINO-L top3 union unchanged full-calibration whole winner')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def scores(r,evaluator,ledger,boundary,out):
    out=Path(out);out.mkdir();(out/'pairs').mkdir();rows=[];cache={};references={};verified={};reused=0
    records=r['records'];g.require(len(records)==100 and [x['source_index'] for x in records]==list(range(100)),
        'Exactly original100 reference records required')
    g.require(len(r['logical_events'])==len(r['reconstruction_results'])==43200,'Complete actual reconstruction list required')
    for i,pin in enumerate(r['reconstruction_results']):
        boundary();result=gate.readpin(pin);event=result['logical_event'];expected=r['logical_events'][i]
        g.require(result['frame_index']==i and event==expected,'Actual reconstruction logical identity changed')
        index=event['source_index'];record=records[index]
        g.require(record['source_id']==event['source_id'],'Reference source identity differs')
        if index not in references:
            target=core.reference_pixels(record);references[index]=(target,evaluator.prepare(target),metric.array_sha(target))
        target,prepared,reference_sha=references[index];recon=result['reconstruction'];archive=recon['image_archive']
        key=core.digest(dict(reference_sha256=reference_sha,reconstruction_sha256=recon['image_sha256'],metric_identity=evaluator.identity))
        archive_key=core.canonical(archive)
        if archive_key not in verified:
            g.require(g.sha(g.inside(archive['path']))==archive['sha256'],'Actual reconstruction archive changed')
            with np.load(archive['path'],allow_pickle=False) as z:image=z['image'].copy()
            g.require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
                (image>=0).all() and (image<=1).all() and metric.array_sha(image)==recon['image_sha256'],
                'Actual reconstruction pixels changed')
            verified[archive_key]=recon['image_sha256']
        else:
            g.require(verified[archive_key]==recon['image_sha256'],'Same archive has conflicting pixel identity')
            image=None
        if key not in cache:
            if image is None:
                with np.load(archive['path'],allow_pickle=False) as z:image=z['image'].copy()
                g.require(metric.array_sha(image)==recon['image_sha256'],'Scored image bytes changed')
            g.require(metric.pair_cache_key(target,image,evaluator.identity)==key,'Metric pair identity differs')
            values=evaluator.score(target,image,record['evaluation_class_index'],prepared)
            pair=out/'pairs'/f'{len(cache):05d}.json';g.write(pair,dict(metrics=values,metric_pair_key=key,
                reference_archive=record['archive'],reconstruction_archive=archive,metric_identity=evaluator.identity))
            cache[key]=(values,core.descriptor(pair))
        else:reused+=1
        values,pair=cache[key]
        rows.append(dict(source_index=index,source_id=event['source_id'],snr_db=event['snr_db'],noise_seed=event['noise_seed'],
            candidate_id=event['candidate_id'],**values,metric_pair_key=key,metric_pair_result=pair,reconstruction_result=pin,
            actual_RX_status=result['actual_RX_status'],actual_received_profile=result['actual_received_profile'],
            actual_header_ok=result['actual_header_ok'],actual_crc_accepted=result['actual_crc_accepted'],
            actual_parser_accepted=result['actual_parser_accepted']))
        if (i+1)%432==0:
            g.write(out/f'progress_{i+1:05d}.json',dict(logical_frames=i+1,actual_unique_pairs=len(cache),reused_pairs=reused,counts=ledger.summary()))
    policy=gate.readpin(r['metric_config']['whole_policy']);finalists=core.finalists(rows,policy);counts=ledger.summary()
    g.require(counts['unresolved']==0 and counts['completed']==counts['reserved'] and counts['caps']==CAPS and
        counts['completed']['model_constructions']==3 and counts['completed']['reference_preparations']==100 and
        counts['completed']['image_scores']==len(cache) and len(cache)+reused==43200 and
        all(counts['completed'][k]<=v for k,v in CAPS.items()),'Complete finite pair-score ledger required')
    for kind in ('dinov2_vitl14_reference','convnext_reference'):g.require(counts['completed'][kind]==100,'Missing reference features')
    for kind in ('dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):
        g.require(counts['completed'][kind]==len(cache),'Incomplete actual metric pair')
    g.require(counts['completed']['lpips_alexnet_backbone_forward']==2*len(cache),'Actual LPIPS backbone count differs')
    g.write(out/'metric_rows.json',rows);g.write(out/'finalists.json',finalists)
    return dict(metric_rows=core.descriptor(out/'metric_rows.json'),finalists=core.descriptor(out/'finalists.json'),
        logical_frames=43200,actual_unique_pairs=len(cache),reused_pairs=reused,counts=counts,
        old_visual_and_PHY_counts_preserved=True,final_strategy_frozen=False)


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
            rows=gate.readpin(result['metric_rows']);finalists=gate.readpin(result['finalists']);policy=gate.readpin(r['metric_config']['whole_policy'])
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and counts['unresolved']==0 and
                counts['completed']==counts['reserved'] and finalists==core.finalists(rows,policy),'Actual metric closure differs')
            registration(path,a.request_sha256,runner);metric_environment(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Metric owner final deadline exceeded')
            completion=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False,final_strategy_frozen=False)
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
        g.write(out/'metric_identity.json',dict(identity=evaluator.identity,metadata=evaluator.metadata,actual_runtime=observed))
        loaded=module.native_linkage(environment,g,previous=before);g.write(out/'native_after_models.json',loaded)
        results=scores(r,evaluator,ledger,boundary,out/'scores')
        torch.cuda.synchronize();g.write(out/'native_final.json',module.native_linkage(environment,g,previous=loaded))
        metric.bound_assets(r['metric_config']);resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            metric_identity=core.descriptor(out/'metric_identity.json'),PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,
            automatic_successor=False,final_strategy_frozen=False)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise


def engine():
    g.require(g.sha(visual.__file__)==VISUAL_SHA and g.sha(metric.__file__)==METRIC_SHA,'Frozen visual/metric implementation changed')
    ns=dict(vars(visual.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


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
