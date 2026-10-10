"""Finite original100 pilot PHY owner; CUDA hidden, no metric/model execution.

This explicit stage registers the complete downstream finite budget but admits
only PHY. Actual identical observations may reuse completed48 or same-run RX;
no old144-profile event is reused. Separate GPU/metric owners remain required.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import h800_ep_pilot_source_v1 as source

g=source.g
gate=source.gate
core=source.core
SCHEMA='H800_EP_ORIGINAL100_PILOT_PHY_V1'
PASS='PASS_H800_ORIGINAL100_PILOT_PHY_ONLY'
SOURCE_SCRIPT_SHA='661d6f1c8c45b3b324c8c488118ccb4c96c3ea8217281bd4d6bb0ebd014403e2'
TOOLS=source.TOOLS+('h800_ep_pilot_source_v1.py',)
METRIC_CAPS=dict(model_constructions=3,reference_preparations=100,image_scores=43200,
    dinov2_vitl14_reference=100,dinov2_vitl14_reconstruction=43200,
    convnext_reference=100,convnext_reconstruction=43200,lpips_pair=43200,
    lpips_alexnet_backbone_forward=86400)


def execution(deadline,seconds):
    g.require(type(seconds) is int and 0<seconds<=21600 and type(deadline) in (int,float) and time.time()<deadline,
        'Independent finite pilot CPU window of at most21600seconds required')
    return dict(deadline_unix=deadline,max_seconds=seconds,CPU_workers=1,CPU_affinity_count=2,CUDA_visible=False)


def source_closure(pins):
    g.require(set(pins)=={'completion','owner_actual_wait'},'Source68 closed owner pair required')
    expected=gate.RT/'qualification/h800_ep_pilot_source68_v1_attempt1'
    g.require(pins['completion']['path']==str(expected/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(expected/'run_owner_actual_wait.json'),'Exact source68 attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==source.PASS and done['actual_children_waited'] is True and done['actual_wait']['success'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,'Actual successful source68 owner wait required')
    worker=gate.readpin(done['worker_completion'])
    g.require(worker['status']==source.PASS and worker['counts']['completed']==worker['counts']['reserved']==source.CAPS and
        worker['counts']['unresolved']==0 and worker['source_count']==100 and len(worker['new68_streams'])==68,
        'Exact missing68 source budget must actually close')
    request=g.checked_json(g.inside(expected/'registered/request.json'),done['request_sha256'])
    prior,old32,prereq=source.prerequisites();records,restore=source.inputs()
    g.require(request['schema']==source.SCHEMA and request['caps']==source.CAPS and request['prerequisites']==prereq and
        request['records']==records and request['restoration']==restore and request['existing32_streams']==old32 and
        worker['existing32_streams']==old32 and request['actual_host_native_binding']==g.native_pin(),
        'Closed source100 input or host binding changed')
    for name,digest in request['tool_bindings'].items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Source68 tool changed: '+name)
    streams=old32+worker['new68_streams']
    g.require([x['source_index'] for x in streams]==list(range(100)), 'Ordered100 fresh H800 stream records required')
    for item in streams:
        meta=gate.readpin(item['metadata']);g.require(meta['source_index']==item['source_index'] and
            meta['archive']==item['archive'] and g.sha(g.inside(item['archive']['path']))==item['archive']['sha256'],
            'Actual fresh source stream metadata/archive changed')
    return records,streams,request,prior


def phy_binding():
    done=g.checked_json(gate.PHY_ROOT/'completion.json',gate.PHY_SHA)
    environment=g.checked_json(gate.PHY_CONTROL/'environment.json',gate.PHY_ENV_SHA)
    g.require(done['status']=='PASS' and done['packet_decode_count']==457 and done['ledger']==dict(total=457,unresolved=0,cap=457),
        'Actual qualified457 PHY required')
    return dict(profile_count=360,catalogue_sha256=done['catalogue_sha256'],backend_identity=done['backend_identity'],
        qualification=dict(path=str(gate.PHY_ROOT/'completion.json'),sha256=gate.PHY_SHA),
        source_bindings_sha256=core.digest(done['source_bindings']),input_bindings_sha256=core.digest(done['input_bindings']),
        CPU_python=dict(path=str(gate.CPU_PYTHON),sha256=g.sha(gate.CPU_PYTHON)),
        admitted_CPU_environment=dict(path=str(gate.PHY_CONTROL/'environment.json'),sha256=gate.PHY_ENV_SHA),
        actual_host_native_binding=g.native_pin(),CUDA_visible=False),environment


def reuse48(identity):
    root=source.LINK_ROOT
    owner=source.LINK_PINS['registered/gpu/completion.json'];done=g.checked_json(root/'registered/gpu/completion.json',owner)
    gpu=gate.readpin(done['worker_completion']);cpu_owner=gate.readpin(gpu['CPU_completion']);cpu=gate.readpin(cpu_owner['worker_completion'])
    certificate=dict(schema='CLOSED_EP48_EXACT_EVENT_REUSE_V1',phy_identity=identity,
        owner_actual_wait=dict(path=str(root/'gpu_owner_actual_wait.json'),sha256=source.LINK_PINS['gpu_owner_actual_wait.json']),
        owner_completion=dict(path=str(root/'registered/gpu/completion.json'),sha256=owner),
        request=dict(path=str(root/'registered/request.json'),sha256=source.LINK_PINS['registered/request.json']),
        physical_packets=cpu['frames'],already_consumed_counts_preserved=True,
        qualification_basis='Same actual qualified backend, full360 catalogue, bound code/runtime; every new waveform and noisy observation is rehashed before event reuse')
    core.external_packets(certificate,identity)
    return certificate


def scientific_caps():
    return dict(logical_frames=43200,actual_packet_decodes=86400,new_TX68=dict(core.SOURCE_CAPS),
        visual=dict(core.VISUAL_CAPS),metrics=METRIC_CAPS,existing32_TX_reused=32,
        encoder_calls=0,training_updates=0,original_calibration_source_indices=list(range(100)),
        new_confirmation_source_reads=0,noise_seeds=[4101],snrs=[4,10,19],candidates=144)


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    records,streams,_,_=source_closure(r['source68_closed']);identity,environment=phy_binding();policy,pp,_=gate.policy_and_static()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and
        r['complete_scientific_caps']==scientific_caps() and r['records']==records and r['streams']==streams and
        r['phy_identity']==identity and r['reuse48']==reuse48(identity) and r['policy']==pp and
        r['frames']==core.frames(records,policy),'Frozen original100 pilot PHY request changed')
    runner=engine();runner.check_tools(r['tool_bindings'])
    g.require(r['GPU_execution_admitted'] is False and r['metrics_execution_admitted'] is False and
        r['automatic_successor'] is False,'Only finite CPU PHY stage is admitted')
    return r,environment


def prepare(a):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh pilot PHY registration required')
    pins=dict(completion=dict(path=str(g.inside(a.source_completion)),sha256=a.source_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.source_owner_wait)),sha256=a.source_owner_wait_sha256))
    records,streams,_,_=source_closure(pins);identity,_=phy_binding();policy,pp,_=gate.policy_and_static()
    e=execution(a.deadline_unix,a.max_seconds);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=e,complete_scientific_caps=scientific_caps(),
        source68_closed=pins,records=records,streams=streams,phy_identity=identity,reuse48=reuse48(identity),policy=pp,
        frames=core.frames(records,policy),tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        root=str(gate.R2),CPU_python=str(gate.CPU_PYTHON),GPU_execution_admitted=False,metrics_execution_admitted=False,
        automatic_successor=False,preparation_packet_decodes=0,preparation_model_calls=0,
        purpose='Original100 calibration pilot actual PHY only; complete downstream caps predeclared, later owners required')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def run(a):
    g.require(g.sys.platform.startswith('linux'),'Actual pilot PHY owner requires Linux')
    path=g.inside(a.request);r,old_environment=registration(path,a.request_sha256);e=r['execution'];out=path.parent/'run'
    shared=g.helper();child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/'controls/h800_ep_pilot_phy_v1/CPU.owner.lock')):
        g.require(not out.exists(),'Pilot PHY already claimed; no replay');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,execution=e))
        try:
            snapshot=shared.cpu_snapshot()
            g.require(snapshot['effective_cpu_cores']>=2 and len(snapshot['allowed_cpus'])>=2 and
                snapshot['available_memory_bytes']>=2*(1<<30),'Insufficient bounded CPU resources')
            ad=dict(cpu_affinity=snapshot['allowed_cpus'][:2])
            g.write(out/'prelaunch_resources.json',snapshot)
            env=gate.controlled_cpu_environment(out,old_environment);g.write(out/'controlled_environment.json',env)
            argv=[r['CPU_python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Deadline expired before launch')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=r['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,
                    cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Pilot PHY child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and
                len(done['results']['logical_frames'])==43200 and done['results']['packet_ledger']['unresolved']==0 and
                done['results']['packet_ledger']['total']<=86400,'Pilot actual PHY completion not closed')
            registration(path,a.request_sha256)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Pilot owner final deadline exceeded')
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


def worker(a):
    path=g.inside(a.request);r,_=registration(path,a.request_sha256);out=path.parent/'run';e=r['execution'];runner=engine()
    launch=runner.worker_identity(out,a.request_sha256,a.owner_pid)
    g.require(launch['argv'][0]==r['CPU_python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2,
        'Actual CPU child must hide CUDA and use exactly2cores')
    stopped=False;started=time.monotonic();last=0.;ledger=None;boundaries=0
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'Pilot PHY stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Pilot PHY execution deadline exceeded')
    def boundary():
        nonlocal last,boundaries
        guard();boundaries+=1
        if time.monotonic()-last>=10:
            last=time.monotonic()
            with (out/'progress.jsonl').open('a') as stream:stream.write(g.json.dumps(dict(logical_frame_boundaries=boundaries,
                packet_ledger=None if ledger is None else ledger.snapshot(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        import ep_phy as phy
        import ep_source_codec as partial
        from t2_ledger import Ledger
        runtime=phy.create_runtime(r['root'],r['phy_identity']['qualification']['path'])
        g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
        streams={item['source_index']:core.link.load_streams(item['archive'],gate.readpin(item['metadata']),partial) for item in r['streams']}
        ledger=Ledger(out/'packet_ledger.sqlite',a.request_sha256,86400)
        results=core.cpu_frames(runtime,r['frames'],streams,partial,phy,ledger,boundary,out/'frames',r['phy_identity'],r['reuse48'])
        ledger.close();ledger=None;guard();registration(path,a.request_sha256)
        g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=results,
            CUDA_initialized=False,neural_model_calls=0,quality_scores=0,automatic_successor=False)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            request_sha256=a.request_sha256,packet_ledger=None if ledger is None else ledger.snapshot()));raise
    finally:
        if ledger is not None:ledger.close()


def engine():
    g.require(g.sha(source.__file__)==SOURCE_SCRIPT_SHA,'Frozen source68 implementation changed')
    runner=source.engine();ns=dict(vars(runner));ns.update(__file__=__file__,TOOLS=TOOLS)
    for name in ('check_tools','worker_identity'):ns[name]=gate.source.host.clone_function(ns[name],ns)
    return source.types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('source-completion','source-completion-sha256','source-owner-wait','source-owner-wait-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();engine()
    result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
