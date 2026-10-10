"""Explicit finite full1000 CPU PHY owner; prepared separately, never auto-run.

Requires actual complete pilot four-metric finalists and closed1000 fresh H800
TX streams. Only original calibration1000 x SNR4/10/19 x seeds4101/2/3 enters.
The per-SNR pilot top3 union original whole winner has at most4 actions.
No GPU, encoder, model, reconstruction, metric, new100 access or auto successor.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import h800_ep_full_source_v1 as source
import h800_ep_full_phy_core_v1 as full
import h800_ep_pilot_phy_v1 as physical
import h800_ep_pilot_visual_v1 as visual
import h800_ep_pilot_metric_owner_v1 as metric_owner

g=source.g
gate=source.gate
core=source.core
SCHEMA=full.SCHEMA
PASS='PASS_H800_ORIGINAL1000_FULL_PHY_ONLY'
SOURCE_SCRIPT_SHA='0f565e6862aa0bda4f165d8a7a578e69458c298853ddad7581398e60ac625d3b'
FULL_CORE_SHA='9891079955cfb2988a05e5a7e7f5ea9bcf2eb653ec47817afd10feb343c99866'
METRIC_OWNER_SHA='8f5b04978b23d8b4525b03599d13a9aa1277872481624fa17c5a0627f43b60e1'
TOOLS=tuple(dict.fromkeys(source.TOOLS+('h800_ep_full_source_v1.py','h800_ep_full_phy_core_v1.py',
    'h800_ep_pilot_phy_v1.py','h800_ep_pilot_visual_v1.py','h800_ep_pilot_metrics_v1.py',
    'h800_ep_pilot_metric_owner_v1.py')))
METRIC_SCHEMA='H800_EP_ORIGINAL100_PILOT_METRICS_V1'
METRIC_PASS='PASS_H800_ORIGINAL100_PILOT_FOUR_METRICS'

def execution(deadline,seconds):
    g.require(type(seconds) is int and 0<seconds<=21600 and type(deadline) in (int,float) and time.time()<deadline,
        'Independent finite full1000 CPU window of at most21600seconds required')
    return dict(deadline_unix=deadline,max_seconds=seconds,CPU_workers=1,CPU_affinity_count=2,CUDA_visible=False)

def source_closure(pins):
    g.require(set(pins)=={'completion','owner_actual_wait'},'Source900 closed owner pair required')
    expected=gate.RT/'qualification/h800_ep_full_source900_v1_attempt1'
    g.require(pins['completion']['path']==str(expected/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(expected/'run_owner_actual_wait.json'),'Exact full_source900 attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==source.PASS and done['actual_children_waited'] is True and done['actual_wait']['success'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,'Actual successful full_source900 owner wait required')
    worker=gate.readpin(done['worker_completion'])
    g.require(done['schema']==worker['schema']==source.SCHEMA and worker['request_sha256']==done['request_sha256'] and
        worker['status']==source.PASS and worker['counts']['completed']==worker['counts']['reserved']==source.CAPS and
        worker['counts']['unresolved']==0 and worker['source_count']==1000 and len(worker['new900_streams'])==900,
        'Exact missing900 source budget must actually close')
    request=g.checked_json(g.inside(expected/'registered/request.json'),done['request_sha256'])
    prior,old100,prereq=source.prerequisites();records,restore=source.inputs()
    g.require(request['schema']==source.SCHEMA and request['caps']==source.CAPS and request['prerequisites']==prereq and
        request['records']==records and request['restoration']==restore and request['existing100_streams']==old100 and
        worker['existing100_streams']==old100 and request['actual_host_native_binding']==g.native_pin(),
        'Closed source1000 input or host binding changed')
    for name,digest in request['tool_bindings'].items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Source900 tool changed: '+name)
    streams=old100+worker['new900_streams']
    g.require([x['source_index'] for x in streams]==list(range(1000)), 'Ordered1000 fresh H800 stream records required')
    for item in streams:
        meta=gate.readpin(item['metadata']);g.require(meta['source_index']==item['source_index'] and
            item['source_id']==records[item['source_index']]['source_id'] and
            meta['archive']==item['archive'] and g.sha(g.inside(item['archive']['path']))==item['archive']['sha256'],
            'Actual fresh source stream metadata/archive changed')
    return records,streams,request,prior

def metric_closure(pins, policy):
    root=gate.RT/'qualification/h800_ep_pilot_metrics_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and
        pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact closed pilot metrics attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait']);worker=gate.readpin(done['worker_completion'])
    g.require(done['schema']==worker['schema']==METRIC_SCHEMA and done['status']==worker['status']==METRIC_PASS and
        done['actual_children_waited'] is True and done['actual_wait']['success'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0 and worker['request_sha256']==done['request_sha256'],
        'Actual complete four-metric owner wait required before full calibration')
    request=g.checked_json(g.inside(root/'registered/request.json'),done['request_sha256'])
    g.require(request['schema']==METRIC_SCHEMA,'Registered complete pilot metric owner required')
    for name,digest in request['tool_bindings'].items():
        g.require(g.sha(Path(__file__).with_name(name))==digest,'Closed metric tool changed: '+name)
    vp=request['pilot_visual_closed'];vr=gate.RT/'qualification/h800_ep_pilot_visual_v1_attempt1'
    validated_visual,validated_worker,validated_cpu,_=metric_owner.visual_closure(vp)
    g.require(set(vp)=={'completion','owner_actual_wait'} and vp['completion']['path']==str(vr/'registered/run/completion.json') and
        vp['owner_actual_wait']['path']==str(vr/'run_owner_actual_wait.json'),'Exact actual pilot reconstruction origin required')
    vd=gate.readpin(vp['completion']);vw=gate.readpin(vp['owner_actual_wait']);vworker=gate.readpin(vd['worker_completion'])
    g.require(vd['status']==vworker['status']==visual.PASS and vd['actual_wait']['success'] is True and
        vd['actual_children_waited'] is True and vd['worker_exit_codes']==[0] and vw['actual_wait'] is True and vw['returncode']==0 and
        vworker['request_sha256']==vd['request_sha256'] and len(vworker['results']['logical_frames'])==43200 and
        vworker['results']['counts']['unresolved']==0 and vworker['results']['counts']['reserved']==vworker['results']['counts']['completed'],
        'All43200 actual pilot reconstructions must be closed')
    vrequest=g.checked_json(g.inside(vr/'registered/request.json'),vd['request_sha256'])
    cpu_request,cpu_worker,_,_,_=visual.cpu_closure(vrequest['pilot_CPU_closed'])
    g.require(vrequest==validated_visual and vworker==validated_worker and cpu_request==validated_cpu and
        vrequest['physical_frames']==cpu_worker['results']['logical_frames'] and request['logical_events']==cpu_request['frames'] and
        request['records']==cpu_request['records'] and request['reconstruction_results']==vworker['results']['logical_frames'] and
        gate.readpin(request['metric_config']['whole_policy'])==policy,'Pilot reconstruction/PHY/metric mapping changed')
    result=worker['results'];counts=result['counts'];pairs=result['actual_unique_pairs']
    g.require(result['logical_frames']==43200 and pairs+result['reused_pairs']==43200 and
        counts['unresolved']==0 and counts['caps']==physical.METRIC_CAPS and counts['reserved']==counts['completed'] and
        counts['completed']['model_constructions']==3 and counts['completed']['reference_preparations']==100 and
        counts['completed']['image_scores']==pairs and all(counts['completed'][k]<=v for k,v in physical.METRIC_CAPS.items()),
        'Complete finite actual pilot metric ledger required')
    metric_rows=gate.readpin(worker['results']['metric_rows']);finalists=gate.readpin(worker['results']['finalists'])
    final=full.shortlist(metric_rows,finalists,policy)
    expected={(r['source_index'],r['snr_db'],r['candidate_id']):p for r,p in
        zip(cpu_request['frames'],vworker['results']['logical_frames'])}
    for row in metric_rows:
        key=row['source_index'],row['snr_db'],row['candidate_id']
        g.require(row['reconstruction_result']==expected[key] and isinstance(row['metric_pair_key'],str) and
            g.SHA_RE.fullmatch(row['metric_pair_key']),'Metric row must belong to its actual pilot reconstruction')
    return final,dict(metric_rows=worker['results']['metric_rows'],finalists=worker['results']['finalists'],
        pilot_visual_closed=vp,pilot_CPU_closed=vrequest['pilot_CPU_closed'])


def reusable_pilot(pins,identity):
    request,worker,_,_,_=visual.cpu_closure(pins)
    root=gate.RT/'qualification/h800_ep_pilot_phy_v1_attempt1';owner=gate.readpin(pins['completion'])
    g.require(request['phy_identity']==identity,'Pilot PHY numeric identity differs')
    return dict(schema='CLOSED_ORIGINAL100_PILOT_PHY_REUSE_V1',phy_identity=identity,
        owner_completion=pins['completion'],owner_actual_wait=pins['owner_actual_wait'],
        request=dict(path=str(root/'registered/request.json'),sha256=owner['request_sha256']),
        closed48=physical.reuse48(identity),previous_call_counts_preserved=True)


def scientific_caps():
    return dict(logical_frames_cap=36000,actual_new_packet_decodes_cap=72000,CPU_workers=1,CPU_affinity_count=2,
        source_indices=list(range(1000)),noise_seeds=[4101,4102,4103],snrs=[4,10,19],finalists_per_snr_max=4,
        stage_order=['source900','fullPHY','fullvisual','fullmetrics'],source900=dict(source.CAPS),
        full_visual=dict(model_load=1,encoder=0,source_tx=0,source_rx=36000,var_render=36000,
            prior_scale=720000,decoder_forward=36000),
        full_metrics=dict(model_constructions=3,reference_preparations=1000,image_scores=36000,
            dinov2_vitl14_reference=1000,dinov2_vitl14_reconstruction=36000,
            convnext_reference=1000,convnext_reconstruction=36000,lpips_pair=36000,
            lpips_alexnet_backbone_forward=72000),
        final_winner_requires_complete_original1000_three_noise_grid=True,
        source_TX_calls=0,encoder_calls=0,model_calls=0,RX_VAR_calls=0,render_calls=0,metric_calls=0,
        training_updates=0,new_confirmation_source_reads=0,old_source_and_pilot_call_counts_preserved=True)


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    records,streams,_,_=source_closure(r['source900_closed']);identity,environment=physical.phy_binding();policy,pp,_=gate.policy_and_static()
    final,metric_pins=metric_closure(r['pilot_metrics_closed'],policy)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and
        r['complete_scientific_caps']==scientific_caps() and r['records']==records and r['streams']==streams and
        r['phy_identity']==identity and r['reuse_pilot']==reusable_pilot(metric_pins['pilot_CPU_closed'],identity) and r['policy']==pp and
        r['metric_materials']==metric_pins and r['finalists']==final and r['frames']==full.frames(records,final,policy),
        'Frozen original1000 full PHY request changed')
    engine().check_tools(r['tool_bindings'])
    g.require(r['GPU_execution_admitted'] is False and r['metrics_execution_admitted'] is False and
        r['automatic_successor'] is False,'Only finite CPU PHY stage is admitted')
    return r,environment


def prepare(a):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh full1000 PHY registration required')
    sp=dict(completion=dict(path=str(g.inside(a.source_completion)),sha256=a.source_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.source_owner_wait)),sha256=a.source_owner_wait_sha256))
    mp=dict(completion=dict(path=str(g.inside(a.metric_completion)),sha256=a.metric_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.metric_owner_wait)),sha256=a.metric_owner_wait_sha256))
    records,streams,_,_=source_closure(sp);identity,_=physical.phy_binding();policy,pp,_=gate.policy_and_static()
    final,metric_pins=metric_closure(mp,policy);rows=full.frames(records,final,policy)
    e=execution(a.deadline_unix,a.max_seconds);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=e,complete_scientific_caps=scientific_caps(),
        source900_closed=sp,pilot_metrics_closed=mp,metric_materials=metric_pins,finalists=final,
        records=records,streams=streams,phy_identity=identity,reuse_pilot=reusable_pilot(metric_pins['pilot_CPU_closed'],identity),policy=pp,
        frames=rows,tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        root=str(gate.R2),CPU_python=str(gate.CPU_PYTHON),GPU_execution_admitted=False,metrics_execution_admitted=False,
        automatic_successor=False,preparation_packet_decodes=0,preparation_model_calls=0,
        purpose='Original1000 full-calibration actual PHY only after complete pilot finalists; independent later owners required')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def run(a):
    g.require(g.sys.platform.startswith('linux'),'Actual full1000 PHY owner requires Linux')
    path=g.inside(a.request);r,old_environment=registration(path,a.request_sha256);e=r['execution'];out=path.parent/'run'
    shared=g.helper();child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/'controls/h800_ep_full_phy_v1/CPU.owner.lock')):
        g.require(not out.exists(),'Full1000 PHY already claimed; no replay');out.mkdir()
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
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Full1000 PHY child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and
                len(done['results']['logical_frames'])==len(r['frames']) and done['results']['packet_ledger']['unresolved']==0 and
                done['results']['packet_ledger']['total']<=72000,'Full actual PHY completion not closed')
            registration(path,a.request_sha256)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Full owner final deadline exceeded')
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
            'Full PHY stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Full PHY execution deadline exceeded')
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
        ledger=Ledger(out/'packet_ledger.sqlite',a.request_sha256,72000)
        policy=gate.readpin(r['policy'])
        results=full.cpu_frames(runtime,r['frames'],r['records'],r['finalists'],policy,streams,partial,phy,ledger,boundary,
            out/'frames',r['phy_identity'],r['reuse_pilot'],gate.readpin)
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
    g.require(g.sha(source.__file__)==SOURCE_SCRIPT_SHA,'Frozen source900 implementation changed')
    g.require(g.sha(full.__file__)==FULL_CORE_SHA,'Frozen full PHY core changed')
    g.require(g.sha(metric_owner.__file__)==METRIC_OWNER_SHA,'Frozen pilot metric owner changed')
    runner=source.engine();ns=dict(vars(runner));ns.update(__file__=__file__,TOOLS=TOOLS)
    for name in ('check_tools','worker_identity'):ns[name]=gate.source.host.clone_function(ns[name],ns)
    return source.types.SimpleNamespace(**ns)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('source-completion','source-completion-sha256','source-owner-wait','source-owner-wait-sha256','metric-completion','metric-completion-sha256','metric-owner-wait','metric-owner-wait-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();engine()
    result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    print(g.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
