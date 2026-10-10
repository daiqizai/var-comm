"""Explicit finite new100 four-arm PHY owner; three isolated CPU providers.

Same fixed100, 3SNR and3noise across four arms. Only CPU PHY is admitted here.
Raw433 KEEP, original T1-144 and EP360 remain distinct complete receive domains.
The owner actually waits each of three sequential2CPU children, then validates
the complete3600 logical outcomes and common7200-call durable packet ledger.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
WCL_SCRIPTS=Path(__file__).resolve().parents[2]/'wcl-evidence-closure-20261009/scripts'
sys.path.insert(0,str(WCL_SCRIPTS))
import ep_new100_phy_core_v1 as core
import ep_new100_confirmation_core_v1 as confirmation
import ep_t1_portable_qualification_v1 as portable
import h800_ep_new100_source_v1 as source
import h800_ep_pilot_phy_v1 as inherited

g=source.g;gate=source.gate
SCHEMA='H800_EP_NEW100_FOUR_ARM_PHY_OWNER_V1'
PASS='PASS_H800_EP_NEW100_FOUR_ARM_PHY_ONLY'
SOURCE_SHA='e3751c723700aa2b6a9cf7c229d418ad52553defb24ec0c07ef9823104cdb8df'
SOURCE_CORE_SHA='c11e379e76615b8a35f8c890cbc731a000c4a88dcf65fc6e31bd587b89a4c5a9'
RAW_CORE_SHA='8c74597d691e3cb545cea76b3f0dc0922dc2efe84e5fa72e9448c3f01a8228d0'
RAW_REQUEST_SHA='9e587fcec3be465d8d6c43a7123400635441918befbbd45a6339f0f208af2faf'
TOOLS=tuple(dict.fromkeys(source.TOOLS+('h800_ep_new100_source_v1.py','ep_new100_phy_core_v1.py',
    'ep_new100_confirmation_core_v1.py','ep_t1_portable_qualification_v1.py','ep_new100_content_io_v1.py',
    'h800_mismatch_phy_core_v1.py','mismatch_plan.py','h800_ep_pilot_phy_v1.py','h800_ep_new100_phy_v1.py')))


def bindings():return {n:g.sha(Path(__file__).with_name(n)) for n in TOOLS}


def execution(deadline,seconds,slots):
    value=inherited.execution(deadline,seconds)
    g.require(len(slots)==len(set(slots))==2 and all(type(x) is int and x>=0 for x in slots),'Explicit coordinated2CPU slots required')
    return dict(value,CPU_slots=slots,provider_order=list(core.GROUPS),concurrent_provider_workers=1)


def source_closure(pins):
    root=gate.RT/'qualification/h800_ep_new100_source_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact actual source100 attempt required')
    done=source.readpin(pins['completion']);wait=source.readpin(pins['owner_actual_wait']);worker=source.readpin(done['worker_completion'])
    g.require(done['schema']==worker['schema']==source.SCHEMA and done['status']==worker['status']==source.PASS and
        done['actual_wait']['success'] is True and done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        wait['actual_wait'] is True and wait['returncode']==0 and worker['request_sha256']==done['request_sha256'],
        'Actual source100 owner/worker wait0 required')
    r=g.checked_json(root/'registered/request.json',done['request_sha256']);records,inputs=source.content_closure(r['content_closed'])
    g.require(r['schema']==source.SCHEMA and r['records']==records and r['inputs']==inputs and r['caps']==source.CAPS and
        r['actual_host_native_binding']==g.native_pin() and r['target_index']==2 and r['source_RX_calls']==r['PHY_calls']==0,
        'Actual new100 source/math/host registration differs')
    source.engine().check_tools(r['tool_bindings']);counts=worker['counts']
    g.require(counts['caps']==source.CAPS and counts['reserved']==counts['completed']==source.CAPS and counts['unresolved']==0 and
        worker['source_count']==100 and len(worker['sources'])==100,'Exactly100 encoder+100TX+1000prior and noRX must close')
    for i,item in enumerate(worker['sources']):
        expected=records[i];meta=source.readpin(item['metadata'])
        g.require(all(item[k]==expected[k] for k in ('source_index','source_id','canonical_source_id','evaluation_class_index','preprocessing_id')) and
            item['source_index']==i and item['actual_new_encoder']==item['actual_new_TX']==1 and item['actual_new_prior']==10 and item['independent_RX_calls']==0 and
            meta['source_index']==i and meta['source_id']==item['source_id'] and meta['archive']==item['archive'] and
            meta['source_assets']==item['source_assets'] and meta['tokens_sha256']==item['tokens_sha256'] and
            meta['old_host_bits_used'] is False and meta['old_host_CDF_used'] is False and len(meta['CDF_trace'])==10,
            'Closed actual source100 streams/encoder identity differs')
        for key in ('source_assets','archive'):
            p=g.inside(item[key]['path']);g.require(p.is_file() and g.sha(p)==item[key]['sha256'],'Closed source arrays changed')
    return r,worker,records,inputs


def provider_bindings(t1_pin):
    ep_identity,environment=inherited.phy_binding()
    raw_request=gate.RT/'qualification/h800_mismatch_phy_v1_attempt1/registered/request.json'
    raw=g.checked_json(raw_request,RAW_REQUEST_SHA);binding=raw['raw_runtime_binding']
    for row in binding['files']:g.require(g.sha(g.inside(row['path']))==row['sha256'],'Original raw provider file changed')
    spec,old,request,_=portable.inspect_binding(t1_pin)
    g.require(spec['actual_project_root']==str(gate.R2) and g.sha(core.raw_original.__file__)==RAW_CORE_SHA,
        'Exact relocated project and raw provider required')
    shared=dict(CPU_environment=ep_identity['admitted_CPU_environment'],CPU_python=ep_identity['CPU_python'],
        actual_host_native_binding=ep_identity['actual_host_native_binding'],CUDA_visible=False)
    providers=dict(raw=dict(group='raw',profile_count=433,catalogue_sha256=binding['catalogue_sha256'],
        aliases_sha256=binding['aliases_sha256'],binding_sha256=confirmation.digest(binding),raw_provider_sha256=RAW_CORE_SHA,
        actual_CPU_qualification=ep_identity['qualification'],original_raw_qualification_preserved=True,**shared),
        whole=dict(group='whole',profile_count=144,catalogue_sha256=old['catalogue_sha256'],backend_identity=old['backend_identity'],
            original241=spec['completion'],relocation=t1_pin,provider_sha256=portable.T1_SHA,current_CPU_qualification=ep_identity['qualification'],
            new_T1_qualification_claimed=False,**shared),
        partial=dict(group='partial',profile_count=360,catalogue_sha256=ep_identity['catalogue_sha256'],
            backend_identity=ep_identity['backend_identity'],qualification=ep_identity['qualification'],**shared))
    return providers,binding,environment


def registration(path,digest):
    g.require(g.sha(source.__file__)==SOURCE_SHA and g.sha(source.core.__file__)==SOURCE_CORE_SHA,'Frozen source100 implementation changed')
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'],r['execution']['CPU_slots'])
    _,worker,records,inputs=source_closure(r['source100_closed']);providers,raw,environment=provider_bindings(r['T1_original241_relocation'])
    selected=source.readpin(inputs['selection']);policy=source.readpin(inputs['policy_bundle']);content=source.readpin(inputs['content_gate'])
    rows=confirmation.frame_grid(selected,inputs['selection'],content,policy)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['records']==records and
        r['sources']==worker['sources'] and r['inputs']==inputs and r['frames']==rows and r['providers']==providers and
        r['raw_binding']==raw and r['complete_scientific_caps']==confirmation.scientific_caps() and r['tool_bindings']==bindings() and
        r['CPU_python']==str(gate.CPU_PYTHON) and r['root']==str(gate.R2) and r['packet_cap']==7200 and
        r['neural_model_calls']==r['encoder_calls']==r['source_TX_calls']==r['source_RX_calls']==r['metric_calls']==0 and
        r['automatic_successor'] is False and r['automatic_retry'] is False,'Frozen finite common100 four-arm PHY registration differs')
    core.validate_grid(rows);return r,environment


def prepare(a):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh new100 PHY registration required')
    sp=dict(completion=dict(path=str(g.inside(a.source_completion)),sha256=a.source_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.source_owner_wait)),sha256=a.source_owner_wait_sha256))
    tp=dict(path=str(g.inside(a.t1_relocation)),sha256=a.t1_relocation_sha256)
    _,worker,records,inputs=source_closure(sp);providers,raw,_=provider_bindings(tp)
    rows=confirmation.frame_grid(source.readpin(inputs['selection']),inputs['selection'],source.readpin(inputs['content_gate']),source.readpin(inputs['policy_bundle']))
    e=execution(a.deadline_unix,a.max_seconds,a.cpu_slots);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',execution=e,source100_closed=sp,T1_original241_relocation=tp,
        records=records,sources=worker['sources'],inputs=inputs,frames=rows,providers=providers,raw_binding=raw,
        complete_scientific_caps=confirmation.scientific_caps(),tool_bindings=bindings(),CPU_python=str(gate.CPU_PYTHON),root=str(gate.R2),
        packet_cap=7200,neural_model_calls=0,encoder_calls=0,source_TX_calls=0,source_RX_calls=0,metric_calls=0,
        automatic_successor=False,automatic_retry=False,historical_RX_reuse=False,
        purpose='Fresh common100 original raw433, original T1whole144 and allowed-partial360 PHY; one shared7200 packet cap')
    path=out/'request.json';g.write(path,r);return core.pin(path)


def run(a):
    path=g.inside(a.request);r,environment=registration(path,a.request_sha256);out=path.parent/'run';e=r['execution']
    g.require(sys.platform.startswith('linux') and set(e['CPU_slots'])<=os.sched_getaffinity(0),'Actual admitted Linux2CPU owner required')
    shared=g.helper();child=None;started=time.monotonic();waits=[];results=[]
    with shared.owner_lock(g.inside(g.BASE/'controls/ep_new100_phy_v1/CPU.owner.lock')):
        g.require(not out.exists(),'New100 PHY already attempted; no replay');out.mkdir()
        g.write(out/'intent.json',dict(request_sha256=a.request_sha256,owner_pid=os.getpid(),execution=e,packet_cap=7200))
        try:
            snapshot=shared.cpu_snapshot();g.write(out/'prelaunch_resources.json',snapshot)
            g.require(snapshot['effective_cpu_cores']>=2 and snapshot['available_memory_bytes']>=2*(1<<30),
                'TwoCPU cores and at least2GiB memory required')
            for group in core.GROUPS:
                phase=out/group;phase.mkdir();env=gate.controlled_cpu_environment(phase,environment)
                g.write(phase/'controlled_environment.json',env)
                argv=[r['CPU_python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,
                    '--owner-pid',str(os.getpid()),'--group',group]
                seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'PHY stage deadline expired')
                with (phase/'child.log').open('xb') as log:
                    child=subprocess.Popen(argv,cwd=r['root'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        preexec_fn=shared.child_limits(e['CPU_slots'],seconds,2))
                    g.write(phase/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,CPU_slots=e['CPU_slots']))
                    waited=shared.wait_owned(child,seconds)
                g.write(phase/'actual_child_wait.json',waited);waits.append(waited)
                g.require(waited['success'] and child.returncode==0,'Actual provider child failed; remaining groups are not launched')
                worker=source.readpin(core.pin(phase/'worker_completion.json'))
                g.require(worker['status']==PASS and worker['request_sha256']==a.request_sha256 and worker['group']==group and
                    worker['CUDA_initialized'] is False,'Actual CPU-only provider closure differs')
                results.append(worker['results']);print(json.dumps(dict(group=group,actual_wait=True,returncode=0,logical=worker['results']['logical_count'])),flush=True)
            logical,ledger=core.close_groups(results,r['frames'],source.readpin)
            import t2_ledger
            actual=t2_ledger.Ledger(out/'packet_ledger.sqlite',a.request_sha256,7200)
            g.require(actual.snapshot()==ledger,'Aggregate durable paid ledger differs');actual.close()
            registration(path,a.request_sha256)
            rows_pin=core.save(out/'logical_rows.json',logical)
            workers=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=dict(logical_rows=rows_pin,groups=results,packet_ledger=ledger),
                logical_count=3600,source_RX_calls=0,neural_model_calls=0,CUDA_initialized=False)
            g.write(out/'worker_completion.json',workers)
            g.require(time.time()<e['deadline_unix'] and time.monotonic()-started<e['max_seconds'],'Final owner deadline exceeded')
            done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.pin(out/'worker_completion.json'),
                actual_children_waited=True,worker_exit_codes=[0,0,0],actual_waits=waits,automatic_successor=False,logical_count=3600,
                packet_ledger=ledger,source_RX_calls=0,neural_model_calls=0)
            g.write(out/'completion.json',done);return done
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_waits=waits,current_child_exit=None if child is None else child.returncode,automatic_successor=False));raise


def worker_identity(path,digest,r,owner_pid,group):
    g.require(group in core.GROUPS,'Registered receiver group required')
    path=g.inside(path);out=path.parent/'run'/group;started=out/'child_started.json'
    for _ in range(20):
        if started.exists():break
        g.require(os.getppid()==owner_pid,'Owner lost before child launch receipt');time.sleep(.05)
    intent=source.readpin(core.pin(out.parent/'intent.json'));launch=source.readpin(core.pin(started))
    argv=[r['CPU_python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),
        '--request-sha256',digest,'--owner-pid',str(owner_pid),'--group',group]
    g.require(path.name=='request.json' and path.parent.name=='registered' and g.sha(path)==digest and
        intent['request_sha256']==digest and intent['owner_pid']==owner_pid and intent['execution']==r['execution'] and
        intent['packet_cap']==7200 and launch['argv']==argv and launch['request_sha256']==digest and
        launch['owner_pid']==owner_pid==os.getppid() and launch['pid']==os.getpid() and
        launch['CPU_slots']==r['execution']['CPU_slots'],'Actual registered request/owner/group/argv identity differs')
    return launch


def worker(a):
    path=g.inside(a.request);r,_=registration(path,a.request_sha256);e=r['execution'];out=path.parent/'run'/a.group
    launch=worker_identity(path,a.request_sha256,r,a.owner_pid,a.group)
    g.require(os.getpid()==launch['pid'] and os.getppid()==a.owner_pid==launch['owner_pid'] and
        launch['request_sha256']==a.request_sha256 and set(os.sched_getaffinity(0))==set(e['CPU_slots']) and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='','Actual bounded hidden-CUDA2CPU child identity required')
    started=time.monotonic();last=0.;boundaries=0;ledger=None;stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def boundary():
        nonlocal last,boundaries
        g.require(not stopped and os.getppid()==a.owner_pid and time.time()<e['deadline_unix'] and time.monotonic()-started<e['max_seconds'] and
            not (g.BASE/'STOP').exists() and not (out/'STOP').exists() and not (out.parent/'STOP').exists(),'PHY stop, owner loss or deadline')
        boundaries+=1
        if time.monotonic()-last>=10:
            last=time.monotonic()
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(boundaries=boundaries,group=a.group,
                ledger=None if ledger is None else ledger.snapshot(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),owner_pid=a.owner_pid,group=a.group,request_sha256=a.request_sha256))
        import t1_phy
        import ep_phy
        import ep_source_codec as partial
        from t2_ledger import Ledger
        boundary();ledger=Ledger(out.parent/'packet_ledger.sqlite',a.request_sha256,7200)
        if a.group=='raw':runtime=core.raw_original.create_runtime(r['raw_binding'],ledger,boundary)
        elif a.group=='whole':runtime=portable.create_runtime(r['T1_original241_relocation'],r['root'],t1_phy,
            r['providers']['whole']['CPU_environment'])
        else:
            runtime=ep_phy.create_runtime(r['root'],r['providers']['partial']['qualification']['path']);runtime._new100_phy_module=ep_phy
        g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
        streams={} if a.group=='raw' else {item['source_index']:inherited.core.link.load_streams(item['archive'],source.readpin(item['metadata']),partial) for item in r['sources']}
        results=core.execute_group(a.group,runtime,r['frames'],r['sources'],streams,r['providers'][a.group],ledger,boundary,out/'frames',t1_phy,g.inside)
        boundary();g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
        g.write(out/'worker_completion.json',dict(schema=SCHEMA,status=PASS,group=a.group,request_sha256=a.request_sha256,results=results,
            CUDA_initialized=False,neural_model_calls=0,source_RX_calls=0));ledger.close();ledger=None
    except BaseException as error:
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            request_sha256=a.request_sha256,packet_ledger=None if ledger is None else ledger.snapshot()));raise
    finally:
        if ledger is not None:ledger.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('source-completion','source-completion-sha256','source-owner-wait','source-owner-wait-sha256','t1-relocation','t1-relocation-sha256','out'):q.add_argument('--'+name,required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=21600)
    q.add_argument('--cpu-slots',type=lambda x:[int(v) for v in x.split(',')],required=True)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True);q.add_argument('--group',choices=tuple(core.GROUPS),required=True)
    a=p.parse_args();result=prepare(a) if a.command=='prepare' else run(a) if a.command=='run' else worker(a)
    if result is not None:print(json.dumps(result),flush=True)


if __name__=='__main__':main()
