"""Explicit prepare, CPU PHY owner, and separate GPU receive/render owner.

Both prerequisites must actually close before prepare. No stage automatically
starts another stage. Existing scientific implementations are never modified.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import types
import h800_ep32_source_gate_v1 as source
import h800_ep48_link_core_v1 as core

g=source.g
SCHEMA=core.SCHEMA
PASS=core.PASS
CAPS=core.CAPS
RT=source.host.RT
R2=g.BASE/'code/VAR_COMM'
SEED=RT/'assets/fixed_existing32_seed_v1'
SOURCE_ROOT=RT/'qualification/h800_ep32_source_gate_v1_attempt1'
PHY_ROOT=R2/'outputs/WCL-ENTROPY-PARTIAL-20261010/H800_PHY457_v2'
PHY_CONTROL=RT/'qualification/h800_ep457_phy_v2_attempt1'
PHY_SHA='ee7a5d9359e82dba62bfdaa3f4d1246815a05f13f393edd23500a6bdd6f6967d'
PHY_ENV_SHA='9a88c004dd4e6639621d5f1cd853646eb5bae435d37a7893c9ffa06859ab8339'
PHY_WAIT_SHA='51f18ccc0c725a4345a297448d2ce23df5c0177e7311b56cad955386728b6496'
SOURCE_DONE_SHA='44f6858f0c696dc1273af18eaeeb78a8befe0f42d997d709f104e31fa7f1491a'
SOURCE_WAIT_SHA='b349fc54219db3fe8378b7615bebadec791b2730e6ec95b768847fbcb4c40fa7'
SOURCE_FIRST4_TX_SHA=(
    '7571987974506ae5f0ce8e45e5ca1de9bdf9f5660bc58b9d93ea1fe4b5f50fee',
    'd04be953a25beff0a1b684576739d121a6d0ea41c22dd8e08654d3f0794bdca1',
    'e55773672a1ab42f37095e8aee0d2d0dfe74aecc847e006dfc534705b1baf0df',
    '8038a62cf23bc8648576b404adac1f12ef0b46525158a07fe49dbcb2374757b0')
SOURCE_IMPLEMENTATION_SHA='5ffd10b0e44e18667b9de87ac5dfbdf01a0a99948d081a8293b61c20815787c7'
TOOLS=source.TOOLS+('h800_ep32_source_gate_v1.py','h800_ep48_link_core_v1.py','ep_phy.py','ep_plan.py','ep_qualify_phy.py')
CPU_PYTHON=RT/'envs/frozen_ldpc_py311/bin/python'


def readpin(pin):return g.checked_json(g.inside(pin['path']),pin['sha256'])


def prerequisites(pins):
    expected={'source_completion':SOURCE_ROOT/'registered/run/completion.json',
        'source_wait':SOURCE_ROOT/'run_owner_actual_wait.json','phy_completion':PHY_ROOT/'completion.json',
        'phy_wait':PHY_CONTROL/'qualify_actual_wait.json','phy_environment':PHY_CONTROL/'environment.json'}
    g.require(set(pins)==set(expected),'Exact component-gate receipt set required')
    for name,path in expected.items():g.require(pins[name]['path']==str(path),'Unexpected component receipt path: '+name)
    for name,digest in dict(source_completion=SOURCE_DONE_SHA,source_wait=SOURCE_WAIT_SHA,
            phy_completion=PHY_SHA,phy_wait=PHY_WAIT_SHA,phy_environment=PHY_ENV_SHA).items():
        g.require(pins[name]['sha256']==digest,'Exact completed component attempt required: '+name)
    values={name:readpin(pin) for name,pin in pins.items()}
    for name in ('source_wait','phy_wait'):
        g.require(values[name]['actual_wait'] is True and values[name]['returncode']==0,'Component owner not actually closed: '+name)
    s=values['source_completion'];p=values['phy_completion']
    g.require(s['status']==source.PASS and s['actual_children_waited'] is True and s['worker_exit_codes']==[0] and
        s['caps']==source.CAPS,'Complete first32 source gate required')
    sw=readpin(s['worker_completion'])
    g.require(sw['status']==source.PASS and sw['counts']['completed']==sw['counts']['reserved']==source.CAPS and
        sw['counts']['unresolved']==0 and len(sw['source_completions'])==32,'First32 model ledger not closed')
    sr=g.checked_json(g.inside(SOURCE_ROOT/'registered/request.json'),s['request_sha256'])
    g.require(sr['schema']==source.SCHEMA and sr['caps']==source.CAPS and sr['input_map']==source.input_map(),
        'First32 input closure changed')
    g.require(pins['phy_completion']['sha256']==PHY_SHA and p['status']=='PASS' and
        p['schema']=='WCL_EP_REAL_PHY_QUALIFICATION_20261010_V1' and p['packet_decode_count']==457 and
        p['actual_header_decodes']==361 and p['actual_body_decodes']==96 and p['ledger']==dict(total=457,unresolved=0,cap=457),
        'Complete actual457 PHY gate required')
    g.require(pins['phy_environment']['sha256']==PHY_ENV_SHA and values['phy_environment']['CUDA_VISIBLE_DEVICES']=='' and
        values['phy_environment']['VIRTUAL_ENV']==str(CPU_PYTHON.parent.parent),'Exact actual LDPC CPU environment required')
    streams=[]
    for i,item in enumerate(sw['source_completions']):
        done=readpin(item)
        g.require(done['source_index']==i and done['status']=='SOURCE_COMPLETE' and done['partial_RX']==18 and
            done['short_whole_RX']==2 and done['prior_scales']==154,'Source-gate scope differs')
        if i<4:
            path=g.inside(Path(item['path']).parent/'actual_TX.json');meta=g.checked_json(path,SOURCE_FIRST4_TX_SHA[i])
            g.require(meta['source_index']==i,'Actual stream source mismatch')
            archive=meta['archive'];g.require(g.sha(g.inside(archive['path']))==archive['sha256'],'Actual source archive changed')
            streams.append(dict(source_index=i,source_id=done['source_id'],metadata=core.descriptor(path),archive=archive))
    return sr,streams,values


def policy_and_static():
    path=g.inside(SEED/'VAR_COMM/outputs/WCL-EVIDENCE-CLOSURE-20261009/T1_entropy_whole/frozen_policy.json')
    policy=g.checked_json(path,core.POLICY_SHA)
    return policy,core.descriptor(path),core.static_binding(policy,SEED,g.checked_json,g.inside)


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest);runner=engine()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and
        r['packet_cap']==96 and r['logical_frames']==48,'Exact independent EP48 registration required')
    source.execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    prior,streams,_=prerequisites(r['prerequisites']);policy,pp,static=policy_and_static()
    g.require(r['stream_inputs']==streams and r['policy']==pp and r['static']==static,'Frozen EP48 input binding differs')
    g.require(r['spec']==runner.inherited_spec(prior,r['execution']['deadline_unix'],900),'Historical visual input closure changed')
    import ep_source_codec as partial
    g.require(r['frames']==core.plan(policy,prior['input_map']['records'][:4],partial),'Fixed48 plan differs')
    runner.check_tools(r['tool_bindings'])
    g.require(r['actual_host_native_binding']==g.native_pin() and
        r['device_identity_binding']==runner.device_from_request(r).device_identity_binding(),'Frozen visual host binding differs')
    g.require(r['automatic_successor'] is False and r['quality_scoring'] is False and r['production_admission'] is False,
        'No scoring, calibration or automatic continuation permitted')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh EP48 registration required')
    pins=g.checked_json(g.inside(a.prerequisites),a.prerequisites_sha256)
    prior,streams,_=prerequisites(pins);policy,pp,static=policy_and_static();e=source.execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    import ep_source_codec as partial
    frames=core.plan(policy,prior['input_map']['records'][:4],partial)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=source.host.DeviceAdapter(dp,a.target_index)
    out.mkdir(parents=True)
    request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,execution=e,caps=CAPS,packet_cap=96,logical_frames=48,
        prerequisites=pins,stream_inputs=streams,policy=pp,static=static,frames=frames,device_receipt=dp,target_index=a.target_index,
        device_identity_binding=device.device_identity_binding(),actual_host_native_binding=g.native_pin(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        CPU_python=str(CPU_PYTHON),quality_scoring=False,training_updates=0,source_TX_calls=0,
        received_truth_available=False,probability_tables_shared=False,automatic_successor=False,production_admission=False,
        purpose='48 actual image links after completed32source and457PHY component gates; original whole winner plus three fixedK targets')
    path=out/'request.json';g.write(path,request);return core.descriptor(path)


def controlled_cpu_environment(out,old):
    env=dict(old);_,controlled=g.helper().controlled_environment(g.BASE,out,2);env.update(controlled)
    (out/'home').mkdir();env.update(HOME=str(out/'home'),PATH=str(CPU_PYTHON.parent)+':/usr/bin:/bin',
        VIRTUAL_ENV=str(CPU_PYTHON.parent.parent),CUDA_VISIBLE_DEVICES='')
    for key in ('GIT_CONFIG_GLOBAL','GIT_CONFIG_SYSTEM'):env[key]='/dev/null'
    g.require(not any(key in env for key in ('PYTHONHOME','PYTHONPATH','CONDA_PREFIX','CONDA_DEFAULT_ENV')),'Unregistered Python path injection')
    return env


def run_owner(a,runner,kind):
    g.require(g.sys.platform.startswith('linux'),'Actual owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256);out=path.parent/kind
    spec=r['spec'];e=r['execution'];shared=g.helper();device=runner.device_from_request(r);started=time.monotonic();child=None
    lock=g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock') if kind=='gpu' else g.BASE/'controls/h800_ep48_v1/CPU.owner.lock'
    with shared.owner_lock(g.inside(lock)):
        g.require(not out.exists(),'Stage already claimed; no replay');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,stage=kind,execution=e))
        try:
            if kind=='gpu':
                cpu=g.checked_json(path.parent/'cpu/completion.json',g.sha(path.parent/'cpu/completion.json'))
                g.require(cpu['status']=='EP48_CPU_ACTUAL48_CLOSED' and cpu['request_sha256']==a.request_sha256 and cpu['actual_wait']['success'],
                    'CPU actual48 owner must close before GPU construction')
                _,_,_,_,extra=g.validate_spec(spec)
                snapshot,ad=runner.wait_prelaunch(device,shared,out,e,started,120);g.write(out/'prelaunch_resources.json',snapshot)
                g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
                _,env=shared.controlled_environment(g.BASE,out,6);(out/'home').mkdir()
                env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=extra['environment']['LD_LIBRARY_PATH'],HOME=str(out/'home'),
                    LANG='C.UTF-8',LC_ALL='C.UTF-8',VIRTUAL_ENV=str(Path(spec['python']).parent.parent),
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',**device.cuda_environment_binding());python=spec['python']
            else:
                env=controlled_cpu_environment(out,readpin(r['prerequisites']['phy_environment']));python=str(CPU_PYTHON)
                cpu=shared.cpu_snapshot();g.require(cpu['effective_cpu_cores']>=2 and len(cpu['allowed_cpus'])>=2 and
                    cpu['available_memory_bytes']>=2*(1<<30),'Insufficient bounded CPU resources')
                ad=dict(cpu_affinity=cpu['allowed_cpus'][:2]);g.write(out/'prelaunch_resources.json',cpu)
            g.write(out/'controlled_environment.json',env)
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'EP48 deadline expired')
            argv=[python,'-B','-u',str(Path(__file__).absolute()),'_'+kind,'--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,cpu_affinity=ad['cpu_affinity']))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'EP48 stage child failed; no retry')
            done=g.checked_json(out/'worker_completion.json',g.sha(out/'worker_completion.json'))
            status=PASS if kind=='gpu' else 'EP48_CPU_ACTUAL48_CLOSED'
            g.require(done['status']==status and done['request_sha256']==a.request_sha256,'Actual worker completion differs')
            registration(path,a.request_sha256)
            if kind=='gpu':core.close_counts(done['counts']);g.validate_spec(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'EP48 owner final deadline exceeded')
            result=dict(schema=SCHEMA,status=status,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,automatic_successor=False,production_admission=False)
            g.write(out/'completion.json',result);return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise


def worker(a,runner,kind):
    path=g.inside(a.request);r=registration(path,a.request_sha256);out=path.parent/kind;spec=r['spec'];e=r['execution']
    intent=g.checked_json(out/'intent.json',g.sha(out/'intent.json'))
    until=time.monotonic()+2
    while not (out/'child_started.json').exists() and time.monotonic()<until:time.sleep(.01)
    launch=g.checked_json(out/'child_started.json',g.sha(out/'child_started.json'))
    python=spec['python'] if kind=='gpu' else str(CPU_PYTHON)
    expected=[python,'-B','-u',str(Path(__file__).absolute()),'_'+kind,'--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(a.owner_pid)]
    g.require(os.getppid()==a.owner_pid and intent['owner_pid']==a.owner_pid and intent['request_sha256']==a.request_sha256 and
        launch['pid']==os.getpid() and launch['argv']==expected,'Actual stage owner/child identity differs')
    stopped=False;started=time.monotonic();backend=None;ledger=None;packet_ledger=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'EP48 stop or parent loss')
        g.require(time.time()<e['deadline_unix'] and time.monotonic()-started<e['max_seconds'],'EP48 execution deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if kind=='gpu' and time.monotonic()-last>=2:
            snapshot,_=runner.device_from_request(r).resource_snapshot(g.helper());last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256,stage=kind))
        import ep_source_codec as partial
        if kind=='cpu':
            g.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==2,'CPU child must hide CUDA and use2CPUs')
            import ep_phy as phy
            from t2_ledger import Ledger
            runtime=phy.create_runtime(spec['project_root'],r['prerequisites']['phy_completion']['path'])
            g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
            streams={item['source_index']:core.load_streams(item['archive'],readpin(item['metadata']),partial) for item in r['stream_inputs']}
            packet_ledger=Ledger(out/'packet_ledger.sqlite',a.request_sha256,96)
            frames=core.cpu_frames(runtime,r['frames'],streams,partial,phy,packet_ledger,boundary,out/'frames')
            counts=packet_ledger.snapshot();packet_ledger.close();packet_ledger=None;guard();registration(path,a.request_sha256)
            g.require(not runtime.torch.cuda.is_initialized(),'CPU PHY initialized CUDA')
            result=dict(schema=SCHEMA,status='EP48_CPU_ACTUAL48_CLOSED',request_sha256=a.request_sha256,frames=frames,
                packet_ledger=counts,actual_frames=48,CUDA_initialized=False,neural_model_calls=0,automatic_successor=False)
        else:
            device=runner.device_from_request(r);device.check_cuda_environment()
            resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
            module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_ep48_original_math')
            Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
            backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
            codec=partial.PartialSourceCodec(spec['project_root']);core.bind_static(codec,r['static'])
            driver=resolver.path(g.OLD+'experiments/content-real-64qam-20261006/h_source_driver.py')
            core.bind_independent_provider(codec,backend,module,driver,g)
            cpu_owner=g.checked_json(path.parent/'cpu/completion.json',g.sha(path.parent/'cpu/completion.json'))
            g.require(cpu_owner['status']=='EP48_CPU_ACTUAL48_CLOSED' and cpu_owner['request_sha256']==a.request_sha256 and
                cpu_owner['actual_wait']['success'],'Actual CPU completion required')
            cpu=readpin(cpu_owner['worker_completion']);g.require(cpu['actual_frames']==48 and cpu['packet_ledger']['unresolved']==0,'CPU frame ledger incomplete')
            rendered=core.gpu_frames(cpu['frames'],codec,backend,partial,ledger,boundary,out/'images')
            backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256);guard()
            counts=core.close_counts(ledger.summary());evidence=[readpin(pin)['evidence'] for pin in rendered]
            positive=sum(x['render_called'] and x['actual_received_profile']['K']>0 for x in evidence)
            # A complete all-fallback execution is retained but cannot qualify partial image transport.
            result=dict(schema=SCHEMA,status=PASS if positive else 'COMPLETE_NO_POSITIVE_K_PATH_NOT_QUALIFIED',request_sha256=a.request_sha256,
                counts=counts,rendered_frames=rendered,logical_frames=48,positive_K_reconstructions=positive,
                fixed_gray_outputs=sum(x['kind']=='gray' for x in evidence),CPU_completion=core.descriptor(path.parent/'cpu/completion.json'),
                source_truth_used=False,TX_CDF_used=False,quality_scoring=False,automatic_successor=False,production_admission=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary(),
            packet_ledger=None if packet_ledger is None else packet_ledger.snapshot()));raise
    finally:
        if packet_ledger is not None:packet_ledger.close()


def engine():
    g.require(g.sha(Path(source.__file__))==SOURCE_IMPLEMENTATION_SHA,'Frozen source gate implementation changed')
    ns=dict(vars(source.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec'):
        ns[name]=source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--prerequisites',required=True);q.add_argument('--prerequisites-sha256',required=True)
    q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',type=int,choices=(0,2),required=True);q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=3600)
    for name in ('run-cpu','run-gpu','_cpu','_gpu'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name.startswith('_'):q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    value=prepare(a,runner) if a.command=='prepare' else run_owner(a,runner,a.command[4:]) if a.command.startswith('run-') else worker(a,runner,a.command[1:])
    print(g.json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
