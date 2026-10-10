"""Bounded missing68 fresh TX stage after the actual closed48 image-link gate.

Only original calibration0032..0099 are encoded. The already completed32 streams
are byte-bound for later pilot reuse. No PHY/RX/render/metrics or automatic next
stage; this stage alone cannot freeze or select a scientific strategy.
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
import h800_ep48_link_gate_v1 as gate
import h800_ep_pilot_core_v1 as core

g=gate.g
SCHEMA='H800_EP_PILOT_SOURCE68_V1'
PASS='PASS_H800_ORIGINAL_CALIBRATION_SOURCE100_AVAILABLE'
CAPS=core.SOURCE_CAPS
RT=gate.RT
LINK_ROOT=RT/'qualification/h800_ep48_link_v1_attempt1'
LINK_PINS={'gpu_owner_actual_wait.json':'1c260f4a573658db4099a68ad9c6c69deec755486bfdde32f1fc57c3e1511eec',
    'registered/gpu/completion.json':'54316154f90392e410b13a2f4195bd629a422963c9a622d62c038df308108c60',
    'registered/request.json':'b5228d730f4bbfb95d1c13621fe68d08a62be701acc17b5dc8d45a9fae5d7b90'}
RESTORE=RT/'incoming/fixed_existing68_seed_v1/hydration_completion.json'
RESTORE_SHA='b89c60faecfe4f522fad243a5a6b1f29d92145df66ad268b4d8abf3b83a1b9d1'
CORE_SHA='3f0ea97b7902de4c24e80ebcfe731b160863ac252ba7ff81948e86b7ca0103cb'
FIRST32_TX_SHA=(
    '7571987974506ae5f0ce8e45e5ca1de9bdf9f5660bc58b9d93ea1fe4b5f50fee',
    'd04be953a25beff0a1b684576739d121a6d0ea41c22dd8e08654d3f0794bdca1',
    'e55773672a1ab42f37095e8aee0d2d0dfe74aecc847e006dfc534705b1baf0df',
    '8038a62cf23bc8648576b404adac1f12ef0b46525158a07fe49dbcb2374757b0',
    '0f9441e1d604cef406e44bd26fc1a99a1f06fb8a3275c6708d29a5d7a5b3937e',
    'bb59ccc28922a37ff41ff5fa5bc0948337cd1aa131f2c98257eebaf77882f734',
    'd33485b6eb4b2821c38e50fe258f92ab0e22c7f989bb705c858860e7064cd502',
    '7b6d1e29d8352cc87d54e3bd72aaba7a63bccc85794e488c9313f95b80e42660',
    '6a1020fc5eeaa6d09fc005f1d46aeef79abe7e5739cadb50269dae5b1045f700',
    '47e9befc717cf0e13414f061452485f324cf7d3325396a5d1e2d0d943317d680',
    'cee916616000b385327bcc34f9f040ff2338d5ac5b3318ab71b8b76d25f732a0',
    '2202ef3611fd0bf68572bf7884c4c8812dc08129cda15ccc6084d9e391a00108',
    '6908d357376c404de85d2f3db07e07adb93541b2552e57bb52141a87438449c1',
    'ccb90d82608622c61fb835a917c559349803a690a50130d45d353464ec7e0f5c',
    '3579c872d3b3b639e03e5aea60d7961bc7ca9d90fb1534b71b679e9d1c94d8ef',
    '1ceeb6392daeb51d9df391b54f9c94ce64ffc47d1354bfbcce30024d045afb92',
    'd569f55fac91d11c92b08792780412ccb2ce9ecc071262310bec06afee3df1e3',
    'c54c04f119fdb0ca14087a73cdf9f0e577397a0486569a76fd35a707de6c5b69',
    '42bedb449e5888e82b044be1db8e8271df19f34ae4c7ff65c386341e1c873d85',
    '7186c38b624dee1bde3819e2b8f585bac65eafcd8c9c536a2af59bfbe6f283a2',
    '5271001b97e947c245f9e93f544e67821b46d1e81920bb430b35c6a039e2583c',
    '50525e04048e2c20fc49602baa22e540fb940ef7c613e8f799ea6b6d5737983a',
    '73b763e7f79e5faa8889563e375e80a257a7e44a47399e8b6f469454833ad787',
    'eb442e3b4554ce598512598511c823b557106dadac890b00b8cb5b3e3c62b494',
    'ccff968e93eb586b5246baa8d28a4a473f139e012945ee6b61c8943a8e9cd226',
    '5736b4da7b10f61b2d249ffada0a98cf081caff97d31216f9c2905590812991f',
    '8d593b1fcfd88210e74790393c596978f9b195116c0926acae9fe812d7316b9d',
    'a7cc5124d77d999576d4a123f6500ab106872833ce691c5b0e9af304649bf692',
    'd855280561b6721743241091d8352091ec679cc3fa31e70b7b0294cb74e1e72c',
    'ab8c5cf1d1d0a13eba532e0992772f22db053568a1d6f28e23f3d7109827de86',
    '7989b17851af1b23744f41bf6ca0bb8824716fc9c572800b6b6ffc799278ac22',
    '7beef6b82c7dc18f379d7b879c3d36622d05f75a13f8a3df3a8f9a7bcd8d70b4',
)
TOOLS=gate.TOOLS+('h800_ep48_link_gate_v1.py','h800_ep_pilot_core_v1.py')


def inputs():
    old=gate.source.input_map();restore=g.checked_json(g.inside(RESTORE),RESTORE_SHA)
    g.require(restore['schema']=='PILOT68_ORIGINAL_ASSETS_RESTORED_V1' and restore['source_indices']==list(range(32,100)) and
        restore['file_count']==136 and restore['source_manifest_sha256']==core.MANIFEST_SHA and
        restore['source_completion_sha256']==core.SOURCE_COMPLETION_SHA,'Exact original68 restoration required')
    relocation={}
    for row in old['records']:
        i=row['source_index'];relocation[core.OLD_ASSET+f'sources/{i:04d}.npz']=row['archive']
        relocation[core.OLD_ASSET+f'source_checkpoints/{i:04d}.json']=row['checkpoint']
    for row in restore['mapping']:
        key=row['original_path'];g.require(key not in relocation,'Restoration duplicates a previously bound input')
        relocation[key]=dict(path=row['actual_path'],sha256=row['sha256'])
    g.require(len(relocation)==200,'Exact100 original archives plus100 checkpoints required')
    root=gate.source.ASSET_ROOT
    records=core.input_records(dict(path=str(root/'manifest.json'),sha256=core.MANIFEST_SHA),
        dict(path=str(root/'completion.json'),sha256=core.SOURCE_COMPLETION_SHA),relocation,g.inside)
    return records,dict(path=str(RESTORE),sha256=RESTORE_SHA)


def prerequisites():
    values={p:g.checked_json(g.inside(LINK_ROOT/p),h) for p,h in LINK_PINS.items()}
    wait=values['gpu_owner_actual_wait.json'];owner=values['registered/gpu/completion.json'];request=values['registered/request.json']
    g.require(wait['actual_wait'] is True and wait['returncode']==0 and owner['status']==gate.PASS and
        owner['actual_wait']['success'] is True and owner['actual_children_waited'] is True and
        owner['request_sha256']==LINK_PINS['registered/request.json'],'Actual closed48 prerequisite required')
    worker=gate.readpin(owner['worker_completion']);gate.core.close_counts(worker['counts'])
    g.require(worker['status']==gate.PASS and worker['logical_frames']==48 and worker['positive_K_reconstructions']>0 and
        worker['request_sha256']==owner['request_sha256'],'Actual positiveK image-link proof required')
    prior,_,_=gate.prerequisites(request['prerequisites'])
    for name,digest in request['tool_bindings'].items():
        g.require(g.sha(Path(__file__).with_name(name))==digest,'Completed48 bound source changed: '+name)
    g.require(len(FIRST32_TX_SHA)==32,'All32 independent closed TX metadata pins required')
    streams=[]
    for i,digest in enumerate(FIRST32_TX_SHA):
        path=g.inside(gate.SOURCE_ROOT/f'registered/run/sources/{i:04d}/actual_TX.json')
        meta=g.checked_json(path,digest);g.require(meta['source_index']==i,'Reused32 source order differs')
        archive=meta['archive'];g.require(g.sha(g.inside(archive['path']))==archive['sha256'],'Reused32 actual stream changed')
        streams.append(dict(source_index=i,source_id=prior['input_map']['records'][i]['source_id'],
            metadata=core.descriptor(path),archive=archive,actual_new_TX=0,origin='CLOSED_EP32_SAME_HOST'))
    return prior,streams,dict(link={p:dict(path=str(LINK_ROOT/p),sha256=h) for p,h in LINK_PINS.items()},
        component_gates=request['prerequisites'],old32_counts_preserved=True)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=gate.source.execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    prior,streams,pins=prerequisites();records,restore=inputs()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and
        r['execution']==e and r['prerequisites']==pins and r['existing32_streams']==streams and
        r['records']==records and r['restoration']==restore,'Frozen missing68 request or input closure differs')
    g.require(r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Historical visual input contract changed')
    runner.check_tools(r['tool_bindings'])
    g.require(r['actual_host_native_binding']==g.native_pin() and
        r['device_identity_binding']==runner.device_from_request(r).device_identity_binding(), 'Host/device binding changed')
    g.require(r['automatic_successor'] is False and r['training_updates']==r['PHY_calls']==r['quality_scores']==0 and
        r['new_source_indices']==list(range(32,100)), 'Bounded source-only scope changed')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh source68 output required')
    e=gate.source.execution(a.deadline_unix,a.max_seconds);prior,streams,pins=prerequisites();records,restore=inputs()
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256)
    device=gate.source.host.DeviceAdapter(dp,a.target_index);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,execution=e,caps=CAPS,
        prerequisites=pins,existing32_streams=streams,records=records,restoration=restore,
        new_source_indices=list(range(32,100)),device_receipt=dp,target_index=a.target_index,
        device_identity_binding=device.device_identity_binding(),actual_host_native_binding=g.native_pin(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        automatic_successor=False,training_updates=0,PHY_calls=0,quality_scores=0,
        preparation_model_calls=0,preparation_source_array_reads=0,
        purpose='Only missing68 original calibration source TX; preserve closed32 and48 budgets')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual source68 owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Source68 stage already claimed; no replay');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,execution=e,caps=CAPS))
        try:
            _,_,_,_,extra=g.validate_spec(spec)
            snapshot,ad=runner.wait_prelaunch(device,shared,out,e,started,120)
            g.write(out/'prelaunch_resources.json',snapshot);g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
            _,controlled=shared.controlled_environment(g.BASE,out,6);env=dict(controlled);(out/'home').mkdir()
            libraries=extra['environment']['LD_LIBRARY_PATH']
            for directory in libraries.split(':'):g.require(g.inside(directory).is_dir(),'Unadmitted native library directory')
            env.update(PATH=str(Path(spec['python']).parent),LD_LIBRARY_PATH=libraries,HOME=str(out/'home'),LANG='C.UTF-8',LC_ALL='C.UTF-8',
                VIRTUAL_ENV=str(Path(spec['python']).parent.parent),CUBLAS_WORKSPACE_CONFIG=':4096:8',**device.cuda_environment_binding())
            g.write(out/'controlled_environment.json',env)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Deadline expired before launch')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,
                    cpu_affinity=ad['cpu_affinity'],started_unix=time.time()))
                waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Source68 child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and done['counts']['completed']==CAPS and
                done['counts']['reserved']==CAPS and done['counts']['unresolved']==0 and len(done['new68_streams'])==68,
                'Source68 actual completion or counts differ')
            registration(path,a.request_sha256,runner);g.validate_spec(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Owner final deadline exceeded')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=core.descriptor(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],caps=CAPS,automatic_successor=False)
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
    device=runner.device_from_request(r);device.check_cuda_environment()
    launch=runner.worker_identity(out,a.request_sha256,a.owner_pid);g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    stopped=False;started=time.monotonic();ledger=None;backend=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'Source68 stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Source68 execution deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_pilot68_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        for name in ('t1_codec_runtime.py','t1_entropy_core.py'):
            relative='experiments/wcl-evidence-closure-20261009/scripts/'+name;actual=g.inside(Path(spec['project_root'])/relative)
            g.require(g.sha(actual)==g.sha(resolver.path(g.OLD+relative))==g.CODE[relative],'Original arithmetic source changed')
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(spec['project_root']);codec.native=backend.native;codec.provider_factory=backend.codec.provider_factory
        (out/'sources').mkdir();completed=[]
        for record in r['records'][32:]:
            completed.append(core.source68(record,backend,codec,partial,ledger,gate.source.load_tokens,out/'sources'/f"{record['source_index']:04d}"))
            g.write(out/f'progress_{len(completed):02d}.json',dict(completed_new_sources=len(completed),counts=ledger.summary()))
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        counts=ledger.summary();g.require(counts['completed']==counts['reserved']==CAPS and counts['unresolved']==0,'Source68 exact budget did not close')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,counts=counts,
            existing32_streams=r['existing32_streams'],new68_streams=completed,source_count=100,
            independent_RX_claim_for_new68=False,PHY_calls=0,quality_scores=0,old_gate_counts_preserved=True,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            request_sha256=a.request_sha256,counts=None if ledger is None else ledger.summary()));raise


def engine():
    g.require(g.sha(core.__file__)==CORE_SHA,'Frozen pilot core changed')
    ns=dict(vars(gate.source.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',type=int,choices=(0,2),required=True);q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=3600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
