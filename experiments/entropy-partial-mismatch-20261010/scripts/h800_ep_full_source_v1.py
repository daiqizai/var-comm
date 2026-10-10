"""Finite fresh TX for original calibration0100..0999; no automatic successor.

Preparation binds the original1000 assets and already closed H800 source100.
Execution is a separate explicit bounded owner. It encodes only the missing900;
no encoder, PHY, RX, rendering, metric evaluation or candidate selection occurs.
Full calibration requires the later complete pilot top3 union original whole
winner shortlist; this TX-only stage is not that selection or calibration run.
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
import numpy as np
import h800_ep_pilot_source_v1 as source100

core=source100.core
gate=source100.gate
g=source100.g
SCHEMA='H800_EP_FULL_SOURCE900_V1'
PASS='PASS_H800_ORIGINAL_CALIBRATION_SOURCE1000_AVAILABLE'
CAPS=dict(model_load=1,encoder=0,source_tx=900,source_rx=0,var_render=0,prior_scale=9000,decoder_forward=0)
RT=source100.RT
SOURCE100_ROOT=RT/'qualification/h800_ep_pilot_source68_v1_attempt1'
SOURCE100_PINS={
    'run_owner_actual_wait.json':'e99988b9b9676f92416179c63be5a3170c65fa8dc8e5c06f1a0c9ab4658180aa',
    'registered/request.json':'61af48c14678bb67bfede5ef601dcee443c2d8bde57b555043ca499b4c50c255',
    'registered/run/completion.json':'2479e9a4c89f55bb7741fd525bbe0f44778ebd8c923a3674401ba595319db0b1',
    'registered/run/worker_completion.json':'25cf3c6e8a945df074eba91d74c93b44d040934788a1f28d25e7bdaf9fe4d69a'}
RESTORE=RT/'incoming/fixed_existing900_seed_v1/hydration_completion.json'
RESTORE_SHA='215af2fe75a7471802700feabf2c8bb3135352b292b76cb41ef0f30aa8097373'
SOURCE100_CODE_SHA='661d6f1c8c45b3b324c8c488118ccb4c96c3ea8217281bd4d6bb0ebd014403e2'
CORE_SHA='3f0ea97b7902de4c24e80ebcfe731b160863ac252ba7ff81948e86b7ca0103cb'
TOOLS=source100.TOOLS+('h800_ep_pilot_source_v1.py',)
MANIFEST_SHA=core.MANIFEST_SHA
SOURCE_COMPLETION_SHA=core.SOURCE_COMPLETION_SHA
OLD_ASSET=core.OLD_ASSET
require=core.require
checked=core.checked
sha=core.sha
save=core.save
descriptor=core.descriptor

def full_input_records(manifest_pin, completion_pin, relocation, inside):
    """Exact original1000 metadata/archive closure, without reading image arrays."""
    require(manifest_pin['sha256'] == MANIFEST_SHA and completion_pin['sha256'] == SOURCE_COMPLETION_SHA,
            'Pinned original1000 calibration metadata required')
    manifest = checked(manifest_pin); completion = checked(completion_pin)
    require(manifest['source_count'] == completion['source_count'] == 1000 and
            len(manifest['records']) == len(manifest['source_ids']) == 1000,
            'Original calibration population changed')
    rows = manifest['records']; result = []
    require(set(relocation) == {OLD_ASSET + f'{folder}/{i:04d}.{suffix}' for i in range(1000) for folder,suffix in [('sources','npz'),('source_checkpoints','json')]}, 'Exact2000 original input mapping required')
    require([r['source_index'] for r in rows] == list(range(1000)) and
            [r['source_id'] for r in rows] == manifest['source_ids'] and
            len({r['source_id'] for r in rows}) == 1000, 'Original calibration full1000 ordering required')
    for i, row in enumerate(rows):
        old_cp = OLD_ASSET + f'source_checkpoints/{i:04d}.json'
        old_npz = OLD_ASSET + f'sources/{i:04d}.npz'
        require(row['checkpoint'] == old_cp and row['archive'] == old_npz,
                'Unexpected original calibration input path')
        cp_pin = relocation[old_cp]; archive = relocation[old_npz]
        require(cp_pin['sha256'] == row['checkpoint_sha256'] == completion['outputs'][old_cp] and
                archive['sha256'] == completion['outputs'][old_npz], 'Original relocation digest differs')
        inside(cp_pin['path']); npz_path = inside(archive['path']); cp = checked(cp_pin)
        require(npz_path.is_file() and sha(npz_path) == archive['sha256'], 'Restored original source bytes differ')
        require(cp['original_calibration_index'] == i and cp['outputs'][old_npz] == archive['sha256'] and all(
                cp[k] == row[k] for k in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256', 'archive')),
                'Original checkpoint identity or asset closure differs')
        require(type(cp['evaluation_class_index']) is int and 0 <= cp['evaluation_class_index'] < 1000,
                'Original evaluation class metadata malformed')
        result.append(dict(source_index=i, source_id=row['source_id'], preprocessing_id=row['preprocessing_id'],
            tokens_sha256=row['tokens_sha256'], evaluation_class_index=cp['evaluation_class_index'],
            archive=dict(path=str(npz_path), sha256=archive['sha256'], bytes=npz_path.stat().st_size),
            checkpoint=cp_pin))
    return result


def source900(record, backend, codec, partial, ledger, load_tokens, out):
    """Generate only missing H800 TX streams; no RX, encoder, render or scoring."""
    index = record['source_index']
    require(type(index) is int and 100 <= index < 1000, 'Only original calibration0100..0999 TX permitted')
    out = Path(out); out.mkdir()
    backend.source_index = index; backend.current = ('TX', 10)
    backend.traces = []; backend.tx_cdfs = {}; backend.received = {}
    backend.boundary()
    tokens = load_tokens(record)
    encoded = ledger.call('source_tx', lambda: codec.encode_endpoints(tokens, partial.all_endpoints()), source=index)
    del tokens
    trace = copy.deepcopy(backend.traces)
    require(len(trace) == 10 and [r['scale'] for r in trace] == list(range(10)) and all(
        r['source'] == index and r['role'] == 'TX' and r['m'] == 10 for r in trace),
        'Ten actual fresh TX priors required')
    endpoints = partial.all_endpoints()
    require(set(encoded) == set(endpoints) and len(endpoints) == 24, 'Exact24 fresh endpoints required')
    arrays = {}; metadata = []
    for m, K in endpoints:
        item = encoded[m, K]; bits = item['bits']
        require(bits.dtype == np.uint8 and bits.ndim == 1 and 2 <= len(bits) <= 1048576 and
                np.isin(bits, (0, 1)).all() and item['arithmetic_bits'] == len(bits) and
                item['m'] == m and item['K'] == K, 'Actual endpoint bits malformed')
        arrays[f'm{m}_K{K}'] = bits.copy()
        metadata.append(dict(m=m, K=K, payload_bits=len(bits), bits_sha256=backend.g.image_sha(bits)))
    archive = out / 'actual_streams.npz'
    with archive.open('xb') as stream: np.savez(stream, **arrays)
    actual = dict(source_index=index, archive=descriptor(archive), endpoints=metadata,
                  CDF_trace=trace, old_host_bits_used=False, old_host_CDF_used=False)
    path = out / 'actual_TX.json'; save(path, actual)
    result = dict(source_index=index, source_id=record['source_id'], metadata=descriptor(path),
                  archive=actual['archive'], actual_new_TX=1, actual_new_prior=10,
                  independent_RX_calls=0, status='FULL_SOURCE_FRESH_TX_COMPLETE_NO_RX_CLAIM')
    save(out / 'completion.json', result)
    backend.traces = []; backend.tx_cdfs = {}; backend.received = {}; backend.boundary()
    return result


def inputs():
    first100,prior_restore=source100.inputs()
    restore=g.checked_json(g.inside(RESTORE),RESTORE_SHA)
    g.require(restore['schema']=='CAL900_ORIGINAL_ASSETS_RESTORED_V1' and
        restore['source_indices']==list(range(100,1000)) and restore['file_count']==1800 and
        restore['source_manifest_sha256']==MANIFEST_SHA and restore['source_completion_sha256']==SOURCE_COMPLETION_SHA,
        'Exact original900 restoration required')
    relocation={}
    for row in first100:
        i=row['source_index'];relocation[OLD_ASSET+f'sources/{i:04d}.npz']=row['archive']
        relocation[OLD_ASSET+f'source_checkpoints/{i:04d}.json']=row['checkpoint']
    for row in restore['mapping']:
        key=row['original_path'];g.require(key not in relocation,'Restoration duplicates an existing input')
        relocation[key]=dict(path=row['actual_path'],sha256=row['sha256'])
    g.require(len(relocation)==2000,'Exact1000 original archives plus1000 checkpoints required')
    root=gate.source.ASSET_ROOT
    records=full_input_records(dict(path=str(root/'manifest.json'),sha256=MANIFEST_SHA),
        dict(path=str(root/'completion.json'),sha256=SOURCE_COMPLETION_SHA),relocation,g.inside)
    g.require(records[:100]==first100,'Previously completed original100 input mapping changed')
    return records,dict(path=str(RESTORE),sha256=RESTORE_SHA,prior68_restore=prior_restore)


def prerequisites():
    prior,first32,old_pins=source100.prerequisites()
    values={p:g.checked_json(g.inside(SOURCE100_ROOT/p),h) for p,h in SOURCE100_PINS.items()}
    wait=values['run_owner_actual_wait.json'];owner=values['registered/run/completion.json']
    worker=values['registered/run/worker_completion.json'];request=values['registered/request.json']
    g.require(wait['actual_wait'] is True and wait['returncode']==0 and owner['status']==source100.PASS and
        owner['actual_wait']['success'] is True and owner['actual_children_waited'] is True and
        owner['worker_exit_codes']==[0] and owner['request_sha256']==SOURCE100_PINS['registered/request.json'],
        'Actual closed source100 owner required')
    g.require(owner['worker_completion']==dict(path=str(SOURCE100_ROOT/'registered/run/worker_completion.json'),
        sha256=SOURCE100_PINS['registered/run/worker_completion.json']) and worker['status']==source100.PASS and
        worker['request_sha256']==owner['request_sha256'] and worker['source_count']==100 and
        worker['counts']['completed']==worker['counts']['reserved']==source100.CAPS and worker['counts']['unresolved']==0,
        'Completed source100 worker and exact68 budget required')
    g.require(request['existing32_streams']==worker['existing32_streams']==first32 and
        request['caps']==source100.CAPS and request['new_source_indices']==list(range(32,100)) and
        request['actual_host_native_binding']==g.native_pin(),'Closed source100 inherited binding changed')
    for name,digest in request['tool_bindings'].items():
        g.require(g.sha(Path(__file__).with_name(name))==digest,'Completed source100 bound source changed: '+name)
    stream_rows=worker['existing32_streams']+worker['new68_streams']
    g.require([x['source_index'] for x in stream_rows]==list(range(100)) and
        len({x['source_id'] for x in stream_rows})==100,'Exact ordered closed100 streams required')
    streams=[]
    for row in stream_rows:
        i=row['source_index'];meta=g.checked_json(g.inside(row['metadata']['path']),row['metadata']['sha256'])
        archive=row['archive'];g.require(meta['source_index']==i and meta['archive']==archive and
            row['source_id']==request['records'][i]['source_id'] and
            g.sha(g.inside(archive['path']))==archive['sha256'],'Reused100 actual source stream changed')
        streams.append(dict(source_index=i,source_id=row['source_id'],metadata=row['metadata'],archive=archive,
            actual_new_TX=0,origin='CLOSED_H800_SOURCE100',previous_call_counts_preserved=True))
    return prior,streams,dict(source100={p:dict(path=str(SOURCE100_ROOT/p),sha256=h) for p,h in SOURCE100_PINS.items()},
        previous_gates=old_pins,old32_and68_counts_preserved=True)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=gate.source.execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    prior,streams,pins=prerequisites();records,restore=inputs()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and
        r['execution']==e and r['prerequisites']==pins and r['existing100_streams']==streams and
        r['records']==records and r['restoration']==restore,'Frozen missing900 request or input closure differs')
    g.require(r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Historical visual input contract changed')
    runner.check_tools(r['tool_bindings'])
    g.require(r['actual_host_native_binding']==g.native_pin() and
        r['device_identity_binding']==runner.device_from_request(r).device_identity_binding(), 'Host/device binding changed')
    g.require(r['target_index']==2 and r['automatic_successor'] is False and r['training_updates']==r['PHY_calls']==r['quality_scores']==0 and
        r['new_source_indices']==list(range(100,1000)), 'Bounded source-only scope changed')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh source900 output required')
    g.require(a.target_index==2,'Source900 must use the source100 qualified physical GPU2')
    e=gate.source.execution(a.deadline_unix,a.max_seconds);prior,streams,pins=prerequisites();records,restore=inputs()
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256)
    device=gate.source.host.DeviceAdapter(dp,a.target_index);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,execution=e,caps=CAPS,
        prerequisites=pins,existing100_streams=streams,records=records,restoration=restore,
        new_source_indices=list(range(100,1000)),device_receipt=dp,target_index=a.target_index,
        device_identity_binding=device.device_identity_binding(),actual_host_native_binding=g.native_pin(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        automatic_successor=False,training_updates=0,PHY_calls=0,quality_scores=0,
        preparation_model_calls=0,preparation_source_array_reads=0,
        purpose='Only missing900 original calibration source TX; preserve closed32,68,48 budgets; no shortlist or full-calibration decisions')
    path=out/'request.json';g.write(path,r);return core.descriptor(path)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual source900 owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'Source900 stage already claimed; no replay');out.mkdir()
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
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Source900 child failed; no automatic retry')
            done=gate.readpin(core.descriptor(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and done['counts']['completed']==CAPS and
                done['counts']['reserved']==CAPS and done['counts']['unresolved']==0 and len(done['new900_streams'])==900,
                'Source900 actual completion or counts differ')
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
            'Source900 stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Source900 execution deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_full900_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        for name in ('t1_codec_runtime.py','t1_entropy_core.py'):
            relative='experiments/wcl-evidence-closure-20261009/scripts/'+name;actual=g.inside(Path(spec['project_root'])/relative)
            g.require(g.sha(actual)==g.sha(resolver.path(g.OLD+relative))==g.CODE[relative],'Original arithmetic source changed')
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(spec['project_root']);codec.native=backend.native;codec.provider_factory=backend.codec.provider_factory
        (out/'sources').mkdir();completed=[]
        for record in r['records'][100:]:
            completed.append(source900(record,backend,codec,partial,ledger,gate.source.load_tokens,out/'sources'/f"{record['source_index']:04d}"))
            g.write(out/f'progress_{len(completed):02d}.json',dict(completed_new_sources=len(completed),counts=ledger.summary()))
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        counts=ledger.summary();g.require(counts['completed']==counts['reserved']==CAPS and counts['unresolved']==0,'Source900 exact budget did not close')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,counts=counts,
            existing100_streams=r['existing100_streams'],new900_streams=completed,source_count=1000,
            independent_RX_claim_for_new900=False,PHY_calls=0,quality_scores=0,old_gate_counts_preserved=True,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            request_sha256=a.request_sha256,counts=None if ledger is None else ledger.summary()));raise


def engine():
    g.require(g.sha(core.__file__)==CORE_SHA,'Frozen pilot core changed')
    g.require(g.sha(source100.__file__)==SOURCE100_CODE_SHA,'Frozen source100 owner changed')
    ns=dict(vars(source100.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',type=int,choices=(2,),required=True);q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=3600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
