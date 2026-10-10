"""First32 calibration source qualification: fresh H800 TX and independent RX.

No PHY, render, training, selection or automatic successor. The historical
two-source spec is an input closure only; execution has its own finite window.
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
import h800_same_host_codec_v1 as same

host=same.host
g=host.HOST
SCHEMA='H800_EP32_PARTIAL_SOURCE_GATE_V1'
PASS='PASS_H800_FIRST32_PARTIAL_SOURCE_CODEC_ONLY'
CAPS=dict(model_load=1,encoder=0,source_tx=32,source_rx=640,var_render=0,prior_scale=4928,decoder_forward=0)
TOOLS=same.TOOLS+('h800_same_host_codec_v1.py','ep_source_codec.py')
SOURCE_PINS={**host.SOURCE_SHA,'h800_single_rx_v1.py':same.HOST_SOURCE_SHA,
    'h800_same_host_codec_v1.py':'04972f2c45d88e61d5d6d1abe528ec3db2838efa045171e520dd59db0171782f',
    'ep_source_codec.py':'65a9f599bef2205c15f864f40ec07f27d8c9d03ed2d318838706f29dd1b3cb4e'}
PROTOCOL_SHA='c458844e95b0e448352067356edb8bd2328d4f881ef1bf0bce87ef7dd4fa62be'
ASSET_ROOT=host.RT/'assets/fixed_existing32_seed_v1/VAR_COMM/outputs/CONTENT-REAL-64QAM-20261006/H/full1000_assets'
OLD_ASSET=g.OLD+'outputs/CONTENT-REAL-64QAM-20261006/H/full1000_assets/'
METADATA_PINS={str(ASSET_ROOT/'manifest.json'):'49bb6b8dfb45ac9fec5968481fe7f1691959a80964972bad1c0bde4e468e3cae',
    str(ASSET_ROOT/'completion.json'):'9760c3f946a10abb1dd7fdc050cfc96832b3e2a31f359f86e7fb2504393a335b',
    str(host.RT/'receipts/fixed_existing32_seed_verified_v1.json'):'db3b531f1b264f68b9e3f84c5232ab0ed47777fd981bd1b2941e28d3e22a2fd9'}
SAME_ROOT=host.RT/'qualification/h800_same_host_codec_v1_attempt1'
SAME_PINS={'run_owner_actual_wait.json':'0586534a4c253fd3172f30fac46af4af3b835db2bdb4644be6120668dadee9c6',
    'registered/run/completion.json':'e358ad3a6f46479e41e168103c8c8bbc2d8d9a80d84bcb2bcc875a8b62bf36dd',
    'registered/run/worker_completion.json':'c649240f1379e77213ad23d0ca435bd4d422cc64191ff296c15db2efff59aa41'}
SAME_REQUEST_SHA='0ff19f5420f2ffc15d2f48c6f24a6a226861b875cde7fdd068bdf29d49ccdc78'


def execution(deadline,seconds):
    g.require(type(seconds) is int and 0<seconds<=3600 and type(deadline) in (int,float) and time.time()<deadline,
        'Independent at-most3600-second EP32 window required')
    return dict(deadline_unix=deadline,max_seconds=seconds,input_contract_is_not_execution_budget=True)


def prerequisites():
    prior,cross=same.cross_failure()
    request=g.checked_json(g.inside(SAME_ROOT/'registered/request.json'),SAME_REQUEST_SHA)
    values={name:g.checked_json(g.inside(SAME_ROOT/name),digest) for name,digest in SAME_PINS.items()}
    owner=values['run_owner_actual_wait.json'];done=values['registered/run/completion.json']
    worker=values['registered/run/worker_completion.json']
    g.require(owner['actual_wait'] is True and owner['returncode']==0 and done['status']==worker['status']==same.PASS,
        'Actual closed same-host prerequisite required')
    g.require(done['actual_children_waited'] is True and done['worker_exit_codes']==[0] and
        worker['counts']['completed']==worker['counts']['reserved']==same.CAPS and worker['counts']['unresolved']==0,
        'Same-host prerequisite counters changed')
    g.require(request['parent_request']==prior['parent_request'],'Same-host parent changed')
    return prior,dict(cross_host_failure=cross,same_host_request=dict(path=str(SAME_ROOT/'registered/request.json'),sha256=SAME_REQUEST_SHA),
        same_host_receipts={n:dict(path=str(SAME_ROOT/n),sha256=h) for n,h in SAME_PINS.items()})


def input_map():
    """Read pinned metadata and hash only the exact first32 original archives."""
    manifest,completion,seed=[g.checked_json(g.inside(p),h) for p,h in METADATA_PINS.items()]
    g.require(manifest['source_count']==completion['source_count']==1000 and len(manifest['records'])==1000 and
        len(manifest['source_ids'])==1000,'Original calibration population changed')
    rows=manifest['records'][:32]
    g.require([r['source_index'] for r in rows]==list(range(32)) and len({r['source_id'] for r in rows})==32 and
        [r['source_id'] for r in rows]==manifest['source_ids'][:32],'First32 ordering or identity differs')
    mapping={r['original']:r for r in seed['mapping']}
    g.require(len(mapping)==len(seed['mapping']),'Ambiguous restored input mapping')
    result=[]
    for i,row in enumerate(rows):
        old_checkpoint=OLD_ASSET+f'source_checkpoints/{i:04d}.json';old_archive=OLD_ASSET+f'sources/{i:04d}.npz'
        g.require(row['checkpoint']==old_checkpoint and row['archive']==old_archive,'Unexpected first32 input path')
        checkpoint_path=g.inside(ASSET_ROOT/f'source_checkpoints/{i:04d}.json')
        checkpoint=g.checked_json(checkpoint_path,row['checkpoint_sha256'])
        restored=mapping[old_archive];archive=g.inside(ASSET_ROOT/f'sources/{i:04d}.npz')
        g.require(restored['restored']==str(archive) and restored['sha256']==completion['outputs'][old_archive]==
            checkpoint['outputs'][old_archive] and completion['outputs'][old_checkpoint]==row['checkpoint_sha256'],
            'Source archive/checkpoint receipt chain differs')
        for key in ('source_index','source_id','preprocessing_id','tokens_sha256','archive'):
            g.require(checkpoint[key]==row[key],'Source checkpoint identity differs: '+key)
        g.require(checkpoint['original_calibration_index']==i and g.SHA_RE.fullmatch(row['tokens_sha256']),
            'Not the original calibration order/token pin')
        g.require(archive.is_file() and archive.stat().st_size==restored['bytes'] and g.sha(archive)==restored['sha256'],
            'Frozen first32 source archive bytes changed')
        result.append(dict(source_index=i,source_id=row['source_id'],preprocessing_id=row['preprocessing_id'],
            tokens_sha256=row['tokens_sha256'],archive=dict(path=str(archive),sha256=restored['sha256'],bytes=restored['bytes']),
            checkpoint=dict(path=str(checkpoint_path),sha256=row['checkpoint_sha256'])))
    return dict(population='ORIGINAL_CALIBRATION_FIRST32_ORDERED',metadata=METADATA_PINS,records=result,
        source_arrays_read_during_mapping=0,old_codec_archives_used=False)


def load_tokens(record):
    import numpy as np
    pin=record['archive'];path=g.inside(pin['path'])
    g.require(g.sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Source bytes changed before token read')
    with np.load(path,allow_pickle=False) as archive:tokens=archive['tokens'].copy()
    g.require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),
        'Frozen source token array malformed')
    g.require(g.hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==record['tokens_sha256'],
        'Frozen first32 token hash differs')
    return tokens


def endpoint_scope(codec):
    tx=codec.all_endpoints();rx=tuple(p for p in tx if p[1]>0)+((4,0),(5,0))
    g.require(len(tx)==24 and len(set(tx))==24 and len(rx)==20 and sum(p[1]>0 for p in rx)==18,
        'Registered endpoint scope differs')
    g.require(32*(10+sum(m+int(K>0) for m,K in rx))==CAPS['prior_scale'],'Frozen prior budget differs')
    return tx,rx


def exercise_source(record,backend,codec,module,ledger,out):
    """Only new actual bits and endpoint fields enter fresh independent RX."""
    import numpy as np
    out=Path(out);out.mkdir();i=record['source_index'];tx_scope,rx_scope=endpoint_scope(module)
    backend.source_index=i;backend.current=('TX',10);backend.traces=[];backend.tx_cdfs={};backend.received={}
    tokens=load_tokens(record);backend.boundary()
    encoded=ledger.call('source_tx',lambda:codec.encode_endpoints(tokens,tx_scope),source=i)
    del tokens
    g.require(set(encoded)==set(tx_scope),'Missing or extra actual TX endpoint')
    tx_trace=copy.deepcopy(backend.traces)
    g.require(len(tx_trace)==10 and [r['scale'] for r in tx_trace]==list(range(10)) and all(
        r['source']==i and r['role']=='TX' and r['m']==10 and g.SHA_RE.fullmatch(r['cdf_sha256']) for r in tx_trace),
        'Ten actual TX prior witnesses required')
    streams={};metadata=[]
    for m,K in tx_scope:
        item=encoded[m,K];bits=item['bits']
        g.require(bits.dtype==np.uint8 and bits.ndim==1 and 2<=len(bits)<=1048576 and np.isin(bits,(0,1)).all() and
            item['arithmetic_bits']==len(bits) and item['m']==m and item['K']==K,'Actual endpoint bits malformed')
        streams[f'm{m}_K{K}']=bits.copy()
        metadata.append(dict(m=m,K=K,payload_bits=len(bits),bits_sha256=g.image_sha(bits),independent_RX_planned=(m,K) in rx_scope))
    path=out/'actual_streams.npz'
    with path.open('xb') as f:np.savez(f,**streams)
    g.write(out/'actual_TX.json',dict(source_index=i,archive=dict(path=str(path),sha256=g.sha(path)),endpoints=metadata,
        CDF_trace=tx_trace,old_host_bits_used=False,old_host_CDF_used=False))
    checks=[]
    for m,K in rx_scope:
        backend.current=('RX',m);backend.traces=[];backend.boundary()
        family=module.PARTIAL_FAMILY if K else module.WHOLE_FAMILY
        rx=ledger.call('source_rx',lambda:codec.decode(family,streams[f'm{m}_K{K}'].copy(),m,K),source=i,m=m,K=K)
        g.require(rx['canonical'] is True and rx['zero_extension_reads']==30,'Independent canonical decode failed')
        rows=copy.deepcopy(backend.traces);scales=m+int(K>0)
        g.require(len(rows)==scales and [r['scale'] for r in rows]==list(range(scales)) and all(
            r['source']==i and r['role']=='RX' and r['m']==m and r['cdf_sha256']==backend.tx_cdfs[r['scale']] for r in rows),
            'Independent RX CDF witnesses differ')
        # Truth is read again only after decode; never an argument to its provider.
        expected=load_tokens(record)[:module.token_count(m,K)].copy()
        check=g.exact(rx['received_tokens'],expected,f's{i}_m{m}_K{K}',out)
        checks.append(dict(m=m,K=K,check=check,CDF_trace=rows))
    result=dict(source_index=i,source_id=record['source_id'],status='SOURCE_COMPLETE',TX_passes=1,
        partial_RX=18,short_whole_RX=2,prior_scales=154,checks=checks,
        probability_tables_shared=False,comparison_tokens_supplied_to_RX=False)
    g.write(out/'completion.json',result);backend.traces=[];backend.tx_cdfs={};backend.received={};backend.boundary()
    return dict(source_index=i,path=str(out/'completion.json'),sha256=g.sha(out/'completion.json'))


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest);runner=engine()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS,'Wrong EP32 registration')
    e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds']);g.require(r['execution']==e,'Execution contract changed')
    prior,pins=prerequisites();g.require(r['prerequisites']==pins and r['parent_request']==prior['parent_request'],'Prerequisite changed')
    g.require(r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Historical input contract changed')
    g.require(r['input_map']==input_map(),'First32 frozen input closure changed')
    runner.check_tools(r['tool_bindings'])
    g.require(r['actual_host_native_binding']==g.native_pin() and
        r['device_identity_binding']==runner.device_from_request(r).device_identity_binding(),'Host binding changed')
    g.require(r['prelaunch_wait_seconds']==120 and r['automatic_successor'] is False and r['production_admission'] is False and
        r['timing_measurement'] is False,'Scope/automatic continuation changed')
    return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh EP32 output required')
    e=execution(a.deadline_unix,a.max_seconds);prior,pins=prerequisites()
    parent=dict(path=str(g.inside(a.parent_request)),sha256=a.parent_request_sha256)
    g.require(parent==prior['parent_request'],'Original parent binding differs')
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    device_pin=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256)
    device=host.DeviceAdapter(device_pin,a.target_index);inputs=input_map();out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,execution=e,caps=CAPS,prerequisites=pins,parent_request=parent,
        input_map=inputs,device_receipt=device_pin,target_index=a.target_index,device_identity_binding=device.device_identity_binding(),
        actual_host_native_binding=g.native_pin(),prelaunch_wait_seconds=120,created_unix=time.time(),
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        purpose='Original calibration first32: fresh actual H800 partial TX and independent RX qualification',
        budget_scope='Frozen EP source gate32TX/576partialRX/64m4m5wholeRX/4928prior; no48frame image or PHY calls',
        preparation_model_calls=0,preparation_source_arrays_read=0,automatic_successor=False,production_admission=False,timing_measurement=False)
    path=out/'request.json';g.write(path,r);return runner.descriptor(path)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual owner requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256);spec=r['spec'];e=r['execution'];out=path.parent/'run'
    shared=g.helper();device=runner.device_from_request(r);child=None;started=time.monotonic()
    with shared.owner_lock(g.inside(g.BASE/('controls/leo_whole_gate_v1/GPU'+str(device.index)+'.owner.lock'))):
        g.require(not out.exists(),'EP32 already claimed; no replay');out.mkdir()
        g.write(out/'intent.json',dict(owner_pid=os.getpid(),request_sha256=a.request_sha256,started_unix=time.time(),caps=CAPS,execution=e))
        try:
            _,_,_,_,extra=g.validate_spec(spec)
            snapshot,ad=runner.wait_prelaunch(device,shared,out,e,started,r['prelaunch_wait_seconds'])
            g.write(out/'prelaunch_resources.json',snapshot);g.write(out/'prelaunch_device_identity.json',device.query_nvml_device_identity())
            _,controlled=shared.controlled_environment(g.BASE,out,6);env=dict(controlled);(out/'home').mkdir()
            libraries=extra['environment']['LD_LIBRARY_PATH']
            for d in libraries.split(':'):g.require(g.inside(d).is_dir(),'Unadmitted native library directory')
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
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'EP32 child failed; preserve ledger without retry')
            done=g.checked_json(out/'worker_completion.json',g.sha(out/'worker_completion.json'))
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and done['counts']['completed']==CAPS and
                done['counts']['unresolved']==0 and len(done['source_completions'])==32,'Completion or counters differ')
            registration(path,a.request_sha256);g.validate_spec(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Owner final deadline exceeded')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=runner.descriptor(out/'worker_completion.json'),
                actual_children_waited=True,worker_exit_codes=[child.returncode],actual_wait=waited,caps=CAPS,
                old_cross_host_failure='PRESERVED_UNCHANGED',production_admission=False,timing_measurement=False,automatic_successor=False)
            g.write(out/'completion.json',result);return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise


def worker(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual worker requires Linux')
    path=g.inside(a.request);r=registration(path,a.request_sha256);device=runner.device_from_request(r)
    device.check_cuda_environment();spec=r['spec'];e=r['execution'];out=path.parent/'run'
    launch=runner.worker_identity(out,a.request_sha256,a.owner_pid);g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    stopped=False;started=time.monotonic();ledger=None;backend=None;shared=g.helper();last=0.;completed=[]
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'EP32 stop or parent loss')
        g.require(time.time()<e['deadline_unix'] and time.monotonic()-started<e['max_seconds'],'EP32 execution deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(shared);last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256,started_unix=time.time()))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_ep32_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        # Bind the unchanged original source arithmetic before importing the extension.
        for name in ('t1_codec_runtime.py','t1_entropy_core.py'):
            relative='experiments/wcl-evidence-closure-20261009/scripts/'+name
            actual=g.inside(Path(spec['project_root'])/relative)
            g.require(g.sha(actual)==g.sha(resolver.path(g.OLD+relative))==g.CODE[relative],'Original source arithmetic differs')
        import ep_source_codec as partial
        g.require(g.sha(partial.__file__)==SOURCE_PINS['ep_source_codec.py'],'Partial extension changed')
        codec=partial.PartialSourceCodec(spec['project_root']);codec.native=backend.native;codec.provider_factory=backend.codec.provider_factory
        (out/'sources').mkdir()
        for record in r['input_map']['records']:
            completed.append(exercise_source(record,backend,codec,partial,ledger,out/'sources'/f"{record['source_index']:04d}"))
            with (out/'source_progress.jsonl').open('a') as stream:stream.write(g.json.dumps(dict(completed_sources=len(completed),
                latest=completed[-1],counts=ledger.summary(),time_unix=time.time()))+'\n')
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256);guard()
        counts=ledger.summary();g.require(counts['completed']==counts['reserved']==CAPS and counts['unresolved']==0,'EP32 budget not closed exactly')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,counts=counts,source_completions=completed,
            actual_TX_passes=32,independent_partial_RX=576,independent_short_whole_RX=64,short_whole_m=[4,5],
            untested_same_host_whole_m=[6,7,8,9],comparison_tokens_supplied_to_RX=False,probability_tables_shared=False,
            old_host_bits_used=False,old_host_CDF_used=False,old_cross_host_failure='PRESERVED_UNCHANGED',
            PHY_calls=0,render_calls=0,production_admission=False,timing_measurement=False,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary(),source_completions=completed));raise


def engine():
    for name,digest in SOURCE_PINS.items():g.require(g.sha(Path(__file__).with_name(name))==digest,'Frozen source changed: '+name)
    g.require(g.sha(Path(__file__).parent.parent/'PROTOCOL.md')==PROTOCOL_SHA,'Frozen protocol changed')
    ns=dict(vars(same.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('worker_identity','wait_prelaunch','descriptor','device_from_request','check_tools','adapt_backend','inherited_spec'):
        ns[name]=host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--parent-request',required=True);q.add_argument('--parent-request-sha256',required=True)
    q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',type=int,choices=(0,2),required=True);q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=3600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine()
    result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
