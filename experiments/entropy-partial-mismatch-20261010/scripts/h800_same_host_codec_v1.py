"""One H800 source TX followed by independent same-host m4 RX; no successor."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import os
import signal
import time
import traceback
import types
import h800_single_rx_v1 as host

g=host.HOST
SCHEMA='H800_SAME_HOST_CODEC_DIAGNOSTIC_V1'
PASS='PASS_H800_SOURCE0_M4_SAME_HOST_CODEC_ONLY'
CAPS=dict(model_load=1,encoder=0,source_tx=1,source_rx=1,var_render=0,prior_scale=13,decoder_forward=0)
TOOLS=host.TOOLS+('h800_single_rx_v1.py',)
HOST_SOURCE_SHA='d2a8410f02078ba44f08d9e2e490174fb6f4549f60df2b4300a56592e6fd2f4a'
CROSS_ROOT=host.RT/'qualification/h800_single_rx_v1_attempt1'
CROSS_REQUEST_SHA='72c4f6c1b345230c0a7e8298622b4c4edbd9b4e43c08a184c306046aa20ad7c4'
CROSS_PINS={
    'run_owner_actual_wait.json':'2e287dc90b75f6809b2cc3773fdd133007825f2dab96c32c8858e82d8eda4b57',
    'registered/run/failure.json':'9e14b0e5ca5133953a297e8ebe5fbacb1eb64cf34200fb5f2f58e2088c813f5d',
    'registered/run/actual_child_wait.json':'0923eff6767843d1cc6bf76f9f3a0661baef44acea81e0f3b3ad5ab1a7c8696a',
    'registered/run/worker_failure.json':'d927ee9ce96752685bbd1e7d32b51d9ed3a79b9c42e67bc58f29af22b61b419e',
    'registered/run/first_CDF_mismatch.json':'30e28f201013dc25d07d73c9bef19f480cf47ee62d14c4b7115394c2b49ca2a8',
}


def cross_failure():
    request=g.checked_json(g.inside(CROSS_ROOT/'registered/request.json'),CROSS_REQUEST_SHA)
    g.require(request['schema']==host.SCHEMA and request['actual_host_native_binding']==g.native_pin(),
        'Fixed cross-host request differs')
    data={name:g.checked_json(g.inside(CROSS_ROOT/name),digest) for name,digest in CROSS_PINS.items()}
    owner=data['run_owner_actual_wait.json'];wait=data['registered/run/actual_child_wait.json']
    g.require(owner['actual_wait'] is True and owner['returncode']==1 and wait['actual_child_waited'] is True and
        wait['child_exit_code']==1 and wait['automatic_retry'] is False,'Cross-host attempt not actually closed')
    counts=data['registered/run/worker_failure.json']['counts']
    completed=dict(model_load=1,encoder=0,source_tx=0,source_rx=0,var_render=0,prior_scale=1,decoder_forward=0)
    g.require(counts['completed']==completed and counts['reserved']==completed|dict(source_rx=1) and counts['unresolved']==1,
        'Cross-host failed ledger differs')
    mismatch=data['registered/run/first_CDF_mismatch.json']
    g.require(mismatch['source']==0 and mismatch['role']=='RX' and mismatch['m']==4 and mismatch['scale']==0 and
        mismatch['cdf_sha256']!=mismatch['expected_TX_CDF_sha256'],'Closed cross-host CDF failure changed')
    return request,dict(request=dict(path=str(CROSS_ROOT/'registered/request.json'),sha256=CROSS_REQUEST_SHA),
        receipts={name:dict(path=str(CROSS_ROOT/name),sha256=digest) for name,digest in CROSS_PINS.items()},
        status='PRESERVED_CROSS_HOST_CDF_MISMATCH_NO_RETRY')


def source_pin(record,resolver):
    asset=record['asset'];path=resolver.path(asset['archive'])
    return dict(source_id=g.IDS[0],source_index=0,asset=copy.deepcopy(asset),
        actual_archive=dict(path=str(path),sha256=g.sha(path)))


def load_tx_tokens(record,resolver,pin):
    """Source-side token input only; pixels and old streams are never read."""
    import numpy as np
    g.require(source_pin(record,resolver)==pin,'Frozen source pin differs before TX')
    with np.load(resolver.path(record['asset']['archive']),allow_pickle=False) as archive:
        tokens=archive['tokens'].copy()
    g.require(tokens.dtype==np.int64 and tokens.shape==(680,) and ((tokens>=0)&(tokens<4096)).all(),
        'Frozen source token array malformed')
    g.require(g.hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()==record['asset']['tokens_sha256'],
        'Frozen TX tokens differ')
    return tokens


def exercise(tokens,backend,ledger,expected_reader,out,rx_exercise):
    """Actual TX bits only enter independent RX; no old bit length or CDF pin."""
    import numpy as np
    out=Path(out);out.mkdir()
    endpoints=ledger.call('source_tx',lambda:backend.source_tx(tokens),source=0)
    g.require(set(endpoints)==set(range(4,10)),'Original TX endpoint scope differs')
    traces=copy.deepcopy(backend.traces)
    g.require(len(traces)==9 and [r['scale'] for r in traces]==list(range(9)) and all(
        r['source']==0 and r['role']=='TX' and r['m']==9 and g.SHA_RE.fullmatch(r['cdf_sha256']) for r in traces),
        'Nine actual H800 TX CDF witnesses required')
    bits=endpoints[4]['bits'].copy()
    g.require(bits.dtype==np.uint8 and bits.ndim==1 and 0<len(bits)<=1048576 and np.isin(bits,(0,1)).all(),
        'Actual same-host m4 bit vector malformed')
    archive=out/'actual_h800_tx_m4.npz'
    with archive.open('xb') as stream:np.savez(stream,bits=bits)
    hashes={row['scale']:row['cdf_sha256'] for row in traces if row['scale']<4}
    metadata=dict(source_index=0,m=4,payload_bits=len(bits),actual_bits_sha256=g.image_sha(bits),
        archive=dict(path=str(archive),sha256=g.sha(archive)),actual_TX_CDF_trace=traces,
        old_bits_used=False,old_TX_CDF_hashes_used=False,probability_tables_shared_with_RX=False)
    g.write(out/'actual_h800_tx_m4.json',metadata)
    # Clear trace instrumentation only. SourceCodec.decode constructs its own
    # IndependentProvider; neither source tokens nor TX probability tables enter it.
    backend.traces=[]
    check=rx_exercise(bits,hashes,backend,ledger,expected_reader,out/'independent_rx')
    return dict(actual_TX=metadata,independent_RX_check=check)


def registration(path,digest):
    r=g.checked_json(g.inside(path),digest)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS,
        'Wrong independent same-host registration')
    g.require(r['source_index']==0 and r['source_id']==g.IDS[0] and r['m']==4 and r['prelaunch_wait_seconds']==120,
        'Same-host source/window scope differs')
    g.require(r['automatic_successor'] is False and r['production_admission'] is False and r['timing_measurement'] is False,
        'No production or successor permission')
    runner=engine();runner.check_tools(r['tool_bindings'])
    prior,pins=cross_failure();g.require(r['cross_host_failure']==pins,'Historical cross-host binding differs')
    g.require(r['parent_request']==prior['parent_request'],'Original v5 parent binding differs')
    expected=runner.inherited_spec(prior,r['spec']['deadline_unix'],r['spec']['max_seconds'])
    g.require(r['spec']==expected and r['actual_host_native_binding']==g.native_pin(),'Only independent deadline may change')
    device=runner.device_from_request(r)
    g.require(r['device_identity_binding']==device.device_identity_binding(),'Current device binding changed')
    g.require(r['source_pin']['source_index']==0 and r['source_pin']['source_id']==g.IDS[0],
        'Wrong frozen TX source identity')
    return r


def prepare(args,runner):
    out=g.inside(args.out);g.require(not out.exists(),'Fresh same-host output required')
    prior,pins=cross_failure();spec=runner.inherited_spec(prior,args.deadline_unix,args.max_seconds)
    parent_pin=dict(path=str(g.inside(args.parent_request)),sha256=args.parent_request_sha256)
    g.require(parent_pin==prior['parent_request'],'Original v5 parent binding differs')
    g.prepared(parent_pin['path'],parent_pin['sha256'])
    device_pin=dict(path=str(g.inside(args.device_receipt)),sha256=args.device_receipt_sha256)
    device=host.DeviceAdapter(device_pin,args.target_index)
    resolver,_,records,_,_=g.validate_spec(spec,verify_environment=False)
    source=source_pin(records[0],resolver);out.mkdir(parents=True)
    request=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',spec=spec,caps=CAPS,cross_host_failure=pins,parent_request=parent_pin,
        source_pin=source,source_index=0,source_id=g.IDS[0],m=4,device_receipt=device_pin,target_index=args.target_index,
        device_identity_binding=device.device_identity_binding(),actual_host_native_binding=g.native_pin(),
        prelaunch_wait_seconds=120,created_unix=time.time(),
        tool_bindings={name:g.sha(Path(__file__).with_name(name)) for name in (*TOOLS,Path(__file__).name)},
        purpose='Actual H800 TX to independent same-host RX; no old-host bitstream compatibility claim',
        budget_scope='Separate model1/TX1/RX1/prior13 migration diagnostic; no EP32/48 budget change',
        preparation_model_calls=0,preparation_source_arrays_read=0,automatic_successor=False,
        production_admission=False,timing_measurement=False)
    path=out/'request.json';g.write(path,request);return runner.descriptor(path)


def worker(args,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual worker requires Linux')
    path=g.inside(args.request);r=registration(path,args.request_sha256);device=runner.device_from_request(r)
    device.check_cuda_environment();spec=r['spec'];out=path.parent/'run'
    launch=runner.worker_identity(out,args.request_sha256,args.owner_pid)
    g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    stopped=False;started=time.monotonic();ledger=None;backend=None;shared=g.helper();last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==args.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'Diagnostic stop or parent loss')
        g.require(time.time()<spec['deadline_unix'] and time.monotonic()-started<spec['max_seconds'],'Diagnostic deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(shared);last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=args.request_sha256,started_unix=time.time()))
        resolver,environment,records,_,extra=g.validate_spec(spec);boundary()
        tokens=load_tx_tokens(records[0],resolver,r['source_pin'])
        ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_same_host_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        checks=exercise(tokens,backend,ledger,lambda:runner.expected_tokens_after_decode(records[0],resolver),
            out/'checks',runner.exercise)
        backend.close();resolver.reverify();environment.reverify();registration(path,args.request_sha256);guard()
        counts=ledger.summary();g.require(counts['completed']==CAPS and counts['unresolved']==0,'Actual same-host budget not closed')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=args.request_sha256,counts=counts,checks=checks,
            old_CDF='UNOBSERVABLE',v5_cross_host_verdict='FAIL_UNCHANGED',h800_cross_host_CDF_failure='PRESERVED_UNCHANGED',
            actual_same_host_TX_RX=True,original32_and48_budgets_unchanged=True,
            comparison_tokens_supplied_to_RX=False,probability_tables_shared=False,
            production_admission=False,timing_measurement=False,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=args.request_sha256,
            error=repr(error),traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary(),
            v5_cross_host_verdict='FAIL_UNCHANGED',h800_cross_host_CDF_failure='PRESERVED_UNCHANGED'))
        raise


def engine():
    g.require(g.sha(Path(host.__file__))==HOST_SOURCE_SHA,'Frozen H800 host adapter changed')
    ns=dict(vars(host.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('run','worker_identity','wait_prelaunch','descriptor','device_from_request','check_tools',
            'adapt_backend','exercise','expected_tokens_after_decode','inherited_spec'):
        ns[name]=host.clone_function(ns[name],ns)
    ns['registration']=registration
    return types.SimpleNamespace(**ns)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--parent-request',required=True);q.add_argument('--parent-request-sha256',required=True)
    q.add_argument('--device-receipt',required=True);q.add_argument('--device-receipt-sha256',required=True)
    q.add_argument('--target-index',type=int,choices=(0,2),required=True);q.add_argument('--out',required=True)
    q.add_argument('--deadline-unix',type=float,required=True);q.add_argument('--max-seconds',type=int,default=900)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args();runner=engine()
    value=prepare(args,runner) if args.command=='prepare' else runner.run(args) if args.command=='run' else worker(args,runner)
    print(g.json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
