"""Finite raw KEEP-state reconstruction; VAR only, no Direct/PHY/metrics.

Actual received hard tokens are the only reconstruction input. Body CRC rejects
remain KEEP. Only a rejected paid header produces fixed0.5 gray. No old-host
image is admitted; identical current actual states may share a reconstruction.
"""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
import os
from pathlib import Path
import signal
import time
import traceback
import types
import numpy as np
import h800_ep_pilot_visual_v1 as original
import h800_mismatch_phy_v1 as physical

g=original.g;gate=original.gate;raw=physical.core
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_VISUAL_V1'
PASS='PASS_H800_RAW100_MISMATCH_RECONSTRUCTIONS_ONLY'
CAPS=dict(model_load=1,encoder=0,source_tx=0,source_rx=0,var_render=4500,prior_scale=45000,decoder_forward=4500)
PHY_SHA='3d01e32bb505d322dadf37aae8c45a7398171ddd38091393340d3b24e91c5ee8'
VISUAL_SHA='d790fb8787ceea4f8c200eb9edaacd226b1903a1a5293513adb6f4c1abb46f7f'
TOOLS=physical.TOOLS+('h800_mismatch_phy_v1.py','h800_ep_pilot_visual_v1.py')
execution=original.execution


def cpu_closure(pins):
    root=gate.RT/'qualification/h800_mismatch_phy_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact mismatch CPU attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==physical.PASS and done['actual_wait']['success'] is True and done['actual_children_waited'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,'Actual mismatch CPU owner wait required')
    request=g.checked_json(root/'registered/request.json',done['request_sha256']);worker=gate.readpin(done['worker_completion'])
    records,frames,binding,closure=physical.original_inputs();identity,_=physical.cpu_identity(binding)
    g.require(request['schema']==physical.SCHEMA and request['records']==records and request['frames']==frames and
        request['raw_runtime_binding']==binding and request['input_closure']==closure and request['CPU_identity']==identity and
        request['complete_scientific_caps']==physical.inputs.future_caps() and request['historical_reuse_admitted']==0,
        'Closed raw mismatch input identity changed')
    physical.engine().check_tools(request['tool_bindings'])
    result=worker['results'];counts=result['packet_ledger']
    g.require(worker['status']==physical.PASS and worker['request_sha256']==done['request_sha256'] and
        worker['CUDA_initialized'] is False and result['logical_count']==5400 and result['physical_count']==4500 and
        len(result['physical_frames'])==4500 and counts['cap']==9000 and counts['unresolved']==0 and counts['total']<=9000 and
        result['historical_reuse_admitted']==0,'Complete finite raw mismatch receive outputs required')
    logical=gate.readpin(result['logical_frames'])
    g.require(len(logical)==5400 and [r['logical_index'] for r in logical]==list(range(5400)), 'Exact5400 logical mappings required')
    for row,event in zip(logical,frames):
        g.require(row['logical_frame_id']==raw.digest(event) and row['source_index']==event['source']['source_index'] and
            row['source_id']==event['source']['source_id'] and row['family']==event['family'] and
            row['actual_snr_db']==event['actual_snr_db'] and row['config_snr_db']==event['config_snr_db'] and
            row['noise_seed']==event['channel']['noise_seed'],'Raw logical frame identity changed')
    return request,worker,logical


def visual_identity(device):
    parent=original.visual_identity(device)
    return dict(schema='H800_RAW_KEEP_RECONSTRUCTION_IDENTITY_V1',device_identity=parent['device_identity'],
        native_binding=parent['native_binding'],model_state_identity=parent['model_state_identity'],
        numeric_settings=parent['numeric_settings'],source_math_SHA=parent['source_math_SHA'],
        render_source_SHA=g.CODE['experiments/content-real-64qam-20261006/h_source_driver.py'],
        receiver_semantics='Original raw full433 hard-token KEEP, unconditional argmax VAR completion, frozen Dc',
        old_host_images_admitted=False)


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    cpu,worker,logical=cpu_closure(r['CPU_closed']);prior=original.source.prerequisites()[0]
    identity=visual_identity(runner.device_from_request(r).device_identity_binding())
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['caps']==CAPS and r['execution']==e and
        r['physical_frames']==worker['results']['physical_frames'] and r['logical_frames']==logical and
        r['raw_runtime_binding']==cpu['raw_runtime_binding'] and r['visual_identity']==identity and
        r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Frozen raw visual request changed')
    g.require(r['source_TX_calls']==r['source_RX_calls']==r['PHY_calls']==r['metric_model_calls']==r['Direct_calls']==0 and
        r['automatic_successor'] is False and r['historical_image_reuse_admitted']==0,'Raw VAR-only scope changed')
    runner.check_tools(r['tool_bindings']);return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh raw visual registration required')
    pins=dict(completion=dict(path=str(g.inside(a.cpu_completion)),sha256=a.cpu_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.cpu_owner_wait)),sha256=a.cpu_owner_wait_sha256))
    cpu,worker,logical=cpu_closure(pins);prior=original.source.prerequisites()[0];e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);g.validate_spec(spec,verify_environment=False)
    g.require(a.target_index==2,'Registered shared single GPU2 required')
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=gate.source.host.DeviceAdapter(dp,a.target_index)
    out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,CPU_closed=pins,
        physical_frames=worker['results']['physical_frames'],logical_frames=logical,raw_runtime_binding=cpu['raw_runtime_binding'],
        visual_identity=visual_identity(device.device_identity_binding()),device_receipt=dp,target_index=2,
        device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        source_TX_calls=0,source_RX_calls=0,PHY_calls=0,metric_model_calls=0,Direct_calls=0,
        historical_image_reuse_admitted=0,automatic_successor=False,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)})
    path=out/'request.json';g.write(path,r);return raw.pin(path)


def parser(binding):
    paths=binding['roles'];known={x['path']:x['sha256'] for x in binding['files']}
    for key in ('action_module','original_receiver_module','catalogue'):
        g.require(g.sha(Path(paths[key]))==known[paths[key]],'Original actual-state parser binding changed')
    raw.load(paths['action_module'],'main_action_space')
    module=raw.load(paths['original_receiver_module'],'_raw_mismatch_visual_actual_parser')
    catalogue=g.json.loads(Path(paths['catalogue']).read_bytes());profiles=catalogue['profiles']
    g.require(len(profiles)==433 and raw.digest(profiles)==binding['catalogue_sha256'],'Full433 visual parse catalogue changed')
    return module,types.SimpleNamespace(entry=lambda pid:copy.deepcopy(profiles[pid]))


def received_tokens(actual,module,catalogue):
    parsed=module.present_actual(actual['header'],actual['body'],catalogue)
    g.require(all(actual[k]==v for k,v in parsed.items()),'Actual raw state does not match hard packet information')
    physical.core.normalized(actual);state=parsed['receiver_state']
    if state['kind']=='gray':return state,None
    parts=[np.asarray(x,dtype=np.int64) for x in state['prefix']]+[np.asarray(state['partial_values'],dtype=np.int64)]
    tokens=np.concatenate(parts)
    g.require(tokens.shape==(g.OFFSETS[state['m']]+state['K'],) and ((tokens>=0)&(tokens<4096)).all(),'Received raw token count/range changed')
    return state,tokens


def actual_state_key(state,identity):
    return raw.digest(dict(actual_state=state,visual_identity=identity))


def render_actual(state,tokens,backend,ledger,boundary):
    boundary()
    if state['kind']=='gray':return np.full((3,256,256),.5,dtype=np.float32)
    def run():
        with backend.native.torch.no_grad():return backend.render_function(backend.native,tokens.copy(),state['m'],state['K'])
    image=ledger.call('var_render',run,source=backend.source_index,m=state['m'],K=state['K'])
    g.require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
        ((image>=0)&(image<=1)).all(),'Original raw VAR reconstruction differs');return image


def images(r,backend,ledger,boundary,out):
    out=Path(out);out.mkdir();module,catalogue=parser(r['raw_runtime_binding']);cache={};physical_rows={}
    for item in r['physical_frames']:
        boundary();packet=gate.readpin(item['actual_RX']);key=item['physical_key'];actual=packet['actual_RX']
        g.require(packet['physical_key']==key==raw.digest(packet['observation_identity']) and packet['state']=='complete' and
            packet['mode']=='new' and not packet['historical_reuse_admitted'],'Only authenticated current physical RX required')
        g.require(key not in physical_rows,'Repeated physical packet')
        state,tokens=received_tokens(actual,module,catalogue);image_key=actual_state_key(state,r['visual_identity'])
        if image_key not in cache:
            backend.source_index=packet['observation_identity']['source_index']
            image=render_actual(state,tokens,backend,ledger,boundary);p=out/'images'/f'{image_key}.npz';p.parent.mkdir(exist_ok=True)
            with p.open('xb') as f:np.savez(f,image=image)
            cache[image_key]=dict(image_archive=raw.pin(p),image_sha256=g.image_sha(image),actual_state_key=image_key,
                actual_state_sha256=actual['actual_state_sha256'],render_called=state['kind']!='gray',first_physical_frame=item['actual_RX'])
        physical_rows[key]=dict(physical_frame=item['actual_RX'],reconstruction=cache[image_key],
            actual_header_ok=actual['header_ok'],actual_crc_accepted=actual['body_crc_accept'],body_attempted=actual['body_decode_complete'],
            actual_RX_status=actual['source_status'],actual_received_profile=dict(profile_id=actual['rx_profile_id'],m=state['m'],K=state['K']),gray=actual['gray'])
    pins=[]
    for row in r['logical_frames']:
        result=physical_rows[row['physical_frame_id']]
        g.require(result['physical_frame']==row['actual_RX'],'Logical frame points to a different actual RX')
        pins.append(raw.save(out/'logical'/f"{row['logical_index']:04d}.json",dict(frame_index=row['logical_index'],logical_event=row,**result)))
    counts=ledger.summary();complete=counts['completed']
    g.require(len(pins)==5400 and len(physical_rows)==4500 and counts['unresolved']==0 and complete==counts['reserved'] and
        counts['caps']==CAPS and all(complete[k]<=v for k,v in CAPS.items()) and complete['model_load']==1 and
        complete['decoder_forward']==complete['var_render'] and complete['prior_scale']==10*complete['var_render'] and
        complete['source_rx']==complete['source_tx']==complete['encoder']==0,'Raw actual rendering budget did not close')
    return dict(logical_frames=pins,counts=counts,physical_count=4500,logical_count=5400,unique_actual_states=len(cache),
        header_reject_physical_frames=sum(row['gray'] for row in physical_rows.values()),historical_image_reuse_admitted=0)


def worker(a,runner):
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    device=runner.device_from_request(r);device.check_cuda_environment();runner.worker_identity(out,a.request_sha256,a.owner_pid)
    started=time.monotonic();stopped=False;ledger=None;backend=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),'Raw visual stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Raw visual deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=10:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'progress.jsonl').open('a') as f:f.write(g.json.dumps(dict(resources=snapshot,counts=None if ledger is None else ledger.summary(),elapsed_seconds=time.monotonic()-started))+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_raw_visual_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        result=images(r,backend,ledger,boundary,out/'reconstructions')
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        done=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,results=result,model_state_identity=dict(g.MODELS),
            source_truth_used=False,Direct_calls=0,PHY_calls=0,quality_scores=0,automatic_successor=False)
        g.write(out/'worker_completion.json',done);return done
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',request_sha256=a.request_sha256,error=repr(error),
            traceback=traceback.format_exc(),counts=None if ledger is None else ledger.summary()));raise


def run(a,runner):
    # Reuse the already-tested resource owner, changing its sole logical-count
    # literal from43200 to5400 in an isolated namespace, never model mathematics.
    tree=ast.parse(inspect.getsource(original.run));matches=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==43200]
    g.require(len(matches)==1,'Original bounded visual owner count guard changed');matches[0].value=5400
    namespace=dict(original.run.__globals__);namespace.update(__file__=__file__,registration=registration,PASS=PASS,SCHEMA=SCHEMA)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
    return namespace['run'](a,runner)


def engine():
    g.require(g.sha(physical.__file__)==PHY_SHA and g.sha(original.__file__)==VISUAL_SHA,'Frozen raw/visual owner changed')
    ns=dict(vars(original.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('cpu-completion','cpu-completion-sha256','cpu-owner-wait','cpu-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(2,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
