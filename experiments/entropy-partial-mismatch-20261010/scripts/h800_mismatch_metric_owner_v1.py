"""Finite four-metric owner for closed raw mismatch reconstructions.

At most4500 source/image pairs, exactly5400 logical rows. Deterministic paired
three-noise arithmetic is exported with no bootstrap, confidence interval or
policy update. Later statistical execution requires its own explicit admission.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import types
import numpy as np
import h800_ep_pilot_metric_owner_v1 as original
import h800_mismatch_visual_v1 as visual
import h800_mismatch_metrics_v1 as metric

g=visual.g;gate=visual.gate;raw=visual.raw
SCHEMA='H800_RAW100_ONE_BIN_MISMATCH_METRIC_OWNER_V1'
PASS='PASS_H800_RAW100_MISMATCH_FOUR_METRICS'
CAPS=metric.CAPS
ORIGINAL_OWNER_SHA='8f5b04978b23d8b4525b03599d13a9aa1277872481624fa17c5a0627f43b60e1'
TOOLS=visual.TOOLS+('h800_mismatch_visual_v1.py','h800_ep_pilot_metric_owner_v1.py','h800_ep_pilot_metrics_v1.py','h800_mismatch_metrics_v1.py')
execution=visual.execution
metric_environment=original.metric_environment
runtime_identity=original.runtime_identity


def visual_closure(pins):
    root=gate.RT/'qualification/h800_mismatch_visual_v1_attempt1'
    g.require(set(pins)=={'completion','owner_actual_wait'} and pins['completion']['path']==str(root/'registered/run/completion.json') and
        pins['owner_actual_wait']['path']==str(root/'run_owner_actual_wait.json'),'Exact raw mismatch visual attempt required')
    done=gate.readpin(pins['completion']);wait=gate.readpin(pins['owner_actual_wait'])
    g.require(done['status']==visual.PASS and done['actual_wait']['success'] is True and done['actual_children_waited'] is True and
        done['worker_exit_codes']==[0] and wait['actual_wait'] is True and wait['returncode']==0,
        'Actual raw visual owner must close before metric admission')
    request=g.checked_json(root/'registered/request.json',done['request_sha256']);worker=gate.readpin(done['worker_completion'])
    cpu,physical,logical=visual.cpu_closure(request['CPU_closed']);device=gate.source.host.DeviceAdapter(request['device_receipt'],request['target_index'])
    g.require(request['schema']==visual.SCHEMA and request['caps']==visual.CAPS and request['logical_frames']==logical and
        request['physical_frames']==physical['results']['physical_frames'] and request['raw_runtime_binding']==cpu['raw_runtime_binding'] and
        request['visual_identity']==visual.visual_identity(device.device_identity_binding()),'Closed raw visual identity changed')
    visual.engine().check_tools(request['tool_bindings']);result=worker['results'];counts=result['counts'];c=counts['completed']
    g.require(worker['status']==visual.PASS and worker['request_sha256']==done['request_sha256'] and
        len(result['logical_frames'])==5400 and result['logical_count']==5400 and result['physical_count']==4500 and
        counts['caps']==visual.CAPS and counts['unresolved']==0 and c==counts['reserved'] and c['model_load']==1 and
        c['decoder_forward']==c['var_render'] and c['prior_scale']==10*c['var_render'] and c['source_tx']==c['source_rx']==c['encoder']==0 and
        all(c[k]<=v for k,v in visual.CAPS.items()) and worker['quality_scores']==0,'Complete actual raw reconstruction closure required')
    return request,worker,cpu,logical


def configuration(spec,device,raw_policy):
    result=original.configuration(spec,device)
    result.update(diagnostic_population=metric.POPULATION,policy_selection=False,raw_frozen_policy=raw_policy)
    return result


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);e=execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    _,worker,cpu,logical=visual_closure(r['visual_closed']);prior=visual.original.source.prerequisites()[0]
    device=runner.device_from_request(r);spec=runner.inherited_spec(prior,e['deadline_unix'],900)
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['spec']==spec and r['records']==cpu['records'] and r['logical_events']==logical and
        r['reconstruction_results']==worker['results']['logical_frames'] and
        r['metric_config']==configuration(spec,device,cpu['input_closure']['original_policy']), 'Frozen raw metric request changed')
    g.require(r['PHY_calls']==r['source_calls']==r['VAR_calls']==r['decoder_calls']==r['training_updates']==r['bootstrap_calls']==0 and
        r['policy_selection'] is False and r['automatic_successor'] is False,'Metric-only diagnostic scope changed')
    runner.check_tools(r['tool_bindings']);metric.bound_assets(r['metric_config']);return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists(),'Fresh raw metric registration required')
    pins=dict(completion=dict(path=str(g.inside(a.visual_completion)),sha256=a.visual_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.visual_owner_wait)),sha256=a.visual_owner_wait_sha256))
    _,worker,cpu,logical=visual_closure(pins);prior=visual.original.source.prerequisites()[0];e=execution(a.deadline_unix,a.max_seconds)
    spec=runner.inherited_spec(prior,e['deadline_unix'],900);metric_environment(spec,verify_environment=False)
    g.require(a.target_index==2,'Registered single GPU2 required')
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=gate.source.host.DeviceAdapter(dp,a.target_index)
    config=configuration(spec,device,cpu['input_closure']['original_policy']);metric.bound_assets(config);out.mkdir(parents=True)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',caps=CAPS,execution=e,spec=spec,visual_closed=pins,
        records=cpu['records'],logical_events=logical,reconstruction_results=worker['results']['logical_frames'],metric_config=config,
        device_receipt=dp,target_index=2,device_identity_binding=device.device_identity_binding(),prelaunch_wait_seconds=120,
        tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        PHY_calls=0,source_calls=0,VAR_calls=0,decoder_calls=0,training_updates=0,bootstrap_calls=0,
        policy_selection=False,automatic_successor=False)
    path=out/'request.json';g.write(path,r);return raw.pin(path)


def scores(r,evaluator,ledger,boundary,out):
    out=Path(out);out.mkdir();cache={};references={};verified={};rows=[];reused=0
    g.require(len(r['records'])==100 and [x['source_index'] for x in r['records']]==list(range(100)) and
        len(r['logical_events'])==len(r['reconstruction_results'])==5400,'Exact100 sources and5400 reconstructed logical events required')
    for i,pin in enumerate(r['reconstruction_results']):
        boundary();result=gate.readpin(pin);event=result['logical_event'];expected=r['logical_events'][i]
        g.require(result['frame_index']==i and event==expected,'Actual raw reconstruction event changed')
        index=event['source_index'];record=r['records'][index];g.require(event['source_id']==record['source_id'],'Reference/source identity differs')
        if index not in references:
            # Same frozen uint8-to-float32/255 expression; only explicit relocated archive changes.
            target=original.core.reference_pixels(dict(record,archive=record['actual_archive']))
            references[index]=(target,evaluator.prepare(target),metric.array_sha(target))
        target,prepared,reference_sha=references[index];recon=result['reconstruction'];archive=recon['image_archive']
        key=raw.digest(dict(reference_sha256=reference_sha,reconstruction_sha256=recon['image_sha256'],metric_identity=evaluator.identity))
        archive_key=raw.digest(archive);image=None
        if archive_key not in verified:
            g.require(g.sha(g.inside(archive['path']))==archive['sha256'],'Actual raw image archive changed')
            with np.load(archive['path'],allow_pickle=False) as z:
                g.require(set(z.files)=={'image'},'Unexpected raw reconstruction archive schema');image=z['image'].copy()
            g.require(image.dtype==np.float32 and image.shape==(3,256,256) and np.isfinite(image).all() and
                ((image>=0)&(image<=1)).all() and metric.array_sha(image)==recon['image_sha256'],'Actual raw RGB bytes differ')
            verified[archive_key]=recon['image_sha256']
        else:g.require(verified[archive_key]==recon['image_sha256'],'Conflicting image SHA for one archive')
        if key not in cache:
            if image is None:
                with np.load(archive['path'],allow_pickle=False) as z:image=z['image'].copy()
                g.require(metric.array_sha(image)==recon['image_sha256'],'Actual scored RGB changed')
            g.require(metric.pair_cache_key(target,image,evaluator.identity)==key,'Source/image metric pair differs')
            values=evaluator.score(target,image,record['evaluation_class_index'],prepared)
            pair=raw.save(out/'pairs'/f'{len(cache):04d}.json',dict(metrics=values,metric_pair_key=key,reference_archive=record['actual_archive'],
                reconstruction_archive=archive,metric_identity=evaluator.identity))
            cache[key]=(values,pair)
        else:reused+=1
        values,pair=cache[key]
        rows.append(dict(source_index=index,source_id=event['source_id'],actual_snr_db=event['actual_snr_db'],config_snr_db=event['config_snr_db'],
            family=event['family'],noise_seed=event['noise_seed'],tx_candidate_id=event['tx_candidate_id'],**values,
            metric_pair_key=key,metric_pair_result=pair,reconstruction_result=pin,actual_RX_status=result['actual_RX_status'],
            actual_received_profile=result['actual_received_profile'],actual_header_ok=result['actual_header_ok'],
            actual_crc_accepted=result['actual_crc_accepted'],body_attempted=result['body_attempted'],gray=result['gray']))
    counts=ledger.summary();c=counts['completed']
    g.require(counts['unresolved']==0 and c==counts['reserved'] and counts['caps']==CAPS and c['model_constructions']==3 and
        c['reference_preparations']==100 and c['image_scores']==len(cache) and len(cache)+reused==5400 and
        all(c[k]<=v for k,v in CAPS.items()),'Complete finite raw metric pair ledger required')
    for kind in ('dinov2_vitl14_reference','convnext_reference'):g.require(c[kind]==100,'Missing source reference features')
    for kind in ('dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):g.require(c[kind]==len(cache),'Missing metric pairs')
    g.require(c['lpips_alexnet_backbone_forward']==2*len(cache),'LPIPS backbone accounting differs')
    summary=metric.paired_outputs(rows)
    return dict(metric_rows=raw.save(out/'metric_rows.json',rows),paired_outputs=raw.save(out/'paired_outputs.json',summary),
        logical_count=5400,actual_unique_pairs=len(cache),reused_pairs=reused,counts=counts,
        bootstrap_calls=0,policy_selection=False,old_visual_and_PHY_counts_preserved=True)


def run(a,runner):
    g.require(g.sys.platform.startswith('linux'),'Actual Linux metric owner required')
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
                LANG='C.UTF-8',LC_ALL='C.UTF-8',VIRTUAL_ENV=str(Path(spec['python']).parent.parent),CUBLAS_WORKSPACE_CONFIG=':4096:8',**device.cuda_environment_binding())
            g.write(out/'controlled_environment.json',env)
            argv=[spec['python'],'-B','-u',str(Path(__file__).absolute()),'_worker','--request',str(path),'--request-sha256',a.request_sha256,'--owner-pid',str(os.getpid())]
            seconds=min(e['max_seconds']-(time.monotonic()-started),e['deadline_unix']-time.time());g.require(seconds>0,'Metric deadline expired')
            with (out/'child.log').open('x') as log:
                child=subprocess.Popen(argv,env=env,cwd=spec['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    preexec_fn=shared.child_limits(ad['cpu_affinity'],seconds,2))
                g.write(out/'child_started.json',dict(pid=child.pid,owner_pid=os.getpid(),request_sha256=a.request_sha256,argv=argv,
                    cpu_affinity=ad['cpu_affinity'],started_unix=time.time()));waited=shared.wait_owned(child,seconds)
            g.write(out/'actual_child_wait.json',waited);g.require(waited['success'],'Raw metric child failed; no automatic retry')
            done=gate.readpin(raw.pin(out/'worker_completion.json'));result=done['results'];rows=gate.readpin(result['metric_rows'])
            g.require(done['status']==PASS and done['request_sha256']==a.request_sha256 and result['logical_count']==5400 and
                result['counts']['unresolved']==0 and result['counts']['completed']==result['counts']['reserved'] and
                gate.readpin(result['paired_outputs'])==metric.paired_outputs(rows),'Actual raw metric closure differs')
            registration(path,a.request_sha256,runner);metric_environment(spec)
            g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'Raw metric final deadline exceeded')
            result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,worker_completion=raw.pin(out/'worker_completion.json'),
                actual_wait=waited,actual_children_waited=True,worker_exit_codes=[child.returncode],automatic_successor=False,bootstrap_calls=0,policy_selection=False)
            g.write(out/'completion.json',result);return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            g.write(out/'failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
                actual_child_waited=child is not None and child.poll() is not None,child_exit_code=None if child is None else child.returncode));raise


def worker(a,runner):
    # Exact original bounded metric process/resource/model procedure; only the
    # registered scope, original compute provider namespace and scores differ.
    ns=dict(original.worker.__globals__);ns.update(__file__=__file__,registration=registration,scores=scores,
        metric=metric,CAPS=CAPS,SCHEMA=SCHEMA,PASS=PASS)
    call=gate.source.host.clone_function(original.worker,ns);return call(a,runner)


def engine():
    g.require(g.sha(original.__file__)==ORIGINAL_OWNER_SHA,'Original bounded metric owner changed')
    ns=dict(vars(visual.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS)
    for name in ('wait_prelaunch','device_from_request','check_tools','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('prepare')
    for name in ('visual-completion','visual-completion-sha256','visual-owner-wait','visual-owner-wait-sha256','device-receipt','device-receipt-sha256','out'):
        q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(2,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=21600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
