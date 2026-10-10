"""Bounded new100 original Encoder/VQ and fresh entropy source preparation.

Requires frozen policies and an actually closed complete content-dedup owner.
One H800 GPU2,2 CPU cores,16GiB allocator,100 encoder+100TX+1000 prior calls;
no independent RX, PHY, rendering, metrics, reselection or automatic successor.
"""
from __future__ import annotations
import argparse
import ast
import inspect
import os
from pathlib import Path
import signal
import time
import traceback
import types
import ep_new100_source_core_v1 as core
import ep_source_population as population
import h800_ep_pilot_source_v1 as original
import h800_shared_gpu_resource_v2 as resources

g=original.g;gate=original.gate;RT=original.RT;CAPS=core.CAPS
SCHEMA='H800_EP_NEW100_SOURCE_V1'
PASS='PASS_H800_EP_NEW100_ENCODER_AND_SOURCE_TX'
ORIGINAL_SHA='661d6f1c8c45b3b324c8c488118ccb4c96c3ea8217281bd4d6bb0ebd014403e2'
CORE_SHA='c11e379e76615b8a35f8c890cbc731a000c4a88dcf65fc6e31bd587b89a4c5a9'
RESOURCE_SHA='0a412ff2705c047ac74426bdb889c55dbad5cb8a669a84bca3b9a8ad728ed7e4'
POPULATION_SHA='4b7fb4a7131a616e1b8f636a1ab631354c8f77035d37ecb937ac1413f16b17f7'
ENCODER_PATH=RT/'assets/ep_new100_source_code_seed_v1/common500_source_assets.py'
CONTENT_OWNER_SHA='e47ea3ef1fba693abb08b887e897c71b43ab523dff2bbba23cc801185d13345b'
TOOLS=original.TOOLS+('h800_ep_pilot_source_v1.py','ep_source_population.py','ep_new100_source_core_v1.py','ep_new100_content_owner_v1.py',
    'h800_shared_gpu_resource_v2.py')


def readpin(pin):
    path=g.inside(pin['path']);g.require(path.is_file() and not path.is_symlink() and g.sha(path)==pin['sha256'],
        'Bound new100 metadata changed: '+str(path))
    if 'bytes' in pin:g.require(path.stat().st_size==pin['bytes'],'Bound new100 metadata size differs')
    return g.json.loads(path.read_bytes())


def content_implementation():
    path=Path(__file__).with_name('ep_new100_content_owner_v1.py')
    g.require(g.sha(path)==CONTENT_OWNER_SHA,'Final frozen content-owner implementation required')
    # Only import the frozen CPU control module; no registration, content worker,
    # selection or preprocessing function is invoked here.
    return g.import_file(path,'_closed_new100_content_owner')


def content_closure(pins):
    g.require(set(pins)=={'completion','owner_actual_wait'},'Content completion and actual owner wait required')
    done=readpin(pins['completion']);wait=readpin(pins['owner_actual_wait'])
    g.require(done['status']=='EP_NEW100_CONTENT_ASSETS_READY_NO_ENCODER' and done['actual_children_waited'] is True and
        done['worker_exit_codes']==[0] and done['actual_child_wait']['actual_wait'] is True and
        done['actual_child_wait']['returncode']==0 and done['actual_child_wait']['interrupted_or_timeout'] is False and
        wait['actual_wait'] is True and wait['returncode']==0 and wait.get('timeout',False) is False and
        wait.get('interrupted_or_timeout',False) is False,
        'Actually closed complete content-dedup owner required before encoding')
    g.require(done['model_calls']==done['packet_calls']==0 and done['automatic_successor'] is False,
        'Content owner must close before any scientific calls')
    implementation=content_implementation();request=readpin(done['request'])
    g.require(request['schema']==implementation.SCHEMA and request['status']=='REGISTERED_NOT_EXECUTED' and
        request['tool_bindings']==implementation.tools(), 'Closed content request/tool identities differ')
    child_path=g.inside(pins['completion']['path']).parent/'actual_child_wait.json'
    child=readpin(dict(path=str(child_path),sha256=g.sha(child_path),bytes=child_path.stat().st_size))
    g.require(child==done['actual_child_wait'],'Actual child wait file and content completion differ')
    worker=readpin(done['worker_completion']);results=done['results']
    g.require(worker['schema']==implementation.SCHEMA and worker['status']=='EP_NEW100_CONTENT_ASSETS_READY_NO_ENCODER' and
        worker['request_sha256']==done['request']['sha256'] and worker['results']==results and
        worker['canonical_reserved']==worker['canonical_completed']==100 and worker['unresolved']==0 and
        worker['model_calls']==worker['packet_calls']==0 and worker['CUDA_initialized'] is False,
        'Content owner/worker output closure differs')
    required={'selection','policy_bundle','registry','content_receipt','content_gate','source_manifest'}
    g.require(required<=set(results),'Complete content/policy/source metadata pins required')
    values={k:readpin(results[k]) for k in required};selection=values['selection'];manifest=values['source_manifest'];policy=values['policy_bundle']
    g.require(all(manifest[k]==results[k] for k in required-{'source_manifest'}) and manifest['model_calls']==0 and
        manifest['original_preprocess']==request['original_preprocess']==values['content_receipt']['original_preprocess'],
        'Source manifest must bind the exact content, policies and original preprocessing')
    g.require(policy['schema']=='EP_NEW100_FOUR_FROZEN_POLICY_PROJECTION_V1' and policy['strategy_selection_uses_new100'] is False and
        policy['full_calibration']['complete_grid_verified'] is True and set(policy['policies'])==set(population.METHODS) and
        all(set(rows)=={'4','10','19'} for rows in policy['policies'].values()), 'All four original frozen policies required')
    proven=population.check_content(values['registry'],results['registry'],selection,results['selection'],
        values['content_receipt'],results['content_receipt'])
    g.require(proven==values['content_gate'] and proven['status']=='EP_NEW100_COMPLETE_CONTENT_DEDUP_PASS' and
        proven['conflicts']==[] and proven['Encoder_calls_allowed'] is True and proven['source_reselection_allowed'] is False,
        'Original complete canonical content gate must PASS without source replacement')
    g.require(selection['source_count']==100 and selection['N']==1024 and selection['SNRs']==[4,10,19] and
        selection['methods']==population.METHODS and selection['noise_seeds']==population.NOISE_SEEDS and
        selection['policy_selection_uses_new100'] is False and selection['source_reselection_allowed'] is False,
        'Frozen new100 confirmation scope changed')
    records=manifest['records']
    g.require(manifest['schema']=='EP_NEW100_SOURCE_CONTENT_ASSETS_V1' and manifest['source_count']==100 and len(records)==100 and
        [r['source_index'] for r in records]==list(range(100)) and [r['source_id'] for r in records]==selection['source_ids'] and
        len({r['canonical_source_id'] for r in records})==100,'Exact fixed new100 source order required')
    for r,expected,actual in zip(records,selection['records'],values['content_receipt']['records']):
        g.require(r['canonical_source_id']==expected['canonical_source_id']==population.canonical_id(r['source_id']) and
            type(r['evaluation_class_index']) is int and 0<=r['evaluation_class_index']<1000 and
            r['original_JPEG']['sha256']==actual['original_file_sha256'] and
            r['preprocessing_id']==r['canonical_pixel_sha256']==actual['pixel_sha256'] and
            r['horizontal_flip_pixel_sha256']==actual['horizontal_flip_pixel_sha256'],'Fixed new source content identity differs')
        for key in ('original_JPEG','pixels_archive'):
            pin=r[key];p=g.inside(pin['path']);g.require(p.is_file() and p.stat().st_size==pin['bytes'] and g.sha(p)==pin['sha256'],
                'Closed new100 source file bytes changed')
    return records,{k:results[k] for k in required}


def registration(path,digest,runner):
    r=g.checked_json(g.inside(path),digest);resources.validate_policy(r);e=gate.source.execution(r['execution']['deadline_unix'],r['execution']['max_seconds'])
    records,inputs=content_closure(r['content_closed']);prior,_,pins=original.prerequisites()
    g.require(r['schema']==SCHEMA and r['status']=='REGISTERED_NOT_EXECUTED' and r['execution']==e and r['caps']==CAPS and
        r['records']==records and r['inputs']==inputs and r['qualified_components']==pins and
        r['spec']==runner.inherited_spec(prior,e['deadline_unix'],900),'Frozen new100 source registration changed')
    g.require(r['encoder_module']==dict(path=str(ENCODER_PATH),sha256=core.ENCODER_SHA) and g.sha(g.inside(ENCODER_PATH))==core.ENCODER_SHA,
        'Exact original Encoder function source required')
    g.require(r['target_index']==2 and r['device_identity_binding']==runner.device_from_request(r).device_identity_binding() and
        r['actual_host_native_binding']==g.native_pin(),'Qualified H800 GPU2 and native identity required')
    g.require(r['automatic_successor'] is False and r['training_updates']==r['PHY_calls']==r['quality_scores']==0 and
        r['source_RX_calls']==r['VAR_render_calls']==r['decoder_calls']==0 and r['source_reselection_allowed'] is False,
        'Finite source-only scope changed')
    runner.check_tools(r['tool_bindings']);return r


def prepare(a,runner):
    out=g.inside(a.out);g.require(not out.exists() and a.target_index==2,'Fresh new100 source output on qualified GPU2 required')
    pins=dict(completion=dict(path=str(g.inside(a.content_completion)),sha256=a.content_completion_sha256),
        owner_actual_wait=dict(path=str(g.inside(a.content_owner_wait)),sha256=a.content_owner_wait_sha256))
    records,inputs=content_closure(pins);prior,_,qualified=original.prerequisites()
    e=gate.source.execution(a.deadline_unix,a.max_seconds);spec=runner.inherited_spec(prior,e['deadline_unix'],900)
    g.validate_spec(spec,verify_environment=False);core.encoder_projection(g.inside(ENCODER_PATH))
    dp=dict(path=str(g.inside(a.device_receipt)),sha256=a.device_receipt_sha256);device=resources.DeviceAdapter(dp,a.target_index)
    r=dict(schema=SCHEMA,status='REGISTERED_NOT_EXECUTED',content_closed=pins,records=records,inputs=inputs,
        qualified_components=qualified,spec=spec,execution=e,caps=CAPS,encoder_module=dict(path=str(ENCODER_PATH),sha256=core.ENCODER_SHA),
        device_receipt=dp,target_index=2,device_identity_binding=device.device_identity_binding(),actual_host_native_binding=g.native_pin(),
        resource_policy=dict(resources.POLICY),resource_guard=core.descriptor(Path(resources.__file__)),
        prelaunch_wait_seconds=120,tool_bindings={n:g.sha(Path(__file__).with_name(n)) for n in (*TOOLS,Path(__file__).name)},
        automatic_successor=False,source_reselection_allowed=False,training_updates=0,PHY_calls=0,quality_scores=0,
        source_RX_calls=0,VAR_render_calls=0,decoder_calls=0,preparation_model_calls=0,preparation_source_arrays_read=0,
        historical_input_contract='Original two-source model/runtime provenance only; new100 source admission is separately closed',
        purpose='Single shared new100 Encoder cache and fresh24 entropy endpoints per image; no new policies')
    out.mkdir(parents=True);path=out/'request.json';g.write(path,r);return core.descriptor(path)


def worker(a,runner):
    path=g.inside(a.request);r=registration(path,a.request_sha256,runner);out=path.parent/'run';spec=r['spec'];e=r['execution']
    device=runner.device_from_request(r);device.bind_resource_evidence(out,a.request_sha256);device.check_cuda_environment()
    launch=runner.worker_identity(out,a.request_sha256,a.owner_pid);g.require(launch['argv'][0]==spec['python'],'Wrong frozen interpreter')
    stopped=False;started=time.monotonic();ledger=None;backend=None;last=0.
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def guard():
        g.require(not stopped and os.getppid()==a.owner_pid and not (out/'STOP').exists() and not (g.BASE/'STOP').exists(),
            'New100 source stop or parent loss')
        g.require(time.monotonic()-started<e['max_seconds'] and time.time()<e['deadline_unix'],'New100 source deadline exceeded')
    def boundary():
        nonlocal last
        guard()
        if time.monotonic()-last>=2:
            snapshot,_=device.resource_snapshot(g.helper());last=time.monotonic()
            with (out/'resources.jsonl').open('a') as stream:stream.write(g.json.dumps(snapshot)+'\n')
    try:
        g.write(out/'worker_claim.json',dict(pid=os.getpid(),request_sha256=a.request_sha256))
        resolver,environment,_,_,extra=g.validate_spec(spec);boundary();ledger=g.Ledger(out/'calls',guard,CAPS)
        module=g.import_file(Path(__file__).with_name('leo_whole_math_v1.py'),'_h800_new100_original_math')
        Backend,proof=runner.adapt_backend(module,device);g.write(out/'device_adapter_ast_proof.json',proof)
        backend=ledger.call('model_load',lambda:Backend(resolver,environment,spec,extra,ledger,boundary,out))
        for name in ('t1_codec_runtime.py','t1_entropy_core.py'):
            relative='experiments/wcl-evidence-closure-20261009/scripts/'+name;actual=g.inside(Path(spec['project_root'])/relative)
            g.require(g.sha(actual)==g.sha(resolver.path(g.OLD+relative))==g.CODE[relative],'Original arithmetic source changed')
        import ep_source_codec as partial
        codec=partial.PartialSourceCodec(spec['project_root']);codec.native=backend.native;codec.provider_factory=backend.codec.provider_factory
        encoder,proof=core.encoder_projection(g.inside(ENCODER_PATH));g.write(out/'original_encoder_AST_proof.json',proof)
        (out/'sources').mkdir();completed=[]
        for record in r['records']:
            completed.append(core.encode_source(record,backend,codec,partial,ledger,encoder,g.inside,out/'sources'/f"{record['source_index']:04d}"))
            g.write(out/f'progress_{len(completed):03d}.json',dict(completed_new_sources=len(completed),counts=ledger.summary()))
        backend.close();resolver.reverify();environment.reverify();registration(path,a.request_sha256,runner);guard()
        counts=ledger.summary();g.require(counts['completed']==counts['reserved']==CAPS and counts['unresolved']==0,'New100 exact source budget did not close')
        result=dict(schema=SCHEMA,status=PASS,request_sha256=a.request_sha256,counts=counts,sources=completed,source_count=100,
            content_closed=r['content_closed'],policy_bundle=r['inputs']['policy_bundle'],independent_RX_claim=False,
            PHY_calls=0,quality_scores=0,training_updates=0,old_gate_counts_preserved=True,automatic_successor=False)
        g.write(out/'worker_completion.json',result);return result
    except BaseException as error:
        if backend is not None:backend.failure_evidence()
        g.write(out/'worker_failure.json',dict(status='STOPPED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),
            request_sha256=a.request_sha256,counts=None if ledger is None else ledger.summary()));raise


def engine():
    g.require(g.sha(original.__file__)==ORIGINAL_SHA and g.sha(core.__file__)==CORE_SHA and g.sha(resources.__file__)==RESOURCE_SHA and
        g.sha(population.__file__)==POPULATION_SHA,'Frozen source owner/codec/population changed')
    ns=dict(vars(original.engine()));ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,TOOLS=TOOLS,DeviceAdapter=resources.DeviceAdapter)
    for name in ('wait_prelaunch','device_from_request','check_tools','adapt_backend','inherited_spec','worker_identity'):
        ns[name]=gate.source.host.clone_function(ns[name],ns)
    return types.SimpleNamespace(**ns)


# Preserve the original bounded owner and actual wait. Only its final source-list
# field/cardinality and binding of the previously audited resource evidence change.
# The shared same-PID resource policy leaves numerical functions/caps unchanged.
run_ns=dict(vars(original));run_ns.update(__file__=__file__,SCHEMA=SCHEMA,PASS=PASS,CAPS=CAPS,registration=registration,core=core)
tree=ast.parse(inspect.getsource(original.run));changed=dict(count=0,field=0,resource_evidence=0)
class New100Scope(ast.NodeTransformer):
    def visit_Constant(self,node):
        if type(node.value) is int and node.value==68:node.value=100;changed['count']+=1
        elif node.value=='new68_streams':node.value='sources';changed['field']+=1
        return node
    def visit_Expr(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='mkdir' and \
                isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='out':
            changed['resource_evidence']+=1
            return [node,ast.parse('device.bind_resource_evidence(out,a.request_sha256)').body[0]]
        return node
tree=New100Scope().visit(tree);g.require(changed==dict(count=1,field=1,resource_evidence=1),'Original owner finite scope/resource binding changed')
exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),run_ns);run=run_ns['run']


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('content-completion','content-completion-sha256','content-owner-wait','content-owner-wait-sha256',
                 'device-receipt','device-receipt-sha256','out'):q.add_argument('--'+name,required=True)
    q.add_argument('--target-index',type=int,choices=(2,),required=True);q.add_argument('--deadline-unix',type=float,required=True)
    q.add_argument('--max-seconds',type=int,default=3600)
    for name in ('run','_worker'):
        q=sub.add_parser(name);q.add_argument('--request',required=True);q.add_argument('--request-sha256',required=True)
        if name=='_worker':q.add_argument('--owner-pid',type=int,required=True)
    a=p.parse_args();runner=engine();result=prepare(a,runner) if a.command=='prepare' else run(a,runner) if a.command=='run' else worker(a,runner)
    print(g.json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
