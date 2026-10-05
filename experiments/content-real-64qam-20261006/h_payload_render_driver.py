"""Registered exclusive-GPU reconstruction of closed H initial_true200 traces.

This stage makes no PHY calls and no policy choices. It starts only after the
separate CPU owner and all its workers exited with a complete receipt chain.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np

STOP = False
DEADLINE = 1791564605.9549868
FINAL_STATUSES = {'RAW_SOURCE_DECODED','ARITHMETIC_SOURCE_DECODED','WIRE_REJECT_GRAY','ARITHMETIC_SOURCE_INVALID_GRAY'}
SEEDS = (6101,6102,6103)


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def verify(bindings):
    for p,s in bindings.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)
def atomic(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not exclusive or not path.exists(),'Existing immutable output: '+str(path))
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def stop(*_):
    global STOP
    STOP=True


def require_native_bindings(runtime,bound):
    directory=Path(runtime);files=list(directory.glob('m1_*.py'))
    require({'m1_common.py','m1_native.py','m1_phy.py','m1_performance.py'}<={p.name for p in files},
            'Incomplete frozen native runtime')
    for p in files:require(bound.get(str(p))==sha(p),'Frozen native execution source unbound: '+str(p))


class CachedReceiver:
    """Per-source deterministic cache of *actual* accepted RX inputs.

    Every trace is revalidated against its actual decoded bits first. Cache
    entries are created only by Receiver.reconstruct, never clean source images.
    Header score/noise/labels/TX correctness cannot choose the cache entry.
    """
    def __init__(self,receiver,identity):
        self.receiver=receiver;self.identity=digest(identity);self.cache={};self.hits=0;self.misses=0
    def reconstruct(self,view,event_id):
        wire=self.receiver._wire(view)
        if wire['gray']:
            result=self.receiver.reconstruct(view);hit=False;key=None;origin=event_id
        else:
            semantic=dict(identity=self.identity,catalogue=self.receiver.catalogue_sha256,
                          profile=wire['profile'],actual_payload=wire['body']['payload'])
            key=digest(semantic)
            if key in self.cache:
                entry=self.cache[key];require(entry['semantic']==semantic,'Receiver cache key collision')
                result=copy.deepcopy(entry['result']);hit=True;origin=entry['origin'];self.hits+=1
            else:
                result=self.receiver.reconstruct(view);hit=False;origin=event_id;self.misses+=1
                self.cache[key]=dict(semantic=semantic,result=copy.deepcopy(result),origin=event_id)
        s=result['summary'];s['receiver_view_sha256']=digest(view)
        s.update(receiver_result_cache_hit=hit,receiver_result_cache_key=key,receiver_cache_origin_event=origin,
                 arithmetic_canonical_executed_this_frame=bool(s['arithmetic_canonical_attempted'] and not hit),
                 suffix_renderer_executed_this_frame=bool(not s['gray'] and not hit),
                 cache_scope='same-source actual received profile+payload and bound visual/receiver identity; no TX truth')
        return result


def load_registered(config_path):
    cfg=read(config_path);reg=read(cfg['registration']);bound=dict(reg['input_bindings'],**reg['source_bindings'])
    require(reg.get('status')=='H_EXECUTION_REVISION_REGISTERED' and reg.get('branch')=='H'
            and reg.get('allowed_stage_ids')==['render'],'Independent H render revision required')
    verify(reg['input_bindings']);verify(reg['source_bindings'])
    require(bound.get(str(Path(config_path).absolute()))==sha(config_path),'Render config not bound')
    required=('protocol','catalogue','shortlist','source200','calibration_registration','S1_completion','S1_assets_completion',
              'cpu_config','cpu_registration','cpu_completion','cpu_owner_config','cpu_owner_registration','cpu_owner_launch',
              'budget_registration','owner_module','wait_module','cpu_driver_module','rx_module','source_driver_module','render_owner_config')
    for k in required:require(bound.get(cfg[k])==sha(cfg[k]),'Required input/source not bound: '+k)
    require(bound.get(str(Path(__file__).absolute()))==sha(__file__),'Render driver unbound')
    out=Path(cfg['out']);hout=Path(cfg['H_out'])
    require(out.is_absolute() and hout in out.parents and out!=hout,'Output must be a distinct H subdirectory')
    require(cfg['overall_deadline_unix']==DEADLINE and 0<cfg['max_seconds']<=86400,'Original deadline/stage cap differs')
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
            and budget['phase_limits']==cfg['phase_limits']==reg['phase_limits'],'Original H budget changed')
    require(Path(cfg['ledger']).is_file(),'Original ledger absent')
    protocol=read(cfg['protocol']);require(protocol['schema']=='H_CODEC_PROTOCOL_V1' and protocol['status']=='FROZEN_BEFORE_DATA','Wrong H protocol')
    for name in ('h64_catalog.py','h64_phy.py','h64_source.py'):
        path=str(Path(cfg['runtime_dir'])/name);require(bound.get(path)==sha(path),'Frozen H receiver source unbound: '+name)
    for name in ('quality_driver.py','source_quality.py'):
        path=str(Path(cfg['uep_runtime'])/name);require(bound.get(path)==sha(path),'Frozen visual adapter unbound')
    require_native_bindings(cfg['native_runtime'],bound)
    for name in ('whole_entropy.py','entropy.py'):
        path=str(Path(cfg['root'])/'src/var_comm'/name);require(bound.get(path)==sha(path),'Original arithmetic source unbound')
    # Import only verified CPU/read-only tools. No backend/model constructors.
    owner=module(cfg['owner_module'],'render_bound_stage_owner');prior=module(cfg['wait_module'],'render_bound_batch_verifier')
    cpu=module(cfg['cpu_driver_module'],'render_bound_cpu_driver');ctx=cpu.load_registered(cfg['cpu_config'])
    require(ctx['cfg']['registration']==cfg['cpu_registration']
            and str(Path(ctx['cfg']['out'])/'completion.json')==cfg['cpu_completion'],'CPU execution scope differs')
    for key in ('protocol','catalogue','shortlist','source200','S1_completion','S1_assets_completion','budget_registration','ledger'):
        require(ctx['cfg'][key]==cfg[key],'GPU and CPU input identity differs: '+key)
    done=read(cfg['cpu_completion']);require(done['status']=='H_INITIAL_TRUE200_CPU_RECEIVE_COMPLETE'
        and done['registration_sha256']==sha(cfg['cpu_registration']) and done['driver_config_sha256']==sha(cfg['cpu_config'])
        and done['budget_registration_sha256']==sha(cfg['budget_registration']) and done['shortlist_sha256']==sha(cfg['shortlist'])
        and done['images_scored'] is False and done['arithmetic_source_decode_complete'] is False
        and done['GPU_used'] is False and done['development_used'] is False,'CPU receive completion incomplete/different')
    verify(done['outputs']);verify(done['input_bindings']);verify(done['source_bindings'])
    for path in (cfg['runtime_dir'],str(Path(cfg['cpu_driver_module']).parent)):sys.path.insert(0,path)
    paths=[Path(ctx['cfg']['out'])/f'worker_{i%2}'/'traces'/f'{i:04d}.json' for i in range(200)]
    events,inventory=cpu.trace_inventory(ctx,paths)
    audit=cpu.audit_ledger(cfg['ledger'],events)
    require(audit==done['ledger_audit'] and inventory['frame_count']==done['frame_count'],
            'Rechecked actual packet ledger/trace inventory differs from CPU completion')
    old=read(cfg['cpu_owner_config']);oldreg=read(cfg['cpu_owner_registration']);launch=read(cfg['cpu_owner_launch'])
    owner.validate_config(old,oldreg,sha(cfg['cpu_owner_config']))
    require(all(s['resource']=='cpu' for s in old['stages']) and old['registration']==cfg['cpu_owner_registration'],
            'CPU prerequisite owner has another scope')
    ident=launch['identity'];expected=[launch['argv'][0],'-B',cfg['owner_module'],'--config',cfg['cpu_owner_config']]
    require(launch['argv']==ident['argv']==expected and launch['registration_sha256']==sha(cfg['cpu_owner_registration'])
            and launch['owner_config_sha256']==sha(cfg['cpu_owner_config']),'CPU owner launch identity differs')
    require(prior.exited(ident,owner.raw_process_state),'CPU owner remains present; independent GPU batch must wait')
    closure=prior.verify_batch(owner,cfg['cpu_owner_config'],cfg['cpu_owner_registration'],ident['uid'],owner.raw_process_state)
    require(closure['bindings'].get(cfg['cpu_completion'])==sha(cfg['cpu_completion']),'CPU owner did not seal this merge completion')
    for child in done['worker_identities']:
        require(any(owner.same_identity(child,record) for record in closure['child_identities']),
                'CPU worker identity missing from exited owner chain')
        require(prior.exited(child,owner.raw_process_state),'CPU packet worker remains present')
    require(len(done['worker_identities'])==2,'CPU worker identity coverage differs')
    for key in ('budget_path','budget_registration','budget_registration_sha256','phase_limits'):
        expected_value=cfg['ledger'] if key=='budget_path' else sha(cfg['budget_registration']) if key=='budget_registration_sha256' else cfg[key]
        require(old[key]==expected_value,'CPU owner budget identity differs')
    snapshot=owner.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    require(snapshot['phase_charged']['initial_true200']==done['ledger']['phase_charged']['initial_true200']<=19200,
            'Paid CPU count changed or exceeds finite scope')
    assets=read(cfg['S1_assets_completion']);s1=read(cfg['S1_completion'])
    require(s1['status']=='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION' and assets['status']=='S1_EXPORT_ASSETS_COMPLETE'
            and assets['source_count']==200 and assets['registration_sha256']==s1['registration_sha256'],'Original S1 assets unavailable')
    require(s1['outputs'].get(cfg['S1_assets_completion'])==sha(cfg['S1_assets_completion']),'S1 root does not bind export receipt')
    verify(s1['outputs']);verify(assets['outputs'])
    ids=read(cfg['source200'])['source_ids'];require(ids==ctx['shortlist']['source_ids'],'Source200 differs')
    candidates=cpu.check_shortlist(ctx['shortlist'])
    require(done['source_count']==200 and done['frame_count']==200*len(candidates)*3,'Incomplete CPU frame count')
    return dict(cfg=cfg,reg=reg,owner=owner,prior=prior,cpu=cpu,cpu_ctx=ctx,completion=done,closure=closure,
                before=snapshot,assets=assets,ids=ids,candidates=candidates,protocol=protocol,catalogue=read(cfg['catalogue']))


def verify_live_visual_owner(ctx,config_path):
    """Require this process to be the bound job of the live exclusive owner."""
    cfg=ctx['cfg'];a=ctx['owner'];owner_cfg=read(cfg['render_owner_config'])
    a.validate_config(owner_cfg,ctx['reg'],sha(cfg['render_owner_config']))
    require(owner_cfg['registration']==cfg['registration'] and len(owner_cfg['stages'])==1,'Unexpected visual owner scope')
    stage=owner_cfg['stages'][0];require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,
                                       'Exactly one registered visual job required')
    job=stage['jobs'][0];expected=job['argv'];entry=a.command_entry(expected)
    offset=2 if expected[1]=='-B' else 1
    require(entry==Path(__file__).resolve() and expected[offset+1:]==['--config',str(Path(config_path).absolute())]
            and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json'),
            'Current process is not the registered render command')
    base=Path(owner_cfg['owner_out']);oid_path=base/'owner_identity.json';oid=read(oid_path)
    require(oid['registration_sha256']==sha(cfg['registration']) and int(oid['pid'])==os.getppid()
            and a.same_identity(oid,a.identity(os.getppid())),'Registered visual owner is not the live parent')
    require(not (base/'failure.json').exists(),'Visual owner has failed')
    path=base/'stages'/'render'/'workers'/job['id']/'launch.json';began=time.monotonic()
    while not path.exists():
        require(time.monotonic()-began<10 and a.same_identity(oid,a.identity(os.getppid())),
                'Owner did not seal current worker identity')
        time.sleep(.1)
    launch=read(path);current=a.identity(os.getpid())
    require(launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='gpu'
            and launch['argv']==expected and a.same_identity(launch['identity'],current)
            and launch['threads']==owner_cfg['gpu_threads'] and set(launch['affinity'])==set(owner_cfg['gpu_affinity'])
            and set(os.sched_getaffinity(0))==set(owner_cfg['gpu_affinity'])
            and os.environ.get('CUDA_VISIBLE_DEVICES')==str(owner_cfg['gpu_device']), 'Visual launch/runtime identity differs')
    gp=base/'stages'/'render'/'gpu_admission.json';admission=read(gp)
    require(admission['status']=='GPU_IDLE_CONFIRMED' and admission['registration_sha256']==sha(cfg['registration']),
            'Original owner did not admit the exclusive GPU')
    return dict(owner_identity=oid,worker_identity=current,bindings={str(p):sha(p) for p in (oid_path,path,gp)})


def load_source(ctx,index):
    cfg=ctx['cfg'];base=Path(ctx['cpu_ctx']['cfg']['out'])/f'worker_{index%2}'
    cp_path=base/'source_checkpoints'/f'{index:04d}.json';trace_path=base/'traces'/f'{index:04d}.json'
    done=ctx['completion'];require(done['outputs'].get(str(cp_path))==sha(cp_path)
        and done['outputs'].get(str(trace_path))==sha(trace_path),'CPU source evidence not sealed')
    cp=read(cp_path);source=read(trace_path)
    require(cp['outputs'].get(str(trace_path))==sha(trace_path) and cp['source_id']==source['source_id']==ctx['ids'][index]
            and cp['source_index']==source['source_index']==index
            and cp['registration_sha256']==source['registration_sha256']==sha(cfg['cpu_registration'])
            and source['status']=='H_INITIAL_CPU_SOURCE_TRACES','CPU trace source provenance differs')
    verify(source['source_bindings'])
    s1cp_path=Path(cfg['S1_assets_completion']).parent/'source_checkpoints'/f'{index:04d}.json'
    require(ctx['assets']['outputs'].get(str(s1cp_path))==sha(s1cp_path),'Original target checkpoint unbound')
    s1cp=read(s1cp_path);verify(s1cp['outputs'])
    require(s1cp['source_id']==ctx['ids'][index] and s1cp['source_index']==index,'Original target identity differs')
    require(s1cp['archive'] in s1cp['outputs'],'Target archive missing from source receipt')
    with np.load(s1cp['archive'],allow_pickle=False) as z:target=z['pixels'].copy()
    require(target.dtype==np.uint8 and target.shape==(3,256,256)
            and hashlib.sha256(target.tobytes()).hexdigest()==s1cp['preprocessing_id'],'Target preprocessing differs')
    return source,target,s1cp['preprocessing_id'],{str(cp_path):sha(cp_path),str(trace_path):sha(trace_path),str(s1cp_path):sha(s1cp_path),
        s1cp['archive']:s1cp['outputs'][s1cp['archive']]}


def render_source(source,target,target_sha,candidates,receiver,rx_api,cache_identity,boundary):
    candidates={c['slot']:c for c in candidates};expected={(s,n) for s in candidates for n in SEEDS};seen=set()
    cache=CachedReceiver(receiver,cache_identity);rows=[];images={};image_by_sha={}
    for frame in sorted(source['frames'],key=lambda f:(f['candidate_slot'],f['noise_seed'])):
        boundary();slot,seed=frame['candidate_slot'],frame['noise_seed']
        require((slot,seed) in expected and (slot,seed) not in seen,'Unknown/duplicate source frame');seen.add((slot,seed));c=candidates[slot]
        require(frame['source_id']==source['source_id'] and frame['source_index']==source['source_index']
                and frame['candidate_id']==c['candidate_id'] and frame['arm']==c['arm'] and frame['snr_db']==c['snr_db'],
                'Frame/source/frozen candidate identity differs')
        result=cache.reconstruct(rx_api.receiver_view(frame),frame['event_id'])
        scores=rx_api.score_reconstruction(result,target,expected_preprocessing_sha256=target_sha)
        diagnostics=rx_api.evaluation_diagnostics(result,frame['evaluation_only'])
        s=result['summary'];require(s['source_decode_complete'] is True and s['source_status'] in FINAL_STATUSES,'Unresolved RX cannot be scored')
        ih=s['image_sha256']
        if ih not in image_by_sha:
            key=f'image_{len(images):04d}';image_by_sha[ih]=key;images[key]=result['image'].copy()
        else:
            key=image_by_sha[ih];require(np.array_equal(images[key],result['image']),'Image content hash collision')
        rows.append(dict(candidate_id=c['candidate_id'],slot=slot,arm=c['arm'],target_m=c['target_m'],K=c.get('K',0),
            q=c['q'],nominal_rate=c['nominal_rate'],source_index=source['source_index'],source_id=source['source_id'],
            snr_db=c['snr_db'],noise_seed=seed,event_id=frame['event_id'],**scores,
            source_status=s['source_status'],gray=s['gray'],receiver_view_sha256=s['receiver_view_sha256'],
            image_key=key,rx_summary=s,evaluation_only=diagnostics))
    require(seen==expected,'Missing actual source frames')
    stats=dict(frames=len(rows),unique_images=len(images),receiver_cache_hits=cache.hits,receiver_cache_misses=cache.misses,
        arithmetic_canonical_calls=sum(r['rx_summary']['arithmetic_canonical_executed_this_frame'] for r in rows),
        suffix_renderer_calls=sum(r['rx_summary']['suffix_renderer_executed_this_frame'] for r in rows),
        gray_frames=sum(r['gray'] for r in rows))
    return rows,images,stats


def validate_source_archive(rows,archive,rx_api):
    with np.load(archive,allow_pickle=False) as z:
        require(set(z.files)=={r['image_key'] for r in rows},'Image archive reference coverage differs')
        hashes={key:rx_api.phy.array_sha(rx_api.validate_image(z[key])) for key in z.files}
    for r in rows:
        s=r['rx_summary'];require(hashes[r['image_key']]==r['image_sha256']==s['image_sha256']
            and r['receiver_view_sha256']==s['receiver_view_sha256'] and s['source_decode_complete'] is True,
            'Per-frame RGB/RX evidence differs')


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];regsha=sha(cfg['registration']);out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    require(not (out/'failure.json').exists(),'Prior failure preserved; new recovery registration required')
    import fcntl
    with (out/'execution.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'completion.json').exists():
            done=read(out/'completion.json');require(done['registration_sha256']==regsha and done['status']=='H_INITIAL_TRUE200_RX_COMPLETE','Existing different completion');verify(done['outputs']);return done
        require(not (out/'attempt.json').exists(),'Partial render attempt requires independent recovery; no blind retry')
        native=None;start=time.monotonic();started=time.time()
        try:
            supervision=verify_live_visual_owner(ctx,config_path)
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists()
                    and time.time()<DEADLINE,'STOP/deadline before visual model construction')
            atomic(out/'attempt.json',dict(status='STARTED',registration_sha256=regsha,config_sha256=sha(config_path),
                predecessor_cpu_registration_sha256=sha(cfg['cpu_registration']),new_packet_decodes=0,policy_selection=False),exclusive=True)
            for path in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src'),str(Path(cfg['source_driver_module']).parent)):
                sys.path.insert(0,path)
            import quality_driver
            from var_comm import whole_entropy,entropy
            from h64_source import reference_primitives
            rx_api=module(cfg['rx_module'],'registered_h_payload_rx')
            module(cfg['source_driver_module'],'h_source_driver')
            native=quality_driver.build_native(Path(cfg['root']),cfg['native_runtime'],stop)
            require(native.loaded['identity']==read(cfg['calibration_registration'])['identity'],'Frozen visual identity changed')
            bound=dict(ctx['reg']['input_bindings'],**ctx['reg']['source_bindings'])
            require(all(bound.get(p)==s for p,s in native.driver_bindings.items()),'Loaded visual source closure differs from registration')
            receiver=rx_api.frozen_native_receiver(native,reference_primitives(whole_entropy,entropy),ctx['catalogue'])
            cache_identity=dict(visual_identity=native.loaded['identity'],receiver_sources=ctx['reg']['source_bindings'],
                                generation=ctx['protocol']['source'],cache_version='EXACT_ACTUAL_RX_PER_SOURCE_V1')
            def boundary():
                require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at frame/source boundary')
                require(time.monotonic()-start<=cfg['max_seconds'] and max(time.time(),started+time.monotonic()-start)<DEADLINE,
                        'Registered render deadline exhausted')
                require(not quality_driver.boundary(native),'Original GPU guard stopped render')
                require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),
                        'Registered GPU owner disappeared/changed')
            outputs={};allrows=[];stats=[]
            for index in range(200):
                boundary();source,target,target_sha,inputs=load_source(ctx,index)
                rows,images,cost=render_source(source,target,target_sha,ctx['candidates'],receiver,rx_api,cache_identity,boundary)
                archive=out/'images'/f'{index:04d}.npz';archive.parent.mkdir(parents=True,exist_ok=True)
                with archive.with_suffix('.tmp').open('xb') as f:np.savez(f,**images)
                os.replace(archive.with_suffix('.tmp'),archive)
                for row in rows:row['image_archive']=str(archive)
                validate_source_archive(rows,archive,rx_api)
                metrics=out/'sources'/f'{index:04d}.json';atomic(metrics,rows,exclusive=True)
                cp=out/'source_checkpoints'/f'{index:04d}.json'
                local={str(archive):sha(archive),str(metrics):sha(metrics)}
                atomic(cp,dict(status='H_INITIAL_RX_SOURCE_COMPLETE',registration_sha256=regsha,config_sha256=sha(config_path),
                    source_id=ctx['ids'][index],source_index=index,frame_count=len(rows),input_bindings=inputs,
                    outputs=local,source_decode_complete=True,images_scored=True,new_packet_decodes=0,**cost),exclusive=True)
                outputs.update(local);outputs[str(cp)]=sha(cp);allrows.extend(rows);stats.append(cost)
                atomic(out/'status.json',dict(status='RUNNING',registration_sha256=regsha,completed_sources=index+1,total_sources=200,
                    completed_frames=len(allrows),elapsed_seconds=time.monotonic()-start))
            require(len(allrows)==ctx['completion']['frame_count'],'Actual frame coverage incomplete')
            boundary();native.frozen()
            framejson=out/'frame_metrics.json';atomic(framejson,allrows,exclusive=True);outputs[str(framejson)]=sha(framejson)
            csvpath=out/'frame_metrics.csv';fields=['candidate_id','slot','arm','target_m','K','q','nominal_rate','source_index','source_id',
                'snr_db','noise_seed','mse','psnr_db','source_status','gray','image_sha256','receiver_view_sha256','image_archive','image_key']
            with csvpath.open('x',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:r[k] for k in fields} for r in allrows)
            outputs[str(csvpath)]=sha(csvpath)
            counts={s:sum(r['source_status']==s for r in allrows) for s in sorted(FINAL_STATUSES)}
            costs={k:sum(x[k] for x in stats) for k in stats[0]};costpath=out/'receiver_cost_counts.json'
            atomic(costpath,dict(status='PREPARATION_CALL_COUNTS_NOT_ONLINE_TIMING',**costs,
                cache='Exact actual RX profile/payload and bound receiver identity per source; no TX-correct shortcut',
                new_packet_decodes=0),exclusive=True);outputs[str(costpath)]=sha(costpath)
            after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
            require(after==ctx['before'],'Image stage changed original packet ledger')
            verify(ctx['reg']['source_bindings']);verify(ctx['reg']['input_bindings']);verify(outputs)
            done=dict(status='H_INITIAL_TRUE200_RX_COMPLETE',registration_sha256=regsha,config_sha256=sha(config_path),
                cpu_completion_sha256=sha(cfg['cpu_completion']),shortlist_sha256=sha(cfg['shortlist']),
                source_count=200,frame_count=len(allrows),source_status_counts=counts,receiver_cost_counts=costs,
                outputs=outputs,input_bindings=ctx['reg']['input_bindings'],source_bindings=ctx['reg']['source_bindings'],
                predecessor_closure_bindings=ctx['closure']['bindings'],images_scored=True,source_decode_complete=True,
                visual_launch_bindings=supervision['bindings'],
                new_packet_decodes=0,budget_unchanged=True,GPU_used=True,policy_selection=False,
                development_used=False,holdout_used=False,H_full_delivery_claimed=False,training_updates=0,
                elapsed_seconds=time.monotonic()-start)
            atomic(out/'completion.json',done,exclusive=True);atomic(out/'status.json',done);return done
        except BaseException as error:
            if not (out/'failure.json').exists():atomic(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',
                registration_sha256=regsha,error=repr(error),traceback=traceback.format_exc(),new_packet_decodes=0),exclusive=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    result=run(args.config);print(json.dumps({k:result[k] for k in ('status','registration_sha256','source_count','frame_count')}))


if __name__=='__main__':main()
