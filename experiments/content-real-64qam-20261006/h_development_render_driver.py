"""Separate single-GPU actual-RX reconstruction of the frozen H18 development grid.

Metered CPU receive must close normally first. This entry makes no PHY call,
policy choice, MAIN reconstruction, training update or clean-source substitution.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

DEADLINE=1791564605.9549868
MAX_ARCHIVE_BYTES=36*1024**3
STORAGE_RESERVE_BYTES=8*1024**3
DONE='H_DEVELOPMENT_RX_COMPLETE'
SCOPE='H_DEVELOPMENT_18_POINTS_ACTUAL_RX_ONLY'
STOP=False
REQUIRED=('owner_module','wait_module','cpu_driver_module','receiver_module','cache_module','source_driver_module',
          'protocol','budget_registration','calibration_registration','numerical_reference','visual_owner_config',
          'static_closure_module','visual_driver_module','rx_adapter_module','visual_source_closure')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting SHA: '+p);result[p]=s
    return result
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound file: '+p)
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def progress(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def stop(*_):
    global STOP
    STOP=True


def original_source_config(group):
    source=group['context']['closures']['source'];owner=source['owner'];reg=source['registration']
    require(len(owner['stages'])==1 and len(owner['stages'][0]['jobs'])==1,'One original dev100 source job required')
    job=owner['stages'][0]['jobs'][0];argv=job['argv']
    require(argv.count('--config')==1,'Actual source config ambiguous')
    path=argv[argv.index('--config')+1]
    require(reg['input_bindings'].get(path)==sha(path),'Actual source config not sealed')
    return read(path),source,path


def visual_identity(source_cfg,source_done,population,numerical,calibration):
    require(numerical['status']=='REAL_NATIVE_QUALIFICATION_PASS' and numerical['synthetic'] is False
        and numerical['frozen_identity']==source_done['frozen_visual_identity']==population['identity']==calibration['identity'],
        'Original source/development/visual identities differ')
    flags=numerical['numerical_runtime']
    require(flags==source_done['numerical_runtime'] and flags['threads']==6 and flags['deterministic'] is True
        and flags['matmul_tf32'] is False and flags['cudnn_tf32'] is False
        and flags['cudnn_benchmark'] is False and flags['precision']=='highest','Original FP32 numerical flags differ')
    return flags


def assert_budget(current,registered,group):
    require(current==registered==group['before'] and current['failed']==current['unresolved']==0
        and current['charged']==164760 and current['phase_charged']['development']==10800
        and current['development_remaining']==2400,'Complete H18 paid budget changed or MAIN reserve consumed')


def load_registered(config_path):
    path=Path(config_path).absolute();cfg=read(path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_DEVELOPMENT_RX_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['branch']=='H' and reg['allowed_stage_ids']==['render'] and reg['source_stage_scope']==SCOPE,
        'Independent development-only visual registration required')
    bound=merge(reg['input_bindings'],reg['source_bindings']);verify(bound)
    for p in (str(path),str(Path(__file__).absolute()),*[cfg[k] for k in REQUIRED]):
        require(bound.get(p)==sha(p),'Required development visual source/input unbound: '+p)
    require(cfg['visual_driver_module']==str(Path(__file__).absolute()),'Registered visual entry differs')
    spec=cfg['cpu_batch'];require(set(spec)=={'config','owner_config','registration','launch','completion'},'Unexpected CPU predecessor')
    for p in spec.values():require(bound.get(p)==sha(p),'Unbound real development CPU predecessor: '+p)
    owner=module(cfg['owner_module'],'development_rx_bound_owner');wait=module(cfg['wait_module'],'development_rx_bound_wait')
    driver=module(cfg['cpu_driver_module'],'development_rx_bound_cpu');adapter=module(cfg['rx_adapter_module'],'development_actual_rx_adapter')
    group=adapter.verify_cpu_closed(spec,driver,owner,wait)
    for k in ('root','runtime_dir','protocol','budget_registration','ledger','phase_limits','owner_module','wait_module','stop_file','H_out'):
        require(cfg[k]==group['cfg'][k],'Visual and physical prerequisites differ: '+k)
    source_cfg,source,source_path=original_source_config(group)
    for k in ('native_runtime','uep_runtime','var_source','dino_source','calibration_registration','numerical_reference',
              'source_driver_module','static_closure_module'):
        require(cfg[k]==source_cfg[k],'Frozen dev100 visual source prerequisite changed: '+k)
    before=owner.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    assert_budget(before,reg['budget_before'],group)
    require(cfg['phase_limits']==reg['phase_limits'],'Visual phase quotas differ')
    hout,out=Path(cfg['H_out']),Path(cfg['out'])
    require(hout.is_absolute() and hout in out.parents and cfg['stop_file']==str(hout/'STOP')
        and cfg['overall_deadline_unix']==DEADLINE and type(cfg['max_seconds'])is int and 0<cfg['max_seconds']<=86400,
        'Output/deadline scope differs')
    require(type(cfg['max_archive_bytes'])is int and 0<cfg['max_archive_bytes']<=MAX_ARCHIVE_BYTES,'Finite RGB archive cap required')
    for old in (Path(group['cfg']['out']),Path(group['cfg']['source_completion']).parent,Path(group['cfg']['asset_completion']).parent):
        require(out!=old and out not in old.parents and old not in out.parents,'New visual output overlaps immutable input')
    require(not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP blocks visual execution')
    numerical,cal=read(cfg['numerical_reference']),read(cfg['calibration_registration'])
    flags=visual_identity(source_cfg,source['done'],group['population'],numerical,cal)
    graph=read(cfg['visual_source_closure']);closure=module(cfg['static_closure_module'],'development_static_visual_closure')
    actual=closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
    require(graph['status']=='EXACT_SOURCE_CLOSURE_MATCH' and graph['source_bindings']==actual
        and closure.compare_bindings(actual,bound)['status']=='EXACT_SOURCE_CLOSURE_MATCH','Complete current visual source closure differs')
    for p,s in source['done']['visual_source_bindings'].items():require(actual.get(p)==s,'Previously executed visual source changed: '+p)
    receiver=module(cfg['receiver_module'],'development_rx_frozen_receiver');cache=module(cfg['cache_module'],'development_rx_frozen_cache')
    return dict(cfg=cfg,reg=reg,bound=bound,owner=owner,wait=wait,cpu=driver,adapter=adapter,group=group,
        source_ids=group['source_ids'],schedule=group['schedule'],core=group['core'],before=before,
        expected_identity=group['population']['identity'],expected_flags=flags,static_bindings=actual,
        rx_api=receiver,cache_api=cache,catalogue=group['catalogue'])


def claim_output(out,regsha):
    out=Path(out)
    if out.exists():require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior visual attempt preserved; no retry')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_DEVELOPMENT_RX_ATTEMPT',registration_sha256=regsha,pid=os.getpid(),started_unix=time.time(),new_packet_decodes=0))


def visual_admission(ctx,config_path):
    cfg=ctx['cfg'];a=ctx['owner'];c=read(cfg['visual_owner_config']);a.validate_config(c,ctx['reg'],sha(cfg['visual_owner_config']))
    require(c['registration']==cfg['registration'] and c['out']==cfg['H_out'] and len(c['stages'])==1,'Independent GPU owner required')
    stage=c['stages'][0];require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,'Exactly one development GPU job required')
    job=stage['jobs'][0];argv=job['argv']
    require(a.command_entry(argv)==Path(__file__).absolute() and argv[1:]==['-B',str(Path(__file__).absolute()),'--config',str(Path(config_path).absolute())]
        and job['id']=='development_render' and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json'),
        'Current GPU command is not registered')
    base=Path(c['owner_out']);ip=base/'owner_identity.json';oid=read(ip)
    require(int(oid['pid'])==os.getppid() and a.same_identity(oid,a.identity(os.getppid()))
        and oid['registration_sha256']==sha(cfg['registration']) and oid['config_sha256']==sha(cfg['visual_owner_config'])
        and not (base/'failure.json').exists(),'Registered exclusive GPU owner is not current live parent')
    lp=base/'stages/render/workers/development_render/launch.json';began=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-began<10 and a.same_identity(oid,a.identity(os.getppid())),'Owner did not seal current GPU worker');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid())
    require(launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='gpu' and launch['argv']==argv
        and a.same_identity(launch['identity'],current) and launch['threads']==c['gpu_threads']==6
        and set(launch['affinity'])==set(c['gpu_affinity'])==set(os.sched_getaffinity(0))
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu_device']),'Visual process identity/resources differ')
    gp=base/'stages/render/gpu_admission.json';g=read(gp)
    require(g['status']=='GPU_IDLE_CONFIRMED' and g['registration_sha256']==sha(cfg['registration']),'Original exclusive GPU admission absent')
    return dict(owner_identity=oid,worker_identity=current,bindings=bind((ip,lp,gp)))


def validate_native(native,ctx):
    require(native.loaded['identity']==ctx['expected_identity'] and native.flags==ctx['expected_flags'],'Frozen visual identity/FP32 flags differ')
    require(native.driver_bindings==ctx['static_bindings']
        and all(ctx['bound'].get(p)==s for p,s in native.driver_bindings.items()),'Loaded complete visual source closure differs')


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];adapter=ctx['adapter'];out=Path(cfg['out']);regsha=sha(cfg['registration'])
    claim_output(out,regsha);began=time.monotonic();started=time.time();native=None
    try:
        supervision=visual_admission(ctx,config_path)
        require(shutil.disk_usage(out).free>=cfg['max_archive_bytes']+STORAGE_RESERVE_BYTES,'Insufficient registered image storage headroom')
        def health():require(not quality_driver.boundary(native),'Original GPU guard requested safe stop')
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at complete source boundary')
            elapsed=time.monotonic()-began
            require(elapsed<cfg['max_seconds'] and max(time.time(),started+elapsed)<DEADLINE,'Original H visual deadline')
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),'Exclusive GPU owner disappeared')
            if native is not None:health()
        boundary()
        for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
        import quality_driver
        from var_comm import whole_entropy,entropy
        from h64_source import reference_primitives
        module(cfg['source_driver_module'],'h_source_driver')
        native=quality_driver.build_native(Path(cfg['root']),cfg['native_runtime'],stop);validate_native(native,ctx)
        receiver=ctx['rx_api'].frozen_native_receiver(native,reference_primitives(whole_entropy,entropy),ctx['catalogue'])
        cache_identity=dict(frozen_visual_identity=native.loaded['identity'],numerical_runtime=native.flags,
            receiver_sources=ctx['reg']['source_bindings'],source_protocol=read(cfg['protocol'])['source'],
            cache_version='DEVELOPMENT100_SAME_SOURCE_ACTUAL_RX_PROFILE_PAYLOAD_V1')
        outputs={};allrows=[];costs=[];imagebytes=0
        for index,sid in enumerate(ctx['source_ids']):
            boundary();source,target,targetsha,inputs=adapter.load_source(ctx['group'],index)
            rows,images,cost=adapter.render_source(source,target,targetsha,ctx['schedule'],ctx['core'],receiver,
                ctx['rx_api'],ctx['cache_api'],cache_identity,health)
            local,size=adapter.write_source(out,index,sid,rows,images,cost,inputs,regsha,ctx['cache_api'],ctx['rx_api'])
            imagebytes+=size;require(imagebytes<=cfg['max_archive_bytes'],'Registered RGB storage cap reached; preserve outputs')
            outputs.update(local);allrows.extend(read(out/'sources'/f'{index:04d}.json'));costs.append(cost)
            progress(out/'status.json',dict(status='RUNNING',registration_sha256=regsha,completed_sources=index+1,total_sources=100,
                completed_frames=len(allrows),image_archive_bytes=imagebytes,elapsed_seconds=time.monotonic()-began))
            del images,source,target
        adapter.validate_complete_rows(allrows,ctx['source_ids']);boundary();native.frozen();validate_native(native,ctx)
        after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        assert_budget(after,ctx['before'],ctx['group'])
        values=out/'frame_metrics.json';save(values,allrows);outputs[str(values)]=sha(values)
        csvpath=out/'frame_metrics.csv';fields=['phase','development_slot','role','candidate_id','arm','target_m','K','q','nominal_rate','snr_db',
            'source_index','source_id','noise_seed','mse','psnr_db','source_status','gray','received_m','received_K','received_mode',
            'received_profile_id','image_sha256','receiver_view_sha256','image_archive','image_key']
        with csvpath.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in allrows)
        outputs[str(csvpath)]=sha(csvpath)
        counts={k:sum(c[k] for c in costs) for k in ('frames','unique_images','receiver_cache_hits','receiver_cache_misses','arithmetic_canonical_calls','suffix_renderer_calls','gray_frames')}
        costpath=out/'receiver_cost_counts.json';save(costpath,dict(status='ACTUAL_CALL_COUNTS_NOT_ONLINE_TIMING',**counts,
            cache_scope='same source actual accepted RX profile/payload, every frame revalidated',image_archive_bytes=imagebytes,new_packet_decodes=0))
        outputs[str(costpath)]=sha(costpath);verify(ctx['bound']);verify(outputs)
        done=dict(status=DONE,registration_sha256=regsha,config_sha256=sha(config_path),source_count=100,frame_count=5400,
            phase_frame_counts={'development':5400},H_policy_snr_points=18,MAIN_frames=0,source_ids=ctx['source_ids'],
            source_status_counts={s:sum(r['source_status']==s for r in allrows) for s in sorted(adapter.FINAL)},
            receiver_cost_counts=counts,image_archive_bytes=imagebytes,outputs=outputs,input_bindings=ctx['reg']['input_bindings'],
            source_bindings=ctx['reg']['source_bindings'],cpu_completion_sha256=sha(cfg['cpu_batch']['completion']),
            finalized_policies_sha256=ctx['group']['done']['finalized_policies_sha256'],
            predecessor_closure_bindings=ctx['group']['closure']['bindings'],supervision=supervision,numerical_runtime=native.flags,
            frozen_visual_identity=native.loaded['identity'],visual_source_bindings=native.driver_bindings,budget_before=ctx['before'],budget_after=after,
            images_scored=True,scored_metrics=['mse','psnr_db'],source_decode_complete=True,arithmetic_source_decode_complete=True,
            policy_selection=False,new_packet_decodes=0,clean_image_proxy_used=False,development_used=True,holdout_used=False,
            training_updates=0,GPU_used=True,MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False,
            unified_neural_metrics_run=False,online_timing_measured=False,elapsed_seconds=time.monotonic()-began)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,traceback=traceback.format_exc(),
            new_packet_decodes=0,original_cpu_and_source_results_preserved=True));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'status':run(a.config)['status']}))


if __name__=='__main__':main()
