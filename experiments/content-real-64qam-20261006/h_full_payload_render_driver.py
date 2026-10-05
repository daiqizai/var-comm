"""Registered single-GPU actual-RX reconstruction of whole+partial calibration1000.

Both metered CPU groups must be complete and reaped. Uses the unchanged
conditional-free frozen receiver, including canonical arithmetic rejection.
Makes no PHY call, policy choice, clean-image substitution or new metric search.
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
import numpy as np

STOP=False
DEADLINE=1791564605.9549868
DONE='H_FULL1000_RX_COMPLETE'
MAX_ARCHIVE_BYTES=36*1024**3
STORAGE_RESERVE_BYTES=8*1024**3


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def progress(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def stop(*_):
    global STOP
    STOP=True


def claim_output(out,regsha):
    out=Path(out)
    if out.exists():require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior visual attempt preserved; independent recovery required')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_FULL1000_RX_ATTEMPT',registration_sha256=regsha,pid=os.getpid(),started_unix=time.time(),new_packet_decodes=0))


def check_storage(out,cap):
    require(shutil.disk_usage(out).free>=cap+STORAGE_RESERVE_BYTES,'Insufficient registered image storage headroom')


def load_registered(config_path):
    cfg=read(config_path);reg=read(cfg['registration']);path=str(Path(__file__).resolve())
    require(reg['source_bindings'].get(cfg['rx_adapter_module'])==sha(cfg['rx_adapter_module'])
        and reg['source_bindings'].get(path)==sha(path),'New full-RX entry/adapter source unbound')
    adapter=module(cfg['rx_adapter_module'],'registered_full1000_rx_adapter');ctx=adapter.load_registered(config_path)
    require(cfg['visual_driver_module']==path,'Registered visual entry differs')
    hout,out=Path(cfg['H_out']),Path(cfg['out'])
    require(hout.is_absolute() and hout in out.parents and cfg['stop_file']==str(hout/'STOP')
        and cfg['overall_deadline_unix']==DEADLINE and 0<cfg['max_seconds']<=86400,'Visual H output/deadline scope differs')
    require(type(cfg['max_archive_bytes']) is int and 0<cfg['max_archive_bytes']<=MAX_ARCHIVE_BYTES,'Registered RGB storage cap required')
    for group in ctx['groups'].values():
        old=Path(group['cfg']['out']);require(out!=old and out not in old.parents and old not in out.parents,'Visual output overlaps metered CPU evidence')
    old=Path(ctx['groups']['whole_calibration']['context']['cfg']['asset_completion']).parent
    require(out!=old and out not in old.parents and old not in out.parents,'Visual output overlaps original source assets')
    ctx['adapter']=adapter;return ctx


def visual_admission(ctx,config_path):
    cfg=ctx['cfg'];a=ctx['owner'];c=read(cfg['visual_owner_config']);a.validate_config(c,ctx['reg'],sha(cfg['visual_owner_config']))
    require(c['registration']==cfg['registration'] and c['out']==cfg['H_out'] and len(c['stages'])==1,'Independent visual owner scope differs')
    stage=c['stages'][0];require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,'Exactly one full visual GPU job required')
    job=stage['jobs'][0];argv=job['argv'];entry=a.command_entry(argv);offset=2 if argv[1]=='-B' else 1
    require(entry==Path(__file__).resolve() and argv[offset+1:]==['--config',str(Path(config_path).resolve())]
        and job['id']=='full1000_render' and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json'),
        'Full visual process is not the registered command')
    base=Path(c['owner_out']);op=base/'owner_identity.json';oid=read(op)
    require(int(oid['pid'])==os.getppid() and a.same_identity(oid,a.identity(os.getppid()))
        and oid['registration_sha256']==sha(cfg['registration']) and oid['config_sha256']==sha(cfg['visual_owner_config'])
        and not (base/'failure.json').exists(),'Registered exclusive visual owner is not the live parent')
    lp=base/'stages/render/workers/full1000_render/launch.json';began=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-began<10 and a.same_identity(oid,a.identity(os.getppid())),'Owner did not seal current visual worker');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid())
    require(launch['registration_sha256']==sha(cfg['registration']) and launch['resource']=='gpu' and launch['argv']==argv
        and a.same_identity(launch['identity'],current) and launch['threads']==c['gpu_threads']==6
        and set(launch['affinity'])==set(c['gpu_affinity'])==set(os.sched_getaffinity(0))
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(c['gpu_device']),'Visual process identity/numeric resource differs')
    gp=base/'stages/render/gpu_admission.json';g=read(gp)
    require(g['status']=='GPU_IDLE_CONFIRMED' and g['registration_sha256']==sha(cfg['registration']),'Original exclusive GPU admission absent')
    return dict(owner_identity=oid,worker_identity=current,bindings={str(p):sha(p) for p in (op,lp,gp)})


def validate_native(native,ctx):
    require(native.loaded['identity']==ctx['calibration']['identity'] and native.flags==ctx['expected_flags'],
        'Frozen visual identity or original FP32 numerical flags differ')
    require(native.driver_bindings==ctx['static_bindings'] and all(ctx['bound'].get(p)==s for p,s in native.driver_bindings.items()),
        'Full loaded visual source closure differs from registration')


def write_source(out,index,sid,rows,images,cost,inputs,regsha,cache_api,rx_api):
    """Seal actual float32 RGB once per source/image value, all42 rows retained."""
    require(len(rows)==42 and all(r['source_index']==index and r['source_id']==sid for r in rows),'Full received source grid differs')
    out=Path(out);archive=out/'images'/f'{index:04d}.npz';archive.parent.mkdir(parents=True,exist_ok=True)
    tmp=archive.with_suffix('.tmp')
    with tmp.open('xb') as f:np.savez(f,**images)
    os.replace(tmp,archive)
    for row in rows:row['image_archive']=str(archive)
    cache_api.validate_source_archive(rows,archive,rx_api)
    values=out/'sources'/f'{index:04d}.json';save(values,rows)
    cp=out/'source_checkpoints'/f'{index:04d}.json';outputs={str(p):sha(p) for p in (archive,values)}
    save(cp,dict(status='H_FULL1000_RX_SOURCE_COMPLETE',registration_sha256=regsha,source_index=index,source_id=sid,
        frame_count=42,phase_frame_counts={'whole_calibration':24,'partial_calibration':18},input_bindings=inputs,outputs=outputs,
        images_scored=True,source_decode_complete=True,**cost))
    outputs[str(cp)]=sha(cp)
    return outputs,archive.stat().st_size


def validate_complete_rows(rows,ids):
    require(len(ids)==len(set(ids))==1000 and len(rows)==42000,'Full received population/frame coverage incomplete')
    expected={(i,s,n) for i in range(1000) for s in range(14) for n in (6101,6102,6103)};seen=set();counts={p:0 for p in ('whole_calibration','partial_calibration')}
    for row in rows:
        key=(row['source_index'],row['full_slot'],row['noise_seed'])
        require(key in expected and key not in seen and row['source_id']==ids[key[0]],'Full received row duplicate/order/population differs');seen.add(key)
        phase='whole_calibration' if key[1]<8 else 'partial_calibration'
        require(row['phase']==phase and row['rx_summary']['source_decode_complete'] is True,'Full actual received state not complete')
        counts[phase]+=1
    require(seen==expected and counts=={'whole_calibration':24000,'partial_calibration':18000},'Final whole/partial source coverage differs')
    return counts


def run(config_path):
    ctx=load_registered(config_path);cfg=ctx['cfg'];adapter=ctx['adapter'];out=Path(cfg['out']);regsha=sha(cfg['registration'])
    claim_output(out,regsha);began=time.monotonic();started=time.time();native=None
    try:
        supervision=visual_admission(ctx,config_path)
        check_storage(out,cfg['max_archive_bytes'])
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP at complete source boundary')
            elapsed=time.monotonic()-began
            require(elapsed<cfg['max_seconds'] and max(time.time(),started+elapsed)<DEADLINE,'Original registered visual deadline')
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),'Exclusive visual owner changed/disappeared')
            if native is not None:health()
        def health():
            require(not quality_driver.boundary(native),'Original GPU guard requested safe stop')
        boundary()
        for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
        import quality_driver
        from var_comm import whole_entropy,entropy
        from h64_source import reference_primitives
        # h_payload_rx imports this exact original module by its public name.
        module(cfg['source_driver_module'],'h_source_driver')
        native=quality_driver.build_native(Path(cfg['root']),cfg['native_runtime'],stop);validate_native(native,ctx)
        receiver=ctx['rx_api'].frozen_native_receiver(native,reference_primitives(whole_entropy,entropy),ctx['catalogue'])
        cache_identity=dict(frozen_visual_identity=native.loaded['identity'],numerical_runtime=native.flags,
            receiver_sources=ctx['reg']['source_bindings'],source_protocol=read(cfg['protocol'])['source'],
            cache_version='FULL1000_SAME_SOURCE_ACTUAL_RX_PROFILE_PAYLOAD_V1')
        outputs={};allrows=[];costs=[];imagebytes=0
        for index,sid in enumerate(ctx['source_ids']):
            boundary();sources,target,targetsha,inputs=adapter.load_source(ctx,index)
            rows,images,cost=adapter.render_source(sources,target,targetsha,ctx['schedule'],ctx['core'],receiver,ctx['rx_api'],
                ctx['cache_api'],cache_identity,health)
            local,size=write_source(out,index,sid,rows,images,cost,inputs,regsha,ctx['cache_api'],ctx['rx_api'])
            imagebytes+=size;require(imagebytes<=cfg['max_archive_bytes'],'Registered image storage cap reached; preserve all outputs')
            outputs.update(local);allrows.extend(rows);costs.append(cost)
            progress(out/'status.json',dict(status='RUNNING',registration_sha256=regsha,completed_sources=index+1,total_sources=1000,
                completed_frames=len(allrows),image_archive_bytes=imagebytes,elapsed_seconds=time.monotonic()-began))
            del images,sources,target
        phase_counts=validate_complete_rows(allrows,ctx['source_ids']);boundary();native.frozen();validate_native(native,ctx)
        after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        require(after==ctx['before'],'Visual stage changed frozen packet budget')
        rows_path=out/'frame_metrics.json';save(rows_path,allrows);outputs[str(rows_path)]=sha(rows_path)
        csvpath=out/'frame_metrics.csv';fields=['phase','full_slot','slot','candidate_id','arm','target_m','K','q','nominal_rate','snr_db',
            'source_index','source_id','noise_seed','mse','psnr_db','source_status','gray','received_m','received_K','received_mode',
            'received_profile_id','image_sha256','receiver_view_sha256','image_archive','image_key']
        with csvpath.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in allrows)
        outputs[str(csvpath)]=sha(csvpath)
        counts={k:sum(c[k] for c in costs) for k in ('frames','unique_images','receiver_cache_hits','receiver_cache_misses','arithmetic_canonical_calls','suffix_renderer_calls','gray_frames')}
        costpath=out/'receiver_cost_counts.json';save(costpath,dict(status='ACTUAL_CALL_COUNTS_NOT_ONLINE_TIMING',**counts,
            cache_scope='same source actual accepted RX profile/payload only, every frame revalidated; no clean-source shortcut',
            image_archive_bytes=imagebytes,new_packet_decodes=0));outputs[str(costpath)]=sha(costpath)
        adapter.verify(ctx['bound']);adapter.verify(outputs)
        done=dict(status=DONE,registration_sha256=regsha,config_sha256=sha(config_path),source_count=1000,frame_count=42000,
            phase_frame_counts=phase_counts,source_ids=ctx['source_ids'],source_status_counts={s:sum(r['source_status']==s for r in allrows) for s in sorted(adapter.FINAL)},
            receiver_cost_counts=counts,image_archive_bytes=imagebytes,outputs=outputs,input_bindings=ctx['reg']['input_bindings'],
            source_bindings=ctx['reg']['source_bindings'],cpu_completion_sha256={p:sha(cfg['cpu_batches'][p]['completion']) for p in adapter.PHASES},
            predecessor_closure_bindings=adapter.merge(*[ctx['groups'][p]['closure']['bindings'] for p in adapter.PHASES]),
            supervision=supervision,numerical_runtime=native.flags,frozen_visual_identity=native.loaded['identity'],
            visual_source_bindings=native.driver_bindings,budget_before=ctx['before'],budget_after=after,
            images_scored=True,source_decode_complete=True,arithmetic_source_decode_complete=True,policy_selection=False,
            new_packet_decodes=0,initial_physical_results_reused=False,clean_image_proxy_used=False,
            development_used=False,holdout_used=False,training_updates=0,GPU_used=True,
            full1000_calibration_complete=False,H_full_delivery_claimed=False,online_timing_measured=False,
            elapsed_seconds=time.monotonic()-began)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,traceback=traceback.format_exc(),
            new_packet_decodes=0,original_cpu_and_source_results_preserved=True));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'status':run(args.config)['status']}))


if __name__=='__main__':main()
