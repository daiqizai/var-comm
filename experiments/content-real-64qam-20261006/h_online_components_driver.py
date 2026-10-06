"""Exclusive, registered H fixed16 online source/visual component measurement.

No physical decoder, waveform, noise, policy selection, or new quality sample.
The frozen timing core is used unchanged. Historical physical event windows
remain separate from freshly measured GPU/source components.
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
import shutil
import signal
import statistics
import sys
import time
import traceback
import numpy as np

DONE='H_FIXED16_ONLINE_COMPONENTS_COMPLETE'
SCOPE='H_FIXED16_18_POINTS_FRESH_ONLINE_COMPONENTS_NO_PHY'
CORE_SHA='31bb602e0204ffd16c1ee213f822ca1339a3df78950fd14775e580f6e4b71e5f'
DEADLINE=1791564605.9549868
STOP=False
MAX_OUTPUT_BYTES=512*1024**2
REQUIRED=('metadata_helper_module','timing_core_module','timing_driver_module','h_metric_driver_module',
    'owner_module','wait_module','cpu_driver_module','visual_owner_config','protocol','budget_registration',
    'quality_driver_module','source_driver_module','receiver_module','codec_module','static_closure_module',
    'visual_source_closure','numerical_reference')
PARITY_FIELDS=('source_status','gray','reason','received_profile_id','received_profile_key','received_m',
    'received_K','received_mode','header_accepted','body_crc_accepted','body_parser_accepted',
    'arithmetic_canonical_attempted','arithmetic_canonical','source_decode_complete','canonical_decode_invalid',
    'catalogue_sha256','receiver_view_sha256','image_sha256','actual_received_tokens_sha256',
    'new_packet_decodes','target_image_used_for_reconstruction','truth_correction','cached_clean_image_used')


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def merge(*maps):
    out={}
    for values in maps:
        for p,s in values.items():
            require(p not in out or out[p]==s,'Conflicting bound input: '+p);out[p]=s
    return out
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound input: '+p)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def stop(*_):
    global STOP
    STOP=True


def inspect_inputs(cfg,bound,*,prior=None):
    """Metadata-only registration admission; no new owner/reg file is required."""
    require(cfg['schema']=='H_ONLINE_COMPONENTS_CONFIG_V1','Independent timing configuration required')
    verify(bound)
    for p in (str(Path(__file__).absolute()),*[cfg[k] for k in REQUIRED if k!='visual_owner_config'],*cfg['metric_batch'].values()):
        require(bound.get(p)==sha(p),'Required timing source/input unbound: '+p)
    require(cfg['timing_driver_module']==str(Path(__file__).absolute()) and sha(cfg['timing_core_module'])==CORE_SHA,
            'Original frozen timing core/new entry differs')
    helper=module(cfg['metadata_helper_module'],'_timing_closed_metric_helpers')
    if prior is None:prior=helper.original_metrics_closed(cfg['metric_batch'],cfg['h_metric_driver_module'])
    else:
        require(prior['spec']==cfg['metric_batch'] and prior['metric_module']==cfg['h_metric_driver_module']
                and prior['closed']['normal_owner_success'] is True,'Wrong in-memory normal metric closure')
        for p in cfg['metric_batch'].values():require(prior['bindings'].get(p)==sha(p),'Predecessor changed during registration')
    pc,mc=prior['context'],prior['config'];visual=pc['render_cfg']
    require(all(bound.get(p)==s for p,s in prior['bindings'].items()),'Normal metric lineage not inherited')
    for k in ('root','H_out','owner_module','wait_module','cpu_driver_module','protocol','budget_registration',
              'ledger','phase_limits','stop_file'):
        require(cfg[k]==mc[k],'Original H execution prerequisite changed: '+k)
    for k in ('runtime_dir','native_runtime','uep_runtime','var_source','dino_source','source_driver_module',
              'receiver_module','numerical_reference','static_closure_module'):
        require(cfg[k]==visual[k],'Original source/receiver prerequisite changed: '+k)
    require(cfg['quality_driver_module']==str(Path(cfg['uep_runtime'])/'quality_driver.py')
            and cfg['codec_module']==str(Path(cfg['runtime_dir'])/'h64_source.py'),
            'Original source-codec/native module path differs')
    prior['metric'].u.assert_budget(prior['before'],prior['before'],pc['group'])
    timing=module(cfg['timing_core_module'],'_registered_h_online_components')
    scope=timing.design(read(cfg['protocol']),read(cfg['budget_registration']),pc['population'],pc['schedule'])
    graph=read(cfg['visual_source_closure']);closure=module(cfg['static_closure_module'],'_timing_source_closure')
    actual=closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
    require(graph['status']=='EXACT_SOURCE_CLOSURE_MATCH' and graph['source_bindings']==actual
            and closure.compare_bindings(actual,bound)['status']=='EXACT_SOURCE_CLOSURE_MATCH',
            'Complete current native source closure differs')
    for p,s in read(visual['visual_source_closure'])['source_bindings'].items():
        require(actual.get(p)==s,'Earlier executed native source changed: '+p)
    require(actual.get(cfg['quality_driver_module'])==sha(cfg['quality_driver_module']),
            'Actual original loader omitted from closure')
    out,hout=Path(cfg['out']),Path(cfg['H_out'])
    require(out.is_absolute() and hout in out.parents and cfg['stop_file']==str(hout/'STOP')
            and cfg['overall_deadline_unix']==DEADLINE and type(cfg['max_seconds']) is int
            and 0<cfg['max_seconds']<=14400,'Original deadline and bounded component job required')
    for old in (mc['out'],visual['out'],pc['group']['cfg']['out']):
        old=Path(old);require(out!=old and out not in old.parents and old not in out.parents,'Timing output overlaps evidence')
    require(not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),'STOP blocks timing')
    rx=module(cfg['receiver_module'],'_timing_original_receiver')
    source=module(cfg['source_driver_module'],'_timing_original_source')
    codec=module(cfg['codec_module'],'_timing_original_codec')
    adapter=module(visual['rx_adapter_module'],'_timing_original_development_adapter')
    ctx=dict(cfg=cfg,bound=bound,helper=helper,prior=prior,owner=pc['owner'],timing=timing,scope=scope,
        group=pc['group'],source_ids=pc['source_ids'],schedule=pc['schedule'],rx=rx,source=source,codec=codec,
        adapter=adapter,static_bindings=actual,before=prior['before'])
    # Validate every fixed case and exact result row before loading models. No
    # source image arrays or calibration qualities are read in this pass.
    metadata={};events={}
    for i in timing.FIXED:
        item=case_metadata(ctx,i);metadata[i]=item;events=merge(events,item['events'])
    ctx.update(case_metadata=metadata,events=events)
    return ctx


def load_registered(config_path):
    path=str(Path(config_path).absolute());cfg=read(path);reg=read(cfg['registration'])
    require(reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H'
            and reg['source_stage_scope']==SCOPE and reg['allowed_stage_ids']==['development'],
            'Independent timing-only execution registration required')
    bound=merge(reg['source_bindings'],reg['input_bindings'])
    for p in (path,cfg['visual_owner_config']):require(bound.get(p)==sha(p),'New timing configuration/owner unbound')
    ctx=inspect_inputs(cfg,bound)
    require(ctx['before']==reg['budget_before'] and cfg['phase_limits']==reg['phase_limits']
            and reg['timing_design']==ctx['scope'],'Registered finite timing design/budget differs')
    ctx['reg']=reg;return ctx


def case_metadata(ctx,index):
    group=ctx['group'];pc=ctx['prior']['context'];sid=ctx['source_ids'][index]
    base=Path(group['cfg']['out'])/f'worker_{index%2}';tp=base/'traces'/f'{index:04d}.json'
    require(group['done']['outputs'].get(str(tp))==sha(tp),'Fixed source actual trace unsealed')
    trace=read(tp);frames=ctx['adapter'].validate_frame_grid(trace,ctx['schedule'],group['core'])
    selected={f['development_slot']:f for f in frames if f['noise_seed']==ctx['timing'].NOISE}
    require(set(selected)==set(range(18)) and all(f['source_index']==index and f['source_id']==sid for f in selected.values()),
            'Fixed noise/source/slot mapping differs')
    rp=Path(pc['render_cfg']['out'])/'sources'/f'{index:04d}.json'
    done=pc['closed']['done'];require(done['outputs'].get(str(rp))==sha(rp),'Actual expected RX rows unsealed')
    rows=read(rp)
    require(len(rows)==54 and {(r['development_slot'],r['noise_seed']) for r in rows}==
            {(s,n) for s in range(18) for n in (6201,6202,6203)},'Actual RX result source grid incomplete')
    expected={r['development_slot']:r for r in rows if r['noise_seed']==ctx['timing'].NOISE}
    for slot,row in expected.items():
        require(row['source_index']==index and row['source_id']==sid and row['event_id']==selected[slot]['event_id']
                and row['source_status']==row['rx_summary']['source_status']
                and row['image_sha256']==row['rx_summary']['image_sha256']
                and row['receiver_view_sha256']==row['rx_summary']['receiver_view_sha256']
                and row['rx_summary']['source_decode_complete'] is True,'Actual expected RX identity differs')
    source=group['context']['source'];record=source['records'][index];sp=record['checkpoint']
    require(source['outputs'].get(sp)==record['sha256']==sha(sp),'Frozen source token parity evidence unsealed')
    scp=read(sp)
    require(scp['source_index']==index and scp['source_id']==sid and scp['encoder_tokens_verified'] is True,
            'Original Encoder/VQ parity receipt differs')
    events={}
    for frame in selected.values():
        kinds=['header']+(['body'] if frame['rx']['body'] is not None else [])
        require(frame['packet_event_ids']==[frame['event_id']+':'+k for k in kinds], 'Actual paid event mapping differs')
        for kind in kinds:events[frame['event_id']+':'+kind]=dict(kind=kind,result_sha256=ctx['timing'].digest(frame['rx'][kind]))
    return dict(frames=selected,expected=expected,tokens_sha256=scp['tokens_sha256'],
        events=events,bindings=bind((tp,rp,sp)))


def verify_rx_parity(value,expected,phy):
    actual=value['receiver_summary'];saved=expected['rx_summary']
    require(all(actual.get(k)==saved.get(k) for k in PARITY_FIELDS)
            and actual.get('canonical_details')==saved.get('canonical_details'),
            'Fresh actual RX/source parse differs from registered reconstruction')
    require(phy.array_sha(value['image'])==actual['image_sha256']==expected['image_sha256'],
            'Fresh actual RX RGB differs from registered reconstruction')
    return True


def measure_case(timing,components,pixels,entry,devctx,frame,expected,expected_tokens_sha,rx,phy,*,synchronize,guard):
    """Original one-warmup/three repeats; each parity check follows its timers."""
    tx_checks=[];rx_checks=[]
    def transmit_source():
        value=components['tx'](pixels,entry,devctx)
        require(hashlib.sha256(b'int64:680\0'+np.asarray(value['tokens'],dtype='<i8').tobytes()).hexdigest()==expected_tokens_sha,
                'Fresh Encoder/VQ differs from original full680-token receipt')
        timing.verify_tx_against_trace(value,frame,phy);tx_checks.append(value['fingerprint']);return value
    view=rx.receiver_view(frame)
    def reconstruct():
        value=components['rx'](copy.deepcopy(view));verify_rx_parity(value,expected,phy)
        rx_checks.append(value['fingerprint']);return value
    tx=timing.repetitions(transmit_source,synchronize=synchronize,guard=guard)
    received=timing.repetitions(reconstruct,synchronize=synchronize,guard=guard)
    require(len(tx_checks)==len(rx_checks)==4,'Every warmup/repetition must independently pass exact parity')
    return dict(development_slot=frame['development_slot'],source_index=frame['source_index'],source_id=frame['source_id'],
        noise_seed=frame['noise_seed'],snr_db=frame['snr_db'],role=frame['role'],candidate_id=frame['candidate_id'],
        arm=frame['arm'],TX=tx,RX=received,TX_summary=timing.summarize_repetitions(tx),
        RX_summary=timing.summarize_repetitions(received),TX_probability_summary=probability_summary(tx),
        RX_probability_summary=probability_summary(received),TX_exact_repetitions=4,RX_exact_repetitions=4,
        tx_profile_id=frame['tx']['profile_id'],tx_actual_m=frame['tx']['actual_m'],tx_K=frame['tx']['K'],
        tx_mode=frame['tx']['mode'],tx_fell_back=frame['tx']['fell_back'],tx_attempts=frame['tx']['attempts'],
        rx_source_status=expected['source_status'],gray=expected['gray'],
        original_image_sha256=expected['image_sha256'],original_receiver_view_sha256=expected['receiver_view_sha256'],
        new_packet_decodes=0,new_noise_draws=0,quality_sample_added=False)


def probability_summary(record):
    require(len(record['warmups'])==1 and len(record['measured'])==3,'Original repetition counts required')
    return {field:dict(samples=[r['probability'][field] for r in record['measured']],
            mean=statistics.mean(r['probability'][field] for r in record['measured']))
            for field in ('probability_seconds','cdf_seconds','provider_calls','advanced_scales')}


def summarize(rows,scope):
    require(len(rows)==288 and {(r['source_index'],r['development_slot']) for r in rows}==
            {(i,s) for i in scope['source_indices'] for s in range(18)} and all(r['noise_seed']==6201 for r in rows),
            'Fixed16 x eighteen timing coverage incomplete')
    components=[]
    for slot in range(18):
        group=[r for r in rows if r['development_slot']==slot]
        for side in ('TX','RX'):
            names=sorted({k for r in group for k in r[side+'_summary']})
            for name in names:
                # Each source contributes its mean of exactly three repeats.
                # An uncalled component costs zero for that source, not N/A.
                values=[r[side+'_summary'].get(name,{}).get('mean_seconds',0.) for r in group]
                components.append(dict(development_slot=slot,side=side,component=name,source_count=16,
                    repetitions_per_source=3,mean_seconds=statistics.mean(values),median_source_seconds=statistics.median(values),
                    source_means=values,warmups_excluded=True,inclusive=name in ('TX_source_total','RX_source_and_image_total')))
            for key,name in (('probability_seconds',side+'_VAR_probability'),('cdf_seconds',side+'_CDF_construction')):
                values=[r[side+'_probability_summary'][key]['mean'] for r in group]
                components.append(dict(development_slot=slot,side=side,component=name,source_count=16,
                    repetitions_per_source=3,mean_seconds=statistics.mean(values),median_source_seconds=statistics.median(values),
                    source_means=values,warmups_excluded=True,inclusive=False,
                    measured_by='Frozen IndependentProvider internal synchronized counters'))
    return dict(status='H_FIXED16_COMPONENT_SUMMARY',component_case_count=288,source_count=16,
        declaration_count=18,components=components,failed_case_count=sum(r['gray'] for r in rows),
        warmups_excluded=True,uncached=True,policy_selection=False,
        PHY_encode=None,exclusive_PHY_decode=None,end_to_end_latency=None,
        unmeasured_status='NOT_MEASURED',historical_PHY_can_be_added_to_GPU_totals=False,
        caveat='Instrumented synchronized uncached components; nested inclusive totals must not be added to their components.')


def gpu_admission(ctx,path):
    cfg=ctx['cfg'];a=ctx['owner'];oc=read(cfg['visual_owner_config']);a.validate_config(oc,ctx['reg'],sha(cfg['visual_owner_config']))
    require(oc['registration']==cfg['registration'] and oc['out']==cfg['H_out'] and len(oc['stages'])==1,
            'Independent timing owner required')
    stage=oc['stages'][0];require(stage['id']=='development' and stage['resource']=='gpu' and len(stage['jobs'])==1,
            'One exclusive timing GPU job required')
    job=stage['jobs'][0];argv=job['argv'];prior=ctx['prior']['closed']['owner']['stages'][0]['jobs'][0]
    require(argv[0]==prior['argv'][0] and argv[1:]==['-B',str(Path(__file__).absolute()),'--config',str(Path(path).absolute())]
            and job['id']=='h_online_components' and job['out']==cfg['out']
            and job['completion']==str(Path(cfg['out'])/'completion.json'),'Registered original UM timing command differs')
    base=Path(oc['owner_out']);ip=base/'owner_identity.json';identity=read(ip)
    require(int(identity['pid'])==os.getppid() and a.same_identity(identity,a.identity(os.getppid()))
            and identity['config_sha256']==sha(cfg['visual_owner_config'])
            and identity['registration_sha256']==sha(cfg['registration']) and not (base/'failure.json').exists(),
            'Live exclusive timing owner absent')
    lp=base/'stages/development/workers/h_online_components/launch.json';started=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-started<10 and a.same_identity(identity,a.identity(os.getppid())),'Worker launch not sealed');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid());gp=base/'stages/development/gpu_admission.json';gate=read(gp)
    require(a.same_identity(current,launch['identity']) and launch['argv']==argv and launch['resource']=='gpu'
            and launch['registration_sha256']==sha(cfg['registration']) and launch['threads']==oc['gpu_threads']==6
            and set(launch['affinity'])==set(oc['gpu_affinity'])==set(os.sched_getaffinity(0))
            and os.getpriority(os.PRIO_PROCESS,0)==15 and os.environ.get('CUDA_VISIBLE_DEVICES')==str(oc['gpu_device']),
            'Original isolated GPU/CPU resource identity differs')
    require(gate['status']=='GPU_IDLE_CONFIRMED' and gate['registration_sha256']==sha(cfg['registration']),
            'Original GPU idle admission absent; P/another live job cannot be overlapped')
    return dict(owner_identity=identity,worker_identity=current,bindings=bind((ip,lp,gp)))


def build_components(ctx):
    cfg=ctx['cfg']
    for p in (cfg['runtime_dir'],cfg['uep_runtime'],str(Path(cfg['root'])/'src')):sys.path.insert(0,p)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    quality=module(cfg['quality_driver_module'],'_timing_original_quality_driver')
    native=quality.build_native(Path(cfg['root']),cfg['native_runtime'],stop)
    numeric=read(cfg['numerical_reference'])
    require(native.loaded['identity']==numeric['frozen_identity']==ctx['prior']['context']['population']['identity']
            and native.flags==numeric['numerical_runtime'] and native.driver_bindings==ctx['static_bindings'],
            'Actual original qualified native identity/FP32/source closure differs')
    native.frozen()
    from var_comm import whole_entropy,entropy
    primitives=ctx['codec'].reference_primitives(whole_entropy,entropy)
    components=ctx['timing'].native_components(native,primitives,ctx['group']['catalogue'],ctx['source'],
        ctx['group']['core'],ctx['rx'],ctx['codec'])
    return native,quality,components


def claim_output(out,regsha):
    out=Path(out)
    if out.exists():require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior timing attempt preserved; no retry')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_ONLINE_COMPONENTS_ATTEMPT',registration_sha256=regsha,pid=os.getpid(),
        created_unix=time.time(),new_packet_decodes=0,new_noise_draws=0))


def run(path):
    ctx=load_registered(path);cfg=ctx['cfg'];out=Path(cfg['out']);regsha=sha(cfg['registration'])
    claim_output(out,regsha);began=time.monotonic();wall=time.time();native=None
    try:
        supervision=gpu_admission(ctx,path)
        require(shutil.disk_usage(out).free>=MAX_OUTPUT_BYTES+8*1024**3,'Timing output storage headroom unavailable')
        health=ctx['prior']['metric'].GPUHealth()
        def guard():
            elapsed=time.monotonic()-began
            require(elapsed<cfg['max_seconds'] and max(time.time(),wall+elapsed)<DEADLINE,'Original finite timing deadline')
            require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),
                    'Exclusive timing owner disappeared')
            health.check(force=True)
            if native is not None:quality.boundary(native)
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists(),
                    'STOP at complete timing source boundary')
            guard()
        boundary()
        history=ctx['timing'].historical_phy_windows(cfg['ledger'],ctx['events'])
        hp=out/'historical_phy_event_windows.json';save(hp,history);outputs=bind((hp,))
        native,quality,components=build_components(ctx);allrows=[];entries={r['development_slot']:r for r in ctx['schedule']}
        for index in ctx['timing'].FIXED:
            boundary();source,pixels,pre,inputs=ctx['adapter'].load_source(ctx['group'],index)
            item=ctx['case_metadata'][index];verify(item['bindings']);rows=[]
            for slot in range(18):
                guard()
                rows.append(measure_case(ctx['timing'],components,pixels,entries[slot],ctx['group']['context']['context'],
                    item['frames'][slot],item['expected'][slot],item['tokens_sha256'],ctx['rx'],ctx['group']['core'].phy,
                    synchronize=native.torch.cuda.synchronize,guard=guard))
            rp=out/'sources'/f'{index:04d}.json';cp=out/'source_checkpoints'/f'{index:04d}.json';save(rp,rows)
            save(cp,dict(status='H_ONLINE_COMPONENT_SOURCE_COMPLETE',registration_sha256=regsha,source_index=index,
                source_id=ctx['source_ids'][index],component_case_count=18,noise_seed=6201,
                warmup_repetitions=1,measured_repetitions=3,outputs=bind((rp,)),
                input_bindings=merge(inputs,item['bindings']),new_packet_decodes=0,new_noise_draws=0,
                timing_repetitions_are_quality_samples=False))
            outputs.update(bind((rp,cp)));allrows.extend(rows)
            require(sum(Path(p).stat().st_size for p in outputs)<=MAX_OUTPUT_BYTES,'Registered bounded timing output exceeded')
            temp=out/'status.tmp';temp.write_text(json.dumps(dict(status='RUNNING',completed_sources=len(allrows)//18,
                total_sources=16,component_cases=len(allrows),elapsed_seconds=time.monotonic()-began))+'\n',encoding='utf-8')
            os.replace(temp,out/'status.json');del source,pixels,rows
        boundary();native.frozen();summary=summarize(allrows,ctx['scope'])
        rp,sp=out/'component_cases.json',out/'component_summary.json';save(rp,allrows);save(sp,summary);outputs.update(bind((rp,sp)))
        flat=[]
        for row in allrows:
            identity={k:row[k] for k in ('source_index','source_id','development_slot','snr_db','noise_seed','role','arm','gray')}
            for side in ('TX','RX'):
                for phase in ('warmups','measured'):
                    for r in row[side][phase]:
                        for component,seconds in r['seconds'].items():
                            flat.append(dict(identity,side=side,phase=r['phase'],repetition=r['repetition'],
                                component=component,seconds=seconds,component_calls=r['component_calls'][component]))
                        for key,name in (('probability_seconds',side+'_VAR_probability'),('cdf_seconds',side+'_CDF_construction')):
                            flat.append(dict(identity,side=side,phase=r['phase'],repetition=r['repetition'],component=name,
                                seconds=r['probability'][key],component_calls=r['probability']['provider_calls']))
        csvpath=out/'component_repeats.csv'
        with csvpath.open('x',encoding='utf-8',newline='') as f:
            fields=list(flat[0]);writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(flat)
            f.flush();os.fsync(f.fileno())
        outputs.update(bind((csvpath,)));verify(ctx['bound']);verify(outputs)
        after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
        ctx['prior']['metric'].u.assert_budget(after,ctx['before'],ctx['group']);boundary();native.frozen()
        done=dict(status=DONE,registration_sha256=regsha,config_sha256=sha(path),source_count=16,component_case_count=288,
            H_policy_snr_points=18,noise_seed=6201,warmup_repetitions=1,measured_repetitions=3,batch_size=1,
            source_indices=list(ctx['timing'].FIXED),source_ids=[ctx['source_ids'][i] for i in ctx['timing'].FIXED],
            exact_TX_parity=True,exact_RX_parity=True,per_component_direction_repetitions=1152,
            outputs=outputs,input_bindings=ctx['reg']['input_bindings'],source_bindings=ctx['reg']['source_bindings'],
            visual_source_bindings=native.driver_bindings,numerical_runtime=native.flags,frozen_visual_identity=native.loaded['identity'],
            budget_before=ctx['before'],budget_after=after,budget_writes=0,new_packet_decodes=0,new_noise_draws=0,
            new_quality_samples=0,new_metric_calls=0,policy_selection=False,development_used=True,holdout_used=False,
            GPU_used=True,online_source_visual_components_measured=True,exclusive_PHY_latency_measured=False,
            PHY_encode_measured=False,end_to_end_latency_measured=False,historical_PHY_scope=history['status'],
            instrumented_uncached_cost_not_optimized_production_latency=True,MAIN_complete=False,
            overall_development_complete=False,H_full_delivery_claimed=False,supervision=supervision,
            own_owner_success_not_yet_certified=True,elapsed_seconds=time.monotonic()-began)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=regsha,
            traceback=traceback.format_exc(),new_packet_decodes=0,new_noise_draws=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);args=p.parse_args()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    result=run(args.config);print(json.dumps({k:result[k] for k in ('status','source_count','component_case_count','new_packet_decodes')}))


if __name__=='__main__':main()
