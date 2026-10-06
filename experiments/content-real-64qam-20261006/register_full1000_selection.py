"""Independent normal-close CPU registration and execution for H final policies.

No job launch, waiting queue, model, PHY decoder or recovery is implemented.
The finished visual run is audited against its historical source closure; new
CPU files never retroactively change that closure or overwrite old inputs.
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

DEADLINE=1791564605.9549868
STOP=False
PHASES={'qualification':2000,'coarse':49152,'refine':12288,'initial_true200':20000,
        'whole_calibration':48000,'partial_calibration':36000,'development':13200,'engineering_reserve':19360}
NAMES=('render_registration','render_owner_config','render_launch','render_config','selector_module','interpretation','prepared_qualification')
EXPECTED=dict(source_count=1000,frame_count=42000,phase_frame_counts={'whole_calibration':24000,'partial_calibration':18000},
    images_scored=True,source_decode_complete=True,arithmetic_source_decode_complete=True,new_packet_decodes=0,
    policy_selection=False,development_used=False,holdout_used=False)
DONE='H_FULL1000_CALIBRATION_SELECTION_COMPLETE'


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(paths):return {str(p):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():require(Path(p).is_absolute() and sha(p)==s,'Changed bound input/source: '+p)
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():require(p not in result or result[p]==s,'Conflicting old/new binding: '+p);result[p]=s
    return result
def pin(row):
    p=Path(row['path']);require(p.is_absolute() and sha(p)==row['sha256'],'Pinned input changed: '+str(p));return p
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def stop(*_):
    global STOP
    STOP=True


def frozen_budget(current,reg,done,owner_done):
    before=reg['budget_before']
    require(current==before==done['budget_before']==done['budget_after']==owner_done['budget'],
        'Registered, visual-before/after, closed owner and current ledger differ')
    require(current['created'] is True and current['unresolved']==0 and current['failed']==0
        and current['charged']==sum(current['phase_charged'].values()) and set(current['phase_charged'])==set(PHASES)
        and all(0<=current['phase_charged'][p]<=cap for p,cap in PHASES.items())
        and current['phase_charged']['development']==0 and current['development_remaining']==13200,
        'Packet ledger not quiescent or development reserve changed')
    # Header CRC rejection may prevent some body calls. Do not invent the total.
    require(24000<=current['phase_charged']['whole_calibration']<=48000
        and 18000<=current['phase_charged']['partial_calibration']<=36000,'Missing original full CPU populations')
    return before


def historical_visual_closure(done,graph,reg):
    require(graph['status']=='EXACT_SOURCE_CLOSURE_MATCH' and graph['source_bindings']==done['visual_source_bindings'],
        'Completed native runtime closure differs from historical registered closure')
    bound=merge(reg['input_bindings'],reg['source_bindings'])
    require(all(bound.get(p)==s for p,s in graph['source_bindings'].items()),'Historical visual dependency was never bound')
    verify(graph['source_bindings'])
    # Deliberately do not enumerate newly prepared Python files after GPU exit.
    return dict(status='COMPLETED_RUNTIME_EQUALS_HISTORICAL_REGISTERED_CLOSURE',
        source_bindings=graph['source_bindings'],new_cpu_sources_added_to_old_run=False,current_source_graph_reenumerated=False)


def normal_render_closed(a,w,paths,state_reader):
    cfg,reg,oc,launch=(read(paths[n]) for n in ('render_config','render_registration','render_owner_config','render_launch'))
    a.validate_config(oc,reg,sha(paths['render_owner_config']))
    require(cfg['registration']==str(paths['render_registration']) and cfg['visual_owner_config']==str(paths['render_owner_config'])
        and oc['registration']==cfg['registration'] and reg['allowed_stage_ids']==['render']
        and reg['source_stage_scope']=='FULL1000_ACTUAL_RX_IMAGES_ONLY' and len(oc['stages'])==1,'Wrong preceding visual scope')
    stage=oc['stages'][0];require(stage['id']=='render' and stage['resource']=='gpu' and len(stage['jobs'])==1,'Exactly one original visual job required')
    job=stage['jobs'][0];require(job['id']=='full1000_render' and job['argv'][1:]==['-B',cfg['visual_driver_module'],'--config',str(paths['render_config'])]
        and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json')
        and job['accepted_statuses']==['H_FULL1000_RX_COMPLETE'] and all(job['receipt_expect'].get(k)==v for k,v in EXPECTED.items()),
        'Frozen owner must call receipt with the full1000/42000 exact expectations')
    ident=launch['identity'];base=Path(oc['owner_out']);ip=base/'owner_identity.json';oid=read(ip)
    require(launch['argv']==ident['argv'] and launch['argv'][1:]==['-B',cfg['owner_module'],'--config',str(paths['render_owner_config'])]
        and launch['registration_sha256']==sha(paths['render_registration']) and launch['owner_config_sha256']==sha(paths['render_owner_config'])
        and a.same_identity(oid,ident) and oid['registration_sha256']==sha(paths['render_registration'])
        and oid['config_sha256']==sha(paths['render_owner_config']),'Original visual owner identity differs')
    for p in (base/'failure.json',base/'registration_failure.json',base/'STOP',Path(cfg['out'])/'failure.json',Path(cfg['out'])/'STOP',Path(cfg['stop_file'])):
        require(not p.exists(),'Visual failure/STOP blocks normal selection: '+str(p))
    require(w.exited(ident,state_reader),'Visual owner still live or unreaped')
    # This calls the original a.receipt with the checked EXPECTED fields and
    # hashes all scientific outputs. It also proves exit0 and closed-log SHA.
    closure=w.verify_batch(a,str(paths['render_owner_config']),str(paths['render_registration']),ident['uid'],state_reader)
    require(len(closure['child_identities'])==1 and w.exited(closure['child_identities'][0],state_reader),'Visual child still live or unproven')
    cp=Path(job['completion']);done=read(cp);od=read(base/'completion.json')
    require(closure['bindings'].get(str(cp))==sha(cp) and done['status']=='H_FULL1000_RX_COMPLETE'
        and done['registration_sha256']==sha(paths['render_registration']) and done['config_sha256']==sha(paths['render_config'])
        and all(done.get(k)==v for k,v in EXPECTED.items()),'Actual closed render completion differs')
    require(done['source_bindings']==reg['source_bindings'] and done['input_bindings']==reg['input_bindings'],
        'Finished visual process changed registered inputs or sources')
    for key in ('predecessor_closure_bindings',):verify(done[key])
    supervision=done['supervision'];verify(supervision['bindings'])
    require(a.same_identity(supervision['owner_identity'],ident)
        and a.same_identity(supervision['worker_identity'],closure['child_identities'][0]),'Visual final supervision differs from actual closed processes')
    graphp=base/'visual_source_closure.json'
    require(reg['input_bindings'].get(str(graphp))==sha(graphp),'Original static visual graph unbound')
    graph=historical_visual_closure(done,read(graphp),reg)
    numerical=read(cfg['numerical_reference']);cal=read(cfg['calibration_registration'])
    require(done['numerical_runtime']==numerical['numerical_runtime'] and done['frozen_visual_identity']==cal['identity'],
        'Actual completed visual numeric/model identity differs')
    closure['bindings']=merge(closure['bindings'],bind((ip,graphp,*[paths[k] for k in ('render_registration','render_owner_config','render_launch','render_config')])),supervision['bindings'])
    return dict(cfg=cfg,reg=reg,old=oc,done=done,owner_done=od,closure=closure,graph=graph,completion=cp)


def source_grid_audit(cfg,done,rows):
    ids=done['source_ids'];require(len(ids)==len(set(ids))==1000 and len(rows)==42000,'Wrong source/frame population')
    groups={i:[] for i in range(1000)}
    for row in rows:
        i=row['source_index'];require(type(i) is int and i in groups and row['source_id']==ids[i],'Source identity differs')
        groups[i].append(row)
    base=Path(cfg['out']);inputs={}
    for i,sid in enumerate(ids):
        cp,values,image=(base/name/f'{i:04d}.{suffix}' for name,suffix in (('source_checkpoints','json'),('sources','json'),('images','npz')))
        require(done['outputs'].get(str(cp))==sha(cp) and done['outputs'].get(str(values))==sha(values)
            and str(image) in done['outputs'],'Actual source archive or checkpoint is unsealed')
        record=read(cp)
        require(record['status']=='H_FULL1000_RX_SOURCE_COMPLETE' and record['registration_sha256']==done['registration_sha256']
            and record['source_index']==i and record['source_id']==sid and record['frame_count']==len(groups[i])==42
            and record['phase_frame_counts']=={'whole_calibration':24,'partial_calibration':18}
            and record['images_scored'] is True and record['source_decode_complete'] is True
            and record['new_packet_decodes']==0 and record['policy_selection'] is False,'Actual source completion differs')
        require(record['outputs']=={str(values):done['outputs'][str(values)],str(image):done['outputs'][str(image)]}
            and read(values)==groups[i] and all(r['image_archive']==str(image) for r in groups[i]),'Per-source and aggregate metrics/archives differ')
        verify(record['input_bindings']);inputs=merge(inputs,record['input_bindings'])
    return dict(status='ALL_1000_SOURCE_RECEIPTS_AND_42000_ROWS_MATCH',source_count=1000,frame_count=42000,
        source_input_bindings=inputs,RGB_arrays_previously_SHA_verified_by_owner_receipt=True,RGB_arrays_redecoded=False)


def predecessor(r):
    paths={k:pin(r[k]) for k in NAMES};cfg=read(paths['render_config']);reg=read(paths['render_registration'])
    bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    for k in ('owner_module','wait_module'):
        require(reg['source_bindings'].get(cfg[k])==sha(cfg[k]),'Original closure verifier unbound')
    for k in ('render_config','render_owner_config','interpretation'):
        require(bound.get(str(paths[k]))==sha(paths[k]),'Original visual input unbound: '+k)
    a=module(cfg['owner_module'],'full_selection_original_owner');w=module(cfg['wait_module'],'full_selection_original_wait')
    ctx=normal_render_closed(a,w,paths,a.raw_process_state)
    current=a.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=True)
    require(cfg['phase_limits']==reg['phase_limits']==PHASES,'Original H quota map changed')
    before=frozen_budget(current,reg,ctx['done'],ctx['owner_done'])
    metrics=Path(cfg['out'])/'frame_metrics.json';require(ctx['done']['outputs'].get(str(metrics))==sha(metrics),'Actual full frame table unsealed')
    rows=read(metrics);audit=source_grid_audit(cfg,ctx['done'],rows)
    physical=read(cfg['cpu_batches']['whole_calibration']['config'])
    artifact_paths={k:physical[old] for k,old in (('shortlist','shortlist'),('selected_whole','selected'),('partial_reference','partial_reference'),('catalogue','catalogue'),('protocol','protocol'))}
    artifact_paths.update(calibration_registration=cfg['calibration_registration'],interpretation=str(paths['interpretation']))
    for p in artifact_paths.values():require(ctx['done']['input_bindings'].get(p)==sha(p),'Selector input not frozen in actual completed render')
    require(a.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),PHASES,quiescent=True)==before,'Budget changed during normal close audit')
    return dict(ctx,paths=paths,owner=a,wait=w,metrics=metrics,audit=audit,before=before,artifact_paths=artifact_paths)


def make_configs(r,ctx,execution):
    rp=execution/'execution_registration.json';cp=execution/'selection_config.json';op=execution/'owner_config.json'
    cfg=dict(schema='H_FULL1000_SELECTION_CONFIG_V1',registration=str(rp),owner_config=str(op),request_path=r['request_path'],
        out=r['selection_out'],root=ctx['cfg']['root'],H_out=ctx['cfg']['H_out'],stop_file=ctx['cfg']['stop_file'],
        max_seconds=r['max_seconds'],overall_deadline_unix=DEADLINE)
    oc=copy.deepcopy(ctx['old']);oc.update(owner_out=str(execution),registration=str(rp))
    oc['stages']=[dict(id='freeze',resource='cpu',requires=[],max_seconds=r['max_seconds'],jobs=[dict(id='full1000_selection',
        argv=[r['python'],'-B',str(Path(__file__).resolve()),'--stage','run','--config',str(cp)],cwd=cfg['root'],out=cfg['out'],
        completion=str(Path(cfg['out'])/'completion.json'),accepted_statuses=[DONE],receipt_expect=dict(source_count=1000,
            measured_frames=42000,whole_count=8,selected_partial_count=2,whole_policy_reselected=False,
            new_packet_decodes=0,new_visual_inference=0,GPU_used=False,development_used=False,holdout_used=False))])]
    return cfg,oc


def register(request_path):
    request_path=Path(request_path).resolve();r=read(request_path)
    require(r['schema']=='H_FULL1000_SELECTION_REGISTRATION_REQUEST_V1' and sys.platform.startswith('linux'),
        'Actual normal-close selection registration requires Linux request')
    require(time.time()<DEADLINE,'Original H deadline exceeded')
    ctx=predecessor(r);paths=ctx['paths'];q=read(paths['prepared_qualification']);verify(q['source_bindings'])
    require(q['status']=='H_FULL1000_SELECTION_CPU_QUALIFICATION_PASS' and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['results'] and all(x['exit_code']==0 for x in q['results']),'CPU-only selector qualification required')
    for p in (Path(__file__).resolve(),paths['selector_module']):require(q['source_bindings'].get(str(p))==sha(p),'Unqualified new selection source')
    e,out,h=Path(r['execution_dir']),Path(r['selection_out']),Path(ctx['cfg']['H_out'])
    for p in (e,out):require(p.is_absolute() and h in p.parents and not p.exists(),'Fresh independent selection path required')
    require(e!=out and e not in out.parents and out not in e.parents,'Selection outputs overlap execution evidence')
    require(Path(r['python']).is_absolute() and Path(r['python']).is_file() and type(r['max_seconds']) is int
        and 0<r['max_seconds']<=3600,'Finite original CPU runtime required')
    sources=merge(ctx['reg']['source_bindings'],q['source_bindings'],bind((Path(__file__).resolve(),paths['selector_module'])))
    inputs=merge(ctx['reg']['input_bindings'],ctx['closure']['bindings'],ctx['done']['outputs'],ctx['audit']['source_input_bindings'],
        bind((request_path,*paths.values())))
    # merge rejects changed old paths; newly qualified CPU sources are separate.
    verify(sources);verify(inputs);r=dict(r,request_path=str(request_path));cfg,oc=make_configs(r,ctx,e)
    e.mkdir()
    try:
        gate=e/'predecessor_closure.json';save(gate,dict(status='NORMAL_FULL1000_GPU_OWNER_AND_WORKER_CLOSED',
            closure=ctx['closure'],historical_visual_closure=ctx['graph'],source_audit=ctx['audit'],budget=ctx['before'],recovery_used=False))
        cp,op,rp=e/'selection_config.json',e/'owner_config.json',e/'execution_registration.json'
        save(cp,cfg);save(op,oc);inputs=merge(inputs,bind((gate,cp,op)))
        reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',revision=e.name,source_stage_scope='FULL1000_CALIBRATION_SELECTION_ONLY',
            allowed_stage_ids=['freeze'],owner_config_sha256=sha(op),phase_limits=PHASES,source_bindings=sources,input_bindings=inputs,
            budget_registration_sha256=sha(ctx['cfg']['budget_registration']),budget_before=ctx['before'],
            render_completion_sha256=sha(ctx['completion']),selector_sha256=sha(paths['selector_module']),
            source_count=1000,measured_frames=42000,whole_count=8,selected_partial_count=2,whole_policy_reselected=False,
            GPU_jobs=0,new_packet_decodes=0,new_visual_inference=0,development_used=False,holdout_started=False,C_started=False,
            H_full_delivery_claimed=False,scientific_protocol_modified=False,future_stage_automatic=False,
            historical_source_closure_verified=True,independent_closeout_accepted=False,created_unix=time.time())
        ctx['owner'].validate_config(oc,reg,sha(op));verify(sources);verify(inputs)
        require(not Path(cfg['stop_file']).exists() and time.time()<DEADLINE,'STOP/deadline changed before seal')
        require(ctx['owner'].budget_snapshot(ctx['cfg']['ledger'],sha(ctx['cfg']['budget_registration']),PHASES,quiescent=True)==ctx['before'],'Budget changed before seal')
        save(rp,reg)
        done=dict(status='H_FULL1000_SELECTION_REGISTERED_NOT_LAUNCHED',registration_sha256=sha(rp),owner_config_sha256=sha(op),
            outputs=bind((gate,cp,op,rp)),workers_started=False,GPU_used=False,new_packet_decodes=0,future_stage_automatic=False,
            required_launch_condition='Check this completion outputs, no registration_failure, all predecessor processes normally closed, STOP absent and exact ledger unchanged.')
        save(e/'registration_completion.json',done);return done
    except BaseException:
        save(e/'registration_failure.json',dict(status='FAILED_PRESERVE_NO_LAUNCH',traceback=traceback.format_exc()));raise


def live_cpu_owner(ctx,cfg,reg,config_path):
    a=ctx['owner'];oc=read(cfg['owner_config']);a.validate_config(oc,reg,sha(cfg['owner_config']))
    require(len(oc['stages'])==1 and oc['stages'][0]['id']=='freeze' and oc['stages'][0]['resource']=='cpu'
        and len(oc['stages'][0]['jobs'])==1,'One CPU selection job only')
    job=oc['stages'][0]['jobs'][0];argv=[job['argv'][0],'-B',str(Path(__file__).resolve()),'--stage','run','--config',str(config_path)]
    require(job['id']=='full1000_selection' and job['argv']==argv and job['out']==cfg['out'],'Current selector command differs')
    base=Path(oc['owner_out']);ip=base/'owner_identity.json';ident=read(ip)
    require(a.same_identity(ident,a.identity(os.getppid())) and ident['registration_sha256']==sha(cfg['registration'])
        and ident['config_sha256']==sha(cfg['owner_config']) and not (base/'failure.json').exists(),'Current parent is not the registered CPU owner')
    lp=base/'stages/freeze/workers/full1000_selection/launch.json';start=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-start<10 and a.same_identity(ident,a.identity(os.getppid())),'CPU owner did not seal current worker');time.sleep(.1)
    launch=read(lp)
    require(a.same_identity(launch['identity'],a.identity(os.getpid())) and launch['argv']==argv and launch['resource']=='cpu'
        and launch['registration_sha256']==sha(cfg['registration']) and launch['threads']==2
        and set(launch['affinity'])==set(oc['cpu_affinities'][0])==set(os.sched_getaffinity(0))
        and os.getpriority(os.PRIO_PROCESS,0)==15,'CPU identity/resources differ')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only environment differs')
    return bind((ip,lp))


def claim_output(out,regsha):
    out=Path(out);require(not out.exists() or (out.is_dir() and not out.is_symlink() and not any(out.iterdir())),
        'Prior selection attempt preserved; no blind retry')
    out.mkdir(parents=True,exist_ok=True);save(out/'attempt.json',dict(status='STARTED',registration_sha256=regsha,new_packet_decodes=0,GPU_used=False))


def run(config_path):
    config_path=Path(config_path).resolve();cfg=read(config_path);reg=read(cfg['registration'])
    require(cfg['schema']=='H_FULL1000_SELECTION_CONFIG_V1' and reg['status']=='H_EXECUTION_REVISION_REGISTERED'
        and reg['source_stage_scope']=='FULL1000_CALIBRATION_SELECTION_ONLY' and reg['allowed_stage_ids']==['freeze'],'Wrong standalone CPU selection registration')
    verify(reg['source_bindings']);verify(reg['input_bindings']);bound=merge(reg['source_bindings'],reg['input_bindings'])
    for p in (config_path,Path(__file__).resolve(),Path(cfg['request_path']),Path(cfg['owner_config'])):
        require(bound.get(str(p))==sha(p),'Selection runtime unbound')
    ctx=predecessor(read(cfg['request_path']));require(ctx['before']==reg['budget_before'],'Packet budget changed since CPU registration')
    supervision=live_cpu_owner(ctx,cfg,reg,config_path);out=Path(cfg['out']);claim_output(out,sha(cfg['registration']));start=time.monotonic()
    try:
        def boundary():
            require(not STOP and not Path(cfg['stop_file']).exists() and not (out/'STOP').exists()
                and time.time()<DEADLINE and time.monotonic()-start<cfg['max_seconds'],'Selection STOP/deadline')
        boundary()
        for p in (ctx['cfg']['runtime_dir'],str(ctx['paths']['selector_module'].parent)):sys.path.insert(0,p)
        selector=module(ctx['paths']['selector_module'],'registered_full1000_pure_selector')
        artifacts={k:{'path':p,'bytes':Path(p).read_bytes()} for k,p in ctx['artifact_paths'].items()}
        result=selector.finalize_calibration(artifacts,ctx['metrics'].read_bytes(),frame_metrics_path=str(ctx['metrics']),render_completion=ctx['done'])
        require(result['status']=='H_FULL1000_CALIBRATION_POLICIES_FINALIZED' and result['measured_frames']==42000
            and len(result['whole_candidates'])==8 and len(result['selected_partial'])==2 and result['whole_policy_reselected'] is False,'Selection scope changed')
        boundary();outputs={}
        for name,value in (('finalized_policies.json',result),('selected_partial.json',result['selected_partial']),
            ('whole_policies_preserved.json',result['whole_candidates']),('fixed_source_controls.json',result['fixed_source_controls']),
            ('development_handoff.json',result['development_plan'])):
            p=out/name;save(p,value);outputs[str(p)]=sha(p)
        csvp=out/'calibration_summary.csv';fields=['full_slot','phase','candidate_id','arm','snr_db','source_count','frame_count',
            'mean_per_source_psnr_db','mean_overall_mse','psnr_of_mean_mse_db','whole_policy_reselected','calibration_not_independent_test']
        with csvp.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in result['all_candidate_summaries'])
        outputs[str(csvp)]=sha(csvp);boundary();live_cpu_owner(ctx,cfg,reg,config_path)
        verify(reg['source_bindings']);verify(reg['input_bindings']);verify(outputs)
        after=ctx['owner'].budget_snapshot(ctx['cfg']['ledger'],sha(ctx['cfg']['budget_registration']),PHASES,quiescent=True)
        require(after==ctx['before'],'Selection changed packet ledger')
        done=dict(status=DONE,registration_sha256=sha(cfg['registration']),config_sha256=sha(config_path),source_count=1000,measured_frames=42000,
            whole_count=8,selected_partial_count=2,whole_policy_reselected=False,outputs=outputs,source_bindings=reg['source_bindings'],
            input_bindings=reg['input_bindings'],predecessor_closure_bindings=ctx['closure']['bindings'],live_launch_bindings=supervision,
            budget_before=ctx['before'],budget_after=after,new_packet_decodes=0,new_visual_inference=0,GPU_used=False,
            development_used=False,holdout_used=False,H_full_delivery_claimed=False,main_full_calibration_complete=False,
            future_stage_automatic=False,elapsed_seconds=time.monotonic()-start)
        save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',registration_sha256=sha(cfg['registration']),
            traceback=traceback.format_exc(),new_packet_decodes=0,GPU_used=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('register','run'),required=True);p.add_argument('--config',required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    result=register(a.config) if a.stage=='register' else run(a.config)
    print(json.dumps({k:result[k] for k in ('status','registration_sha256')}))


if __name__=='__main__':main()
