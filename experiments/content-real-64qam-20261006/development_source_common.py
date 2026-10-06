"""Bound, source-only preparation of the original development100 population.

No wire decoder or neural module is imported by the admission functions.
Selection must have closed normally before either preparation stage.
"""
from pathlib import Path
import os
import sys
import time
import h_development_driver as d

read,sha,verify,bind,merge,module,require=d.read,d.sha,d.verify,d.bind,d.merged,d.module,d.require
COUNT=100
DEADLINE=d.DEADLINE
ASSETS_DONE='H_DEVELOPMENT100_CPU_ASSETS_READY_SOURCE_ENCODING_INCOMPLETE'
SOURCE_DONE='H_DEVELOPMENT100_SOURCE_CODEC_COMPLETE'
SCOPES={'assets':'DEVELOPMENT100_CPU_ASSETS_ONLY','source':'DEVELOPMENT100_SOURCE_CODEC_ONLY'}
BASE_REQUIRED=('protocol','development_registration','calibration_registration','selection_completion',
    'source_qualification','owner_config','owner_module','wait_module','budget_registration',
    'common_module','population_module','assets_module','assets_validation_module','full_source_module')
GPU_REQUIRED=('asset_completion','asset_manifest','assets_registration','static_closure_module','numerical_reference',
    'source_driver_module')


def save(path,value):
    import json
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def claim(out,regsha,kind):
    out=Path(out)
    if out.exists():require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior preparation evidence forbids retry')
    else:out.mkdir(parents=True)
    save(out/'attempt.json',dict(status='H_DEVELOPMENT100_PREPARATION_ATTEMPT',kind=kind,
        registration_sha256=regsha,pid=os.getpid(),new_packet_decodes=0))


def population(pop,calibration):
    require(pop['stage']==pop['calibration_or_development']=='m1_development','Original development registration required')
    ids=pop['source_ids'];pp=pop['preprocessing_ids'];rows=pop['data_bindings']
    require(len(ids)==len(set(ids))==len(pp)==100 and isinstance(rows,list) and len(rows)==100,'Original ordered100 list required')
    require(calibration['stage']==calibration['calibration_or_development']=='m1_calibration'
        and len(calibration['source_ids'])==len(set(calibration['source_ids']))==1000
        and not set(ids)&set(calibration['source_ids']),'Development/calibration population overlap')
    require(pop['identity']==calibration['identity'],'Original visual identity differs across populations')
    for i,r in enumerate(rows):
        require(set(r)=={'index','rgb_sha256','source_npz_sha256'} and r['index']==i
            and r['rgb_sha256']==pp[i] and all(isinstance(r[k],str) and len(r[k])==64 for k in ('rgb_sha256','source_npz_sha256')),
            'Development list/preprocessing identity changed')
    return list(ids)


def validate_manifest(manifest,done,pop):
    require(done['status']==manifest['status']==ASSETS_DONE and done['source_count']==manifest['source_count']==100
        and manifest['source_ids']==pop['source_ids'] and manifest['preprocessing_ids']==pop['preprocessing_ids']
        and manifest['visual_identity']==pop['identity'] and manifest['original_data_bindings']==pop['data_bindings'],
        'Original development asset identity/order differs')
    for v in (done,manifest):
        require(v['source_codec_complete'] is False and v['pending_source_encoding_count']==100
            and v['development_used'] is True and v['holdout_used'] is False and v['new_packet_decodes']==0,
            'CPU assets falsely claim completed source encoding')
    records=manifest['records'];require(len(records)==100,'Incomplete original100 assets')
    for i,r in enumerate(records):
        require(r['source_index']==i and r['source_id']==pop['source_ids'][i] and r['preprocessing_id']==pop['preprocessing_ids'][i]
            and done['outputs'].get(r['checkpoint'])==r['checkpoint_sha256'],'Asset record is not sealed in original order')
    return records


def budget_unchanged(before,reg):
    require(before.get('created') is True and before.get('unresolved')==before.get('failed')==0
        and before==reg['budget_before'] and before['phase_charged']['development']==0
        and before['development_remaining']==13200,'Source preparation budget changed or development already consumed')


def load_registered(config_path,kind,entry_path):
    p=Path(config_path).resolve();cfg=read(p);reg=read(cfg['registration'])
    require(kind in SCOPES and cfg['schema']=='H_DEVELOPMENT100_PREPARATION_CONFIG_V1' and cfg['kind']==kind
        and reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H'
        and reg['allowed_stage_ids']==['source'] and reg['source_stage_scope']==SCOPES[kind], 'Independent source-preparation registration required')
    verify(reg['source_bindings']);verify(reg['input_bindings']);bound=merge(reg['source_bindings'],reg['input_bindings'])
    for key in BASE_REQUIRED+(GPU_REQUIRED if kind=='source' else ('population_manifest','token_archive','original_pixel_root')):
        if key=='original_pixel_root':continue
        require(bound.get(cfg[key])==sha(cfg[key]),'Unbound preparation input: '+key)
    for f in (str(p),str(Path(entry_path).resolve()),str(Path(__file__).resolve()),str(Path(d.__file__).resolve())):
        require(bound.get(f)==sha(f),'Unbound actual preparation entry/helper: '+f)
    require(cfg['phase_limits']==reg['phase_limits']==d.PHASES and cfg['overall_deadline_unix']==DEADLINE
        and 0<cfg['max_seconds']<=86400 and cfg['stop_file']==str(Path(cfg['H_out'])/'STOP')
        and Path(cfg['H_out']) in Path(cfg['out']).parents,'Preparation output/resource scope differs')
    for f in (Path(cfg['stop_file']),Path(cfg['out'])/'STOP',Path(cfg['out'])/'failure.json'):
        require(not f.exists(),'STOP/failed preparation requires separate diagnosis: '+str(f))
    protocol=read(cfg['protocol'])
    require(protocol['schema']=='H_CODEC_PROTOCOL_V1' and protocol['status']=='FROZEN_BEFORE_DATA'
        and protocol['source']['whole_targets']==[6,7,8,9] and protocol['source']['sizes']==[1,2,3,4,5,6,8,10,13,16]
        and protocol['source']['ordering']=='public raster' and protocol['source']['new_training']==0,'Frozen H source protocol changed')
    pop,cal=read(cfg['development_registration']),read(cfg['calibration_registration']);ids=population(pop,cal)
    qual=read(cfg['source_qualification']);verify(qual['source_bindings'])
    require(qual['status']=='H_DEVELOPMENT100_SOURCE_QUALIFICATION_PASS' and qual['new_packet_decodes']==0
        and qual['results'] and all(r['exit_code']==0 for r in qual['results'])
        and qual['source_bindings'].get(str(Path(entry_path).resolve()))==sha(entry_path), 'Actual source entry not CPU qualified')
    api=module(cfg['owner_module'],'dev_source_owner');wait=module(cfg['wait_module'],'dev_source_wait')
    require(set(cfg['prerequisites'])==({'selection'} if kind=='assets' else {'selection','assets'}),'Exact predecessor batches required')
    closures={}
    for name,spec in cfg['prerequisites'].items():
        for f in spec.values():require(bound.get(f)==sha(f),'Unbound predecessor '+name)
        expected=(dict(status='H_FULL1000_CALIBRATION_SELECTION_COMPLETE',source_count=1000,measured_frames=42000,
            whole_count=8,selected_partial_count=2,whole_policy_reselected=False,development_used=False,new_packet_decodes=0)
            if name=='selection' else dict(status=ASSETS_DONE,source_count=100,new_packet_decodes=0))
        closure=d.closed_batch(api,wait,spec,cfg['owner_module'],expected,api.raw_process_state)
        require(closure['owner']['root']==cfg['root'] and closure['owner']['out']==cfg['H_out']
            and closure['owner']['budget_path']==cfg['ledger'],'Predecessor workspace/budget differs')
        closures[name]=closure
    require(cfg['prerequisites']['selection']['completion']==cfg['selection_completion'],'Selection completion path differs')
    # The normal full1000 selector is itself gated by the normal previous visual
    # batch; its source/input closure, outputs, processes and logs are rechecked.
    budget=read(cfg['budget_registration'])
    require(budget['status']=='FROZEN' and budget['branch']=='H' and budget['total_cap']==200000
        and budget['phase_limits']==cfg['phase_limits'],'Original independent H budget required')
    before=api.budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    budget_unchanged(before,reg)
    for closure in closures.values():require(closure['owner_done']['budget']==before,'Predecessor budget no longer current')
    ctx=dict(cfg=cfg,reg=reg,bound=bound,pop=pop,cal=pop,ids=ids,owner=api,wait=wait,closures=closures,before=before,
        assets_api=module(cfg['assets_validation_module'],'dev_frozen_asset_validation'),
        codec_runner=module(cfg['full_source_module'],'dev_frozen_codec_runner'))
    if kind=='source':
        done=read(cfg['asset_completion']);manifest=read(cfg['asset_manifest']);ar=read(cfg['assets_registration'])
        require(cfg['prerequisites']['assets']['completion']==cfg['asset_completion'] and cfg['prerequisites']['assets']['registration']==cfg['assets_registration']
            and done['registration_sha256']==sha(cfg['assets_registration']) and done['outputs'].get(cfg['asset_manifest'])==sha(cfg['asset_manifest'])
            and ar['source_stage_scope']==SCOPES['assets'],'CPU asset lineage differs')
        for mapping in (done['outputs'],done['source_bindings'],done['input_bindings']):verify(mapping)
        for key in ('development_registration','calibration_registration','source_qualification'):
            require(done['input_bindings'].get(cfg[key])==sha(cfg[key]),'Asset preparation did not bind '+key)
        ctx.update(records=validate_manifest(manifest,done,pop),assets_done=done,manifest=manifest)
        for directory,names in ((cfg['runtime_dir'],('h64_catalog.py','h64_source.py')),
            (cfg['uep_runtime'],('quality_driver.py','source_quality.py')),
            (str(Path(cfg['root'])/'src/var_comm'),('whole_entropy.py','entropy.py'))):
            for name in names:
                f=str(Path(directory)/name);require(bound.get(f)==sha(f),'Unbound frozen codec/model helper: '+f)
        native_files=list(Path(cfg['native_runtime']).glob('m1_*.py'))
        require({'m1_common.py','m1_native.py','m1_phy.py','m1_performance.py'}<={f.name for f in native_files},'Native runtime incomplete')
        for f in native_files:require(bound.get(str(f))==sha(f),'Unbound native helper')
        closure=module(cfg['static_closure_module'],'development_static_source_closure')
        static=closure.collect_bindings(cfg['root'],cfg['native_runtime'],cfg['var_source'],cfg['dino_source'],cfg['uep_runtime'])
        require(closure.compare_bindings(static,bound)['status']=='EXACT_SOURCE_CLOSURE_MATCH','New visual source closure unbound')
        ref=read(cfg['numerical_reference']);require(ref['status']=='REAL_NATIVE_QUALIFICATION_PASS'
            and cfg['numerical_reference_field']==['numerical_runtime'],'Original numerical qualification required')
        flags=ref['numerical_runtime'];require(flags['threads']==6 and flags['interop_threads']==2 and flags['deterministic'] is True
            and flags['matmul_tf32'] is False and flags['cudnn_tf32'] is False,'Original FP32 qualification differs')
        ctx.update(static_bindings=static,expected_flags=flags)
    return ctx


def live_owner(ctx,config_path,entry_path,kind):
    cfg=ctx['cfg'];a=ctx['owner'];oc=read(cfg['owner_config']);a.validate_config(oc,ctx['reg'],sha(cfg['owner_config']))
    resource='cpu' if kind=='assets' else 'gpu'
    require(oc['registration']==cfg['registration'] and oc['out']==cfg['H_out'] and len(oc['stages'])==1,'Preparation owner differs')
    stage=oc['stages'][0];require(stage['id']=='source' and stage['resource']==resource and len(stage['jobs'])==1,'Single source-only owner required')
    job=stage['jobs'][0];argv=job['argv'];entry=a.command_entry(argv);offset=2 if argv[1]=='-B' else 1
    require(entry==Path(entry_path).resolve() and argv[offset+1:]==['--config',str(Path(config_path).resolve())]
        and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json'),'Actual source command differs')
    base=Path(oc['owner_out']);ip=base/'owner_identity.json';ident=read(ip)
    require(a.same_identity(ident,a.identity(os.getppid())) and ident['registration_sha256']==sha(cfg['registration'])
        and ident['config_sha256']==sha(cfg['owner_config']) and not (base/'failure.json').exists(),'Actual parent is not bound owner')
    lp=base/'stages/source/workers'/job['id']/'launch.json';start=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-start<10 and a.same_identity(ident,a.identity(os.getppid())),'Worker launch not sealed');time.sleep(.1)
    launch=read(lp);current=a.identity(os.getpid())
    require(a.same_identity(launch['identity'],current) and launch['registration_sha256']==sha(cfg['registration'])
        and launch['argv']==argv and launch['resource']==resource,'Actual child launch differs')
    files=[ip,lp]
    if resource=='gpu':
        require(launch['threads']==oc['gpu_threads']==6 and set(launch['affinity'])==set(oc['gpu_affinity'])==set(os.sched_getaffinity(0))
            and os.environ.get('CUDA_VISIBLE_DEVICES')==str(oc['gpu_device']),'GPU numerical resource scope differs')
        gp=base/'stages/source/gpu_admission.json';g=read(gp);files.append(gp)
        require(g['status']=='GPU_IDLE_CONFIRMED' and g['registration_sha256']==sha(cfg['registration']),'Exclusive GPU admission absent')
    else:
        require(launch['threads']==2 and set(launch['affinity'])==set(cfg['cpu_affinity'])==set(os.sched_getaffinity(0))
            and os.getpriority(os.PRIO_PROCESS,0)==15 and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU preparation limits differ')
        require(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'CPU threads differ')
    return dict(owner_identity=ident,worker_identity=current,bindings=bind(files))


def source_boundary(ctx,supervision,started,monotonic_start,stop=False):
    cfg=ctx['cfg'];elapsed=time.monotonic()-monotonic_start
    require(not stop and not Path(cfg['stop_file']).exists() and not (Path(cfg['out'])/'STOP').exists(),'STOP at source boundary')
    require(elapsed<cfg['max_seconds'] and max(time.time(),started+elapsed)<DEADLINE,'Original source deadline')
    require(ctx['owner'].same_identity(supervision['owner_identity'],ctx['owner'].identity(os.getppid())),'Source owner changed')


def final_budget(ctx):
    cfg=ctx['cfg'];after=ctx['owner'].budget_snapshot(cfg['ledger'],sha(cfg['budget_registration']),cfg['phase_limits'],quiescent=True)
    require(after==ctx['before'],'Source-only stage changed packet ledger');verify(ctx['bound']);return after
