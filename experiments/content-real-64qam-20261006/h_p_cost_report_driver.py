"""Registered read-only export of already completed H/P statistics and H costs.

Metadata admission precedes table reads. All original files remain immutable;
this entry performs no statistical resampling, policy selection or model call.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

DONE='H_P_COMPONENT_REPORT_COMPLETE_MAIN_PENDING'
SCOPE='H_P_FROZEN_TABLE_AND_COMPONENT_COST_EXPORT_ONLY'
JOB='h_p_component_report'
QUALIFIED='H_P_COST_REPORT_CPU_INTERFACE_QUALIFICATION_PASS'
QUALIFICATION_MODULES=['test_h_p_cost_report','test_h_p_cost_report_driver','test_register_h_p_cost_report']
QUALIFICATION_COUNT=24
PROJECTION_SHA='6bcf40c835b7b0ee1e90adf3f66857979b1c2f11a8d1a47ee58c058d546d8545'
CPU_SHA='b4935e3c58b742d37fc288f3d8fe213fd7b64a3e65bc78865486549bbb9d0e13'
EXPECTED=dict(source_count=100,H_frames=5400,P_frames=600,H_points=18,P_points=2,
    H_internal_comparisons=26,H_P_comparisons=18,common_metrics=18,missing_P_metrics=3,
    timing_sources=16,timing_cases=288,policy_selection=False,development_used=True,holdout_used=False,
    GPU_used=False,new_packet_decodes=0,new_model_calls=0,new_metric_calls=0,new_noise_draws=0,
    new_bootstrap_replicates=0,new_comparisons=0,MAIN_complete=False,H_success_evaluated=False,
    H_full_delivery_claimed=False,overall_development_complete=False)
SPEC_KEYS={'config','owner_config','registration','launch','completion'}
BATCH_STATUS={'H':'H18_DEVELOPMENT_DESCRIPTIVE_AGGREGATION_COMPLETE',
    'HP':'H_P_FIXED_SOURCE_COMPARISON_COMPLETE_MAIN_PENDING','timing':'H_FIXED16_ONLINE_COMPONENTS_COMPLETE'}
COMPONENT_DEFINITIONS={
    'TX_visual_encoding':'Original fresh Encoder/VQ only; measured before TX_source_total starts.',
    'TX_source_total':'Frozen select_payload including source coding and fallback; excludes Encoder/VQ and all header/FEC/modulation/channel work.',
    'RX_source_and_image_total':'Actual received source parsing/decoding and suffix VAR/image reconstruction; excludes the earlier PHY packet reception.',
    'nested_components':'Integer arithmetic, probability setup/teardown, VAR probability and CDF counters overlap their enclosing source timers; do not add twice.',
    'end_to_end':'NOT_MEASURED; no reconstructed whole-chain total is computed.',
    'PHY':'Historical concurrent event windows with bookkeeping only; no isolated encode/decode measurement.'}
_HASH_CACHE=None


@contextmanager
def hashing_session():
    """One operation only; every use stats immutable inputs, first read checks both ends."""
    global _HASH_CACHE
    outer=_HASH_CACHE is None
    if outer:_HASH_CACHE={}
    try:yield
    finally:
        if outer:_HASH_CACHE=None


def cached_operation(function):
    @wraps(function)
    def call(*args,**kwargs):
        with hashing_session():return function(*args,**kwargs)
    return call


def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    p=Path(p);before=p.stat();key=(str(p.absolute()),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    if _HASH_CACHE is not None and key in _HASH_CACHE:return _HASH_CACHE[key]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed during SHA read: '+str(p))
    value=h.hexdigest()
    if _HASH_CACHE is not None:_HASH_CACHE[key]=value
    return value
def bind(paths):return {str(Path(p).absolute()):sha(p) for p in paths}
def verify(values):
    for p,s in values.items():
        require(Path(p).is_absolute() and re.fullmatch('[0-9a-f]{64}',s) and sha(p)==s,'Changed binding: '+p)
def merge(*maps):
    out={}
    for values in maps:
        for p,s in values.items():require(p not in out or out[p]==s,'Conflicting binding: '+p);out[p]=s
    return out
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def budget_ok(value):
    require(value['charged']==164760 and value['phase_charged']['development']==10800
        and value['development_remaining']==2400 and value['failed']==value['unresolved']==0,
        'Original quiescent H budget required; MAIN reserve remains unused')
def csv_read(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    require(rows,'Nonempty table required');fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def contained(p,base):
    require(Path(base).resolve() in Path(p).resolve().parents,'Output escaped original result directory')


def sealed_file(batch,name):
    p=batch['data_dir']/name
    require(batch['science']['outputs'].get(str(p))==sha(p),'Required original output is not sealed: '+str(p))
    return p


def normal_batch(request,key,api,cpu,wait):
    spec=request['batches'][key];require(set(spec)==SPEC_KEYS,'Exact original normal batch fields required')
    for p in spec.values():require(request['input_bindings'].get(p)==sha(p),'Unbound original batch input: '+p)
    cfg,reg=read(spec['config']),read(spec['registration'])
    require(cfg['registration']==spec['registration'] and cfg.get('owner_config',cfg.get('visual_owner_config'))==spec['owner_config'],
            'Original scientific config/owner differs')
    require(reg['input_bindings'].get(spec['config'])==sha(spec['config']),'Original config not registered')
    for k in ('source_bindings','input_bindings'):verify(reg[k])
    for p,s in reg['source_bindings'].items():require(request['source_bindings'].get(p)==s,'Original source map must be inherited')
    for p,s in reg['input_bindings'].items():require(request['input_bindings'].get(p)==s,'Original input map must be inherited')
    bound=merge(reg['source_bindings'],reg['input_bindings'])
    for name in ('owner','cpu','wait'):
        p=request['modules'][name];require(bound.get(p)==sha(p),'Closure API was not bound by original execution: '+name)
    closed=cpu.closed_batch(api,wait,dict(config=spec['owner_config'],registration=spec['registration'],
        launch=spec['launch'],completion=spec['completion']),request['modules']['owner'],
        dict(status=BATCH_STATUS[key],policy_selection=False,development_used=True,holdout_used=False,
             new_packet_decodes=0,MAIN_complete=False,H_full_delivery_claimed=False),api.raw_process_state)
    done=closed['done'];require(done['config_sha256']==sha(spec['config']),'Scientific config SHA changed')
    require(done['budget_before']==done['budget_after']==closed['owner_done']['budget'],'Original normal batch budget changed')
    budget_ok(done['budget_after']);science=done;data_dir=Path(cfg['out'])
    if key=='H':
        original=done['original_aggregation_receipt'];p=Path(original['path'])
        require(p==data_dir/'statistics/completion.json' and original['sha256']==sha(p)
            and done['outputs'].get(str(p))==sha(p) and done['original_aggregation_receipt_modified'] is False,
            'Original H scientific receipt must remain separate from owner envelope')
        science=read(p);data_dir=p.parent
        require(science['status']==done['status'] and science['budget_before']==science['budget_after']==done['budget_after'],
                'Original H aggregation receipt differs')
        for p,s in science['outputs'].items():require(done['outputs'].get(p)==s,'Unsealed original H table')
    for p in science['outputs']:contained(p,data_dir)
    for k in ('input_bindings','source_bindings','outputs'):verify(science[k])
    return dict(spec=spec,cfg=cfg,reg=reg,closed=closed,done=done,science=science,data_dir=data_dir,
        bindings=merge(closed['bindings'],reg['input_bindings'],science['input_bindings'],science['outputs'],bind(spec.values())))


def metadata_admission(batches):
    """Read frozen identities/proofs only, never summary, paired or cost values."""
    h,hp,t=(batches[k] for k in ('H','HP','timing'));hd,hpd,td=(b['science'] for b in (h,hp,t))
    require(hd['source_count']==100 and hd['frame_count']==5400 and hd['comparison_count']==26
        and hpd['source_count']==100 and hpd['H_frames']==5400 and hpd['P_frames']==600
        and hpd['comparison_count']==18 and hpd['admitted_metric_count']==18 and hpd['missing_P_metric_count']==3,
        'Completed original H/H-P scope differs')
    hpreq=read(hp['cfg']['request']);hm=hpreq['H_request']['metric_batch']
    require(h['cfg']['metric_config']==hm['config'] and t['cfg']['metric_batch']==hm,'Different original H metrics lineage')
    ids=hpd['source_ids'];require(len(ids)==len(set(ids))==100 and hpd['exact_original_pixel_pairs']==100
        and hpd['noise_seeds']=={'H':[6201,6202,6203],'P':[2001,2002,2003]}
        and hpd['original_rows_modified'] is False,'Original source/noise grid changed')
    hplan=read(sealed_file(h,'comparison_plan.json'));plan=read(sealed_file(hp,'comparison_plan.json'))
    admission=read(sealed_file(hp,'common_metric_admission.json'))
    require(plan==read(hp['cfg']['comparison_plan']) and admission==read(hp['cfg']['admission'])
        and hpd['comparison_plan_sha256']==sha(hp['cfg']['comparison_plan'])
        and hpd['common_metric_admission_sha256']==sha(hp['cfg']['admission'])
        and plan['H_catalog']==hplan['catalog'],'Precomparison plan/admission differs from original frozen H catalog')
    require(len(hplan['comparisons'])==26 and len(plan['comparisons'])==18
        and hplan['selection'] is False and plan['selection'] is False,'Original fixed comparisons required')
    proof=read(sealed_file(hp,'normalization_proof.json'));bridge=read(sealed_file(hp,'target_pixel_bridge.json'))
    require(proof['source_ids']==ids and proof['metrics_recomputed'] is False and proof['comparisons_selected'] is False
        and proof['frame_level_noise_pairing_claimed'] is False and len(bridge)==100,'Sealed source normalization proof differs')
    for i,row in enumerate(bridge):
        require(row['source_index']==i and row['source_id']==ids[i] and row['pixel_equality']=='EXACT_UINT8_AND_FLOAT32_CONVERSION'
            and row['shape']==[3,256,256] and row['channel_order']=='RGB_CHW' and row['new_metric_calls']==0
            and set(row['original_reference_sha256'])=={'H','P'},'Exact original pixel bridge is absent')
        for checksum in [row['reference_sha256'],*row['original_reference_sha256'].values()]:
            require(re.fullmatch('[0-9a-f]{64}',checksum),'Original hash domain proof missing')
        for branch in ('H','P'):
            verify(row['branch_input_bindings'][branch])
            for p,s in row['branch_input_bindings'][branch].items():
                require(hpd['input_bindings'].get(p)==s,'Pixel source proof not bound to H/P completion')
    fixed=[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95]
    require(td['source_indices']==fixed and td['source_ids']==[ids[i] for i in fixed]
        and td['source_count']==16 and td['component_case_count']==288 and td['H_policy_snr_points']==18
        and td['exact_TX_parity'] is td['exact_RX_parity'] is True and td['noise_seed']==6201
        and td['warmup_repetitions']==1 and td['measured_repetitions']==3,'Complete original fixed16 timing required')
    for i in fixed:
        cp=read(sealed_file(t,f'source_checkpoints/{i:04d}.json'))
        require(cp['status']=='H_ONLINE_COMPONENT_SOURCE_COMPLETE' and cp['source_index']==i and cp['source_id']==ids[i]
            and cp['registration_sha256']==td['registration_sha256'] and cp['component_case_count']==18
            and cp['noise_seed']==6201 and cp['timing_repetitions_are_quality_samples'] is False,
            'Missing or changed fixed16 timing checkpoint')
        verify(cp['outputs']);verify(cp['input_bindings'])
        for p,s in cp['outputs'].items():require(td['outputs'].get(p)==s,'Unsealed timing source result')
    return dict(source_ids=ids,H_plan=hplan,HP_plan=plan,admission=admission,
        exact_original_pixel_pairs=100,quality_rows_read=False,component_values_read=False)


@cached_operation
def preflight(request):
    require(request['schema']=='H_P_COST_REPORT_REQUEST_V1' and set(request['batches'])=={'H','HP','timing'}
        and set(request['modules'])=={'owner','cpu','wait','projection'},'Exact report request required')
    sources=request['source_bindings'];verify(sources);verify(request['input_bindings'])
    for p in [*request['modules'].values(),str(Path(__file__).absolute())]:
        require(sources.get(p)==sha(p),'Report/closure source not bound: '+p)
    require(sha(request['modules']['projection'])==PROJECTION_SHA and sha(request['modules']['cpu'])==CPU_SHA,
            'Frozen projection or original normal-closure checker changed')
    api=module(request['modules']['owner'],'hpcost_original_owner')
    cpu=module(request['modules']['cpu'],'hpcost_original_cpu')
    wait=module(request['modules']['wait'],'hpcost_original_wait')
    # Isolated loaded module objects only. The SHA algorithm/expected values
    # remain exact; duplicate traversals use the same operation-local stat cache.
    api.sha=sha;cpu.sha=sha
    projection=module(request['modules']['projection'],'hpcost_frozen_projection')
    batches={k:normal_batch(request,k,api,cpu,wait) for k in ('H','HP','timing')}
    owners=[b['closed']['owner'] for b in batches.values()];base=owners[1]
    for own in owners:
        for k in ('root','out','budget_path','budget_registration','budget_registration_sha256','phase_limits'):
            require(own[k]==base[k],'Report stages do not share original H protocol/budget')
    require(base['root']==request['root'],'Request root differs')
    before=api.budget_snapshot(base['budget_path'],base['budget_registration_sha256'],base['phase_limits'],quiescent=True)
    budget_ok(before)
    require(all(b['science']['budget_before']==b['science']['budget_after']==before for b in batches.values()),
            'H budget advanced after original completed batches')
    metadata=metadata_admission(batches)
    bound=merge(request['input_bindings'],*(b['bindings'] for b in batches.values()))
    python=batches['HP']['closed']['owner_identity']['argv'][0]
    require(python==batches['H']['closed']['owner_identity']['argv'][0]==batches['timing']['closed']['owner_identity']['argv'][0]
        and '/UNIFIED-METRICS-' in python.replace('\\','/'),'Original common UM interpreter required')
    return dict(request=request,sources=sources,bindings=bound,batches=batches,metadata=metadata,
        owner=api,cpu=cpu,wait=wait,projection=projection,template_owner=base,python=python,before=before)


def unchanged(x):
    verify(x['sources']);verify(x['bindings']);own=x['template_owner']
    require(x['owner'].budget_snapshot(own['budget_path'],own['budget_registration_sha256'],own['phase_limits'],quiescent=True)==x['before'],
            'Report must not change packet budget')
    for batch in x['batches'].values():
        for d in (Path(batch['closed']['owner']['owner_out']),Path(batch['cfg']['out']),Path(own['out'])):
            require(not (d/'STOP').exists() and not (d/'failure.json').exists(),'New predecessor STOP/failure')
    if 'execution_owner' in x:
        for d in (Path(x['execution_owner']['owner_out']),Path(x['cfg']['out'])):
            require(not(d/'STOP').exists() and not(d/'failure.json').exists(),'Current report stopped/failed')


def project_tables(x):
    h,hp,t=(x['batches'][k] for k in ('H','HP','timing'))
    def load(b,m):return {k:(csv_read if name.endswith('.csv') else read)(sealed_file(b,name)) for k,name in m.items()}
    hv=load(h,dict(point_catalog='point_catalog.csv',tx_source_scale='tx_source_scale.csv',
        rx_received_scale='rx_received_scale.csv',receiver_status='receiver_status.csv'))
    pv=load(hp,dict(point_identity='point_identity.json',common_metric_admission='common_metric_admission.json',
        comparison_plan='comparison_plan.json',summary='summary.csv',paired='paired.csv'))
    tv=load(t,dict(component_summary='component_summary.json',historical_phy_event_windows='historical_phy_event_windows.json'))
    for v,b in ((hv,h),(pv,hp),(tv,t)):v['completion']=b['science']
    return x['projection'].project(pv,hv,tv)


def export(x,out):
    """Copy original numeric tables byte-for-byte; only report layout/cost projection are new."""
    view=project_tables(x);out=Path(out);copies={}
    files={'HP':['summary.csv','paired.csv','common_metric_admission.json','comparison_plan.json','point_identity.json',
                 'target_pixel_bridge.json','metric_availability.json','normalization_proof.json'],
        'H':['summary.csv','paired.csv','point_catalog.csv','comparison_plan.json','tx_source_scale.csv','rx_received_scale.csv',
             'receiver_status.csv','classifier_transition.csv','wire_diagnostics.csv','metric_provenance.json','cost_scope.json'],
        'timing':['component_summary.json','historical_phy_event_windows.json']}
    for key,names in files.items():
        dest=out/key;dest.mkdir()
        for name in names:
            src=sealed_file(x['batches'][key],name);target=dest/name
            require(not target.exists() and src.stat().st_size<10*1024**2,'Fresh light text copy required')
            shutil.copyfile(src,target);require(sha(src)==sha(target),'Original table bytes changed')
            copies[str(target)]=dict(original_path=str(src),original_sha256=sha(src),export_sha256=sha(target))
    csv_write(out/'h_components.csv',view['costs']['components'])
    report=x['projection'].markdown(view)
    report=report.replace('本文件由纯表格投影生成；正式输出前仍须外部核验各阶段正常退出及全部输入输出 SHA。',
        '本次导出已核验原 H 汇总、H/P 统计和计时阶段正常退出及全部原输出 SHA。原表逐字节保留；新导出任务的正常退出另由其监督程序确认。')
    report=report.replace('TX/RX 总计是包含子项的计时，不与子项相加；VAR 概率与 CDF 子项单列。',
        'TX_visual_encoding 是独立的 Encoder/VQ 时间；TX_source_total 从随后 select_payload 开始，仅覆盖源编码与回退，不含 Encoder/VQ、头部编码、FEC 或调制。'
        'RX_source_and_image_total 覆盖实际源解码及图像恢复，不含先前的 PHY 接收。VAR 概率、CDF 和整数算术是嵌套子项，不重复相加，也不把字段名 total 解释为整链耗时。')
    (out/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    save(out/'copy_inventory.json',dict(status='ORIGINAL_TABLE_BYTES_PRESERVED',files=copies,
        new_bootstrap_replicates=0,new_comparisons=0))
    save(out/'scope.json',{k:view[k] for k in ('MAIN','final_system_conclusion','H_success_evaluated','missing_P_metrics','interpretation')})
    save(out/'component_definitions.json',COMPONENT_DEFINITIONS)
    return view


@cached_operation
def registered_context(path,admitted=None):
    path=Path(path).absolute();cfg=read(path);reg=read(cfg['registration']);own=read(cfg['owner_config'])
    verify(reg['source_bindings']);verify(reg['input_bindings'])
    for p in (str(path),cfg['request'],cfg['owner_config'],cfg['qualification']):
        require(reg['input_bindings'].get(p)==sha(p),'Registered report input absent: '+p)
    request=read(cfg['request']);x=preflight(request) if admitted is None else admitted
    require(x['request']==request,'Already admitted metadata belongs to another request')
    q=read(cfg['qualification'])
    for k in ('source_bindings','input_bindings','outputs'):verify(q[k])
    require(q['status']==QUALIFIED and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['actual_quality_rows_read'] is False and q['python']==x['python'],'Actual CPU interface qualification required')
    argv=[x['python'],'-B','-m','unittest','-v',*QUALIFICATION_MODULES]
    require(q['test_count']==cfg['qualification_test_count']==QUALIFICATION_COUNT
        and q['results']==[dict(argv=argv,exit_code=0,test_count=QUALIFICATION_COUNT)],
            'Qualification test inventory differs')
    log=Path(cfg['qualification']).parent/'tests.log';text=log.read_text(encoding='utf-8',errors='replace')
    require(q['outputs'].get(str(log))==sha(log) and f'Ran {QUALIFICATION_COUNT} tests' in text
        and text.rstrip().endswith('OK'),'Actual qualification test log differs')
    for p,s in x['sources'].items():require(q['source_bindings'].get(p)==reg['source_bindings'].get(p)==s,'Unqualified report source')
    for p,s in x['bindings'].items():require(reg['input_bindings'].get(p)==s,'Unregistered predecessor evidence')
    require(reg['source_stage_scope']==SCOPE and reg['budget_before']==x['before'],'Report registration scope differs')
    x['owner'].validate_config(own,reg,sha(cfg['owner_config']))
    require(own['owner_out']==str(Path(cfg['owner_config']).parent) and own['registration']==cfg['registration']
        and len(own['stages'])==1,'Separate report owner required')
    s=own['stages'][0];require(s['id']=='report' and s['resource']=='cpu' and len(s['jobs'])==1,'Single CPU report stage required')
    job=s['jobs'][0];require(job['id']==JOB and job['argv']==[x['python'],'-B',str(Path(__file__).absolute()),'--config',str(path)]
        and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json')
        and job['accepted_statuses']==[DONE] and job['receipt_expect']==EXPECTED,'Report job/receipt contract differs')
    x.update(cfg=cfg,reg=reg,execution_owner=own,job=job);unchanged(x);return x


def live_owner(x):
    api=x['owner'];cfg=x['cfg'];own=x['execution_owner'];base=Path(own['owner_out']);ip=base/'owner_identity.json'
    ident=read(ip);require(api.same_identity(ident,api.identity(os.getppid()))
        and ident['registration_sha256']==sha(cfg['registration']) and ident['config_sha256']==sha(cfg['owner_config']),
        'Actual current report owner required')
    lp=base/'stages/report/workers'/JOB/'launch.json';began=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-began<10 and api.same_identity(ident,api.identity(os.getppid())),'Worker launch not sealed');time.sleep(.1)
    launched=read(lp)
    require(api.same_identity(launched['identity'],api.identity(os.getpid())) and launched['argv']==x['job']['argv']
        and launched['registration_sha256']==sha(cfg['registration']) and launched['resource']=='cpu' and launched['threads']==2
        and set(launched['affinity'])==set(own['cpu_affinities'][0])==os.sched_getaffinity(0)
        and os.getpriority(os.PRIO_PROCESS,0)==15,'Actual CPU worker/resource identity differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only2threads required')
    return bind((ip,lp))


def claim(out,regsha):
    out=Path(out);require(not out.exists() or out.is_dir() and not out.is_symlink() and not any(out.iterdir()),
        'Only fresh or owner-precreated empty output accepted; preserve previous attempts')
    out.mkdir(parents=True,exist_ok=True);save(out/'attempt.json',dict(status='READ_ONLY_REPORT_STARTED',registration_sha256=regsha))


@cached_operation
def run(path):
    x=registered_context(path);supervision=live_owner(x);out=Path(x['cfg']['out']);began=time.monotonic()
    claim(out,sha(x['cfg']['registration']))
    try:
        unchanged(x);export(x,out);unchanged(x)
        require(time.monotonic()-began<x['cfg']['max_seconds'],'Finite report deadline exceeded')
        closure={k:dict(normal_owner_success=True,owner_identity=b['closed']['owner_identity'],
            worker_identities=b['closed']['children'],bindings=b['closed']['bindings']) for k,b in x['batches'].items()}
        save(out/'normal_predecessor_closure.json',closure)
        inputs=merge(x['bindings'],supervision,bind((path,x['cfg']['request'],x['cfg']['qualification'])))
        save(out/'input_inventory.json',dict(input_bindings=inputs,source_bindings=x['sources'],budget=x['before']))
        outputs=bind(p for p in out.rglob('*') if p.is_file() and p.name!='attempt.json')
        value=dict(status=DONE,**EXPECTED,registration_sha256=sha(x['cfg']['registration']),config_sha256=sha(path),
            source_ids=x['metadata']['source_ids'],outputs=outputs,input_bindings=inputs,source_bindings=x['reg']['source_bindings'],
            budget_before=x['before'],budget_after=x['before'],original_results_modified=False,
            predecessor_normal_owners_verified=True,own_owner_success_not_yet_certified=True,
            final_system_conclusion='UNRESOLVED_MAIN_NOT_COMPLETE',future_stage_automatic=False,elapsed_seconds=time.monotonic()-began)
        verify(outputs);unchanged(x);save(out/'completion.json',value);return value
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();v=run(a.config)
    print(json.dumps(dict(status=v['status'],MAIN_complete=False,new_packet_decodes=0)))
