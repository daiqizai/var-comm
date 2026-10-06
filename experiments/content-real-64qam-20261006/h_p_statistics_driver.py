"""Registered CPU-only H/P identity bridge and fixed source-paired statistics.

Metadata admission and the eighteen pairs are sealed before score rows load.
Original rows, seeds, policies, image hashes and numerical methods are retained.
This entry does not load a model, decode PHY, choose an H winner or admit MAIN.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h_p_result_normalization as n
import h_p_main_source_statistics as statistics

NORMALIZER_SHA='2ec18f85dfb04fb28b269117aae160262e5fbd09359471ba017e3f248ee3427f'
TEMPLATE_SHA='5a298cc15fbc8d53efb46d9e655666941d81682e6301eb13d52b317889fae944'
DONE='H_P_FIXED_SOURCE_COMPARISON_COMPLETE_MAIN_PENDING'
SCOPE='H18_P2_SOURCE_PAIRED_DESCRIPTIVE_STATISTICS_NO_RESELECTION'
JOB='h_p_descriptive_statistics'
QUALIFIED='H_P_STATISTICS_CPU_QUALIFICATION_PASS'
EXPECTED=dict(source_count=100,H_frames=5400,P_frames=600,normalized_frames=6000,H_points=18,P_points=2,
    comparison_count=18,admitted_metric_count=18,missing_P_metric_count=3,
    bootstrap_replicates=10000,bootstrap_seed=2026100605,policy_selection=False,development_used=True,
    holdout_used=False,GPU_used=False,new_packet_decodes=0,new_metric_calls=0,new_noise_draws=0,
    P_complete=True,MAIN_complete=False,H_success_evaluated=False,H_full_delivery_claimed=False,
    overall_development_complete=False,frame_level_noise_pairing_claimed=False)
require=n.require; read=n.read; sha=n.sha; verify=n.verify; identity=n.identity


def bind(paths): return {str(Path(p).absolute()):sha(p) for p in paths}
def merge(*maps):
    out={}
    for values in maps:
        for p,s in values.items():
            require(p not in out or out[p]==s,'Conflicting binding: '+p);out[p]=s
    return out
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def csv_file(path,rows):
    require(rows,'Expected nonempty table')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def check_budget(v):
    require(v['charged']==164760 and v['phase_charged']['development']==10800
        and v['development_remaining']==2400 and v['failed']==v['unresolved']==0,
        'Closed H164760 budget and untouched2400 reserve required')


def metadata_admission(template,h_done,p_done,p_registration,paths):
    """Identity-only, deliberately no frame/source-quality row arguments."""
    require(template['status']=='H_P_METRIC_METADATA_AUDIT_PREPARED_NOT_FINAL_ADMISSION'
        and template['current_admitted_count']==0 and set(template['metrics'])==set(n.METRICS),
        'Original complete21-metric metadata audit required')
    hm=h_done['metric_metadata'];hc=hm['independent'];pc=p_done['classifier_identity']
    require({k:v for k,v in hc.items() if k!='policy_sha256'}=={k:v for k,v in pc.items() if k!='policy_sha256'},
        'Actual H/P ConvNeXt definition/model/preprocessing/precision/batch differs')
    require(hc['classification_batch_size']==pc['classification_batch_size']==1
        and hc['used_for_selection'] is pc['used_for_selection'] is False,'Independent B1 classifier required')
    require(h_done['numerical_runtime']==p_done['numerical_runtime']==p_registration['numerical_runtime'],
        'Frozen original numerical runtime differs')
    require(hm['metric_batch_size']==1 and p_registration['metric_batch_size']==16
        and p_registration['old_metric_values_retained'] is True
        and p_registration['new_metrics_used_for_selection'] is False,'Historical batch/value protocol differs')
    require(hm['metric_evaluator']==read(paths['P_metadata']),'Actual H evaluator differs from original P metadata')
    require(p_registration['modelmanifest_sha256']==sha(paths['P_manifest']), 'Original P modelmanifest differs')
    require(hm['metric_evaluator']['metrics'] and p_done['legacy_scalars_preserved'] is True
        and p_done['RGB_exact_parity'] is True and p_done['ConvNeXt_scored'] is True,'Real P replay proof absent')
    hb=merge(h_done['input_bindings'],h_done['source_bindings'],bind([paths['H_completion']]))
    pb=merge(p_done['input_bindings'],p_done['source_bindings'],bind([paths['P_completion']]))
    result=dict(status='FROZEN_H_P_COMMON_METRIC_ADMISSION',used_for_selection=False,metrics={},input_bindings={},
        generated_from_identity_only=True,quality_rows_used_for_admission=False,
        prepared_template_sha256=TEMPLATE_SHA,
        actual_classifier_identities=dict(H=hc,P=pc),
        scope='Exact definitions/weights/preprocessing; documented numerical methods retained without bitwise-equivalence claim')
    for metric in n.METRICS:
        entry=copy.deepcopy(template['metrics'][metric])
        if metric in n.P_MISSING:
            require(entry['status']=='MISSING_IN_P','Original P missing metrics changed')
            result['metrics'][metric]=dict(status='MISSING_IN_P',reason=entry['reason']);continue
        require(entry['metadata_audit_status']=='MATCHED_FROZEN_METADATA_WAIT_ACTUAL_GATES'
            and entry['branches']['H']['identity']==entry['branches']['P']['identity'],'Unaudited metric identity')
        entry.update(status='ADMITTED',numerical_difference_admitted=True,
            numerical_difference_note='Descriptive comparison preserves H float64 PSNR, P original float32 PSNR; H B1, original P legacy quality B3 and unified cached B1/qualified B16, P new ConvNeXt B1. No bitwise-equivalence claim.',
            reason='Identity verified from sealed original metadata; actual normal closure and exact pixel bridge remain mandatory execution gates')
        entry.pop('pending_gates',None)
        for branch,proof in entry['branches'].items():
            proof['evidence'].update(bind([paths['H_completion' if branch=='H' else 'P_completion']]))
            actual=hb if branch=='H' else pb
            for p,s in proof['evidence'].items():
                require(actual.get(p)==s,'Metric identity evidence not sealed by '+branch+': '+p)
            result['input_bindings']=merge(result['input_bindings'],proof['evidence'])
        result['metrics'][metric]=entry
    n.admit_metrics(result,{'H':dict(bindings=hb),'P':dict(bindings=pb)})
    require(sum(v['status']=='ADMITTED' for v in result['metrics'].values())==18,'Exactly18 existing common metrics required')
    return result


def comparison_plan(hplan):
    """Frozen calibration identities only; no score values or winner selection."""
    catalog=hplan['catalog']; require(len(catalog)==18 and {r['development_slot'] for r in catalog}==set(range(18)),
        'Complete frozen18 H point identities required')
    pairs=[]
    for row in sorted(catalog,key=lambda r:r['development_slot']):
        slot=row['development_slot'];snr=row['snr_db'];point=f'H18_SLOT_{slot:02d}'
        require(row['point_id']==point and snr in (13,19),'Original point/SNR identity differs')
        pairs.append(dict(comparison_id=point+'_MINUS_P1024_SNR_'+str(snr),method=point,
            reference=f'P1024_SNR_{snr}',snr_db=snr,role=row['role'],arm=row['arm'],
            candidate_id=row['candidate_id'],interpretation='FIXED_POINT_DESCRIPTIVE_H_MINUS_P_NOT_H_WINNER_SELECTION'))
    require(len(pairs)==18 and len({(r['method'],r['reference']) for r in pairs})==18,'Exactly18 pairs, not36')
    return dict(status='FROZEN_H18_TO_SAME_SNR_P18_PAIRS_BEFORE_SCORE_ROWS',comparisons=pairs,
        H_catalog=catalog,source_count=100,noise_seeds={'H':list(n.NOISE['H']),'P':list(n.NOISE['P'])},
        comparison_count=18,selection=False,MAIN_complete=False,H_success_evaluated=False,
        bootstrap_replicates=10000,bootstrap_seed=2026100605,
        multiplicity='Predeclared descriptive pointwise95% intervals, no multiplicity-adjusted winner claim')


def historical_graph(cfg,reg,done):
    """Closed P evidence is historical; later added Python is not retroactive."""
    bound=merge(reg['source_bindings'],reg['input_bindings']);path=cfg['visual_source_closure']
    require(bound.get(path)==sha(path),'Original P graph unbound')
    graph=read(path);verify(graph['source_bindings'])
    require(graph['status']=='EXACT_SOURCE_CLOSURE_MATCH' and graph['source_bindings']==done['visual_source_bindings']
        and all(bound.get(p)==s for p,s in graph['source_bindings'].items()),'Original executed P graph changed')
    return graph


def normal_p_metadata(spec,module_path,hctx,hclosed,h_spec):
    """Original normal owner and immutable P inputs, without current enumeration.

    The frozen replay load_registered is a pre-execution gate and enumerates
    new tracked Python. It must not be reused to validate a completed old run.
    No source check is removed: every original bound byte and graph is checked.
    """
    require(set(spec)=={'config','owner_config','registration','launch','completion'} and sha(module_path)==n.P_DRIVER_SHA,
        'Exact P600 batch and frozen entry required')
    d=n.module(module_path,'hp_closed_p_driver');c=read(spec['config']);reg=read(spec['registration'])
    bound=merge(reg['source_bindings'],reg['input_bindings']);verify(bound)
    require(reg['status']=='H_EXECUTION_REVISION_REGISTERED' and reg['branch']=='H'
        and reg['source_stage_scope']==d.SCOPE and reg['allowed_stage_ids']==['development'], 'Wrong historical P registration')
    for p in [spec['config'],*[c[k] for k in d.REQUIRED],*c['metric_batch'].values()]:
        require(bound.get(p)==sha(p),'Original P dependency unbound: '+p)
    require(c['schema']=='P600_RECEIVED_REPLAY_CONFIG_V1' and c['replay_driver_module']==module_path
        and c['metric_batch']==h_spec and c['h_metric_driver_module']==hctx['cfg']['metric_driver_module'], 'Historical H/P dependency differs')
    require(c['registration']==spec['registration'] and c['visual_owner_config']==spec['owner_config'],'P batch lineage differs')
    for k in ('root','H_out','owner_module','wait_module','cpu_driver_module','ledger','budget_registration','phase_limits','stop_file','protocol'):
        require(c[k]==hctx['cfg'][k],'Historical H/P execution differs: '+k)
    expected=dict(status=d.DONE,source_count=100,frame_count=600,P_policy_snr_points=2,N=1024,
        snrs_db=[13,19],noise_seeds=list(n.NOISE['P']),RGB_exact_parity=True,legacy_scalars_preserved=True,
        ConvNeXt_scored=True,source_predictions=100,reconstruction_predictions=600,new_packet_decodes=0,
        new_noise_draws=0,budget_writes=0,policy_selection=False,development_used=True,holdout_used=False,
        MAIN_complete=False,overall_development_complete=False,H_full_delivery_claimed=False)
    a=hctx['owner'];closed=hctx['cpu'].closed_batch(a,hctx['wait'],
        dict(config=spec['owner_config'],registration=spec['registration'],launch=spec['launch'],completion=spec['completion']),
        c['owner_module'],expected,a.raw_process_state)
    owner=closed['owner'];done=closed['done'];stages=owner['stages'];job=stages[0]['jobs'][0]
    require(len(stages)==len(stages[0]['jobs'])==1 and stages[0]['id']=='development' and stages[0]['resource']=='gpu'
        and job['id']=='p600_received_replay' and job['argv'][1:]==['-B',module_path,'--config',spec['config']]
        and job['out']==c['out'],'Original closed P GPU command differs')
    require(done['registration_sha256']==sha(spec['registration']) and done['config_sha256']==sha(spec['config'])
        and done['budget_before']==done['budget_after']==hctx['before']==closed['owner_done']['budget']==reg['budget_before']
        and done['metric_predecessor_completion_sha256']==sha(h_spec['completion']), 'P identity/budget differs')
    graph=historical_graph(c,reg,done)
    for k in ('source_bindings','input_bindings','outputs'):verify(done[k])
    for p,s in merge(hctx['bound'],hclosed['bindings'],hclosed['done']['outputs']).items():
        require(bound.get(p)==s,'P did not seal original H lineage: '+p)
    audit=read(c['latent_completion']);inventory=read(c['latent_inventory'])
    require(audit['outputs']=={c['latent_inventory']:sha(c['latent_inventory'])},'Original latent audit output changed')
    d.audited_input_paths(c,audit)
    for p,s in merge(audit['input_bindings'],audit['source_bindings']).items():require(bound.get(p)==s,'Original latent input unbound')
    require(done['source_ids']==hctx['source_ids'] and inventory['input_bindings'].get(c['selected_P_policy'])
        ==sha(c['selected_P_policy']),'Historical P population/selection differs')
    # Do not inspect legacy quality values during qualify/register. Their file
    # hashes are checked above; full unchanged scalar parity is worker work.
    ctx=dict(cfg=c,reg=reg,owner=a,bound=bound,driver=d,H_context=hctx,audit=audit,inventory=inventory,
        before=hctx['before'],static_bindings=graph['source_bindings'])
    return ctx,closed


def load_closed_p(ctx,closed):
    """Same six-row source and scalar proof as frozen n.load_p after closure."""
    c=ctx['cfg'];done=closed['done'];base=Path(c['out']);rows=[]
    core=n.module(c['core_module'],'hp_actual_closed_p_core')
    cctx=core.context(ctx['audit'],ctx['inventory'],read(c['scalar_inventory']),read(c['metrics_registration']),
        read(c['numerical_reference']),inventory_sha256=sha(c['latent_inventory']))
    require(done['source_ids']==cctx['source_ids'] and cctx['selected_P_sha256']==sha(c['selected_P_policy']),
        'Original P scalar population/policy differs')
    population,manifest,assets=ctx['driver'].original_assets(ctx['H_context'],cctx)
    for i,sid in enumerate(done['source_ids']):
        cp=base/'source_checkpoints'/f'{i:04d}.json';rp=base/'sources'/f'{i:04d}.json'
        for p in (cp,rp):require(done['outputs'].get(str(p))==sha(p),'P source output not sealed')
        checkpoint=read(cp);verify(checkpoint['outputs']);verify(checkpoint['input_bindings'])
        require(checkpoint['status']=='P600_RECEIVED_REPLAY_SOURCE_COMPLETE' and checkpoint['source_index']==i
            and checkpoint['source_id']==sid and checkpoint['frame_count']==checkpoint['RGB_parity_frames']==checkpoint['ConvNeXt_frames']==6
            and checkpoint['registration_sha256']==done['registration_sha256']
            and checkpoint['outputs'].get(str(rp))==sha(rp),'Original P checkpoint differs')
        part=read(rp);require(len(part)==6,'Incomplete P source');rows.extend(part)
    flat=base/'frame_metrics.json';require(done['outputs'].get(str(flat))==sha(flat) and rows==read(flat),'P flat rows differ')
    core.validate_complete(cctx,rows)
    return dict(branch='P',normal_owner_success=True,rows=rows,source_ids=done['source_ids'],policy_sha256=cctx['selected_P_sha256'],
        completion=done,population=population,manifest=manifest,assets=assets,
        bindings=merge(ctx['bound'],closed['bindings'],done['outputs']),context=ctx)


def preflight(request):
    """Registered metadata/normal closure only; never read H/P score output rows."""
    require(request['schema']=='H_P_FIXED_STATISTICS_REQUEST_V1','Wrong H/P request')
    verify(request['source_bindings']);verify(request['input_bindings'])
    for p in (str(Path(__file__).absolute()),str(Path(n.__file__).absolute()),str(Path(statistics.__file__).absolute()),
              request['H_aggregate'],request['P_driver'],request['H_request']['metric_driver']):
        require(request['source_bindings'].get(p)==sha(p),'New or historical entry unbound: '+p)
    require(sha(n.__file__)==NORMALIZER_SHA and sha(statistics.__file__)==n.STATISTICS_SHA
        and sha(request['H_aggregate'])==n.H_AGGREGATE_SHA and sha(request['P_driver'])==n.P_DRIVER_SHA,
        'Frozen normalization/statistic/branch entry differs')
    require(sha(request['metadata_template'])==TEMPLATE_SHA
        and request['input_bindings'].get(request['metadata_template'])==TEMPLATE_SHA,'Original audited metadata template differs')
    hs=request['H_request']['metric_batch'];ps=request['P_batch']
    for p in list(hs.values())+list(ps.values()):require(request['input_bindings'].get(p)==sha(p),'Branch batch unbound')
    template=read(request['metadata_template']);verify(template['input_bindings'])
    paths={k:v['original_path'] for k,v in template['evidence_files'].items()}
    require(paths['H_science']==hs['completion'],'H identity audit refers to another result')
    proof=dict(H_completion=hs['completion'],P_completion=ps['completion'],P_metadata=paths['P_metadata'],P_manifest=paths['P_manifest'])
    # This mapping uses only metadata, before loaders can inspect scalar caches.
    admission=metadata_admission(template,read(hs['completion']),read(ps['completion']),read(paths['P_registration']),proof)
    a=n.module(request['H_aggregate'],'hp_frozen_h_aggregate');hctx,hclosed=a.normal_metrics_closed(request['H_request'])
    plan=comparison_plan(a.comparison_plan(hctx['schedule']))
    pctx,pclosed=normal_p_metadata(ps,request['P_driver'],hctx,hclosed,hs)
    require(hctx['source_ids']==pclosed['done']['source_ids'] and hctx['before']==pctx['before'], 'Original populations/budgets differ')
    check_budget(hctx['before'])
    python=hclosed['owner']['stages'][0]['jobs'][0]['argv'][0]
    require(python==pclosed['owner']['stages'][0]['jobs'][0]['argv'][0], 'Original H/P UM environment differs')
    bindings=merge(request['input_bindings'],hctx['bound'],hclosed['bindings'],pctx['bound'],pclosed['bindings'],
        hclosed['done']['outputs'],pclosed['done']['outputs'],admission['input_bindings'])
    return dict(request=request,a=a,hctx=hctx,hclosed=hclosed,pctx=pctx,pclosed=pclosed,admission=admission,plan=plan,
        owner=hctx['owner'],before=hctx['before'],python=python,bindings=bindings,sources=request['source_bindings'])


def unchanged(x):
    verify(x['sources']);verify(x['bindings']);c=x['hctx']['cfg']
    require(not Path(c['stop_file']).exists(),'H STOP blocks CPU comparison')
    require(time.time()<x['hclosed']['owner']['overall_deadline_unix'],'Original H deadline expired')
    if 'cfg' in x:
        require(not (Path(x['cfg']['out'])/'STOP').exists() and not (Path(x['cfg']['owner_config']).parent/'STOP').exists(),
            'New execution/output STOP blocks CPU comparison')
    require(x['owner'].budget_snapshot(c['ledger'],sha(c['budget_registration']),c['phase_limits'],quiescent=True)==x['before'],
        'H ledger changed during CPU comparison')


def registered_context(config_path):
    cfg=read(config_path);reg=read(cfg['registration']);owner=read(cfg['owner_config'])
    require(cfg['schema']=='H_P_FIXED_STATISTICS_CONFIG_V1' and cfg['max_seconds']==7200,'Wrong comparison configuration')
    verify(reg['source_bindings']);verify(reg['input_bindings'])
    for p in (str(Path(config_path).absolute()),cfg['request'],cfg['owner_config'],cfg['admission'],cfg['comparison_plan'],cfg['qualification']):
        require(reg['input_bindings'].get(p)==sha(p),'Execution input unsealed: '+p)
    require(reg['source_stage_scope']==SCOPE and reg['allowed_stage_ids']==['report'] and reg['GPU_jobs']==0
        and reg['new_packet_decodes']==0 and reg['policy_selection'] is False,'Wrong CPU statistics scope')
    x=preflight(read(cfg['request']));require(reg['budget_before']==x['before'],'Registered budget differs')
    for p,s in x['sources'].items():require(reg['source_bindings'].get(p)==s,'Unregistered source')
    require(read(cfg['admission'])==x['admission'] and read(cfg['comparison_plan'])==x['plan'],'Admission/pairs changed after sealing')
    q=read(cfg['qualification']);require(q['status']==QUALIFIED and q['GPU_used'] is False and q['new_packet_decodes']==0
        and q['actual_quality_rows_read'] is False and q['python']==x['python'],'Actual CPU qualification missing')
    for name in ('source_bindings','input_bindings','outputs'):verify(q[name])
    for p,s in x['sources'].items():require(q['source_bindings'].get(p)==s,'Unqualified statistics source')
    require(len(q['results'])==1 and q['results'][0]['exit_code']==0 and q['results'][0]['test_count']==13,
        'The exact thirteen synthetic driver tests must pass')
    log=Path(cfg['qualification']).parent/'tests.log'
    require(q['outputs'].get(str(log))==sha(log) and 'Ran 13 tests' in log.read_text(encoding='utf-8')
        and log.read_text(encoding='utf-8').rstrip().endswith('OK'),'Unexpected comparison qualification output')
    require(owner['owner_out']==str(Path(cfg['owner_config']).parent) and owner['registration']==cfg['registration'],'Wrong owner paths')
    x['owner'].validate_config(owner,reg,sha(cfg['owner_config']))
    require(len(owner['stages'])==1 and owner['stages'][0]['id']=='report' and owner['stages'][0]['resource']=='cpu'
        and len(owner['stages'][0]['jobs'])==1,'Exactly one CPU report stage required')
    job=owner['stages'][0]['jobs'][0]
    require(job['id']==JOB and job['argv']==[x['python'],'-B',str(Path(__file__).absolute()),'--config',str(Path(config_path).absolute())]
        and job['out']==cfg['out'] and job['completion']==str(Path(cfg['out'])/'completion.json')
        and job['accepted_statuses']==[DONE] and job['receipt_expect']==EXPECTED,'Wrong CPU command or completion contract')
    x.update(cfg=cfg,reg=reg,execution_owner=owner,job=job);unchanged(x);return x


def live_owner(x):
    a=x['owner'];c=x['cfg'];own=x['execution_owner'];base=Path(own['owner_out']);ip=base/'owner_identity.json';v=read(ip)
    require(a.same_identity(v,a.identity(os.getppid())) and v['registration_sha256']==sha(c['registration'])
        and v['config_sha256']==sha(c['owner_config']) and not(base/'failure.json').exists(),'Current registered CPU owner absent')
    lp=base/'stages/report/workers'/JOB/'launch.json';start=time.monotonic()
    while not lp.exists():
        require(time.monotonic()-start<10 and a.same_identity(v,a.identity(os.getppid())),'Worker launch not sealed');time.sleep(.1)
    launch=read(lp)
    require(a.same_identity(launch['identity'],a.identity(os.getpid())) and launch['argv']==x['job']['argv']
        and launch['registration_sha256']==sha(c['registration']) and launch['resource']=='cpu' and launch['threads']==2
        and set(launch['affinity'])==set(own['cpu_affinities'][0])==os.sched_getaffinity(0)
        and os.getpriority(os.PRIO_PROCESS,0)==15,'Current CPU worker identity/resources differ')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only two-thread environment required')
    return bind([ip,lp])


def calculate(h,p,admission,plan):
    targets=n.bridge_targets(h,p);view=n.normalize(h,p,targets,admission)
    require(len(view['metrics'])==18 and len(plan['comparisons'])==18,'Fixed18 metrics/pairs required')
    for row in plan['comparisons']:
        require(view['points'][row['method']]['snr_db']==view['points'][row['reference']]['snr_db']==row['snr_db'],
            'Loaded point differs from predeclared same-SNR pair')
    result=statistics.summarize(view['rows'],view['source_ids'],view['points'],view['metrics'],
        [(r['method'],r['reference']) for r in plan['comparisons']])
    labels={(r['method'],r['reference']):r for r in plan['comparisons']}
    for row in result['paired']:row.update({k:v for k,v in labels[row['method'],row['reference']].items() if k not in row})
    require(len(result['summary'])==360 and len(result['paired'])==324 and len(result['source_means'])==36000,
        'Full frozen H18/P2 output grid required')
    return view,result


def claim_output(out,registration_sha256):
    out=Path(out)
    require(not out.exists() or out.is_dir() and not out.is_symlink() and not any(out.iterdir()),'Prior attempt preserved; no retry')
    out.mkdir(parents=True,exist_ok=True)
    save(out/'attempt.json',dict(status='H_P_CPU_COMPARISON_STARTED',registration_sha256=registration_sha256))


def run(config_path):
    x=registered_context(config_path);supervision=live_owner(x);cfg=x['cfg'];out=Path(cfg['out']);started=time.monotonic()
    claim_output(out,sha(cfg['registration']))
    try:
        for name,value in [('common_metric_admission.json',x['admission']),('comparison_plan.json',x['plan'])]:save(out/name,value)
        # No quality row is loaded until actual normal closures and sealed metadata/pairs pass.
        req=x['request'];h=n.load_h(req['H_request'],req['H_aggregate']);p=load_closed_p(x['pctx'],x['pclosed'])
        view,result=calculate(h,p,x['admission'],x['plan']);unchanged(x)
        require(time.monotonic()-started<cfg['max_seconds'],'CPU comparison deadline exceeded')
        save(out/'target_pixel_bridge.json',view['target_bridge']);save(out/'metric_availability.json',view['availability'])
        save(out/'point_identity.json',view['points'])
        for name in ('summary','paired','source_means'):csv_file(out/(name+'.csv'),result[name])
        save(out/'normalization_proof.json',{k:v for k,v in view.items() if k not in ('rows','target_bridge','availability','points')})
        report=('## H与P的固定点开发集比较\n\n同一100源；H保留6201/6202/6203，P保留2001/2002/2003。先分别按源平均三噪声，再进行10000次源级配对bootstrap（种子2026100605）。18个H点各与同信噪比P比较，未选择H赢家。\n\n'
          '保留18项共同指标；P缺少MSE、ResNet概率和confidently_wrong，不补值。H的PSNR保留float64计算，P保留原float32标量；原神经指标批量差异在common_metric_admission.json明确登记。原图像素逐值一致，两支原始参考哈希同时保留。\n\n'
          'whole_system的H64-R仅为整尺度赢家；RAW_PARTIAL_SYSTEM两个点同样保留，不能把较弱的整尺度raw64称为最强raw64。区间为预先指定的描述性点级95%区间，未作多重比较校正。MAIN尚缺；此处不判断H总体成功，也不以开发集选择新的整尺度/部分尺度策略。在线运行成本需独立计时结果，不能由本次CPU统计耗时代替。\n')
        (out/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
        unchanged(x);outputs=bind(p for p in out.iterdir() if p.name not in ('attempt.json','completion.json'))
        allinputs=merge(x['bindings'],h['bindings'],p['bindings'],supervision,bind([cfg['request'],cfg['admission'],cfg['comparison_plan'],str(config_path)]))
        done=dict(status=DONE,**EXPECTED,registration_sha256=sha(cfg['registration']),config_sha256=sha(config_path),
            input_bindings=allinputs,source_bindings=x['sources'],outputs=outputs,budget_before=x['before'],budget_after=x['before'],
            elapsed_seconds=time.monotonic()-started,source_ids=view['source_ids'],noise_seeds={'H':list(n.NOISE['H']),'P':list(n.NOISE['P'])},
            exact_original_pixel_pairs=100,original_rows_modified=False,common_metric_admission_sha256=sha(cfg['admission']),
            comparison_plan_sha256=sha(cfg['comparison_plan']),own_owner_success_not_yet_certified=True,
            final_system_conclusion='UNRESOLVED_MAIN_NOT_COMPLETE',multiple_comparisons=result['multiple_comparisons'])
        verify(allinputs);save(out/'completion.json',done);return done
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    done=run(args.config);print(json.dumps(dict(status=done['status'],source_count=done['source_count'],comparison_count=18)))
