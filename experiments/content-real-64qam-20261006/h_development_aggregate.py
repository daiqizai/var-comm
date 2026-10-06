"""CPU aggregation of the closed H18 development scores; no selection or models.

This standalone, single-attempt entry requires the normal metric owner to have
exited. It cannot admit P/MAIN, run a decoder or declare the H success criterion.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h_development_statistics as statistics

STATISTICS_SHA='674acc273a86a8c544ed4623d971a5f6ead5ba92e81fdc52080b1fc30295f2f6'
METRIC_DRIVER_SHA='24ccbe493c2e44793fd87324cfbece2d4051239ba741bf4129dea48ea0ba37d5'
DONE='H18_DEVELOPMENT_DESCRIPTIVE_AGGREGATION_COMPLETE'
ARMS=('H16-R','H16-A','H64-R','H64-A')
ROLES=('H_WHOLE_SYSTEM','H_RAW_PARTIAL_SYSTEM','H_FIXED_M7_ATTRIBUTION')
FINAL=('RAW_SOURCE_DECODED','ARITHMETIC_SOURCE_DECODED','WIRE_REJECT_GRAY','ARITHMETIC_SOURCE_INVALID_GRAY')
METRICS=('mse','psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine',
    'dists','dreamsim','ms_ssim','resnet50_top1_label','resnet50_top1_source_prediction',
    'resnet50_top1_probability','semantic_error','confidently_wrong','convnext_top1_label',
    'convnext_top1_source_prediction','convnext_source_prediction_agreement',
    'convnext_source_correct_to_wrong','convnext_source_wrong_to_correct','dino_mismatched','dino_specificity')
TX_KEYS=('tx_target_m','tx_actual_m','tx_actual_K','tx_source_mode','tx_fell_back','tx_profile_id')
RX_KEYS=('source_status','rx_accepted_m','rx_accepted_K','rx_declared_profile_id','gray','rx_canonical_accepted')


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def verify(mapping):
    for p,s in mapping.items():
        require(Path(p).is_absolute() and len(s)==64 and sha(p)==s,'Bound input changed: '+p)
def bind(paths):return {str(Path(p).absolute()):sha(p) for p in paths}
def merge(*maps):
    result={}
    for values in maps:
        for p,s in values.items():
            require(p not in result or result[p]==s,'Conflicting binding: '+p);result[p]=s
    return result
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def write_csv(path,rows):
    require(rows,'No empty table may silently stand for complete results')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def comparison_plan(schedule):
    """No measured row is an input. Select only identities fixed by calibration."""
    require(len(schedule)==18 and {s['development_slot'] for s in schedule}==set(range(18)), 'Frozen18 required')
    catalog=[];keys={};points={}
    for s in sorted(schedule,key=lambda x:x['development_slot']):
        slot=s['development_slot'];c=s['candidate'];role=s['role'];snr=c['snr_db'];arm=c['arm']
        require(s['status']=='FROZEN_POLICY_READY' and s['phase']=='development' and snr in (13,19), 'Unready development policy')
        expected=ROLES[0] if slot<8 else ROLES[1] if slot<10 else ROLES[2]
        require(role==expected,'Original slot role changed')
        if role!=ROLES[1]:require(arm in ARMS,'Unexpected whole/control arm')
        key=(role,snr,arm if role!=ROLES[1] else 'partial')
        require(key not in keys,'Duplicate arm/SNR identity')
        p=f'H18_SLOT_{slot:02d}';keys[key]=p
        points[p]=dict(snr_db=snr,role=role,candidate_id=c['candidate_id'])
        catalog.append(dict(point_id=p,development_slot=slot,arm=arm,**points[p]))
    comparisons=[]
    def add(kind,method,reference):
        comparisons.append(dict(comparison_id=f'{kind}:{method}-minus-{reference}',kind=kind,
            method=method,reference=reference,snr_db=points[method]['snr_db']))
    for snr in (13,19):
        for role in (ROLES[0],ROLES[2]):
            for a,b in (('H16-A','H16-R'),('H64-A','H64-R'),('H64-R','H16-R'),('H64-A','H16-A')):
                add('FACTOR_WHOLE' if role==ROLES[0] else 'FACTOR_FIXED_M7',keys[role,snr,a],keys[role,snr,b])
        for arm in ARMS:add('WHOLE_MINUS_OWN_FIXED_M7',keys[ROLES[0],snr,arm],keys[ROLES[2],snr,arm])
        add('PARTIAL_MINUS_RAW64_WHOLE',keys[ROLES[1],snr,'partial'],keys[ROLES[0],snr,'H64-R'])
    require(len(comparisons)==26,'Predetermined26 comparisons required')
    return dict(status='FROZEN_FROM_DECLARED_SCHEDULE_BEFORE_METRIC_ROWS',points=points,catalog=catalog,
        comparisons=comparisons,selection=False,multiplicity='Exploratory descriptive contrasts; unadjusted percentile95 intervals.',
        registered_H_success_evaluated=False)


def fallback_tables(rows,source_ids,plan):
    # Full coverage validation must precede deduplication of source-side choices.
    statistics.validate(rows,source_ids,plan['points'],('psnr_db',))
    by=defaultdict(list)
    for r in rows:by[r['point_id']].append(r)
    txrows=[];rxrows=[];failure=[];conditional=[];wire=[]
    for p,group in by.items():
        source_groups=defaultdict(list)
        for r in group:source_groups[r['source_index']].append(r)
        tx=Counter()
        for i,items in source_groups.items():
            choices={tuple(r[k] for k in TX_KEYS) for r in items}
            require(len(choices)==1,'Source payload changed across paired noises')
            value=next(iter(choices));require(type(value[0]) is int and type(value[1]) is int and type(value[2]) is int
                and 1<=value[1]<=value[0]<=10 and value[2]>=0 and type(value[4]) is bool
                and value[4]==(value[1]<value[0]),'Invalid transmitted scale/fallback')
            if plan['points'][p]['role']==ROLES[2]:
                require(value[0:3]==(7,7,0) and value[4] is False and all(r['nominal_rate']=='1/2' for r in items),'Fixedm7 control changed')
            tx[value]+=1
        rx=Counter();statuses=Counter()
        for r in group:
            require(r['source_status'] in FINAL and r['rx_summary']['source_decode_complete'] is True,'Unresolved receiver state cannot become gray')
            accepted=r['source_status'] in FINAL[:2]
            require(type(r['gray']) is bool and r['gray']==(not accepted) and r['rx_canonical_accepted']==accepted,
                'Actual accepted/gray identity differs')
            if accepted:require(type(r['rx_accepted_m']) is int and type(r['rx_accepted_K']) is int,'Accepted RX scale absent')
            else:require(r['rx_accepted_m'] is None and r['rx_accepted_K'] is None,'Rejected frame cannot claim accepted source scale')
            rx[tuple(r[k] for k in RX_KEYS)]+=1;statuses[r['source_status']]+=1
        identity=dict(point_id=p,**plan['points'][p])
        for v,n in sorted(tx.items(),key=lambda x:repr(x[0])):
            txrows.append(dict(identity,**dict(zip(TX_KEYS,v)),count=n,denominator=100,unit='source, one source choice before channel noise'))
        for v,n in sorted(rx.items(),key=lambda x:repr(x[0])):
            rxrows.append(dict(identity,**dict(zip(RX_KEYS,v)),count=n,denominator=300,unit='actual received frame, all failures retained'))
        for status in FINAL:failure.append(dict(identity,source_status=status,count=statuses[status],denominator=300))
        for field in ('header_correct','parsed_wire_matches_transmission','accepted_wire_mismatch',
                      'canonical_decode_invalid','reconstructed_after_accepted_wire_mismatch'):
            values=[r['evaluation_only'][field] for r in group]
            require(all(type(v) is bool for v in values),'Exact post-RX diagnostic booleans required')
            wire.append(dict(identity,diagnostic=field,count=sum(values),denominator=300,
                scope='POST_RX_WIRE_DIAGNOSTIC_NOT_SEMANTIC_CORRECTNESS'))
        for classifier in ('resnet50','convnext'):
            for original_correct in (True,False):
                eligible=[r for r in group if (r[classifier+'_source_prediction']==r['true_class'])==original_correct]
                n=sum((r[classifier+'_prediction']!=r['true_class']) if original_correct else
                      (r[classifier+'_prediction']==r['true_class']) for r in eligible)
                conditional.append(dict(identity,classifier=classifier,transition='source_correct_to_wrong' if original_correct else 'source_wrong_to_correct',
                    count=n,denominator=len(eligible),source_denominator=len(eligible)//3,
                    fraction=n/len(eligible) if eligible else None,status='MEASURED' if eligible else 'NO_ELIGIBLE_ORIGINAL_SOURCES'))
    return dict(tx_source_scale=txrows,rx_received_scale=rxrows,receiver_status=failure,classifier_transition=conditional,wire_diagnostics=wire)


def summarize(rows,source_ids,plan,metrics=METRICS):
    result=statistics.summarize(rows,source_ids,plan['points'],metrics,[(c['method'],c['reference']) for c in plan['comparisons']])
    labels={(c['method'],c['reference']):c for c in plan['comparisons']}
    for row in result['paired']:
        c=labels[row['method'],row['reference']];row.update(comparison_id=c['comparison_id'],comparison_kind=c['kind'],
            interpretation='DESCRIPTIVE_EXPLORATORY_NOT_H_SUCCESS_TEST')
    result.update(fallback_tables(rows,source_ids,plan));return result


def normal_metrics_closed(request):
    verify(request['input_bindings']);verify(request['source_bindings'])
    for path in (str(Path(__file__).absolute()),str(Path(statistics.__file__).absolute()),request['metric_driver']):
        require(request['source_bindings'].get(path)==sha(path),'Aggregator or frozen source unbound')
    require(sha(statistics.__file__)==STATISTICS_SHA and sha(request['metric_driver'])==METRIC_DRIVER_SHA,'Frozen statistic/metric source changed')
    spec=request['metric_batch']
    require(set(spec)=={'config','owner_config','registration','launch','completion'},'Exact normal metric batch required')
    for path in spec.values():require(request['input_bindings'].get(path)==sha(path),'Metric predecessor not pinned')
    driver=module(request['metric_driver'],'h18_aggregation_frozen_metrics')
    ctx=driver.load_registered(spec['config']);cfg=ctx['cfg']
    require(cfg['registration']==spec['registration'] and cfg['visual_owner_config']==spec['owner_config'], 'Metric owner lineage changed')
    expected=dict(status=driver.DONE,source_count=100,frame_count=5400,H_policy_snr_points=18,MAIN_frames=0,
        new_packet_decodes=0,policy_selection=False,development_used=True,holdout_used=False,
        unified_neural_metrics_run=True,statistical_aggregation_run=False,H_full_delivery_claimed=False,
        overall_development_complete=False,P_complete=False,MAIN_complete=False,online_timing_measured=False)
    closed=ctx['cpu'].closed_batch(ctx['owner'],ctx['wait'],dict(config=spec['owner_config'],registration=spec['registration'],
        launch=spec['launch'],completion=spec['completion']),cfg['owner_module'],expected,ctx['owner'].raw_process_state)
    old=closed['owner'];done=closed['done']
    require(len(old['stages'])==1 and old['stages'][0]['id']=='development' and old['stages'][0]['resource']=='gpu'
        and len(old['stages'][0]['jobs'])==1,'One actual closed metric job required')
    job=old['stages'][0]['jobs'][0]
    require(job['id']=='development_metrics' and job['argv'][1:]==['-B',request['metric_driver'],'--config',spec['config']]
        and job['out']==cfg['out'],'Closed metric command changed')
    require(done['registration_sha256']==sha(spec['registration']) and done['config_sha256']==sha(spec['config'])
        and done['source_ids']==ctx['source_ids'] and done['noise_seeds']==list(statistics.SEEDS), 'Metric identity/population changed')
    driver.u.assert_budget(ctx['before'],done['budget_before'],ctx['group'])
    require(done['budget_after']==ctx['before']==closed['owner_done']['budget'],'Metric budget changed')
    verify(done['outputs']);verify(done['input_bindings']);verify(done['source_bindings'])
    return ctx,closed


def load_rows(ctx,closed):
    """Use JSON to preserve booleans/nulls; verify each source seal and wire trace."""
    cfg=ctx['cfg'];done=closed['done'];base=Path(cfg['out']);rows=[];seen={}
    for i,sid in enumerate(ctx['source_ids']):
        cp=base/'source_checkpoints'/f'{i:04d}.json';rp=base/'sources'/f'{i:04d}.json'
        for p in (cp,rp):require(done['outputs'].get(str(p))==sha(p),'Metric source output not sealed')
        c=read(cp)
        require(c['status']=='H_DEVELOPMENT_METRIC_SOURCE_COMPLETE' and c['source_index']==i and c['source_id']==sid
            and c['frame_count']==54 and c['registration_sha256']==sha(cfg['registration'])
            and c['metric_evaluator_identity']==done['metric_evaluator_identity'] and c['new_packet_decodes']==0,
            'Metric source checkpoint differs')
        verify(c['outputs']);verify(c['input_bindings']);require(c['outputs'].get(str(rp))==sha(rp),'Rows absent from checkpoint')
        values=read(rp);ctx['adapter'].validate_source_rows(values,i,sid,ctx['schedule'])
        trace=Path(ctx['group']['cfg']['out'])/f'worker_{i%2}'/'traces'/f'{i:04d}.json'
        require(c['input_bindings'].get(str(trace))==ctx['group']['done']['outputs'].get(str(trace))==sha(trace),'CPU trace not bound to scores')
        checked=ctx['adapter'].attach_wire_accounting(values,read(trace))
        require(checked==values,'Stored TX/RX accounting differs from actual wire')
        for r in values:
            require(r['metric_evaluator_identity']==done['metric_evaluator_identity'] and r['metric_batch_size']==1,'Mixed evaluator identity')
        rows.extend(values);seen.update(bind((cp,rp,trace)))
    p=base/'frame_metrics.json';require(done['outputs'].get(str(p))==sha(p),'Complete frame JSON not sealed')
    require(rows==read(p),'Flattened frame output differs from all100 checkpoints')
    return rows,merge(seen,bind((p,)))


def run(request_path):
    request_path=str(Path(request_path).absolute());r=read(request_path)
    require(r['schema']=='H18_DESCRIPTIVE_AGGREGATION_REQUEST_V1' and r['comparisons']=='FROZEN_26_FACTOR_CONTRASTS'
        and r['P_admitted'] is False and r['MAIN_admitted'] is False,'H-only request required')
    out=Path(r['out']);require(out.is_absolute() and not out.exists(),'New output required; prior attempts are immutable')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only two-thread environment required')
    ctx,closed=normal_metrics_closed(r)
    forbidden=[Path(ctx['cfg']['out']),Path(ctx['render_cfg']['out']),Path(ctx['group']['cfg']['out'])]
    require(all(out!=p and out not in p.parents and p not in out.parents for p in forbidden),'Output overlaps sealed science data')
    out.mkdir(parents=True);save(out/'attempt.json',dict(status='H18_CPU_AGGREGATION_ATTEMPT',request_sha256=sha(request_path)))
    start=time.monotonic()
    try:
        # This is deliberately persisted before load_rows touches any measured score.
        plan=comparison_plan(ctx['schedule']);save(out/'comparison_plan.json',plan)
        save(out/'frozen_schedule.json',dict(schedule=ctx['schedule'],source_ids=ctx['source_ids'],
            finalized_policy_sha256=sha(ctx['group']['cfg']['finalized']),source_bindings=bind((ctx['group']['cfg']['finalized'],))))
        rows,extra=load_rows(ctx,closed);result=summarize(rows,ctx['source_ids'],plan)
        write_csv(out/'point_catalog.csv',plan['catalog'])
        for name in ('summary','paired','source_means','tx_source_scale','rx_received_scale','receiver_status','classifier_transition','wire_diagnostics'):
            write_csv(out/(name+'.csv'),result[name])
        for role,name in zip(ROLES,('whole_system_summary','partial_system_summary','fixed_m7_summary')):
            write_csv(out/(name+'.csv'),[x for x in result['summary'] if x['role']==role])
        done=closed['done'];render=ctx['closed']['done'];costpath=Path(ctx['cfg']['out'])/'metric_cost_counts.json'
        require(done['outputs'].get(str(costpath))==sha(costpath),'Metric cost count file unbound')
        cost=dict(scope='ACTUAL_PREPARATION_AND_EVALUATION_STAGES_NOT_ONLINE',render_elapsed_seconds=render['elapsed_seconds'],
            metrics_elapsed_seconds=done['elapsed_seconds'],metric_cost_counts=read(costpath),
            online_TX='NOT_MEASURED',online_RX='NOT_MEASURED',exclusive_PHY='NOT_MEASURED',end_to_end_latency='NOT_MEASURED',
            new_packet_decodes=0)
        save(out/'cost_scope.json',cost)
        save(out/'metric_provenance.json',dict(metadata=done['metric_metadata'],evaluator_identity=done['metric_evaluator_identity'],
            numerical_runtime=done['numerical_runtime'],metrics=list(METRICS),prediction_indices_not_averaged=True,
            F_recovery_error='NOT_MEASURED_NO_REGISTERED_LATENT_RECOVERY_METRIC',FID='DEFERRED_HOLDOUT',KID='DEFERRED_HOLDOUT'))
        save(out/'missing_comparators.json',dict(P='MISSING_NOT_ADMITTED',MAIN='MISSING_NOT_ADMITTED',
            MAIN_required_points=[dict(modulation=q,snr_db=s) for q in ('QPSK','16QAM') for s in (13,19)],
            P_future_noise_seeds=[2001,2002,2003],H_noise_seeds=list(statistics.SEEDS),
            future_join='Independently averaged source means; never relabel noise seeds or claim common noise observations.',
            H_success='NOT_EVALUATED_WITHOUT_FULL_MAIN_AND_P_EVIDENCE',H_winner_selected=False))
        report='''# H18 开发集统计（仅 H）\n\n100 张原 development 图，每点使用 6201、6202、6203 三个噪声；18 个已冻结点全部保留。\n先对每源三个噪声取均值，再按源 bootstrap 10,000 次（seed 2026100605）。\n\n- 整尺度系统、部分尺度系统与固定 m7/K0/rate1/2 归因对照分别列表；不按开发集挑选 H 策略。\n- 26 个因子对照由冻结的 18 点名单事先生成；区间为描述性、探索性比较，未作多重比较校正。它们不替代原登记的 H 成功条件。\n- TX 尺度回退按每点 100 个源计数；RX 接收尺度与全部失败按每点 300 帧计数。拒收不从分母剔除。\n- P 与 MAIN 尚未纳入。本表不能宣称 H 优于完整主系统，也不能把 S1 桥接点当成完整 MAIN。\n- 成本表仅记录已完成生成/评测阶段的实耗时与缓存计数；在线 TX/RX、独占 PHY 及端到端耗时尚未测量。\n- 本批是开发集，未读取 holdout。分类转移另报源分类正确/错误各自的条件分母，空分母记缺项。\n'''
        (out/'REPORT.md').write_text(report,encoding='utf-8',newline='\n')
        inputs=merge(r['input_bindings'],ctx['bound'],closed['bindings'],done['outputs'],extra,bind((request_path,)))
        verify(inputs);verify(r['source_bindings'])
        after=ctx['owner'].budget_snapshot(ctx['cfg']['ledger'],sha(ctx['cfg']['budget_registration']),ctx['cfg']['phase_limits'],quiescent=True)
        require(after==ctx['before'],'No PHY budget change permitted during aggregation')
        outputs=bind(p for p in out.iterdir() if p.is_file() and p.name!='attempt.json')
        completion=dict(status=DONE,scope='H18_ONLY_DESCRIPTIVE_STATISTICS',source_count=100,frame_count=5400,points=18,
            noise_seeds=list(statistics.SEEDS),bootstrap_replicates=statistics.REPLICATES,bootstrap_seed=statistics.BOOTSTRAP_SEED,
            comparison_count=26,policy_selection=False,development_used=True,holdout_used=False,GPU_used=False,new_packet_decodes=0,
            P_complete=False,MAIN_complete=False,H_success_evaluated=False,H_full_delivery_claimed=False,
            overall_development_complete=False,online_latency_measured=False,budget_before=ctx['before'],budget_after=after,
            input_bindings=inputs,source_bindings=r['source_bindings'],outputs=outputs,elapsed_seconds=time.monotonic()-start)
        save(out/'completion.json',completion);return completion
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',request_sha256=sha(request_path),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    print(run(a.request)['status'])
if __name__=='__main__':main()
