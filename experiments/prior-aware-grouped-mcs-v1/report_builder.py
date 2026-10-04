"""Publish measured UEP tables and untouched-pixel fixed16 panels; CPU only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import math
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import source_quality as q
from score_actual import SNRS,SEEDS,write_csv

FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
HEADLINE=('psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine','dists',
          'convnext_source_prediction_agreement','F_sq_error_zero_erasure_proxy')
LABELS=dict(psnr_db='PSNR ↑',lpips_alex='LPIPS ↓',dinov2_vitl14_cosine='DINOv2-L ↑',
            clip_image_cosine='CLIP ↑',dists='DISTS ↓',convnext_source_prediction_agreement='ConvNeXt 一致率 ↑',
            F_sq_error_zero_erasure_proxy='F 误差/零擦除代理 ↓')


def read_csv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def admit_bundle(folder,*,actual=False):
    folder=Path(folder).resolve();completion=q.read(folder/'completion.json');registration=q.read(folder/'registration.json')
    q.require(completion['registration_sha256']==q.sha(folder/'registration.json'),'Report scoring registration differs')
    if actual:q.require(completion['status']=='UEP_ACTUAL_SCORING_COMPLETE' and completion['sources']==100,'Incomplete actual score bundle')
    outputs=completion['outputs']
    for path,digest in outputs.items():q.require(q.sha(path)==digest,'Scored output changed: '+path)
    checkpoints={};rows=[]
    for index in range(100):
        path=folder/'source_checkpoints'/('%04d.json'%index)
        if not path.exists():
            q.require(not actual,'Missing actual source checkpoint');continue
        q.require(outputs.get(str(path))==q.sha(path),'Unbound comparison source checkpoint')
        cp=q.read(path)
        q.require(cp['source_index']==index and cp['payload_sha256']==q.identity({k:v for k,v in cp.items() if k!='payload_sha256'}),
                  'Comparison source checkpoint seal differs')
        q.require(cp.get('binding')==q.identity(registration),'Comparison checkpoint registration differs')
        checkpoints[index]=cp;rows.extend(cp['rows'])
    return dict(folder=folder,completion=completion,registration=registration,checkpoints=checkpoints,rows=rows,
        bindings={str(folder/'completion.json'):q.sha(folder/'completion.json'),str(folder/'registration.json'):q.sha(folder/'registration.json'),**outputs})


def interval(values,draws):
    values=np.asarray(values,dtype=np.float64)
    q.require(values.shape==(100,) and np.isfinite(values).all(),'Complete 100-source vector required')
    low,high=np.quantile(values[draws].mean(1),[.025,.975])
    return dict(mean=float(values.mean()),ci_low=float(low),ci_high=float(high),sources=100,frames=300,
                bootstrap_unit='source_mean_over_three_noise_repeats',bootstrap_seed=20261002,bootstrap_replicates=10000)


def pair_p(actual,p_rows):
    """Only complete actually scored P points; never interpolate or impute."""
    p={};seen=set()
    for row in p_rows:
        if row.get('method') not in ('P','P1024') or int(row['N'])!=1024:continue
        key=(int(row['source_index']),int(row['snr_db']),int(row['noise_seed']))
        q.require(key not in seen,'Repeated P comparison frame');seen.add(key);p[key]=row
    groups=defaultdict(list)
    for row in actual:groups[int(row['snr_db']),row['method']].append(row)
    draws=np.random.default_rng(20261002).integers(0,100,(10000,100));paired=[];p_summary=[];availability=[]
    expected={(i,s) for i in range(100) for s in SEEDS}
    for snr in SNRS:
        by={(i,s):row for (i,n,s),row in p.items() if n==snr}
        complete=set(by)==expected
        availability.append(dict(N=1024,snr_db=snr,method='P1024',frames=len(by),complete_paired_population=complete,
                                 status='MEASURED_COMPLETE' if complete else 'MISSING_OR_INCOMPLETE_NOT_PAIRED'))
        if not complete:continue
        for metric in HEADLINE:
            if not all(row.get(metric) not in (None,'') and math.isfinite(float(row[metric])) for row in by.values()):continue
            values=np.asarray([float(by[i,s][metric]) for i in range(100) for s in SEEDS]).reshape(100,3).mean(1)
            p_summary.append(dict(N=1024,snr_db=snr,method='P1024',metric=metric,**interval(values,draws)))
            for (n,method),rows in groups.items():
                if n!=snr:continue
                a={(int(r['source_index']),int(r['noise_seed'])):r for r in rows}
                q.require(set(a)==expected,'Incomplete actual method for P comparison')
                for key in expected:
                    q.require(a[key]['source_id']==by[key]['source_id'] and a[key]['preprocessing_id']==by[key]['preprocessing_id']
                              and a[key]['reference_sha256']==by[key]['reference_sha256'],'P comparison reference differs')
                if not all(r.get(metric) not in (None,'') for r in a.values()):continue
                delta=np.asarray([float(a[i,s][metric])-float(by[i,s][metric]) for i in range(100) for s in SEEDS]).reshape(100,3).mean(1)
                paired.append(dict(N=1024,snr_db=snr,method_A=method,method_B='P1024',metric=metric,delta='A_minus_B',
                    pairing='same source and registered noise seed; original P and UEP channel noise namespaces differ',**interval(delta,draws)))
    return paired,p_summary,availability


def load_images(checkpoint):
    proof=checkpoint['float_reconstructions'];q.require(q.sha(proof['path'])==proof['sha256'],'Fixed-example float archive changed')
    with np.load(proof['path'],allow_pickle=False) as archive:
        q.require(set(archive.files)=={'images','source_rgb','row_ids','image_slots'},'Fixed-example archive schema differs')
        images=archive['images'].copy();source=archive['source_rgb'].copy();slots=archive['image_slots'].tolist();ids=archive['row_ids'].tolist()
    expected=[r.get('row_id',r.get('replay_row_id')) for r in checkpoint['rows']]
    q.require(ids==expected and len(slots)==len(expected),'Fixed-example row mapping differs')
    reference=q.rgb_sha(source);mapping={}
    for row,slot in zip(checkpoint['rows'],slots):
        q.require(type(slot)is int and 0<=slot<len(images) and row['image_sha256']==q.rgb_sha(images[slot])
                  and row['reference_sha256']==reference,'Fixed-example image/reference association differs')
        mapping[int(row['snr_db']),int(row['noise_seed']),row['method']]=(images[slot],row)
    return source,mapping


def display_rgb(image):
    q.rgb_sha(image)
    # Only the unavoidable float-to-PNG quantization; no resizing, enhancement,
    # spatial crop, sharpening or content modification.
    return np.clip(np.rint(image.transpose(1,2,0)*255),0,255).astype(np.uint8)


def make_panel(examples,methods,path,title):
    """Paste each measured 256x256 display image at native resolution."""
    tile=256;gap=12;top=62;bottom=34;width=gap+(tile+gap)*len(methods);height=top+(tile+bottom)*len(examples)+gap
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.load_default(size=18)
    except TypeError:
        try:font=ImageFont.truetype('DejaVuSans.ttf',18)
        except OSError:font=ImageFont.load_default()
    draw.text((gap,5),title,fill='black',font=font)
    for column,method in enumerate(methods):draw.text((gap+column*(tile+gap),32),method,fill='black',font=font)
    used=[]
    for row,example in enumerate(examples):
        y=top+row*(tile+bottom)
        for column,method in enumerate(methods):
            x=gap+column*(tile+gap);image=example['images'].get(method)
            if image is None:
                draw.rectangle((x,y,x+tile-1,y+tile-1),fill='#eeeeee');draw.text((x+40,y+110),'NOT MEASURED',fill='black',font=font)
            else:
                canvas.paste(Image.fromarray(display_rgb(image)),(x,y));used.append(dict(source_index=example['source_index'],method=method,image_sha256=q.rgb_sha(image),box=[x,y,x+tile,y+tile]))
            draw.text((x,y+tile+4),'source %02d'%example['source_index'],fill='black',font=font)
    Path(path).parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    with Image.open(path) as final:
        pixels=np.asarray(final.convert('RGB'))
        for cell in used:
            example=next(v for v in examples if v['source_index']==cell['source_index']);x0,y0,x1,y1=cell['box']
            q.require(np.array_equal(pixels[y0:y1,x0:x1],display_rgb(example['images'][cell['method']])), 'Saved panel changed image pixels')
    q.require(Path(path).stat().st_size<10_000_000,'Publication panel reaches the 10,000,000-byte repository limit')
    return dict(path=str(Path(path).resolve()),sha256=q.sha(path),cells=used,native_resolution=True,noise_seed=2001)


def fixed_examples(actual,pbundle,fixed,result):
    q.require(fixed['status']=='FROZEN_FIXED_EXAMPLES' and fixed['source_indices']==list(FIXED),'Original fixed16 selection differs')
    records={r['source_index']:r for r in fixed['records']};sources=actual['registration']['source_identity'];panels=[];originals=[]
    q.require(set(records)==set(FIXED),'Fixed example record set differs')
    for index in FIXED:
        expected=sources[index];record=records[index]
        for key in ('source_id','preprocessing_id'):
            alias='image_id' if key=='source_id' and 'source_id' not in record else key
            q.require(record[alias]==expected[key],'Fixed example identity differs from scored sources')
    cached={}
    for index in FIXED:
        source,images=load_images(actual['checkpoints'][index]);pimages={}
        if pbundle and index in pbundle['checkpoints']:
            psource,pimages=load_images(pbundle['checkpoints'][index]);q.require(np.array_equal(source,psource),'P fixed-example reference differs')
        cached[index]=(source,images,pimages)
        path=result/'originals'/('%04d.png'%index);path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(display_rgb(source)).save(path);originals.append(dict(source_index=index,path=str(path),sha256=q.sha(path),float_reference_sha256=q.rgb_sha(source)))
    for snr in SNRS:
        for kind,methods in [('main',('Original','P1024','B0','B1','B2','B3','B4')),
                             ('matched',('Original','P1024','B3','matched_B1','matched_B2','matched_B3'))]:
            for page in range(4):
                examples=[]
                for index in FIXED[page*4:(page+1)*4]:
                    source,images,pimages=cached[index];rendered={'Original':source}
                    for method in methods[1:]:
                        values=(pimages.get((snr,2001,'P1024')) or pimages.get((snr,2001,'P'))) if method=='P1024' else images.get((snr,2001,method))
                        if values is not None:rendered[method]=values[0]
                    examples.append(dict(source_index=index,images=rendered))
                path=result/'figures'/('%s_N1024_snr%02d_page%d.png'%(kind,snr,page+1))
                panels.append(make_panel(examples,methods,path,'N1024 | %d dB | noise 2001 | %s'%(snr,kind)))
    return panels,originals


def fmt(value):
    return '—' if value in (None,'') else '%.4f'%float(value)


def markdown_table(rows,columns):
    answer=['| '+' | '.join(label for _,label in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:answer.append('| '+' | '.join(str(row.get(key,'—')).replace('|','/') for key,_ in columns)+' |')
    return '\n'.join(answer)


def publish_source_tables(scored,result):
    """Expand the full row table through the repository-safe chunked publisher."""
    from release_tables import copy_light
    copy_light(Path(scored)/'metrics_per_frame.csv.gz',Path(result)/'metrics_per_frame.csv')
    for source in ('convnext_summary.csv','convnext_transitions.csv','development_source_quality.csv'):
        shutil.copyfile(Path(scored)/source,Path(result)/source)


def calibration_scope(scored,policies):
    """Bind the pre-development cost decision and disclose any frozen screen."""
    parent=Path(scored).resolve().parent;path=parent/'stage_a_cost_decision.json'
    q.require(policies['source_count']==1000 and policies['development_read'] is False,'Final1000 calibration policy required')
    if not path.exists():
        return dict(status='COST_DECISION_NOT_PRESENT',final_calibration_sources=1000,
            full_grid_optimum_claimed=False),{},'最终策略使用完整1000张校准图。未提供阶段A成本决策记录，本报告不声称完整网格最优。'
    cost=q.read(path);pilot=parent/'quality_pilot'/'benchmark.json'
    q.require(type(cost['screen300']) is bool and cost['development_read'] is False
              and cost['quality_pilot_sha256']==q.sha(pilot),'Stage A cost decision or pilot binding differs')
    from artifact_bridge import combined_cost_rule
    expected=combined_cost_rule(cost['Q_hours'],cost['coarse_remaining_hours'],cost['refine_hours'],cost['actual_hours'],cost['report_hours'])
    q.require(all(cost[k]==v for k,v in expected.items()),'Stage A cost decision contradicts the frozen rule')
    q.require(q.read(pilot)['estimated_full1000_source_Q_hours']==cost['Q_hours'],'Stage A measured Q estimate differs')
    bindings={str(path):q.sha(path),str(pilot):q.sha(pilot)}
    scope=dict(status='CALIBRATION_SCOPE_VERIFIED',final_calibration_sources=1000,screen300=cost['screen300'],
        stage_a_cost_decision=cost,full_grid_optimum_claimed=False)
    if cost['screen300']:
        shortlist=parent/'shortlist_profiles.json';receipt_path=parent/'shortlist_profiles.json.receipt.json'
        screen=parent/'screen300_policies.json';receipt=q.read(receipt_path);profiles=q.read(shortlist)
        q.require(receipt['status']=='FIXED300_SHORTLIST_FROZEN' and receipt['source_count']==300
                  and receipt['development_read'] is False and receipt['shortlist_sha256']==q.sha(shortlist)
                  and receipt['screen_sha256']==q.sha(screen) and receipt['candidates']==len(profiles)
                  and policies['profile_input_sha256']==q.identity(profiles), 'Frozen shortlist is not the final policy candidate set')
        screen_doc=q.read(screen)
        q.require(screen_doc['source_count']==300 and screen_doc['stage']=='screen' and screen_doc['development_read'] is False
                  and screen_doc['synthetic'] is False,'Initial fixed300 screening receipt differs')
        bindings.update({str(p):q.sha(p) for p in (shortlist,receipt_path,screen)})
        scope.update(screen_calibration_sources=300,final_candidate_count=len(profiles),shortlist_receipt=receipt,
                     claim='Full1000 optimum over the pre-frozen shortlist only')
        description=('阶段A预计总成本 %.2f 小时超过登记的48小时阈值，先用原序前300张校准图完成全候选初筛；'
            '按预先登记规则冻结 top3、强基线、独立验证邻居及匹配对照，共%d个候选，再在完整1000张校准图上复核。'
            '本轮选策最优仅指这份预先冻结 shortlist 内的最优，不能称为完整1000张上的全网格最优。')%(cost['estimated_total_hours'],len(profiles))
    else:
        scope.update(claim='Final1000 selection within registered candidates')
        description=('阶段A预计总成本 %.2f 小时未超过登记的48小时阈值，直接在完整1000张校准图上评估登记候选集合，'
                     '未触发300张初筛。')%cost['estimated_total_hours']
    description+=' 成本规则为 max(Q实测估计,剩余粗BLER估计)+精化+实链路+报告；规划余量不是已经发生的耗时。'
    return scope,bindings,description


def build(root,scored,policies_path,validation_path,gate_path,fixed_path,timing_path,*,p_scored=None):
    root=Path(root).resolve();actual=admit_bundle(scored,actual=True)
    p=admit_bundle(p_scored) if p_scored else None
    policies=q.read(policies_path);validation_doc=q.read(validation_path);gate=q.read(gate_path);fixed=q.read(fixed_path)
    scope,scope_bindings,scope_text=calibration_scope(scored,policies)
    timing=q.read(timing_path);timing_regpath=Path(timing_path).parent/'registration.json';timing_reg=q.read(timing_regpath)
    q.require(timing['status']=='UEP_UNCACHED_RECEIVER_COST_COMPLETE' and timing['source_indices']==list(FIXED)
              and timing['uncached_image_receiver'] is True and timing['uncached_TX_entropy'] is True
              and timing['registration_sha256']==q.sha(timing_regpath)
              and timing_reg['model_identity']==actual['registration']['visual_identity'], 'Current frozen uncached receiver/TX timing is required')
    q.require(timing['input_bindings'].get(str(Path(policies_path).resolve()))==q.sha(policies_path), 'Timing uses a different selected policy')
    for path,digest in timing['outputs'].items():q.require(q.sha(path)==digest,'Timing checkpoint changed')
    q.require(actual['registration']['selected_policies_sha256']==q.sha(policies_path),'Report policies differ from scored policies')
    result=root/'results/prior_aware_uep_20261004';result.mkdir(parents=True,exist_ok=True)
    report=root/'reports/uep_prior_aware_20261004.md';report.parent.mkdir(parents=True,exist_ok=True)
    inputs=dict(actual['bindings']);inputs.update(p['bindings'] if p else {})
    inputs.update({str(Path(path).resolve()):q.sha(path) for path in (policies_path,validation_path,gate_path,fixed_path,timing_path,timing_regpath)})
    inputs.update(timing['outputs'])
    inputs.update(scope_bindings)
    inputs[str(Path(__file__).resolve())]=q.sha(__file__)
    main=read_csv(Path(scored)/'metrics_summary.csv');paired=read_csv(Path(scored)/'metrics_paired.csv')
    against_p,p_summary,p_availability=pair_p(actual['rows'],p['rows'] if p else []);main+=p_summary
    write_csv(result/'complete_main_table.csv',main);write_csv(result/'paired_method_intervals.csv',paired)
    write_csv(result/'paired_against_P1024.csv',against_p);write_csv(result/'P1024_coverage.csv',p_availability)
    publish_source_tables(scored,result)
    panels,originals=fixed_examples(actual,p,fixed,result)
    q.write(result/'figures_manifest.json',dict(source_indices=list(FIXED),noise_seed=2001,panels=panels,originals=originals,
                                               image_source='original saved float32 RGB, display quantization only',input_bindings=inputs))
    compact_policies={k:v for k,v in policies.items() if k!='per_source_predictions'}
    compact_policies['omitted_large_per_source_prediction_rows']=len(policies.get('per_source_predictions',[]))
    compact_policies['full_policy_path']=str(Path(policies_path).resolve());compact_policies['full_policy_sha256']=q.sha(policies_path)
    q.write(result/'selected_policies.json',compact_policies);q.write(result/'model_validation.json',validation_doc);q.write(result/'extension_gate.json',gate)
    q.write(result/'timing.json',timing);write_csv(result/'uncached_timing_summary.csv',timing['summary'])
    q.write(result/'calibration_scope.json',scope)
    q.write(result/'model_metadata.json',dict(visual=actual['registration']['visual_identity'],
        metrics=actual['registration']['metric_identity'],independent_classifier=actual['registration']['independent_classifier'],
        current_study_selection_usage='DINOv2 ViT-L/14 is the UEP calibration objective; other scores do not select/refine profiles; ConvNeXt is independent development validation',
        historical_metadata_note='The unchanged original evaluator metadata describes its historical evaluation-only use; current UEP usage is explicitly overridden above'))
    long={(int(r['snr_db']),r['method'],r['metric']):r for r in main};table=[]
    for snr in SNRS:
        for method in ('P1024','B0','B1','B2','B3','B4'):
            row=dict(snr=snr,method=method)
            for metric in HEADLINE:row[metric]=fmt(long.get((snr,method,metric),{}).get('mean'))
            table.append(row)
    matched=[]
    for snr in SNRS:
        for method in ('B3','matched_B1','matched_B2','matched_B3'):
            matched.append(dict(snr=snr,method=method,**{m:fmt(long.get((snr,method,m),{}).get('mean')) for m in HEADLINE[:3]}))
    policy_rows=[]
    for cell in policies['cells']:
        if cell['N']!=1024 or cell['snr_db'] not in SNRS:continue
        for family,item in cell['independent_optima'].items():
            if item['status']!='CALIBRATION_SELECTED':
                policy_rows.append(dict(snr=cell['snr_db'],method=family,status=item['status']));continue
            profile=item['selected']['profile']
            policy_rows.append(dict(snr=cell['snr_db'],method=family,status=item['status'],G=profile['G'],m=profile['m'],K=profile['K'],j=profile['j'],
                group_MCS='/'.join(g['modulation']+' r'+str(g['nominal_rate']) for g in profile['groups']),stable_id=profile['stable_id']))
    write_csv(result/'selected_policy_table.csv',policy_rows)
    costs=[dict(source_index=i,physical_frames=cp['physical_frames'],unique_images=cp['unique_images'],
        render_total_seconds=cp['render_total_seconds'],metrics_total_seconds=cp['metrics_total_seconds'],
        source_total_seconds=cp['source_total_seconds']) for i,cp in actual['checkpoints'].items()]
    write_csv(result/'offline_costs.csv',costs)
    costs_by={(int(r['snr_db']),r['method'],r['component']):r for r in timing['summary']}
    timing_table=[dict(snr=snr,method=method,
        TX_ms=fmt(costs_by.get((snr,method,'TX_entropy'),{}).get('mean_ms')),
        RX_ms=fmt(costs_by.get((snr,method,'RX_image'),{}).get('mean_ms')),
        gray_frames=costs_by.get((snr,method,'RX_image'),{}).get('gray_frames','—'))
        for snr in SNRS for method in ('B0','B1','B2','B3','B4')]
    bullets=[]
    for metric in ('dinov2_vitl14_cosine','convnext_source_prediction_agreement','psnr_db','lpips_alex'):
        values=[r for r in paired if r['method_A']=='B3' and r['method_B']=='B0' and r['metric']==metric]
        if values:
            bullets.append('- '+LABELS.get(metric,metric)+'，B3−B0：'+ '；'.join('%s dB %s [%s, %s]'%(r['snr_db'],fmt(r['mean']),fmt(r['ci_low']),fmt(r['ci_high'])) for r in values)+'。')
    text=['# 先验感知 UEP：冻结 N1024 实链路结果','',
        '校准选策与开发集评测已分离。主表使用原 100 张 development 图像、每图 3 个登记噪声种子，在 4/7/10/13 dB 完成实际付费码流解码；失败帧也计入均值。',
        '', '## 校准样本与候选范围','',scope_text,
        '', '## 主要结果','',*bullets,'',
        '**门槛结果：** `'+str(gate.get('status','未提供结论'))+'`。本结果来自已多次使用的 development 集，门槛不是独立 holdout 确认。',
        '', '## 完整主表摘要','',markdown_table(table,[('snr','SNR/dB'),('method','方法')]+[(m,LABELS[m]) for m in HEADLINE]),
        '', '全部已算指标及配对区间见 [完整主表](../results/prior_aware_uep_20261004/complete_main_table.csv) 与 [方法配对区间](../results/prior_aware_uep_20261004/paired_method_intervals.csv)。空白表示该点或指标未完成测量，不插值、不以别的 SNR 代替。',
        '', '## 相同源信息与分组边界的对照','',
        markdown_table(matched,[('snr','SNR/dB'),('method','方法')]+[(m,LABELS[m]) for m in HEADLINE[:3]]),
        '', 'matched_B1/B2/B3 保持 B3 的 (m,K,j)。B3 退化为单组时，这组对照标为不适用；同一波形或旁路复用不算独立方法增益。B4 根据 direct Q 选策，但最终仍用同一个 VAR 接收器。',
        '', '## 对 P1024 的取舍','',
        '[P1024 覆盖清单](../results/prior_aware_uep_20261004/P1024_coverage.csv) 与 [完整源配对差值](../results/prior_aware_uep_20261004/paired_against_P1024.csv)。仅完整 100×3 的同源点进入配对表。P 使用原信道噪声命名空间，UEP 使用新实链路噪声命名空间；相同种子编号不代表噪声向量逐值相同。',
        '', '## 功率、失败与指标口径','',
        '- 全部实际 N=1024；68 次 QPSK 头、正文 CRC/LDPC/rate matching 与已知填充符号均计费。纯 QPSK 逐帧 E=2048；含 16QAM 的方案按固定星座平均 Es=2，逐帧实际能量另存，不能宣称与 QPSK 严格逐帧同能量。',
        '- 头或第一组失败输出 RGB0.5；第二组失败只用实际接受的第一组前缀。CRC 误接受按实际错误 token 重建，不替换成干净前缀。所有非灰图由同一冻结 Dc 解码。',
        '- DINOv2 ViT-L/14（无 register，1024维 CLS）是本轮校准选择目标；原 DINOv2 ViT-S/14 保留。OpenAI CLIP ViT-L/14（224px）的图像余弦、LPIPS-Alex、DISTS、DreamSim ensemble v0.2.0、MS-SSIM、ResNet-50 两种准确率与错配检查仍报。独立 ConvNeXt-Tiny IMAGENET1K_V1 仅用于冻结策略后的开发评测，未参与候选精化。',
        '- F 误差为 32×16×16 latent 的平方误差和。无接收 latent 的灰图，条件 F 误差留空；全帧代理用零擦除的 F 误差明确计入，不能把该代理称为灰图的真实 latent。',
        '- 置信区间先对每源三个噪声求均值，再按源配对 bootstrap 10,000 次，种子 20261002。未为寻求显著性追加样本。',
        '', '## 预测模型验证与接收代价','',
        '[模型验证](../results/prior_aware_uep_20261004/model_validation.json)、[扩展门槛](../results/prior_aware_uep_20261004/extension_gate.json) 与 [离线总耗时](../results/prior_aware_uep_20261004/offline_costs.csv)。开发 clean Q 只评估最终冻结码本需要的状态，用来区分源分布变化和链路预测误差，不回流选策。',
        '', '### 本轮无最终图像缓存的实测代价','',
        markdown_table(timing_table,[('snr','SNR/dB'),('method','方法'),('TX_ms','TX 熵序/ms'),('RX_ms','接收图像/ms'),('gray_frames','16帧中灰图数')]),
        '', '固定16张、noise2001；每个实际独立波形先热身1次，再正式3次，CUDA同步，先对重复求均值。正式接收每次清空最终图像与KV缓存；重复方法共享同一波形的测量记录。TX仅含已知粗token前缀到完整熵排序，K0为不需要；图像接收从实际CRC接受payload到CPU RGB，含VAR与Dc，指标计算排除在计时外。',
        '', 'PHY/LDPC编解码在旧实链路ledger中没有同步计时，单列为未测，因此不拼接端到端时延。旧链路约60–75ms仅作历史参考，不并入本轮统计。详细重复值见 [本轮计时](../results/prior_aware_uep_20261004/timing.json)。',
        '', 'offline_costs.csv 是含多状态缓存、去重和指标的离线批量耗时，另列于上述无缓存接收时延。',
        '', '## 固定 16 张样例','',
        '沿用原固定样例及 noise=2001。图像按原 256×256 像素直接拼接，只进行 PNG 显示所需的浮点量化。完整 16 张原图在 [originals](../results/prior_aware_uep_20261004/originals/)。',
    ]
    for snr in SNRS:
        text += ['', '### %d dB'%snr,'', '![%d dB 第一页](../results/prior_aware_uep_20261004/figures/main_N1024_snr%02d_page1.png)'%(snr,snr)]
    text+=['','所有主方法与 matched 图页见 [图像清单](../results/prior_aware_uep_20261004/figures_manifest.json)。','']
    report.write_text('\n'.join(text),encoding='utf-8')
    outputs={str(path):q.sha(path) for path in result.rglob('*') if path.is_file() and path.name!='publication_manifest.json'}
    outputs[str(report)]=q.sha(report)
    q.require(all(Path(path).stat().st_size<10_000_000 for path in outputs),'A publication file reaches the 10,000,000-byte repository limit')
    for path,digest in inputs.items():q.require(q.sha(path)==digest,'Input changed during report publication')
    manifest=dict(status='UEP_STATIC_REPORT_COMPLETE',input_bindings=inputs,outputs=outputs,source_indices=list(FIXED),
        generated_model_outputs=False,pixels_modified=False,display_quantization='round float32 RGB*255 to uint8',
        missing_points_imputed=False,automatic_git_push=False)
    q.write(result/'publication_manifest.json',manifest)
    return dict(report=str(report),results=str(result),files=len(outputs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','scored','policies','model-validation','gate','fixed-examples','timing'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--p-scored',type=Path);args=parser.parse_args()
    print(build(args.root,args.scored,args.policies,args.model_validation,args.gate,args.fixed_examples,args.timing,p_scored=args.p_scored))


if __name__=='__main__':main()
