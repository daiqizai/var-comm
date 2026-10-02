"""Independent CPU analysis of frozen-image metric sidecars.

No experiment imports, reconstruction, model loading, calibration or selection.
The statistical unit is a source image; noise repeats are averaged first.
Clean oracle and source-only rate curves each retain one seed0 row per source.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULT=ROOT/'results/unified_metrics_20261002'
SEED=20261002
REPLICATES=10000
SNRS=(1,4,7,13,19)
NOISE_SEEDS=(2001,2002,2003)
GROUP_FIELDS=('experiment','scope','N','phy_family','snr_db','method','projection','control',
              'output_role','decoder_id','label_conditioned')
NEW_METRICS=dict(clip_image_cosine='higher',dinov2_vitl14_cosine='higher',dists='lower',dreamsim='lower',ms_ssim='higher',
    resnet50_top1_label='higher',resnet50_top1_source_prediction='higher')
LEGACY_METRICS=dict(psnr_db='higher',lpips_alex='lower',dino_cosine='higher',dino_specificity='higher')
METRICS={**LEGACY_METRICS,**NEW_METRICS}
PAIRED_FIELDS=('contrast','group_A','group_B','experiment_A','scope_A','N_A','phy_A','snr_A',
    'method_A','projection_A','control_A','decoder_A','experiment_B','scope_B','N_B','phy_B',
    'snr_B','method_B','projection_B','control_B','decoder_B','metric','direction','mean',
    'ci_low','ci_high','n_sources','delta','is_main_conclusion','comparison_scope','aggregation')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows,fields=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows and fields is None:raise ValueError('Empty measured output: '+str(p))
    fields=list(fields) if fields is not None else list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def boolean(x):
    if isinstance(x,(bool,np.bool_)):return bool(x)
    s=str(x).strip().lower()
    if s in ('1','1.0','true'):return True
    if s in ('0','0.0','false'):return False
    raise ValueError('Explicit boolean required: '+str(x))


def number(x,required=True):
    if x is None or str(x).strip().lower() in ('','none','null','nan'):
        if required:raise ValueError('Required finite metric missing')
        return None
    value=float(x)
    if not np.isfinite(value):raise ValueError('Nonfinite observation')
    return value


def integer(x):
    value=number(x)
    if int(value)!=value:raise ValueError('Integer identity required')
    return int(value)


def digest(x):
    value=str(x).lower()
    if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('SHA256 identity required')
    return value


def group(row):
    result=[]
    for name in GROUP_FIELDS:
        if name not in row:raise ValueError('Missing group identity: '+name)
        value=row[name]
        if name=='N':value='' if value in ('',None) else str(integer(value))
        elif name=='snr_db':value='clean' if str(value).lower()=='clean' else str(integer(value))
        elif name=='label_conditioned':value='true' if boolean(value) else 'false'
        else:value='' if value is None else str(value)
        result.append(value)
    return tuple(result)


def context(key):
    return dict(zip(GROUP_FIELDS,key))


def group_id(key):
    return hashlib.sha256(json.dumps(context(key),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def available(reg):
    states=reg.get('metric_availability')
    if not isinstance(states,dict):raise ValueError('Registered metric availability required')
    answer=set()
    for metric in NEW_METRICS:
        rec=states.get(metric)
        status=rec.get('status') if isinstance(rec,dict) else rec
        if status in ('AVAILABLE','EVALUATED','COMPLETE','READY'):answer.add(metric)
        elif status not in ('NOT_RUN','NOT_AVAILABLE','NOT_EVALUATED','UNAVAILABLE_NOT_REGISTERED'):
            raise ValueError('Metric availability missing/unknown: '+metric)
    return answer


def metric(row,name,availability):
    if name in NEW_METRICS and name not in availability:
        if number(row.get(name),False) is not None:raise ValueError('Unregistered metric value present: '+name)
        return None
    if name=='dino_specificity':
        mismatch=number(row.get('dino_mismatched'),False)
        return None if mismatch is None else number(row['dino_cosine'])-mismatch
    return number(row.get(name))


def original_sources(baseline,expected_sources=100):
    identities={}
    for row in baseline:
        sid=str(row['source_id']);index=integer(row['source_index'])
        if sid in identities:raise ValueError('Original source baseline duplicated')
        prediction=integer(row['resnet50_source_prediction']);truth=integer(row['true_class_index'])
        if not 0<=prediction<1000 or not 0<=truth<1000:raise ValueError('ImageNet index mapping outside0..999')
        correct=number(row['resnet50_source_top1_label'])
        if correct!=float(prediction==truth):raise ValueError('Original source classifier accuracy contradicts predictions/labels')
        if not row.get('preprocessing_id'):raise ValueError('Source preprocessing missing')
        identities[sid]=dict(index=index,preprocessing_id=str(row['preprocessing_id']),
            reference_sha256=digest(row['reference_sha256']),prediction=prediction,truth=truth,correct=correct)
    if len(identities)!=expected_sources or sorted(r['index'] for r in identities.values())!=list(range(expected_sources)):
        raise ValueError('Original source population is incomplete or indices duplicate')
    sources=sorted(identities,key=lambda sid:identities[sid]['index'])
    return sources,identities


def validate(rows,baseline,reg,expected_sources=100,expected_snrs=SNRS):
    sources,identities=original_sources(baseline,expected_sources);availability=available(reg)
    if reg.get('source_ids')!=sources:raise ValueError('Scoring registration source order differs')
    groups=defaultdict(list);flags={}
    for row in rows:
        sid=str(row['source_id'])
        if sid not in identities:raise ValueError('Unregistered source observation')
        identity=identities[sid]
        if (integer(row['source_index']),str(row['preprocessing_id']),digest(row['reference_sha256']))!=(identity['index'],identity['preprocessing_id'],identity['reference_sha256']):
            raise ValueError('Observation source/reference identity differs')
        digest(row['image_sha256'])
        if digest(row.get('modelmanifest_sha256',reg['modelmanifest_sha256']))!=digest(reg['modelmanifest_sha256']):
            raise ValueError('Per-frame metric model/transform identity differs')
        if not boolean(row.get('replay_parity_passed',reg.get('replay_parity_passed',False))):raise ValueError('Frozen reconstruction replay parity not passed')
        key=group(row);snr=key[4]
        if not key[0] or not key[1] or not key[5] or not key[9]:raise ValueError('Experiment/scope/method/decoder identity missing')
        if snr!='clean' and int(snr) not in expected_snrs:raise ValueError('Unregistered SNR')
        if key[2] and int(key[2]) not in (512,1024):raise ValueError('Unregistered paid budget')
        conditioned=boolean(row['label_conditioned']);main=boolean(row['is_main_conclusion'])
        if key[5].startswith('D_C_') and not conditioned:raise ValueError('D_C class-label access omitted')
        is_d0=key[9]=='D0' or key[8].lower().startswith('d0') or key[5].endswith('_D0')
        is_oracle=key[0]=='M2_ORACLE' or key[8].lower()=='oracle'
        is_rate=key[0]=='M1_RATE' or key[3]=='source_only'
        if is_rate and (key[0]!='M1_RATE' or key[2]!='' or key[3]!='source_only' or snr!='clean'):
            raise ValueError('Source-only rate curve requires M1_RATE, no paid N, source_only PHY and clean seed0')
        if (conditioned or is_d0 or is_oracle or is_rate or key[8].lower()=='reference') and main:
            raise ValueError('Class-conditioned/D0/oracle/source-only reference entered the main conclusion')
        if key in flags and flags[key]!=main:raise ValueError('Main/supplementary role changed within group')
        flags[key]=main
        for name in METRICS:
            value=metric(row,name,availability)
            if name in ('resnet50_top1_label','resnet50_top1_source_prediction') and value is not None and value not in (0.,1.):
                raise ValueError('Classification accuracy must be0/1 per frame')
        if {'resnet50_top1_label','resnet50_top1_source_prediction'}<=availability:
            predicted=integer(row['resnet50_prediction']);original=integer(row['resnet50_source_prediction'])
            if not 0<=predicted<1000 or original!=identity['prediction']:raise ValueError('Classifier source prediction/index changed')
            if number(row['resnet50_top1_label'])!=float(predicted==identity['truth']) or number(row['resnet50_top1_source_prediction'])!=float(predicted==original):
                raise ValueError('Reconstruction classifier flags contradict predictions')
            if number(row['resnet50_source_top1_label'])!=identity['correct']:raise ValueError('Original-source baseline changed between reconstructions')
        groups[key].append(row)
    if not groups:raise ValueError('No scored reconstructions')
    expected=reg.get('expected_groups')
    if not isinstance(expected,list) or not expected:raise ValueError('Frozen expected method/group inventory missing')
    expected_keys=[group(x) for x in expected]
    if len(set(expected_keys))!=len(expected_keys) or set(groups)!=set(expected_keys):raise ValueError('Method/scope inventory incomplete or duplicated')
    # The inventory is sealed before scoring. Actual-link gates and projection
    # choices can legitimately admit only some SNR groups; do not invent rows
    # outside that inventory or treat absence of an unapproved group as failure.
    for key,frames in groups.items():
        seeds=(0,) if key[4]=='clean' else NOISE_SEEDS
        seen=[(str(r['source_id']),integer(r['noise_seed'])) for r in frames]
        if len(set(seen))!=len(seen) or set(seen)!={(sid,seed) for sid in sources for seed in seeds}:
            raise ValueError('Incomplete/duplicate source-noise grid: '+str(key))
    return dict(groups),sources,identities,availability,flags


class Bootstrap:
    def __init__(self,replicates=REPLICATES,seed=SEED):
        self.replicates=replicates;self.seed=seed;self.indices={}
    def interval(self,values):
        values=np.asarray(values,dtype=np.float64)
        if values.ndim!=1:raise ValueError('One source mean per bootstrap observation required')
        n=len(values)
        if n not in self.indices:self.indices[n]=np.random.default_rng(self.seed).integers(0,n,(self.replicates,n))
        valid=np.isfinite(values)
        if not valid.any():return dict(mean=None,ci_low=None,ci_high=None,n_sources=0)
        draws=values[self.indices[n]];counts=np.isfinite(draws).sum(1)
        means=np.divide(np.nansum(draws,axis=1),counts,out=np.full(self.replicates,np.nan),where=counts>0)
        lo,hi=np.percentile(means[np.isfinite(means)],[2.5,97.5])
        return dict(mean=float(values[valid].mean()),ci_low=float(lo),ci_high=float(hi),n_sources=int(valid.sum()))


def source_values(frames,sources,name,availability):
    values=defaultdict(list)
    for row in frames:
        value=metric(row,name,availability)
        if value is not None:values[str(row['source_id'])].append(value)
    return np.asarray([np.mean(values[sid]) if values[sid] else np.nan for sid in sources],dtype=np.float64),values


def summarize(groups,sources,availability,flags,bootstrap):
    summary=[];per_source=[];means={}
    for key,frames in sorted(groups.items()):
        meta=dict(**context(key),group_id=group_id(key),is_main_conclusion=flags[key])
        for name,direction in METRICS.items():
            values,observations=source_values(frames,sources,name,availability)
            means[key,name]=values;interval=bootstrap.interval(values)
            summary.append(dict(**meta,metric=name,direction=direction,**interval,n_frames=sum(map(len,observations.values())),
                n_expected_frames=len(frames),status='EVALUATED' if interval['n_sources'] else 'NOT_EVALUATED',
                aggregation='one_clean_output_per_source' if key[4]=='clean' else 'mean_noise_within_source_then_equal_source_mean'))
            for i,sid in enumerate(sources):
                if np.isfinite(values[i]):per_source.append(dict(**meta,source_id=sid,source_index=i,metric=name,
                    value=float(values[i]),n_noise=len(observations[sid])))
    return summary,per_source,means


def contrasts(groups,reg):
    """Inventory-based contrasts; no metric value is consulted to choose pairs."""
    explicit=reg.get('contrasts')
    if explicit is not None:
        result=[]
        for rec in explicit:
            a,b=group(rec['group_A']),group(rec['group_B'])
            if a not in groups or b not in groups or a==b:raise ValueError('Registered contrast lacks observed groups')
            result.append((a,b,str(rec.get('name',group_id(a)+'_minus_'+group_id(b)))))
        return result
    buckets=defaultdict(list);result=set()
    for key in groups:
        buckets[key[:5]+(key[6],)].append(key)
    for candidates in buckets.values():
        baselines=[k for k in candidates if k[5] in ('P512','P1024','whole_policy') or k[7]=='UNGUIDED' or k[5]=='UNGUIDED']
        if not baselines:baselines=[k for k in candidates if k[5] in ('D_U_QPSK','D_U_16QAM') and k[9]!='D0']
        for b in baselines:
            for a in candidates:
                if a!=b:result.add((a,b))
        for a in candidates:
            if a[5]=='entropy_policy':
                for b in candidates:
                    if b[5] in ('raster_at_entropy','random_at_entropy','oracle_at_entropy','legacy_policy'):result.add((a,b))
            if a[0]=='M1_RATE' and a[5]=='rate_curve_entropy':
                for b in candidates:
                    if b[5] in ('rate_curve_raster','rate_curve_random','rate_curve_oracle'):result.add((a,b))
    # P and digital can have different PHY tags; this remains a source-paired
    # comparison of their frozen output distributions, not identical waveforms.
    for a in groups:
        for b in groups:
            if b[5] in ('P512','P1024') and b[9]!='D0' and a!=b and a[:3]==b[:3] and a[4]==b[4]:
                if a[2] and a[5] not in ('P512','P1024'):result.add((a,b))
    return [(a,b,group_id(a)+'_minus_'+group_id(b)) for a,b in sorted(result)]


def paired(groups,sources,reg,means,flags,bootstrap):
    table=[]
    for a,b,name in contrasts(groups,reg):
        for metric_name,direction in METRICS.items():
            values=means[a,metric_name]-means[b,metric_name]
            ci=bootstrap.interval(values)
            table.append(dict(contrast=name,group_A=group_id(a),group_B=group_id(b),
                experiment_A=a[0],scope_A=a[1],N_A=a[2],phy_A=a[3],snr_A=a[4],method_A=a[5],projection_A=a[6],control_A=a[7],decoder_A=a[9],
                experiment_B=b[0],scope_B=b[1],N_B=b[2],phy_B=b[3],snr_B=b[4],method_B=b[5],projection_B=b[6],control_B=b[7],decoder_B=b[9],
                metric=metric_name,direction=direction,**ci,delta='A_minus_B',
                is_main_conclusion=bool(flags[a] and flags[b] and a[10]==b[10]=='false'),
                comparison_scope='same_source_means; physical_noise_and_waveform_identity_not_assumed',
                aggregation='paired_difference_between_clean_source_outputs' if a[4]==b[4]=='clean' else 'paired_difference_after_mean_noise_within_source'))
    return table


def verify_registration(reg,allow_synthetic=False):
    if (reg.get('synthetic',False) and not allow_synthetic) or reg.get('training_updates')!=0:
        raise ValueError('Actual frozen zero-training scoring required; synthetic fixtures require explicit opt-in')
    if reg.get('original_pipeline_complete') is not True:raise ValueError('Original GPU pipeline must be complete before sidecar scoring')
    for name in ('input_tables','frozen_policy_bindings','pipeline_completion_bindings'):
        entries=reg.get(name)
        if not isinstance(entries,dict) or not entries:raise ValueError('Missing frozen '+name)
        for path,h in entries.items():
            if sha(path)!=h:raise ValueError('Frozen '+name+' changed: '+path)
    manifest=Path(reg['modelmanifest_path'])
    if sha(manifest)!=reg['modelmanifest_sha256']:raise ValueError('Metric model/transform manifest changed')
    return read_json(manifest)


def report(path,summary,baseline_ci,reg,availability,rows,groups):
    counts=defaultdict(lambda:dict(groups=0,frames=0))
    for key,frames in groups.items():counts[key[0]]['groups']+=1;counts[key[0]]['frames']+=len(frames)
    lines=['# 冻结结果的统一补充指标','',
        f"原图ResNet50 ImageNet1K V2 top1准确率：{baseline_ci['mean']:.4f}，源配对bootstrap95%区间 [{baseline_ci['ci_low']:.4f}, {baseline_ci['ci_high']:.4f}]。分母为100张原图，每图只计一次。",'',
        '| 研究 | 完整方法/范围组 | 重建评分帧 |','|---|---:|---:|']
    for name,count in sorted(counts.items()):lines.append(f"| {name} | {count['groups']} | {count['frames']} |")
    lines+=['','有噪组先平均同一源的三次噪声，再对100个源配对抽样10,000次，seed20261002。clean oracle及M1源编码率曲线都按100源×一次seed0独立分组，不复制成三噪声。300次重建不是300张独立图像。区间是重复使用这100张development图上的描述性区间，不含训练种子不确定性，也未做多重比较校正。','',
        'PSNR、LPIPS-Alex和原dino_cosine（DINOv2 ViT-S/14）来自封存原表。新增dinov2_vitl14_cosine独立使用DINOv2 ViT-L/14标准无registers、1024维CLS余弦，不替换原S/14数值；两者使用同一224方形bicubic、align_corners=False及ImageNet归一化。补充指标没有反向选择m/K/lambda/alpha、checkpoint或策略。逐帧表保留原始字段及replay误差，分析只读取评分。','',
        '| 补充指标 | 方向 | 说明 |','|---|---|---|',
        '| OpenAI CLIP ViT-L/14 image cosine | ↑ | 重建图与原图的图像embedding余弦，不是文本相似度 |',
        '| DINOv2 ViT-L/14 image cosine | ↑ | 新增标准无registers模型的1024维CLS余弦；原DINOv2 ViT-S/14单列保留 |',
        '| DISTS | ↓ | 以冻结VGG16特征度量结构与纹理差异 |',
        '| DreamSim default ensemble | ↓ | 实际manifest登记DINO/CLIP/OpenCLIP ViT-B/16与适配权重 |',
        '| MS-SSIM | ↑ | 多尺度结构相似度，使用登记的数据范围与实现 |',
        '| ResNet50 true-label top1 | ↑ | 重建预测与真实ImageNet标签一致的比例 |',
        '| ResNet50 source-prediction agreement | ↑ | 重建预测与原图预测一致，不能代替真实标签准确率 |','',
        'D_C显式标为label_conditioned，不纳入主要结论；D0是参考解码器结果。M1_RATE源编码率曲线在同一m/K下比较entropy与raster/random/oracle，属于源端参考，不代表付费有噪链路。M2免费oracle与实际付费链路按scope分开，oracle结果不能证明实际链路收益。header失败的灰图也完整评分，不按协议成功或新增指标有效性挑帧。','',
        '评测骨干不是全部相互独立：新增DINOv2 ViT-L/14与原S/14属同一模型家族，DreamSim复用DINO/CLIP家族；DISTS的VGG16与旧LPIPS的Alex不同。它们没有与编码、通信或重建共享骨干；但原DINOv2 ViT-S/14参与部分既有校准/门槛约束，不能称为从未用于参数选择的独立验证。新增CLIP/DINOv2 ViT-L/14/DISTS/ResNet/DreamSim/MS-SSIM没有进入本轮选模选参。旧连续模型使用LPIPS-Alex训练损失，因此该指标也不能当作完全独立的验证。实际关系与权重/变换见模型manifest和METRIC_PROTOCOL；多指标一致仍不能排除全部评测偏差。','',
        '**KID：DEFERRED_HOLDOUT。FID：NOT_EVALUATED。** 目前只有100张反复使用的源图；三噪声不会增加独立图像数。分布指标留待更多独立holdout，当前不以同源重复重建扩充分布样本。','',
        '完整输出：[每组汇总](metrics_summary.csv)、[逐源均值](metrics_source_means.csv)、[源配对差](metrics_paired_intervals.csv)、[原图分类基线](source_classification_baseline.json)。','',
        f"模型与预处理manifest SHA256：`{reg['modelmanifest_sha256']}`。"]
    unavailable=sorted(set(NEW_METRICS)-availability)
    if unavailable:lines+=['','未评测指标：'+', '.join(unavailable)+'；状态及原因保留在登记文件，未填造数值。']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze(result=RESULT,replicates=REPLICATES,allow_synthetic=False):
    if not allow_synthetic and replicates!=REPLICATES:raise ValueError('Formal bootstrap is fixed to10000 source resamples')
    result=Path(result);regpath=result/'metrics_registration.json';reg=read_json(regpath)
    verify_registration(reg,allow_synthetic)
    framepath=result/'metrics_per_frame.csv';baselinepath=result/'source_baseline.csv'
    rows=read_csv(framepath);baseline=read_csv(baselinepath)
    groups,sources,identities,availability,flags=validate(rows,baseline,reg)
    bootstrap=Bootstrap(replicates)
    summary,per_source,means=summarize(groups,sources,availability,flags,bootstrap)
    comparison=paired(groups,sources,reg,means,flags,bootstrap)
    outputs={}
    for name,table in [('metrics_summary.csv',summary),('metrics_source_means.csv',per_source)]:
        p=result/name;write_csv(p,table);outputs[str(p)]=sha(p)
    p=result/'metrics_paired_intervals.csv';write_csv(p,comparison,PAIRED_FIELDS);outputs[str(p)]=sha(p)
    baseline_ci=bootstrap.interval([identities[sid]['correct'] for sid in sources])
    p=result/'source_classification_baseline.json';write_json(p,dict(**baseline_ci,model='torchvision ResNet50 IMAGENET1K_V2',
        sources=100,source_ids=sources,one_original_prediction_per_source=True,bootstrap_replicates=replicates,bootstrap_seed=SEED));outputs[str(p)]=sha(p)
    p=result/'METRICS_REPORT.md';report(p,summary,baseline_ci,reg,availability,rows,groups);outputs[str(p)]=sha(p)
    receipt=dict(status='COMPLETE',synthetic=bool(reg.get('synthetic',False)),training_updates=0,policy_selection_updates=0,
        sources=100,frames=len(rows),groups=len(groups),bootstrap_replicates=replicates,bootstrap_seed=SEED,
        statistical_unit='source_image',noise_aggregation='within_source_before_paired_bootstrap',
        KID=dict(status='DEFERRED_HOLDOUT',reason='100 reused sources; repeated noise is not an independent sample'),
        FID=dict(status='NOT_EVALUATED',reason='Deferred distribution evaluation on additional independent holdout'),
        evaluated_metrics=sorted(availability),not_evaluated_metrics=sorted(set(NEW_METRICS)-availability),
        inputs={str(p):sha(p) for p in [regpath,framepath,baselinepath,Path(reg['modelmanifest_path'])]},outputs=outputs)
    write_json(result/'metrics_analysis_completion.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=RESULT)
    parser.add_argument('--allow-synthetic',action='store_true')
    args=parser.parse_args();analyze(args.results_dir,allow_synthetic=args.allow_synthetic)
