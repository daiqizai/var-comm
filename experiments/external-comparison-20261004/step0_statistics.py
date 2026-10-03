"""Summarize admitted historical measurements without model inference."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from step0_cache_export import read, sha, seal, require, STUDIES
from step0_reference_prepare import admitted


def select(row, study):
    meta=json.loads(row['history_metadata_json'])
    snr=float(row['snr_db'])
    seed=int(row.get(meta.get('noise_seed_field','noise_seed'),row.get('seed',-1)))
    if snr not in (7,13) or seed not in (2001,2002,2003): return None
    method=meta.get('method_id',meta.get('method',row['method']))
    N=int(meta.get('N',row.get('N',row.get('complex_uses',-1))))
    if study==STUDIES[0] and method in ('P2048','P3060'): group='P'
    elif study==STUDIES[1] and method=='P4084_N4084_seed2026092304': group='P'
    elif study==STUDIES[2] and N in (2048,3060,4084) and method in (f'final_raw_QPSK_N{N}_Dc',f'final_arithmetic_QPSK_N{N}_Dc'):
        group='Digital raw / Dc' if method.startswith('final_raw_') else 'Digital arithmetic / Dc'
    elif study==STUDIES[3] and row['method'] in ('raw_adaptive','arithmetic_adaptive'):
        group='Legacy raw / D0' if row['method']=='raw_adaptive' else 'Legacy arithmetic / D0'
    else: return None
    decoder=meta.get('decoder_id',meta.get('decoder'))
    require(decoder==('D0' if group.startswith('Legacy') else 'Dc'),'Decoder changed')
    require(meta.get('label_conditioned') is (group!='P'),'Class conditioning changed')
    if group=='P': require(meta.get('training_seed')==2026092304,'Preselected P seed changed')
    prediction=int(row['new_resnet50_prediction']); original=int(row['new_resnet50_source_prediction'])
    require(int(row['new_resnet50_top1_source_prediction'])==int(prediction==original),'Prediction consistency differs')
    columns=meta.get('original_metric_columns',{'psnr_db':'psnr_db','lpips_alex':'lpips_alex'})
    return dict(group=group,N=N,snr_db=int(snr),noise_seed=seed,source_index=int(row['history_source_index']),
        source_id=row['history_source_id'],history_row_id=row['history_row_id'],decoder=decoder,
        true_class_paid=group!='P',training_seed=meta.get('training_seed'),
        semantic_error=int(prediction!=original),true_label_error=1-int(row['new_resnet50_top1_label']),
        psnr_db=float(row[columns['psnr_db']]),lpips_alex=float(row[columns['lpips_alex']]),
        dinov2_vitl14_cosine=float(row['new_dinov2_vitl14_cosine']),
        clip_image_cosine=float(row['new_clip_image_cosine']),dists=float(row['new_dists']),
        dreamsim=float(row['new_dreamsim']))


def run(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    inputs={};rows=[]
    for study in STUDIES:
        _,receipt,bindings=admitted(root,study);inputs.update(bindings)
        p=root/'results/historical_metrics_r2_20261003'/study/'metrics_per_frame.csv'
        require(receipt['bindings'].get(str(p))==sha(p),'Inherited metrics CSV differs: '+str(p))
        inputs[str(p)]=sha(p)
        with p.open() as stream:
            for row in csv.DictReader(stream):
                chosen=select(row,study)
                if chosen: rows.append(chosen)
    require(len(rows)==6600 and len({r['history_row_id'] for r in rows})==6600,'Selected 100x3 coverage differs')
    groups=collections.defaultdict(list)
    for row in rows: groups[row['group'],row['N'],row['snr_db']].append(row)
    require(len(groups)==22,'Budget/protocol cell scope differs')
    bootstrap=np.random.default_rng(20261002).integers(0,100,(10000,100))
    metrics=('semantic_error','true_label_error','psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim')
    summary=[]
    for (group,N,snr),cell in sorted(groups.items()):
        require({(r['source_index'],r['noise_seed']) for r in cell}=={(i,s) for i in range(100) for s in (2001,2002,2003)},'Cell lacks paired sources/noise')
        require(len(cell)==300,'Duplicate cell observations')
        for metric in metrics:
            means=np.asarray([np.mean([r[metric] for r in cell if r['source_index']==i]) for i in range(100)])
            require(np.isfinite(means).all(),'Nonfinite measured metric')
            lo,hi=np.quantile(means[bootstrap].mean(1),[.025,.975])
            summary.append(dict(group=group,N=N,snr_db=snr,metric=metric,mean=float(means.mean()),ci_low=float(lo),ci_high=float(hi),
                sources=100,frames=300,decoder=cell[0]['decoder'],true_class_paid=group!='P',classification_main_eligible=group=='P',
                training_seed=cell[0]['training_seed'],all_frames_error_free=(sum(r['semantic_error'] for r in cell)==0)))
    for filename,data in [('per_frame.csv',rows),('summary.csv',summary)]:
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lookup={(r['group'],r['N'],r['snr_db'],r['metric']):r for r in summary}
    text=['# 大带宽语义错误：已有结果汇总','',
        '使用已完成且已核验的重建评测，100源 × 3噪声，7/13 dB。先按源平均三次噪声，再对源配对 bootstrap 10,000 次（种子20261002）。',
        '语义错误指独立 ResNet-50 对重建图的预测与对原图的预测不一致。它是分类代理指标，不能证明图中每项内容正确。',
        '旧数字策略使用付费真实类别；其分类结果受真实标签条件影响，只作附表。旧 N3060 的 D0 单列。P 固定训练 seed 2026092304，未混合其他 seed。',
        '原图相同，但旧 N3060 浮点归一化与后续协议存在约1ULP差异；原记录保留。不把不同协议间变化归因于带宽一个因素。',
        '', '| 方法 | N | SNR | 语义错误率 [95% CI] | PSNR | LPIPS | DINOv2-L/14 |', '|---|---:|---:|---:|---:|---:|---:|']
    for group,N,snr in sorted(groups):
        values={k:lookup[group,N,snr,k] for k in metrics};e=values['semantic_error']
        text.append(f"| {group} | {N} | {snr} | {100*e['mean']:.1f}% [{100*e['ci_low']:.1f}, {100*e['ci_high']:.1f}] | {values['psnr_db']['mean']:.2f} | {values['lpips_alex']['mean']:.3f} | {values['dinov2_vitl14_cosine']['mean']:.3f} |")
    zero=[(g,n,s) for g,n,s in groups if lookup[g,n,s,'semantic_error']['all_frames_error_free']]
    text+=['', '按全部300帧是否均与原图预测一致，零错误单元：'+(str(zero) if zero else '没有。不能宣称在某个 N 语义错误已经消失。'),
        '', '本次不增加模型推断；自信地错需要新的 softmax 概率，当前旧表没有该字段，明确留空。无类别 N2048 数字策略也不能用这些付费类别点替代。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    colors={'P':'#172B4D','Digital raw / Dc':'#B04A00','Digital arithmetic / Dc':'#007B78','Legacy raw / D0':'#AD668E','Legacy arithmetic / D0':'#705AB3'}
    for ax,snr in zip(axes,(7,13)):
        for group,color in colors.items():
            data=sorted([v for v in summary if v['group']==group and v['snr_db']==snr and v['metric']=='semantic_error'],key=lambda v:v['N'])
            y=np.array([v['mean'] for v in data])*100
            error=np.array([[v['mean']-v['ci_low'] for v in data],[v['ci_high']-v['mean'] for v in data]])*100
            ax.errorbar([v['N'] for v in data],y,yerr=error,label=group+(' [paid class]' if group!='P' else ''),color=color,marker='o',capsize=3,linestyle='-' if group=='P' else '--')
        ax.set_title(f'{snr} dB');ax.set_xlabel('Total complex channel uses N');ax.set_xticks([2048,3060,4084]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Disagreement with original prediction (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Historical protocols: classification proxy; paid-class digital is descriptive')
    fig.tight_layout(rect=(0,.14,1,.96))
    for ext in ('png','pdf','svg'): fig.savefig(out/f'semantic_error_resource.{ext}',dpi=160)
    plt.close(fig)
    files=['per_frame.csv','summary.csv','report.md','semantic_error_resource.png','semantic_error_resource.pdf','semantic_error_resource.svg']
    seal(out/'completion.json',dict(status='STEP0_STATISTICS_COMPLETE',rows=6600,cells=22,bootstrap=10000,source_unit=True,
        input_bindings=inputs,code_sha256=sha(__file__),outputs={n:sha(out/n) for n in files},model_inference=False))
    print('STEP0_STATISTICS_COMPLETE',6600,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.root,a.out)
