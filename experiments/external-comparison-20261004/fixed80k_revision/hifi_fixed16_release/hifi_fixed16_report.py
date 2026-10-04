"""Paired exploratory report of the original fixed16 images; full run stays paused."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import uuid
import numpy as np

HERE=Path(__file__).resolve().parent
METHODS=('SwinJSCC_new_shared','HiFiDiffCom_SwinJSCC')
LABELS={METHODS[0]:'SwinJSCC',METHODS[1]:'HiFi-DiffCom + Swin'}
FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
BUDGETS=(1024,2048);SNRS=(7,13);SEEDS=(2001,)
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim',
    'ms_ssim','dino_specificity','resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')
CHECKPOINT='8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21'

def require(value,message):
    if not value:raise RuntimeError(message)

def paths(root):
    root=Path(root).resolve();base=root/'outputs/EXTERNAL-COMPARISON-20261004';child=base/'fixed80k_revision/hifi_fixed16_release'
    return dict(root=root,base=base,child=child,config=child/'config.json',out=child/'evaluation',
        result=root/'results/external_comparison_20261004/fixed80k_revision/hifi_fixed16_release/evaluation')

def dependencies(root):
    p=paths(root);sys.path.insert(0,str(p['base']/'runtime'));sys.path.insert(0,str(HERE.parent))
    import external_eval_common as c
    import external_eval_report as original
    import fixed80k_adapter as fixed
    return c,original,fixed

def boolean(value):
    if value in (True,'True','true','1'):return True
    if value in (False,'False','false','0'):return False
    raise RuntimeError('Explicit boolean required')

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))

def key(row):return int(row['source_index']),int(row['N']),int(row['snr_db']),int(row['noise_seed']),row['method']

def validate_rows(rows):
    expected={(i,n,s,2001,m) for i in FIXED for n in BUDGETS for s in SNRS for m in METHODS}
    indexed={key(row):row for row in rows}
    require(len(rows)==128 and len(indexed)==128 and set(indexed)==expected,'Exactly the registered 16-source/128-row paired population is required')
    for item,row in indexed.items():
        require(not boolean(row['synthetic']) and not boolean(row['label_conditioned']) and not boolean(row['audit_used_to_control_receiver']),
            'Invalid scientific row')
        require(int(row['selected_step'])==80000 and row['selected_checkpoint_sha256']==CHECKPOINT,'Wrong fixed Swin checkpoint')
        require(float(row['E'])==2*item[1] and abs(float(row['actual_energy'])-2*item[1])<.02,'Paid energy accounting differs')
        require(all(math.isfinite(float(row[m])) for m in METRICS),'Missing/nonfinite unified metric')
        if item[-1]==METHODS[0]:require(int(row['NFE'])==0,'Swin row has posterior steps')
        else:
            other=indexed[item[:-1]+(METHODS[0],)]
            require(row['observed_sha256']==other['observed_sha256']
                and boolean(row['header_accepted'])==boolean(other['header_accepted']),'Receivers did not share identical observed waveform/header')
            if boolean(row['header_accepted']):require(boolean(row['complete_posterior_schedule'])
                and int(row['NFE'])==int(row['t_start']) and int(row['NFE'])>=2,'Full registered posterior required')
            else:require(int(row['NFE'])==0 and row['fallback']=='fixed_gray_0.5'
                and row['image_sha256']==other['image_sha256'],'Failed headers must yield identical gray outputs')
    return indexed

def validate_summary(rows,indexed):
    expected={(n,s,m,metric) for n in BUDGETS for s in SNRS for m in METHODS for metric in METRICS};answer={}
    for row in rows:
        item=int(row['N']),int(row['snr_db']),row['method'],row['metric']
        require(item in expected and item not in answer,'Unexpected/duplicate metric summary')
        require(int(row['n_sources'])==16 and int(row['n_frames'])==16,'Small fixed16 summary population differs')
        values=[float(row[x]) for x in ('mean','ci_low','ci_high')]
        require(all(math.isfinite(v) for v in values) and values[1]<=values[2],'Invalid bootstrap interval')
        actual=math.fsum(float(indexed[i,item[0],item[1],2001,item[2]][item[3]]) for i in FIXED)/16
        require(abs(actual-values[0])<1e-10,'Summary mean differs from actual fixed16 frames');answer[item]=row
    require(set(answer)==expected,'All 104 metric summaries required');return answer

def validate_pairs(rows,indexed):
    expected={(n,s,metric) for n in BUDGETS for s in SNRS for metric in METRICS};answer={}
    for row in rows:
        item=int(row['N_A']),int(row['snr_A']),row['metric']
        require(item in expected and item not in answer and row['method_A']==METHODS[1] and row['method_B']==METHODS[0],
            'Paired interval is not HiFi minus Swin')
        require(int(row['N_B'])==item[0] and int(row['snr_B'])==item[1]
            and row['comparison_scope']=='same_source_noise_seed_and_identical_measured_full_waveform','Unmatched paired interval')
        require(int(row['n_sources'])==16,'Paired interval must use only the registered 16 source pairs')
        values=[float(row[x]) for x in ('mean','ci_low','ci_high')]
        require(all(math.isfinite(v) for v in values) and values[1]<=values[2],'Invalid paired bootstrap interval')
        actual=math.fsum(float(indexed[i,item[0],item[1],2001,METHODS[1]][item[2]])
            -float(indexed[i,item[0],item[1],2001,METHODS[0]][item[2]]) for i in FIXED)/16
        require(abs(actual-values[0])<1e-10,'Paired delta differs from actual 16 matching source pairs');answer[item]=row
    require(set(answer)==expected,'All 52 paired metric intervals required');return answer

def timing_table(rows):
    groups=defaultdict(list)
    for row in rows:groups[int(row['N']),int(row['snr_db']),row['method']].append(row)
    answer=[]
    for (n,s,m),frames in sorted(groups.items()):
        require(len(frames)==16,'Receiver timing requires 16 actual frames per condition/method')
        rx=np.asarray([float(r['RX_seconds']) for r in frames]);accepted=np.asarray([boolean(r['header_accepted']) for r in frames])
        require(np.isfinite(rx).all() and (rx>=0).all(),'Invalid receiver timing')
        nfe=[int(r['NFE']) for r in frames if boolean(r['header_accepted'])]
        answer.append(dict(N=n,snr_db=s,method=m,n_frames=16,RX_mean_seconds=float(rx.mean()),RX_median_seconds=float(np.median(rx)),
            RX_p95_seconds=float(np.percentile(rx,95)),header_rejected_frames=int((~accepted).sum()),
            NFE_min_accepted=min(nfe) if nfe else '',NFE_max_accepted=max(nfe) if nfe else '',
            scope='actual fixed16 frame timings; not the final separately registered common-calibration benchmark'))
    require(len(answer)==8,'Receiver timing condition inventory differs');return answer

def figure_population(root,indexed,reconstruction,records):
    p=paths(root);c,original,_=dependencies(root)
    # Source indices retain the original 0..99 identities; the list is not
    # renumbered 0..15 merely because this release uses a smaller population.
    fixed=c.read(p['base']/'fixed_examples.json')
    require(fixed['source_indices']==list(FIXED),'Original fixed16 selection changed')
    by_index={int(row['source_index']):row for row in records}
    for chosen in fixed['records']:
        index=chosen['source_index'];require(index in by_index and chosen=={k:by_index[index][k] for k in chosen},'Fixed source identity differs')
    from hifi16_common import validate_source
    population={};bindings={}
    for index in FIXED:
        path=p['out']/'source_checkpoints'/f'{index:04d}.json'
        require(reconstruction['outputs'].get(str(path))==c.sha(path),'Figure receipt is not a completed reconstruction output')
        value=validate_source(c.read(path),reconstruction['binding'],index);proof=value['float_reconstructions'];archive=Path(proof['path'])
        require(reconstruction['outputs'].get(str(archive))==c.sha(archive)==proof['sha256'],'Figure archive differs')
        with np.load(archive,allow_pickle=False) as data:
            source=c.pixels(data['source_rgb']).copy();images=data['images'];slots=data['image_slots'].tolist();pairs={}
            require(len(value['rows'])==len(slots)==8,'Eight paired image rows per fixed source required')
            for row,slot in zip(value['rows'],slots):
                item=key(row);require(item in indexed and item[0]==index,'Figure row identity differs')
                image=c.pixels(images[slot]);scored=indexed[item]
                require(c.rgb_sha(image)==scored['image_sha256']==row['image_sha256']
                    and c.rgb_sha(source)==scored['reference_sha256']==row['reference_sha256'],'Figure and metric pixels differ')
                pairs[item[1],item[2],item[4]]=image.copy()
            require(len(pairs)==8,'Figure population incomplete')
        population[index]=(source,pairs);bindings.update({str(path):c.sha(path),str(archive):proof['sha256']})
    return population,bindings

def plot(output,indexed,population,progress,stopped,labels=None):
    import hashlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    def label(value):
        index=int(value)
        return (labels[index]+' ['+str(index)+']') if labels is not None else 'class '+str(index)
    def wrapped(value):return '\n'.join(textwrap.fill(line,39) for line in value.splitlines())
    output.mkdir(parents=True,exist_ok=True);figures=[];cells=[]
    for n in BUDGETS:
        for snr in SNRS:
            pdf_path=output/f'hifi_fixed16_N{n}_SNR{snr}.pdf';pages=[]
            with PdfPages(pdf_path,metadata={'Title':f'Exploratory fixed16; N{n}; SNR{snr}','CreationDate':None,'ModDate':None}) as pdf:
                for page in range(4):
                    if stopped():raise InterruptedError('Requested pause at complete figure page')
                    indices=FIXED[page*4:page*4+4]
                    fig,axes=plt.subplots(4,3,figsize=(12,17.2),dpi=120)
                    fig.subplots_adjust(left=.04,right=.98,top=.89,bottom=.145,hspace=.75,wspace=.095)
                    for col,title in enumerate(('Original','SwinJSCC: exact 80k','HiFi-DiffCom + Swin')):axes[0,col].set_title(title,fontsize=13,weight='bold')
                    for pos,index in enumerate(indices):
                        source,pairs=population[index];first=indexed[index,n,snr,2001,METHODS[0]]
                        for col,image in enumerate((source,pairs[n,snr,METHODS[0]],pairs[n,snr,METHODS[1]])):
                            ax=axes[pos,col];ax.imshow(image.transpose(1,2,0),interpolation='nearest',vmin=0,vmax=1)
                            ax.set_xticks([]);ax.set_yticks([])
                            for spine in ax.spines.values():spine.set_visible(False)
                            if col==0:caption=f"Source {index:02d}\nTrue: {label(first['true_class_index'])}\nOriginal R50: {label(first['resnet50_source_prediction'])}"
                            else:
                                row=indexed[index,n,snr,2001,METHODS[col-1]]
                                caption=(f"PSNR {float(row['psnr_db']):.2f}  LPIPS {float(row['lpips_alex']):.3f}\n"
                                    f"DINO-L {float(row['dinov2_vitl14_cosine']):.3f}\nR50: {label(row['resnet50_prediction'])}\n"
                                    f"p={float(row['resnet50_top1_probability']):.3f} | NFE {row['NFE']} | "
                                    +('header OK' if boolean(row['header_accepted']) else 'gray fallback'))
                                cells.append(dict(source_index=index,N=n,snr_db=snr,noise_seed=2001,method=METHODS[col-1],
                                    image_sha256=row['image_sha256'],reference_sha256=row['reference_sha256'],
                                    observed_sha256=row['observed_sha256'],replay_row_id=row['replay_row_id']))
                            ax.set_xlabel(wrapped(caption),fontsize=9,labelpad=6)
                    fig.suptitle(f'N = {n} | SNR = {snr} dB | fixed examples {page*4+1}-{page*4+4}/16',fontsize=16,y=.968,weight='bold')
                    fig.text(.5,.937,'Both receivers see the same waveform | paid header | E = 2N | no free class/text',ha='center',fontsize=10)
                    fig.text(.5,.065,'Exploratory fixed16 subset, noise seed 2001; not a representative 100-source result.\n'
                        'DINO-L = DINOv2 ViT-L/14. R50 class differences are automated diagnostics, not human semantic judgments.\n'
                        'Exact 80k checkpoint; training paused at 81,551; budget-truncated, no convergence claim.\n'
                        'HiFi completes the registered posterior schedule. Full evaluation remains paused after this release.',ha='center',fontsize=9,linespacing=1.4)
                    path=output/f'hifi_fixed16_N{n}_SNR{snr}_page{page+1}.png'
                    fig.savefig(path,dpi=120,facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
                    pages.append(dict(path=str(path),sha256=sha(path),source_indices=list(indices)))
                    progress('RUNNING',figures_complete=len(figures)*4+page+1,figures_expected=16)
            figures.append(dict(N=n,snr_db=snr,pdf=str(pdf_path),pdf_sha256=sha(pdf_path),png_pages=pages))
    return figures,cells

def report_text(summary,paired,timing,figures,original):
    lines=['# HiFi-DiffCom 第一档：固定 16 张图的配对评测','',
        '**本报告为探索性固定样例评测。** 原登记的 16 张图片不是随机代表样本，本报告不能代表完整 100 张源图，也不能代替人工语义判断。'
        '完成并推送这一档后，完整评测保持暂停，等待下一步决定。','',
        '范围为固定 16 图 × 噪声种子 2001 × N1024/N2048 × SNR 7/13 dB：共 **64 个真实信道帧、128 个 Swin/HiFi 配对输出**。'
        '每一对使用完全相同的实际接收波形和付费头部，未加入 1 dB 或其他噪声种子的未测点。','',
        '## 模型、资源与指标','',
        'Swin 固定为用户指定的第 80,000 步模型；原训练实际在第 81,551 步安全暂停。'
        '`user_requested_pause=True`、`budget_truncated=True`，原收敛协议未完成，不宣称收敛。'
        'HiFi 使用同一 Swin 似然和冻结 ImageNet256 无条件 ADM，执行完整登记后验采样，不以短工程探针代替结果。',
        'N1024 正文/头部为 768/256 个复数符号；N2048 为 1664/384。每帧 E=2N，排名掩码、功率、CRC、tail、冗余都计费；'
        '头部失败时两方法均输出固定灰图并计入指标。没有免费类别或文本条件。',
        '保留 DINOv2 ViT-S/14，并报告 DINOv2 ViT-L/14、CLIP ViT-L/14、DISTS、DreamSim、MS-SSIM、LPIPS-Alex、PSNR 和独立 ResNet-50 任务指标。'
        'DINO 错配特异性使用原登记的负例映射，不因缩小样本而更换负例。外部 RGB 方法没有可比的 F 恢复误差；本小样本不报告 FID/KID。',
        '95% 区间以这 16 张固定源图为单位 bootstrap 10,000 次（种子 20261002），每张只有一个噪声实现。'
        '方法差值在同图同波形的配对上重采样。这些小样本区间仅描述本固定集合，不能表示对 ImageNet 总体或其他信道噪声的可靠泛化区间。','']
    for title,metrics in [('像素与语义',('psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine')),
        ('感知及原 DINO 指标',('dists','dreamsim','ms_ssim','dino_cosine','dino_specificity')),
        ('分类诊断',('resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong'))]:
        lines+=['## '+title,'','固定集合均值 [95% 源图 bootstrap 区间]。','']
        lines+=original.markdown_table(['N','SNR','方法']+[original.METRIC_LABELS[m] for m in metrics],
            [[n,s,LABELS[m]]+[original.interval(summary[n,s,m,k]) for k in metrics] for n in BUDGETS for s in SNRS for m in METHODS])+['']
    lines+=['分类器与真标签、与原图预测的一致率分别列出。`semantic_error` 是与原图分类器预测不一致；'
        '`confidently_wrong` 还要求重建图 top-1 概率 ≥0.5。它们不是人工语义判决，分类概率也不等同于经过校准的正确性概率。','',
        '## 配对差值：HiFi − Swin','',
        '正差值对 PSNR、DINO、CLIP 等越大越好的指标有利；负差值对 LPIPS、DISTS 和错误率等越小越好的指标有利。'
        '某一指标改善不能解释为全面优势。','']
    main=('psnr_db','lpips_alex','dinov2_vitl14_cosine','semantic_error','confidently_wrong')
    lines+=original.markdown_table(['N','SNR']+[original.METRIC_LABELS[m] for m in main],
        [[n,s]+[original.interval(paired[n,s,m]) for m in main] for n in BUDGETS for s in SNRS])
    lines+=['','[全部 52 项配对区间](metrics_paired_intervals.csv) · [全部 104 项均值和区间](metrics_summary.csv) · '
        '[128 个逐帧指标](metrics_per_frame.csv)','',
        '## 实际接收耗时','',
        '包含头部解码、同步后的实际重建和失败帧；不含模型加载、TX 或指标计算。既有缓存保留原实际计时。'
        '这是本固定集合的推断记录，不是最终相同校准集上的独立跨方法成本实验；没有宣称独立接收器的最小显存。','']
    lines+=original.markdown_table(['N','SNR','方法','均值/秒','中位数/秒','p95/秒','头部拒收/16','接受帧 NFE'],
        [[v['N'],v['snr_db'],LABELS[v['method']],f"{v['RX_mean_seconds']:.3f}",f"{v['RX_median_seconds']:.3f}",
            f"{v['RX_p95_seconds']:.3f}",v['header_rejected_frames'],f"{v['NFE_min_accepted']}-{v['NFE_max_accepted']}"] for v in timing])
    lines+=['','## 原图、Swin 与 HiFi 对照','',
        '全部固定 16 图均展示，顺序沿用原登记，不按本轮质量挑图。以下 128 个重建单元直接来自已计分的浮点输出。','']
    for item in figures:
        lines += [f"### N{item['N']} · {item['snr_db']} dB",'',f"[16 图 PDF](figures/{Path(item['pdf']).name})",'']
        for page in item['png_pages']:lines += [f"![原图与两种接收结果](figures/{Path(page['path']).name})",'']
    return '\n'.join(lines)

def generate(root,progress,stopped):
    p=paths(root);c,original,fixed=dependencies(root);out=p['out'];result=p['result'];config=c.read(p['config'])
    require(Path(config['output'])==out and Path(config['result'])==result,'Registered first-tier result paths differ')
    fixed.install(root)
    from hifi16_common import validate_config
    validate_config(config)
    cp=out/'completion.json';done=c.read(cp)
    require(done.get('status')=='HIFI_FIXED16_EVALUATION_COMPLETE' and done.get('rows')==128
        and done.get('sources')==16 and done.get('metric_groups')==8 and done.get('synthetic') is False
        and done.get('full_comparison_complete') is False,'All real first-tier metric outputs required')
    c.verify(done['bindings']);c.verify(done['outputs'])
    rp=out/'reconstruction_completion.json';reconstruction=c.read(rp)
    require(reconstruction.get('status')=='HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE' and reconstruction.get('rows')==128
        and reconstruction.get('sources')==16 and reconstruction.get('physical_frames')==64
        and reconstruction.get('synthetic') is False,'Complete paired reconstruction required')
    c.verify(reconstruction['bindings']);c.verify(reconstruction['outputs'])
    require(done['reconstruction_completion_sha256']==c.sha(rp),'Metrics do not match completed reconstruction')
    selected,selection=fixed.validate_selection(config['training_output'],root)
    regpath=out/'reconstruction_registration.json';registration=c.read(regpath)
    indexed=validate_rows(read_csv(result/'metrics_per_frame.csv'))
    summary=validate_summary(read_csv(result/'metrics_summary.csv'),indexed)
    paired=validate_pairs(read_csv(result/'metrics_paired_intervals.csv'),indexed)
    timing=timing_table(list(indexed.values()))
    population,images=figure_population(root,indexed,reconstruction,registration['source_identity'])
    import torchvision
    from torchvision.models import ResNet50_Weights
    import torchvision.models._meta as category_source
    labels=list(ResNet50_Weights.IMAGENET1K_V2.meta['categories'])
    require(len(labels)==1000 and all(isinstance(v,str) and v for v in labels),'Independent classifier label metadata differs')
    labelpath=out/'label_metadata.json';labelsource=Path(category_source.__file__).resolve()
    c.seal(labelpath,dict(status='INDEPENDENT_RESNET50_V2_IMAGENET_LABEL_METADATA',categories=labels,
        categories_sha256=c.identity(labels),torchvision_version=torchvision.__version__,
        source='ResNet50_Weights.IMAGENET1K_V2.meta.categories',source_bindings={str(labelsource):c.sha(labelsource)},
        network_lookup=False,model_inference=False))
    bindings={str(Path(__file__).resolve()):c.sha(__file__),str(p['config']):c.sha(p['config']),
        str(Path(original.__file__).resolve()):c.sha(original.__file__),str(Path(c.__file__).resolve()):c.sha(c.__file__),
        str(cp):c.sha(cp),str(rp):c.sha(rp),str(regpath):c.sha(regpath),str(labelpath):c.sha(labelpath),str(labelsource):c.sha(labelsource),
        str(p['base']/'fixed_examples.json'):c.sha(p['base']/'fixed_examples.json'),**selection,**images,**done['outputs'],**reconstruction['bindings']}
    reportreg=out/'report_registration.json';c.seal(reportreg,dict(status='HIFI_FIXED16_REPORT_REGISTERED',input_bindings=bindings,
        source_indices=list(FIXED),sources=16,rows=128,physical_frames=64,methods=list(METHODS),selected_step=80000,
        exploratory_fixed_subset=True,full_comparison_complete=False,synthetic=False))
    reportdone=out/'report_completion.json'
    if reportdone.exists():
        previous=c.read(reportdone);require(previous['report_registration_sha256']==c.sha(reportreg),'Existing first-tier report identity differs')
        c.verify(previous['outputs']);return previous
    figures,cells=plot(result/'figures',indexed,population,progress,stopped,labels)
    require(len(figures)==4 and len(cells)==128,'All 128 reconstruction cells required')
    timings=result/'receiver_timing.csv';c.write_csv(timings,timing)
    manifest=result/'figures_manifest.json';c.seal(manifest,dict(status='HIFI_FIXED16_VERIFIED_FIGURES',figures=figures,cells=cells,
        source_indices=list(FIXED),noise_seed=2001,original_float_pixels=True,input_bindings=images))
    report=result/'HIFI_FIXED16_REPORT.md';report.write_text(report_text(summary,paired,timing,figures,original),encoding='utf-8')
    outputs={str(file):c.sha(file) for file in (timings,manifest,report,reportreg,labelpath)}
    for figure in figures:outputs[figure['pdf']]=figure['pdf_sha256'];outputs.update({v['path']:v['sha256'] for v in figure['png_pages']})
    c.verify(bindings)
    value=dict(status='HIFI_FIXED16_REPORT_COMPLETE',sources=16,physical_frames=64,rows=128,metric_groups=8,
        paired_metric_groups=52,fixed_sources=16,png_figures=16,pdf_figures=4,figure_cells=128,
        full_comparison_complete=False,exploratory_fixed_subset=True,full_evaluation_resume_authorized=False,
        synthetic=False,selected_step=80000,actual_training_pause_step=81551,original_training_finished=False,
        report_registration_sha256=c.sha(reportreg),evaluation_completion_sha256=c.sha(cp),input_bindings=bindings,
        outputs=outputs,sample_selection_uses_quality=False)
    c.seal(reportdone,value);progress('COMPLETE',rows=128,full_comparison_complete=False);return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--launch-id')
    args=parser.parse_args();p=paths(args.root);c,_,_=dependencies(args.root);p['out'].mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(p['out']/'report.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    token=args.launch_id or uuid.uuid4().hex
    def progress(status,**fields):c.write(p['out']/'report_status.json',dict(status=status,pid=os.getpid(),launch_id=token,
        safe_pause_handler_installed=True,updated=time.time(),**fields))
    progress('STARTING')
    try:
        require(not (p['out']/'report_failure.json').exists(),'Previous report failure needs review')
        generate(args.root,progress,lambda:stop[0])
    except InterruptedError as error:progress('PAUSED',reason=str(error));raise SystemExit(75)
    except Exception as error:
        c.write(p['out']/'report_failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False));raise

if __name__=='__main__':main()
