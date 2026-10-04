"""Publishable single-method Swin report; no fabricated paired-method result."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np

HERE=Path(__file__).resolve().parent
METHOD='SwinJSCC_new_shared'
FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
SNRS=(1,7,13);BUDGETS=(1024,2048);SEEDS=(2001,2002,2003)
METRICS=('psnr_db','lpips_alex','dino_cosine','dinov2_vitl14_cosine','clip_image_cosine','dists','dreamsim',
    'ms_ssim','dino_specificity','resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong')

def require(value,message):
    if not value:raise RuntimeError(message)

def dependencies(root):
    runtime=Path(root)/'outputs/EXTERNAL-COMPARISON-20261004/runtime';sys.path.insert(0,str(runtime))
    sys.path.insert(0,str(HERE.parent))
    import external_eval_common as c
    import external_eval_report as original
    import fixed80k_adapter as fixed
    return c,original,fixed

def paths(root):
    root=Path(root).resolve();base=root/'outputs/EXTERNAL-COMPARISON-20261004'
    child=base/'fixed80k_revision/swin_early_release'
    return dict(root=root,base=base,child=child,out=child/'evaluation',config=child/'config.json',
        result=root/'results/external_comparison_20261004/fixed80k_revision/swin_early_release/evaluation')

def boolean(value):
    if value in (True,'True','true','1'):return True
    if value in (False,'False','false','0'):return False
    raise RuntimeError('Explicit boolean required')

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))

def key(row):return int(row['source_index']),int(row['N']),int(row['snr_db']),int(row['noise_seed']),row['method']

def validate_rows(rows):
    expected={(i,n,s,seed,METHOD) for i in range(100) for n in BUDGETS for s in SNRS for seed in SEEDS}
    indexed={key(row):row for row in rows}
    require(len(rows)==1800 and len(indexed)==1800 and set(indexed)==expected,'Exact 1,800-row Swin population required')
    for item,row in indexed.items():
        require(not boolean(row['synthetic']) and not boolean(row['label_conditioned'])
            and not boolean(row['audit_used_to_control_receiver']),'Invalid unconditional scientific row')
        require(int(row['selected_step'])==80000 and row['selected_checkpoint_sha256']==
            '8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21','Wrong evaluated Swin checkpoint')
        require(float(row['E'])==2*item[1] and abs(float(row['actual_energy'])-2*item[1])<.02,'Paid physical energy differs')
        require(all(math.isfinite(float(row[m])) for m in METRICS),'Nonfinite/missing unified metric')
        require(int(row['NFE'])==0,'Swin-only rows cannot contain posterior sampling')
        if not boolean(row['header_accepted']):require(row['fallback']=='fixed_gray_0.5','CRC failure fallback differs')
    return indexed

def validate_summary(rows,indexed):
    expected={(n,s,METHOD,m) for n in BUDGETS for s in SNRS for m in METRICS};answer={}
    for row in rows:
        group=int(row['N']),int(row['snr_db']),row['method'],row['metric']
        require(group in expected and group not in answer,'Summary group differs')
        require(int(row['n_sources'])==100 and int(row['n_frames'])==300,'Summary population differs')
        values=[float(row[x]) for x in ('mean','ci_low','ci_high')]
        require(all(math.isfinite(x) for x in values) and values[1]<=values[2],'Invalid source-bootstrap interval')
        actual=math.fsum(float(indexed[i,group[0],group[1],seed,METHOD][group[3]]) for i in range(100) for seed in SEEDS)/300
        require(abs(actual-values[0])<1e-10,'Summary mean does not match actual frames')
        answer[group]=row
    require(set(answer)==expected,'All 78 unified metric groups required');return answer

def figure_population(root,indexed,reconstruction,records):
    p=paths(root);c,original,_=dependencies(root)
    original.fixed_selection(c.read(p['base']/'fixed_examples.json'),records)
    population={};bindings={}
    for index in FIXED:
        path=p['out']/'source_checkpoints'/f'{index:04d}.json'
        require(reconstruction['outputs'].get(str(path))==c.sha(path),'Figure source receipt is not a completed output')
        value=c.read(path);proof=value['float_reconstructions'];archive=Path(proof['path'])
        require(value['binding']==reconstruction['binding'] and value['source_index']==index,'Source receipt binding differs')
        require(value['payload_sha256']==c.identity({k:v for k,v in value.items() if k!='payload_sha256'}),'Source payload checksum differs')
        c.verify(value['frame_bindings'])
        require(c.sha(archive)==proof['sha256'] and reconstruction['outputs'].get(str(archive))==proof['sha256'],
            'Figure archive is not a completed output')
        with np.load(archive,allow_pickle=False) as data:
            require(set(data.files)=={'source_rgb','images','image_slots','row_ids'},'Unexpected image archive schema')
            target=c.pixels(data['source_rgb']).copy();images=data['images'];slots=data['image_slots'].tolist();ids=data['row_ids'].tolist()
            require(len(value['rows'])==len(slots)==len(ids)==18 and slots==proof['image_slots'],'Wrong Swin source frame count')
            require(ids==proof['row_ids'],'Image row ordering differs')
            pairs={}
            for row,slot,row_id in zip(value['rows'],slots,ids):
                item=key(row);require(item in indexed and item[0]==index and row['replay_row_id']==row_id,'Image row identity differs')
                require(type(slot)is int and 0<=slot<len(images),'Invalid image slot')
                image=c.pixels(images[slot]);scored=indexed[item]
                require(c.rgb_sha(image)==scored['image_sha256']==row['image_sha256'],'Image pixels differ from scored pixels')
                require(c.rgb_sha(target)==scored['reference_sha256']==row['reference_sha256'],'Source metric target differs')
                if item[3]==2001:pairs[item[1],item[2]]=image.copy()
            require(set(pairs)=={(n,s) for n in BUDGETS for s in SNRS},'Fixed figure cells incomplete')
        population[index]=(target,pairs);bindings.update({str(path):c.sha(path),str(archive):proof['sha256']})
    return population,bindings

def plot(output,indexed,population,progress,stopped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # Hash helpers are dependency-free; the already verified pixel population
    # is the sole source of all plotted content.
    import hashlib
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    output.mkdir(parents=True,exist_ok=True);figures=[];cells=[]
    for n in BUDGETS:
        for snr in SNRS:
            pdf_path=output/f'swin_fixed16_N{n}_SNR{snr}.pdf';pages=[]
            with PdfPages(pdf_path,metadata={'Title':f'Exact80k Swin; N{n}; SNR{snr}','CreationDate':None,'ModDate':None}) as pdf:
                for page in range(4):
                    if stopped():raise InterruptedError('Pause requested at figure page boundary')
                    indices=FIXED[page*4:page*4+4]
                    fig,axes=plt.subplots(4,2,figsize=(8.5,14.5),dpi=120)
                    fig.subplots_adjust(left=.065,right=.97,top=.9,bottom=.13,hspace=.43,wspace=.12)
                    axes[0,0].set_title('Original',fontsize=14,weight='bold');axes[0,1].set_title('SwinJSCC: exact 80k',fontsize=14,weight='bold')
                    for pos,index in enumerate(indices):
                        source,pairs=population[index];row=indexed[index,n,snr,2001,METHOD]
                        for col,image in enumerate((source,pairs[n,snr])):
                            ax=axes[pos,col];ax.imshow(image.transpose(1,2,0),interpolation='nearest',vmin=0,vmax=1)
                            ax.set_xticks([]);ax.set_yticks([])
                            for spine in ax.spines.values():spine.set_visible(False)
                            if col==0:caption=f"Source {index:02d} | true class {row['true_class_index']}\nOriginal R50 class {row['resnet50_source_prediction']}"
                            else:
                                caption=(f"PSNR {float(row['psnr_db']):.2f}  LPIPS {float(row['lpips_alex']):.3f}  DINO-L {float(row['dinov2_vitl14_cosine']):.3f}\n"
                                    f"R50 class {row['resnet50_prediction']} | p={float(row['resnet50_top1_probability']):.3f}\n"
                                    +('Header accepted' if boolean(row['header_accepted']) else 'Header rejected: gray fallback'))
                                cells.append(dict(source_index=index,N=n,snr_db=snr,noise_seed=2001,method=METHOD,
                                    image_sha256=row['image_sha256'],reference_sha256=row['reference_sha256'],replay_row_id=row['replay_row_id']))
                            ax.set_xlabel(caption,fontsize=9,labelpad=5)
                    fig.suptitle(f'N = {n} | SNR = {snr} dB | fixed sources {page*4+1}-{page*4+4}/16',fontsize=15,y=.965,weight='bold')
                    fig.text(.5,.933,'Paid header included | E = 2N | no class/text side information',ha='center',fontsize=10)
                    fig.text(.5,.055,'Original registered fixed16 sources; noise seed 2001; no image-quality selection.\n'
                        'DINO-L = DINOv2 ViT-L/14. R50 uses an independent ImageNet classifier.\n'
                        'User selected exact 80k; training safely paused at 81,551; no convergence claim.\n'
                        'Swin-only release. HiFi and the full matched comparison are still pending.',ha='center',fontsize=9,linespacing=1.4)
                    path=output/f'swin_fixed16_N{n}_SNR{snr}_page{page+1}.png'
                    fig.savefig(path,dpi=120,facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)
                    pages.append(dict(path=str(path),sha256=sha(path),source_indices=list(indices)))
                    progress('RUNNING',figures_complete=len(figures)*4+page+1,figures_expected=24)
            figures.append(dict(N=n,snr_db=snr,pdf=str(pdf_path),pdf_sha256=sha(pdf_path),png_pages=pages))
    return figures,cells

def report_text(summary,timing,figures,original):
    lines=['# SwinJSCC：用户指定 80k 检查点评测','',
        '**本次仅发布 SwinJSCC。** HiFi-DiffCom 和完整五方法对照仍在进行；本报告不宣称方法间优劣。','',
        '评测为 N1024/N2048 × 1/7/13 dB × 100 张 development 源图 × 三个噪声种子（2001、2002、2003），共 1,800 个真实传输输出。',
        '模型固定为用户指定的第 80,000 步。训练实际在第 81,551 步安全暂停，后续权重保留但不进入本表。'
        '原 240k 上限与平台期规则未完成；`user_requested_pause=True`、`budget_truncated=True`，不宣称收敛。',
        '同一个全新 MSE SwinJSCC SA+RA 模型覆盖两个资源点。80k 的原完整校准结果用于核验，模型没有根据 development 重新挑选。','',
        '## 资源和评测口径','',
        'N1024 包括正文 768、头部 256 个复数符号；N2048 包括正文 1664、头部 384。逐帧总能量 E=2N，头部的掩码排名、功率、CRC、tail 和冗余均计费。'
        '头部拒收输出固定灰图并计入指标。无免费类别或文本。',
        '保留 DINOv2 ViT-S/14，并增加 DINOv2 ViT-L/14；CLIP 为 ViT-L/14。分类器为独立的 ImageNet ResNet-50，分别统计与真标签和与原图预测一致的比例。'
        'Swin 编解码器没有使用 DINO 或 CLIP 作为视觉骨干。',
        '区间沿用原评测方法：先在每张源图内平均三次噪声，再对 100 张源图 bootstrap 10,000 次（种子 20261002）。'
        '这里报告单方法指标区间，不生成不存在的方法间配对差值。RGB 外部方法不具备可比的 F 恢复误差；FID/KID 留待更大 holdout。','']
    for title,metrics in [('像素与语义',('psnr_db','lpips_alex','dinov2_vitl14_cosine','clip_image_cosine')),
        ('感知及原 DINO 指标',('dists','dreamsim','ms_ssim','dino_cosine','dino_specificity')),
        ('任务表现',('resnet50_top1_label','resnet50_top1_source_prediction','semantic_error','confidently_wrong'))]:
        lines+=['## '+title,'','均值 [95% 源图 bootstrap 区间]。','']
        lines+=original.markdown_table(['N','SNR dB']+[original.METRIC_LABELS[m] for m in metrics],
            [[n,s]+[original.interval(summary[n,s,METHOD,m]) for m in metrics] for n in BUDGETS for s in SNRS])+['']
    lines+=['`semantic_error` 表示与分类器原图预测不一致；`confidently_wrong` 还要求重建图 top-1 概率 ≥0.5。二者不是人工判断，也不是经过概率校准的置信度。','',
        '## 本次推断的接收耗时','',
        '以下是此次实际逐帧推断记录，包含头部解码与失败帧；不含模型加载、TX 或指标计算。最终相同校准图集上的跨方法成本测量仍待完整报告。','']
    lines+=original.markdown_table(['N','SNR','接收均值 / 秒','接收中位数 / 秒','接收 p95 / 秒','头部拒收 / 300'],
        [[v['N'],v['snr_db'],f"{v['RX_mean_seconds']:.5f}",f"{v['RX_median_seconds']:.5f}",f"{v['RX_p95_seconds']:.5f}",v['header_rejected_frames']] for v in timing])
    lines+=['','## 固定样例','',
        '原登记的 16 张源图，每张展示噪声种子 2001。全部 96 个 Swin 单元均来自已计分的浮点重建图，没有按结果挑图。','']
    for item in figures:
        lines += [f"### N{item['N']} · {item['snr_db']} dB",'',f"[全部 16 张样例 PDF](figures/{Path(item['pdf']).name})",'']
        for page in item['png_pages']:lines += [f"![固定样例](figures/{Path(page['path']).name})",'']
    return '\n'.join(lines)

def generate(root,progress,stopped):
    p=paths(root);c,original,fixed=dependencies(root);out=p['out'];result=p['result']
    config=c.read(p['config']);require(Path(config['output'])==out and Path(config['result'])==result,'Early release paths differ')
    fixed.install(root)
    from swin_early_common import validate_config
    validate_config(config)
    complete_path=out/'completion.json';complete=c.read(complete_path)
    require(complete.get('status')=='SWIN_EARLY_EVALUATION_COMPLETE' and complete.get('rows')==1800
        and complete.get('sources')==100 and complete.get('metric_groups')==6 and complete.get('synthetic') is False,
        'All 1,800 Swin unified metric rows must finish')
    require(complete.get('full_comparison_complete') is False,'Early result cannot finish the full comparison')
    c.verify(complete['bindings']);c.verify(complete['outputs'])
    rp=out/'reconstruction_completion.json';reconstruction=c.read(rp)
    require(reconstruction.get('status')=='SWIN_EARLY_RECONSTRUCTIONS_COMPLETE' and reconstruction.get('rows')==1800
        and reconstruction.get('sources')==100 and reconstruction.get('synthetic') is False,'Missing full Swin reconstruction')
    c.verify(reconstruction['bindings']);c.verify(reconstruction['outputs'])
    require(complete['reconstruction_completion_sha256']==c.sha(rp),'Metric/reconstruction identity differs')
    selected,selection=fixed.validate_selection(config['training_output'],root)
    registration_path=out/'reconstruction_registration.json';registration=c.read(registration_path)
    indexed=validate_rows(read_csv(result/'metrics_per_frame.csv'))
    summary=validate_summary(read_csv(result/'metrics_summary.csv'),indexed)
    timing=original.timing_table(list(indexed.values()))
    population,images=figure_population(root,indexed,reconstruction,registration['source_identity'])
    bindings={str(Path(__file__).resolve()):c.sha(__file__),str(p['config']):c.sha(p['config']),
        str(Path(original.__file__).resolve()):c.sha(original.__file__),str(Path(c.__file__).resolve()):c.sha(c.__file__),
        str(complete_path):c.sha(complete_path),str(rp):c.sha(rp),str(registration_path):c.sha(registration_path),
        str(p['base']/'fixed_examples.json'):c.sha(p['base']/'fixed_examples.json'),**selection,**images,
        **complete['outputs'],**reconstruction['bindings']}
    reg=dict(status='SWIN_EARLY_REPORT_REGISTERED',input_bindings=bindings,source_indices=list(FIXED),rows=1800,
        sources=100,methods=[METHOD],selected_step=80000,full_comparison_complete=False,synthetic=False)
    regpath=out/'report_registration.json';c.seal(regpath,reg)
    donepath=out/'report_completion.json'
    if donepath.exists():
        done=c.read(donepath);require(done['report_registration_sha256']==c.sha(regpath),'Existing early report identity differs')
        c.verify(done['outputs']);return done
    figures,cells=plot(result/'figures',indexed,population,progress,stopped)
    require(len(figures)==6 and len(cells)==96,'Fixed Swin figure coverage differs')
    timing_path=result/'receiver_timing.csv';c.write_csv(timing_path,timing)
    manifest=result/'figures_manifest.json';c.seal(manifest,dict(status='SWIN_EARLY_FIXED16_FIGURES',figures=figures,cells=cells,
        source_indices=list(FIXED),noise_seed=2001,original_float_pixels=True,input_bindings=images))
    report=result/'SWIN_EARLY_REPORT.md';report.write_text(report_text(summary,timing,figures,original),encoding='utf-8')
    outputs={str(f):c.sha(f) for f in (timing_path,manifest,report,regpath)}
    for figure in figures:
        outputs[figure['pdf']]=figure['pdf_sha256'];outputs.update({v['path']:v['sha256'] for v in figure['png_pages']})
    c.verify(bindings)
    done=dict(status='SWIN_EARLY_REPORT_COMPLETE',sources=100,physical_frames=1800,rows=1800,metric_groups=6,
        fixed_sources=16,png_figures=24,pdf_figures=6,figure_cells=96,full_comparison_complete=False,synthetic=False,
        selected_step=80000,actual_training_pause_step=81551,original_training_finished=False,
        report_registration_sha256=c.sha(regpath),evaluation_completion_sha256=c.sha(complete_path),
        input_bindings=bindings,outputs=outputs,sample_selection_uses_quality=False)
    c.seal(donepath,done);progress('COMPLETE',rows=1800,full_comparison_complete=False);return done

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
        require(not (p['out']/'report_failure.json').exists(),'Prior report failure requires review')
        generate(args.root,progress,lambda:stop[0])
    except InterruptedError as error:progress('PAUSED',reason=str(error));raise SystemExit(75)
    except Exception as error:
        c.write(p['out']/'report_failure.json',dict(status='FAILED_REQUIRES_REVIEW',error=repr(error),automatic_retry=False));raise

if __name__=='__main__':main()
