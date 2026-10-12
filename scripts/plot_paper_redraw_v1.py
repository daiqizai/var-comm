"""Registry-bound publication figures. Reads published values; never runs models.

Reproduce from the repository root with python scripts/plot_paper_redraw_v1.py.
The separately registered eight-frame example entry supplies M04 if needed.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, re, shutil, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/figures/redraw_v1'
CONFIG=ROOT/'experiments/figure-redraw-20261012'
REG=json.loads((CONFIG/'figure_registry.json').read_text())
METHODS={r['key']:r for r in REG['methods']}
METRICS=REG['metrics']
METRIC_KEYS=[m['key'] for m in METRICS]
ALL_SNRS=[1,4,7,10,13,19]
THREE_SNRS=[4,10,19]
INPUTS={}; DELIVERED=[]
REVIEW=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(rel):
    p=ROOT/rel; INPUTS[rel]=sha(p)
    with p.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    return [dict(r,input_csv=rel,input_sha256=INPUTS[rel]) for r in rows]
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def write_json(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.titlesize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'lines.linewidth':1.5,'lines.markersize':4,'svg.fonttype':'none',
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':True})
def export(fig,folder,stem):
    fig.canvas.draw()
    for fmt in ['pdf','svg','png']:
        fig.savefig(folder/f'{stem}.{fmt}',dpi=600,facecolor='white')
    if REVIEW is not None:REVIEW.savefig(fig)
    # A print-proof copy, not an altered reconstruction supplied as a result.
    with Image.open(folder/f'{stem}.png') as im:
        im.convert('L').save(folder/f'{stem}_grayscale.png')
    plt.close(fig)
def metadata(folder,ident,rows,caption,extra=None):
    write_csv(folder/'plot_data.csv',rows)
    (folder/'caption.tex').write_text(caption+'\n',encoding='utf-8')
    record=dict(figure_id=ident,reviewed_commit=REG['reviewed_commit'],
        registry_sha256=sha(CONFIG/'figure_registry.json'),script_sha256=sha(__file__),
        input_bindings={r['input_csv']:r['input_sha256'] for r in rows if 'input_csv' in r},
        bootstrap_calls=0,model_calls=0,smoothing=False,display_only_scaling=True,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.pdf','.svg','.png']})
    if extra:record.update(extra)
    write_json(folder/'provenance.json',record)
    DELIVERED.append(dict(id=ident,path=str(folder.relative_to(ROOT)),status='EXPORTED',files=list(record['outputs'])))

def normalize(rows,mapping,identity='method',reference=None):
    result=[]
    for row in rows:
        if row['metric'] not in METRIC_KEYS:continue
        value=row[identity]
        if value not in mapping:continue
        if reference and row.get('reference')!=reference.get(value):continue
        result.append(dict(row,display_method=mapping[value]))
    return result

def validate(rows,keys,snrs,count,paired=False):
    seen=set()
    for r in rows:
        key=(r['display_method'],r['metric'],int(float(r['snr_db'])))
        assert key not in seen,('duplicate',key);seen.add(key)
        assert int(r['source_count'])==count and int(r['noise_count'])==3,r
        values=[float(r[k]) for k in ['ci_low','mean','ci_high']]
        assert all(math.isfinite(v) for v in values) and values[0]<=values[1]<=values[2],r
        if paired:assert r['delta_definition'] in ['method minus reference','method - reference','method-reference'],r
    expected={(k,m,s) for k in keys for m in METRIC_KEYS for s in snrs}
    assert seen==expected,dict(missing=sorted(expected-seen),unexpected=sorted(seen-expected))

QUALITY_NOTE=('DINOv2-L denotes ViT-L/14 cosine similarity. Prediction agreement is agreement with the '
    'source-image ConvNeXt prediction, not ground-truth classification accuracy. Shading shows the published '
    'pointwise 95\% source-bootstrap intervals after averaging three noise realizations per source. '
    'All failed frames remain included. No intervals were recomputed.')
DELTA_NOTE=('Differences are method minus reference; intervals are the already published source-paired '
    '95\% bootstrap intervals after averaging three noise realizations per source. Positive PSNR, DINOv2-L, '
    'and agreement differences are favorable; negative LPIPS differences are favorable. Agreement differences '
    'are percentage points. No marginal confidence intervals were subtracted. Zero and adverse results remain included.')

def panels(ident,stem,rows,keys,snrs,count,cohort,caption,paired=False,subset=None,labels=None,singles=False):
    validate(rows,keys,snrs,count,paired)
    folder=OUT/ident;folder.mkdir(parents=True,exist_ok=True)
    metrics=METRICS if subset is None else [m for m in METRICS if m['key'] in subset]
    def render(ms,name,single=False):
        n=len(ms);nrows=2 if n==4 else 1;ncols=1 if single else 2
        fig,axs=plt.subplots(nrows,ncols,figsize=(3.5,2.65) if single else (7,4.45 if n==4 else 2.75),squeeze=False)
        fig.subplots_adjust(left=.17 if single else .105,right=.98,bottom=.23 if (single or n==2) else .14,
                            top=.70 if single else (.78 if ident.startswith('S15') else .80 if len(keys)>2 else .84),hspace=.40,wspace=.35)
        for panel,(ax,metric) in enumerate(zip(axs.flat,ms)):
            extent=[]
            for key in keys:
                s=METHODS[key]; rr=sorted([r for r in rows if r['display_method']==key and r['metric']==metric['key']],key=lambda r:float(r['snr_db']))
                x=np.array([float(r['snr_db']) for r in rr]);scale=metric['display_scale']
                y,lo,hi=[np.array([float(r[k])*scale for r in rr]) for k in ['mean','ci_low','ci_high']]
                label=(labels or {}).get(key,s['legend'])
                args=dict(color=s['color'],linestyle=s['linestyle'],marker=s['marker'],markerfacecolor=s.get('markerface',s['color']),label=label)
                if paired:ax.errorbar(x,y,yerr=np.array([y-lo,hi-y]),capsize=2.5,elinewidth=1,**args)
                else:
                    ax.plot(x,y,**args);ax.fill_between(x,lo,hi,color=s['color'],alpha=.12,linewidth=0)
                    if key=='swin_80k':
                        # The last measured segment spans outside the old 1--13 dB range.
                        ax.plot(x[-2:],y[-2:],color=s['color'],ls='--',lw=1.5)
                        ax.plot(x[-1],y[-1],marker=s['marker'],mfc='white',mec=s['color'],ls='none')
                extent.extend(lo);extent.extend(hi)
            if paired:
                ax.axhline(0,color='#999999',lw=.7,zorder=0);extent.append(0)
            lower,upper=min(extent),max(extent);span=upper-lower
            pad=span*.10 if span else max(abs(lower)*.05,.001)
            ax.set_ylim(lower-pad,upper+pad)
            ax.set_xlim(min(snrs)-.65,max(snrs)+.65);ax.set_xticks(snrs)
            ax.set_xlabel('SNR (dB)');ax.set_ylabel(metric['delta_label' if paired else 'label']+(' ↓' if metric['better']=='lower' else ' ↑'))
            ax.grid(axis='y',color='#E6E6E6',linewidth=.6);ax.set_axisbelow(True)
            if not single:ax.set_title(f'({chr(97+panel)})',loc='left',pad=5)
            ax.tick_params(direction='out',length=3,width=.6)
        handles,legends=axs.flat[0].get_legend_handles_labels()
        fig.legend(handles,legends,loc='upper center',bbox_to_anchor=(.54 if not single else .53,.985),
                   ncol=1 if single else 2,frameon=False,columnspacing=1.6,handlelength=2.8,fontsize=7.5 if single else 8)
        if ident.startswith('S15'):
            fig.text(.54,.89,('NeST-Com (raw)' if count==500 else 'NeST-Com (entropy-coded)')+' − reference',ha='center',fontsize=9)
        fig.text(.54 if not single else .55,.018,f'{cohort}  ·  N = '+('2048' if ident.startswith('S06') else '1024'),ha='center',fontsize=7.5,color='#555555')
        export(fig,folder,name)
    render(metrics,stem)
    if singles:
        for metric in metrics:render([metric],stem+'_'+metric['key'],True)
    metadata(folder,ident,rows,caption,dict(cohort=cohort,source_count=count,noise_count=3,paired=paired,
        row_count=len(rows),main_metrics=[m['key'] for m in metrics],validation='EXACT_GRID_NO_DUPLICATES_FINITE_ORDERED_INTERVALS'))

def system():
    folder=OUT/'M01';folder.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(7,3.45));ax=fig.add_axes([.01,.015,.98,.97]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,label,color='#EEF3F8',size=7.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004,rounding_size=.009',ec='#617083',fc=color,lw=.7))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
    def arrow(start,end,dashed=False,color='#52606D'):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color,linestyle='--' if dashed else '-'))
    ax.text(.003,.977,'(a) Transmission and reconstruction',weight='bold',fontsize=9)
    box(.45,.884,.31,.07,'Offline calibration → configuration table',size=7.5)
    ax.text(.41,.92,'N, SNR',ha='right',fontsize=8);arrow((.415,.918),(.445,.918))
    top=['Input\nimage','Frozen visual\nencoder + VQ','Prefix\nselection (m, K)','Source coding\nraw / entropy','Packetization\nand CRC','LDPC +\nrate matching','QAM + frame\nassembly']
    xs=np.linspace(.008,.866,7);w=.125;h=.105;y=.695
    for j,label in enumerate(top):
        box(xs[j],y,w,h,label)
        if j<6:arrow((xs[j]+w,y+h/2),(xs[j+1]-.003,y+h/2))
    arrow((.51,.88),(xs[2]+w/2,.805),True);arrow((.69,.88),(xs[5]+w/2,.805),True)
    box(.405,.55,.21,.10,'Optional TX VAR\nprobabilities',color='#F2EDF6',size=7.5)
    arrow((.51,.654),(.51,.689),True)
    ax.text(.76,.616,'Protected header + payload\n+ known padding',fontsize=7.5,ha='center',va='center')
    ax.text(.76,.55,r'$N_{\rm header}+N_{\rm body}+N_{\rm idle}=N$',fontsize=8,ha='center')
    # A bent route keeps the channel arrow distinct from the optional TX prior.
    arrow((.93,.69),(.93,.508));ax.text(.945,.594,'AWGN',fontsize=7.5,rotation=90,ha='center',va='center')
    bottom=['Reconstructed\nimage','Frozen image\ndecoder $D_c$','VAR\ncompletion','Source decoding\n+ error rules','Header +\npayload decoding']
    bx=[.008,.184,.36,.57,.80];bw=[.145,.145,.175,.195,.19]
    for j,label in enumerate(bottom):
        box(bx[j],.386,bw[j],.115,label,color='#F1F5F1')
        if j>0:arrow((bx[j]-.003,.442),(bx[j-1]+bw[j-1]+.003,.442))
    ax.text(.65,.329,'Raw: KEEP hard tokens on body CRC rejection.\nEntropy: frozen protocol-specific failure rules.',fontsize=7,ha='center',va='center')
    ax.text(.007,.262,'(b) Boundary-scale recovery',weight='bold',fontsize=9)
    blocks=['Received coarse\nscale prefix','Predict boundary\nscale','Overwrite received\nboundary positions','Update\naccumulated latent','Predict\nfiner scales']
    xx=np.linspace(.008,.805,5)
    for j,label in enumerate(blocks):
        box(xx[j],.087,.182,.105,label,color='#EAF2F8' if j in [0,2] else '#F4ECDA')
        if j<4:arrow((xx[j]+.182,.14),(xx[j+1]-.003,.14))
    ax.text(.5,.028,'Received tokens condition finer scales; other positions in the same boundary scale are not recomputed.',ha='center',fontsize=7.5)
    export(fig,folder,'fig01_system_overview')
    caption=('System overview of NeST-Com. The frozen visual encoder produces a multiscale residual-quantized '
        'representation with scale widths 1,2,3,4,5,6,8,10,13,16. A preselected prefix (m,K) is coded as fixed '
        '12-bit indices or using optional transmitter-side VAR-conditional entropy coding. The paid header '
        'communicates the decoding profile; it is not free side information. Known padding may carry energy. '
        'The raw protocol retains actual hard-decision body tokens after body CRC rejection (KEEP); entropy '
        'decoding uses its own frozen rejection rules. Receiver prefix overwrites change the accumulated latent '
        'and condition finer scales, without recomputing other positions in the same boundary scale. '
        'The transmitter knows the lookup SNR and selects its prefix before transmission; arbitrary mid-codeword '
        'air-interface interruption is not supported. Dc was adapted previously and is now frozen.')
    rows=[dict(component='scale_prefix',complete_scales=m,tokens=t,raw_bits=12*t) for m,t in [(6,91),(7,155),(8,255),(9,424),(10,680)]]
    metadata(folder,'M01',rows,caption,dict(kind='editable conceptual schematic',scientific_values_generated=False))

def curves():
    raw=read_csv('results/main_raw64_20261007/final_common500_r6/summary.csv')
    new=read_csv('results/entropy_partial_new100_20261011/summary.csv')
    paired=read_csv('results/entropy_partial_new100_20261011/paired.csv')
    main=read_csv('paper/figures/final_paper_comparisons_20261012/plot_data.csv')
    for row in main:
        assert sha(ROOT/row['source_csv'])==row['source_sha256'],row['source_csv']
        assert row['population']=='common500' and int(row['N'])==1024
    rows=normalize(main,{'proposed':'nest_raw','latent':'latent_final','swin':'swin_final','bpg':'bpg_adaptive'})
    panels('M02','fig02_main_common500',rows,['nest_raw','latent_final','swin_final','bpg_adaptive'],ALL_SNRS,500,'Common 500 sources',
        'Reconstruction quality at N=1024 complex channel uses per image on common500, with 500 sources and three '
        'noise realizations per source. Latent-space JSCC uses the selected 100k checkpoint; adapted SwinJSCC uses '
        '200k, including weight-initialized additional training over 1--19 dB with a reinitialized optimizer, '
        'sampling order, and RNG. Swin reached its additional training budget while still improving; convergence '
        'is not claimed. The BPG baseline uses adaptive resolution. '+QUALITY_NOTE,singles=True)
    delta=normalize(paired,{'RAW_PARTIAL':'nest_raw','EC_VAR_PARTIAL':'nest_entropy'},reference={'RAW_PARTIAL':'RAW_WHOLE','EC_VAR_PARTIAL':'EC_VAR_WHOLE'})
    caption=('Benefit of allowing partial scales within each source-coding interface on the same 100-source '
        'confirmation set, N=1024. Raw coding compares NeST-Com (raw) minus Whole-scale prefix + VAR (raw); '
        'VAR-conditional entropy coding compares the corresponding entropy-coded partial and whole-scale '
        'systems. Partial configurations may fall back to K=0. Each within-interface comparison retains the same '
        'modulation and coding permissions. Interfaces also differ in packet format and failure handling; their '
        'increments are not added or interpreted as a pure orthogonal factorial ablation. '+DELTA_NOTE)
    labels={'nest_raw':'Raw coding','nest_entropy':'VAR-conditional entropy coding'}
    panels('M03','fig03_partial_scale_gains_new100',delta,['nest_raw','nest_entropy'],THREE_SNRS,100,'New100 confirmation set',caption,True,['psnr_db','dinov2_vitl14_cosine'],labels)
    panels('S03_paired','s03_partial_gains_four_metrics',delta,['nest_raw','nest_entropy'],THREE_SNRS,100,'New100 confirmation set',caption,True,labels=labels)
    mapping={'RAW_WHOLE':'whole_raw','RAW_PARTIAL':'nest_raw','EC_VAR_WHOLE':'whole_entropy','EC_VAR_PARTIAL':'nest_entropy'}
    panels('S03','s03_four_digital_modes_new100',normalize(new,mapping),list(mapping.values()),THREE_SNRS,100,'New100 confirmation set',
        'Four frozen digital modes on the same new100 population. Whole-scale prefix permits K=0, not necessarily '
        'all ten scales. Raw and entropy interfaces have different packet and failure rules. '+QUALITY_NOTE)
    ids={f'RAW64_{a}_VAR_COMPLETION_SNR_{s}':k for a,k in [('WHOLE','whole_raw'),('PARTIAL','nest_raw')] for s in ALL_SNRS}
    panels('S02','s02_raw_whole_partial_common500',normalize(raw,ids,'point_id'),['whole_raw','nest_raw'],ALL_SNRS,500,'Common 500 sources',
        'Whole-scale and partial-scale raw transmission with the same frozen VAR and Dc, modulation and coding '
        'permissions. Partial transmission permits a whole-scale fallback; K is not forced positive. '+QUALITY_NOTE)
    ids={f'RAW64_PARTIAL_{a}_SNR_{s}':k for a,k in [('VAR_COMPLETION','nest_raw'),('DIRECT_DC','partial_no_completion')] for s in ALL_SNRS}
    panels('S05','s05_same_received_tokens_completion',normalize(raw,ids,'point_id'),['partial_no_completion','nest_raw'],ALL_SNRS,500,'Common 500 sources',
        'Both arms use identical actually received tokens and the same frozen Dc. Direct decoding sets missing '
        'residual contributions to zero; missing codebook indices are not set to zero and no additional source '
        'truth is supplied. '+QUALITY_NOTE)
    newbase=read_csv('results/baseline_strength_20261011/paired_supplement_v1/new100/summary.csv')
    rows=normalize(new,{'EC_VAR_PARTIAL':'nest_entropy'})+normalize(newbase,{'P100000':'latent_final','Swin200000':'swin_final'})
    panels('S04','s04_entropy_vs_final_baselines_new100',rows,['nest_entropy','latent_final','swin_final'],THREE_SNRS,100,'Existing new100 sources',
        'Entropy-coded NeST-Com and frozen final continuous baselines (100k latent-space JSCC and 200k adapted '
        'SwinJSCC) on the same previously released new100 sources. This later baseline evaluation is not a new '
        'simultaneously blinded confirmation study. '+QUALITY_NOTE)
    n2048=read_csv('results/wcl_evidence_closure_20261009/T6_budget2048/v1/summary.csv')
    rows=normalize(n2048,{'N2048_RAW_WHOLE_VAR_COMPLETION':'whole_raw','N2048_RAW_PARTIAL_VAR_COMPLETION':'nest_raw','N2048_EC_VAR_WHOLE_VAR_COMPLETION':'whole_entropy'},'method_id')
    panels('S06','s06_budget2048_confirm100',rows,['whole_raw','nest_raw','whole_entropy'],THREE_SNRS,100,'N2048 confirmation 100',
        'Three digital schemes at N=2048 on their fixed 100-source confirmation set. No entropy-coded partial '
        'curve is available for this budget. Coincident 19 dB outputs remain included; values are not subtracted '
        'from a different population at N1024. '+QUALITY_NOTE)
    for pop,num,snrs,key,method in [('common500',500,ALL_SNRS,'nest_raw','RAW_PARTIAL'),('new100',100,THREE_SNRS,'nest_entropy','EC_VAR_PARTIAL')]:
        table=read_csv(f'results/baseline_strength_20261011/paired_supplement_v1/{pop}/paired.csv')
        rr=[dict(r,display_method={'P100000':'latent_final','Swin200000':'swin_final'}[r['reference']]) for r in table if r['method']==method and r['reference'] in ['P100000','Swin200000']]
        panels('S15_'+pop,'s15_paired_external_'+pop,rr,['latent_final','swin_final'],snrs,num,pop,
            METHODS[key]['legend']+' minus each final baseline on '+pop+'. '+DELTA_NOTE,True,
            labels={'latent_final':'vs. Latent-space JSCC','swin_final':'vs. SwinJSCC (adapted)'})

def training_figures():
    rows=read_csv('results/baseline_strength_20261011/training_figures_v1/plot_data.csv')
    for branch,key,stop in [('P1024','latent_final',100000),('SwinJSCC (adapted)','swin_final',200000)]:
        selected=[r for r in rows if r['branch']==branch and r['series'] in ['training_loss_500_update_block_mean','calibration_objective']]
        ident='S09_'+key;folder=OUT/ident;folder.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(7,2.65));fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.84,wspace=.40)
        for ax,series,ylabel in zip(axes,['training_loss_500_update_block_mean','calibration_objective'],
              ['Training objective (500-update mean)','Calibration objective' if branch=='P1024' else 'Calibration RGB MSE']):
            rr=sorted([r for r in selected if r['series']==series],key=lambda r:float(r['global_step']))
            ax.plot([float(r['global_step'])/1000 for r in rr],[float(r['value']) for r in rr],color=METHODS[key]['color'],marker='o' if series.startswith('calibration') else None,markersize=3,lw=1.3)
            ax.axvline(stop/1000,color='#555555',ls='--',lw=.8)
            for change in ([60,70] if branch=='P1024' else [120,140]):
                ax.axvline(change,color='#AAAAAA',ls=':',lw=.65)
            ax.set(xlabel='Training updates (k)',ylabel=ylabel);ax.grid(axis='y',color='#E6E6E6',lw=.6)
        fig.suptitle(METHODS[key]['legend']+(' · empirical plateau at 100k' if stop==100000 else ' · budget stop at 200k'),fontsize=9,y=.975)
        fig.text(.5,.025,'Calibration only; no holdout-based model selection',ha='center',fontsize=7.5)
        export(fig,folder,'s09_learning_'+key)
        metadata(folder,ident,selected,'Training loss uses all registered non-overlapping 500-update means. Calibration '
            'retains every published point. Latent-space JSCC resumes the full original 40k state and reaches its '
            'empirical plateau at 100k. Swin starts from 80k weights with a fresh optimizer, sampling and RNG, '
            'extends training to 1--19 dB, and stops at 200k by budget while improving. Objectives differ across '
            'methods and cannot be compared by their absolute values. Dashed lines mark selected checkpoints; '
            'dotted lines mark learning-rate transitions (60k/70k for latent-space JSCC; 120k/140k for Swin).')
    data=read_csv('results/baseline_strength_20261011/automatic_followthrough_v1/milestones/descriptive_summary.csv')
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#555555']
    for prefix,name in [('P','Latent-space JSCC'),('Swin','SwinJSCC (adapted)')]:
        rr=[r for r in data if r['method'].startswith(prefix)];folder=OUT/('S09_milestones_'+prefix);folder.mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(2,2,figsize=(7,4.45));fig.subplots_adjust(left=.105,right=.98,bottom=.14,top=.82,hspace=.40,wspace=.35)
        for ax,m in zip(axes.flat,METRICS):
            for s,c,marker,ls in zip(ALL_SNRS,colors,['o','s','^','v','D','P'],['-','--','-.',':','--','-.']):
                own=sorted([r for r in rr if int(r['snr_db'])==s and r['metric']==m['key']],key=lambda r:int(r['method'].removeprefix(prefix)))
                assert len(own)==4
                xx=[int(r['method'].removeprefix(prefix))/1000 for r in own];yy=[float(r['mean'])*m['display_scale'] for r in own]
                ax.plot(xx,yy,color=c,marker=marker,ls=ls,label=f'{s} dB',lw=1.2,markersize=3)
                if prefix=='Swin' and s==19:ax.plot(xx[0],yy[0],color=c,marker=marker,mfc='white',ls='none',ms=4)
            ax.set(xlabel='Training updates (k)',ylabel=m['label']);ax.grid(axis='y',color='#E6E6E6',lw=.6)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(.55,.99))
        fig.text(.54,.015,name+' · calibration1000 · descriptive means',ha='center',fontsize=7.5)
        export(fig,folder,'s09_milestones_'+prefix)
        metadata(folder,'S09_milestones_'+prefix,rr,'Frozen checkpoint diagnostics on calibration1000, averaging three '
            'noise realizations per source. These are descriptive means, not holdout results; no confidence '
            'intervals were computed for this plot. Old Swin80k at 19 dB is outside its old training and calibration range.')

def bottleneck():
    rows=read_csv('results/baseline_strength_20261011/automatic_followthrough_v1/bottleneck/descriptive_summary.csv')
    keys=['F_direct_Dc_free_latent_reference','P40000_zero_noise','P40000_19dB','P100000_zero_noise','P100000_19dB']
    names=['Full-latent reference','40k, noise removed','40k, 19 dB','100k, noise removed','100k, 19 dB']
    folder=OUT/'S10';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(7,4.5));fig.subplots_adjust(left=.23,right=.98,bottom=.16,top=.92,wspace=.30,hspace=.50)
    for j,(ax,m) in enumerate(zip(axes.flat,METRICS)):
        own=[next(r for r in rows if r['method']==k and r['metric']==m['key']) for k in keys]
        ax.scatter([float(r['mean'])*m['display_scale'] for r in own],range(5),c=['#272727','#B64342','#B64342','#B64342','#B64342'],s=24)
        ax.set_yticks(range(5),names if j%2==0 else ['']*5,fontsize=8);ax.invert_yaxis()
        ax.set_xlabel(m['label']);ax.grid(axis='x',color='#E6E6E6',lw=.6)
    fig.text(.5,.025,'Latent-space JSCC bottleneck diagnostic · fixed calibration100 · descriptive means',ha='center',fontsize=7.5)
    export(fig,folder,'s10_latent_bottleneck')
    metadata(folder,'S10',rows,'Full-latent reference and latent-space JSCC bottleneck diagnostics on the fixed first '
        '100 calibration sources. Noise removal retains the N1024 communication bottleneck and the 19 dB '
        'receiver condition. The free full-latent reference has no equal communication budget and is not an '
        'achievable N1024 upper bound. Only published descriptive means are shown; no uncertainty was fabricated.')

def lengths():
    rows=[r for r in read_csv('results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/source_lengths.csv') if int(r['complete_scale_m'])==9 and r['population']=='calibration']
    assert len(rows)==2000
    folder=OUT/'S07';folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.19,right=.97,bottom=.21,top=.72)
    for family,key,label in [('EC_STATIC_WHOLE','whole_static','Static entropy coding'),('EC_VAR_WHOLE','whole_entropy','VAR-conditional entropy coding')]:
        rr=[r for r in rows if r['family']==family];assert len(rr)==1000 and len({r['source_id'] for r in rr})==1000
        assert all(r['length_status']=='ACTUAL_ENCODED_LENGTH' and r['arithmetic_stream_includes_flush']=='True' for r in rr)
        values=np.sort([int(r['arithmetic_bits']) for r in rr]);s=METHODS[key]
        ax.step(values,np.arange(1,1001)/1000,where='post',label=label,color=s['color'],ls=s['linestyle'])
    assert {r['raw12_bits'] for r in rows}=={'5088'}
    ax.plot([5088,5088],[0,1],color=METHODS['nest_raw']['color'],label='Fixed-length token coding')
    ax.set(xlabel='Encoded source bits per image',ylabel='Empirical CDF',ylim=(0,1.02));ax.grid(axis='y',color='#E6E6E6',lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',frameon=False,fontsize=7.5)
    export(fig,folder,'s07_actual_source_lengths_m9')
    metadata(folder,'S07',rows,'Actual terminated source-stream lengths for the same complete nine-scale prefix '
        '(424 tokens), on calibration1000. Fixed-length coding takes 5088 source bits. Arithmetic lengths include '
        'stream termination, not estimated cross entropy. No wireless capacity line is inferred from total N.')

def qualitative():
    result=ROOT/'results/figure_redraw_20261012/final_examples_v1'
    if not (result/'completion.json').exists():
        DELIVERED.append(dict(id='M04',status='WAITING_FOR_EIGHT_FIXED_FRAMES',path=str(result.relative_to(ROOT))));return
    done=json.loads((result/'completion.json').read_text());receipts=json.loads((result/'receipts.json').read_text())
    assert done['frames']==8 and done['receipts_sha256']==sha(result/'receipts.json')
    assert json.loads((result/'actual_wait.json').read_text())['returncode']==0
    old=ROOT/'paper/figures/user_selected_examples_20261012';plan=json.loads((old/'selection_and_integrity.json').read_text())
    arrays=[];records=[]
    for letter,index in zip('abcd',[21,92,95,75]):
        row=[]
        for key,name in [('original','original'),('swin_final','Swin200000'),('latent_final','P100000'),('nest_raw','partial_scale_with_var')]:
            if key in ['original','nest_raw']:
                path=old/'individual'/f'example_{letter}_{name}.png';assert sha(path)==plan['individual_sha256'][path.name]
                receipt=dict(reused=True,path=str(path),sha256=sha(path),noise_seed=6201 if key=='nest_raw' else '')
            else:
                receipt=next(r for r in receipts if r['source_index']==index and r['method']==name)
                path=Path(receipt['png']['path']);assert sha(path)==receipt['png']['sha256']
            image=np.asarray(Image.open(path));assert image.shape==(256,256,3) and image.dtype==np.uint8
            row.append(image);records.append(dict(source_index=index,method=key,N=1024,snr_db=13,receipt=json.dumps(receipt,sort_keys=True)))
        arrays.append(row)
    folder=OUT/'M04';folder.mkdir(parents=True,exist_ok=True)
    labels=['Original','SwinJSCC\n(adapted)','Latent-space\nJSCC','NeST-Com\n(raw)']
    for n,stem in [(3,'fig04_qualitative_final_checkpoints'),(4,'fig04_qualitative_all_four')]:
        fig,axes=plt.subplots(n,4,figsize=(7,n*1.70+.48));fig.subplots_adjust(left=.04,right=.997,bottom=.005,top=1-.40/(n*1.7+.48),wspace=.025,hspace=.025)
        for i in range(n):
            for j in range(4):
                ax=axes[i,j];ax.imshow(arrays[i][j],interpolation='none');ax.axis('off')
                if i==0:ax.set_title(labels[j],fontsize=8.5,pad=5)
                if j==0:ax.text(-.025,.5,f'({chr(97+i)})',transform=ax.transAxes,ha='right',va='center',fontsize=8)
        export(fig,folder,stem)
    metadata(folder,'M04',records,'Fixed development examples at N=1024 and 13 dB. Main rows are the white dog, '
        'broccoli and pastries; the full four-row panel additionally includes the shopping cart. The four sources '
        '(21,92,95,75) were selected by the user after inspecting a 16-source historical gallery, not sampled as '
        'representative performance examples. Final latent-space JSCC100k and adapted SwinJSCC200k each generated '
        'their own transmitted waveform and actual received observation, with fixed first noise seed2001. '
        'Unchanged digital results reuse the verified seed6201 observations. Cross-family noise draws differ; '
        'no per-method best-noise selection was performed. All sources and operating points match. No image '
        'enhancement is applied; 600dpi describes figure export, not reconstruction resolution.',
        dict(inference_receipt_sha256=sha(result/'receipts.json'),source_indices=[21,92,95,75],
             actual_new_inference_frames=8,historical_source_and_digital_binding_sha256=sha(old/'selection_and_integrity.json')))
    # The mechanism panel uses the independently verified SAME received state.
    folder=OUT/'S05_photos';folder.mkdir(parents=True,exist_ok=True)
    proofs=plan['mechanism_proofs'];photo_rows=[]
    fig,axes=plt.subplots(4,3,figsize=(7,9.55));fig.subplots_adjust(left=.035,right=.995,bottom=.005,top=.953,wspace=.025,hspace=.025)
    for i,(letter,index) in enumerate(zip('abcd',[21,92,95,75])):
        pair=[p for p in proofs if p['source_index']==index and 'PARTIAL' in p['method']];assert len(pair)==2
        for field in ['actual_state_sha256','received_tokens_sha256','model_identity_sha256']:assert pair[0][field]==pair[1][field]
        for j,(name,label) in enumerate([('original','Original'),('partial_scale_without_completion','Direct decoding'),('partial_scale_with_var','VAR completion')]):
            path=old/'individual'/f'example_{letter}_{name}.png';assert sha(path)==plan['individual_sha256'][path.name]
            photo_rows.append(dict(source_index=index,method=label,N=1024,snr_db=13,path=str(path),sha256=sha(path)))
            axes[i,j].imshow(np.asarray(Image.open(path)),interpolation='none');axes[i,j].axis('off')
            if i==0:axes[i,j].set_title(label,fontsize=9)
    export(fig,folder,'s05_identical_received_tokens_photos')
    metadata(folder,'S05_photos',photo_rows,'Original, direct decoding and VAR completion at N=1024,13 dB on the '
        'same user-selected development sources. Last two columns share actual received tokens and frozen Dc. '
        'Direct decoding zeros missing residual contributions, not unknown codebook indices. No extra ground '
        'truth is supplied. These images are reused without altering their pixels.',dict(received_state_proofs=proofs))
    folder=OUT/'S16';folder.mkdir(parents=True,exist_ok=True);history_rows=[]
    fig,axes=plt.subplots(4,5,figsize=(7,6.05));fig.subplots_adjust(left=.015,right=.995,bottom=.007,top=.917,wspace=.025,hspace=.025)
    columns=[('original','Original'),('swin','SwinJSCC\n(adapted, 80k)'),('continuous_latent_jscc','Latent-space\nJSCC (40k)'),('hifi_diffcom','HiFi-DiffCom\n(adapted)'),('partial_scale_with_var','NeST-Com\n(raw)')]
    for i,(letter,index) in enumerate(zip('abcd',[21,92,95,75])):
        for j,(name,label) in enumerate(columns):
            path=old/'individual'/f'example_{letter}_{name}.png';assert sha(path)==plan['individual_sha256'][path.name]
            axes[i,j].imshow(np.asarray(Image.open(path)),interpolation='none');axes[i,j].axis('off')
            if i==0:axes[i,j].set_title(label,fontsize=7.5)
            history_rows.append(dict(source_index=index,method=label.replace('\n',' '),N=1024,snr_db=13,path=str(path),sha256=sha(path)))
    export(fig,folder,'s16_historical_hifi_selected')
    metadata(folder,'S16',history_rows,'Historical development examples at N1024,13 dB. Adapted Swin80k and '
        'HiFi-DiffCom share the historical received observation. The latent baseline is40k. These images are '
        'not the final100k/200k baselines in M02/M04. The user chose these four sources after viewing16 '
        'candidates; all prior galleries remain preserved. No new HiFi recovery or image enhancement was run.',
        dict(historical_selection_sha256=sha(old/'selection_and_integrity.json')))
    native=OUT/'M04/native';native.mkdir(exist_ok=True)
    for letter,index in zip('abcd',[21,92,95,75]):
        for name in ['original','partial_scale_with_var']:
            source=old/'individual'/f'example_{letter}_{name}.png';dest=native/f'{index:04d}_{name}.png'
            shutil.copyfile(source,dest);assert sha(source)==sha(dest)
        for name in ['P100000','Swin200000']:
            receipt=next(r for r in receipts if r['source_index']==index and r['method']==name)
            source=Path(receipt['png']['path']);dest=native/f'{index:04d}_{name}.png'
            shutil.copyfile(source,dest);assert sha(source)==sha(dest)

def extra_published():
    # Historical Kodak versions are resolved explicitly, never via unversioned aliases.
    rows=normalize(read_csv('results/generalization_kodak_20261009/analysis_v1/summary.csv'),
                   {'VAR_UNCONDITIONAL':'nest_raw','P1024':'latent_40k','SWIN80K':'swin_80k','BPG_ADAPTIVE':'bpg_adaptive'})
    panels('S12','s12_kodak256_crops',rows,['nest_raw','latent_40k','swin_80k','bpg_adaptive'],THREE_SNRS,24,'Kodak24 · 256×256 center crops',
        'Frozen-policy transfer to one deterministic 256x256 center crop from each of the 24 Kodak photographs. '
        'The latent-space checkpoint is historical40k and the adapted Swin checkpoint is historical80k. '
        '19 dB is outside the old Swin training and calibration range; no13 dB point was measured. '
        'This is not native-resolution Kodak transmission and does not establish absence from all pretrained '
        'training corpora. The raw method uses unconditional VAR. '+QUALITY_NOTE)
    # The historical report embeds the exact published paired rows, not rounded Markdown.
    rel='experiments/entropy-partial-mismatch-20261010/reports/leo_mismatch_statistics_v2.json'
    report=json.loads((ROOT/rel).read_text());assert report['actual_owner']['returncode']==0
    rows=[dict(r,input_csv=rel,input_sha256=sha(ROOT/rel)) for r in report['paired_results'] if r['metric']=='psnr_db']
    assert len(rows)==18
    folder=OUT/'S11';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(7,2.8));fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.72,wspace=.40)
    for ax,snr in zip(axes,[4,7,10]):
        for family,key in [('WHOLE','whole_raw'),('PARTIAL','nest_raw')]:
            rr=sorted([r for r in rows if r['family']==family and r['actual_snr_db']==snr],key=lambda r:r['config_snr_db'])
            assert len(rr)==3 and all(r['source_count']==100 and r['noise_count']==3 for r in rr)
            x=[r['config_snr_db']-snr for r in rr];y=np.array([r['mean'] for r in rr])
            lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr]);assert (lo<=y).all() and (y<=hi).all()
            s=METHODS[key];ax.errorbar(x,y,yerr=[y-lo,hi-y],capsize=2,color=s['color'],marker=s['marker'],ls=s['linestyle'],mfc=s.get('markerface',s['color']),label=s['legend'])
        ax.axhline(0,color='#999999',lw=.7);ax.set_xticks([-3,0,3]);ax.set_xlabel('Tx SNR offset (dB)');ax.set_title(f'True SNR = {snr} dB',fontsize=8.5);ax.grid(axis='y',color='#E6E6E6',lw=.6)
    axes[0].set_ylabel('PSNR change from matched (dB)')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False)
    fig.text(.53,.02,'Post-hoc first100 common500 sources · N = 1024',ha='center',fontsize=7.5)
    export(fig,folder,'s11_one_bin_tx_mismatch')
    metadata(folder,'S11',rows,'One-bin transmitter configuration mismatch on the first100 already-used common500 '
        'sources, three noises per source. Offset is lookup SNR minus true SNR; the receiver uses the true noise '
        'variance. Differences use the same-method matched configuration at the same true SNR as reference. '
        'Published paired intervals are reused. This is not a model of typical deployment error. Raw body CRC '
        'rejection keeps hard tokens and is not a gray-frame event; no protection threshold was retuned.')
    # A compact frozen raw configuration table; entropy has distinct framing.
    rows=[r for r in read_csv('results/wcl_evidence_closure_20261009/T4_resources_cost/summary_completed_v2/resource_summary.csv') if r['method'] in ['RAW_WHOLE','RAW_PARTIAL']]
    assert len(rows)==12
    folder=OUT/'S08';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(7,2.75));fig.subplots_adjust(left=.035,right=.99,bottom=.12,top=.82,wspace=.12)
    for ax,method,key in zip(axes,['RAW_WHOLE','RAW_PARTIAL'],['whole_raw','nest_raw']):
        rr=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['snr_db']))
        ax.axis('off');ax.set_title(METHODS[key]['legend'],fontsize=8.5)
        table=ax.table(cellText=[[r[k] for k in ['snr_db','m','K','body_modulation','ldpc_effective_rate_fraction']] for r in rr],colLabels=['SNR (dB)','m','K','Modulation','k/n'],cellLoc='center',loc='center',colWidths=[.22,.10,.12,.25,.25])
        table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.3)
        for (i,j),cell in table.get_celld().items():
            cell.set_linewidth(.35);cell.set_edgecolor('#BBBBBB')
            if i==0:cell.set_facecolor('#EEF3F8')
    export(fig,folder,'s08_frozen_raw_configurations')
    metadata(folder,'S08',rows,'Frozen raw configurations for N1024. m is the number of complete scales and K '
        'the next-scale raster-prefix token count. k/n is the actual LDPC information-to-transmitted-bit ratio, '
        'not a nominal rate. Whole and partial modes share modulation/coding permissions. This raw table does '
        'not assign its68-complex-use header to entropy-coded modes.')
    fig,axes=plt.subplots(1,2,figsize=(7,2.8));fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.72,wspace=.35)
    for ax,method,key in zip(axes,['RAW_WHOLE','RAW_PARTIAL'],['whole_raw','nest_raw']):
        rr=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['snr_db']))
        bottom=np.zeros(6);xs=[int(r['snr_db']) for r in rr]
        for field,label,color in [('header_complex_uses','Protected header','#426B8C'),('body_transmitted_complex_uses','Coded payload','#A9C1D3'),('idle_complex_uses','Known padding','#D8D8D8')]:
            v=np.array([float(r[field]) for r in rr]);ax.bar(xs,v,bottom=bottom,width=1.6,label=label,color=color,edgecolor='white',lw=.4);bottom+=v
        assert np.array_equal(bottom,np.full(6,1024));ax.set_xticks(xs);ax.set(xlabel='SNR (dB)',ylim=(0,1100));ax.set_title(METHODS[key]['legend'],fontsize=8)
    axes[0].set_ylabel('Complex channel uses per image');fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='upper center',frameon=False,fontsize=7.5)
    export(fig,folder,'s08_raw_frame_accounting')
    # Refresh provenance after the second independent export.
    DELIVERED.pop();metadata(folder,'S08',rows,(folder/'caption.tex').read_text()+' Frame bars show actual channel-use accounting; all components sum to1024.')
    # Only the two requested perception metrics available in the completed reference suite.
    rel='results/external_comparison_20261004/reference_metrics_cpu_full/summary.csv'
    rows=[r for r in read_csv(rel) if r['metric'] in ['lpips_alex','dinov2_vitl14_cosine']]
    assert len(rows)==12 and all(r['sources']=='100' for r in rows)
    folder=OUT/'S13';folder.mkdir(parents=True,exist_ok=True)
    order=['unrelated','same_class','gaussian_blur_sigma1','gaussian_noise_sigma2_255','jpeg_quality90','identity']
    names=['Unrelated\nimage','Same-class\nimage','Gaussian\nblur','Mild Gaussian\nnoise','JPEG\nquality90','Identical\nimage']
    fig,axes=plt.subplots(1,2,figsize=(7,3.0));fig.subplots_adjust(left=.09,right=.985,bottom=.38,top=.94,wspace=.28)
    for ax,m in zip(axes,[METRICS[1],METRICS[2]]):
        rr=[next(r for r in rows if r['condition']==c and r['metric']==m['key']) for c in order]
        y=np.array([float(r['mean']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr])
        ax.errorbar(range(6),y,yerr=[y-lo,hi-y],fmt='o',color='#0F4D92',capsize=2)
        ax.set_xticks(range(6),names,fontsize=7.5,rotation=35,ha='right');ax.set_ylabel(m['label']);ax.grid(axis='y',color='#E6E6E6',lw=.6)
    export(fig,folder,'s13_metric_reference_levels')
    metadata(folder,'S13',rows,'Published metric-reference means and intervals on100 fixed development originals, '
        'one observation per condition/source, no channel or noise triplication. Donors were fixed before '
        'scoring and same-class donors came from training/calibration. These are metric-scale references, '
        'not universal semantic safety thresholds. LPIPS-Alex and DINOv2-L ViT-L/14 are shown. ConvNeXt was not '
        'part of this reference suite, and the historical ResNet agreement is not relabeled as ConvNeXt.')

def main():
    global REVIEW
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['all','qualitative'],default='all');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);style()
    if a.only=='all':
        with PdfPages(OUT/'figure_review_print_size.pdf') as review:
            REVIEW=review
            system();curves();training_figures();bottleneck();lengths();qualitative();extra_published()
        REVIEW=None
    else:qualitative()
    write_json(OUT/('delivery_index.json' if a.only=='all' else 'qualitative_delivery_index.json'),DELIVERED)
    pending=[dict(id='MT01',status='WAITING_FOR_ORIGINAL_EXCLUSIVE_TIMING_WINDOW'),dict(id='S14',status='WAITING_FOR_ACTUAL_UNIFIED_TIMING'),
             dict(id='S01',status='OPTIONAL_VERSION_CHECK',reason='Source A reports historical DINO; not relabeled DINOv2-L. Current mechanism is illustrated in M01.'),
             dict(id='S17',status='OPTIONAL_PROTOCOL_MATCH_CHECK',reason='No additional frozen ablation curves are mixed into current-method attribution without matching protocol evidence.')]
    write_json(OUT/'pending.json',pending)
    table=OUT/'MT01';table.mkdir(exist_ok=True)
    (table/'README.md').write_text('Numerical table pending actual six-method timing under the original all-GPU '
        'exclusive-resource gate. Column layout: Method | Transmitter(ms) | Receiver(ms) | Software end-to-end(ms). '
        'Use19dB for every method; full4/10/19dB breakdown accompanies it. No old-device times or numeric '
        'placeholders are supplied. Registration is not completion.\n',encoding='utf-8')
    (OUT/'README.md').write_text('# NeST-Com paper figure redraw v1\n\n'
        'All numerical figures read full-precision published CSV values and existing intervals. No new training, '
        'model selection, scoring or bootstrap is performed by this script. Every figure folder contains PDF, '
        'editable-text SVG, 600dpi PNG, grayscale proof, plot_data.csv, caption.tex and provenance.json.\n\n'
        'M04 separately binds eight actual final-checkpoint inference frames, preserving the chosen sources and '
        'first-noise rule. Old HiFi examples remain historical and are not relabeled as final baselines.\n\n'
        'MT01 and S14 require actual same-environment timing under the original exclusive-resource gate; no '
        'publication-ready numerical table is fabricated while waiting. The review PDF contains figures at '
        'their specified physical width; it is a figure proof, not a compiled five-page manuscript.\n\n'
        'Reproduce from the repository root:\n\n```sh\npython scripts/plot_paper_redraw_v1.py\n```\n',encoding='utf-8')
    print(json.dumps(DELIVERED,indent=2))

if __name__=='__main__':main()
