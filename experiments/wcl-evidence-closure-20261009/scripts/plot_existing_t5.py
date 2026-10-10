#!/usr/bin/env python3
"""Render already completed, SHA-bound data and images; no scientific execution."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from PIL import Image

COMMIT='252176e041758ecb2d3e81b6fde5b587e7e17bb7'
SNRS=(1,4,7,10,13,19)
FIXED=(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
METRICS=(('psnr_db','PSNR','PSNR (dB) ↑','ΔPSNR (dB) ↑',1,'psnr'),
 ('lpips_alex','LPIPS-Alex','LPIPS ↓','ΔLPIPS ↓',1,'lpips'),
 ('dinov2_vitl14_cosine','DINOv2 ViT-L/14','Cosine similarity ↑','ΔDINOv2-L ↑',1,'dinov2_l'),
 ('convnext_top1_source_prediction','ConvNeXt prediction agreement','Prediction agreement (%) ↑','ΔPrediction agreement\n(percentage points) ↑',100,'convnext_agreement'))
METHODS=(('RAW64_WHOLE_VAR_COMPLETION','Complete-scale raw + VAR','#555555','--','s'),
 ('RAW64_PARTIAL_VAR_COMPLETION','Partial-scale raw + VAR (NeST-Com)','#0072B2','-','o'))
EXPECTED={'summary.csv':'5899fe10ccf2e7fe93ef36c59b9e14c6622f542d58ee1c0bbf0c8332a5577859',
 'paired.csv':'dd8221b1fe8aa9d86f9cbf78ea281fd2da860e2ffdcd6fb83daa487eebc0c235',
 'points.json':'1efbbf1cc4d1dc4359f005bc234a324e830072dc4e1a4b57b5ad4eac92fbc9ce',
 'pairs.json':'524f95dfec6278bdfe9b8fc03765bd2f678854816e1ad2e73a38bb854a485468'}
PROVENANCE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,expected=None):
    p=Path(p);h=sha(p)
    assert expected is None or h==expected,f'Input hash differs: {p}'
    PROVENANCE[str(p)]={'sha256':h,'bytes':p.stat().st_size};return p
def read(p):return json.loads(bind(p).read_text(encoding='utf-8'))
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def readcsv(p):
    with bind(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(p,rows):
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def one(xs):
    assert len(xs)==1,f'Expected one matched record, found {len(xs)}';return xs[0]
def locate(root,rel,local):
    for p in [root/rel,root/local]:
        if p.exists():return p
    raise FileNotFoundError(rel)
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
      'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
      'axes.linewidth':.6,'grid.linewidth':.5,'grid.color':'#E3E3E3',
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'wcl-t5-'+COMMIT,
      'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
      'axes.unicode_minus':True,'path.simplify':False})
def savefig(fig,out,stem):
    # Lines, confidence bands and text remain vector in PDF/SVG; photos are raster.
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'{stem}.{ext}',dpi=600 if ext=='png' else 150,facecolor='white')

def gather_curves(root):
    data=locate(root,'results/main_raw64_20261007/final_common500_r6',
      '.research/main_raw64_20261007/take_over_v1/final_publication_r6/actual_staging_r6/results/main_raw64_20261007/final_common500_r6')
    for name,h in EXPECTED.items():bind(data/name,h)
    summary=readcsv(data/'summary.csv');paired=readcsv(data/'paired.csv')
    points=read(data/'points.json');pairs=read(data/'pairs.json')
    pairkeys={(p['method'],p['reference']) for p in (pairs.values() if isinstance(pairs,dict) else pairs)}
    selected=[]
    for metric,_,_,_,scale,_ in METRICS:
        for snr in SNRS:
            for family,label,*_ in METHODS:
                method=f'{family}_SNR_{snr}'
                r=one([r for r in summary if r['point_id']==method and r['metric']==metric])
                assert int(r['snr_db'])==snr and method in points
                assert (int(r['source_count']),int(r['noise_count']),int(r['frame_count']))==(500,3,1500)
                assert Decimal(r['ci_low'])<=Decimal(r['mean'])<=Decimal(r['ci_high'])
                selected.append(dict(r,source_csv='summary.csv',figure='same_prior',method=method,reference='',public_method=label,display_scale=str(scale),**{f'plot_{k}':str(Decimal(r[k])*scale) for k in ('mean','ci_low','ci_high')}))
            method=f'RAW64_PARTIAL_VAR_COMPLETION_SNR_{snr}';ref=f'RAW64_WHOLE_VAR_COMPLETION_SNR_{snr}'
            r=one([r for r in paired if r['method']==method and r['reference']==ref and r['metric']==metric])
            assert (method,ref) in pairkeys and r['delta_definition']=='method minus reference'
            assert (int(r['source_count']),int(r['noise_count']),int(r['frame_count']))==(500,3,1500)
            assert Decimal(r['ci_low'])<=Decimal(r['mean'])<=Decimal(r['ci_high'])
            selected.append(dict(r,source_csv='paired.csv',figure='paired',public_method='Partial-scale minus complete-scale',display_scale=str(scale),**{f'plot_{k}':str(Decimal(r[k])*scale) for k in ('mean','ci_low','ci_high')}))
    assert all(Decimal(r['mean'])==Decimal(r['ci_low'])==Decimal(r['ci_high'])==0 for r in selected if r['figure']=='paired' and int(r['snr_db'])==7)
    return selected

def draw_panel(ax,rows,metric,paired):
    key,title,ylabel,delta,scale,slug=metric
    allvals=[]
    for family,label,color,ls,marker in (METHODS[1:] if paired else METHODS):
        r=[one([r for r in rows if r['figure']==('paired' if paired else 'same_prior') and r['metric']==key and int(r['snr_db'])==snr and r['method']==f'{family}_SNR_{snr}']) for snr in SNRS]
        y,lo,hi=[np.array([float(z[f'plot_{k}']) for z in r]) for k in ('mean','ci_low','ci_high')]
        allvals.extend(lo);allvals.extend(hi)
        if paired:ax.errorbar(SNRS,y,yerr=[y-lo,hi-y],color=color,lw=1.3,marker=marker,ms=4.2,capsize=2.8,elinewidth=1.0,zorder=3)
        else:
            ax.fill_between(SNRS,lo,hi,color=color,alpha=.13,linewidth=0)
            ax.plot(SNRS,y,color=color,ls=ls,marker=marker,lw=1.35,ms=4.2,label=label,zorder=3)
    if paired:ax.axhline(0,color='#777777',ls='--',lw=.8,zorder=1);allvals.append(0)
    lo,hi=min(allvals),max(allvals);pad=max((hi-lo)*.12,1e-5)
    ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(.2,19.8);ax.set_xticks(SNRS)
    ax.set_xlabel('SNR (dB)');ax.set_ylabel(delta if paired else ylabel);ax.set_title(title,pad=7)
    ax.grid(True,alpha=1);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)

def plot_curves(rows,out):
    for paired,stem in [(False,'fig_t5_same_prior_N1024'),(True,'fig_t5_partial_minus_complete')]:
        fig,axes=plt.subplots(2,2,figsize=(7,5.15))
        for ax,metric in zip(axes.flat,METRICS):draw_panel(ax,rows,metric,paired)
        title='Partial-scale minus complete-scale, N = 1024' if paired else 'Same-prior raw-token comparison, N = 1024'
        fig.suptitle(title,y=.982,fontsize=10)
        top=.82 if not paired else .875
        if not paired:
            handles=[Line2D([],[],color=c,ls=l,marker=m,lw=1.3,ms=4,label=name) for _,name,c,l,m in METHODS]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False)
        fig.subplots_adjust(left=.115,right=.985,bottom=.10,top=top,wspace=.38,hspace=.53)
        savefig(fig,out,stem);plt.close(fig)
        for metric in METRICS:
            fig,ax=plt.subplots(figsize=(3.5,2.75));draw_panel(ax,rows,metric,paired)
            if not paired:
                ax.legend(loc='upper center',bbox_to_anchor=(.5,1.36),frameon=False,fontsize=7.3,ncol=1,handlelength=2.5)
                ax.set_title(metric[1],pad=6);fig.subplots_adjust(left=.20,right=.97,bottom=.18,top=.71)
            else:fig.subplots_adjust(left=.24,right=.97,bottom=.18,top=.86)
            savefig(fig,out,stem+'_'+metric[-1]);plt.close(fig)

def display(x):
    assert x.dtype==np.float32 and x.shape==(3,256,256) and np.isfinite(x).all()
    assert x.min()>=0 and x.max()<=1
    return np.rint(x.transpose(1,2,0)*np.float32(255)).astype(np.uint8)

def gather_images(root):
    fixed=locate(root,'outputs/MAIN-RAW64-20261007/development_fixed16_cached_display_v2',
      '.research/main_raw64_20261007/take_over_v1/current/development_fixed16_cached_display_v2')
    fm=read(fixed/'examples_manifest.json');fc=read(fixed/'completion.json')
    assert fm['source_indices']==list(FIXED) and not fc['holdout_used'] and fc['all_reported_methods_frozen_scope']
    for remote,h in fc['outputs'].items():bind(fixed/remote.split('/development_fixed16_cached_display_v2/')[-1],h)
    bpgdir=root/'paper/figures/a5_adaptive_six_methods_N1024_13dB_group02'
    bpg=read(bpgdir/'collection.json');bind(bpgdir/'adaptive16_float_inputs.npz',bpg['adaptive_float_sha256'])
    assert bpg['fixed16']==list(FIXED) and bpg['old_five_columns_pixel_identical']
    with np.load(bpgdir/'adaptive16_float_inputs.npz',allow_pickle=False) as z:
        bpgimages=z['images'].copy();bpgrefs=z['references'].copy()
    assert bpgimages.shape==(16,3,256,256)
    arrays={};records=[]
    def png(p):
        with Image.open(p) as im:x=np.asarray(im.convert('RGB')).copy()
        assert x.shape==(256,256,3);return x
    def append(page,order,public,internal,snr,image,seed,frame=None,cache=None):
        row={'fixed_order':order+1,'source_index':page['source_index'],'source_id':page['source_id'],'population':'development','N':1024,'snr_db':snr,'public_method':public,'internal_mapping':internal,'noise_seed':seed,'cache':str(cache or ''),'display_uint8_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'reuse':'REUSE_EXACT','main_subset':order<4}
        if frame:
            row.update({k:frame.get(k,'') for k in ['image_sha256','actual_state_sha256','candidate_id','observation_sha256','noise_namespace']})
        records.append(row)
    for order,page in enumerate(fm['pages']):
        idx=page['source_index'];d=fixed/f'{order:02d}_source_{idx:04d}'
        original=png(d/'reference.png');arrays[(idx,'original','all')]=original
        append(page,order,'Original image','source','all',original,'',cache=d/'reference.png')
        assert np.array_equal(display(bpgrefs[order]),original),'BPG source pixels differ'
        br=one([r for r in bpg['rows'] if r['source_index']==idx]);assert br['source_id']==page['source_id'] and br['N']==1024 and br['snr_db']==13 and br['population']=='development'
        bimage=display(bpgimages[order]);arrays[(idx,'bpg',13)]=bimage
        append(page,order,'Adaptive BPG + LDPC','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024',13,bimage,br['noise_seed'],br,bpgdir/'adaptive16_float_inputs.npz')
        for internal,public,alias in [('WHOLE VAR','Complete-scale transmission with VAR completion','whole'),('PARTIAL direct Dc','Partial-scale transmission without completion','direct'),('PARTIAL VAR','Partial-scale transmission with VAR completion (NeST-Com)','partial'),('P baseline','Latent-space continuous JSCC','latent'),('Swin80k','SwinJSCC-80k (adapted)','swin')]:
            snrs=[1,10,13,19] if alias=='partial' else [1,10,19] if alias in ('whole','direct') else [13]
            for snr in snrs:
                frame=one([r for r in page['frames'] if r['column']==internal and r['snr_db']==snr])
                f=d/f'snr{snr}_{internal.replace(" ","_")}.png';image=png(f);arrays[(idx,alias,snr)]=image
                assert frame['noise_seed']==(6201 if alias in ('whole','direct','partial') else 2001)
                append(page,order,public,internal,snr,image,frame['noise_seed'],frame,f)
        for snr in (1,10,13,19):
            a=one([r for r in page['frames'] if r['column']=='PARTIAL VAR' and r['snr_db']==snr]);b=one([r for r in page['frames'] if r['column']=='PARTIAL direct Dc' and r['snr_db']==snr])
            assert a['actual_state_sha256']==b['actual_state_sha256'],'Partial arms do not share actual tokens'
    return arrays,records

def image_page(arrays,sources,snr,external,start):
    labels=['Original image','Adaptive BPG\n+ LDPC','SwinJSCC-80k\n(adapted)','Latent-space\ncontinuous JSCC','Partial-scale + VAR\n(NeST-Com)'] if external else ['Original image','Complete-scale\ntransmission + VAR','Partial-scale transmission\nwithout completion','Partial-scale transmission\n+ VAR (NeST-Com)']
    aliases=['original','bpg','swin','latent','partial'] if external else ['original','whole','direct','partial']
    nc=len(aliases);width=7.;left=.31;right=.05;gap=.045;rowgap=.05;top=.85;bottom=.12
    cell=(width-left-right-(nc-1)*gap)/nc;height=top+bottom+4*cell+3*rowgap
    fig=plt.figure(figsize=(width,height))
    fig.text(.52,1-.15/height,'Fixed development examples',ha='center',fontsize=9)
    fig.text(.52,1-.34/height,f'N = 1024, SNR = {snr} dB; original positions {start+1}–{start+4}',ha='center',fontsize=8.5)
    for c,label in enumerate(labels):
        fig.text((left+c*(cell+gap)+cell/2)/width,1-.66/height,label,ha='center',va='center',fontsize=8.2 if external else 8.1,linespacing=1.2)
    for r,idx in enumerate(sources):
        y=bottom+(3-r)*(cell+rowgap)
        fig.text(.145/width,(y+cell/2)/height,f'{start+r+1:02d}',ha='center',va='center',fontsize=8)
        for c,alias in enumerate(aliases):
            x=arrays[(idx,alias,'all' if alias=='original' else snr)]
            ax=fig.add_axes([(left+c*(cell+gap))/width,y/height,cell/width,cell/height])
            ax.imshow(x,interpolation='nearest',rasterized=True);ax.set_axis_off()
    return fig

def plot_images(arrays,out):
    for external,snr in [(False,1),(False,10),(False,19),(True,13)]:
        stem=('fig_t5_external' if external else 'fig_t5_mechanism')+f'_N1024_{snr}dB'
        with PdfPages(out/f'{stem}_all16.pdf') as pdf:
            for start in range(0,16,4):
                fig=image_page(arrays,FIXED[start:start+4],snr,external,start)
                savefig(fig,out,stem+f'_page{start//4+1:02d}');pdf.savefig(fig);plt.close(fig)

CAPTIONS=r'''% Direct reuse of existing estimates and registered qualitative images.
\newcommand{\WCLSamePriorCaption}{Complete-scale and partial-scale raw-token transmission with the same frozen VAR prior and decoder at $N=1024$, including paid auxiliary information. Means and pointwise 95\% confidence intervals are copied from the published 500-source holdout CSV, after each source averages its three frozen noise realizations. The existing 10,000-replicate source bootstrap is reused without resampling or multiplicity correction. PSNR, LPIPS-Alex, DINOv2 ViT-L/14 cosine similarity, and ConvNeXt prediction agreement with the source prediction are shown; agreement is not classification accuracy. Both policies have the same modulation and coding permissions, and partial-scale transmission may fall back to a complete-scale configuration. These are the original frozen raw-token policies, not the expanded T2 controls or the T1 entropy-coded baseline.}
\newcommand{\WCLPairedCaption}{Existing source-paired differences for partial-scale minus complete-scale raw-token transmission with VAR completion. Error bars are the published paired 95\% confidence intervals, not differences between single-method intervals. Positive differences favor partial-scale transmission except for LPIPS, where negative values are favorable. ConvNeXt agreement differences are percentage points. All six SNR points, the exact zero differences at 7~dB and the small or zero differences at 13~dB are retained. No original statistic was recomputed. The 19~dB agreement interval includes zero.}
\newcommand{\WCLMechanismCaption}{Previously registered fixed development examples at $N=1024$ and SNR 1, 10, or 19~dB. Columns show the original, complete-scale transmission with VAR completion, partial-scale transmission without completion, and partial-scale transmission with VAR completion (NeST-Com). Original source order and raw-channel seed 6201 are retained. The two partial-scale arms use the same actually received tokens and frozen decoder $D_c$. Without completion, missing residual contributions are zero; unknown tokens are not filled with codebook index zero and the receiver obtains no extra source truth. The first four registered entries form the main-text subset; all 16 entries are shown on four pages per SNR. These already-viewed development examples are not an independent holdout or a selection of favorable noise. The entropy-coded column will be added only after T1 has actual outputs.}
\newcommand{\WCLExternalCaption}{The same 16 registered development sources at $N=1024$ and 13~dB, shown in original order. Columns show the original, adaptive BPG + LDPC, SwinJSCC-80k (adapted), latent-space continuous JSCC, and partial-scale raw transmission with VAR completion (NeST-Com). Existing noise seeds are 2001 for the three external systems and 6201 for the raw-token system, with their original distinct physical noise namespaces. Matching source images and operating points do not imply identical received observations across different waveforms. No method-specific best-noise selection, enhancement, new inference, channel simulation, metric evaluation, or bootstrap was performed. The Swin result describes the fixed adapted 80k checkpoint, not an official optimum or proof of convergence. Missing external 10/19~dB examples are not replaced with another SNR. HiFi examples remain separate in their existing exact subset.}
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);ap.add_argument('--output-dir',type=Path,default=Path('results/wcl_evidence_closure_20261009/T5_figures'));args=ap.parse_args()
    root=args.root.resolve();out=args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
    if out.exists() and any(out.iterdir()):raise RuntimeError('Use an empty output directory; existing results are never overwritten')
    out.mkdir(parents=True,exist_ok=True);style()
    rows=gather_curves(root);arrays,samples=gather_images(root)
    csvout(out/'plot_data.csv',rows);csvout(out/'sample_manifest.csv',samples)
    plot_curves(rows,out);plot_images(arrays,out)
    (out/'captions.tex').write_text(CAPTIONS,encoding='utf-8')
    (out/'README.md').write_text('''# WCL T5: existing evidence, new presentation

Original common500 statistics: commit `252176e041758ecb2d3e81b6fde5b587e7e17bb7`. Supplemental cached images come from the science snapshot `14b09ecd72984fb39c683d1a62bcd3221c69142f`. No original scientific file is changed; all curves use original CSV precision and existing intervals.

`fig_t5_same_prior_N1024` explicitly shows the complete-scale and partial-scale raw+VAR curves. `fig_t5_partial_minus_complete` shows their existing paired differences. Each has four independent single-column panels. PDF/SVG data plots contain vector lines, bands, markers and text; SVG text remains editable. PNG files are 600dpi.

Mechanism samples cover1/10/19dB, with16 fixed development sources in four four-row pages perSNR. External five-column samples cover13dB on all16 of the same sources. A combined `all16.pdf` accompanies each set. Source photographs in PDF/SVG remain embedded raster images with vector titles. Main-text examples use the first four entries in the historical list, without outcome selection; full pages preserve all16. These are development examples, not holdout images. Internal method IDs are kept only in metadata/code.

`plot_data.csv` preserves raw input numbers and explicit display conversions. ConvNeXt prediction agreement is displayed as percent; differences are percentage points. LPIPS difference is never sign-flipped. `sample_manifest.csv` binds original order, sourceID, method, working point, seed, cached image and actual-state identity. Public labels contain no internal P/H/MAIN/WHOLE shorthand.

The complete external10/19dB fixed16 views are not generated because matching Swin/adaptiveBPG caches were not established locally. No13dB image substitutes for them. T1 entropy-coded examples await actual new output. Swin19dB remains out of training/calibration range wherever shown in the pre-existing external curves; no19dB Swin image is shown here.

Reproduce from the repository root into a new output directory:

```text
python experiments/wcl-evidence-closure-20261009/scripts/plot_existing_t5.py --output-dir results/wcl_evidence_closure_20261009/T5_figures_replot
```

The script reads standard repository/cache locations on the original host and known local synchronized locations on this workstation. All source bindings and outputs are in `completion.json`. It imports only numpy, matplotlib and Pillow; no model, PHY or bootstrap module is called.
''',encoding='utf-8')
    for path,binding in PROVENANCE.items():assert sha(path)==binding['sha256'],'Input changed during rendering'
    outputs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    write(out/'completion.json',{'status':'COMPLETE','script_sha256':sha(__file__),'original_statistics_commit':COMMIT,'latest_science_commit':'14b09ecd72984fb39c683d1a62bcd3221c69142f','source_bindings':PROVENANCE,'outputs':outputs,'plot_rows':len(rows),'sample_manifest_rows':len(samples),'fixed16':list(FIXED),'source_population':'development','curve_source_count':500,'curve_noise_count':3,'new_inference':0,'new_channel':0,'new_metric_calls':0,'new_bootstrap':0,'input_files_modified':False,'png_dpi':600,'svg_text_editable':True,'pdf_data_plots_are_vector':True,'matplotlib':matplotlib.__version__,'numpy':np.__version__})
    print(json.dumps({'status':'COMPLETE','files':len(outputs)+1,'plot_rows':len(rows),'sample_rows':len(samples),'output':str(out),'new_scientific_calls':0}))

if __name__=='__main__':main()
