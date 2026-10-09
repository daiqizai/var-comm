"""Export existing Kodak CSV statistics; never bootstrap or run any model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

METHODS=('VAR_UNCONDITIONAL','P1024','BPG_ADAPTIVE','SWIN80K')
METRICS=('psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction')
SNRS=(4,10,19)
DISPLAY={
 'VAR_UNCONDITIONAL':('Partial-scale digital + unconditional VAR','#0072B2','-','o'),
 'P1024':('Latent continuous JSCC','#D55E00','--','s'),
 'BPG_ADAPTIVE':('Adaptive-resolution BPG + LDPC','#CC79A7',(0,(5,1,1,1)),'v'),
 'SWIN80K':('SwinJSCC-80k (adapted)','#009E73','-.','^'),
}
PANELS=(('PSNR','PSNR (dB) ↑','ΔPSNR (dB) ↑',1),('LPIPS','LPIPS ↓','ΔLPIPS ↓',1),
 ('DINOv2-L','Cosine similarity ↑','ΔDINOv2-L ↑',1),
 ('ConvNeXt','Source-prediction agreement (%) ↑','ΔSource-prediction agreement\n(percentage points) ↑',100))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def csv_read(path):
    with Path(path).open(encoding='utf8',newline='')as f:rows=list(csv.DictReader(f))
    for row in rows:
        row['snr_db']=int(row['snr_db'])
        for key in ('mean','ci_low','ci_high'):row[key]=float(row[key])
    return rows
def require(ok,message):
    if not ok:raise ValueError(message)

def run(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    data,out=Path(data).resolve(),Path(out).resolve()
    require(not out.exists(),'Use a new figure directory; preserve prior scientific results')
    done=read(data/'completion.json');require(done['status']=='KODAK24_SOURCE_STATISTICS_AND_PLOTS_COMPLETE_V1','Completed statistics required')
    tables={};inputs={str(data/'completion.json'):sha(data/'completion.json')}
    for name,expected in [('summary',48),('paired',36)]:
        path=data/(name+'.csv');matches=[h for p,h in done['outputs'].items()if Path(p).name==path.name]
        require(len(matches)==1 and sha(path)==matches[0],'Existing statistic hash changed')
        inputs[str(path)]=matches[0];tables[name]=csv_read(path);require(len(tables[name])==expected,'Exact table size')
        keys=set()
        for row in tables[name]:
            key=(row['method'],row.get('reference',''),row['snr_db'],row['metric'])
            require(key not in keys and row['metric']in METRICS and row['snr_db']in SNRS,'Duplicate or unexpected row');keys.add(key)
            require(int(row['source_count'])==24 and int(row['noise_count'])==3 and int(row['frame_count'])==72
                and np.isfinite([row[k]for k in ('mean','ci_low','ci_high')]).all()
                and row['ci_low']<=row['mean']<=row['ci_high'],'Scope/interval changed')
            if name=='paired':require(row['method']=='VAR_UNCONDITIONAL'and row['delta_definition']=='method minus reference','Unchanged paired direction')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,
        'figure.facecolor':'white','savefig.facecolor':'white','path.simplify':False})
    plot_data=[];layout=[]
    for kind,table in [('main',tables['summary']),('paired',tables['paired'])]:
        paired=kind=='paired';fig,axes=plt.subplots(2,2,figsize=(7,5.2),dpi=150)
        fig.subplots_adjust(left=.12,right=.985,bottom=.10,top=.84,wspace=.40,hspace=.54)
        shown=METHODS[1:]if paired else METHODS
        handles=[Line2D([],[],label=('VAR − '+DISPLAY[m][0])if paired else DISPLAY[m][0],color=DISPLAY[m][1],
            linestyle=DISPLAY[m][2],marker=DISPLAY[m][3],linewidth=1.3,markersize=4)for m in shown]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=1 if paired else 2,
            fontsize=7.5,frameon=False,handlelength=2.6,columnspacing=1.2,labelspacing=.35)
        for ax,metric,panel,letter in zip(axes.flat,METRICS,PANELS,'abcd'):
            title,ylabel,dlabel,scale=panel;lows=[];highs=[]
            for method in shown:
                key='reference'if paired else'method'
                group=sorted([r for r in table if r[key]==method and r['metric']==metric],key=lambda r:r['snr_db'])
                require([r['snr_db']for r in group]==list(SNRS),'Exactly three numerical SNR points')
                x=np.asarray(SNRS);y=np.asarray([r['mean']for r in group])*scale
                lo=np.asarray([r['ci_low']for r in group])*scale;hi=np.asarray([r['ci_high']for r in group])*scale
                lows.extend(lo);highs.extend(hi);_,color,line,marker=DISPLAY[method]
                if paired:
                    ax.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),color=color,linestyle='None',
                        marker=None,capsize=3,elinewidth=.9,zorder=2)
                else:ax.fill_between(x,lo,hi,color=color,alpha=.12,linewidth=0,zorder=1)
                if method=='SWIN80K':
                    ax.plot(x[:2],y[:2],color=color,linestyle=line,marker=marker,markersize=4,linewidth=1.3,zorder=3)
                    ax.plot(x[1:],y[1:],color=color,linestyle='--',linewidth=1.3,zorder=3)
                    ax.plot(x[-1:],y[-1:],color=color,linestyle='None',marker=marker,markersize=5,
                        markerfacecolor='white',markeredgewidth=1.2,zorder=4)
                else:ax.plot(x,y,color=color,linestyle=line,marker=marker,markersize=4,linewidth=1.3,zorder=3)
                plot_data.extend(dict(figure=kind,**r,display_scale=scale,plot_mean=r['mean']*scale,
                    plot_ci_low=r['ci_low']*scale,plot_ci_high=r['ci_high']*scale)for r in group)
            if paired:ax.axhline(0,color='#606060',linestyle='--',linewidth=.8);lows.append(0);highs.append(0)
            low,high=min(lows),max(highs);span=high-low if high>low else max(abs(high)*.1,.01)
            ax.set_ylim(low-.09*span,high+.09*span);ax.set_xlim(3.3,19.7);ax.set_xticks(SNRS);ax.set_xlabel('SNR (dB)')
            ax.set_ylabel(dlabel if paired else ylabel);ax.set_title(f'({letter}) {title}',loc='left',fontsize=9,pad=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.grid(color='#DFE3E6',linewidth=.55);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
        texts=[t for ax in fig.axes for t in [ax.xaxis.label,ax.yaxis.label,ax._left_title,*ax.get_xticklabels(),*ax.get_yticklabels()]]
        boxes=[t.get_window_extent(renderer)for t in texts if t.get_visible()and t.get_text()]
        boxes.extend(l.get_window_extent(renderer)for l in fig.legends)
        require(all(b.x0>=-.5 and b.y0>=-.5 and b.x1<=bounds.x1+.5 and b.y1<=bounds.y1+.5 for b in boxes),'Text clipping: '+kind)
        for ext in ('pdf','svg','png'):fig.savefig(out/f'kodak24_N1024_{kind}.{ext}',dpi=600 if ext=='png'else 150)
        layout.append(dict(figure=kind,text_bounds_pass=True,Swin_19dB_hollow=True,Swin_10_to_19dB_dashed=True))
        plt.close(fig)
    fields=list(dict.fromkeys(k for row in plot_data for k in row))
    with(out/'plot_data.csv').open('w',encoding='utf8',newline='')as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(plot_data)
    captions=(data/'captions.tex').read_text(encoding='utf8')
    old="Swin's 19 dB reference remains outside its training/calibration range."
    require(old in captions,'Expected existing caption statement')
    captions=captions.replace(old,old+' The Swin-referenced contrast also uses a hollow 19 dB marker and a dashed 10--19 dB segment.')
    (out/'captions.tex').write_text(captions,encoding='utf8')
    readme='''# Kodak24 paper figures

These figures read the completed 48 summary rows and 36 source-paired differences from `results/generalization_kodak_20261009/analysis_v1`. No bootstrap, model call, channel draw, policy selection, or change to original results occurs. Both figures preserve Swin's 19 dB point using a hollow marker and a dashed 10-to-19 dB segment because that point is outside its training/calibration range.

All 24 predetermined center crops, three noises per source, and failure outputs are included. DINO-L uses dinov2_vitl14_cosine. ConvNeXt is source-prediction agreement, displayed as percent; paired increments are percentage points. LPIPS differences remain proposed minus reference. The small crop benchmark and pointwise intervals do not establish universal superiority or training-data non-overlap.

Files: main and paired 2x2 figures, each vector PDF/editable-text SVG/600-dpi PNG, plus captions and actual plotting rows. The original analysis exports and completion are preserved. Figures still require actual PNG visual inspection, recorded separately.

Reproduce into a new directory:

`python experiments/generalization_kodak_20261009/plot_only.py --data results/generalization_kodak_20261009/analysis_v1 --out NEW_FIGURE_DIRECTORY`
'''
    (out/'README.md').write_text(readme,encoding='utf8')
    formats={}
    for kind in ('main','paired'):
        pdf=out/f'kodak24_N1024_{kind}.pdf';svg=out/f'kodak24_N1024_{kind}.svg'
        root=ET.parse(svg).getroot();ns='{http://www.w3.org/2000/svg}'
        counts=dict(pdf_image_xobjects=pdf.read_bytes().count(b'/Subtype /Image'),
            svg_image_elements=len(root.findall('.//'+ns+'image')),svg_text_elements=len(root.findall('.//'+ns+'text')))
        require(counts['pdf_image_xobjects']==counts['svg_image_elements']==0 and counts['svg_text_elements']>0,'True vector/text export required')
        formats[kind]=counts
    require(all(sha(p)==h for p,h in inputs.items()),'Original input changed')
    outputs={str(p):sha(p)for p in out.iterdir()if p.is_file()}
    (out/'validation.json').write_text(json.dumps(dict(status='COMPLETE_PLOT_ONLY',inputs=inputs,outputs=outputs,
        script_sha256=sha(__file__),source_count=24,noise_count=3,plotted_rows=84,layout=layout,formats=formats,
        new_bootstrap_calls=0,new_model_calls=0,new_PHY_calls=0,new_channel_draws=0,
        original_analysis_unchanged=True,actual_PNG_review='PENDING'),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='COMPLETE_PLOT_ONLY',formats=6,plotted_rows=84,out=str(out))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();run(args.data,args.out)
