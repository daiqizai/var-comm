"""Frozen descriptive B1 development plots, no inference or resampling."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=False)
    def rows(name):return list(csv.DictReader((root/name).open(encoding='utf8')))
    style={'F0_FINE_TOKEN':('Fine-token F0','#4D4D4D','s','--'),'R_scalar':('Scalar residual','#B64342','^','-.'),'R_progressive':('Progressive residual','#0F4D92','o','-')}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=300,facecolor='white')
        plt.close(fig)
    summary=rows('source_rd_summary.csv');fig,axes=plt.subplots(1,3,figsize=(10.3,3.3));handles=[]
    for method,(label,color,marker,ls)in style.items():
        vv=sorted([v for v in summary if v['method']==method],key=lambda v:float(v['budget']))
        for ax,metric,ylabel in zip(axes,['psnr_db','lpips_alex','D'],['PSNR (dB) ↑','LPIPS-Alex ↓','MSE + 0.1 LPIPS ↓']):
            h,=ax.plot([float(v['budget'])for v in vv],[float(v[metric])for v in vv],label=label,color=color,marker=marker,ls=ls,lw=1.4,ms=5)
            ax.set_xlabel('Net source capacity (bits)');ax.set_ylabel(ylabel);ax.set_xticks([449,547,608]);ax.grid(axis='y',alpha=.18)
            if ax is axes[0]:handles.append(h)
    fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,1),fontsize=9);fig.suptitle('Selected checkpoints · 200 development sources · actual terminated streams',fontsize=9,y=.88);fig.subplots_adjust(left=.07,right=.99,bottom=.18,top=.75,wspace=.36);save(fig,'source_budget_comparison')
    curve=rows('checkpoint_curve.csv');anchor=rows('anchor_rd.csv');fig,axes=plt.subplots(1,2,figsize=(9.5,3.6));handles=[]
    for order,color,marker in [('scalar','#B64342','^'),('progressive','#0F4D92','o')]:
        vv=[v for v in curve if v['order']==order];steps=sorted({int(v['step'])for v in vv});ds=[np.mean([float(v['D'])for v in vv if int(v['step'])==step])for step in steps]
        h,=axes[0].plot(steps,ds,color=color,marker=marker,label=order.capitalize());handles.append(h)
        aa=sorted([v for v in anchor if v['order']==order],key=lambda v:int(v['j']));axes[1].plot([float(v['actual_bits'])for v in aa],[float(v['D'])for v in aa],color=color,marker=marker,ms=3.5,lw=1)
        for j in [0,64,128,192,256]:
            v=next(v for v in aa if int(v['j'])==j);axes[1].annotate(str(j),(float(v['actual_bits']),float(v['D'])),xytext=(3,3 if order=='scalar'else-9),textcoords='offset points',fontsize=7)
    axes[0].set(xlabel='Training updates',ylabel='Development selection objective ↓',title='(a) All five checkpoint evaluations');axes[0].set_xticks([1000,2000,3000,4000,5000]);axes[1].set(xlabel='Mean actual total AC bits',ylabel='MSE + 0.1 LPIPS ↓',title='(b) Fixed 16-source, 17-anchor diagnostic')
    for ax in axes:ax.grid(axis='y',alpha=.18)
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.5,1));fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.8,wspace=.28);save(fig,'training_and_anchor_rd')
    extra=rows('diagnostic_five_metric_summary.csv');fig,axes=plt.subplots(1,3,figsize=(10.3,3.3));handles=[]
    for method,(label,color,marker,ls)in style.items():
        vv=sorted([v for v in extra if v['method']==method],key=lambda v:float(v['budget']))
        for ax,key,labely in zip(axes,['dinov2_vitl14_cosine','dreamsim','clip_image_cosine'],['DINOv2 cosine ↑','DreamSim ↓','CLIP-I cosine ↑']):
            h,=ax.plot([float(v['budget'])for v in vv],[float(v[key])for v in vv],color=color,marker=marker,ls=ls,label=label,ms=5,lw=1.4);ax.set(xlabel='Net source capacity (bits)',ylabel=labely);ax.set_xticks([449,547,608]);ax.grid(axis='y',alpha=.18)
            if ax is axes[0]:handles.append(h)
    fig.legend(handles=handles,loc='upper center',ncol=3,bbox_to_anchor=(.5,1));fig.suptitle('Fixed first 16 development sources · exploratory diagnostics, no confidence intervals',fontsize=9,y=.88);fig.subplots_adjust(left=.07,right=.99,bottom=.18,top=.75,wspace=.36);save(fig,'diagnostic_metrics')
    (out/'provenance.json').write_text(json.dumps(dict(inputs={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in ['source_rd_summary.csv','checkpoint_curve.csv','anchor_rd.csv','diagnostic_five_metric_summary.csv']},bootstrap='none',axis_values='raw CSV, no jitter or smoothing',anchors='17 preset j values; not all257 image decodes',formats=['PDF','SVG editable text','PNG300dpi']),indent=2))
if __name__=='__main__':
    import sys
    main(sys.argv[1])
