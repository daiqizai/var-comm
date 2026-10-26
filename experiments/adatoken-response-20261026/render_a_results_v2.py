"""Deterministic publication plots of closed CSVs; no statistics or model calls."""
from pathlib import Path
import argparse,csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
METRICS=('psnr_db','lpips_alex','dreamsim','clip_image_cosine')
LABELS=('PSNR (dB) ↑','LPIPS-Alex ↓','DreamSim ↓','CLIP-I ↑')
ARMS=('WHOLE','HALF','QUARTER','EIGHTH')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8')))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main(root):
 data=root/'results/adatoken_response_20261026_v1/a_delivery';out=root/'results/adatoken_response_20261026_v1/figures_a_v2';out.mkdir(exist_ok=False)
 summary=rows(data/'summary.csv');paired=rows(data/'paired_differences.csv');assert len(summary)==48 and len(paired)==72
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
 files=[];plotted=[]
 def export(fig,name):
  fig.canvas.draw()
  for ax in fig.axes:
   ticks=[t for x,t in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0]<=x<=ax.get_xlim()[1]]+[t for y,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=y<=ax.get_ylim()[1]]
   for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ticks]:
    if not t.get_visible()or not t.get_text():continue
    b=t.get_window_extent(fig.canvas.get_renderer());assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(name,t.get_text(),b)
  for suffix in('pdf','svg','png'):
   p=out/(name+'.'+suffix);fig.savefig(p,dpi=600);files.append(pin(p))
  p=out/(name+'_print_preview.png');fig.savefig(p,dpi=150);files.append(pin(p));plt.close(fig)
 for grayscale in(False,True):
  fig,axs=plt.subplots(2,2,figsize=(178/25.4,110/25.4));fig.subplots_adjust(left=.105,right=.98,bottom=.105,top=.855,wspace=.28,hspace=.47)
  for i,(ax,metric,label)in enumerate(zip(axs.flat,METRICS,LABELS)):
   for snr,color,marker,ls in [(4,'#767676','s','--'),(10,'#3775BA','o','-'),(19,'#5E966B','^','-.')]:
    rs=[next(v for v in summary if v['metric']==metric and v['arm']==a and int(v['snr_db'])==snr)for a in ARMS];y=np.array([float(v['mean'])for v in rs]);lo=np.array([float(v['ci_low'])for v in rs]);hi=np.array([float(v['ci_high'])for v in rs]);assert np.all(lo<=y)and np.all(y<=hi)
    ax.errorbar(np.arange(4),y,yerr=[y-lo,hi-y],label=f'{snr} dB',color='#272727'if grayscale else color,marker=marker,linestyle=ls,linewidth=1.5,markersize=4.2,capsize=2,elinewidth=.85,markerfacecolor='white'if snr==4 else('#272727'if grayscale else color),zorder=3)
    if not grayscale:plotted.extend([dict(figure='a_quality_all_snr',**v)for v in rs])
   ax.set_xticks(range(4),['Whole','1/2','1/4','1/8']);ax.set_xlim(-.2,3.2);ax.set_title(f'({chr(97+i)}) {label}',loc='left',pad=7);ax.grid(axis='y',color='#E8E8E8',linewidth=.5,zorder=0)
   if i>=2:ax.set_xlabel('Endpoint set')
  fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,.99))
  if grayscale:
   p=out/'a_quality_all_snr_grayscale.png';fig.savefig(p,dpi=150);files.append(pin(p));plt.close(fig)
  else:export(fig,'a_quality_all_snr')
 fig,axs=plt.subplots(2,2,figsize=(178/25.4,106/25.4));fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.90,wspace=.31,hspace=.54)
 comps=('HALF_minus_WHOLE','QUARTER_minus_HALF','EIGHTH_minus_QUARTER')
 for i,(ax,metric,label)in enumerate(zip(axs.flat,METRICS,LABELS)):
  rs=[next(v for v in paired if v['metric']==metric and v['comparison']==c and int(v['snr_db'])==10)for c in comps];y=np.array([float(v['mean'])for v in rs]);lo=np.array([float(v['ci_low'])for v in rs]);hi=np.array([float(v['ci_high'])for v in rs])
  ax.axhline(0,color='#767676',linestyle='--',linewidth=.7,zorder=1);ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='o',color='#3775BA',markersize=4.5,capsize=3,elinewidth=1,zorder=3);ax.set_xticks(range(3),['Half − whole','Quarter − half','Eighth − quarter']);ax.set_xlim(-.35,2.35);ax.set_title(f'({chr(97+i)}) Δ{label}',loc='left',pad=7)
  low=min(0,lo.min());high=max(0,hi.max());pad=(high-low)*.13;ax.set_ylim(low-pad,high+pad);ax.grid(axis='y',color='#E8E8E8',linewidth=.5,zorder=0);plotted.extend([dict(figure='a_incremental_10db',**v)for v in rs])
 export(fig,'a_incremental_10db')
 with(out/'plotted_values.json').open('x')as f:json.dump(plotted,f,indent=2,allow_nan=False)
 with(out/'manifest.json').open('x')as f:json.dump(dict(inputs=[pin(data/n)for n in('summary.csv','paired_differences.csv','efficiency.csv')],script=pin(Path(__file__)),dimensions_mm={'a_quality_all_snr':[178,110],'a_incremental_10db':[178,106]},fonts_pt={'body':8.5,'ticks':8,'panel':9},source_statistics_recomputed=False,new_inference=False,files=files),f,indent=2)
 print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
