"""Deterministic rendering of closed summaries only; no statistical computation."""
from shared import *
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis_data import METHODS

def main():
 done=read(O/'metrics_v1/completion.json');assert done['status']=='N2048_TWO_METHOD_COMPARISON_COMPLETE'
 for d in done['outputs']:verify(d)
 out=O/'figures_v1';out.mkdir()
 with (O/'metrics_v1/summary.csv').open()as f:rows=list(csv.DictReader(f))
 r=read(REG);jobs={j['key']:j for j in r['formal_jobs']};frames=read(O/'formal/logical_frames.json')
 style=dict(width_mm=180,height_mm=105,font='DejaVu Sans',axis_label_pt=8.5,tick_pt=7.5,title_pt=9,line_pt=1.5,colors={METHODS[0]:'#3775BA',METHODS[1]:'#B64342'},skill='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md')
 put(out/'style.json',style)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,axes=plt.subplots(2,4,figsize=(180/25.4,105/25.4));fig.subplots_adjust(left=.072,right=.988,bottom=.11,top=.84,wspace=.45,hspace=.56)
 panels=[('psnr_db','PSNR (dB) ↑',1),('lpips_alex','LPIPS ↓',1),('dinov2_vitl14_cosine','DINO-L ↑',1),('convnext_top1_source_prediction','Agreement (%) ↑',100),('dists','DISTS ↓',1),('dreamsim','DreamSim ↓',1),('ms_ssim','MS-SSIM ↑',1),(None,'Delivered objects (%) ↑',1)]
 for pi,(ax,(metric,label,factor))in enumerate(zip(axes.flat,panels)):
  for mi in [1,0]:
   method=METHODS[mi];snrs=[4,10,19];c=style['colors'][method]
   if metric:
    rr=[next(v for v in rows if v['method']==method and v['metric']==metric and int(v['snr_db'])==snr)for snr in snrs]
    y=np.array([float(v['mean'])for v in rr])*factor;lo=np.array([float(v['ci_low'])for v in rr])*factor;hi=np.array([float(v['ci_high'])for v in rr])*factor
    ax.errorbar(snrs,y,yerr=np.array([y-lo,hi-y]),fmt='none',ecolor=c,elinewidth=.9,capsize=2,capthick=.8,zorder=2)
   else:
    y=np.array([100*sum(f['accepted']for f in frames if f['codec']==mi and jobs[f['key']]['snr_db']==snr)/300 for snr in snrs])
   ax.plot(snrs,y,color=c,linestyle='-' if mi==0 else '--',marker='o' if mi==0 else 's',markersize=4.2 if mi==0 else 5.6,markerfacecolor=c if mi==0 else 'white',markeredgewidth=.9,linewidth=1.5,zorder=4 if mi==0 else 3)
  ax.set_title('(%s) %s'%(chr(97+pi),label),loc='left',pad=6);ax.set_xticks([4,10,19]);ax.set_xlim(2.5,20.5);ax.set_xlabel('SNR (dB)',labelpad=2);ax.tick_params(length=3,pad=2)
  if metric is None:ax.set_ylim(-5,105);ax.set_yticks([0,50,100])
 handles=[Line2D([],[],color=style['colors'][METHODS[0]],marker='o',markersize=4.2,linewidth=1.5,label='NeST-Com (full m10)'),Line2D([],[],color=style['colors'][METHODS[1]],marker='s',markerfacecolor='white',markersize=5.6,linestyle='--',linewidth=1.5,label='ARPC (full e7)')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.985),ncol=2,fontsize=8.5)
 fig.text(.53,.895,'N = 2048 · two equal-protection subframes · fixed 100 sources',ha='center',fontsize=8)
 outputs=[]
 for ext in ['pdf','svg','png']:
  p=out/('n2048_full_comparison.'+ext);fig.savefig(p,dpi=600,facecolor='white');outputs.append(pin(p))
 fig.savefig(out/'paper_size_preview.png',dpi=120,facecolor='white');outputs.append(pin(out/'paper_size_preview.png'))
 plt.close(fig)
 caption='NeST-Com full m10 and adapted ARPC full e7 on the same fixed external100 sources with N=2048, split into two equal-protection N1024 subframes at 64-QAM and rate5/6. Each subframe pays its header, length, CRC and96-bit reassembly record. Both source objects must be received completely; rejected pairs are gray and remain in every denominator. Points show source means after averaging three noise realizations; error bars are pointwise95% source-bootstrap intervals. Agreement is the fraction of reconstructions whose ConvNeXt top-1 prediction matches the source prediction. The last panel shows observed delivery fractions among300 objects per method/SNR without an added reliability interval. Curves coincide at4 and10dB because all objects fail for both methods. No channel normalization, policy retuning or model training was performed.'
 (out/'caption_en.txt').write_text(caption+'\n');outputs.append(pin(out/'caption_en.txt'))
 put(out/'manifest.json',dict(status='EXPORTED_REQUIRES_VISUAL_REVIEW',source=pin(__file__),summary=pin(O/'metrics_v1/summary.csv'),physical=pin(O/'formal/logical_frames.json'),outputs=outputs,size_mm=[180,105],vector_curves_in_pdf_svg=True,bootstrap_calls=0,new_scoring=0))
if __name__=='__main__':main()
