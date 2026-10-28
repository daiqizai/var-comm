"""Reproduce descriptive pilot figures from published CSVs; no inference/resampling."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]
O=R/'results/nextscale_selective_blocks_20261028_v1'
def rows(n):return list(csv.DictReader((O/n).open()))
def main():
 F=O/'figures_v2';F.mkdir(exist_ok=False)
 nom=json.loads((O/'nomination.json').read_text());overall=rows('p3_candidate_overall.csv');snrs=rows('p3_candidate_snr_summary.csv');mech=rows('selective_mechanism_contribution.csv');closure=json.loads((O/'p3_closure.json').read_text())
 selected=nom['selected'];codes={};nt=ns=0
 for c in selected:
  if c['kind']=='THREE':nt+=1;codes[c['candidate_id']]=f'T{nt:02d}'
  else:ns+=1;codes[c['candidate_id']]=f'S{ns:02d}'
 with(F/'candidate_labels.csv').open('x',newline='')as f:
  w=csv.writer(f);w.writerow(['label','candidate_id']);w.writerows((v,k)for k,v in codes.items())
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'lines.linewidth':1.3,'lines.markersize':4,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
 fig,ax=plt.subplots(1,2,figsize=(7.16,3.9),gridspec_kw={'width_ratios':[1,1.35]});fig.subplots_adjust(left=.09,right=.98,top=.86,bottom=.24,wspace=.32)
 y=np.arange(len(selected))
 actual=[float(next(r['dino_cosine']for r in overall if r['candidate_id']==c['candidate_id']and r['mode']==('SEL'if c['kind']=='THREE'else'SINGLE')))for c in selected]
 proxy=[c['J_proxy']for c in selected]
 ax[0].plot(proxy,y,'x',color='#767676',label='Packet proxy')
 ax[0].scatter(actual,y,s=20,c=['#0F4D92'if c['kind']=='THREE'else'#B64342'for c in selected],label='Actual full frames',zorder=3)
 ax[0].set_yticks(y,[codes[c['candidate_id']]for c in selected]);ax[0].invert_yaxis();ax[0].set_xlabel('Overall DINO cosine, J');ax[0].set_title('(a) All nominated candidates',loc='left');ax[0].set_xlim(-.025,.435);ax[0].grid(axis='x',color='.9',linewidth=.5)
 ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=7.5)
 best=closure['best_measured_SEL']['candidate_id'];single=closure['best_measured_single']['candidate_id']
 for cid,mode,color,mark,style,label in [(single,'SINGLE','#B64342','s','-',f'{codes[single]} single'),(best,'SEL','#0F4D92','o','-',f'{codes[best]} SEL'),(best,'SEQ','#42949E','^','--',f'{codes[best]} SEQ'),(best,'ALL','#767676','d',':',f'{codes[best]} ALL')]:
  values=[float(next(r['dino_cosine']for r in snrs if r['candidate_id']==cid and r['mode']==mode and int(r['snr_db'])==s))for s in [0,2,4]]
  ax[1].plot([0,2,4],values,color=color,marker=mark,linestyle=style,label=label)
 ax[1].set_xticks([0,2,4]);ax[1].set_xlabel('Actual SNR (dB)');ax[1].set_ylabel('DINO cosine');ax[1].set_ylim(-.025,.455);ax[1].grid(color='.9',linewidth=.5);ax[1].set_title('(b) Best measured single and SEL',loc='left');ax[1].legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=7.5)
 fig.suptitle('32 calibration sources · fixed TX policy · N = 1024',fontsize=9.5,y=.97)
 for ext in ['pdf','svg','png']:fig.savefig(F/('p3_candidate_comparison.'+ext),dpi=300)
 plt.close(fig)
 triples=[c for c in selected if c['kind']=='THREE'];data=[]
 for c in triples:
  data.append([next(r for r in mech if r['candidate_id']==c['candidate_id']and r['metric']=='dino_cosine'and int(r['snr_db'])==s)for s in [0,2,4]])
 freq=np.array([[float(r['frequency'])*100 for r in rr]for rr in data]);delta=np.array([[float(r['SEL_minus_SEQ'])for r in rr]for rr in data])
 fig,axes=plt.subplots(1,2,figsize=(7.16,3.8));fig.subplots_adjust(left=.09,right=.97,bottom=.08,top=.88,wspace=.26)
 for idx,(a,arr,cmap,title)in enumerate(zip(axes,[freq,delta],['Blues','RdBu_r'],['(a) C success / A fail / B success','(b) Actual SEL − SEQ contribution'])):
  limit=max(float(np.max(np.abs(arr)))*1.02,1e-6);lo=0 if idx==0 else-limit
  im=a.imshow(arr,cmap=cmap,vmin=lo,vmax=limit,aspect='auto');a.set_xticks([0,1,2],[0,2,4]);a.set_yticks(range(len(triples)),[codes[c['candidate_id']]for c in triples]);a.set_xlabel('Actual SNR (dB)');a.set_title(title,loc='left',fontsize=8)
  for i in range(arr.shape[0]):
   for j in range(3):a.text(j,i,f'{arr[i,j]:.1f}%'if idx==0 else f'{arr[i,j]:+.4f}',ha='center',va='center',fontsize=7.7,color='white'if abs(arr[i,j])/limit>.6 else'black')
  cb=fig.colorbar(im,ax=a,orientation='horizontal',fraction=.06,pad=.19,aspect=35);cb.set_label('Frequency (%)'if idx==0 else'Delta DINO cosine',fontsize=8);cb.ax.tick_params(labelsize=7)
 fig.suptitle('All 11 three-block candidates · 96 frames per cell',fontsize=9.5,y=.98)
 for ext in ['pdf','svg','png']:fig.savefig(F/('p3_selective_mechanism.'+ext),dpi=300)
 plt.close(fig)
 inputs=['nomination.json','p3_candidate_overall.csv','p3_candidate_snr_summary.csv','selective_mechanism_contribution.csv','p3_closure.json']
 (F/'provenance.json').write_text(json.dumps(dict(inputs={n:hashlib.sha256((O/n).read_bytes()).hexdigest()for n in inputs},font_pt=8.5,size_inches=[[7.16,3.9],[7.16,3.8]],CI='No new CI/bootstrap; pilot descriptive results',labels='candidate_labels.csv'),indent=2)+'\n')
 print('Created 2 descriptive figures, PDF/SVG/PNG and data identities.')
if __name__=='__main__':main()
