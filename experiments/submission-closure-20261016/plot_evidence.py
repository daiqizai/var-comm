"""Deterministic paper-layout preparation from closed outputs; no model/metric calls."""
from pathlib import Path
import sys,csv,json,hashlib,os
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');O=R/'results/submission_closure_20261016_v1';T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/submission_closure_20261016_v1')
os.environ['MPLCONFIGDIR']=str(T/'plot_cache');os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.5,'savefig.facecolor':'white'})
out=O/'figures';out.mkdir(exist_ok=False)
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf8')))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
figs=[];inputs=[]
def load(p):inputs.append(pin(p));return rows(p)
def save(fig,name,caption):
 for ext in ['pdf','svg','png']:
  fig.savefig(out/(name+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.07)
 figs.append(dict(name=name,caption=caption,files=[pin(out/(name+'.'+ext))for ext in ['pdf','svg','png']]))
 plt.close(fig);print('FIGURE',name,flush=True)
names={'EC_VAR_PARTIAL':'NeST-Com (entropy)','RAW_PARTIAL':'NeST-Com (raw)','EC_VAR_WHOLE':'Whole-scale entropy + VAR','RAW_WHOLE':'Whole-scale raw + VAR','P100000':'Latent JSCC (100k)','Swin200000':'SwinJSCC (adapted, 200k)','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'Adaptive BPG + LDPC','HiFiSwin200000':'HiFi (Swin200k)'}
methods=['EC_VAR_PARTIAL','RAW_PARTIAL','EC_VAR_WHOLE','P100000','Swin200000','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'];colors=['#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7','#5E5E5E'];markers=['o','s','^','D','v','P']
old=load(R/'results/entropy_main_common500_20261013/analysis_v1/summary_all_methods.csv');new=load(R/'results/quality_metrics_supplement_20261014_v1/common500/summary.csv')
allrows=[]
for origin,source in [('original_four',old),('supplement_three',new)]:
 for r in source:allrows.append(dict(method=r['method'],snr_db=r['snr_db'],metric=r['metric'],mean=r['mean'],ci_low=r.get('ci_low',r.get('CI95_low')),ci_high=r.get('ci_high',r.get('CI95_high')),source_count='500',noise_count='3',logical_frames='1500',provenance=origin,new_metric_or_bootstrap_calls=0))
with(out/'common500_all_metrics.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
fig,axes=plt.subplots(2,2,figsize=(7.2,5.8));spec=[('psnr_db','PSNR (dB) ↑',1),('lpips_alex','LPIPS-Alex ↓',1),('dreamsim','DreamSim ensemble ↓',1),('convnext_top1_source_prediction','ConvNeXt source agreement (%) ↑',100)]
for ax,(metric,label,scale)in zip(axes.flat,spec):
 for method,color,marker in zip(methods,colors,markers):
  rs=sorted([r for r in allrows if r['method']==method and r['metric']==metric],key=lambda x:int(x['snr_db']));assert len(rs)==3
  x=[int(r['snr_db'])for r in rs];y=[float(r['mean'])*scale for r in rs];lo=[float(r['ci_low'])*scale for r in rs];hi=[float(r['ci_high'])*scale for r in rs]
  ax.plot(x,y,label=names[method],color=color,marker=marker,ms=4,lw=1.5);ax.fill_between(x,lo,hi,color=color,alpha=.09,linewidth=0)
 ax.set(xlabel='SNR (dB)',ylabel=label,xticks=[4,10,19])
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.0),frameon=False);fig.suptitle('Existing common500 evidence · provisional m10 comparator scope',fontsize=11);fig.tight_layout(rect=(0,.13,1,.95))
save(fig,'common500_main_provisional','Original common500, 500 sources × 3 noises at each SNR, N=1024. Existing means and source-level 95% intervals reused unchanged; no new bootstrap. Complete m10 fits calibration sources and requires comparator-scope correction before this layout is used as a final main figure.')
u=load(R/'results/entropy_main_common500_20261013/analysis_v1/length_summary.csv')
fig,ax=plt.subplots(figsize=(4.8,3.2))
for m,c in [('EC_VAR_PARTIAL',colors[0]),('EC_VAR_WHOLE',colors[2])]:
 rs=sorted([r for r in u if r.get('method')==m],key=lambda r:int(r['snr_db']));assert len(rs)==3
 key=next(k for k in rs[0]if 'utilization' in k and 'mean' in k)
 ax.plot([int(r['snr_db'])for r in rs],[float(r[key])*100 for r in rs],'-o',color=c,label=names[m])
ax.set(xlabel='SNR (dB)',ylabel='Arithmetic stream / (k − 29) (%)',xticks=[4,10,19]);ax.legend(frameon=False);fig.tight_layout()
save(fig,'source_capacity_utilization','Actual arithmetic stream bits divided by available arithmetic source capacity k−29. 500 sources per point; this is source-capacity utilization, not wireless-symbol saving. Existing frozen endpoints; m10 correction pending.')
summ=load(O/'fixed16/summary.csv');fm=['RAW_PARTIAL','P100000','Swin200000','HiFiSwin200000'];fc=['#0072B2','#009E73','#CC79A7','#D55E00'];fl=['NeST-Com\n(raw)','Latent JSCC\n100k','SwinJSCC\n200k','HiFi\nSwin200k']
fig,axes=plt.subplots(2,2,figsize=(7.2,5.4))
for ax,metric,label in zip(axes.flat,['psnr_db','dists','dreamsim','ms_ssim'],['PSNR (dB) ↑','DISTS ↓','DreamSim ensemble ↓','MS-SSIM ↑']):
 rs=[next(r for r in summ if r['method']==m and r['metric']==metric)for m in fm]
 ax.bar(range(4),[float(r['mean'])for r in rs],color=fc,width=.65,alpha=.8,label='Mean')
 ax.scatter(range(4),[float(r['median'])for r in rs],marker='_',s=150,color='black',lw=1.7,label='Median',zorder=5)
 ax.set(xticks=range(4),xticklabels=fl,ylabel=label);ax.tick_params(axis='x',labelsize=8)
axes[0,0].legend(frameon=False);fig.suptitle('All 16 fixed development sources · N=1024 · 13 dB',fontsize=11);fig.tight_layout(rect=(0,0,1,.95))
save(fig,'fixed16_descriptive','All original 16 development sources, native float32 images. Bars are means; black marks are medians, not confidence intervals. Raw uses channel seed6201; continuous methods seed2001; HiFi posterior23. Only Swin and HiFi share the actual RX observation. No new bootstrap or population-level superiority claim.')
# Native arrays are used only for uniform display; no metric computation here.
reg=json.loads((O/'metric_registration.json').read_bytes());inputs.append(pin(O/'metric_registration.json'));display={};reference={}
for r in reg['pairs']:
 p=Path(r['archive']['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['archive']['sha256']
 with np.load(p,allow_pickle=False)as z:a=z['reconstruction'].copy()
 if a.ndim==4:a=a[0]
 assert a.shape==(3,256,256)and a.dtype==np.float32
 display[r['source_index'],r['method']]=np.rint(np.clip(a,0,1).transpose(1,2,0)*255).astype(np.uint8)
 if r['source_index']not in reference:
  p=Path(r['reference']['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['reference']['sha256']
  with np.load(p,allow_pickle=False)as z:a=z['pixels'].copy()
  assert a.dtype==np.uint8 and a.shape==(3,256,256);reference[r['source_index']]=a.transpose(1,2,0)
def gallery(indices,name,title):
 fig,axes=plt.subplots(len(indices),5,figsize=(7.2,1.5*len(indices)+.45),squeeze=False)
 for i,idx in enumerate(indices):
  for j,m in enumerate(['source']+fm):
   ax=axes[i,j];ax.imshow(reference[idx]if m=='source'else display[idx,m],interpolation='nearest');ax.grid(False);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   if i==0:ax.set_title('Source'if m=='source'else fl[j-1],fontsize=8)
   if j==0:ax.set_ylabel(f'Index {idx}',fontsize=8)
 fig.suptitle(title,fontsize=10);fig.tight_layout(pad=.25,rect=(0,0,1,.95))
 save(fig,name,'Existing native float images, all methods uniformly clipped and rounded to RGB uint8 for display only. No sharpening, recoloring, or method-specific processing. N=1024,13dB; raw6201/continuous2001/HiFi23; only Swin and HiFi same actual observation. Fixed source order, not quality selection.')
assert reg['source_indices'][:2]==[0,25]
gallery([0,25],'fixed16_first_two','First two sources in the original fixed16 registration · 13 dB')
for start in range(0,16,4):gallery(reg['source_indices'][start:start+4],f'fixed16_gallery_{start//4+1}','Complete fixed16 gallery · original order · 13 dB')
# Structural diagram: separate TX entropy model cost; shared single-codeword wire protocol.
fig,ax=plt.subplots(figsize=(7.2,3.9));ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,text,color='#EDF4F7'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc=color,ec='#4B6470',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#34454D',lw=.9))
box(.1,2.55,1.3,1,'Source RGB\nFrozen tokenizer');box(1.8,4.1,2.3,1.1,'Entropy TX\nVAR integer CDF\nActual finish + endpoint');box(1.8,.9,2.3,1.1,'Raw TX\nFixed-width tokens\nNo VAR probability model')
box(4.6,2.3,2.1,1.5,'Paid header + payload\nOne LDPC codeword\nAWGN / actual RX\nN=1024, nominal Es=2')
box(7.2,3.8,2.55,1.4,'Entropy RX\nLength / CRC / arithmetic\nchecks; reject → gray')
box(7.2,.8,2.55,1.2,'Raw RX\nRegistered CRC / KEEP rule\nActual received tokens')
box(7.2,2.35,2.55,1.0,'Shared VAR completion\nFrozen adapted decoder D_c','#F4EBDD')
arrow((1.4,3.05),(1.8,4.65));arrow((1.4,3.05),(1.8,1.45));arrow((4.1,4.65),(4.6,3.25));arrow((4.1,1.45),(4.6,2.75));arrow((6.7,3.25),(7.2,4.5));arrow((6.7,2.75),(7.2,1.4));arrow((8.47,3.8),(8.47,3.35));arrow((8.47,2),(8.47,2.35))
ax.text(5,.22,'Entropy: 68 header + 956 body symbols; length 13 bit + CRC 16 bit.\nRaw uses its frozen paid profile; failed frames remain in every denominator.',ha='center',va='center',fontsize=8)
fig.tight_layout();save(fig,'single_codeword_system','Single-codeword implementations only. Entropy TX adds VAR probability evaluation; raw TX does not. RX consumes actual observations, paid profiles and shared models; source truth is audit-only. Raw and entropy failure rules are explicitly different. Frozen raw whole19 uses idle symbols, so 956 occupied body symbols is not asserted for every raw profile.')
json.dump(dict(status='RENDERED_PENDING_VISUAL_REVIEW',new_model_calls=0,new_metric_calls=0,new_bootstrap=0,provisional_main_due_m10=True,inputs=inputs,figures=figs), (out/'manifest.json').open('x'),indent=2)
(out/'captions.md').write_text('\n\n'.join('## '+x['name']+'\n\n'+x['caption']for x in figs),encoding='utf8')
