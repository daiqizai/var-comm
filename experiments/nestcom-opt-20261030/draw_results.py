"""Scientific figures from frozen CSVs and pinned native arrays; no image enhancement."""
from opt_common import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def rows(path):
 with Path(path).open()as f:return list(csv.DictReader(f))
def main():
 out=D/'delivery/figures';out.mkdir(parents=True,exist_ok=False);provenance=[]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
 def export(fig,name,inputs,note):
  for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.055)
  provenance.append(dict(name=name,inputs=[pin(p)for p in inputs],description=note,files=[pin(out/(name+'.'+x))for x in ['pdf','svg','png']],data_not_recomputed=True));plt.close(fig)
 data=rows(D/'delivery/route_summary.csv');fig,axes=plt.subplots(1,2,figsize=(7.16,4.5),sharey=True);y=np.arange(len(data))
 for ax,metric,title in zip(axes,['normalized_lpips','normalized_dreamsim'],['LPIPS change (%)','DreamSim change (%)']):
  values=np.array([float(v[metric])-1 for v in data])*100;colors=['#3775BA'if v['route']=='D'else'#B64342'if x>0 else'#42949E'for v,x in zip(data,values)]
  ax.barh(y,values,color=colors,height=.7);ax.axvline(0,color='#767676',lw=.8);ax.set_xlabel(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 axes[0].set_yticks(y,[v['candidate']for v in data]);axes[0].invert_yaxis();axes[0].set_title('FIT100 core: mean of cell-normalized metrics',loc='left');axes[1].set_title('Negative is favorable',loc='left');fig.tight_layout()
 export(fig,'development_routes',[D/'delivery/route_summary.csv'],'All first-round candidates; means normalized within each core workpoint then averaged. Screening means, not blind test. C guard failure and D confirmation in decision table.')
 summary=rows(D/'final_common500_delivery/summary.csv');methods=['original_NeST','final_NeST','WHOLE_final_receiver','Ada_frozen'];labels=['Original NeST','NeST + tokenwise fitting','WHOLE, same receiver','Ada, fixed-frame adaptation'];colors=['#767676','#3775BA','#9A4D8E','#42949E'];styles=['--','-','-.',':'];markers=['o','s','^','D']
 for N in [1024,512]:
  fig,axes=plt.subplots(2,2,figsize=(7.16,4.6));handles=[]
  for ax,metric,title in zip(axes.flat,['psnr_db','lpips_alex','dreamsim','clip_image_cosine'],['PSNR (dB) ↑','LPIPS-Alex ↓','DreamSim ↓','CLIP-I ↑']):
   for method,label,color,style,marker in zip(methods,labels,colors,styles,markers):
    vs=sorted([v for v in summary if v['method']==method and int(v['N'])==N and v['metric']==metric],key=lambda v:int(v['snr_db']));x=np.array([int(v['snr_db'])for v in vs]);mean=np.array([float(v['mean'])for v in vs]);lo=np.array([float(v['ci95_low'])for v in vs]);hi=np.array([float(v['ci95_high'])for v in vs])
    h=ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],color=color,ls=style,marker=marker,ms=3.8,lw=1.15,elinewidth=.75,capsize=2,label=label)
    if metric=='psnr_db':handles.append(h)
   ax.set_xticks([4,7,10,19]if N==1024 else[4,10,19]);ax.set_xlabel('SNR (dB)');ax.set_ylabel(title);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
  fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.025),fontsize=8);fig.tight_layout(rect=(0,0,1,.93))
  export(fig,f'common500_N{N}',[D/'final_common500_delivery/summary.csv',D/'frozen_final.json'],'Four frozen systems. Equal-source means and pointwise95% source-bootstrap CI after averaging three native channel repetitions. Historical common500 regression; no configuration selection.')
 capacity=rows(D/'final_common500_delivery/endpoint_and_failure_summary.csv');fig,axes=plt.subplots(1,2,figsize=(7.16,2.55))
 for ax,N in zip(axes,[1024,512]):
  for method,label,color,style,marker in zip(methods[:3],labels[:3],colors[:3],styles[:3],markers[:3]):
   vs=sorted([v for v in capacity if v['method']==method and int(v['N'])==N],key=lambda v:int(v['snr_db']));ax.plot([int(v['snr_db'])for v in vs],[100*float(v['mean_source_utilization'])for v in vs],ls=style,marker=marker,color=color,ms=4,label=label)
  ax.set_title(f'N = {N}');ax.set_xlabel('SNR (dB)');ax.set_ylabel('Source-bit utilization (%)');ax.set_xticks([4,7,10,19]if N==1024 else[4,10,19]);ax.set_ylim(0,103);ax.grid(axis='y',alpha=.12)
 axes[0].legend(fontsize=7.5,loc='lower right');fig.tight_layout();export(fig,'source_bit_utilization',[D/'final_common500_delivery/endpoint_and_failure_summary.csv'],'Source arithmetic bits / body information capacity after29bit overhead; not total airtime efficiency. All logical positions retained.')
 frames=rows(D/'final_common500_delivery/frame_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(7.16,2.55))
 for ax,snr in zip(axes,[7,10]):
  for method,label,color,style in zip(methods[:3],labels[:3],colors[:3],styles[:3]):
   vs=[v for v in frames if v['method']==method and int(v['N'])==1024 and int(v['snr_db'])==snr]
   x=[]
   for v in vs:
    ep=json.loads(v['endpoint']);m,K=ep;x.append(OFFSETS[m]+K)
   x=np.sort(x);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,ls=style,label=label)
  ax.set_title(f'N = 1024, {snr} dB');ax.set_xlabel('Transmitted token count');ax.set_ylabel('Empirical cumulative fraction');ax.grid(alpha=.12)
 axes[0].legend(fontsize=7,loc='lower right');fig.tight_layout();export(fig,'transmitted_endpoint_distribution',[D/'final_common500_delivery/frame_metrics.csv'],'All transmitted selections including subsequently rejected frames; three registered repetitions per source retained. Distribution, not per-image quality monotonicity.')
 ef=rows(D/'delivery/E_selection_frequency.csv');fig,axes=plt.subplots(1,2,figsize=(7.16,2.7))
 for i,snr in enumerate([7,10]):
  vs=[v for v in ef if int(v['snr_db'])==snr];axes[0].bar(np.arange(3)+(i-.5)*.34,[int(v['sources'])for v in vs],width=.34,label=f'{snr} dB',color=['#3775BA','#42949E'][i])
 axes[0].set_xticks(range(3),['e0: longest','e1: shorter','e2: shorter']);axes[0].set_ylabel('Selected sources / 100');axes[0].legend()
 ed=next(v for v in data if v['candidate']=='E');axes[1].bar(['LPIPS','DreamSim'],[100*(float(ed[m])-1)for m in ['normalized_lpips','normalized_dreamsim']],color='#B64342');axes[1].axhline(0,color='#767676',lw=.7);axes[1].set_ylabel('Core normalized change (%)');axes[1].set_title('Actual channel, all sources retained')
 fig.tight_layout();export(fig,'E_selection_and_quality',[D/'delivery/E_selection_frequency.csv',D/'delivery/route_summary.csv'],'FIT100 transmitter decisions before channel observation. Positive quality-distance changes are unfavorable; one actual rejection remains in denominator. These are screening means, not simultaneous confidence statements.')
 timing=rows(D/'delivery/timing_and_compute.csv');tx=[v for v in timing if v.get('component')=='TX_total'and v.get('snr_db')=='pooled_core'];rx=[v for v in timing if v.get('component')=='RX_total'and v.get('snr_db')=='pooled_core']
 if tx and rx:
  fig,axes=plt.subplots(1,2,figsize=(7.16,3.4));names=[v['variant']for v in tx]
  for ax,vs,title in zip(axes,[tx,rx],['TX (seconds)','RX (seconds)']):
   lookup={v['variant']:v for v in vs};vs=[lookup[n]for n in names];ax.barh(np.arange(len(names)),[float(v['mean_seconds'])for v in vs],color=['#3775BA'if n=='D_token'else'#767676'for n in names]);ax.set_yticks(np.arange(len(names)),names);ax.invert_yaxis();ax.set_xlabel(title);ax.grid(axis='x',alpha=.12)
  fig.tight_layout();export(fig,'deployment_time',[D/'delivery/timing_and_compute.csv',D/'benchmark_protocol.json'],'16 fixed FIT sources, two core workpoints,2warmups+3measured repeats,batch1,same GPU/2CPU affinity. Component-summed TX/RX means in shared environment; E includes actual candidate generation and LPIPS.')
 # Mechanism diagnostic sources were frozen before final reconstructions, never ranked by gain.
 rule=read(D/'display_source_rule.json');ids=rule['C_mechanism']['source_ids'];sources={s['source_id']:s for s in read(L/'inputs/fit100.json')};logical=read(L/'fit_C_rx_plan/logical.json');outputs={(v['key'],v['variant']):v for v in read(L/'fit_C/outputs.json')};jobs={v['key']:v['projection']for v in read(L/'fit_C_rx_plan/inputs.json')};manifest=[]
 fig,axes=plt.subplots(2,5,figsize=(7.16,3.35));titles=['Source','Received boundary mask','Original receiver','No-content control','Content corrector']
 for i,sid in enumerate(ids):
  s=sources[sid];f=next(v for v in logical if v['source_id']==sid and v['method']=='NEST'and v['N']==1024 and v['snr_db']==7);p=jobs[f['received_key']]
  with np.load(checked(s['archive']),allow_pickle=False)as z:source=z['pixels'].copy().transpose(1,2,0)/255
  axes[i,0].imshow(source);m,K=p['m'],p['K'];size=SIZES[m]if m<10 else 16;mask=(np.arange(size*size)<K).reshape(size,size);axes[i,1].imshow(mask,cmap=ListedColormap(['#F0AD68','#3775BA']),vmin=0,vmax=1,interpolation='nearest');axes[i,1].set_xlabel(f'm={m}, K={K}; blue: received',fontsize=7)
  for j,variant in enumerate(['original','C_no_content_1000','C_content_1000'],2):
   v=outputs[(f['received_key'],variant)];image=np.full((3,256,256),.5,np.float32)if v['image']is None else np.load(checked(v['image']),allow_pickle=False)['image'];assert asha(image)==v['pixel_sha256'];axes[i,j].imshow(image.transpose(1,2,0));manifest.append(dict(source_id=sid,source_index=s['source_index'],N=1024,snr_db=7,noise_seed=4101,variant=variant,image=v['image'],pixel_sha256=v['pixel_sha256'],source_array=s['archive'],reference_sha256=s['reference_sha256'],m=m,K=K))
  axes[i,0].set_xlabel(f'FIT hash order {i+1}; index {s["source_index"]}',fontsize=7)
 for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([]);[sp.set_visible(False)for sp in ax.spines.values()]
 for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=7.7)
 fig.tight_layout(pad=.3);save(out/'C_mechanism_native_manifest.json',manifest);export(fig,'C_mechanism_fixed_development',[out/'C_mechanism_native_manifest.json',D/'display_source_rule.json'],'Two fixed FIT hash-order development sources, N1024/7, actual first channel. Full native fields, no enhanced images. C failed the N512/19 guard; this illustration is not an adoption result.')
 save(D/'delivery/figure_provenance.json',dict(figures=provenance,NEW200_contact_sheet='BLOCKED: raw unused validation pool unavailable',known_Pastries_Dog_final_contact='BLOCKED: exact legacy source IDs are outside formal common500; no integer-index substitution',scientific_figure_skill='scientific-figure-making local installed instructions; actual data-driven Matplotlib, no generative images',visual_review='PENDING_REAL_OPEN'))
 print('FIGURES_EXPORTED',len(provenance))
if __name__=='__main__':main()
