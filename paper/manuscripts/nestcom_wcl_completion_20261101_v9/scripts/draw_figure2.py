from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path(__file__).resolve().parents[1]
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
e1=read(OUT/'inputs/v9_package/inputs/e1_summary.csv');e2=read(OUT/'inputs/v9_package/inputs/e2_summary.csv')
plotted=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.4,'axes.titlesize':9,'legend.fontsize':7.6,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.0,4.05))
styles={
 'final_NeST':dict(label='NeST-Com',color='#3775BA',marker='o',linestyle='-',linewidth=1.65,zorder=7),
 'WHOLE_final_receiver':dict(label='WHOLE',color='#5c8db7',marker='s',linestyle='--',linewidth=1.3,zorder=6),
 'Ada_frozen':dict(label='Ada (adapted)',color='#42949E',marker='v',linestyle='-.',linewidth=1.4,zorder=5),
 'ARPC':dict(label='ARPC + LDPC',color='#9A4D8E',marker='+',linestyle='--',linewidth=.95,zorder=2),
 'Latent_JSCC':dict(label='Latent JSCC',color='#B64342',marker='s',linestyle=':',linewidth=.95,zorder=2),
 'SwinJSCC':dict(label='SwinJSCC',color='#5E966B',marker='^',linestyle='-.',linewidth=.95,zorder=2),
 'BPG_LDPC':dict(label='BPG + LDPC',color='#777777',marker='D',linestyle=':',linewidth=.95,zorder=2)}
order=['final_NeST','WHOLE_final_receiver','Ada_frozen','ARPC','Latent_JSCC','SwinJSCC','BPG_LDPC']
for col,metric in enumerate(['dists','dreamsim']):
 ax=axs[0,col]
 for method in order:
  rows=sorted([r for r in e1 if r['method']==method and r['N']=='1024' and r['metric']==metric],key=lambda r:float(r['snr_db']))
  assert len(rows)==4
  plotted.extend(dict(panel='ab',experiment='fixed_symbols',method=method,metric=metric,N=1024,x=float(r['snr_db']),mean=float(r['mean']),ci95_low=float(r['ci95_low']),ci95_high=float(r['ci95_high'])) for r in rows)
  x=np.array([float(r['snr_db']) for r in rows]);y=np.array([float(r['mean']) for r in rows]);lo=np.array([float(r['ci95_low']) for r in rows]);hi=np.array([float(r['ci95_high']) for r in rows])
  ax.errorbar(x,y,yerr=[y-lo,hi-y],markersize=3.2,capsize=1.7,elinewidth=.65,markerfacecolor='white' if method!='final_NeST' else styles[method]['color'],**styles[method])
 ax.set_xticks([4,7,10,19]);ax.set_xlabel('SNR (dB)');ax.set_ylabel(('DISTS' if metric=='dists' else 'DreamSim')+r' $\downarrow$');ax.grid(axis='y',alpha=.2)
 ax.set_title(('(a)' if col==0 else '(b)')+r' Fixed symbol budget: $N=1024$',loc='left')
 ax.set_ylim(bottom=0)
 ax=axs[1,col]
 for method,stylekey in [('Token_NeST','final_NeST'),('WHOLE','WHOLE_final_receiver'),('Ada','Ada_frozen')]:
  rows=sorted([r for r in e2 if r['method']==method and r['metric']==metric],key=lambda r:float(r['B_bits']))
  assert len(rows)==6
  plotted.extend(dict(panel='cd',experiment='equal_information_bits',method=method,metric=metric,N='',x=float(r['B_bits']),mean=float(r['mean']),ci95_low=float(r['ci95_low']),ci95_high=float(r['ci95_high'])) for r in rows)
  x=np.array([float(r['B_bits']) for r in rows]);y=np.array([float(r['mean']) for r in rows]);lo=np.array([float(r['ci95_low']) for r in rows]);hi=np.array([float(r['ci95_high']) for r in rows])
  ax.errorbar(x,y,yerr=[y-lo,hi-y],markersize=3.2,capsize=1.7,elinewidth=.65,markerfacecolor='white' if method!='Token_NeST' else styles[stylekey]['color'],**styles[stylekey])
 ax.set_xticks([620,1302,1940,2577]);ax.set_xlabel('Total information budget (bits)');ax.set_ylabel(('DISTS' if metric=='dists' else 'DreamSim')+r' $\downarrow$');ax.grid(axis='y',alpha=.2)
 ax.set_title(('(c)' if col==0 else '(d)')+' Equal information budgets',loc='left');ax.set_ylim(bottom=0)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.0),frameon=False,columnspacing=1.7,handlelength=2.7)
fig.subplots_adjust(left=.09,right=.985,bottom=.11,top=.84,wspace=.28,hspace=.49)
for ext in ['pdf','svg','png']: fig.savefig(OUT/'figures'/('figure2.'+ext),dpi=600)
fig.savefig(OUT/'figures/figure2_print_size.png',dpi=150)
plt.close(fig)
assert len(plotted)==92
with (OUT/'figures/figure2_data.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
from PIL import Image
Image.open(OUT/'figures/figure2.png').convert('L').save(OUT/'figures/figure2_grayscale.png')
