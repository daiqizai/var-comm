"""Read-only plotting of the closed 32-source actual-frame results."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    root=Path(root);out=root/'figures_v1';out.mkdir(exist_ok=False)
    def rows(name):return list(csv.DictReader((root/name).open(encoding='utf8')))
    overall=rows('wireless_overall.csv');account=rows('source_accounting.csv')
    spec=[('S_FILL','SINGLE','Fine-fill single','#0F4D92'),('S_PROTECT','SINGLE','Protect single','#3775BA'),('OLD_STRONG_SINGLE','SINGLE','Original single','#767676'),('COMPACT_TOKEN','SEL','Compact token-wise','#42949E'),('COMPACT_FIXED','SEL','Compact fixed K=12','#9A4D8E'),('SHARED_OLD','SEL','Shared old payload','#B64342'),('OLD_BEST_THREE','SEL','Original three blocks','#4D4D4D')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.85),gridspec_kw={'width_ratios':[1.25,1]});export=[]
    for i,(arm,mode,label,color)in enumerate(spec):
        v=next(r for r in overall if(r['arm'],r['mode'])==(arm,mode));aa=[r for r in account if r['arm']==arm]
        if aa:source,fields,padding=[np.mean([float(r[k])for r in aa])for k in('source_bits','fields','padding')]
        elif arm=='OLD_STRONG_SINGLE':source,fields,padding=572.34375,29,35.65625
        else:source,fields,padding=422.9375,87,126.0625
        axes[0].barh(i,source,color=color,height=.62,label='Source AC'if i==0 else None)
        axes[0].barh(i,fields,left=source,color='#4D4D4D',height=.62,label='Fields'if i==0 else None)
        axes[0].barh(i,padding,left=source+fields,color='#E0E0E0',edgecolor='#767676',linewidth=.4,hatch='///',height=.62,label='Padding'if i==0 else None)
        j=float(v['dino_cosine']);axes[1].plot(j,i,'o',color=color,ms=6);axes[1].text(j+.0018,i,f'{j:.4f}',va='center',fontsize=8.5)
        export.append(dict(arm=arm,mode=mode,source_bits=source,fields=fields,padding=padding,DINO_J=j))
    for ax in axes:ax.set_ylim(6.6,-.6);ax.set_yticks(range(7))
    axes[0].set_yticklabels([x[2]for x in spec]);axes[1].set_yticklabels([])
    axes[0].set_xlim(0,675);axes[1].set_xlim(.34,.451);axes[1].axvline(float(next(r['dino_cosine']for r in overall if r['arm']=='S_FILL')),color='#0F4D92',ls=':',lw=.8)
    axes[0].set_xlabel('Mean LDPC information bits per image');axes[1].set_xlabel('DINOv2 cosine J ↑')
    axes[0].set_title('(a) Actual source and overhead accounting',loc='left',fontsize=10);axes[1].set_title('(b) Actual-frame quality',loc='left',fontsize=10)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8)
    fig.suptitle('N=1024 · 32 development sources · 0/2/4 dB equally weighted',fontsize=10,y=.99)
    fig.subplots_adjust(left=.22,right=.98,bottom=.23,top=.86,wspace=.16)
    for ext in('pdf','svg','png'):fig.savefig(out/('accounting_quality.'+ext),dpi=300,facecolor='white')
    with(out/'plot_data.csv').open('w',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    (out/'provenance.json').write_text(json.dumps(dict(inputs={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in['wireless_overall.csv','source_accounting.csv']},size_inches=[9.3,3.85],bootstrap='none; descriptive fixed-development-source means',gray_fields='same overhead convention, no wireless symbols reclaimed from padding'),indent=2))
    plt.close(fig)
if __name__=='__main__':
    import sys
    main(sys.argv[1])
