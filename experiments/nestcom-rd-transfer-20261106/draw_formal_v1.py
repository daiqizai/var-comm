"""All five metrics, all matched budgets, and a separate full-codec comparison."""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
    root=Path(a.results);out=root/'figures';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((root/'summary.csv').open()));assert len(rows)==50
    caps=list(csv.DictReader((root/'capacity.csv').open()));key={(v['method'],v['condition'],v['metric']):v for v in rows}
    methods=['original_nest','trained5000'];labels=['Original NeST','RD-adapted NeST (step 5000)']
    colors=['#767676','#3775BA'];markers=['s','o'];linestyles=['--','-']
    metrics=[('psnr_db','PSNR ↑ (dB)'),('lpips_alex','LPIPS-Alex ↓'),('dists','DISTS ↓'),('dreamsim','DreamSim ↓'),('clip_i','CLIP-I ↑')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    provenance=dict(summary_sha256=hashlib.sha256((root/'summary.csv').read_bytes()).hexdigest(),
        capacity_sha256=hashlib.sha256((root/'capacity.csv').read_bytes()).hexdigest(),pointwise_CI=50,mean_points=50,
        style_skill='scientific-figure-making; design-theory.md; physical-sized fonts and minimal axes',
        scientific_image_transforms=False)
    for full in [False,True]:
        fig,axes=plt.subplots(1,5,figsize=(12.6,3.1));fig.subplots_adjust(left=.05,right=.99,bottom=.26,top=.74,wspace=.42)
        for ax,(metric,title) in zip(axes,metrics):
            ax.set_title(title,pad=9);ax.grid(axis='y',color='#E8E8E8',linewidth=.6);ax.set_axisbelow(True)
            for j,m in enumerate(methods):
                conditions=['full'] if full else ['591','1273','1911','2548']
                rr=[key[m,c,metric] for c in conditions];ys=np.array([float(v['mean']) for v in rr]);lo=np.array([float(v['ci_low']) for v in rr]);hi=np.array([float(v['ci_high']) for v in rr])
                xs=[j] if full else [(int(c)+41)/1000 for c in conditions]
                ax.errorbar(xs,ys,yerr=np.vstack([ys-lo,hi-ys]),color=colors[j],marker=markers[j],
                    linestyle='none' if full else linestyles[j],linewidth=1.4,markersize=4.5,capsize=2,elinewidth=.8,label=labels[j])
            ax.margins(y=.13)
            if full:ax.set_xticks([0,1],['Original','Step 5000']);ax.set_xlim(-.5,1.5)
            else:
                ax.set_xticks([.632,1.314,1.952,2.589],['0.632','1.314','1.952','2.589']);ax.tick_params(axis='x',rotation=30)
                ax.set_xlabel('Packet budget (kbit)',labelpad=4)
        handles,ls=axes[0].get_legend_handles_labels();fig.legend(handles,ls,loc='upper center',bbox_to_anchor=(.5,.985),ncol=2)
        if full:
            bits={v['method']:float(v['mean_source_bits']) for v in caps if v['condition']=='full'}
            note=f"All 680 tokens; unconstrained source bits: original {bits['original_nest']:.1f}, step 5000 {bits['trained5000']:.1f}. Not an equal-rate point."
        else:note='500 fixed held-out sources; actual terminated streams plus 41 paid signaling/CRC bits; clean packets, no new PHY.'
        fig.text(.5,.025,note,ha='center',fontsize=8.5)
        stem='full_representation' if full else 'matched_budgets'
        for ext in ['pdf','svg','png']:
            target=out/(stem+'.'+ext)
            fig.savefig(target,dpi=300,facecolor='white')
            if ext=='svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()).rstrip()+'\n')
        plt.close(fig)
    (out/'plot_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'captions.md').write_text('# Fixed-step original NeST comparison\n\nMatched budgets: four equal total information-packet limits (source capacity plus 12-bit profile, 13-bit length and 16-bit CRC). All five metrics and all pointwise 95% source-bootstrap intervals are retained. This is a clean-packet source-codec evaluation, not a new wireless-frame experiment.\n\nFull representation: both systems independently decode all 680 tokens. Actual terminated arithmetic-code length is reported separately; this comparison measures the full-reconstruction ceiling, not equal-bit performance. Points and error bars use all 500 fixed held-out sources.\n')

if __name__=='__main__':main()
