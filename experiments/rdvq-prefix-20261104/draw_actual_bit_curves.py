"""Publication-size plots from closed numerical CSVs only; no placeholder data."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
COLORS={'R0':'#767676','R1':'#B64342','R2':'#42949E','R3':'#3775BA','R3_estimated':'#9A4D8E'}
STYLES={'R0':(':','x'),'R1':('--','s'),'R2':('--','v'),'R3':('-','o'),'R3_estimated':('-.','^')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p1',required=True);ap.add_argument('--p2');ap.add_argument('--output',required=True);a=ap.parse_args();O=Path(a.output);O.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    inputs={};exports=[]
    def load(path):
        path=Path(path);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return list(csv.DictReader(path.open()))
    def export(fig,name,caption):
        fig.canvas.draw();fig.savefig(O/(name+'.pdf'));fig.savefig(O/(name+'.svg'));fig.savefig(O/(name+'.png'),dpi=600)
        fig.savefig(O/(name+'_print.png'),dpi=150)
        with Image.open(O/(name+'_print.png')) as im:im.convert('L').save(O/(name+'_grayscale.png'))
        exports.append(dict(name=name,size_inches=list(fig.get_size_inches()),caption=caption));plt.close(fig)
    def panel(ax,metric,label):
        ax.set_title(label);ax.set_xlabel('Source packet budget (kbit)');ax.set_xticks([.384,1,2,3,4]);ax.set_xlim(.32,4.16);ax.grid(axis='y',color='#e8e8e8',linewidth=.5);ax.set_axisbelow(True)
    def line(ax,rows,color,style,marker,label):
        rows=sorted(rows,key=lambda v:int(v['budget_bits']));assert len(rows)==11
        x=np.array([int(v['budget_bits'])/1000 for v in rows]);y=np.array([float(v['mean']) for v in rows]);lo=np.array([float(v['ci_low']) for v in rows]);hi=np.array([float(v['ci_high']) for v in rows]);assert np.isfinite([x,y,lo,hi]).all()
        assert (lo<=hi).all()
        ax.plot(x,y,color=color,linestyle=style,marker=marker,markersize=3,linewidth=1,label=label)
        ax.errorbar(x,(lo+hi)/2,yerr=(hi-lo)/2,fmt='none',ecolor=color,elinewidth=.55,capsize=1.6)
        return len(rows)
    p1=load(a.p1);n=0;fig,axes=plt.subplots(1,2,figsize=(7.16,2.8));fig.subplots_adjust(left=.07,right=.985,bottom=.19,top=.74,wspace=.25)
    names=[('longest_feasible','Longest feasible','#3775BA','-','o'),('calibrated_target','Calibrated target','#42949E','--','s'),('source_informed_DISTS_bound','Source-informed reference','#9A4D8E','-.','^')]
    for ax,m,title in zip(axes,['dists','dreamsim'],['(a) DISTS (lower is better)','(b) DreamSim (lower is better)']):
        panel(ax,m,title)
        for key,label,c,ls,mark in names:n+=line(ax,[v for v in p1 if v['policy']==key and v['metric']==m and int(v['budget_bits'])>0],c,ls,mark,label)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.0))
    export(fig,'p1_actual_packet_curves','Frozen R0, 500 fixed test sources; actual independently terminated packets. Pointwise95% source-bootstrap CIs. The source-informed DISTS reference uses the TX source and candidate reconstructions, not future noise. It is expensive, not a learned controller. All11 budgets include72-bit source wrappers; these are not wireless-frame costs.')
    if a.p2:
        p2=load(a.p2)
        for seed in [2026110401,2026110402,2026110403]:
            for metrics,tag in [(['dists','dreamsim'],'primary'),(['psnr_db','lpips_alex'],'auxiliary')]:
                fig,axes=plt.subplots(1,2,figsize=(7.16,2.8));fig.subplots_adjust(left=.07,right=.985,bottom=.19,top=.74,wspace=.25)
                present=[]
                for ax,m in zip(axes,metrics):
                    panel(ax,m,m.replace('_',' ').upper()+(' (higher is better)' if m=='psnr_db' else ' (lower is better)'))
                    for arm in ['R0','R1','R2','R3_estimated','R3']:
                        s=0 if arm=='R0' else seed;rows=[v for v in p2 if v['arm']==arm and int(v['seed'])==s and v['metric']==m and int(v['budget_bits'])>0]
                        if not rows:continue
                        ls,mark=STYLES[arm];n+=line(ax,rows,COLORS[arm],ls,mark,arm.replace('_estimated',' estimated'));present.append(arm)
                handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,1.0))
                export(fig,f'p2_{tag}_seed{seed}',f'Fixed4,000-update checkpoints, training seed{seed}; R0 has no new training. All500 test sources, actual longest-feasible native prefixes, all11 budgets and failed/NULL denominators retained. Error bars resample sources conditional on this checkpoint; training-seed spread is a separate table. No curve or budget is selected for favorable performance.')
    (O/'figure_identity.json').write_text(json.dumps(dict(input_sha256=inputs,exports=exports,plotted_mean_and_CI_records=n,style_skill='scientific-figure-making local SKILL.md + design-theory.md + common-patterns.md',dpi=600,vector_text='SVG text, PDF TrueType',visual_review='PENDING'),indent=2))
if __name__=='__main__':main()
