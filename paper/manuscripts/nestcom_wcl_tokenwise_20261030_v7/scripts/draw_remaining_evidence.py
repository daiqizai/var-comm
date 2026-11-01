"""Plot already-closed E2 or E6 CSVs; never launches or estimates experiments.

Requires a fresh, hash-bound execution_closure.json copied after the remote
statistics owner and child have actually exited. No figures are generated from
incomplete receivers or merely present intermediate files.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(stage):
    d=P/'inputs'/stage;c=json.loads((d/'execution_closure.json').read_text())
    assert c['actual_wait_returncode']==0 and c['owner_gone'] and c['child_gone'] and not c['failure_present']
    assert c['summary_sha256']==sha(d/'summary.csv')
    with (d/'summary.csv').open(newline='',encoding='utf8') as f:r=list(csv.DictReader(f))
    return d,r,c
def one(rs,**k):
    a=[r for r in rs if all(str(r[x])==str(v) for x,v in k.items())]
    assert len(a)==1,(k,len(a));return a[0]
styles={'WHOLE':('#767676','s','--'),'Token_NeST':('#3775BA','o','-'),'Ada':('#42949E','^','-.')}
labels={'WHOLE':'WHOLE','Token_NeST':'NeST-Com','Ada':'Ada (adapted)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
def axis(ax,title,xlabel):
    ax.set_title(title,loc='left',fontsize=9,pad=6);ax.set_xlabel(xlabel,labelpad=2)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E4E7E9',lw=.45);ax.set_axisbelow(True);ax.tick_params(length=2.5,width=.6,pad=2)
def series(ax,x,y,lo,hi,method,conditional=False):
    assert all(l<=m<=h for l,m,h in zip(lo,y,hi))
    color,marker,ls=styles[method]
    if not conditional:ax.errorbar(x,y,yerr=[[m-l for m,l in zip(y,lo)],[h-m for h,m in zip(hi,y)]],fmt='none',ecolor=color,elinewidth=.65,capsize=1.7,zorder=2)
    line,=ax.plot(x,y,color=color,marker=marker,ls=':' if conditional else ls,lw=1.2,ms=3.4 if conditional else 4.3,mfc='white',mew=.7,zorder=3)
    assert list(line.get_xdata())==x and list(line.get_ydata())==y
def export(fig,stem,record):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
    for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/(stem+'.'+ext),dpi=600,facecolor='white')
    fig.savefig(P/'figures'/(stem+'_print_size.png'),dpi=150,facecolor='white')
    record.update(no_scientific_calls=True,no_estimation=True,visual_review='PENDING',SVG_text_editable=True)
    (P/'validation'/(stem+'_data_check.json')).write_text(json.dumps(record,indent=2)+'\n')
def e2():
    d,rs,c=load('e2');assert len(rs)==90
    x=[620,1212,1302,1940,2248,2577];fig,axs=plt.subplots(1,2,figsize=(7.16,2.8));fig.subplots_adjust(left=.07,right=.985,bottom=.30,top=.80,wspace=.23)
    for ax,(metric,title) in zip(axs,[('dists','(a) DISTS ↓'),('dreamsim','(b) DreamSim ↓')]):
        for method in styles:
            vs=[one(rs,method=method,B_bits=b,metric=metric) for b in x]
            assert all(v['n_sources']=='500' and int(v['logical'])==(1500 if method=='Ada' else 500) for v in vs)
            series(ax,x,[float(v['mean']) for v in vs],[float(v['ci95_low']) for v in vs],[float(v['ci95_high']) for v in vs],method)
        axis(ax,title,'Total information budget B (bit)');ax.set_xticks(x);ax.tick_params(axis='x',labelrotation=55);ax.set_xlim(550,2650)
    handles=[Line2D([],[],color=s[0],marker=s[1],ls=s[2],mfc='white',label=labels[m]) for m,s in styles.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False)
    export(fig,'supp_figure_s1_information_budget',dict(input_sha256=sha(d/'summary.csv'),closure=c,points=36,pointwise_CIs=36,unequal_numeric_x=True,generation_repeats_not_channel_repeats=True))
def e6():
    d,rs,c=load('e6');assert len(rs)==15
    # Actual method names are verified rather than inferred from plot labels.
    aliases={'WHOLE':'WHOLE','Token_NeST':'Token_NeST','Ada':'Ada'}
    observed={r['method'] for r in rs};assert observed==set(aliases.values()),observed
    x=[8.0,8.5,9.0,9.5,10.0];fig,axs=plt.subplots(1,3,figsize=(7.16,2.8));fig.subplots_adjust(left=.07,right=.985,bottom=.23,top=.78,wspace=.30)
    for ax,(metric,title) in zip(axs,[('frame_rejection_rate','(a) Rejection rate'),('dists','(b) DISTS ↓'),('dreamsim','(c) DreamSim ↓')]):
        for method in styles:
            vs=[]
            for snr in x:
                a=[v for v in rs if v['method']==aliases[method] and float(v['snr_db'])==snr];assert len(a)==1;vs.append(a[0])
            assert all(v['n_sources']=='100' and v['logical_frames']=='2000' for v in vs)
            series(ax,x,[float(v[metric]) for v in vs],[float(v[metric+'_ci95_low']) for v in vs],[float(v[metric+'_ci95_high']) for v in vs],method)
            if metric!='frame_rejection_rate':
                valid=[(xx,v) for xx,v in zip(x,vs) if int(v['total_success'])>0]
                xx=[xx for xx,v in valid];yy=[float(v[metric+'_successful_only']) for xx,v in valid]
                series(ax,xx,yy,yy,yy,method,True)
        axis(ax,title,'Actual SNR (dB)');ax.set_xticks(x);ax.set_xlim(7.9,10.1)
        if metric=='frame_rejection_rate':ax.set_ylim(-.025,1.025)
    handles=[Line2D([],[],color=s[0],marker=s[1],ls=s[2],mfc='white',label=labels[m]) for m,s in styles.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False)
    fig.text(.5,.01,'Dotted quality lines: successful receptions only; no point when all receptions fail.',ha='center',fontsize=7.5)
    export(fig,'supp_figure_s2_threshold',dict(input_sha256=sha(d/'summary.csv'),closure=c,overall_points=45,source_cluster_CIs=45,conditional_values='descriptive with changing successful-event denominator',actual_frames=30000))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['e2','e6']);args=ap.parse_args()
    {'e2':e2,'e6':e6}[args.stage]()
