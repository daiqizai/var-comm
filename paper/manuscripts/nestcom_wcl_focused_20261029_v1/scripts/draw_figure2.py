"""Two panels from frozen means/CIs; deterministic rendering, no estimation."""
from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
P=Path(__file__).resolve().parents[1]
style=json.loads((P/'style.json').read_text());data=list(csv.DictReader((P/'inputs/common500_summary.csv').open()))
aliases={'EC_VAR_PARTIAL':'NEST','NEST_PARTIAL':'NEST','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024':'BPG','BPG_ADAPTIVE':'BPG'}
for r in data:r['method']=aliases.get(r['method'],r['method'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(7.16,2.40));fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.77,wspace=.20)
all_rows=[];handles={}
for ax,(metric,title,limits) in zip(axes,[('psnr_db','(a) PSNR (dB) ↑',(13,25)),('dreamsim','(b) DreamSim ↓',(0,.65))]):
 for method,s in style['methods'].items():
  v=[next(r for r in data if r['method']==method and r['snr_db']==str(n)and r['metric']==metric)for n in(4,10,19)]
  x=[4,10,19];y=[float(r['mean'])for r in v];lo=[float(r['ci_low'])for r in v];hi=[float(r['ci_high'])for r in v]
  assert all(l<=m<=h and limits[0]<=l<=h<=limits[1]for l,m,h in zip(lo,y,hi))
  line=s['linestyle'];line=(line[0],tuple(line[1]))if isinstance(line,list)else line
  opt=dict(color=s['color'],marker=s['marker'],linestyle=line,linewidth=s['linewidth'],markersize=s['markersize'],markerfacecolor=s['color']if s['filled']else'white',markeredgewidth=.75)
  ax.errorbar(x,y,yerr=[[m-l for m,l in zip(y,lo)],[h-m for h,m in zip(hi,y)]],fmt='none',ecolor=s['color'],elinewidth=.65,capsize=1.7,capthick=.65,zorder=2)
  artist,=ax.plot(x,y,**opt,zorder=3);assert list(artist.get_xdata())==x and list(artist.get_ydata())==y
  if metric=='psnr_db':handles[method]=Line2D([],[],label=s['label'],**opt)
  all_rows.extend(v)
 ax.set_title(title,loc='left',fontsize=9,pad=5);ax.set_xlim(3,20);ax.set_ylim(*limits);ax.set_xticks([4,10,19]);ax.set_xlabel('SNR (dB)',labelpad=2)
 ax.grid(axis='y',color='#E4E7E9',linewidth=.45);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,width=.6,pad=2)
axes[0].set_yticks([13,16,19,22,25]);axes[1].set_yticks([0,.2,.4,.6])
legend_order=['NEST','P100000','ARPC_ADAPTED','Swin200000','ADA_PREDCLASS_FIXEDFRAME','BPG']
fig.legend(handles=[handles[k] for k in legend_order],loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,columnspacing=1.8,handlelength=2.5,labelspacing=.55)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for text in fig.findobj(matplotlib.text.Text):
 if text.get_visible()and text.get_text():
  b=text.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(text.get_text(),b.bounds)
for ext in('pdf','svg','png'):fig.savefig(P/'figures'/('figure2.'+ext),dpi=600,facecolor='white')
fig.savefig(P/'figures/figure2_print_size.png',dpi=150,facecolor='white')
Image.open(P/'figures/figure2_print_size.png').convert('L').save(P/'figures/figure2_grayscale.png')
with(P/'figures/figure2_data.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
(P/'figures/figure2_validation.json').write_text(json.dumps(dict(points=36,confidence_intervals=36,all_artist_coordinates_exact=True,all_intervals_inside_axes=True,size_inches=[7.16,2.40],no_bootstrap=True),indent=2)+'\n')
