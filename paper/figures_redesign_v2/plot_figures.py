"""Draw closed means/intervals. No model, image reconstruction, scores, or resampling."""
from pathlib import Path
import csv,json,hashlib,os,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parent
S=json.loads((ROOT/'figure_style_v2.json').read_text())
C=json.loads((ROOT/'resolved_style.json').read_text())
M=json.loads((ROOT/'methods.json').read_text())
exports=[];layout_checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):
 with (ROOT/'plot_inputs'/n).open(newline='')as f:return list(csv.DictReader(f))
def configure():
 font=None;availability={}
 for name in S['font_fallback']:
  try:availability[name]=font_manager.findfont(name,fallback_to_default=False)
  except ValueError:availability[name]=None
  if font is None and availability[name]:font=name
 assert font
 plt.rcParams.update({'font.family':font,'font.size':7.3,'axes.labelsize':8.3,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
  'legend.fontsize':7.3,'axes.linewidth':.7,'axes.edgecolor':S['palette']['axis'],'text.color':S['palette']['text'],
  'axes.labelcolor':S['palette']['text'],'xtick.color':S['palette']['axis'],'ytick.color':S['palette']['axis'],
  'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'nestcom-redesign-v2',
  'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','axes.unicode_minus':True})
 return dict(actual_font=font,font_path=availability[font],font_file_sha256=sha(Path(availability[font])),availability=availability,matplotlib=matplotlib.__version__,font_files_distributed=False)
def frame(ax):
 ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
 ax.yaxis.grid(True,color=S['palette']['grid'],linewidth=.45)
 ax.tick_params(direction='out',length=2.3,width=.65,pad=2)
 ax.set_xlabel('SNR (dB)',labelpad=3)
def title(fig,ax,letter,label,y_mm):
 # A common two-line title band keeps all four plotting rectangles aligned.
 w,h=fig.get_size_inches()*25.4
 left=ax.get_position().x0
 fig.text(left,y_mm/h,f'({letter})',fontsize=8.8,ha='left',va='top')
 fig.text(left+5.3/w,y_mm/h,label,fontsize=8.3,ha='left',va='top',linespacing=1.05)
def draw2(ax,spec):
 data=read('fig2_plot_data.csv');frame(ax)
 for order,m in enumerate(M):
  rr=sorted([r for r in data if r['method']==m['label']and r['metric']==spec['id']],key=lambda r:int(r['snr_db']))
  assert len(rr)==3
  x=[int(r['snr_db'])for r in rr];y=[float(r['mean'])for r in rr]
  low=[float(r['ci_low'])for r in rr];high=[float(r['ci_high'])for r in rr]
  assert all(spec['ylim'][0]<=a<=b<=spec['ylim'][1]for a,b in zip(low,high))
  z=5 if order==0 else 4
  # Circle3.6 and hollow square4.2 retain both outer contours at overlapping10dB LPIPS.
  ax.plot(x,y,color=m['color'],ls=m['linestyle'],lw=m['linewidth'],zorder=z)
  ax.errorbar(x,y,yerr=[[b-a for a,b in zip(low,y)],[b-a for a,b in zip(y,high)]],fmt='none',
   ecolor=m['color'],elinewidth=.9,capsize=2,capthick=.9,zorder=7)
  ax.plot(x,y,ls='none',marker=m['marker'],ms=m['markersize'],mfc='white'if m['fill']=='none'else m['color'],
   mec=m['color']if m['fill']=='none'else'white',mew=.85 if m['fill']=='none'else .55,zorder=6 if order==0 else 5.5)
 ax.set(xlim=C['fig2_xlim'],ylim=spec['ylim'],xticks=[4,10,19],yticks=spec['yticks'])
def handles2():
 return [Line2D([],[],label=m['label'],color=m['color'],ls=m['linestyle'],lw=m['linewidth'],marker=m['marker'],ms=m['markersize'],
  mfc='white'if m['fill']=='none'else m['color'],mec=m['color']if m['fill']=='none'else'white',mew=.85 if m['fill']=='none'else .55)for m in M]
def draw3(ax,metric):
 data=read('fig3_plot_data.csv');frame(ax)
 ax.axhline(0,color='#8A8A8A',lw=.7,ls='--',dashes=(3,2),zorder=2)
 for name,s in S['paired_series'].items():
  rr=sorted([r for r in data if r['method']==name and r['metric']==metric],key=lambda r:int(r['snr_db']));assert len(rr)==3
  x=[i+s['offset']for i in range(3)];y=[float(r['mean'])for r in rr];lo=[float(r['ci_low'])for r in rr];hi=[float(r['ci_high'])for r in rr]
  assert all(C['fig3_limits'][metric][0]<a<=b<C['fig3_limits'][metric][1]for a,b in zip(lo,hi))
  ax.plot(x,y,ls='none',marker=s['marker'],ms=3.6,color=s['color'],mfc=s['facecolor'],mec=s['color']if name=='Raw'else'white',mew=.85 if name=='Raw'else .55,zorder=4)
  # Draw true endpoints above glyphs, with caps wider than markers: no minimum CI height.
  ax.errorbar(x,y,yerr=[[b-a for a,b in zip(lo,y)],[b-a for a,b in zip(y,hi)]],fmt='none',
   ecolor=s['color'],elinewidth=.9,capsize=2.6,capthick=.9,zorder=5)
 ax.set(xlim=[-.42,2.42],xticks=[0,1,2],xticklabels=['4','10','19'],ylim=C['fig3_limits'][metric],yticks=C['fig3_ticks'][metric])
 ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'if metric=='dreamsim'else'%.1f'))
def handles3():
 return [Line2D([],[],label=n,ls='none',marker=s['marker'],ms=3.6,mfc=s['facecolor'],mec=s['color']if n=='Raw'else'white',mew=.85 if n=='Raw'else.55)for n,s in S['paired_series'].items()]
def export(fig,name,w,h):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
 overflow=[]
 for obj in fig.findobj(Text):
  if not obj.get_visible()or not obj.get_text():continue
  b=obj.get_window_extent(renderer)
  if b.x0<-.5 or b.y0<-.5 or b.x1>canvas.x1+.5 or b.y1>canvas.y1+.5:overflow.append(obj.get_text())
 assert not overflow,(name,overflow)
 legends=[]
 for leg in fig.legends:
  b=leg.get_window_extent(renderer);overlap=[i for i,a in enumerate(fig.axes)if b.overlaps(a.get_window_extent(renderer))]
  assert not overlap,(name,'legend overlaps axes',overlap)
  legends.append(dict(bounds_pixels=list(b.bounds),overlaps_axes=False))
 layout_checks.append(dict(figure=name,canvas_mm=[w,h],text_overflow=overflow,legends=legends,axes=[list(a.get_position().bounds)for a in fig.axes]))
 for ext in ['pdf','svg','png']:
  p=ROOT/(name+'.'+ext);p.parent.mkdir(exist_ok=True)
  meta={'Creator':'NeST-Com deterministic figure redraw; existing statistics only'}
  if ext=='pdf':meta.update(CreationDate=None,ModDate=None)
  if ext=='svg':meta.update(Date=None)
  fig.savefig(p,dpi=600,bbox_inches=None,metadata=meta)
  if ext=='svg':p.write_text('\n'.join(line.rstrip()for line in p.read_text().splitlines())+'\n',encoding='utf8')
  exports.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),width_mm=w,height_mm=h))
 plt.close(fig)
def main():
 fonts=configure();specs=C['metrics'];W=C['fig2_width_mm'];H=C['fig2_height_mm']
 fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160)
 left,right,gap,bottom,top=8.8,3,9.3,10.8,37.7
 pw=(W-left-right-3*gap)/4
 labels=['PSNR (dB) ↑','LPIPS ↓','DreamSim ↓','Prediction\nagreement (%) ↑']
 for j,spec in enumerate(specs):
  ax=fig.add_axes([(left+j*(pw+gap))/W,bottom/H,pw/W,(top-bottom)/H]);draw2(ax,spec);title(fig,ax,'abcd'[j],labels[j],44.4)
 fig.legend(handles=handles2(),loc='upper center',bbox_to_anchor=(.5,1-.5/H),ncol=4,frameon=False,handlelength=1.7,handletextpad=.4,columnspacing=.95,borderaxespad=0)
 export(fig,'fig2_main_1x4',W,H)
 W=C['fig3_width_mm'];H=C['fig3_height_mm'];left,right,gap,bottom,top=9.2,3.1,12.0,10.8,33.8;pw=(W-left-right-gap)/2
 fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160)
 for j,metric in enumerate(['psnr_db','dreamsim']):
  ax=fig.add_axes([(left+j*(pw+gap))/W,bottom/H,pw/W,(top-bottom)/H]);draw3(ax,metric)
  title(fig,ax,'ab'[j],['PSNR gain\n(dB)','DreamSim\nreduction'][j],40.9)
 fig.legend(handles=handles3(),loc='upper center',bbox_to_anchor=(.5,1-.7/H),ncol=2,frameon=False,handletextpad=.35,columnspacing=1.4,borderaxespad=0)
 export(fig,'fig3_paired_two_panels',W,H)
 # Separate editable panels keep native font/line sizes, not scaled crops.
 for j,spec in enumerate(specs):
  W,H=50.,55.;fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160);ax=fig.add_axes([9/W,11/H,38/W,30/H]);draw2(ax,spec);title(fig,ax,'abcd'[j],labels[j],49)
  export(fig,'panels/fig2_'+spec['file'],W,H)
 for j,metric in enumerate(['psnr_db','dreamsim']):
  W,H=45.,50.;fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=160);ax=fig.add_axes([9/W,11/H,33/W,26/H]);draw3(ax,metric);title(fig,ax,'ab'[j],['PSNR gain (dB)','DreamSim\nreduction'][j],45)
  export(fig,'panels/fig3_'+('psnr_gain'if j==0 else'dreamsim_reduction'),W,H)
 (ROOT/'export_manifest.json').write_text(json.dumps(dict(fonts=fonts,exports=exports,layout_checks=layout_checks),indent=2)+'\n')
 print(json.dumps(dict(exports=len(exports),font=fonts['actual_font'],status='RENDERED_REQUIRES_VISUAL_REVIEW')))
if __name__=='__main__':main()
