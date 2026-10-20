"""Render a separate CLIP-I supplement from closed CSV; never calculate scores."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image
import re

def main():
 assert read(O/'delivery/completion.json')['status']=='COMPLETE_CLIP_I_FIVE_METHOD_COMMON500'
 data=O/'delivery/summary.csv'
 with data.open()as f:rows=list(csv.DictReader(f))
 assert len(rows)==15 and all(v['metric']=='clip_image_cosine'for v in rows)
 original=R/'paper/wcl_figures_tables_20261020_v1/style.json'
 style=read(original)
 out=O/'figures_v2';out.mkdir()
 save(out/'style.json',dict(width_mm=178,height_mm=76,axis_pt=9,tick_pt=8,legend_pt=8.5,
     style_source=desc(original),methods=style['methods'],source_skill='scientific-figure-making',
     metric_scale=1,no_percent=True))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,
     'ytick.labelsize':8,'legend.fontsize':8.5,'axes.linewidth':.7,'axes.spines.top':False,
     'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(178/25.4,76/25.4))
 fig.subplots_adjust(left=.105,right=.975,bottom=.19,top=.78)
 handles=[]
 for m in METHODS:
  s=style['methods'][m];g=sorted([v for v in rows if v['method']==m],key=lambda v:int(v['snr_db']))
  xs=np.array([float(v['snr_db'])for v in g]);ys=np.array([float(v['mean'])for v in g])
  lows=np.array([float(v['ci_low'])for v in g]);highs=np.array([float(v['ci_high'])for v in g])
  assert list(xs)==[4,10,19]and np.all(lows<=ys)and np.all(highs>=ys)
  ls=tuple(s['linestyle'])if isinstance(s['linestyle'],list)else s['linestyle']
  ax.errorbar(xs,ys,yerr=np.stack([ys-lows,highs-ys]),fmt='none',ecolor=s['color'],elinewidth=1.1,capsize=4,capthick=1.1,zorder=2)
  ax.plot(xs,ys,color=s['color'],linestyle=ls,linewidth=s['linewidth'],zorder=3)
  ax.plot(xs,ys,linestyle='none',marker=s['marker'],markersize=s['markersize'],markeredgecolor=s['color'],
      markeredgewidth=1.05,markerfacecolor=s['color']if s['filled']else'white',zorder=4)
  handles.append(Line2D([],[],color=s['color'],linestyle=ls,linewidth=s['linewidth'],
      marker=s['marker'],markersize=s['markersize'],markerfacecolor=s['color']if s['filled']else'white',
      markeredgewidth=1.05,label=s['label']))
 ax.set_xticks([4,10,19]);ax.set_xlim(3,20)
 low=min(float(v['ci_low'])for v in rows)
 ax.set_ylim(min(.5,low-.02),1)
 ax.set_xlabel('SNR (dB)');ax.set_ylabel('CLIP-I ↑')
 ax.grid(axis='y',color='#e6e6e6',linewidth=.5,zorder=0)
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.96),ncol=5,frameon=False,
     columnspacing=1.2,handlelength=2.6,handletextpad=.5)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bbox=fig.bbox
 outside=[]
 for t in fig.findobj(Text):
  if t.get_visible()and t.get_text():
   b=t.get_window_extent(renderer)
   if b.x0<bbox.x0-.5 or b.y0<bbox.y0-.5 or b.x1>bbox.x1+.5 or b.y1>bbox.y1+.5:outside.append(t.get_text())
 assert not outside,outside
 for ext in ['pdf','svg','png']:
  fig.savefig(out/('clip_i_common500.'+ext),dpi=600,facecolor='white')
 fig.savefig(out/'print_size.png',dpi=120,facecolor='white')
 with Image.open(out/'clip_i_common500.png')as im:im.convert('L').save(out/'grayscale.png')
 with Image.open(out/'print_size.png')as im:im.convert('L').save(out/'print_size_grayscale.png')
 svg=(out/'clip_i_common500.svg').read_text()
 assert '<image'not in svg and '<text' in svg
 plt.close(fig)
 caption="""CLIP-I on the original common500 at N=1024. CLIP-I denotes CLIP image–image
cosine similarity between the source and its actual received reconstruction,
using OpenAI CLIP ViT-L/14 with the frozen224-input floating-point preprocessing
and FP32 inference. Higher values indicate greater image-embedding similarity.
Points are source means after averaging each source's three original noise
realizations; error bars are pointwise95% intervals obtained from the existing
10000×500 source-bootstrap index bank. NeST-Com19dB uses the corrected T1-C m10
version; Latent JSCC and SwinJSCC use their final100k and200k checkpoints.
All original receiver outcomes retain their weight. Noise realizations differ
between historical method namespaces. This supplement does not select models
or policies and excludes development examples and N2048.
"""
 (out/'caption.md').write_text(caption)
 save(out/'export_audit.json',dict(source=desc(data),figure_mm=[178,76],dpi=600,
     label_clipping=[],SVG_raster_images=0,editable_SVG_text=True,visual_review_required=True,
     scientific_compute=0,statistic_recompute=0,skill_path='C:/Users/11946/.codex/skills/scientific-figure-making/SKILL.md'))
 save(out/'manifest.json',dict(files=[desc(p)for p in sorted(out.iterdir())if p.is_file()]))
 print(json.dumps(desc(out/'manifest.json')))
if __name__=='__main__':main()
