"""Retain the two chosen source IDs; show each current policy's first event."""
from pathlib import Path
import json,csv,hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
P=Path(__file__).resolve().parents[1]
data=json.loads((P/'inputs/fig3_unified_20261103/native_mapping.json').read_text())
selection=json.loads((P/'inputs/fig3_unified_20261103/selection.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
width,height=6.0,2.77
fig=plt.figure(figsize=(width,height));cols=['Source','WHOLE','SwinJSCC','Ada-TokenCom','NeST-Com'];used=[]
# Five full-field square images per row; no cropping or anisotropic stretch.
side=1.12;left=.04;gap=.08;bottoms=[1.49,.16]
for row in [1,2]:
 for col in range(5):
  v=next(v for v in data if v['row']==row and v['column']==col)
  path=P/v['display']['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==v['display']['sha256']
  im=Image.open(path);assert im.size==(256,256)
  ax=fig.add_axes([(left+col*(side+gap))/width,bottoms[row-1]/height,side/width,side/height]);ax.imshow(im,interpolation='none');ax.axis('off')
  if row==1:ax.text(.5,1.028,cols[col],ha='center',va='bottom',transform=ax.transAxes,fontsize=8.5)
  label=f"({'ab'[row-1]}) {v['label']}"if col==0 else f"{v['dists']:.4f} / {v['dreamsim']:.4f}"
  ax.text(.5,-.025,label,ha='center',va='top',transform=ax.transAxes,fontsize=7.5,fontweight=('bold' if col==4 else 'normal'));used.append(v)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible()and t.get_text():
  b=t.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('figure3.'+ext),dpi=600,facecolor='white')
svg=P/'figures/figure3.svg';svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')
fig.savefig(P/'figures/figure3_print_size.png',dpi=150,facecolor='white')
Image.open(P/'figures/figure3_print_size.png').convert('L').save(P/'figures/figure3_grayscale.png')
(P/'validation/figure3_data_check.json').write_text(json.dumps(dict(rows=2,columns=5,N=1024,snr_db=10,repeat_index=0,size_inches=[width,height],full_native_field_of_view=True,display_metrics='DISTS / DreamSim, lower is better',display_precision=4,headers_pt=8.5,metrics_pt=7.5,no_new_scientific_calls=True,selection=selection,source_ids=[v['source_id']for v in used if v['method']=='Source'],visual_review='PENDING'),indent=2)+'\n')
print(json.dumps(dict(rows=2,columns=5,inches=[width,height],full_field_images=len(used))))
