"""Selected 7dB frozen repeat-zero native reconstructions, full field of view."""
from pathlib import Path
import json,csv,hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
P=Path(__file__).resolve().parents[1]
data=json.loads((P/'inputs/f1/selected_native_mapping.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6.0,3.35));cols=['Source','WHOLE','Ada (adapted)','NeST-Com'];used=[]
# Four square images per row; no cropping, interpolation or anisotropic stretch.
side=1.40;left=.05;gap=.10;bottoms=[1.77,.17]
for row in [1,2]:
 for col in range(4):
  v=next(v for v in data if v['row']==row and v['column']==col)
  path=P/'figures/individual'/Path(v['display']['path']).name;assert hashlib.sha256(path.read_bytes()).hexdigest()==v['display']['sha256']
  im=Image.open(path);assert im.size==(256,256)
  ax=fig.add_axes([(left+col*(side+gap))/6,bottoms[row-1]/3.35,side/6,side/3.35]);ax.imshow(im,interpolation='none');ax.axis('off')
  if row==1:ax.text(.5,1.028,cols[col],ha='center',va='bottom',transform=ax.transAxes,fontsize=8.5)
  label=f"({'ab'[row-1]}) {v['label']}"if col==0 else f"{v['dists']:.3f} / {v['dreamsim']:.3f}"
  ax.text(.5,-.025,label,ha='center',va='top',transform=ax.transAxes,fontsize=8.0);used.append(v)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible()and t.get_text():
  b=t.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('figure3.'+ext),dpi=600,facecolor='white')
fig.savefig(P/'figures/figure3_print_size.png',dpi=150,facecolor='white')
(P/'validation/figure3_data_check.json').write_text(json.dumps(dict(rows=2,columns=4,N=1024,snr_db=7,repeat_index=0,size_inches=[6,3.35],full_native_field_of_view=True,display_metrics='DISTS / DreamSim, lower is better',no_new_scientific_calls=True,selection='Selected illustrative advantage examples',noise_namespace='Original repeat0: WHOLE/NeST 6201, Ada 2001; no cross-method same-noise claim',source_ids=[v['source_id']for v in used if v['method']=='Source'],visual_review='PENDING'),indent=2)+'\n')
