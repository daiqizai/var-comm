"""Display frozen median-rank and minimum-DISTS-gain sources; no inference."""
from pathlib import Path
import json,hashlib
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
data=json.loads((P/'inputs/examples/mapping.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(6.5,4.85))
used=[]
for row,selection in enumerate(('median','worst')):
 for col,method in enumerate(('Source','WHOLE','Token')):
  v=next(x for x in data if x['selection']==selection and x['method']==method)
  path=P/'inputs/examples'/Path(v['display']['path']).name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==v['display']['sha256']
  im=Image.open(path);assert im.size==(256,256)
  ax=fig.add_axes([(.15+2.13*col)/6.5,(2.64-2.39*row)/4.85,2.0/6.5,2.0/4.85])
  ax.imshow(im,interpolation='none');ax.axis('off')
  if row==0:ax.text(.5,1.025,'NeST-Com' if method=='Token' else method,ha='center',va='bottom',transform=ax.transAxes)
  label=('Median rank (source 209)' if selection=='median' else 'DISTS degradation (source 39)') if method=='Source' else f"{v['dists']:.3f} / {v['dreamsim']:.3f}"
  ax.text(.5,-.035,label,ha='center',va='top',transform=ax.transAxes,fontsize=8.5)
  used.append(v)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
  b=t.get_window_extent(renderer)
  assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,(t.get_text(),b.bounds)
for ext in ('pdf','svg','png'):fig.savefig(P/'figures'/('supp_examples_distribution.'+ext),dpi=300,facecolor='white')
(P/'validation/distribution_examples_check.json').write_text(json.dumps(dict(inputs=used,selection_rule='Median rank and minimum source-mean DISTS gain, preregistered before extraction; fixed first channel repeat',N=1024,snr_db=10,metrics='Displayed native-output DISTS / DreamSim',new_inference=0,new_metrics=0,visual_review='PENDING'),indent=2)+'\n')
plt.close(fig)
